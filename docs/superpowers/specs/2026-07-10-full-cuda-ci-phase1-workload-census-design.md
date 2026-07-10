# Full-CUDA CI Phase 1 Workload Census Design

## Status

Approved direction for the earliest incomplete phase in `goal-5.6.md`.
Phase 0 is frozen at commit `93ae843` and supplies the immutable logical CI
trace and graph-semantic oracle used here.

## Decision

Build the census offline from the Phase 0 oracle. Do not instrument or modify
the skeleton execution path.

The census records two distinct quantities:

```text
canonical_global_unique_target_s_count = 110,617
s_affinity_executed_mgcv_fit_count      = 273,284
```

`110,617` is the canonical number of distinct `target|S` keys in the logical
CI workload. `273,284` is a route-specific executed fit count from the current
S-affinity worker layout. The latter must not be labeled as a unique-key count.

## Why This Approach

### Approach A: offline canonical census (selected)

Derive every logical test and residual key from the immutable Phase 0 trace,
then fit each unique residual key once with the legacy mgcv authority.

Benefits:

- does not alter scheduler, replay, CI decisions, or timings;
- produces the global canonical key set rather than worker-local observations;
- supports deterministic sharding, checkpointing, and restart;
- makes the Phase 2 setup contract consume a stable machine-readable input.

Cost:

- requires a separate exhaustive mgcv pass over 110,617 keys;
- setup diagnostics add work beyond the existing residual-only fit path.

### Approach B: instrument the live skeleton run

Capture fit/setup metadata inside residual-provider workers.

Rejected for Phase 1 because worker-local duplication would obscure the global
key set, IPC would be substantial, and instrumentation could perturb the route
whose behavior the census is supposed to describe.

### Approach C: structural census plus sampled numerical cases

Record every key structurally but collect full mgcv metadata only for selected
cases.

Rejected because Phase 1 explicitly requires every high-condition,
rank-deficient, near-constant, multi-penalty, and near-alpha case to be
identified. Sampling cannot prove that coverage.

## Inputs

The standard runner consumes only pinned Phase 0 and canonical fixture inputs:

```text
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/manifest.json
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/logical_ci_trace.rds
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/deletion_trace.csv
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/adjacency.rds
fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds
```

The runner validates the Phase 0 manifest hashes and canonical contract before
building or resuming any shard.

## Architecture

Add three focused files:

```text
fastkpc/R/full_cuda_ci_workload_census.R
fastkpc/tools/run_full_cuda_ci_workload_census.R
fastkpc/tests/test_full_cuda_ci_workload_census.R
```

The module has four independent responsibilities:

1. normalize and validate Phase 0 inputs;
2. construct logical-test and residual-request tables;
3. execute restartable legacy-mgcv metadata shards;
4. merge shards and write the standard artifact schema and summaries.

## Stage A: Logical CI Census

Produce one row for each of the 240,489 canonical logical CI tests.

Required fields:

```text
logical_sequence_id
source_task_index
level
x
y
S_key
S_size
formula_route
reference_p_value
alpha
deletes_edge
selected_sepset
absolute_distance_from_alpha
log_distance_from_alpha
residual_key_x
residual_key_y
```

Definitions:

- `formula_route = unconditional` for `S_size = 0`;
- `formula_route = full_smooth` for `1 <= S_size <= 2`;
- `formula_route = additive_smooth` for `S_size > 2`;
- `selected_sepset` is true only for the accepted deleting test recorded by
  the canonical deletion trace;
- log distance uses a finite floor for zero p-values and records the floor in
  the manifest.

Stage A must reproduce exactly:

```text
logical_test_count              = 240,489
conditional_logical_test_count  = 238,276
conditional_residual_requests   = 476,552
unique_S_count                  = 8,634
```

## Stage B: Canonical Residual Requests

Expand each conditional logical test into `x|S` and `y|S` requests, then dedupe
by the exact canonical key:

```text
target index + sorted S + canonical data/config identity
```

The canonical residual-request table records:

```text
residual_key
target
S_key
S_size
formula_route
same_S_group_id
same_S_group_size
request_multiplicity
first_logical_sequence_id
last_logical_sequence_id
level
```

The global unique count must be `110,617`. The artifact also records the
historical S-affinity executed fit count (`273,284`) as a separate route metric.

## Stage C: Legacy mgcv Metadata

Fit each unique key exactly once with the current authoritative legacy formula
and default `mgcv::gam` selection semantics.

Formula construction must match `regrXonS`:

```text
|S| <= 2: target ~ s(S1, ..., Sk)
|S| > 2:  target ~ s(S1) + ... + s(Sk)
```

No fixed-sp replacement, approximate smoother, or CUDA path is allowed in this
phase.

Per-key metadata:

```text
fit_status
fit_error
fit_time_ms
formula
method
optimizer
family
link
converged
selected_sp
GCV_Cp_score
EDF
coefficient_rank
model_matrix_nrow
model_matrix_ncol
model_matrix_rank
model_matrix_condition
penalty_count
penalty_block_dimensions
penalty_ranks
constraint_dimensions
conditioning_rank
conditioning_condition
conditioning_rank_deficient
near_constant_conditioning_count
target_sd
target_near_constant
residual_hash
fitted_hash
model_matrix_hash
coefficient_hash
mgcv_version
R_version
```

Large vectors and matrices are not stored in the final artifact. Only hashes,
dimensions, ranks, conditions, and scalar fit metadata are retained.

Condition estimation must be deterministic and bounded. Exact SVD is not
required for every well-conditioned case; the method and thresholds are
recorded in the manifest. Non-finite or rank-deficient cases are retained as
explicit census rows, not dropped.

## Sharding and Restart

The numerical pass is partitioned by deterministic hash of `residual_key`.

```text
shard_id = hash(residual_key) modulo shard_count
```

Each shard writes atomically:

```text
shards/shard_<id>.rds
shards/shard_<id>.summary.json
```

A completed shard is reused only when its manifest matches:

```text
data hash
Phase 0 oracle hash
mgcv version
R version
formula semantics version
metadata schema version
shard count
shard id
```

Unix workers may use `parallel::mclapply`. The parent only merges completed
rows; it does not share mutable mgcv objects across processes. The default
worker count comes from an explicit environment variable and is recorded.

## Artifact

Standard output:

```text
fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/
  manifest.json
  summary.json
  summary.md
  commands.txt
  environment.txt
  graph_agreement.csv
  sepset_agreement.csv
  n_edgetests.csv
  first_divergence.json
  fallbacks.csv
  stage_timing.csv
  raw_runs.csv
  logical_ci_tests.rds
  logical_ci_tests.csv
  residual_requests.rds
  residual_requests.csv
  residual_metadata.rds
  residual_metadata.csv
  counts_by_s_size.csv
  counts_by_penalty_count.csv
  counts_by_model_dimension.csv
  counts_by_condition_bucket.csv
  same_s_group_distribution.csv
  near_alpha_tests.csv
  unsupported_envelope.csv
  shards/
```

CSV outputs are for inspection. RDS files preserve exact integer, logical, and
list-column data without lossy delimiter parsing.

## Fail-Closed Rules

The runner exits nonzero and writes `pass=false` when any of these occur:

```text
Phase 0 manifest/hash mismatch
logical trace count or canonical-plan mismatch
request count or unique-key count mismatch
duplicate residual metadata key
missing shard
mgcv fit error
non-finite required metadata without an explicit supported status
graph semantic mismatch against Phase 0
unknown fallback or approximate backend count != 0
```

Every failed key remains in `residual_metadata` with its error and shard id.

## Correctness and Exit Gates

The census is observational. It must not execute a new skeleton or alter the
Phase 0 oracle.

Hard gates:

```text
logical tests                         = 240,489
conditional residual requests         = 476,552
canonical global unique target|S      = 110,617
unique S groups                       = 8,634
S-affinity executed mgcv fits         = 273,284 (separate route metric)
mgcv metadata rows                    = 110,617
mgcv fit errors                       = 0
SHD                                   = 0
normalized sepsets identical          = TRUE
n.edgetests exact                     = TRUE
canonical deletion trace identical    = TRUE
```

The summary must include distributions by `|S|`, penalty count, model
dimension, rank/condition bucket, same-S group size, near-alpha distance, and
time weighted by `|S|` and penalty count.

## Tests

Focused tests use a small canonical trace and verify:

- logical-test expansion and selected-sepset marking;
- exact request multiplicity and global deduplication;
- formula routing at `|S| = 1`, `2`, and `3`;
- same-S group IDs and sizes;
- metadata extraction for single- and multi-penalty cases;
- rank-deficient and near-constant rows remain explicit;
- shard resume rejects mismatched manifests;
- duplicate/missing shard rows fail closed;
- standard artifact schema and first-divergence output.

A real-subset test uses keys drawn from the canonical Phase 0 trace. The final
351x48 artifact is mandatory for Phase 1 completion.

## Non-Goals

Phase 1 does not:

- change the skeleton scheduler or replay order;
- cache or accelerate production residual fits;
- define `PreparedSSetup` (Phase 2);
- run fixed-sp C++/CUDA solves (Phase 3);
- change the legacy oracle or tolerance gates.
