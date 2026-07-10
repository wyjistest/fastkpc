# Full-CUDA CI Phase 1 Workload Census Design

## Status

Amended after expert review. Phase 0 is frozen at commit `93ae843` and
supplies the immutable logical CI trace and graph-semantic oracle. Phase 1
implementation beyond the structural census must follow this amended contract.

The previous implementation plan is superseded until this design is approved
and a replacement plan is written.

## Decision

Build the census offline from the Phase 0 oracle. Do not instrument or modify
the live skeleton execution path.

Keep canonical workload counts separate from route-specific execution metrics:

```text
canonical_global_unique_conditional_target_s_count = 110,617
s_affinity_executed_mgcv_fit_count                  = 273,284
```

`110,617` is the canonical number of distinct conditional `target|S` keys in
the complete logical workload. `273,284` is a historical fit count produced by
one S-affinity worker/cache layout. It is not a unique-key count and is not a
Phase 1 hard gate unless a future hash-protected route trace makes it
independently recountable.

## Why Offline Census

The selected approach derives every logical test and residual key from the
immutable Phase 0 trace, then fits each canonical residual key once with the
pinned legacy mgcv authority.

Benefits:

- no change to scheduler, replay, CI decisions, or production timings;
- a global canonical key corpus instead of worker-local cache observations;
- deterministic sharding, checkpointing, and restart;
- a stable machine-readable input for Phase 2 `PreparedSSetup` work;
- explicit separation of response-independent setup and target-specific fit
  metadata.

Live skeleton instrumentation is rejected because worker-local duplication
obscures the canonical key set and instrumentation can perturb the route being
described. Sampling-only numerical metadata is rejected because Phase 1 must
identify every high-risk case in the canonical corpus.

## Phase 0 Input Contract

The standard runner receives an oracle artifact directory and the canonical
dataset path. It must call one unified Phase 0 oracle loader. The loader must
require and independently validate all of these inputs before any census shard
is built or resumed:

```text
manifest.json
summary.json
adjacency.rds
sepsets.rds
n_edgetests.csv
logical_ci_trace.rds
deletion_trace.csv
pmax.rds
graph_agreement.csv
sepset_agreement.csv
first_divergence.json
fallbacks.csv
canonical dataset
```

The canonical paths are currently:

```text
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/
fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds
```

The source-controlled input contract v1 pins Phase 0 source commit `93ae843`
and these file SHA-256 values:

```text
dataset/cancer_RD-causalDiscoveryInput.rds  e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036
oracle/adjacency.rds                        6701a033e821f8433842ae825f67715b6c2349e3c36515b45296f125ff7e1d4e
oracle/deletion_trace.csv                   00eeb3fe843e9f868b133cef9f573430ea9d49fe7b0dd53ba3be51e2e9e94486
oracle/fallbacks.csv                        dc45430a89ad1c4fb85cf9f4c63b4babe89df8cf2b505759e7389f7b5462beca
oracle/first_divergence.json                b29373f20a56d99ce76ef4c21f2d508a13ed22ed009a30c902a1e6d23e46fef6
oracle/graph_agreement.csv                  f1eabdbc578607ee0b124a6dcd3fd8250077493472605583889eb532cd3b85d5
oracle/logical_ci_trace.rds                 b777c5dc1b9acad08c133ccd668eae5e9444a89d481c7f8adf9b2c2a0dd6cda5
oracle/manifest.json                        f907559586c4b766f483bdc01b4074d93ce2c8b80972c3199c4493848b2b8750
oracle/n_edgetests.csv                      6c0e1ccb14c9721e7056aa91e851065877965ab33869867dd130c7ca3d503058
oracle/pmax.rds                             2fafe1f5084dcb86114adfb86d06855350d36872e59795b6ee604bf4c6e19df5
oracle/sepset_agreement.csv                 9e57978d03fa0e62526b884e0566d2671d2c74417f117fd420b0fb4afb85a256
oracle/sepsets.rds                          69853449f95e1486ef237a2b1bd7c3a99d94cac4c0f202d7c509c890a49e1ca6
oracle/summary.json                         eec6724d9fd69671399783b565c2dd8bbdbc3a4e553ba742ac781d483354ade7
```

The dataset RDS file hash above protects the input bytes. It is distinct from
the canonical normalized data-matrix identity used in residual keys:

```text
dataset_file_sha256       = e03cbfafed3336f3da7725878e92af24556b5ebdc6227b94ea84469e62e94036
dataset_sha256_in_key     = 971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7
```

`dataset_sha256_in_key` is the Phase 0 `data_hash`, recomputed after coercing
the canonical 351x48 matrix to double with the Phase 0 hashing helper. It must
not be replaced by the RDS file hash.

Construct `oracle_input_bundle_sha256` by sorting the logical paths above with
bytewise ascending order, serializing each row as
`<logical_path><TAB><lowercase_sha256><LF>`, concatenating the rows with a
final LF already supplied by the last row, and hashing the raw UTF-8 bytes.
Contract v1 requires:

```text
oracle_input_bundle_sha256 = 7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e
```

The loader must not trust `summary.json` or `manifest.json` `pass` fields as a
substitute for validation. It must:

1. match every required file against a versioned, source-controlled
   `phase1_oracle_input_contract` containing its exact SHA-256 and the frozen
   Phase 0 source commit;
2. recompute the canonical dataset SHA-256 and dimensions;
3. normalize adjacency, sepsets, deletion trace, and logical trace through the
   Phase 0 contract helpers;
4. recompute adjacency, normalized-sepset, and deletion-trace hashes and match
   the pinned canonical contract;
5. validate logical row counts against `n_edgetests.csv` and rebuild the
   canonical layer plan independently;
6. require all graph/sepset/test/deletion agreement rows to pass;
7. require unknown, approximate, fallback, and backend error counts to be zero;
8. cross-check the recomputed facts against `summary.json` and `manifest.json`;
9. record the SHA-256 of every input file and a combined
   `oracle_input_bundle_sha256` in the Phase 1 manifest.

The historical `manifest.json::source_result_path` and
`source_result_hash` are provenance fields, not additional Phase 1 inputs.
Do not reopen that mutable/ignored result RDS. Do not call
`fastkpc_full_cuda_validate_canonical_fixture()` on the reduced skeleton loaded
from the artifact, because that object no longer carries `per.level.log` and
would reconstruct a different deletion-trace representation from sepsets.
Validate the normalized data, adjacency, sepsets, CSV deletion trace, and
`n.edgetests` separately against the canonical semantic contract. Coerce the
CSV deletion-trace `p_value` column back to numeric before hashing. The exact
Phase 0 oracle bundle hashes and these independently recomputed semantics
replace the external source-result dependency.

The Phase 1 artifact reports inherited oracle evidence with explicit scope:

```text
oracle_inherited_graph_gate = TRUE
new_candidate_graph_gate    = NOT_APPLICABLE
```

Phase 1 does not execute a new skeleton and therefore must not claim that it
performed a new candidate graph comparison. Copied graph, sepset, deletion,
and `n.edgetests` files must carry `gate_scope = phase0_oracle_inherited`.

## Architecture

Use focused modules with explicit boundaries:

```text
fastkpc/R/full_cuda_ci_workload_census.R
fastkpc/tools/run_full_cuda_ci_workload_census.R
fastkpc/tests/test_full_cuda_ci_workload_census.R
fastkpc/tests/test_full_cuda_ci_workload_census_parity.R
fastkpc/tests/test_full_cuda_ci_workload_census_restart.R
fastkpc/tests/test_full_cuda_ci_workload_census_real_subset.R
```

Responsibilities:

1. load and validate the complete Phase 0 oracle bundle;
2. construct logical-test and canonical residual-request tables;
3. prove offline single-target fitting matches the actual `regrXonS` layout;
4. execute restartable legacy-mgcv metadata shards;
5. validate and compress response-independent same-S setup metadata;
6. classify numerical risk and write the standard fail-closed artifact.

## Stage A: Logical CI Census

Produce one row for each canonical logical CI test.

Required fields:

```text
logical_sequence_id
source_sequence_id
source_task_index
level
x
y
S_key
S_size
formula_class
reference_p_value
alpha
reference_decision
reference_independent
deletes_edge
selected_sepset
signed_distance_from_alpha
absolute_distance_from_alpha
signed_log_ratio_from_alpha
absolute_log_distance_from_alpha
residual_key_x
residual_key_y
```

Reuse the existing compatibility enum from
`fastkpc/R/mgcv_compat_contract.R`:

```text
direct-ci
full-smooth
additive-smooth
```

Level 0 is direct CI and is not part of the GAM residual corpus:

```text
formula_class  = direct-ci
residual_key_x = NA_character_
residual_key_y = NA_character_
```

Conditional routing is:

```text
1 <= |S| <= 2  -> full-smooth
|S| > 2        -> additive-smooth
```

### Decision and alpha-distance contract

The pinned legacy replay threshold is `p >= alpha`. For every finite Phase 0
p-value, the structural census must verify that this threshold result equals
the explicit `deletes_edge` decision in the trace. It then records:

```text
reference_independent = deletes_edge
reference_decision    = independent | dependent
```

The canonical Phase 0 trace contains only finite p-values. A future non-finite
p-value is recorded as `reference_decision = nonfinite`,
`reference_independent = NA`, and fails the canonical Phase 1 gate rather than
being silently coerced.

Use these versioned formulas:

```text
signed_distance_from_alpha = reference_p_value - alpha
absolute_distance_from_alpha = abs(signed_distance_from_alpha)

signed_log_ratio_from_alpha =
    log(max(reference_p_value, p_floor) / alpha)

absolute_log_distance_from_alpha =
    abs(signed_log_ratio_from_alpha)
```

`p_floor = .Machine$double.xmin` for schema v1 and must be written to the
manifest.

The complete Stage A table is authenticated with
`full-cuda-ci-logical-census-v1`: hash the ordered field vector above, every
column in canonical row order, the dataset identity/dimensions, and `p_floor`
using portable metadata serialization. The frozen 351x48 value is:

```text
canonical_logical_census_hash = c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634
```

Canonical validation recomputes this hash, so any lineage, p-value, decision,
distance, order, or residual-key drift fails closed.

### Trace lineage and selected sepsets

Retain both `source_sequence_id` and `source_task_index` from the logical
trace. Do not assume that `deletion_trace.csv::source_sequence_id` is a logical
test identifier: the current Phase 0 deletion trace may use deletion-order
lineage.

`selected_sepset` is the explicit logical-trace deletion decision. Validate it
against the deletion trace using the canonical tuple:

```text
level, unordered(edge_x, edge_y), sorted S
```

There must be a one-to-one match between logical rows with
`deletes_edge = TRUE` and canonical deletion rows. Ambiguous, missing, or extra
matches fail closed.

Stage A canonical counts:

```text
logical_test_count                 = 240,489
conditional_logical_test_count     = 238,276
conditional_residual_request_count = 476,552
unique_conditional_S_count         = 8,634
```

## Stage B: Canonical Conditional Residual Requests

Expand every conditional logical test into `x|S` and `y|S`, then deduplicate
by a frozen key payload. Do not create `target|empty-S` keys.

The canonical request table records:

```text
residual_key_payload
residual_key_sha256
target
S_key
S_size
formula_class
same_S_group_id
same_S_group_size
request_multiplicity
first_logical_sequence_id
last_logical_sequence_id
first_level
last_level
```

### Residual key serialization v1

The UTF-8 payload has exactly these fields, in this order, separated by LF and
terminated by one final LF:

```text
schema_version=full-cuda-ci-residual-key-v1
dataset_sha256=971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7
n=<base-10 integer>
p=<base-10 integer>
target_index=<base-10 integer>
sorted_S=<comma-separated base-10 integers>
formula_class=<full-smooth|additive-smooth>
family=gaussian
link=identity
method=GCV.Cp
optimizer=mgcv-default
gamma=1
select=false
scale=mgcv-default
weights=none
offset=none
formula_semantics_version=kpcalg_regrXonS_v1
mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1
```

Rules:

- no locale-dependent formatting, whitespace padding, scientific notation,
  or platform newline conversion;
- `sorted_S` is strictly increasing and contains no duplicates;
- hash the raw UTF-8 payload bytes with SHA-256;
- one hash mapping to two payloads is a collision error;
- one payload mapping to two hashes is a serialization error.

`same_S_group_id` is the SHA-256 of this response-independent payload, using
the same byte rules:

```text
schema_version=full-cuda-ci-same-s-key-v1
dataset_sha256=971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7
n=<base-10 integer>
p=<base-10 integer>
sorted_S=<comma-separated base-10 integers>
formula_class=<full-smooth|additive-smooth>
family=gaussian
link=identity
method=GCV.Cp
optimizer=mgcv-default
gamma=1
select=false
scale=mgcv-default
weights=none
offset=none
formula_semantics_version=kpcalg_regrXonS_v1
mgcv_semantics_version=legacy-mgcv-gam-default-selection-v1
```

The global corpus hash is the SHA-256 of the sorted residual SHA-256 values
joined with LF and terminated by LF.

The frozen 351x48 contract produces:

```text
canonical_key_corpus_hash = b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa
```

Canonical Stage B gates:

```text
sum(request_multiplicity)                                  = 476,552
nrow(residual_requests)                                    = 110,617
canonical_global_unique_conditional_target_s_count         = 110,617
length(unique(same_S_group_id))                            = 8,634
canonical_key_corpus_hash                                  = b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa
```

The historical route metric is recorded only as provenance:

```text
s_affinity_executed_mgcv_fit_count = 273284
metric_role                        = historical_route_metric
hard_gate                          = false
provenance_status                  = no_hash_protected_route_trace
```

## Stage C: Legacy Layout Parity Subset

Before the full numerical census, prove that the offline one-target fit is
equivalent to the actual `regrXonS` two-target data layout.

For each parity case:

1. call the real `regrXonS(cbind(x, y), S_data)` and retain its residuals;
2. reproduce the exact legacy `data.frame(cbind(X, S))`, `x1...` naming, and
   formula construction to retain both `mgcv::gam` fit objects;
3. fit `x|S` and `y|S` independently through the offline census helper;
4. run the same legacy dCov gamma CI authority on both residual pairs.

The required deterministic case set covers:

```text
canonical |S| = 1
canonical |S| = 2
canonical |S| = 3
canonical large additive multi-penalty case
canonical conditional logical test nearest alpha
synthetic rank-deficient conditioning case
synthetic near-constant target/conditioner case
```

Compare:

```text
selected sp in canonical penalty order
residual vector
fitted vector
GCV/Cp score
EDF
downstream CI p-value
downstream independence decision
```

For the pinned R/mgcv environment, schema v1 requires exact hashes for numeric
residual and fitted vectors and exact canonicalized numeric values for selected
sp, GCV/Cp, EDF, and the downstream p-value. Decision equality is mandatory.
No implementation may silently relax this gate. If pinned mgcv produces a
documented last-bit layout difference, stop and amend the specification with
measured absolute/relative tolerances before continuing.

Feasibility probes on the pinned canonical environment (`R 4.4.1`, `mgcv 1.9.1`)
already produced exact equality for canonical `|S| = 1,2,3,7`, the canonical
conditional test nearest alpha, and synthetic rank-deficient and near-constant
cases. The committed parity test must reproduce this evidence; the probe does
not replace the test or final artifact.

## Stage D: Legacy mgcv Setup and Target-Fit Metadata

Fit each canonical residual key exactly once with the authoritative legacy
formula and default `mgcv::gam` selection semantics. No fixed-sp replacement,
approximate smoother, or CUDA path is allowed.

Execution may conservatively extract setup observations per target. The final
schema must separate response-independent same-S setup from target-specific fit
state.

### `same_s_setup_metadata` - 8,634 rows

Required fields include:

```text
same_S_group_id
S_key
S_size
formula_class
representative_residual_key_sha256
formula_semantics_version
model_matrix_nrow
model_matrix_ncol
model_matrix_hash
model_matrix_rank
model_matrix_condition
penalty_count
penalty_block_dimensions
penalty_ranks
penalty_offsets
penalty_hashes
penalty_nullity
constraint_dimensions
constraint_rank
constraint_nullspace_dimension
constraint_hash
H_dimensions
H_hash
weights_policy
offset_policy
smooth_classes
basis_dimensions
conditioning_rank
conditioning_condition
near_constant_conditioning_count
setup_fingerprint
mgcv_version
R_version
```

### `target_fit_metadata` - 110,617 rows

Required fields include:

```text
residual_key_sha256
same_S_group_id
setup_fingerprint
shard_id
target
fit_status
fit_error
fit_time_ms
formula
method
optimizer
family
link
selected_sp
selected_sp_names
selected_sp_hash
GCV_Cp_score
EDF
convergence_fields
warning_classes
warning_messages
coefficient_rank
coefficient_all_finite
fitted_all_finite
residual_all_finite
penalized_system_condition_at_selected_sp
target_sd
target_near_constant
coefficient_hash
fitted_hash
residual_hash
target_fit_fingerprint
```

### Phase 1 setup fingerprint contract

The setup fingerprint must be response-independent. The current generic
`fastkpc_setup_fingerprint()` cannot be reused unchanged because existing call
sites may set `input_p` from the local fit frame: the real two-target
`regrXonS` layout has `|S| + 2` columns while an offline one-target frame has
`|S| + 1`. That layout difference is not a setup difference.

Phase 1 uses `full-cuda-ci-same-s-setup-fingerprint-v1`. Its fixed UTF-8
payload contains, in order:

```text
schema_version=full-cuda-ci-same-s-setup-fingerprint-v1
dataset_sha256=971f1e0784817c6febfc84154f19540be02a92f6e4acad784dc7f92b979f1df7
same_S_group_id=<lowercase SHA-256>
n=351
canonical_input_p=48
formula_class=<full-smooth|additive-smooth>
family=gaussian
link=identity
method=GCV.Cp
optimizer=mgcv-default
gamma=1
select=false
scale=mgcv-default
model_matrix_hash=<metadata numeric hash v1>
penalty_hashes=<comma-separated hashes in mgcv penalty order>
penalty_offsets=<comma-separated base-10 integers>
constraint_hash=<metadata numeric hash v1>
H_hash=<metadata numeric hash v1 or NONE>
weights_policy=none
offset_policy=none
rank_metadata_hash=<metadata object hash v1>
```

Hash this payload with the same raw UTF-8 SHA-256 rule as residual keys.
Exclude target index, response values/hash, response name, local fit-frame
column count, selected sp, coefficients, fitted values, and residuals.

`metadata numeric hash v1` coerces numeric arrays to double, removes names and
dimnames, retains dimensions and list order, serializes the normalized object
with portable R serialization version 2, and hashes the returned raw bytes
with SHA-256. `metadata object hash v1` applies the same serialization/hash
rule to normalized integer/logical/list metadata. All hash schema versions are
recorded in the manifest.

Frame and shard-payload authentication use
`full-cuda-ci-frame-hash-v2`/`full-cuda-ci-shard-payload-hash-v2`. They retain
nested list/vector names because convergence and provenance semantics depend
on names such as `converged`, `outer.info`, `source`, and `value`; only matrix
dimnames are removed.

Within every `same_S_group_id`, all per-target setup observations must have
exactly one value for:

```text
model_matrix_hash
penalty_hashes
constraint_hash
setup_fingerprint
```

Any violation fails closed and prevents compression to the 8,634-row setup
table. The setup and target-fit tables join through `same_S_group_id` and
`setup_fingerprint`.

Feasibility probes across representative canonical `|S| = 1,2,3,7` groups and
multiple targets produced one exact serialized value per group for `X`,
penalty blocks, offsets, constraints, `H`, and setup rank. The committed
invariant tests and full census must reproduce this evidence.

Large vectors and matrices are not stored in the final artifact. Store exact
numeric hashes, dimensions, ranks, conditions, and scalar/list metadata.

## Numerical Risk Contract

All risk rules are versioned and written to the manifest.

### Thresholds and estimators

```text
risk_schema_version       = full-cuda-ci-risk-v1
near_constant_threshold   = sqrt(.Machine$double.eps)
rank_estimator             = lapack-svd-v1
rank_tolerance             = max(dim(A)) * max(singular_values(A)) *
                             .Machine$double.eps
condition_estimator        = lapack-svd-ratio-v1
high_condition_threshold   = 1e12
near_alpha_tau             = log(2)
p_floor                    = .Machine$double.xmin
```

For `lapack-svd-v1`, remove names/dimnames, coerce to double, and compute
singular values with `La.svd(A, nu = 0, nv = 0)$d`. Apply these rules in order:

1. if either dimension is zero, `rank = 0`, `condition = NA`, and bucket is
   `not_applicable_empty`;
2. if any matrix entry or singular value is non-finite, `rank = NA`,
   `condition = NA`, and bucket is `nonfinite_unknown`;
3. if the largest singular value is zero, `rank = 0`, `condition = Inf`, and
   bucket is `rank_deficient_inf`;
4. otherwise set `tol = max(dim(A)) * s_max * .Machine$double.eps` and
   `rank = sum(s > tol)`;
5. if `rank < min(dim(A))`, set `condition = Inf`; otherwise set
   `condition = s_max / s_min`.

For model, conditioning, and penalized-system matrices in this corpus,
rank-deficient means rank is below the expected column/system dimension, even
when a generic rectangular SVD would regard the row rank as complete.

Condition buckets:

```text
not_applicable_empty
finite_lt_1e4
finite_1e4_to_lt_1e8
finite_1e8_to_lt_1e12
finite_ge_1e12
rank_deficient_inf
nonfinite_unknown
```

Near-alpha buckets use `absolute_log_distance_from_alpha`:

```text
exact_boundary
le_1e_minus_12
le_1e_minus_9
le_1e_minus_6
le_1e_minus_3
le_log_1_01
le_log_1_1
le_log_2
farther
```

Assign near-alpha buckets from top to bottom; the first matching boundary
wins. `near_alpha = TRUE` exactly when
`absolute_log_distance_from_alpha <= log(2)`.

For near-constant classification, compute the ordinary sample standard
deviation after coercing the vector to double:

```text
sd_v = sqrt(sum((v - mean(v))^2) / (length(v) - 1))
near_constant = !is.finite(sd_v) || sd_v <= sqrt(.Machine$double.eps)
```

Vectors of length less than two have `sd_v = NA` and are near-constant. The
canonical dataset has no missing values; a missing value in a future corpus is
non-finite metadata and fails the canonical gate.

### Metadata policies

The final metadata gate independently recomputes every target-risk field from
the authenticated setup and target-fit tables. Stored risk flags and condition
buckets are outputs to verify, not self-validating evidence.
`near_constant_target` is derived from persisted `target_sd` and the frozen
threshold, and the stored `target_near_constant` boolean must agree exactly.

- Selected sp is stored in mgcv penalty order with original names preserved;
  canonical comparisons use the ordered numeric vector and a separate names
  vector.
- Capture warning class and message in emission order with
  `withCallingHandlers`; warnings are recorded and muffled after capture, not
  promoted to errors.
- Record convergence fields with provenance from the exact fit slots that
  exist (`converged`, `outer.info`, `mgcv.conv`). Do not invent a value when a
  slot is absent.
- Preserve explicit `NA`, `NaN`, and `Inf` classifications. Non-finite required
  metadata without an allowed risk/status classification fails closed.
- Compute `penalized_system_condition_at_selected_sp` in the constraint
  nullspace for `X'WX + P(sp) + H`, where `W` is the exact mgcv setup weight
  diagonal and each penalty block is embedded at its mgcv offset.
- Compute `penalty_nullity`, `constraint_rank`, and
  `constraint_nullspace_dimension` with the frozen rank rule.

Let `C` be the extracted constraint matrix and let `Z` be the deterministic
right-nullspace basis obtained from its SVD under the frozen rank tolerance.
Then:

```text
constraint_rank               = rank(C)
constraint_nullspace_dimension = ncol(X) - constraint_rank
P_unit                         = sum of all embedded penalty blocks at weight 1
penalty_nullity                = ncol(Z) - rank(Z' P_unit Z)
P(selected_sp)                 = sum_j selected_sp[j] * embedded_penalty[j]
A(selected_sp)                 = Z' (X' W X + P(selected_sp) + H) Z
```

Treat absent `C` as a zero-row matrix with `Z = I`; treat absent `H` as the
zero matrix. Penalty order and offsets are exactly those returned by the pinned
mgcv setup object.

### Row-level risk artifacts

Write:

```text
risk_cases.rds
risk_cases.csv
field_coverage.csv
```

Each risk row identifies a residual key or logical test and contains:

```text
high_condition
rank_deficient
near_constant_target
near_constant_conditioner
multi_penalty
near_alpha
mgcv_warning
mgcv_nonconverged
nonfinite_metadata
```

`field_coverage.csv` records, per metadata field, total rows, present rows,
finite rows where applicable, required status, and coverage ratio.

## Sharding and Restart

Sort the complete `residual_key_sha256` corpus lexicographically, assign a
one-based rank, and use:

```text
shard_id = (sorted_rank - 1) %% shard_count
```

Do not use implementation-dependent integer hashes or modulo arithmetic on R
integers.

Each shard writes atomically through a temporary file followed by rename:

```text
shards/shard_<id>.rds
shards/shard_<id>.summary.json
```

A completed shard is reusable only when its manifest matches:

```text
canonical_key_corpus_hash
canonical_logical_census_hash
expected_key_count_for_shard
expected_key_hash_for_shard
dataset_sha256
oracle_input_bundle_sha256
source_commit
R version
mgcv version
BLAS identity
LAPACK identity
BLAS thread count
formula semantics version
mgcv semantics version
risk-threshold config hash
metadata schema version
shard count
shard id
```

The completion summary also records SHA-256 hashes for the setup-observation,
target-fit, and target-risk tables plus a combined shard payload hash. Resume
and merge recompute these hashes after semantic validation. A finite scalar
change is corruption even when row counts and key joins remain valid.

Unix workers may use `parallel::mclapply`. The parent merges only completed
immutable shard rows. Duplicate shards, duplicate keys, missing keys, wrong
corpus hashes, stale configuration, and partial files fail closed.

## Artifact

Standard output:

```text
fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/
  manifest.json
  summary.json
  summary.md
  commands.txt
  environment.txt
  oracle_input_hashes.csv
  graph_agreement.csv
  sepset_agreement.csv
  n_edgetests.csv
  deletion_trace.csv
  first_divergence.json
  fallbacks.csv
  stage_timing.csv
  raw_runs.csv
  logical_ci_tests.rds
  logical_ci_tests.csv
  residual_requests.rds
  residual_requests.csv
  legacy_layout_parity_cases.rds
  legacy_layout_parity_results.csv
  same_s_setup_metadata.rds
  same_s_setup_metadata.csv
  target_fit_metadata.rds
  target_fit_metadata.csv
  risk_cases.rds
  risk_cases.csv
  field_coverage.csv
  counts_by_s_size.csv
  counts_by_penalty_count.csv
  counts_by_model_dimension.csv
  counts_by_condition_bucket.csv
  same_s_group_distribution.csv
  near_alpha_tests.csv
  unsupported_envelope.csv
  shards/
```

CSV files are for inspection. RDS files preserve integer, logical, ordered
numeric, and list-column data without delimiter loss.

## Fail-Closed Rules

The runner exits nonzero and writes `pass = false` when any of these occur:

```text
Phase 0 input, canonical contract, or inherited graph gate mismatch
logical trace count, layer plan, lineage, or decision mismatch
canonical logical-census hash mismatch
deletion/selected-sepset one-to-one mismatch
conditional request count, unique-key count, or S-group count mismatch
residual key serialization inconsistency or SHA-256 collision
legacy two-target versus offline one-target parity failure
same-S setup invariant violation
duplicate, missing, stale, partial, or wrong-corpus shard
shard payload or merged metadata authentication mismatch
missing or misjoined target/setup/risk per-key lineage
stored target-risk semantics disagree with independently recomputed risk
mgcv fit error
required non-finite metadata or output evidence without an exact risk status
incomplete required field coverage
unknown fallback or approximate backend count != 0
```

Every failed fit key remains in `target_fit_metadata` with its error and shard
id. No error row may be silently dropped during merge.

## Correctness and Exit Gates

Canonical structural hard gates:

```text
logical tests                                      = 240,489
conditional logical tests                          = 238,276
conditional residual requests                      = 476,552
canonical global unique conditional target|S       = 110,617
unique conditional S groups                        = 8,634
sum(request_multiplicity)                           = 476,552
```

Numerical hard gates:

```text
legacy layout parity cases pass                     = TRUE
same-S setup metadata rows                          = 8,634
target fit metadata rows                            = 110,617
setup observation rows before same-S compression    = 110,617
target risk rows before risk-case filtering          = 110,617
target/request, target/setup, and target/risk joins  = exact
target risk semantics                                = exact
target near-constant semantics                       = exact
authenticated metadata tables                       = exact
warning and non-finite classification                = exact
same-S invariant violations                         = 0
mgcv fit errors                                     = 0
required field coverage                             = 100%
```

Inherited Phase 0 hard gates:

```text
oracle_inherited_graph_gate                         = TRUE
oracle inherited SHD                                = 0
oracle inherited normalized sepsets identical       = TRUE
oracle inherited n.edgetests exact                  = TRUE
oracle inherited deletion trace identical           = TRUE
new_candidate_graph_gate                            = NOT_APPLICABLE
```

Historical route sanity metric, not a hard gate:

```text
s_affinity_executed_mgcv_fit_count                  = 273,284
```

The summary must include distributions by `|S|`, penalty count, model
dimension, rank/condition bucket, same-S group size, near-alpha distance, and
fit time weighted by `|S|` and penalty count.

## Approved Implementation Sequence

Implementation must proceed as six independently reviewable commits:

1. Structural census: Phase 0 loader, logical table, request expansion,
   canonical key serialization, deduplication, and structural count gates. No
   mgcv fits.
2. Legacy parity subset: metadata extractor plus real `regrXonS` layout parity
   cases.
3. Setup/fit schema: per-target extraction, same-S invariant validation, and
   the 8,634-row/110,617-row split tables.
4. Shard/restart qualification: interruption, resume, duplicate shard, wrong
   corpus, missing shard, and atomic-write tests.
5. Scaled dry runs: deterministic fixed-prefix corpus runs that measure memory,
   disk, warnings, throughput, and merge determinism.
6. Full census: run all 110,617 keys and generate the final Phase 1 artifact.

The complete mgcv census must not start before stages 1-5 pass.

## Tests

Focused tests must cover:

- complete Phase 0 input loading and independent inherited-gate validation;
- source lineage, exact decision recording, alpha formulas, and non-finite
  fail-closed behavior;
- level-0 direct-CI rows with `NA` residual keys;
- exact request multiplicity, canonical payload bytes, SHA-256, collision
  checks, and global deduplication;
- formula routing at `|S| = 1`, `2`, and `3` using the shared enum;
- real-layout parity for all required canonical and synthetic cases;
- setup/fit split and same-S invariant failures;
- rank-deficient, near-constant, high-condition, multi-penalty, near-alpha,
  warning, and nonconverged risk rows;
- shard interruption/resume, duplicate/missing/wrong-corpus rejection, and
  atomic writes;
- deterministic scaled merge and standard artifact schema.

The final 351x48 artifact is mandatory for Phase 1 completion.

## Non-Goals

Phase 1 does not:

- execute a new skeleton or claim a new candidate graph gate;
- change scheduler, replay order, residual values, or dCov authority;
- cache or accelerate production residual fits;
- define the final `PreparedSSetup` ABI for Phase 2;
- run fixed-sp C++/CUDA solves;
- relax the legacy oracle or numerical gates.
