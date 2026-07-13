# Full-CUDA CI Phase 3 Persistent Fixed-SP Runtime Design

## Status

Phase 0 through Phase 2 are complete. The authenticated Phase 2 artifact at
`fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/` contains all 8,634
canonical `PreparedSSetup` objects and all 110,617 canonical `TargetState`
rows. This design freezes the Phase 3 runtime contract before implementation
planning.

Phase 3 may land in independently reviewable 3A, 3B, and 3C commits. The phase
is not complete until the stable path and both full artifacts pass. A
Cholesky-only subset, a repeated single-target bridge, or an inherited graph
gate is not Phase 3 completion.

## Decision

Build one process-local, single-GPU `CudaRuntimeContext` and one short-lived
`PreparedSGpuHandle` per canonical `PreparedSSetup`. The context owns the CUDA
stream, cuBLAS handle, cuSOLVER handle, reusable workspace arenas, and compact
status buffers. A prepared handle owns only response-independent setup buffers
and setup identity. A solve accepts one setup, many target columns, and one
target-specific smoothing-parameter vector per target.

The runtime has three declared double-precision solve routes:

```text
finite condition < 1e8:
  true batched normal-system Cholesky

finite condition >= 1e8 and < 1e12:
  augmented least-squares QR

condition >= 1e12, rank deficient, or nonfinite:
  augmented SVD with deterministic rank truncation
```

The condition value is the authenticated Phase 1
`penalized_system_condition_at_selected_sp` for the oracle-sp campaign. A
request without an authenticated condition classification takes the
conservative SVD route. It never defaults to Cholesky.

The solve returns a generation-checked device-resident residual batch token.
It does not implicitly copy residuals or fitted values to the host. An
explicit shadow-only materializer exists for oracle comparison. A later CUDA
dCov consumer can wait on the token event and consume the device pointer
without changing the residual ABI.

Use the existing iteration and qualification corpora for development and
qualification. Run the full 8,634-setup, 110,617-target oracle-sp artifact only
after qualification passes, then run the full canonical logical-CI replay and
graph gate.

## Why This Architecture

The current CUDA fixed-sp implementation is a one-target prototype. Every
target currently:

- creates and destroys a cuSOLVER handle;
- performs thirteen CUDA allocations including factor workspace;
- uploads `X`, `Z`, and `XtX_null` again;
- uploads one assembled penalty and one RHS;
- synchronizes after system construction and again after residual creation;
- copies theta, coefficients, fitted values, residuals, and RSS to the host;
- frees every device buffer.

The current same-setup native entry point is a C++ `for` loop around that
single-target function and correctly reports `true_batched_kernel = FALSE`.
It does not own persistent device state.

The canonical Phase 2 corpus has small, bounded setup dimensions and useful
same-S batches:

```text
rows per setup                         = 351
coefficient/nullspace dimension        = 10 through 64
targets per setup                      = 2 through 47
penalty count                          = 1, 3, 4, 5, 6, or 7
maximum augmented-system row count     = 407
```

The numerical risk is not confined to a few outliers:

```text
Cholesky route (< 1e8)                 = 73,158 targets
augmented QR route [1e8, 1e12)         =  4,210 targets
augmented SVD route                    = 33,249 targets
rank-deficient targets                 =  1,162 targets
```

A persistent Cholesky-only runtime would therefore leave roughly thirty
percent of the corpus without a valid stable route. It cannot be promoted as
the compatible authority.

## Alternatives Considered

### Process-global implicit setup cache

An implicit cache keyed by setup fingerprint would reduce the R API surface,
but it hides ownership and makes PID changes, fork safety, device selection,
artifact replacement, test isolation, and teardown difficult to prove. It is
rejected.

### Artifact-native monolithic executor

A single native command that loads all shards, schedules all targets, solves
them, runs dCov, and publishes the artifact could eventually reduce host
overhead. It would combine artifact authentication, scheduling, numerical
authority, and graph replay before any one boundary had independent parity
evidence. It is rejected for Phase 3.

### Shared runtime plus explicit prepared handles

This is the selected design. It follows the repository's existing external
pointer and finalizer pattern, exposes resource counts directly, allows one
setup and one target batch to be tested in isolation, and gives the future
dCov stage a stable device-residual interface.

## Phase Boundary

Phase 3 owns:

- authenticated consumption of Phase 0, Phase 1, and Phase 2 artifacts;
- a versioned host-to-native `PreparedSSetup` DTO;
- persistent stream, cuBLAS, and cuSOLVER resources;
- one-time setup uploads and reusable workspace;
- true batched Cholesky for the declared safe envelope;
- augmented QR and SVD stable routes;
- target-specific selected-sp and status handling;
- device-resident residual ownership;
- full fixed-sp numerical parity and full logical-CI shadow replay;
- resource-lifecycle and truthful batching diagnostics;
- restartable, authenticated Phase 3 artifacts.

Phase 3 does not own:

- smoothing-parameter selection on CUDA;
- rebuilding mgcv basis/setup semantics;
- replacing oracle-selected `sp`;
- a production device-pointer dCov implementation;
- multiple concurrent in-flight residual batches;
- multi-GPU scheduling;
- default-route promotion before both full artifacts pass.

Those remain later roadmap work.

## Authoritative Inputs

The Phase 3 runner opens these inputs as one authenticated lineage:

```text
Phase 0 oracle artifact
  manifest and summary
  logical_ci_trace.rds
  deletion trace
  adjacency and sepsets
  n.edgetests evidence

Phase 1 workload census artifact
  manifest and summary
  target_fit_metadata.rds
  risk_cases.rds
  condition and rank configuration

Phase 2 Prepared-S artifact
  manifest and summary
  prepared setup and target indexes
  all 64 setup/target shards
  iteration and qualification corpora

canonical normalized 351x48 dataset
```

The loader must call the completed Phase 1 and Phase 2 artifact validators. It
must independently verify:

```text
phase2_complete                    = TRUE
PreparedSSetup count               = 8,634
TargetState count                  = 110,617
Phase 2 shard count                = 64
canonical and selected corpus hash equality
dataset SHA-256 and dimensions
R, mgcv, provider, and formula lineage
Phase 0 logical trace and graph evidence hashes
```

Phase 3 joins Phase 1 target metadata to Phase 2 TargetState by canonical
`residual_key_sha256`. Missing, duplicate, or conflicting rows fail before GPU
initialization.

The runner must not deserialize the complete setup RDS merely to run a small
subset. It reads the iteration or qualification key set first, maps sorted
PreparedSKey rank to the existing 64-shard assignment, and loads only the
required shards. The full run streams the 64 shards in deterministic order.

## Host Catalog API

Add a read-only consumer boundary with this logical shape:

```text
catalog <- open_full_cuda_prepared_s_catalog(
  phase0_dir,
  phase1_dir,
  phase2_dir,
  data_path,
  require_full = TRUE
)

iterator <- iter_full_cuda_fixed_sp_batches(
  catalog,
  scope = "iteration" | "qualification" | "full"
)
```

Each iterator item contains exactly one PreparedSKey and:

```text
prepared_setup
target_state_rows
Y                              n x target_count
SP                             penalty_count x target_count
projected_rhs                  coefficient_dim x target_count
nullspace_projected_rhs        null_dim x target_count
condition_bucket               one per target
coefficient_rank               one per target
identity and lineage hashes
```

`SP` row order must exactly match `penalty_sp_labels` and
`penalty_sp_indices`. Targets from different PreparedSKeys cannot share a
batch.

The R loader authenticates RDS and list-column semantics. C++ does not parse
arbitrary R objects or recompute fingerprints from R serialization. The R
adapter emits a versioned, plain native DTO only after validation.

## Native DTO

Schema version:

```text
full-cuda-ci-prepared-s-native-dto-v1
```

Identity fields:

```text
dataset_sha256
prepared_s_key_sha256
same_S_group_id
phase1_setup_fingerprint
provider_fingerprint
semantic_fingerprint
representation_fingerprint
prepared_s_setup_schema_version
native_dto_schema_version
```

Static numeric fields:

```text
n
coefficient_dim
null_dim
penalty_count
X
constraint_mode
constraint_nullspace
gram_matrix
nullspace_gram_matrix
penalty_blocks
penalty_offsets
penalty_ranks
penalty_sp_indices
H
weights
weights_policy
offset
offset_policy
```

The adapter uses aliases instead of materializing duplicate identity objects:

```text
constraint_mode = identity:
  Z aliases identity; no p x p identity transfer

nullspace_gram_policy = alias-gram:
  nullspace Gram aliases gram_matrix

weights_policy = none-or-unit:
  no weight vector transfer

offset_policy = none-or-zero:
  no offset vector transfer
```

Unsupported policy values fail at handle creation. They cannot select a CPU
fallback. The canonical Phase 2 artifact must have zero unsupported setups.

## Runtime Components

### CudaRuntimeContext

One context exists per R process and device for a Phase 3 run. It owns:

```text
device id
creator PID
one non-default CUDA stream
one cuBLAS handle bound to the stream
one cuSOLVER handle bound to the stream
one reusable double workspace arena
one reusable integer/status arena
one reusable pointer arena
one completion event pool
resource and timing counters
context generation
```

The Phase 3 v1 runner is single-threaded with respect to one context. Public
calls reject concurrent use. cuBLAS uses double precision without TF32 or
reduced-precision math.

The context exposes an explicit reserve operation. The full runner reserves
the canonical maxima before the measured solve pass:

```text
n                      = 351
null_dim               = 64
target_count           = 47
penalty_count          = 7
augmented_rows         = 407
```

After reserve/warm-up, no solve may grow the workspace. A larger request fails
closed rather than allocating in the target hot path.

### PreparedSGpuHandle

One handle represents exactly one authenticated PreparedSKey. It owns:

```text
shared runtime context ownership
identity header and creator PID/device
X or weighted X
constraint nullspace when non-identity
X_null = X Z, or alias X under identity constraints
Gram/nullspace Gram
projected penalty components
projected H
penalty roots for augmented solves
setup dimensions and ranks
setup H2D counters and bytes
handle generation
```

The handle is created once, used for all targets in that setup, and destroyed
after its logical tests have been consumed. Static setup values are uploaded
once. Target data must not be stored in the handle.

Penalty roots are created once per setup. For each projected positive
semidefinite penalty `P_j`, retain a root `R_j` satisfying:

```text
R_j' R_j = P_j
```

Use symmetric eigendecomposition and the Phase 1 frozen rank tolerance:

```text
rank_tol(A) = max(dim(A)) * max(singular_values(A)) * double_epsilon
```

Retain the exact Phase 2 penalty rank and fail if the independently derived
root rank disagrees. Build an analogous root for non-null `H`.

### DeviceResidualBatch

The solve returns an external-pointer token containing:

```text
shared ownership of setup handle and context
device id
residual device pointer
optional fitted/coefficient device pointers
n and target_count
leading dimensions
target key order
completion event
owner generation
batch generation
per-target route and status metadata
```

The output buffers belong to the setup handle workspace. V1 permits one
in-flight batch per handle. A subsequent solve increments the generation and
invalidates older tokens. Every consumer checks the generation before use.

An explicit `materialize_for_shadow()` waits on the completion event and
copies requested outputs to host. Normal `solve_batch()` never returns numeric
residual or fitted vectors to R.

## Required Native API Shape

The internal C++ API should have this shape:

```text
shared_ptr<CudaRuntimeContext> create_fixed_sp_runtime(options)
void reserve_fixed_sp_runtime(context, capacities)
RuntimeInfo fixed_sp_runtime_info(context)

shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu(context, dto)
PreparedSInfo prepared_s_gpu_info(handle)

FixedSpBatchResult solve_fixed_sp_batch(
  handle,
  Y,
  SP,
  nullspace_rhs,
  route_metadata,
  output_mask
)

ShadowMaterializedBatch materialize_fixed_sp_shadow(
  residual_token,
  output_mask
)
```

R `.Call` wrappers return tagged external pointers with registered finalizers.
The minimum public R wrapper set is:

```text
fixed_sp_cuda_runtime_create()
fixed_sp_cuda_runtime_reserve()
fixed_sp_cuda_runtime_info()
fixed_sp_cuda_runtime_free()

fixed_sp_cuda_prepared_create()
fixed_sp_cuda_prepared_info()
fixed_sp_cuda_prepared_free()

fixed_sp_cuda_solve_batch()
fixed_sp_cuda_residual_info()
fixed_sp_cuda_materialize_shadow()
```

The old single-target entry point becomes a compatibility adapter that creates
or receives a prepared handle and calls `solve_batch()` with one target. It
must not retain an independent solver implementation.

## Numerical Formulation

For setup matrix `X`, constraint nullspace `Z`, selected smoothing vector
`sp`, penalty blocks `S_j`, and optional fixed penalty `H`, define:

```text
X0    = weighted(X Z)
P_j   = Z' S_j Z
P_H   = Z' H Z
A     = X0' X0 + sum_j sp_j P_j + P_H
b     = X0' weighted(y - offset)
```

The Cholesky route solves `A theta = b` in double precision. The stable routes
do not solve these normal equations. They solve the augmented system:

```text
B = [ X0
      sqrt(sp_1) R_1
      ...
      sqrt(sp_k) R_k
      R_H ]

c = [ weighted(y - offset)
      0 ]

theta = argmin ||B theta - c||_2
beta  = Z theta
fitted = X beta + offset
residual = y - fitted
```

All kernel inputs and outputs are double precision. No TF32, mixed precision,
iterative approximation, or host solve is permitted.

## Route Selection

The oracle-sp runner maps the Phase 1 target condition bucket to a route:

```text
finite_lt_1e4 or finite_1e4_to_lt_1e8:
  CHOLESKY_BATCHED

finite_1e8_to_lt_1e12:
  AUGMENTED_QR

finite_ge_1e12:
  AUGMENTED_SVD

rank_deficient_inf or nonfinite_unknown:
  AUGMENTED_SVD
```

`coefficient_rank < null_dim` always forces SVD. Missing or unauthenticated
route metadata also forces SVD.

The route thresholds are part of the artifact manifest and cannot be changed
by an environment variable during a qualifying or full run.

### Batched Cholesky

For all Cholesky targets in one setup batch:

1. upload `Y`, `SP`, and RHS matrices once;
2. build all `A_target` matrices with one fused/strided kernel family;
3. construct one device pointer array;
4. call `cusolverDnDpotrfBatched` once;
5. call `cusolverDnDpotrsBatched` once, or one equivalent true batched solve;
6. build beta, fitted, residual, and compact RSS/status outputs with batched
   kernels;
7. record one completion event.

This path may report `true_batched_kernel = TRUE` only when at least two
targets actually enter the batched factor/solve calls. A one-target call is a
native batch API call but not a true-batched kernel.

If `potrf` reports a non-positive pivot, that target's Cholesky output is
discarded and the target is explicitly rerouted to the CUDA SVD path. The
transition increments `stable_reroute_count`. It is not a CPU fallback and is
never silent.

### Augmented QR

For the finite `[1e8, 1e12)` full-rank bucket:

1. construct the augmented matrix and RHS in reusable device workspace;
2. call cuSOLVER double `geqrf`;
3. apply `Q'` with `ormqr`;
4. solve the triangular system with cuBLAS/cuSOLVER;
5. inspect the diagonal rank guard using the frozen tolerance;
6. reroute to SVD if the guard rejects the factorization.

The v1 QR implementation may process stable targets sequentially inside one
native batch call while reusing the same stream, handles, and workspace. Such
targets report `true_batched_kernel = FALSE`.

### Augmented SVD

For the high-condition, rank-deficient, rejected-QR, and unclassified cases:

1. construct the augmented matrix and RHS in reusable device workspace;
2. call a deterministic cuSOLVER double SVD path;
3. truncate singular directions at
   `max(m, q) * sigma_max * double_epsilon`;
4. form the minimum-norm solution entirely on device;
5. report effective rank and singular-value diagnostics.

The SVD implementation must use a deterministic sorted singular-value order.
It may process targets sequentially within the native batch call in v1 and
must report `true_batched_kernel = FALSE` for those targets.

## Mixed Batch Execution

A same-S target batch may contain all three routes. The runtime performs one
input upload and partitions target indices on device or in compact host
metadata. Route execution order is fixed:

```text
Cholesky -> QR -> SVD -> residual/fitted finalization
```

Output columns remain in the canonical input target order. Route partitioning
must not reorder the public result or the logical-CI replay.

The batch reports both aggregate and per-target truth:

```text
native_batch_call
true_batched_kernel
true_batched_target_count
cholesky_target_count
qr_target_count
svd_target_count
stable_reroute_count
```

An aggregate `true_batched_kernel = TRUE` means every successful target in the
public batch was covered by true batched numerical execution and the batch had
at least two targets. A mixed batch containing sequential QR or SVD targets
reports `true_batched_kernel = FALSE`, even when its Cholesky subset used a
batched factor/solve. `true_batched_target_count` and per-target route fields
report that partial batching without overstating the whole batch.

## Status and Error Contract

Versioned target statuses:

```text
OK_CHOLESKY_BATCHED
OK_CHOLESKY_SINGLE
OK_AUGMENTED_QR
OK_AUGMENTED_SVD
ERR_NONFINITE_INPUT
ERR_SP_SHAPE_OR_ORDER
ERR_ROUTE_METADATA
ERR_STABLE_PATH_NOT_IMPLEMENTED
ERR_QR_FAILED
ERR_SVD_FAILED
ERR_NONFINITE_OUTPUT
ERR_INTERNAL_CUDA
```

Only `OK_*` rows contain valid output. Any non-OK target fails the artifact.
`ERR_STABLE_PATH_NOT_IMPLEMENTED` is a Phase 3A milestone-only status used to
prove that high-risk targets fail closed before 3C exists. Its count must be
zero in the iteration gate after 3C, in qualification, and in both full
artifacts.

Batch-level errors throw before returning a token:

```text
wrong external-pointer tag or freed object
wrong PID or device
context/handle generation mismatch
schema or fingerprint mismatch
setup/target dimension mismatch
target key order mismatch
y or selected-sp hash mismatch
workspace capacity exceeded after warm-up
CUDA allocation, stream, handle, or event failure
```

There is no CPU result field, no approximate result field, and no
`allow_fallback` option in the Phase 3 authority API.

## Fork, Device, and Lifetime Safety

CUDA state is process-local. Context, prepared handle, and residual token each
record creator PID and device id. Every public call checks both. A handle
created before `fork()` is invalid in the child and fails before issuing a
CUDA call.

External-pointer finalizers are idempotent. Explicit free clears the external
pointer and increments its generation. Context destruction waits for owned
completion events only after child handles/tokens release shared ownership.

The runner sets and records one explicit device. Device changes during a run
fail. Multi-GPU execution requires a later design.

## Device-Resident Contract

The primary solve path:

- uploads setup data once per PreparedSKey;
- uploads only target `Y`, `SP`, and RHS per batch;
- leaves residuals on device;
- returns compact status and timing metadata only;
- never calls `cudaDeviceSynchronize` after every small kernel;
- uses one final completion event per batch when host status is required;
- for a batch containing declared QR targets, permits at most one additional
  compact QR checkpoint so the host can enqueue required CUDA SVD reroutes;
- never synchronizes or copies diagnostics once per target.

The shadow materializer is an explicit observer. It increments:

```text
shadow_materialize_call_count
shadow_materialize_target_count
shadow_d2h_bytes
```

The full numerical and graph artifacts may use it to compare every result with
the oracle. They must label the copy as shadow observation and cannot claim
that a production dCov consumer executed device-resident. The runtime ABI,
not the shadow observer, establishes that residuals can remain resident until
a future device-pointer dCov consumer is connected.

## Required Diagnostics

Runtime lifecycle:

```text
runtime_context_create_count
runtime_context_destroy_count
stream_create_count
stream_destroy_count
cublas_handle_create_count
cublas_handle_destroy_count
cusolver_handle_create_count
cusolver_handle_destroy_count
workspace_reserve_count
workspace_grow_count
workspace_bytes
```

Setup lifecycle:

```text
prepared_handle_create_count
prepared_handle_destroy_count
setup_h2d_upload_count
setup_h2d_bytes
penalty_root_build_count
penalty_root_rank_mismatch_count
```

Solve lifecycle:

```text
batch_call_count
target_count
true_batched_batch_count
true_batched_target_count
cholesky_target_count
qr_target_count
svd_target_count
stable_reroute_count
per_target_allocation_count_after_warmup
per_target_handle_create_count
unknown_fallback_count
cpu_fallback_count
approximate_backend_count
nonfinite_output_count
```

Transfer and synchronization:

```text
target_h2d_count and bytes
implicit_residual_d2h_count and bytes
shadow_materialize_count and bytes
batch_event_record_count
batch_event_wait_count
qr_checkpoint_record_count
qr_checkpoint_wait_count
target_level_stable_sync_count
cuda_device_synchronize_count
```

Timing:

```text
runtime_create_ms
runtime_reserve_ms
setup_create_ms
setup_h2d_ms
penalty_root_ms
target_h2d_ms
build_system_ms
cholesky_ms
qr_ms
svd_ms
residual_finalize_ms
shadow_materialize_ms
total_batch_ms
```

All counters are recomputed by the artifact gate from per-batch records where
possible. Caller-supplied aggregate booleans are not trusted.

## Operational Hard Gates

After the explicit canonical-capacity warm-up:

```text
stream_create_count during solves                 = 0
cublas_handle_create_count during solves          = 0
cusolver_handle_create_count during solves        = 0
workspace_grow_count during solves                = 0
per_target_allocation_count_after_warmup          = 0
per_target_handle_create_count                    = 0
implicit_residual_d2h_count                       = 0
cuda_device_synchronize_count during solves       = 0
target_level_stable_sync_count                    = 0
unknown_fallback_count                            = 0
cpu_fallback_count                                = 0
approximate_backend_count                         = 0
```

Across the full run:

```text
runtime context creates                           = 1
setup H2D uploads                                 = 8,634
prepared handle creates                           = 8,634
all prepared handles destroyed                    = TRUE
all residual tokens released                      = TRUE
```

## Numerical Validation Gates

The Phase 2 fixed-sp `C_magic` adapter remains the oracle. Phase 3 compares
coefficients, fitted values, residuals, and RSS for every selected target.

GPU operation ordering is different from the Phase 2 R/mgcv call, so exact
hash equality remains a reported metric but is not assumed. The initial hard
ceilings are deliberately no weaker than the current native CUDA fixed-sp
test envelope:

```text
residual max absolute difference       < 1e-7
residual relative L2 difference        < 1e-7
fitted max absolute difference         < 1e-7
fitted relative L2 difference          < 1e-7
all coefficients/fitted/residuals      finite
```

For vector reference `r` and candidate `c`, relative L2 is frozen as:

```text
sqrt(sum((c - r)^2)) / max(sqrt(sum(r^2)), 1e-300)
```

The qualification artifact records route- and condition-bucket maxima. A
condition-specific relaxation requires a spec amendment and cannot waive any
CI decision flip. Environment variables cannot alter tolerances in a
qualifying or full run.

For the qualification logical-CI corpus:

```text
legacy dCov max absolute p-value difference < 1e-10
legacy dCov decision flips                  = 0
near-alpha decision flips                   = 0
```

Repeated execution on the same declared GPU, CUDA, driver, and library
environment must reproduce route/status fields exactly and report zero
numeric drift between repeated Phase 3 runs.

## Development and Qualification Corpora

The existing authenticated corpora already cover all three routes and all
canonical penalty counts.

### Iteration gate

```text
PreparedSSetup groups   = 44
TargetState rows        = 270
logical dCov pairs      = 44

Cholesky targets        = 172
QR targets              = 31
SVD targets             = 67
```

This is the normal development loop. It must include resource lifecycle,
external-pointer misuse, mixed-route ordering, repeat solve, and explicit
shadow materialization tests.

### Extended qualification

```text
PreparedSSetup groups   = 2,061
TargetState rows        = 6,143
logical dCov pairs      = 3,808
near-alpha tests        = 1,478

Cholesky targets        = 3,889
QR targets              = 190
SVD targets             = 2,064
```

Qualification must pass before any full run. It is the gate for stable-path
coverage, all penalty counts, dimensions 10 through 64, rank-deficient cases,
near-alpha decisions, resource counters, and artifact restart behavior.

### Full oracle-sp corpus

```text
PreparedSSetup groups   = 8,634
TargetState rows        = 110,617

Cholesky targets        = 73,158
QR targets              = 4,210
SVD targets             = 33,249
```

The full run is last. It cannot be replaced by extrapolation from the
qualification corpus.

## Required Artifacts

### Fixed-sp CUDA oracle-sp artifact

Path:

```text
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/
```

Required payload:

```text
manifest.json
summary.json
commands.txt
environment.txt
input_hashes.csv
route_config.json
runtime_lifecycle.csv
setup_results.csv and .rds
target_parity.csv and .rds
risk_cases.csv and .rds
qualification_dcov_parity.csv and .rds
resource_metrics.csv
stage_timing.csv
fallbacks.csv
failures.csv
shards/
```

The artifact runs all 110,617 targets. It stores compact per-target parity,
route, status, hashes, and error metrics; it does not store all residual
vectors. It runs dCov parity over the authenticated qualification logical-CI
corpus.

### Full fixed-sp CUDA shadow artifact

Path:

```text
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/
```

Required payload:

```text
manifest.json
summary.json
commands.txt
environment.txt
input_hashes.csv
logical_ci_parity.csv and .rds
deletion_trace.csv
sepset_agreement.csv
n_edgetests.csv
adjacency.rds
first_divergence.json
runtime_lifecycle.csv
resource_metrics.csv
stage_timing.csv
fallbacks.csv
failures.csv
```

The runner executes candidate residuals for every target required by the
complete Phase 0 logical trace, computes candidate legacy dCov p-values, and
replays decisions in canonical `logical_sequence_id` order. It compares every
decision before asserting graph equality.

The pinned dCov comparator is the correctness-qualified legacy C++ Spectra
backend. The full artifact requires zero backend errors, zero Spectra
fallbacks, and zero p-value decision flips across all 240,489 logical tests.

Required graph fields:

```text
candidate_graph_gate             = TRUE
logical_test_count               = 240,489
edge_count                       = 110
oracle_edge_count                = 110
SHD                              = 0
normalized_sepsets_identical     = TRUE
n_edgetests_exact                = TRUE
deletion_trace_exact             = TRUE
decision_flip_count              = 0
first_divergence                 = NOT_APPLICABLE
```

## Sharding, Resume, and Publication

Use 64 deterministic shards. Sort `prepared_s_key_sha256` in radix order and
assign:

```text
shard_id = (sorted_rank - 1) %% 64
```

Each shard summary binds:

```text
Phase 0/1/2 manifest hashes
canonical PreparedSKey and TargetState corpus hashes
route configuration hash
runtime ABI and solver version
CUDA toolkit, driver, GPU model, and compute capability
R and package source commit
expected setup and target keys
payload SHA-256
```

Resume validates the complete identity before reusing a shard. Wrong corpus,
wrong route config, wrong device environment, missing pair, duplicate shard,
or payload mismatch fails closed.

Publish through a staging directory. Remove or move any prior completion
marker before publishing. Publish payload files first, `manifest.json`
penultimate, and `summary.json` last. A completed-artifact validator
recomputes semantic hashes, counts, route totals, resource totals, and all 64
shard pairs.

## Test Plan

### Non-CUDA contract tests

- authenticate and stream iteration/qualification/full batches;
- reject malformed DTO fields and reordered SP labels;
- reject wrong dataset/setup/target fingerprints;
- reject missing or duplicate Phase 1 condition metadata;
- verify canonical route counts and deterministic shard assignment;
- reject caller-forged aggregate pass flags.

### CUDA resource and ABI tests

- create/reserve/info/free context;
- create/info/free prepared handle;
- solve one and many targets through the same implementation;
- solve twice and prove no post-warm-up allocation or handle creation;
- reject freed, stale-generation, wrong-device, and wrong-PID handles;
- prove old residual tokens become stale after the next solve;
- prove explicit shadow materialization is the only residual D2H route.

### Numerical unit tests

- safe single- and multi-target Cholesky;
- mixed target-specific multi-penalty SP matrices;
- QR bucket setup;
- high-condition SVD setup;
- rank-deficient SVD setup;
- QR-to-SVD and Cholesky-to-SVD declared reroutes;
- target order preservation in mixed batches;
- exact route/status repeatability.

### Real-corpus gates

Run in order:

```text
clean CUDA native build
contract and resource tests
44-setup iteration corpus
2,061-setup qualification corpus
qualification dCov and near-alpha gate
restart/resume and publication tests
110,617-target oracle-sp artifact
full logical-CI and graph shadow artifact
git diff --check
```

Every CUDA test is gated by `FASTKPC_RUN_CUDA_TESTS=1`. The standard Phase 3
runner must record the exact command and device id.

## Staged Delivery

### Phase 3A - Persistent resources and safe single-target adapter

Deliver the catalog/DTO, runtime context, prepared handle, reserve/warm-up,
device token, and safe Cholesky path. High-risk targets return an explicit
`stable path not implemented` failure in 3A tests; they never return a normal
equation result. 3A is an implementation milestone, not a phase exit.

### Phase 3B - True same-S multi-target Cholesky

Replace repeated single-target calls with fused system construction and true
batched factor/solve for Cholesky targets. Preserve canonical target order and
truthful per-target batching diagnostics. The single-target wrapper becomes a
batch-size-one adapter.

### Phase 3C - Stable augmented QR/SVD

Add one-time penalty roots, augmented QR, augmented SVD, deterministic rank
handling, and mixed-route batches. Remove the 3A high-risk unsupported status
from the canonical corpus. Run iteration and qualification gates.

### Phase 3 full closure

Generate and authenticate both required full artifacts. Update `goal-5.6.md`
only after their independent validators pass.

## Exit Condition

Phase 3 is complete only when current evidence proves all of the following:

```text
all 8,634 PreparedSSetup objects consumed
all 110,617 TargetState rows solved
route counts exactly 73,158 / 4,210 / 33,249
all outputs finite
numeric tolerances pass
qualification and full decision flips = 0
unknown/CPU/approximate fallback counts = 0
post-warm-up per-target allocation and handle creation = 0
setup upload count = 8,634
implicit residual D2H count = 0
true_batched diagnostics are truthful
fixed_sp_cuda_oracle_sp_v1 completed and authenticated
fixed_sp_cuda_full_shadow_v1 completed and authenticated
edge_count = 110 / 110
SHD = 0
normalized sepsets identical
n.edgetests exact
deletion trace exact
```

Oracle-selected smoothing parameters remain authoritative after Phase 3.
CUDA smoothing selection begins in Phase 4 and later phases.
