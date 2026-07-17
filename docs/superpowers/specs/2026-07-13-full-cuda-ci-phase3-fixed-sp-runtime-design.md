# Full-CUDA CI Phase 3 Persistent Fixed-SP Runtime Design

## Status

Phase 0 through Phase 2 are complete. The authenticated Phase 2 artifact at
`fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/` contains all 8,634
canonical `PreparedSSetup` objects and all 110,617 canonical `TargetState`
rows. This design originally froze the Phase 3 runtime contract before
implementation planning; the status paragraphs below record later history and
the Phase 3C correction without rewriting that baseline.

Phase 3 may land in independently reviewable 3A, 3B, and 3C commits. The phase
is not complete until the stable path and both full artifacts pass. A
Cholesky-only subset, a repeated single-target bridge, or an inherited graph
gate is not Phase 3 completion.

The original implementation history started from `main` at or after `37ce594`;
`42ef3ef` remains provenance only. The current correction branch already
contains Phase 3C Tasks 1-4, the stacked-root Task 5 commit `8bd6142`, and its
follow-up fixes through `5b5cee0`. Preserve that history. This design amendment
reopens and corrects Task 5 with follow-up commits; it is not a branch-reset or
history-rewrite instruction.

The Phase 3C correction in this revision supersedes any earlier text that used
a CPU stacked augmented SVD as a route-specific numerical reference. Phase 2
`C_magic` remains the sole fitted/residual numerical oracle for every route.
Persistent independent penalty roots remain valid setup diagnostics and QR
inputs, but every planned or rerouted SVD solve uses the target-specific
aggregate-penalty root defined below.

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
  aggregate-penalty augmented SVD with deterministic rank truncation
```

The condition value is the authenticated Phase 1
`penalized_system_condition_at_selected_sp` for the oracle-sp campaign. A
request without an authenticated condition classification takes the
conservative SVD route. It never defaults to Cholesky.

The solve returns a leased, generation-checked device-resident residual batch
token. It does not implicitly copy residuals or fitted values to the host. An
explicit shadow-only materializer exists for oracle comparison. A later CUDA
dCov consumer can wait on the token event, register its own completion event,
and consume the device pointer without changing the residual ABI. The output
slot cannot be reused until the token lease is explicitly released and every
registered consumer event has completed.

Use the existing iteration and qualification corpora for development and
qualification. Run the full 8,634-setup, 110,617-target oracle-sp artifact only
after qualification passes, then run the full canonical logical-CI replay and
graph gate.

## Why This Architecture

At the original design baseline, the CUDA fixed-sp implementation was a
one-target prototype. Every target then:

- creates and destroys a cuSOLVER handle;
- performs thirteen CUDA allocations including factor workspace;
- uploads `X`, `Z`, and `XtX_null` again;
- uploads one assembled penalty and one RHS;
- synchronizes after system construction and again after residual creation;
- copies theta, coefficients, fitted values, residuals, and RSS to the host;
- frees every device buffer.

The same-setup native entry point at that baseline was a C++ `for` loop around
that single-target function and correctly reported
`true_batched_kernel = FALSE`.
It does not own persistent device state.

The canonical Phase 2 corpus has small, bounded setup dimensions and useful
same-S batches:

```text
rows per setup                         = 351
coefficient/nullspace dimension        = 10 through 64
targets per setup                      = 2 through 47
penalty count                          = 1, 3, 4, 5, 6, or 7
maximum stacked-QR logical row count   = 407
maximum aggregate-SVD workspace rows   = 415
```

The numerical risk is not confined to a few outliers:

```text
planned Cholesky route (< 1e8)         = 73,158 targets
planned augmented QR [1e8, 1e12)       =  4,210 targets
planned augmented SVD                  = 33,249 targets
coefficient-rank-deficient targets     =      1 target
nonfinite condition targets            =  1,162 targets
```

A persistent Cholesky-only runtime would therefore leave roughly thirty
percent of the corpus without a valid stable route. It cannot be promoted as
the compatible authority.

Phase 3C iteration evidence also rules out treating the mathematically more
accurate independent-root stacking as a legacy-compatibility oracle. It matched
a CPU stacked augmented SVD to about `4e-12` for the
target whose key starts `03de91f`, yet its fitted/residual result differed from
`C_magic` by `0.0594444`. The same formulation failed `C_magic` parity for
`14 / 67` iteration SVD targets, with maximum residual absolute error
`0.52305231`. A CPU upper-DPSTF2-compatible aggregate-root prototype matched
the LAPACK rank and pivot for `67 / 67` targets and reduced the maximum
`C_magic` error to about `9.03e-10`. This evidence selects legacy aggregate
penalty semantics; it does not promote the CPU prototype to a numerical
oracle.

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
oracle_projected_rhs           coefficient_dim x target_count, shadow only
oracle_nullspace_projected_rhs null_dim x target_count, shadow only
condition_bucket               one per target
coefficient_rank               one per target
planned_route                  one per target
identity and lineage hashes
```

`SP` row order must exactly match `penalty_sp_labels` and
`penalty_sp_indices_zero_based`. Targets from different PreparedSKeys cannot
share a batch. The two oracle RHS matrices are retained only to compare CUDA's
internally computed `X0' * Y` result; neither is forwarded through the
production native solve ABI.

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
penalty_offsets_zero_based
penalty_ranks
penalty_sp_indices_zero_based
penalty_sp_labels
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

The Phase 2 R representation may use one-based indices. The authenticated R
adapter converts offsets and smoothing-parameter indices exactly once at the
DTO boundary and emits zero-based integers to C++/CUDA. Native code never
performs a second base conversion. DTO validation reconstructs the first,
last, and multi-penalty blocks and requires exact equality with the Phase 2
oracle matrices; negative or out-of-range offsets/indices fail closed.

Phase 3 v1 also freezes the canonical identity penalty-to-SP mapping:

```text
penalty_sp_indices_zero_based = 0:(penalty_count - 1)
sp_mapping = NULL
min_sp = NULL
```

The R adapter and native host-view validator both enforce this invariant.
Numerical builders may index `SP` by penalty ordinal only after that check.
Any non-identity mapping fails closed and requires a later schema amendment.

Selected smoothing parameters satisfy:

```text
is.finite(SP) and SP >= 0
```

`sp = 0` is a legal fixed-parameter boundary. Negative and nonfinite values
are invalid.

Hash responsibility is split explicitly:

```text
R/catalog adapter:
  dataset and artifact lineage
  PreparedSKey and residual target key
  Y hash and oracle selected-SP hash

C++/CUDA runtime:
  DTO schema, shape, order, dtype, finite/range checks
  route metadata and device execution status
```

The native runtime must not claim that it validated a hash that was not
provided through its host view.

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
one user-provided cuBLAS workspace
one completion event pool
resource and timing counters
context generation
```

The Phase 3 v1 runner is single-threaded with respect to one context. Public
calls reject concurrent use. cuBLAS uses double precision without TF32 or
reduced-precision math.

Determinism is configured in code, not inherited from the environment. The
runtime sets and queries:

```text
cusolverDnSetDeterministicMode(..., CUSOLVER_DETERMINISTIC_RESULTS)
cublasSetMathMode(..., CUBLAS_PEDANTIC_MATH)
cublasSetAtomicsMode(..., CUBLAS_ATOMICS_NOT_ALLOWED)
```

Initialization order is fixed. Bind the non-default stream first, then install
the user-provided cuBLAS workspace:

```text
cublasSetStream(...)
cublasSetWorkspace(...)
cusolverDnSetStream(...)
```

Changing the stream requires reinstalling the cuBLAS workspace before any
operation. Runtime info records the queried deterministic/math/atomics modes,
workspace size and alignment, CUDA toolkit and driver versions, GPU compute
capability, and SM count. RSS, norm, and error reductions use fixed-order
trees and never unordered cross-block `atomicAdd` accumulation.

The context exposes an explicit reserve operation. The full runner reserves
the canonical maxima before the measured solve pass:

```text
n                      = 351
null_dim               = 64
target_count           = 47
penalty_count          = 7
augmented_rows         = 407
```

`augmented_rows` remains the caller-facing logical capacity for stacked QR; R
callers must not inflate it to guess an aggregate root rank. The C++ reserve
first merges every logical capacity with the context's existing capacities,
then computes, with checked integer addition:

```text
effective_stable_rows = max(merged.augmented_rows,
                            merged.n + merged.null_dim)
```

The within-capacity fast path must include
`effective_stable_rows <= stable_workspace.max_rows`; stable probe/growth checks
use `effective_stable_rows`, not `merged.augmented_rows`. The context retains
`merged.augmented_rows` as the logical QR capacity, while
`augmented_workspace_bytes` is computed from the internal stable capacity as
`sizeof(double) * stable_workspace.max_rows * stable_workspace.max_q`.
Canonical reserve therefore yields `max(407, 351 + 64) = 415` stable rows.
This makes declared SVD targets and Cholesky/QR targets rerouted from mixed or
true-batched batches executable without any caller observing or guessing a
device-produced rank.

The stable reserve also includes one `q x q` aggregate-penalty factor buffer,
`2q` DPSTF2 work values, and fixed-capacity per-target aggregate diagnostics.
For reserved target capacity `T` and coefficient/nullspace capacity `Q`, append
the following to the existing six stable integer arrays (`geqrf_info`,
`ormqr_info`, `qr_rank`, `qr_reroute`, `qr_finite_status`, and `svd_info`) in
both the device integer arena and its pinned-host mirror:

```text
aggregate_root_rank[T]
aggregate_factor_call_count[T]
aggregate_b_build_count[T]
aggregate_pivots[T * Q]              # target-major, fixed stride Q
```

The aggregate addition is exactly `T * (Q + 3)` integers; the complete stable
compact prefix is `6T + T(Q + 3) = T(Q + 9)` integers. With the existing
`3T + 1` legacy integer prefix and one shared stable `info`, the mirrored full
integer prefix is `T(Q + 12) + 2` integers. The per-target pivot slice is both
factor scratch and retained diagnostic storage; there is no extra hidden
`Q`-integer allocation.

Reserve device doubles in this order before the reusable stable workspace:
`sigma_max[T]`, `smallest_retained_sigma[T]`, and `aggregate_dstop[T]`. The
pinned-host status arena mirrors the full enlarged integer prefix first, rounds
its byte offset up to `alignof(double)`, and only then stores those same `3T`
doubles in the same order. Thus the host-double alignment is recomputed after
the `T(Q + 3)` aggregate integers, never from the old six-array boundary.
These buffers are reserved before warm-up and reused for every executed SVD
target. QR needs at most the logical `407` rows; SVD has the fixed internal
`n + q = 415` slots, so no rank D2H is needed to size a cuSOLVER call.

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
exact resident projected H matrix `P_H`
persistent individual penalty/H roots for QR and setup diagnostics
setup dimensions and ranks
setup H2D counters and bytes
handle generation
```

The handle is created once, used for all targets in that setup, and destroyed
after its logical tests have been consumed. Static setup values are uploaded
once. Target data must not be stored in the handle.

Individual penalty roots are created once per setup. For each projected
positive semidefinite penalty `P_j`, retain a root `R_j` satisfying:

```text
R_j' R_j = P_j
```

Use symmetric eigendecomposition and the Phase 1 frozen rank tolerance:

```text
rank_tol(A) = max(dim(A)) * max(singular_values(A)) * double_epsilon
```

Retain the exact Phase 2 penalty rank and fail if the independently derived
root rank disagrees. Build an analogous root for non-null `H`.

These persistent roots preserve the existing reconstruction/rank diagnostics
and supply the QR row bands. They are not the numerical root used by an SVD
target. The resident projected matrices `P_j` and `P_H` remain authoritative
inputs to each target-specific aggregate factorization.

When `H` is non-null, retain its exact projected `q x q` matrix `P_H = Z' H Z`
on the prepared handle in addition to the persistent `H` root. Aggregate SVD
construction starts from that exact resident matrix. It must not reconstruct
`P_H` as a crossproduct of the eigendecomposition-derived `H` root. When `H` is
null, `P_H` is the exact zero matrix by contract.

The existing test-only prepared/static shadow API makes this ownership directly
observable. `PreparedSStaticShadow` contains `has_H` and a column-major
`projected_H` vector of length `q * q`; the R
`C_fixed_sp_cuda_test_prepared_static_shadow` result exposes it as a `q x q`
matrix for non-null `H` and `NULL` otherwise. `PreparedSInfo` separately reports
`projected_H_test_shadow_d2h_count` and
`projected_H_test_shadow_d2h_bytes`. A successful non-null-H materialization
increments them by one and exactly `q * q * sizeof(double)` respectively; it
does not fold this observation into root-shadow counters. Production setup and
solve code never calls this API, and production aggregate matrix/root D2H
remains zero.

The prepared handle's static setup payload does not mix in target-dependent
outputs. At handle creation it is paired with one separate
`TransientResidualSlot` provisioned outside the solve hot path:

```text
coefficients, fitted, residual, and RSS buffers
solve completion event
optional registered consumer completion event
generation
lease state
```

The slot may be reused by that handle only after its current lease has been
released and any registered consumer event has completed.

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
slot generation
per-target planned/executed route, reroute reason, and solver status metadata
exclusive lease on one TransientResidualSlot
```

V1 permits one in-flight batch per handle. A solve attempted while a prior
token still owns the slot, or while its registered consumer event is
incomplete, fails before writing output with `ERR_OUTPUT_SLOT_BUSY`. Explicit
token release relinquishes the lease only after the consumer event is
complete. The next successful solve increments the slot generation; a
released token from the prior generation remains inspectable for compact
metadata but every device access or shadow materialization fails as
`STALE_TOKEN`.

Any solve failure before a valid token is returned restores the slot to
`FREE`; exception paths cannot strand a lease.

Every acquired slot is initialized to quiet NaN for the requested public
output region before route execution. Successful routes overwrite their
canonical columns. Non-OK target columns remain invalid, and the shadow
materializer returns them only as explicit R `NA_real_`; no error token can
expose bytes from a prior solve.

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
  route_metadata,
  output_mask
)

void release_fixed_sp_residual(residual_token)
void register_fixed_sp_consumer_event(residual_token, cuda_event)

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
fixed_sp_cuda_residual_release()
fixed_sp_cuda_residual_free()
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

`b` is production CUDA work. The solve uploads `Y` and `SP`, then computes
all RHS columns from resident `X0` with cuBLAS on the runtime stream. A Phase 2
CPU projected RHS may be compared as a shadow oracle, but it is not accepted
as a production solve input. Artifact diagnostics report
`rhs_authority = "cuda-x0-transpose-y"` and
`full_cuda_data_plane = TRUE`.

The Cholesky route solves `A theta = b` in double precision. The QR and SVD
routes do not solve these normal equations, but their penalty-row construction
is intentionally different.

The QR route uses the persistent independent roots:

```text
B_QR = [ X0
         sqrt(sp_1) R_1
         ...
         sqrt(sp_k) R_k
         R_H ]
```

Every target whose `executed_route` is `AUGMENTED_SVD`, including a declared
Cholesky/QR reroute, first forms the aggregate projected penalty in canonical
penalty order:

```text
P(sp) = P_H + sum_j sp_j P_j
```

It then applies the upper LAPACK `DPSTRF`/`DPSTF2` pivoted-Cholesky semantics
to obtain rank `r`, pivot permutation `Pi`, leading upper factor rows `U_r`,
and the unrepresented rank-revealed remainder `E_r`:

```text
Pi' P(sp) Pi = U_r' U_r + E_r
R_aggregate   = U_r Pi'
B_SVD         = [ X0
                  R_aggregate ]
```

`R_aggregate` is an `r x q` compact root whose columns are scattered back to
original canonical coefficient order. Therefore
`R_aggregate' R_aggregate = P(sp) - Pi E_r Pi'` in exact arithmetic, plus
floating-point factorization error. It is the DPSTRF rank-revealed approximation
selected by `dstop`, not equality with `P(sp)` when `r < q`. Physically, the SVD
workspace appends `q` root slots after `X0`, writes the compact root into the
first `r`, and zeros the remaining `q - r`. This fixed `n + q` layout is
mathematically `[X0; R_aggregate]`, keeps the existing deterministic SVD
implementation, and does not require an intervening rank D2H checkpoint. The
common augmented right-hand side is:

```text
c = [ weighted(y - offset)
      0 ]

theta = argmin ||B_route theta - c||_2
beta  = Z theta
fitted = X beta + offset
residual = y - fitted
```

The aggregate factorization freezes the default LAPACK stopping value exactly:

```text
unit_roundoff = std::numeric_limits<double>::epsilon() / 2
dstop         = q * unit_roundoff * max(diag(P(sp)))
```

The maximum is taken from the initial diagonal. Pivots initialize in canonical
coefficient order. At each upper-DPSTF2 step, update accumulated squared row
terms and candidate diagonals in increasing remaining position, choose the
first position attaining the maximum (including the first-pivot tie), apply the
upper-storage row/column segment swaps followed by the work and pivot swaps,
take the pivot square root, and update the remaining upper row in that fixed
order. Stop before a pivot `<= dstop` or a nonfinite pivot; the completed-step
count is `aggregate_penalty_root_rank`. No alternate epsilon, absolute-value
diagonal scale, tie rule, lower-factor order, or post-hoc rank tolerance is
permitted.

The augmented-SVD solve rank follows authoritative `C_magic` `fit_magic`
semantics, independently of the penalty-root rank:

```text
svd_rank_tol       = sqrt(std::numeric_limits<double>::epsilon())
svd_rank_threshold = sigma_max * svd_rank_tol
effective_rank     = count(sigma_i >= svd_rank_threshold)
```

Directions with `sigma_i < svd_rank_threshold` are truncated; equality is
retained. There is no row-count multiplier. The `gesvdj` convergence tolerance
(`1e-12` in the Phase 3C runtime) controls Jacobi iteration convergence only;
it is not the solve-rank tolerance. The aggregate-root CPU prototype reporting
about `9.03e-10` maximum `C_magic` error used this `C_magic` threshold.

CUDA builds `P(sp)` from resident projected matrices and device-resident `SP`,
factors it, scatters the compact root, and builds `B_SVD` entirely on the
runtime stream. The path performs no solve-time allocation, CPU fallback,
factor/root D2H transfer, or host-selected pivot/rank input. A deterministic
one-block kernel with at most 64 threads is acceptable for the canonical
`q <= 64` envelope. Device-produced rank/pivot values may be copied only with
the existing compact post-solve batch diagnostics; the host never feeds them
back into the solve. A CPU stacked independent-root augmented SVD may be
recorded as an internal implementation diagnostic, but it cannot supply
fitted/residual references, rank authority, pass/fail gates, or
`numeric_reference`.

The aggregate buffer is separate from `B_SVD` and retains the upper factor and
pivot vector after the first root emission. `gesvdj` overwrites `B_SVD` while
producing the retained `U`, singular values, and `V`. The existing one-step
coefficient correction therefore rebuilds `c` and re-emits the same
original-order aggregate root into `B_SVD`, then uses the retained SVD factors;
it does not reaggregate or refactor `P(sp)`. Exactly one aggregate factorization
and exactly two `B/c` builds are performed per executed SVD target. Device
per-target counters are authoritative: an executed SVD target has
`aggregate_factor_call_count = 1` and `aggregate_b_build_count = 2`; every
non-SVD target has `0` and `0`. The batch/global
`aggregate_penalty_factor_count` and `aggregate_svd_b_build_count` values are
recomputed sums of those vectors, never independent counters used as sole
lifecycle evidence.

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

For all planned Cholesky targets in one setup batch:

1. upload `Y` and `SP` matrices once;
2. compute all RHS columns as resident `X0' * Y` with one cuBLAS operation;
3. build all `A_target` matrices with one fused/strided kernel family;
4. construct device pointer arrays;
5. call `cusolverDnDpotrfBatched` once;
6. read the per-target factor info through one compact checkpoint and remove
   failed factors;
7. when at least one factor succeeded, call `cusolverDnDpotrsBatched` once,
   or one equivalent true batched solve;
8. read its scalar info through one compact solve checkpoint;
9. build beta, fitted, residual, and compact RSS/status outputs with batched
   kernels;
10. record one completion event.

This path may report `true_batched_kernel = TRUE` only when at least two
targets actually enter the batched factor/solve calls. A one-target call is a
native batch API call but not a true-batched kernel.

If `potrf` reports a non-positive pivot, that target's Cholesky output is
discarded and the target is explicitly rerouted to the CUDA SVD path. The
transition increments `stable_reroute_count`. It is not a CPU fallback and is
never silent.

`potrfBatched` owns one `infoArray` element per attempted target.
`potrsBatched` instead owns one scalar `info` for the whole API call and
supports `nrhs = 1` in this layout. A nonzero `potrs` info is a batch/API
failure: it fails the public solve and is never mapped to one target or
interpreted as a numerical reroute. Factor and solve info use distinct device
cells; `potrs` may never overwrite an unchecked `potrf` result. If no factor
succeeds, the runtime skips `potrsBatched` entirely.

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

1. aggregate `P_H + sum_j sp_j P_j` in reusable device workspace;
2. apply the deterministic upper-DPSTF2-compatible device factorization and
   emit `R_aggregate` in original coefficient order;
3. emit `[X0; R_aggregate; zero padding]` and its RHS in the fixed
   `n + q` reusable device workspace;
4. call a deterministic cuSOLVER double SVD path;
5. truncate only singular directions with
   `sigma_i < sigma_max * sqrt(double_epsilon)`;
6. form the first minimum-norm solution entirely on device;
7. rebuild `c` and re-emit the retained aggregate factor into `B_SVD` after
   `gesvdj` overwrites the first build, without another aggregate factorization;
8. apply the existing one-step correction with the retained SVD factors;
9. report aggregate-root rank/pivot separately from effective augmented-SVD
   rank and singular-value diagnostics.

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

Every target records separate planning and execution fields:

```text
planned_route
executed_route
reroute_reason
solver_status
```

The authenticated condition census freezes planned counts. Declared CUDA
stability reroutes are allowed in full artifacts, but they must conserve the
corpus exactly:

```text
executed_cholesky = planned_cholesky - cholesky_to_svd_count
executed_qr       = planned_qr - qr_to_svd_count
executed_svd      = planned_svd + cholesky_to_svd_count + qr_to_svd_count
```

Iteration and qualification currently require both reroute counts to be zero.
Full closure may accept a nonzero declared reroute only when route
conservation, solver status, numerical gates, zero decision flips, and SHD 0
all pass. No field named only `route_count` may stand for both meanings.
The only v1 nonempty reroute reasons are
`CHOLESKY_NON_POSITIVE_PIVOT` and `QR_RANK_GUARD_REJECTED`.

The batch reports both aggregate and per-target truth:

```text
native_batch_call
true_batched_kernel
true_batched_target_count
planned_cholesky_target_count
planned_qr_target_count
planned_svd_target_count
executed_cholesky_target_count
executed_qr_target_count
executed_svd_target_count
cholesky_to_svd_count
qr_to_svd_count
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
ERR_OUTPUT_SLOT_BUSY or STALE_TOKEN lifetime violation
schema or fingerprint mismatch
setup/target dimension mismatch
target key order mismatch
R-adapter y or selected-sp hash mismatch
workspace capacity exceeded after warm-up
CUDA allocation, stream, handle, or event failure
nonzero batched `potrs` scalar info
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

Residual `release` and residual `free` are distinct operations. `release`
ends the output-slot lease while preserving compact token metadata for stale
generation checks; `free` releases if needed, destroys the host token, and
clears the external pointer. A token with an incomplete registered consumer
event cannot relinquish its slot.

The runner sets and records one explicit device. Device changes during a run
fail. Multi-GPU execution requires a later design.

## Device-Resident Contract

The primary solve path:

- uploads setup data once per PreparedSKey;
- uploads only target `Y` and `SP` per batch;
- computes RHS columns on device from resident `X0`;
- leaves residuals on device;
- returns compact status and timing metadata only;
- never calls `cudaDeviceSynchronize` after every small kernel;
- uses one final completion event per batch when host status is required;
- permits one compact Cholesky-factor checkpoint and one scalar `potrs`
  checkpoint per affected public batch;
- for a batch containing declared QR targets, permits at most one additional
  compact QR checkpoint so the host can enqueue required CUDA SVD reroutes;
- keeps aggregate-penalty construction, pivoting, root emission, and SVD
  consumption on device without an intervening host checkpoint;
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
cusolver_deterministic_mode
cublas_math_mode
cublas_atomics_mode
cublas_user_workspace_installed
cublas_workspace_bytes and alignment
cuda_toolkit_version and driver_version
gpu_compute_capability and sm_count
```

Setup lifecycle:

```text
prepared_handle_create_count
prepared_handle_destroy_count
setup_h2d_upload_count
setup_h2d_bytes
penalty_root_build_count
penalty_root_rank_mismatch_count
projected_H_test_shadow_d2h_count
projected_H_test_shadow_d2h_bytes
```

The penalty-root fields describe only persistent individual penalty/H roots;
they are not aggregate SVD-root counters. The projected-H fields are explicitly
test-observer traffic and remain unchanged during production solves.

Solve lifecycle:

```text
batch_call_count
target_count
true_batched_batch_count
true_batched_target_count
planned_cholesky_target_count
planned_qr_target_count
planned_svd_target_count
executed_cholesky_target_count
executed_qr_target_count
executed_svd_target_count
cholesky_to_svd_count
qr_to_svd_count
stable_reroute_count
output_slot_acquire_count
output_slot_release_count
output_slot_busy_count
stale_token_reject_count
invalid_output_init_count
per_target_allocation_count_after_warmup
per_target_handle_create_count
unknown_fallback_count
cpu_fallback_count
approximate_backend_count
nonfinite_output_count
aggregate_penalty_factor_count
aggregate_svd_b_build_count
aggregate_penalty_root_rank per target
aggregate_penalty_root_pivot per target
aggregate_factor_call_count per target
aggregate_b_build_count per target
aggregate_dstop per target
effective_rank per target (augmented-system SVD rank)
```

For `T = target_count` and `Q = null_dim`, `DeviceResidualInfo` has these exact
aggregate shapes:

```text
aggregate_penalty_root_rank_native     int[T]
aggregate_factor_call_count            int[T]
aggregate_b_build_count                int[T]
aggregate_penalty_root_pivot_native    int[T * Q]
aggregate_dstop                        double[T]
```

Non-SVD device entries initialize to rank/pivot `-1`, count `0/0`, and quiet
NaN `dstop`. An executed SVD writes rank in `[0,Q]`, one full native zero-based
permutation in its fixed-stride pivot row, finite nonnegative `dstop`, and counts
`1/2`. At the R boundary, root rank and both count fields are integer vectors of
length `T`; rank is `NA_integer_` for non-SVD entries. `aggregate_dstop` is a
numeric vector of length `T` with `NA_real_` for non-SVD entries.
`aggregate_penalty_root_pivot` is a list of length `T`: each non-SVD entry is
`integer(0)`, and each executed-SVD entry is a length-`Q` permutation converted
exactly once to one-based canonical coefficient positions.

After all declared and rerouted SVD work is enqueued, the one SVD batch
checkpoint copies the contiguous enlarged integer diagnostics once and the
contiguous `3T` double diagnostics once into their pinned-host mirrors, then
waits once. No aggregate field triggers a target-level copy or wait. This
compact diagnostic copy is not an aggregate matrix/root copy, so
`aggregate_penalty_root_d2h_count` remains zero. A test-only CPU LAPACK
comparison may validate rank/pivot but may never provide production factor
metadata.

Artifact gates recompute and require:

```text
for every target i:
  executed_route[i] == AUGMENTED_SVD =>
    aggregate_factor_call_count[i] = 1 and aggregate_b_build_count[i] = 2
  executed_route[i] != AUGMENTED_SVD =>
    aggregate_factor_call_count[i] = 0 and aggregate_b_build_count[i] = 0

aggregate_penalty_factor_count = sum(aggregate_factor_call_count)
aggregate_svd_b_build_count    = sum(aggregate_b_build_count)
aggregate_penalty_factor_count = executed_svd_target_count
aggregate_svd_b_build_count    = 2 * executed_svd_target_count
```

Transfer and synchronization:

```text
target_h2d_count and bytes
rhs_device_build_count and bytes
rhs_authority and full_cuda_data_plane
implicit_residual_d2h_count and bytes
aggregate_penalty_root_d2h_count and bytes
shadow_materialize_count and bytes
batch_event_record_count
batch_event_wait_count
cholesky_factor_checkpoint_record_count
cholesky_factor_checkpoint_wait_count
cholesky_solve_checkpoint_record_count
cholesky_solve_checkpoint_wait_count
qr_checkpoint_record_count
qr_checkpoint_wait_count
svd_checkpoint_record_count
svd_checkpoint_wait_count
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
aggregate_penalty_root_d2h_count                  = 0
cuda_device_synchronize_count during solves       = 0
target_level_stable_sync_count                    = 0
unknown_fallback_count                            = 0
cpu_fallback_count                                = 0
approximate_backend_count                         = 0
cusolver deterministic mode                       = enabled
cuBLAS math mode                                   = pedantic
cuBLAS atomics mode                                = not allowed
cuBLAS user workspace installed                    = TRUE
full_cuda_data_plane                               = TRUE
rhs_authority                                      = cuda-x0-transpose-y
```

Within each execution session that computes at least one shard:

```text
runtime context creates                           = 1
runtime context destroys                          = 1
all prepared handles destroyed                    = TRUE
all residual tokens released                      = TRUE
all output-slot leases released                   = TRUE
```

A pure resume session that reuses an already complete artifact creates zero
CUDA contexts. An uninterrupted canonical full run therefore has one session
and one context. A restartable artifact may contain shards from multiple
cleanly closed sessions; its aggregate context-create count equals its
execution-session count, while setup uploads and prepared-handle creates
across accepted shards must still total exactly `8,634`. Shards referencing an
incomplete session are not reusable and must be recomputed.

## Numerical Validation Gates

The Phase 2 fixed-sp `C_magic` adapter remains the sole numerical oracle for
every route. Phase 3 compares coefficients, fitted values, residuals, and RSS
for every selected target. Every target record, including all planned or
rerouted SVD targets, must report `numeric_reference = "mgcv-fixed-sp"`. A CPU
stacked augmented SVD is diagnostic only and cannot replace any of these
comparisons.

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
environment must reproduce planned/executed route, reroute-reason, and
solver-status fields exactly and report zero
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

planned Cholesky targets = 73,158
planned QR targets       = 4,210
planned SVD targets      = 33,249
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
planned/executed route, reroute reason, solver status, hashes, and error
metrics; it does not store all residual vectors. It runs dCov parity over the
authenticated qualification logical-CI corpus.

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
- prove zero-based penalty offsets/SP indices reconstruct exact first, last,
  and multi-penalty matrices and reject out-of-range values;
- require the v1 identity penalty-to-SP mapping and reject permutations;
- accept synthetic `sp = 0` and reject negative/nonfinite smoothing values;
- reject wrong dataset/setup/target fingerprints;
- reject missing or duplicate Phase 1 condition metadata;
- verify canonical planned-route counts, executed-route conservation, and
  deterministic shard assignment;
- reject caller-forged aggregate pass flags.

### CUDA resource and ABI tests

- create/reserve/info/free context;
- create/info/free prepared handle;
- solve one and many targets through the same implementation;
- reject a second solve with `ERR_OUTPUT_SLOT_BUSY` while the first token owns
  the slot;
- release the first lease, solve again, and prove the first token is stale;
- prove no post-warm-up allocation or handle creation across released solves;
- call canonical reserve with logical `augmented_rows = 407` and require exact
  internal `augmented_workspace_bytes = 8 * 415 * 64`; exercise merged `n/q`
  growth and equal-reserve reuse against the internal
  `max(merged_augmented_rows, merged_n + merged_null_dim)` formula;
- for narrow `n = 1`, `augmented_rows = 1`, and `q` in `{63, 64}` reserves,
  require exact `augmented_workspace_bytes = 8 * (1 + q) * q`, then repeat each
  reserve and require no allocation, workspace query, or workspace growth;
- require the exact null-H prepared/static-shadow schema `X_null`, `gram`,
  `projected_penalties`, `projected_H`, with `projected_H = NULL`; preserve the
  other exact field values and the existing event-resource and prepared-info
  counter assertions, including zero projected-H test-shadow D2H deltas;
- reject freed, stale-generation, wrong-device, and wrong-PID handles;
- prove an incomplete registered consumer event prevents slot reuse;
- prove a non-OK token materializes only explicit NA after prior safe output;
- prove explicit shadow materialization is the only residual D2H route.

### Numerical unit tests

- safe single- and multi-target Cholesky;
- per-target `potrfBatched` info and scalar batch-level `potrsBatched` info;
- mixed target-specific multi-penalty SP matrices;
- CUDA-computed RHS parity against the Phase 2 shadow RHS;
- QR bucket setup;
- high-condition SVD setup;
- rank-deficient SVD setup;
- all 67 iteration SVD targets against `C_magic` at each of the four fitted and
  residual `< 1e-7` gates;
- aggregate-penalty root rank and pivot against test-only CPU LAPACK upper
  pivoted Cholesky, including a synthetic equal-pivot tie that selects the
  first remaining canonical position;
- a constrained synthetic `q = 3` non-null-H setup with
  `P_H = diag(1, 0.60*q*epsilon, 0.90*q*epsilon)` and zero target SP: both small
  directions are omitted by the `q*epsilon` eigendecomposition-derived
  `H_root`, but retained by the aggregate factor's `q*(epsilon/2)` stop. Require
  the prepared/static shadow to equal exact resident `P_H`, require its separate
  test-D2H delta to be one matrix, prove `crossprod(H_root) != P_H`, and require
  GPU aggregate rank/pivot to match exact-`P_H` LAPACK (`3`, `c(1,3,2)`) rather
  than the truncated-root reconstruction (`1`, canonical trailing order);
- explicit separation of aggregate-root rank from effective augmented-SVD
  rank;
- C_magic solve-rank truncation at
  `sigma_max * sqrt(double_epsilon)`, separately from `gesvdj` convergence;
- one aggregate factorization and two aggregate `B/c` emissions per executed
  SVD target, with the second emission reusing the retained factor; assert the
  per-target vectors are exactly `1/2` for SVD and `0/0` for non-SVD before
  checking their recomputed iteration `67/134` and qualification `2,064/4,128`
  sums;
- Cholesky-to-SVD and QR-to-SVD declared reroutes, each with exact per-target
  factor/build lifecycle counts of `1/2` for the rerouted target and `0/0` for
  every non-SVD target in that reroute regression;
- mixed-route and forced true-batch-reroute tests that reserve only their logical
  QR row capacity, still execute SVD reroutes within the internal `n + q`
  capacity, and perform no solve-time growth;
- target order preservation in mixed batches;
- exact planned/executed route, reroute-reason, and solver-status repeatability.

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
leased device token, CUDA-built RHS, and safe Cholesky path. High-risk targets
return an explicit `stable path not implemented` failure in 3A tests; they
never return a normal equation result. 3A is an implementation milestone, not
a phase exit.

### Phase 3B - True same-S multi-target Cholesky

Replace repeated single-target calls with fused system construction and true
batched factor/solve for Cholesky targets. Preserve canonical target order and
truthful per-target batching diagnostics. The single-target wrapper becomes a
batch-size-one adapter.

### Phase 3C - Stable augmented QR/SVD

Add one-time individual penalty roots for QR/diagnostics, augmented QR,
target-specific aggregate-penalty augmented SVD, deterministic DPSTF2 pivoting
and SVD rank handling, and mixed-route batches. Remove the 3A high-risk
unsupported status from the canonical corpus. Run iteration and qualification
gates with `C_magic` as the sole numerical oracle.

### Phase 3 full closure

Generate and authenticate both required full artifacts. Update `goal-5.6.md`
only after their independent validators pass.

## Exit Condition

Phase 3 is complete only when current evidence proves all of the following:

```text
all 8,634 PreparedSSetup objects consumed
all 110,617 TargetState rows solved
planned route counts exactly 73,158 / 4,210 / 33,249
executed route counts satisfy declared reroute conservation
every reroute has a reason and successful CUDA SVD status
every executed SVD uses a device-built DPSTF2-compatible aggregate root
every target numeric_reference = mgcv-fixed-sp
all outputs finite
numeric tolerances pass
qualification and full decision flips = 0
unknown/CPU/approximate fallback counts = 0
post-warm-up per-target allocation and handle creation = 0
setup upload count = 8,634
implicit residual D2H count = 0
all output-slot leases released
CUDA-built RHS authority and deterministic modes verified
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
