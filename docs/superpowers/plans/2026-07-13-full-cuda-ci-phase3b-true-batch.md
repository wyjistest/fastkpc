# Full-CUDA CI Phase 3B True Same-S Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated safe single-target Cholesky calls with one same-S native batch that uploads target matrices once, builds all safe systems with fused kernels, executes true batched Cholesky, preserves canonical target order, and reports partial versus whole-batch batching truthfully.

**Architecture:** Extend the Phase 3A runtime and prepared handle; do not create a second solver. One public batch uploads `Y` and `SP` once, computes all RHS columns as resident `X0' * Y`, partitions canonical target indices into Cholesky and not-yet-implemented stable routes, gathers the safe subset into contiguous factor/RHS buffers, calls `cusolverDnDpotrfBatched` and `cusolverDnDpotrsBatched`, then scatters coefficients/fitted/residuals back to canonical columns. Stable targets remain explicit `ERR_STABLE_PATH_NOT_IMPLEMENTED` until Phase 3C.

**Tech Stack:** Phase 3A catalog/DTO/runtime, C++17, CUDA C++17, CUDA Runtime, cuBLAS, cuSOLVER batched Cholesky, Rcpp `.Call`, Phase 2 `C_magic` oracle.

---

## Preconditions

Start only after the complete Phase 3A plan passes and `goal-5.6.md` records
the Phase 3A milestone. Re-run the Phase 3A iteration gate before editing.

The implementation branch must contain `main` at or after `37ce594`, the
Phase 3 review-amendment commit, and the accepted Phase 3A commits. The
pre-plan production-code baseline remains `42ef3ef` for provenance only.

Phase 3B does not implement QR, SVD, penalty roots, qualification, or full
artifacts. It must preserve the Phase 3A fail-closed result for stable targets.

The authenticated iteration corpus fixes these Phase 3B counts:

```text
setup batches                         = 44
all-safe setup batches                = 5
mixed setup batches                   = 33
all-stable setup batches              = 6

safe Cholesky targets                 = 172
setup groups containing safe targets  = 38
true-batched Cholesky subgroups       = 26
true-batched successful targets       = 160
single Cholesky targets               = 12
stable-not-implemented targets        = 98

whole public batches allowed to report
true_batched_kernel = TRUE             = 5
```

## Files

Modify:

```text
fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp
fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp
fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu
fastkpc/src/r_api_cuda.cpp
fastkpc/R/cuda_native.R
fastkpc/R/full_cuda_ci_fixed_sp_runtime.R
goal-5.6.md
```

Create:

```text
fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3b_iteration.R
```

## Spec Coverage for This Subplan

```text
one same-S native batch API                         existing Phase 3A API
one H2D upload phase with Y/SP only                Tasks 1-3
one CUDA X0'Y RHS build per public batch           Task 2
fused target-specific system construction          Task 2
true batched potrf/potrs                           Task 3
batched beta/fitted/residual finalization           Task 4
canonical output order and mixed route truth        Task 5
real iteration route/resource/numeric gate          Task 6
clean build, regression, and roadmap record         Task 7
```

Stable QR/SVD coverage remains Phase 3C.

## Task 1: Extend Truthful Batch Diagnostics

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R`

- [ ] **Step 1: Write the failing diagnostic-schema test**

Create the test by sourcing the Phase 3 R files, opening the iteration catalog,
and selecting a setup with at least three Cholesky targets and more than one
penalty:

```r
source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP Phase 3B true batch\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = TRUE)
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "workload_census_351x48_v1"),
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "prepared_s_contract_v1"),
  file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
            "cancer_RD-causalDiscoveryInput.rds")
)
scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "iteration")
batches <- fastkpc_full_cuda_fixed_sp_batches(catalog, scope)
candidate <- which(vapply(batches, function(batch) {
  sum(batch$planned_route == "CHOLESKY_BATCHED") >= 3L &&
    length(batch$setup$penalty_blocks) > 1L
}, logical(1L)))[[1L]]
batch <- batches[[candidate]]
safe <- which(batch$planned_route == "CHOLESKY_BATCHED")
safe <- safe[[1L]]
batch$target_rows <- batch$target_rows[safe, , drop = FALSE]
batch$Y <- batch$Y[, safe, drop = FALSE]
batch$SP <- batch$SP[, safe, drop = FALSE]
batch$oracle_nullspace_rhs <-
  batch$oracle_nullspace_rhs[, safe, drop = FALSE]
batch$planned_route <- batch$planned_route[safe]
dto <- fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
native <- fastkpc_full_cuda_fixed_sp_native_batch(batch, dto)

runtime <- fixed_sp_cuda_runtime_create(0L)
on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
fixed_sp_cuda_runtime_reserve(runtime, 351L, 64L, 47L, 7L, 407L)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)
on.exit(try(fixed_sp_cuda_prepared_free(handle), silent = TRUE), add = TRUE)

token <- fixed_sp_cuda_solve_batch(
  handle, native$Y, native$SP, native$planned_route, native$target_keys
)
on.exit(try(fixed_sp_cuda_residual_free(token), silent = TRUE), add = TRUE)
info <- fixed_sp_cuda_residual_info(token)

required <- c(
  "native_batch_call", "true_batched_kernel",
  "true_batched_subgroup_count", "true_batched_attempted_target_count",
  "true_batched_target_count", "cholesky_single_target_count",
  "potrf_batched_call_count", "potrs_batched_call_count",
  "target_batch_h2d_call_count", "target_h2d_copy_count",
  "canonical_output_order_exact", "planned_route", "executed_route",
  "reroute_reason", "solver_status"
)
assert_true(length(setdiff(required, names(info))) == 0L,
            "Phase 3B diagnostic schema")

cat("PASS Phase 3B diagnostic schema\n")
```

- [ ] **Step 2: Run the test and verify missing diagnostics**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
```

Expected: FAIL because the Phase 3A token info lacks the Phase 3B fields.

- [ ] **Step 3: Add diagnostic fields with frozen meanings**

Extend `DeviceResidualInfo`:

```cpp
int batch_call_count = 0;
int true_batched_subgroup_count = 0;
int true_batched_attempted_target_count = 0;
int cholesky_single_target_count = 0;
int potrf_batched_call_count = 0;
int potrs_batched_call_count = 0;
int target_batch_h2d_call_count = 0;
int target_h2d_copy_count = 0;
std::size_t target_h2d_bytes = 0;
bool canonical_output_order_exact = false;
```

Freeze the meanings:

```text
true_batched_attempted_target_count:
  targets passed to potrfBatched, including a target later rejected by info

true_batched_target_count:
  attempted targets that completed potrf/potrs and have OK_CHOLESKY_BATCHED

true_batched_kernel:
  TRUE only if target_count >= 2 and every public target has
  OK_CHOLESKY_BATCHED

true_batched_subgroup_count:
  1 when a safe subgroup of at least two targets entered batched factor/solve,
  otherwise 0
```

Expose every field in `C_fixed_sp_cuda_residual_info`. Initialize them to zero
or false in the Phase 3A implementation.

- [ ] **Step 4: Build and verify the schema test passes**

Run the test. It may still use repeated single solves at this point, but the
new fields must exist and remain truthful (`true_batched_kernel = FALSE`).

- [ ] **Step 5: Commit diagnostic schema**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/r_api_cuda.cpp \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
git commit -m "diag: add truthful fixed-sp CUDA batch metrics"
```

## Task 2: Fuse Target Upload and System Construction

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R`

- [ ] **Step 1: Add pre-factor batch assertions**

Replace Task 1's `safe <- safe[[1L]]` line with the complete safe index vector
so the test submits all safe targets from the selected setup. After the solve,
assert:

```r
assert_true(info$native_batch_call, "one native batch call")
assert_true(info$batch_call_count == 1L, "one public solve call")
assert_true(info$target_batch_h2d_call_count == 1L,
            "one target batch upload phase")
assert_true(info$target_h2d_copy_count == 2L,
            "one Y and one SP copy")
assert_true(info$target_h2d_bytes ==
              8 * (length(native$Y) + length(native$SP)),
            "target H2D byte accounting")
assert_true(info$rhs_device_build_count == 1L &&
              identical(info$rhs_authority, "cuda-x0-transpose-y") &&
              isTRUE(info$full_cuda_data_plane),
            "one CUDA RHS build")
```

Add a temporary assertion that
`info$true_batched_subgroup_count == 0L`; Task 3 will replace it with `1L`.

- [ ] **Step 2: Run the test and verify upload-count failure**

Expected: FAIL because Phase 3A uploads and solves target columns one at a
time.

- [ ] **Step 3: Copy Y and SP once, then build RHS on CUDA**

In `solve_fixed_sp_batch()`:

```cpp
const std::size_t y_count =
  static_cast<std::size_t>(batch.n) * batch.target_count;
const std::size_t sp_count =
  static_cast<std::size_t>(batch.penalty_count) * batch.target_count;

check_cuda(cudaMemcpyAsync(
  d_Y, batch.Y, sizeof(double) * y_count,
  cudaMemcpyHostToDevice, context->stream
), "copy Phase 3B Y batch");
check_cuda(cudaMemcpyAsync(
  d_SP, batch.SP, sizeof(double) * sp_count,
  cudaMemcpyHostToDevice, context->stream
), "copy Phase 3B SP batch");

const double one = 1.0;
const double zero = 0.0;
check_cublas(cublasDgemm(
  context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
  batch.null_dim, batch.target_count, batch.n,
  &one, handle->d_X_null, batch.n, d_Y, batch.n,
  &zero, d_rhs, batch.null_dim
), "build Phase 3B CUDA RHS batch");
```

Set H2D counts/bytes from the two exact input sizes. Set
`rhs_device_build_count = 1`, `rhs_authority = "cuda-x0-transpose-y"`, and
`full_cuda_data_plane = TRUE`. Compare `d_rhs` with
`batch$oracle_nullspace_rhs` only through explicit shadow observation. Do not
issue a per-target input copy or accept a CPU RHS pointer in the native host
view.

- [ ] **Step 4: Add safe-index and fused system-build kernels**

Add device buffers in the reserved integer arena:

```text
d_safe_target_indices[target_capacity]
d_stable_target_indices[target_capacity]
d_potrf_info[target_capacity]
d_potrs_info[1]
```

Update the Phase 3B reserve formula before warm-up to
`int_required = 3 * targets + 1` for safe indices, stable indices,
per-target factor info, and the scalar solve info. This remains one reserve
allocation and cannot grow during solves.

Copy the compact route partition once. Add:

```cpp
__global__ void build_fixed_sp_systems_kernel(
    const double* gram,
    const double* projected_penalties,
    const double* SP,
    const int* safe_target_indices,
    int safe_count,
    int penalty_count,
    int q,
    double* systems) {
  const int matrix_index = blockIdx.z;
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (matrix_index >= safe_count || row >= q || col >= q) return;
  const int target = safe_target_indices[matrix_index];
  const std::size_t element = row + static_cast<std::size_t>(col) * q;
  double value = gram[element];
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    value += SP[penalty + static_cast<std::size_t>(target) * penalty_count] *
      projected_penalties[element +
        static_cast<std::size_t>(penalty) * q * q];
  }
  systems[element + static_cast<std::size_t>(matrix_index) * q * q] = value;
}
```

The kernel's ordinal SP lookup depends on the Phase 3A adapter invariant
`penalty_sp_indices_zero_based[penalty] == penalty`; handle creation must have
rejected every non-identity mapping.

Use a `16 x 16` block and `grid.z = safe_count`. Add a gather kernel that
copies canonical RHS columns into contiguous `d_theta` columns in safe-index
order.

Until Task 3, loop over the already-built contiguous systems and call the
Phase 3A single `potrf/potrs` operations without additional H2D copies or
allocations. Set each successful target to `OK_CHOLESKY_SINGLE` and set
`cholesky_single_target_count = safe_count`. This keeps the Task 2 commit
numerically valid while all true-batch counters remain zero.

- [ ] **Step 5: Run the test**

Expected: upload assertions PASS; numerical output remains Phase 3A repeated
single behavior until Task 3.

- [ ] **Step 6: Commit fused upload/build**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
git commit -m "perf: fuse fixed-sp target upload and system construction"
```

## Task 3: Execute True Batched Cholesky

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R`

The two test-only forced-info hooks declared below cross the native/R boundary.
Declare their runtime entry points in `mgcv_fixed_sp_runtime.hpp`, expose and
register the `.Call` shims in `r_api_cuda.cpp`, and keep dynamic symbols
disabled. These hooks are test controls only and must not receive production R
wrappers.

- [ ] **Step 1: Replace temporary assertions with true-batch gates**

Assert for the selected all-safe batch:

```r
assert_true(info$true_batched_subgroup_count == 1L,
            "one true-batched subgroup")
assert_true(info$true_batched_attempted_target_count == ncol(native$Y),
            "all targets attempted in batched Cholesky")
assert_true(info$true_batched_target_count == ncol(native$Y),
            "all targets completed in batched Cholesky")
assert_true(info$cholesky_single_target_count == 0L,
            "no repeated single Cholesky")
assert_true(isTRUE(info$true_batched_kernel),
            "all-safe multi-target batch is truly batched")
assert_true(all(info$solver_status == "OK_CHOLESKY_BATCHED"),
            "batched target statuses")
```

Add test-only `C_fixed_sp_cuda_test_force_next_potrf_info` and
`C_fixed_sp_cuda_test_force_next_potrs_info` hooks. First force every factor
info positive and prove the runtime skips `potrsBatched`. Then set the solve
scalar to `-1`, call the same public batch, and require a batch-level error
containing `Phase 3B batched potrs info`; no token may be returned and no
per-target reroute counter may change. Clear the hook immediately after the
assertion.

```r
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
.Call("C_fixed_sp_cuda_test_force_next_potrf_info",
      rep.int(1L, ncol(native$Y)), PACKAGE = "fastkpc_cuda")
all_factor_fail_token <- fixed_sp_cuda_solve_batch(
  handle, native$Y, native$SP, native$planned_route, native$target_keys
)
all_factor_fail_info <- fixed_sp_cuda_residual_info(all_factor_fail_token)
assert_true(all_factor_fail_info$potrf_batched_call_count == 1L &&
              all_factor_fail_info$potrs_batched_call_count == 0L &&
              all_factor_fail_info$cholesky_to_svd_count == ncol(native$Y) &&
              all(all_factor_fail_info$solver_status ==
                    "ERR_STABLE_PATH_NOT_IMPLEMENTED"),
            "all factor failures skip zero-sized potrs")
fixed_sp_cuda_residual_release(all_factor_fail_token)
fixed_sp_cuda_residual_free(all_factor_fail_token)
.Call("C_fixed_sp_cuda_test_force_next_potrf_info", integer(),
      PACKAGE = "fastkpc_cuda")
.Call("C_fixed_sp_cuda_test_force_next_potrs_info", -1L,
      PACKAGE = "fastkpc_cuda")
potrs_error <- tryCatch(
  fixed_sp_cuda_solve_batch(
    handle, native$Y, native$SP, native$planned_route, native$target_keys
  ),
  error = identity
)
assert_true(inherits(potrs_error, "error") &&
              grepl("Phase 3B batched potrs info",
                    conditionMessage(potrs_error), fixed = TRUE),
            "potrs scalar info is a batch failure")
.Call("C_fixed_sp_cuda_test_force_next_potrs_info", 0L,
      PACKAGE = "fastkpc_cuda")
token <- fixed_sp_cuda_solve_batch(
  handle, native$Y, native$SP, native$planned_route, native$target_keys
)
```

- [ ] **Step 2: Run and verify true-batch failure**

Expected: FAIL because the Phase 3A solver does not call batched cuSOLVER.

- [ ] **Step 3: Build device pointer arrays**

Add:

```cpp
__global__ void make_fixed_sp_pointer_arrays(
    double* systems,
    double* theta,
    int q,
    int count,
    double** system_ptrs,
    double** theta_ptrs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  system_ptrs[index] = systems + static_cast<std::size_t>(index) * q * q;
  theta_ptrs[index] = theta + static_cast<std::size_t>(index) * q;
}
```

Launch it once after fused system/RHS construction.

- [ ] **Step 4: Call batched factor and solve once**

For `safe_count >= 2`:

```cpp
check_cuda(cudaMemsetAsync(
  d_potrf_info, 0, sizeof(int) * safe_count, context->stream
), "zero Phase 3B batched info");
check_cusolver(cusolverDnDpotrfBatched(
  context->solver, CUBLAS_FILL_MODE_UPPER, q,
  d_system_ptrs, q, d_potrf_info, safe_count
), "Phase 3B batched potrf");
```

Copy the `safe_count` factor-info integers asynchronously to the preallocated
host info vector, record the handle event, and wait on that event. Mark failed
factors `ERR_STABLE_PATH_NOT_IMPLEMENTED`, set
`reroute_reason = "CHOLESKY_NON_POSITIVE_PIVOT"`, increment
`cholesky_to_svd_count` and `stable_reroute_count`, and build compact
system/theta pointer arrays for only
the successful factors. Then call:

```cpp
check_cuda(cudaMemsetAsync(
  d_potrs_info, 0, sizeof(int), context->stream
), "zero Phase 3B potrs scalar info");
check_cusolver(cusolverDnDpotrsBatched(
  context->solver, CUBLAS_FILL_MODE_UPPER, q, 1,
  d_success_system_ptrs, q, d_success_theta_ptrs, q,
  d_potrs_info, successful_factor_count
), "Phase 3B batched potrs");
```

Call that block only when `successful_factor_count > 0`. When every factor
fails, issue no zero-sized `potrsBatched` call: Phase 3B returns the declared
stable-not-implemented rows, while Phase 3C proceeds directly to the SVD
reroute phase for those targets.

Reserve these distinct status buffers:

```text
d_potrf_info[target_capacity]  # one factor status per target
d_potrs_info[1]                # one scalar API status for the whole solve
```

For `safe_count == 1`, retain the Phase 3A single `potrf/potrs` path and status
`OK_CHOLESKY_SINGLE`, set `executed_route = "CHOLESKY_BATCHED"`, and keep the
reroute reason empty. For `safe_count == 0`, issue no factorization.

- [ ] **Step 5: Interpret factor and scalar solve status correctly**

After `potrsBatched`, copy exactly one scalar `d_potrs_info` value and
record/wait on the handle event a second time. If it is nonzero, fail the
entire public solve as a cuSOLVER API/batch error; do not assign it to a
target, do not reroute any target, and do not return a partially valid token.
The error path restores the transient slot to `FREE` before throwing so a
subsequent valid solve can proceed.
Successful targets from an original `safe_count >= 2` become
`OK_CHOLESKY_BATCHED` with `executed_route = "CHOLESKY_BATCHED"` and an empty
reroute reason. Record two compact event waits and zero
`cudaDeviceSynchronize` calls. Only positive per-target `potrf_info[i]`
declares a Cholesky-to-SVD reroute. Do not expose any failed output.

- [ ] **Step 6: Run the true-batch test**

Expected: diagnostic/status assertions PASS. Numerical parity is added in
Task 4.

- [ ] **Step 7: Commit batched factor/solve**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp \
  fastkpc/src/r_api_cuda.cpp \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
git commit -m "perf: add true batched fixed-sp Cholesky"
```

## Task 4: Batch Coefficients, Fitted Values, and Residuals

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R`

Task 2 already preserved numerical parity with repeated per-target output
finalization, so Task 4's RED gate must distinguish that bridge from true
batched output work. Add and expose these truthful per-token diagnostics:

```text
coefficient_batch_finalize_call_count
fitted_batch_finalize_call_count
residual_rss_batch_finalize_call_count
per_target_output_finalize_call_count
batch_output_finalized_target_count
```

For the all-safe test requesting every output, require the three batch call
counts to equal one, the per-target count to equal zero, and the finalized
target count to equal the number of successful batched targets. The existing
Task 3 implementation must report zero batch-finalization calls before this
task, providing the RED failure without weakening the already-established
numeric parity gate.

- [ ] **Step 1: Add all-target numerical parity assertions**

Materialize the batch and compare every column with the Phase 2 oracle:

```r
shadow <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("coefficients", "fitted", "residuals", "rss")
)
relative_l2 <- function(candidate, reference) {
  sqrt(sum((candidate - reference)^2)) /
    max(sqrt(sum(reference^2)), 1e-300)
}

for (target_index in seq_len(ncol(native$Y))) {
  materialized <- fastkpc_full_cuda_materialize_target_state(
    batch$target_rows[target_index, , drop = FALSE],
    catalog$data, batch$setup$dataset_sha256
  )
  oracle <- fastkpc_mgcv_magic_fixed_sp_from_prepared(
    batch$setup, materialized
  )
  assert_true(max(abs(
    shadow$residuals[, target_index] - oracle$residuals
  )) < 1e-7, "batched residual max abs parity")
  assert_true(relative_l2(
    shadow$residuals[, target_index], oracle$residuals
  ) < 1e-7, "batched residual relative L2 parity")
  assert_true(max(abs(
    shadow$fitted[, target_index] - oracle$fitted
  )) < 1e-7, "batched fitted max abs parity")
  assert_true(relative_l2(
    shadow$fitted[, target_index], oracle$fitted
  ) < 1e-7, "batched fitted relative L2 parity")
}
```

- [ ] **Step 2: Run and verify batch-finalization diagnostic failure**

Expected: FAIL because Task 3 still launches per-target output operations and
does not report batched output finalization.

- [ ] **Step 3: Add canonical-index finalization kernels**

Add one kernel family with a two-dimensional grid over row/coefficient and
safe position. For each safe position, look up the canonical target index and:

```text
theta_safe -> beta_canonical[:, target]
X * beta_canonical[:, target] -> fitted_canonical[:, target]
Y[:, target] - fitted -> residual_canonical[:, target]
residual^2 reduction -> rss_canonical[target]
```

Use the prepared handle's canonical output buffers. Do not allocate or return
a safe-subgroup-only output matrix.

- [ ] **Step 4: Make RSS deterministic**

Use one block per target and a fixed binary-tree reduction order in shared
memory. Do not use cross-block `atomicAdd`. Record one RSS value in canonical
target order.

- [ ] **Step 5: Add and run deterministic repeat assertions**

After materializing the first solve, compute:

```r
first_hashes <- c(
  coefficients = fastkpc_full_cuda_census_metadata_hash(shadow$coefficients),
  fitted = fastkpc_full_cuda_census_metadata_hash(shadow$fitted),
  residuals = fastkpc_full_cuda_census_metadata_hash(shadow$residuals),
  rss = fastkpc_full_cuda_census_metadata_hash(shadow$rss)
)
fixed_sp_cuda_residual_release(token)
fixed_sp_cuda_residual_free(token)
token <- fixed_sp_cuda_solve_batch(
  handle, native$Y, native$SP, native$planned_route, native$target_keys
)
shadow_repeat <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("coefficients", "fitted", "residuals", "rss")
)
repeat_hashes <- c(
  coefficients = fastkpc_full_cuda_census_metadata_hash(
    shadow_repeat$coefficients
  ),
  fitted = fastkpc_full_cuda_census_metadata_hash(shadow_repeat$fitted),
  residuals = fastkpc_full_cuda_census_metadata_hash(
    shadow_repeat$residuals
  ),
  rss = fastkpc_full_cuda_census_metadata_hash(shadow_repeat$rss)
)
assert_true(identical(first_hashes, repeat_hashes),
            "same-environment batched output hashes")
```

Run the true-batch test. Expected: PASS with exact repeat hashes.

- [ ] **Step 6: Commit batched output finalization**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/r_api_cuda.cpp \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
git commit -m "feat: finalize fixed-sp CUDA batch outputs in canonical order"
```

## Task 5: Support Mixed Cholesky and Stable-Error Batches Truthfully

**Files:**
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R`
- Modify if the gate exposes a defect: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`

- [ ] **Step 1: Write the mixed-batch regression test**

Select an iteration setup containing at least two Cholesky targets and at least
one stable target. Submit the complete batch without reordering. Assert:

```r
info <- fixed_sp_cuda_residual_info(token)
shadow <- fixed_sp_cuda_materialize_shadow(
  token, outputs = c("fitted", "residuals")
)
safe <- which(native$planned_route == "CHOLESKY_BATCHED")
stable <- which(native$planned_route != "CHOLESKY_BATCHED")

assert_true(info$true_batched_subgroup_count == 1L,
            "mixed batch has one batched safe subgroup")
assert_true(info$true_batched_target_count == length(safe),
            "mixed safe target count")
assert_true(!isTRUE(info$true_batched_kernel),
            "mixed batch must not claim whole-batch true batching")
assert_true(all(info$solver_status[safe] == "OK_CHOLESKY_BATCHED"),
            "mixed safe statuses")
assert_true(all(info$solver_status[stable] ==
                  "ERR_STABLE_PATH_NOT_IMPLEMENTED"),
            "mixed stable statuses")
assert_true(isTRUE(info$canonical_output_order_exact),
            "mixed canonical order")
assert_true(all(is.finite(shadow$residuals[, safe, drop = FALSE])),
            "safe mixed residuals are finite")
assert_true(all(is.na(shadow$residuals[, stable, drop = FALSE])),
            "invalid stable outputs are explicit NA in shadow materialization")
```

The test must also compare every safe column to its Phase 2 oracle and verify
that `info$target_keys` exactly equals `native$target_keys`.

- [ ] **Step 2: Run the mixed-order/status characterization gate**

Task 2 through Task 4 already initialize invalid public outputs and scatter the
safe subgroup by canonical index, so this regression test may pass on its
first run. Record that baseline result. Modify the runtime only if the test
exposes a concrete mixed-order, status, or masking defect; do not manufacture
a RED failure by changing already-correct behavior.

- [ ] **Step 3: Preserve Phase 3A invalid-output initialization**

Retain the Phase 3A batch-start quiet-NaN initialization for coefficients,
fitted, residuals, and RSS across all public targets. Only successful routes
overwrite their canonical columns. Shadow materialization continues to
convert non-OK target columns to R `NA_real_`. The mixed-batch test is a
regression gate for this earlier contract, not the first implementation of it.

- [ ] **Step 4: Preserve canonical metadata order**

Store target keys, planned/executed routes, reroute reasons, and solver
statuses in public input order. Safe/stable
partition arrays are internal only. Set:

```cpp
info.canonical_output_order_exact = true;
info.true_batched_kernel =
  batch.target_count >= 2 &&
  std::all_of(info.solver_statuses.begin(), info.solver_statuses.end(),
              [](FixedSpStatus value) {
                return value == FixedSpStatus::OkCholeskyBatched;
              });
```

- [ ] **Step 5: Run mixed and all-safe tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
```

Expected: both PASS.

- [ ] **Step 6: Commit mixed-batch semantics**

```bash
git add fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
# Add mgcv_fixed_sp_runtime.cu only if the characterization gate required a fix.
git commit -m "test: gate truthful mixed fixed-sp batch semantics"
```

## Task 6: Gate Phase 3B on the Real Iteration Corpus

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3b_iteration.R`

- [ ] **Step 1: Write the failing Phase 3B iteration gate**

Create a runner test that calls one public solve per each of the 44 setup
batches and recomputes this summary from per-target/per-batch records:

```r
expected <- list(
  setup_count = 44L,
  target_count = 270L,
  batch_call_count = 44L,
  all_safe_batch_count = 5L,
  mixed_batch_count = 33L,
  all_stable_batch_count = 6L,
  cholesky_ok_count = 172L,
  true_batched_subgroup_count = 26L,
  true_batched_target_count = 160L,
  cholesky_single_target_count = 12L,
  whole_batch_true_batched_count = 5L,
  stable_not_implemented_count = 98L,
  setup_h2d_upload_count = 44L,
  target_batch_h2d_call_count = 44L,
  target_h2d_copy_count = 88L,
  rhs_device_build_count = 44L,
  full_cuda_data_plane = TRUE,
  invalid_output_init_count = 44L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L
)
```

Also require all 172 safe targets to pass residual/fitted max-absolute and
relative-L2 `< 1e-7` against the Phase 2 oracle.

- [ ] **Step 2: Run and verify missing runner failure**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3b_iteration.R
```

Expected: FAIL with missing
`fastkpc_run_full_cuda_fixed_sp_phase3b_iteration`.

- [ ] **Step 3: Implement one-batch-per-setup iteration runner**

Add `fastkpc_run_full_cuda_fixed_sp_phase3b_iteration()` that:

```text
opens/authenticates the catalog once
creates/reserves one runtime once
for each PreparedSKey in radix order:
  creates one prepared handle
  calls fixed_sp_cuda_solve_batch once with the complete target batch
  records info before shadow materialization
  explicitly materializes only for oracle comparison
  appends per-target numeric/status rows and one batch metric row
  releases token and handle
recomputes summary from those rows
```

Do not derive expected counts from the implementation's aggregate summary.
The test compares recomputed values to the frozen constants above.

- [ ] **Step 4: Run Phase 3B iteration gate**

Expected: PASS with exact counts and no resource/fallback violations.

- [ ] **Step 5: Commit real-corpus Phase 3B gate**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3b_iteration.R
git commit -m "test: gate true fixed-sp CUDA batches on iteration corpus"
```

## Task 7: Phase 3B Verification and Roadmap Record

**Files:**
- Modify: `goal-5.6.md`

- [ ] **Step 1: Clean-build the CUDA library**

```bash
rm -f fastkpc/build/*.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
```

Expected: clean build exits zero.

- [ ] **Step 2: Run Phase 3A regressions**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3a_solve.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
```

Expected: all PASS.

- [ ] **Step 3: Run all Phase 3B tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3b_iteration.R
```

Expected: all PASS.

- [ ] **Step 4: Run existing CUDA regression tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
```

Expected: all PASS. The legacy bridge remains truthfully false because it is
not yet the Phase 3 persistent API.

- [ ] **Step 5: Run hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no staged `fastkpc/artifacts` symlink.

- [ ] **Step 6: Record Phase 3B without claiming Phase 3 completion**

Append to the Phase 3 roadmap status:

```text
Phase 3B complete:
  one same-S native Y/SP upload phase per setup batch
  one CUDA X0'Y RHS build per setup batch
  fused target-specific system construction
  true batched potrf/potrs
  canonical mixed-batch output order
  iteration true-batched targets = 160
  iteration single safe targets = 12
  iteration stable targets remain explicit errors = 98
  post-warm-up allocation/handle creation = 0

Active next task:
  Phase 3C penalty roots and augmented QR/SVD
```

- [ ] **Step 7: Commit the Phase 3B record**

```bash
git add goal-5.6.md
git commit -m "docs: record full CUDA CI Phase 3B milestone"
```

- [ ] **Step 8: Request code and spec-conformance review**

Review against the Phase 3 design and this plan. Resolve findings and rerun
Tasks 7.1 through 7.5 before Phase 3C begins.
