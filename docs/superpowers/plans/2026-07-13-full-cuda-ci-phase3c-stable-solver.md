# Full-CUDA CI Phase 3C Stable QR/SVD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the persistent fixed-sp CUDA numerical service by building one-time penalty roots, solving the declared stable buckets with augmented QR/SVD, returning valid residuals for every canonical iteration and qualification target, and proving zero dCov/near-alpha decision flips.

**Architecture:** Extend the Phase 3B runtime. Prepared handles build and retain projected penalty roots once. The context reserves one maximum-size augmented workspace and one persistent `gesvdjInfo_t`. QR and SVD targets execute sequentially inside the same native batch call while reusing the stream, cuBLAS/cuSOLVER handles, and workspace; Cholesky targets retain true batching. All routes scatter to the same canonical device output buffers and remain truthful in per-target diagnostics.

**Tech Stack:** Phase 3A/3B runtime, C++17, CUDA C++17, CUDA Runtime, cuBLAS `Dtrsv/Dgemv`, cuSOLVER `Dsyevd/Dgeqrf/Dormqr/Dgesvdj`, Rcpp `.Call`, Phase 2 `C_magic` oracle, legacy C++ Spectra dCov comparator.

---

## Preconditions

Start only after Phase 3B verification and review pass. Re-run the Phase 3B
iteration test before editing.

Frozen canonical counts:

```text
iteration:
  setups                     = 44
  penalty root matrices      = 159
  penalty root rows          = 1,424
  explicit constraints / H   = 0 / 0
  Cholesky / QR / SVD        = 172 / 31 / 67
  SVD finite-high / nonfinite= 59 / 8

qualification:
  setups                     = 2,061
  targets                    = 6,143
  penalty root matrices      = 6,272
  penalty root rows          = 63,552
  explicit constraints / H   = 0 / 0
  Cholesky / QR / SVD        = 3,889 / 190 / 2,064
  SVD finite-high / nonfinite= 902 / 1,162
  logical dCov pairs         = 3,808
  conditional near-alpha     = 1,478

qualification batching:
  all-safe / mixed / all-stable batches = 723 / 652 / 686
  true-batched Cholesky subgroups        = 806
  true-batched Cholesky targets          = 3,320
  single Cholesky targets                = 569
```

Phase 3C must reduce `ERR_STABLE_PATH_NOT_IMPLEMENTED` to zero in iteration
and qualification. It does not run the full 110,617-target or graph artifact;
that is the closure plan.

## Files

Create:

```text
fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh
  Internal augmented-system/root/QR/SVD declarations used only by the runtime.

fastkpc/src/cuda/mgcv_fixed_sp_stable.cu
  Penalty-root construction, augmented matrix kernels, QR and SVD solves.

fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R

fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R
```

Modify:

```text
fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp
fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp
fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu
fastkpc/tools/build_cuda_native.sh
fastkpc/src/r_api_cuda.cpp
fastkpc/R/cuda_native.R
fastkpc/R/full_cuda_ci_fixed_sp_runtime.R
fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.hpp
fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.cu
fastkpc/R/mgcv_extract_oracle.R
goal-5.6.md
```

## Spec Coverage for This Subplan

```text
persistent stable workspace and SVD parameters          Task 1
one-time projected penalty roots and rank validation    Task 2
augmented B/c construction                              Task 3
finite [1e8,1e12) QR solve and rank guard               Task 4
high-condition/rank-deficient SVD solve                 Task 5
three-route mixed batches and full iteration gate       Task 6
6,143-target qualification numeric/resource gate        Task 7
3,808 dCov and 1,478 near-alpha decision gate           Task 8
legacy single API convergence on one solver             Task 9
clean verification and roadmap record                   Task 10
```

The two required full artifacts remain the closure plan.

## Task 1: Reserve Persistent Stable-Solver Resources

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Create: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Create: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Modify: `fastkpc/tools/build_cuda_native.sh`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R`

- [ ] **Step 1: Add failing stable-resource lifecycle assertions**

Extend the runtime lifecycle test after canonical reserve:

```r
info <- fixed_sp_cuda_runtime_info(runtime)
required <- c(
  "gesvdj_info_create_count", "gesvdj_info_destroy_count",
  "eigen_workspace_bytes", "qr_workspace_bytes", "svd_workspace_bytes",
  "augmented_workspace_bytes", "stable_workspace_grow_count"
)
assert_true(length(setdiff(required, names(info))) == 0L,
            "stable runtime diagnostic schema")
assert_true(info$gesvdj_info_create_count == 1L,
            "one persistent gesvdjInfo")
assert_true(info$qr_workspace_bytes > 0 && info$svd_workspace_bytes > 0,
            "stable workspaces reserved")
assert_true(info$eigen_workspace_bytes > 0,
            "penalty eigensolver workspace reserved")
assert_true(info$augmented_workspace_bytes >= 8 * 407 * 64,
            "maximum augmented matrix reserved")
assert_true(info$stable_workspace_grow_count == 0L,
            "stable workspace does not grow after reserve")
```

- [ ] **Step 2: Run and verify missing stable diagnostics**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: FAIL on missing fields.

- [ ] **Step 3: Define stable workspace views**

Create `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`:

```cpp
#ifndef FASTKPC_MGCV_FIXED_SP_STABLE_CUH
#define FASTKPC_MGCV_FIXED_SP_STABLE_CUH

#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace fastkpc {

struct FixedSpStableWorkspace {
  double* B = nullptr;
  double* c = nullptr;
  double* tau = nullptr;
  double* eigen_work = nullptr;
  double* qr_work = nullptr;
  double* svd_work = nullptr;
  double* singular_values = nullptr;
  double* U = nullptr;
  double* V = nullptr;
  double* scaled_projection = nullptr;
  double* diagonal = nullptr;
  int* info = nullptr;
  int eigen_lwork = 0;
  int qr_lwork = 0;
  int ormqr_lwork = 0;
  int svd_lwork = 0;
  int max_rows = 0;
  int max_q = 0;
};

}  // namespace fastkpc

#endif
```

- [ ] **Step 4: Create one persistent Jacobi-SVD parameter object**

In the runtime context constructor:

```cpp
check_cusolver(cusolverDnCreateGesvdjInfo(&svd_params),
               "create Phase 3C gesvdjInfo");
check_cusolver(cusolverDnXgesvdjSetTolerance(svd_params, 1e-12),
               "set Phase 3C SVD convergence tolerance");
check_cusolver(cusolverDnXgesvdjSetMaxSweeps(svd_params, 100),
               "set Phase 3C SVD max sweeps");
check_cusolver(cusolverDnXgesvdjSetSortEig(svd_params, 1),
               "sort Phase 3C singular values");
```

Destroy it exactly once in the same-PID context destructor. Add lifecycle
counters to `FixedSpRuntimeInfo` and the R info wrapper.

- [ ] **Step 5: Query and reserve maximum eigensolver/QR/SVD workspace**

During canonical reserve, use a reserve-time probe allocation for
`m = 407`, `q = 64` and query:

```cpp
cusolverDnDsyevd_bufferSize(
  solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
  max_q, probe_square, max_q, probe_sigma, &eigen_lwork
);
cusolverDnDgeqrf_bufferSize(
  solver, max_rows, max_q, probe_B, max_rows, &qr_lwork
);
cusolverDnDormqr_bufferSize(
  solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
  max_rows, 1, max_q, probe_B, max_rows, probe_tau,
  probe_rhs, max_rows, &ormqr_lwork
);
cusolverDnDgesvdj_bufferSize(
  solver, CUSOLVER_EIG_MODE_VECTOR, 1,
  max_rows, max_q, probe_B, max_rows, probe_sigma,
  probe_U, max_rows, probe_V, max_q,
  &svd_lwork, svd_params
);
```

Reserve `eigen_lwork` for projected-penalty/H eigendecomposition,
`max(qr_lwork, ormqr_lwork)` for QR, and `svd_lwork` for SVD, plus
`B`, `c`, `tau`, `sigma`, economy `U`, full `V`, scaled projection, diagonal,
and info. The legacy cuSOLVER buffer-size values are element counts, so expose
workspace diagnostics in bytes only after multiplying by `sizeof(double)`.
Free all reserve probes before returning. A second equal reserve performs no
allocation or query.

Reserve compact per-target device arrays for QR/SVD `info`, QR rank, SVD
effective rank, sigma extrema, and reroute flags from the existing integer and
double arenas. Stable solvers write these arrays on the persistent stream.
They are copied to the host only at batch checkpoints; no target may trigger a
private stream synchronization or diagnostic D2H copy.

- [ ] **Step 6: Add the stable object to the build**

Compile `mgcv_fixed_sp_stable.cu` before `mgcv_fixed_sp_runtime.cu`, link its
object before the runtime object, and include the new internal header from the
runtime implementation.

- [ ] **Step 7: Clean-build and run lifecycle tests**

```bash
rm -f fastkpc/build/mgcv_fixed_sp_stable.o \
  fastkpc/build/mgcv_fixed_sp_runtime.o \
  fastkpc/build/r_api_cuda.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: PASS with one persistent SVD parameter object and no post-reserve
growth.

- [ ] **Step 8: Commit stable resources**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/tools/build_cuda_native.sh fastkpc/src/r_api_cuda.cpp \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
git commit -m "feat: reserve persistent fixed-sp QR and SVD resources"
```

## Task 2: Build and Validate Penalty Roots Once per Setup

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R`

- [ ] **Step 1: Write a failing root reconstruction test**

Open the iteration catalog, create one runtime, and for each of the 44 setup
batches create a prepared handle. Add a test-only explicit root materializer:

```r
roots <- fixed_sp_cuda_prepared_materialize_roots_for_test(handle)
assert_true(length(roots$penalty_roots) == dto$penalty_count,
            "one root per penalty")
for (penalty_index in seq_len(dto$penalty_count)) {
  block <- dto$penalty_blocks[[penalty_index]]
  offset <- dto$penalty_offsets[[penalty_index]]
  full <- matrix(0, dto$coefficient_dim, dto$coefficient_dim)
  rows <- offset:(offset + nrow(block) - 1L)
  full[rows, rows] <- block
  Z <- if (identical(dto$constraint_mode, "identity")) {
    diag(dto$coefficient_dim)
  } else {
    dto$constraint_nullspace
  }
  projected <- crossprod(Z, full %*% Z)
  reconstructed <- crossprod(roots$penalty_roots[[penalty_index]])
  assert_true(max(abs(reconstructed - projected)) < 1e-10,
              "penalty root reconstruction")
  assert_true(nrow(roots$penalty_roots[[penalty_index]]) ==
                dto$penalty_ranks[[penalty_index]],
              "penalty root rank")
}
assert_true(is.null(roots$H_root), "canonical corpus has no H root")
```

Aggregate and require:

```r
assert_true(total_root_matrix_count == 159L,
            "iteration penalty root matrix count")
assert_true(total_root_row_count == 1424L,
            "iteration penalty root row count")
assert_true(total_rank_mismatch_count == 0L,
            "iteration penalty root ranks")
```

Add a synthetic setup test with an explicit constraint whose nullspace does
not reduce the declared smooth-penalty rank and a positive-semidefinite `H`.
Require `crossprod(root) == Z' P Z` for the smooth penalty and
`crossprod(H_root) == Z' H Z` for the fixed penalty. This synthetic case is
the contract evidence for branches absent from the canonical corpus, whose
44 iteration and 2,061 qualification setups all have identity constraints and
`H = NULL`.

- [ ] **Step 2: Run and verify missing root materializer**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
```

Expected: FAIL with missing root API.

- [ ] **Step 3: Add root storage and diagnostics to prepared handles**

Extend the prepared handle with:

```cpp
double* d_penalty_roots = nullptr;
std::vector<int> penalty_root_offsets;
std::vector<int> penalty_root_ranks;
double* d_H_root = nullptr;
int H_root_rank = 0;
int penalty_root_build_count = 0;
int penalty_root_rank_mismatch_count = 0;
std::size_t penalty_root_bytes = 0;
double penalty_root_build_ms = 0.0;
```

Allocate the contiguous smooth-root storage at handle creation. The total
smooth root rows are `sum(penalty_ranks)` and every root has `q` columns.
Store it as one column-major `(total_root_rows x q)` matrix with row-band
offsets for each penalty. If `H` is non-null, independently reserve a
column-major `(q x q)` capacity after projecting `H` through `Z`; only the
first derived `H_root_rank` rows are logically live.

- [ ] **Step 4: Implement one-time symmetric eigendecomposition per penalty**

Phase 3A already uploads each projected `q x q` smooth penalty. For each
projected penalty:

1. copy its `q x q` matrix to stable scratch;
2. call `cusolverDnDsyevd` with vectors and lower fill;
3. launch a device validation/root kernel over the ascending eigenvalues and
   column-major eigenvectors;
4. define `tol = q * max(abs(eigenvalues)) * double_epsilon`;
5. reject eigenvalues `< -tol` as a non-PSD penalty;
6. retain eigenvalues `> tol` and require the count to equal the authenticated
   Phase 2 penalty rank;
7. launch a kernel that writes root row `r` as
   `sqrt(lambda_r) * eigenvector_r'` in deterministic ascending retained-index
   order into the root matrix's row band.

The setup creation fails if any rank differs. It cannot continue with an
approximate root. Canonicalize each retained eigenvector sign before writing:
find its first maximum-absolute component and make that component nonnegative.
This does not change `R'R` and removes an avoidable repeat-run ambiguity.

Queue all smooth-penalty eigendecompositions and device validation records,
then record/wait one setup-build event and copy only compact `info`, PSD, and
derived-rank fields. Do not copy eigenvalue vectors to the host. Reusing the
single eigensolver scratch is safe because all operations are ordered on the
persistent stream.

If `H` is present, extend `PreparedSHostView`/the private prepared handle with
the fixed penalty, project it as `Z' H Z`, and apply the same PSD/tolerance/root
procedure. `H` has no Phase 2 smooth-rank field, so its derived rank is frozen
by the same formula and reported as `H_root_rank`; a non-PSD or nonfinite `H`
fails handle creation. Keep smooth-root matrix count and H-root matrix count as
separate diagnostics so the canonical `159` and `6,272` counts retain their
meaning.

- [ ] **Step 5: Implement the test-only root materializer**

Add `.Call`/R wrapper:

```r
fixed_sp_cuda_prepared_materialize_roots_for_test <- function(handle) {
  .Call("C_fixed_sp_cuda_prepared_materialize_roots_for_test", handle,
        PACKAGE = "fastkpc_cuda")
}
```

It validates PID/device/generation, waits on the setup event, copies roots, and
increments a setup-shadow D2H counter. Production solve code never calls it.

- [ ] **Step 6: Run root, prepared-handle, and lifecycle tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
```

Expected: all PASS; exact iteration smooth-root matrix/row counts
`159 / 1,424`, zero mismatch, and exact synthetic smooth/H reconstruction.

- [ ] **Step 7: Commit penalty roots**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
git commit -m "feat: build fixed-sp CUDA penalty roots once per setup"
```

## Task 3: Construct the Augmented System on Device

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R`

- [ ] **Step 1: Write a failing augmented-system test**

Select one iteration QR target and one SVD target. Add a test-only function
that builds and materializes `B` and `c` without solving. For each target:

```r
augmented <- fixed_sp_cuda_build_augmented_for_test(
  handle, native$Y[, target_index], native$SP[, target_index],
  target_index = target_index
)
root_rows <- sum(dto$penalty_ranks) + roots$H_root_rank
assert_true(identical(dim(augmented$B),
                      c(dto$n + root_rows, dto$null_dim)),
            "augmented dimensions")
X_null <- if (identical(dto$constraint_mode, "identity")) {
  dto$X
} else {
  dto$X %*% dto$constraint_nullspace
}
assert_true(identical(dto$weights_policy, "none-or-unit") &&
              identical(dto$offset_policy, "none-or-zero"),
            "Phase 3 v1 neutral weight/offset policies")
X0 <- X_null
assert_true(max(abs(augmented$B[seq_len(dto$n), , drop = FALSE] - X0)) <
              1e-12, "augmented design rows")
expected_c <- native$Y[, target_index]
assert_true(max(abs(augmented$c[seq_len(dto$n)] - expected_c)) < 1e-12,
            "augmented response rows")
assert_true(all(augmented$c[-seq_len(dto$n)] == 0),
            "augmented penalty RHS is zero")

penalty_from_B <- crossprod(
  augmented$B[-seq_len(dto$n), , drop = FALSE]
)
expected_penalty <- matrix(0, dto$null_dim, dto$null_dim)
for (penalty_index in seq_len(dto$penalty_count)) {
  root <- roots$penalty_roots[[penalty_index]]
  expected_penalty <- expected_penalty +
    native$SP[penalty_index, target_index] * crossprod(root)
}
if (!is.null(roots$H_root)) {
  expected_penalty <- expected_penalty + crossprod(roots$H_root)
}
assert_true(max(abs(penalty_from_B - expected_penalty)) < 1e-8,
            "augmented selected-sp penalty")
```

Run the same assertions on Task 2's synthetic explicit-constraint/non-null-H
setup. Its expected row count includes the derived `H_root_rank`.

- [ ] **Step 2: Run and verify missing builder**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R
```

Expected: FAIL with missing
`fixed_sp_cuda_build_augmented_for_test`.

- [ ] **Step 3: Implement deterministic augmented build kernels**

Add an internal descriptor:

```cpp
struct AugmentedSystemView {
  double* B = nullptr;
  double* c = nullptr;
  int leading_dimension = 0;
  int rows = 0;
  int cols = 0;
  int target_index = -1;
};
```

The build kernel writes:

```text
rows [0, n):              X_null
next penalty_rank[j]:     sqrt(SP[j,target]) * root_j
optional final H rows:    root_H

c[0:n]                    weighted(Y[:,target] minus offset)
c[n:rows]                 0
```

Reject nonpositive/nonfinite SP before launch. Use fixed row offsets from the
prepared handle. `B` is column-major with `leading_dimension = rows`; every
kernel indexes `B[row + col * rows]`. Phase 3 v1 accepts only the authenticated
`none-or-unit` weight and `none-or-zero` offset policies, so the displayed
weighted formula reduces exactly to `X_null` and `Y` for every canonical
setup. Any other policy fails at handle creation before launch. Build into the
context stable workspace and issue no allocation or host solve.

- [ ] **Step 4: Add the test-only materializer**

Expose only under the internal test-prefixed `.Call` name. It waits on the
handle event and copies `B/c`; production APIs cannot request augmented
matrices.

- [ ] **Step 5: Run augmented and root tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
```

Expected: both PASS for QR and SVD representatives.

- [ ] **Step 6: Commit augmented construction**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R
git commit -m "feat: construct fixed-sp augmented systems on CUDA"
```

## Task 4: Implement the Augmented QR Route

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R`

- [ ] **Step 1: Write a failing real-iteration QR test**

Select all 31 iteration QR targets. Solve their complete setup batches through
the public mixed-batch API and assert for each QR target:

```r
assert_true(info$status[[target_index]] == "OK_AUGMENTED_QR",
            "QR status")
assert_true(info$route_executed[[target_index]] == "AUGMENTED_QR",
            "QR executed route")
assert_true(!info$target_true_batched[[target_index]],
            "QR must not claim true batching")
assert_true(info$qr_rank[[target_index]] == dto$null_dim,
            "QR full rank")
assert_true(max(abs(shadow$residuals[, target_index] - oracle$residuals)) <
              1e-7, "QR residual max abs parity")
assert_true(relative_l2(shadow$residuals[, target_index], oracle$residuals) <
              1e-7, "QR residual relative L2 parity")
assert_true(max(abs(shadow$fitted[, target_index] - oracle$fitted)) <
              1e-7, "QR fitted max abs parity")
assert_true(relative_l2(shadow$fitted[, target_index], oracle$fitted) <
              1e-7, "QR fitted relative L2 parity")
```

Aggregate `qr_target_count == 31L` and `qr_to_svd_reroute_count == 0L`.

- [ ] **Step 2: Run and verify stable-not-implemented failure**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
```

Expected: FAIL because QR targets still return the Phase 3A milestone status.

- [ ] **Step 3: Implement QR factorization and Q-transpose RHS**

For each QR target in canonical order:

```cpp
check_cusolver(cusolverDnDgeqrf(
  solver, rows, q, workspace.B, rows, workspace.tau,
  workspace.qr_work, workspace.qr_lwork, workspace.info
), "Phase 3C augmented geqrf");
check_cusolver(cusolverDnDormqr(
  solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
  rows, 1, q, workspace.B, rows, workspace.tau,
  workspace.c, rows, workspace.qr_work,
  workspace.ormqr_lwork, workspace.info
), "Phase 3C augmented ormqr");
```

Queue a one-element device copy from `workspace.info` into that target's
`geqrf_info` slot immediately after `Dgeqrf`, and another into `ormqr_info`
immediately after `Dormqr`, before the shared info cell is reused.

After `ormqr`, call `cublasDtrsv` on the upper-triangular `R`; the first `q`
values of `c` become the candidate theta. Launch a compact device kernel that
copies `info`, scans `abs(diag(R))` directly from column-major `B`, computes
the QR rank below, and writes per-target rank/reroute fields. Before reusing
`c`, immediately launch the canonical beta/fitted/residual finalization into
that target's output column. Those outputs are provisional until the rank
checkpoint accepts them and are overwritten by SVD on a declared reroute. Do
not copy a diagonal vector or synchronize per target.

- [ ] **Step 4: Apply the frozen rank guard**

Compute:

```text
max_diag = max(abs(diag(R)))
rank_tol = max(rows, q) * max_diag * double_epsilon
qr_rank  = count(abs(diag(R)) > rank_tol)
```

If `qr_rank != q`, rebuild `B/c`, increment `qr_to_svd_reroute_count`, and
call the Task 5 SVD routine. Until Task 5 lands, return
`ERR_STABLE_PATH_NOT_IMPLEMENTED` for that declared reroute. The 31 iteration
QR targets must not reroute.

Process all declared QR targets before reading any rank result. Record one QR
checkpoint event for the public batch, wait once, and copy only compact
`geqrf_info/ormqr_info/rank/reroute` arrays. A nonzero solver info is
`ERR_QR_FAILED`;
only a successful factorization rejected by the rank guard is a declared CUDA
QR-to-SVD reroute. This batch-level checkpoint is separately counted and is
never reported as `cudaDeviceSynchronize`.

- [ ] **Step 5: Solve R theta and finalize canonical output**

The triangular solve already issued in Step 3 is:

```cpp
check_cublas(cublasDtrsv(
  blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
  CUBLAS_DIAG_NON_UNIT, q, workspace.B, rows,
  workspace.c, 1
), "Phase 3C QR triangular solve");
```

After the checkpoint, mark accepted provisional columns
`OK_AUGMENTED_QR`. Provisional output for a rejected column is never exposed
and is overwritten by Task 5's SVD result. Assert in the test that QR
checkpoint record/wait counts are bounded by public batches containing QR
targets, not by the 31 QR target count.

- [ ] **Step 6: Run QR and Phase 3B regressions**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
```

Expected: QR test and all Phase 3B tests PASS.

- [ ] **Step 7: Commit QR route**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
git commit -m "feat: solve stable fixed-sp targets with augmented CUDA QR"
```

## Task 5: Implement the Augmented SVD Route

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R`

- [ ] **Step 1: Write a failing SVD corpus test**

Select all 67 iteration SVD targets and require:

```r
assert_true(svd_target_count == 67L, "iteration SVD count")
assert_true(svd_finite_high_count == 59L, "finite high-condition SVD count")
assert_true(svd_nonfinite_count == 8L, "nonfinite/rank-deficient SVD count")
assert_true(all(status == "OK_AUGMENTED_SVD"), "SVD statuses")
assert_true(all(!target_true_batched), "SVD truthfulness")
assert_true(all(svd_info == 0L), "SVD cuSOLVER info")
assert_true(all(effective_rank == expected_augmented_rank),
            "SVD augmented-system rank")
assert_true(residual_max_abs_diff < 1e-7 && residual_relative_l2 < 1e-7,
            "SVD residual parity")
assert_true(fitted_max_abs_diff < 1e-7 && fitted_relative_l2 < 1e-7,
            "SVD fitted parity")
```

For the 67-target test only, build each CPU augmented `B` with the same roots
and compute `expected_augmented_rank` from
`max(nrow(B), ncol(B)) * max(svd(B)$d) * .Machine$double.eps`. The per-target
table also carries the authenticated Phase 1 condition value and mgcv
`coefficient_rank`, but that rank is route metadata, not the SVD truncation
oracle. Record `effective_rank - coefficient_rank` diagnostically; do not
force equality between two ranks defined by different matrices/tolerances.

- [ ] **Step 2: Run and verify stable-not-implemented failure**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
```

Expected: FAIL because SVD targets are not solved.

- [ ] **Step 3: Implement economy Jacobi SVD**

For each SVD target, rebuild the augmented system and call:

```cpp
check_cusolver(cusolverDnDgesvdj(
  solver, CUSOLVER_EIG_MODE_VECTOR, 1,
  rows, q, workspace.B, rows,
  workspace.singular_values,
  workspace.U, rows,
  workspace.V, q,
  workspace.svd_work, workspace.svd_lwork,
  workspace.info, svd_params
), "Phase 3C augmented gesvdj");
```

Immediately after each queued SVD, launch device work that copies `info` to the
target status slot and consumes that target's singular values before the
workspace is reused. Nonzero info is `ERR_SVD_FAILED` and fails the gate.

Do not call `cusolverDnXgesvdjGetSweeps` or
`cusolverDnXgesvdjGetResidual` per target: those host-observation APIs would
serialize the stream and violate the persistent batch contract. The hard
convergence evidence is device `info == 0` plus finite singular values and
numeric parity. The persistent `gesvdjInfo_t` still freezes tolerance, maximum
sweeps, and sorting configuration in runtime/manifest diagnostics.

- [ ] **Step 4: Apply deterministic singular-rank truncation**

Compute on device:

```text
rank_tol = max(rows, q) * sigma_max * double_epsilon
effective_rank = count(sigma_i > rank_tol)
scaled_i = (U[:,i]' c) / sigma_i when sigma_i > rank_tol, else 0
theta = V * scaled
```

Use cuBLAS `Dgemv` for `U'c` and `V*scaled`; use a deterministic elementwise
kernel for truncation/division. With `econ = 1` and `rows >= q`, cuSOLVER
stores economy `U` as `rows x q` and right singular vectors as columns of the
`q x q` matrix `V`; therefore the two GEMVs are exactly `U'c` followed by
`V*scaled`. Do not transpose `V`, invert, or solve normal equations.

- [ ] **Step 5: Finalize outputs and diagnostics**

Reuse canonical output kernels. Set `OK_AUGMENTED_SVD`, effective rank,
`sigma_max`, smallest retained sigma, SVD info, and
`target_true_batched = FALSE`. Copy the compact SVD diagnostics once after all
declared and rerouted SVD targets in the public batch finish; no target-level
host wait is allowed.

- [ ] **Step 6: Run SVD, QR, root, and Phase 3B tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
```

Expected: all PASS; all 98 formerly unsupported iteration targets now solve.

- [ ] **Step 7: Commit SVD route**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
git commit -m "feat: solve difficult fixed-sp targets with augmented CUDA SVD"
```

## Task 6: Close the Three-Route Iteration Gate

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R`

- [ ] **Step 1: Write the full iteration gate**

Run all 44 setup batches through one runtime and require:

```r
expected <- list(
  setup_count = 44L,
  target_count = 270L,
  penalty_root_matrix_count = 159L,
  penalty_root_row_count = 1424L,
  H_root_matrix_count = 0L,
  cholesky_count = 172L,
  qr_count = 31L,
  svd_count = 67L,
  true_batched_target_count = 160L,
  cholesky_single_target_count = 12L,
  whole_batch_true_batched_count = 5L,
  stable_not_implemented_count = 0L,
  stable_reroute_count = 0L,
  non_ok_status_count = 0L,
  root_rank_mismatch_count = 0L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
```

Require every target's residual/fitted max-absolute and relative-L2 error
`< 1e-7`, finite outputs, canonical key order, and exact repeated route/status
fields. Require GPU SVD effective rank to equal the independently computed CPU
augmented-system rank under the frozen threshold. Record Phase 1
`coefficient_rank` separately. QR/SVD checkpoint waits may occur only once per
affected public batch and may never scale with QR/SVD target count.

- [ ] **Step 2: Run and verify missing Phase 3C runner**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
```

Expected: FAIL with missing
`fastkpc_run_full_cuda_fixed_sp_phase3c_iteration`.

- [ ] **Step 3: Implement the Phase 3C iteration runner**

Base it on the Phase 3B runner but require all statuses `OK_*`, collect
route-specific diagnostics, and recompute all aggregate counts from target,
batch, setup, and runtime rows. Keep explicit shadow materialization separate
from solve metrics.

- [ ] **Step 4: Run the full iteration gate twice**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
```

Expected: both runs PASS and their target route/status/numeric hashes are
identical.

- [ ] **Step 5: Commit the iteration closure**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
git commit -m "test: close full CUDA fixed-sp iteration parity"
```

## Task 7: Run the 6,143-Target Qualification Gate

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Create: `fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R`

- [ ] **Step 1: Write a failing qualification summary test**

The test invokes the runner into a temporary output directory and validates:

```r
expected <- list(
  setup_count = 2061L,
  target_count = 6143L,
  penalty_root_matrix_count = 6272L,
  penalty_root_row_count = 63552L,
  H_root_matrix_count = 0L,
  cholesky_count = 3889L,
  qr_count = 190L,
  svd_count = 2064L,
  svd_finite_high_count = 902L,
  svd_nonfinite_count = 1162L,
  all_safe_batch_count = 723L,
  mixed_batch_count = 652L,
  all_stable_batch_count = 686L,
  true_batched_subgroup_count = 806L,
  true_batched_target_count = 3320L,
  cholesky_single_target_count = 569L,
  whole_batch_true_batched_count = 723L,
  non_ok_status_count = 0L,
  stable_not_implemented_count = 0L,
  stable_reroute_count = 0L,
  root_rank_mismatch_count = 0L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
```

It also requires route- and condition-bucket numeric maxima `< 1e-7`, all
outputs finite, GPU/CPU augmented-SVD ranks exact under the frozen threshold,
all cuSOLVER info zero, and no target-level stable-path synchronization.

- [ ] **Step 2: Run and verify missing qualification runner**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
```

Expected: FAIL because the runner/artifact does not exist.

- [ ] **Step 3: Implement the qualification runner**

`fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R` reads these fixed
environment variables with defaults:

```text
FASTKPC_FULL_CUDA_PHASE3_DEVICE=0
FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1
FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1
FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1
FASTKPC_FULL_CUDA_PHASE3_DATA=fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_qualification_v1
```

It authenticates the catalog, runs qualification scope in PreparedSKey radix
order, and writes through a staging directory:

```text
summary.json
manifest.json
target_parity.rds and .csv
batch_metrics.rds and .csv
setup_metrics.rds and .csv
runtime_metrics.csv
stage_timing.csv
fallbacks.csv
failures.csv
commands.txt
environment.txt
```

Write `manifest.json` penultimate and `summary.json` last. The test recomputes
all counts from the RDS rows and rejects caller-supplied pass booleans.

- [ ] **Step 4: Run qualification**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
FASTKPC_FULL_CUDA_PHASE3_DEVICE=0 \
FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR=fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR=fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1 \
FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR=fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1 \
FASTKPC_FULL_CUDA_PHASE3_DATA=fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds \
FASTKPC_FULL_CUDA_PHASE3_OUTPUT=fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_qualification_v1 \
  Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
```

Expected: PASS with the exact counts above. This is the first multi-thousand
target run; do not start a full 110,617-target run here.

- [ ] **Step 5: Commit qualification runner/gate**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
git commit -m "test: qualify stable fixed-sp CUDA on 6143 targets"
```

## Task 8: Gate Qualification dCov and Near-Alpha Decisions

**Files:**
- Modify: `fastkpc/R/full_cuda_ci_fixed_sp_runtime.R`
- Modify: `fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R`
- Create: `fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R`

- [ ] **Step 1: Write the failing dCov/decision test**

Read the qualification artifact and the authenticated
`qualification_logical_tests.rds`. Require:

```r
assert_true(nrow(dcov) == 3808L, "qualification logical dCov count")
assert_true(sum(dcov$near_alpha) == 1478L,
            "qualification near-alpha count")
assert_true(max(dcov$absolute_p_value_difference) < 1e-10,
            "qualification dCov p-value tolerance")
assert_true(sum(dcov$decision_flip) == 0L,
            "qualification decision flips")
assert_true(sum(dcov$near_alpha & dcov$decision_flip) == 0L,
            "near-alpha decision flips")
assert_true(sum(dcov$backend_error) == 0L, "dCov backend errors")
assert_true(sum(dcov$spectra_fallback) == 0L, "Spectra fallbacks")
```

- [ ] **Step 2: Run and verify missing dCov evidence**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
```

Expected: FAIL because qualification output lacks dCov rows.

- [ ] **Step 3: Compute pinned legacy C++ Spectra dCov parity**

For each qualification logical test in canonical sequence order:

1. look up the two materialized candidate residual vectors by residual key;
2. call the correctness-qualified legacy C++ dCov gamma backend with Spectra;
3. compare to the Phase 0/2 reference p-value;
4. derive the decision using the recorded reference decision convention;
5. record p-value drift, decision flip, near-alpha flag, backend error, and
   Spectra fallback.

During the qualification solve, retain a bounded in-memory residual registry
for the 6,143 qualification keys (`6143 * 351 * 8`, about 16.5 MiB of raw
doubles). Populate it only through the explicit shadow materializer, key it by
authenticated `residual_key_sha256`, and reject missing, duplicate, or
conflicting vectors. Consume the registry for dCov before publication, then
drop it; do not add residual vectors to the artifact payload. This makes the
lookup in step 1 concrete without weakening the device-resident production
ABI.

Use:

```text
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
```

The runner stores `qualification_dcov_parity.rds/.csv` and folds independently
recomputed dCov fields into the qualification summary.

- [ ] **Step 4: Run dCov and qualification tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
```

Expected: both PASS with zero flips/errors/fallbacks.

- [ ] **Step 5: Commit qualification decision gate**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tools/run_full_cuda_ci_fixed_sp_qualification.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
git commit -m "test: gate stable fixed-sp CUDA dCov decisions"
```

## Task 9: Converge the Legacy Single CUDA API on the Stable Runtime

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.hpp`
- Modify: `fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R`

- [ ] **Step 1: Add adapter diagnostics to the existing test**

Append:

```r
assert_true(identical(gpu$diagnostics$runtime_version,
                      "full-cuda-ci-fixed-sp-runtime-v1"),
            "legacy single delegates to stable runtime")
assert_true(isTRUE(gpu$diagnostics$compatibility_transient_context),
            "legacy adapter declares transient context")
assert_true(identical(gpu$diagnostics$route_executed, "AUGMENTED_SVD"),
            "unclassified compatibility call conservatively uses SVD")
assert_true(gpu$diagnostics$cpu_fallback_count == 0L,
            "compatibility adapter has no CPU fallback")
```

- [ ] **Step 2: Run and verify missing diagnostics**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
```

Expected: FAIL on `runtime_version`.

- [ ] **Step 3: Replace the independent legacy solver with an adapter**

Retain the public result shape. Build a temporary runtime and a synthetic
private compatibility prepared view with:

```text
Gram                 = XtX_null
penalty_count        = 1
projected penalty    = assembled penalty_null
SP                   = matrix(1.0, 1, 1)
RHS                  = Xty_null
route metadata       = unauthenticated -> AUGMENTED_SVD
```

The private adapter also receives the existing `X`, `y`, and `Z`, constructs
`X_null = X Z`, and builds the aggregate penalty root used by the augmented
SVD. Before creating CUDA state, independently recompute
`crossprod(X_null)` and `crossprod(X_null, y)` and require parity with the
caller-supplied `XtX_null` and `Xty_null` under a fixed `1e-12` relative/absolute
check. This adapter must not forge canonical Phase 2 fingerprints or pass its
synthetic view through the authenticated artifact constructor.

Call `solve_fixed_sp_batch(target_count = 1)`, explicitly materialize legacy
fields, and destroy token/handle/context. Remove the independent per-target
allocation/Cholesky implementation from `mgcv_extract_fixed_sp_cuda.cu`.

The adapter is excluded from persistent operational metrics and reports
`compatibility_transient_context = TRUE`.

- [ ] **Step 4: Run legacy and Phase 3 regressions**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
```

Expected: all PASS. The old same-setup bridge remains explicitly false until
it is replaced by a caller that can provide authenticated Phase 3 route/setup
metadata.

- [ ] **Step 5: Commit one stable solver implementation**

```bash
git add fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.hpp \
  fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/mgcv_extract_oracle.R \
  fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
git commit -m "refactor: delegate fixed-sp CUDA singles to stable runtime"
```

## Task 10: Phase 3C Verification and Roadmap Record

**Files:**
- Modify: `goal-5.6.md`

- [ ] **Step 1: Clean-build CUDA**

```bash
rm -f fastkpc/build/*.o fastkpc/build/fastkpc_cuda.so
bash fastkpc/tools/build_cuda_native.sh
```

Expected: clean build exits zero.

- [ ] **Step 2: Run all Phase 3 contract/resource tests**

```bash
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_catalog.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_native_dto.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_prepared_handle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_misuse.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_augmented_system.R
```

Expected: all PASS.

- [ ] **Step 3: Run all Phase 3 numerical tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_qualification.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_dcov.R
```

Expected: all PASS.

- [ ] **Step 4: Run existing CUDA regressions and hygiene**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_native_batch_bridge.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
git diff --check
git status --short
```

Expected: tests PASS, no whitespace errors, and no staged artifact symlink.

- [ ] **Step 5: Record Phase 3C without claiming full Phase 3 closure**

Append:

```text
Phase 3C complete:
  one-time penalty roots, qualification matrices/rows 6,272 / 63,552
  augmented QR and deterministic augmented SVD
  iteration 270/270 targets OK
  qualification 6,143/6,143 targets OK
  route counts 3,889 / 190 / 2,064 exact
  dCov 3,808 pairs, near-alpha 1,478, decision flips 0
  unknown/CPU/approximate fallback 0

Active next task:
  generate fixed_sp_cuda_oracle_sp_v1 and fixed_sp_cuda_full_shadow_v1
```

- [ ] **Step 6: Commit Phase 3C record**

```bash
git add goal-5.6.md
git commit -m "docs: record full CUDA CI Phase 3C qualification"
```

- [ ] **Step 7: Request code and spec-conformance review**

Review all 3A/3B/3C code against the Phase 3 design. Resolve findings and rerun
Tasks 10.1 through 10.4 before starting full artifact closure.
