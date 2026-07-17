# Full-CUDA CI Phase 3C Stable QR/SVD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the persistent fixed-sp CUDA numerical service by retaining one-time individual penalty roots for QR/diagnostics, building target-specific legacy-compatible aggregate roots for SVD, returning valid residuals for every canonical iteration and qualification target, and proving zero dCov/near-alpha decision flips.

**Architecture:** Extend the Phase 3B runtime. Prepared handles retain resident projected penalties plus the existing one-time individual roots. QR stacks those roots; every executed SVD target instead aggregates `P_H + sum_j sp_j P_j` from device SP, applies deterministic upper-DPSTF2-compatible pivoted Cholesky entirely on device, scatters the compact root to original coefficient order, and solves `[X0; R_aggregate]`. Callers reserve QR's logical augmented rows; C++ internally reserves `max(augmented_rows, n + q)` stable rows after merging capacities. The context reserves all factor/augmented storage and one persistent `gesvdjInfo_t`; Cholesky remains truly batched while QR/SVD targets reuse the same stream, handles, and workspace sequentially.

**Tech Stack:** Phase 3A/3B runtime, C++17, CUDA C++17, CUDA Runtime, cuBLAS `Dtrsv/Dgemv`, cuSOLVER `Dsyevd/Dgeqrf/Dormqr/Dgesvdj`, deterministic CUDA DPSTF2-compatible factor kernel, test-only CPU LAPACK `DPSTRF/DPSTF2`, Rcpp `.Call`, Phase 2 `C_magic` oracle, legacy C++ Spectra dCov comparator.

---

## Preconditions

This document is now an amended execution record, not a fresh pre-implementation
plan. Preserve the current branch history: Task 1 landed as `9be58b6`, Task 2
as `2b78cc7` plus `e383611`, Task 3 as `50bb61d`, Task 4 as `1812384` plus
`4c1f521`, and the original stacked-root Task 5 as `8bd6142` plus `2574980` and
`5b5cee0`. Do not reset, rewrite, or replay those commits.

The correction reopens Task 5 from that current baseline. The documentation
commit `docs: align stable SVD with legacy aggregate penalty semantics` lands
after the existing Task 1-5 history; a follow-up implementation commit corrects
Task 5 in place. Phase 2 `C_magic` is the sole fitted/residual oracle for all
routes, and a CPU stacked augmented SVD is an internal diagnostic only.

The correction is evidence-driven. Independent stacked roots failed `C_magic`
for `14 / 67` iteration SVD targets (maximum residual absolute error
`0.52305231`). The target-key prefix `03de91f` matched stacked CPU augmented SVD
at about `4e-12` but differed from `C_magic` by `0.0594444`. A CPU
upper-DPSTF2-compatible aggregate-root prototype matched LAPACK rank/pivot for
`67 / 67` targets and reached about `9.03e-10` maximum `C_magic` error.

Frozen canonical counts:

```text
iteration:
  setups                     = 44
  penalty root matrices      = 159
  penalty root rows          = 1,424
  explicit constraints / H   = 0 / 0
  planned Cholesky / QR / SVD = 172 / 31 / 67
  SVD finite-high / nonfinite= 59 / 8

qualification:
  setups                     = 2,061
  targets                    = 6,143
  penalty root matrices      = 6,272
  penalty root rows          = 63,552
  explicit constraints / H   = 0 / 0
  planned Cholesky / QR / SVD = 3,889 / 190 / 2,064
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

Route accounting is never implicit. The condition census supplies
`planned_route`; the solver writes `executed_route`, `reroute_reason`, and
`solver_status`. Iteration and qualification require zero Cholesky-to-SVD and
QR-to-SVD reroutes. The full closure plan uses the safe-reroute policy defined
in the design spec and validates route conservation rather than overloading
one route-count field.

## Files

Create:

```text
fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh
  Internal augmented-system/root/QR/SVD declarations used only by the runtime.

fastkpc/src/cuda/mgcv_fixed_sp_stable.cu
  Individual-root construction, QR augmented kernels, target aggregate
  penalty/root kernels, and QR/SVD solves.

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
persistent stable workspace and SVD parameters           Task 1 (landed)
one-time individual penalty roots and rank validation    Task 2
landed stacked B/c construction; QR authority            Task 3
finite [1e8,1e12) QR solve and rank guard               Task 4
aggregate-root SVD correction and resource delta         Task 5 (reopened)
planned/executed routes and three-route iteration gate  Task 6
6,143-target qualification numeric/resource gate        Task 7
3,808 dCov and 1,478 near-alpha decision gate           Task 8
legacy single API convergence on one solver             Task 9
clean verification and roadmap record                   Task 10
```

The two required full artifacts remain the closure plan.

The historical commit sequence above remains intact. The correction sequence
is: this documentation amendment, one reopened-Task-5 implementation/test
commit containing the aggregate resource/handle/factor changes, then Task 6's
all-route `C_magic` gate and removal of the temporary route-specific CPU
augmented fitted/residual reference. The existing `8bd6142` legitimately
returned `OK_AUGMENTED_SVD` under the now-superseded stacked semantics; correct
it with a follow-up commit rather than pretending it never landed. Tasks 1-4
below remain implementation provenance. Any new workspace or prepared-handle
storage required by the amendment belongs to reopened Task 5.

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
assert_true(info$augmented_workspace_bytes == 8 * 415 * 64,
            "internal stable matrix reserves max(407, 351 + 64) rows")
assert_true(info$stable_workspace_grow_count == 0L,
            "stable workspace does not grow after reserve")
```

Task 1 originally landed this byte assertion at `407` rows. The snippet above
shows its required final form after reopened Task 5; the caller reserve argument
remains logical `augmented_rows = 407`, and implementation ownership for the
`415` internal capacity remains Task 5.

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

This landed `1e-12` value controls `gesvdj` convergence only. Reopened Task 5
uses the separate `C_magic` solve-rank tolerance
`sqrt(std::numeric_limits<double>::epsilon())`.

- [ ] **Step 5: Query and reserve maximum eigensolver/QR/SVD workspace**

The landed Task 1 implementation used a reserve-time probe allocation for
`m = 407`, `q = 64` and queried:

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

Reopened Task 5 replaces only that landed `max_rows` probe/view input with
`effective_stable_rows`; callers still provide the logical QR row capacity.

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
  offset <- dto$penalty_offsets_zero_based[[penalty_index]] + 1L
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

These roots remain persistent QR inputs and setup diagnostics. Do not pass them
to the production SVD builder; Task 5 aggregates the resident projected
penalty matrices directly from device SP.

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

zero_sp <- native$SP[, target_index]
zero_sp[[1L]] <- 0
augmented_zero <- fixed_sp_cuda_build_augmented_for_test(
  handle, native$Y[, target_index], zero_sp,
  target_index = target_index
)
expected_zero_penalty <- matrix(0, dto$null_dim, dto$null_dim)
for (penalty_index in seq_len(dto$penalty_count)) {
  expected_zero_penalty <- expected_zero_penalty +
    zero_sp[[penalty_index]] *
      crossprod(roots$penalty_roots[[penalty_index]])
}
if (!is.null(roots$H_root)) {
  expected_zero_penalty <-
    expected_zero_penalty + crossprod(roots$H_root)
}
assert_true(max(abs(
  crossprod(augmented_zero$B[-seq_len(dto$n), , drop = FALSE]) -
    expected_zero_penalty
)) < 1e-8, "zero SP produces zero-scaled penalty rows")
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

Reject negative/nonfinite SP before launch; `SP == 0` is legal and emits a
zero-scaled root block. Direct `SP[j, target]` indexing is valid only under the
Phase 3A identity penalty-to-SP invariant. Use fixed row offsets from the
prepared handle. `B` is column-major with `leading_dimension = rows`; every
kernel indexes `B[row + col * rows]`. Phase 3 v1 accepts only the authenticated
`none-or-unit` weight and `none-or-zero` offset policies, so the displayed
weighted formula reduces exactly to `X_null` and `Y` for every canonical
setup. Any other policy fails at handle creation before launch. Build into the
context stable workspace and issue no allocation or host solve.

The landed builder remains the QR numerical builder and a valid stacked-root
implementation diagnostic. Reopened Task 5 adds a separate aggregate builder;
corrected production `AUGMENTED_SVD` must not call this stacked builder.

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
assert_true(info$solver_status[[target_index]] == "OK_AUGMENTED_QR",
            "QR status")
assert_true(info$planned_route[[target_index]] == "AUGMENTED_QR",
            "QR planned route")
assert_true(info$executed_route[[target_index]] == "AUGMENTED_QR",
            "QR executed route")
assert_true(info$reroute_reason[[target_index]] == "",
            "QR did not reroute")
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

Aggregate `planned_qr_target_count == 31L`,
`executed_qr_target_count == 31L`, and `qr_to_svd_count == 0L`.

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

If `qr_rank != q`, discard the stacked QR `B`, build Task 5's target aggregate
penalty/root and SVD `B/c`, increment `qr_to_svd_count`, set
`executed_route = AUGMENTED_SVD` and
`reroute_reason = "QR_RANK_GUARD_REJECTED"`, and
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

## Task 5: Reopen and Correct the Aggregate-Penalty Augmented SVD Route

**Files:**
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh`
- Modify: `fastkpc/src/cuda/mgcv_fixed_sp_stable.cu`
- Modify: `fastkpc/src/r_api_cuda.cpp`
- Modify: `fastkpc/R/cuda_native.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R`
- Modify: `fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R`

- [ ] **Step 1: Write the failing all-target aggregate-SVD test**

Select all 67 iteration SVD targets, run the public solve, and call the Phase 2
`C_magic` adapter once per target. Require each target independently to pass all
four fitted/residual gates; an aggregate maximum alone is insufficient:

```r
assert_true(planned_svd_target_count == 67L, "iteration planned SVD count")
assert_true(executed_svd_target_count == 67L,
            "iteration executed SVD count")
assert_true(svd_finite_high_count == 59L, "finite high-condition SVD count")
assert_true(svd_nonfinite_count == 8L, "nonfinite/rank-deficient SVD count")
assert_true(all(solver_status == "OK_AUGMENTED_SVD"), "SVD statuses")
assert_true(all(planned_route == "AUGMENTED_SVD"),
            "declared SVD planned routes")
assert_true(all(executed_route == "AUGMENTED_SVD"),
            "declared SVD executed routes")
assert_true(all(!target_true_batched), "SVD truthfulness")
assert_true(all(svd_info == 0L), "SVD cuSOLVER info")
assert_true(all(numeric_reference == "mgcv-fixed-sp"),
            "C_magic is the only SVD numeric reference")
assert_true(all(residual_max_abs_diff < 1e-7),
            "all SVD residual max-absolute gates")
assert_true(all(residual_relative_l2_diff < 1e-7),
            "all SVD residual relative-L2 gates")
assert_true(all(fitted_max_abs_diff < 1e-7),
            "all SVD fitted max-absolute gates")
assert_true(all(fitted_relative_l2_diff < 1e-7),
            "all SVD fitted relative-L2 gates")
assert_true(all(aggregate_penalty_root_rank >= 0L) &&
              all(aggregate_penalty_root_rank <= coefficient_dim),
            "aggregate root ranks are valid")
assert_true(all(effective_rank >= 0L) &&
              all(effective_rank <= coefficient_dim),
            "augmented SVD ranks are valid and separately named")
assert_true(identical(info$aggregate_factor_call_count,
                      rep.int(1L, length(info$executed_route))),
            "every selected SVD target factors exactly once")
assert_true(identical(info$aggregate_b_build_count,
                      rep.int(2L, length(info$executed_route))),
            "every selected SVD target builds B/c exactly twice")
assert_true(length(info$aggregate_penalty_root_rank) == 67L &&
              length(info$aggregate_penalty_root_pivot) == 67L &&
              all(lengths(info$aggregate_penalty_root_pivot) == dto$null_dim) &&
              length(info$aggregate_dstop) == 67L &&
              all(is.finite(info$aggregate_dstop)) &&
              all(info$aggregate_dstop >= 0),
            "aggregate diagnostic API shapes")
```

For each of the 67 targets, independently reconstruct
`P = P_H + sum_j sp_j P_j` on CPU from the authenticated DTO after the CUDA
solve has completed, then invoke test-only CPU LAPACK through:

```r
cpu_factor <- suppressWarnings(chol(P, pivot = TRUE, tol = -1))
cpu_root_rank <- as.integer(attr(cpu_factor, "rank"))
cpu_pivot <- as.integer(attr(cpu_factor, "pivot"))

assert_true(info$aggregate_penalty_root_rank[[target_index]] == cpu_root_rank,
            "aggregate root rank matches CPU LAPACK")
assert_true(identical(
  info$aggregate_penalty_root_pivot[[target_index]], cpu_pivot
), "aggregate root pivot matches CPU LAPACK")
```

`chol(..., pivot = TRUE, tol = -1)` is test-only LAPACK structural evidence.
Neither its root, rank, pivot, nor fitted/residual result is passed to CUDA or
used as production solve authority. Add a synthetic equal-leading-diagonal
case and require the first remaining canonical position to win every pivot
tie. Add a second exact diagonal case whose next pivot lies strictly between
`q * (epsilon / 2) * maxdiag` and `q * epsilon * maxdiag`; require the LAPACK
rank so using unhalved `epsilon` fails the test.

Separately form the test-only CPU aggregate-root augmented matrix with the same
zero-row padding as CUDA. Compute its solve-rank reference exactly as
`C_magic::fit_magic` does:

```r
svd_rank_tol <- sqrt(.Machine$double.eps)
svd_rank_threshold <- max(aggregate_svd$d) * svd_rank_tol
expected_effective_rank <- sum(aggregate_svd$d >= svd_rank_threshold)
```

Require `effective_rank == expected_effective_rank`. Keep
`aggregate_penalty_root_rank`, `effective_rank`, authenticated Phase 1
`coefficient_rank`, and their differences as distinct diagnostics; never force
equality among ranks defined by different matrices or tolerances. The CPU
67-target aggregate prototype that reached about `9.03e-10` maximum `C_magic`
error used this threshold. A stacked independent-root CPU SVD may be recorded
under an explicitly non-authoritative diagnostic name, but none of its
fitted/residual values may populate an expected value or pass/fail assertion.

Retain the route/resource gates in this RED test: exact planned/executed route
counts and statuses, zero reroutes, zero post-warm-up allocation/workspace
growth, zero CPU/unknown/approximate fallback, zero target-level sync, zero
aggregate matrix/root D2H, exactly one compact SVD checkpoint per affected
public batch (zero for an unaffected batch),
`aggregate_penalty_factor_count ==
sum(info$aggregate_factor_call_count) == 67L`,
`aggregate_svd_b_build_count == sum(info$aggregate_b_build_count) == 134L`, and
unchanged canonical output order. In the mixed-route and forced true-batch
reroute regressions, apply the same target mask and require exact `1/2` counts
for every executed SVD target, including reroutes, and exact `0/0` for every
non-SVD target. Also require all five aggregate vectors/lists to have outer
length `T`, with non-SVD rank/dstop `NA`, non-SVD pivots `integer(0)`, and every
executed-SVD pivot a length-`Q` one-based permutation.

Keep the canonical caller reserve at logical `augmented_rows = 407L`, then
extend the lifecycle RED assertions to require exactly
`augmented_workspace_bytes == 8 * max(407L, 351L + 64L) * 64L`, i.e. internal
`m = 415`, plus nonzero `aggregate_factor_workspace_bytes`. In the existing
cross-dimension reserve sequence, merging `n = 400` into the prior `q = 64`
must grow the stable row capacity to `464`, and the following explicit
`n = 400, q = 64` reserve must reuse it without another allocation or query.
Mixed and true-batch tests continue passing only their logical QR row capacity;
their declared/rerouted SVD work must fit `n + q` with zero solve-time growth.

Extend the synthetic constrained/non-null-H case with `q = 3`, zero target SP,
and exact projected
`P_H = diag(1, 0.60*q*.Machine$double.eps,
0.90*q*.Machine$double.eps)`. Its eigendecomposition-derived `H_root` drops both
small directions at `q*epsilon`, while upper DPSTF2 retains both at
`q*(epsilon/2)`. Require the test-only prepared/static shadow's `projected_H`
matrix to equal `P_H`, a separate projected-H test-shadow D2H delta of one call
and exactly `8*q*q` bytes, and `crossprod(H_root) != projected_H`. Require the
device aggregate rank/pivot to follow exact `projected_H` (`3`, `c(1L,3L,2L)`)
and not the truncated reconstruction (`1`, canonical trailing pivot order).
Production aggregate/root D2H counters remain zero.

- [ ] **Step 2: Run and verify the landed stacked-SVD failure**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
```

Expected: FAIL because landed commit `8bd6142` uses stacked roots, lacks the
aggregate rank/pivot diagnostics, and misses `C_magic` on the known `14 / 67`
targets; lifecycle and synthetic-H assertions also fail on missing aggregate
storage/exact projected-H ownership. This is the required RED for the follow-up
correction; do not reset to the earlier `ERR_STABLE_PATH_NOT_IMPLEMENTED`
milestone and do not accept the route-specific CPU augmented reference.

- [ ] **Step 3: Build and factor the aggregate penalty entirely on device**

As the reopened-Task-5 resource delta, extend the landed Task 1 workspace with:

```cpp
double* aggregate_penalty_factor = nullptr;  // q x q, overwritten by U
double* aggregate_factor_work = nullptr;     // 2q DPSTF2 work values
```

Do not change the caller meaning of `FixedSpCapacities::augmented_rows`: it is
the logical stacked-QR capacity and remains `407` for the canonical runner.
Compute merged logical capacities before the reserve fast path, then derive:

```cpp
const int effective_stable_rows = checked_max(
    merged_capacities.augmented_rows,
    checked_add(merged_capacities.n, merged_capacities.null_dim));
```

The within-capacity check includes
`effective_stable_rows <= stable_workspace.max_rows`. The stable-dimension
growth check, probe allocation, probe view, cuSOLVER workspace queries, final
stable workspace view, and `augmented_workspace_bytes` all use
`effective_stable_rows`; `capacities.augmented_rows` retains only the merged QR
logical value. Canonical reserve is therefore `max(407, 351 + 64) = 415` rows.
Because this happens inside C++, every declared SVD and every mixed/true-batch
Cholesky/QR reroute gets `n + q` capacity without an R caller guessing the
device root rank. Equal/smaller reserves allocate and query nothing, and
solve-time growth remains forbidden.

Reserve the `q x q` factor and `2q` work doubles before warm-up and expose
`aggregate_factor_workspace_bytes = sizeof(double) * (q*q + 2*q)`. For reserved
target capacity `T` and nullspace capacity `Q`, preserve the existing six stable
integer arrays in their current order, then append exactly this layout to both
the device integer arena and its pinned-host mirror:

```text
aggregate_root_rank[T]
aggregate_factor_call_count[T]
aggregate_b_build_count[T]
aggregate_pivots[T * Q]          # target-major, native zero-based
```

This is exactly `T * (Q + 3)` additional integers. The current target's
fixed-stride pivot row is the factor's pivot scratch and retained diagnostic;
do not reserve a second standalone `Q`-integer pivot buffer. Including the
existing six arrays, the stable compact count is `T * (Q + 9)`. Including the
existing legacy `3T + 1` prefix and shared stable `info`, the full integer arena
and mirrored pinned integer prefix require `T * (Q + 12) + 2` integers.

Add `aggregate_dstop[T]` after the existing device `sigma_max[T]` and
`smallest_retained_sigma[T]` arrays. In pinned host storage, first lay out the
entire enlarged integer prefix, recompute the byte alignment upward to
`alignof(double)`, and then mirror `sigma_max[T]`,
`smallest_retained_sigma[T]`, and `aggregate_dstop[T]` as contiguous `3T`
doubles. Update all capacity/binding assertions to these exact offsets. This is
a follow-up to `9be58b6`, not a rewrite of that commit.

Extend the prepared handle with an exact resident projected-H buffer:

```cpp
double* d_projected_H = nullptr;  // exact q x q P_H when H is non-null
```

At prepared-handle creation, upload the already computed exact `Z' H Z` matrix
once and retain it separately from `d_H_root`; free it with the handle and count
its bytes in setup H2D/resource diagnostics. A non-null-H aggregate starts from
`d_projected_H`. It must never reconstruct `P_H` as `crossprod(d_H_root)`.
When `H` is null, initialize the aggregate factor buffer to exact zero. Extend
the synthetic constrained/non-null-H SVD test to prove the aggregate uses the
exact projected matrix while the existing individual H-root diagnostic remains
unchanged.

Extend `PreparedSStaticShadow` with `has_H` and a column-major
`projected_H[q*q]` field, and extend the existing test-only
`C_fixed_sp_cuda_test_prepared_static_shadow` R result with a `q x q`
`projected_H` matrix or `NULL`. Add separate `PreparedSInfo` fields
`projected_H_test_shadow_d2h_count` and
`projected_H_test_shadow_d2h_bytes`; increment them only after a successful
non-null-H shadow copy, by one and exactly `q*q*sizeof(double)`. Do not combine
them with root-shadow counters. No production path calls this observer.

Add an SVD-only entry point whose inputs are resident projected `P_j`, resident
`P_H`, and the device pointer to that target's SP column. It accepts no host
root, rank, or pivot metadata. In ascending canonical penalty ordinal, build
the upper triangle of:

```text
P(sp) = P_H + sum_j SP[j,target] * P_j
```

Use one deterministic block of at most 64 threads for canonical `q <= 64`, or
an equivalently ordered kernel. Freeze:

```cpp
const double unit_roundoff =
    std::numeric_limits<double>::epsilon() / 2.0;
const double dstop =
    static_cast<double>(q) * unit_roundoff * max_initial_diagonal;
```

Write that value to the target's device `aggregate_dstop` slot. Before route
execution, initialize all aggregate ranks/pivots to `-1`, both per-target
lifecycle counts to zero, and aggregate dstop to quiet NaN so non-SVD entries
are unambiguous.

Implement upper `DPSTF2` operation order, not a merely algebraically equivalent
factorization:

1. initialize pivots to canonical `0:(q - 1)` and accumulated dot products to
   zero;
2. select the initial maximum diagonal with strict `>` comparisons so ties
   retain the first position; zero/nonfinite initial maxima stop at rank zero;
3. at each step `j > 0`, update `work[i] += A[j-1,i]^2` and
   `candidate[i] = A[i,i] - work[i]` for `i = j:(q - 1)` in increasing order;
4. select the first maximum candidate, and stop before a nonfinite pivot or a
   pivot `<= dstop`;
5. when pivoting, apply the upper-storage diagonal, leading-column,
   trailing-row, and middle cross-segment swaps in LAPACK order, then swap the
   accumulated work and pivot entries;
6. take the pivot square root, update the remaining upper row in fixed
   left-to-right DPSTF2 order, and continue.

Write `aggregate_penalty_root_rank` and the full native zero-based pivot vector
to device diagnostics. For rank `r`, scatter the leading upper factor rows back
to original coefficient columns:

```text
R_aggregate[row, pivot[col]] = U[row, col],  row < r
```

For `r < q`, this is LAPACK's rank-revealed approximation:
`crossprod(R_aggregate)` omits the DPSTF2 remainder and is not required to equal
`P(sp)`. Gate rank/pivot against CPU LAPACK and fitted/residual output against
`C_magic`; do not add an exact aggregate crossproduct assertion.

Write those `r` rows after `X0` and zero the remaining `q - r` root slots. Use
the fixed host-known `rows = n + q`, preserving the existing deterministic SVD
enqueue without a rank D2H checkpoint while the logical root remains compact.
Retain the factor and pivots in `aggregate_penalty_factor` after this first
emission, and increment that target's device
`aggregate_factor_call_count` exactly once. No allocation, CPU fallback,
factor/root D2H, host wait, or host factor metadata is permitted between
aggregation and SVD consumption.

- [ ] **Step 4: Run the existing deterministic economy Jacobi SVD**

For each SVD target, consume `[X0; R_aggregate; zero padding]` and call:

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
sweeps, and sorting configuration in runtime/manifest diagnostics. Its
`1e-12` tolerance is a Jacobi convergence control only and must never be reused
as the solve-rank threshold.

- [ ] **Step 5: Apply deterministic singular-rank truncation**

Compute on device:

```text
svd_rank_tol = sqrt(double_epsilon)
rank_threshold = sigma_max * svd_rank_tol
effective_rank = count(sigma_i >= rank_threshold)
scaled_i = (U[:,i]' c) / sigma_i when sigma_i >= rank_threshold, else 0
theta = V * scaled
```

Use cuBLAS `Dgemv` for `U'c` and `V*scaled`; use a deterministic elementwise
kernel for truncation/division. With `econ = 1` and `rows >= q`, cuSOLVER
stores economy `U` as `rows x q` and right singular vectors as columns of the
`q x q` matrix `V`; therefore the two GEMVs are exactly `U'c` followed by
`V*scaled`. `effective_rank` is the rank of this padded augmented SVD; it is not
`aggregate_penalty_root_rank`. This exactly follows `C_magic::fit_magic`:
singular values strictly below `sigma_max * sqrt(double_epsilon)` are dropped,
with equality retained and no row-count multiplier. Do not transpose `V`,
invert, solve normal equations, or substitute a CPU coefficient vector.

- [ ] **Step 6: Re-emit B/c once for the existing coefficient correction**

`gesvdj` overwrites the first `B` build. Keep `aggregate_penalty_factor`, its
pivots/rank, `U`, singular values, and `V` live. Rebuild `c`, rewrite `X0`, and
re-emit the same retained aggregate root plus zero padding into `B`; do not
reaggregate or refactor `P(sp)`, and do not call `gesvdj` again. Then preserve
the landed one-step correction:

```text
correction_residual = c - B theta
correction_scaled_i = (U[:,i]' correction_residual) / sigma_i
                      when sigma_i >= rank_threshold, else 0
theta = theta + V correction_scaled
```

Use the same `C_magic` singular mask as the first solution. Record exactly two
`B/c` builds and one aggregate factorization in that target's device counters.
Increment `aggregate_b_build_count` after each successful emission, yielding
exact per-SVD values `aggregate_factor_call_count = 1` and
`aggregate_b_build_count = 2`; initialized non-SVD entries remain `0/0`. The
batch/global `aggregate_penalty_factor_count` and
`aggregate_svd_b_build_count` are recomputed host sums of those vectors. The
second build is root re-emission from the retained factor, not a second factor
event.

- [ ] **Step 7: Finalize outputs and diagnostics**

Reuse canonical output kernels. Set `OK_AUGMENTED_SVD`, effective rank,
`sigma_max`, smallest retained sigma, SVD info, and
`target_true_batched = FALSE`. Store native pivots in fixed-stride
`target_count x q` device metadata. Define `DeviceResidualInfo` with
`aggregate_penalty_root_rank`, `aggregate_factor_call_count`, and
`aggregate_b_build_count` integer vectors of length `T`, one flattened native
`aggregate_penalty_root_pivot` integer vector of length `T*Q`, and one
`aggregate_dstop` double vector of length `T`. Copy the contiguous enlarged
integer region once and the contiguous three-double-array region once into
pinned host storage at the single batch SVD checkpoint after all
declared/rerouted SVD work is enqueued, then wait once; no target-level host
wait or diagnostic copy is allowed.

The R wrapper validates every shape. It returns rank and count integer vectors
of length `T`, with non-SVD rank converted from `-1` to `NA_integer_` and counts
left at `0/0`; it returns `aggregate_dstop` as a numeric length-`T` vector with
non-SVD NaN converted to `NA_real_`. It converts the flattened pivots to a list
of length `T`: `integer(0)` for each non-SVD target and one length-`Q`, one-based
canonical permutation for each executed SVD target. This metadata copy is not a
matrix/root materialization, and `aggregate_penalty_root_d2h_count` remains
zero. No factor metadata is returned to CUDA.

For a planned SVD target, set `executed_route = AUGMENTED_SVD` with an empty
reroute reason. For a Cholesky or QR reroute, preserve its original
`planned_route`, set `executed_route = AUGMENTED_SVD`, retain exactly
`CHOLESKY_NON_POSITIVE_PIVOT` or `QR_RANK_GUARD_REJECTED`, and require the
final `solver_status = OK_AUGMENTED_SVD` before counting the reroute as
successfully executed.

- [ ] **Step 8: Run SVD, QR, root, and Phase 3B tests**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_qr.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
```

Expected: all PASS; all 98 formerly unsupported iteration targets now solve,
all 67 SVD targets use `C_magic`, LAPACK aggregate rank/pivot matches `67 / 67`,
aggregate-root rank remains separate from effective SVD rank, and lifecycle
evidence records per-target SVD `1/2`, non-SVD `0/0`, and recomputed totals of
exactly 67 factors / 134 `B/c` builds. Canonical, mixed, and true-batch-reroute
reserves pass logical QR rows while internal stable rows cover `n + q` without
solve-time growth.

- [ ] **Step 9: Commit the aggregate-root SVD correction**

```bash
git add fastkpc/src/cuda/mgcv_fixed_sp_runtime_types.hpp \
  fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cuh \
  fastkpc/src/cuda/mgcv_fixed_sp_stable.cu \
  fastkpc/src/r_api_cuda.cpp fastkpc/R/cuda_native.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_runtime_lifecycle.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_penalty_roots.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_svd.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_true_batch.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_mixed_batch.R
git commit -m "fix: align fixed-sp SVD with aggregate penalty semantics"
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
  planned_cholesky_count = 172L,
  planned_qr_count = 31L,
  planned_svd_count = 67L,
  executed_cholesky_count = 172L,
  executed_qr_count = 31L,
  executed_svd_count = 67L,
  cholesky_to_svd_count = 0L,
  qr_to_svd_count = 0L,
  true_batched_target_count = 160L,
  cholesky_single_target_count = 12L,
  whole_batch_true_batched_count = 5L,
  stable_not_implemented_count = 0L,
  stable_reroute_count = 0L,
  non_ok_status_count = 0L,
  root_rank_mismatch_count = 0L,
  aggregate_penalty_factor_count = 67L,
  aggregate_svd_b_build_count = 134L,
  aggregate_penalty_root_rank_mismatch_count = 0L,
  aggregate_penalty_root_pivot_mismatch_count = 0L,
  aggregate_penalty_root_d2h_count = 0L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  invalid_output_init_count = 44L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
```

Call Phase 2 `C_magic` exactly once for every one of the 270 targets. Set
`numeric_reference = "mgcv-fixed-sp"` for all rows and require each row's
residual/fitted max-absolute and relative-L2 error `< 1e-7`. Delete the
route-specific branch that replaces SVD `reference_fitted` or
`reference_residuals` with CPU stacked augmented-SVD output. Such output may
remain only in explicitly non-authoritative diagnostic fields and cannot
affect an error, expected value, hash authority, or gate.

Require finite outputs, canonical key order, and exact repeated
planned/executed route, reroute-reason, and solver-status fields. Every target
row contains `planned_route`, `executed_route`, `reroute_reason`,
`solver_status`, and `numeric_reference`. Recompute and require route
conservation even though both reroute counts are zero. For all 67 SVD rows,
recompute target `P(sp)` and test-only CPU LAPACK rank/pivot as in Task 5;
require exact `aggregate_penalty_root_rank` and
`aggregate_penalty_root_pivot`, then separately require GPU `effective_rank`
to equal the CPU aggregate-root padded augmented-system rank under
`sigma_max * sqrt(.Machine$double.eps)`, with equality retained. Record Phase 1
`coefficient_rank` separately and never conflate the three ranks.

Recompute `aggregate_penalty_factor_count` and
`aggregate_svd_b_build_count` as sums of the per-target diagnostic vectors.
Before checking the `67/134` sums, require every one of the 67 executed SVD rows
to have `aggregate_factor_call_count == 1L` and
`aggregate_b_build_count == 2L`, and every one of the 203 non-SVD rows to have
exactly `0L/0L`. Global totals are recomputed evidence, not substitutes for
these row invariants. Preserve
`root_rank_mismatch_count` as the existing setup-time individual-root
diagnostic; aggregate rank/pivot mismatch counters are separate target-time
diagnostics. Retain every route/resource gate: QR/SVD checkpoint waits occur at
most once per affected public batch and never scale with target count; no
post-warm-up allocation, workspace growth, target-level sync, aggregate
matrix/root D2H, fallback, decision relaxation, or determinism exception is
allowed.

The iteration runner continues to pass the census-derived logical QR maximum
`augmented_rows = 407L`. Require exact canonical runtime evidence
`augmented_workspace_bytes == 8 * 415L * 64L` and zero post-warm-up growth; the
runner must not add `q` or infer a device aggregate rank at any R reserve call.

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
from solve metrics. The CPU LAPACK aggregate factor and optional stacked CPU
SVD are test-shadow diagnostics only. The runner always computes parity and
numeric hashes against the authoritative `C_magic` fitted/residual vectors.

- [ ] **Step 4: Run the full iteration gate twice**

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
```

Expected: both runs PASS and their target planned/executed-route,
reroute-reason, solver-status, and numeric hashes are identical.

- [ ] **Step 5: Commit the all-route C_magic iteration closure**

```bash
git add fastkpc/R/full_cuda_ci_fixed_sp_runtime.R \
  fastkpc/tests/test_mgcv_fixed_sp_cuda_phase3c_iteration.R
git commit -m "test: close fixed-sp iteration parity against C_magic"
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
  planned_cholesky_count = 3889L,
  planned_qr_count = 190L,
  planned_svd_count = 2064L,
  executed_cholesky_count = 3889L,
  executed_qr_count = 190L,
  executed_svd_count = 2064L,
  cholesky_to_svd_count = 0L,
  qr_to_svd_count = 0L,
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
  aggregate_penalty_factor_count = 2064L,
  aggregate_svd_b_build_count = 4128L,
  aggregate_penalty_root_rank_mismatch_count = 0L,
  aggregate_penalty_root_pivot_mismatch_count = 0L,
  aggregate_penalty_root_d2h_count = 0L,
  workspace_grow_count_after_warmup = 0L,
  per_target_allocation_count_after_warmup = 0L,
  per_target_handle_create_count = 0L,
  cuda_device_synchronize_count = 0L,
  target_level_stable_sync_count = 0L,
  implicit_residual_d2h_count = 0L,
  all_output_slot_leases_released = TRUE,
  invalid_output_init_count = 2061L,
  cpu_fallback_count = 0L,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L
)
```

It also requires `numeric_reference = "mgcv-fixed-sp"` for every target,
route- and condition-bucket numeric maxima `< 1e-7`, all outputs finite,
device aggregate-root rank/pivot exact against test-only CPU LAPACK,
GPU/CPU aggregate-root padded augmented-SVD effective ranks exact under the
`C_magic` `sigma_max * sqrt(double_epsilon)` threshold, per-target
`aggregate_factor_call_count/aggregate_b_build_count` exactly `1/2` on all
2,064 SVD rows and `0/0` on all 4,079 non-SVD rows, all cuSOLVER info zero, and
no target-level stable-path synchronization. Recompute the qualification
`2,064/4,128` factor/build totals by summing those vectors; aggregate totals are
not sole lifecycle evidence. CPU structural checks never replace `C_magic`
fitted/residual authority.

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
assert_true(identical(gpu$diagnostics$planned_route, "AUGMENTED_SVD") &&
              identical(gpu$diagnostics$executed_route, "AUGMENTED_SVD"),
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
Y                    = y
oracle RHS only      = Xty_null
route metadata       = unauthenticated -> AUGMENTED_SVD
```

The private adapter also receives the existing `X`, `y`, and `Z`, constructs
`X_null = X Z`, then delegates aggregate-penalty construction, upper
DPSTF2-compatible factorization, original-order root emission, and augmented
SVD to the same device path as Task 5. It must not build a host aggregate root.
Before creating CUDA state, independently recompute
`crossprod(X_null)` and `crossprod(X_null, y)` and require parity with the
caller-supplied `XtX_null` and `Xty_null` under a fixed `1e-12` relative/absolute
check. `Xty_null` is compatibility validation evidence only; the production
solve computes `X_null' * y` on CUDA. This adapter must not forge canonical
Phase 2 fingerprints or pass its
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
  one-time individual QR roots, qualification matrices/rows 6,272 / 63,552
  augmented QR and deterministic aggregate-penalty augmented SVD
  aggregate root rank/pivot exact against test-only CPU LAPACK
  C_magic numeric reference for every route and target
  C_magic sqrt(double epsilon) SVD rank threshold
  aggregate SVD one factor / two B builds per executed target
  iteration 270/270 targets OK
  qualification 6,143/6,143 targets OK
  planned routes 3,889 / 190 / 2,064 exact
  executed routes 3,889 / 190 / 2,064; declared reroutes 0 / 0
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
