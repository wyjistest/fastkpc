# Fast kPC CUDA Residual Kernels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the `fastSpline` residual fit used by the CUDA skeleton/WAN-PDAG path from CPU-only execution to an opt-in CUDA residual device, while preserving CPU equivalence, fallback behavior, campaign diagnostics, and unchanged `kpcalg/R` files.

**Architecture:** Keep the existing CPU `fastSpline` solver as the numerical reference and add a CUDA implementation behind explicit `residual_device` plumbing. The first layer exposes standalone CUDA residual fit APIs for validation; the second layer integrates the same CUDA residual path into the CUDA skeleton residual cache; the final layer surfaces the device selection through `fast_kpc()`, campaign output, reports, and CLI without changing legacy `kpcalg::kpc()`.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5, cuBLAS, cuSOLVER, existing `fastkpc/R/native.R`, `fastkpc/R/cuda_native.R`, `fastkpc/R/fast_kpc.R`, existing exact dCov CUDA batch backend, base R tests and report writer.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-14-fast-kpc-cuda-residual-kernels-goal-execution.md: add an opt-in CUDA fastSpline residual device, validate standalone residual equivalence against the CPU fastSpline reference, integrate CUDA residuals into the CUDA skeleton/WAN-PDAG residual cache, extend public wrapper/campaign/report/CLI diagnostics, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `520000`.

Do not mark the goal complete until every criterion in Phase 13 is satisfied.

## Current Baseline

The previous goal completed:

```text
fast_kpc()
fastkpc_result contract
validation scenarios
run_fastkpc_validation_campaign()
write_fastkpc_validation_report()
run_fast_kpc.R
run_validation_campaign.R
CPU/CUDA WAN-PDAG public wrapper validation
legacy missing-package diagnostics
```

The final campaign from the previous goal passed:

```text
40/40 runs ok
20/20 CPU-vs-CUDA PDAG identical
max CPU-vs-CUDA pMax diff about 9.214851e-15
legacy unavailable reason: missing package(s): pcalg, graph
kpcalg/R MD5 all OK
```

Current important implementation facts:

```text
fast_skeleton_cuda_backend(..., residual_backend = "fastSpline") already works.
It still computes fastSpline residuals on CPU via ResidualCache -> compute_residuals_with_backend().
CUDA currently accelerates batched exact dCov only.
The existing fastSpline basis uses quantile knots, normalized cubic radial basis columns, tensor basis for |S|=2, and additive basis for |S|>2.
The existing fastSpline solver scans a log lambda grid and solves small penalized systems in double precision.
```

## Scope

In scope:

- Add a standalone CUDA fastSpline residual fit API:

```r
fastspline_residual_cuda(y,
                         S,
                         fastspline_params = list(),
                         fallback = TRUE)
```

- Add a batch CUDA residual API for skeleton integration:

```r
fastspline_residual_batch_cuda(data,
                               targets,
                               conditioning_sets,
                               fastspline_params = list(),
                               fallback = TRUE)
```

- Add `residual_device = c("auto", "cpu", "cuda")` to CUDA skeleton/WAN-PDAG wrappers and the public `fast_kpc()` wrapper.
- Keep `residual_backend = "fastSpline"` as the statistical backend name.
- Record the execution device separately as `residual_device`, with values `cpu`, `cuda`, or `cuda-fallback-cpu`.
- Integrate CUDA residuals into the CUDA skeleton residual cache.
- Keep CPU skeleton and CPU WAN-PDAG behavior unchanged.
- Preserve CPU-vs-CUDA graph equality under `residual_device = "cuda"` within tight numeric tolerances.
- Add validation campaign/report/CLI support for `residual_device`.
- Add benchmark helpers that compare CPU residuals vs CUDA residuals in standalone residual fits and full CUDA skeleton runs.
- Keep all old tests passing.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not implement layer-batched PC scheduling across many CI tests beyond the existing CUDA skeleton batching.
- Do not implement multi-GPU scheduling.
- Do not replace `kpcalg::kpc()`.
- Do not change legacy `kpcalg/R/*.R`.
- Do not change the mathematical fastSpline basis or GCV criterion in this goal.
- Do not require pcalg/graph for package-independent tests.
- Do not remove the CPU residual path.
- Do not rewrite WAN-PDAG orientation rules.

## Design Contract

### Residual Device Contract

`residual_backend` describes the statistical residual model:

```text
linear
fastSpline
```

`residual_device` describes where that residual model is executed:

```text
cpu
cuda
auto
```

Resolved runtime values:

```text
cpu
cuda
cuda-fallback-cpu
```

Rules:

```text
1. residual_device="cpu" always uses the existing CPU residual backend.
2. residual_device="cuda" uses CUDA residual kernels for fastSpline and errors clearly if CUDA is unavailable and fallback=FALSE.
3. residual_device="auto" uses CUDA residual kernels only when CUDA is available and the selected backend supports CUDA residuals.
4. linear residuals remain CPU-only in this goal.
5. If residual_backend="linear" and residual_device="cuda", resolve to cpu with diagnostic reason "linear residual CUDA device is not implemented".
6. If CUDA fastSpline fails and fallback=TRUE, return cpu residuals and record residual_device="cuda-fallback-cpu".
7. Existing calls that omit residual_device must behave as before except for additional diagnostics fields.
```

### Standalone CUDA Residual Result Contract

`fastspline_residual_cuda()` returns:

```text
residuals
fitted
selected_lambda
gcv
rss
edf
design_cols
ridge_attempts
backend
residual_backend
residual_device
fallback_used
diagnostics
```

Required values:

```text
backend = "cuda"
residual_backend = "fastSpline"
residual_device in c("cuda", "cuda-fallback-cpu")
length(residuals) == length(y)
length(fitted) == length(y)
all residuals finite
all fitted finite
selected_lambda finite and positive
gcv finite
rss finite and non-negative
edf finite and positive
design_cols positive integer
ridge_attempts non-negative integer
```

### Batch CUDA Residual Result Contract

`fastspline_residual_batch_cuda()` returns:

```text
residuals
fitted
selected_lambda
gcv
rss
edf
design_cols
ridge_attempts
residual_device
fallback_used
diagnostics
```

Shapes:

```text
residuals: n x batch numeric matrix
fitted: n x batch numeric matrix
selected_lambda: length batch
gcv: length batch
rss: length batch
edf: length batch
design_cols: length batch
ridge_attempts: length batch
```

`targets` are one-based R column indices. `conditioning_sets` is a list of one-based integer vectors.

### Skeleton Integration Contract

For CUDA skeleton/WAN-PDAG wrappers:

```r
fast_skeleton_cuda_backend(..., residual_device = c("auto", "cpu", "cuda"))
fast_kpc_wanpdag_cuda(..., residual_device = c("auto", "cpu", "cuda"))
fast_kpc(..., residual_device = c("auto", "cpu", "cuda"))
```

Required result additions:

```text
skeleton$residual_device
skeleton$residual_device_requested
skeleton$residual_device_reason
skeleton$residual_cache$residual_device
orientation$residual_device
orientation$residual_device_requested
```

Public `fastkpc_result$config` must add:

```text
residual_device_requested
residual_device_used
```

Public `fastkpc_result$diagnostics` must add:

```text
cuda_residual_available
cuda_residual_reason
```

### Numerical Tolerances

Standalone residual equivalence:

```text
max(abs(cpu$residuals - cuda$residuals)) < 1e-7 for |S| in 0,1,2,3 on deterministic small/medium fixtures.
max(abs(cpu$fitted - cuda$fitted)) < 1e-7.
abs(cpu$rss - cuda$rss) / max(1, cpu$rss) < 1e-8.
selected lambda must be identical or adjacent on the lambda grid.
```

Graph-level equivalence:

```text
CUDA skeleton with residual_device="cuda" must match residual_device="cpu":
  adjacency identical
  sepsets identical after sorting
  n.edgetests identical
  max(abs(pMax_cuda - pMax_cpu)) < 1e-7

Public fast_kpc engine="cuda" with residual_device="cuda" must match engine="cuda" residual_device="cpu" on the fixed smoke scenarios:
  skeleton adjacency identical
  orientation pdag identical
  orientation counts identical
```

### CUDA Implementation Contract

The CUDA implementation must:

```text
1. Use CPU code only for deterministic knot/penalty metadata construction unless the task explicitly moves that piece to CUDA.
2. Evaluate the design matrix on GPU.
3. Form XtX and Xty on GPU.
4. Scan the lambda grid using double precision small-system solves.
5. Use FP64 for the small system and GCV quantities.
6. Disable TF32 for any cuBLAS path used for residual numerics.
7. Return explicit fallback diagnostics if CUDA allocation/solve fails and fallback=TRUE.
8. Throw a clear error if CUDA allocation/solve fails and fallback=FALSE.
```

Recommended implementation split:

```text
CPU:
  parse parameters
  build knot centers
  build penalty matrix
  prepare conditioning set metadata

CUDA:
  design evaluation
  XtX/Xty formation
  lambda-grid solve/GCV
  fitted/residual vector evaluation
```

## File Structure

Create:

- `fastkpc/src/cuda/fastspline_residual_cuda.hpp`  
  C++ interface for CUDA fastSpline residual fits.

- `fastkpc/src/cuda/fastspline_residual_cuda.cu`  
  CUDA kernels and cuBLAS/cuSOLVER orchestration for single and batched residual fits.

- `fastkpc/R/cuda_residual_validation.R`  
  R validation and benchmark helpers for CUDA residuals.

- `fastkpc/tests/test_cuda_fastspline_residual_kernel.R`
- `fastkpc/tests/test_cuda_fastspline_residual_batch.R`
- `fastkpc/tests/test_cuda_residual_device_skeleton.R`
- `fastkpc/tests/test_fastkpc_residual_device_public_api.R`
- `fastkpc/tests/test_cuda_residual_device_campaign.R`
- `fastkpc/tests/test_cuda_residual_device_report_cli.R`
- `fastkpc/tests/test_cuda_residual_benchmark.R`
- `fastkpc/tests/test_cuda_residual_docs_contract.R`

Modify:

- `fastkpc/src/fastkpc_types.hpp`  
  Add residual-device fields to skeleton options/results.

- `fastkpc/src/skeleton_engine_cuda.hpp`
- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Route fastSpline residual cache misses to CUDA residuals when requested.

- `fastkpc/src/r_api_cuda.cpp`  
  Register standalone residual APIs and pass residual-device options through CUDA skeleton/WAN-PDAG exports.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile `fastspline_residual_cuda.cu` and link `-lcublas -lcusolver`.

- `fastkpc/R/cuda_native.R`  
  Add R wrappers and `residual_device` args.

- `fastkpc/R/fast_kpc.R`  
  Add `residual_device` config, diagnostics, and pass-through.

- `fastkpc/R/validation_campaign.R`  
  Add `residual_devices`, table columns, and CPU/CUDA residual-device comparison rows.

- `fastkpc/R/report_writer.R`  
  Include residual-device columns in Markdown/CSV outputs automatically.

- `fastkpc/tools/run_fast_kpc.R`
- `fastkpc/tools/run_validation_campaign.R`  
  Add CLI flags.

- `fastkpc/README.md`
- `fastkpc/reports/README.md`

Do not modify:

- `kpcalg/R/*.R`
- `kpcalg/MD5`

## Phase 0: Baseline Verification

Purpose: prove the current public wrapper/campaign stage is green before changing CUDA residual internals.

- [ ] Run:

```bash
pwd
Rscript -e 'cat("R ", as.character(getRversion()), "\n", sep=""); for (p in c("Rcpp","RcppArmadillo","mgcv","pcalg","graph","RSpectra")) cat(p, ": ", requireNamespace(p, quietly=TRUE), "\n", sep="")'
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
Rcpp, RcppArmadillo, and mgcv are TRUE.
CUDA toolkit and at least one GPU are visible.
pcalg, graph, and RSpectra availability is recorded.
```

- [ ] Run:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
Rscript fastkpc/tests/test_full_framework_smoke.R
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

Expected:

```text
CUDA build succeeds.
CPU sourceCpp build succeeds.
All listed tests print PASS.
Every kpcalg/R MD5 line reports OK.
```

## Phase 1: Standalone CUDA Residual Kernel Red Test

Purpose: define the standalone single-fit API before writing CUDA residual code.

- [ ] Create `fastkpc/tests/test_cuda_fastspline_residual_kernel.R`.

Test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(101)
n <- 96
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
z3 <- runif(n, -2, 2)
y <- sin(z1) + cos(z2) + 0.25 * z3 + rnorm(n, sd = 0.08)

cases <- list(
  empty = matrix(numeric(0), nrow = n, ncol = 0),
  one = cbind(z1 = z1),
  two = cbind(z1 = z1, z2 = z2),
  three = cbind(z1 = z1, z2 = z2, z3 = z3)
)

params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

for (name in names(cases)) {
  S <- cases[[name]]
  cpu <- fastspline_residual(y, S, fastspline_params = params)
  cuda <- fastspline_residual_cuda(y, S, fastspline_params = params,
                                   fallback = FALSE)

  assert_true(is.list(cuda), paste(name, "CUDA result should be a list"))
  assert_true(cuda$backend == "cuda", paste(name, "backend should be cuda"))
  assert_true(cuda$residual_backend == "fastSpline",
              paste(name, "residual backend should be fastSpline"))
  assert_true(cuda$residual_device == "cuda",
              paste(name, "residual_device should be cuda"))
  assert_true(identical(cuda$fallback_used, FALSE),
              paste(name, "fallback should not be used"))
  assert_true(length(cuda$residuals) == n, paste(name, "residual length"))
  assert_true(length(cuda$fitted) == n, paste(name, "fitted length"))
  assert_true(all(is.finite(cuda$residuals)), paste(name, "finite residuals"))
  assert_true(all(is.finite(cuda$fitted)), paste(name, "finite fitted"))
  assert_true(is.finite(cuda$selected_lambda) && cuda$selected_lambda > 0,
              paste(name, "selected lambda should be positive"))
  assert_true(is.finite(cuda$gcv), paste(name, "gcv should be finite"))
  assert_true(is.finite(cuda$rss) && cuda$rss >= 0,
              paste(name, "rss should be finite"))
  assert_true(is.finite(cuda$edf) && cuda$edf > 0,
              paste(name, "edf should be positive"))
  assert_true(cuda$design_cols == cpu$design_cols,
              paste(name, "design_cols should match CPU"))

  assert_true(max_abs_diff(cuda$residuals, cpu$residuals) < 1e-7,
              paste(name, "residuals should match CPU"))
  assert_true(max_abs_diff(cuda$fitted, cpu$fitted) < 1e-7,
              paste(name, "fitted values should match CPU"))
  rel_rss <- abs(cuda$rss - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste(name, "rss should match CPU"))
}

cat("test_cuda_fastspline_residual_kernel.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
```

Expected:

```text
The test fails because fastspline_residual_cuda() does not exist yet.
```

## Phase 2: Implement Standalone CUDA Residual API

Purpose: add the minimum CUDA residual implementation that matches the CPU reference for one fit at a time.

- [ ] Create `fastkpc/src/cuda/fastspline_residual_cuda.hpp`.

Required interface:

```cpp
#ifndef FASTKPC_FASTSPLINE_RESIDUAL_CUDA_HPP
#define FASTKPC_FASTSPLINE_RESIDUAL_CUDA_HPP

#include "../fastspline_basis.hpp"
#include "../fastspline_solver.hpp"

#include <Rcpp.h>
#include <string>
#include <vector>

struct FastSplineCudaDiagnostics {
  bool cuda_used;
  bool fallback_used;
  std::string reason;
};

struct FastSplineCudaFit {
  FastSplineFit fit;
  FastSplineCudaDiagnostics diagnostics;
};

FastSplineCudaFit fit_fastspline_residuals_cuda(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const FastSplineParams& params,
  bool fallback);

std::vector<FastSplineCudaFit> fit_fastspline_residuals_cuda_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

#endif
```

- [ ] Create `fastkpc/src/cuda/fastspline_residual_cuda.cu`.

Implementation requirements:

```text
Use cuda_runtime.h, cublas_v2.h, cusolverDn.h.
Use Rcpp only at the host API boundary.
Use make_fastspline_design() for metadata parity at first.
Upload design.X, design.P, and y to device.
Use cuBLAS Dgemm/Dgemv or custom kernels to form XtX, Xty, fitted.
For each lambda, form A = XtX + lambda * P + ridge matrix in double precision.
Use cuSOLVER double Cholesky or LU fallback to solve A beta = Xty.
Compute fitted, residuals, rss, edf, and GCV.
Keep the CPU tie-break rule: lower GCV wins; if GCV ties within 1e-14, smaller lambda wins.
If any CUDA stage fails and fallback=TRUE, call fit_fastspline_residuals() and set fallback_used=TRUE.
If any CUDA stage fails and fallback=FALSE, throw runtime_error("CUDA fastSpline residual fit failed: <stage>").
```

The first working implementation may use one host loop over lambda values. It must not change the CPU `fastSpline` basis or solver.

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Required changes:

```text
Compile fastkpc/src/cuda/fastspline_residual_cuda.cu into fastkpc/build/fastspline_residual_cuda.o.
Link fastspline_residual_cuda.o into fastkpc/build/fastkpc_cuda.so.
Add -lcublas -lcusolver to the final link command.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Add include:

```cpp
#include "cuda/fastspline_residual_cuda.hpp"
```

Add `fit_to_list()` helper that returns all fields required by the standalone residual result contract.

Add exported C call:

```cpp
extern "C" SEXP C_fastspline_residual_cuda(SEXP ys,
                                           SEXP Ss,
                                           SEXP fastspline_paramss,
                                           SEXP fallbacks)
```

Behavior:

```text
Validate y numeric vector.
Validate S numeric matrix with nrow(S) == length(y), allowing zero columns.
Build data = cbind(y, S).
Use target=0 and conditioning_set=1:(ncol(data)-1) in zero-based C++ indices.
Call fit_fastspline_residuals_cuda(data, 0, cond, params, fallback).
Return the contract list.
```

Register it:

```cpp
{"C_fastspline_residual_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_cuda), 4},
```

- [ ] Modify `fastkpc/R/cuda_native.R`.

Add:

```r
fastspline_residual_cuda <- function(y, S, fastspline_params = list(),
                                     fallback = TRUE) {
  load_fastkpc_cuda_native()
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  .Call("C_fastspline_residual_cuda", as.numeric(y), S, fastspline_params,
        isTRUE(fallback), PACKAGE = "fastkpc_cuda")
}
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
```

Expected:

```text
CUDA build succeeds.
test_cuda_fastspline_residual_kernel.R prints PASS.
```

## Phase 3: Batch CUDA Residual Red Test

Purpose: define the batch residual API needed by skeleton cache integration.

- [ ] Create `fastkpc/tests/test_cuda_fastspline_residual_batch.R`.

Test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(102)
n <- 90
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
z3 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.08),
  x2 = cos(z1) + rnorm(n, sd = 0.08),
  x3 = sin(z2) + cos(z3) + rnorm(n, sd = 0.08),
  x4 = z1 * z2 + rnorm(n, sd = 0.08),
  x5 = rnorm(n)
)

targets <- c(1L, 2L, 3L, 4L)
conditioning_sets <- list(integer(0), 1L, c(1L, 2L), c(1L, 2L, 3L))
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)

assert_true(is.matrix(batch$residuals), "batch residuals should be a matrix")
assert_true(identical(dim(batch$residuals), c(n, length(targets))),
            "batch residual matrix dimension should match")
assert_true(is.matrix(batch$fitted), "batch fitted should be a matrix")
assert_true(identical(dim(batch$fitted), c(n, length(targets))),
            "batch fitted matrix dimension should match")
assert_true(length(batch$selected_lambda) == length(targets),
            "lambda length should match batch")
assert_true(length(batch$gcv) == length(targets), "gcv length should match batch")
assert_true(length(batch$rss) == length(targets), "rss length should match batch")
assert_true(length(batch$edf) == length(targets), "edf length should match batch")
assert_true(all(batch$residual_device == "cuda"),
            "all batch residual devices should be cuda")
assert_true(!any(batch$fallback_used), "fallback should not be used")

for (k in seq_along(targets)) {
  target <- targets[[k]]
  S_idx <- conditioning_sets[[k]]
  y <- data[, target]
  S <- if (length(S_idx) == 0L) {
    matrix(numeric(0), nrow = n, ncol = 0)
  } else {
    data[, S_idx, drop = FALSE]
  }
  cpu <- fastspline_residual(y, S, fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], cpu$residuals) < 1e-7,
              paste("batch residual", k, "should match CPU"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("batch fitted", k, "should match CPU"))
  rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste("batch rss", k, "should match CPU"))
}

cat("test_cuda_fastspline_residual_batch.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
```

Expected:

```text
The test fails because fastspline_residual_batch_cuda() does not exist yet.
```

## Phase 4: Implement Batch CUDA Residual API

Purpose: expose a batch interface without changing skeleton logic yet.

- [ ] Extend `fastkpc/src/r_api_cuda.cpp`.

Add exported C call:

```cpp
extern "C" SEXP C_fastspline_residual_batch_cuda(SEXP datas,
                                                 SEXP targetss,
                                                 SEXP conditioning_setss,
                                                 SEXP fastspline_paramss,
                                                 SEXP fallbacks)
```

Behavior:

```text
Validate data is numeric matrix and finite.
Validate targets are one-based integer indices in 1:ncol(data).
Validate conditioning_sets is a list with length equal to length(targets).
Convert targets and conditioning sets to zero-based C++ indices.
Call fit_fastspline_residuals_cuda_batch().
Return matrices/vectors following the batch result contract.
```

Register:

```cpp
{"C_fastspline_residual_batch_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_batch_cuda), 5},
```

- [ ] Extend `fastkpc/R/cuda_native.R`.

Add:

```r
fastspline_residual_batch_cuda <- function(data, targets, conditioning_sets,
                                           fastspline_params = list(),
                                           fallback = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fastspline_residual_batch_cuda", data, as.integer(targets),
        conditioning_sets, fastspline_params, isTRUE(fallback),
        PACKAGE = "fastkpc_cuda")
}
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
```

Expected:

```text
Both CUDA residual tests print PASS.
```

## Phase 5: Residual Device Skeleton Red Test

Purpose: define how CUDA residuals integrate into the CUDA skeleton cache.

- [ ] Create `fastkpc/tests/test_cuda_residual_device_skeleton.R`.

Test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

compare_sepsets_exact <- function(a, b) {
  for (i in seq_along(a)) {
    for (j in seq_along(a[[i]])) {
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) return(FALSE)
    }
  }
  TRUE
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(103)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.12),
  x2 = cos(z1) + rnorm(n, sd = 0.12),
  x3 = sin(z2) + rnorm(n, sd = 0.12),
  x4 = z1 * z2 + rnorm(n, sd = 0.12),
  x5 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

cuda_cpu_residual <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cpu",
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_cuda_residual <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  residual_cache = TRUE,
  batch_size = 0,
  fastspline_params = params
)

assert_true(cuda_cuda_residual$backend == "cuda",
            "skeleton backend should remain cuda")
assert_true(cuda_cuda_residual$residual_backend == "fastSpline",
            "residual_backend should be fastSpline")
assert_true(cuda_cuda_residual$residual_device == "cuda",
            "residual_device should be cuda")
assert_true(cuda_cuda_residual$residual_device_requested == "cuda",
            "requested residual device should be recorded")
assert_true(cuda_cuda_residual$residual_cache$residual_device == "cuda",
            "cache residual device should be cuda")
assert_true(cuda_cuda_residual$residual_cache$hits > 0,
            "CUDA residual cache should still have hits")
assert_true(cuda_cuda_residual$residual_cache$computations <
              cuda_cuda_residual$residual_cache$requests,
            "cache computations should be lower than requests")

assert_true(identical(cuda_cuda_residual$adjacency, cuda_cpu_residual$adjacency),
            "CUDA residual adjacency should match CPU residual")
assert_true(compare_sepsets_exact(cuda_cuda_residual$sepsets,
                                  cuda_cpu_residual$sepsets),
            "CUDA residual sepsets should match CPU residual")
assert_true(identical(cuda_cuda_residual$n.edgetests,
                      cuda_cpu_residual$n.edgetests),
            "CUDA residual n.edgetests should match CPU residual")
assert_true(max_abs_diff(cuda_cuda_residual$pMax, cuda_cpu_residual$pMax) < 1e-7,
            "CUDA residual pMax should match CPU residual")

linear_cuda_requested <- fast_skeleton_cuda_backend(
  data, alpha, 1L,
  residual_backend = "linear",
  residual_device = "cuda",
  residual_cache = TRUE
)
assert_true(linear_cuda_requested$residual_device == "cpu",
            "linear residual_device cuda request should resolve to cpu")
assert_true(grepl("linear residual CUDA device is not implemented",
                  linear_cuda_requested$residual_device_reason, fixed = TRUE),
            "linear cuda request should record reason")

cat("test_cuda_residual_device_skeleton.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
```

Expected:

```text
The test fails because fast_skeleton_cuda_backend() does not accept residual_device yet.
```

## Phase 6: Integrate Residual Device Into CUDA Skeleton

Purpose: use CUDA residual fits on residual-cache misses in the CUDA skeleton path.

- [ ] Modify `fastkpc/src/fastkpc_types.hpp`.

Add to `SkeletonOptions`:

```cpp
std::string residual_device_requested;
bool cuda_residual_fallback;
```

Add to `SkeletonResult`:

```cpp
std::string residual_device;
std::string residual_device_requested;
std::string residual_device_reason;
```

- [ ] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

Implementation requirements:

```text
Keep current CPU residual cache behavior when residual_device_requested is empty, "auto" resolving to CPU for linear, or "cpu".
For residual_backend_name == "fastSpline" and residual_device_requested in c("cuda", "auto"), use fit_fastspline_residuals_cuda() on cache misses.
Store cached residual vectors exactly as before so hit/miss semantics remain stable.
Set result.residual_device, result.residual_device_requested, and result.residual_device_reason.
For linear + residual_device_requested="cuda", use CPU residuals and reason "linear residual CUDA device is not implemented".
For CUDA fastSpline fallback, set result.residual_device="cuda-fallback-cpu" and reason from diagnostics.
```

Recommended minimal design:

```text
Do not modify the generic ResidualCache class in this phase.
Create a CUDA-specific cache wrapper in skeleton_engine_cuda.cpp that mirrors ResidualCacheKey and stats behavior, because CPU sourceCpp must not link CUDA.
Use make_residual_cache_key() for key parity.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Update:

```text
C_fast_skeleton_cuda_backend arity from 9 to 11.
C_fast_kpc_wanpdag_cuda arity from 12 to 14.
```

New args:

```text
residual_devices
cuda_residual_fallbacks
```

Pass through:

```cpp
skeleton_options.residual_device_requested = residual_device;
skeleton_options.cuda_residual_fallback = cuda_residual_fallback;
```

Add result fields in `skeleton_result_to_list()`:

```text
residual_device
residual_device_requested
residual_device_reason
residual_cache$residual_device
```

Add orientation result fields for CUDA WAN-PDAG:

```text
orientation$residual_device
orientation$residual_device_requested
```

The orientation stage may remain CPU residual in this goal, but the result must record it honestly:

```text
orientation$residual_device = "cpu"
orientation$residual_device_requested = skeleton requested device
```

- [ ] Modify `fastkpc/R/cuda_native.R`.

Update signatures:

```r
fast_skeleton_cuda_backend <- function(data, alpha, max_conditioning_size,
                                       residual_backend = "linear",
                                       residual_device = c("auto", "cpu", "cuda"),
                                       residual_cache = TRUE,
                                       index = 1,
                                       legacy_index = TRUE,
                                       batch_size = 0,
                                       fastspline_params = list(),
                                       cuda_residual_fallback = TRUE)
```

```r
fast_kpc_wanpdag_cuda <- function(data, alpha, max_conditioning_size,
                                  residual_backend = "fastSpline",
                                  residual_device = c("auto", "cpu", "cuda"),
                                  residual_cache = TRUE,
                                  index = 1,
                                  legacy_index = TRUE,
                                  batch_size = 0,
                                  orient_collider = TRUE,
                                  solve_confl = FALSE,
                                  rules = c(TRUE, TRUE, TRUE),
                                  fastspline_params = list(),
                                  cuda_residual_fallback = TRUE)
```

Use `match.arg(residual_device)`.

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
```

Expected:

```text
New residual-device skeleton test prints PASS.
Existing CUDA fastSpline skeleton and WAN-PDAG CUDA tests still print PASS.
```

## Phase 7: Public fast_kpc Residual Device Test

Purpose: expose residual-device selection through the public wrapper without breaking existing callers.

- [ ] Create `fastkpc/tests/test_fastkpc_residual_device_public_api.R`.

Test content:

```r
source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

set.seed(104)
n <- 100
z <- seq(-2, 2, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + rnorm(n, sd = 0.1),
  x3 = cos(z) + rnorm(n, sd = 0.1),
  x4 = z^2 + rnorm(n, sd = 0.1)
)

cpu_residual <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cpu",
  graph_stage = "wanpdag",
  seed = 104
)

cuda_residual <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  graph_stage = "wanpdag",
  seed = 104
)

assert_true(inherits(cuda_residual, "fastkpc_result"),
            "result should have fastkpc_result class")
assert_true(cuda_residual$config$residual_device_requested == "cuda",
            "config should record requested residual device")
assert_true(cuda_residual$config$residual_device_used == "cuda",
            "config should record used residual device")
assert_true(cuda_residual$skeleton$residual_device == "cuda",
            "skeleton should record cuda residual device")
assert_true(cuda_residual$skeleton$residual_cache$residual_device == "cuda",
            "cache should record cuda residual device")
assert_true(isTRUE(cuda_residual$diagnostics$cuda_residual_available),
            "diagnostics should report cuda residual availability")

assert_true(identical(cuda_residual$skeleton$adjacency,
                      cpu_residual$skeleton$adjacency),
            "public CUDA residual skeleton should match CPU residual")
assert_true(max_abs_diff(cuda_residual$skeleton$pMax,
                         cpu_residual$skeleton$pMax) < 1e-7,
            "public CUDA residual pMax should match CPU residual")
assert_true(identical(cuda_residual$orientation$pdag,
                      cpu_residual$orientation$pdag),
            "public CUDA residual pdag should match CPU residual")

default_result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "fastSpline",
  graph_stage = "skeleton"
)
assert_true(default_result$config$residual_device_requested == "auto",
            "default residual_device should be auto")
assert_true(default_result$config$residual_device_used == "cpu",
            "CPU engine should use CPU residual device")

cat("test_fastkpc_residual_device_public_api.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
```

Expected:

```text
The test fails because fast_kpc() does not accept residual_device yet.
```

## Phase 8: Implement Public Wrapper Residual Device

Purpose: propagate residual-device selection through public wrapper result contracts.

- [ ] Modify `fastkpc/R/fast_kpc.R`.

Update `fast_kpc()` signature:

```r
residual_device = c("auto", "cpu", "cuda"),
cuda_residual_fallback = TRUE,
```

Required behavior:

```text
Use match.arg(residual_device).
For engine_used == "cpu", residual_device_used must be "cpu".
For engine_used == "cuda", pass residual_device and cuda_residual_fallback to CUDA wrappers.
For graph_stage="skeleton", preserve orientation=NULL behavior.
Add config fields residual_device_requested, residual_device_used, cuda_residual_fallback.
Add diagnostics fields cuda_residual_available and cuda_residual_reason.
Add metrics unchanged.
Update validate_fastkpc_result() to require the new config fields.
Update print/summary to include residual_device_used.
```

- [ ] Update `fastkpc/tests/test_fastkpc_result_contract.R` expected config fields:

```r
"residual_device_requested"
"residual_device_used"
"cuda_residual_fallback"
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
```

Expected:

```text
All four tests print PASS.
```

## Phase 9: Campaign/Report/CLI Residual Device Test

Purpose: make residual-device comparisons first-class validation artifacts.

- [ ] Create `fastkpc/tests/test_cuda_residual_device_campaign.R`.

Test content:

```r
source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(105),
  n_values = c(70),
  scenarios = c("chain", "additive"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cpu", "cuda"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = TRUE
)

assert_true("residual_device" %in% names(campaign$runs),
            "runs should include residual_device")
assert_true(all(c("cpu", "cuda") %in% campaign$runs$residual_device),
            "runs should include cpu and cuda residual devices")
assert_true("residual_device_diffs" %in% names(campaign),
            "campaign should include residual_device_diffs")
assert_true(is.data.frame(campaign$residual_device_diffs),
            "residual_device_diffs should be data.frame")
assert_true(nrow(campaign$residual_device_diffs) == 2L,
            "one residual-device diff row per scenario")
assert_true(all(campaign$residual_device_diffs$pdag_identical),
            "CPU/CUDA residual-device pdag should match")
assert_true(all(campaign$residual_device_diffs$skeleton_adjacency_identical),
            "CPU/CUDA residual-device skeleton should match")
assert_true(all(campaign$residual_device_diffs$max_abs_pmax_diff < 1e-7),
            "CPU/CUDA residual-device pMax diff should be tiny")
assert_true(campaign$summary$total_runs == nrow(campaign$runs),
            "summary total_runs should match")

cat("test_cuda_residual_device_campaign.R: PASS\n")
```

- [ ] Create `fastkpc/tests/test_cuda_residual_device_report_cli.R`.

Test content:

```r
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(106),
  n_values = c(60),
  scenarios = c("chain"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cpu", "cuda"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE
)

output_dir <- tempfile("fastkpc-cuda-residual-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(file.path(output_dir, "residual_device_diffs.csv")),
            "report should write residual_device_diffs.csv")
assert_true(file.exists(artifacts$summary_md), "summary.md should exist")

summary_text <- paste(readLines(artifacts$summary_md, warn = FALSE), collapse = "\n")
assert_true(grepl("## Residual Device", summary_text, fixed = TRUE),
            "summary should contain residual device section")

report_dir <- tempfile("fastkpc-cuda-residual-cli-")
status <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", report_dir,
    "--seeds", "107",
    "--n-values", "60",
    "--scenarios", "chain",
    "--engines", "cuda",
    "--residual-backends", "fastSpline",
    "--residual-devices", "cpu,cuda",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--legacy", "FALSE")
)
assert_true(identical(status, 0L), "campaign CLI should accept residual-devices")
assert_true(file.exists(file.path(report_dir, "residual_device_diffs.csv")),
            "campaign CLI should write residual_device_diffs.csv")

input <- tempfile(fileext = ".csv")
output <- tempfile(fileext = ".rds")
z <- seq(-2, 2, length.out = 70)
utils::write.csv(data.frame(x1 = z, x2 = sin(z), x3 = cos(z), x4 = z^2),
                 input, row.names = FALSE)
status_one <- system2(
  "Rscript",
  c("fastkpc/tools/run_fast_kpc.R",
    "--input", input,
    "--output", output,
    "--engine", "cuda",
    "--residual-backend", "fastSpline",
    "--residual-device", "cuda",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--graph-stage", "wanpdag")
)
assert_true(identical(status_one, 0L), "single CLI should accept residual-device")
result <- readRDS(output)
assert_true(result$config$residual_device_requested == "cuda",
            "single CLI result should record residual-device request")

cat("test_cuda_residual_device_report_cli.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_cuda_residual_device_report_cli.R
```

Expected:

```text
The tests fail because campaign/report/CLI do not include residual_device yet.
```

## Phase 10: Implement Campaign, Report, And CLI Residual Device Support

Purpose: make residual-device validation reproducible through the public campaign layer.

- [ ] Modify `fastkpc/R/validation_campaign.R`.

Update signature:

```r
run_fastkpc_validation_campaign(...,
                                residual_devices = c("auto"),
                                ...)
```

Implementation requirements:

```text
Add residual_device to expand.grid.
Pass residual_device into fast_kpc().
Add residual_device to runs, graph_metrics, timings, cache, orientation_counts, errors.
Add campaign$residual_device_diffs.
For residual_device_diffs, match scenario, seed, n, engine="cuda", residual_backend="fastSpline", residual_device cpu vs cuda.
Compare pdag, skeleton adjacency, pMax, orientation counts.
Add summary fields residual_device_diff_rows, residual_device_pdag_identical, max_residual_device_pmax_diff.
Keep existing campaign calls that omit residual_devices working.
```

Required `campaign$residual_device_diffs` columns:

```text
scenario
seed
n
engine
residual_backend
pdag_identical
skeleton_adjacency_identical
max_abs_pmax_diff
orientation_counts_identical
status
```

- [ ] Modify `fastkpc/R/report_writer.R`.

Add artifact:

```text
residual_device_diffs.csv
```

Add Markdown heading:

```text
## Residual Device
```

- [ ] Modify `fastkpc/tools/run_fast_kpc.R`.

Add argument:

```text
--residual-device auto|cpu|cuda
```

Pass it to `fast_kpc()`.

- [ ] Modify `fastkpc/tools/run_validation_campaign.R`.

Add argument:

```text
--residual-devices comma-separated residual device names
```

Default:

```text
auto
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_cuda_residual_device_report_cli.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
```

Expected:

```text
All five tests print PASS.
```

## Phase 11: CUDA Residual Benchmark And Validation Helpers

Purpose: provide reusable diagnostics for future larger validation campaigns.

- [ ] Create `fastkpc/tests/test_cuda_residual_benchmark.R`.

Test content:

```r
source("fastkpc/R/cuda_residual_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

validation <- validate_cuda_fastspline_residuals(seed = 108, n = 96)
assert_true(is.data.frame(validation$cases), "validation cases should be data.frame")
assert_true(all(validation$cases$status == "ok"), "all validation cases should be ok")
assert_true(all(validation$cases$max_abs_residual_diff < 1e-7),
            "residual diffs should be tiny")
assert_true(all(validation$cases$max_abs_fitted_diff < 1e-7),
            "fitted diffs should be tiny")

bench <- benchmark_cuda_fastspline_residuals(seed = 109, n = 160, repeats = 2)
required <- c("case", "device", "repeat", "elapsed_sec", "residual_backend",
              "residual_device", "status")
assert_true(all(required %in% names(bench$timings)),
            "benchmark timings should have required columns")
assert_true(all(bench$timings$status == "ok"), "benchmark timings should be ok")
assert_true(any(bench$timings$residual_device == "cuda"),
            "benchmark should include cuda residual device")
assert_true(any(bench$timings$residual_device == "cpu"),
            "benchmark should include cpu residual device")

cat("test_cuda_residual_benchmark.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_benchmark.R
```

Expected:

```text
The test fails because cuda_residual_validation.R does not exist yet.
```

- [ ] Create `fastkpc/R/cuda_residual_validation.R`.

Required functions:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

validate_cuda_fastspline_residuals <- function(seed = 108, n = 96)
benchmark_cuda_fastspline_residuals <- function(seed = 109, n = 160, repeats = 3)
```

Validation fixture:

```text
Generate z1,z2,z3 and y deterministically.
Run cases empty, one, two, three.
Compare CPU fastspline_residual() to fastspline_residual_cuda().
Return cases data frame and raw result list.
```

Benchmark fixture:

```text
Run CPU and CUDA standalone residual fits for one, two, and three conditioning variables.
Repeat requested number of times.
Return timings data frame and summary.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_benchmark.R
```

Expected:

```text
test_cuda_residual_benchmark.R prints PASS.
```

## Phase 12: Documentation

Purpose: document residual-device behavior and generated artifacts.

- [ ] Create `fastkpc/tests/test_cuda_residual_docs_contract.R`.

Test content:

```r
readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports_readme <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                        collapse = "\n")

assert_contains <- function(text, pattern) {
  if (!grepl(pattern, text, fixed = TRUE)) {
    stop("missing required text: ", pattern, call. = FALSE)
  }
}

required_readme <- c(
  "CUDA Residual Device",
  "residual_device",
  "fastspline_residual_cuda",
  "fastspline_residual_batch_cuda",
  "CUDA residual kernels are opt-in",
  "linear residual CUDA device is not implemented",
  "CUDA residual fallback",
  "kpcalg::kpc() is not replaced",
  "kpcalg/R/*.R files are not modified"
)
for (pattern in required_readme) assert_contains(readme, pattern)

required_reports <- c(
  "residual_device_diffs.csv",
  "residual_device",
  "cuda-fallback-cpu"
)
for (pattern in required_reports) assert_contains(reports_readme, pattern)

cat("test_cuda_residual_docs_contract.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_docs_contract.R
```

Expected:

```text
The test fails until README files are updated.
```

- [ ] Update `fastkpc/README.md`.

Add sections:

```text
CUDA Residual Device
Standalone CUDA Residual API
Residual Device In fast_kpc
CUDA Residual Validation
CUDA Residual Known Limits
```

Required statements:

```text
CUDA residual kernels are opt-in.
linear residual CUDA device is not implemented.
CUDA residual fallback can resolve to cuda-fallback-cpu.
kpcalg::kpc() is not replaced.
kpcalg/R/*.R files are not modified.
```

- [ ] Update `fastkpc/reports/README.md`.

Add:

```text
residual_device_diffs.csv
residual_device
cuda-fallback-cpu
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_docs_contract.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
```

Expected:

```text
Both docs tests print PASS.
```

## Phase 13: Completion Criteria

The goal is complete only when all criteria are true:

```text
1. Standalone CUDA residual API exists:
   - fastspline_residual_cuda()
   - fastspline_residual_batch_cuda()
   - C_fastspline_residual_cuda
   - C_fastspline_residual_batch_cuda

2. CUDA residual implementation exists:
   - fastkpc/src/cuda/fastspline_residual_cuda.hpp
   - fastkpc/src/cuda/fastspline_residual_cuda.cu
   - build script compiles and links it
   - cuBLAS/cuSOLVER linked successfully

3. CUDA skeleton integration exists:
   - fast_skeleton_cuda_backend(..., residual_device=...)
   - fast_kpc_wanpdag_cuda(..., residual_device=...)
   - residual cache records residual_device
   - CPU residual path still available

4. Public wrapper support exists:
   - fast_kpc(..., residual_device=...)
   - config records residual_device_requested/residual_device_used
   - diagnostics records cuda_residual_available/cuda_residual_reason

5. Campaign/report/CLI support exists:
   - run_fastkpc_validation_campaign(..., residual_devices=...)
   - residual_device_diffs table
   - residual_device_diffs.csv artifact
   - --residual-device and --residual-devices CLI flags

6. All new tests pass:
   - fastkpc/tests/test_cuda_fastspline_residual_kernel.R
   - fastkpc/tests/test_cuda_fastspline_residual_batch.R
   - fastkpc/tests/test_cuda_residual_device_skeleton.R
   - fastkpc/tests/test_fastkpc_residual_device_public_api.R
   - fastkpc/tests/test_cuda_residual_device_campaign.R
   - fastkpc/tests/test_cuda_residual_device_report_cli.R
   - fastkpc/tests/test_cuda_residual_benchmark.R
   - fastkpc/tests/test_cuda_residual_docs_contract.R

7. Prior public wrapper/campaign/report tests still pass.

8. Prior CUDA dCov/skeleton/fastSpline/WAN-PDAG tests still pass.

9. Standalone CUDA residual differences are below tolerance:
   - max_abs_residual_diff < 1e-7
   - max_abs_fitted_diff < 1e-7
   - relative rss diff < 1e-8

10. CUDA residual skeleton graph equality holds:
   - adjacency identical vs CPU residual path
   - sepsets identical vs CPU residual path
   - pMax max abs diff < 1e-7
   - WAN-PDAG pdag identical in smoke scenarios

11. Linear residuals with residual_device="cuda" resolve to CPU with explicit reason.

12. Fallback behavior is explicit and testable.

13. kpcalg/R/*.R files remain unchanged by MD5.
```

Required final verification command:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
Rscript fastkpc/tests/test_cuda_residual_benchmark.R
Rscript fastkpc/tests/test_orientation_matrix.R
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_regrvonps_native.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
Rscript fastkpc/tests/test_wanpdag_benchmark.R
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
Rscript fastkpc/tests/test_validation_scenarios.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_cuda_residual_device_report_cli.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
Rscript fastkpc/tests/test_fastkpc_reproducibility.R
Rscript fastkpc/tests/test_fastkpc_legacy_diagnostics.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
Rscript fastkpc/tests/test_cuda_residual_docs_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
Rscript -e 'source("fastkpc/R/validation_campaign.R"); source("fastkpc/R/report_writer.R"); c <- run_fastkpc_validation_campaign(seeds=c(201,202), n_values=c(80), scenarios=c("chain","fork","collider","independent","additive"), engines=c("cuda"), residual_backends=c("fastSpline"), residual_devices=c("cpu","cuda"), legacy=TRUE, benchmark=TRUE); print(c$summary); print(c$residual_device_diffs); stopifnot(all(c$runs$status == "ok")); stopifnot(all(c$residual_device_diffs$max_abs_pmax_diff < 1e-7, na.rm=TRUE)); d <- tempfile("fastkpc-cuda-residual-final-report-"); a <- write_fastkpc_validation_report(c, d); print(a); stopifnot(file.exists(file.path(d, "summary.md"))); stopifnot(file.exists(file.path(d, "campaign.rds"))); stopifnot(file.exists(file.path(d, "residual_device_diffs.csv")))'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
Exact build commands used.
Exact test commands run.
Pass/fail result for every new and existing test group.
Standalone CUDA residual max residual/fitted/RSS differences.
CUDA residual skeleton pMax maximum difference vs CPU residual path.
Public fast_kpc() residual_device smoke result.
Campaign summary.
Residual-device diff table summary and max_abs_pmax_diff maximum.
Fallback diagnostics if any fallback occurred.
Report artifact directory and files written.
CLI smoke status.
kpcalg/R MD5 result.
```

## Later Goals

Create separate goals only after this CUDA residual device goal is complete.

### Later Goal I: Layer-Batched PC Scheduler

Objective:

```text
Replace one-test-at-a-time skeleton orchestration with a layer-batched scheduler that groups residual tasks and dCov tasks across candidate edges while preserving stable PC replay semantics.
```

### Later Goal J: Larger Reproducibility Report

Objective:

```text
Run a larger validation campaign across more seeds, n values, graph shapes, residual backends, residual devices, and public wrapper configurations, then write a versioned report under fastkpc/reports.
```

### Later Goal K: Multi-GPU Scheduling

Objective:

```text
Split layer-batched skeleton work across the two RTX 4090 devices with deterministic replay and per-device diagnostics.
```

## Execution Rules For Codex

- Use TDD: write each listed test before implementing the corresponding code.
- Keep all new functionality under `fastkpc/`.
- Do not alter `kpcalg/R/*.R`.
- Keep `fast_kpc()` opt-in; do not replace `kpcalg::kpc()`.
- Keep CPU residual behavior as the reference.
- Keep CUDA residuals opt-in through `residual_device`.
- Treat CUDA fallback as a recorded diagnostic, not a silent behavior.
- Treat missing `pcalg`/`graph` as a recorded diagnostic, not a hard failure.
- Run CUDA tests serially because the local build artifacts and loaded shared object are shared.
- Do not initialize git in this workspace unless the user asks.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-14-fast-kpc-cuda-residual-kernels-goal-execution.md: add an opt-in CUDA fastSpline residual device, validate standalone residual equivalence against the CPU fastSpline reference, integrate CUDA residuals into the CUDA skeleton/WAN-PDAG residual cache, extend public wrapper/campaign/report/CLI diagnostics, and keep kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
