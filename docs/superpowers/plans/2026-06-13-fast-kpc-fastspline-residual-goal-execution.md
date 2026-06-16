# Fast kPC fastSpline Residual Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in fastSpline residual backend to `fastkpc`, integrate it with the existing residual cache, CPU skeleton, and CUDA skeleton, and produce graph-level validation against both the current linear MVP backend and legacy `mgcv` residualization.

**Architecture:** Keep exact dCov, CUDA dCov batching, skeleton replay, and the existing residual cache behavior stable. Add a residual backend registry with two production backends for this stage: existing `linear` and new `fastSpline`; add an R-only legacy `mgcv` validation harness for comparison, not as the default native backend. Validate this statistically and graph-wise with explicit difference reports rather than hiding expected smoothing differences.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5 for existing batched dCov, current per-run residual cache, cubic B-spline basis, ridge-regularized penalized least squares, fixed lambda grid selected by GCV.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-fastspline-residual-goal-execution.md: add an opt-in fastSpline residual backend with residual-cache integration, CPU/CUDA skeleton support, and graph-level validation against linear and legacy mgcv references, keeping legacy kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `260000`.

This goal is intentionally larger than the previous ones. Do not mark it complete until all completion criteria in Phase 9 are satisfied. Mark it blocked only if the same blocker prevents progress for three consecutive goal turns and cannot be resolved locally.

## Preconditions

The residual cache goal must already be complete.

Required current artifacts:

```text
fastkpc/R/native.R
fastkpc/R/cuda_native.R
fastkpc/R/diff_report.R
fastkpc/R/legacy_runner.R
fastkpc/R/residual_validation.R
fastkpc/src/residual_backend.hpp
fastkpc/src/residual_backend.cpp
fastkpc/src/residual_cache.hpp
fastkpc/src/residual_cache.cpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/rcpp_exports.cpp
fastkpc/src/r_api_cuda.cpp
fastkpc/tests/test_residual_cache_core.R
fastkpc/tests/test_skeleton_residual_cache.R
fastkpc/tests/test_cuda_residual_cache.R
```

Required environment:

```text
R 4.4.1
Rcpp installed
RcppArmadillo installed
mgcv installed for legacy validation
CUDA toolkit available at /usr/local/cuda/bin/nvcc
NVIDIA driver able to run CUDA kernels
```

If `mgcv` is unavailable, native fastSpline work may continue but the goal cannot be marked complete until legacy mgcv validation either passes or the missing dependency is explicitly documented in the completion report and a user approves skipping legacy validation. If CUDA is unavailable, the goal cannot be complete because CUDA skeleton fastSpline integration is in scope.

## Scope

In scope for this goal:

- Introduce a residual backend registry that supports:

```text
linear
fastSpline
```

- Implement fastSpline residualization in native C++:

```text
1D conditioning set: cubic B-spline basis + second-difference penalty + GCV lambda selection.
2D conditioning set: tensor-product cubic B-spline basis + separable second-difference penalty + GCV lambda selection.
>2D conditioning set: additive sum of per-variable cubic B-spline bases + block second-difference penalty + GCV lambda selection.
```

- Integrate fastSpline with the existing per-run residual cache.
- Add CPU skeleton wrapper and CUDA skeleton wrapper that accept `residual_backend = "linear"` or `"fastSpline"`.
- Preserve existing linear backend outputs and tests.
- Add R validation helpers comparing:

```text
linear vs fastSpline
legacy mgcv residuals vs fastSpline residuals
legacy mgcv graph vs fastSpline graph where pcalg is available
CPU fastSpline skeleton vs CUDA fastSpline skeleton
```

- Add small benchmark/report helpers measuring:

```text
residual computation count
cache hits
CPU skeleton wall time
CUDA skeleton wall time
legacy mgcv residualization wall time on fixed residual tasks
```

- Keep `kpcalg/R/*.R` unchanged.

Out of scope for this goal:

- Do not implement GPU residualization.
- Do not implement CUDA B-spline kernels.
- Do not implement multi-GPU scheduling.
- Do not migrate `udag2wanpdag()`.
- Do not replace exported `kpcalg::kpc()`.
- Do not make fastSpline the default backend.
- Do not claim graph equality with mgcv as a hard requirement; report graph differences explicitly.
- Do not support HSIC or permutation tests.

## Design Contract

### Residual Backend Registry

Add a residual backend enum/descriptor layer.

Required native backend names:

```text
linear
fastSpline
```

Required descriptor fields:

```text
name
params
cache_key
```

Backend parameter strings must be deterministic and include all values that can affect residuals.

Required default backend params:

```text
linear:
  intercept=true;ridge=1e-8

fastSpline:
  degree=3;knots=10;lambda_grid=1e-4:1e4:25;ridge=1e-8;mode=auto
```

The residual cache key must include backend name and params. A cached `linear` residual must never be reused for `fastSpline`, and a `fastSpline` residual with different params must not be reused.

### fastSpline Numerical Contract

For a target vector `y` and conditioning matrix `S`, compute residuals:

```text
residual = y - fitted
```

where `fitted` comes from penalized least squares:

```text
min_beta ||y - X beta||^2 + lambda * beta' P beta
```

Use an intercept column in every design.

Basis rules:

```text
|S| = 1:
  X = [1, cubic_bspline(S1, knots = 10)]
  P = second_difference_penalty_for_basis

|S| = 2:
  X = [1, tensor(cubic_bspline(S1), cubic_bspline(S2))]
  P = kron(D2'D2, I) + kron(I, D2'D2)

|S| > 2:
  X = [1, cubic_bspline(S1), cubic_bspline(S2), one block per conditioning variable through cubic_bspline(Sd)]
  P = block diagonal matrix with one D2'D2 block per conditioning variable and zero intercept row/column
```

Lambda selection:

```text
lambda_grid = exp(seq(log(1e-4), log(1e4), length.out = 25))
For each lambda:
  A = X'X + lambda * P + ridge * I_penalized
  beta = solve(A, X'y)
  RSS = ||y - X beta||^2
  edf = trace(X solve(A, X'))
  GCV = n * RSS / (n - edf)^2
Choose the lambda with minimum finite GCV.
Tie-break by choosing the smaller lambda.
```

Ridge:

```text
ridge = 1e-8
Do not penalize the intercept in P.
Apply ridge only to non-intercept coefficients.
If Cholesky or solve fails, increase ridge by factors of 100 up to 1e-4.
If all lambdas fail, throw a clear error containing "fastSpline solve failed".
```

Centering/scaling:

```text
Sort-independent: rows must remain in original sample order.
For each conditioning variable, use quantile-based boundary knots from that variable.
If a conditioning variable is constant, its spline block must degrade to a zero-width smooth without crashing. The intercept remains.
```

Accuracy targets:

```text
Residual mean absolute value < 1e-8 when an intercept is included.
On smooth synthetic y = sin(z) + noise, fastSpline RSS must be lower than linear RSS.
On additive smooth y = sin(z1) + cos(z2) + noise, fastSpline RSS must be lower than linear RSS.
```

### Legacy mgcv Validation Contract

Legacy `mgcv` is a validation reference, not a required graph-equality target.

Validation must report:

```text
residual_correlation
relative_residual_l2
p_value_correlation
adjacency diff
sepset diff
pMax diff
n.edgetests diff
```

Required pass thresholds for fixed small scenarios:

```text
|S| = 1 smooth scenario:
  residual_correlation >= 0.97
  relative_residual_l2 <= 0.35

|S| = 2 smooth scenario:
  residual_correlation >= 0.85
  relative_residual_l2 <= 0.60
```

Graph differences from mgcv are allowed but must be reported. CPU fastSpline and CUDA fastSpline, however, must be graph-identical up to pMax tolerance.

### Skeleton Behavior Contract

Existing wrappers remain available:

```r
fast_skeleton_cpp()
fast_skeleton_cuda()
fast_skeleton_cpp_cached()
fast_skeleton_cuda_cached()
```

Add backend-aware wrappers:

```r
fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                          residual_backend = "linear",
                          residual_cache = TRUE,
                          index = 1,
                          legacy_index = TRUE,
                          fastspline_params = list())

fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           residual_backend = "linear",
                           residual_cache = TRUE,
                           index = 1,
                           legacy_index = TRUE,
                           batch_size = 0,
                           fastspline_params = list())
```

Backend-aware wrappers must return:

```text
backend: "cpu" or "cuda"
residual_backend: "linear" or "fastSpline"
residual_backend_params: deterministic string
residual_cache: stats list
```

Existing cached wrappers may call backend-aware wrappers with `residual_backend = "linear"` if return compatibility is preserved.

### CUDA Contract

CUDA skeleton fastSpline does not compute residuals on GPU. It must:

```text
compute/cache fastSpline residuals on CPU
pack residual vectors into CUDA dCov batches
preserve deterministic replay semantics
match CPU fastSpline skeleton adjacency, sepsets, and n.edgetests
have max_abs_pmax_diff < 1e-8 against CPU fastSpline skeleton
```

## File Structure

Create these files:

- `fastkpc/src/fastspline_basis.hpp`
- `fastkpc/src/fastspline_basis.cpp`  
  Cubic B-spline basis construction, quantile knot selection, tensor/additive design matrix helpers, and penalty matrix builders.

- `fastkpc/src/fastspline_solver.hpp`
- `fastkpc/src/fastspline_solver.cpp`  
  Penalized least squares, lambda grid, GCV scoring, ridge retry, fitted/residual output, and diagnostics.

- `fastkpc/src/residual_backend_registry.hpp`
- `fastkpc/src/residual_backend_registry.cpp`  
  Backend name parsing, descriptor construction, params serialization, backend dispatch to linear or fastSpline.

- `fastkpc/R/fastspline_validation.R`  
  Residual-level, p-value-level, graph-level, and benchmark validation helpers.

- `fastkpc/tests/test_fastspline_basis.R`
- `fastkpc/tests/test_fastspline_solver.R`
- `fastkpc/tests/test_residual_backend_registry.R`
- `fastkpc/tests/test_skeleton_fastspline_cpu.R`
- `fastkpc/tests/test_skeleton_fastspline_cuda.R`
- `fastkpc/tests/test_fastspline_mgcv_validation.R`
- `fastkpc/tests/test_fastspline_benchmark.R`

Modify these files:

- `fastkpc/src/residual_backend.hpp`
- `fastkpc/src/residual_backend.cpp`  
  Add backend dispatch and keep linear behavior unchanged.

- `fastkpc/src/residual_cache.hpp`
- `fastkpc/src/residual_cache.cpp`  
  Cache must use generic backend descriptor/options instead of hard-coded linear helper.

- `fastkpc/src/fastkpc_types.hpp`  
  Add residual backend name/params to `SkeletonOptions` and result metadata.

- `fastkpc/src/skeleton_engine.cpp`
- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Route residual requests through selected backend and cache.

- `fastkpc/src/rcpp_exports.cpp`
- `fastkpc/src/r_api_cuda.cpp`  
  Add backend-aware entry points and test helpers.

- `fastkpc/R/native.R`
- `fastkpc/R/cuda_native.R`  
  Add backend-aware wrappers and fastSpline helper wrappers.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile/link new fastSpline and registry sources into CUDA shared object.

- `fastkpc/README.md`  
  Document fastSpline API, validation, benchmark, and known limits.

Do not modify:

- `kpcalg/R/*.R`
- `gpu-dcov/*` except for running validation

## Phase 0: Baseline Audit

Purpose: prove current CPU/CUDA/residual-cache baseline is green before adding a new backend.

- [ ] Run:

```bash
pwd
find fastkpc -maxdepth 3 -type f | sort
find kpcalg/R -maxdepth 1 -type f | sort
Rscript -e 'cat("R ", as.character(getRversion()), "\n", sep=""); for (p in c("Rcpp","RcppArmadillo","mgcv")) cat(p, ": ", requireNamespace(p, quietly=TRUE), "\n", sep="")'
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
Rcpp, RcppArmadillo, and mgcv are TRUE.
CUDA toolkit and at least one GPU are visible.
```

- [ ] Run:

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
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/residual_validation.R"); print(validate_cpu_residual_cache()); print(validate_cuda_residual_cache())'
```

Expected:

```text
All existing tests pass.
Residual cache validation reports adjacency_identical TRUE for CPU and CUDA.
```

## Phase 1: fastSpline Basis

Purpose: implement deterministic spline design/penalty construction independently of skeleton code.

- [ ] Create `fastkpc/tests/test_fastspline_basis.R`.

The test must call a new Rcpp helper:

```r
fastspline_basis_selftest(data)
```

Use:

```r
set.seed(41)
n <- 80
z1 <- sort(runif(n, -2, 2))
z2 <- rnorm(n)
constant <- rep(3, n)
data <- cbind(z1 = z1, z2 = z2, constant = constant)
```

The returned list must include:

```text
one_d:
  nrow
  ncol
  row_sums_close_to_one
  finite
  penalty_dim
  penalty_symmetric

two_d:
  nrow
  ncol
  finite
  penalty_dim
  penalty_symmetric

additive:
  nrow
  ncol
  finite
  penalty_dim
  penalty_symmetric

constant:
  finite
  non_intercept_cols_all_zero_or_constant
```

Assertions:

```text
one_d$nrow == n
one_d$ncol >= 6
one_d$row_sums_close_to_one TRUE
one_d$finite TRUE
one_d$penalty_dim == one_d$ncol
one_d$penalty_symmetric TRUE
two_d$nrow == n
two_d$ncol > one_d$ncol
two_d$finite TRUE
two_d$penalty_dim == two_d$ncol
additive$nrow == n
additive$ncol > one_d$ncol
additive$finite TRUE
constant$finite TRUE
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_basis.R
```

Expected:

```text
The test fails because fastspline_basis_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/fastspline_basis.hpp` and `fastkpc/src/fastspline_basis.cpp`.

Required C++ structs:

```cpp
struct FastSplineParams {
  int degree;
  int knots;
  double lambda_min;
  double lambda_max;
  int lambda_count;
  double ridge;
  std::string mode;
};

struct FastSplineDesign {
  std::vector<double> X;      // row-major n by p
  std::vector<double> P;      // row-major p by p
  int n;
  int p;
};
```

Required functions:

```cpp
FastSplineParams default_fastspline_params();
std::string serialize_fastspline_params(const FastSplineParams& params);

std::vector<double> quantile_knots(const std::vector<double>& x, int knots);
std::vector<double> cubic_bspline_basis(const std::vector<double>& x,
                                        const FastSplineParams& params,
                                        int* n_basis);
std::vector<double> second_difference_penalty(int n_basis);

FastSplineDesign make_fastspline_design(const Rcpp::NumericMatrix& data,
                                        const std::vector<int>& conditioning_set,
                                        const FastSplineParams& params);
```

Design requirements:

```text
X must include an intercept as column 0.
P must have zero penalty for intercept row/column.
Use row-major storage inside C++ helpers.
For 1D, non-intercept B-spline rows should sum to approximately 1 for non-constant input.
For 2D, tensor basis columns are pairwise products of 1D basis columns.
For >2D, additive basis concatenates 1D bases for each conditioning variable.
For constant input, avoid NaN/Inf and use deterministic finite basis values.
```

- [ ] Add `fastspline_basis_selftest()` to `fastkpc/src/rcpp_exports.cpp` and wrapper in `fastkpc/R/native.R`.

Required wrapper:

```r
fastspline_basis_selftest <- function(data) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fastspline_basis_selftest_export(data)
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_basis.R
```

Expected:

```text
test_fastspline_basis.R prints PASS.
```

## Phase 2: fastSpline Solver

Purpose: fit penalized spline residuals and prove it improves over linear residuals on smooth data.

- [ ] Create `fastkpc/tests/test_fastspline_solver.R`.

Required tests:

```text
1. 1D smooth scenario: y = sin(z) + noise. fastSpline RSS < 0.75 * linear RSS.
2. 2D smooth scenario: y = sin(z1) + cos(z2) + noise. fastSpline RSS < 0.80 * linear RSS.
3. Additive 3D smooth scenario: y = sin(z1) + cos(z2) + 0.5*z3^2 + noise. fastSpline RSS < 0.85 * linear RSS.
4. Residual mean is within 1e-8 of zero in all scenarios.
5. Selected lambda is finite and inside [1e-4, 1e4].
6. edf is finite and between 1 and design column count.
7. Constant conditioning variable does not crash and returns finite residuals.
```

Required Rcpp helper:

```r
fastspline_solver_selftest()
```

Return list:

```text
one_d = list(fastspline_rss, linear_rss, residual_mean, selected_lambda, edf, design_cols)
two_d = same fields
three_d = same fields
constant = list(finite_residuals, residual_mean, selected_lambda)
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_solver.R
```

Expected:

```text
The test fails because fastspline_solver_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/fastspline_solver.hpp` and `fastkpc/src/fastspline_solver.cpp`.

Required C++ structs:

```cpp
struct FastSplineFit {
  std::vector<double> residuals;
  std::vector<double> fitted;
  double selected_lambda;
  double gcv;
  double rss;
  double edf;
  int design_cols;
  int ridge_attempts;
};
```

Required functions:

```cpp
FastSplineFit fit_fastspline_residuals(const Rcpp::NumericMatrix& data,
                                       int target,
                                       const std::vector<int>& conditioning_set,
                                       const FastSplineParams& params);

std::vector<double> lambda_grid(const FastSplineParams& params);
```

Implementation requirements:

```text
Use double precision throughout.
Use normal equations with Cholesky or a robust fallback.
Do not introduce new external dependencies beyond Rcpp/RcppArmadillo.
Compute edf = trace(X solve(A, X')) using A inverse or repeated solves.
Choose minimum finite GCV.
Tie-break by smaller lambda.
Throw "fastSpline solve failed" if all lambdas fail.
```

Suggested implementation approach:

```text
Use RcppArmadillo for matrix algebra in fastspline_solver.cpp.
Keep basis construction in standard C++ vectors and convert to arma::mat only in solver.
```

- [ ] Update build paths.

Required:

```text
sourceCpp CPU path must compile fastspline_basis.cpp and fastspline_solver.cpp.
CUDA build script must compile/link them too because CUDA skeleton may request fastSpline residuals on CPU.
```

- [ ] Add `fastspline_solver_selftest()` export and R wrapper.

Run:

```bash
Rscript fastkpc/tests/test_fastspline_solver.R
```

Expected:

```text
test_fastspline_solver.R prints PASS.
```

## Phase 3: Backend Registry And Cache Integration

Purpose: make residual cache backend-aware and preserve linear behavior.

- [ ] Create `fastkpc/tests/test_residual_backend_registry.R`.

Required tests:

```text
1. list_residual_backends() returns c("linear", "fastSpline").
2. linear backend residuals match existing fast_residual_cache_selftest direct residual path.
3. fastSpline backend residuals differ from linear on smooth nonlinear data.
4. residual cache key separates linear and fastSpline.
5. residual cache key separates fastSpline params with different knots.
6. fastSpline cached repeated request hits cache and computations < requests.
7. Unknown backend raises "Unknown residual backend".
```

Required R wrappers:

```r
list_residual_backends()
fast_residual_backend_selftest()
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_residual_backend_registry.R
```

Expected:

```text
The test fails because list_residual_backends() or fast_residual_backend_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/residual_backend_registry.hpp` and `fastkpc/src/residual_backend_registry.cpp`.

Required API:

```cpp
enum class ResidualBackendKind {
  Linear,
  FastSpline
};

struct ResidualBackendConfig {
  ResidualBackendKind kind;
  std::string name;
  std::string params;
  FastSplineParams fastspline;
};

std::vector<std::string> list_residual_backend_names();
ResidualBackendConfig make_residual_backend_config(
  const std::string& name,
  const FastSplineParams& fastspline_params);

std::vector<double> compute_residuals_with_backend(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const ResidualBackendConfig& config);
```

Required behavior:

```text
linear dispatches to compute_linear_residuals(data, target, conditioning_set).
fastSpline dispatches to fit_fastspline_residuals(data, target, conditioning_set, config.fastspline).residuals.
Unknown backend throws "Unknown residual backend".
Config params string is deterministic.
```

- [ ] Modify `fastkpc/src/residual_cache.hpp` and `fastkpc/src/residual_cache.cpp`.

Required changes:

```text
ResidualCacheOptions must hold ResidualBackendConfig or equivalent name/params/kind.
ResidualCache::get() must dispatch through compute_residuals_with_backend().
Existing linear_residual_cache_options(TRUE/FALSE) must still exist for compatibility.
Add backend_residual_cache_options(name, params, enabled).
```

- [ ] Modify `fastkpc/src/fastkpc_types.hpp`.

Required additions:

```text
SkeletonOptions.residual_backend_name
SkeletonOptions.fastspline_params or serialized fastSpline options
SkeletonResult.residual_backend_params
```

- [ ] Add exports/wrappers:

```r
list_residual_backends()
fast_residual_backend_selftest()
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
```

Expected:

```text
All three tests print PASS.
```

## Phase 4: CPU Skeleton fastSpline Integration

Purpose: run the CPU skeleton with fastSpline residuals as an opt-in backend.

- [ ] Create `fastkpc/tests/test_skeleton_fastspline_cpu.R`.

Use fixed nonlinear scenario:

```r
set.seed(51)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.2),
  x2 = cos(z1) + rnorm(n, sd = 0.2),
  x3 = sin(z2) + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2
```

Required tests:

```text
1. fast_skeleton_cpp_backend(data, alpha, max_ord, residual_backend="linear", residual_cache=TRUE) matches fast_skeleton_cpp_cached(data, alpha, max_ord, residual_cache=TRUE) adjacency exactly.
2. linear backend pMax matches cached linear pMax within 1e-10.
3. fastSpline backend returns backend "cpu" and residual_backend "fastSpline".
4. fastSpline backend cache stats have hits > 0 and computations < requests.
5. fastSpline backend adjacency is symmetric with FALSE diagonal.
6. fastSpline backend pMax is symmetric with diagonal 1.
7. Running fastSpline twice on the same data returns identical adjacency, sepsets, n.edgetests, and pMax within 1e-12.
8. fastSpline and linear graph differences are summarized with summarize_graph_diff().
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
```

Expected:

```text
The test fails because fast_skeleton_cpp_backend() does not exist yet.
```

- [ ] Modify `fastkpc/src/skeleton_engine.cpp`.

Required behavior:

```text
ResidualCache is constructed from SkeletonOptions.residual_backend_name and fastSpline params.
Conditional residuals use selected backend through cache.
Unconditional tests remain unchanged.
Existing run_skeleton_exact behavior with default/linear options is unchanged.
SkeletonResult includes residual_backend and residual_backend_params.
```

- [ ] Modify `fastkpc/src/rcpp_exports.cpp`.

Add export:

```cpp
Rcpp::List fast_skeleton_cpp_backend_export(Rcpp::NumericMatrix data,
                                            double alpha,
                                            int max_conditioning_size,
                                            double index,
                                            bool legacy_index,
                                            bool residual_cache,
                                            std::string residual_backend,
                                            Rcpp::List fastspline_params);
```

The export must:

```text
Parse fastspline_params with defaults.
Reject unknown residual_backend.
Return normal skeleton result plus residual_backend_params.
```

- [ ] Modify `fastkpc/R/native.R`.

Required wrapper:

```r
fast_skeleton_cpp_backend <- function(data, alpha, max_conditioning_size,
                                      residual_backend = "linear",
                                      residual_cache = TRUE,
                                      index = 1,
                                      legacy_index = TRUE,
                                      fastspline_params = list()) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_skeleton_cpp_backend_export(
    data, as.numeric(alpha), as.integer(max_conditioning_size),
    as.numeric(index), isTRUE(legacy_index), isTRUE(residual_cache),
    as.character(residual_backend), fastspline_params
  )
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_skeleton_mvp.R
```

Expected:

```text
All three tests print PASS.
```

## Phase 5: CUDA Skeleton fastSpline Integration

Purpose: allow CUDA dCov batching to use fastSpline residuals computed/cached on CPU.

- [ ] Create `fastkpc/tests/test_skeleton_fastspline_cuda.R`.

Use the same fixed nonlinear scenario from Phase 4.

Required tests:

```text
1. fast_skeleton_cuda_backend(data, alpha, max_ord, residual_backend="linear", residual_cache=TRUE) matches fast_skeleton_cuda_cached(data, alpha, max_ord, residual_cache=TRUE) adjacency.
2. CUDA linear backend pMax matches cached CUDA linear pMax within 1e-8.
3. CUDA fastSpline backend returns backend "cuda" and residual_backend "fastSpline".
4. CUDA fastSpline backend cache stats have hits > 0 and computations < requests.
5. CUDA fastSpline adjacency, sepsets, and n.edgetests match CPU fastSpline.
6. CUDA fastSpline pMax differs from CPU fastSpline by less than 1e-8.
7. batch_size = 1 and batch_size = 0 produce identical CUDA fastSpline adjacency, sepsets, n.edgetests and pMax within 1e-8.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
```

Expected:

```text
The test fails because fast_skeleton_cuda_backend() does not exist yet.
```

- [ ] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

Required behavior:

```text
Use backend-aware ResidualCache while packing conditional task residuals.
Do not change CUDA dCov batching or deterministic replay.
CPU fastSpline residuals are computed before copying residual matrices to GPU.
Existing CUDA linear behavior remains unchanged.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Add `.Call` entry:

```cpp
C_fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                             index, legacy_index, batch_size,
                             residual_cache, residual_backend,
                             fastspline_params)
```

Register it in `R_init_fastkpc_cuda()`.

- [ ] Modify `fastkpc/R/cuda_native.R`.

Required wrapper:

```r
fast_skeleton_cuda_backend <- function(data, alpha, max_conditioning_size,
                                       residual_backend = "linear",
                                       residual_cache = TRUE,
                                       index = 1,
                                       legacy_index = TRUE,
                                       batch_size = 0,
                                       fastspline_params = list()) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_backend", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache), as.character(residual_backend),
        fastspline_params, PACKAGE = "fastkpc_cuda")
}
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Required behavior:

```text
Compile/link fastspline_basis.cpp, fastspline_solver.cpp, and residual_backend_registry.cpp into fastkpc_cuda.so.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

Expected:

```text
All three tests print PASS.
```

## Phase 6: Legacy mgcv Residual Validation

Purpose: quantify how fastSpline compares to legacy `regrXonS()` without requiring graph equality.

- [ ] Create `fastkpc/tests/test_fastspline_mgcv_validation.R`.

Required helper:

```r
validate_fastspline_against_mgcv(seed = 61, n = 160)
```

It must return:

```text
one_d:
  residual_correlation
  relative_residual_l2
  fastspline_rss
  mgcv_rss
  dcov_pvalue_abs_diff

two_d:
  same fields

graph:
  available
  reason_if_unavailable
  diff
  max_abs_pmax_diff
  adjacency_added_count
  adjacency_removed_count
```

Test requirements:

```text
1. If mgcv is missing, fail with "mgcv is required for this validation".
2. one_d$residual_correlation >= 0.97.
3. one_d$relative_residual_l2 <= 0.35.
4. two_d$residual_correlation >= 0.85.
5. two_d$relative_residual_l2 <= 0.60.
6. dcov_pvalue_abs_diff is finite for one_d and two_d.
7. graph section exists and records availability. If pcalg is missing, graph$available is FALSE with reason containing "pcalg".
8. If graph$available is TRUE, diff contains adjacency, pMax, sepsets, and n_edgetests sections.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
```

Expected:

```text
The test fails because validate_fastspline_against_mgcv() does not exist yet.
```

- [ ] Create `fastkpc/R/fastspline_validation.R`.

Required functions:

```r
fastkpc_mgcv_residual <- function(y, S)
fastkpc_fastspline_residual <- function(y, S, fastspline_params = list())
validate_fastspline_against_mgcv <- function(seed = 61, n = 160)
compare_fastspline_linear_graph <- function(seed = 51, n = 120, alpha = 0.2, max_conditioning_size = 2)
compare_fastspline_cpu_cuda_graph <- function(seed = 51, n = 120, alpha = 0.2, max_conditioning_size = 2)
```

Implementation requirements:

```text
fastkpc_mgcv_residual() uses fastkpc_legacy_env() and env$regrXonS().
fastkpc_fastspline_residual() calls a native helper for one target and conditioning matrix.
Graph comparison functions use summarize_graph_diff().
Do not require pcalg for residual-level mgcv validation.
If pcalg is not installed, graph legacy comparison returns available = FALSE.
```

- [ ] Add native helper and R wrapper:

```r
fastspline_residual <- function(y, S, fastspline_params = list())
```

It must:

```text
Use fastSpline backend only.
Return residual vector and diagnostics.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
```

Expected:

```text
test_fastspline_mgcv_validation.R prints PASS.
```

## Phase 7: Benchmark And Reporting

Purpose: provide a small repeatable benchmark that shows where time is going and confirms cache behavior.

- [ ] Create `fastkpc/tests/test_fastspline_benchmark.R`.

Required helper:

```r
benchmark_fastspline_backends(seed = 71, n = 180, alpha = 0.2, max_conditioning_size = 2)
```

Return:

```text
list(
  timings = data.frame(backend, engine, elapsed_sec),
  cache = data.frame(backend, engine, requests, hits, computations),
  graph = list(
    linear_vs_fastspline_cpu = summarize_graph_diff(linear_cpu_result, fastspline_cpu_result),
    fastspline_cpu_vs_cuda = summarize_graph_diff(fastspline_cpu_result, fastspline_cuda_result)
  )
)
```

Test requirements:

```text
1. timings has rows for linear/cpu, fastSpline/cpu, fastSpline/cuda.
2. elapsed_sec values are finite and positive.
3. cache rows have hits > 0 for cached skeleton runs.
4. fastSpline CPU-vs-CUDA adjacency is identical.
5. fastSpline CPU-vs-CUDA max pMax diff < 1e-8.
```

- [ ] Add helper to `fastkpc/R/fastspline_validation.R`.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastspline_benchmark.R
```

Expected:

```text
test_fastspline_benchmark.R prints PASS.
```

## Phase 8: Documentation And Build Hygiene

Purpose: leave the fastSpline backend usable and clearly scoped.

- [ ] Update `fastkpc/README.md`.

Required sections:

```text
fastSpline Residual Backend Scope
fastSpline API
fastSpline Tests
fastSpline Validation
fastSpline Benchmark
fastSpline Known Limits
```

Known limits must explicitly state:

```text
fastSpline is opt-in and not the default backend.
fastSpline is not mgcv and graph differences from legacy mgcv can occur.
fastSpline residuals are computed on CPU even when CUDA dCov is used.
No CUDA spline kernels are implemented in this goal.
No WAN-PDAG migration is implemented in this goal.
kpcalg::kpc() is not replaced.
```

- [ ] Confirm build scripts and wrappers reference new sources/API.

Run:

```bash
rg -n "fastspline|fastSpline|residual_backend_registry|fast_skeleton_.*_backend" fastkpc/tools/build_cuda_native.sh fastkpc/src/rcpp_exports.cpp fastkpc/src/r_api_cuda.cpp fastkpc/R/native.R fastkpc/R/cuda_native.R fastkpc/R/fastspline_validation.R fastkpc/README.md
```

Expected:

```text
Each of these files has at least one matching line, except fastkpc/README.md may have multiple documentation matches. Missing output for any listed source or wrapper file means the implementation is not wired into that build/API surface yet.
```

## Phase 9: Completion Criteria

The goal is complete only when all of these are true:

```text
1. Existing tests still pass:
   - fastkpc/tests/test_dcov_exact.R
   - fastkpc/tests/test_skeleton_mvp.R
   - fastkpc/tests/test_diff_report.R
   - fastkpc/tests/test_cuda_build_contract.R
   - fastkpc/tests/test_dcov_cuda_batch.R
   - fastkpc/tests/test_skeleton_cuda_batch.R
   - fastkpc/tests/test_residual_cache_core.R
   - fastkpc/tests/test_skeleton_residual_cache.R
   - fastkpc/tests/test_cuda_residual_cache.R

2. New fastSpline tests pass:
   - fastkpc/tests/test_fastspline_basis.R
   - fastkpc/tests/test_fastspline_solver.R
   - fastkpc/tests/test_residual_backend_registry.R
   - fastkpc/tests/test_skeleton_fastspline_cpu.R
   - fastkpc/tests/test_skeleton_fastspline_cuda.R
   - fastkpc/tests/test_fastspline_mgcv_validation.R
   - fastkpc/tests/test_fastspline_benchmark.R

3. CPU fastSpline skeleton is deterministic:
   - repeated adjacency identical
   - repeated sepsets identical
   - repeated n.edgetests identical
   - repeated max_abs_pmax_diff < 1e-12

4. CPU fastSpline vs CUDA fastSpline validation reports:
   - adjacency_identical TRUE
   - sepsets_identical TRUE
   - n_edgetests_identical TRUE
   - max_abs_pmax_diff < 1e-8
   - cache_stats$hits > 0
   - cache_stats$computations < cache_stats$requests

5. mgcv residual validation reports:
   - one_d$residual_correlation >= 0.97
   - one_d$relative_residual_l2 <= 0.35
   - two_d$residual_correlation >= 0.85
   - two_d$relative_residual_l2 <= 0.60
   - graph comparison is either available with a diff report or unavailable with an explicit pcalg reason.

6. Benchmark helper returns finite positive timings and cache hit stats.

7. fastkpc/README.md documents fastSpline API, validation, benchmark, and limits.

8. No kpcalg/R/*.R file has been modified.
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
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/fastspline_validation.R"); print(compare_fastspline_linear_graph()); print(compare_fastspline_cpu_cuda_graph()); print(validate_fastspline_against_mgcv()); print(benchmark_fastspline_backends())'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
The exact build commands used.
The exact test commands run.
The pass/fail result of each test.
CPU fastSpline deterministic validation result.
CPU-vs-CUDA fastSpline graph diff and max pMax diff.
mgcv residual validation metrics.
Benchmark timing table.
Cache stats for fastSpline CPU and CUDA runs.
The kpcalg/R MD5 result.
```

## Later Goals

Create separate goals after this fastSpline residual backend goal is complete.

### Later Goal D: WAN-PDAG Migration

Objective:

```text
Migrate kpcalg's udag2wanpdag generalized transitive orientation step to the C++ scheduler, including batched regrVonPS-style residual independence checks and fastSpline residual cache reuse.
```

### Later Goal E: CUDA Residual Kernels

Objective:

```text
Move the expensive fastSpline basis evaluation and batched small-system solves to CUDA after the CPU fastSpline backend and graph validation are stable.
```

## Execution Rules For Codex

- Work in small commits if the workspace is a git repository. If it is not a git repository, do not initialize one unless the user asks.
- Prefer adding files under `fastkpc/` over modifying legacy files.
- Do not alter `kpcalg/R/*.R`.
- Do not delete or rewrite `gpu-dcov/`; use it as a numeric reference.
- Follow TDD: write each new fastSpline test before implementation.
- Keep fastSpline opt-in.
- Do not make graph differences from mgcv disappear by weakening tests; report them.
- If fastSpline CPU and CUDA differ, debug residual vectors first, then dCov p-values, then skeleton replay.
- Do not implement CUDA spline kernels, WAN-PDAG migration, HSIC, permutation tests, or `kpcalg::kpc()` replacement in this goal.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-fastspline-residual-goal-execution.md: add an opt-in fastSpline residual backend with residual-cache integration, CPU/CUDA skeleton support, and graph-level validation against linear and legacy mgcv references, keeping legacy kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
