# Fast kPC HSIC CI Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in HSIC gamma and HSIC permutation conditional-independence backends to fastkpc without modifying `kpcalg/R`, preserving existing dCov, residual, scheduler, CUDA, WAN-PDAG, validation, report, and CLI behavior.

**Architecture:** Keep the existing fastkpc skeleton/WAN-PDAG engines as the graph execution surface and introduce a narrow independence-test abstraction underneath each CI evaluation. Implement HSIC gamma and permutation numerics in native CPU C++ first, use existing residual backends for conditional tests, expose `ci_method`, `hsic_params`, and `permutation_params` through R wrappers and public `fast_kpc()`, and make CUDA engines resolve unsupported HSIC CUDA execution to CPU with explicit diagnostics rather than silently changing graph semantics.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, existing `fastkpc/src/skeleton_engine.cpp`, existing `fastkpc/src/skeleton_engine_cuda.cpp`, existing `fastkpc/src/wanpdag_engine.cpp`, existing validation/report/CLI tooling, legacy reference behavior from `kpcalg/R/hsicgamma.R`, `kpcalg/R/hsicperm.R`, `kpcalg/R/hsictest.R`, and `kpcalg/R/kernelCItest.R`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-16-fast-kpc-hsic-ci-backends-goal-execution.md: add opt-in native CPU HSIC gamma and HSIC permutation CI backends, expose ci_method/hsic_params/permutation_params through native wrappers, public fast_kpc(), validation campaigns, reports, and CLIs, preserve existing dCov/CUDA/WAN-PDAG behavior, record deterministic diagnostics and CPU resolution for CUDA HSIC requests, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `760000`.

This goal is intentionally large. It covers mathematical native HSIC implementation, test-first reference comparisons against legacy R formulas, skeleton/WAN-PDAG integration, public API design, validation campaign/report/CLI expansion, deterministic permutation handling, documentation, benchmark helpers, and full regression verification.

Do not mark the goal complete until every item in "Completion Criteria" is proven by current-state evidence. Mark the goal blocked only if the same local blocker repeats for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Current Baseline

The previous completed fastkpc stages provide:

```text
CPU exact dCov gamma backend
CUDA batched dCov gamma backend
CPU linear and fastSpline residual backends
CUDA fastSpline residual single and true-batched groups
Layer-batched CUDA skeleton scheduler
Native WAN-PDAG orientation
CUDA WAN-PDAG orientation residual/dCov execution for fastSpline
Public fast_kpc() wrapper
Validation campaign, report writer, and CLI tooling
Legacy kpcalg R files kept unchanged
```

Current implementation facts that matter:

```text
Skeleton CI tests currently compute dCov gamma p-values only.
Conditional tests residualize x and y on S, then evaluate dCov on residuals.
CUDA skeleton evaluates dCov gamma in batches and can prefetch CUDA residuals.
WAN-PDAG generalized orientation uses regrVonPS residual-vs-S tests.
Public fast_kpc() exposes residual_backend/residual_device/scheduler/orientation_residual_device but not ci_method.
Validation campaigns compare engines, residual backends, residual devices, schedulers, and orientation residual devices, but not independence-test methods.
Legacy kpcalg has R implementations of hsic.gamma(), hsic.perm(), hsic.test(), and kernelCItest() that must be used as behavioral references, not edited.
```

Important baseline contract:

```text
Existing default fast_kpc() calls keep dCov gamma behavior.
Existing tests continue to pass without requiring HSIC arguments.
CUDA dCov/residual/orientation paths remain valid.
kpcalg/R/*.R files remain unchanged.
HSIC support is opt-in in fastkpc.
```

## Why This Goal Is Next

The fastkpc backend now covers most of the dCov gamma execution path from skeleton through WAN-PDAG orientation. The remaining user-visible method gap in README is HSIC/permutation support. Adding HSIC as a staged, opt-in CI backend increases semantic coverage without forcing a risky change to the default graph behavior.

This is a better next slice than multi-GPU scheduling or replacing `kpcalg::kpc()` because:

```text
1. It is independently testable against existing legacy R functions.
2. It improves statistical method coverage exposed by kpcalg::kernelCItest().
3. It can reuse existing residual backends and graph engines.
4. It can preserve CUDA behavior by resolving HSIC CUDA requests to CPU first.
5. It creates clean abstraction seams for later GPU HSIC work.
```

This goal should not attempt a full GPU HSIC kernel. The first correct slice is native CPU HSIC gamma/permutation plus plumbing, diagnostics, and campaign/report coverage.

## Scope

In scope:

- Add explicit independence-test method controls:

```r
ci_method = c("dcc.gamma", "hsic.gamma", "hsic.perm")
hsic_params = list(sig = 1, kernel = "rbf", bandwidth = "legacy")
permutation_params = list(replicates = 100, seed = NULL, include_observed = TRUE)
ci_diagnostics = TRUE
```

- Preserve existing dCov gamma as the default:

```text
ci_method default: "dcc.gamma"
index default: 1
legacy_index default: TRUE
```

- Implement native CPU HSIC gamma:

```text
Gaussian/RBF kernel
legacy sigma semantics compatible with kpcalg/R/hsicgamma.R
centered kernel matrices
HSIC statistic
legacy gamma approximation fields
finite p-value output
diagnostics for mean, variance, shape, scale/rate, sample count
```

- Implement native CPU HSIC permutation:

```text
Observed HSIC statistic
Deterministic permutation generator when seed is supplied
R RNG-compatible path when seed is NULL and called from R
include_observed behavior matching hsic.perm() default p-value convention
replicate statistic vector when requested
diagnostics for replicate count and seed behavior
```

- Add a focused C++ CI method abstraction:

```text
CiMethod enum or equivalent string resolver
CiOptions in SkeletonOptions
CiResult diagnostics payload
CPU evaluator used by skeleton and WAN-PDAG paths
CUDA skeleton resolver that records HSIC CPU execution/fallback reason
```

- Make conditional HSIC use existing residualization:

```text
S empty: test original x and y vectors
S non-empty: residualize x and y using selected residual_backend, then test residual vectors
```

- Add public wrapper fields and result diagnostics:

```text
config$ci_method
config$ci_method_requested
config$ci_backend
config$ci_backend_reason
config$hsic_params
config$permutation_params
skeleton$ci_method
skeleton$ci_backend
skeleton$ci_diagnostics
orientation$ci_method
orientation$ci_diagnostics
diagnostics$ci_method_available
```

- Extend validation campaigns, reports, and CLI tooling:

```text
ci_methods dimension
ci_method_diffs.csv
ci_method_diagnostics.csv
summary.md sections for CI Method and CI Method Diagnostics
--ci-method / --ci-methods CLI options
--hsic-sig
--hsic-kernel
--permutation-replicates
--permutation-seed
--ci-diagnostics
```

- Add validation helpers comparing:

```text
native HSIC gamma vs legacy hsic.gamma()
native HSIC permutation vs reference R implementation with fixed seed/permutations
dCov gamma vs HSIC gamma graph differences as reported differences
HSIC gamma CPU engine vs CUDA engine CPU-resolved path
HSIC permutation reproducibility with fixed seed
WAN-PDAG HSIC gamma pipeline smoke
```

- Add benchmark helpers:

```text
benchmark_hsic_backends()
benchmark_ci_methods()
non-strict timing tables
no hard speedup requirement
```

- Keep all current tests passing.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not replace exported `kpcalg::kpc()`.
- Do not modify any file under `kpcalg/R`.
- Do not implement GPU HSIC kernels.
- Do not implement HSIC cluster in this goal.
- Do not implement conditional HSIC cluster behavior.
- Do not change default dCov gamma graph semantics.
- Do not change exact dCov gamma p-value math.
- Do not change fastSpline fitting math.
- Do not parallelize WAN-PDAG graph mutation.
- Do not implement multi-GPU scheduling.
- Do not require `kernlab` for fastkpc native HSIC tests, except optional legacy comparison tests that skip with explicit diagnostics when unavailable.
- Do not initialize git if this workspace is not already a git repository.

## Design Contract

### Public API Contract

Recommended `fast_kpc()` signature after this goal:

```r
fast_kpc <- function(data,
                     alpha = 0.2,
                     max_conditioning_size = 2,
                     engine = c("auto", "cuda", "cpu"),
                     ci_method = c("dcc.gamma", "hsic.gamma", "hsic.perm"),
                     residual_backend = c("fastSpline", "linear"),
                     residual_device = c("auto", "cpu", "cuda"),
                     orientation_residual_device = c("auto", "cpu", "cuda"),
                     scheduler = c("auto", "layer", "legacy"),
                     graph_stage = c("wanpdag", "skeleton"),
                     residual_cache = TRUE,
                     index = 1,
                     legacy_index = TRUE,
                     batch_size = 0,
                     residual_batch_size = 0,
                     orientation_batch_size = 0,
                     scheduler_diagnostics = TRUE,
                     orientation_diagnostics = TRUE,
                     ci_diagnostics = TRUE,
                     orient_collider = TRUE,
                     solve_confl = FALSE,
                     rules = c(TRUE, TRUE, TRUE),
                     fastspline_params = list(),
                     hsic_params = list(sig = 1, kernel = "rbf"),
                     permutation_params = list(replicates = 100, seed = NULL,
                                               include_observed = TRUE),
                     cuda_residual_fallback = TRUE,
                     validate = FALSE,
                     benchmark = FALSE,
                     legacy = FALSE,
                     labels = NULL,
                     seed = NULL)
```

Compatibility rules:

```text
1. Existing calls that omit ci_method remain dcc.gamma.
2. Existing calls that omit hsic_params remain valid.
3. Existing calls that omit permutation_params remain valid.
4. ci_method controls skeleton CI tests and WAN-PDAG generalized residual-vs-S tests.
5. residual_backend still controls conditional residualization.
6. residual_device still controls supported skeleton residual execution.
7. orientation_residual_device still controls supported orientation residual execution.
8. ci_method="dcc.gamma" keeps current CUDA dCov behavior.
9. ci_method="hsic.gamma" with engine="cuda" resolves CI backend to CPU in this goal and records reason.
10. ci_method="hsic.perm" with engine="cuda" resolves CI backend to CPU in this goal and records reason.
```

Recommended native wrapper signatures:

```r
fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                          ci_method = "dcc.gamma",
                          residual_backend = "linear",
                          residual_cache = TRUE,
                          index = 1,
                          legacy_index = TRUE,
                          fastspline_params = list(),
                          hsic_params = list(),
                          permutation_params = list(),
                          ci_diagnostics = TRUE)
```

```r
fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           ci_method = "dcc.gamma",
                           residual_backend = "linear",
                           residual_device = c("auto", "cpu", "cuda"),
                           residual_cache = TRUE,
                           index = 1,
                           legacy_index = TRUE,
                           batch_size = 0,
                           residual_batch_size = 0,
                           scheduler = c("auto", "layer", "legacy"),
                           scheduler_diagnostics = TRUE,
                           ci_diagnostics = TRUE,
                           fastspline_params = list(),
                           hsic_params = list(),
                           permutation_params = list(),
                           cuda_residual_fallback = TRUE)
```

```r
fast_kpc_wanpdag_cuda(data, alpha, max_conditioning_size,
                      ci_method = "dcc.gamma",
                      residual_backend = "fastSpline",
                      residual_device = c("auto", "cpu", "cuda"),
                      orientation_residual_device = c("auto", "cpu", "cuda"),
                      residual_cache = TRUE,
                      index = 1,
                      legacy_index = TRUE,
                      batch_size = 0,
                      residual_batch_size = 0,
                      orientation_batch_size = 0,
                      scheduler = c("auto", "layer", "legacy"),
                      scheduler_diagnostics = TRUE,
                      orientation_diagnostics = TRUE,
                      ci_diagnostics = TRUE,
                      orient_collider = TRUE,
                      solve_confl = FALSE,
                      rules = c(TRUE, TRUE, TRUE),
                      fastspline_params = list(),
                      hsic_params = list(),
                      permutation_params = list(),
                      cuda_residual_fallback = TRUE)
```

### CI Method Resolution Contract

Add a focused resolver:

```text
requested: dcc.gamma / hsic.gamma / hsic.perm
engine_used: cpu / cuda
graph_stage: skeleton / wanpdag
ci_backend_requested: auto/native/cuda can remain internal in this goal
ci_backend_used: native-cpu / cuda-dcov / cuda-resolved-cpu
ci_backend_reason: empty string when direct, non-empty when resolved/fallback
```

Resolution rules:

```text
engine=cpu, ci_method=dcc.gamma -> native-cpu
engine=cpu, ci_method=hsic.gamma -> native-cpu
engine=cpu, ci_method=hsic.perm -> native-cpu
engine=cuda, ci_method=dcc.gamma -> cuda-dcov for unconditional dCov batches and current CUDA path
engine=cuda, ci_method=hsic.gamma -> native-cpu with reason "CUDA HSIC gamma backend is not implemented in this stage"
engine=cuda, ci_method=hsic.perm -> native-cpu with reason "CUDA HSIC permutation backend is not implemented in this stage"
```

Public result fields:

```text
config$ci_method_requested
config$ci_method
config$ci_backend
config$ci_backend_reason
skeleton$ci_method
skeleton$ci_backend
skeleton$ci_backend_reason
orientation$ci_method
orientation$ci_backend
orientation$ci_backend_reason
```

### Native HSIC Math Contract

The first native HSIC implementation should be dense and deterministic. It does not need incomplete Cholesky in the first pass because the goal is fastkpc-native correctness, not reproducing every approximation detail of `kernlab::inchol()`.

Dense native HSIC contract:

```text
Input vectors x and y have length n.
n must be greater than 5 for gamma approximation.
Values must be finite.
Kernel is Gaussian/RBF.
K_ij = exp(-sigma * squared_distance(x_i, x_j)) when using kernlab-compatible sigma.
Default sig=1 maps to kernlab rbfdot(sigma=1/sig), so sigma = 1 / sig.
Centered matrices use H = I - 1/n.
HSIC = sum((HKH) * (HLH)) / n^2.
```

Gamma approximation contract:

```text
Return p.value in [0, 1].
Return statistic named HSIC.
Return estimate named HSIC.
Return estimates vector with HSIC, HSIC mean, HSIC variance.
Return finite diagnostics when input is valid.
If variance is non-positive or non-finite, return p.value=1 and record reason.
```

Legacy comparison contract:

```text
Native dense HSIC is not required to bit-match legacy incomplete-Cholesky HSIC.
Validation compares rank/behavior and loose tolerances on small fixed fixtures.
When numCol >= n, legacy inchol should be closer; use that setting for stricter tests when kernlab is available.
Graph equality versus dCov is not required.
Graph differences are reported.
```

### HSIC Permutation Contract

Permutation p-value behavior:

```text
observed = HSIC(x, y)
replicates[k] = HSIC(x, y[perm_k])
include_observed=TRUE: p = mean(c(replicates, observed) >= observed)
include_observed=FALSE: p = mean(replicates >= observed)
```

Determinism:

```text
If permutation_params$seed is non-null, native permutation order is deterministic independent of global R RNG.
If permutation_params$seed is null, R wrapper may use R RNG state and should be reproducible under set.seed().
Validation tests must use explicit seed for native deterministic tests.
```

Diagnostics:

```text
permutation_replicates
permutation_seed
permutation_include_observed
permutation_min
permutation_mean
permutation_max
```

### Conditional Test Contract

For every CI method:

```text
if S is empty:
  evaluate method on original columns x and y
if S is non-empty:
  residualize x and y on S using selected residual_backend
  evaluate method on residual vectors
```

This intentionally matches the current dCov residualization pattern. It does not implement HSIC cluster conditional testing.

### Diagnostics Contract

Add skeleton-level diagnostics:

```text
ci_method
ci_backend
ci_backend_reason
ci_tests
ci_dcov_gamma_tests
ci_hsic_gamma_tests
ci_hsic_perm_tests
ci_hsic_permutation_replicates
ci_cpu_tests
ci_cuda_tests
ci_cuda_resolved_cpu_tests
ci_nonfinite_pvalues
```

Add optional per-test diagnostics when `ci_diagnostics = TRUE`:

```text
level
test_id
x
y
conditioning_size
ci_method
ci_backend
statistic
p_value
hsic_mean
hsic_variance
permutation_replicates
reason
```

Keep this bounded. Per-test diagnostics may be a data frame stored under:

```text
skeleton$ci_diagnostics$tests
skeleton$ci_diagnostics$summary
```

WAN-PDAG orientation diagnostics should include:

```text
orientation$ci_method
orientation$ci_backend
orientation$ci_backend_reason
orientation$ci_diagnostics$regrvonps_hsic_gamma_tests
orientation$ci_diagnostics$regrvonps_hsic_perm_tests
orientation$ci_diagnostics$regrvonps_dcc_gamma_tests
```

### Validation Tolerances

Numerical tolerances:

```text
Native HSIC gamma p-values are finite and in [0, 1].
Dense native HSIC statistic is stable across repeated calls within 1e-12.
Native HSIC gamma vs legacy hsic.gamma with numCol >= n: statistic relative difference <= 0.15 on small fixtures.
Native HSIC permutation with fixed seed repeats exactly.
HSIC permutation p-value changes when seed changes on a fixture with replicates >= 20.
dCov gamma default graph output remains identical to pre-goal output under existing tests.
```

Graph tolerances:

```text
dcc.gamma vs hsic.gamma graph equality is not required.
CPU hsic.gamma vs CUDA engine CPU-resolved hsic.gamma graph equality is required.
CPU hsic.perm fixed-seed vs repeated CPU hsic.perm fixed-seed graph equality is required.
WAN-PDAG hsic.gamma CPU vs CUDA engine CPU-resolved pdag equality is required.
```

## File Structure

Expected created files:

```text
fastkpc/src/ci_method.hpp
fastkpc/src/ci_method.cpp
fastkpc/src/hsic_cpu.hpp
fastkpc/src/hsic_cpu.cpp
fastkpc/R/hsic_validation.R
fastkpc/tests/test_hsic_native_gamma.R
fastkpc/tests/test_hsic_native_permutation.R
fastkpc/tests/test_hsic_skeleton_cpu.R
fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
fastkpc/tests/test_hsic_wanpdag_pipeline.R
fastkpc/tests/test_fastkpc_ci_method_public_api.R
fastkpc/tests/test_ci_method_campaign_report_cli.R
fastkpc/tests/test_hsic_docs_contract.R
fastkpc/tests/test_hsic_benchmark.R
```

Expected modified files:

```text
fastkpc/src/fastkpc_types.hpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/wanpdag_engine.cpp
fastkpc/src/regrvonps_native.hpp
fastkpc/src/regrvonps_native.cpp
fastkpc/src/regrvonps_device.cpp
fastkpc/src/r_api_cuda.cpp
fastkpc/src/rcpp_exports.cpp
fastkpc/R/native.R
fastkpc/R/cuda_native.R
fastkpc/R/fast_kpc.R
fastkpc/R/validation_campaign.R
fastkpc/R/report_writer.R
fastkpc/R/wanpdag_validation.R
fastkpc/tools/build_cuda_native.sh
fastkpc/tools/run_fast_kpc.R
fastkpc/tools/run_validation_campaign.R
fastkpc/README.md
fastkpc/reports/README.md
```

Files that must remain unchanged:

```text
kpcalg/R/dcovgamma.R
kpcalg/R/hsicgamma.R
kpcalg/R/hsicperm.R
kpcalg/R/hsictest.R
kpcalg/R/kernelCItest.R
kpcalg/R/kpc.R
all other kpcalg/R/*.R files
```

## Phase 0: Baseline Audit And Red-Line Checks

Purpose: establish current behavior and avoid accidental default changes.

- [ ] Run:

```bash
pwd
git rev-parse --show-toplevel 2>&1 || true
find docs/superpowers/plans -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected:

```text
Current directory is /data/wenyujianData/kpcalg.
Workspace may report not a git repository.
Existing plan files include this document's predecessor.
```

- [ ] Run the current default public wrapper tests before editing:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Record legacy HSIC source behavior without modifying it:

```bash
Rscript -e 'source("kpcalg/R/hsicgamma.R"); source("kpcalg/R/hsicperm.R"); set.seed(201); x <- runif(30); y <- sin(x) + rnorm(30, sd=0.1); print(hsic.gamma(x,y,numCol=30)$p.value); set.seed(202); print(hsic.perm(x,y,p=20,numCol=30)$p.value)'
```

Expected:

```text
Command either prints p-values or errors because kernlab is unavailable.
If kernlab is unavailable, later legacy-comparison tests must skip with explicit reason.
```

## Phase 1: TDD Red Tests For Native HSIC Numerics

Purpose: define native HSIC behavior before implementation.

- [ ] Create `fastkpc/tests/test_hsic_native_gamma.R`.

Required test content:

```r
source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(211)
n <- 48
x <- seq(-2, 2, length.out = n)
y_dep <- sin(x) + rnorm(n, sd = 0.05)
y_ind <- sample(y_dep)

build_fastkpc_native(rebuild = TRUE)

dep <- fast_hsic_gamma_cpp(x, y_dep, sig = 1)
ind <- fast_hsic_gamma_cpp(x, y_ind, sig = 1)
repeat_dep <- fast_hsic_gamma_cpp(x, y_dep, sig = 1)

assert_true(is.list(dep), "HSIC gamma result should be a list")
assert_true(is.finite(dep$statistic), "HSIC statistic should be finite")
assert_true(is.finite(dep$p.value), "HSIC p-value should be finite")
assert_true(dep$p.value >= 0 && dep$p.value <= 1,
            "HSIC p-value should be in [0, 1]")
assert_true(dep$statistic > ind$statistic,
            "dependent fixture should have larger HSIC statistic")
assert_true(abs(dep$statistic - repeat_dep$statistic) < 1e-12,
            "HSIC gamma should repeat exactly")
assert_true(all(c("hsic", "mean", "variance", "shape", "scale") %in%
                  names(dep$diagnostics)),
            "HSIC gamma diagnostics should include gamma fields")

cat("test_hsic_native_gamma.R: PASS\n")
```

- [ ] Run the red test:

```bash
Rscript fastkpc/tests/test_hsic_native_gamma.R
```

Expected:

```text
FAIL because fast_hsic_gamma_cpp does not exist.
```

- [ ] Create `fastkpc/tests/test_hsic_native_permutation.R`.

Required test content:

```r
source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(212)
n <- 40
x <- seq(-1, 1, length.out = n)
y <- x^2 + rnorm(n, sd = 0.04)

build_fastkpc_native(rebuild = TRUE)

a <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 99L,
                        include_observed = TRUE)
b <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 99L,
                        include_observed = TRUE)
c <- fast_hsic_perm_cpp(x, y, sig = 1, replicates = 30L, seed = 100L,
                        include_observed = TRUE)

assert_true(is.finite(a$p.value), "HSIC permutation p-value should be finite")
assert_true(a$p.value >= 0 && a$p.value <= 1,
            "HSIC permutation p-value should be in [0, 1]")
assert_true(length(a$replicates) == 30L,
            "HSIC permutation should return requested replicates")
assert_true(identical(a$replicates, b$replicates),
            "HSIC permutation fixed seed should repeat exactly")
assert_true(!identical(a$replicates, c$replicates),
            "HSIC permutation different seed should change replicate order")
assert_true(a$diagnostics$replicates == 30L,
            "HSIC permutation diagnostics should record replicate count")

cat("test_hsic_native_permutation.R: PASS\n")
```

- [ ] Run the red test:

```bash
Rscript fastkpc/tests/test_hsic_native_permutation.R
```

Expected:

```text
FAIL because fast_hsic_perm_cpp does not exist.
```

## Phase 2: Native HSIC CPU Implementation

Purpose: add dense native HSIC functions with a small R-facing export.

- [ ] Create `fastkpc/src/hsic_cpu.hpp`.

Required declarations:

```cpp
#ifndef FASTKPC_HSIC_CPU_HPP
#define FASTKPC_HSIC_CPU_HPP

#include <string>
#include <vector>

struct HsicOptions {
  double sig;
  int permutation_replicates;
  int permutation_seed;
  bool has_permutation_seed;
  bool include_observed;
};

struct HsicResult {
  double statistic;
  double p_value;
  double mean;
  double variance;
  double shape;
  double scale;
  std::vector<double> replicates;
  std::string method;
  std::string reason;
};

HsicOptions default_hsic_options();

HsicResult hsic_gamma_cpu(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const HsicOptions& options);

HsicResult hsic_permutation_cpu(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const HsicOptions& options);

#endif
```

- [ ] Create `fastkpc/src/hsic_cpu.cpp`.

Implementation requirements:

```text
1. Validate equal vector lengths.
2. Require n > 5 for gamma.
3. Require finite values.
4. Build dense RBF kernels using sigma = 1 / sig.
5. Center kernels by subtracting row means, column means, and adding grand mean.
6. Compute HSIC as sum(Kc * Lc) / n^2.
7. Compute gamma mean/variance using stable dense approximations.
8. Use R::pgamma(statistic, shape, scale, false, false).
9. Use std::mt19937 for explicit seed permutation.
10. Return reason when variance is invalid.
```

Suggested helper names:

```text
validate_hsic_vectors
rbf_kernel
center_kernel
hsic_statistic_from_centered
gamma_approximation
permuted_copy
```

- [ ] Modify `fastkpc/src/rcpp_exports.cpp`.

Add includes:

```cpp
#include "hsic_cpu.hpp"
```

Add list conversion helper:

```cpp
Rcpp::List hsic_result_to_list(const HsicResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("statistic") = result.statistic,
    Rcpp::Named("estimate") = result.statistic,
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("HSIC") = result.statistic,
      Rcpp::Named("HSIC mean") = result.mean,
      Rcpp::Named("HSIC variance") = result.variance
    ),
    Rcpp::Named("p.value") = result.p_value,
    Rcpp::Named("replicates") =
      Rcpp::NumericVector(result.replicates.begin(), result.replicates.end()),
    Rcpp::Named("method") = result.method,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("hsic") = result.statistic,
      Rcpp::Named("mean") = result.mean,
      Rcpp::Named("variance") = result.variance,
      Rcpp::Named("shape") = result.shape,
      Rcpp::Named("scale") = result.scale,
      Rcpp::Named("reason") = result.reason
    )
  );
}
```

Add exported functions:

```cpp
// [[Rcpp::export]]
Rcpp::List fast_hsic_gamma_cpp_export(Rcpp::NumericVector xs,
                                      Rcpp::NumericVector ys,
                                      double sig = 1.0) {
  HsicOptions options = default_hsic_options();
  options.sig = sig;
  std::vector<double> x(xs.begin(), xs.end());
  std::vector<double> y(ys.begin(), ys.end());
  return hsic_result_to_list(hsic_gamma_cpu(x, y, options));
}

// [[Rcpp::export]]
Rcpp::List fast_hsic_perm_cpp_export(Rcpp::NumericVector xs,
                                     Rcpp::NumericVector ys,
                                     double sig = 1.0,
                                     int replicates = 100,
                                     int seed = NA_INTEGER,
                                     bool include_observed = true) {
  HsicOptions options = default_hsic_options();
  options.sig = sig;
  options.permutation_replicates = replicates;
  options.has_permutation_seed = seed != NA_INTEGER;
  options.permutation_seed = options.has_permutation_seed ? seed : 0;
  options.include_observed = include_observed;
  std::vector<double> x(xs.begin(), xs.end());
  std::vector<double> y(ys.begin(), ys.end());
  return hsic_result_to_list(hsic_permutation_cpu(x, y, options));
}
```

- [ ] Modify `fastkpc/R/native.R`.

Add wrappers:

```r
fast_hsic_gamma_cpp <- function(x, y, sig = 1) {
  build_fastkpc_native()
  fast_hsic_gamma_cpp_export(as.numeric(x), as.numeric(y), as.numeric(sig))
}

fast_hsic_perm_cpp <- function(x, y, sig = 1, replicates = 100L,
                               seed = NA_integer_,
                               include_observed = TRUE) {
  build_fastkpc_native()
  fast_hsic_perm_cpp_export(as.numeric(x), as.numeric(y), as.numeric(sig),
                            as.integer(replicates), as.integer(seed),
                            isTRUE(include_observed))
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_hsic_native_gamma.R
Rscript fastkpc/tests/test_hsic_native_permutation.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 3: CI Method Abstraction In C++

Purpose: stop hard-coding dCov inside skeleton and orientation numerical checks.

- [ ] Create `fastkpc/src/ci_method.hpp`.

Required declarations:

```cpp
#ifndef FASTKPC_CI_METHOD_HPP
#define FASTKPC_CI_METHOD_HPP

#include "hsic_cpu.hpp"

#include <string>
#include <vector>

enum class CiMethodKind {
  DcovGamma,
  HsicGamma,
  HsicPermutation
};

struct CiMethodOptions {
  CiMethodKind method;
  std::string method_name;
  double dcov_index;
  bool legacy_index;
  HsicOptions hsic;
};

struct CiMethodResult {
  double p_value;
  double statistic;
  double hsic_mean;
  double hsic_variance;
  int permutation_replicates;
  std::string method_name;
  std::string backend;
  std::string reason;
};

CiMethodKind parse_ci_method_kind(const std::string& method);

CiMethodOptions make_ci_method_options(const std::string& method,
                                       double index,
                                       bool legacy_index,
                                       const HsicOptions& hsic);

CiMethodResult evaluate_ci_vectors(const std::vector<double>& x,
                                   const std::vector<double>& y,
                                   const CiMethodOptions& options);

#endif
```

- [ ] Create `fastkpc/src/ci_method.cpp`.

Implementation requirements:

```text
parse_ci_method_kind("dcc.gamma") -> DcovGamma
parse_ci_method_kind("hsic.gamma") -> HsicGamma
parse_ci_method_kind("hsic.perm") -> HsicPermutation
unknown method throws "Unknown CI method: <value>"
evaluate_ci_vectors dispatches to dcov_exact_pvalue, hsic_gamma_cpu, or hsic_permutation_cpu
```

- [ ] Modify `fastkpc/src/fastkpc_types.hpp`.

Add to `SkeletonOptions`:

```cpp
std::string ci_method_name;
HsicOptions hsic_options;
bool ci_diagnostics_enabled;
```

Add to `SkeletonResult`:

```cpp
std::string ci_method;
std::string ci_backend;
std::string ci_backend_reason;
int ci_tests;
int ci_dcov_gamma_tests;
int ci_hsic_gamma_tests;
int ci_hsic_perm_tests;
int ci_hsic_permutation_replicates;
int ci_cpu_tests;
int ci_cuda_tests;
int ci_cuda_resolved_cpu_tests;
int ci_nonfinite_pvalues;
```

- [ ] Modify `fastkpc/src/skeleton_engine.cpp`.

Replace `ci_pvalue_exact()` with a result-returning evaluator:

```text
ci_test_exact(data, x, y, conditioning_set, options, residual_cache)
```

Behavior:

```text
S empty: extract original vectors.
S non-empty: residual_cache->get x and y.
Build CiMethodOptions from SkeletonOptions.
Call evaluate_ci_vectors().
Return p_value and diagnostics.
Increment SkeletonResult CI counters.
```

- [ ] Preserve pMax/delete semantics:

```text
pMax updates still use p_value.
Non-finite p_value follows existing na_delete rule.
Sepsets and n.edgetests remain unchanged for ci_method="dcc.gamma".
```

- [ ] Run existing dCov skeleton tests:

```bash
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_residual_cache_core.R
```

Expected:

```text
Every test prints PASS.
```

## Phase 4: R Native Wrappers For CI Method

Purpose: expose CI method options through CPU native wrappers.

- [ ] Modify `fastkpc/src/rcpp_exports.cpp`.

Add parser:

```text
parse_hsic_params(Rcpp::List values)
parse_permutation_params(Rcpp::List values)
make_skeleton_options(... ci_method, hsic_params, permutation_params, ci_diagnostics ...)
```

For missing values:

```text
hsic sig default 1
permutation replicates default 100
permutation seed absent unless supplied
include_observed default TRUE
ci_diagnostics default TRUE
```

- [ ] Modify exported CPU skeleton functions:

```text
fast_skeleton_cpp_backend_export
fast_kpc_wanpdag_cpp_export
fast_orient_wanpdag_cpp_export
```

Add CI arguments to R-facing wrappers while preserving existing calls by providing R defaults in `fastkpc/R/native.R`.

- [ ] Modify result conversion helpers:

Add skeleton fields:

```r
ci_method
ci_backend
ci_backend_reason
ci_diagnostics = list(summary = list(...))
```

Add orientation fields:

```r
ci_method
ci_backend
ci_backend_reason
ci_diagnostics
```

- [ ] Modify `fastkpc/R/native.R`.

Add parameters to:

```text
fast_skeleton_cpp_backend()
fast_orient_wanpdag_cpp()
fast_kpc_wanpdag_cpp()
```

Parameter defaults:

```r
ci_method = "dcc.gamma"
hsic_params = list()
permutation_params = list()
ci_diagnostics = TRUE
```

- [ ] Create `fastkpc/tests/test_hsic_skeleton_cpu.R`.

Required assertions:

```text
1. fast_skeleton_cpp_backend(..., ci_method="hsic.gamma") returns skeleton$ci_method == "hsic.gamma".
2. pMax is finite off diagonal.
3. adjacency is symmetric.
4. repeated hsic.gamma calls are identical.
5. hsic.perm with fixed seed repeats exactly.
6. hsic.perm with different seed can change pMax on at least one pair or records different diagnostics.
7. dcc.gamma default call still reports ci_method == "dcc.gamma".
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 5: CUDA Resolution For HSIC Methods

Purpose: make CUDA wrappers accept HSIC methods without pretending GPU HSIC exists.

- [ ] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

Resolution behavior:

```text
ci_method="dcc.gamma": keep existing CUDA dCov path.
ci_method="hsic.gamma": use CPU-native CI evaluator inside CUDA skeleton control flow.
ci_method="hsic.perm": use CPU-native CI evaluator inside CUDA skeleton control flow.
ci_backend="native-cpu"
ci_backend_reason="CUDA HSIC gamma backend is not implemented in this stage"
ci_cuda_resolved_cpu_tests increments by evaluated test count.
```

Implementation constraint:

```text
Do not batch HSIC through dcov_batch_cuda.
Do not change layer scheduler replay order.
Do not use CUDA dCov p-values for HSIC methods.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Parse and pass:

```text
ci_method
hsic_params
permutation_params
ci_diagnostics
```

Update `C_fast_skeleton_cuda_backend` and `C_fast_kpc_wanpdag_cuda` arity and registration.

- [ ] Modify `fastkpc/R/cuda_native.R`.

Add same parameters to:

```text
fast_skeleton_cuda_backend()
fast_kpc_wanpdag_cuda()
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Add objects:

```text
hsic_cpu.o
ci_method.o
```

Link them into `fastkpc/build/fastkpc_cuda.so`.

- [ ] Create `fastkpc/tests/test_hsic_skeleton_cuda_resolution.R`.

Required assertions:

```text
1. CUDA wrapper accepts ci_method="hsic.gamma".
2. CUDA result records ci_method == "hsic.gamma".
3. CUDA result records ci_backend == "native-cpu".
4. CUDA result records non-empty ci_backend_reason mentioning CUDA HSIC gamma.
5. CPU hsic.gamma skeleton adjacency equals CUDA engine CPU-resolved hsic.gamma adjacency.
6. CPU hsic.gamma pMax equals CUDA engine CPU-resolved hsic.gamma pMax within 1e-12.
7. ci_method="hsic.perm" with fixed seed repeats exactly under CUDA wrapper.
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
```

Expected:

```text
Build succeeds.
Both tests print PASS.
```

## Phase 6: WAN-PDAG And Orientation CI Method Integration

Purpose: make WAN-PDAG generalized orientation use the selected CI method consistently.

- [ ] Modify `fastkpc/src/orientation_types.hpp`.

Add to `OrientationOptions`:

```cpp
std::string ci_method_name;
HsicOptions hsic_options;
bool ci_diagnostics_enabled;
```

Add to `OrientationResult`:

```cpp
std::string ci_method;
std::string ci_backend;
std::string ci_backend_reason;
int regrvonps_dcc_gamma_tests;
int regrvonps_hsic_gamma_tests;
int regrvonps_hsic_perm_tests;
```

- [ ] Modify `fastkpc/src/regrvonps_native.cpp`.

Replace direct `dcov_exact_pvalue(residuals, other, ...)` call with `evaluate_ci_vectors()`.

Counter behavior:

```text
ci_method="dcc.gamma": increment regrvonps_dcc_gamma_tests
ci_method="hsic.gamma": increment regrvonps_hsic_gamma_tests
ci_method="hsic.perm": increment regrvonps_hsic_perm_tests
```

- [ ] Modify `fastkpc/src/regrvonps_device.cpp`.

For orientation residual device CUDA:

```text
ci_method="dcc.gamma": keep CUDA dCov batch behavior.
ci_method="hsic.gamma": compute residuals as requested, then evaluate HSIC on CPU vectors; record ci_backend native-cpu.
ci_method="hsic.perm": compute residuals as requested, then evaluate HSIC permutation on CPU vectors; record ci_backend native-cpu.
```

This preserves CUDA residual acceleration while acknowledging HSIC CI evaluation is CPU in this goal.

- [ ] Modify `fastkpc/src/wanpdag_engine.cpp`.

Initialize and copy orientation CI fields into `OrientationResult`.

- [ ] Create `fastkpc/tests/test_hsic_wanpdag_pipeline.R`.

Required assertions:

```text
1. fast_kpc_wanpdag_cpp(..., ci_method="hsic.gamma") returns orientation$ci_method == "hsic.gamma".
2. fast_kpc_wanpdag_cuda(..., ci_method="hsic.gamma") accepts the method.
3. CUDA engine CPU-resolved HSIC pdag equals CPU HSIC pdag.
4. Orientation counts are identical.
5. regrvonps_hsic_gamma_tests > 0 on a fixture that exercises generalized orientation.
6. hsic.perm with fixed seed repeats pdag and event count exactly.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_hsic_wanpdag_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
```

Expected:

```text
Every test prints PASS.
```

## Phase 7: Public fast_kpc API

Purpose: expose CI method without breaking current public calls.

- [ ] Modify `fastkpc/R/fast_kpc.R`.

Add to signature:

```r
ci_method = c("dcc.gamma", "hsic.gamma", "hsic.perm"),
ci_diagnostics = TRUE,
hsic_params = list(),
permutation_params = list(),
```

Add config fields:

```r
ci_method_requested
ci_method
ci_backend
ci_backend_reason
ci_diagnostics
hsic_params
permutation_params
```

Pass CI fields to CPU and CUDA wrappers for both skeleton and WAN-PDAG graph stages.

- [ ] Update `validate_fastkpc_result()`.

Required config fields:

```text
ci_method_requested
ci_method
ci_backend
ci_backend_reason
ci_diagnostics
```

- [ ] Update `fastkpc_result_summary()` and `print.fastkpc_result()`.

Add compact display:

```text
ci_method: hsic.gamma
ci_backend: native-cpu
```

- [ ] Create `fastkpc/tests/test_fastkpc_ci_method_public_api.R`.

Required assertions:

```text
1. fast_kpc(..., ci_method="hsic.gamma", engine="cpu") returns fastkpc_result.
2. config$ci_method == "hsic.gamma".
3. skeleton$ci_method == "hsic.gamma".
4. graph_stage="skeleton" works.
5. graph_stage="wanpdag" works.
6. engine="cuda", ci_method="hsic.gamma" records native-cpu or cuda-resolved-cpu backend reason.
7. ci_method default remains "dcc.gamma".
8. existing residual_device and orientation_residual_device config fields remain present.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
```

Expected:

```text
Every test prints PASS.
```

## Phase 8: Validation Helpers

Purpose: provide focused HSIC validation and benchmark helpers.

- [ ] Create `fastkpc/R/hsic_validation.R`.

Required functions:

```r
validate_hsic_native_gamma <- function(seed = 221, n = 60, sig = 1)
validate_hsic_native_permutation <- function(seed = 222, n = 50,
                                             replicates = 40)
compare_ci_methods_graphs <- function(seed = 223, n = 100,
                                      methods = c("dcc.gamma", "hsic.gamma"))
compare_hsic_cpu_cuda_resolution <- function(seed = 224, n = 90)
benchmark_hsic_backends <- function(seed = 225, n = 120, repeats = 2)
```

Required return tables:

```text
validate_hsic_native_gamma()$metrics
validate_hsic_native_permutation()$metrics
compare_ci_methods_graphs()$diffs
compare_hsic_cpu_cuda_resolution()$metrics
benchmark_hsic_backends()$timings
benchmark_hsic_backends()$summary
```

- [ ] Create `fastkpc/tests/test_hsic_benchmark.R`.

Required assertions:

```text
1. validate_hsic_native_gamma() returns finite native p-value.
2. validate_hsic_native_permutation() returns deterministic fixed-seed replicates.
3. compare_hsic_cpu_cuda_resolution() reports adjacency_identical TRUE.
4. benchmark_hsic_backends() returns CPU timing rows for hsic.gamma and hsic.perm.
5. Benchmark has no strict speedup assertion.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_hsic_benchmark.R
```

Expected:

```text
Test prints PASS.
```

## Phase 9: Validation Campaign, Report Writer, And CLI

Purpose: make HSIC visible in reproducible campaign artifacts.

- [ ] Modify `fastkpc/R/validation_campaign.R`.

Add argument:

```r
ci_methods = c("dcc.gamma")
hsic_params = list()
permutation_params = list()
ci_diagnostics = TRUE
```

Add grid dimension:

```text
ci_method
```

Update run IDs:

```text
scenario-seed-n-engine-ci_method-residual_backend-residual_device-scheduler-orientation_device
```

Add flattened diagnostics:

```text
ci_method_diagnostics
```

Columns:

```text
run_id
scenario
seed
n
engine
ci_method
ci_backend
ci_backend_reason
residual_backend
residual_device
scheduler
ci_tests
ci_dcov_gamma_tests
ci_hsic_gamma_tests
ci_hsic_perm_tests
ci_hsic_permutation_replicates
ci_cpu_tests
ci_cuda_tests
ci_cuda_resolved_cpu_tests
ci_nonfinite_pvalues
```

Add diff table:

```text
ci_method_diffs
```

Compare matched `dcc.gamma` vs `hsic.gamma` and `hsic.gamma` vs `hsic.perm` when both exist. Graph equality is reported, not required.

- [ ] Modify `fastkpc/R/report_writer.R`.

Add artifacts:

```text
ci_method_diffs.csv
ci_method_diagnostics.csv
```

Add Markdown sections:

```markdown
## CI Method
## CI Method Diagnostics
```

- [ ] Modify `fastkpc/tools/run_fast_kpc.R`.

Add options:

```text
--ci-method
--hsic-sig
--permutation-replicates
--permutation-seed
--ci-diagnostics
```

Print:

```text
ci_method=<value>
ci_backend=<value>
ci_tests=<value>
ci_hsic_gamma_tests=<value>
ci_hsic_perm_tests=<value>
```

- [ ] Modify `fastkpc/tools/run_validation_campaign.R`.

Add options:

```text
--ci-methods
--hsic-sig
--permutation-replicates
--permutation-seed
--ci-diagnostics
```

- [ ] Create `fastkpc/tests/test_ci_method_campaign_report_cli.R`.

Required assertions:

```text
1. Campaign accepts ci_methods=c("dcc.gamma","hsic.gamma").
2. Campaign object has ci_method_diffs.
3. Campaign object has ci_method_diagnostics.
4. Report writes ci_method_diffs.csv.
5. Report writes ci_method_diagnostics.csv.
6. Summary markdown contains "CI Method".
7. run_fast_kpc.R accepts --ci-method hsic.gamma.
8. run_validation_campaign.R accepts --ci-methods dcc.gamma,hsic.gamma.
9. CLI output prints ci_method=hsic.gamma.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_ci_method_campaign_report_cli.R
```

Expected:

```text
Test prints PASS.
```

## Phase 10: Documentation And Contract Tests

Purpose: document public controls and keep docs synchronized with behavior.

- [ ] Modify `fastkpc/README.md`.

Add section:

```markdown
## HSIC CI Methods

fastkpc supports opt-in native CPU HSIC gamma and HSIC permutation CI methods
through `ci_method`.
```

Required text:

```text
ci_method
dcc.gamma
hsic.gamma
hsic.perm
hsic_params
permutation_params
ci_diagnostics
ci_method_diffs.csv
ci_method_diagnostics.csv
CUDA HSIC resolves to CPU in this stage
kpcalg/R/*.R files are not modified
```

- [ ] Modify `fastkpc/reports/README.md`.

Add artifacts:

```text
ci_method_diffs.csv
ci_method_diagnostics.csv
```

Explain that graph differences across CI methods are expected and reported.

- [ ] Create `fastkpc/tests/test_hsic_docs_contract.R`.

Required assertions:

```text
README mentions HSIC CI Methods.
README mentions ci_method.
README mentions hsic.gamma.
README mentions hsic.perm.
README mentions hsic_params.
README mentions permutation_params.
README mentions CUDA HSIC resolves to CPU.
reports README mentions ci_method_diffs.csv.
reports README mentions ci_method_diagnostics.csv.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_hsic_docs_contract.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
Rscript fastkpc/tests/test_cuda_residual_docs_contract.R
```

Expected:

```text
Every test prints PASS.
```

## Phase 11: Reproducibility And Determinism

Purpose: avoid adding flaky permutation behavior.

- [ ] Add reproducibility assertions to `fastkpc/tests/test_fastkpc_reproducibility.R`.

Required checks:

```text
1. fast_kpc(..., ci_method="hsic.gamma") repeated with same seed returns identical skeleton adjacency and pMax.
2. fast_kpc(..., ci_method="hsic.perm", permutation seed fixed) repeated returns identical skeleton adjacency and pMax.
3. validation campaign with ci_methods=c("hsic.gamma") repeated returns identical runs table except elapsed time.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_reproducibility.R
```

Expected:

```text
Test prints PASS.
```

## Phase 12: Full Verification Campaign

Purpose: prove the new HSIC CI backend is complete and does not regress earlier stages.

- [ ] Run clean builds:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
CUDA native build succeeds.
CPU native build succeeds.
```

- [ ] Run new HSIC focused tests:

```bash
Rscript fastkpc/tests/test_hsic_native_gamma.R
Rscript fastkpc/tests/test_hsic_native_permutation.R
Rscript fastkpc/tests/test_hsic_skeleton_cpu.R
Rscript fastkpc/tests/test_hsic_skeleton_cuda_resolution.R
Rscript fastkpc/tests/test_hsic_wanpdag_pipeline.R
Rscript fastkpc/tests/test_fastkpc_ci_method_public_api.R
Rscript fastkpc/tests/test_ci_method_campaign_report_cli.R
Rscript fastkpc/tests/test_hsic_docs_contract.R
Rscript fastkpc/tests/test_hsic_benchmark.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run existing public/WAN-PDAG tests:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
Rscript fastkpc/tests/test_wanpdag_orientation_device_validation.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run existing CUDA skeleton/residual/scheduler tests:

```bash
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run existing CPU/non-CUDA regression tests:

```bash
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_orientation_matrix.R
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_diff_report.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run validation helper smoke:

```bash
Rscript -e 'source("fastkpc/R/hsic_validation.R"); x <- compare_hsic_cpu_cuda_resolution(); print(x$metrics); stopifnot(isTRUE(x$metrics$adjacency_identical)); stopifnot(isTRUE(x$metrics$pmax_close))'
```

Expected:

```text
No error.
Metrics show CPU/CUDA HSIC gamma resolution equivalence.
```

- [ ] Run compact CI-method validation campaign:

```bash
rm -rf fastkpc/reports/ci_method_smoke
Rscript fastkpc/tools/run_validation_campaign.R \
  --engines cpu,cuda \
  --residual-backends fastSpline \
  --residual-devices cpu \
  --ci-methods dcc.gamma,hsic.gamma \
  --schedulers legacy \
  --seeds 11 \
  --n-values 80 \
  --scenarios chain,additive \
  --legacy FALSE \
  --output-dir fastkpc/reports/ci_method_smoke
Rscript -e 'x <- read.csv("fastkpc/reports/ci_method_smoke/ci_method_diagnostics.csv"); print(x[, c("scenario", "engine", "ci_method", "ci_backend", "ci_tests")]); stopifnot(any(x$ci_method == "hsic.gamma"), any(x$ci_tests > 0))'
```

Expected:

```text
Campaign completes.
Report directory contains ci_method_diffs.csv.
Report directory contains ci_method_diagnostics.csv.
At least one HSIC row records ci_tests > 0.
```

- [ ] Verify legacy package files remain unchanged:

```bash
cd kpcalg
md5sum -c MD5 | rg '^R/'
cd ..
```

Expected:

```text
Every kpcalg/R MD5 line reports OK.
```

## Completion Criteria

The goal is complete only when all criteria are true:

```text
1. fast_hsic_gamma_cpp() and fast_hsic_perm_cpp() exist and return finite p-values on fixed fixtures.
2. Native HSIC gamma is deterministic across repeated calls.
3. Native HSIC permutation is deterministic with fixed seed and changes replicate order with different seed.
4. Skeleton CPU backend accepts ci_method="dcc.gamma", "hsic.gamma", and "hsic.perm".
5. Existing dcc.gamma default skeleton behavior remains unchanged under existing tests.
6. CUDA skeleton accepts HSIC methods and records CPU resolution reason rather than pretending GPU HSIC exists.
7. WAN-PDAG CPU and CUDA wrappers accept HSIC methods and preserve CPU-vs-CUDA CPU-resolved graph equality.
8. Public fast_kpc() accepts ci_method, hsic_params, permutation_params, and ci_diagnostics.
9. fastkpc_result config/result validation includes CI method fields.
10. Validation campaign supports ci_methods as a dimension.
11. Report writer emits ci_method_diffs.csv and ci_method_diagnostics.csv.
12. CLI tools accept CI method options and print useful CI counters.
13. README and reports README document HSIC CI methods and artifacts.
14. HSIC benchmark/validation helpers exist and pass smoke tests.
15. All commands in Phase 12 pass in the local environment.
16. kpcalg/R MD5 checks remain OK.
```

## Notes For Future Workers

Keep the implementation conservative:

```text
Do not change the default CI method.
Do not edit kpcalg/R files.
Do not require graph equality between dCov and HSIC.
Do not add CUDA HSIC kernels in this goal.
Do not hide CUDA HSIC CPU resolution.
Do not let permutation tests become flaky.
Use explicit seeds in every permutation test.
Prefer dense native HSIC correctness before optimizing.
```

The dense HSIC implementation is allowed to differ from legacy incomplete-Cholesky approximations. Document that distinction clearly and validate it with bounded behavioral comparisons rather than pretending it is bit-identical to `kernlab::inchol()`.

The first implementation should make future GPU HSIC possible by centralizing CI method dispatch. The dispatch boundary is the main architectural deliverable; the math functions and public plumbing are the behavioral deliverables.
