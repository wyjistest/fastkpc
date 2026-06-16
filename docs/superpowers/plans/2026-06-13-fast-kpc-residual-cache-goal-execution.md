# Fast kPC Residual Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared residual cache to the staged `fastkpc` CPU and CUDA skeleton backends so repeated conditional tests reuse residual vectors keyed by target variable, conditioning set, backend, and backend parameters.

**Architecture:** Keep the graph scheduler and exact dCov backends unchanged in behavior. Extract residualization into a small C++ residual backend/cache layer, then route both `run_skeleton_exact()` and `run_skeleton_cuda_batch()` through it. Expose cache statistics to R and validate that cached and uncached runs produce identical graphs while reducing residualization work.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, existing CPU exact dCov backend, existing CUDA batched dCov backend, current CPU linear residualization MVP as the first cached residual backend.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-residual-cache-goal-execution.md: add a shared residual cache for CPU and CUDA skeleton paths, preserving graph behavior and keeping legacy kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `140000`.

Do not mark the goal complete until all completion criteria in Phase 6 are satisfied. Mark the goal blocked only if the same blocker prevents progress for three consecutive goal turns and cannot be resolved locally.

## Preconditions

The previous CUDA batched dCov goal must already be complete.

Required current artifacts:

```text
fastkpc/R/native.R
fastkpc/R/cuda_native.R
fastkpc/R/cuda_validation.R
fastkpc/R/diff_report.R
fastkpc/R/legacy_runner.R
fastkpc/src/dcov_exact_cpu.cpp
fastkpc/src/dcov_exact_cpu.hpp
fastkpc/src/skeleton_engine.cpp
fastkpc/src/skeleton_engine.hpp
fastkpc/src/skeleton_engine_cuda.cpp
fastkpc/src/skeleton_engine_cuda.hpp
fastkpc/src/r_api_cuda.cpp
fastkpc/src/rcpp_exports.cpp
fastkpc/tests/test_skeleton_mvp.R
fastkpc/tests/test_skeleton_cuda_batch.R
fastkpc/tools/build_cuda_native.sh
fastkpc/tools/clean_cuda_native.sh
```

Required environment:

```text
R 4.4.1
Rcpp installed
CUDA toolkit available at /usr/local/cuda/bin/nvcc
NVIDIA driver able to run CUDA kernels
```

If CUDA is unavailable, CPU residual cache work may still be implemented, but the goal cannot be marked complete until CUDA skeleton cache tests also pass.

## Scope

In scope for this goal:

- Add a C++ residual cache keyed by:

```text
target variable
sorted conditioning set
residual backend name
residual backend parameter string
sample count
variable count
```

- Add cache statistics:

```text
requests
hits
misses
computations
stored_vectors
stored_values
backend_name
enabled
```

- Route CPU exact skeleton conditional residualization through the cache.
- Route CUDA skeleton conditional residualization through the same cache while preserving CUDA batched dCov replay semantics.
- Add R wrappers to enable/disable cache and return cache statistics.
- Add tests showing cached and uncached CPU/CUDA skeleton outputs are identical.
- Add tests showing cache hits occur on a fixed scenario with repeated residual requests.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope for this goal:

- Do not implement fastSpline.
- Do not implement CUDA GAM.
- Do not implement GPU residualization.
- Do not implement legacy `mgcv` equivalence as a production backend.
- Do not migrate `udag2wanpdag()`.
- Do not replace exported `kpcalg::kpc()`.
- Do not support HSIC or permutation tests.

## Design Contract

### Residual Backend Contract

The first cached residual backend is the existing CPU linear MVP residualization currently implemented as:

```cpp
std::vector<double> residualize_lm(const Rcpp::NumericMatrix& data,
                                   int target,
                                   const std::vector<int>& conditioning_set);
```

This goal must keep the numeric output of `residualize_lm()` unchanged.

Add a backend descriptor:

```text
backend_name = "linear"
backend_params = "intercept=true;ridge=1e-8"
```

The current solver adds a small ridge only when a pivot is near singular. Keep that behavior unchanged in this goal; the parameter string documents compatibility, not a new tunable interface.

### Cache Semantics

For unconditional tests, no residual is requested and no cache event is recorded.

For conditional tests, each p-value requires two residual vectors:

```text
residual(target = x, S)
residual(target = y, S)
```

Cache key canonicalization:

```text
conditioning_set must be sorted ascending before key construction.
target is zero-based in C++ keys.
backend_name and backend_params are part of the key.
nrow(data) and ncol(data) are part of the key.
```

Cache value:

```text
std::vector<double> residuals length n
```

Cache behavior:

```text
enabled = FALSE: always compute residuals, record requests and computations, do not store or hit.
enabled = TRUE: return stored residuals on hit; compute and store on miss.
```

Cache lifetime:

```text
One cache object per skeleton run.
No global cache across R calls.
No pointer or persistent GPU context in this goal.
```

### Skeleton Behavior Contract

Cached and uncached runs must return graph-identical results:

```text
adjacency identical
sepsets identical after sorting each set
n.edgetests identical
pMax max absolute difference < 1e-10 for CPU cached vs CPU uncached
pMax max absolute difference < 1e-8 for CUDA cached vs CPU uncached
```

The cache must not change task enumeration, stable-level snapshots, deletion replay, pMax updates, or sepset recording.

### R API Contract

Add these wrappers:

```r
fast_skeleton_cpp_cached(data, alpha, max_conditioning_size,
                         index = 1, legacy_index = TRUE,
                         residual_cache = TRUE)

fast_skeleton_cuda_cached(data, alpha, max_conditioning_size,
                          index = 1, legacy_index = TRUE,
                          batch_size = 0,
                          residual_cache = TRUE)
```

Return value must extend existing skeleton result lists with:

```text
backend: "cpu" or "cuda"
residual_backend: "linear"
residual_cache: list(
  enabled,
  requests,
  hits,
  misses,
  computations,
  stored_vectors,
  stored_values,
  backend_name
)
```

Existing wrappers `fast_skeleton_cpp()` and `fast_skeleton_cuda()` must remain available and keep their current behavior. They may call the cached implementation with `residual_cache = FALSE` internally if output compatibility is preserved.

## File Structure

Create these files:

- `fastkpc/src/residual_cache.hpp`
- `fastkpc/src/residual_cache.cpp`  
  Residual cache key, cache stats, cache object, and cached residual accessor.

- `fastkpc/src/residual_backend.hpp`
- `fastkpc/src/residual_backend.cpp`  
  Linear residual backend descriptor and a small wrapper around existing `residualize_lm()`.

- `fastkpc/R/residual_validation.R`  
  CPU/CUDA cached-vs-uncached graph validation helpers.

- `fastkpc/tests/test_residual_cache_core.R`
- `fastkpc/tests/test_skeleton_residual_cache.R`
- `fastkpc/tests/test_cuda_residual_cache.R`

Modify these files:

- `fastkpc/src/fastkpc_types.hpp`  
  Add residual cache options and stats to `SkeletonOptions` and `SkeletonResult`.

- `fastkpc/src/dcov_exact_cpu.hpp`
- `fastkpc/src/dcov_exact_cpu.cpp`  
  Keep `residualize_lm()` available; remove unused private helpers only if needed.

- `fastkpc/src/skeleton_engine.cpp`  
  Route conditional residualization through cache-aware helpers.

- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Route conditional residualization through the same cache-aware helpers.

- `fastkpc/src/rcpp_exports.cpp`  
  Add CPU cached entry point and serialize residual cache stats.

- `fastkpc/src/r_api_cuda.cpp`  
  Add CUDA cached entry point and serialize residual cache stats.

- `fastkpc/R/native.R`  
  Add `fast_skeleton_cpp_cached()`.

- `fastkpc/R/cuda_native.R`  
  Add `fast_skeleton_cuda_cached()`.

- `fastkpc/tools/build_cuda_native.sh`  
  Include new residual cache/backend C++ sources.

- `fastkpc/README.md`  
  Document residual cache API, tests, stats, and known limits.

Do not modify these files:

- `kpcalg/R/*.R`
- `gpu-dcov/*` except for reading or running validation

## Phase 0: Baseline Audit

Purpose: prove the CPU and CUDA skeleton baselines still work before adding cache behavior.

- [ ] Run:

```bash
pwd
find fastkpc -maxdepth 3 -type f | sort
find kpcalg/R -maxdepth 1 -type f | sort
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
CUDA toolkit and at least one GPU are visible.
```

- [ ] Run:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/cuda_validation.R"); print(validate_cuda_skeleton_scenario())'
```

Expected:

```text
Existing CPU and CUDA skeleton tests pass.
validate_cuda_skeleton_scenario() reports adjacency_identical TRUE, sepsets_identical TRUE, n_edgetests_identical TRUE.
```

## Phase 1: Residual Cache Core

Purpose: create a testable cache object before touching skeleton logic.

- [ ] Create `fastkpc/tests/test_residual_cache_core.R`.

The test must check through a new Rcpp test entry point:

```text
1. Same target and same conditioning set in different order map to one cache key.
2. Different target variables map to different keys.
3. Different backend params map to different keys.
4. With cache enabled, a repeated residual request increments hits and does not increment computations.
5. With cache disabled, repeated residual requests increment computations and leave stored_vectors at 0.
6. Cached residual values equal direct residualize_lm() values within 1e-12.
```

Required temporary R API for this test:

```r
fast_residual_cache_selftest(data)
```

It must return:

```text
list(
  key_order_invariant = TRUE,
  target_distinct = TRUE,
  params_distinct = TRUE,
  enabled_stats = list(...),
  disabled_stats = list(...),
  max_abs_residual_diff = numeric(1)
)
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_residual_cache_core.R
```

Expected:

```text
The test fails because fast_residual_cache_selftest() does not exist yet.
```

- [ ] Create `fastkpc/src/residual_backend.hpp` and `fastkpc/src/residual_backend.cpp`.

Required C++ API:

```cpp
struct ResidualBackendDescriptor {
  std::string name;
  std::string params;
};

ResidualBackendDescriptor linear_residual_backend_descriptor();

std::vector<double> compute_linear_residuals(const Rcpp::NumericMatrix& data,
                                             int target,
                                             const std::vector<int>& conditioning_set);
```

Behavior:

```text
linear_residual_backend_descriptor().name == "linear"
linear_residual_backend_descriptor().params == "intercept=true;ridge=1e-8"
compute_linear_residuals() delegates to residualize_lm() without changing numeric behavior.
```

- [ ] Create `fastkpc/src/residual_cache.hpp` and `fastkpc/src/residual_cache.cpp`.

Required C++ API:

```cpp
struct ResidualCacheOptions {
  bool enabled;
  std::string backend_name;
  std::string backend_params;
};

struct ResidualCacheStats {
  bool enabled;
  int requests;
  int hits;
  int misses;
  int computations;
  int stored_vectors;
  int stored_values;
  std::string backend_name;
};

struct ResidualCacheKey {
  int target;
  std::vector<int> conditioning_set;
  int n_rows;
  int n_cols;
  std::string backend_name;
  std::string backend_params;
};

class ResidualCache {
 public:
  explicit ResidualCache(ResidualCacheOptions options);
  const std::vector<double>& get(const Rcpp::NumericMatrix& data,
                                 int target,
                                 const std::vector<int>& conditioning_set);
  ResidualCacheStats stats() const;
};
```

Required behavior:

```text
Sort conditioning_set inside key construction.
If enabled is FALSE, compute into an internal scratch vector and return it by const reference.
If enabled is TRUE, store computed vectors in a map keyed by ResidualCacheKey.
requests increments on every get().
hits increments only when returning a stored vector.
misses increments only for enabled cache misses.
computations increments whenever compute_linear_residuals() is called.
stored_vectors equals the number of vectors in the map.
stored_values equals stored_vectors * n_rows.
```

- [ ] Add `fast_residual_cache_selftest()` to `fastkpc/src/rcpp_exports.cpp` and wrapper in `fastkpc/R/native.R`.

Required behavior:

```text
Use a small fixed sequence of residual requests to prove key and stats behavior.
Return the list described above.
Do not expose this as a package API; it is a local validation helper.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_residual_cache_core.R
```

Expected:

```text
test_residual_cache_core.R prints PASS.
```

## Phase 2: CPU Skeleton Cache Integration

Purpose: route CPU exact skeleton conditional residualization through the cache while preserving graph behavior.

- [ ] Create `fastkpc/tests/test_skeleton_residual_cache.R`.

The test must check:

```text
1. fast_skeleton_cpp_cached(..., residual_cache = FALSE) matches fast_skeleton_cpp() exactly.
2. fast_skeleton_cpp_cached(..., residual_cache = TRUE) matches uncached adjacency exactly.
3. Cached and uncached pMax differ by less than 1e-10.
4. Cached and uncached sepsets match after sorting each conditioning set.
5. Cached and uncached n.edgetests are identical.
6. Cached run reports residual_cache$enabled TRUE.
7. Cached run reports hits > 0 on a fixed scenario with max_conditioning_size = 2.
8. Cached run reports computations < requests.
9. Uncached run reports residual_cache$enabled FALSE and hits == 0.
```

Use this fixed scenario:

```r
set.seed(31)
n <- 90
z1 <- rnorm(n)
z2 <- rnorm(n)
data <- cbind(
  x1 = z1 + rnorm(n, sd = 0.2),
  x2 = z1 - z2 + rnorm(n, sd = 0.2),
  x3 = z2 + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_residual_cache.R
```

Expected:

```text
The test fails because fast_skeleton_cpp_cached() does not exist yet.
```

- [ ] Modify `fastkpc/src/fastkpc_types.hpp`.

Required additions:

```cpp
struct ResidualCacheStats;

struct SkeletonOptions {
  ...
  bool residual_cache_enabled;
};

struct SkeletonResult {
  ...
  ResidualCacheStats residual_cache_stats;
  std::string residual_backend;
};
```

If including `ResidualCacheStats` directly causes circular includes, define a simple serializable stats struct in `fastkpc_types.hpp` and convert from cache stats at the boundary.

- [ ] Modify `fastkpc/src/skeleton_engine.cpp`.

Required behavior:

```text
At the start of run_skeleton_exact(), create one ResidualCache with enabled = options.residual_cache_enabled.
For conditional tests, request residual vectors through ResidualCache::get().
Do not use the cache for unconditional tests.
Do not change edge enumeration or deletion semantics.
Store cache stats and residual_backend in SkeletonResult before returning.
```

- [ ] Modify `fastkpc/src/rcpp_exports.cpp`.

Required behavior:

```text
Add fast_skeleton_cpp_cached_export(data, alpha, max_conditioning_size, index, legacy_index, residual_cache).
Add residual_cache list and residual_backend to returned R list.
Keep fast_skeleton_cpp_export available and behavior-compatible.
```

- [ ] Modify `fastkpc/R/native.R`.

Required wrapper:

```r
fast_skeleton_cpp_cached <- function(data, alpha, max_conditioning_size,
                                     index = 1, legacy_index = TRUE,
                                     residual_cache = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_skeleton_cpp_cached_export(
    data,
    as.numeric(alpha),
    as.integer(max_conditioning_size),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache)
  )
}
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_skeleton_mvp.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 3: CUDA Skeleton Cache Integration

Purpose: route CUDA skeleton residual packing through the same cache.

- [ ] Create `fastkpc/tests/test_cuda_residual_cache.R`.

The test must check:

```text
1. fast_skeleton_cuda_cached(..., residual_cache = FALSE) matches fast_skeleton_cuda() adjacency.
2. fast_skeleton_cuda_cached(..., residual_cache = TRUE) matches fast_skeleton_cpp_cached(..., residual_cache = FALSE) adjacency.
3. Cached CUDA and uncached CPU pMax differ by less than 1e-8.
4. Cached CUDA and uncached CPU sepsets match.
5. Cached CUDA and uncached CPU n.edgetests are identical.
6. Cached CUDA run reports backend "cuda".
7. Cached CUDA run reports residual_cache$enabled TRUE.
8. Cached CUDA run reports hits > 0 and computations < requests on the fixed max_conditioning_size = 2 scenario.
9. batch_size = 1 and batch_size = 0 produce identical cached CUDA graph outputs.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

Expected:

```text
The test fails because fast_skeleton_cuda_cached() does not exist yet.
```

- [ ] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

Required behavior:

```text
At the start of run_skeleton_cuda_batch(), create one ResidualCache with enabled = options.residual_cache_enabled.
For conditional tasks, fill_task_vectors() must request residual vectors through the shared ResidualCache.
Do not cache unconditional raw columns.
Do not change CUDA batch task enumeration or replay semantics.
Store cache stats and residual_backend in SkeletonResult before returning.
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Required behavior:

```text
Add C_fast_skeleton_cuda_cached(data, alpha, max_conditioning_size, index, legacy_index, batch_size, residual_cache).
Return residual_cache list and residual_backend in the R result.
Keep C_fast_skeleton_cuda available and behavior-compatible.
Register the new .Call symbol in R_init_fastkpc_cuda().
```

- [ ] Modify `fastkpc/R/cuda_native.R`.

Required wrapper:

```r
fast_skeleton_cuda_cached <- function(data, alpha, max_conditioning_size,
                                      index = 1, legacy_index = TRUE,
                                      batch_size = 0,
                                      residual_cache = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_cached", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache),
        PACKAGE = "fastkpc_cuda")
}
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Required behavior:

```text
Compile and link residual_cache.cpp and residual_backend.cpp into fastkpc/build/fastkpc_cuda.so.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 4: Residual Cache Validation Report

Purpose: make cache behavior and graph equivalence visible.

- [ ] Create `fastkpc/R/residual_validation.R`.

Required functions:

```r
validate_cpu_residual_cache(seed = 31, n = 90, alpha = 0.2, max_conditioning_size = 2)
validate_cuda_residual_cache(seed = 31, n = 90, alpha = 0.2, max_conditioning_size = 2, batch_size = 0)
```

Each function must return:

```text
list(
  diff = summarize_graph_diff(uncached, cached),
  max_abs_pmax_diff,
  adjacency_identical,
  sepsets_identical,
  n_edgetests_identical,
  cache_stats
)
```

`cache_stats` must include:

```text
enabled
requests
hits
misses
computations
stored_vectors
stored_values
backend_name
```

- [ ] Add residual cache validation command to `fastkpc/README.md`.

Required command:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/residual_validation.R"); print(validate_cpu_residual_cache()); print(validate_cuda_residual_cache())'
```

Expected documented values:

```text
adjacency_identical TRUE
sepsets_identical TRUE
n_edgetests_identical TRUE
cache_stats$hits > 0
cache_stats$computations < cache_stats$requests
```

- [ ] Run the validation command.

Expected:

```text
Both CPU and CUDA validation reports meet the documented values.
```

## Phase 5: Documentation And Build Hygiene

Purpose: leave the residual cache path clear for the next agent.

- [ ] Update `fastkpc/README.md`.

Required sections:

```text
Residual Cache Scope
Residual Cache API
Residual Cache Tests
Residual Cache Validation
Residual Cache Known Limits
```

Known limits must explicitly state:

```text
The cache is per skeleton run and is not global across R calls.
The only production residual backend in this goal is the CPU linear MVP backend.
The cache does not implement mgcv equivalence.
The cache does not implement GPU residualization.
The cache does not change exported kpcalg::kpc().
```

- [ ] Confirm build scripts include new sources.

Run:

```bash
rg -n "residual_cache|residual_backend" fastkpc/tools/build_cuda_native.sh fastkpc/src/rcpp_exports.cpp fastkpc/src/r_api_cuda.cpp fastkpc/R/native.R fastkpc/R/cuda_native.R fastkpc/README.md
```

Expected:

```text
All listed files reference the residual cache where required.
```

## Phase 6: Completion Criteria

The goal is complete only when all of these are true:

```text
1. Previous CPU and CUDA tests still pass:
   - fastkpc/tests/test_dcov_exact.R
   - fastkpc/tests/test_skeleton_mvp.R
   - fastkpc/tests/test_diff_report.R
   - fastkpc/tests/test_cuda_build_contract.R
   - fastkpc/tests/test_dcov_cuda_batch.R
   - fastkpc/tests/test_skeleton_cuda_batch.R

2. New residual cache tests pass:
   - fastkpc/tests/test_residual_cache_core.R
   - fastkpc/tests/test_skeleton_residual_cache.R
   - fastkpc/tests/test_cuda_residual_cache.R

3. CPU residual cache validation reports:
   - adjacency_identical TRUE
   - sepsets_identical TRUE
   - n_edgetests_identical TRUE
   - max_abs_pmax_diff < 1e-10
   - cache_stats$hits > 0
   - cache_stats$computations < cache_stats$requests

4. CUDA residual cache validation reports:
   - adjacency_identical TRUE
   - sepsets_identical TRUE
   - n_edgetests_identical TRUE
   - max_abs_pmax_diff < 1e-8
   - cache_stats$hits > 0
   - cache_stats$computations < cache_stats$requests

5. fastkpc/README.md documents residual cache API, tests, validation, and limits.

6. No kpcalg/R/*.R file has been modified.
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
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/residual_validation.R"); print(validate_cpu_residual_cache()); print(validate_cuda_residual_cache())'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
The exact build commands used.
The exact test commands run.
The pass/fail result of each test.
CPU residual cache stats.
CUDA residual cache stats.
CPU/CUDA max pMax differences.
Whether any graph-level differences were observed.
The kpcalg/R MD5 result.
```

## Later Goals

Create separate goals after this residual cache goal is complete.

### Later Goal C: fastSpline Residual Backend

Objective:

```text
Add a B-spline penalized least-squares residual backend as an explicit statistical replacement for mgcv, using the residual cache interface introduced here and validating graph-level differences against legacy mgcv.
```

### Later Goal D: WAN-PDAG Migration

Objective:

```text
Migrate kpcalg's udag2wanpdag generalized transitive orientation step to the C++ scheduler, including batched regrVonPS-style residual independence checks.
```

## Execution Rules For Codex

- Work in small commits if the workspace is a git repository. If it is not a git repository, do not initialize one unless the user asks.
- Prefer adding files under `fastkpc/` over modifying legacy files.
- Do not alter `kpcalg/R/*.R`.
- Do not delete or rewrite `gpu-dcov/`; use it as a numeric reference.
- Follow TDD: write each new residual cache test before implementation.
- Do not make graph changes acceptable merely because cache stats improve.
- If cached and uncached graphs differ, debug residual vectors first, then p-values, then skeleton replay.
- Do not implement fastSpline, CUDA GAM, GPU residualization, residual persistence across R calls, WAN-PDAG migration, HSIC, or permutation tests in this goal.
- Do not replace exported `kpcalg::kpc()` in this goal.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-13-fast-kpc-residual-cache-goal-execution.md: add a shared residual cache for CPU and CUDA skeleton paths, preserving graph behavior and keeping legacy kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
