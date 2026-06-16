# Fast kPC CUDA WAN-PDAG Orientation Residuals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move WAN-PDAG generalized orientation residual and dCov work onto the CUDA residual/dCov stack with explicit orientation-device controls, while preserving CPU orientation semantics, graph equality, diagnostics, campaign/report artifacts, and unchanged `kpcalg/R` files.

**Architecture:** Keep the completed CUDA skeleton path, layer scheduler, true-batched fastSpline residual solver, and exact dCov CUDA backend as reusable building blocks. Add a narrow orientation execution layer around `regrVonPS`: CPU orientation remains the semantic reference, while CUDA orientation can materialize target residuals through the CUDA fastSpline batch API and evaluate the residual-vs-S dCov tests in CUDA batches. Public wrappers gain `orientation_residual_device` and `orientation_batch_size` controls, campaign/report tooling surfaces orientation-device diffs and diagnostics, and legacy replay/order-sensitive WAN-PDAG graph updates remain sequential and deterministic.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5, existing `fastkpc/src/wanpdag_engine.cpp`, existing `fastkpc/src/regrvonps_native.cpp`, existing `fastkpc/src/cuda/fastspline_residual_cuda.*`, existing `fastkpc/src/cuda/dcov_batch_cuda.*`, existing validation campaign/report tooling, local shell build scripts under `fastkpc/tools/`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-16-fast-kpc-cuda-wanpdag-orientation-residuals-goal-execution.md: add opt-in CUDA WAN-PDAG orientation residual/dCov execution for generalized regrVonPS calls, expose orientation_residual_device and orientation_batch_size controls, preserve CPU orientation graph semantics and deterministic replay, extend public wrapper/campaign/report/CLI diagnostics, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `820000`.

This goal is intentionally large. It should run long enough to cover C++ API boundary design, CUDA orientation residual materialization, batched dCov inside `regrVonPS`, R public API plumbing, validation campaign/report/CLI expansion, benchmark artifacts, documentation, and full regression verification.

Do not mark the goal complete until every criterion in "Completion Criteria" is satisfied. Mark the goal blocked only if the same local blocker prevents progress for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Current Baseline

The previous completed goals provide:

```text
fast_kpc() public wrapper
CPU and CUDA skeleton backends
CUDA dCov batch backend
CPU fastSpline residual backend
CUDA fastSpline single residual backend
true-batched CUDA fastSpline residual groups
layer-batched CUDA skeleton scheduler
residual_device controls for CUDA skeleton residual cache
residual_batch_size controls for skeleton residual prefetch
WAN-PDAG CPU/CUDA pipeline wrappers
validation campaign/report/CLI tooling
```

Current implementation facts that matter for this goal:

```text
fast_kpc_wanpdag_cuda() runs the CUDA skeleton, then calls orient_wanpdag_native().
orient_wanpdag_native() is CPU-side and uses ResidualCache plus regrvonps_native().
regrvonps_native() computes one target residual V | (S union parents(V)).
For each node in S, regrvonps_native() evaluates dcov_exact_pvalue(residuals, data[, node]).
CUDA WAN-PDAG wrapper currently annotates orientation$residual_device = "cpu".
WAN-PDAG orientation graph updates are sequential because each accepted generalized orientation mutates the PDAG and neighborhoods.
The completed true-batched fastSpline CUDA API can materialize many residual fits, but orientation currently does not use it.
The completed CUDA dCov batch API can evaluate many vector pairs, but orientation currently uses CPU dCov.
```

Important baseline contract:

```text
CPU WAN-PDAG orientation is the semantic reference.
CUDA skeleton pMax differences must not change orientation semantics.
Existing fast_kpc_wanpdag_cuda() output must remain valid for calls that omit new orientation arguments.
kpcalg/R/*.R must remain unchanged.
```

## Why This Goal Is Next

The fast path now accelerates skeleton CI work substantially:

```text
1. Skeleton dCov tests are CUDA-batched.
2. Skeleton residual prefetch is layer-batched.
3. fastSpline residual groups are true-batched on CUDA.
```

The remaining CPU island inside the CUDA WAN-PDAG pipeline is generalized orientation. `orient_wanpdag_native()` can call `regrvonps_native()` many times; each call can trigger a fastSpline residual fit and multiple exact dCov p-values. That work is still CPU-only, so a full CUDA skeleton-to-WAN-PDAG run still returns to CPU for orientation residualization and dCov testing.

This goal should not parallelize graph mutation. It should accelerate the numerical work inside each deterministic `regrVonPS` check while preserving the same search order and PDAG updates.

## Scope

In scope:

- Add explicit orientation residual execution controls:

```r
orientation_residual_device = c("auto", "cpu", "cuda")
orientation_batch_size = 0
orientation_diagnostics = TRUE
```

- Keep existing `residual_device` as the skeleton residual-device control.
- Make `orientation_residual_device = "auto"` resolve to:

```text
engine="cuda" and residual_backend="fastSpline": "cuda"
engine="cuda" and residual_backend="linear": "cpu"
engine="cpu": "cpu"
graph_stage="skeleton": "cpu" or "none" in result diagnostics
```

- Add CUDA-capable orientation execution for generalized `regrVonPS` calls:

```text
target residual V | (S union parents(V)) can use CUDA fastSpline residuals
dCov tests residual vs every node in S can use CUDA dCov batch
```

- Keep collider orientation and orientation rule applications on CPU because they are graph-rule logic, not numerical residual/dCov work.
- Keep graph search, neighborhood updates, and accepted orientation replay sequential and deterministic.
- Add C++ orientation diagnostics:

```text
orientation_residual_device
orientation_residual_device_requested
orientation_residual_device_reason
orientation_batch_size_requested
orientation_batch_size_used
regrvonps_calls
regrvonps_cuda_calls
regrvonps_cpu_calls
orientation_dcov_batches
orientation_dcov_pairs
orientation_residual_fits
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
orientation_cache_requests
orientation_cache_hits
orientation_cache_computations
```

- Add R validation helpers comparing:

```text
CPU orientation vs CUDA orientation on the same CUDA skeleton
orientation_residual_device="cpu" vs "cuda"
orientation_batch_size=1 vs automatic batch sizing
residual_device="cpu" vs "cuda" combined with orientation_residual_device="cuda"
```

- Extend public `fast_kpc()` config/result contract.
- Extend validation campaign, report writer, and CLIs with orientation-device dimensions and diffs.
- Add benchmark helpers that quantify orientation residual/dCov time and batch counters without requiring a strict speedup.
- Keep old tests passing.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not replace exported legacy `kpcalg::kpc()`.
- Do not modify any file under `kpcalg/R`.
- Do not parallelize WAN-PDAG graph mutation or change orientation search order.
- Do not change collider orientation semantics.
- Do not change `check_immor`, orientation rules, `solve_confl`, or `unf_vect` behavior.
- Do not change exact dCov gamma statistics or `legacy_index` semantics.
- Do not implement multi-GPU scheduling.
- Do not implement HSIC or permutation tests.
- Do not make full mgcv-equivalent CUDA GAM a requirement.
- Do not remove CPU orientation path or CPU residual cache.
- Do not initialize git if the workspace is not already a git repository.

## Design Contract

### Public API Contract

Recommended `fast_kpc()` signature after this goal:

```r
fast_kpc <- function(data,
                     alpha = 0.2,
                     max_conditioning_size = 2,
                     engine = c("auto", "cuda", "cpu"),
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
                     orient_collider = TRUE,
                     solve_confl = FALSE,
                     rules = c(TRUE, TRUE, TRUE),
                     fastspline_params = list(),
                     cuda_residual_fallback = TRUE,
                     validate = FALSE,
                     benchmark = FALSE,
                     legacy = FALSE,
                     labels = NULL,
                     seed = NULL)
```

Recommended CUDA WAN-PDAG wrapper signature:

```r
fast_kpc_wanpdag_cuda <- function(data, alpha, max_conditioning_size,
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
                                  orient_collider = TRUE,
                                  solve_confl = FALSE,
                                  rules = c(TRUE, TRUE, TRUE),
                                  fastspline_params = list(),
                                  cuda_residual_fallback = TRUE)
```

Compatibility rules:

```text
1. Existing calls that omit orientation_residual_device remain valid.
2. Existing calls that omit orientation_batch_size remain valid.
3. residual_device remains the skeleton residual-device control.
4. orientation_residual_device controls only WAN-PDAG orientation residual/dCov work.
5. graph_stage="skeleton" accepts orientation arguments but ignores them with config diagnostics.
6. engine="cpu" with orientation_residual_device="cuda" errors clearly unless the wrapper resolves it before native execution.
7. residual_backend="linear" with orientation_residual_device="cuda" resolves to "cpu" and records reason "linear orientation residual CUDA device is not implemented".
```

### Orientation Device Resolution Contract

Add a focused resolver:

```text
requested: auto/cpu/cuda
engine_used: cpu/cuda
graph_stage: skeleton/wanpdag
residual_backend: linear/fastSpline
cuda_available: TRUE/FALSE
fallback: TRUE/FALSE
```

Resolved values:

```text
cpu
cuda
cuda-fallback-cpu
none
```

Rules:

```text
1. graph_stage="skeleton" resolves orientation device to "none".
2. engine_used="cpu" resolves orientation device to "cpu".
3. residual_backend="linear" resolves orientation device to "cpu".
4. orientation_residual_device="cpu" resolves to "cpu".
5. orientation_residual_device="cuda" resolves to "cuda" only when engine_used="cuda", residual_backend="fastSpline", and CUDA is available.
6. orientation_residual_device="auto" resolves to "cuda" under the same supported CUDA conditions, otherwise "cpu".
7. If CUDA orientation residual work fails and cuda_residual_fallback=TRUE, affected calls fall back to CPU and the orientation result records "cuda-fallback-cpu".
8. If CUDA orientation residual work fails and cuda_residual_fallback=FALSE, the run errors with a message prefixed by "CUDA WAN-PDAG orientation failed".
```

### C++ Option And Result Contract

Extend `OrientationOptions` with:

```cpp
std::string orientation_residual_device_requested;
bool cuda_residual_fallback;
int orientation_batch_size;
bool orientation_diagnostics_enabled;
```

Add:

```cpp
struct OrientationDiagnostics {
  std::string orientation_residual_device;
  std::string orientation_residual_device_requested;
  std::string orientation_residual_device_reason;
  int orientation_batch_size_requested;
  int orientation_batch_size_used;
  int regrvonps_calls;
  int regrvonps_cuda_calls;
  int regrvonps_cpu_calls;
  int orientation_dcov_batches;
  int orientation_dcov_pairs;
  int orientation_residual_fits;
  int orientation_cuda_residual_fits;
  int orientation_cpu_fallback_fits;
  int orientation_cache_requests;
  int orientation_cache_hits;
  int orientation_cache_computations;
};
```

Extend `OrientationResult` with:

```cpp
std::string residual_device;
std::string residual_device_requested;
std::string residual_device_reason;
OrientationDiagnostics diagnostics;
```

Backward compatibility:

```text
Existing R result fields orientation$residual_backend, orientation$residual_backend_params,
orientation$residual_cache, orientation$counts, and orientation$events must remain.
New fields are additive.
```

### RegrVonPS CUDA Contract

Existing CPU behavior:

```text
1. parents = parents_of(pdag, p, V)
2. conditioning_set = sorted_unique_union(S, parents)
3. residuals = residual_cache->get(data, V, conditioning_set)
4. for each node in S:
     p_value = dcov_exact_pvalue(residuals, data[, node])
     reject_count += p_value < alpha
```

CUDA behavior must preserve the same result:

```text
1. Same parents and conditioning_set.
2. Same target residual semantics.
3. Same p_values order as S order.
4. Same reject_count threshold.
5. Same cache stats semantics.
6. Same first_or_nan() event p.value behavior.
```

Add a new function:

```cpp
RegrVonPsResult regrvonps_device(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  OrientationResidualCache* residual_cache,
  OrientationDiagnostics* diagnostics);
```

The function chooses CPU or CUDA based on resolved orientation device and can call the existing CPU implementation for CPU mode.

### Orientation Residual Cache Contract

The existing `ResidualCache` is CPU-only. This goal should not mutate it into a CUDA-specific class in place. Add a focused adapter:

```cpp
class OrientationResidualCache {
 public:
  OrientationResidualCache(const OrientationOptions& options);
  const std::vector<double>& get(const Rcpp::NumericMatrix& data,
                                 int target,
                                 const std::vector<int>& conditioning_set,
                                 FastSplineCudaDiagnostics* cuda_diagnostics);
  ResidualCacheStats stats() const;
  std::string residual_device() const;
  std::string residual_device_requested() const;
  std::string residual_device_reason() const;
};
```

Rules:

```text
1. CPU mode delegates to existing compute_residuals_with_backend() semantics.
2. CUDA fastSpline mode may call fit_fastspline_residuals_cuda() for single orientation calls.
3. If later orientation prefetch groups are added inside the same goal, they may call fit_fastspline_residuals_cuda_batch_result().
4. Cache key must include target, sorted conditioning set, n, p, residual backend name, backend params, and resolved orientation residual device.
5. Cache counters retain current meaning: requests, hits, misses/computations, stored vectors.
6. A CUDA fallback inserts CPU-computed residuals and records residual_device = "cuda-fallback-cpu".
```

### Orientation dCov Batch Contract

Inside one `regrVonPS` call, if `S` has length `m > 0`, CUDA mode should evaluate:

```text
x matrix: n x m, every column is target residuals
y matrix: n x m, column k is data[, S[k]]
```

Rules:

```text
1. orientation_batch_size <= 0 means evaluate all pairs for the regrVonPS call in one CUDA dCov batch.
2. orientation_batch_size = 1 must produce identical p_values to automatic sizing within tolerance.
3. The p_values vector order must match S order.
4. Non-finite p-values follow existing dCov behavior; do not add new NA deletion semantics.
5. Empty S does not call dCov and returns reject_count = 0.
```

### WAN-PDAG Replay And Determinism Contract

Do not batch across graph-mutating orientation decisions in this goal.

Required sequencing:

```text
1. Collider orientation runs first exactly as today.
2. Generalized orientation loop visits V in the existing order.
3. Subsets S are enumerated in the existing order.
4. Reverse checks over W and S2 keep the existing order.
5. Each regrVonPS numeric check may use CUDA internally.
6. PDAG mutation occurs immediately after the same accepted condition as CPU reference.
7. Rules are applied after accepted generalized orientation exactly as today.
```

Equality requirements:

```text
orientation_residual_device="cuda" pdag == orientation_residual_device="cpu" pdag
orientation counts identical
event count identical
event rule/phase/accepted sequence identical
max abs event p.value diff < 1e-7 for finite p-values
```

### Diagnostics Contract

`orientation` result must include:

```text
residual_device
residual_device_requested
residual_device_reason
diagnostics
```

`orientation$diagnostics` fields:

```text
orientation_residual_device
orientation_residual_device_requested
orientation_residual_device_reason
orientation_batch_size_requested
orientation_batch_size_used
regrvonps_calls
regrvonps_cuda_calls
regrvonps_cpu_calls
orientation_dcov_batches
orientation_dcov_pairs
orientation_residual_fits
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
orientation_cache_requests
orientation_cache_hits
orientation_cache_computations
```

Required identities:

```text
regrvonps_calls == regrvonps_cuda_calls + regrvonps_cpu_calls
orientation_cache_requests == orientation$residual_cache$requests
orientation_cache_hits == orientation$residual_cache$hits
orientation_cache_computations == orientation$residual_cache$computations
orientation_dcov_pairs >= number of finite p_values emitted by generalized regrVonPS CUDA calls
```

### Validation Tolerances

Standalone `regrVonPS` CPU-vs-CUDA:

```text
parents identical
conditioning_set identical
reject_count identical
length p_values identical
max(abs(cpu$p_values - cuda$p_values)) < 1e-7
```

WAN-PDAG CPU-vs-CUDA orientation on same skeleton:

```text
pdag identical
counts identical
event accepted sequence identical
max finite event p.value diff < 1e-7
```

Full `fast_kpc()` public wrapper:

```text
engine="cuda", residual_device="cuda", orientation_residual_device="cuda"
matches orientation_residual_device="cpu" on deterministic scenarios.
```

## File Structure

Create these files:

- `fastkpc/src/orientation_residual_cache.hpp`  
  Orientation-specific residual cache adapter with CPU/CUDA residual-device resolution and cache stats.

- `fastkpc/src/orientation_residual_cache.cpp`  
  CPU residual computation, CUDA fastSpline residual materialization, fallback handling, cache keying, and diagnostics updates.

- `fastkpc/src/regrvonps_device.hpp`  
  Device-aware `regrVonPS` declarations, p-value comparison helpers, and diagnostics structs if they are not placed in `orientation_types.hpp`.

- `fastkpc/src/regrvonps_device.cpp`  
  CPU/CUDA dispatch for `regrVonPS`, CPU compatibility wrapper, and CUDA dCov batch packing for residual-vs-S tests.

- `fastkpc/tests/test_regrvonps_cuda_orientation_device.R`  
  Standalone `regrVonPS` CPU-vs-CUDA p-value and diagnostics tests.

- `fastkpc/tests/test_wanpdag_cuda_orientation_device.R`  
  WAN-PDAG CPU-orientation vs CUDA-orientation graph and event equivalence tests.

- `fastkpc/tests/test_fastkpc_orientation_device_public_api.R`  
  Public `fast_kpc()` and `fast_kpc_wanpdag_cuda()` orientation-device config/result contract tests.

- `fastkpc/tests/test_orientation_device_campaign_report_cli.R`  
  Campaign/report/CLI tests for orientation-device dimensions and artifacts.

- `fastkpc/tests/test_orientation_device_benchmark.R`  
  Non-strict benchmark smoke for orientation residual/dCov counters and timings.

- `fastkpc/tests/test_orientation_device_docs_contract.R`  
  README/report documentation contract tests for orientation-device options and artifacts.

Modify these files:

- `fastkpc/src/orientation_types.hpp`  
  Add orientation residual device options and diagnostics result fields.

- `fastkpc/src/regrvonps_native.hpp`  
  Add declarations needed by device-aware wrapper or expose reusable CPU helper safely.

- `fastkpc/src/regrvonps_native.cpp`  
  Keep CPU behavior stable; optionally refactor shared p-value result assembly without changing output.

- `fastkpc/src/wanpdag_engine.hpp`  
  Add CUDA/device-aware orientation entry points if separate from `orient_wanpdag_native()`.

- `fastkpc/src/wanpdag_engine.cpp`  
  Use `OrientationResidualCache` and `regrvonps_device()` in generalized orientation checks.

- `fastkpc/src/r_api_cuda.cpp`  
  Extend `.Call` CUDA WAN-PDAG arguments and orientation result conversion.

- `fastkpc/src/rcpp_exports.cpp`  
  If CPU sourceCpp exports need orientation diagnostics helpers, update declarations.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile/link new C++ source files.

- `fastkpc/R/cuda_native.R`  
  Add orientation residual arguments to CUDA wrappers.

- `fastkpc/R/native.R`  
  Add CPU wrapper acceptance and result normalization for orientation diagnostics if needed.

- `fastkpc/R/fast_kpc.R`  
  Add public config fields, argument validation, result metrics, and print/summary exposure.

- `fastkpc/R/wanpdag_validation.R`  
  Add CPU-vs-CUDA orientation-device validation and benchmark helpers.

- `fastkpc/R/validation_campaign.R`  
  Add orientation device dimensions, diffs, and diagnostic tables.

- `fastkpc/R/report_writer.R`  
  Write orientation-device CSV artifacts and Markdown sections.

- `fastkpc/R/scheduler_validation.R`  
  Include orientation-device counters in full CUDA scheduler benchmarks where relevant.

- `fastkpc/tools/run_fast_kpc.R`  
  Add `--orientation-residual-device`, `--orientation-batch-size`, and `--orientation-diagnostics`.

- `fastkpc/tools/run_validation_campaign.R`  
  Add `--orientation-residual-devices`, `--orientation-batch-size`, and report artifact handling.

- `fastkpc/README.md`  
  Document orientation residual device controls, diagnostics, validation, benchmarks, and limits.

- `fastkpc/reports/README.md`  
  Document orientation-device report artifacts.

Do not modify:

- `kpcalg/R/*.R`

## Phase 0: Baseline Audit And Guardrails

Purpose: confirm the true-batched skeleton/residual baseline before changing WAN-PDAG orientation.

- [ ] Run:

```bash
pwd
test -e docs/superpowers/plans/2026-06-15-fast-kpc-true-batched-fastspline-cuda-goal-execution.md && echo previous-plan-ok
test -e fastkpc/src/wanpdag_engine.cpp && echo wanpdag-source-ok
test -e fastkpc/src/regrvonps_native.cpp && echo regrvonps-source-ok
test -e fastkpc/src/cuda/fastspline_batched_solver.cu && echo true-batch-source-ok
test -e fastkpc/tests/test_wanpdag_cuda_pipeline.R && echo wanpdag-cuda-test-ok
```

Expected:

```text
/data/wenyujianData/kpcalg
previous-plan-ok
wanpdag-source-ok
regrvonps-source-ok
true-batch-source-ok
wanpdag-cuda-test-ok
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R
```

Expected:

```text
CUDA native build succeeds.
CPU native sourceCpp build succeeds.
Every listed test prints PASS.
```

- [ ] Run:

```bash
cd kpcalg
md5sum -c MD5 | rg '^R/'
cd ..
```

Expected:

```text
Every kpcalg/R MD5 line reports OK.
```

If a baseline test fails, use systematic debugging. Do not implement orientation-device changes until the failure is understood and either fixed in fastkpc-owned files or documented as an unrelated environmental blocker.

## Phase 1: TDD Red Tests For Orientation Device Contracts

Purpose: define behavior before touching production C++/R code.

- [ ] Create `fastkpc/tests/test_regrvonps_cuda_orientation_device.R`.

Required test behavior:

```text
1. Build CPU and CUDA native code.
2. Construct a deterministic 5-column nonlinear data matrix.
3. Construct a small PDAG where V has at least one parent and S has at least two nodes.
4. Call a new R helper or exported wrapper for CPU regrVonPS.
5. Call the CUDA orientation-device path for the same V and S.
6. Assert parents identical.
7. Assert conditioning_set identical.
8. Assert reject_count identical.
9. Assert p_values length identical.
10. Assert max_abs_pvalue_diff < 1e-7.
11. Assert diagnostics regrvonps_cuda_calls == 1.
12. Assert diagnostics orientation_dcov_batches > 0.
```

Suggested test scaffold:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(601)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.08),
  x2 = cos(z1) + rnorm(n, sd = 0.08),
  x3 = z1 * z2 + rnorm(n, sd = 0.08),
  x4 = sin(z2) + rnorm(n, sd = 0.08),
  x5 = rnorm(n)
)
pdag <- matrix(0L, ncol(data), ncol(data))
pdag[1, 3] <- 1L
pdag[3, 4] <- 1L
pdag[4, 3] <- 1L
V <- 3L
S <- c(2L, 4L)
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

cpu <- fast_regrvonps_orientation_device(
  data, pdag, V, S,
  residual_backend = "fastSpline",
  orientation_residual_device = "cpu",
  fastspline_params = params
)
cuda <- fast_regrvonps_orientation_device(
  data, pdag, V, S,
  residual_backend = "fastSpline",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0,
  fastspline_params = params,
  cuda_residual_fallback = FALSE
)

assert_true(identical(cpu$parents, cuda$parents), "parents should match")
assert_true(identical(cpu$conditioning_set, cuda$conditioning_set),
            "conditioning sets should match")
assert_true(identical(cpu$reject_count, cuda$reject_count),
            "reject counts should match")
assert_true(length(cpu$p_values) == length(cuda$p_values),
            "p-value lengths should match")
assert_true(max_abs_diff(cpu$p_values, cuda$p_values) < 1e-7,
            "p-values should match")
assert_true(cuda$diagnostics$regrvonps_cuda_calls == 1L,
            "CUDA call should be recorded")
assert_true(cuda$diagnostics$orientation_dcov_batches > 0L,
            "CUDA dCov batches should be recorded")

cat("test_regrvonps_cuda_orientation_device.R: PASS\n")
```

Expected before implementation:

```text
FAIL because fast_regrvonps_orientation_device() does not exist.
```

- [ ] Create `fastkpc/tests/test_wanpdag_cuda_orientation_device.R`.

Required test behavior:

```text
1. Run fast_kpc_wanpdag_cuda() twice on the same data:
   a. orientation_residual_device="cpu"
   b. orientation_residual_device="cuda"
2. Use residual_device="cuda" for skeleton in both runs.
3. Use scheduler="layer" and residual_batch_size=0.
4. Assert skeleton adjacency and pMax match between runs.
5. Assert orientation pdag identical.
6. Assert orientation counts identical.
7. Assert event accepted/rule/phase sequence identical.
8. Assert max finite event p.value diff < 1e-7.
9. Assert cuda orientation diagnostics record regrvonps_cuda_calls > 0 if generalized checks occur.
10. Assert orientation residual_device is "cuda" or "cuda-fallback-cpu" with an explicit reason.
```

Expected before implementation:

```text
FAIL because fast_kpc_wanpdag_cuda() does not accept orientation_residual_device.
```

- [ ] Create `fastkpc/tests/test_fastkpc_orientation_device_public_api.R`.

Required test behavior:

```text
fast_kpc(..., engine="cuda", graph_stage="wanpdag",
         residual_device="cuda", orientation_residual_device="cuda")
returns config fields:
  orientation_residual_device_requested
  orientation_residual_device_used
  orientation_batch_size
  orientation_diagnostics
and orientation result fields:
  residual_device
  residual_device_requested
  diagnostics
```

Expected before implementation:

```text
FAIL because fast_kpc() does not accept orientation_residual_device.
```

## Phase 2: Add Orientation Options, Diagnostics, And R Result Shape

Purpose: add the data contract without changing CPU behavior.

- [ ] Modify `fastkpc/src/orientation_types.hpp`.

Add:

```cpp
struct OrientationDiagnostics {
  std::string orientation_residual_device;
  std::string orientation_residual_device_requested;
  std::string orientation_residual_device_reason;
  int orientation_batch_size_requested;
  int orientation_batch_size_used;
  int regrvonps_calls;
  int regrvonps_cuda_calls;
  int regrvonps_cpu_calls;
  int orientation_dcov_batches;
  int orientation_dcov_pairs;
  int orientation_residual_fits;
  int orientation_cuda_residual_fits;
  int orientation_cpu_fallback_fits;
  int orientation_cache_requests;
  int orientation_cache_hits;
  int orientation_cache_computations;
};
```

Extend `OrientationOptions`:

```cpp
std::string orientation_residual_device_requested;
bool cuda_residual_fallback;
int orientation_batch_size;
bool orientation_diagnostics_enabled;
```

Extend `OrientationResult`:

```cpp
std::string residual_device;
std::string residual_device_requested;
std::string residual_device_reason;
OrientationDiagnostics diagnostics;
```

- [ ] Modify `fastkpc/src/wanpdag_engine.cpp`.

In `default_orientation_options()` initialize:

```cpp
options.orientation_residual_device_requested = "cpu";
options.cuda_residual_fallback = true;
options.orientation_batch_size = 0;
options.orientation_diagnostics_enabled = true;
```

Initialize result defaults in `orient_wanpdag_native()`:

```cpp
result.residual_device = "cpu";
result.residual_device_requested =
  options.orientation_residual_device_requested.empty() ?
  "cpu" : options.orientation_residual_device_requested;
result.residual_device_reason = "";
result.diagnostics = make_orientation_diagnostics(...);
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Extend `orientation_result_to_list()` to include:

```text
residual_device
residual_device_requested
residual_device_reason
diagnostics
```

The diagnostics list must include all fields from `OrientationDiagnostics`.

- [ ] Build:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
Both native builds succeed.
```

- [ ] Run existing tests to verify additive result shape did not break CPU behavior:

```bash
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
```

Expected:

```text
All listed tests print PASS.
The new red tests still fail because public orientation device args are not wired yet.
```

## Phase 3: Add Orientation Residual Cache Adapter

Purpose: make residual materialization selectable without disturbing existing skeleton cache.

- [ ] Create `fastkpc/src/orientation_residual_cache.hpp`.

Required declarations:

```cpp
#ifndef FASTKPC_ORIENTATION_RESIDUAL_CACHE_HPP
#define FASTKPC_ORIENTATION_RESIDUAL_CACHE_HPP

#include "orientation_types.hpp"
#include "residual_backend_registry.hpp"
#include "residual_cache.hpp"
#include "cuda/fastspline_residual_cuda.hpp"

#include <Rcpp.h>
#include <map>
#include <string>
#include <vector>

class OrientationResidualCache {
 public:
  OrientationResidualCache(const OrientationOptions& options,
                           int n,
                           int p);

  const std::vector<double>& get(const Rcpp::NumericMatrix& data,
                                 int target,
                                 const std::vector<int>& conditioning_set,
                                 FastSplineCudaDiagnostics* cuda_diagnostics);

  ResidualCacheStats stats() const;
  const ResidualBackendConfig& backend() const;
  const std::string& residual_device() const;
  const std::string& residual_device_requested() const;
  const std::string& residual_device_reason() const;

 private:
  std::vector<double> compute(const Rcpp::NumericMatrix& data,
                              int target,
                              const std::vector<int>& conditioning_set,
                              FastSplineCudaDiagnostics* cuda_diagnostics);
};

#endif
```

- [ ] Create `fastkpc/src/orientation_residual_cache.cpp`.

Required behavior:

```text
1. Resolve requested device from OrientationOptions.
2. linear backend always uses CPU and records reason when CUDA requested.
3. fastSpline + cuda uses fit_fastspline_residuals_cuda().
4. CUDA fallback sets residual_device to "cuda-fallback-cpu".
5. Cache key uses make_residual_cache_key() semantics plus resolved device.
6. stats() returns requests, hits, misses, computations, stored vectors, stored values.
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Add compile line:

```sh
"$CXX" $COMMON_CXX -c "$ROOT/src/orientation_residual_cache.cpp" -o "$BUILD/orientation_residual_cache.o"
```

Add link input:

```sh
"$BUILD/orientation_residual_cache.o" \
```

- [ ] If CPU sourceCpp build needs the new file, modify `fastkpc/R/native.R` source list or `fastkpc/src/rcpp_exports.cpp` build path according to existing patterns.

- [ ] Build:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
Both native builds succeed.
```

## Phase 4: Add Device-Aware RegrVonPS

Purpose: accelerate the numerical work inside each deterministic orientation check.

- [ ] Create `fastkpc/src/regrvonps_device.hpp`.

Required declarations:

```cpp
#ifndef FASTKPC_REGRVONPS_DEVICE_HPP
#define FASTKPC_REGRVONPS_DEVICE_HPP

#include "orientation_residual_cache.hpp"
#include "regrvonps_native.hpp"

RegrVonPsResult regrvonps_device(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& pdag,
  int p,
  int V,
  const std::vector<int>& S,
  const OrientationOptions& options,
  OrientationResidualCache* residual_cache,
  OrientationDiagnostics* diagnostics);

#endif
```

- [ ] Create `fastkpc/src/regrvonps_device.cpp`.

Implement CPU branch:

```text
Use OrientationResidualCache::get() for the target residual.
Use dcov_exact_pvalue() for each node in S.
Increment diagnostics regrvonps_calls and regrvonps_cpu_calls.
Preserve RegrVonPsResult fields exactly.
```

Implement CUDA branch:

```text
1. Use OrientationResidualCache::get() to materialize target residuals.
2. If S is empty, return reject_count = 0 without dCov.
3. Pack xmat as repeated residual columns.
4. Pack ymat as data columns S.
5. Split into chunks by orientation_batch_size.
6. Call dcov_batch_cuda() per chunk.
7. Append p_values in S order.
8. Increment orientation_dcov_batches and orientation_dcov_pairs.
9. Increment regrvonps_cuda_calls.
```

Required fallback:

```text
If CUDA dCov fails and cuda_residual_fallback=TRUE, recompute p-values with CPU dCov and increment regrvonps_cpu_calls for the affected call.
If CUDA dCov fails and cuda_residual_fallback=FALSE, throw with "CUDA WAN-PDAG orientation failed".
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Compile/link `regrvonps_device.cpp`.

- [ ] Build:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Expected:

```text
CUDA native build succeeds.
```

## Phase 5: Wire Device-Aware RegrVonPS Into WAN-PDAG Engine

Purpose: preserve graph semantics while changing the numeric backend used by orientation checks.

- [ ] Modify `fastkpc/src/wanpdag_engine.cpp`.

Replace local CPU `ResidualCache` with:

```cpp
OrientationResidualCache residual_cache(options, data.nrow(), p);
```

Replace calls:

```cpp
regrvonps_native(data, result.pdag, p, V, S, options, &residual_cache)
```

with:

```cpp
regrvonps_device(data, result.pdag, p, V, S, options,
                 &residual_cache, &result.diagnostics)
```

Apply the same replacement for reverse `W, S2` checks.

- [ ] Preserve existing increments:

```text
result.regrvonps_calls must still equal the number of regrVonPS checks.
result.events must be pushed in the same places.
result.generalized_orientations must update in the same places.
```

- [ ] After the loop, copy cache stats:

```cpp
result.residual_cache_requests = stats.requests;
result.residual_cache_hits = stats.hits;
result.residual_cache_computations = stats.computations;
result.residual_device = residual_cache.residual_device();
result.residual_device_requested = residual_cache.residual_device_requested();
result.residual_device_reason = residual_cache.residual_device_reason();
result.diagnostics.orientation_cache_requests = stats.requests;
result.diagnostics.orientation_cache_hits = stats.hits;
result.diagnostics.orientation_cache_computations = stats.computations;
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
```

Expected:

```text
The standalone regrVonPS CUDA orientation-device test prints PASS.
```

## Phase 6: R API And Public Wrapper Plumbing

Purpose: expose orientation controls without breaking existing calls.

- [ ] Modify `fastkpc/R/cuda_native.R`.

Extend `fast_kpc_wanpdag_cuda()`:

```r
orientation_residual_device = c("auto", "cpu", "cuda"),
orientation_batch_size = 0,
orientation_diagnostics = TRUE
```

Add `match.arg(orientation_residual_device)` and pass these to `.Call`.

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Extend `C_fast_kpc_wanpdag_cuda` argument list and registration arity.

Set:

```cpp
orientation_options.orientation_residual_device_requested =
  Rcpp::as<std::string>(orientation_residual_devices);
orientation_options.cuda_residual_fallback =
  Rcpp::as<bool>(cuda_residual_fallbacks);
orientation_options.orientation_batch_size =
  Rf_asInteger(orientation_batch_sizes);
orientation_options.orientation_diagnostics_enabled =
  Rcpp::as<bool>(orientation_diagnosticss);
```

Remove the old forced R annotation:

```cpp
orientation_list["residual_device"] = "cpu";
orientation_list["residual_device_requested"] = residual_device;
```

The orientation result conversion should now use the native result fields.

- [ ] Modify `fastkpc/R/fast_kpc.R`.

Add arguments:

```r
orientation_residual_device = c("auto", "cpu", "cuda")
orientation_batch_size = 0
orientation_diagnostics = TRUE
```

Add config fields:

```text
orientation_residual_device_requested
orientation_residual_device_used
orientation_batch_size
orientation_diagnostics
```

Update `validate_fastkpc_result()` required config fields.

Update `fastkpc_graph_metrics()` to include orientation diagnostics counters:

```text
orientation_dcov_batches
orientation_dcov_pairs
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
```

- [ ] Modify `fastkpc/R/native.R` CPU wrappers to accept orientation arguments when practical, but resolve CPU path to `orientation_residual_device_used = "cpu"` so public calls do not fail merely because the engine is CPU.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_orientation_device_public_api.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
```

Expected:

```text
All listed tests print PASS.
```

## Phase 7: WAN-PDAG Orientation Equivalence Tests

Purpose: prove graph semantics are unchanged.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
```

Expected:

```text
PASS with pdag identical, counts identical, event sequence identical, finite event p-value diffs < 1e-7.
```

- [ ] Add or extend `fastkpc/R/wanpdag_validation.R`.

Add:

```r
compare_wanpdag_orientation_devices <- function(seed = 602,
                                                n = 140,
                                                alpha = 0.2,
                                                max_conditioning_size = 2,
                                                residual_device = "cuda",
                                                orientation_batch_sizes = c(1L, 0L),
                                                fastspline_params = list(knots = 8,
                                                                         lambda_count = 17,
                                                                         ridge = 1e-8)) {
  ...
}
```

Required return:

```text
cpu_orientation
cuda_orientation
batch_size_diffs
metrics
diagnostics
```

Required metrics:

```text
pdag_identical
orientation_counts_identical
event_sequence_identical
max_abs_event_pvalue_diff
orientation_dcov_batches
orientation_dcov_pairs
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
```

- [ ] Add benchmark helper:

```r
benchmark_wanpdag_orientation_devices <- function(seed = 603,
                                                  n = 180,
                                                  repeats = 3,
                                                  orientation_residual_devices = c("cpu", "cuda")) {
  ...
}
```

Benchmark must not enforce a speedup threshold.

- [ ] Run validation helper smoke:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); x <- compare_wanpdag_orientation_devices(); print(x$metrics); stopifnot(isTRUE(x$metrics$pdag_identical)); stopifnot(isTRUE(x$metrics$orientation_counts_identical)); stopifnot(x$metrics$orientation_dcov_batches > 0)'
```

Expected:

```text
No error.
Metrics show CUDA orientation dCov batches > 0 when generalized checks occur.
```

## Phase 8: Campaign, Report Writer, And CLI Integration

Purpose: make orientation-device behavior visible in long validation runs.

- [ ] Modify `fastkpc/R/validation_campaign.R`.

Add campaign grid dimension:

```r
orientation_residual_devices = c("auto")
```

Add `orientation_batch_size` and `orientation_diagnostics` parameters.

Update run id:

```text
scenario-seed-n-engine-residual_backend-residual_device-orientation_residual_device-scheduler
```

Add run columns:

```text
orientation_residual_device
orientation_batch_size
orientation_dcov_batches
orientation_dcov_pairs
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
```

Add table:

```text
orientation_device_diffs
```

Required columns:

```text
scenario
seed
n
engine
residual_backend
residual_device
scheduler
left_orientation_residual_device
right_orientation_residual_device
pdag_identical
orientation_counts_identical
event_sequence_identical
max_abs_event_pvalue_diff
status
```

Add table:

```text
orientation_device_diagnostics
```

Required columns:

```text
run_id
scenario
seed
n
engine
residual_backend
residual_device
orientation_residual_device
scheduler
orientation_batch_size
regrvonps_calls
regrvonps_cuda_calls
regrvonps_cpu_calls
orientation_dcov_batches
orientation_dcov_pairs
orientation_residual_fits
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
orientation_cache_requests
orientation_cache_hits
orientation_cache_computations
```

- [ ] Modify `fastkpc/R/report_writer.R`.

Write:

```text
orientation_device_diffs.csv
orientation_device_diagnostics.csv
```

Add Markdown sections:

```markdown
## Orientation Device
## Orientation Device Diagnostics
```

- [ ] Modify `fastkpc/tools/run_fast_kpc.R`.

Add CLI options:

```text
--orientation-residual-device
--orientation-batch-size
--orientation-diagnostics
```

Print when diagnostics exist:

```text
orientation_residual_device=<value>
orientation_dcov_batches=<value>
orientation_dcov_pairs=<value>
orientation_cuda_residual_fits=<value>
orientation_cpu_fallback_fits=<value>
```

- [ ] Modify `fastkpc/tools/run_validation_campaign.R`.

Add CLI options:

```text
--orientation-residual-devices
--orientation-batch-size
--orientation-diagnostics
```

- [ ] Create `fastkpc/tests/test_orientation_device_campaign_report_cli.R`.

Required assertions:

```text
1. Campaign accepts orientation_residual_devices=c("cpu","cuda").
2. Campaign has orientation_device_diffs.
3. Campaign has orientation_device_diagnostics.
4. Report writes orientation_device_diffs.csv.
5. Report writes orientation_device_diagnostics.csv.
6. Summary markdown contains "Orientation Device".
7. run_fast_kpc.R accepts --orientation-residual-device cuda.
8. run_validation_campaign.R accepts --orientation-residual-devices cpu,cuda.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_device_campaign_report_cli.R
```

Expected:

```text
PASS.
```

## Phase 9: Documentation And Contract Tests

Purpose: document the new public surface and report artifacts.

- [ ] Modify `fastkpc/README.md`.

Add section:

```markdown
## CUDA WAN-PDAG Orientation Residuals
```

Document:

```text
orientation_residual_device
orientation_batch_size
orientation_diagnostics
CPU reference behavior
CUDA fallback behavior
diagnostic fields
validation command
benchmark command
known limits
```

Minimum validation command:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); x <- compare_wanpdag_orientation_devices(); print(x$metrics)'
```

Minimum benchmark command:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(benchmark_wanpdag_orientation_devices(repeats=3)$summary)'
```

- [ ] Modify `fastkpc/reports/README.md`.

Add artifact descriptions:

```text
orientation_device_diffs.csv
orientation_device_diagnostics.csv
```

- [ ] Create `fastkpc/tests/test_orientation_device_docs_contract.R`.

Required assertions:

```text
README contains "CUDA WAN-PDAG Orientation Residuals".
README contains "orientation_residual_device".
README contains "orientation_batch_size".
README contains "compare_wanpdag_orientation_devices".
reports README contains orientation_device_diffs.csv.
reports README contains orientation_device_diagnostics.csv.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_device_docs_contract.R
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
```

Expected:

```text
All listed tests print PASS.
```

## Phase 10: Benchmarks And Non-Strict Performance Diagnostics

Purpose: quantify orientation offload without making performance brittle.

- [ ] Create `fastkpc/tests/test_orientation_device_benchmark.R`.

Required behavior:

```text
1. Source wanpdag_validation.R.
2. Run benchmark_wanpdag_orientation_devices(seed=604, n=120, repeats=2).
3. Assert timings is a data frame.
4. Assert summary is a data frame.
5. Assert cpu and cuda orientation devices are present.
6. Assert graph_equal is TRUE.
7. Assert CUDA diagnostic rows report orientation_dcov_batches > 0 when generalized checks occur.
8. Do not assert speedup threshold.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_orientation_device_benchmark.R
Rscript fastkpc/tests/test_wanpdag_benchmark.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 11: Full Verification Campaign

Purpose: prove the new orientation-device slice is complete and does not regress earlier slices.

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

- [ ] Run new focused tests:

```bash
Rscript fastkpc/tests/test_regrvonps_cuda_orientation_device.R
Rscript fastkpc/tests/test_wanpdag_cuda_orientation_device.R
Rscript fastkpc/tests/test_fastkpc_orientation_device_public_api.R
Rscript fastkpc/tests/test_orientation_device_campaign_report_cli.R
Rscript fastkpc/tests/test_orientation_device_docs_contract.R
Rscript fastkpc/tests/test_orientation_device_benchmark.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run existing WAN-PDAG and public wrapper tests:

```bash
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
Rscript fastkpc/tests/test_wanpdag_benchmark.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
```

Expected:

```text
Every test prints PASS.
Legacy validation may report missing package diagnostics if pcalg/graph are unavailable, but the test must still pass.
```

- [ ] Run existing CUDA skeleton/residual regression tests:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
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
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run validation helper smoke:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); x <- compare_wanpdag_orientation_devices(); print(x$metrics); stopifnot(isTRUE(x$metrics$pdag_identical)); stopifnot(isTRUE(x$metrics$orientation_counts_identical)); stopifnot(x$metrics$orientation_dcov_batches > 0)'
```

Expected:

```text
No error.
Printed metrics show orientation_dcov_batches > 0 for CUDA orientation.
```

- [ ] Run compact orientation-device validation campaign:

```bash
rm -rf fastkpc/reports/orientation_device_smoke
Rscript fastkpc/tools/run_validation_campaign.R \
  --engines cuda \
  --residual-backends fastSpline \
  --residual-devices cuda \
  --orientation-residual-devices cpu,cuda \
  --schedulers layer \
  --seeds 11 \
  --n-values 80 \
  --scenarios chain,additive \
  --legacy FALSE \
  --output-dir fastkpc/reports/orientation_device_smoke
Rscript -e 'x <- read.csv("fastkpc/reports/orientation_device_smoke/orientation_device_diagnostics.csv"); print(x[, c("scenario", "orientation_residual_device", "orientation_dcov_batches", "orientation_dcov_pairs")]); stopifnot(any(x$orientation_residual_device == "cuda"), any(x$orientation_dcov_batches > 0))'
```

Expected:

```text
Campaign completes.
Report directory contains orientation_device_diffs.csv.
Report directory contains orientation_device_diagnostics.csv.
At least one CUDA orientation row records orientation_dcov_batches > 0.
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
1. fast_kpc() accepts orientation_residual_device, orientation_batch_size, and orientation_diagnostics.
2. fast_kpc_wanpdag_cuda() accepts and passes orientation residual device controls to native code.
3. OrientationResult includes residual_device, residual_device_requested, residual_device_reason, and diagnostics.
4. regrVonPS CUDA path preserves parents, conditioning_set, reject_count, and p_values within tolerance.
5. WAN-PDAG CUDA orientation-device path preserves pdag, orientation counts, event sequence, and finite event p-values within tolerance versus CPU orientation.
6. orientation_batch_size=1 matches automatic orientation batching.
7. orientation_residual_device="auto" resolves predictably and records reasons for CPU resolution/fallback.
8. Campaign output includes orientation_device_diffs and orientation_device_diagnostics.
9. Report writer emits orientation_device_diffs.csv and orientation_device_diagnostics.csv.
10. CLI tools accept orientation-device options and print useful counters.
11. README and reports README document the new orientation-device controls and artifacts.
12. All commands in Phase 11 pass in the local environment.
13. kpcalg/R MD5 checks remain OK.
```

## Notes For Future Workers

Keep the first implementation narrow:

```text
Accelerate numerical work inside regrVonPS.
Do not batch or parallelize graph mutation.
Do not alter orientation search order.
Keep CPU orientation as the reference.
Use additive diagnostics for every fallback.
Prefer correctness and deterministic equivalence over broader GPU scheduling.
```

If generalized orientation produces no `regrVonPS` calls on a small fixture, adjust the fixture rather than weakening diagnostics. The validation scenarios must exercise at least one CUDA orientation dCov batch so the new device path is proven by behavior, not just by API shape.
