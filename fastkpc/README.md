# fastkpc

Staged fast kPC backend for this workspace.

This directory is intentionally separate from `kpcalg/R`. The legacy package
files stay unchanged while the fast backend is developed and validated.

## Current Scope

Implemented:

- R exact distance-covariance gamma reference: `R/dcov_exact.R`
- Legacy R baseline helpers: `R/legacy_runner.R`
- C++ exact distance-covariance p-value backend
- C++ skeleton MVP with stable level semantics
- R wrappers around the native entry points
- Tests for exact dCov and skeleton behavior
- Opt-in CUDA batched exact dCov backend: `R/cuda_native.R`
- Opt-in CUDA skeleton path using per-level batched dCov and deterministic replay
- CPU-vs-CUDA validation helpers: `R/cuda_validation.R`
- Per-run residual cache for CPU and CUDA skeleton paths
- Residual cache validation helpers: `R/residual_validation.R`
- Opt-in `fastSpline` residual backend on CPU
- Backend-aware CPU/CUDA skeleton wrappers
- fastSpline validation and benchmark helpers: `R/fastspline_validation.R`
- Opt-in native WAN-PDAG orientation stage
- CPU/CUDA skeleton-to-WAN-PDAG pipeline wrappers
- WAN-PDAG validation and benchmark helpers: `R/wanpdag_validation.R`
- Opt-in CUDA `fastSpline` residual device for skeleton residual cache
- Layer-batched CUDA scheduler with legacy-order replay diagnostics
- True-batched CUDA `fastSpline` residual groups for scheduler residual prefetch
- Opt-in CUDA WAN-PDAG orientation residual/dCov execution for fastSpline
- Scheduler validation and benchmark helpers: `R/scheduler_validation.R`
- Opt-in native CPU HSIC gamma and HSIC permutation CI methods
- CUDA HSIC kernels for `hsic.gamma` and fixed-seed `hsic.perm`
- CUDA HSIC skeleton and WAN-PDAG orientation diagnostics
- CUDA HSIC validation and benchmark helpers: `R/hsic_cuda_validation.R`
- Operational precision-ladder reports, timing attribution, routing policy,
  compatibility envelope, workload structure stats, and true-batched-kernel
  decision artifacts for the fastSplineCUDA / mgcvExtractGPU stack

Not implemented:

- Full CUDA GAM replacement for mgcv
- Multi-GPU scheduling
- Replacement of exported `kpcalg::kpc()`
- `tprsApproxCUDA`, unless future attribution evidence reverses the current
  defer decision

## Operational precision ladder

The public backend positioning is:

```text
precision = "fast":
    fastSplineCUDA
    fastest approximate CUDA primary backend
    not mgcv-compatible

precision = "compatible":
    mgcvExtractGPU where the version/semantic envelope is supported
    mgcvExtractCPU / legacy mgcv fallback otherwise

precision = "hybrid":
    fastSplineCUDA primary
    mgcvExtractGPU near-alpha verifier
    canonical replay preserved
```

`mgcvExtractGPU` is a version-pinned compatibility bridge: mgcv constructs the
restricted setup and fastkpc uses GPU numerical paths where supported. It is not
a full mgcv clone and it is not a pure GPU approximation backend.

The precision `mgcvExtractGPU` executor now uses same-setup x/y pair batching
for selected fixed-sp CUDA solves after per-target spectral GCV selection.
It also uses an on-demand same-S prepared setup/spectral cache inside each run,
so later misses reuse response-free design and spectral state while preserving
target-specific `X'y`, GCV, `sp`, and solve work. This is not eager same-S group
planning/batching, it is not capacity-bounded prepared-cache eviction, and it
is not a true fused/batched GPU kernel. Diagnostics must keep
`true_batched_kernel = false` until a fused kernel exists.

Native CUDA validation remains opt-in. Running
`FASTKPC_RUN_CUDA_TESTS=1 fastkpc/tools/run_mgcv_gate_b_tests.sh` exercises the
native precision E2E gate and writes CPU/GPU parity evidence through
`fastkpc_run_native_cuda_precision_parity()`.

`fastSplineCUDA` remains the frozen approximate baseline. `tprsApproxCUDA`
remains deferred unless projection-floor, oracle-lambda, timing, and graph-level
evidence justify building a new pure GPU approximation.

Operational artifacts can be generated with:

```bash
Rscript fastkpc/tools/run_precision_ladder_summary_report.R
Rscript fastkpc/tools/run_precision_ladder_timing_campaign.R
Rscript fastkpc/tools/run_hybrid_policy_calibration_report.R
Rscript fastkpc/tools/run_workload_structure_stats.R
Rscript fastkpc/tools/run_true_batched_kernel_decision.R
```

CUDA-specific tests remain opt-in. GitHub Actions are intentionally absent
unless reintroduced by explicit request.

The precision policy and skeleton data plane are now wired into `fast_kpc()`:

```text
precision = "fast":
    preserves fastSpline primary execution

precision = "compatible":
    routes through the authoritative resolver
    fails closed when semantic/version/runtime envelope checks fail
    executes CPU and CUDA skeleton data-plane slices for |S| <= 2
    uses mgcvExtractGPU where supported
    falls back through mgcvExtractCPU/GCVBridge and legacy mgcv

precision = "hybrid":
    keeps fastSpline primary execution
    executes near-alpha verifier residualization for skeleton |S| <= 2
    records verifier and fallback receipts
    preserves canonical replay
    uses verifier p-values in real skeleton edge/sepset decisions
```

The default remains the existing behavior until held-out validation is accepted.
Diagnostics distinguish `backend_planned` from `backend_executed`;
`backend_used` refers to the actual executor. Current precision data-plane scope
is skeleton only, CPU/CUDA, and single-penalty `|S| <= 2`. On-demand same-S
prepared setup/spectral caching is implemented for the precision GPU executor;
eager same-S group planning/batching, capacity-bounded prepared-cache eviction,
WAN-PDAG, `|S| > 2` multi-penalty GCV, and true fused/batched
`mgcvExtractGPU` kernels remain future work pending native CUDA parity and
timing evidence.

## Build

The first slice uses `Rcpp::sourceCpp()` rather than a package build system.
The wrapper builds the native code on first use:

```bash
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

`sourceCpp()` creates object files in `fastkpc/src` during this local build.
Those files are build artifacts, not source deliverables; rebuild them with the
command above after removing them.

## CUDA Scope

The CUDA backend is opt-in through `fastkpc/R/cuda_native.R`. It adds:

- `fast_dcov_batch_cuda()`: evaluates many exact dCov gamma tests in one CUDA call.
- `fast_skeleton_cuda()`: runs the C++ skeleton MVP with per-level CUDA dCov batches.

The CUDA skeleton preserves the CPU exact MVP's stable-level replay semantics:
candidate tests are batched for evaluation, then replayed in deterministic CPU
order so adjacency, sepsets, pMax, and `n.edgetests` match the CPU reference.

## Layer-batched CUDA scheduler

The CUDA skeleton backend supports an explicit scheduler mode:

```r
fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           scheduler = "layer")
fast_kpc(data, engine = "cuda", scheduler = "layer")
```

Scheduler modes:

```text
scheduler="layer": build one task plan per PC skeleton level, prefetch residuals,
                   evaluate dCov in CUDA batches, then replay in legacy order.
scheduler="legacy": keep the previous CUDA skeleton loop as a reference path.
scheduler="auto": use layer scheduling for CUDA skeleton execution and legacy
                  scheduling for CPU execution.
```

`scheduler="layer"` preserves legacy replay order. It may evaluate speculative
tasks, but only accepted replayed tests update adjacency, sepsets, pMax, and
`n.edgetests`. `scheduler="legacy"` remains available as a reference path for
graph-level comparisons.

`residual_backend` and `residual_device` keep their previous meanings:
`residual_backend` selects the statistical residual model, while
`residual_device` selects where supported residual fits execute. The
`residual_batch_size` option controls materialization grouping for residual
vectors; it does not change the statistical residual model. `batch_size`
continues to control dCov batch grouping.

## Scheduler diagnostics

CUDA skeleton results include:

```text
scheduler
scheduler_requested
scheduler_diagnostics$summary
scheduler_diagnostics$levels
scheduler_diagnostics$batches
scheduler_diagnostics$residuals
```

The diagnostics distinguish planned work, evaluated work, replayed tests, and
ignored speculative work. The summary records counts such as `tasks_planned`,
`tasks_evaluated`, `tests_replayed`, `tasks_ignored_after_delete`,
`dcov_batches`, `unique_residual_requests`, and `residual_batches`.

## CI methods

The default conditional-independence method remains exact dCov gamma:

```r
fast_kpc(data, ci_method = "dcc.gamma")
```

HSIC methods are opt-in. Set `ci_diagnostics = TRUE` to keep CI method counts
and backend-resolution fields in the result:

```r
fast_kpc(data, ci_method = "hsic.gamma", hsic_params = list(sig = 1))
fast_kpc(data, ci_method = "hsic.perm",
         permutation_params = list(replicates = 100, seed = 42,
                                   include_observed = TRUE))
```

Native HSIC uses dense RBF Gram matrices on CPU. Conditional tests reuse the
selected residual backend: raw vectors are tested when the conditioning set is
empty, and residual vectors are tested otherwise. CUDA dCov remains the CUDA
path for `ci_method = "dcc.gamma"`.

## CUDA HSIC kernels

CUDA HSIC kernels are available for CUDA runs with `ci_method = "hsic.gamma"`
and for `ci_method = "hsic.perm"` when an explicit permutation seed is supplied.
Successful GPU execution records `ci_backend = "cuda-hsic"` and positive CUDA
HSIC counters such as `ci_hsic_cuda_batches` and `ci_hsic_cuda_pairs`.

`hsic.perm` on CUDA requires an explicit permutation seed. If the seed is
missing, the run uses native-cpu fallback and records the reason
`CUDA HSIC permutation requires explicit seed in this stage`. Other unsupported
conditions, such as configured sample-size limits, also use `ci_backend =
"native-cpu"` with `ci_backend_reason` unless `cuda_memory_fallback = FALSE`.

Useful CUDA HSIC controls and diagnostics:

```r
fast_kpc(
  data,
  engine = "cuda",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1,
                     cuda_max_n = 2048,
                     cuda_max_batch_pairs = 64,
                     cuda_memory_fallback = TRUE)
)

fast_kpc(
  data,
  engine = "cuda",
  ci_method = "hsic.perm",
  permutation_params = list(replicates = 100, seed = 42,
                            include_observed = TRUE)
)
```

Top-level results expose `config$ci_backend`, `config$cuda_hsic_used`,
`diagnostics$cuda_hsic_available`, and `diagnostics$cuda_hsic_reason`.
Skeleton results expose `skeleton$ci_backend`,
`skeleton$ci_backend_reason`, and `skeleton$ci_diagnostics`.
WAN-PDAG orientation results expose `orientation$ci_backend`,
`orientation$ci_backend_reason`, and `orientation$ci_diagnostics`.

Validation reports include `hsic_cuda_backend_diagnostics.csv`,
`hsic_cuda_cpu_fallbacks.csv`, and `hsic_cuda_perf.csv`.
The legacy `kpcalg/R/*.R` files are not modified.

Validation campaigns and reports include `ci_method`, `ci_method_diffs.csv`,
`ci_method_diagnostics.csv`, `hsic_cuda_backend_diagnostics.csv`,
`hsic_cuda_cpu_fallbacks.csv`, and `hsic_cuda_perf.csv`. Helper functions in
`R/hsic_validation.R` validate native HSIC numerics and deterministic
permutation behavior. Helper functions in `R/hsic_cuda_validation.R` validate
CUDA HSIC numerics, fixed-seed CUDA permutation behavior, CPU/CUDA skeleton
agreement, and non-strict benchmark timings.

Validation campaigns can compare schedulers with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = "cuda",
  schedulers = c("legacy", "layer")
)
```

Generated reports include `scheduler_diffs.csv`, `scheduler_levels.csv`,
`scheduler_batches.csv`, `scheduler_residuals.csv`, and
`true_batched_residuals.csv`.

## True-Batched CUDA fastSpline Residuals

`fastspline_residual_batch_cuda()` groups residual requests by sample count,
fastSpline parameter set, and design column count. Compatible groups with two or
more fits run through a grouped CUDA path for crossproducts, penalized systems,
cuSOLVER batched Cholesky solves, GCV scoring, and residual computation.
Singleton groups keep using the single-fit CUDA path.

Batch results include `batch_diagnostics`:

```text
requested_fits
groups
true_batched_groups
true_batched_fits
single_fit_calls
cpu_fallback_fits
unique_designs
duplicate_design_fits
max_fits_per_design
group_table
```

Layer scheduler summaries include the corresponding residual counters:

```text
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
cuda_residual_unique_designs
cuda_residual_duplicate_design_fits
cuda_residual_max_fits_per_design
```

`residual_batch_size = 0` lets a scheduler level materialize all unique
compatible residual requests together. `residual_batch_size = 1` remains useful
as a one-at-a-time validation reference. If a requested design exceeds the
current true-batch solver limit or a CUDA batch solve fails, `fallback = TRUE`
uses the CPU fastSpline residual path and records the fallback reason; with
`fallback = FALSE`, the batch call errors.

Run a focused validation:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); x <- validate_cuda_fastspline_residual_batch(); print(x$cases); print(x$batch_diagnostics[c("groups","true_batched_groups","true_batched_fits","single_fit_calls")])'
```

Run a non-strict timing smoke:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); print(benchmark_cuda_fastspline_residual_batch(repeats=3)$summary)'
```

## CUDA WAN-PDAG Orientation Residuals

WAN-PDAG graph mutation remains sequential and deterministic, but the numerical
work inside generalized `regrVonPS` checks can now run on CUDA for CUDA
fastSpline WAN-PDAG runs:

```r
fast_kpc(
  data,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0,
  graph_stage = "wanpdag"
)
```

`orientation_residual_device` controls only the orientation residual/dCov work.
It does not change skeleton residual-device selection. Supported values are:

```text
orientation_residual_device = "auto" | "cpu" | "cuda"
```

`"auto"` resolves to CUDA for CUDA fastSpline WAN-PDAG runs and to CPU for CPU
engine, skeleton-only runs, or linear residuals. A linear request for
`orientation_residual_device = "cuda"` resolves to CPU with reason
`linear orientation residual CUDA device is not implemented`.

`orientation_batch_size = 0` batches all residual-vs-S dCov pairs in one
orientation check. `orientation_batch_size = 1` is the one-at-a-time validation
reference. The graph search order, accepted orientations, rule replay, and pdag
updates remain CPU-ordered and deterministic.

`orientation_diagnostics = TRUE` returns orientation-device resolution,
`regrVonPS` CPU/CUDA counters, residual fit counters, dCov batch/pair counters,
fallback counters, and orientation cache counters.

Orientation diagnostics are returned in:

```text
result$config$orientation_residual_device_requested
result$config$orientation_residual_device_used
result$config$orientation_residual_device_reason
result$orientation$residual_device
result$orientation$diagnostics
```

Key counters include:

```text
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

Validation campaigns can compare orientation devices with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = "cuda",
  orientation_residual_devices = c("cpu", "cuda")
)
```

Generated reports include `orientation_device_diffs.csv` and
`orientation_device_diagnostics.csv`.
`kpcalg/R` files are not modified by fastkpc staged backend work.

## CUDA Build

Build CUDA native code:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Or build through R:

```bash
Rscript -e 'source("fastkpc/R/cuda_native.R"); build_fastkpc_cuda_native(rebuild=TRUE)'
```

The build uses `/usr/local/cuda/bin/nvcc`, `-arch=sm_89`, and writes
`fastkpc/build/fastkpc_cuda.so`.

`fastkpc/build/*.o` and `fastkpc/build/fastkpc_cuda.so` are local build
artifacts. They can be removed with:

```bash
bash fastkpc/tools/clean_cuda_native.sh
```

## CUDA Tests

Run CUDA tests serially:

```bash
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
```

## CPU-vs-CUDA Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/cuda_validation.R"); print(validate_cuda_dcov_batch()); print(validate_cuda_skeleton_scenario())'
```

Expected fixed-scenario graph behavior:

```text
adjacency_identical TRUE
sepsets_identical TRUE
n_edgetests_identical TRUE
max_abs_pmax_diff < 1e-8
```

## Residual Cache Scope

The residual cache is shared by the CPU exact skeleton and CUDA skeleton paths.
It caches conditional residual vectors within one skeleton run using:

```text
target variable
sorted conditioning set
residual backend name
residual backend parameters
sample count
variable count
```

Unconditional tests do not use the cache. The supported native residual
backends are:

```text
linear:     intercept=true;ridge=1e-8
fastSpline: degree=3;knots=10;lambda_grid=1e-4:1e4:25;ridge=1e-8;mode=auto
```

## Residual Cache API

CPU:

```r
source("fastkpc/R/native.R")
fast_skeleton_cpp_cached(data, alpha, max_conditioning_size,
                         residual_cache = TRUE)
```

CUDA:

```r
source("fastkpc/R/cuda_native.R")
fast_skeleton_cuda_cached(data, alpha, max_conditioning_size,
                          batch_size = 0,
                          residual_cache = TRUE)
```

Both return the normal skeleton result plus:

```text
residual_backend
residual_cache$enabled
residual_cache$requests
residual_cache$hits
residual_cache$misses
residual_cache$computations
residual_cache$stored_vectors
residual_cache$stored_values
residual_cache$backend_name
```

## Residual Cache Tests

Run:

```bash
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

## Residual Cache Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/residual_validation.R"); print(validate_cpu_residual_cache()); print(validate_cuda_residual_cache())'
```

Expected:

```text
adjacency_identical TRUE
sepsets_identical TRUE
n_edgetests_identical TRUE
cache_stats$hits > 0
cache_stats$computations < cache_stats$requests
```

## Residual Cache Known Limits

- The cache is per skeleton run and is not global across R calls.
- The cache does not implement mgcv equivalence.
- The cache does not implement GPU residualization.
- The cache does not change exported `kpcalg::kpc()`.

## fastSpline Residual Backend Scope

`fastSpline` is an opt-in residual backend for conditional tests. It uses a
native C++ cubic spline design and ridge-regularized penalized least squares:

```text
|S| = 1: cubic spline basis
|S| = 2: tensor-product cubic spline basis
|S| > 2: additive cubic spline blocks
lambda: fixed log grid selected by GCV
```

CUDA skeleton runs may use `fastSpline`, but residuals are still computed on
CPU and then packed into CUDA dCov batches. No CUDA spline kernels are
implemented in this stage.

## fastSpline API

CPU backend-aware skeleton:

```r
source("fastkpc/R/native.R")
fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                          residual_backend = "fastSpline",
                          residual_cache = TRUE)
```

CUDA backend-aware skeleton:

```r
source("fastkpc/R/cuda_native.R")
fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           residual_backend = "fastSpline",
                           residual_cache = TRUE,
                           batch_size = 0)
```

One-off residual fit:

```r
source("fastkpc/R/native.R")
fastspline_residual(y, S)
```

Both skeleton wrappers return the normal skeleton result plus:

```text
residual_backend
residual_backend_params
residual_cache
```

## fastSpline Tests

Run:

```bash
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
```

## fastSpline Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/fastspline_validation.R"); print(validate_fastspline_against_mgcv()); print(compare_fastspline_linear_graph()); print(compare_fastspline_cpu_cuda_graph())'
```

Expected fixed-scenario behavior:

```text
mgcv one-dimensional residual correlation >= 0.97
mgcv two-dimensional residual correlation >= 0.85
CPU-vs-CUDA fastSpline adjacency identical
CPU-vs-CUDA fastSpline max_abs_pmax_diff < 1e-8
```

Graph differences from legacy `mgcv` are reported, not hidden.

## fastSpline Benchmark

Run:

```bash
Rscript -e 'source("fastkpc/R/fastspline_validation.R"); print(benchmark_fastspline_backends())'
```

The helper returns timing rows, cache stats, a linear-vs-fastSpline graph diff,
and a CPU-vs-CUDA fastSpline graph diff.

## fastSpline Known Limits

- fastSpline is opt-in and not the default backend.
- fastSpline is not mgcv and graph differences from legacy mgcv can occur.
- fastSpline residuals are computed on CPU even when CUDA dCov is used.
- No CUDA spline kernels are implemented in this goal.
- No WAN-PDAG migration is implemented in this goal.
- `kpcalg::kpc()` is not replaced.

## WAN-PDAG Orientation Scope

WAN-PDAG orientation is implemented as an opt-in fastkpc stage. It consumes the
native skeleton result, builds a partially directed integer adjacency matrix,
orients collider triples, applies the native orientation rules, and runs the
generalized transitive orientation checks with the existing residual backend and
per-run residual cache.

The orientation engine lives under `fastkpc/src` and does not modify legacy
package files. kpcalg/R/*.R files are not modified, and kpcalg::kpc() is not replaced.

## WAN-PDAG API

CPU skeleton plus native orientation:

```r
source("fastkpc/R/native.R")
fast_kpc_wanpdag_cpp(data, alpha, max_conditioning_size,
                     residual_backend = "fastSpline",
                     residual_cache = TRUE)
```

Orient an existing fastkpc skeleton:

```r
source("fastkpc/R/native.R")
fast_orient_wanpdag_cpp(skeleton_result, data,
                        residual_backend = "fastSpline",
                        residual_cache = TRUE)
```

CUDA skeleton plus the same native orientation engine:

```r
source("fastkpc/R/cuda_native.R")
fast_kpc_wanpdag_cuda(data, alpha, max_conditioning_size,
                      residual_backend = "fastSpline",
                      residual_cache = TRUE,
                      batch_size = 0)
```

The returned object has:

```text
skeleton
orientation$pdag
orientation$events
orientation$counts
orientation$residual_cache
```

## WAN-PDAG Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(validate_wanpdag_against_legacy()); print(compare_wanpdag_cpu_cuda())'
```

`validate_wanpdag_against_legacy()` compares against
`kpcalg::udag2wanpdag()` when `pcalg` and `graph` are available. If those
packages are missing, the helper returns `available = FALSE` with an explicit
missing-package reason and still reports native event counts, cache stats, and
a deterministic hand-written fixture check.

`compare_wanpdag_cpu_cuda()` verifies that CPU and CUDA skeleton paths feed the
same native orientation stage and reports pdag equality, orientation-count
equality, skeleton pMax drift, and cache stats.

## WAN-PDAG Benchmark

Run:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(benchmark_wanpdag_pipelines())'
```

The helper returns timing rows, orientation cache rows, orientation counts, a
CPU-vs-CUDA WAN-PDAG diff, and a linear-vs-fastSpline WAN-PDAG diff.

## WAN-PDAG Known Limits

- WAN-PDAG orientation is opt-in.
- `kpcalg::kpc()` is not replaced.
- kpcalg/R/*.R files are not modified.
- HSIC gamma and HSIC permutation tests are implemented as opt-in native CPU
  CI methods and as CUDA HSIC kernels for supported CUDA runs.
- CUDA orientation residual/dCov execution is opt-in and limited to fastSpline.
- WAN-PDAG graph mutation remains sequential; CUDA is used only for numerical checks.
- `unfVect` conservative/majority exclusions are accepted as input but not fully implemented in native orientation.

## Public fast_kpc API

`R/fast_kpc.R` provides the opt-in public wrapper over the completed fastkpc
CPU/CUDA skeleton and WAN-PDAG stages:

```r
source("fastkpc/R/fast_kpc.R")
result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 2,
  engine = "auto",
  residual_backend = "fastSpline",
  graph_stage = "wanpdag"
)
```

`engine = "auto"` uses CUDA only when `fastkpc_cuda_available()` succeeds;
otherwise it falls back to CPU. `graph_stage = "skeleton"` returns the skeleton
only, while `graph_stage = "wanpdag"` also returns native orientation output.

## fastkpc_result

`fast_kpc()` returns class `fastkpc_result`. The normalized result contains:

```text
config
data_info
engine
skeleton
orientation
metrics
timings
cache
validation
benchmark
diagnostics
```

Helpers:

```r
validate_fastkpc_result(result)
fastkpc_result_summary(result)
fastkpc_extract_skeleton(result)
fastkpc_extract_pdag(result)
```

`print.fastkpc_result()` and `summary.fastkpc_result()` provide compact
interactive views.

## Validation Campaign

`R/validation_scenarios.R` defines deterministic synthetic scenarios:

```text
chain
fork
collider
independent
additive
```

`R/validation_campaign.R` runs reproducible CPU/CUDA and residual-backend
comparisons:

```r
source("fastkpc/R/validation_campaign.R")
campaign <- run_fastkpc_validation_campaign(
  seeds = c(11, 12, 13),
  n_values = c(80, 140),
  scenarios = c("chain", "fork", "collider", "independent", "additive"),
  engines = c("cpu", "cuda"),
  residual_backends = c("linear", "fastSpline"),
  legacy = TRUE,
  benchmark = TRUE
)
```

The campaign reports run rows, graph metrics, pairwise diffs, CPU-vs-CUDA
diffs, linear-vs-fastSpline diffs, legacy diagnostics, timing rows, cache rows,
orientation counts, and errors. Missing `pcalg` or `graph` packages are
reported as explicit legacy diagnostics, not as package-independent failures.

## Validation Reports

`R/report_writer.R` writes campaign artifacts:

```r
source("fastkpc/R/report_writer.R")
write_fastkpc_validation_report(campaign, "fastkpc/reports/example")
```

`write_fastkpc_validation_report()` writes `summary.md`, CSV tables for each
campaign section, and `campaign.rds`.

## Command Line Tools

Run one public-wrapper job from CSV:

```bash
Rscript fastkpc/tools/run_fast_kpc.R \
  --input data.csv \
  --output result.rds \
  --engine cpu \
  --residual-backend fastSpline \
  --alpha 0.2 \
  --max-conditioning-size 1 \
  --graph-stage wanpdag
```

Run a validation campaign and write a report:

```bash
Rscript fastkpc/tools/run_validation_campaign.R \
  --output-dir fastkpc/reports/example \
  --seeds 11,12 \
  --n-values 80 \
  --scenarios chain,independent \
  --engines cpu,cuda \
  --residual-backends linear,fastSpline \
  --orientation-residual-devices cpu,cuda \
  --alpha 0.2 \
  --max-conditioning-size 1 \
  --legacy TRUE
```

## Public Wrapper Known Limits

- kpcalg::kpc() is not replaced.
- kpcalg/R/*.R files are not modified.
- CUDA residual kernels are opt-in.
- Validation campaign reports graph differences; it does not force equality.
- Legacy comparison requires pcalg and graph.

## CUDA Residual Device

CUDA residual kernels are opt-in through the `residual_device` argument. The
statistical residual model remains selected by `residual_backend`, while
`residual_device` selects where supported residual fits run:

```text
residual_backend = "fastSpline"
residual_device = "auto" | "cpu" | "cuda"
```

`residual_device = "cuda"` uses CUDA fastSpline residual fitting in the CUDA
skeleton path. `residual_device = "cpu"` keeps the previous CPU residual path.
`residual_device = "auto"` resolves to CUDA for CUDA fastSpline skeleton runs
when CUDA is available. linear residual CUDA device is not implemented; linear
residual requests resolve to CPU with an explicit diagnostic reason.

## Standalone CUDA Residual API

Standalone helpers are available for validating residual numerics directly:

```r
source("fastkpc/R/cuda_native.R")
fit <- fastspline_residual_cuda(y, S, fallback = TRUE)
batch <- fastspline_residual_batch_cuda(data, targets, conditioning_sets)
```

`fastspline_residual_cuda()` returns residuals, fitted values, selected lambda,
GCV, RSS, EDF, design column count, ridge attempts, backend, residual backend,
residual device, fallback status, and diagnostics.

`fastspline_residual_batch_cuda()` returns the same quantities for a batch of
target/conditioning-set residual fits.

## Residual Device In fast_kpc

The public wrapper accepts:

```r
fast_kpc(data,
         engine = "cuda",
         residual_backend = "fastSpline",
         residual_device = "cuda")
```

`fastkpc_result$config` records `residual_device_requested` and
`residual_device_used`. CUDA residual fallback can resolve the used value to
`cuda-fallback-cpu` when fallback is enabled and a CUDA residual fit fails.

## CUDA Residual Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); print(validate_cuda_fastspline_residuals()); print(benchmark_cuda_fastspline_residuals(repeats=2))'
```

Expected fixed-scenario behavior:

```text
max_abs_residual_diff < 1e-7
max_abs_fitted_diff < 1e-7
relative_rss_diff < 1e-8
```

Validation campaigns can compare residual devices with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = c("cpu", "cuda")
)
```

Generated reports include `residual_device_diffs.csv`.

## CUDA Residual Known Limits

- CUDA residual kernels are opt-in.
- linear residual CUDA device is not implemented.
- CUDA residual fallback is explicit and may report `cuda-fallback-cpu`.
- CUDA WAN-PDAG orientation residuals are opt-in and limited to fastSpline.
- kpcalg::kpc() is not replaced.
- kpcalg/R/*.R files are not modified.

## mgcv-Compatible Residual Oracle And Hybrid Verification

fastSpline CUDA is a high-throughput approximate backend. It is useful as the
primary fast path for batched kPC conditional-independence workloads, but it is
not mgcv-equivalent and graph differences from legacy `mgcv::gam()` residuals
can occur.

mgcvExtractCPU is a version-pinned extraction oracle for compatibility
validation. It is restricted to the `kpcalg::regrXonS()` residualization
surface, may depend on `mgcv` internals, and is not the final portable product
backend. The protected baseline is:

```text
Baseline: mgcv Gate B fixed-sp self-solve + hybrid canonical replay
Commit: 5da2313
Tag: mgcv-gate-b-v1
```

`fastkpc_mgcv_extract_capabilities()` returns a machine-readable capability
and version boundary object for diagnostics, bug reports, and campaign output.

The mgcv fixed-sp reference calls `mgcv::gam(..., sp=sp, fit=TRUE)`.
The mgcvExtract fixed-sp self-solve uses `mgcv::gam(fit=FALSE)` setup data,
then solves the fixed-sp Gaussian penalized least-squares problem inside
fastkpc. For mgcv parity, that setup solve follows mgcv's all-fixed
`L/lsp0` semantics and records the version-pinned `mgcv-C-magic-fixed-sp`
kernel path in diagnostics; it does not call `mgcv::gam(fit=TRUE)` or
`mgcv::magic()`. mgcvExtractGCVBridge uses mgcv for smoothing-parameter
selection and fastkpc only for the fixed-sp solve; it is not a self-contained
GCV implementation.

The restricted compatibility target is:

```text
|S| == 0: direct CI test; no residualization
|S| <= 2: X_i ~ s(S variables jointly)
|S| > 2:  X_i ~ s(S_1) + s(S_2) + ... + s(S_k)
family: Gaussian identity
output: residuals only
```

The default s(s1, s2) is not a tensor-product smooth. It is mgcv's default
isotropic smooth semantics. Tensor-product smooths such as `te`, `ti`, or `t2`
are not part of legacy `kpcalg::regrXonS()` formula construction.

The near-alpha verifier runs a fast primary backend first and verifies tests
whose p-values are close to alpha on a log scale. Verification may replace the
p-value source, but it must preserve canonical edge and sepset replay order.

Gate B campaign:
`fastkpc/tools/run_mgcv_gate_b_campaign.sh` runs fixed-sp setup/self-solve
parity scenarios across formula classes, smoothing-parameter scales, sample
sizes, collinearity, near-constant conditioning variables, and tied values. It
writes `mgcv_gate_b_fixed_sp_campaign.csv`.

canonical hybrid replay:
the near-alpha verifier may replace p-values but not replay order. Primary rows
define `canonical_test_order_id`; verifier rows are joined by that id and
replayed deterministically. Sepsets are recorded from the canonical first
separating set after p-value replacement.

hybrid calibration campaign:
`fastkpc/tools/run_hybrid_calibration_campaign.sh` evaluates tau values such as
`log(1.5)`, `log(2)`, `log(3)`, and `log(5)` and writes
`hybrid_calibration_summary.csv` plus `hybrid_policy_summary.txt`. The summary
tracks near-alpha verifier rate, decision flip reduction, skeleton/sepset/WAN-
PDAG drift proxies, runtime proxies, speedup versus legacy, and a recommended
tau per scenario group.

graph-level golden snapshots:
`fastkpc_hybrid_golden_snapshots()` provides deterministic canonical replay
snapshots for linear, nonlinear additive, pairwise full-smooth, and near-alpha
flip scenarios. These protect test order, p-value source selection, edge
deletion logs, sepsets, skeleton adjacency, and WAN-PDAG adjacency summaries.

Non-goals:

```text
No full mgcv clone
No bamGPU
No non-Gaussian family
No summary.gam/vcov/SE/prediction interval compatibility
No GAMM
No by-smooth or factor-smooth support
No pretending fastSpline is mgcv-equivalent
No sharing smoothing parameters across targets
```

## Tests

Run all first-slice tests:

```bash
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
```

Run these commands serially. The current first-slice native wrapper uses
`Rcpp::sourceCpp()`, and parallel test processes can race while building object
files in `fastkpc/src`.

Run the external GPU numeric baseline:

```bash
Rscript gpu-dcov/validate.R
```

## Behavior Notes

The exact dCov backend removes the old `RSpectra::eigs()` truncation from
`kpcalg::dcov.gamma`. It computes the full double-centered distance covariance
statistic:

```text
nV2 = sum((HKH) * (HLH)) / n
```

The `legacy_index` option controls the old kpcalg index behavior:

- `legacy_index = TRUE`: preserve old behavior and ignore the distance exponent.
- `legacy_index = FALSE`: apply the documented distance exponent.

The skeleton MVP uses a native linear residualization path for conditional
tests. This validates scheduler mechanics and is not a replacement for the
legacy `mgcv` residual backend.

`R/legacy_runner.R` provides optional helpers for running the legacy kpcalg R
functions from `kpcalg/R` as validation baselines. They do not modify or replace
the legacy package files.
