# Fast kPC True-Batched fastSpline CUDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current loop-based CUDA fastSpline residual batch API with a true batched CUDA implementation that preserves CPU/CUDA graph equivalence, feeds the layer scheduler residual prefetch path, and exposes diagnostics proving residual batches are no longer evaluated one fit at a time.

**Architecture:** Keep the existing fastSpline basis, penalty, lambda grid, GCV selection, R API names, and scheduler interfaces stable. Add an internal shape-grouped CUDA batch solver behind `fit_fastspline_residuals_cuda_batch()`: host code builds existing designs, groups fits by common `(n, design_cols, params)`, packs matrices once per group, launches batched kernels for crossproducts, penalized systems, solves, GCV selection, and final residuals, then returns results in original request order. The scheduler continues to call the same batch API, but gains diagnostics that distinguish true batched groups from single-fit fallback.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5, cuBLAS/cuSOLVER or an internal small dense SPD batched solver adapter, existing `fastkpc/src/cuda/fastspline_residual_cuda.cu`, existing layer-batched scheduler, existing validation campaign/report tooling, local shell build scripts under `fastkpc/tools/`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-15-fast-kpc-true-batched-fastspline-cuda-goal-execution.md: replace loop-based fastspline_residual_batch_cuda with a shape-grouped true batched CUDA fastSpline residual solver, integrate true-batch diagnostics into the layer scheduler residual prefetch path, extend R validation/campaign/report/CLI coverage, preserve CPU/CUDA graph equivalence, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `760000`.

Do not mark the goal complete until every criterion in "Completion Criteria" is satisfied. Mark the goal blocked only if the same local blocker prevents progress for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Current Baseline

The previous completed stages provide:

```text
fastspline_residual_cuda()
fastspline_residual_batch_cuda()
fast_skeleton_cuda_backend(..., residual_backend="fastSpline", residual_device="cuda")
layer-batched CUDA skeleton scheduler with residual prefetch/materialization
residual_batch_size scheduler option
validation campaign/report/CLI scheduler and residual-device dimensions
CUDA build script fastkpc/tools/build_cuda_native.sh
```

Important current implementation facts:

```text
fastspline_residual_batch_cuda() is exported and returns matrix-shaped batch results.
fit_fastspline_residuals_cuda_batch() currently loops over fit_fastspline_residuals_cuda() once per fit.
The layer scheduler already batches unique residual requests before dCov packing.
CudaSkeletonResidualCache::prefetch_level() calls fit_fastspline_residuals_cuda_batch() when residual_backend="fastSpline" and resolved residual_device is "cuda".
The existing single-fit CUDA path builds XtX/Xty on device, scans ridge/lambda, solves by Cholesky, computes residuals, and falls back to CPU when allowed.
The CPU fastSpline solver remains the numerical reference.
WAN-PDAG orientation residuals remain CPU-side.
```

The bottleneck this plan targets:

```text
The scheduler can now collect many residual fits at once, but the CUDA residual batch implementation still performs one complete CUDA fit per residual request. That causes repeated allocations, transfers, solver setup, kernel launches, and host synchronizations inside each scheduler level.
```

## Scope

In scope:

- Keep public R functions stable:

```r
fastspline_residual_cuda(y, S, fastspline_params = list(), fallback = TRUE)
fastspline_residual_batch_cuda(data, targets, conditioning_sets,
                               fastspline_params = list(), fallback = TRUE)
fast_skeleton_cuda_backend(..., residual_batch_size = 0,
                           scheduler = c("auto", "layer", "legacy"))
fast_kpc(..., residual_device = c("auto", "cpu", "cuda"),
         scheduler = c("auto", "layer", "legacy"),
         residual_batch_size = 0)
```

- Add internal true batched CUDA execution for supported fastSpline shape groups.
- Group batch requests by common design dimensions instead of requiring all requests to share a conditioning-set size.
- Preserve original request order in returned batch matrices and metadata vectors.
- Preserve current fallback behavior:

```text
fallback=FALSE: unsupported or failing CUDA batch work errors clearly.
fallback=TRUE: failed CUDA batch work falls back to CPU for affected fits and records diagnostics.
```

- Add top-level batch diagnostics to `fastspline_residual_batch_cuda()`:

```text
requested_fits
groups
true_batched_groups
true_batched_fits
single_fit_calls
cpu_fallback_fits
max_group_size
min_group_size
cholesky_backend
batch_mode
```

- Add scheduler diagnostics for residual batch execution:

```text
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
```

- Add validation helpers comparing:

```text
single fit CUDA vs batch CUDA for identical fits
CPU fastSpline vs true batched CUDA fastSpline
batch_size=1 residual materialization vs automatic residual batching
layer scheduler residual_device="cpu" vs residual_device="cuda"
legacy CUDA scheduler vs layer CUDA scheduler with true batched residuals
```

- Extend validation campaign, report writer, and CLIs to surface true-batch residual diagnostics.
- Add benchmark helpers showing residual batch amortization without making a strict speedup a correctness requirement.
- Keep all existing tests passing.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not change the mathematical fastSpline basis, penalty, lambda grid, ridge escalation rule, or GCV criterion.
- Do not replace exported legacy `kpcalg::kpc()`.
- Do not modify any file under `kpcalg/R`.
- Do not move WAN-PDAG orientation residuals to CUDA.
- Do not implement multi-GPU scheduling.
- Do not change exact dCov gamma statistics, `legacy_index` semantics, scheduler replay semantics, or sepset semantics.
- Do not require a speedup threshold for correctness. Use speed metrics as diagnostics only.
- Do not initialize git if the workspace is not already a git repository.

## Design Contract

### Definition Of True Batched For This Goal

`fit_fastspline_residuals_cuda_batch()` is true batched when a supported group of two or more fits:

```text
1. Does not call fit_fastspline_residuals_cuda() once per fit.
2. Allocates and transfers the packed group data once per group.
3. Launches group kernels whose grid includes a fit dimension.
4. Evaluates lambda/ridge candidates for the group through one grouped solve path per candidate.
5. Produces diagnostics with single_fit_calls == 0 for the supported group.
```

The implementation may still loop over:

```text
shape groups
ridge attempts
lambda values
```

Those loops are part of the statistical selection algorithm and do not violate the true-batch requirement.

### Shape Grouping Contract

Each requested fit has:

```text
target column
conditioning set
FastSplineDesign X
FastSplineDesign P
n
design_cols
fastSpline params
original request index
```

Group key:

```text
n
design_cols
degree
knots
lambda_min
lambda_max
lambda_count
ridge
mode
```

Rules:

```text
1. Requests with different design_cols must be in different groups.
2. Requests with different lambda grids must be in different groups.
3. Requests with different design matrix contents may still be in the same group as long as dimensions and params match.
4. Empty conditioning sets are supported as design_cols=1.
5. One-dimensional additive designs are supported.
6. Two-dimensional tensor designs are supported for the test/default parameter sizes already used by the project.
7. Conditioning sets of size greater than two use the existing additive design path and are supported when design_cols is within the solver adapter limit.
8. A group of size one may use the single-fit CUDA implementation but must count single_fit_calls=1.
9. A group of size greater than one must use the true batched path or fail/fallback according to the fallback option.
```

### Data Layout Contract

Use column-major matrices at the R boundary as before. Internally, pack group matrices explicitly:

```text
X_stack:     group_size x n x p in row-major-per-fit layout
P_stack:     group_size x p x p in column-major-per-fit layout
y_stack:     group_size x n
XtX_stack:   group_size x p x p in column-major-per-fit layout
Xty_stack:   group_size x p
A_stack:     group_size x p x p in column-major-per-fit layout
beta_stack:  group_size x p
Ainv_stack:  group_size x p x p when needed for edf
```

Required indexing helpers must live in CUDA source, not be duplicated ad hoc in kernels:

```cpp
__device__ __host__ inline std::size_t fit_matrix_offset(int fit, int row, int col,
                                                         int rows, int cols);
__device__ __host__ inline std::size_t fit_vector_offset(int fit, int row, int rows);
```

### Solver Adapter Contract

Add an internal adapter for batched SPD solves:

```cpp
struct BatchedSpdSolveResult {
  std::vector<int> info;
  std::string backend;
};
```

Required operations:

```text
factor A for every active fit in a group
solve A * beta = Xty for every active fit
solve A * Ainv = I for every active fit
record per-fit info
```

The adapter may use cuSOLVER batched APIs when available or an internal small dense SPD kernel. The public and test contract is the same either way:

```text
backend is a non-empty string.
info[k] == 0 means the fit candidate was solved.
non-zero info skips that fit candidate and continues the lambda/ridge scan.
No supported group of size > 1 may be silently rerouted to per-fit CUDA calls.
```

The implementation must support at least:

```text
design_cols = 1
design_cols = 1 + knots for knots in c(6, 8, 10)
design_cols = 1 + knots * knots for knots in c(6, 8, 10)
design_cols = 1 + m * knots for m in c(3, 4) and knots in c(6, 8, 10)
```

If a user requests a larger unsupported `design_cols` and `fallback=FALSE`, error with:

```text
CUDA fastSpline batch unsupported design_cols=<value> for true batched solve
```

If the same condition occurs with `fallback=TRUE`, use CPU fallback for that group and record the reason.

### Lambda And Ridge Selection Contract

For each fit, match the CPU/single-CUDA selection semantics:

```text
1. Start ridge at params.ridge.
2. For a ridge attempt, scan lambda_grid(params) in order.
3. For each successful candidate, compute beta, fitted, residuals, rss, edf, denominator, and gcv.
4. Skip non-finite candidates and candidates with denominator <= 1e-8.
5. Select the smallest gcv.
6. If abs(gcv - best_gcv) <= 1e-14, select the smaller lambda.
7. If at least one finite candidate was found for a fit at the current ridge, that fit completes at that ridge.
8. Only fits with no finite candidate continue to the next ridge.
9. Stop ridge escalation after the same effective upper bound used by the existing CPU and CUDA solvers.
```

Implementation note for efficiency:

```text
Store best_beta, best_lambda, best_gcv, best_rss, best_edf, and best_ridge_attempts per fit on device or in small host arrays. Compute final fitted/residuals once from best_beta after selection.
```

### Diagnostics Contract

`fastspline_residual_batch_cuda()` must keep existing return fields and add:

```text
batch_diagnostics
```

`batch_diagnostics` fields:

```text
requested_fits
groups
true_batched_groups
true_batched_fits
single_fit_calls
cpu_fallback_fits
max_group_size
min_group_size
cholesky_backend
batch_mode
group_table
```

`group_table` columns:

```text
group_id
n
design_cols
fit_count
true_batched
single_fit_calls
cpu_fallback_fits
cholesky_backend
status
reason
```

Per-fit diagnostics must keep existing fields and add:

```text
batch_group_id
batch_position
true_batched
cholesky_backend
```

Allowed per-fit `residual_device` values remain:

```text
cuda
cuda-fallback-cpu
```

### Scheduler Integration Contract

The layer scheduler must continue to call a batch residual materialization function once per residual batch. It must record:

```text
residual_batches: number of scheduler residual materialization batches
cuda_residual_batch_groups: sum of fastSpline CUDA shape groups inside scheduler residual batches
cuda_residual_true_batched_groups: sum of true batched groups
cuda_residual_true_batched_fits: sum of fits handled by true batched groups
cuda_residual_single_fit_calls: sum of per-fit CUDA calls used inside batch API
cuda_residual_cpu_fallback_fits: sum of CPU fallback fits inside batch API
```

Required invariants for `scheduler="layer"`, `residual_backend="fastSpline"`, `residual_device="cuda"`, and `residual_batch_size=0` on validation scenarios:

```text
cuda_residual_true_batched_groups > 0 when any residual batch has at least two compatible fits.
cuda_residual_true_batched_fits > 0 when conditional tests exist.
cuda_residual_single_fit_calls == 0 for supported grouped validation scenarios.
residual_cache$computations == scheduler_diagnostics$summary$unique_residual_requests.
```

### Numeric Equivalence Contract

Standalone residual equivalence:

```text
max(abs(batch$residuals[, k] - cpu$residuals)) < 1e-7
max(abs(batch$fitted[, k] - cpu$fitted)) < 1e-7
relative rss diff < 1e-8
selected_lambda matches CPU or produces residual/fitted differences within tolerance
```

Skeleton equivalence:

```text
CUDA layer true-batch adjacency == CUDA layer residual_batch_size=1 adjacency
CUDA layer true-batch sepsets == CUDA layer residual_batch_size=1 sepsets
CUDA layer true-batch n.edgetests == CUDA layer residual_batch_size=1 n.edgetests
max(abs(pMax_true_batch - pMax_one_at_a_time)) < 1e-7
```

Public wrapper equivalence:

```text
fast_kpc(..., engine="cuda", scheduler="layer", residual_device="cuda", residual_batch_size=0)
must match residual_batch_size=1 for graph outputs on deterministic scenarios.
```

## File Structure

Create these files:

- `fastkpc/src/cuda/fastspline_batched_solver.hpp`  
  Internal structs and declarations for shape-grouped true batched CUDA fastSpline residual solving.

- `fastkpc/src/cuda/fastspline_batched_solver.cu`  
  CUDA kernels, solver adapter, shape-group execution, and per-fit result assembly for true batched fastSpline residual fits.

- `fastkpc/tests/test_cuda_fastspline_true_batch_contract.R`  
  Failing-first contract test for top-level batch diagnostics and no single-fit calls on compatible batch groups.

- `fastkpc/tests/test_cuda_fastspline_batch_grouping.R`  
  Mixed-shape batch test proving grouping preserves original order and CPU equivalence.

- `fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R`  
  Scheduler-level test proving residual prefetch uses true batched CUDA groups and preserves graph results.

- `fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R`  
  Campaign/report/CLI contract test for true-batch residual diagnostics.

- `fastkpc/tests/test_true_batched_fastspline_benchmark.R`  
  Non-strict benchmark smoke test recording batch amortization metrics.

Modify these files:

- `fastkpc/src/cuda/fastspline_residual_cuda.hpp`  
  Add batch result and diagnostics structs while preserving existing function names.

- `fastkpc/src/cuda/fastspline_residual_cuda.cu`  
  Route batch calls through the true batched implementation, keep single-fit CUDA path intact, and keep CPU fallback behavior.

- `fastkpc/src/skeleton_task_scheduler.hpp`  
  Add true-batch residual diagnostic counters to `SchedulerDiagnostics`.

- `fastkpc/src/skeleton_task_scheduler.cpp`  
  Initialize the new diagnostic counters.

- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Consume batch diagnostics from residual prefetch and aggregate them into scheduler diagnostics.

- `fastkpc/src/r_api_cuda.cpp`  
  Return top-level `batch_diagnostics`, per-fit diagnostics additions, and scheduler summary additions.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile and link `fastspline_batched_solver.cu`.

- `fastkpc/R/cuda_native.R`  
  Keep function signatures stable and ensure returned diagnostics are passed through unchanged.

- `fastkpc/R/cuda_residual_validation.R`  
  Add validation and benchmark helpers for true batched fastSpline residuals.

- `fastkpc/R/scheduler_validation.R`  
  Include true-batch residual counters in scheduler benchmarks and validation summaries.

- `fastkpc/R/validation_campaign.R`  
  Include true-batch residual diagnostic dimensions in campaign output.

- `fastkpc/R/report_writer.R`  
  Write true-batch residual diagnostics CSV artifacts and Markdown sections.

- `fastkpc/tools/run_fast_kpc.R`  
  Print true-batch residual diagnostics when present.

- `fastkpc/tools/run_validation_campaign.R`  
  Add report artifacts for true-batch residual diagnostics.

- `fastkpc/README.md`  
  Document true batched CUDA fastSpline residuals, diagnostics, validation commands, and limits.

- `fastkpc/reports/README.md`  
  Document new report CSV artifacts.

Do not modify:

- `kpcalg/R/*.R`

## Phase 0: Baseline Audit And Guardrails

Purpose: verify the scheduler/residual-device baseline before changing CUDA batch behavior.

- [ ] Run:

```bash
pwd
test -e docs/superpowers/plans/2026-06-14-fast-kpc-layer-batched-scheduler-goal-execution.md && echo previous-plan-ok
test -e fastkpc/src/cuda/fastspline_residual_cuda.cu && echo cuda-residual-source-ok
test -e fastkpc/src/skeleton_engine_cuda.cpp && echo scheduler-source-ok
test -e fastkpc/tests/test_cuda_fastspline_residual_batch.R && echo batch-test-ok
```

Expected:

```text
/data/wenyujianData/kpcalg
previous-plan-ok
cuda-residual-source-ok
scheduler-source-ok
batch-test-ok
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_fastkpc_scheduler_public_api.R
```

Expected:

```text
CUDA native build succeeds.
CPU sourceCpp build succeeds.
Each listed R test prints PASS.
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

If any baseline test fails, use systematic debugging before implementing this plan. Fix only fastkpc-owned regressions; do not alter `kpcalg/R`.

## Phase 1: Write Failing Tests For True-Batch Contracts

Purpose: lock external behavior before CUDA implementation work.

- [ ] Create `fastkpc/tests/test_cuda_fastspline_true_batch_contract.R`.

Required test content:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x
residual_values <- function(fit) fit$residuals %||% fit$residual
max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(501)
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

targets <- c(1L, 2L, 3L, 4L)
conditioning_sets <- list(5L, 5L, 5L, 5L)
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

batch <- fastspline_residual_batch_cuda(
  data,
  targets = targets,
  conditioning_sets = conditioning_sets,
  fastspline_params = params,
  fallback = FALSE
)

diag <- batch$batch_diagnostics
assert_true(is.list(diag), "batch diagnostics should be present")
assert_true(identical(as.integer(diag$requested_fits), length(targets)),
            "requested_fits should match input batch length")
assert_true(as.integer(diag$true_batched_groups) >= 1L,
            "compatible batch should use at least one true batched group")
assert_true(identical(as.integer(diag$true_batched_fits), length(targets)),
            "all compatible fits should be true batched")
assert_true(identical(as.integer(diag$single_fit_calls), 0L),
            "compatible batch must not call the single-fit CUDA path")
assert_true(identical(as.integer(diag$cpu_fallback_fits), 0L),
            "fallback should not be used")
assert_true(is.data.frame(diag$group_table), "group_table should be a data frame")
assert_true(all(diag$group_table$true_batched),
            "every compatible group should be marked true_batched")

for (k in seq_along(targets)) {
  cpu <- fastspline_residual(data[, targets[[k]]],
                             data[, conditioning_sets[[k]], drop = FALSE],
                             fastspline_params = params)
  assert_true(max_abs_diff(batch$residuals[, k], residual_values(cpu)) < 1e-7,
              paste("residual", k, "should match CPU"))
  assert_true(max_abs_diff(batch$fitted[, k], cpu$fitted) < 1e-7,
              paste("fitted", k, "should match CPU"))
  rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
  assert_true(rel_rss < 1e-8, paste("rss", k, "should match CPU"))
  assert_true(isTRUE(batch$diagnostics[[k]]$true_batched),
              paste("fit", k, "should be marked true_batched"))
}

cat("test_cuda_fastspline_true_batch_contract.R: PASS\n")
```

- [ ] Run the new test and verify it fails before implementation:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
```

Expected before implementation:

```text
FAIL because batch_diagnostics is missing or single_fit_calls is not zero.
```

- [ ] Create `fastkpc/tests/test_cuda_fastspline_batch_grouping.R`.

Required behavior:

```text
Use a mixed batch with conditioning sets integer(0), one variable, two variables, and three variables.
Assert batch_diagnostics$groups >= 3.
Assert output columns remain in input order.
Assert each residual/fitted column matches CPU within tolerance.
Assert group_table fit_count sums to requested_fits.
Assert every per-fit diagnostic has batch_group_id and batch_position.
```

Use this command:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
```

Expected before implementation:

```text
FAIL because grouping diagnostics are missing.
```

- [ ] Create `fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R`.

Required behavior:

```text
Build one deterministic nonlinear data set with n >= 120 and p >= 5.
Run fast_skeleton_cuda_backend() with residual_backend="fastSpline", residual_device="cuda", scheduler="layer", residual_batch_size=0.
Run the same call with residual_batch_size=1.
Assert adjacency, sepsets, n.edgetests, and pMax tolerance equivalence.
Assert scheduler summary contains cuda_residual_true_batched_groups > 0.
Assert scheduler summary contains cuda_residual_single_fit_calls == 0 for the automatic residual batch run.
```

Use this command:

```bash
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
```

Expected before implementation:

```text
FAIL because scheduler true-batch diagnostics are missing.
```

## Phase 2: Add Batch Diagnostic Types Without Changing Computation

Purpose: expose diagnostic shape first so R/API/tests can be developed independently from solver internals.

- [ ] Modify `fastkpc/src/cuda/fastspline_residual_cuda.hpp`.

Add:

```cpp
struct FastSplineCudaBatchDiagnostics {
  int requested_fits;
  int groups;
  int true_batched_groups;
  int true_batched_fits;
  int single_fit_calls;
  int cpu_fallback_fits;
  int max_group_size;
  int min_group_size;
  std::string cholesky_backend;
  std::string batch_mode;
  std::vector<int> group_id;
  std::vector<int> group_n;
  std::vector<int> group_design_cols;
  std::vector<int> group_fit_count;
  std::vector<int> group_true_batched;
  std::vector<int> group_single_fit_calls;
  std::vector<int> group_cpu_fallback_fits;
  std::vector<std::string> group_cholesky_backend;
  std::vector<std::string> group_status;
  std::vector<std::string> group_reason;
};

struct FastSplineCudaBatchResult {
  std::vector<FastSplineCudaFit> fits;
  FastSplineCudaBatchDiagnostics diagnostics;
};

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_batch_result(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);
```

Preserve the existing compatibility declaration:

```cpp
std::vector<FastSplineCudaFit> fit_fastspline_residuals_cuda_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);
```

- [ ] Modify `FastSplineCudaDiagnostics` in the same header.

Add:

```cpp
int batch_group_id;
int batch_position;
bool true_batched;
std::string cholesky_backend;
```

Initialize these fields in single-fit code:

```text
batch_group_id = -1
batch_position = 0
true_batched = false
cholesky_backend = "single-fit-cusolver"
```

- [ ] Modify `fastkpc/src/cuda/fastspline_residual_cuda.cu`.

Temporarily implement `fit_fastspline_residuals_cuda_batch_result()` by wrapping the existing loop and populating diagnostics honestly:

```text
requested_fits = targets.size()
groups = targets.size()
true_batched_groups = 0
true_batched_fits = 0
single_fit_calls = targets.size()
cpu_fallback_fits = count fallback_used
batch_mode = "loop"
```

The compatibility wrapper returns `result.fits`.

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Add helper:

```cpp
Rcpp::DataFrame fastspline_batch_group_table_to_df(
  const FastSplineCudaBatchDiagnostics& diagnostics);
```

Add helper:

```cpp
Rcpp::List fastspline_batch_diagnostics_to_list(
  const FastSplineCudaBatchDiagnostics& diagnostics);
```

Add per-fit diagnostic fields in `C_fastspline_residual_batch_cuda()`:

```text
batch_group_id
batch_position
true_batched
cholesky_backend
```

Add top-level return field:

```cpp
Rcpp::Named("batch_diagnostics") =
  fastspline_batch_diagnostics_to_list(batch_result.diagnostics)
```

- [ ] Build and run diagnostics-focused tests:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
```

Expected:

```text
Existing batch test passes.
True-batch contract test still fails only because true_batched_groups is 0 or single_fit_calls is nonzero.
```

## Phase 3: Add Shape Group Planning And Host Packing

Purpose: create deterministic shape groups and packed host buffers while still using old computation until the batched solver is ready.

- [ ] Create `fastkpc/src/cuda/fastspline_batched_solver.hpp`.

Required declarations:

```cpp
#ifndef FASTKPC_FASTSPLINE_BATCHED_SOLVER_HPP
#define FASTKPC_FASTSPLINE_BATCHED_SOLVER_HPP

#include "fastspline_residual_cuda.hpp"

#include <Rcpp.h>
#include <string>
#include <vector>

struct FastSplineBatchRequest {
  int original_index;
  int target;
  std::vector<int> conditioning_set;
  FastSplineDesign design;
};

struct FastSplineBatchGroup {
  int group_id;
  int n;
  int design_cols;
  std::vector<FastSplineBatchRequest> requests;
};

std::vector<FastSplineBatchGroup> make_fastspline_batch_groups(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params);

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_true_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback);

#endif
```

- [ ] Create `fastkpc/src/cuda/fastspline_batched_solver.cu`.

Implement `make_fastspline_batch_groups()`:

```text
1. Validate targets and conditioning_sets sizes before grouping.
2. Build FastSplineDesign for each request with make_fastspline_design().
3. Use a std::map string key so group order is deterministic by first request order.
4. Store original_index for result reconstruction.
5. Preserve request order inside each group by input position.
```

Key string fields:

```text
n
design.p
params.degree
params.knots
params.lambda_min
params.lambda_max
params.lambda_count
params.ridge
params.mode
```

- [ ] Add host packing helpers in `fastspline_batched_solver.cu`.

Required helper behavior:

```text
pack_group_X() returns group_size * n * p doubles.
pack_group_P() returns group_size * p * p doubles.
pack_group_y() returns group_size * n doubles.
All helpers throw clear dimension errors if a request does not match the group dimensions.
```

- [ ] Modify `fastkpc/tools/build_cuda_native.sh`.

Add compile line:

```sh
"$NVCC" -O3 -arch=sm_89 -Xcompiler -fPIC -std=c++17 \
  $COMMON_INC -c "$ROOT/src/cuda/fastspline_batched_solver.cu" \
  -o "$BUILD/fastspline_batched_solver.o"
```

Add link input before `fastspline_residual_cuda.o`:

```sh
"$BUILD/fastspline_batched_solver.o" \
```

- [ ] Temporarily route `fit_fastspline_residuals_cuda_batch_result()` through grouping but still compute fits with existing single-fit calls.

Expected diagnostic changes:

```text
groups reflects shape grouping.
group_table has one row per shape group.
single_fit_calls still equals requested_fits.
true_batched_groups remains 0.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
```

Expected:

```text
The grouping test still fails only on true_batched expectations if it requires them.
The group count, original order, and CPU equivalence assertions pass.
```

## Phase 4: Implement Batched CUDA Kernels And Solver Adapter

Purpose: add the actual group-level GPU execution path.

- [ ] In `fastkpc/src/cuda/fastspline_batched_solver.cu`, add device/host indexing helpers:

```cpp
__device__ __host__ inline std::size_t matrix_offset(int fit,
                                                     int row,
                                                     int col,
                                                     int rows,
                                                     int cols) {
  return (static_cast<std::size_t>(fit) * rows * cols) +
         (static_cast<std::size_t>(row) * cols) + col;
}

__device__ __host__ inline std::size_t colmajor_square_offset(int fit,
                                                              int row,
                                                              int col,
                                                              int p) {
  return (static_cast<std::size_t>(fit) * p * p) +
         (static_cast<std::size_t>(col) * p) + row;
}
```

- [ ] Add batched crossproduct kernels:

```text
batched_xtx_kernel: grid dimensions include fit, a, b and compute XtX per fit.
batched_xty_kernel: grid dimensions include fit, col and compute Xty per fit.
```

Required checks:

```text
No writes outside group_size * p * p or group_size * p.
Use double accumulation.
Synchronize and check cudaGetLastError() after launches.
```

- [ ] Add batched system build kernels:

```text
batched_build_system_kernel copies XtX + lambda * P + ridge diagonal into A.
batched_identity_kernel writes one identity matrix per active fit.
```

Rules:

```text
Ridge is added only on diagonal entries row > 0, matching existing CPU/CUDA solvers.
Inactive fits are skipped.
```

- [ ] Add batched fitted/residual/rss/edf/GCV kernels:

```text
batched_fitted_residual_kernel
batched_rss_kernel
batched_edf_kernel
batched_update_best_kernel
```

`batched_update_best_kernel` must implement:

```text
candidate accepted when !best_found
candidate accepted when gcv < best_gcv
candidate accepted when abs(gcv - best_gcv) <= 1e-14 and lambda < best_lambda
```

- [ ] Implement the solver adapter in the same file.

Required adapter function:

```cpp
BatchedSpdSolveResult solve_batched_spd_systems(double* d_A,
                                                double* d_rhs,
                                                double* d_Ainv,
                                                int group_size,
                                                int p,
                                                const std::vector<int>& active);
```

The adapter must:

```text
1. Factor A for every active fit.
2. Solve beta RHS for every active fit.
3. Solve inverse RHS for every active fit.
4. Return info for every fit in the group.
5. Set backend to a stable non-empty value such as "cusolver-batched" or "internal-small-spd".
```

- [ ] Add a design-cols guard in the true batch path.

Minimum supported values:

```text
p <= 128
```

If the final implementation supports a larger safe limit, encode that exact integer in one constant:

```cpp
constexpr int kMaxTrueBatchedDesignCols = 128;
```

- [ ] Build after kernels compile:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Expected:

```text
built: /data/wenyujianData/kpcalg/fastkpc/build/fastkpc_cuda.so
```

## Phase 5: Route Batch API Through True Batched Groups

Purpose: replace the loop-based batch implementation for supported groups.

- [ ] Implement `fit_fastspline_residuals_cuda_true_batch()`.

Required flow:

```text
1. Validate input lengths.
2. Build shape groups.
3. Allocate output vector with targets.size() elements.
4. For each group:
   a. If group size == 1, call fit_fastspline_residuals_cuda() and count one single_fit_call.
   b. If group design_cols exceeds supported limit and fallback=FALSE, throw.
   c. If group design_cols exceeds supported limit and fallback=TRUE, compute affected fits with CPU fallback and count cpu_fallback_fits.
   d. Otherwise run the true batched CUDA group solver.
   e. Scatter group results back to original_index.
5. Populate batch diagnostics and group table vectors.
```

- [ ] The true batched group solver must return per-fit `FastSplineCudaFit` values with:

```text
fit.residuals
fit.fitted
fit.selected_lambda
fit.gcv
fit.rss
fit.edf
fit.design_cols
fit.ridge_attempts
diagnostics.cuda_used = true
diagnostics.fallback_used = false
diagnostics.true_batched = true
diagnostics.batch_group_id = group_id
diagnostics.batch_position = position within group
diagnostics.cholesky_backend = solver adapter backend
```

- [ ] Modify `fit_fastspline_residuals_cuda_batch_result()` in `fastspline_residual_cuda.cu`:

```text
Call fit_fastspline_residuals_cuda_true_batch().
If that call throws and fallback=TRUE, return CPU fallback fits for all requested fits with batch diagnostics showing cpu_fallback_fits=requested_fits and the captured reason.
If that call throws and fallback=FALSE, rethrow with prefix "CUDA fastSpline residual batch failed: ".
```

- [ ] Keep `fit_fastspline_residuals_cuda_batch()` as a compatibility wrapper:

```cpp
return fit_fastspline_residuals_cuda_batch_result(
  data, targets, conditioning_sets, params, fallback).fits;
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
```

Expected:

```text
All four tests print PASS.
The true-batch contract test reports single_fit_calls == 0 for its compatible group.
```

## Phase 6: Integrate True-Batch Diagnostics Into Scheduler

Purpose: make scheduler residual prefetch prove it is using the new batch path.

- [ ] Modify `fastkpc/src/skeleton_task_scheduler.hpp`.

Add to `SchedulerDiagnostics`:

```cpp
int cuda_residual_batch_groups;
int cuda_residual_true_batched_groups;
int cuda_residual_true_batched_fits;
int cuda_residual_single_fit_calls;
int cuda_residual_cpu_fallback_fits;
```

- [ ] Modify `fastkpc/src/skeleton_task_scheduler.cpp`.

Initialize all new fields to zero in `make_scheduler_diagnostics()`.

- [ ] Modify `fastkpc/src/skeleton_engine_cuda.cpp`.

In `CudaSkeletonResidualCache::prefetch_level()`, replace:

```cpp
const std::vector<FastSplineCudaFit> fits =
  fit_fastspline_residuals_cuda_batch(...);
```

with:

```cpp
const FastSplineCudaBatchResult batch_result =
  fit_fastspline_residuals_cuda_batch_result(...);
const std::vector<FastSplineCudaFit>& fits = batch_result.fits;
```

Aggregate:

```text
diagnostics->cuda_residual_batch_groups += batch_result.diagnostics.groups
diagnostics->cuda_residual_true_batched_groups += batch_result.diagnostics.true_batched_groups
diagnostics->cuda_residual_true_batched_fits += batch_result.diagnostics.true_batched_fits
diagnostics->cuda_residual_single_fit_calls += batch_result.diagnostics.single_fit_calls
diagnostics->cuda_residual_cpu_fallback_fits += batch_result.diagnostics.cpu_fallback_fits
```

- [ ] Modify `fastkpc/src/r_api_cuda.cpp`.

Add the new fields to `scheduler_diagnostics_to_list()` summary:

```text
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
```

Expected:

```text
All listed tests print PASS.
Layer scheduler graph outputs match one-at-a-time residual materialization.
Scheduler summary reports true-batch residual counters.
```

## Phase 7: Extend R Validation And Benchmark Helpers

Purpose: make true-batch residual behavior reproducible outside unit tests.

- [ ] Modify `fastkpc/R/cuda_residual_validation.R`.

Add:

```r
validate_cuda_fastspline_residual_batch <- function(seed = 510,
                                                    n = 128,
                                                    p = 6,
                                                    fastspline_params = list(knots = 8,
                                                                             lambda_count = 17,
                                                                             ridge = 1e-8)) {
  build_fastkpc_native()
  build_fastkpc_cuda_native()
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    sin(z1) + stats::rnorm(n, sd = 0.08),
    cos(z1) + stats::rnorm(n, sd = 0.08),
    z1 * z2 + stats::rnorm(n, sd = 0.08),
    sin(z2) + stats::rnorm(n, sd = 0.08),
    cos(z2) + stats::rnorm(n, sd = 0.08),
    stats::rnorm(n)
  )
  targets <- c(1L, 2L, 3L, 4L, 5L)
  conditioning_sets <- list(6L, 6L, c(1L, 2L), c(1L, 2L), c(1L, 2L, 3L))
  batch <- fastspline_residual_batch_cuda(
    data, targets, conditioning_sets,
    fastspline_params = fastspline_params,
    fallback = FALSE
  )
  rows <- vector("list", length(targets))
  for (k in seq_along(targets)) {
    S <- data[, conditioning_sets[[k]], drop = FALSE]
    cpu <- fastspline_residual(data[, targets[[k]]], S,
                               fastspline_params = fastspline_params)
    residual_diff <- max(abs(as.numeric(batch$residuals[, k]) -
                               as.numeric(fastkpc_residual_values(cpu))))
    fitted_diff <- max(abs(as.numeric(batch$fitted[, k]) -
                             as.numeric(cpu$fitted)))
    rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
    rows[[k]] <- data.frame(
      fit = k,
      target = targets[[k]],
      conditioning_size = length(conditioning_sets[[k]]),
      status = if (residual_diff < 1e-7 && fitted_diff < 1e-7 && rel_rss < 1e-8) "ok" else "diff",
      max_abs_residual_diff = residual_diff,
      max_abs_fitted_diff = fitted_diff,
      relative_rss_diff = rel_rss,
      true_batched = isTRUE(batch$diagnostics[[k]]$true_batched),
      stringsAsFactors = FALSE
    )
  }
  list(
    config = list(seed = seed, n = n, p = p,
                  fastspline_params = fastspline_params),
    cases = do.call(rbind, rows),
    batch_diagnostics = batch$batch_diagnostics,
    raw = batch
  )
}
```

- [ ] Add `benchmark_cuda_fastspline_residual_batch()` in the same file.

Required output:

```text
timings data frame with mode in c("single_loop", "true_batch")
summary data frame grouped by mode
batch_diagnostics from true_batch run
```

Benchmark must not fail when true batch is slower. It only fails if the true batch run errors or returns non-equivalent residuals.

- [ ] Modify `fastkpc/R/scheduler_validation.R`.

Add true-batch counters to rows produced by `benchmark_layer_scheduler()`:

```r
cuda_residual_batch_groups =
  as.integer(scheduler_summary$cuda_residual_batch_groups %||% 0L)
cuda_residual_true_batched_groups =
  as.integer(scheduler_summary$cuda_residual_true_batched_groups %||% 0L)
cuda_residual_true_batched_fits =
  as.integer(scheduler_summary$cuda_residual_true_batched_fits %||% 0L)
cuda_residual_single_fit_calls =
  as.integer(scheduler_summary$cuda_residual_single_fit_calls %||% 0L)
cuda_residual_cpu_fallback_fits =
  as.integer(scheduler_summary$cuda_residual_cpu_fallback_fits %||% 0L)
```

- [ ] Add a validation smoke command:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); x <- validate_cuda_fastspline_residual_batch(); print(x$cases); print(x$batch_diagnostics[c("groups","true_batched_groups","single_fit_calls")])'
```

Expected:

```text
All cases have status ok.
true_batched_groups is positive.
single_fit_calls is zero for compatible grouped fits.
```

## Phase 8: Extend Campaign, Report Writer, CLI, And Docs

Purpose: make true-batch diagnostics visible in long validation runs and generated artifacts.

- [ ] Modify `fastkpc/R/validation_campaign.R`.

For each run with `engine_used == "cuda"` and a skeleton section, extract scheduler summary fields:

```text
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
```

Add them to the run-level campaign data frame. Missing fields must become zero integers.

- [ ] Add a campaign-level table:

```text
true_batched_residuals
```

Required columns:

```text
scenario
engine
residual_backend
residual_device
scheduler
residual_batch_size
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
status
```

- [ ] Modify `fastkpc/R/report_writer.R`.

Write:

```text
true_batched_residuals.csv
```

Add a Markdown section headed:

```markdown
## True-Batched CUDA fastSpline Residuals
```

The section must report:

```text
total true-batched groups
total true-batched fits
total single-fit CUDA calls
total CPU fallback fits
```

- [ ] Modify `fastkpc/tools/run_fast_kpc.R`.

When result skeleton scheduler diagnostics summary includes true-batch counters, print:

```text
cuda_residual_true_batched_groups=<value>
cuda_residual_true_batched_fits=<value>
cuda_residual_single_fit_calls=<value>
cuda_residual_cpu_fallback_fits=<value>
```

- [ ] Modify `fastkpc/tools/run_validation_campaign.R`.

Ensure the generated report directory includes `true_batched_residuals.csv` when CUDA scheduler runs are present.

- [ ] Modify `fastkpc/README.md`.

Add a section:

```markdown
## True-Batched CUDA fastSpline Residuals
```

Document:

```text
fastspline_residual_batch_cuda()
batch_diagnostics
scheduler true-batch counters
residual_batch_size=0 vs residual_batch_size=1 validation
fallback behavior
known design_cols limit
```

- [ ] Modify `fastkpc/reports/README.md`.

Add `true_batched_residuals.csv` to the artifact list with a one-sentence description.

- [ ] Create `fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R`.

Required assertions:

```text
run_fastkpc_validation_campaign() includes true-batch residual counters.
write_fastkpc_validation_report() writes true_batched_residuals.csv.
The Markdown report contains "True-Batched CUDA fastSpline Residuals".
The CLI command completes and prints cuda_residual_true_batched_groups.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
Rscript fastkpc/tests/test_layer_scheduler_docs_contract.R
```

Expected:

```text
All listed tests print PASS.
```

## Phase 9: Add Non-Strict Benchmarks And Regression Guards

Purpose: quantify the change without making performance noisy tests brittle.

- [ ] Create `fastkpc/tests/test_true_batched_fastspline_benchmark.R`.

Required behavior:

```text
Source cuda_residual_validation.R.
Run benchmark_cuda_fastspline_residual_batch(seed=511, n=160, repeats=2).
Assert timings is a data frame.
Assert both single_loop and true_batch modes are present.
Assert true_batch status is ok.
Assert batch_diagnostics$true_batched_groups > 0.
Do not assert a speedup threshold.
```

- [ ] Add a manual benchmark command to `fastkpc/README.md`:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); print(benchmark_cuda_fastspline_residual_batch(repeats=3)$summary)'
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_true_batched_fastspline_benchmark.R
Rscript fastkpc/tests/test_layer_scheduler_benchmark.R
Rscript fastkpc/tests/test_cuda_residual_benchmark.R
```

Expected:

```text
All listed tests print PASS.
Benchmark output records timings and true-batch counters.
```

## Phase 10: Full Verification Campaign

Purpose: prove correctness across existing CPU, CUDA, scheduler, residual, wrapper, report, and docs contracts.

- [ ] Run CUDA build from clean state:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Expected:

```text
built: /data/wenyujianData/kpcalg/fastkpc/build/fastkpc_cuda.so
```

- [ ] Run CPU native build:

```bash
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
No error.
```

- [ ] Run focused new tests:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
Rscript fastkpc/tests/test_cuda_fastspline_batch_grouping.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_true_batched_fastspline_campaign_report_cli.R
Rscript fastkpc/tests/test_true_batched_fastspline_benchmark.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run existing CUDA residual and scheduler tests:

```bash
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run public wrapper and campaign tests:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_fastkpc_scheduler_public_api.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_layer_scheduler_campaign_report_cli.R
Rscript fastkpc/tests/test_full_framework_smoke.R
```

Expected:

```text
Every test prints PASS.
```

- [ ] Run CPU/non-CUDA regression tests that should not be affected:

```bash
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
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
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); x <- validate_cuda_fastspline_residual_batch(); stopifnot(all(x$cases$status == "ok")); stopifnot(x$batch_diagnostics$true_batched_groups > 0); stopifnot(x$batch_diagnostics$single_fit_calls == 0); print(x$batch_diagnostics[c("groups","true_batched_groups","true_batched_fits","single_fit_calls")])'
```

Expected:

```text
No error.
Printed diagnostics show positive true_batched_groups and zero single_fit_calls.
```

- [ ] Run a compact validation campaign:

```bash
Rscript fastkpc/tools/run_validation_campaign.R \
  --engines cuda \
  --residual-backends fastSpline \
  --residual-devices cuda \
  --schedulers layer \
  --seeds 11 \
  --n-values 80 \
  --scenarios chain,additive \
  --legacy FALSE \
  --output-dir fastkpc/reports/true_batched_fastspline_smoke
```

Expected:

```text
Campaign completes.
Report directory contains true_batched_residuals.csv.
CSV has at least one row with cuda_residual_true_batched_groups > 0.
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
1. fastspline_residual_batch_cuda() returns batch_diagnostics with the documented fields.
2. Compatible grouped batch calls report true_batched_groups > 0 and single_fit_calls == 0.
3. Mixed-shape batch calls preserve original output order and report a correct group_table.
4. Standalone true-batched residuals match CPU residuals/fitted values within the documented tolerances.
5. Layer scheduler residual prefetch aggregates true-batch counters into scheduler_diagnostics$summary.
6. CUDA layer scheduler with residual_batch_size=0 matches residual_batch_size=1 graph outputs within tolerance.
7. Validation campaign and report writer produce true_batched_residuals.csv.
8. README and reports README document true-batch diagnostics and known limits.
9. All commands in Phase 10 pass in the local environment, except optional legacy package comparisons may report explicit missing-package diagnostics if pcalg/graph are unavailable.
10. kpcalg/R MD5 checks remain OK.
```

## Notes For Future Workers

Keep the first implementation conservative:

```text
Use existing host-side make_fastspline_design().
Batch only the linear algebra and final residual computation.
Keep the single-fit CUDA function as a reference and fallback path.
Use diagnostics to make every fallback visible.
Prefer correctness and clear grouping over a broad solver surface.
```

Do not hide unsupported matrix sizes behind silent single-fit CUDA loops. A group of size greater than one either runs through the true batched path or records an explicit CPU fallback/error according to `fallback`.
