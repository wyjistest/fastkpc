# Fast kPC Layer-Batched Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current ad hoc CUDA skeleton loop with a deterministic layer-batched scheduler that preplans each PC skeleton level, prefetches unique residual vectors, evaluates dCov in large CUDA batches, replays decisions in legacy order, and reports task/residual/batch diagnostics without changing graph results.

**Architecture:** Extract task enumeration and replay semantics into a focused scheduler layer used by the CUDA skeleton backend. The scheduler builds a complete per-level task plan from the stable adjacency snapshot, materializes residual vectors through the existing CPU/CUDA residual backends, packs dCov column batches, and replays p-values in original order so adjacency, sepsets, pMax, and `n.edgetests` remain equivalent to the existing CPU and CUDA reference paths. Public R APIs gain opt-in scheduler controls and report diagnostics, while legacy `kpcalg/R` files remain untouched.

**Tech Stack:** R 4.4.1, Rcpp/RcppArmadillo, C++17, CUDA 12.5, existing `fastkpc/src/skeleton_engine_cuda.cpp`, existing exact dCov CUDA backend, existing fastSpline CUDA residual device, base R validation/report tooling, local shell build scripts under `fastkpc/tools/`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-14-fast-kpc-layer-batched-scheduler-goal-execution.md: add a deterministic layer-batched CUDA skeleton scheduler with per-level task plans, unique residual prefetch/materialization, CUDA dCov batch packing, legacy-order replay, public scheduler controls, validation campaign/report/CLI diagnostics, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `780000`.

This goal is intentionally larger than the prior residual-device stage. It should run long enough to cover scheduler extraction, API plumbing, diagnostics, validation campaign changes, benchmark artifacts, documentation, and full regression verification.

Do not mark the goal complete until every criterion in "Completion Criteria" is satisfied. Mark the goal blocked only if the same local blocker prevents progress for three consecutive goal turns and no meaningful validation or implementation work remains possible.

## Current Baseline

The previous completed goal added:

```text
fastspline_residual_cuda()
fastspline_residual_batch_cuda()
residual_device = auto/cpu/cuda plumbing
CUDA skeleton residual cache support for fastSpline residual_device="cuda"
fast_kpc() public wrapper residual_device support
validation campaign residual_device diffs
report writer residual_device_diffs.csv
CLI --residual-device and --residual-devices
CUDA residual docs/tests/benchmarks
```

Final previous-stage validation showed:

```text
Standalone CUDA residual max_abs_residual_diff about 2.331468e-15
CUDA skeleton residual-device pMax max diff about 1.110223e-15
Final residual-device campaign 20/20 runs ok
Residual-device PDAG identical in all compared runs
max_residual_device_pmax_diff about 5.562217e-14
kpcalg/R MD5 all OK
```

Current implementation facts to preserve:

```text
fast_skeleton_cuda_backend(..., residual_backend = "fastSpline", residual_device = "cuda") works.
run_skeleton_cuda_batch() enumerates all level tasks speculatively, computes p-values in CUDA dCov batches, and replays them in deterministic order.
Residual vectors are currently requested inside fill_task_vectors() while each dCov batch is packed.
CudaSkeletonResidualCache computes each cache miss one at a time.
fit_fastspline_residuals_cuda_batch() exists at the R/native boundary, but its current C++ implementation loops over fit_fastspline_residuals_cuda() one fit at a time.
WAN-PDAG orientation residuals remain CPU-side.
Existing CPU skeleton is the numerical reference for adjacency/sepsets/pMax/n.edgetests.
```

Important consequence:

```text
The next bottleneck is not the dCov kernel alone. It is the scheduling shape around thousands of CI tests: task enumeration, repeated residual lookups, per-miss CUDA residual calls, small batch packing, and limited diagnostics about accepted vs speculative work.
```

## Scope

In scope:

- Add an explicit CUDA skeleton scheduler mode:

```r
scheduler = c("auto", "layer", "legacy")
```

- Keep the existing CUDA loop available as `scheduler = "legacy"` until validation proves the layer scheduler is equivalent.
- Make `scheduler = "auto"` resolve to:

```text
CUDA skeleton path: "layer"
CPU skeleton path: "legacy"
```

- Add per-level task planning with deterministic task IDs and task order.
- Add unique residual request collection before dCov packing for each level.
- Add residual prefetch/materialization for CUDA skeleton levels:

```text
linear residuals: CPU residual backend, same semantics as before
fastSpline + residual_device="cpu": CPU fastSpline residual backend
fastSpline + residual_device="cuda": existing CUDA fastSpline residual backend
fastSpline + residual_device="auto": same device resolution as previous goal
```

- Preserve existing residual cache statistics semantics:

```text
requests: task-level residual vector uses
hits: task-level uses served by an already materialized vector
misses: unique residual requests that required materialization
computations: unique residual fits actually run
stored_vectors: unique vectors retained by cache
stored_values: stored_vectors * n
```

- Add scheduler-specific diagnostics separate from residual cache stats:

```text
scheduler
scheduler_requested
levels
tasks_planned
tasks_evaluated
tests_replayed
tasks_ignored_after_delete
dcov_batches
dcov_batch_size_requested
dcov_batch_size_used
residual_requests
unique_residual_requests
residual_batches
residual_batch_size_requested
residual_batch_size_used
max_level_tasks
max_level_unique_residuals
```

- Add R validation helpers comparing:

```text
legacy CUDA scheduler vs layer CUDA scheduler
CPU exact skeleton vs layer CUDA scheduler
residual_device="cpu" vs residual_device="cuda"
batch_size=1 vs automatic dCov batch sizing
residual_batch_size=1 vs automatic residual materialization
```

- Extend public `fast_kpc()` with scheduler controls and result config fields.
- Extend validation campaign, report writer, and CLIs to include scheduler dimensions and scheduler diffs.
- Add benchmark helpers that quantify scheduler overhead and speculative work.
- Keep old tests passing.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not replace exported legacy `kpcalg::kpc()`.
- Do not modify any file under `kpcalg/R`.
- Do not implement multi-GPU scheduling.
- Do not move WAN-PDAG orientation residuals to CUDA.
- Do not change dCov gamma statistics or `legacy_index` semantics.
- Do not change the fastSpline basis, penalty, lambda grid, or GCV criterion.
- Do not require `pcalg` or `graph` for package-independent tests.
- Do not make true batched cuSOLVER fastSpline fitting a completion requirement for this goal. The layer scheduler must call the existing residual backend through a batch/materialization interface and expose diagnostics that make a later true-batched solver goal measurable.
- Do not initialize git if the workspace is not already a git repository.

## Design Contract

### Scheduler Modes

The public scheduler option has three accepted values:

```text
auto
layer
legacy
```

Resolution rules:

```text
1. scheduler="legacy" uses the current run_skeleton_cuda_batch() behavior with minimal internal refactoring.
2. scheduler="layer" uses the new per-level scheduler implementation.
3. scheduler="auto" resolves to "layer" for CUDA skeleton execution.
4. scheduler="auto" resolves to "legacy" for CPU skeleton execution because CPU scheduling is not being rewritten in this goal.
5. Existing calls that omit scheduler must remain valid.
6. Existing calls that pass batch_size must keep controlling dCov batch size.
```

Recommended R signatures after this goal:

```r
fast_skeleton_cuda_backend <- function(data, alpha, max_conditioning_size,
                                       residual_backend = "linear",
                                       residual_device = c("auto", "cpu", "cuda"),
                                       residual_cache = TRUE,
                                       index = 1,
                                       legacy_index = TRUE,
                                       batch_size = 0,
                                       residual_batch_size = 0,
                                       scheduler = c("auto", "layer", "legacy"),
                                       scheduler_diagnostics = TRUE,
                                       fastspline_params = list(),
                                       cuda_residual_fallback = TRUE)
```

```r
fast_kpc <- function(data,
                     alpha = 0.2,
                     max_conditioning_size = 2,
                     engine = c("auto", "cuda", "cpu"),
                     residual_backend = c("fastSpline", "linear"),
                     residual_device = c("auto", "cpu", "cuda"),
                     scheduler = c("auto", "layer", "legacy"),
                     graph_stage = c("wanpdag", "skeleton"),
                     residual_cache = TRUE,
                     index = 1,
                     legacy_index = TRUE,
                     batch_size = 0,
                     residual_batch_size = 0,
                     scheduler_diagnostics = TRUE,
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

The argument order should keep existing arguments stable as far as practical. New options may be inserted near `batch_size` because they control execution scheduling.

### Task Identity Contract

Each layer task must have a stable identity:

```cpp
struct LayerCiTask {
  int task_id;
  int level;
  int edge_x;
  int edge_y;
  int orientation_x;
  int orientation_y;
  std::vector<int> conditioning_set;
  int edge_key;
};
```

Rules:

```text
1. task_id is zero-based within the level and equals position in replay order.
2. level equals conditioning set size.
3. edge_x < edge_y for the undirected skeleton edge under test.
4. orientation_x/orientation_y preserve whether the task came from neighbors of x or neighbors of y.
5. conditioning_set values are zero-based C++ column indices.
6. edge_key is edge_x * p + edge_y.
7. Enumeration order must match the current CUDA scheduler order.
```

### Level Plan Contract

Each PC skeleton level produces:

```cpp
struct LayerPlan {
  int level;
  int p;
  std::vector<int> adjacency_snapshot;
  std::vector<LayerCiTask> tasks;
  int unconditional_tasks;
  int conditional_tasks;
  int unique_residual_requests;
};
```

Planning rules:

```text
1. The plan is built from the stable adjacency snapshot at the start of the level.
2. The plan enumerates both orientation directions exactly as the current CUDA implementation does.
3. The plan includes speculative tasks that may be ignored during replay after an edge deletion.
4. The plan must not inspect p-values or mutate graph state.
5. Empty levels are valid and must still record diagnostics.
```

### Residual Request Contract

Conditional tasks need two residual vectors, one for `orientation_x` and one for `orientation_y`.

Unique residual requests are keyed by:

```text
target column
conditioning set
n
p
residual backend name
residual backend params
resolved residual device
```

C++ sketch:

```cpp
struct LayerResidualRequest {
  int request_id;
  int target;
  std::vector<int> conditioning_set;
  std::string key;
};
```

Rules:

```text
1. conditioning_set must be normalized the same way ResidualCacheKey normalizes it.
2. A repeated task-level residual use increments requests and hits when materialized.
3. A first unique materialization increments misses and computations.
4. residual_cache=FALSE still computes correct values but does not reuse vectors across requests.
5. residual_cache=TRUE reuses vectors within and across levels when keys match.
6. The new scheduler may prefetch unique requests before dCov packing, but cache stats must remain interpretable at the task-use level.
```

### dCov Batch Contract

After residual vectors are materialized, each level is evaluated in dCov batches:

```text
X: n x batch matrix
Y: n x batch matrix
one column pair per LayerCiTask
```

Rules:

```text
1. Unconditional tasks use original data columns.
2. Conditional tasks use materialized residual vectors.
3. batch_size <= 0 means automatic dCov batch sizing.
4. batch_size = 1 must produce the same graph and pMax as automatic sizing.
5. dCov p-values are attached back to task_id positions.
6. Non-finite p-values must follow existing na_delete behavior.
```

### Replay Contract

Replay is the graph-semantic boundary. It must preserve observable behavior.

Replay rules:

```text
1. Iterate tasks in task_id order.
2. For each edge, accept tests until the first p >= alpha deletion event.
3. Ignore later speculative tasks for an edge once that edge is marked deleted for the current level.
4. Update pMax only from accepted replayed tests.
5. Increment n.edgetests only for accepted replayed tests.
6. Store sepset from the accepted deletion task exactly as the current CPU/CUDA reference does.
7. Apply all deletions after the level replay completes.
8. Record tasks_ignored_after_delete = tasks_evaluated - tests_replayed.
```

Required equality after replay:

```text
layer CUDA adjacency == legacy CUDA adjacency
layer CUDA sepsets == legacy CUDA sepsets
layer CUDA n.edgetests == legacy CUDA n.edgetests
max(abs(layer CUDA pMax - legacy CUDA pMax)) < 1e-8 for linear residuals
max(abs(layer CUDA pMax - legacy CUDA pMax)) < 1e-7 for fastSpline residuals
```

### Diagnostics Contract

Every CUDA skeleton result must include:

```text
scheduler
scheduler_requested
scheduler_diagnostics
```

`scheduler_diagnostics` must be a list with:

```text
summary
levels
batches
residuals
```

Minimum `summary` fields:

```text
scheduler
scheduler_requested
levels
tasks_planned
tasks_evaluated
tests_replayed
tasks_ignored_after_delete
dcov_batches
residual_requests
unique_residual_requests
residual_batches
max_level_tasks
max_level_unique_residuals
```

Minimum `levels` columns:

```text
level
tasks_planned
tasks_evaluated
tests_replayed
tasks_ignored_after_delete
deletions
unconditional_tasks
conditional_tasks
unique_residual_requests
dcov_batches
residual_batches
```

Minimum `batches` columns:

```text
level
batch_id
kind
start_task_id
task_count
n
status
```

`kind` must be one of:

```text
dcov
residual
```

Minimum `residuals` columns:

```text
level
request_id
target
conditioning_size
residual_backend
residual_device
materialized
fallback_used
reason
```

The legacy scheduler can return diagnostics with the same shape and fewer rows if exact per-request detail is not available, but it must still report `scheduler = "legacy"` and aggregate counts.

## File Structure

Create these files:

- `fastkpc/src/skeleton_task_scheduler.hpp`  
  Shared scheduler types, task planning declarations, diagnostics structs, and replay result structs.

- `fastkpc/src/skeleton_task_scheduler.cpp`  
  Deterministic task enumeration, residual request collection, replay helpers, and diagnostics aggregation that do not call CUDA directly.

- `fastkpc/R/scheduler_validation.R`  
  R helpers for layer-vs-legacy scheduler comparisons, scheduler benchmark runs, and scheduler diagnostic validation.

- `fastkpc/tests/test_layer_scheduler_task_plan.R`  
  Native/R-level checks for deterministic task planning and diagnostic schema.

- `fastkpc/tests/test_cuda_layer_scheduler_equivalence.R`  
  CUDA layer scheduler vs legacy CUDA and CPU skeleton graph equivalence.

- `fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R`  
  fastSpline residual_device CPU/CUDA comparisons with layer residual materialization diagnostics.

- `fastkpc/tests/test_fastkpc_scheduler_public_api.R`  
  Public `fast_kpc()` scheduler config/result contract tests.

- `fastkpc/tests/test_layer_scheduler_campaign_report_cli.R`  
  Validation campaign, report writer, and CLI tests for scheduler dimensions.

- `fastkpc/tests/test_layer_scheduler_benchmark.R`  
  Non-strict benchmark smoke test that records timings and speculative work counters.

- `fastkpc/tests/test_layer_scheduler_docs_contract.R`  
  README/report documentation contract test for scheduler options and artifacts.

Modify these files:

- `fastkpc/src/fastkpc_types.hpp`  
  Add scheduler options and scheduler diagnostic fields to `SkeletonOptions` and `SkeletonResult`.

- `fastkpc/src/skeleton_engine_cuda.hpp`  
  Add scheduler-aware function declarations or extend `run_skeleton_cuda_batch()`.

- `fastkpc/src/skeleton_engine_cuda.cpp`  
  Keep legacy execution available and add layer-batched execution path.

- `fastkpc/src/r_api_cuda.cpp`  
  Extend `.Call` entry points with scheduler/residual batch arguments and result conversion.

- `fastkpc/tools/build_cuda_native.sh`  
  Compile and link `skeleton_task_scheduler.cpp`.

- `fastkpc/R/cuda_native.R`  
  Add scheduler arguments to CUDA wrappers.

- `fastkpc/R/fast_kpc.R`  
  Add scheduler config, result contract fields, and print/summary exposure.

- `fastkpc/R/validation_campaign.R`  
  Add scheduler dimensions and scheduler diff tables.

- `fastkpc/R/report_writer.R`  
  Write scheduler diagnostics CSV artifacts and Markdown report sections.

- `fastkpc/tools/run_fast_kpc.R`  
  Add `--scheduler`, `--residual-batch-size`, and `--scheduler-diagnostics`.

- `fastkpc/tools/run_validation_campaign.R`  
  Add `--schedulers`, `--residual-batch-size`, and report scheduler diff artifacts.

- `fastkpc/README.md`  
  Document scheduler modes, diagnostics, validation commands, and known limits.

- `fastkpc/reports/README.md`  
  Document scheduler report artifacts.

Do not modify:

- `kpcalg/R/*.R`

## Phase 0: Baseline Audit And Guardrails

Purpose: verify the previous stage still passes before making scheduler changes.

- [ ] Run:

```bash
pwd
find docs/superpowers/plans -maxdepth 1 -type f | sort
find fastkpc/src -maxdepth 2 -type f | sort
find fastkpc/R -maxdepth 1 -type f | sort
find fastkpc/tests -maxdepth 1 -type f | sort
```

Expected:

```text
Working directory is /data/wenyujianData/kpcalg.
The previous CUDA residual goal document exists.
fastkpc/src/cuda/fastspline_residual_cuda.cu exists.
fastkpc/R/cuda_residual_validation.R exists.
No kpcalg/R edits are needed.
```

- [ ] Run:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
```

Expected:

```text
All listed tests print PASS.
CUDA native build succeeds.
CPU sourceCpp build succeeds.
```

- [ ] Run:

```bash
cd kpcalg
md5sum -c MD5 | rg '^R/'
cd ..
```

Expected:

```text
All kpcalg/R MD5 checks print OK.
```

If any baseline test fails, use systematic debugging. Do not implement scheduler changes until the baseline failure is understood and either fixed in fastkpc-owned files or documented as an unrelated environmental blocker.

## Phase 1: Add Scheduler Types And Option Fields

Purpose: introduce the scheduler vocabulary without changing behavior.

- [ ] Create `fastkpc/src/skeleton_task_scheduler.hpp` with focused structs.

Required public declarations:

```cpp
#ifndef FASTKPC_SKELETON_TASK_SCHEDULER_HPP
#define FASTKPC_SKELETON_TASK_SCHEDULER_HPP

#include <string>
#include <vector>

struct LayerCiTask {
  int task_id;
  int level;
  int edge_x;
  int edge_y;
  int orientation_x;
  int orientation_y;
  std::vector<int> conditioning_set;
  int edge_key;
};

struct LayerResidualRequest {
  int request_id;
  int target;
  std::vector<int> conditioning_set;
  std::string key;
};

struct LayerPlan {
  int level;
  int p;
  std::vector<int> adjacency_snapshot;
  std::vector<LayerCiTask> tasks;
  int unconditional_tasks;
  int conditional_tasks;
  int unique_residual_requests;
};

struct LayerDiagnosticsLevel {
  int level;
  int tasks_planned;
  int tasks_evaluated;
  int tests_replayed;
  int tasks_ignored_after_delete;
  int deletions;
  int unconditional_tasks;
  int conditional_tasks;
  int unique_residual_requests;
  int dcov_batches;
  int residual_batches;
};

struct SchedulerDiagnostics {
  std::string scheduler;
  std::string scheduler_requested;
  int levels;
  int tasks_planned;
  int tasks_evaluated;
  int tests_replayed;
  int tasks_ignored_after_delete;
  int dcov_batches;
  int residual_requests;
  int unique_residual_requests;
  int residual_batches;
  int max_level_tasks;
  int max_level_unique_residuals;
  std::vector<LayerDiagnosticsLevel> per_level;
};

LayerPlan make_layer_plan(const std::vector<int>& adjacency_snapshot,
                          int p,
                          int level);

std::vector<LayerResidualRequest> collect_unique_residual_requests(
  const LayerPlan& plan,
  int n,
  int p,
  const std::string& residual_backend,
  const std::string& residual_backend_params,
  const std::string& residual_device);

#endif
```

- [ ] Create `fastkpc/src/skeleton_task_scheduler.cpp` with deterministic enumeration copied from the current CUDA ordering.

Required behavior:

```text
make_layer_plan(snapshot, p, ord) must enumerate x-side combinations first, then y-side combinations, for each edge x < y.
collect_unique_residual_requests() must return stable request_id order by first use in task order.
```

- [ ] Modify `fastkpc/src/fastkpc_types.hpp`.

Add to `SkeletonOptions`:

```cpp
std::string scheduler_requested;
int residual_batch_size;
bool scheduler_diagnostics_enabled;
```

Add to `SkeletonResult`:

```cpp
std::string scheduler;
std::string scheduler_requested;
SchedulerDiagnostics scheduler_diagnostics;
```

Include:

```cpp
#include "skeleton_task_scheduler.hpp"
```

- [ ] Initialize the new fields in every existing `SkeletonOptions` construction in `fastkpc/src/r_api_cuda.cpp`.

Required default values:

```cpp
options.scheduler_requested = "legacy";
options.residual_batch_size = 0;
options.scheduler_diagnostics_enabled = true;
```

- [ ] Update `fastkpc/tools/build_cuda_native.sh` to compile and link `skeleton_task_scheduler.cpp`.

Add compile line near other C++ sources:

```sh
"$CXX" $COMMON_CXX -c "$ROOT/src/skeleton_task_scheduler.cpp" -o "$BUILD/skeleton_task_scheduler.o"
```

Add link input:

```sh
"$BUILD/skeleton_task_scheduler.o" \
```

- [ ] Build to catch type integration errors:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Expected:

```text
built: /data/wenyujianData/kpcalg/fastkpc/build/fastkpc_cuda.so
```

## Phase 2: Expose Scheduler Diagnostics Without Behavior Change

Purpose: make the legacy CUDA path report scheduler fields before replacing execution.

- [ ] Modify `fastkpc/src/r_api_cuda.cpp` result conversion.

Add fields to `skeleton_result_to_list()`:

```cpp
Rcpp::Named("scheduler") = result.scheduler,
Rcpp::Named("scheduler_requested") = result.scheduler_requested,
Rcpp::Named("scheduler_diagnostics") =
  scheduler_diagnostics_to_list(result.scheduler_diagnostics)
```

Add helper conversion functions:

```text
scheduler_diagnostics_summary_to_list()
scheduler_diagnostics_levels_to_data_frame()
scheduler_diagnostics_batches_to_data_frame()
scheduler_diagnostics_residuals_to_data_frame()
scheduler_diagnostics_to_list()
```

The `batches` and `residuals` data frames may be empty in this phase but must have the required columns.

- [ ] Modify `run_skeleton_cuda_batch()` legacy path to fill aggregate diagnostics.

Required legacy diagnostic values:

```text
scheduler = "legacy"
scheduler_requested = options.scheduler_requested
levels = max_order + 1
tasks_planned = total tasks enumerated across levels
tasks_evaluated = total tasks passed to dcov_batch_cuda()
tests_replayed = sum(result.n_edge_tests)
tasks_ignored_after_delete = tasks_evaluated - tests_replayed
dcov_batches = number of dcov_batch_cuda() calls
residual_requests = residual_cache.requests
unique_residual_requests = residual_cache.computations
residual_batches = residual_cache.computations
max_level_tasks = max tasks size over levels
max_level_unique_residuals = max residual computations over levels if available, else 0
```

- [ ] Add R wrapper arguments in `fastkpc/R/cuda_native.R` while keeping old calls valid:

```r
scheduler = c("auto", "layer", "legacy")
residual_batch_size = 0
scheduler_diagnostics = TRUE
```

For this phase, resolve `scheduler="auto"` to `"legacy"` before calling C++ so behavior is unchanged.

- [ ] Add `fastkpc/tests/test_layer_scheduler_task_plan.R`.

Minimum assertions:

```r
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/validation_scenarios.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

build_fastkpc_cuda_native(rebuild = TRUE)
scenario <- fastkpc_fixed_scenario()
result <- fast_skeleton_cuda_backend(
  scenario$data,
  alpha = scenario$alpha,
  max_conditioning_size = scenario$max_conditioning_size,
  scheduler = "legacy",
  scheduler_diagnostics = TRUE
)

diag <- result$scheduler_diagnostics
assert_true(result$scheduler == "legacy", "legacy scheduler should be recorded")
assert_true(result$scheduler_requested == "legacy", "requested scheduler should be recorded")
assert_true(is.list(diag), "scheduler diagnostics should be a list")
assert_true(is.list(diag$summary), "scheduler summary should be present")
assert_true(is.data.frame(diag$levels), "scheduler levels should be a data.frame")
assert_true(diag$summary$tasks_planned >= sum(result$n.edgetests),
            "planned tasks should cover replayed tests")
assert_true(diag$summary$tasks_ignored_after_delete ==
              diag$summary$tasks_evaluated - diag$summary$tests_replayed,
            "ignored count should match evaluated minus replayed")
cat("test_layer_scheduler_task_plan.R: PASS\n")
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_layer_scheduler_task_plan.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

Expected:

```text
All tests print PASS.
```

## Phase 3: Implement Layer Plan Enumeration And Replay

Purpose: make the scheduler own task planning and replay while still using existing residual fill and dCov evaluation mechanics.

- [ ] Add a new internal C++ execution function in `fastkpc/src/skeleton_engine_cuda.cpp`:

```cpp
SkeletonResult run_skeleton_cuda_layer_scheduler(const Rcpp::NumericMatrix& data,
                                                 const SkeletonOptions& options,
                                                 int dcov_batch_size);
```

- [ ] Refactor existing anonymous `SkeletonTask` usage so `LayerCiTask` is used for both planning and replay in the new path.

Keep the old legacy function available:

```cpp
SkeletonResult run_skeleton_cuda_legacy(const Rcpp::NumericMatrix& data,
                                        const SkeletonOptions& options,
                                        int batch_size);
```

Then make exported `run_skeleton_cuda_batch()` dispatch:

```cpp
if (resolved_scheduler == "layer") {
  return run_skeleton_cuda_layer_scheduler(data, options, batch_size);
}
return run_skeleton_cuda_legacy(data, options, batch_size);
```

- [ ] Implement replay in a helper that is independent of CUDA:

```cpp
void replay_layer_pvalues(const LayerPlan& plan,
                          const std::vector<double>& pvalues,
                          const SkeletonOptions& options,
                          int p,
                          std::vector<int>* delete_edges,
                          SkeletonResult* result,
                          int* tests_replayed,
                          std::vector<LevelDeletion>* level_log);
```

Replay must follow the contract above. It must update pMax and sepsets only for accepted replayed tests.

- [ ] Add a direct layer-vs-legacy test in `fastkpc/tests/test_cuda_layer_scheduler_equivalence.R`.

Use at least two scenarios:

```text
fastkpc_fixed_scenario()
a generated nonlinear 5-variable scenario with max_conditioning_size = 2
```

Required assertions:

```r
legacy <- fast_skeleton_cuda_backend(..., scheduler = "legacy")
layer <- fast_skeleton_cuda_backend(..., scheduler = "layer")
cpu <- fast_skeleton_cpp_backend(...)

identical(layer$adjacency, legacy$adjacency)
identical(layer$n.edgetests, legacy$n.edgetests)
compare_sepsets_exact(layer$sepsets, legacy$sepsets)
max(abs(layer$pMax - legacy$pMax)) < 1e-8
identical(layer$adjacency, cpu$adjacency)
identical(layer$n.edgetests, cpu$n.edgetests)
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
```

Expected:

```text
All tests print PASS.
```

## Phase 4: Add Unique Residual Prefetch For Layer Scheduler

Purpose: move residual materialization out of dCov batch packing and into a per-level prefetch step.

- [ ] Add a `LayerResidualStore` inside `fastkpc/src/skeleton_engine_cuda.cpp` or a focused helper file if the source becomes hard to read.

Required responsibilities:

```text
1. Accept LayerResidualRequest objects in stable order.
2. Materialize unique residual vectors.
3. Preserve task-level cache stats semantics.
4. Provide const vector access by target + conditioning set key.
5. Record residual diagnostic rows.
```

Suggested C++ shape:

```cpp
class LayerResidualStore {
 public:
  LayerResidualStore(const Rcpp::NumericMatrix& data,
                     const ResidualBackendConfig& backend,
                     const std::string& requested_device,
                     const std::string& resolved_device,
                     bool cache_enabled,
                     bool fallback,
                     int residual_batch_size);

  void register_task_use(int target, const std::vector<int>& conditioning_set);
  void prefetch_level(const std::vector<LayerResidualRequest>& requests);
  const std::vector<double>& get(int target,
                                 const std::vector<int>& conditioning_set) const;
  ResidualCacheStats stats() const;
  std::vector<ResidualDiagnosticRow> diagnostics() const;
};
```

The exact class name can differ, but the boundary must remain small and testable.

- [ ] Implement residual prefetch policy.

Rules:

```text
1. For unconditional tasks, no residual request is created.
2. For conditional tasks, register both orientation_x and orientation_y residual uses.
3. With residual_cache=TRUE, materialize each unique request at most once.
4. With residual_cache=FALSE, materialize per task use or emulate uncached semantics while retaining correct graph output.
5. residual_batch_size <= 0 means all unique residual requests for a level can be submitted as one materialization group.
6. residual_batch_size = 1 must be equivalent to automatic residual materialization.
7. If backend is fastSpline and resolved device is cuda, call `fit_fastspline_residuals_cuda_batch()` for prefetch groups.
8. If backend is fastSpline and resolved device is cpu, call the CPU backend for each unique request.
9. If backend is linear, use CPU linear residualization and record residual_device="cpu".
```

- [ ] Preserve previous device-resolution semantics.

Required behavior:

```text
residual_backend="linear", residual_device="cuda" -> residual_device="cpu" with reason "linear residual CUDA device is not implemented"
residual_backend="fastSpline", residual_device="cuda", fallback=TRUE -> cuda or cuda-fallback-cpu
residual_backend="fastSpline", residual_device="cpu" -> cpu
```

- [ ] Update layer dCov packing to read residuals from `LayerResidualStore`.

Required packing rules:

```text
For each task in the dCov batch:
  if conditioning_set.empty():
    x = original data column orientation_x
    y = original data column orientation_y
  else:
    x = residual_store.get(orientation_x, conditioning_set)
    y = residual_store.get(orientation_y, conditioning_set)
```

- [ ] Add `fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R`.

Minimum assertions:

```r
cpu_device <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cpu",
  scheduler = "layer",
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_auto_batch <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 0,
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_one_batch <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 1,
  residual_cache = TRUE,
  fastspline_params = params
)

identical(cuda_auto_batch$adjacency, cpu_device$adjacency)
identical(cuda_auto_batch$n.edgetests, cpu_device$n.edgetests)
max(abs(cuda_auto_batch$pMax - cpu_device$pMax)) < 1e-7
identical(cuda_one_batch$adjacency, cuda_auto_batch$adjacency)
max(abs(cuda_one_batch$pMax - cuda_auto_batch$pMax)) < 1e-8
cuda_auto_batch$scheduler == "layer"
cuda_auto_batch$scheduler_diagnostics$summary$unique_residual_requests > 0
cuda_auto_batch$scheduler_diagnostics$summary$residual_batches > 0
cuda_auto_batch$residual_cache$computations <=
  cuda_auto_batch$residual_cache$requests
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
```

Expected:

```text
All tests print PASS.
```

## Phase 5: Auto Batch Sizing And Scheduler Stats

Purpose: make automatic batch sizing explicit and inspectable.

- [ ] Add small helper functions in C++:

```cpp
int resolve_dcov_batch_size(int requested_batch_size,
                            int n,
                            int planned_tasks);

int resolve_residual_batch_size(int requested_residual_batch_size,
                                int unique_residual_requests);
```

Rules:

```text
1. requested > 0 returns requested.
2. requested <= 0 returns planned_tasks for dCov when planned_tasks > 0.
3. requested <= 0 returns unique_residual_requests for residual materialization when unique_residual_requests > 0.
4. Empty inputs return 1 for diagnostic consistency but do not run batches.
```

This is intentionally simple. Device-memory-aware sizing can be a later performance goal.

- [ ] Record batch diagnostic rows for every dCov batch.

Required `kind="dcov"` row values:

```text
level
batch_id
kind = "dcov"
start_task_id
task_count
n
status = "ok" or error reason
```

- [ ] Record residual diagnostic rows for every unique residual request.

Required fields:

```text
level
request_id
target
conditioning_size
residual_backend
residual_device
materialized
fallback_used
reason
```

- [ ] Validate aggregate identities in C++ before returning:

```text
tasks_ignored_after_delete == tasks_evaluated - tests_replayed
tasks_planned >= tasks_evaluated
tests_replayed == sum(n_edge_tests)
unique_residual_requests <= residual_requests when residual_cache=TRUE and conditional tasks exist
```

If an identity fails, throw a clear error with the identity name.

- [ ] Extend `test_layer_scheduler_task_plan.R` to validate:

```r
levels <- result$scheduler_diagnostics$levels
summary <- result$scheduler_diagnostics$summary
assert_true(sum(levels$tasks_evaluated) == summary$tasks_evaluated,
            "level evaluated counts should sum to summary")
assert_true(sum(levels$tests_replayed) == summary$tests_replayed,
            "level replay counts should sum to summary")
assert_true(summary$tests_replayed == sum(result$n.edgetests),
            "summary replayed tests should match n.edgetests")
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_layer_scheduler_task_plan.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
```

Expected:

```text
All tests print PASS.
```

## Phase 6: Public API And Result Contract

Purpose: make the scheduler controllable from the public wrapper and stable in result objects.

- [ ] Update `fastkpc/R/cuda_native.R`.

Required behavior:

```text
fast_skeleton_cuda()
fast_skeleton_cuda_cached()
fast_skeleton_cuda_backend()
fast_kpc_wanpdag_cuda()
```

must accept scheduler-related arguments where appropriate:

```r
scheduler = c("auto", "layer", "legacy")
residual_batch_size = 0
scheduler_diagnostics = TRUE
```

`fast_skeleton_cuda()` and `fast_skeleton_cuda_cached()` can default to `scheduler="legacy"` for backwards compatibility unless they explicitly expose the new argument.

- [ ] Update `.Call` signatures in `fastkpc/src/r_api_cuda.cpp`.

Required native inputs for backend and WAN-PDAG CUDA calls:

```text
scheduler
residual_batch_size
scheduler_diagnostics
```

Update `call_methods[]` arity exactly.

- [ ] Update `fastkpc/R/fast_kpc.R`.

Add config fields:

```text
scheduler_requested
scheduler_used
residual_batch_size
scheduler_diagnostics
```

Add result metrics:

```text
tasks_planned
tasks_evaluated
tests_replayed
tasks_ignored_after_delete
unique_residual_requests
dcov_batches
residual_batches
```

Update `fastkpc_result_summary()` and `print.fastkpc_result()` to include scheduler used.

- [ ] Ensure CPU engine behavior is explicit:

```text
engine="cpu", scheduler="layer" must not silently pretend CPU layer scheduling exists.
```

Acceptable behavior:

```text
Option A: error clearly with "layer scheduler is only implemented for CUDA skeleton execution"
Option B: resolve to legacy with diagnostic reason "CPU layer scheduler is not implemented"
```

Use Option B for public `fast_kpc(engine="cpu", scheduler="auto")`, and Option A for explicit `engine="cpu", scheduler="layer"` to avoid misleading users.

- [ ] Add `fastkpc/tests/test_fastkpc_scheduler_public_api.R`.

Minimum assertions:

```r
cuda_layer <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 0,
  graph_stage = "skeleton",
  fastspline_params = params
)

cuda_legacy <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "legacy",
  graph_stage = "skeleton",
  fastspline_params = params
)

cuda_layer$config$scheduler_requested == "layer"
cuda_layer$config$scheduler_used == "layer"
cuda_layer$skeleton$scheduler == "layer"
identical(cuda_layer$skeleton$adjacency, cuda_legacy$skeleton$adjacency)
max(abs(cuda_layer$skeleton$pMax - cuda_legacy$skeleton$pMax)) < 1e-7
inherits(cuda_layer, "fastkpc_result")
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_fastkpc_scheduler_public_api.R
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
```

Expected:

```text
All tests print PASS.
```

## Phase 7: Validation Campaign, Report Writer, And CLI

Purpose: make scheduler differences visible in the same report system used by earlier goals.

- [ ] Update `fastkpc/R/validation_campaign.R`.

Add argument:

```r
schedulers = c("auto")
```

Update run IDs to include scheduler:

```text
scenario-seed-n-engine-residual_backend-residual_device-scheduler
```

Add `scheduler` column to:

```text
runs
graph_metrics
timings
cache
orientation_counts
errors
```

Add a scheduler-diff table:

```r
fastkpc_campaign_scheduler_diffs <- function(results) { ... }
```

Minimum columns:

```text
scenario
seed
n
engine
residual_backend
residual_device
left_scheduler
right_scheduler
pdag_identical
skeleton_adjacency_identical
max_abs_pmax_diff
orientation_counts_identical
status
```

Compare at least:

```text
legacy vs layer for engine="cuda"
auto vs layer for engine="cuda" when both are present
```

- [ ] Update campaign summary fields:

```text
scheduler_diff_rows
scheduler_pdag_identical
max_scheduler_pmax_diff
layer_runs
legacy_runs
```

- [ ] Update `fastkpc/R/report_writer.R`.

Write new artifacts:

```text
scheduler_diffs.csv
scheduler_levels.csv
scheduler_batches.csv
scheduler_residuals.csv
```

Add Markdown sections:

```text
## Scheduler
## Scheduler Levels
## Scheduler Batches
## Scheduler Residuals
```

- [ ] Update CLI `fastkpc/tools/run_fast_kpc.R`.

Add flags:

```text
--scheduler
--residual-batch-size
--scheduler-diagnostics
```

The single-run CLI output RDS must contain config fields:

```text
scheduler_requested
scheduler_used
residual_batch_size
```

- [ ] Update CLI `fastkpc/tools/run_validation_campaign.R`.

Add flags:

```text
--schedulers
--residual-batch-size
--scheduler-diagnostics
```

Use comma-separated parser consistent with existing CLI list args.

- [ ] Add `fastkpc/tests/test_layer_scheduler_campaign_report_cli.R`.

Minimum test flow:

```r
campaign <- run_fastkpc_validation_campaign(
  scenarios = c("chain"),
  seeds = c(11),
  ns = c(80),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cuda"),
  schedulers = c("legacy", "layer"),
  max_conditioning_size = 2,
  fastspline_params = list(knots = 7, lambda_count = 13)
)

assert_true("scheduler_diffs" %in% names(campaign), "campaign should include scheduler_diffs")
assert_true(nrow(campaign$scheduler_diffs) >= 1L, "scheduler diffs should have rows")
assert_true(all(campaign$scheduler_diffs$skeleton_adjacency_identical),
            "scheduler comparison should preserve skeleton adjacency")

out <- tempfile("fastkpc-scheduler-report-")
artifacts <- write_fastkpc_validation_report(campaign, out)
assert_true(file.exists(file.path(out, "scheduler_diffs.csv")),
            "report should write scheduler_diffs.csv")
assert_true(file.exists(file.path(out, "scheduler_levels.csv")),
            "report should write scheduler_levels.csv")
```

Also run both CLIs in the test with small data and assert output files exist.

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_layer_scheduler_campaign_report_cli.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_cuda_residual_device_report_cli.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
```

Expected:

```text
All tests print PASS.
```

## Phase 8: Scheduler Benchmark And Speculative Work Accounting

Purpose: produce evidence about whether the scheduler is reducing overhead and where work remains speculative.

- [ ] Add benchmark helper in `fastkpc/R/scheduler_validation.R`:

```r
benchmark_layer_scheduler <- function(seed = 407,
                                      n = 500,
                                      p = 8,
                                      alpha = 0.2,
                                      max_conditioning_size = 2,
                                      residual_backend = "fastSpline",
                                      residual_device = "cuda",
                                      schedulers = c("legacy", "layer"),
                                      batch_sizes = c(1L, 0L),
                                      residual_batch_sizes = c(1L, 0L),
                                      fastspline_params = list(knots = 8,
                                                               lambda_count = 17,
                                                               ridge = 1e-8)) {
  ...
}
```

Return a list:

```text
runs
summary
graph_equal
```

Minimum `runs` columns:

```text
scheduler
batch_size
residual_batch_size
elapsed_sec
skeleton_edges
tasks_planned
tasks_evaluated
tests_replayed
tasks_ignored_after_delete
dcov_batches
unique_residual_requests
residual_batches
residual_cache_requests
residual_cache_computations
```

- [ ] Add a deterministic benchmark scenario generator inside the helper, not a new dependency.

Use smooth nonlinear variables with shared latent sources:

```r
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  sin(z1) + noise,
  cos(z1) + noise,
  z1 * z2 + noise,
  sin(z2) + noise,
  cos(z2) + noise,
  z1 + noise,
  z2 + noise,
  independent_noise
)
```

- [ ] Add `fastkpc/tests/test_layer_scheduler_benchmark.R`.

This test must not require a speedup threshold. It must assert:

```text
benchmark returns data frames
both legacy and layer scheduler rows are present
all runs completed
layer vs legacy graph_equal is TRUE
tasks_planned >= tests_replayed
dcov_batches > 0
```

- [ ] Run:

```bash
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_layer_scheduler_benchmark.R
```

Expected:

```text
test_layer_scheduler_benchmark.R: PASS
```

## Phase 9: Documentation And Contract Tests

Purpose: document the new scheduler in the developer-facing fastkpc docs.

- [ ] Update `fastkpc/README.md`.

Add sections:

```text
Layer-batched CUDA scheduler
Scheduler modes
Scheduler diagnostics
Residual prefetch semantics
Validation commands
Known limits
```

Required wording points:

```text
1. scheduler="layer" preserves legacy replay order.
2. scheduler="legacy" remains available as a reference path.
3. scheduler diagnostics distinguish planned/evaluated/replayed/ignored work.
4. residual_batch_size controls materialization grouping, not the statistical residual model.
5. residual_backend and residual_device keep their previous meanings.
6. WAN-PDAG orientation residuals remain CPU-side in this stage.
7. kpcalg/R files are not modified by fastkpc staged backend work.
```

- [ ] Update `fastkpc/reports/README.md`.

Document artifacts:

```text
scheduler_diffs.csv
scheduler_levels.csv
scheduler_batches.csv
scheduler_residuals.csv
```

- [ ] Add `fastkpc/tests/test_layer_scheduler_docs_contract.R`.

Minimum assertions:

```r
readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports <- paste(readLines("fastkpc/reports/README.md", warn = FALSE), collapse = "\n")

must <- c(
  "Layer-batched CUDA scheduler",
  "scheduler=\"layer\"",
  "scheduler=\"legacy\"",
  "residual_batch_size",
  "scheduler_diagnostics",
  "planned",
  "replayed",
  "WAN-PDAG orientation residuals remain CPU-side"
)

for (needle in must) {
  if (!grepl(needle, readme, fixed = TRUE)) {
    stop("README missing: ", needle, call. = FALSE)
  }
}

for (needle in c("scheduler_diffs.csv", "scheduler_levels.csv",
                 "scheduler_batches.csv", "scheduler_residuals.csv")) {
  if (!grepl(needle, reports, fixed = TRUE)) {
    stop("reports README missing: ", needle, call. = FALSE)
  }
}

cat("test_layer_scheduler_docs_contract.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_layer_scheduler_docs_contract.R
```

Expected:

```text
test_layer_scheduler_docs_contract.R: PASS
```

## Phase 10: Full Regression Verification

Purpose: prove the stage is complete and did not regress previous work.

- [ ] Run the CUDA and CPU builds from clean state:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

Expected:

```text
CUDA build succeeds.
CPU sourceCpp build succeeds.
```

- [ ] Run old core tests:

```bash
set -e
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_regrvonps_native.R
Rscript fastkpc/tests/test_orientation_matrix.R
Rscript fastkpc/tests/test_orientation_rules.R
Rscript fastkpc/tests/test_wanpdag_engine_core.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
```

Expected:

```text
All tests print PASS.
```

- [ ] Run old CUDA tests:

```bash
set -e
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_kernel.R
Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_fastkpc_residual_device_public_api.R
Rscript fastkpc/tests/test_cuda_residual_device_campaign.R
Rscript fastkpc/tests/test_cuda_residual_device_report_cli.R
Rscript fastkpc/tests/test_cuda_residual_benchmark.R
Rscript fastkpc/tests/test_cuda_residual_docs_contract.R
```

Expected:

```text
All tests print PASS.
```

- [ ] Run public wrapper/report tests:

```bash
set -e
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
if [ -f fastkpc/tests/test_fastkpc_public_wrapper_validation.R ]; then
  Rscript fastkpc/tests/test_fastkpc_public_wrapper_validation.R
fi
Rscript fastkpc/tests/test_validation_scenarios.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
Rscript fastkpc/tests/test_fastkpc_legacy_diagnostics.R
```

Expected:

```text
All existing tests that are present print PASS.
The optional public wrapper validation test is skipped only when that file is absent in this workspace.
```

- [ ] Run new scheduler tests:

```bash
set -e
Rscript fastkpc/tests/test_layer_scheduler_task_plan.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_equivalence.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_residual_prefetch.R
Rscript fastkpc/tests/test_fastkpc_scheduler_public_api.R
Rscript fastkpc/tests/test_layer_scheduler_campaign_report_cli.R
Rscript fastkpc/tests/test_layer_scheduler_benchmark.R
Rscript fastkpc/tests/test_layer_scheduler_docs_contract.R
```

Expected:

```text
All new tests print PASS.
```

- [ ] Run a final scheduler validation campaign from R:

```bash
Rscript - <<'RS'
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

campaign <- run_fastkpc_validation_campaign(
  scenarios = c("chain", "fork"),
  seeds = c(101, 202),
  ns = c(80, 120),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cpu", "cuda"),
  schedulers = c("legacy", "layer"),
  max_conditioning_size = 2,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

summary <- campaign$summary
print(summary)

if (!identical(summary$error_runs, 0L)) {
  stop("campaign has error runs", call. = FALSE)
}
if (summary$scheduler_diff_rows < 4L) {
  stop("scheduler diff rows unexpectedly low", call. = FALSE)
}
if (summary$scheduler_pdag_identical < 4L) {
  stop("scheduler PDAG comparisons should be identical", call. = FALSE)
}
if (is.finite(summary$max_scheduler_pmax_diff) &&
    summary$max_scheduler_pmax_diff > 1e-7) {
  stop("scheduler pMax diff too large", call. = FALSE)
}

out <- tempfile("fastkpc-layer-scheduler-final-report-")
artifacts <- write_fastkpc_validation_report(campaign, out)
print(artifacts)

required <- c("summary.md", "campaign.rds", "scheduler_diffs.csv",
              "scheduler_levels.csv", "scheduler_batches.csv",
              "scheduler_residuals.csv")
missing <- required[!file.exists(file.path(out, required))]
if (length(missing) > 0L) {
  stop("missing report artifact: ", paste(missing, collapse = ", "),
       call. = FALSE)
}
RS
```

Expected:

```text
No errors.
summary$error_runs is 0.
scheduler PDAG comparisons are identical.
Report artifact paths are printed.
```

- [ ] Run final `kpcalg/R` guard:

```bash
cd kpcalg
md5sum -c MD5 | rg '^R/'
cd ..
```

Expected:

```text
Every printed kpcalg/R line ends with OK.
```

## Completion Criteria

This goal is complete only when all of the following are true:

```text
1. scheduler="layer" exists for CUDA skeleton and WAN-PDAG public paths.
2. scheduler="legacy" remains available as a reference path.
3. scheduler="auto" resolves predictably and is recorded in result config.
4. Layer scheduler graph outputs match legacy CUDA outputs:
   adjacency identical
   sepsets identical
   n.edgetests identical
   pMax max absolute diff below tolerance
5. Layer scheduler graph outputs match CPU exact skeleton outputs on validation scenarios.
6. fastSpline residual_device="cuda" works under layer scheduling.
7. residual_batch_size=1 and residual_batch_size=0 are graph-equivalent.
8. Existing residual cache stats remain interpretable and old residual cache tests pass.
9. Scheduler diagnostics appear in skeleton results, fast_kpc results, validation campaign objects, report CSVs, and Markdown.
10. CLI tools accept scheduler arguments and produce scheduler artifacts.
11. All old tests listed in Phase 10 pass.
12. All new scheduler tests pass.
13. Final scheduler validation campaign completes with zero error runs.
14. kpcalg/R MD5 checks pass.
```

## Expected Final Status Summary

When the goal is finished, report these items to the user:

```text
1. The new plan file executed.
2. New scheduler modes and defaults.
3. Whether layer scheduler is now the CUDA auto path.
4. Final layer-vs-legacy max pMax diff.
5. Final scheduler campaign counts:
   total runs
   ok runs
   error runs
   scheduler diff rows
   scheduler PDAG identical count
   max scheduler pMax diff
6. Final report artifact directory.
7. kpcalg/R MD5 status.
8. Any performance observations from benchmark_layer_scheduler().
```

Do not claim speedup unless benchmark results support it. It is acceptable for this stage to be a correctness and observability stage if timings show the current `fit_fastspline_residuals_cuda_batch()` loop remains the dominant cost.

## Notes For The Executing Agent

- Prefer small, focused edits. The easiest way to break this stage is to combine enumeration, residual materialization, dCov packing, and replay into one large block.
- Treat replay as sacred. If replay changes, graph outputs can change even when p-values are numerically identical.
- Keep scheduler diagnostics additive. Existing user-facing result fields should not disappear.
- If C++ source organization becomes awkward, split helper code into a small file rather than making `skeleton_engine_cuda.cpp` much larger.
- If `Rcpp::sourceCpp()` has trouble linking a new C++ file for CPU-only builds, keep scheduler code in the CUDA native build path first. The CPU engine does not need to use the new scheduler in this goal.
- The current workspace may not be a git repository. Do not initialize git unless explicitly instructed by the user.
