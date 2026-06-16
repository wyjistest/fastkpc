# Fast kPC Public Wrapper And Validation Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a stable opt-in public `fast_kpc()` wrapper over the completed CPU/CUDA skeleton plus WAN-PDAG pipeline, then build a reproducible validation campaign and report layer that makes graph-level differences, cache behavior, timing, and legacy availability explicit.

**Architecture:** Keep all native C++/CUDA kernels, skeleton engines, residual backends, and WAN-PDAG orientation code stable. Add R-level orchestration, result normalization, scenario generation, campaign execution, report writing, and command-line tooling under `fastkpc/`; this stage packages the existing backend into a usable framework entry point without replacing `kpcalg::kpc()` or modifying `kpcalg/R/*.R`.

**Tech Stack:** R 4.4.1, existing `fastkpc/R/native.R`, `fastkpc/R/cuda_native.R`, `fastkpc/R/wanpdag_validation.R`, C++17/CUDA 12.5 backends already built by prior goals, base R data frames/lists for reports, optional `pcalg`/`graph`/`RSpectra` for legacy comparison diagnostics.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-14-fast-kpc-public-wrapper-validation-goal-execution.md: add a stable opt-in public fast_kpc() wrapper over the completed CPU/CUDA WAN-PDAG backend, implement reproducible validation campaign/report tooling, compare CPU/CUDA/residual-backend/legacy diagnostics, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `460000`.

This goal is intentionally large. It should run long enough to cover public API design, validation infrastructure, report generation, CLI tooling, documentation, and full regression verification. Do not mark the goal complete until every criterion in Phase 13 is satisfied.

## Current Baseline

The previous goal completed these capabilities:

```text
fast_kpc_wanpdag_cpp()
fast_orient_wanpdag_cpp()
fast_kpc_wanpdag_cuda()
native orientation matrix/rules/regrVonPS/WAN-PDAG engine
wanpdag validation helpers
CPU-vs-CUDA WAN-PDAG equality checks
WAN-PDAG benchmark helpers
README WAN-PDAG docs
```

The final verification from that goal passed:

```text
All existing exact dCov, skeleton, CUDA, residual cache, fastSpline tests
All WAN-PDAG tests
CUDA build script
CPU sourceCpp build
validate_wanpdag_against_legacy()
compare_wanpdag_cpu_cuda()
benchmark_wanpdag_pipelines()
kpcalg/R MD5 checks
```

Known environment state at the time this plan was written:

```text
pcalg unavailable
graph unavailable
CUDA available
Rcpp/RcppArmadillo/mgcv available
workspace is not necessarily a git repository
```

If `pcalg` or `graph` remains unavailable, the new validation campaign must preserve explicit missing-package diagnostics and continue package-independent checks.

## Scope

In scope:

- Add a high-level public wrapper:

```r
fast_kpc(data,
         alpha = 0.2,
         max_conditioning_size = 2,
         engine = c("auto", "cuda", "cpu"),
         residual_backend = c("fastSpline", "linear"),
         graph_stage = c("wanpdag", "skeleton"),
         residual_cache = TRUE,
         index = 1,
         legacy_index = TRUE,
         batch_size = 0,
         orient_collider = TRUE,
         solve_confl = FALSE,
         rules = c(TRUE, TRUE, TRUE),
         fastspline_params = list(),
         validate = FALSE,
         benchmark = FALSE,
         legacy = FALSE,
         labels = NULL,
         seed = NULL)
```

- Add a normalized result object with class `fastkpc_result`.
- Add print/summary helpers for `fastkpc_result`.
- Add stable result contract helpers:

```r
as_fastkpc_result()
validate_fastkpc_result()
fastkpc_result_summary()
fastkpc_graph_metrics()
fastkpc_extract_pdag()
fastkpc_extract_skeleton()
```

- Add deterministic scenario generation for validation.
- Add a multi-scenario validation campaign runner.
- Add CPU-vs-CUDA, linear-vs-fastSpline, repeated-run determinism, and legacy availability/diff sections.
- Add report writer that saves reproducible RDS/CSV/Markdown artifacts.
- Add command-line tools for smoke runs and validation campaign runs.
- Add documentation and docs-contract tests.
- Preserve all existing APIs and tests.
- Keep `kpcalg/R/*.R` unchanged.

Out of scope:

- Do not replace exported `kpcalg::kpc()`.
- Do not modify any file under `kpcalg/R`.
- Do not implement CUDA residual kernels.
- Do not implement multi-GPU scheduling.
- Do not add HSIC/permutation/cluster tests.
- Do not require legacy graph equality when `pcalg`/`graph` are missing.
- Do not hide graph differences caused by exact dCov, fastSpline-vs-mgcv, or orientation changes; report them.
- Do not create a package build system in this goal.
- Do not initialize git if the workspace is not a git repository.

## Design Contract

### Public Wrapper Contract

`fast_kpc()` must:

```text
1. Accept numeric matrix/data.frame input.
2. Convert data to a double matrix without changing row order.
3. Preserve or assign column labels.
4. Resolve engine:
   - "cpu" always uses fast_kpc_wanpdag_cpp() or fast_skeleton_cpp_backend().
   - "cuda" always uses fast_kpc_wanpdag_cuda() or fast_skeleton_cuda_backend().
   - "auto" uses CUDA when fastkpc_cuda_available() is TRUE, otherwise CPU.
5. Resolve residual_backend:
   - "fastSpline" default.
   - "linear" accepted.
6. Resolve graph_stage:
   - "wanpdag" returns skeleton and orientation.
   - "skeleton" returns skeleton only and orientation = NULL.
7. Return class c("fastkpc_result", "list").
8. Attach timing for total, skeleton, orientation, validation, benchmark when measured.
9. Attach config exactly enough to reproduce the run.
10. Attach diagnostics for CUDA availability, legacy availability, package versions, and backend params.
```

`fast_kpc()` must not call legacy `kpcalg::kpc()`.

### Result Contract

The result must contain:

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

Required `config` fields:

```text
alpha
max_conditioning_size
engine_requested
engine_used
residual_backend
graph_stage
residual_cache
index
legacy_index
batch_size
orient_collider
solve_confl
rules
fastspline_params
validate
benchmark
legacy
seed
```

Required `data_info` fields:

```text
n
p
labels
has_missing
all_finite
storage_mode
```

Required `metrics` fields:

```text
skeleton_edge_count
directed_edge_count
undirected_edge_count
bidirected_edge_count
orientation_event_count
generalized_orientation_count
max_pmax
min_nonzero_pmax
```

Required `timings` columns:

```text
stage
elapsed_sec
```

Required `cache` sections:

```text
skeleton
orientation
```

If a section does not apply, it must be a named list with zero counters rather than missing.

### Validation Campaign Contract

Add a campaign runner:

```r
run_fastkpc_validation_campaign(seeds = c(11, 12, 13),
                                n_values = c(80, 140),
                                scenarios = c("chain", "fork", "collider", "independent", "additive"),
                                engines = c("cpu", "cuda"),
                                residual_backends = c("linear", "fastSpline"),
                                alpha = 0.2,
                                max_conditioning_size = 2,
                                legacy = TRUE,
                                benchmark = TRUE,
                                output_dir = NULL)
```

Return contract:

```text
config
runs
graph_metrics
pairwise_diffs
cpu_cuda
linear_fastspline
legacy
timings
cache
orientation_counts
errors
artifacts
summary
```

`runs` data frame required columns:

```text
run_id
scenario
seed
n
p
engine
residual_backend
status
error_message
skeleton_edge_count
directed_edge_count
undirected_edge_count
bidirected_edge_count
elapsed_total_sec
```

`cpu_cuda` data frame required columns:

```text
scenario
seed
n
residual_backend
pdag_identical
skeleton_adjacency_identical
max_abs_pmax_diff
orientation_counts_identical
status
```

`legacy` data frame required columns:

```text
scenario
seed
n
available
reason_if_unavailable
native_engine
native_residual_backend
pdag_exact
directed_added
directed_removed
undirected_added
undirected_removed
max_abs_pdag_diff
status
```

### Report Contract

Add:

```r
write_fastkpc_validation_report(campaign, output_dir)
```

It must write:

```text
summary.md
runs.csv
graph_metrics.csv
pairwise_diffs.csv
cpu_cuda.csv
linear_fastspline.csv
legacy.csv
timings.csv
cache.csv
orientation_counts.csv
errors.csv
campaign.rds
```

The Markdown report must contain these headings:

```text
# fastkpc Validation Campaign
## Configuration
## Summary
## CPU vs CUDA
## Linear vs fastSpline
## Legacy Diagnostics
## Timings
## Cache
## Errors
## Reproduction
```

The report writer must create `output_dir` if absent and must not overwrite unrelated files.

### CLI Contract

Add tools:

```text
fastkpc/tools/run_fast_kpc.R
fastkpc/tools/run_validation_campaign.R
```

`run_fast_kpc.R` must support:

```text
--input path/to/data.csv
--output path/to/result.rds
--engine cpu|cuda|auto
--residual-backend linear|fastSpline
--alpha numeric
--max-conditioning-size integer
--graph-stage skeleton|wanpdag
```

`run_validation_campaign.R` must support:

```text
--output-dir path/to/report-dir
--seeds comma-separated integers
--n-values comma-separated integers
--scenarios comma-separated names
--engines comma-separated engine names
--residual-backends comma-separated backend names
--alpha numeric
--max-conditioning-size integer
--legacy TRUE|FALSE
```

No new non-base-R command-line parsing dependency is allowed in this goal. Implement a small local parser.

## File Structure

Create:

- `fastkpc/R/fast_kpc.R`  
  Public wrapper, result normalization, print/summary helpers, extractors, config normalization.

- `fastkpc/R/validation_scenarios.R`  
  Deterministic synthetic scenario generator and scenario metadata.

- `fastkpc/R/validation_campaign.R`  
  Campaign execution, pairwise comparisons, CPU/CUDA and backend diff tables.

- `fastkpc/R/report_writer.R`  
  CSV/RDS/Markdown report writer.

- `fastkpc/tools/run_fast_kpc.R`  
  CLI for one dataset.

- `fastkpc/tools/run_validation_campaign.R`  
  CLI for reproducible validation campaigns.

- `fastkpc/reports/README.md`  
  Static docs describing generated report artifacts.

- `fastkpc/tests/test_fast_kpc_public_api.R`
- `fastkpc/tests/test_fastkpc_result_contract.R`
- `fastkpc/tests/test_validation_scenarios.R`
- `fastkpc/tests/test_validation_campaign_smoke.R`
- `fastkpc/tests/test_report_writer.R`
- `fastkpc/tests/test_fastkpc_cli_tools.R`
- `fastkpc/tests/test_fastkpc_reproducibility.R`
- `fastkpc/tests/test_fastkpc_legacy_diagnostics.R`
- `fastkpc/tests/test_fastkpc_docs_contract.R`
- `fastkpc/tests/test_full_framework_smoke.R`

Modify:

- `fastkpc/README.md`  
  Add public wrapper, campaign, report, CLI, known limits.

- `fastkpc/R/wanpdag_validation.R`  
  Only if reusable diff helpers need tiny exports or compatibility fixes. Prefer not to move existing behavior.

Do not modify:

- `kpcalg/R/*.R`
- Native C++/CUDA sources unless an integration test exposes a real bug in existing wrappers.

## Phase 0: Baseline Verification

Purpose: prove the completed WAN-PDAG stage is green before wrapping it.

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

- [ ] Run current WAN-PDAG baseline:

```bash
set -e
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
Rscript fastkpc/tests/test_wanpdag_cuda_pipeline.R
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
Rscript fastkpc/tests/test_wanpdag_benchmark.R
Rscript fastkpc/tests/test_wanpdag_docs_contract.R
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

## Phase 1: Public Wrapper Red Test

Purpose: define the user-facing API and result shape before writing wrapper code.

- [ ] Create `fastkpc/tests/test_fast_kpc_public_api.R`.

Test content:

```r
source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

data <- cbind(
  x1 = seq(-pi, pi, length.out = 90),
  x2 = sin(seq(-pi, pi, length.out = 90)),
  x3 = cos(seq(-pi, pi, length.out = 90)),
  x4 = rnorm(90)
)

result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "fastSpline",
  graph_stage = "wanpdag",
  seed = 101
)

assert_true(inherits(result, "fastkpc_result"), "result should have fastkpc_result class")
assert_true(result$config$engine_used == "cpu", "engine_used should be cpu")
assert_true(result$config$residual_backend == "fastSpline", "residual backend should be fastSpline")
assert_true(is.list(result$skeleton), "skeleton should be present")
assert_true(is.list(result$orientation), "orientation should be present")
assert_true(is.integer(result$orientation$pdag), "orientation pdag should be integer")
assert_true(identical(dim(result$orientation$pdag), c(ncol(data), ncol(data))),
            "pdag dimension should match variable count")
assert_true(is.data.frame(result$timings), "timings should be a data.frame")
assert_true(all(c("stage", "elapsed_sec") %in% names(result$timings)),
            "timings should have stage and elapsed_sec")
assert_true(is.list(result$metrics), "metrics should be present")
assert_true(is.list(result$diagnostics), "diagnostics should be present")

skeleton_only <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "linear",
  graph_stage = "skeleton"
)
assert_true(is.null(skeleton_only$orientation), "skeleton graph_stage should not orient")
assert_true(skeleton_only$config$graph_stage == "skeleton", "graph_stage should be recorded")

cat("test_fast_kpc_public_api.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
```

Expected:

```text
The test fails because fastkpc/R/fast_kpc.R does not exist yet.
```

## Phase 2: Implement Public Wrapper

Purpose: add the smallest high-level wrapper that passes the API test while delegating to existing backend wrappers.

- [ ] Create `fastkpc/R/fast_kpc.R`.

Implementation requirements:

```r
source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

fastkpc_zero_cache <- function() {
  list(requests = 0L, hits = 0L, computations = 0L)
}

fastkpc_elapsed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed = max(elapsed, .Machine$double.eps))
}

fastkpc_normalize_data <- function(data, labels = NULL) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (is.null(labels)) {
    labels <- colnames(data)
    if (is.null(labels)) labels <- paste0("V", seq_len(ncol(data)))
  }
  colnames(data) <- labels
  list(
    data = data,
    info = list(
      n = nrow(data),
      p = ncol(data),
      labels = labels,
      has_missing = anyNA(data),
      all_finite = all(is.finite(data)),
      storage_mode = storage.mode(data)
    )
  )
}
```

Wrapper behavior:

```text
Use match.arg for engine, residual_backend, graph_stage.
For engine="auto", call fastkpc_cuda_available() inside tryCatch; choose "cuda" only on TRUE.
For graph_stage="wanpdag":
  - CPU: call fast_kpc_wanpdag_cpp().
  - CUDA: call fast_kpc_wanpdag_cuda().
For graph_stage="skeleton":
  - CPU: call fast_skeleton_cpp_backend().
  - CUDA: call fast_skeleton_cuda_backend().
Do not call legacy kpcalg functions.
```

Add:

```r
fastkpc_graph_metrics <- function(result)
as_fastkpc_result <- function(raw, config, data_info, elapsed_total_sec)
validate_fastkpc_result <- function(result)
fastkpc_result_summary <- function(result)
fastkpc_extract_pdag <- function(result)
fastkpc_extract_skeleton <- function(result)
print.fastkpc_result <- function(x, ...)
summary.fastkpc_result <- function(object, ...)
```

`validate_fastkpc_result()` must stop with clear messages:

```text
"fastkpc_result missing config"
"fastkpc_result missing skeleton"
"fastkpc_result pdag dimension mismatch"
"fastkpc_result timing table invalid"
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_wanpdag_cpu_pipeline.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 3: Result Contract Tests

Purpose: lock the normalized result object so later campaign/report code can depend on it.

- [ ] Create `fastkpc/tests/test_fastkpc_result_contract.R`.

Test content:

```r
source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

data <- cbind(
  a = seq(-2, 2, length.out = 100),
  b = sin(seq(-2, 2, length.out = 100)),
  c = rnorm(100),
  d = cos(seq(-2, 2, length.out = 100))
)

result <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
                   engine = "cpu", residual_backend = "fastSpline")

required_top <- c("config", "data_info", "engine", "skeleton", "orientation",
                  "metrics", "timings", "cache", "validation", "benchmark",
                  "diagnostics")
missing_top <- setdiff(required_top, names(result))
assert_true(length(missing_top) == 0L,
            paste("missing result fields:", paste(missing_top, collapse = ", ")))

required_config <- c("alpha", "max_conditioning_size", "engine_requested",
                     "engine_used", "residual_backend", "graph_stage",
                     "residual_cache", "index", "legacy_index", "batch_size",
                     "orient_collider", "solve_confl", "rules",
                     "fastspline_params", "validate", "benchmark", "legacy",
                     "seed")
assert_true(all(required_config %in% names(result$config)),
            "config should include all required fields")

required_metrics <- c("skeleton_edge_count", "directed_edge_count",
                      "undirected_edge_count", "bidirected_edge_count",
                      "orientation_event_count", "generalized_orientation_count",
                      "max_pmax", "min_nonzero_pmax")
assert_true(all(required_metrics %in% names(result$metrics)),
            "metrics should include all required fields")

assert_true(validate_fastkpc_result(result), "validate_fastkpc_result should return TRUE")
assert_true(is.character(capture.output(print(result))), "print method should produce text")
summary_value <- summary(result)
assert_true(is.list(summary_value), "summary should return a list")
assert_true(identical(fastkpc_extract_skeleton(result), result$skeleton),
            "skeleton extractor should return skeleton")
assert_true(identical(fastkpc_extract_pdag(result), result$orientation$pdag),
            "pdag extractor should return pdag")

bad <- result
bad$config <- NULL
err <- tryCatch(validate_fastkpc_result(bad), error = conditionMessage)
assert_true(grepl("missing config", err), "missing config should be rejected")

cat("test_fastkpc_result_contract.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_result_contract.R
```

Expected:

```text
The test initially fails because contract helpers are incomplete, then passes after implementing missing fields.
```

- [ ] If the test fails, complete `fastkpc/R/fast_kpc.R` so every required field is present and validated.

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_fast_kpc_public_api.R
```

Expected:

```text
Both tests print PASS.
```

## Phase 4: Validation Scenario Generator

Purpose: build deterministic graph/data scenarios for campaign testing.

- [ ] Create `fastkpc/tests/test_validation_scenarios.R`.

Test content:

```r
source("fastkpc/R/validation_scenarios.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

names <- fastkpc_scenario_names()
required <- c("chain", "fork", "collider", "independent", "additive")
assert_true(all(required %in% names), "required scenario names should exist")

for (scenario in required) {
  a <- generate_fastkpc_scenario(scenario = scenario, seed = 10, n = 80)
  b <- generate_fastkpc_scenario(scenario = scenario, seed = 10, n = 80)
  c <- generate_fastkpc_scenario(scenario = scenario, seed = 11, n = 80)
  assert_true(is.matrix(a$data), paste(scenario, "data should be matrix"))
  assert_true(nrow(a$data) == 80, paste(scenario, "nrow should match"))
  assert_true(ncol(a$data) >= 4, paste(scenario, "should have at least four variables"))
  assert_true(identical(a$data, b$data), paste(scenario, "same seed should reproduce"))
  assert_true(!identical(a$data, c$data), paste(scenario, "different seed should differ"))
  assert_true(is.matrix(a$truth$adjacency), paste(scenario, "truth adjacency should exist"))
  assert_true(identical(dim(a$truth$adjacency), c(ncol(a$data), ncol(a$data))),
              paste(scenario, "truth adjacency dimension should match"))
  assert_true(is.character(a$description), paste(scenario, "description should exist"))
}

err <- tryCatch(generate_fastkpc_scenario("not-a-scenario", seed = 1, n = 20),
                error = conditionMessage)
assert_true(grepl("Unknown fastkpc validation scenario", err),
            "unknown scenario should fail clearly")

cat("test_validation_scenarios.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_validation_scenarios.R
```

Expected:

```text
The test fails because fastkpc/R/validation_scenarios.R does not exist yet.
```

- [ ] Create `fastkpc/R/validation_scenarios.R`.

Required functions:

```r
fastkpc_scenario_names <- function()
generate_fastkpc_scenario <- function(scenario, seed, n)
```

Scenario definitions:

```text
chain:
  x1 = z + noise
  x2 = sin(x1) + noise
  x3 = x2^2 + noise
  x4 = independent noise
  truth edges: 1->2, 2->3

fork:
  x1 = z + noise
  x2 = sin(x1) + noise
  x3 = cos(x1) + noise
  x4 = independent noise
  truth edges: 1->2, 1->3

collider:
  x1 independent
  x2 independent
  x3 = sin(x1) + cos(x2) + noise
  x4 independent noise
  truth edges: 1->3, 2->3

independent:
  x1, x2, x3, x4 independent noise
  truth edges: none

additive:
  x1 = z1 + noise
  x2 = z2 + noise
  x3 = sin(x1) + cos(x2) + noise
  x4 = x3 + 0.2 * noise
  truth edges: 1->3, 2->3, 3->4
```

Return:

```r
list(
  name = scenario,
  seed = seed,
  n = n,
  data = data,
  truth = list(adjacency = truth, pdag = truth_pdag),
  description = description
)
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_validation_scenarios.R
```

Expected:

```text
test_validation_scenarios.R prints PASS.
```

## Phase 5: Campaign Runner Red Test

Purpose: define campaign output tables before implementation.

- [ ] Create `fastkpc/tests/test_validation_campaign_smoke.R`.

Test content:

```r
source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(21, 22),
  n_values = c(70),
  scenarios = c("chain", "independent"),
  engines = c("cpu", "cuda"),
  residual_backends = c("linear", "fastSpline"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = TRUE,
  benchmark = TRUE,
  output_dir = NULL
)

required <- c("config", "runs", "graph_metrics", "pairwise_diffs",
              "cpu_cuda", "linear_fastspline", "legacy", "timings", "cache",
              "orientation_counts", "errors", "artifacts", "summary")
missing <- setdiff(required, names(campaign))
assert_true(length(missing) == 0L,
            paste("campaign missing fields:", paste(missing, collapse = ", ")))

assert_true(is.data.frame(campaign$runs), "runs should be data.frame")
assert_true(nrow(campaign$runs) == 2L * 1L * 2L * 2L * 2L,
            "runs row count should equal seeds*n_values*scenarios*engines*backends")
assert_true(all(campaign$runs$status == "ok"), "all smoke campaign runs should be ok")
assert_true(is.data.frame(campaign$cpu_cuda), "cpu_cuda should be data.frame")
assert_true(all(campaign$cpu_cuda$max_abs_pmax_diff < 1e-8, na.rm = TRUE),
            "CPU-vs-CUDA pMax diff should be tiny")
assert_true(any(campaign$cpu_cuda$pdag_identical), "at least one CPU-vs-CUDA pdag should match")
assert_true(is.data.frame(campaign$legacy), "legacy should be data.frame")
if (!all(campaign$legacy$available)) {
  assert_true(any(grepl("pcalg|graph", campaign$legacy$reason_if_unavailable)),
              "legacy unavailable rows should mention pcalg/graph")
}
assert_true(is.list(campaign$summary), "summary should exist")
assert_true(campaign$summary$total_runs == nrow(campaign$runs),
            "summary total_runs should match runs")

cat("test_validation_campaign_smoke.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_validation_campaign_smoke.R
```

Expected:

```text
The test fails because fastkpc/R/validation_campaign.R does not exist yet.
```

## Phase 6: Implement Campaign Runner

Purpose: run many fastkpc configurations deterministically and produce comparison tables.

- [ ] Create `fastkpc/R/validation_campaign.R`.

At top:

```r
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_scenarios.R")
source("fastkpc/R/wanpdag_validation.R")
```

Required helpers:

```r
fastkpc_run_id <- function(scenario, seed, n, engine, residual_backend)
fastkpc_safe_run <- function(expr)
fastkpc_flatten_cache <- function(result, run_id)
fastkpc_flatten_timings <- function(result, run_id)
fastkpc_flatten_orientation_counts <- function(result, run_id)
fastkpc_campaign_pairwise_diffs <- function(results)
fastkpc_campaign_cpu_cuda <- function(results)
fastkpc_campaign_linear_fastspline <- function(results)
fastkpc_campaign_legacy <- function(results, scenarios, alpha, max_conditioning_size, legacy)
fastkpc_campaign_summary <- function(campaign)
run_fastkpc_validation_campaign <- function(...)
```

Implementation details:

```text
Use expand.grid over seeds, n_values, scenarios, engines, residual_backends.
For each row:
  - generate scenario data.
  - call fast_kpc(... graph_stage="wanpdag").
  - capture errors into status/error_message; do not stop the campaign.
  - store full result internally long enough to compute diffs.
  - emit compact rows into data frames.
For CPU/CUDA diffs:
  - match scenario, seed, n, residual_backend.
  - compare skeleton adjacency, pMax, pdag, orientation counts.
For linear/fastSpline diffs:
  - match scenario, seed, n, engine.
  - compare pdag matrices and edge summaries.
For legacy:
  - when legacy=FALSE, return available=FALSE and reason_if_unavailable="legacy disabled".
  - when packages are missing, return available=FALSE and missing package reason.
  - when packages are available, call validate_wanpdag_against_legacy() for CPU fastSpline rows.
```

Error handling:

```text
If one run fails, campaign$status for that run is "error" and error_message stores conditionMessage.
Campaign function itself only stops for invalid arguments, not for a failed backend run.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_validation_campaign_smoke.R
```

Expected:

```text
test_validation_campaign_smoke.R prints PASS.
```

- [ ] Run regression:

```bash
Rscript fastkpc/tests/test_fast_kpc_public_api.R
Rscript fastkpc/tests/test_fastkpc_result_contract.R
Rscript fastkpc/tests/test_validation_scenarios.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
```

Expected:

```text
All four tests print PASS.
```

## Phase 7: Reproducibility Contract

Purpose: guarantee repeated wrapper and campaign runs are stable for fixed inputs.

- [ ] Create `fastkpc/tests/test_fastkpc_reproducibility.R`.

Test content:

```r
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

data <- cbind(
  x1 = seq(-2, 2, length.out = 100),
  x2 = sin(seq(-2, 2, length.out = 100)),
  x3 = cos(seq(-2, 2, length.out = 100)),
  x4 = seq(-2, 2, length.out = 100)^2
)

a <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
              engine = "cpu", residual_backend = "fastSpline", seed = 9)
b <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1L,
              engine = "cpu", residual_backend = "fastSpline", seed = 9)

assert_true(identical(a$skeleton$adjacency, b$skeleton$adjacency),
            "skeleton adjacency should repeat exactly")
assert_true(identical(a$skeleton$sepsets, b$skeleton$sepsets),
            "sepsets should repeat exactly")
assert_true(max(abs(a$skeleton$pMax - b$skeleton$pMax)) == 0,
            "pMax should repeat exactly for same engine")
assert_true(identical(a$orientation$pdag, b$orientation$pdag),
            "pdag should repeat exactly")
assert_true(identical(a$orientation$counts, b$orientation$counts),
            "orientation counts should repeat exactly")

campaign_a <- run_fastkpc_validation_campaign(
  seeds = c(31),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)
campaign_b <- run_fastkpc_validation_campaign(
  seeds = c(31),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)

strip_elapsed <- function(campaign) {
  campaign$runs$elapsed_total_sec <- 0
  campaign$timings$elapsed_sec <- 0
  campaign
}
assert_true(identical(strip_elapsed(campaign_a)$runs, strip_elapsed(campaign_b)$runs),
            "campaign run rows should repeat except timing")

cat("test_fastkpc_reproducibility.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_reproducibility.R
```

Expected:

```text
test_fastkpc_reproducibility.R prints PASS.
```

## Phase 8: Legacy Diagnostics Test

Purpose: make missing legacy packages and available legacy comparisons first-class campaign outputs.

- [ ] Create `fastkpc/tests/test_fastkpc_legacy_diagnostics.R`.

Test content:

```r
source("fastkpc/R/validation_campaign.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(41),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = TRUE
)

required <- c("scenario", "seed", "n", "available", "reason_if_unavailable",
              "native_engine", "native_residual_backend", "pdag_exact",
              "directed_added", "directed_removed", "undirected_added",
              "undirected_removed", "max_abs_pdag_diff", "status")
assert_true(all(required %in% names(campaign$legacy)),
            "legacy table should have required columns")

if (!all(campaign$legacy$available)) {
  assert_true(all(nzchar(campaign$legacy$reason_if_unavailable)),
              "unavailable legacy rows should have reasons")
  assert_true(any(grepl("pcalg|graph", campaign$legacy$reason_if_unavailable)),
              "missing package reason should mention pcalg or graph")
} else {
  assert_true(all(campaign$legacy$status == "ok"), "available legacy rows should be ok")
  assert_true(all(is.finite(campaign$legacy$max_abs_pdag_diff)),
              "available legacy rows should have finite diff")
}

disabled <- run_fastkpc_validation_campaign(
  seeds = c(41),
  n_values = c(70),
  scenarios = c("chain"),
  engines = c("cpu"),
  residual_backends = c("fastSpline"),
  legacy = FALSE
)
assert_true(all(disabled$legacy$available == FALSE), "legacy disabled should be unavailable")
assert_true(all(disabled$legacy$reason_if_unavailable == "legacy disabled"),
            "legacy disabled reason should be explicit")

cat("test_fastkpc_legacy_diagnostics.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_legacy_diagnostics.R
```

Expected:

```text
test_fastkpc_legacy_diagnostics.R prints PASS.
```

## Phase 9: Report Writer

Purpose: make campaign output persistent and easy to inspect.

- [ ] Create `fastkpc/tests/test_report_writer.R`.

Test content:

```r
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(51),
  n_values = c(60),
  scenarios = c("chain", "independent"),
  engines = c("cpu"),
  residual_backends = c("linear", "fastSpline"),
  legacy = TRUE
)

output_dir <- tempfile("fastkpc-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)

required_files <- c("summary.md", "runs.csv", "graph_metrics.csv",
                    "pairwise_diffs.csv", "cpu_cuda.csv",
                    "linear_fastspline.csv", "legacy.csv", "timings.csv",
                    "cache.csv", "orientation_counts.csv", "errors.csv",
                    "campaign.rds")
for (file in required_files) {
  path <- file.path(output_dir, file)
  assert_true(file.exists(path), paste(file, "should exist"))
  assert_true(file.info(path)$size > 0, paste(file, "should be non-empty"))
}

summary_text <- paste(readLines(file.path(output_dir, "summary.md"), warn = FALSE),
                      collapse = "\n")
required_headings <- c("# fastkpc Validation Campaign", "## Configuration",
                       "## Summary", "## CPU vs CUDA", "## Linear vs fastSpline",
                       "## Legacy Diagnostics", "## Timings", "## Cache",
                       "## Errors", "## Reproduction")
for (heading in required_headings) {
  assert_true(grepl(heading, summary_text, fixed = TRUE),
              paste("summary.md missing", heading))
}

loaded <- readRDS(file.path(output_dir, "campaign.rds"))
assert_true(is.list(loaded) && is.data.frame(loaded$runs), "campaign.rds should reload")
assert_true(is.list(artifacts) && length(artifacts) >= length(required_files),
            "writer should return artifact paths")

cat("test_report_writer.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_report_writer.R
```

Expected:

```text
The test fails because fastkpc/R/report_writer.R does not exist yet.
```

- [ ] Create `fastkpc/R/report_writer.R`.

Required functions:

```r
source("fastkpc/R/validation_campaign.R")

fastkpc_write_csv <- function(x, path)
fastkpc_markdown_table <- function(df, max_rows = 12)
fastkpc_campaign_markdown <- function(campaign)
write_fastkpc_validation_report <- function(campaign, output_dir)
```

Implementation rules:

```text
Use utils::write.csv(row.names = FALSE).
Use saveRDS for campaign.rds.
For empty data frames, write CSV with headers and zero rows.
For Markdown, include compact tables using base R text formatting.
Do not require knitr, rmarkdown, data.table, or tidyverse.
Return a named list of artifact paths.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_report_writer.R
```

Expected:

```text
test_report_writer.R prints PASS.
```

## Phase 10: CLI Tools

Purpose: let long validation runs be started without writing R code.

- [ ] Create `fastkpc/tests/test_fastkpc_cli_tools.R`.

Test content:

```r
assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

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
    "--engine", "cpu",
    "--residual-backend", "fastSpline",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--graph-stage", "wanpdag")
)
assert_true(identical(status_one, 0L), "run_fast_kpc.R should exit 0")
assert_true(file.exists(output), "run_fast_kpc.R should write result RDS")
result <- readRDS(output)
assert_true(inherits(result, "fastkpc_result"), "CLI result should be fastkpc_result")

report_dir <- tempfile("fastkpc-cli-report-")
status_campaign <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", report_dir,
    "--seeds", "61",
    "--n-values", "60",
    "--scenarios", "chain,independent",
    "--engines", "cpu",
    "--residual-backends", "linear,fastSpline",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--legacy", "TRUE")
)
assert_true(identical(status_campaign, 0L), "run_validation_campaign.R should exit 0")
assert_true(file.exists(file.path(report_dir, "summary.md")),
            "campaign CLI should write summary.md")
assert_true(file.exists(file.path(report_dir, "campaign.rds")),
            "campaign CLI should write campaign.rds")

cat("test_fastkpc_cli_tools.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
```

Expected:

```text
The test fails because CLI tools do not exist yet.
```

- [ ] Create `fastkpc/tools/run_fast_kpc.R`.

Required implementation:

```r
#!/usr/bin/env Rscript
source("fastkpc/R/fast_kpc.R")

parse_args <- function(args) {
  out <- list()
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) stop("Invalid argument: ", key, call. = FALSE)
    name <- sub("^--", "", key)
    if (i == length(args)) stop("Missing value for ", key, call. = FALSE)
    out[[name]] <- args[[i + 1L]]
    i <- i + 2L
  }
  out
}
```

Required behavior:

```text
Read CSV with utils::read.csv(check.names = FALSE).
Require --input and --output.
Default engine=auto, residual-backend=fastSpline, alpha=0.2, max-conditioning-size=2, graph-stage=wanpdag.
Call fast_kpc().
saveRDS(result, output).
Print one line: "wrote: <output>".
Exit non-zero on invalid arguments.
```

- [ ] Create `fastkpc/tools/run_validation_campaign.R`.

Required behavior:

```text
Use the same parse_args pattern.
Require --output-dir.
Default seeds=11, n-values=80, scenarios=chain, engines=cpu, residual-backends=fastSpline.
Parse comma-separated values using strsplit(..., fixed=TRUE).
Call run_fastkpc_validation_campaign().
Call write_fastkpc_validation_report().
Print one line: "wrote report: <output_dir>".
Exit non-zero on invalid arguments.
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
```

Expected:

```text
test_fastkpc_cli_tools.R prints PASS.
```

## Phase 11: Framework Smoke Test

Purpose: run the public wrapper and campaign in one small end-to-end check.

- [ ] Create `fastkpc/tests/test_full_framework_smoke.R`.

Test content:

```r
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

scenario <- generate_fastkpc_scenario("additive", seed = 71, n = 90)
single <- fast_kpc(
  scenario$data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "auto",
  residual_backend = "fastSpline",
  validate = TRUE,
  benchmark = TRUE,
  legacy = TRUE,
  seed = 71
)

assert_true(inherits(single, "fastkpc_result"), "single run should return fastkpc_result")
assert_true(single$config$engine_used %in% c("cpu", "cuda"), "engine_used should be concrete")
assert_true(is.list(single$validation), "validation section should exist")
assert_true(is.list(single$benchmark), "benchmark section should exist")

campaign <- run_fastkpc_validation_campaign(
  seeds = c(71),
  n_values = c(70),
  scenarios = c("chain", "fork", "collider"),
  engines = c("cpu", "cuda"),
  residual_backends = c("fastSpline"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = TRUE,
  benchmark = TRUE
)

assert_true(all(campaign$runs$status == "ok"), "framework smoke campaign runs should be ok")
assert_true(any(campaign$cpu_cuda$pdag_identical), "some CPU/CUDA pdag rows should match")
output_dir <- tempfile("fastkpc-framework-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(artifacts$summary_md), "framework report summary should exist")

cat("test_full_framework_smoke.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_full_framework_smoke.R
```

Expected:

```text
test_full_framework_smoke.R prints PASS.
```

## Phase 12: Documentation

Purpose: document the public API and campaign workflow.

- [ ] Create `fastkpc/tests/test_fastkpc_docs_contract.R`.

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
  "Public fast_kpc API",
  "fast_kpc(",
  "fastkpc_result",
  "Validation Campaign",
  "run_fastkpc_validation_campaign",
  "Validation Reports",
  "write_fastkpc_validation_report",
  "Command Line Tools",
  "run_fast_kpc.R",
  "run_validation_campaign.R",
  "kpcalg::kpc() is not replaced",
  "kpcalg/R/*.R files are not modified"
)
for (pattern in required_readme) assert_contains(readme, pattern)

required_reports <- c(
  "fastkpc reports",
  "summary.md",
  "runs.csv",
  "cpu_cuda.csv",
  "legacy.csv",
  "campaign.rds"
)
for (pattern in required_reports) assert_contains(reports_readme, pattern)

cat("test_fastkpc_docs_contract.R: PASS\n")
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
```

Expected:

```text
The test fails until README and reports README are updated.
```

- [ ] Update `fastkpc/README.md`.

Add sections:

```text
Public fast_kpc API
fastkpc_result
Validation Campaign
Validation Reports
Command Line Tools
Public Wrapper Known Limits
```

Required known-limit statements:

```text
kpcalg::kpc() is not replaced.
kpcalg/R/*.R files are not modified.
CUDA residual kernels are not implemented.
Validation campaign reports graph differences; it does not force equality.
Legacy comparison requires pcalg and graph.
```

- [ ] Create `fastkpc/reports/README.md`.

Required content:

```markdown
# fastkpc reports

Generated validation campaign reports are written here or to a user-specified
output directory. Report directories contain:

- `summary.md`
- `runs.csv`
- `graph_metrics.csv`
- `pairwise_diffs.csv`
- `cpu_cuda.csv`
- `linear_fastspline.csv`
- `legacy.csv`
- `timings.csv`
- `cache.csv`
- `orientation_counts.csv`
- `errors.csv`
- `campaign.rds`
```

- [ ] Run:

```bash
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
```

Expected:

```text
test_fastkpc_docs_contract.R prints PASS.
```

## Phase 13: Completion Criteria

The goal is complete only when all criteria are true:

```text
1. Public wrapper exists:
   - fastkpc/R/fast_kpc.R
   - fast_kpc()
   - class fastkpc_result
   - print/summary/extractor/validation helpers

2. Validation campaign exists:
   - fastkpc/R/validation_scenarios.R
   - fastkpc/R/validation_campaign.R
   - deterministic scenarios
   - run_fastkpc_validation_campaign()
   - CPU/CUDA table
   - linear/fastSpline table
   - legacy diagnostics table

3. Report writer exists:
   - fastkpc/R/report_writer.R
   - write_fastkpc_validation_report()
   - summary.md and CSV/RDS artifacts

4. CLI tools exist and pass smoke tests:
   - fastkpc/tools/run_fast_kpc.R
   - fastkpc/tools/run_validation_campaign.R

5. All new tests pass:
   - fastkpc/tests/test_fast_kpc_public_api.R
   - fastkpc/tests/test_fastkpc_result_contract.R
   - fastkpc/tests/test_validation_scenarios.R
   - fastkpc/tests/test_validation_campaign_smoke.R
   - fastkpc/tests/test_report_writer.R
   - fastkpc/tests/test_fastkpc_cli_tools.R
   - fastkpc/tests/test_fastkpc_reproducibility.R
   - fastkpc/tests/test_fastkpc_legacy_diagnostics.R
   - fastkpc/tests/test_fastkpc_docs_contract.R
   - fastkpc/tests/test_full_framework_smoke.R

6. Prior WAN-PDAG tests still pass:
   - fastkpc/tests/test_wanpdag_engine_core.R
   - fastkpc/tests/test_wanpdag_cpu_pipeline.R
   - fastkpc/tests/test_wanpdag_cuda_pipeline.R
   - fastkpc/tests/test_wanpdag_legacy_validation.R
   - fastkpc/tests/test_wanpdag_benchmark.R
   - fastkpc/tests/test_wanpdag_docs_contract.R

7. Prior fastSpline/CUDA/exact tests still pass.

8. Report writer produces all required files in a temp output directory.

9. CLI tools exit 0 on smoke inputs.

10. CPU-vs-CUDA campaign rows report max_abs_pmax_diff < 1e-8 for successful matched rows.

11. Legacy diagnostics report available=FALSE with explicit pcalg/graph reason when those packages are missing, or available=TRUE with diff metrics when installed.

12. README and reports README document the new public API and campaign workflow.

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
Rscript fastkpc/tests/test_validation_scenarios.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_report_writer.R
Rscript fastkpc/tests/test_fastkpc_cli_tools.R
Rscript fastkpc/tests/test_fastkpc_reproducibility.R
Rscript fastkpc/tests/test_fastkpc_legacy_diagnostics.R
Rscript fastkpc/tests/test_fastkpc_docs_contract.R
Rscript fastkpc/tests/test_full_framework_smoke.R
Rscript -e 'source("fastkpc/R/validation_campaign.R"); source("fastkpc/R/report_writer.R"); c <- run_fastkpc_validation_campaign(seeds=c(101,102), n_values=c(80), scenarios=c("chain","fork","collider","independent","additive"), engines=c("cpu","cuda"), residual_backends=c("linear","fastSpline"), legacy=TRUE, benchmark=TRUE); print(c$summary); print(c$cpu_cuda); d <- tempfile("fastkpc-final-report-"); a <- write_fastkpc_validation_report(c, d); print(a); stopifnot(file.exists(file.path(d, "summary.md"))); stopifnot(file.exists(file.path(d, "campaign.rds")))'
cd kpcalg
md5sum -c MD5 | rg '^R/'
```

When marking this goal complete, report:

```text
Exact build commands used.
Exact test commands run.
Pass/fail result for every new and existing test group.
fast_kpc() CPU/CUDA smoke result.
Campaign summary table.
CPU-vs-CUDA campaign max_abs_pmax_diff maximum.
Legacy diagnostics availability and reason.
Report artifact directory and files written.
CLI smoke status.
kpcalg/R MD5 result.
```

## Later Goals

Create separate goals only after this public wrapper and validation campaign goal is complete.

### Later Goal H: CUDA Residual Kernels

Objective:

```text
Move fastSpline basis evaluation, Gram formation, and batched small-system solves to CUDA after the public wrapper and validation campaign are stable.
```

### Later Goal I: Layer-Batched PC Scheduler

Objective:

```text
Replace one-test-at-a-time skeleton orchestration with a layer-batched scheduler that groups residual tasks and dCov tasks across candidate edges while preserving stable PC replay semantics.
```

### Later Goal J: Larger Reproducibility Report

Objective:

```text
Run a larger validation campaign across more seeds, n values, graph shapes, and residual backends, then write a versioned report under fastkpc/reports.
```

## Execution Rules For Codex

- Use TDD: write each listed test before implementing the corresponding code.
- Keep all new functionality under `fastkpc/`.
- Prefer R-level orchestration in this goal; avoid native C++/CUDA changes unless an existing backend bug blocks the public wrapper.
- Do not alter `kpcalg/R/*.R`.
- Keep `fast_kpc()` opt-in; do not replace `kpcalg::kpc()`.
- Keep reports deterministic except for elapsed time values.
- Treat missing `pcalg`/`graph` as a recorded diagnostic, not a hard failure for package-independent tests.
- Run CUDA tests serially because the local build artifacts are shared.
- Do not initialize git in this workspace unless the user asks.

## Final Handoff Prompt

After this plan is saved, the next Codex run can be started with:

```text
Create a goal with objective: "Build the next staged fast kPC backend slice from docs/superpowers/plans/2026-06-14-fast-kpc-public-wrapper-validation-goal-execution.md: add a stable opt-in public fast_kpc() wrapper over the completed CPU/CUDA WAN-PDAG backend, implement reproducible validation campaign/report tooling, compare CPU/CUDA/residual-backend/legacy diagnostics, and keep kpcalg/R files unchanged."
```

Then execute the plan with `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
