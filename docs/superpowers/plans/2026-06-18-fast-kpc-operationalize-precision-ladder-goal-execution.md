# Operationalize and Calibrate fastkpc Precision Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the implemented fastSplineCUDA / mgcvExtractGPU precision ladder into an operational backend policy with decision reports, timing attribution, routing rules, compatibility boundaries, and workload evidence.

**Architecture:** Keep `fastSplineCUDA` frozen as the high-throughput approximate primary backend. Treat `mgcvExtractGPU` as an mgcv setup anchored compatibility bridge and verifier, with honest diagnostics for fixed-sp, single-penalty GCV, and same-setup native batch paths. Build reporting, timing, routing, and compatibility envelopes around the existing campaigns before investing in new CUDA kernels.

**Tech Stack:** R campaign/report code under `fastkpc/R`, CLI runners under `fastkpc/tools`, native CUDA diagnostics from `fastkpc/src`, CSV/Markdown artifacts under `fastkpc/artifacts` or `fastkpc/reports`, tests as executable `Rscript fastkpc/tests/*.R`.

---

## Current Baseline

Checkpoint:

```text
Commit: 5d73cc6
Meaning:
    native same-setup mgcvExtractGPU batch bridge exists
    multiple target residuals can be submitted through one .Call
    diagnostics explicitly say true_batched_kernel = false
```

This is the first point where the project can be described as a precision
ladder rather than a single approximate CUDA path plus a CPU oracle.

The previous precision-ladder goal established:

```text
fastSplineCUDA:
    high-throughput pure GPU approximate backend

mgcvExtractGPUFixedSP:
    mgcv setup anchored fixed-sp CUDA solve

mgcvExtractGPUGCV:
    single-penalty |S| = 1 / |S| = 2 smoothing selection
    direct grid and spectral/Demmler-Reinsch-style scoring

mgcvExtractGPU same-setup batch:
    one native .Call with shared X/Z/XtX and multi-target Y/Xty
    diagnostics explicitly report true_batched_kernel = false

hybrid:
    fastSplineCUDA primary
    mgcvExtractGPU near-alpha verifier
    canonical replay preserved

tprsApproxCUDA:
    deferred by go/no-go memo until evidence justifies it
```

The next stage is not a new residual algorithm. It is an operationalization layer that decides when to use each backend, explains time/accuracy tradeoffs, and makes compatibility boundaries fail closed.

## Backend Positioning

The public narrative should stay explicit:

```text
fastSplineCUDA:
    fastest pure GPU approximate residual backend
    primary candidate for precision = "fast"
    not mgcv-compatible

mgcvExtractGPUFixedSP:
    mgcv setup anchored fixed-sp GPU solve
    compatibility and numerical parity gate
    not a new smoothing-selection backend

mgcvExtractGPUGCV:
    mgcv setup anchored single-penalty smoothing-selection backend
    supported first for |S| = 1 and |S| = 2
    candidate verifier / compatible backend where the envelope permits

same-setup native batch:
    reduces R/native call overhead and setup repetition
    still performs repeated per-target CUDA solves internally
    not a true fused or batched GPU kernel

hybrid:
    fastSplineCUDA primary
    mgcvExtractGPU verifier near alpha
    canonical replay preserved

legacy mgcv:
    authoritative compatibility reference

tprsApproxCUDA:
    deferred pure GPU approximation
    only reconsidered after projection-floor, oracle-lambda, timing, and graph evidence
```

This naming distinction is part of the compatibility contract:

```text
mgcvExtractGPU is a compatibility bridge.
tprsApproxCUDA would be a more accurate pure GPU approximation.
fastSplineCUDA should remain a frozen approximate baseline until reports justify a replacement.
```

## Non-goals

```text
No new residual basis.
No tprsApproxCUDA implementation.
No multi-penalty GPU GCV.
No full mgcv clone.
No bamGPU.
No mutation of fastSplineCUDA baseline semantics.
No claiming true batched/fused mgcvExtractGPU kernel while diagnostics say true_batched_kernel = false.
No changing canonical replay order.
No making mgcvExtractGPU the default backend without timing and graph-level evidence.
No hiding unsupported |S| > 2 additive-smooth cases behind a compatibility claim.
No reusing raw mgcv sp values across different basis/penalty parameterizations.
```

Deferred work:

```text
multi-penalty GPU GCV:
    defer until workload stats show |S| > 2 dominates verifier/runtime cost

true fused/batched mgcvExtractGPU kernel:
    defer until timing shows linear_solve_ms dominates and targets_per_setup is high

tprsApproxCUDA:
    defer until mgcvExtractGPU materially improves graph drift, mgcv CPU setup is the bottleneck,
    and projection-floor/oracle-lambda data show a pure GPU TPRS-like approximation can keep most accuracy
```

## Phase 1: Precision-ladder summary report

Create a readable summary layer over existing CSV artifacts. The report should be designed for backend decisions, not just test evidence.

### Required inputs

```text
precision ladder attribution CSVs:
    mgcv_residual_compatibility.csv
    mgcv_ci_compatibility.csv
    mgcv_graph_compatibility.csv
    precision_ladder_attribution_summary.csv

mgcvExtractGPU graph campaign CSVs:
    mgcv_extract_gpu_graph_comparison.csv
    mgcv_extract_gpu_hybrid_diagnostics.csv
    mgcv_extract_gpu_graph_summary.csv

tprsApproxCUDA decision artifacts:
    tprs_approx_cuda_decision.csv
    tprs_approx_cuda_go_no_go_memo.md
```

### Required summary table

The top-level Markdown report must include a backend comparison table with:

```text
backend
role
supported formula class
supported |S|
residual rel-L2 p50
residual rel-L2 p95
residual rel-L2 max
log-p drift p50
log-p drift p95
near-alpha flip rate
skeleton SHD
sepset mismatch rate
WAN-PDAG mismatch
setup time
solve time
CI time
end-to-end runtime
speedup vs legacy
recommended use
```

### Required stratification

All summaries should be stratifiable by:

```text
|S| = 1
|S| = 2
|S| > 2
number of targets sharing setup
n
basis dimension / null-space dimension
CI method: dCov / HSIC
case class: well-conditioned / difficult
```

### Deliverables

```text
fastkpc/R/precision_ladder_report.R
fastkpc/tools/run_precision_ladder_summary_report.R
fastkpc/tools/run_precision_ladder_summary_report.sh
fastkpc/tests/test_precision_ladder_summary_report.R
```

### Implementation tasks

- [ ] **Step 1: Add the failing report test**

Create `fastkpc/tests/test_precision_ladder_summary_report.R` with a synthetic
artifact bundle. The test should not require CUDA or the local cancer dataset.

Required assertions:

```r
source("fastkpc/R/precision_ladder_report.R")

out <- fastkpc_write_precision_ladder_summary_report(
  residual_metrics = residual_metrics,
  ci_metrics = ci_metrics,
  graph_metrics = graph_metrics,
  gpu_graph_summary = gpu_graph_summary,
  tprs_decision = tprs_decision,
  output_dir = tempdir()
)

stopifnot(file.exists(out$report_path))
stopifnot(file.exists(out$summary_csv))

txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
stopifnot(grepl("mgcvExtractGPU is a compatibility bridge", txt, fixed = TRUE))
stopifnot(grepl("same-setup native batch is not a true fused/batched GPU kernel", txt, fixed = TRUE))
stopifnot(grepl("tprsApproxCUDA", txt, fixed = TRUE))
stopifnot(grepl("Decision: defer", txt, fixed = TRUE))
stopifnot(grepl("|S| = 1", txt, fixed = TRUE))
stopifnot(grepl("|S| = 2", txt, fixed = TRUE))
stopifnot(grepl("|S| > 2", txt, fixed = TRUE))
stopifnot(grepl("targets sharing setup", txt, fixed = TRUE))
stopifnot(grepl("basis dimension", txt, fixed = TRUE))
stopifnot(grepl("CI method", txt, fixed = TRUE))

summary <- read.csv(out$summary_csv, stringsAsFactors = FALSE)
required_cols <- c(
  "backend", "role", "supported_formula_class", "supported_S",
  "residual_rel_l2_p50", "residual_rel_l2_p95", "residual_rel_l2_max",
  "log_p_drift_p50", "log_p_drift_p95", "near_alpha_flip_rate",
  "skeleton_shd", "sepset_mismatch_rate", "wanpdag_mismatch",
  "setup_time", "solve_time", "ci_time", "end_to_end_runtime",
  "speedup_vs_legacy", "recommended_use"
)
stopifnot(all(required_cols %in% names(summary)))
stopifnot(all(c(
  "fastSplineCUDA",
  "mgcvExtractGPUFixedSP",
  "mgcvExtractGPUGCV",
  "hybrid-fastSplineCUDA-mgcvExtractGPU",
  "legacy-mgcv"
) %in% summary$backend))
```

Run:

```bash
Rscript fastkpc/tests/test_precision_ladder_summary_report.R
```

Expected before implementation:

```text
cannot open file 'fastkpc/R/precision_ladder_report.R'
```

- [ ] **Step 2: Implement the summary builder**

Create `fastkpc/R/precision_ladder_report.R` with:

```r
fastkpc_precision_ladder_backend_summary <- function(
  residual_metrics = NULL,
  ci_metrics = NULL,
  graph_metrics = NULL,
  gpu_graph_summary = NULL,
  tprs_decision = NULL
)

fastkpc_write_precision_ladder_summary_report <- function(
  residual_metrics = NULL,
  ci_metrics = NULL,
  graph_metrics = NULL,
  gpu_graph_summary = NULL,
  tprs_decision = NULL,
  output_dir = file.path("fastkpc", "artifacts", "precision_ladder_summary")
)
```

Rules:

```text
Do not invent timings that are not measured yet.
Use NA_real_ for setup_time / solve_time / ci_time until Phase 2.
Map available campaign runtime to end_to_end_runtime.
Compute speedup_vs_legacy only when a legacy runtime exists in the same input.
Keep all required backends in the table even when some metrics are NA.
```

- [ ] **Step 3: Add a CLI runner**

Create `fastkpc/tools/run_precision_ladder_summary_report.R`:

```r
source("fastkpc/R/precision_ladder_report.R")

output_dir <- Sys.getenv(
  "FASTKPC_PRECISION_LADDER_SUMMARY_DIR",
  file.path("fastkpc", "artifacts", "precision_ladder_summary")
)

out <- fastkpc_write_precision_ladder_summary_report(output_dir = output_dir)
cat("precision ladder summary report:", out$report_path, "\n")
cat("precision ladder backend summary:", out$summary_csv, "\n")
```

Create `fastkpc/tools/run_precision_ladder_summary_report.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_precision_ladder_summary_report.R "$@"
```

- [ ] **Step 4: Verify Phase 1**

Run:

```bash
Rscript fastkpc/tests/test_precision_ladder_summary_report.R
Rscript fastkpc/tools/run_precision_ladder_summary_report.R
Rscript -e 'for (f in list.files("fastkpc/R", "\\.R$", full.names=TRUE)) { cat("parse", f, "\n"); parse(f) }'
```

Expected:

```text
All commands exit 0.
The report path and summary CSV path are printed by the runner.
```

### Gate

```text
The report can be regenerated from CSV artifacts.
The report states that mgcvExtractGPU is a compatibility bridge.
The report states that same-setup native batch is not a true fused/batched GPU kernel.
The report recommends no tprsApproxCUDA implementation unless the decision CSV says otherwise.
```

## Phase 2: Timing attribution

Add timing diagnostics that explain where wall time is spent. This is the main evidence needed before deciding whether a true batched GPU kernel is worth building.

### Required timing fields

Each backend/campaign row should be able to report:

```text
mgcv_setup_cpu_ms
setup_cache_lookup_ms
setup_cache_hit
host_to_device_ms
spectral_prepare_ms
gcv_score_ms
linear_solve_ms
residual_materialize_ms
device_to_host_ms
ci_test_ms
canonical_replay_ms
total_ms
```

### Required context fields

```text
backend
mode
solve_source
native_gpu_solve_used
true_batched_kernel
targets_per_setup
setup_reuse_count
gcv_grid_points
gcv_grid_boundary_hit
condition_estimate
fallback_reason
setup_fingerprint
target_fingerprint
```

### Timing interpretation rules

The report must classify the main bottleneck:

```text
mgcv setup dominated:
    mgcv_setup_cpu_ms is the largest component

GCV dominated:
    spectral_prepare_ms + gcv_score_ms dominate

linear solve dominated:
    linear_solve_ms dominates and targets_per_setup is large

CI dominated:
    ci_test_ms dominates

replay dominated:
    canonical_replay_ms dominates
```

### Deliverables

```text
fastkpc/R/precision_ladder_timing.R
fastkpc/tools/run_precision_ladder_timing_campaign.R
fastkpc/tools/run_precision_ladder_timing_campaign.sh
fastkpc/tests/test_precision_ladder_timing_schema.R
fastkpc/tests/test_precision_ladder_timing_report.R
```

### Implementation tasks

- [ ] **Step 1: Add timing schema tests**

Create `fastkpc/tests/test_precision_ladder_timing_schema.R`.

Required test shape:

```r
source("fastkpc/R/precision_ladder_timing.R")

row <- fastkpc_precision_ladder_timing_row(
  backend = "mgcvExtractGPUFixedSP",
  mode = "fixed-sp",
  solve_source = "cuda-fixed-sp",
  native_gpu_solve_used = TRUE,
  true_batched_kernel = FALSE,
  targets_per_setup = 4L,
  mgcv_setup_cpu_ms = 3,
  setup_cache_lookup_ms = 1,
  host_to_device_ms = 2,
  linear_solve_ms = 10,
  residual_materialize_ms = 2,
  device_to_host_ms = 1,
  ci_test_ms = 5,
  canonical_replay_ms = 1
)

required <- c(
  "backend", "mode", "solve_source", "native_gpu_solve_used",
  "true_batched_kernel", "targets_per_setup", "setup_reuse_count",
  "mgcv_setup_cpu_ms", "setup_cache_lookup_ms", "setup_cache_hit",
  "host_to_device_ms", "spectral_prepare_ms", "gcv_score_ms",
  "linear_solve_ms", "residual_materialize_ms", "device_to_host_ms",
  "ci_test_ms", "canonical_replay_ms", "total_ms",
  "gcv_grid_points", "gcv_grid_boundary_hit", "condition_estimate",
  "fallback_reason", "setup_fingerprint", "target_fingerprint",
  "timing_accounting_note"
)
stopifnot(all(required %in% names(row)))
stopifnot(row$total_ms >= 25)
stopifnot(row$true_batched_kernel == FALSE)
stopifnot(row$targets_per_setup == 4L)
```

Run:

```bash
Rscript fastkpc/tests/test_precision_ladder_timing_schema.R
```

Expected before implementation:

```text
cannot open file 'fastkpc/R/precision_ladder_timing.R'
```

- [ ] **Step 2: Implement timing row and bottleneck classifier**

Create `fastkpc/R/precision_ladder_timing.R` with:

```r
fastkpc_precision_ladder_timing_row <- function(
  backend,
  mode,
  solve_source,
  native_gpu_solve_used = FALSE,
  true_batched_kernel = FALSE,
  targets_per_setup = 1L,
  setup_reuse_count = NA_integer_,
  mgcv_setup_cpu_ms = NA_real_,
  setup_cache_lookup_ms = NA_real_,
  setup_cache_hit = NA,
  host_to_device_ms = NA_real_,
  spectral_prepare_ms = NA_real_,
  gcv_score_ms = NA_real_,
  linear_solve_ms = NA_real_,
  residual_materialize_ms = NA_real_,
  device_to_host_ms = NA_real_,
  ci_test_ms = NA_real_,
  canonical_replay_ms = NA_real_,
  total_ms = NA_real_,
  gcv_grid_points = NA_integer_,
  gcv_grid_boundary_hit = NA,
  condition_estimate = NA_real_,
  fallback_reason = NA_character_,
  setup_fingerprint = NA_character_,
  target_fingerprint = NA_character_
)

fastkpc_classify_timing_bottleneck <- function(row)
```

Classifier outputs:

```text
mgcv_setup_dominated
gcv_dominated
linear_solve_dominated
ci_dominated
replay_dominated
unclassified
```

- [ ] **Step 3: Add timing report test**

Create `fastkpc/tests/test_precision_ladder_timing_report.R`.

Required assertions:

```r
source("fastkpc/R/precision_ladder_timing.R")

rows <- rbind(
  fastkpc_precision_ladder_timing_row(
    backend = "mgcvExtractGPUFixedSP",
    mode = "fixed-sp",
    solve_source = "cuda-fixed-sp",
    native_gpu_solve_used = TRUE,
    true_batched_kernel = FALSE,
    targets_per_setup = 8L,
    linear_solve_ms = 90,
    total_ms = 120
  ),
  fastkpc_precision_ladder_timing_row(
    backend = "legacy-mgcv",
    mode = "reference",
    solve_source = "mgcv",
    mgcv_setup_cpu_ms = 80,
    total_ms = 100
  )
)

out <- fastkpc_write_precision_ladder_timing_report(rows, output_dir = tempdir())
stopifnot(file.exists(out$csv_path))
stopifnot(file.exists(out$report_path))

txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
stopifnot(grepl("true batched solve kernel", txt, fixed = TRUE))
stopifnot(grepl("same-setup batch", txt, fixed = TRUE))
stopifnot(grepl("linear_solve_dominated", txt, fixed = TRUE))
```

- [ ] **Step 4: Add timing campaign runner**

Create `fastkpc/tools/run_precision_ladder_timing_campaign.R` and `.sh`.

First version may synthesize rows from existing diagnostics when detailed timers
are absent, but it must mark missing components as `NA_real_` and include
`timing_accounting_note`.

- [ ] **Step 5: Verify Phase 2**

Run:

```bash
Rscript fastkpc/tests/test_precision_ladder_timing_schema.R
Rscript fastkpc/tests/test_precision_ladder_timing_report.R
Rscript fastkpc/tools/run_precision_ladder_timing_campaign.R
```

Expected:

```text
All commands exit 0.
The timing CSV includes true_batched_kernel = FALSE for same-setup native batch.
```

### Gate

```text
Timing rows are written to CSV.
Every timing row has total_ms >= sum of known component times or records timing_accounting_note.
same-setup batch reports targets_per_setup > 1 and true_batched_kernel = false.
The report can say whether a true batched solve kernel is likely useful.
```

## Phase 3: Hybrid routing policy calibration

Formalize public routing modes and calibrate the default hybrid verifier band.

### Public modes

```r
precision = "fast"
precision = "compatible"
precision = "hybrid"
```

### Routing semantics

```text
fast:
    primary backend = fastSplineCUDA
    verifier = none
    compatibility claim = approximate only

compatible:
    if |S| <= 2 and setup is single-penalty:
        backend = mgcvExtractGPUGCV
    otherwise:
        backend = mgcvExtractCPU / legacy mgcv fallback

hybrid:
    primary backend = fastSplineCUDA
    verifier = mgcvExtractGPUGCV where supported
    fallback = mgcvExtractCPU / legacy mgcv where unsupported
    replay = canonical legacy order
```

### First-version verifier trigger

Keep the default transparent:

```text
trigger if abs(log(p_primary / alpha)) <= tau
```

Campaign-calibrate:

```text
tau in log(1.5), log(2), log(3), log(5)
alpha in 0.01, 0.05, 0.10
```

### Future risk flags

Record but do not use for default routing until calibrated:

```text
|S| = 2 joint smooth
fastSpline residual diagnostics abnormal
GCV grid boundary hit
condition estimate high
primary backend fallback
primary p-value NaN / Inf
```

### Deliverables

```text
fastkpc/R/backend_routing_policy.R
fastkpc/R/hybrid_policy_calibration_report.R
fastkpc/tools/run_hybrid_policy_calibration_report.R
fastkpc/tests/test_backend_routing_policy.R
fastkpc/tests/test_hybrid_policy_calibration_report.R
```

### Implementation tasks

- [ ] **Step 1: Add routing policy tests**

Create `fastkpc/tests/test_backend_routing_policy.R`.

Required assertions:

```r
source("fastkpc/R/backend_routing_policy.R")

fast_route <- fastkpc_select_backend_route(
  precision = "fast",
  S_size = 2L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE
)
stopifnot(fast_route$primary_backend == "fastSplineCUDA")
stopifnot(is.na(fast_route$verifier_backend))
stopifnot(fast_route$compatibility_claim == "approximate")

compatible_route <- fastkpc_select_backend_route(
  precision = "compatible",
  S_size = 2L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE
)
stopifnot(compatible_route$primary_backend == "mgcvExtractGPUGCV")
stopifnot(compatible_route$compatibility_claim == "mgcv-setup-anchored")
stopifnot(compatible_route$canonical_replay_required == TRUE)

fallback_route <- fastkpc_select_backend_route(
  precision = "compatible",
  S_size = 4L,
  single_penalty = FALSE,
  mgcv_extract_gpu_supported = FALSE
)
stopifnot(fallback_route$primary_backend %in% c("mgcvExtractCPU", "legacy-mgcv"))
stopifnot(fallback_route$fallback_reason != "")

hybrid_route <- fastkpc_select_backend_route(
  precision = "hybrid",
  S_size = 1L,
  single_penalty = TRUE,
  mgcv_extract_gpu_supported = TRUE,
  tau = log(2)
)
stopifnot(hybrid_route$primary_backend == "fastSplineCUDA")
stopifnot(hybrid_route$verifier_backend == "mgcvExtractGPUGCV")
stopifnot(hybrid_route$canonical_replay_required == TRUE)
```

Run:

```bash
Rscript fastkpc/tests/test_backend_routing_policy.R
```

Expected before implementation:

```text
cannot open file 'fastkpc/R/backend_routing_policy.R'
```

- [ ] **Step 2: Implement routing policy**

Create `fastkpc/R/backend_routing_policy.R` with:

```r
fastkpc_select_backend_route <- function(
  precision = c("fast", "compatible", "hybrid"),
  S_size,
  single_penalty,
  mgcv_extract_gpu_supported,
  tau = log(2),
  fallback_backend = "legacy-mgcv"
)

fastkpc_near_alpha_trigger <- function(primary_p, alpha, tau)
```

Behavior:

```text
precision = "fast":
    primary_backend = "fastSplineCUDA"
    verifier_backend = NA
    never selects mgcvExtractGPU

precision = "compatible":
    supported single-penalty |S| <= 2 -> "mgcvExtractGPUGCV"
    unsupported -> fallback_backend
    never selects fastSplineCUDA as a compatibility substitute

precision = "hybrid":
    primary_backend = "fastSplineCUDA"
    supported verifier -> "mgcvExtractGPUGCV"
    unsupported verifier -> fallback_backend
    canonical_replay_required = TRUE
```

- [ ] **Step 3: Add hybrid calibration report test**

Create `fastkpc/tests/test_hybrid_policy_calibration_report.R`.

Required assertions:

```r
source("fastkpc/R/hybrid_policy_calibration_report.R")

campaign <- data.frame(
  tau = c(log(1.5), log(2), log(3), log(5)),
  alpha = 0.05,
  num_tests_total = 100L,
  num_verified = c(5L, 10L, 20L, 40L),
  num_primary_decision_flips_vs_legacy = 12L,
  num_hybrid_decision_flips_vs_legacy = c(8L, 4L, 3L, 3L),
  skeleton_shd_primary = 5L,
  skeleton_shd_hybrid = c(4L, 2L, 2L, 2L),
  runtime_primary = 1,
  runtime_hybrid = c(1.1, 1.3, 1.8, 3.0),
  runtime_legacy = 10
)

out <- fastkpc_write_hybrid_policy_calibration_report(campaign, output_dir = tempdir())
stopifnot(file.exists(out$report_path))
stopifnot(file.exists(out$summary_csv))
summary <- read.csv(out$summary_csv, stringsAsFactors = FALSE)
stopifnot("selected_default_tau" %in% names(summary))
stopifnot(any(summary$selected_default_tau == log(2)))
txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
stopifnot(grepl("default tau", txt, ignore.case = TRUE))
stopifnot(grepl("canonical replay", txt, fixed = TRUE))
```

- [ ] **Step 4: Implement calibration report**

Create `fastkpc/R/hybrid_policy_calibration_report.R` with:

```r
fastkpc_select_default_tau <- function(campaign)
fastkpc_write_hybrid_policy_calibration_report <- function(campaign, output_dir)
```

Default selection rule for the first version:

```text
Among rows that reduce hybrid decision flips versus primary and keep runtime_hybrid <= 2x runtime_primary,
choose the smallest tau with maximal flip reduction within one flip of the best row.
If no row improves flips, return NA and recommend no default hybrid verification.
```

- [ ] **Step 5: Verify Phase 3**

Run:

```bash
Rscript fastkpc/tests/test_backend_routing_policy.R
Rscript fastkpc/tests/test_hybrid_policy_calibration_report.R
```

Expected:

```text
Both commands exit 0.
```

### Gate

```text
precision = "fast" never calls mgcvExtractGPU.
precision = "compatible" never silently uses fastSplineCUDA for supported mgcvExtractGPU cases.
precision = "hybrid" preserves canonical replay order.
Default tau is selected from campaign evidence and written to a report.
Diagnostics expose p_source_used for every CI test.
```

## Phase 4: Compatibility envelope

Make mgcvExtractGPU fail closed outside the version and semantic boundary that has been validated.

### Capability object

Extend the existing capability/fingerprint system with:

```text
supported_R_versions
supported_mgcv_versions
observed_R_version
observed_mgcv_version
supported_family = gaussian_identity
supported_formula_classes = full-smooth, additive-smooth
supported_single_penalty_modes = |S| = 1, |S| = 2
setup_fingerprint_schema_version
cuda_backend_version
native_same_setup_batch_version
spectral_gcv_version
compatibility_status
compatibility_action
```

### Fail-closed behavior

```text
compatibility_status = "supported":
    run requested mgcvExtractGPU path

compatibility_status = "canary":
    warn and allow only if allow_canary = TRUE

compatibility_status = "unsupported":
    warn and fallback to mgcvExtractCPU / legacy mgcv
```

### Deliverables

```text
fastkpc/R/mgcv_extract_compatibility_envelope.R
fastkpc/tests/test_mgcv_extract_compatibility_envelope.R
fastkpc/tests/test_mgcv_extract_fail_closed.R
```

### Implementation tasks

- [ ] **Step 1: Add compatibility envelope test**

Create `fastkpc/tests/test_mgcv_extract_compatibility_envelope.R`.

Required assertions:

```r
source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

env <- fastkpc_mgcv_extract_gpu_capabilities(
  observed_R_version = "4.5.0",
  observed_mgcv_version = "1.9-4",
  observed_cuda_backend_version = "mgcvExtractGPU-v1"
)

required <- c(
  "backend", "role", "supported_R_versions", "supported_mgcv_versions",
  "observed_R_version", "observed_mgcv_version",
  "supported_family", "supported_formula_classes",
  "supported_single_penalty_modes", "setup_fingerprint_schema_version",
  "cuda_backend_version", "native_same_setup_batch_version",
  "spectral_gcv_version", "compatibility_status", "compatibility_action"
)
stopifnot(all(required %in% names(env)))
stopifnot(env$backend == "mgcvExtractGPU")
stopifnot(env$role == "version-pinned compatibility bridge")
stopifnot(env$supported_family == "gaussian_identity")
```

- [ ] **Step 2: Add fail-closed tests**

Create `fastkpc/tests/test_mgcv_extract_fail_closed.R`.

Required assertions:

```r
source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

supported <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.5.0",
  observed_mgcv_version = "1.9-4",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = FALSE
)
stopifnot(supported$compatibility_status == "supported")
stopifnot(supported$compatibility_action == "run-mgcvExtractGPU")

unsupported <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.6.0",
  observed_mgcv_version = "1.10-0",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = FALSE
)
stopifnot(unsupported$compatibility_status == "unsupported")
stopifnot(unsupported$compatibility_action == "fallback")
stopifnot(grepl("4.6.0", unsupported$warning_message, fixed = TRUE))
stopifnot(grepl("1.10-0", unsupported$warning_message, fixed = TRUE))

canary <- fastkpc_check_mgcv_extract_gpu_compatibility(
  observed_R_version = "4.6.0",
  observed_mgcv_version = "1.10-0",
  supported_R_versions = "4.5.0",
  supported_mgcv_versions = "1.9-4",
  allow_canary = TRUE
)
stopifnot(canary$compatibility_status == "canary")
stopifnot(canary$compatibility_action == "warn-and-run")
```

- [ ] **Step 3: Implement compatibility envelope**

Create `fastkpc/R/mgcv_extract_compatibility_envelope.R` with:

```r
fastkpc_mgcv_extract_gpu_capabilities <- function(
  observed_R_version = R.version$major,
  observed_mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
    as.character(utils::packageVersion("mgcv"))
  } else {
    NA_character_
  },
  observed_cuda_backend_version = "mgcvExtractGPU-v1",
  supported_R_versions = c("4.5.0"),
  supported_mgcv_versions = c("1.9-4")
)

fastkpc_check_mgcv_extract_gpu_compatibility <- function(
  observed_R_version,
  observed_mgcv_version,
  supported_R_versions,
  supported_mgcv_versions,
  allow_canary = FALSE
)
```

Do not throw for unsupported versions in the checker. Return a structured
fallback decision so public wrappers can decide how to route.

- [ ] **Step 4: Verify Phase 4**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_compatibility_envelope.R
Rscript fastkpc/tests/test_mgcv_extract_fail_closed.R
```

Expected:

```text
Both commands exit 0.
Unsupported versions return fallback decisions with explicit warning_message.
```

### Gate

```text
Unsupported R/mgcv versions do not silently run mgcvExtractGPU.
Warnings include observed version, supported version, backend requested, and fallback backend.
Compatibility status is included in public diagnostics.
```

## Phase 5: Real workload structure statistics

Measure whether true batched kernels or multi-penalty GCV would matter on real workloads before building them.

### Required workload stats

```text
dataset_id
n
p
alpha
max_conditioning_level
conditioning_level
num_ci_tests
num_unique_S
num_same_setup_groups
targets_per_setup_p50
targets_per_setup_p95
targets_per_setup_max
num_tests_by_S_size
runtime_by_S_size
near_alpha_tests_by_S_size
verifier_calls_by_S_size
mgcvExtractGPU_supported_tests
mgcvExtractGPU_unsupported_tests
```

### Real dataset target

Support the current local dataset path when available:

```text
/data/wenyujianData/zhuData/2025/causalDiscoveryInput.RData
```

The runner must also support synthetic fallback data so tests do not depend on local private data.

### Deliverables

```text
fastkpc/R/workload_structure_stats.R
fastkpc/tools/run_workload_structure_stats.R
fastkpc/tests/test_workload_structure_stats.R
```

### Implementation tasks

- [ ] **Step 1: Add workload stats test**

Create `fastkpc/tests/test_workload_structure_stats.R`.

Required assertions:

```r
source("fastkpc/R/workload_structure_stats.R")

test_plan <- data.frame(
  canonical_test_order_id = seq_len(8),
  x = c(1, 1, 2, 2, 3, 3, 4, 4),
  y = c(2, 3, 3, 4, 4, 5, 5, 6),
  S_key = c("1", "1", "1,2", "1,2", "1,2,3", "1,2,3", "2", "2"),
  S_size = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L),
  conditioning_level = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L),
  near_alpha = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE),
  verifier_called = c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE),
  mgcvExtractGPU_supported = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

stats <- fastkpc_workload_structure_stats(
  test_plan = test_plan,
  dataset_id = "synthetic-unit",
  n = 100L,
  p = 6L,
  alpha = 0.05,
  max_conditioning_level = 3L
)

required <- c(
  "dataset_id", "n", "p", "alpha", "max_conditioning_level",
  "conditioning_level", "num_ci_tests", "num_unique_S",
  "num_same_setup_groups", "targets_per_setup_p50",
  "targets_per_setup_p95", "targets_per_setup_max",
  "num_tests_by_S_size", "runtime_by_S_size",
  "near_alpha_tests_by_S_size", "verifier_calls_by_S_size",
  "mgcvExtractGPU_supported_tests", "mgcvExtractGPU_unsupported_tests"
)
stopifnot(all(required %in% names(stats)))
stopifnot(sum(stats$num_ci_tests) == 8L)
stopifnot(sum(stats$mgcvExtractGPU_unsupported_tests) == 2L)

out <- fastkpc_write_workload_structure_stats(stats, output_dir = tempdir())
stopifnot(file.exists(out$csv_path))
stopifnot(file.exists(out$report_path))
txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
stopifnot(grepl("same-setup multiplicity", txt, fixed = TRUE))
stopifnot(grepl("|S| > 2", txt, fixed = TRUE))
```

- [ ] **Step 2: Implement workload stats**

Create `fastkpc/R/workload_structure_stats.R` with:

```r
fastkpc_workload_structure_stats <- function(
  test_plan,
  dataset_id,
  n,
  p,
  alpha,
  max_conditioning_level
)

fastkpc_write_workload_structure_stats <- function(stats, output_dir)
```

Counting rules:

```text
num_unique_S:
    count unique S_key per conditioning level

num_same_setup_groups:
    count S_key groups with at least two tests

targets_per_setup_*:
    count rows per S_key within each conditioning level

mgcvExtractGPU_supported_tests:
    sum TRUE mgcvExtractGPU_supported

mgcvExtractGPU_unsupported_tests:
    sum FALSE mgcvExtractGPU_supported
```

If runtime columns are absent, write `runtime_by_S_size = NA_real_` and state
that full CI was not run.

- [ ] **Step 3: Add local/synthetic runner**

Create `fastkpc/tools/run_workload_structure_stats.R`.

Behavior:

```text
If FASTKPC_WORKLOAD_RDATA is set, load that file.
Else if /data/wenyujianData/zhuData/2025/causalDiscoveryInput.RData exists, load it.
Else create a deterministic synthetic test plan.
Do not run full expensive CI just to collect structure stats.
Write CSV and Markdown to fastkpc/artifacts/workload_structure_stats.
```

The runner should accept private local input but keep tests independent from it.

- [ ] **Step 4: Verify Phase 5**

Run:

```bash
Rscript fastkpc/tests/test_workload_structure_stats.R
Rscript fastkpc/tools/run_workload_structure_stats.R
```

Expected:

```text
Both commands exit 0.
The runner prints whether it used local RData or synthetic fallback.
```

### Gate

```text
Stats can be generated without running full expensive CI.
Stats report whether |S| > 2 dominates wall time.
Stats report whether same-setup multiplicity is high enough to justify true batched kernels.
```

## Phase 6: True batched GPU kernel decision

Use Phase 2 and Phase 5 evidence to decide whether to implement a real fused/batched mgcvExtractGPU kernel.

### Decision criteria

Proceed only if:

```text
linear_solve_ms dominates total_ms
targets_per_setup_p95 is high enough to amortize GPU batch overhead
mgcv setup is not the dominant bottleneck
CI time is not the dominant bottleneck
same-setup native batch bridge is materially faster than per-target R/native calls
graph-level accuracy benefit remains meaningful
```

Defer if:

```text
mgcv setup dominates
GCV scoring dominates
CI test time dominates
|S| > 2 unsupported cases dominate verifier workload
targets_per_setup is usually 1
```

### Deliverables

```text
fastkpc/R/true_batched_kernel_decision.R
fastkpc/tools/run_true_batched_kernel_decision.R
fastkpc/tests/test_true_batched_kernel_decision.R
```

### Implementation tasks

- [ ] **Step 1: Add decision test**

Create `fastkpc/tests/test_true_batched_kernel_decision.R`.

Required assertions:

```r
source("fastkpc/R/true_batched_kernel_decision.R")

timing <- data.frame(
  backend = "mgcvExtractGPUFixedSP",
  mode = "same-setup-native-batch",
  true_batched_kernel = FALSE,
  targets_per_setup = 16L,
  mgcv_setup_cpu_ms = 5,
  gcv_score_ms = 5,
  linear_solve_ms = 80,
  ci_test_ms = 5,
  total_ms = 100
)

workload <- data.frame(
  dataset_id = "synthetic-unit",
  targets_per_setup_p95 = 16,
  mgcvExtractGPU_supported_tests = 80L,
  mgcvExtractGPU_unsupported_tests = 5L,
  near_alpha_tests_by_S_size = 20L
)

decision <- fastkpc_true_batched_kernel_decision(
  timing = timing,
  workload = workload
)
stopifnot(decision$decision %in% c("proceed", "defer"))
stopifnot(decision$decision == "proceed")
stopifnot(grepl("linear_solve_ms", decision$rationale, fixed = TRUE))

defer_timing <- timing
defer_timing$mgcv_setup_cpu_ms <- 80
defer_timing$linear_solve_ms <- 5
defer_timing$total_ms <- 100
defer_decision <- fastkpc_true_batched_kernel_decision(
  timing = defer_timing,
  workload = workload
)
stopifnot(defer_decision$decision == "defer")
stopifnot(grepl("mgcv setup", defer_decision$rationale, fixed = TRUE))

out <- fastkpc_write_true_batched_kernel_decision(decision, output_dir = tempdir())
stopifnot(file.exists(out$csv_path))
stopifnot(file.exists(out$report_path))
txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
stopifnot(grepl("true batched mgcvExtractGPU kernel", txt, fixed = TRUE))
```

- [ ] **Step 2: Implement decision artifact**

Create `fastkpc/R/true_batched_kernel_decision.R` with:

```r
fastkpc_true_batched_kernel_decision <- function(
  timing,
  workload,
  linear_solve_fraction_threshold = 0.5,
  targets_per_setup_p95_threshold = 4,
  unsupported_fraction_threshold = 0.25
)

fastkpc_write_true_batched_kernel_decision <- function(decision, output_dir)
```

Proceed only when:

```text
linear_solve_ms / total_ms >= linear_solve_fraction_threshold
targets_per_setup_p95 >= targets_per_setup_p95_threshold
unsupported_fraction <= unsupported_fraction_threshold
mgcv_setup_cpu_ms is not the largest component
ci_test_ms is not the largest component
```

Otherwise defer and name the dominant blocking reason.

- [ ] **Step 3: Add runner**

Create `fastkpc/tools/run_true_batched_kernel_decision.R`.

Inputs:

```text
FASTKPC_TIMING_CSV
FASTKPC_WORKLOAD_STATS_CSV
```

If inputs are missing, write a defer decision with rationale:

```text
insufficient timing/workload evidence
```

- [ ] **Step 4: Verify Phase 6**

Run:

```bash
Rscript fastkpc/tests/test_true_batched_kernel_decision.R
Rscript fastkpc/tools/run_true_batched_kernel_decision.R
```

Expected:

```text
Both commands exit 0.
A decision CSV and Markdown report are written before any fused kernel work starts.
```

### Gate

```text
Decision artifact is written before any fused/batched mgcvExtractGPU kernel work starts.
Decision artifact states proceed/defer and cites timing/workload evidence.
```

## Documentation Updates

After Phases 1-6, update `README.md` and the relevant `fastkpc/README.md`
section with the operational backend positioning:

```text
precision = "fast":
    fastSplineCUDA

precision = "compatible":
    mgcvExtractGPU where supported
    mgcvExtractCPU / legacy mgcv fallback otherwise

precision = "hybrid":
    fastSplineCUDA primary
    mgcvExtractGPU near-alpha verifier
    canonical replay preserved
```

The docs must include:

```text
mgcvExtractGPU is a version-pinned compatibility bridge.
same-setup native batch is not a true fused/batched GPU kernel.
tprsApproxCUDA remains deferred unless evidence reverses the decision.
CUDA-specific tests remain opt-in.
GitHub Actions are intentionally absent unless reintroduced by explicit request.
```

## Recommended Issues

```text
Issue 1: Add precision-ladder summary report generator
Issue 2: Add timing attribution schema and timing campaign
Issue 3: Add timing bottleneck classifier
Issue 4: Calibrate hybrid default tau from graph campaign evidence
Issue 5: Add public backend routing policy for fast / compatible / hybrid
Issue 6: Add mgcvExtractGPU compatibility envelope and fail-closed fallback
Issue 7: Add real workload structure statistics
Issue 8: Add same-setup multiplicity report
Issue 9: Add true batched kernel go/no-go decision artifact
Issue 10: Update README with operational backend positioning
```

## Acceptance Gates

### Gate A: Reports exist

```text
Precision-ladder summary report is generated from CSV artifacts.
Timing report is generated from timing diagnostics.
Hybrid policy report selects a default tau.
```

### Gate B: Routing is explicit

```text
precision = "fast" maps to fastSplineCUDA only.
precision = "compatible" maps to mgcvExtractGPU where supported and mgcvExtractCPU/legacy fallback otherwise.
precision = "hybrid" maps to fastSplineCUDA primary plus mgcvExtractGPU verifier near alpha.
```

### Gate C: Compatibility is fail-closed

```text
Unsupported R/mgcv/CUDA/version combinations warn and fallback.
Diagnostics include compatibility status and fallback reason.
```

### Gate D: Real workload stats exist

```text
The project can report |S| distribution, same-setup multiplicity, near-alpha rates, and verifier-support coverage for real or synthetic workload input.
```

### Gate E: Kernel decision is evidence-based

```text
True batched mgcvExtractGPU kernel work is either justified or deferred by a decision artifact.
No fused kernel implementation starts before this decision artifact exists.
```

## Success Criteria

```text
1. A user can read one Markdown report and understand backend speed/accuracy tradeoffs.
2. Timing diagnostics identify the dominant bottleneck for each backend mode.
3. Hybrid default tau is chosen from campaign evidence, not a hard-coded guess.
4. Public backend routing exposes fast / compatible / hybrid modes with honest fallback diagnostics.
5. mgcvExtractGPU refuses unsupported compatibility envelopes by fallback, not silent execution.
6. Real workload stats show whether true batched kernels or multi-penalty GCV are worth building.
7. tprsApproxCUDA remains deferred unless new evidence reverses the decision.
```

## Implementation Order

```text
1. Report summary first, because it reveals missing metrics.
2. Timing attribution second, because routing decisions need wall-time evidence.
3. Hybrid routing third, because default policy depends on report/timing data.
4. Compatibility envelope fourth, before public default use.
5. Workload statistics fifth, to decide where future engineering effort matters.
6. True batched kernel decision last, after timing and workload evidence exist.
```

## Verification Commands

```bash
Rscript fastkpc/tests/test_precision_ladder_summary_report.R
Rscript fastkpc/tests/test_precision_ladder_timing_schema.R
Rscript fastkpc/tests/test_precision_ladder_timing_report.R
Rscript fastkpc/tests/test_backend_routing_policy.R
Rscript fastkpc/tests/test_hybrid_policy_calibration_report.R
Rscript fastkpc/tests/test_mgcv_extract_compatibility_envelope.R
Rscript fastkpc/tests/test_mgcv_extract_fail_closed.R
Rscript fastkpc/tests/test_workload_structure_stats.R
Rscript fastkpc/tests/test_true_batched_kernel_decision.R
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

CUDA-specific verification remains opt-in:

```bash
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_mgcv_extract_gpu_native_fixed_sp.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_mgcv_extract_gpu_gcv_single_penalty.R
```
