# Operationalize fastkpc precision ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the implemented fastSplineCUDA / mgcvExtractGPU precision ladder into an operational backend policy with reports, timing attribution, routing rules, compatibility boundaries, and workload evidence.

**Architecture:** Keep `fastSplineCUDA` frozen as the high-throughput approximate primary backend. Treat `mgcvExtractGPU` as an mgcv setup anchored compatibility bridge and verifier, with honest diagnostics for fixed-sp, single-penalty GCV, and same-setup native batch paths. Build reporting and routing layers around the existing campaigns before investing in new CUDA kernels.

**Tech Stack:** R campaign/report code under `fastkpc/R`, CLI runners under `fastkpc/tools`, native CUDA diagnostics from `fastkpc/src`, CSV/Markdown artifacts under `fastkpc/artifacts` or `fastkpc/reports`, tests as executable `Rscript fastkpc/tests/*.R`.

---

## Current Baseline

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

### Gate

```text
Decision artifact is written before any fused/batched mgcvExtractGPU kernel work starts.
Decision artifact states proceed/defer and cites timing/workload evidence.
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
