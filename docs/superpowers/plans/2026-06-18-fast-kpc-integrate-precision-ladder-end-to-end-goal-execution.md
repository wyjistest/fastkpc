# Integrate and Validate fastkpc Precision Ladder End to End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate `precision = "fast" / "compatible" / "hybrid"` into real fastkpc skeleton and WAN-PDAG execution, with fail-closed routing, canonical replay, trace diagnostics, cache-aware workload evidence, held-out hybrid validation, and evidence-based fused-kernel decisions.

**Architecture:** Keep `fastSplineCUDA` as the frozen approximate execution path and `mgcvExtractGPU` as a version-pinned compatibility bridge. Introduce one authoritative resolver that combines semantic, version, runtime, and setup-support checks, then make `fast_kpc()` and scheduler diagnostics consume the resolver instead of duplicating support logic. Add trace instrumentation and held-out validation around existing execution paths before starting any new backend or fused CUDA kernel work.

**Tech Stack:** R orchestration under `fastkpc/R`, public API in `fastkpc/R/fast_kpc.R`, hybrid replay in `fastkpc/R/hybrid_verifier.R`, policy modules from `7c73846`, CLI runners under `fastkpc/tools`, executable tests under `fastkpc/tests`, generated local artifacts under ignored `fastkpc/artifacts`.

---

## Current Baseline

Commit `7c73846` completed the precision-ladder policy layer:

```text
backend_routing_policy.R
hybrid_policy_calibration_report.R
mgcv_extract_compatibility_envelope.R
precision_ladder_report.R
precision_ladder_timing.R
workload_structure_stats.R
true_batched_kernel_decision.R
```

That commit is intentionally a policy/reporting baseline. The next target is
to prove the policies are actually consumed by the public execution path.

## Non-goals

```text
No new residual basis.
No tprsApproxCUDA.
No multi-penalty GPU GCV.
No full mgcv port.
No bamGPU.
No true fused/batched mgcvExtractGPU kernel.
No default switch to precision = "hybrid" before held-out validation.
No silent compatibility downgrade.
No change to canonical replay order.
```

## File Structure

Create:

```text
fastkpc/R/precision_backend_resolver.R
fastkpc/R/precision_execution_trace.R
fastkpc/R/hybrid_heldout_validation.R
fastkpc/R/failure_injection_scenarios.R
fastkpc/tools/run_precision_ladder_e2e_validation.R
fastkpc/tools/run_precision_ladder_e2e_validation.sh
fastkpc/tests/test_precision_backend_resolver.R
fastkpc/tests/test_precision_fast_mode_e2e.R
fastkpc/tests/test_precision_compatible_fail_closed.R
fastkpc/tests/test_precision_hybrid_e2e_replay.R
fastkpc/tests/test_precision_execution_trace.R
fastkpc/tests/test_workload_structure_cache_aware.R
fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
fastkpc/tests/test_hybrid_heldout_validation.R
fastkpc/tests/test_precision_failure_injection.R
```

Modify:

```text
fastkpc/R/backend_routing_policy.R
fastkpc/R/mgcv_extract_compatibility_envelope.R
fastkpc/R/workload_structure_stats.R
fastkpc/R/true_batched_kernel_decision.R
fastkpc/R/hybrid_policy_calibration_report.R
fastkpc/R/fast_kpc.R
fastkpc/tools/run_mgcv_gate_b_tests.sh
README.md
fastkpc/README.md
```

## Phase 1: Authoritative Precision Backend Resolver

Build one resolver that owns all support checks. No scheduler or public wrapper
should separately decide that `mgcvExtractGPU` is supported.

### Public API

```r
fastkpc_resolve_backend_request <- function(
  precision = c("fast", "compatible", "hybrid"),
  alpha,
  tau,
  S,
  formula_class,
  penalty_count,
  family,
  link,
  setup_fingerprint,
  runtime_capabilities,
  fallback_backend = "legacy-mgcv",
  allow_canary = FALSE
)
```

Return a named list:

```text
precision
primary_backend
verifier_backend
compatibility_status
compatibility_action
compatibility_claim
near_alpha_policy
canonical_replay_required
fallback_backend
fallback_reason
supported_checks
unsupported_checks
setup_fingerprint
runtime_capabilities
```

### Task 1: Resolver Red Test

**Files:**
- Create: `fastkpc/tests/test_precision_backend_resolver.R`
- Create: `fastkpc/R/precision_backend_resolver.R`
- Modify: `fastkpc/R/backend_routing_policy.R`
- Modify: `fastkpc/R/mgcv_extract_compatibility_envelope.R`

- [ ] **Step 1: Write the failing resolver test**

```r
fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/precision_backend_resolver.R")

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

fast_route <- fastkpc_resolve_backend_request(
  precision = "fast", alpha = 0.05, tau = log(2), S = c(1L, 2L),
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-a", runtime_capabilities = caps
)
assert_true(fast_route$primary_backend == "fastSplineCUDA",
            "fast mode must use fastSplineCUDA")
assert_true(is.na(fast_route$verifier_backend),
            "fast mode must not select a verifier")
assert_true(fast_route$compatibility_claim == "approximate",
            "fast mode must be approximate")

compatible <- fastkpc_resolve_backend_request(
  precision = "compatible", alpha = 0.05, tau = log(2), S = c(1L, 2L),
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-a", runtime_capabilities = caps
)
assert_true(compatible$primary_backend == "mgcvExtractGPUGCV",
            "supported compatible mode should select mgcvExtractGPUGCV")
assert_true(compatible$compatibility_status == "supported",
            "supported compatible mode should be supported")

bad_family <- fastkpc_resolve_backend_request(
  precision = "compatible", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "binomial", link = "logit",
  setup_fingerprint = "setup-b", runtime_capabilities = caps
)
assert_true(bad_family$primary_backend == "legacy-mgcv",
            "unsupported compatible mode must fall back")
assert_true(bad_family$compatibility_action == "fallback",
            "unsupported compatible mode must fail closed")
assert_true(grepl("family", bad_family$fallback_reason, fixed = TRUE),
            "fallback reason should name family")

hybrid_bad_cuda <- fastkpc_resolve_backend_request(
  precision = "hybrid", alpha = 0.05, tau = log(2), S = 1L,
  formula_class = "full-smooth", penalty_count = 1L,
  family = "gaussian", link = "identity",
  setup_fingerprint = "setup-c",
  runtime_capabilities = modifyList(caps, list(cuda_available = FALSE))
)
assert_true(hybrid_bad_cuda$primary_backend == "fastSplineCUDA",
            "hybrid primary remains fastSplineCUDA")
assert_true(hybrid_bad_cuda$verifier_backend == "legacy-mgcv",
            "unsupported hybrid verifier should fall back")
assert_true(hybrid_bad_cuda$canonical_replay_required,
            "hybrid must require canonical replay")

cat("PASS precision backend resolver\n")
```

- [ ] **Step 2: Run test to verify red**

Run:

```bash
Rscript fastkpc/tests/test_precision_backend_resolver.R
```

Expected:

```text
cannot open file 'fastkpc/R/precision_backend_resolver.R'
```

- [ ] **Step 3: Implement resolver**

Create `fastkpc/R/precision_backend_resolver.R`:

```r
source("fastkpc/R/backend_routing_policy.R")
source("fastkpc/R/mgcv_extract_compatibility_envelope.R")

fastkpc_precision_runtime_capabilities <- function() {
  list(
    R_version = paste(R.version$major, R.version$minor, sep = "."),
    mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
      as.character(utils::packageVersion("mgcv"))
    } else {
      NA_character_
    },
    cuda_available = tryCatch(isTRUE(fastkpc_cuda_available()),
                              error = function(e) FALSE),
    cuda_device_capability = NA_character_,
    mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
  )
}

fastkpc_resolve_backend_request <- function(
  precision = c("fast", "compatible", "hybrid"),
  alpha,
  tau,
  S,
  formula_class,
  penalty_count,
  family,
  link,
  setup_fingerprint,
  runtime_capabilities,
  fallback_backend = "legacy-mgcv",
  allow_canary = FALSE
) {
  precision <- match.arg(precision)
  checks <- fastkpc_check_mgcv_extract_gpu_compatibility(
    observed_R_version = runtime_capabilities$R_version,
    observed_mgcv_version = runtime_capabilities$mgcv_version,
    family = family,
    link = link,
    formula_class = formula_class,
    S_size = length(S),
    penalty_count = penalty_count,
    setup_fingerprint_schema_version =
      runtime_capabilities$setup_fingerprint_schema_version,
    cuda_available = runtime_capabilities$cuda_available,
    mgcvExtractGPU_backend_version =
      runtime_capabilities$mgcvExtractGPU_backend_version,
    spectral_gcv_version = runtime_capabilities$spectral_gcv_version,
    allow_canary = allow_canary
  )
  supported <- checks$compatibility_status == "supported"
  route <- fastkpc_select_backend_route(
    precision = precision,
    S_size = length(S),
    single_penalty = penalty_count == 1L,
    mgcv_extract_gpu_supported = supported,
    tau = tau,
    fallback_backend = fallback_backend
  )
  if (precision == "compatible" && !supported) {
    route$primary_backend <- fallback_backend
  }
  if (precision == "hybrid" && !supported) {
    route$verifier_backend <- fallback_backend
  }
  route$compatibility_status <- checks$compatibility_status
  route$compatibility_action <- checks$compatibility_action
  route$fallback_backend <- fallback_backend
  route$fallback_reason <- if (supported) "" else checks$warning_message
  route$near_alpha_policy <- list(alpha = alpha, tau = tau)
  route$supported_checks <- checks$supported_checks
  route$unsupported_checks <- checks$unsupported_checks
  route$setup_fingerprint <- setup_fingerprint
  route$runtime_capabilities <- runtime_capabilities
  route
}
```

Extend `fastkpc_check_mgcv_extract_gpu_compatibility()` to validate:

```text
R version
mgcv version
family == gaussian
link == identity
formula class in full-smooth / additive-smooth
|S| <= 2 for GPU GCV
penalty_count == 1
setup fingerprint schema version
mgcvExtractGPU backend version
spectral GCV version
CUDA availability
```

- [ ] **Step 4: Run resolver test**

Run:

```bash
Rscript fastkpc/tests/test_precision_backend_resolver.R
```

Expected:

```text
PASS precision backend resolver
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/precision_backend_resolver.R \
  fastkpc/R/backend_routing_policy.R \
  fastkpc/R/mgcv_extract_compatibility_envelope.R \
  fastkpc/tests/test_precision_backend_resolver.R
git commit -m "feat: add authoritative precision backend resolver"
```

## Phase 2: Public Precision Modes in Real Execution

Expose `precision` on `fast_kpc()` and make the result diagnostics record the
resolved route. This phase does not need to replace every residual path with
mgcvExtractGPU immediately; the gate is that public execution consults the
resolver and records the decision consistently.

### Task 2: fast Mode E2E

**Files:**
- Modify: `fastkpc/R/fast_kpc.R`
- Test: `fastkpc/tests/test_precision_fast_mode_e2e.R`

- [ ] **Step 1: Write fast mode test**

```r
source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(101)
data <- matrix(rnorm(80 * 5), 80, 5)

legacy_fast <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", residual_backend = "fastSpline",
  graph_stage = "skeleton", seed = 101
)
precision_fast <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "fast",
  graph_stage = "skeleton", seed = 101
)

assert_true(identical(legacy_fast$skeleton$adjacency,
                      precision_fast$skeleton$adjacency),
            "precision fast must preserve existing fastSpline execution")
assert_true(precision_fast$config$precision == "fast",
            "config should record precision")
assert_true(precision_fast$config$precision_route$primary_backend == "fastSplineCUDA" ||
              precision_fast$config$precision_route$primary_backend == "fastSplineCPU",
            "fast route should record fastSpline primary")
assert_true(precision_fast$config$precision_route$compatibility_claim == "approximate",
            "fast route should be approximate")

cat("PASS precision fast mode e2e\n")
```

- [ ] **Step 2: Run to verify red**

Run:

```bash
Rscript fastkpc/tests/test_precision_fast_mode_e2e.R
```

Expected:

```text
unused argument (precision = "fast")
```

- [ ] **Step 3: Add `precision` to `fast_kpc()`**

Modify signature:

```r
precision = c("legacy", "fast", "compatible", "hybrid"),
tau = log(2),
precision_diagnostics = TRUE,
runtime_capabilities = NULL,
allow_canary_mgcv_extract = FALSE,
```

Resolution rules:

```text
precision = "legacy":
    preserve existing behavior exactly

precision = "fast":
    residual_backend = "fastSpline"
    execution otherwise follows existing engine/device path

precision = "compatible":
    call resolver and record route
    if unsupported, route to legacy/mgcv fallback diagnostics
    do not silently use fastSpline as compatibility substitute

precision = "hybrid":
    call resolver and record route
    preserve existing fastSpline primary path
    attach verifier/fallback diagnostics
```

Add config fields:

```text
precision
precision_requested
precision_route
backend_requested
backend_used
verifier_backend
compatibility_status
compatibility_action
fallback_reason
canonical_replay_required
```

For this phase, when `engine = "cpu"` and `precision = "fast"`, allow
`primary_backend = "fastSplineCPU"` in diagnostics while preserving
`compatibility_claim = "approximate"`.

- [ ] **Step 4: Run test**

```bash
Rscript fastkpc/tests/test_precision_fast_mode_e2e.R
```

Expected:

```text
PASS precision fast mode e2e
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/fast_kpc.R fastkpc/tests/test_precision_fast_mode_e2e.R
git commit -m "feat: expose precision fast execution mode"
```

### Task 3: Compatible Fail-Closed E2E

**Files:**
- Modify: `fastkpc/R/fast_kpc.R`
- Test: `fastkpc/tests/test_precision_compatible_fail_closed.R`

- [ ] **Step 1: Write compatible fail-closed test**

```r
source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(102)
data <- matrix(rnorm(70 * 5), 70, 5)
caps <- list(
  R_version = "unsupported-R",
  mgcv_version = "unsupported-mgcv",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps
)

assert_true(result$config$precision == "compatible",
            "config should record compatible mode")
assert_true(result$config$compatibility_action == "fallback",
            "unsupported compatible mode must fall back")
assert_true(grepl("unsupported", result$config$fallback_reason, fixed = TRUE),
            "fallback reason should be public")
assert_true(result$config$backend_used != "fastSplineCUDA",
            "compatible fallback must not silently use fastSplineCUDA")

cat("PASS precision compatible fail closed\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_precision_compatible_fail_closed.R
```

Expected:

```text
unused argument (precision = "compatible")
```

- [ ] **Step 3: Implement fail-closed compatible routing in `fast_kpc()`**

When resolver returns fallback:

```text
config$backend_requested = "mgcvExtractGPUGCV"
config$backend_used = fallback backend
config$compatibility_action = "fallback"
config$fallback_reason is non-empty
```

If no implemented legacy residual backend exists in this local fast path, the
first version may execute the existing CPU path while diagnostics must record
that compatibility fallback occurred. It must not claim `mgcvExtractGPU` ran.

- [ ] **Step 4: Run test**

```bash
Rscript fastkpc/tests/test_precision_compatible_fail_closed.R
```

Expected:

```text
PASS precision compatible fail closed
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/fast_kpc.R fastkpc/tests/test_precision_compatible_fail_closed.R
git commit -m "feat: fail closed for compatible precision mode"
```

### Task 4: Hybrid E2E Replay Diagnostics

**Files:**
- Modify: `fastkpc/R/fast_kpc.R`
- Modify: `fastkpc/R/hybrid_verifier.R`
- Test: `fastkpc/tests/test_precision_hybrid_e2e_replay.R`

- [ ] **Step 1: Write hybrid e2e replay test**

```r
source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(103)
data <- matrix(rnorm(90 * 6), 90, 6)

primary <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "fast",
  graph_stage = "skeleton", seed = 103
)
hybrid <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "hybrid", tau = log(2),
  graph_stage = "skeleton", seed = 103
)

assert_true(hybrid$config$precision == "hybrid",
            "config should record hybrid precision")
assert_true(isTRUE(hybrid$config$canonical_replay_required),
            "hybrid must require canonical replay")
assert_true(is.data.frame(hybrid$diagnostics$precision_trace),
            "hybrid result should include precision trace")
trace <- hybrid$diagnostics$precision_trace
required <- c("run_id", "canonical_test_order_id", "backend_requested",
              "backend_used", "p_source_used", "fallback_reason")
assert_true(all(required %in% names(trace)),
            "precision trace should expose p-value source and fallback")
assert_true(identical(hybrid$skeleton$adjacency, primary$skeleton$adjacency) ||
              all(order(trace$canonical_test_order_id) ==
                    seq_along(trace$canonical_test_order_id)),
            "hybrid trace must preserve canonical ordering")

cat("PASS precision hybrid e2e replay\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_precision_hybrid_e2e_replay.R
```

Expected:

```text
precision_trace should expose p-value source and fallback
```

- [ ] **Step 3: Add hybrid precision trace stub from real result diagnostics**

Do not invent p-values. The first version may aggregate existing scheduler
diagnostics into trace rows when raw CI p-values are not exposed. Required
fields:

```text
run_id
scenario_id
dataset_hash
conditioning_level
canonical_test_order_id
setup_fingerprint
target_id
backend_requested
backend_used
verifier_backend
fallback_reason
primary_p
verifier_p
p_used
p_source_used
decision_before_verify
decision_after_verify
canonical_replay_required
```

Use `NA_real_` for p-values that the current native path does not expose yet,
but keep the field and source/fallback diagnostics.

- [ ] **Step 4: Run test**

```bash
Rscript fastkpc/tests/test_precision_hybrid_e2e_replay.R
```

Expected:

```text
PASS precision hybrid e2e replay
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/fast_kpc.R fastkpc/R/hybrid_verifier.R \
  fastkpc/tests/test_precision_hybrid_e2e_replay.R
git commit -m "feat: trace hybrid precision replay diagnostics"
```

## Phase 3: Real Execution Trace Instrumentation

Turn timing schema into trace rows created by the public execution path.

### Task 5: Execution Trace Rows

**Files:**
- Create: `fastkpc/R/precision_execution_trace.R`
- Modify: `fastkpc/R/precision_ladder_timing.R`
- Modify: `fastkpc/R/fast_kpc.R`
- Test: `fastkpc/tests/test_precision_execution_trace.R`

- [ ] **Step 1: Write trace test**

```r
source("fastkpc/R/precision_execution_trace.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

row <- fastkpc_precision_trace_row(
  run_id = "run-1",
  scenario_id = "unit",
  dataset_hash = "hash",
  conditioning_level = 1L,
  canonical_test_order_id = 3L,
  setup_fingerprint = "setup",
  target_id = "x1",
  backend_requested = "mgcvExtractGPUGCV",
  backend_used = "legacy-mgcv",
  verifier_backend = "legacy-mgcv",
  compatibility_action = "fallback",
  fallback_reason = "unsupported mgcv version",
  p_source_used = "legacy-mgcv",
  mgcv_setup_cpu_ms = 1,
  linear_solve_ms = 2,
  total_ms = 5
)

required <- c(
  "run_id", "scenario_id", "dataset_hash", "conditioning_level",
  "canonical_test_order_id", "setup_fingerprint", "target_id",
  "backend_requested", "backend_used", "verifier_backend",
  "compatibility_action", "fallback_reason", "CUDA_device",
  "git_sha", "primary_p", "verifier_p", "p_used", "p_source_used",
  "decision_before_verify", "decision_after_verify",
  "mgcv_setup_cpu_ms", "setup_cache_lookup_ms", "host_to_device_ms",
  "spectral_prepare_ms", "gcv_score_ms", "linear_solve_ms",
  "residual_materialize_ms", "device_to_host_ms", "ci_test_ms",
  "canonical_replay_ms", "total_ms"
)
assert_true(all(required %in% names(row)), "trace row missing required fields")
assert_true(row$total_ms >= 5, "trace should preserve total timing")

cat("PASS precision execution trace\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_precision_execution_trace.R
```

Expected:

```text
cannot open file 'fastkpc/R/precision_execution_trace.R'
```

- [ ] **Step 3: Implement trace row and append helpers**

Create:

```r
fastkpc_precision_trace_row <- function(...)
fastkpc_precision_trace_from_result <- function(result, route, run_id, scenario_id)
fastkpc_write_precision_trace <- function(trace, output_dir)
```

`fast_kpc()` should attach:

```r
result$diagnostics$precision_trace
```

when `precision_diagnostics = TRUE`.

- [ ] **Step 4: Run test and affected e2e tests**

```bash
Rscript fastkpc/tests/test_precision_execution_trace.R
Rscript fastkpc/tests/test_precision_hybrid_e2e_replay.R
```

Expected:

```text
PASS precision execution trace
PASS precision hybrid e2e replay
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/precision_execution_trace.R fastkpc/R/fast_kpc.R \
  fastkpc/R/precision_ladder_timing.R fastkpc/tests/test_precision_execution_trace.R
git commit -m "feat: add real precision execution trace rows"
```

## Phase 4: Cache-aware Workload Statistics

Fix multiplicity so it estimates GPU batch width, not merely CI tests per `S`.

### Task 6: Cache-aware Workload Stats

**Files:**
- Modify: `fastkpc/R/workload_structure_stats.R`
- Test: `fastkpc/tests/test_workload_structure_cache_aware.R`

- [ ] **Step 1: Write cache-aware test**

```r
source("fastkpc/R/workload_structure_stats.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

requests <- data.frame(
  setup_fingerprint = c("s1", "s1", "s1", "s2", "s2"),
  canonical_test_order_id = c(1L, 1L, 2L, 3L, 3L),
  target_id = c("x", "y", "x", "a", "b"),
  cache_hit = c(FALSE, FALSE, TRUE, FALSE, TRUE),
  device_solve_called = c(TRUE, TRUE, FALSE, TRUE, FALSE),
  S_size = c(1L, 1L, 1L, 2L, 2L),
  conditioning_level = c(1L, 1L, 1L, 2L, 2L),
  stringsAsFactors = FALSE
)

stats <- fastkpc_cache_aware_workload_stats(
  residual_requests = requests,
  dataset_id = "unit",
  n = 50L,
  p = 4L,
  alpha = 0.05,
  max_conditioning_level = 2L
)

required <- c("ci_tests_per_setup", "raw_residual_requests_per_setup",
              "unique_targets_per_setup", "uncached_targets_per_setup",
              "device_solve_calls_per_setup", "cache_hit_rate",
              "setup_fingerprint")
assert_true(all(required %in% names(stats)), "cache-aware fields missing")
s1 <- stats[stats$setup_fingerprint == "s1", , drop = FALSE]
assert_true(s1$ci_tests_per_setup == 2L, "s1 has two CI tests")
assert_true(s1$raw_residual_requests_per_setup == 3L, "s1 has three requests")
assert_true(s1$unique_targets_per_setup == 2L, "s1 has two unique targets")
assert_true(s1$uncached_targets_per_setup == 2L, "s1 has two uncached requests")
assert_true(s1$device_solve_calls_per_setup == 2L, "s1 has two device solves")

cat("PASS workload structure cache aware\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_workload_structure_cache_aware.R
```

Expected:

```text
could not find function "fastkpc_cache_aware_workload_stats"
```

- [ ] **Step 3: Implement cache-aware stats**

Add:

```r
fastkpc_cache_aware_workload_stats <- function(
  residual_requests,
  dataset_id,
  n,
  p,
  alpha,
  max_conditioning_level
)
```

Group by `setup_fingerprint`, not `S_key`.

- [ ] **Step 4: Run tests**

```bash
Rscript fastkpc/tests/test_workload_structure_stats.R
Rscript fastkpc/tests/test_workload_structure_cache_aware.R
```

Expected:

```text
PASS workload structure stats
PASS workload structure cache aware
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/workload_structure_stats.R \
  fastkpc/tests/test_workload_structure_cache_aware.R
git commit -m "feat: add cache-aware workload multiplicity stats"
```

## Phase 5: Scenario-aligned Fused Kernel Decision

Prevent mixing the slowest timing row from one scenario with the highest
multiplicity workload row from another.

### Task 7: Scenario-aligned Decision

**Files:**
- Modify: `fastkpc/R/true_batched_kernel_decision.R`
- Test: `fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R`

- [ ] **Step 1: Write scenario-aligned test**

```r
source("fastkpc/R/true_batched_kernel_decision.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

timing <- data.frame(
  scenario_id = c("a", "b"),
  dataset_id = c("d1", "d2"),
  backend = "mgcvExtractGPUFixedSP",
  conditioning_level = c(1L, 1L),
  linear_solve_ms = c(90, 5),
  mgcv_setup_cpu_ms = c(5, 80),
  ci_test_ms = c(5, 5),
  total_ms = c(100, 100)
)
workload <- data.frame(
  scenario_id = c("a", "b"),
  dataset_id = c("d1", "d2"),
  backend = "mgcvExtractGPUFixedSP",
  conditioning_level = c(1L, 1L),
  uncached_targets_per_setup_p95 = c(10, 20),
  supported_wall_time_fraction = c(0.9, 0.9),
  evidence_runs = c(2L, 2L)
)

decision <- fastkpc_true_batched_kernel_decision_scenario_aligned(
  timing = timing,
  workload = workload,
  min_evidence_runs = 2L
)
assert_true(decision$decision %in% c("proceed", "defer", "insufficient-evidence"),
            "decision should be enumerated")
assert_true(decision$decision == "defer",
            "mixed evidence should defer because setup-dominated scenario has equal weight")
assert_true("scenario_id" %in% names(decision$evidence),
            "decision should retain scenario evidence")

cat("PASS true batched kernel scenario-aligned decision\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
```

Expected:

```text
could not find function "fastkpc_true_batched_kernel_decision_scenario_aligned"
```

- [ ] **Step 3: Implement scenario-aligned decision**

Add:

```r
fastkpc_true_batched_kernel_decision_scenario_aligned <- function(
  timing,
  workload,
  by = c("scenario_id", "dataset_id", "backend", "conditioning_level"),
  linear_solve_fraction_threshold = 0.5,
  uncached_targets_per_setup_p95_threshold = 4,
  supported_wall_time_fraction_threshold = 0.75,
  min_evidence_runs = 3L
)
```

Rules:

```text
join timing/workload by scenario keys
compute per-scenario decision evidence
weight by total_ms
return proceed only when weighted criteria pass
return insufficient-evidence when joined rows or evidence_runs are below threshold
otherwise defer with dominant reason
```

- [ ] **Step 4: Run tests**

```bash
Rscript fastkpc/tests/test_true_batched_kernel_decision.R
Rscript fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
```

Expected:

```text
PASS true batched kernel decision
PASS true batched kernel scenario-aligned decision
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/true_batched_kernel_decision.R \
  fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
git commit -m "feat: align batched-kernel decision by scenario"
```

## Phase 6: Held-out Hybrid Tau Validation

Keep calibration and validation separate. The result before deployment is an
experimental recommended tau, not a universal default.

### Task 8: Held-out Validation

**Files:**
- Create: `fastkpc/R/hybrid_heldout_validation.R`
- Modify: `fastkpc/R/hybrid_policy_calibration_report.R`
- Test: `fastkpc/tests/test_hybrid_heldout_validation.R`

- [ ] **Step 1: Write held-out test**

```r
source("fastkpc/R/hybrid_heldout_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

cal <- data.frame(
  tau = c(log(1.5), log(2), log(3)),
  decision_flip_rate_primary = c(0.12, 0.12, 0.12),
  decision_flip_rate_hybrid = c(0.08, 0.04, 0.04),
  skeleton_shd_primary = c(5, 5, 5),
  skeleton_shd_hybrid = c(4, 2, 2),
  sepset_mismatch_hybrid = c(0.2, 0.1, 0.1),
  wanpdag_mismatch_hybrid = c(4, 2, 2),
  verification_rate = c(0.05, 0.12, 0.25),
  runtime_ratio = c(1.1, 1.3, 1.8)
)
held <- data.frame(
  tau = log(2),
  decision_flip_rate_primary = 0.10,
  decision_flip_rate_hybrid = 0.05,
  skeleton_shd_primary = 4,
  skeleton_shd_hybrid = 2,
  sepset_mismatch_primary = 0.2,
  sepset_mismatch_hybrid = 0.1,
  wanpdag_mismatch_primary = 3,
  wanpdag_mismatch_hybrid = 1,
  verification_rate = 0.12,
  runtime_ratio = 1.35
)

result <- fastkpc_validate_hybrid_tau_heldout(
  calibration = cal,
  heldout = held,
  max_runtime_ratio = 2
)
assert_true(result$selected_tau == log(2), "held-out validation should keep log(2)")
assert_true(result$heldout_pass, "held-out graph metrics should pass")
assert_true(grepl("experimental", result$recommendation, fixed = TRUE),
            "recommendation should remain experimental")

cat("PASS hybrid heldout validation\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_hybrid_heldout_validation.R
```

Expected:

```text
cannot open file 'fastkpc/R/hybrid_heldout_validation.R'
```

- [ ] **Step 3: Implement held-out validation**

Create:

```r
fastkpc_select_tau_lexicographic <- function(calibration, max_runtime_ratio = 2)
fastkpc_validate_hybrid_tau_heldout <- function(calibration, heldout, max_runtime_ratio = 2)
fastkpc_write_hybrid_heldout_validation_report <- function(result, output_dir)
```

Lexicographic rule:

```text
1. hybrid graph metrics must be <= primary metrics
2. maximize decision flip rate reduction
3. choose lower verification_rate within near-equal graph loss
4. require runtime_ratio <= budget
```

- [ ] **Step 4: Run test**

```bash
Rscript fastkpc/tests/test_hybrid_heldout_validation.R
```

Expected:

```text
PASS hybrid heldout validation
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/hybrid_heldout_validation.R \
  fastkpc/R/hybrid_policy_calibration_report.R \
  fastkpc/tests/test_hybrid_heldout_validation.R
git commit -m "feat: validate hybrid tau on held-out scenarios"
```

## Phase 7: Failure Injection and Fail-Closed Tests

Actively test unsupported versions, CUDA unavailable, stale setup, NaN p-values,
and verifier failures.

### Task 9: Failure Injection

**Files:**
- Create: `fastkpc/R/failure_injection_scenarios.R`
- Modify: `fastkpc/R/precision_backend_resolver.R`
- Modify: `fastkpc/R/hybrid_verifier.R`
- Test: `fastkpc/tests/test_precision_failure_injection.R`

- [ ] **Step 1: Write failure injection test**

```r
source("fastkpc/R/failure_injection_scenarios.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

cases <- fastkpc_precision_failure_cases()
required_cases <- c("unsupported_mgcv_version", "unsupported_R_version",
                    "cuda_unavailable", "setup_fingerprint_mismatch",
                    "nan_primary_p", "verifier_failure")
assert_true(all(required_cases %in% names(cases)), "missing failure cases")

results <- fastkpc_run_precision_failure_injection(cases)
assert_true(all(results$compatibility_action %in% c("fallback", "warn-and-run")),
            "failure cases must not silently run unsupported GPU")
assert_true(all(nzchar(results$fallback_reason)),
            "failure cases must expose fallback reason")
assert_true(all(results$canonical_replay_preserved),
            "failure cases must preserve canonical replay")
assert_true(all(c("backend_requested", "backend_used", "p_source_used") %in%
                  names(results)),
            "failure diagnostics must expose backend and p-value source")

cat("PASS precision failure injection\n")
```

- [ ] **Step 2: Run to verify red**

```bash
Rscript fastkpc/tests/test_precision_failure_injection.R
```

Expected:

```text
cannot open file 'fastkpc/R/failure_injection_scenarios.R'
```

- [ ] **Step 3: Implement failure cases**

Create:

```r
fastkpc_precision_failure_cases <- function()
fastkpc_run_precision_failure_injection <- function(cases)
```

Each row must include:

```text
case_id
backend_requested
backend_used
compatibility_action
fallback_reason
primary_p
verifier_p
p_used
p_source_used
decision_before_verify
decision_after_verify
canonical_replay_preserved
```

- [ ] **Step 4: Run test**

```bash
Rscript fastkpc/tests/test_precision_failure_injection.R
```

Expected:

```text
PASS precision failure injection
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/failure_injection_scenarios.R \
  fastkpc/R/precision_backend_resolver.R fastkpc/R/hybrid_verifier.R \
  fastkpc/tests/test_precision_failure_injection.R
git commit -m "test: add precision fail-closed injection scenarios"
```

## Phase 8: End-to-end Validation Runner and Docs

Bundle resolver, public modes, trace, cache-aware stats, held-out validation, and
decision artifacts into one local runner.

### Task 10: E2E Validation Runner

**Files:**
- Create: `fastkpc/tools/run_precision_ladder_e2e_validation.R`
- Create: `fastkpc/tools/run_precision_ladder_e2e_validation.sh`
- Modify: `fastkpc/tools/run_mgcv_gate_b_tests.sh`
- Modify: `README.md`
- Modify: `fastkpc/README.md`

- [ ] **Step 1: Add runner**

Create `fastkpc/tools/run_precision_ladder_e2e_validation.R`:

```r
source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/precision_execution_trace.R")
source("fastkpc/R/workload_structure_stats.R")
source("fastkpc/R/hybrid_heldout_validation.R")
source("fastkpc/R/true_batched_kernel_decision.R")

output_dir <- Sys.getenv(
  "FASTKPC_PRECISION_E2E_DIR",
  file.path("fastkpc", "artifacts", "precision_ladder_e2e")
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(501)
data <- matrix(rnorm(80 * 6), 80, 6)
fast <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                 precision = "fast", engine = "cpu", graph_stage = "skeleton")
compatible <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                       precision = "compatible", engine = "cpu",
                       graph_stage = "skeleton")
hybrid <- fast_kpc(data, alpha = 0.2, max_conditioning_size = 1,
                   precision = "hybrid", engine = "cpu",
                   graph_stage = "skeleton")

trace <- do.call(rbind, list(
  fast$diagnostics$precision_trace,
  compatible$diagnostics$precision_trace,
  hybrid$diagnostics$precision_trace
))
utils::write.csv(trace, file.path(output_dir, "precision_execution_trace.csv"),
                 row.names = FALSE)

summary <- data.frame(
  precision = c("fast", "compatible", "hybrid"),
  backend_used = c(fast$config$backend_used,
                   compatible$config$backend_used,
                   hybrid$config$backend_used),
  compatibility_action = c(fast$config$compatibility_action,
                           compatible$config$compatibility_action,
                           hybrid$config$compatibility_action),
  fallback_reason = c(fast$config$fallback_reason,
                      compatible$config$fallback_reason,
                      hybrid$config$fallback_reason),
  skeleton_edges = c(fast$metrics$skeleton_edge_count,
                     compatible$metrics$skeleton_edge_count,
                     hybrid$metrics$skeleton_edge_count)
)
utils::write.csv(summary, file.path(output_dir, "precision_e2e_summary.csv"),
                 row.names = FALSE)
cat("precision ladder e2e artifacts:", output_dir, "\n")
```

Create `fastkpc/tools/run_precision_ladder_e2e_validation.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_precision_ladder_e2e_validation.R "$@"
```

- [ ] **Step 2: Make executable and run**

```bash
chmod +x fastkpc/tools/run_precision_ladder_e2e_validation.sh
Rscript fastkpc/tools/run_precision_ladder_e2e_validation.R
```

Expected:

```text
precision ladder e2e artifacts: fastkpc/artifacts/precision_ladder_e2e
```

- [ ] **Step 3: Add new tests to gate runner**

Append to `fastkpc/tools/run_mgcv_gate_b_tests.sh`:

```bash
Rscript fastkpc/tests/test_precision_backend_resolver.R
Rscript fastkpc/tests/test_precision_fast_mode_e2e.R
Rscript fastkpc/tests/test_precision_compatible_fail_closed.R
Rscript fastkpc/tests/test_precision_hybrid_e2e_replay.R
Rscript fastkpc/tests/test_precision_execution_trace.R
Rscript fastkpc/tests/test_workload_structure_cache_aware.R
Rscript fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
Rscript fastkpc/tests/test_hybrid_heldout_validation.R
Rscript fastkpc/tests/test_precision_failure_injection.R
```

- [ ] **Step 4: Update docs**

Update `README.md` and `fastkpc/README.md` to say:

```text
precision policy is integrated into fast_kpc()
precision = "fast" preserves fastSpline primary execution
precision = "compatible" fails closed when envelope checks fail
precision = "hybrid" records verifier/fallback diagnostics and preserves canonical replay
default precision remains legacy/existing behavior until held-out validation is accepted
true fused/batched mgcvExtractGPU kernel remains blocked on scenario-aligned evidence
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/tools/run_precision_ladder_e2e_validation.R \
  fastkpc/tools/run_precision_ladder_e2e_validation.sh \
  fastkpc/tools/run_mgcv_gate_b_tests.sh README.md fastkpc/README.md
git commit -m "chore: add precision ladder e2e validation runner"
```

## Final Verification

Run:

```bash
Rscript fastkpc/tests/test_precision_backend_resolver.R
Rscript fastkpc/tests/test_precision_fast_mode_e2e.R
Rscript fastkpc/tests/test_precision_compatible_fail_closed.R
Rscript fastkpc/tests/test_precision_hybrid_e2e_replay.R
Rscript fastkpc/tests/test_precision_execution_trace.R
Rscript fastkpc/tests/test_workload_structure_cache_aware.R
Rscript fastkpc/tests/test_true_batched_kernel_decision_scenario_aligned.R
Rscript fastkpc/tests/test_hybrid_heldout_validation.R
Rscript fastkpc/tests/test_precision_failure_injection.R
Rscript fastkpc/tools/run_precision_ladder_e2e_validation.R
Rscript -e 'for (f in list.files("fastkpc/R", "\\.R$", full.names=TRUE)) { parse(f) }; cat("PASS parse all R files\n")'
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

CUDA-specific validation remains opt-in:

```bash
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_mgcv_extract_gpu_same_setup_batch.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_mgcv_extract_gpu_gcv_single_penalty.R
```

## Acceptance Gates

### Gate A: `precision = "fast"`

```text
Matches existing fastSpline execution result and order.
Diagnostics state compatibility_claim = approximate.
Does not call mgcvExtractGPU.
```

### Gate B: `precision = "compatible"`

```text
Supported envelope selects mgcvExtractGPU route.
Unsupported envelope fails closed to legacy/mgcv fallback.
Never silently substitutes fastSplineCUDA for compatibility.
```

### Gate C: `precision = "hybrid"`

```text
Primary remains fastSplineCUDA / fastSpline path.
Verifier is mgcvExtractGPU when supported.
Verifier falls back to legacy/mgcv when unsupported.
Canonical replay is preserved.
Public diagnostics explain p-value source and fallback.
```

### Gate D: Trace and Workload Evidence

```text
Every public run can emit precision_trace rows.
Trace includes backend_requested, backend_used, p_source_used, fallback_reason.
Workload stats use setup_fingerprint and cache-aware residual-request counts.
```

### Gate E: Decision Safety

```text
True fused/batched kernel decision is scenario-aligned.
Held-out tau validation reports experimental tau, not universal default.
Failure injection proves fail-closed behavior for unsupported and error states.
```

## Recommended Issues

```text
Issue 1: Add authoritative precision backend resolver
Issue 2: Add precision argument to fast_kpc() with legacy-compatible default
Issue 3: Attach precision route diagnostics to fastkpc_result
Issue 4: Add precision execution trace rows from real runs
Issue 5: Add cache-aware setup_fingerprint workload statistics
Issue 6: Make true batched kernel decision scenario-aligned
Issue 7: Split hybrid tau calibration and held-out validation
Issue 8: Add failure injection tests for fail-closed routing
Issue 9: Add precision ladder end-to-end validation runner
Issue 10: Update docs with integrated precision modes
```

## Success Criteria

```text
1. fast_kpc(..., precision = "fast") preserves current fastSpline behavior.
2. fast_kpc(..., precision = "compatible") routes through one resolver and fails closed.
3. fast_kpc(..., precision = "hybrid") preserves canonical replay and exposes p-value sources.
4. Diagnostics can explain every fallback and every backend actually used.
5. Workload multiplicity is based on cache-aware target requests, not only CI tests per S.
6. Fused-kernel decisions align timing and workload by scenario.
7. Hybrid tau recommendations are validated on held-out scenarios.
8. No new backend work starts from this plan.
```
