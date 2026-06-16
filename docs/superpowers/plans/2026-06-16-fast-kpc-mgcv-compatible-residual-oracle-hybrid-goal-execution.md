# Fast kPC mgcv-Compatible Residual Oracle And Hybrid Decision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a restricted, version-pinned `mgcv` extraction oracle for `kpcalg::regrXonS()` compatibility, add residual/p-value/graph validation metrics, and introduce a near-alpha verifier policy that combines fastSpline CUDA throughput with legacy-compatible decisions.

**Architecture:** Keep `fastSpline` CUDA as the high-throughput approximate primary backend, and add a separate `mgcvExtractCPU` oracle family that depends on `mgcv` setup internals only for compatibility analysis and verification. The implementation first freezes legacy residual semantics and diagnostics, then builds fixed-smoothing-parameter extraction parity, then GCV bridge parity, then same-S multi-target CPU extraction, and only produces a CUDA design memo after CPU oracle evidence justifies it. Hybrid verification may replace p-values near alpha, but graph edge deletion and sepset selection must replay in canonical legacy order.

**Tech Stack:** R 4.4.1, `mgcv` 1.9-x, base R, optional `pcalg`/`graph`/`RSpectra`/`energy`/`kernlab` legacy dependencies, existing `fastkpc/R/legacy_runner.R`, `fastkpc/R/fastspline_validation.R`, `fastkpc/R/validation_campaign.R`, `fastkpc/R/native.R`, `fastkpc/R/cuda_native.R`, existing C++17/Rcpp/CUDA fastSpline and CI backends, and unchanged `kpcalg/R/*.R` legacy source files.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next fastkpc compatibility slice from docs/superpowers/plans/2026-06-16-fast-kpc-mgcv-compatible-residual-oracle-hybrid-goal-execution.md: freeze kpcalg::regrXonS residual semantics, add backend/setup/target fingerprinting, add residual/p-value/graph/WAN-PDAG compatibility campaign metrics, implement a version-pinned mgcvExtractFixedSP CPU oracle, add a mgcvExtractGCVBridge path, implement same-S multi-target CPU extraction diagnostics, add a near-alpha verifier policy over fastSpline CUDA that preserves canonical replay order, document the backend taxonomy and non-goals, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `900000`.

Do not mark the goal complete until every item in "Completion Criteria" is proven by current-state evidence. Mark the goal blocked only if the same local blocker repeats for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Baseline From Previous Goals

The current fastkpc implementation already includes:

```text
Native CPU dCov gamma and HSIC gamma/permutation CI methods
CUDA dCov CI batches
CUDA HSIC gamma and fixed-seed permutation CI batches
fastSpline residual backend on CPU
fastSpline residual device on CUDA
true-batched fastSpline CUDA residual groups
residual cache
level/layer scheduler diagnostics
WAN-PDAG opt-in native/CUDA paths
validation campaign/report/CLI infrastructure
fastSpline-vs-mgcv residual validation helpers
kpcalg/R files intentionally unchanged
```

This goal does not revisit the already-completed batching/cache/CUDA-fastSpline layer. It adds compatibility and decision-safety infrastructure above it.

## Current Empirical Context

The current `fastSpline` CUDA path matches CPU `fastSpline` at floating-point noise:

```text
CPU-vs-CUDA fastSpline residual max abs diff: about 1e-15
CPU-vs-CUDA fastSpline selected lambda: identical in fixed validation cases
CPU-vs-CUDA fastSpline RSS: identical in fixed validation cases
```

The current `fastSpline` path is close to but not equivalent to legacy `mgcv::gam()`:

```text
1D validation residual correlation: about 0.9895
1D validation relative residual L2: about 0.1446
1D dCov p-value absolute difference: about 0.1001

2D validation residual correlation: about 0.9792
2D validation relative residual L2: about 0.2036
2D dCov p-value absolute difference: about 0.00815
```

Therefore the project must not describe `fastSpline` as mgcv-compatible. It is a fast approximate backend.

## Backend Taxonomy

The public and internal documentation must distinguish four residual/backend families:

```text
legacy mgcv
  Direct kpcalg-compatible reference path through kpcalg/R and mgcv::gam().
  Slow, authoritative for compatibility testing, optional when legacy packages are installed.

fastSpline CUDA
  High-throughput approximate residual backend.
  Useful as primary fast path.
  Not mgcv-equivalent.

mgcvExtractCPU
  Version-pinned extraction oracle / bridge for compatibility validation.
  May depend on mgcv internals such as setup objects, model matrix, penalties,
  constraints, rank metadata, or magic-style inputs.
  Not the final product backend.

mgcvSubset
  Future restricted portable implementation of kpcalg::regrXonS residual semantics.
  Starts on CPU only after extraction oracle gates pass.
  CUDA design is deferred until CPU evidence justifies it.
```

Required diagnostic wording:

```text
fastSpline: "approximate"
mgcvExtractCPU: "version-pinned extraction oracle"
mgcvSubsetCPU: "restricted mgcv-compatible subset"
mgcvSubsetCUDA: "future design, not implemented in this goal"
```

## Non-Goals

- Do not implement a full `mgcv::gam()` clone.
- Do not implement `bamGPU`.
- Do not implement generic GAM acceleration.
- Do not implement non-Gaussian families.
- Do not implement `summary.gam`, `vcov`, standard errors, prediction intervals, ANOVA, or plotting compatibility.
- Do not implement GAMM.
- Do not implement by-smooth or factor-smooth support.
- Do not replace default `s(s1, s2)` with tensor-product semantics.
- Do not call `fastSpline` mgcv-equivalent.
- Do not share smoothing parameters across target variables.
- Do not change exported `kpcalg::kpc()`.
- Do not modify `kpcalg/R/*.R`.
- Do not initialize git if this workspace is not already a git repository.

## Critical Semantics Contract

### Direct CI Path

`|S| == 0` is not residualization:

```text
kernelCItest(x, y, S = integer(), suffStat)
  -> direct HSIC / dCov on x and y
  -> no regrXonS call
  -> no mgcvExtract residualization
```

### Conditional CI Path

`|S| > 0` uses `kpcalg::regrXonS()` semantics:

```text
kernelCItest(x, y, S, suffStat)
  -> residuals <- regrXonS(cbind(x, y), S)
  -> resx <- residuals[, 1]
  -> resy <- residuals[, 2]
  -> HSIC / dCov on residuals
```

### Residual Formula Semantics

The source of truth is `kpcalg/R/regrXonS.R`, not stale documentation:

```text
if |S| <= 2:
    X_i ~ s(S variables jointly)

if |S| > 2:
    X_i ~ s(S_1) + s(S_2) + ... + s(S_k)
```

Default `mgcv::s(s1, s2)` is an isotropic thin plate regression spline smooth. It is not a tensor-product smooth. Tensor-product smooths are `te`, `ti`, or `t2`, and are not part of the legacy `regrXonS()` formula construction.

### Per-Target Fit Invariant

`regrXonS(cbind(x, y), S)` performs separate `mgcv::gam()` fits:

```text
x ~ s(S)
y ~ s(S)
```

Each target has its own selected smoothing parameters, score, rank behavior, warnings, fitted values, and residuals. Same-S batching may reuse setup-level data, but it must not reuse one target's `sp` for another target.

## Fingerprinting Contract

Separate setup and target fingerprints. Do not put target-specific values into the setup fingerprint, or same-S setup reuse will be defeated.

### setup_fingerprint

Include values that should be shared for same conditioning setup:

```text
R_version
mgcv_version
fastkpc_version_or_source_hash
kpcalg_compatibility_mode
backend_family
backend_version
formula_class: full-smooth or additive-smooth
conditioning_set_as_set
conditioning_variable_order_used_in_formula
n
input_p
k
bs
method
optimizer
gamma
select
family
scale_setting
na_action
weights_policy
intercept_policy
model_matrix_hash_if_extracted
penalty_hashes_if_extracted
constraint_hash_if_extracted
rank_metadata_if_extracted
setup_warning_classes
```

### target_fingerprint

Include values that differ by target:

```text
target_variable_id
target_vector_hash
sp_input
sp_output
selected_sp
score
edf
rank_if_target_specific
target_warning_classes
residual_hash_optional_for_debug
fitted_hash_optional_for_debug
```

### CI replay identifiers

Every CI test diagnostic row must include:

```text
canonical_test_order_id
conditioning_level
x
y
S_order_used_for_test
S_set_key
alpha
ci_method
primary_backend
verifier_backend_if_any
```

Cache keys may normalize `S` as a set, but canonical replay order must preserve the legacy test and subset ordering.

## File Structure Plan

Create or modify these files. The exact file list can shrink if equivalent local helpers already exist, but the responsibilities must remain separated.

```text
fastkpc/R/mgcv_compat_contract.R
  Defines frozen kpcalg regrXonS semantics helpers, formula-class resolution,
  setup/target fingerprint helpers, and canonical diagnostic field helpers.

fastkpc/R/mgcv_extract_oracle.R
  Implements mgcvExtractFixedSP and mgcvExtractGCVBridge helpers.
  May depend on mgcv internals and must record mgcv version fingerprints.

fastkpc/R/mgcv_extract_validation.R
  Fixed scenarios and validation summaries for legacy mgcv, mgcvExtractFixedSP,
  mgcvExtractGCVBridge, and fastSpline comparisons.

fastkpc/R/hybrid_verifier.R
  Near-alpha trigger policy, p-value source selection, and canonical replay
  helpers over already-computed CI diagnostics.

fastkpc/R/validation_campaign.R
  Extend existing campaign outputs with compatibility metrics and hybrid fields.

fastkpc/R/fastspline_validation.R
  Reuse or lightly extend existing mgcv comparison helpers; do not mix
  fastSpline equivalence claims into mgcvExtract naming.

fastkpc/R/native.R
fastkpc/R/cuda_native.R
  Source new R helpers if the project pattern requires explicit source lines.

fastkpc/tests/test_mgcv_compat_contract.R
  Tests formula semantics, direct-CI vs residual path, and fingerprints.

fastkpc/tests/test_mgcv_extract_fixed_sp.R
  Tests fixed-sp extraction parity gates.

fastkpc/tests/test_mgcv_extract_gcv_bridge.R
  Tests GCV bridge diagnostics and practical parity where mgcv is available.

fastkpc/tests/test_mgcv_extract_batch_cpu.R
  Tests same-S multi-target extraction keeps per-target sp.

fastkpc/tests/test_hybrid_near_alpha_policy.R
  Tests log-scale near-alpha triggers, p-source replacement, and canonical order.

fastkpc/tests/test_compatibility_campaign_metrics.R
  Tests campaign artifacts include residual, CI, graph, WAN-PDAG, and hybrid fields.

fastkpc/tests/test_mgcv_compat_docs_contract.R
  Tests README/docs state taxonomy and non-goals honestly.

fastkpc/README.md
  Document backend taxonomy, mgcvExtractCPU scope, hybrid verifier policy,
  validation commands, and non-goals.
```

## Task 1: Freeze Legacy regrXonS Semantics Contract

**Files:**
- Create: `fastkpc/R/mgcv_compat_contract.R`
- Modify: `fastkpc/R/native.R`
- Modify: `fastkpc/R/cuda_native.R`
- Test: `fastkpc/tests/test_mgcv_compat_contract.R`

- [x] **Step 1: Write the failing contract test**

Create `fastkpc/tests/test_mgcv_compat_contract.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) {
  if (!isTRUE(value)) fail(message)
}
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

assert_equal(
  fastkpc_regrxons_formula_class(integer()),
  "direct-ci",
  "|S| == 0 must be direct CI, not residualization"
)

assert_equal(
  fastkpc_regrxons_formula_class(3L),
  "full-smooth",
  "|S| == 1 must use joint/full smooth"
)

assert_equal(
  fastkpc_regrxons_formula_class(c(3L, 4L)),
  "full-smooth",
  "|S| == 2 must use joint/full smooth"
)

assert_equal(
  fastkpc_regrxons_formula_class(c(3L, 4L, 5L)),
  "additive-smooth",
  "|S| > 2 must use additive smooth"
)

sem <- fastkpc_regrxons_semantics(c(4L, 2L), target = 1L, n = 20L, p = 5L)
assert_equal(sem$formula_class, "full-smooth", "formula class")
assert_equal(sem$conditioning_variable_order_used_in_formula, c(4L, 2L),
             "formula order must preserve caller order")
assert_equal(sem$conditioning_set_as_set, c(2L, 4L),
             "set key must be sorted independently")
assert_true(grepl("kpcalg_regrXonS_v1", sem$compatibility_mode),
            "compatibility mode must be explicit")

setup <- fastkpc_setup_fingerprint(sem, mgcv_version = "1.9-4",
                                   model_matrix_hash = "XHASH",
                                   penalty_hashes = c("S1", "S2"),
                                   constraint_hash = "CHASH",
                                   rank_metadata = "rank=7")
target_a <- fastkpc_target_fingerprint(target = 1L, y_hash = "YA",
                                       selected_sp = c(0.1, 1.0),
                                       score = 12.5, edf = 3.2)
target_b <- fastkpc_target_fingerprint(target = 2L, y_hash = "YB",
                                       selected_sp = c(0.2, 2.0),
                                       score = 13.5, edf = 4.2)

assert_true(!grepl("YA|YB|0.1|0.2", setup$fingerprint),
            "setup fingerprint must not contain target-specific values")
assert_true(!identical(target_a$fingerprint, target_b$fingerprint),
            "target fingerprint must differ across targets")

cat("PASS mgcv compatibility contract\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_contract.R
```

Expected:

```text
Error in source("fastkpc/R/mgcv_compat_contract.R") :
  cannot open file ...
```

or an equivalent failure because the contract file/functions do not exist.

- [x] **Step 3: Implement the minimal contract helpers**

Create `fastkpc/R/mgcv_compat_contract.R`:

```r
fastkpc_regrxons_formula_class <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("direct-ci")
  if (length(S) <= 2L) return("full-smooth")
  "additive-smooth"
}

fastkpc_regrxons_semantics <- function(S, target, n, p,
                                       compatibility_mode = "kpcalg_regrXonS_v1") {
  S <- as.integer(S)
  list(
    compatibility_mode = compatibility_mode,
    target = as.integer(target),
    formula_class = fastkpc_regrxons_formula_class(S),
    conditioning_variable_order_used_in_formula = S,
    conditioning_set_as_set = sort(unique(S)),
    n = as.integer(n),
    p = as.integer(p),
    family = "gaussian_identity",
    output = "residuals_only"
  )
}

fastkpc_hash_object <- function(x) {
  raw <- serialize(x, NULL, version = 2)
  paste(as.character(tools::md5sum(charToRaw(paste(raw, collapse = "")))), collapse = "")
}

fastkpc_collapse_key <- function(x) {
  if (length(x) == 0L) return("")
  paste(as.character(x), collapse = "|")
}

fastkpc_setup_fingerprint <- function(semantics,
                                      R_version = as.character(getRversion()),
                                      mgcv_version = if (requireNamespace("mgcv", quietly = TRUE)) {
                                        as.character(utils::packageVersion("mgcv"))
                                      } else {
                                        "unavailable"
                                      },
                                      backend_family = "mgcvExtractCPU",
                                      backend_version = "v1",
                                      k = NA_integer_,
                                      bs = "tp",
                                      method = "GCV.Cp",
                                      optimizer = NA_character_,
                                      gamma = 1,
                                      select = FALSE,
                                      scale_setting = "mgcv-default",
                                      na_action = "na.fail",
                                      weights_policy = "none",
                                      intercept_policy = "mgcv-default",
                                      model_matrix_hash = "",
                                      penalty_hashes = character(),
                                      constraint_hash = "",
                                      rank_metadata = "",
                                      setup_warning_classes = character()) {
  fields <- list(
    R_version = R_version,
    mgcv_version = mgcv_version,
    backend_family = backend_family,
    backend_version = backend_version,
    compatibility_mode = semantics$compatibility_mode,
    formula_class = semantics$formula_class,
    conditioning_set_as_set = fastkpc_collapse_key(semantics$conditioning_set_as_set),
    conditioning_variable_order_used_in_formula =
      fastkpc_collapse_key(semantics$conditioning_variable_order_used_in_formula),
    n = semantics$n,
    p = semantics$p,
    k = k,
    bs = bs,
    method = method,
    optimizer = optimizer,
    gamma = gamma,
    select = select,
    family = semantics$family,
    scale_setting = scale_setting,
    na_action = na_action,
    weights_policy = weights_policy,
    intercept_policy = intercept_policy,
    model_matrix_hash = model_matrix_hash,
    penalty_hashes = fastkpc_collapse_key(penalty_hashes),
    constraint_hash = constraint_hash,
    rank_metadata = rank_metadata,
    setup_warning_classes = fastkpc_collapse_key(setup_warning_classes)
  )
  list(fields = fields, fingerprint = fastkpc_hash_object(fields))
}

fastkpc_target_fingerprint <- function(target, y_hash, sp_input = NULL,
                                       sp_output = NULL, selected_sp = NULL,
                                       score = NA_real_, edf = NA_real_,
                                       rank_if_target_specific = NA,
                                       target_warning_classes = character(),
                                       residual_hash = "",
                                       fitted_hash = "") {
  fields <- list(
    target_variable_id = as.integer(target),
    target_vector_hash = y_hash,
    sp_input = sp_input,
    sp_output = sp_output,
    selected_sp = selected_sp,
    score = score,
    edf = edf,
    rank_if_target_specific = rank_if_target_specific,
    target_warning_classes = fastkpc_collapse_key(target_warning_classes),
    residual_hash = residual_hash,
    fitted_hash = fitted_hash
  )
  list(fields = fields, fingerprint = fastkpc_hash_object(fields))
}
```

- [x] **Step 4: Source the contract file from public R wrappers if needed**

If `fastkpc/R/native.R` and `fastkpc/R/cuda_native.R` are used as source entrypoints in tests, add near the top of each file:

```r
if (file.exists("fastkpc/R/mgcv_compat_contract.R")) {
  source("fastkpc/R/mgcv_compat_contract.R")
}
```

Do not add this line if another central source loader already handles new R files.

- [x] **Step 5: Run the contract test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_contract.R
```

Expected:

```text
PASS mgcv compatibility contract
```

## Task 2: Add Compatibility Campaign Metrics Skeleton

**Files:**
- Create: `fastkpc/R/mgcv_extract_validation.R`
- Modify: `fastkpc/R/validation_campaign.R`
- Test: `fastkpc/tests/test_compatibility_campaign_metrics.R`

- [x] **Step 1: Write the failing metrics test**

Create `fastkpc/tests/test_compatibility_campaign_metrics.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

sample <- fastkpc_empty_compatibility_campaign_metrics()
required_residual <- c("scenario", "target", "S_key", "backend",
                       "residual_correlation", "relative_l2",
                       "max_abs_diff", "mean_diff", "sd_ratio",
                       "selected_sp", "edf", "score",
                       "setup_fingerprint", "target_fingerprint")
required_ci <- c("canonical_test_order_id", "x", "y", "S_key",
                 "conditioning_level", "p_legacy", "p_backend",
                 "log_p_ratio", "decision_legacy", "decision_backend",
                 "decision_flip", "distance_to_alpha_log",
                 "backend_used", "fallback_triggered", "verifier_backend")
required_graph <- c("scenario", "backend", "skeleton_shd",
                    "skeleton_precision", "skeleton_recall", "skeleton_f1",
                    "edge_deletion_mismatch", "sepset_mismatch_rate",
                    "first_separating_set_mismatch",
                    "wanpdag_orientation_mismatch",
                    "arrowhead_agreement", "near_alpha_tests",
                    "verifier_calls", "verifier_decision_changes")

assert_true(all(required_residual %in% names(sample$residual)),
            "residual metric columns missing")
assert_true(all(required_ci %in% names(sample$ci)),
            "CI metric columns missing")
assert_true(all(required_graph %in% names(sample$graph)),
            "graph metric columns missing")

p <- fastkpc_log_distance_to_alpha(p = 0.10, alpha = 0.05)
assert_true(abs(p - log(2)) < 1e-12, "log alpha distance must be log(p/alpha)")

cat("PASS compatibility campaign metrics\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_compatibility_campaign_metrics.R
```

Expected: failure because `fastkpc/R/mgcv_extract_validation.R` or metric helpers do not exist.

- [x] **Step 3: Implement empty metric schemas and helpers**

Create `fastkpc/R/mgcv_extract_validation.R`:

```r
fastkpc_log_distance_to_alpha <- function(p, alpha) {
  p <- max(as.numeric(p), .Machine$double.xmin)
  alpha <- max(as.numeric(alpha), .Machine$double.xmin)
  log(p / alpha)
}

fastkpc_empty_residual_compatibility_metrics <- function() {
  data.frame(
    scenario = character(),
    target = integer(),
    S_key = character(),
    backend = character(),
    residual_correlation = numeric(),
    relative_l2 = numeric(),
    max_abs_diff = numeric(),
    mean_diff = numeric(),
    sd_ratio = numeric(),
    selected_sp = character(),
    edf = numeric(),
    score = numeric(),
    setup_fingerprint = character(),
    target_fingerprint = character(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_ci_compatibility_metrics <- function() {
  data.frame(
    canonical_test_order_id = integer(),
    x = integer(),
    y = integer(),
    S_key = character(),
    conditioning_level = integer(),
    p_legacy = numeric(),
    p_backend = numeric(),
    log_p_ratio = numeric(),
    decision_legacy = logical(),
    decision_backend = logical(),
    decision_flip = logical(),
    distance_to_alpha_log = numeric(),
    backend_used = character(),
    fallback_triggered = logical(),
    verifier_backend = character(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_graph_compatibility_metrics <- function() {
  data.frame(
    scenario = character(),
    backend = character(),
    skeleton_shd = integer(),
    skeleton_precision = numeric(),
    skeleton_recall = numeric(),
    skeleton_f1 = numeric(),
    edge_deletion_mismatch = integer(),
    sepset_mismatch_rate = numeric(),
    first_separating_set_mismatch = integer(),
    wanpdag_orientation_mismatch = integer(),
    arrowhead_agreement = numeric(),
    near_alpha_tests = integer(),
    verifier_calls = integer(),
    verifier_decision_changes = integer(),
    stringsAsFactors = FALSE
  )
}

fastkpc_empty_compatibility_campaign_metrics <- function() {
  list(
    residual = fastkpc_empty_residual_compatibility_metrics(),
    ci = fastkpc_empty_ci_compatibility_metrics(),
    graph = fastkpc_empty_graph_compatibility_metrics()
  )
}
```

- [x] **Step 4: Source validation helpers from campaign code if needed**

If `fastkpc/R/validation_campaign.R` is the central campaign entrypoint, add near its top:

```r
if (file.exists("fastkpc/R/mgcv_extract_validation.R")) {
  source("fastkpc/R/mgcv_extract_validation.R")
}
```

- [x] **Step 5: Run the metrics test**

Run:

```bash
Rscript fastkpc/tests/test_compatibility_campaign_metrics.R
```

Expected:

```text
PASS compatibility campaign metrics
```

## Task 3: Implement mgcvExtractFixedSP Oracle

**Files:**
- Create: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/R/mgcv_extract_validation.R`
- Test: `fastkpc/tests/test_mgcv_extract_fixed_sp.R`

This task may use `mgcv` internals. It must record the exact `mgcv` version and must skip with a clear message when `mgcv` is unavailable.

- [x] **Step 1: Write the fixed-sp test**

Create `fastkpc/tests/test_mgcv_extract_fixed_sp.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(2101)
n <- 80
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
y <- sin(s1) + cos(s2) + stats::rnorm(n, sd = 0.1)
data <- data.frame(y = y, s1 = s1, s2 = s2)

legacy <- mgcv::gam(y ~ s(s1, s2), data = data, method = "GCV.Cp")
fixed <- fastkpc_mgcv_extract_fixed_sp(
  formula = y ~ s(s1, s2),
  data = data,
  sp = legacy$sp,
  method = "GCV.Cp"
)

assert_true(identical(fixed$backend_family, "mgcvExtractCPU"),
            "backend family must be mgcvExtractCPU")
assert_true(identical(fixed$mode, "fixed-sp"),
            "mode must be fixed-sp")
assert_true(length(fixed$residuals) == n, "residual length")
assert_true(max(abs(fixed$residuals - stats::residuals(legacy))) < 1e-6,
            "fixed-sp residuals must match legacy practical tolerance")
assert_true(max(abs(fixed$fitted - stats::fitted(legacy))) < 1e-6,
            "fixed-sp fitted values must match legacy practical tolerance")
assert_true(nchar(fixed$setup_fingerprint$fingerprint) > 0,
            "setup fingerprint required")
assert_true(nchar(fixed$target_fingerprint$fingerprint) > 0,
            "target fingerprint required")

cat("PASS mgcv extract fixed-sp\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected: failure because `fastkpc/R/mgcv_extract_oracle.R` or `fastkpc_mgcv_extract_fixed_sp()` does not exist.

- [x] **Step 3: Implement a conservative fixed-sp bridge**

Create `fastkpc/R/mgcv_extract_oracle.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")

fastkpc_require_mgcv <- function() {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for mgcvExtractCPU", call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_mgcv_hash_numeric <- function(x) {
  fastkpc_hash_object(round(as.numeric(x), digits = 14))
}

fastkpc_mgcv_extract_fixed_sp <- function(formula, data, sp,
                                          method = "GCV.Cp",
                                          target = 1L,
                                          S = integer(),
                                          k = NA_integer_,
                                          bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)

  fit <- mgcv::gam(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    fit = TRUE
  )

  residuals <- as.numeric(stats::residuals(fit))
  fitted <- as.numeric(stats::fitted(fit))
  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(residuals), p = ncol(data))
  setup <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "fixed-sp-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = if (!is.null(fit$model)) fastkpc_hash_object(names(fit$model)) else "",
    penalty_hashes = fastkpc_hash_object(fit$sp),
    constraint_hash = "",
    rank_metadata = paste0("rank=", fit$rank)
  )
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(model.response(model.frame(fit))),
    sp_input = sp,
    sp_output = fit$sp,
    selected_sp = fit$sp,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank_if_target_specific = fit$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp",
    formula = formula,
    method = method,
    sp = fit$sp,
    residuals = residuals,
    fitted = fitted,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank = fit$rank,
    setup_fingerprint = setup,
    target_fingerprint = target_fp,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )
}
```

This implementation is a bridge, not a final independent solver. If the project later accesses `gam(..., fit=FALSE)` and `magic()` directly, keep this API stable and improve internals behind it.

- [x] **Step 4: Run the fixed-sp test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
PASS mgcv extract fixed-sp
```

- [x] **Step 5: Record the Gate B interpretation in comments/docs**

Add this note to `fastkpc/R/mgcv_extract_oracle.R` above `fastkpc_mgcv_extract_fixed_sp()`:

```r
# Gate B only proves that the extraction bridge can reproduce residuals when
# mgcv supplies setup/sp semantics. It does not prove that fastkpc can construct
# mgcv's basis, penalties, constraints, rank behavior, or optimizer independently.
```

## Task 4: Implement mgcvExtractGCVBridge

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Test: `fastkpc/tests/test_mgcv_extract_gcv_bridge.R`

- [x] **Step 1: Write the GCV bridge test**

Create `fastkpc/tests/test_mgcv_extract_gcv_bridge.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(2102)
n <- 90
s1 <- stats::runif(n, -2, 2)
y <- sin(s1) + stats::rnorm(n, sd = 0.1)
data <- data.frame(y = y, s1 = s1)

legacy <- mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")
bridge <- fastkpc_mgcv_extract_gcv_bridge(
  formula = y ~ s(s1),
  data = data,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)

assert_true(identical(bridge$backend_family, "mgcvExtractCPU"),
            "backend family")
assert_true(identical(bridge$mode, "gcv-bridge"),
            "mode")
assert_true(max(abs(bridge$residuals - stats::residuals(legacy))) < 1e-6,
            "GCV bridge residuals should match direct legacy fit")
assert_true(max(abs(log(bridge$sp) - log(legacy$sp))) < 1e-8,
            "GCV bridge selected sp should match direct legacy fit")

cat("PASS mgcv extract GCV bridge\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
```

Expected: failure because `fastkpc_mgcv_extract_gcv_bridge()` does not exist.

- [x] **Step 3: Implement the bridge**

Append to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_mgcv_extract_gcv_bridge <- function(formula, data,
                                            method = "GCV.Cp",
                                            target = 1L,
                                            S = integer(),
                                            k = NA_integer_,
                                            bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  legacy <- mgcv::gam(formula = formula, data = data, method = method)
  fixed <- fastkpc_mgcv_extract_fixed_sp(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  fixed$mode <- "gcv-bridge"
  fixed$legacy_score <- if (!is.null(legacy$gcv.ubre)) as.numeric(legacy$gcv.ubre) else NA_real_
  fixed$legacy_edf <- if (!is.null(legacy$edf)) sum(legacy$edf) else NA_real_
  fixed$legacy_rank <- legacy$rank
  fixed
}
```

- [x] **Step 4: Run the GCV bridge test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
```

Expected:

```text
PASS mgcv extract GCV bridge
```

- [x] **Step 5: Record the Gate C split**

Add a short comment near the GCV bridge:

```r
# mgcvExtractGCVBridge may call mgcv to select smoothing parameters. A future
# mgcvPortGCVPrototype must be validated separately and should not inherit this
# bridge's strict parity gate until optimizer details are implemented.
```

## Task 5: Implement Same-S Multi-Target CPU Extraction

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Test: `fastkpc/tests/test_mgcv_extract_batch_cpu.R`

- [x] **Step 1: Write the batch CPU test**

Create `fastkpc/tests/test_mgcv_extract_batch_cpu.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(2103)
n <- 100
s1 <- stats::runif(n, -2, 2)
x <- sin(s1) + stats::rnorm(n, sd = 0.1)
y <- cos(s1) + stats::rnorm(n, sd = 0.1)
data <- data.frame(x = x, y = y, s1 = s1)

batch <- fastkpc_mgcv_extract_batch(
  Y = as.matrix(data[, c("x", "y")]),
  S_data = data.frame(s1 = s1),
  S = 3L,
  target_ids = c(1L, 2L),
  formula_class = "full-smooth",
  method = "GCV.Cp"
)

assert_true(is.matrix(batch$residuals), "residuals must be matrix")
assert_true(all(dim(batch$residuals) == c(n, 2L)), "residual matrix shape")
assert_true(length(batch$sp) == 2L, "sp must be per target")
assert_true(length(unique(vapply(batch$target_fingerprints, `[[`, character(1), "fingerprint"))) == 2L,
            "target fingerprints must differ")
assert_true(nchar(batch$setup_fingerprint$fingerprint) > 0,
            "shared setup fingerprint required")

legacy_x <- mgcv::gam(x ~ s(s1), data = data, method = "GCV.Cp")
legacy_y <- mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")

assert_true(max(abs(batch$residuals[, 1] - stats::residuals(legacy_x))) < 1e-6,
            "x residuals")
assert_true(max(abs(batch$residuals[, 2] - stats::residuals(legacy_y))) < 1e-6,
            "y residuals")

cat("PASS mgcv extract batch CPU\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
```

Expected: failure because batch extraction does not exist.

- [x] **Step 3: Implement batch extraction using per-target fits**

Append to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_mgcv_extract_batch <- function(Y, S_data, S,
                                       target_ids = seq_len(ncol(Y)),
                                       formula_class = NULL,
                                       method = "GCV.Cp") {
  fastkpc_require_mgcv()
  Y <- as.matrix(Y)
  S_data <- as.data.frame(S_data)
  if (is.null(colnames(Y))) colnames(Y) <- paste0("target", seq_len(ncol(Y)))
  if (is.null(colnames(S_data))) colnames(S_data) <- paste0("s", seq_len(ncol(S_data)))
  if (is.null(formula_class)) formula_class <- fastkpc_regrxons_formula_class(S)

  n <- nrow(Y)
  q <- ncol(Y)
  residuals <- matrix(NA_real_, n, q)
  fitted <- matrix(NA_real_, n, q)
  sp <- vector("list", q)
  score <- rep(NA_real_, q)
  edf <- rep(NA_real_, q)
  ranks <- rep(NA_integer_, q)
  target_fps <- vector("list", q)

  rhs <- if (identical(formula_class, "additive-smooth")) {
    paste(sprintf("s(%s)", colnames(S_data)), collapse = " + ")
  } else {
    sprintf("s(%s)", paste(colnames(S_data), collapse = ", "))
  }

  sem <- fastkpc_regrxons_semantics(S = S, target = target_ids[1],
                                    n = n, p = q + ncol(S_data))
  setup_fp <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "batch-gcv-bridge-v1",
    method = method,
    model_matrix_hash = fastkpc_hash_object(list(S_data = S_data, rhs = rhs))
  )

  for (j in seq_len(q)) {
    local_data <- cbind(data.frame(.target = Y[, j]), S_data)
    form <- stats::as.formula(paste(".target ~", rhs))
    fit <- fastkpc_mgcv_extract_gcv_bridge(
      formula = form,
      data = local_data,
      method = method,
      target = target_ids[j],
      S = S
    )
    residuals[, j] <- fit$residuals
    fitted[, j] <- fit$fitted
    sp[[j]] <- fit$sp
    score[j] <- fit$score
    edf[j] <- fit$edf
    ranks[j] <- fit$rank
    target_fps[[j]] <- fit$target_fingerprint
  }

  colnames(residuals) <- colnames(Y)
  colnames(fitted) <- colnames(Y)
  list(
    backend_family = "mgcvExtractCPU",
    mode = "batch-gcv-bridge",
    residuals = residuals,
    fitted = fitted,
    sp = sp,
    score = score,
    edf = edf,
    rank = ranks,
    setup_fingerprint = setup_fp,
    target_fingerprints = target_fps,
    formula_class = formula_class,
    method = method
  )
}
```

This is an intentionally conservative bridge. It establishes API shape and diagnostics before optimizing setup reuse. Later work may reuse extracted setup more aggressively behind the same return contract.

- [x] **Step 4: Run the batch CPU test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
```

Expected:

```text
PASS mgcv extract batch CPU
```

## Task 6: Add Near-Alpha Verifier Policy

**Files:**
- Create: `fastkpc/R/hybrid_verifier.R`
- Test: `fastkpc/tests/test_hybrid_near_alpha_policy.R`

- [x] **Step 1: Write the policy test**

Create `fastkpc/tests/test_hybrid_near_alpha_policy.R`:

```r
source("fastkpc/R/hybrid_verifier.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) fail(message)
}

policy <- fastkpc_hybrid_policy(alpha = 0.05, tau = log(3),
                                primary = "fastSplineCUDA",
                                verifier = "mgcvExtractCPU")

assert_true(fastkpc_near_alpha(0.05, policy), "alpha itself must trigger")
assert_true(fastkpc_near_alpha(0.05 / 3, policy), "lower band must trigger")
assert_true(fastkpc_near_alpha(0.05 * 3, policy), "upper band must trigger")
assert_true(!fastkpc_near_alpha(0.001, policy), "far lower p must not trigger")
assert_true(!fastkpc_near_alpha(0.9, policy), "far upper p must not trigger")

tests <- data.frame(
  canonical_test_order_id = c(3L, 1L, 2L),
  primary_p = c(0.9, 0.051, 0.001),
  verifier_p = c(NA_real_, 0.20, NA_real_),
  stringsAsFactors = FALSE
)
resolved <- fastkpc_apply_hybrid_policy(tests, policy)
assert_equal(resolved$canonical_test_order_id, c(3L, 1L, 2L),
             "policy must preserve input/canonical replay order")
assert_true(resolved$near_alpha_triggered[2], "second row near alpha")
assert_equal(resolved$p_source_used[2], "mgcvExtractCPU", "verifier source used")
assert_true(abs(resolved$p_used[2] - 0.20) < 1e-12, "verifier p used")
assert_equal(resolved$p_source_used[1], "fastSplineCUDA", "primary source used")

cat("PASS hybrid near-alpha policy\n")
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_near_alpha_policy.R
```

Expected: failure because `fastkpc/R/hybrid_verifier.R` does not exist.

- [x] **Step 3: Implement the policy helpers**

Create `fastkpc/R/hybrid_verifier.R`:

```r
fastkpc_hybrid_policy <- function(enabled = TRUE,
                                  alpha = 0.05,
                                  tau = log(3),
                                  primary = "fastSplineCUDA",
                                  verifier = "mgcvExtractCPU",
                                  always_verify_nan = TRUE,
                                  always_verify_boundary = TRUE) {
  list(
    enabled = isTRUE(enabled),
    alpha = as.numeric(alpha),
    tau = as.numeric(tau),
    primary = as.character(primary),
    verifier = as.character(verifier),
    always_verify_nan = isTRUE(always_verify_nan),
    always_verify_boundary = isTRUE(always_verify_boundary)
  )
}

fastkpc_near_alpha <- function(p, policy) {
  if (!isTRUE(policy$enabled)) return(FALSE)
  if (!is.finite(p)) return(isTRUE(policy$always_verify_nan))
  p <- max(as.numeric(p), .Machine$double.xmin)
  alpha <- max(as.numeric(policy$alpha), .Machine$double.xmin)
  abs(log(p / alpha)) <= policy$tau + 1e-12
}

fastkpc_apply_hybrid_policy <- function(test_rows, policy) {
  out <- as.data.frame(test_rows, stringsAsFactors = FALSE)
  out$near_alpha_triggered <- vapply(out$primary_p, fastkpc_near_alpha,
                                    logical(1), policy = policy)
  has_verifier <- "verifier_p" %in% names(out) & is.finite(out$verifier_p)
  use_verifier <- out$near_alpha_triggered & has_verifier
  out$p_used <- out$primary_p
  out$p_used[use_verifier] <- out$verifier_p[use_verifier]
  out$p_source_used <- policy$primary
  out$p_source_used[use_verifier] <- policy$verifier
  out$decision_before_verify <- out$primary_p > policy$alpha
  out$decision_after_verify <- out$p_used > policy$alpha
  out$verification_reason <- ""
  out$verification_reason[out$near_alpha_triggered] <- "near-alpha"
  out
}
```

- [x] **Step 4: Run the policy test**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_near_alpha_policy.R
```

Expected:

```text
PASS hybrid near-alpha policy
```

## Task 7: Integrate Hybrid Diagnostics Into Campaign Outputs

**Files:**
- Modify: `fastkpc/R/validation_campaign.R`
- Modify: `fastkpc/R/mgcv_extract_validation.R`
- Test: `fastkpc/tests/test_compatibility_campaign_metrics.R`
- Test: `fastkpc/tests/test_hybrid_near_alpha_policy.R`

- [x] **Step 1: Extend CI metric rows with hybrid fields**

In `fastkpc/R/mgcv_extract_validation.R`, add:

```r
fastkpc_make_ci_compatibility_row <- function(canonical_test_order_id,
                                              x, y, S,
                                              conditioning_level,
                                              p_legacy,
                                              p_backend,
                                              alpha,
                                              backend_used,
                                              fallback_triggered = FALSE,
                                              verifier_backend = "") {
  p_legacy_safe <- max(as.numeric(p_legacy), .Machine$double.xmin)
  p_backend_safe <- max(as.numeric(p_backend), .Machine$double.xmin)
  data.frame(
    canonical_test_order_id = as.integer(canonical_test_order_id),
    x = as.integer(x),
    y = as.integer(y),
    S_key = paste(sort(as.integer(S)), collapse = "|"),
    conditioning_level = as.integer(conditioning_level),
    p_legacy = as.numeric(p_legacy),
    p_backend = as.numeric(p_backend),
    log_p_ratio = log(p_backend_safe / p_legacy_safe),
    decision_legacy = as.numeric(p_legacy) > alpha,
    decision_backend = as.numeric(p_backend) > alpha,
    decision_flip = (as.numeric(p_legacy) > alpha) != (as.numeric(p_backend) > alpha),
    distance_to_alpha_log = fastkpc_log_distance_to_alpha(p_backend, alpha),
    backend_used = as.character(backend_used),
    fallback_triggered = isTRUE(fallback_triggered),
    verifier_backend = as.character(verifier_backend),
    stringsAsFactors = FALSE
  )
}
```

- [x] **Step 2: Add a row-level test to the existing metrics test**

Append to `fastkpc/tests/test_compatibility_campaign_metrics.R`:

```r
row <- fastkpc_make_ci_compatibility_row(
  canonical_test_order_id = 7L,
  x = 1L,
  y = 2L,
  S = c(4L, 3L),
  conditioning_level = 2L,
  p_legacy = 0.04,
  p_backend = 0.08,
  alpha = 0.05,
  backend_used = "fastSplineCUDA",
  fallback_triggered = TRUE,
  verifier_backend = "mgcvExtractCPU"
)
assert_true(row$decision_flip, "decision flip should be TRUE")
assert_true(identical(row$S_key, "3|4"), "S key should be sorted")
```

- [x] **Step 3: Run campaign metrics tests**

Run:

```bash
Rscript fastkpc/tests/test_compatibility_campaign_metrics.R
Rscript fastkpc/tests/test_hybrid_near_alpha_policy.R
```

Expected:

```text
PASS compatibility campaign metrics
PASS hybrid near-alpha policy
```

- [x] **Step 4: Wire schemas into validation campaign artifacts**

Update `fastkpc/R/validation_campaign.R` so compact campaigns can include an optional `compatibility` list:

```r
compatibility <- fastkpc_empty_compatibility_campaign_metrics()
```

When no compatibility run is requested, write empty data frames with the required columns rather than omitting the section.

## Task 8: Add Documentation Contract

**Files:**
- Modify: `fastkpc/README.md`
- Test: `fastkpc/tests/test_mgcv_compat_docs_contract.R`

- [x] **Step 1: Write the docs contract test**

Create `fastkpc/tests/test_mgcv_compat_docs_contract.R`:

```r
text <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")

fail <- function(message) stop(message, call. = FALSE)
assert_grepl <- function(pattern, message) {
  if (!grepl(pattern, text, fixed = TRUE)) fail(message)
}

assert_grepl("fastSpline CUDA is a high-throughput approximate backend",
             "README must describe fastSpline CUDA as approximate")
assert_grepl("mgcvExtractCPU is a version-pinned extraction oracle",
             "README must describe mgcvExtractCPU as oracle")
assert_grepl("No full mgcv clone",
             "README must state full mgcv clone non-goal")
assert_grepl("No bamGPU",
             "README must state bamGPU non-goal")
assert_grepl("default s(s1, s2) is not a tensor-product smooth",
             "README must state smooth semantics")
assert_grepl("near-alpha verifier",
             "README must document hybrid verifier")

cat("PASS mgcv compatibility docs contract\n")
```

- [x] **Step 2: Run the docs test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R
```

Expected: failure until README text is added.

- [x] **Step 3: Add README section**

Add a section to `fastkpc/README.md`:

```markdown
## mgcv-Compatible Residual Oracle And Hybrid Verification

fastSpline CUDA is a high-throughput approximate backend. It is useful as the
primary fast path for batched kPC conditional-independence workloads, but it is
not mgcv-equivalent and graph differences from legacy `mgcv::gam()` residuals
can occur.

mgcvExtractCPU is a version-pinned extraction oracle for compatibility
validation. It is restricted to the `kpcalg::regrXonS()` residualization
surface, may depend on `mgcv` internals, and is not the final portable product
backend.

The restricted compatibility target is:

```text
|S| == 0: direct CI test; no residualization
|S| <= 2: X_i ~ s(S variables jointly)
|S| > 2:  X_i ~ s(S_1) + s(S_2) + ... + s(S_k)
family: Gaussian identity
output: residuals only
```

The default `s(s1, s2)` is not a tensor-product smooth. It is mgcv's default
isotropic smooth semantics. Tensor-product smooths such as `te`, `ti`, or `t2`
are not part of legacy `kpcalg::regrXonS()` formula construction.

The near-alpha verifier runs a fast primary backend first and verifies tests
whose p-values are close to alpha on a log scale. Verification may replace the
p-value source, but it must preserve canonical edge and sepset replay order.

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
```

- [x] **Step 4: Run the docs test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R
```

Expected:

```text
PASS mgcv compatibility docs contract
```

## Task 9: Add Validation Commands And Completion Audit

**Files:**
- Modify: `fastkpc/README.md`
- Modify: this plan if implementation learns a stricter command is needed

- [x] **Step 1: Run new focused tests**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_contract.R
Rscript fastkpc/tests/test_compatibility_campaign_metrics.R
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
Rscript fastkpc/tests/test_hybrid_near_alpha_policy.R
Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R
```

Expected:

```text
All tests print PASS or SKIP for missing optional mgcv/legacy packages.
No test fails.
```

- [x] **Step 2: Run existing related regression tests**

Run:

```bash
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_validation_campaign_smoke.R
Rscript fastkpc/tests/test_fastkpc_legacy_diagnostics.R
Rscript fastkpc/tests/test_wanpdag_legacy_validation.R
```

Expected:

```text
Existing tests pass or skip only because optional legacy packages are unavailable.
```

- [x] **Step 3: Run CUDA smoke only if CUDA native build is already available**

If CUDA native artifacts are available, run:

```bash
Rscript fastkpc/tests/test_cuda_residual_device_skeleton.R
Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
Rscript fastkpc/tests/test_hsic_cuda_skeleton_backend.R
```

Expected:

```text
CUDA tests pass.
No new mgcv compatibility helper changes CUDA backend truthfulness diagnostics.
```

- [x] **Step 4: Verify kpcalg/R remains unchanged**

Run:

```bash
cd kpcalg && md5sum -c MD5 | rg '^R/'
```

Expected:

```text
All kpcalg/R files report OK.
```

If the workspace lacks `rg`, use:

```bash
cd kpcalg && md5sum -c MD5 | grep '^R/'
```

## Completion Criteria

The goal is complete only when all of these are true:

```text
1. The regrXonS compatibility contract exists and distinguishes |S| == 0 direct CI from |S| > 0 residualization.
2. setup_fingerprint and target_fingerprint are separate and tested.
3. Formula semantics follow source code: |S| <= 2 full/joint smooth, |S| > 2 additive smooth.
4. Documentation states default s(s1, s2) is not a tensor-product smooth.
5. mgcvExtractFixedSP exists and records version-pinned oracle diagnostics.
6. Gate B wording is documented: fixed-sp parity on mgcv-provided setup does not prove portable mgcv basis construction.
7. mgcvExtractGCVBridge exists separately from any future mgcvPortGCVPrototype.
8. Same-S multi-target CPU extraction exists and keeps per-target smoothing parameters.
9. Compatibility campaign schemas include residual-level, CI-test-level, graph-level, WAN-PDAG/hybrid fields.
10. Near-alpha verifier policy exists and preserves canonical replay order.
11. Hybrid diagnostics expose primary_p, verifier_p, p_source_used, near_alpha_triggered, decision_before_verify, decision_after_verify, canonical_test_order_id, backend_primary, backend_verifier, and verification_reason where applicable.
12. README documents backend taxonomy, non-goals, validation commands, and hybrid policy.
13. New focused tests pass or explicitly skip only for missing optional packages.
14. Existing fastSpline/mgcv validation and campaign smoke tests still pass.
15. kpcalg/R MD5 audit passes.
```

## Stop Conditions

Stop and debug before proceeding if any of these happen:

```text
1. Fixed-sp extraction cannot reproduce legacy residuals on simple 1D or 2D Gaussian smooths.
2. setup_fingerprint changes when only the target vector changes.
3. same-S batch extraction reuses selected sp across targets.
4. near-alpha verification changes graph state in result-arrival order instead of canonical replay order.
5. README or diagnostics imply fastSpline is mgcv-equivalent.
6. Any implementation requires modifying kpcalg/R files.
```

## Future Goals After This Plan

Do not implement these in this goal. Write separate plans after the oracle/hybrid evidence exists:

```text
1. mgcvPortCPU-v1:
   progressively port restricted basis/penalty construction away from mgcv internals.

2. mgcvPortGCVPrototype:
   implement restricted Gaussian GCV optimizer and compare against mgcvExtractGCVBridge.

3. mgcvSubsetCUDA design:
   design CUDA batching only after CPU oracle shows useful graph-stability benefit.

4. hybrid graph replay integration:
   integrate verifier p-value replacement into the native/CUDA scheduler replay layer if
   the R-level policy and campaign metrics prove the approach.
```

## Self-Review Checklist For Plan Executor

Before marking this goal complete, check:

```text
No unresolved placeholder markers were introduced.
No code path describes fastSpline as mgcv-equivalent.
No test requires pcalg when it can skip cleanly.
No optional mgcv-dependent test fails hard when mgcv is unavailable.
No target-specific field is included in setup_fingerprint.
No smoothing parameter is shared across targets.
No kpcalg/R file changed.
```
