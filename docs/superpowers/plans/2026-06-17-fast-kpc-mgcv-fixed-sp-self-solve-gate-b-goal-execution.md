# Fast kPC mgcv Fixed-SP Self-Solve Gate B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current `mgcvExtractFixedSP` bridge from a second `mgcv::gam()` refit into a real Gate B self-solve: use `mgcv::gam(fit = FALSE)` setup components, assemble the fixed smoothing-parameter penalized Gaussian system inside fastkpc, solve it without calling `mgcv::gam()` for fitted values, and prove fitted/residual parity against a version-pinned mgcv fixed-sp reference.

**Architecture:** Keep the existing mgcv-based bridge as an explicit reference function, add a separate setup extraction layer, add standalone penalty assembly and constrained Gaussian fixed-sp solve helpers, then route the new `fastkpc_mgcv_extract_fixed_sp_solve()` through those helpers. The old `fastkpc_mgcv_extract_fixed_sp()` remains as a compatibility alias during this goal, while tests and diagnostics make it impossible to confuse the mgcv refit reference with fastkpc's self-solve Gate B.

**Tech Stack:** R 4.4.x, `mgcv` 1.9-x, base R matrix algebra, existing `fastkpc/R/mgcv_compat_contract.R`, existing `fastkpc/R/mgcv_extract_oracle.R`, existing mgcv compatibility tests, optional legacy `kpcalg`/`pcalg` dependencies for later graph-level checks, and unchanged `kpcalg/R/*.R` legacy source files.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next fastkpc mgcv compatibility slice from docs/superpowers/plans/2026-06-17-fast-kpc-mgcv-fixed-sp-self-solve-gate-b-goal-execution.md: preserve the existing mgcv fixed-sp refit bridge under an honest reference name, add mgcv setup extraction from gam(fit=FALSE), implement standalone penalty assembly for G$S/G$off/H, implement a restricted Gaussian identity fixed-sp constrained self-solve, add fastkpc_mgcv_extract_fixed_sp_solve() as the real Gate B oracle, update GCVBridge to consume the self-solve while marking sp_source/gcv_source honestly, add strict fixed-sp parity tests across |S|=1, |S|=2, and additive |S|=3 cases, add failure diagnostics for penalty/off/constraint/rank drift, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `650000`.

Do not mark the goal complete until every item in "Completion Criteria" is proven by fresh command output. Mark the goal blocked only if the same local blocker repeats for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Baseline From Previous Goals

The current repository already has the broad compatibility scaffold:

```text
fastkpc/R/mgcv_compat_contract.R
fastkpc/R/mgcv_extract_oracle.R
fastkpc/R/mgcv_extract_validation.R
fastkpc/R/hybrid_verifier.R
fastkpc/tests/test_mgcv_compat_contract.R
fastkpc/tests/test_mgcv_extract_fixed_sp.R
fastkpc/tests/test_mgcv_extract_gcv_bridge.R
fastkpc/tests/test_mgcv_extract_batch_cpu.R
fastkpc/tests/test_hybrid_near_alpha_policy.R
fastkpc/tests/test_compatibility_campaign_metrics.R
```

The important current limitation is narrow and explicit:

```text
fastkpc_mgcv_extract_fixed_sp()
  currently calls mgcv::gam(formula, data, sp = sp, method = method, fit = TRUE)
  and returns mgcv fitted/residuals.

This proves that mgcv can refit mgcv.
It does not prove that fastkpc understands G$X, G$S, G$off, G$C, G$rank,
penalty assembly, constraints, or fixed-sp Gaussian solving.
```

This goal converts that limitation into a real Gate B.

## Non-Goals

- Do not implement a full `mgcv::gam()` clone.
- Do not implement `bamGPU`.
- Do not implement generic GAM acceleration.
- Do not implement smoothing parameter optimization from scratch.
- Do not implement `mgcvPortGCVPrototype`.
- Do not implement CUDA for mgcvSubset.
- Do not implement non-Gaussian families.
- Do not implement non-identity links.
- Do not implement `summary.gam`, `vcov`, standard errors, prediction intervals, ANOVA, or plotting compatibility.
- Do not implement GAMM.
- Do not implement by-smooth or factor-smooth support.
- Do not replace default `s(s1, s2)` with tensor-product semantics.
- Do not call `fastSpline` mgcv-equivalent.
- Do not share smoothing parameters across target variables.
- Do not modify `kpcalg/R/*.R`.
- Do not change exported legacy `kpcalg::kpc()`.

## Gate B Definition

Gate B is passed only when this statement is true:

```text
same formula
same data
same mgcv version
same mgcv setup object from gam(fit = FALSE)
same extracted model matrix X
same extracted response y
same extracted penalty matrices S
same extracted penalty offsets off
same extracted fixed penalty H when present
same extracted constraint matrix C when present
same fixed positive smoothing parameters sp

fastkpc self-solve -> same practical fitted values and residuals as
mgcv::gam(..., sp = sp, fit = TRUE)
```

The self-solve path must not call `mgcv::gam(..., fit = TRUE)` to compute its coefficients, fitted values, or residuals. It may call the mgcv refit reference only for comparison and diagnostics.

## Restricted First-Version Contract

The self-solve implementation in this goal supports only:

```text
family = gaussian()
identity link
weights = NULL or all 1
offset = NULL or all 0
min.sp = NULL
paraPen = NULL
select = FALSE
fixed positive sp only
method passed only for setup/reference consistency
```

The implementation must reject unsupported input clearly:

```text
sp has any NA/NaN/Inf
sp has any value <= 0
length(sp) != length(G$S)
weights are present and not all equal to 1
offset is present and not all equal to 0
G$L or other nontrivial smoothing-parameter mapping is required
G$paraPen is present
family is not gaussian identity
```

If a future mgcv object contains a component that changes this contract, the self-solve should fail with a diagnostic message rather than silently producing residual drift.

## Naming Contract

This goal must make naming honest:

```r
fastkpc_mgcv_gam_fixed_sp_reference()
```

Calls `mgcv::gam(..., sp = sp, fit = TRUE)`. It is a reference/refit path, not fastkpc self-solve.

```r
fastkpc_mgcv_extract_setup()
```

Calls `mgcv::gam(..., fit = FALSE)` and extracts setup components. It may depend on mgcv internals and must record the mgcv version.

```r
fastkpc_assemble_penalty()
```

Builds the full coefficient penalty matrix from compact `S`, `off`, `sp`, and optional `H`.

```r
fastkpc_solve_gaussian_penalized_fixed_sp()
```

Solves the restricted fixed-sp Gaussian penalized least squares problem inside fastkpc.

```r
fastkpc_mgcv_extract_fixed_sp_solve()
```

The real Gate B entrypoint. It uses mgcv setup plus fastkpc self-solve, and compares to the reference only for diagnostics.

```r
fastkpc_mgcv_extract_fixed_sp()
```

Keep as a compatibility wrapper during this goal. It may call the new self-solve once tests are migrated, but it must expose a `mode` or diagnostic that distinguishes alias behavior from the old mgcv refit bridge.

## File Structure Plan

Modify these files:

```text
fastkpc/R/mgcv_extract_oracle.R
  Add reference naming, setup extraction, penalty assembly, restricted solver,
  true fixed-sp self-solve, and honest GCVBridge source diagnostics.

fastkpc/tests/test_mgcv_extract_fixed_sp.R
  Replace the current "mgcv refit equals mgcv" test with true self-solve parity
  cases. Keep compatibility-wrapper checks.

fastkpc/tests/test_mgcv_extract_gcv_bridge.R
  Assert GCVBridge uses mgcv for sp selection and fastkpc for fixed-sp solving.

fastkpc/tests/test_mgcv_extract_batch_cpu.R
  Assert batch path keeps per-target sp and can use the new self-solve bridge
  without sharing smoothing parameters.

fastkpc/tests/test_mgcv_penalty_assembly.R
  New focused tests for S/off/H assembly and explicit failure messages.

fastkpc/tests/test_mgcv_extract_setup_contract.R
  New focused tests for setup extraction, supported/unsupported cases, and
  required setup diagnostics.

fastkpc/README.md
  Update only if current text implies mgcvExtractFixedSP is already a self-solve.
```

Do not modify these files:

```text
kpcalg/R/*.R
fastkpc/src/*.cpp
fastkpc/src/cuda/*.cu
```

This goal is CPU-only R compatibility work.

## Implementation Details

### Fixed-SP Objective

For the restricted Gaussian identity case, solve:

```text
min_beta ||y - X beta||^2 + beta' P beta

P = H + sum_j sp[j] * S_full[j]
```

where each compact `S[[j]]` is placed into the full coefficient matrix using `off[j]`:

```text
idx_j = off[j]:(off[j] + nrow(S[[j]]) - 1)
P[idx_j, idx_j] += sp[j] * S[[j]]
```

If `C` is present, solve subject to:

```text
C beta = 0
```

The first implementation may use a QR null-space projection:

```text
Z spans null(C)
beta = Z theta
theta solves (Z'X'XZ + Z'PZ) theta = Z'X'y
```

If drift remains after correct penalty assembly and constraints, add diagnostics before changing solvers. The likely later reason is mgcv's more careful rank-truncated SVD behavior, but this goal should not start with a complex solver before proving the extraction contract.

### Required Diagnostics

Every self-solve result should include:

```text
backend_family = "mgcvExtractCPU"
mode = "fixed-sp-self-solve"
reference_mode = "mgcv-gam-fixed-sp-reference"
sp_source = "fixed-input"
solve_source = "fastkpc-fixed-sp"
gcv_source = "none"
is_self_contained_gcv = FALSE
mgcv_version
formula
method
sp
coefficients
fitted
residuals
reference_coefficients
reference_fitted
reference_residuals
max_abs_fitted_diff
relative_l2_fitted_diff
max_abs_residual_diff
relative_l2_residual_diff
setup_diagnostics
penalty_diagnostics
constraint_diagnostics
rank_diagnostics
setup_fingerprint
target_fingerprint
```

Every GCV bridge result should include:

```text
mode = "gcv-bridge"
sp_source = "mgcv"
solve_source = "fastkpc-fixed-sp"
gcv_source = "mgcv"
is_self_contained_gcv = FALSE
legacy_score
legacy_edf
legacy_rank
```

## Task 1: Add Focused Setup Extraction Contract Tests

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Create: `fastkpc/tests/test_mgcv_extract_setup_contract.R`

- [ ] **Step 1: Write the setup extraction test**

Create `fastkpc/tests/test_mgcv_extract_setup_contract.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

set.seed(31701)
n <- 70
data <- data.frame(
  y = sin(seq_len(n) / 7) + stats::rnorm(n, sd = 0.05),
  s1 = stats::runif(n, -2, 2)
)

legacy <- mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")
setup <- fastkpc_mgcv_extract_setup(
  formula = y ~ s(s1),
  data = data,
  sp = legacy$sp,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)

assert_true(is.matrix(setup$X), "setup must expose model matrix X")
assert_true(is.numeric(setup$y), "setup must expose response y")
assert_true(length(setup$y) == n, "response length must match input rows")
assert_true(length(setup$S) == length(legacy$sp), "penalty count must match sp count")
assert_true(length(setup$off) == length(setup$S), "off count must match penalty count")
assert_true(all(is.finite(setup$sp)), "setup sp must be finite")
assert_true(all(setup$sp > 0), "setup sp must be fixed positive")
assert_equal(setup$family, "gaussian_identity", "family contract")
assert_equal(setup$weights_policy, "none-or-unit", "weights policy")
assert_equal(setup$offset_policy, "none-or-zero", "offset policy")
assert_true(nchar(setup$setup_fingerprint$fingerprint) > 0,
            "setup fingerprint required")

bad <- tryCatch(
  fastkpc_mgcv_extract_setup(
    formula = y ~ s(s1),
    data = data,
    sp = -1,
    method = "GCV.Cp"
  ),
  error = function(e) e
)
assert_true(inherits(bad, "error"), "negative sp must fail")
assert_true(grepl("fixed positive", conditionMessage(bad)),
            "negative sp error must explain fixed positive requirement")

cat("PASS mgcv extract setup contract\n")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
```

Expected:

```text
Error ... could not find function "fastkpc_mgcv_extract_setup"
```

- [ ] **Step 3: Implement setup extraction**

Add these helpers to `fastkpc/R/mgcv_extract_oracle.R` after `fastkpc_mgcv_selected_sp()`:

```r
fastkpc_stop_unsupported_setup <- function(message) {
  stop(paste0("Unsupported mgcv fixed-sp setup: ", message), call. = FALSE)
}

fastkpc_validate_fixed_positive_sp <- function(sp, expected_length = NULL) {
  if (is.null(sp) || length(sp) == 0L) {
    fastkpc_stop_unsupported_setup("sp must be supplied for fixed-sp self-solve")
  }
  sp <- as.numeric(sp)
  if (any(!is.finite(sp)) || any(sp <= 0)) {
    fastkpc_stop_unsupported_setup("sp must contain fixed positive finite values")
  }
  if (!is.null(expected_length) && length(sp) != expected_length) {
    fastkpc_stop_unsupported_setup(
      paste0("length(sp) must equal length(G$S); got ", length(sp),
             " and expected ", expected_length)
    )
  }
  sp
}

fastkpc_setup_weights_policy <- function(G) {
  w <- G$w
  if (is.null(w)) return(list(policy = "none-or-unit", w = NULL))
  w <- as.numeric(w)
  if (length(w) == 0L || all(abs(w - 1) < 1e-12)) {
    return(list(policy = "none-or-unit", w = NULL))
  }
  fastkpc_stop_unsupported_setup("non-unit weights are not supported in Gate B v1")
}

fastkpc_setup_offset_policy <- function(G) {
  offset <- G$offset
  if (is.null(offset)) return(list(policy = "none-or-zero", offset = NULL))
  offset <- as.numeric(offset)
  if (length(offset) == 0L || all(abs(offset) < 1e-12)) {
    return(list(policy = "none-or-zero", offset = NULL))
  }
  fastkpc_stop_unsupported_setup("non-zero offsets are not supported in Gate B v1")
}

fastkpc_mgcv_extract_setup <- function(formula, data, sp,
                                       method = "GCV.Cp",
                                       target = 1L,
                                       S = integer(),
                                       k = NA_integer_,
                                       bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  sp <- fastkpc_validate_fixed_positive_sp(sp)

  G <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    sp = sp,
    method = method,
    fit = FALSE
  )

  if (is.null(G$X) || is.null(G$y) || is.null(G$S) || is.null(G$off)) {
    fastkpc_stop_unsupported_setup("G must contain X, y, S, and off")
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(G$S))
  weights_info <- fastkpc_setup_weights_policy(G)
  offset_info <- fastkpc_setup_offset_policy(G)

  if (!is.null(G$L) && length(G$L) > 0L) {
    if (is.matrix(G$L) && any(abs(G$L - diag(nrow(G$L), ncol(G$L))) > 1e-12)) {
      fastkpc_stop_unsupported_setup("non-identity smoothing parameter mapping G$L")
    }
  }
  if (!is.null(G$paraPen) && length(G$paraPen) > 0L) {
    fastkpc_stop_unsupported_setup("paraPen is not supported in Gate B v1")
  }

  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(G$y), p = ncol(data))
  setup_fp <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "setup-fixed-sp-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = fastkpc_hash_object(round(as.numeric(G$X), digits = 14)),
    penalty_hashes = vapply(G$S, fastkpc_hash_object, character(1)),
    constraint_hash = fastkpc_hash_object(G$C),
    rank_metadata = paste0("rank=", paste(G$rank, collapse = "|"))
  )

  list(
    G = G,
    X = G$X,
    y = as.numeric(G$y),
    S = G$S,
    off = as.integer(G$off),
    C = G$C,
    rank = G$rank,
    H = G$H,
    w = weights_info$w,
    offset = offset_info$offset,
    sp = sp,
    formula = formula,
    method = method,
    family = "gaussian_identity",
    weights_policy = weights_info$policy,
    offset_policy = offset_info$policy,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    setup_fingerprint = setup_fp
  )
}
```

- [ ] **Step 4: Run the setup extraction test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
```

Expected:

```text
PASS mgcv extract setup contract
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_setup_contract.R
git commit -m "test: add mgcv fixed-sp setup extraction contract"
```

## Task 2: Rename The Existing mgcv Fixed-SP Refit Bridge

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_fixed_sp.R`

- [ ] **Step 1: Add a test for honest reference naming**

Append this block near the top of `fastkpc/tests/test_mgcv_extract_fixed_sp.R` after the data setup:

```r
ref <- fastkpc_mgcv_gam_fixed_sp_reference(
  formula = y ~ s(s1, s2),
  data = data,
  sp = legacy$sp,
  method = "GCV.Cp",
  target = 1L,
  S = c(2L, 3L)
)

assert_true(identical(ref$mode, "mgcv-gam-fixed-sp-reference"),
            "reference mode must name mgcv refit honestly")
assert_true(identical(ref$solve_source, "mgcv"),
            "reference solve source must be mgcv")
assert_true(max(abs(ref$residuals - stats::residuals(legacy))) < 1e-6,
            "mgcv fixed-sp reference residuals must match legacy selected-sp fit")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
Error ... could not find function "fastkpc_mgcv_gam_fixed_sp_reference"
```

- [ ] **Step 3: Extract the old implementation into the reference function**

In `fastkpc/R/mgcv_extract_oracle.R`, add:

```r
fastkpc_mgcv_gam_fixed_sp_reference <- function(formula, data, sp,
                                                method = "GCV.Cp",
                                                target = 1L,
                                                S = integer(),
                                                k = NA_integer_,
                                                bs = "tp") {
  fastkpc_require_mgcv()
  data <- as.data.frame(data)
  sp <- fastkpc_validate_fixed_positive_sp(sp)

  fit <- mgcv::gam(
    formula = formula,
    data = data,
    family = stats::gaussian(),
    sp = sp,
    method = method,
    fit = TRUE
  )

  residuals <- as.numeric(stats::residuals(fit))
  fitted <- as.numeric(stats::fitted(fit))
  coefficients <- as.numeric(stats::coef(fit))
  selected_sp <- fastkpc_mgcv_selected_sp(fit, fallback = sp)
  response <- if (!is.null(fit$y)) fit$y else model.response(stats::model.frame(fit))

  sem <- fastkpc_regrxons_semantics(S = S, target = target,
                                    n = length(residuals), p = ncol(data))
  setup <- fastkpc_setup_fingerprint(
    sem,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    backend_family = "mgcvExtractCPU",
    backend_version = "mgcv-gam-fixed-sp-reference-v1",
    k = k,
    bs = bs,
    method = method,
    model_matrix_hash = tryCatch(
      fastkpc_hash_object(round(as.numeric(stats::predict(fit, type = "lpmatrix")), digits = 14)),
      error = function(e) ""
    ),
    penalty_hashes = fastkpc_mgcv_penalty_hashes(fit),
    constraint_hash = fastkpc_hash_object(lapply(fit$smooth, `[[`, "C")),
    rank_metadata = paste0("rank=", fit$rank)
  )
  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(response),
    sp_input = sp,
    sp_output = selected_sp,
    selected_sp = selected_sp,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank_if_target_specific = fit$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "mgcv-gam-fixed-sp-reference",
    solve_source = "mgcv",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    formula = formula,
    method = method,
    coefficients = coefficients,
    sp = selected_sp,
    residuals = residuals,
    fitted = fitted,
    score = if (!is.null(fit$gcv.ubre)) as.numeric(fit$gcv.ubre) else NA_real_,
    edf = if (!is.null(fit$edf)) sum(fit$edf) else NA_real_,
    rank = fit$rank,
    fit = fit,
    setup_fingerprint = setup,
    target_fingerprint = target_fp,
    mgcv_version = as.character(utils::packageVersion("mgcv"))
  )
}
```

- [ ] **Step 4: Keep the old function as a compatibility wrapper**

Replace the body of `fastkpc_mgcv_extract_fixed_sp()` with:

```r
fastkpc_mgcv_extract_fixed_sp <- function(formula, data, sp,
                                          method = "GCV.Cp",
                                          target = 1L,
                                          S = integer(),
                                          k = NA_integer_,
                                          bs = "tp") {
  out <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  out$mode <- "fixed-sp-compat-reference"
  out$compatibility_alias_for <- "fastkpc_mgcv_gam_fixed_sp_reference"
  out
}
```

This wrapper may be switched to the self-solve in a later task after parity is proven.

- [ ] **Step 5: Run the existing fixed-sp test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected initially after wrapper mode change:

```text
FAIL mode must be fixed-sp
```

Update that old assertion to accept the compatibility reference mode until Task 4 changes the wrapper:

```r
assert_true(fixed$mode %in% c("fixed-sp-compat-reference", "fixed-sp-self-solve"),
            "mode must identify compatibility reference or self-solve")
```

Run again:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
PASS mgcv extract fixed-sp
```

- [ ] **Step 6: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_fixed_sp.R
git commit -m "refactor: name mgcv fixed-sp refit as reference"
```

## Task 3: Implement And Test Penalty Assembly

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Create: `fastkpc/tests/test_mgcv_penalty_assembly.R`

- [ ] **Step 1: Write the penalty assembly test**

Create `fastkpc/tests/test_mgcv_penalty_assembly.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal_num <- function(actual, expected, message, tol = 1e-12) {
  if (max(abs(actual - expected)) > tol) {
    fail(paste0(message, ": max diff ", max(abs(actual - expected))))
  }
}

S <- list(
  matrix(c(2, 0, 0, 3), 2, 2),
  matrix(5, 1, 1)
)
H <- diag(c(0.1, 0.2, 0.3, 0.4))
P <- fastkpc_assemble_penalty(
  p = 4L,
  S = S,
  off = c(2L, 4L),
  sp = c(10, 2),
  H = H
)

expected <- H
expected[2:3, 2:3] <- expected[2:3, 2:3] + 10 * S[[1]]
expected[4, 4] <- expected[4, 4] + 2 * S[[2]][1, 1]
assert_equal_num(P, expected, "assembled penalty")

bad_length <- tryCatch(
  fastkpc_assemble_penalty(p = 4L, S = S, off = c(2L), sp = c(10, 2)),
  error = function(e) e
)
assert_true(inherits(bad_length, "error"), "off length mismatch must fail")
assert_true(grepl("length\\(S\\).*length\\(off\\).*length\\(sp\\)",
                  conditionMessage(bad_length)),
            "length mismatch message must mention S/off/sp")

bad_bounds <- tryCatch(
  fastkpc_assemble_penalty(p = 3L, S = S, off = c(2L, 4L), sp = c(10, 2)),
  error = function(e) e
)
assert_true(inherits(bad_bounds, "error"), "out-of-bounds penalty must fail")
assert_true(grepl("outside coefficient dimension", conditionMessage(bad_bounds)),
            "bounds error must mention coefficient dimension")

bad_square <- tryCatch(
  fastkpc_assemble_penalty(p = 4L, S = list(matrix(1, 2, 1)), off = 1L, sp = 1),
  error = function(e) e
)
assert_true(inherits(bad_square, "error"), "non-square penalty must fail")
assert_true(grepl("square", conditionMessage(bad_square)),
            "non-square error must mention square")

cat("PASS mgcv penalty assembly\n")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_penalty_assembly.R
```

Expected:

```text
Error ... could not find function "fastkpc_assemble_penalty"
```

- [ ] **Step 3: Implement penalty assembly**

Add to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_assemble_penalty <- function(p, S, off, sp, H = NULL) {
  p <- as.integer(p)
  if (length(p) != 1L || is.na(p) || p <= 0L) {
    stop("p must be a positive scalar coefficient dimension", call. = FALSE)
  }
  if (length(S) != length(off) || length(S) != length(sp)) {
    stop("length(S), length(off), and length(sp) must match", call. = FALSE)
  }
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(S))

  P <- matrix(0, p, p)
  if (!is.null(H)) {
    H <- as.matrix(H)
    if (!identical(dim(H), c(p, p))) {
      stop("H must have dimension p x p", call. = FALSE)
    }
    P <- P + H
  }

  for (j in seq_along(S)) {
    Sj <- as.matrix(S[[j]])
    if (nrow(Sj) != ncol(Sj)) {
      stop("Each penalty matrix S[[j]] must be square", call. = FALSE)
    }
    kj <- nrow(Sj)
    idx <- seq.int(as.integer(off[j]), length.out = kj)
    if (min(idx) < 1L || max(idx) > p) {
      stop("Penalty block indexed by off is outside coefficient dimension",
           call. = FALSE)
    }
    P[idx, idx] <- P[idx, idx, drop = FALSE] + sp[j] * Sj
  }

  P
}
```

- [ ] **Step 4: Run the penalty assembly test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_penalty_assembly.R
```

Expected:

```text
PASS mgcv penalty assembly
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_penalty_assembly.R
git commit -m "feat: assemble mgcv fixed-sp penalties"
```

## Task 4: Implement The Restricted Gaussian Fixed-SP Self-Solve

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_fixed_sp.R`

- [ ] **Step 1: Replace the fixed-sp parity test with self-solve expectations**

Update `fastkpc/tests/test_mgcv_extract_fixed_sp.R` so it tests `fastkpc_mgcv_extract_fixed_sp_solve()` directly:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
relative_l2 <- function(a, b) {
  denom <- sqrt(sum(as.numeric(b)^2))
  if (denom == 0) return(sqrt(sum(as.numeric(a - b)^2)))
  sqrt(sum(as.numeric(a - b)^2)) / denom
}

run_case <- function(name, formula, data, S) {
  legacy <- mgcv::gam(formula, data = data, method = "GCV.Cp")
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = "GCV.Cp",
    target = 1L,
    S = S
  )
  solved <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = legacy$sp,
    method = "GCV.Cp",
    target = 1L,
    S = S
  )

  assert_true(identical(ref$mode, "mgcv-gam-fixed-sp-reference"),
              paste(name, "reference mode"))
  assert_true(identical(solved$mode, "fixed-sp-self-solve"),
              paste(name, "self-solve mode"))
  assert_true(identical(solved$solve_source, "fastkpc-fixed-sp"),
              paste(name, "solve source"))
  assert_true(identical(solved$sp_source, "fixed-input"),
              paste(name, "sp source"))
  assert_true(identical(solved$gcv_source, "none"),
              paste(name, "gcv source"))
  assert_true(isFALSE(solved$is_self_contained_gcv),
              paste(name, "not self-contained gcv"))

  max_fit <- max(abs(solved$fitted - ref$fitted))
  max_res <- max(abs(solved$residuals - ref$residuals))
  rel_fit <- relative_l2(solved$fitted, ref$fitted)
  rel_res <- relative_l2(solved$residuals, ref$residuals)

  assert_true(max_fit < 1e-5,
              paste(name, "fitted max abs diff too large:", max_fit))
  assert_true(max_res < 1e-5,
              paste(name, "residual max abs diff too large:", max_res))
  assert_true(rel_fit < 1e-5,
              paste(name, "fitted relative L2 too large:", rel_fit))
  assert_true(rel_res < 1e-5,
              paste(name, "residual relative L2 too large:", rel_res))
  assert_true(nchar(solved$setup_fingerprint$fingerprint) > 0,
              paste(name, "setup fingerprint required"))
  assert_true(nchar(solved$target_fingerprint$fingerprint) > 0,
              paste(name, "target fingerprint required"))
}

set.seed(2101)
n <- 90
s1 <- stats::runif(n, -2, 2)
s2 <- stats::runif(n, -2, 2)
s3 <- stats::runif(n, -2, 2)
y <- sin(s1) + cos(s2) + 0.2 * s3 + stats::rnorm(n, sd = 0.1)
data <- data.frame(y = y, s1 = s1, s2 = s2, s3 = s3)

run_case("|S|=1", y ~ s(s1), data, S = 2L)
run_case("|S|=2", y ~ s(s1, s2), data, S = c(2L, 3L))
run_case("|S|=3 additive", y ~ s(s1) + s(s2) + s(s3), data, S = c(2L, 3L, 4L))

cat("PASS mgcv extract fixed-sp self-solve\n")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
Error ... could not find function "fastkpc_mgcv_extract_fixed_sp_solve"
```

- [ ] **Step 3: Implement the null-space helper**

Add to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_constraint_nullspace <- function(C, p, tol = sqrt(.Machine$double.eps)) {
  if (is.null(C) || length(C) == 0L) return(diag(p))
  C <- as.matrix(C)
  if (nrow(C) == 0L) return(diag(p))
  if (ncol(C) != p) {
    stop("Constraint matrix C must have ncol(C) equal to coefficient dimension",
         call. = FALSE)
  }
  qrCt <- qr(t(C), tol = tol)
  Q <- qr.Q(qrCt, complete = TRUE)
  rC <- qrCt$rank
  if (rC >= p) {
    stop("Constraint matrix leaves no free coefficient space", call. = FALSE)
  }
  Q[, seq.int(rC + 1L, p), drop = FALSE]
}
```

- [ ] **Step 4: Implement the restricted Gaussian solver**

Add to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_solve_gaussian_penalized_fixed_sp <- function(
    X, y, S, off, sp, C = NULL, H = NULL, w = NULL,
    tol = sqrt(.Machine$double.eps)) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  if (nrow(X) != length(y)) {
    stop("nrow(X) must equal length(y)", call. = FALSE)
  }
  p <- ncol(X)
  P <- fastkpc_assemble_penalty(p = p, S = S, off = off, sp = sp, H = H)

  if (is.null(w)) {
    Xw <- X
    yw <- y
  } else {
    w <- as.numeric(w)
    if (length(w) != length(y)) {
      stop("length(w) must equal length(y)", call. = FALSE)
    }
    if (any(!is.finite(w)) || any(w < 0)) {
      stop("weights must be finite and nonnegative", call. = FALSE)
    }
    sw <- sqrt(w)
    Xw <- X * sw
    yw <- y * sw
  }

  Z <- fastkpc_constraint_nullspace(C = C, p = p, tol = tol)
  XZ <- Xw %*% Z
  A <- crossprod(XZ) + crossprod(Z, P %*% Z)
  b <- crossprod(XZ, yw)

  theta <- as.numeric(qr.solve(A, b, tol = tol))
  as.numeric(Z %*% theta)
}
```

- [ ] **Step 5: Implement the fixed-sp self-solve entrypoint**

Add to `fastkpc/R/mgcv_extract_oracle.R`:

```r
fastkpc_relative_l2_diff <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  denom <- sqrt(sum(b^2))
  if (denom == 0) return(sqrt(sum((a - b)^2)))
  sqrt(sum((a - b)^2)) / denom
}

fastkpc_mgcv_extract_fixed_sp_solve <- function(formula, data, sp,
                                                method = "GCV.Cp",
                                                target = 1L,
                                                S = integer(),
                                                k = NA_integer_,
                                                bs = "tp",
                                                tol = sqrt(.Machine$double.eps)) {
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  setup <- fastkpc_mgcv_extract_setup(
    formula = formula,
    data = data,
    sp = ref$sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )

  beta <- fastkpc_solve_gaussian_penalized_fixed_sp(
    X = setup$X,
    y = setup$y,
    S = setup$S,
    off = setup$off,
    sp = setup$sp,
    C = setup$C,
    H = setup$H,
    w = setup$w,
    tol = tol
  )
  fitted <- as.numeric(setup$X %*% beta)
  residuals <- as.numeric(setup$y - fitted)
  response <- setup$y

  target_fp <- fastkpc_target_fingerprint(
    target = target,
    y_hash = fastkpc_mgcv_hash_numeric(response),
    sp_input = sp,
    sp_output = setup$sp,
    selected_sp = setup$sp,
    score = NA_real_,
    edf = NA_real_,
    rank_if_target_specific = setup$rank,
    residual_hash = fastkpc_mgcv_hash_numeric(residuals),
    fitted_hash = fastkpc_mgcv_hash_numeric(fitted)
  )

  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp-self-solve",
    reference_mode = ref$mode,
    solve_source = "fastkpc-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    formula = formula,
    method = method,
    coefficients = beta,
    fitted = fitted,
    residuals = residuals,
    sp = setup$sp,
    reference_coefficients = ref$coefficients,
    reference_fitted = ref$fitted,
    reference_residuals = ref$residuals,
    max_abs_fitted_diff = max(abs(fitted - ref$fitted)),
    relative_l2_fitted_diff = fastkpc_relative_l2_diff(fitted, ref$fitted),
    max_abs_residual_diff = max(abs(residuals - ref$residuals)),
    relative_l2_residual_diff = fastkpc_relative_l2_diff(residuals, ref$residuals),
    setup_diagnostics = list(
      n = nrow(setup$X),
      p = ncol(setup$X),
      penalty_count = length(setup$S),
      off = setup$off,
      has_C = !is.null(setup$C) && length(setup$C) > 0L,
      has_H = !is.null(setup$H),
      weights_policy = setup$weights_policy,
      offset_policy = setup$offset_policy
    ),
    penalty_diagnostics = list(
      sp = setup$sp,
      penalty_dims = lapply(setup$S, dim)
    ),
    constraint_diagnostics = list(
      C_dim = if (is.null(setup$C)) c(0L, ncol(setup$X)) else dim(as.matrix(setup$C))
    ),
    rank_diagnostics = list(rank = setup$rank),
    setup = setup,
    reference = ref,
    setup_fingerprint = setup$setup_fingerprint,
    target_fingerprint = target_fp,
    mgcv_version = setup$mgcv_version
  )
}
```

- [ ] **Step 6: Run the fixed-sp self-solve test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
PASS mgcv extract fixed-sp self-solve
```

If it fails with residual/fitted drift, do not loosen tolerance first. Print and inspect:

```r
str(solved$setup_diagnostics)
str(solved$penalty_diagnostics)
str(solved$constraint_diagnostics)
str(solved$rank_diagnostics)
solved$max_abs_fitted_diff
solved$relative_l2_fitted_diff
solved$max_abs_residual_diff
solved$relative_l2_residual_diff
```

Then debug in this order:

```text
1. sp length and ordering
2. off indexing and compact S block placement
3. H inclusion
4. C nullspace handling
5. weights/offset rejection
6. rank deficiency and qr.solve behavior
```

- [ ] **Step 7: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_fixed_sp.R
git commit -m "feat: solve mgcv fixed-sp setup inside fastkpc"
```

## Task 5: Switch The Compatibility Wrapper To The Self-Solve

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_fixed_sp.R`

- [ ] **Step 1: Add wrapper behavior assertions**

Append to `fastkpc/tests/test_mgcv_extract_fixed_sp.R`:

```r
wrapper <- fastkpc_mgcv_extract_fixed_sp(
  formula = y ~ s(s1),
  data = data,
  sp = mgcv::gam(y ~ s(s1), data = data, method = "GCV.Cp")$sp,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)
assert_true(identical(wrapper$mode, "fixed-sp-self-solve"),
            "compatibility wrapper should now use fixed-sp self-solve")
assert_true(identical(wrapper$compatibility_alias_for,
                      "fastkpc_mgcv_extract_fixed_sp_solve"),
            "wrapper must name the self-solve alias target")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
Error ... compatibility wrapper should now use fixed-sp self-solve
```

- [ ] **Step 3: Switch the wrapper**

Replace `fastkpc_mgcv_extract_fixed_sp()` with:

```r
fastkpc_mgcv_extract_fixed_sp <- function(formula, data, sp,
                                          method = "GCV.Cp",
                                          target = 1L,
                                          S = integer(),
                                          k = NA_integer_,
                                          bs = "tp") {
  out <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = sp,
    method = method,
    target = target,
    S = S,
    k = k,
    bs = bs
  )
  out$compatibility_alias_for <- "fastkpc_mgcv_extract_fixed_sp_solve"
  out
}
```

- [ ] **Step 4: Run the fixed-sp test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
PASS mgcv extract fixed-sp self-solve
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_fixed_sp.R
git commit -m "refactor: route fixed-sp alias to self-solve"
```

## Task 6: Make GCVBridge Consume The Self-Solve Honestly

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_gcv_bridge.R`

- [ ] **Step 1: Strengthen the GCVBridge test**

Update `fastkpc/tests/test_mgcv_extract_gcv_bridge.R` to assert:

```r
assert_true(identical(bridge$mode, "gcv-bridge"),
            "bridge mode")
assert_true(identical(bridge$sp_source, "mgcv"),
            "GCVBridge sp source must be mgcv")
assert_true(identical(bridge$gcv_source, "mgcv"),
            "GCVBridge gcv source must be mgcv")
assert_true(identical(bridge$solve_source, "fastkpc-fixed-sp"),
            "GCVBridge solve source must be fastkpc fixed-sp")
assert_true(isFALSE(bridge$is_self_contained_gcv),
            "GCVBridge must not claim self-contained GCV")
assert_true(max(abs(bridge$residuals - stats::residuals(legacy))) < 1e-5,
            "GCVBridge residuals must match legacy practical tolerance")
```

- [ ] **Step 2: Run the GCVBridge test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
```

Expected failure before implementation:

```text
Error ... GCVBridge sp source must be mgcv
```

or a similar missing-field failure.

- [ ] **Step 3: Update GCVBridge implementation**

Modify `fastkpc_mgcv_extract_gcv_bridge()` so the core call is:

```r
legacy <- mgcv::gam(formula = formula, data = data,
                    family = stats::gaussian(), method = method)
fixed <- fastkpc_mgcv_extract_fixed_sp_solve(
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
fixed$sp_source <- "mgcv"
fixed$gcv_source <- "mgcv"
fixed$solve_source <- "fastkpc-fixed-sp"
fixed$is_self_contained_gcv <- FALSE
fixed$legacy_score <- if (!is.null(legacy$gcv.ubre)) as.numeric(legacy$gcv.ubre) else NA_real_
fixed$legacy_edf <- if (!is.null(legacy$edf)) sum(legacy$edf) else NA_real_
fixed$legacy_rank <- legacy$rank
fixed$legacy_sp <- legacy$sp
fixed
```

- [ ] **Step 4: Run GCVBridge and fixed-sp tests**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
```

Expected:

```text
PASS mgcv extract fixed-sp self-solve
PASS mgcv extract GCV bridge
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_gcv_bridge.R
git commit -m "feat: bridge mgcv-selected sp through fixed-sp self-solve"
```

## Task 7: Preserve Same-S Multi-Target Semantics

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Modify: `fastkpc/tests/test_mgcv_extract_batch_cpu.R`

- [ ] **Step 1: Strengthen batch test source diagnostics**

Update `fastkpc/tests/test_mgcv_extract_batch_cpu.R` so each target fit asserts:

```r
assert_true(all(vapply(batch$target_fingerprints, function(fp) {
  nchar(fp$fingerprint) > 0
}, logical(1))), "all target fingerprints required")
assert_true(length(unique(vapply(batch$target_fingerprints, `[[`, character(1), "fingerprint"))) ==
              ncol(batch$residuals),
            "target fingerprints must differ across targets")
assert_true(identical(batch$solve_source, "fastkpc-fixed-sp"),
            "batch extraction must use fixed-sp self-solve through GCVBridge")
assert_true(identical(batch$sp_source, "mgcv"),
            "batch GCVBridge sp source must be mgcv per target")
```

- [ ] **Step 2: Run batch test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
```

Expected failure before implementation:

```text
Error ... batch extraction must use fixed-sp self-solve through GCVBridge
```

- [ ] **Step 3: Add batch-level source fields**

In `fastkpc_mgcv_extract_batch()`, add to the returned list:

```r
solve_source = "fastkpc-fixed-sp",
sp_source = "mgcv",
gcv_source = "mgcv",
is_self_contained_gcv = FALSE,
```

Keep the per-target loop using `fastkpc_mgcv_extract_gcv_bridge()`. Do not share `sp` across targets.

- [ ] **Step 4: Run batch and GCV tests**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
```

Expected:

```text
PASS mgcv extract GCV bridge
PASS mgcv extract batch CPU
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_batch_cpu.R
git commit -m "test: preserve per-target mgcv bridge semantics in batch"
```

## Task 8: Add A CPU-Only mgcv Gate Runner

**Files:**
- Create: `fastkpc/tools/run_mgcv_gate_b_tests.sh`

- [ ] **Step 1: Create the runner**

Create `fastkpc/tools/run_mgcv_gate_b_tests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

Rscript fastkpc/tests/test_mgcv_compat_contract.R
Rscript fastkpc/tests/test_mgcv_extract_setup_contract.R
Rscript fastkpc/tests/test_mgcv_penalty_assembly.R
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x fastkpc/tools/run_mgcv_gate_b_tests.sh
```

- [ ] **Step 3: Run the gate**

Run:

```bash
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

Expected:

```text
PASS mgcv compatibility contract
PASS mgcv extract setup contract
PASS mgcv penalty assembly
PASS mgcv extract fixed-sp self-solve
PASS mgcv extract GCV bridge
PASS mgcv extract batch CPU
```

- [ ] **Step 4: Commit**

```bash
git add fastkpc/tools/run_mgcv_gate_b_tests.sh
git commit -m "test: add mgcv Gate B runner"
```

## Task 9: Documentation Update

**Files:**
- Modify: `fastkpc/README.md`

- [ ] **Step 1: Search for stale wording**

Run:

```bash
rg -n "mgcvExtract|FixedSP|fixed-sp|self-solve|oracle|bridge|mgcv-compatible" fastkpc/README.md docs/superpowers/plans
```

Expected: identify any wording that implies the old fixed-sp bridge is already a self-solve.

- [ ] **Step 2: Update README wording only if stale**

If `fastkpc/README.md` mentions `mgcvExtractFixedSP`, ensure it distinguishes:

```text
mgcv fixed-sp reference:
    calls mgcv::gam(..., sp=sp, fit=TRUE)

mgcvExtract fixed-sp self-solve:
    uses mgcv setup from gam(fit=FALSE), then solves fixed-sp Gaussian
    penalized least squares inside fastkpc

GCVBridge:
    mgcv selects sp; fastkpc self-solves at selected sp; not self-contained GCV
```

Required exact phrase somewhere in README if mgcvExtract is documented:

```text
mgcvExtractGCVBridge uses mgcv for smoothing-parameter selection and fastkpc only for the fixed-sp solve; it is not a self-contained GCV implementation.
```

- [ ] **Step 3: Run docs contract if present**

Run:

```bash
if test -f fastkpc/tests/test_mgcv_compat_docs_contract.R; then Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R; fi
```

Expected:

```text
PASS mgcv compatibility docs contract
```

- [ ] **Step 4: Commit if README changed**

If README changed:

```bash
git add fastkpc/README.md
git commit -m "docs: clarify mgcv fixed-sp self-solve scope"
```

If README did not change, do not create an empty commit.

## Task 10: Final Verification

**Files:**
- No source changes unless verification exposes a real bug.

- [ ] **Step 1: Parse all R files**

Run:

```bash
Rscript -e 'for (f in list.files("fastkpc/R", "\\\\.R$", full.names = TRUE)) { cat("parse", f, "\\n"); parse(f) }'
```

Expected: command exits `0` after printing each parsed file.

- [ ] **Step 2: Run the mgcv Gate B runner**

Run:

```bash
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

Expected:

```text
PASS mgcv compatibility contract
PASS mgcv extract setup contract
PASS mgcv penalty assembly
PASS mgcv extract fixed-sp self-solve
PASS mgcv extract GCV bridge
PASS mgcv extract batch CPU
```

- [ ] **Step 3: Run the broader non-CUDA R test set**

Run:

```bash
for f in fastkpc/tests/test_*.R; do
  case "$f" in
    *cuda*) echo "SKIP $f";;
    *) echo "RUN $f"; Rscript "$f";;
  esac
done
```

Expected: all non-CUDA tests pass or explicitly skip only when a documented optional dependency is unavailable.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git status --short
git diff --stat
git diff -- fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
Only intended mgcv Gate B files changed.
No kpcalg/R files changed.
```

- [ ] **Step 5: Final commit**

If all previous tasks were committed individually, no final commit is needed. If uncommitted verification/doc edits remain:

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests fastkpc/tools fastkpc/README.md
git commit -m "test: complete mgcv fixed-sp Gate B validation"
```

## Completion Criteria

This goal is complete only when current evidence proves all of the following:

```text
1. fastkpc_mgcv_gam_fixed_sp_reference() exists and clearly calls mgcv refit.
2. fastkpc_mgcv_extract_setup() exists and extracts G$X/G$y/G$S/G$off/G$C/G$rank/H/w.
3. Unsupported Gate B v1 inputs fail clearly.
4. fastkpc_assemble_penalty() is separately tested for compact S/off/H assembly.
5. fastkpc_solve_gaussian_penalized_fixed_sp() solves the restricted Gaussian system.
6. fastkpc_mgcv_extract_fixed_sp_solve() does not use mgcv::gam(fit=TRUE) for fitted/residuals.
7. Fixed-sp self-solve parity passes for |S|=1, |S|=2, and additive |S|=3 cases.
8. GCVBridge records sp_source="mgcv", gcv_source="mgcv", solve_source="fastkpc-fixed-sp".
9. Batch extraction preserves per-target sp and target fingerprints.
10. The mgcv Gate B runner passes.
11. Non-CUDA R tests pass.
12. No files under kpcalg/R are modified.
```

## Failure Triage

If fixed-sp parity fails, debug in this order:

```text
1. Confirm setup$sp equals the reference fit sp vector.
2. Confirm length(setup$S) == length(setup$off) == length(setup$sp).
3. Confirm each off[j] block placement stays inside ncol(setup$X).
4. Confirm H is included only when present and has p x p dimensions.
5. Confirm C is either absent, empty, or has ncol(C) == ncol(X).
6. Confirm the null-space basis dimension is p - rank(C).
7. Confirm weights are NULL/unit and offsets are NULL/zero.
8. Compare crossprod(X) + P between self-solve diagnostics and a direct local reconstruction.
9. If the system is rank deficient, add an SVD-based solver comparison before changing tolerances.
```

Do not respond to parity failure by weakening the tolerance until the diagnostics show the remaining difference is numerical rank behavior rather than penalty/setup interpretation.

## Future Work After This Goal

Only after this Gate B passes should the project move to:

```text
mgcvExtractGCVBridge compatibility campaign
near-alpha verifier integration into skeleton replay
mgcvPortGCVPrototype
mgcvSubsetCPU portable basis/penalty reproduction
mgcvSubsetCUDA batching design
```

The immediate next stage after this goal should not be CUDA. It should use the new fixed-sp self-solve to measure how much legacy residual and p-value drift is explained by setup/solver parity versus smoothing-parameter selection.
