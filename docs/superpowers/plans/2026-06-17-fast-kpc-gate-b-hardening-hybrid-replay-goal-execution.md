# Fast kPC Gate B Hardening And Hybrid Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the new mgcv fixed-sp Gate B into a regression wall, separate pure fastkpc self-solve from mgcv reference comparison, prove GCVBridge is "mgcv chooses sp, fastkpc solves", and connect the near-alpha verifier policy to canonical CI replay so verifier p-values can safely protect skeleton/sepset decisions without changing order semantics.

**Architecture:** Keep CUDA and fastSpline kernels unchanged. Add a CPU-only mgcv Gate B campaign layer around the existing `mgcv::gam(fit=FALSE)` setup self-solve, then add an R-level canonical replay harness that joins primary and verifier p-values by `canonical_test_order_id`, applies the near-alpha policy, and replays edge deletion/sepset selection deterministically. Only after the R harness proves replay semantics should later work consider pushing hybrid verification into the C++/CUDA scheduler.

**Tech Stack:** R 4.4.x, `mgcv` 1.9-x, base R matrix algebra, existing `fastkpc/R/mgcv_extract_oracle.R`, existing `fastkpc/R/hybrid_verifier.R`, existing `fastkpc/R/mgcv_extract_validation.R`, existing C++/CUDA skeleton outputs as reference artifacts, CSV output through base R, and unchanged `kpcalg/R/*.R`.

---

## Goal Objective For Codex

Use this exact goal objective when starting a Codex goal:

```text
Build the next fastkpc compatibility slice from docs/superpowers/plans/2026-06-17-fast-kpc-gate-b-hardening-hybrid-replay-goal-execution.md: add a Gate B fixed-sp campaign runner with CSV output, split the pure mgcv setup self-solve core from mgcv reference comparison, add structural tests preventing self-solve from relying on mgcv::gam(fit=TRUE) or mgcv::magic(), prove GCVBridge residuals equal fixed-sp self-solve at mgcv-selected sp, implement canonical near-alpha hybrid replay over CI test rows, add graph-level hybrid replay tests for edge deletion and sepset order, produce first small compatibility campaign artifacts, document the protected baseline, and keep kpcalg/R files unchanged.
```

Recommended token budget for the goal run: `760000`.

Do not mark the goal complete until every item in "Completion Criteria" is proven by current command output and generated artifacts. Mark the goal blocked only if the same local blocker repeats for three consecutive goal turns and no meaningful implementation or validation work remains possible.

## Baseline Checkpoint

The previous checkpoint is:

```text
8d072ea feat: add mgcv fixed-sp self-solve gate
```

It established:

```text
fastkpc_mgcv_extract_setup()
fastkpc_mgcv_gam_fixed_sp_reference()
fastkpc_assemble_penalty()
fastkpc_solve_gaussian_penalized_fixed_sp()
fastkpc_mgcv_extract_fixed_sp_solve()
fastkpc_mgcv_extract_gcv_bridge()
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

The important property now true:

```text
mgcv gives setup
fastkpc assembles penalty
fastkpc solves fixed-sp Gaussian penalized least squares
fastkpc residuals match mgcv fixed-sp refit in basic |S|=1, |S|=2, and additive |S|=3 tests
```

The next problem is not CUDA performance. The next problem is turning that basic Gate B into a protected baseline and wiring it into decision safety.

## Non-Goals

- Do not implement a self-contained GCV optimizer.
- Do not implement `mgcvPortGCVPrototype`.
- Do not implement `mgcvSubsetCUDA`.
- Do not implement a full `mgcv::gam()` clone.
- Do not implement `bamGPU`.
- Do not add new fastSpline CUDA kernels.
- Do not move graph control flow to GPU.
- Do not modify `kpcalg/R/*.R`.
- Do not replace exported legacy `kpcalg::kpc()`.
- Do not change default fastSpline math, lambda grid, penalty, or GCV rule.
- Do not make fastSpline claim mgcv equivalence.

## Deliverables

This goal must produce:

```text
1. Gate B campaign runner and CSV output
2. Pure setup self-solve core separated from reference comparison
3. Structural tests that protect self-solve from mgcv::gam(fit=TRUE) and mgcv::magic()
4. GCVBridge parity tests against fixed-sp self-solve at mgcv-selected sp
5. Canonical hybrid replay implementation for CI test rows
6. Graph-level hybrid replay tests for edge deletion and sepset order
7. First small compatibility campaign artifacts
8. README/docs update describing the protected Gate B and hybrid replay scope
```

## File Structure Plan

Create or modify these files:

```text
fastkpc/R/mgcv_extract_oracle.R
  Split pure setup self-solve from reference comparison.
  Keep reference function explicit.
  Keep GCVBridge source diagnostics honest.

fastkpc/R/mgcv_gate_b_campaign.R
  New Gate B campaign scenarios, metrics, CSV writer, and command-friendly runner.

fastkpc/R/hybrid_verifier.R
  Extend policy helper into primary/verifier join and canonical replay.

fastkpc/R/mgcv_extract_validation.R
  Extend compatibility metric schema for Gate B campaign and hybrid replay rows.

fastkpc/R/validation_campaign.R
  Add first small compatibility campaign wrapper if needed.
  Do not disrupt existing validation campaign outputs.

fastkpc/tools/run_mgcv_gate_b_campaign.R
  CLI-style R script that runs fixed scenarios and writes CSV artifacts.

fastkpc/tools/run_mgcv_gate_b_campaign.sh
  Shell wrapper for the campaign.

fastkpc/tools/run_mgcv_gate_b_tests.sh
  Extend to include new structural and replay tests.

fastkpc/tests/test_mgcv_self_solve_purity.R
  Tests self-solve core does not call mgcv reference paths.

fastkpc/tests/test_mgcv_gate_b_campaign.R
  Tests campaign runner output schema and passing basic cases.

fastkpc/tests/test_mgcv_extract_gcv_bridge.R
  Strengthen parity against self-solve at mgcv-selected sp.

fastkpc/tests/test_hybrid_canonical_replay.R
  Tests primary/verifier join, p replacement, canonical replay, edge deletion, and sepset order.

fastkpc/tests/test_hybrid_graph_replay_policy.R
  Tests graph-level outcomes before and after verifier substitution.

fastkpc/README.md
  Document protected Gate B campaign and hybrid replay scope.
```

Do not modify:

```text
kpcalg/R/*.R
fastkpc/src/cuda/*.cu
```

Only modify C++ scheduler code in this goal if the R-level replay harness cannot express a required invariant. The preferred first implementation is R-level and testable without rebuilding CUDA.

## Core Contracts

### Pure Self-Solve Contract

The pure self-solve core must consume only:

```text
setup from fastkpc_mgcv_extract_setup()
fixed positive sp
tol
```

It may call:

```text
fastkpc_assemble_penalty()
fastkpc_solve_gaussian_penalized_fixed_sp()
base R matrix algebra
```

It must not call:

```text
mgcv::gam(..., fit = TRUE)
mgcv::magic()
fastkpc_mgcv_gam_fixed_sp_reference()
```

Reference comparison belongs outside the pure solve core.

### GCVBridge Contract

GCVBridge means:

```text
mgcv::gam(..., method = method) chooses sp
fastkpc_mgcv_extract_setup(..., sp = legacy$sp) extracts setup
fastkpc self-solve computes fitted/residuals
```

Required diagnostic fields:

```text
mode = "gcv-bridge"
sp_source = "mgcv"
gcv_source = "mgcv"
solve_source = "fastkpc-fixed-sp"
is_self_contained_gcv = FALSE
```

### Hybrid Replay Contract

Hybrid verification may replace p-values. It must not replace replay order.

Required behavior:

```text
1. primary rows define the canonical test plan and canonical_test_order_id
2. verifier rows are joined by canonical_test_order_id
3. verifier row order is ignored
4. p_used is primary_p unless near-alpha and verifier_p is finite
5. decisions are replayed in canonical order
6. first canonical separating set deletes the edge
7. later tests for an already-deleted edge are ignored
8. sepset records the conditioning set from the canonical first accepted deletion
```

Required diagnostic fields:

```text
canonical_test_order_id
conditioning_level
x
y
S_key
primary_p
verifier_p
p_used
p_source_used
near_alpha_triggered
decision_before_verify
decision_after_verify
edge_deleted
edge_already_deleted
sepset_recorded
verifier_backend
verification_reason
```

## Task 1: Split Pure Setup Self-Solve From Reference Comparison

**Files:**
- Modify: `fastkpc/R/mgcv_extract_oracle.R`
- Create: `fastkpc/tests/test_mgcv_self_solve_purity.R`

- [ ] **Step 1: Write the purity test**

Create `fastkpc/tests/test_mgcv_self_solve_purity.R`:

```r
source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_oracle.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(41001)
n <- 80
data <- data.frame(
  y = sin(seq_len(n) / 9) + stats::rnorm(n, sd = 0.08),
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

ref_before <- fastkpc_mgcv_gam_fixed_sp_reference
magic_available <- exists("magic", envir = asNamespace("mgcv"), inherits = FALSE)

fastkpc_mgcv_gam_fixed_sp_reference <- function(...) {
  stop("reference path must not be called by pure setup solve", call. = FALSE)
}
assignInNamespace(
  "gam",
  function(...) stop("mgcv::gam must not be called by pure setup solve", call. = FALSE),
  ns = "mgcv"
)
if (magic_available) {
  assignInNamespace(
    "magic",
    function(...) stop("mgcv::magic must not be called by pure setup solve", call. = FALSE),
    ns = "mgcv"
  )
}

solution <- fastkpc_mgcv_solve_setup_fixed_sp(setup)
assert_true(identical(solution$mode, "fixed-sp-setup-self-solve"),
            "pure setup solve mode")
assert_true(identical(solution$solve_source, "fastkpc-fixed-sp"),
            "pure setup solve source")
assert_true(length(solution$residuals) == n, "residual length")
assert_true(length(solution$fitted) == n, "fitted length")
assert_true(length(solution$coefficients) == ncol(setup$X), "coefficient length")

fastkpc_mgcv_gam_fixed_sp_reference <- ref_before

cat("PASS mgcv self-solve purity\n")
```

Note: If `assignInNamespace()` cannot patch locked mgcv bindings on the local R build, replace the namespace patch with a static test that reads `body(fastkpc_mgcv_solve_setup_fixed_sp)` and rejects `mgcv::gam`, `mgcv::magic`, and `fastkpc_mgcv_gam_fixed_sp_reference`. The test must still prove the core function exists and returns fitted/residuals from a pre-extracted setup.

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_self_solve_purity.R
```

Expected:

```text
Error ... could not find function "fastkpc_mgcv_solve_setup_fixed_sp"
```

- [ ] **Step 3: Add the pure setup self-solve function**

Add to `fastkpc/R/mgcv_extract_oracle.R` below `fastkpc_relative_l2_diff()`:

```r
fastkpc_mgcv_solve_setup_fixed_sp <- function(setup,
                                              sp = setup$sp,
                                              tol = sqrt(.Machine$double.eps)) {
  sp <- fastkpc_validate_fixed_positive_sp(sp, expected_length = length(setup$S))
  beta <- fastkpc_solve_gaussian_penalized_fixed_sp(
    X = setup$X,
    y = setup$y,
    S = setup$S,
    off = setup$off,
    sp = sp,
    C = setup$C,
    H = setup$H,
    w = setup$w,
    tol = tol
  )
  fitted <- as.numeric(setup$X %*% beta)
  residuals <- as.numeric(setup$y - fitted)
  list(
    backend_family = "mgcvExtractCPU",
    mode = "fixed-sp-setup-self-solve",
    solve_source = "fastkpc-fixed-sp",
    sp_source = "fixed-input",
    gcv_source = "none",
    is_self_contained_gcv = FALSE,
    coefficients = beta,
    fitted = fitted,
    residuals = residuals,
    sp = sp,
    setup_fingerprint = setup$setup_fingerprint,
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
      sp = sp,
      penalty_dims = lapply(setup$S, dim)
    ),
    constraint_diagnostics = list(
      C_dim = if (is.null(setup$C)) c(0L, ncol(setup$X)) else dim(as.matrix(setup$C))
    ),
    rank_diagnostics = list(rank = setup$rank)
  )
}
```

- [ ] **Step 4: Refactor `fastkpc_mgcv_extract_fixed_sp_solve()`**

Change `fastkpc_mgcv_extract_fixed_sp_solve()` so it calls:

```r
solution <- fastkpc_mgcv_solve_setup_fixed_sp(setup = setup, sp = setup$sp, tol = tol)
```

Then build reference comparison diagnostics outside the pure solve core:

```r
fitted <- solution$fitted
residuals <- solution$residuals
beta <- solution$coefficients
```

Keep the returned public fields unchanged:

```text
mode = "fixed-sp-self-solve"
reference_mode = ref$mode
solve_source = "fastkpc-fixed-sp"
```

- [ ] **Step 5: Run purity and fixed-sp tests**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_self_solve_purity.R
Rscript fastkpc/tests/test_mgcv_extract_fixed_sp.R
```

Expected:

```text
PASS mgcv self-solve purity
PASS mgcv extract fixed-sp self-solve
```

- [ ] **Step 6: Commit**

```bash
git add fastkpc/R/mgcv_extract_oracle.R fastkpc/tests/test_mgcv_self_solve_purity.R
git commit -m "refactor: split pure mgcv setup self-solve"
```

## Task 2: Add GCVBridge Parity Against Fixed-SP Self-Solve

**Files:**
- Modify: `fastkpc/tests/test_mgcv_extract_gcv_bridge.R`

- [ ] **Step 1: Add parity assertions**

Append to `fastkpc/tests/test_mgcv_extract_gcv_bridge.R`:

```r
self <- fastkpc_mgcv_extract_fixed_sp_solve(
  formula = y ~ s(s1),
  data = data,
  sp = legacy$sp,
  method = "GCV.Cp",
  target = 1L,
  S = 2L
)

assert_true(max(abs(bridge$residuals - self$residuals)) < 1e-10,
            "GCVBridge residuals must equal self-solve at mgcv-selected sp")
assert_true(max(abs(bridge$fitted - self$fitted)) < 1e-10,
            "GCVBridge fitted values must equal self-solve at mgcv-selected sp")
assert_true(identical(bridge$sp_source, "mgcv"),
            "GCVBridge sp source must remain mgcv")
assert_true(identical(bridge$gcv_source, "mgcv"),
            "GCVBridge gcv source must remain mgcv")
assert_true(identical(bridge$solve_source, "fastkpc-fixed-sp"),
            "GCVBridge solve source must remain fastkpc fixed-sp")
```

- [ ] **Step 2: Run the test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_extract_gcv_bridge.R
```

Expected:

```text
PASS mgcv extract GCV bridge
```

- [ ] **Step 3: Commit**

```bash
git add fastkpc/tests/test_mgcv_extract_gcv_bridge.R
git commit -m "test: pin GCV bridge to fixed-sp self-solve"
```

## Task 3: Add Gate B Campaign Runner

**Files:**
- Create: `fastkpc/R/mgcv_gate_b_campaign.R`
- Create: `fastkpc/tests/test_mgcv_gate_b_campaign.R`
- Create: `fastkpc/tools/run_mgcv_gate_b_campaign.R`
- Create: `fastkpc/tools/run_mgcv_gate_b_campaign.sh`
- Modify: `fastkpc/tools/run_mgcv_gate_b_tests.sh`

- [ ] **Step 1: Write the campaign test**

Create `fastkpc/tests/test_mgcv_gate_b_campaign.R`:

```r
source("fastkpc/R/mgcv_gate_b_campaign.R")

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP mgcv not installed\n")
  quit(save = "no", status = 0)
}

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-gate-b-campaign-")
dir.create(out_dir, recursive = TRUE)

campaign <- fastkpc_run_mgcv_gate_b_campaign(
  seeds = c(11, 12),
  n_values = c(80),
  sp_grid = c("selected", "small", "medium", "large"),
  output_dir = out_dir
)

required <- c(
  "scenario_id", "seed", "n", "S_size", "formula_class", "sp_source",
  "sp", "edf_reference", "rank_setup", "constraint_rank", "penalty_rank",
  "coef_rel_l2", "fitted_rel_l2", "residual_rel_l2",
  "max_abs_residual_diff", "condition_number_proxy",
  "pass_gate_b", "warning_message"
)
missing <- setdiff(required, names(campaign$fixed_sp))
assert_true(length(missing) == 0L,
            paste("missing campaign fields:", paste(missing, collapse = ", ")))
assert_true(nrow(campaign$fixed_sp) > 0L, "campaign should produce rows")
assert_true(all(campaign$fixed_sp$pass_gate_b), "basic campaign rows should pass Gate B")
assert_true(file.exists(file.path(out_dir, "mgcv_gate_b_fixed_sp_campaign.csv")),
            "fixed-sp campaign CSV should be written")

read_back <- utils::read.csv(file.path(out_dir, "mgcv_gate_b_fixed_sp_campaign.csv"))
assert_true(nrow(read_back) == nrow(campaign$fixed_sp),
            "CSV row count should match in-memory campaign")

cat("PASS mgcv Gate B campaign\n")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_gate_b_campaign.R
```

Expected:

```text
Error in file(filename, "r", encoding = encoding) :
  cannot open file 'fastkpc/R/mgcv_gate_b_campaign.R'
```

- [ ] **Step 3: Implement campaign scenarios**

Create `fastkpc/R/mgcv_gate_b_campaign.R`:

```r
source("fastkpc/R/mgcv_extract_oracle.R")

fastkpc_gate_b_formula <- function(S_size) {
  if (S_size == 1L) stats::as.formula("y ~ s(s1)")
  else if (S_size == 2L) stats::as.formula("y ~ s(s1, s2)")
  else stats::as.formula(
    paste("y ~", paste(sprintf("s(s%d)", seq_len(S_size)), collapse = " + "))
  )
}

fastkpc_gate_b_formula_class <- function(S_size) {
  if (S_size <= 2L) "full-smooth" else "additive-smooth"
}

fastkpc_gate_b_scenario_data <- function(seed, n, S_size, scenario_id) {
  set.seed(seed)
  z <- stats::runif(n, -2, 2)
  S <- matrix(stats::rnorm(n * S_size), n, S_size)
  if (identical(scenario_id, "mild_collinearity") && S_size >= 2L) {
    S[, 2] <- S[, 1] + stats::rnorm(n, sd = 0.03)
  }
  if (identical(scenario_id, "near_constant")) {
    S[, 1] <- 0.001 * stats::rnorm(n)
  }
  if (identical(scenario_id, "tied_values")) {
    S[, 1] <- round(S[, 1], digits = 1)
  }
  colnames(S) <- paste0("s", seq_len(S_size))
  y <- sin(z) + rowSums(S[, seq_len(S_size), drop = FALSE]) / max(1, S_size) +
    stats::rnorm(n, sd = 0.08)
  data.frame(y = y, S, check.names = FALSE)
}

fastkpc_gate_b_sp_value <- function(source, selected_sp, length_out) {
  if (identical(source, "selected")) return(selected_sp)
  value <- switch(
    source,
    small = 1e-4,
    medium = 1,
    large = 1e4,
    stop("unknown sp source: ", source, call. = FALSE)
  )
  rep(value, length_out)
}

fastkpc_matrix_rank <- function(x, tol = sqrt(.Machine$double.eps)) {
  if (is.null(x) || length(x) == 0L) return(0L)
  qr(as.matrix(x), tol = tol)$rank
}

fastkpc_condition_proxy <- function(A) {
  values <- tryCatch(svd(A, nu = 0, nv = 0)$d, error = function(e) numeric())
  values <- values[is.finite(values) & values > .Machine$double.eps]
  if (length(values) == 0L) return(Inf)
  max(values) / min(values)
}

fastkpc_gate_b_row <- function(scenario_id, seed, n, S_size, sp_source) {
  data <- fastkpc_gate_b_scenario_data(seed, n, S_size, scenario_id)
  formula <- fastkpc_gate_b_formula(S_size)
  selected <- mgcv::gam(formula, data = data, method = "GCV.Cp")
  sp <- fastkpc_gate_b_sp_value(sp_source, selected$sp, length(selected$sp))
  ref <- fastkpc_mgcv_gam_fixed_sp_reference(
    formula = formula,
    data = data,
    sp = sp,
    method = "GCV.Cp",
    target = 1L,
    S = seq.int(2L, length.out = S_size)
  )
  solved <- fastkpc_mgcv_extract_fixed_sp_solve(
    formula = formula,
    data = data,
    sp = sp,
    method = "GCV.Cp",
    target = 1L,
    S = seq.int(2L, length.out = S_size)
  )
  setup <- solved$setup
  P <- fastkpc_assemble_penalty(ncol(setup$X), setup$S, setup$off, setup$sp, setup$H)
  coef_rel <- fastkpc_relative_l2_diff(solved$coefficients, ref$coefficients)
  fitted_rel <- fastkpc_relative_l2_diff(solved$fitted, ref$fitted)
  residual_rel <- fastkpc_relative_l2_diff(solved$residuals, ref$residuals)
  max_abs_res <- max(abs(solved$residuals - ref$residuals))
  pass <- is.finite(fitted_rel) && is.finite(residual_rel) &&
    fitted_rel <= 1e-5 && residual_rel <= 1e-5 && max_abs_res <= 1e-5
  data.frame(
    scenario_id = scenario_id,
    seed = as.integer(seed),
    n = as.integer(n),
    S_size = as.integer(S_size),
    formula_class = fastkpc_gate_b_formula_class(S_size),
    sp_source = sp_source,
    sp = paste(signif(sp, 8), collapse = "|"),
    edf_reference = as.numeric(ref$edf),
    rank_setup = as.integer(setup$rank[1]),
    constraint_rank = fastkpc_matrix_rank(setup$C),
    penalty_rank = fastkpc_matrix_rank(P),
    coef_rel_l2 = coef_rel,
    fitted_rel_l2 = fitted_rel,
    residual_rel_l2 = residual_rel,
    max_abs_residual_diff = max_abs_res,
    condition_number_proxy = fastkpc_condition_proxy(crossprod(setup$X) + P),
    pass_gate_b = pass,
    warning_message = "",
    stringsAsFactors = FALSE
  )
}

fastkpc_run_mgcv_gate_b_campaign <- function(
    seeds = c(11, 12, 13),
    n_values = c(80, 200, 500),
    S_sizes = c(1L, 2L, 3L, 4L),
    scenarios = c("baseline", "mild_collinearity", "near_constant", "tied_values"),
    sp_grid = c("selected", "small", "medium", "large"),
    output_dir = NULL) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for Gate B campaign", call. = FALSE)
  }
  rows <- list()
  for (scenario in scenarios) {
    for (seed in seeds) {
      for (n in n_values) {
        for (S_size in S_sizes) {
          for (sp_source in sp_grid) {
            row <- tryCatch(
              fastkpc_gate_b_row(scenario, seed, n, S_size, sp_source),
              error = function(e) data.frame(
                scenario_id = scenario,
                seed = as.integer(seed),
                n = as.integer(n),
                S_size = as.integer(S_size),
                formula_class = fastkpc_gate_b_formula_class(S_size),
                sp_source = sp_source,
                sp = "",
                edf_reference = NA_real_,
                rank_setup = NA_integer_,
                constraint_rank = NA_integer_,
                penalty_rank = NA_integer_,
                coef_rel_l2 = NA_real_,
                fitted_rel_l2 = NA_real_,
                residual_rel_l2 = NA_real_,
                max_abs_residual_diff = NA_real_,
                condition_number_proxy = NA_real_,
                pass_gate_b = FALSE,
                warning_message = conditionMessage(e),
                stringsAsFactors = FALSE
              )
            )
            rows[[length(rows) + 1L]] <- row
          }
        }
      }
    }
  }
  fixed_sp <- do.call(rbind, rows)
  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(
      fixed_sp,
      file.path(output_dir, "mgcv_gate_b_fixed_sp_campaign.csv"),
      row.names = FALSE
    )
  }
  list(fixed_sp = fixed_sp, output_dir = output_dir)
}
```

- [ ] **Step 4: Create command wrappers**

Create `fastkpc/tools/run_mgcv_gate_b_campaign.R`:

```r
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) args[[1]] else file.path("fastkpc", "artifacts", "mgcv_gate_b")
source("fastkpc/R/mgcv_gate_b_campaign.R")
result <- fastkpc_run_mgcv_gate_b_campaign(output_dir = output_dir)
cat("wrote:", file.path(output_dir, "mgcv_gate_b_fixed_sp_campaign.csv"), "\n")
cat("rows:", nrow(result$fixed_sp), "\n")
cat("pass_gate_b:", sum(result$fixed_sp$pass_gate_b), "\n")
if (!all(result$fixed_sp$pass_gate_b)) {
  failing <- result$fixed_sp[!result$fixed_sp$pass_gate_b, ]
  print(utils::head(failing[, c("scenario_id", "seed", "n", "S_size", "sp_source", "warning_message")], 10))
  quit(save = "no", status = 1)
}
```

Create `fastkpc/tools/run_mgcv_gate_b_campaign.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
out="${1:-fastkpc/artifacts/mgcv_gate_b}"
Rscript fastkpc/tools/run_mgcv_gate_b_campaign.R "$out"
```

Run:

```bash
chmod +x fastkpc/tools/run_mgcv_gate_b_campaign.sh
```

- [ ] **Step 5: Add campaign test to Gate B runner**

Append to `fastkpc/tools/run_mgcv_gate_b_tests.sh`:

```bash
Rscript fastkpc/tests/test_mgcv_self_solve_purity.R
Rscript fastkpc/tests/test_mgcv_gate_b_campaign.R
```

- [ ] **Step 6: Run campaign tests**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_gate_b_campaign.R
fastkpc/tools/run_mgcv_gate_b_campaign.sh /tmp/fastkpc-mgcv-gate-b
```

Expected:

```text
PASS mgcv Gate B campaign
wrote: /tmp/fastkpc-mgcv-gate-b/mgcv_gate_b_fixed_sp_campaign.csv
```

- [ ] **Step 7: Commit**

```bash
git add fastkpc/R/mgcv_gate_b_campaign.R fastkpc/tests/test_mgcv_gate_b_campaign.R fastkpc/tools/run_mgcv_gate_b_campaign.R fastkpc/tools/run_mgcv_gate_b_campaign.sh fastkpc/tools/run_mgcv_gate_b_tests.sh
git commit -m "test: add mgcv Gate B fixed-sp campaign"
```

## Task 4: Implement Canonical Hybrid Replay

**Files:**
- Modify: `fastkpc/R/hybrid_verifier.R`
- Create: `fastkpc/tests/test_hybrid_canonical_replay.R`

- [ ] **Step 1: Write canonical replay test**

Create `fastkpc/tests/test_hybrid_canonical_replay.R`:

```r
source("fastkpc/R/hybrid_verifier.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

policy <- fastkpc_hybrid_policy(alpha = 0.05, tau = log(3),
                                primary = "fastSplineCPU",
                                verifier = "mgcvExtractCPU")

primary <- data.frame(
  canonical_test_order_id = c(1L, 2L, 3L, 4L),
  conditioning_level = c(1L, 1L, 1L, 1L),
  x = c(1L, 1L, 1L, 2L),
  y = c(2L, 2L, 2L, 3L),
  S_key = c("3", "4", "5", "1"),
  primary_p = c(0.051, 0.90, 0.20, 0.001),
  stringsAsFactors = FALSE
)
verifier <- data.frame(
  canonical_test_order_id = c(4L, 1L),
  verifier_p = c(0.70, 0.001),
  verifier_backend = c("mgcvExtractCPU", "mgcvExtractCPU"),
  stringsAsFactors = FALSE
)

resolved <- fastkpc_apply_hybrid_verifier(primary, verifier, policy)
assert_equal(resolved$canonical_test_order_id, c(1L, 2L, 3L, 4L),
             "resolved rows must follow primary canonical order")
assert_true(resolved$near_alpha_triggered[1], "row 1 should trigger near alpha")
assert_true(resolved$p_used[1] < policy$alpha,
            "row 1 verifier should prevent deletion")
assert_true(resolved$p_used[2] > policy$alpha,
            "row 2 primary should delete edge")
assert_true(resolved$p_used[4] > policy$alpha,
            "row 4 verifier should delete edge despite primary")

replay <- fastkpc_replay_canonical_ci_decisions(
  resolved,
  alpha = policy$alpha,
  p = 5L
)
assert_true(replay$adjacency[1, 2] == FALSE && replay$adjacency[2, 1] == FALSE,
            "edge 1-2 should be deleted")
assert_equal(replay$sepsets[[1]][[2]], 4L,
             "edge 1-2 sepset must be canonical first accepted S=4")
assert_true(replay$diagnostics$edge_deleted[2],
            "row 2 should delete edge 1-2")
assert_true(replay$diagnostics$edge_already_deleted[3],
            "row 3 should be ignored after canonical deletion")
assert_true(replay$adjacency[2, 3] == FALSE && replay$adjacency[3, 2] == FALSE,
            "edge 2-3 should be deleted by verifier row")
assert_equal(replay$sepsets[[2]][[3]], 1L,
             "edge 2-3 sepset must be S=1")

cat("PASS hybrid canonical replay\n")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_canonical_replay.R
```

Expected:

```text
Error ... could not find function "fastkpc_apply_hybrid_verifier"
```

- [ ] **Step 3: Implement verifier join**

Add to `fastkpc/R/hybrid_verifier.R`:

```r
fastkpc_apply_hybrid_verifier <- function(primary_rows, verifier_rows, policy) {
  primary <- as.data.frame(primary_rows, stringsAsFactors = FALSE)
  verifier <- as.data.frame(verifier_rows, stringsAsFactors = FALSE)
  required_primary <- c("canonical_test_order_id", "x", "y", "S_key", "primary_p")
  missing_primary <- setdiff(required_primary, names(primary))
  if (length(missing_primary) > 0L) {
    stop("primary rows missing fields: ", paste(missing_primary, collapse = ", "),
         call. = FALSE)
  }
  if (nrow(verifier) == 0L) {
    verifier <- data.frame(
      canonical_test_order_id = integer(),
      verifier_p = numeric(),
      verifier_backend = character(),
      stringsAsFactors = FALSE
    )
  }
  required_verifier <- c("canonical_test_order_id", "verifier_p")
  missing_verifier <- setdiff(required_verifier, names(verifier))
  if (length(missing_verifier) > 0L) {
    stop("verifier rows missing fields: ", paste(missing_verifier, collapse = ", "),
         call. = FALSE)
  }

  order_key <- primary$canonical_test_order_id
  verifier_idx <- match(order_key, verifier$canonical_test_order_id)
  primary$verifier_p <- NA_real_
  has_match <- !is.na(verifier_idx)
  primary$verifier_p[has_match] <- verifier$verifier_p[verifier_idx[has_match]]
  primary$verifier_backend <- ""
  if ("verifier_backend" %in% names(verifier)) {
    primary$verifier_backend[has_match] <-
      verifier$verifier_backend[verifier_idx[has_match]]
  }

  resolved <- fastkpc_apply_hybrid_policy(primary, policy)
  resolved$verifier_backend[resolved$p_source_used == policy$verifier &
                              !nzchar(resolved$verifier_backend)] <- policy$verifier
  resolved
}
```

- [ ] **Step 4: Implement canonical replay**

Add to `fastkpc/R/hybrid_verifier.R`:

```r
fastkpc_parse_S_key <- function(S_key) {
  if (is.null(S_key) || !nzchar(S_key)) return(integer())
  as.integer(strsplit(as.character(S_key), "\\|", fixed = FALSE)[[1]])
}

fastkpc_replay_canonical_ci_decisions <- function(test_rows, alpha, p,
                                                  initial_adjacency = NULL) {
  rows <- as.data.frame(test_rows, stringsAsFactors = FALSE)
  rows <- rows[order(rows$canonical_test_order_id), , drop = FALSE]
  if (is.null(initial_adjacency)) {
    adjacency <- matrix(TRUE, p, p)
    diag(adjacency) <- FALSE
  } else {
    adjacency <- as.matrix(initial_adjacency)
    storage.mode(adjacency) <- "logical"
  }
  sepsets <- replicate(p, replicate(p, integer(), simplify = FALSE), simplify = FALSE)
  rows$edge_deleted <- FALSE
  rows$edge_already_deleted <- FALSE
  rows$sepset_recorded <- ""

  edge_done <- new.env(parent = emptyenv())
  for (i in seq_len(nrow(rows))) {
    x <- as.integer(rows$x[i])
    y <- as.integer(rows$y[i])
    key <- paste(sort(c(x, y)), collapse = "-")
    if (isTRUE(edge_done[[key]]) || !isTRUE(adjacency[x, y])) {
      rows$edge_already_deleted[i] <- TRUE
      next
    }
    pval <- rows$p_used[i]
    if (!is.finite(pval)) pval <- 0
    if (pval >= alpha) {
      S <- fastkpc_parse_S_key(rows$S_key[i])
      adjacency[x, y] <- FALSE
      adjacency[y, x] <- FALSE
      sepsets[[x]][[y]] <- S
      sepsets[[y]][[x]] <- S
      rows$edge_deleted[i] <- TRUE
      rows$sepset_recorded[i] <- rows$S_key[i]
      edge_done[[key]] <- TRUE
    }
  }
  list(adjacency = adjacency, sepsets = sepsets, diagnostics = rows)
}
```

- [ ] **Step 5: Run canonical replay test**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_canonical_replay.R
```

Expected:

```text
PASS hybrid canonical replay
```

- [ ] **Step 6: Commit**

```bash
git add fastkpc/R/hybrid_verifier.R fastkpc/tests/test_hybrid_canonical_replay.R
git commit -m "feat: add canonical hybrid CI replay"
```

## Task 5: Add Graph-Level Hybrid Replay Test

**Files:**
- Create: `fastkpc/tests/test_hybrid_graph_replay_policy.R`

- [ ] **Step 1: Write graph-level test**

Create `fastkpc/tests/test_hybrid_graph_replay_policy.R`:

```r
source("fastkpc/R/hybrid_verifier.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

policy <- fastkpc_hybrid_policy(alpha = 0.05, tau = log(3),
                                primary = "fastSplineCPU",
                                verifier = "mgcvExtractCPU")

primary <- data.frame(
  canonical_test_order_id = c(1L, 2L, 3L),
  conditioning_level = c(1L, 1L, 1L),
  x = c(1L, 1L, 1L),
  y = c(2L, 2L, 3L),
  S_key = c("3", "4", "2"),
  primary_p = c(0.051, 0.90, 0.049),
  stringsAsFactors = FALSE
)

primary_only <- fastkpc_apply_hybrid_verifier(
  primary,
  data.frame(canonical_test_order_id = integer(), verifier_p = numeric()),
  fastkpc_hybrid_policy(enabled = FALSE, alpha = 0.05,
                        primary = "fastSplineCPU", verifier = "mgcvExtractCPU")
)
primary_graph <- fastkpc_replay_canonical_ci_decisions(primary_only, alpha = 0.05, p = 4L)

verifier <- data.frame(
  canonical_test_order_id = c(3L, 1L),
  verifier_p = c(0.20, 0.001),
  verifier_backend = c("mgcvExtractCPU", "mgcvExtractCPU"),
  stringsAsFactors = FALSE
)
hybrid <- fastkpc_apply_hybrid_verifier(primary, verifier, policy)
hybrid_graph <- fastkpc_replay_canonical_ci_decisions(hybrid, alpha = 0.05, p = 4L)

assert_true(primary_graph$adjacency[1, 2] == FALSE,
            "primary alone deletes edge 1-2")
assert_true(hybrid_graph$adjacency[1, 2] == FALSE,
            "hybrid still deletes edge 1-2 through later canonical row")
assert_true(identical(primary_graph$sepsets[[1]][[2]], 3L),
            "primary alone records first near-alpha sepset")
assert_true(identical(hybrid_graph$sepsets[[1]][[2]], 4L),
            "hybrid records later canonical sepset after verifier prevents row 1")
assert_true(primary_graph$adjacency[1, 3] == TRUE,
            "primary alone keeps edge 1-3")
assert_true(hybrid_graph$adjacency[1, 3] == FALSE,
            "hybrid verifier deletes edge 1-3")
assert_true(sum(hybrid$decision_before_verify != hybrid$decision_after_verify) == 2L,
            "hybrid should record two verifier-induced decision changes")

cat("PASS hybrid graph replay policy\n")
```

- [ ] **Step 2: Run graph-level test**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_graph_replay_policy.R
```

Expected:

```text
PASS hybrid graph replay policy
```

- [ ] **Step 3: Add tests to Gate B runner**

Append to `fastkpc/tools/run_mgcv_gate_b_tests.sh`:

```bash
Rscript fastkpc/tests/test_hybrid_canonical_replay.R
Rscript fastkpc/tests/test_hybrid_graph_replay_policy.R
```

- [ ] **Step 4: Commit**

```bash
git add fastkpc/tests/test_hybrid_graph_replay_policy.R fastkpc/tools/run_mgcv_gate_b_tests.sh
git commit -m "test: add graph-level hybrid replay coverage"
```

## Task 6: Add First Compatibility Campaign Artifacts

**Files:**
- Create: `fastkpc/R/hybrid_compatibility_campaign.R`
- Create: `fastkpc/tests/test_hybrid_compatibility_campaign.R`
- Create: `fastkpc/tools/run_hybrid_compatibility_campaign.R`
- Create: `fastkpc/tools/run_hybrid_compatibility_campaign.sh`

- [ ] **Step 1: Write campaign artifact test**

Create `fastkpc/tests/test_hybrid_compatibility_campaign.R`:

```r
source("fastkpc/R/hybrid_compatibility_campaign.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-hybrid-compat-")
dir.create(out_dir, recursive = TRUE)

campaign <- fastkpc_run_hybrid_compatibility_campaign(output_dir = out_dir)
files <- c(
  "mgcv_residual_compatibility.csv",
  "mgcv_ci_compatibility.csv",
  "mgcv_graph_compatibility.csv",
  "hybrid_near_alpha_diagnostics.csv"
)
for (file in files) {
  assert_true(file.exists(file.path(out_dir, file)),
              paste("missing artifact", file))
}
assert_true(is.data.frame(campaign$ci), "CI artifact should be data.frame")
assert_true(is.data.frame(campaign$graph), "graph artifact should be data.frame")
assert_true("decision_flip_rate" %in% names(campaign$summary),
            "summary should include decision flip rate")
assert_true("near_alpha_fraction" %in% names(campaign$summary),
            "summary should include near-alpha fraction")
assert_true("verifier_decision_changes" %in% names(campaign$summary),
            "summary should include verifier decision changes")

cat("PASS hybrid compatibility campaign\n")
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_compatibility_campaign.R
```

Expected:

```text
Error ... cannot open file 'fastkpc/R/hybrid_compatibility_campaign.R'
```

- [ ] **Step 3: Implement small deterministic campaign**

Create `fastkpc/R/hybrid_compatibility_campaign.R`:

```r
source("fastkpc/R/hybrid_verifier.R")
source("fastkpc/R/mgcv_extract_validation.R")

fastkpc_hybrid_demo_rows <- function(alpha = 0.05) {
  data.frame(
    canonical_test_order_id = seq_len(6),
    conditioning_level = c(0L, 0L, 1L, 1L, 2L, 2L),
    x = c(1L, 1L, 1L, 2L, 2L, 3L),
    y = c(2L, 3L, 2L, 3L, 4L, 4L),
    S_key = c("", "", "3", "1", "1|3", "1|2"),
    p_legacy = c(0.90, 0.001, 0.001, 0.08, 0.20, 0.02),
    primary_p = c(0.90, 0.001, 0.051, 0.04, 0.20, 0.02),
    verifier_p = c(NA, NA, 0.001, 0.08, NA, NA),
    stringsAsFactors = FALSE
  )
}

fastkpc_run_hybrid_compatibility_campaign <- function(output_dir,
                                                      alpha = 0.05,
                                                      tau = log(3)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- fastkpc_hybrid_demo_rows(alpha)
  policy <- fastkpc_hybrid_policy(alpha = alpha, tau = tau,
                                  primary = "fastSplineCPU",
                                  verifier = "mgcvExtractCPU")
  verifier <- rows[is.finite(rows$verifier_p),
                   c("canonical_test_order_id", "verifier_p")]
  verifier$verifier_backend <- "mgcvExtractCPU"
  resolved <- fastkpc_apply_hybrid_verifier(
    rows[, c("canonical_test_order_id", "conditioning_level", "x", "y", "S_key", "primary_p")],
    verifier,
    policy
  )
  replay <- fastkpc_replay_canonical_ci_decisions(resolved, alpha = alpha, p = 4L)
  ci <- data.frame(
    canonical_test_order_id = rows$canonical_test_order_id,
    x = rows$x,
    y = rows$y,
    S_key = rows$S_key,
    conditioning_level = rows$conditioning_level,
    p_legacy = rows$p_legacy,
    p_backend = rows$primary_p,
    p_hybrid = resolved$p_used,
    decision_legacy = rows$p_legacy > alpha,
    decision_backend = rows$primary_p > alpha,
    decision_hybrid = resolved$p_used > alpha,
    decision_flip = (rows$p_legacy > alpha) != (rows$primary_p > alpha),
    hybrid_flip = (rows$p_legacy > alpha) != (resolved$p_used > alpha),
    near_alpha_triggered = resolved$near_alpha_triggered,
    p_source_used = resolved$p_source_used,
    stringsAsFactors = FALSE
  )
  residual <- data.frame(
    scenario = "demo",
    backend = c("fastSplineCPU", "mgcvExtractCPU"),
    residual_correlation = c(0.99, 1.0),
    relative_l2 = c(0.15, 1e-8),
    stringsAsFactors = FALSE
  )
  graph <- data.frame(
    scenario = "demo",
    backend = "hybrid",
    skeleton_shd = sum(ci$hybrid_flip),
    near_alpha_tests = sum(resolved$near_alpha_triggered),
    verifier_calls = sum(is.finite(resolved$verifier_p)),
    verifier_decision_changes = sum(resolved$decision_before_verify != resolved$decision_after_verify),
    stringsAsFactors = FALSE
  )
  summary <- data.frame(
    decision_flip_rate = mean(ci$decision_flip),
    near_alpha_fraction = mean(resolved$near_alpha_triggered),
    verifier_decision_changes = sum(resolved$decision_before_verify != resolved$decision_after_verify),
    stringsAsFactors = FALSE
  )
  utils::write.csv(residual, file.path(output_dir, "mgcv_residual_compatibility.csv"), row.names = FALSE)
  utils::write.csv(ci, file.path(output_dir, "mgcv_ci_compatibility.csv"), row.names = FALSE)
  utils::write.csv(graph, file.path(output_dir, "mgcv_graph_compatibility.csv"), row.names = FALSE)
  utils::write.csv(replay$diagnostics, file.path(output_dir, "hybrid_near_alpha_diagnostics.csv"), row.names = FALSE)
  utils::write.csv(summary, file.path(output_dir, "hybrid_summary.csv"), row.names = FALSE)
  list(residual = residual, ci = ci, graph = graph,
       hybrid = replay$diagnostics, summary = summary,
       output_dir = output_dir)
}
```

This first campaign is intentionally deterministic and small. It proves artifact shape and decision accounting. A later goal may replace the demo p-values with full skeleton execution traces.

- [ ] **Step 4: Create command wrappers**

Create `fastkpc/tools/run_hybrid_compatibility_campaign.R`:

```r
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) args[[1]] else file.path("fastkpc", "artifacts", "hybrid_compatibility")
source("fastkpc/R/hybrid_compatibility_campaign.R")
result <- fastkpc_run_hybrid_compatibility_campaign(output_dir = output_dir)
cat("wrote hybrid compatibility artifacts:", output_dir, "\n")
print(result$summary)
```

Create `fastkpc/tools/run_hybrid_compatibility_campaign.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
out="${1:-fastkpc/artifacts/hybrid_compatibility}"
Rscript fastkpc/tools/run_hybrid_compatibility_campaign.R "$out"
```

Run:

```bash
chmod +x fastkpc/tools/run_hybrid_compatibility_campaign.sh
```

- [ ] **Step 5: Run campaign artifact test**

Run:

```bash
Rscript fastkpc/tests/test_hybrid_compatibility_campaign.R
fastkpc/tools/run_hybrid_compatibility_campaign.sh /tmp/fastkpc-hybrid-compat
```

Expected:

```text
PASS hybrid compatibility campaign
wrote hybrid compatibility artifacts: /tmp/fastkpc-hybrid-compat
```

- [ ] **Step 6: Commit**

```bash
git add fastkpc/R/hybrid_compatibility_campaign.R fastkpc/tests/test_hybrid_compatibility_campaign.R fastkpc/tools/run_hybrid_compatibility_campaign.R fastkpc/tools/run_hybrid_compatibility_campaign.sh
git commit -m "feat: add first hybrid compatibility campaign artifacts"
```

## Task 7: Documentation Update

**Files:**
- Modify: `fastkpc/README.md`
- Modify: `fastkpc/tests/test_mgcv_compat_docs_contract.R`

- [ ] **Step 1: Strengthen docs contract**

Append to `fastkpc/tests/test_mgcv_compat_docs_contract.R`:

```r
assert_grepl("Gate B campaign",
             "README must mention Gate B campaign")
assert_grepl("canonical hybrid replay",
             "README must mention canonical hybrid replay")
assert_grepl("verifier may replace p-values but not replay order",
             "README must state verifier replay invariant")
```

- [ ] **Step 2: Run docs test and verify failure**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R
```

Expected failure mentioning missing Gate B campaign or canonical replay wording.

- [ ] **Step 3: Update README**

Add to the mgcv-compatible section of `fastkpc/README.md`:

```text
Gate B campaign
  `fastkpc/tools/run_mgcv_gate_b_campaign.sh` runs fixed-sp setup/self-solve
  parity scenarios across formula classes, smoothing-parameter scales, sample
  sizes, collinearity, near-constant conditioning variables, and tied values.
  It writes `mgcv_gate_b_fixed_sp_campaign.csv`.

Canonical hybrid replay
  The near-alpha verifier may replace p-values but not replay order. Primary
  rows define `canonical_test_order_id`; verifier rows are joined by that id
  and replayed deterministically. Sepsets are recorded from the canonical first
  separating set after p-value replacement.
```

- [ ] **Step 4: Run docs test**

Run:

```bash
Rscript fastkpc/tests/test_mgcv_compat_docs_contract.R
```

Expected:

```text
PASS mgcv compatibility docs contract
```

- [ ] **Step 5: Commit**

```bash
git add fastkpc/README.md fastkpc/tests/test_mgcv_compat_docs_contract.R
git commit -m "docs: document Gate B campaign and hybrid replay"
```

## Task 8: Final Verification

**Files:**
- No source changes unless verification reveals a real bug.

- [ ] **Step 1: Parse all R files**

Run:

```bash
Rscript -e 'for (f in list.files("fastkpc/R", "\\\\.R$", full.names = TRUE)) { cat("parse", f, "\\n"); parse(f) }'
```

Expected: exit code `0`.

- [ ] **Step 2: Run Gate B test runner**

Run:

```bash
fastkpc/tools/run_mgcv_gate_b_tests.sh
```

Expected output includes:

```text
PASS mgcv self-solve purity
PASS mgcv Gate B campaign
PASS hybrid canonical replay
PASS hybrid graph replay policy
```

- [ ] **Step 3: Run campaign wrappers**

Run:

```bash
fastkpc/tools/run_mgcv_gate_b_campaign.sh /tmp/fastkpc-mgcv-gate-b-final
fastkpc/tools/run_hybrid_compatibility_campaign.sh /tmp/fastkpc-hybrid-compat-final
```

Expected files:

```text
/tmp/fastkpc-mgcv-gate-b-final/mgcv_gate_b_fixed_sp_campaign.csv
/tmp/fastkpc-hybrid-compat-final/mgcv_residual_compatibility.csv
/tmp/fastkpc-hybrid-compat-final/mgcv_ci_compatibility.csv
/tmp/fastkpc-hybrid-compat-final/mgcv_graph_compatibility.csv
/tmp/fastkpc-hybrid-compat-final/hybrid_near_alpha_diagnostics.csv
```

- [ ] **Step 4: Run non-CUDA tests**

Run:

```bash
set -e
for f in fastkpc/tests/test_*.R; do
  case "$f" in
    *cuda*) echo "SKIP $f";;
    *) echo "RUN $f"; Rscript "$f";;
  esac
done
```

Expected: all non-CUDA tests pass; CUDA-named tests are skipped by this command.

- [ ] **Step 5: Inspect changed files**

Run:

```bash
git status --short
git diff --stat
test -z "$(git status --short kpcalg/R)" && echo "kpcalg/R clean" || git status --short kpcalg/R
```

Expected:

```text
Only intended fastkpc/docs files changed.
kpcalg/R clean
```

- [ ] **Step 6: Final commit**

If previous tasks were not already committed:

```bash
git add fastkpc/R fastkpc/tests fastkpc/tools fastkpc/README.md docs/superpowers/plans
git commit -m "feat: harden mgcv Gate B and hybrid replay"
```

## Completion Criteria

This goal is complete only when current evidence proves:

```text
1. fastkpc_mgcv_solve_setup_fixed_sp() exists and consumes pre-extracted setup.
2. The pure setup solve path is protected against mgcv::gam(fit=TRUE), mgcv::magic(), and reference-path leakage.
3. fastkpc_mgcv_extract_fixed_sp_solve() still returns reference comparison diagnostics.
4. GCVBridge residuals equal fixed-sp self-solve residuals at mgcv-selected sp.
5. Gate B campaign runner exists and writes mgcv_gate_b_fixed_sp_campaign.csv.
6. Gate B campaign covers |S|=1, |S|=2, additive |S|=3/4, selected/small/medium/large sp, multiple n values, collinearity, near-constant S, and tied S values.
7. Hybrid verifier join ignores verifier return order and restores canonical primary order.
8. Canonical replay records edge deletion and sepset from the first accepted canonical separating set after p-value replacement.
9. Graph-level hybrid test shows verifier can prevent and induce deletions without changing canonical replay semantics.
10. First compatibility campaign artifacts are written with residual, CI, graph, and hybrid near-alpha diagnostics CSVs.
11. README documents Gate B campaign and canonical hybrid replay invariants.
12. R parse passes.
13. Gate B runner passes.
14. Non-CUDA tests pass.
15. kpcalg/R remains unchanged.
```

## Failure Triage

If Gate B campaign fails:

```text
1. Inspect warning_message for unsupported setup.
2. Check sp length and ordering.
3. Check S/off/H penalty assembly.
4. Check C rank and nullspace dimension.
5. Check near-constant/tied-value rank behavior.
6. Add scenario-specific diagnostics before loosening tolerances.
```

If hybrid replay fails:

```text
1. Verify primary rows contain canonical_test_order_id.
2. Verify verifier rows join by canonical_test_order_id only.
3. Verify p_used uses verifier only when near-alpha and verifier_p is finite.
4. Verify replay sorts by canonical_test_order_id.
5. Verify edge_done prevents later tests from replacing sepset.
6. Verify S_key parsing preserves integer conditioning sets.
```

If campaign artifacts exist but graph values look wrong:

```text
1. Compare decision_before_verify and decision_after_verify.
2. Count verifier-induced decision changes.
3. Compare primary-only replay against hybrid replay.
4. Inspect sepset_recorded per canonical row.
```

## Future Work After This Goal

After this goal, the next meaningful work is:

```text
1. Replace demo hybrid compatibility p-values with real layer scheduler CI traces.
2. Add optional mgcvExtractGCVBridge verifier calls for actual near-alpha tests.
3. Compare legacy mgcv, fastSpline, mgcvExtract, and hybrid on small graph scenarios.
4. Use campaign evidence to decide whether mgcvPortGCVPrototype is worth starting.
```

Still defer:

```text
self-contained GCV optimizer
mgcvSubsetCUDA
new fastSpline CUDA kernels
bamGPU
full mgcv clone
```
