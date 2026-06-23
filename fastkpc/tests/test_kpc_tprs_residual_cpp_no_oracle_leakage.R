source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP kpcTprsResidualCPP no-oracle-leakage: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

with_mgcv_forbidden <- function(expr) {
  ns <- asNamespace("mgcv")
  calls <- new.env(parent = emptyenv())
  calls$count <- 0L
  blocker <- function(...) {
    calls$count <- calls$count + 1L
    stop("forbidden mgcv oracle call", call. = FALSE)
  }
  traced <- character()
  for (name in c("gam", "smoothCon", "magic")) {
    if (exists(name, envir = ns, inherits = FALSE)) {
      trace(name, tracer = blocker, print = FALSE, where = ns)
      traced <- c(traced, name)
    }
  }
  on.exit({
    for (name in rev(traced)) {
      try(untrace(name, where = ns), silent = TRUE)
    }
  }, add = TRUE)
  value <- force(expr)
  list(value = value, forbidden_calls = calls$count)
}

set.seed(62340)
n <- 70L
s <- stats::runif(n, -2, 2)
data <- cbind(
  x = sin(s) + stats::rnorm(n, sd = 0.04),
  y = cos(s) + stats::rnorm(n, sd = 0.04),
  z = s + stats::rnorm(n, sd = 0.02),
  w = 0.4 * sin(s) - 0.2 * cos(s) + stats::rnorm(n, sd = 0.06)
)

receipt_guard <- with_mgcv_forbidden(
  fastkpc_execute_ci_kpc_tprs_residual_cpp(
    data = data, x = 1L, y = 2L, S = 3L,
    ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
    hsic_params = list(), permutation_params = list(),
    route = list(setup_fingerprint = "S:3"), role = "primary"
  )
)
assert_true(receipt_guard$forbidden_calls == 0L,
            "candidate executor must not call mgcv oracle functions")
assert_true(receipt_guard$value$residual_backend_executed ==
              "kpcTprsResidualCPP",
            "candidate executor should run standalone backend")

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1",
  kpcTprsResidualCPP_supported = TRUE,
  kpcTprsResidualCPP_backend_version = "kpcTprsResidualCPP-v1"
)
switch_guard <- with_mgcv_forbidden(
  fast_kpc(
    data,
    alpha = 0.05,
    max_conditioning_size = 1,
    engine = "cpu",
    precision = "compatible",
    graph_stage = "skeleton",
    runtime_capabilities = caps
  )
)
trace <- switch_guard$value$diagnostics$precision_trace
conditional <- trace[nzchar(trace$S_key), , drop = FALSE]
assert_true(switch_guard$forbidden_calls == 0L,
            "limited switch candidate path must not call mgcv oracle functions")
assert_true(nrow(conditional) > 0L,
            "no-oracle-leakage test should exercise conditional rows")
assert_true(all(conditional$backend_executed == "kpcTprsResidualCPP"),
            "conditional rows should execute standalone backend")

cat("PASS kpcTprsResidualCPP no-oracle-leakage\n")
