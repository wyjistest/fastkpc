source("fastkpc/R/kpc_tprs_residual_cpp.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(62497)
n <- 80L
s <- stats::runif(n, -2, 2)
y <- sin(1.3 * s) + 0.2 * s + stats::rnorm(n, sd = 0.04)
S <- matrix(s + stats::rnorm(n, sd = 0.01), ncol = 1L)

full <- fastkpc_kpc_tprs_gcv_candidate(
  y = y,
  S = S,
  diagnostics_level = "full"
)
compact <- fastkpc_kpc_tprs_gcv_candidate(
  y = y,
  S = S,
  diagnostics_level = "compact"
)

assert_true(is.data.frame(full$grid) && nrow(full$grid) > 0L,
            "full diagnostics should keep GCV grid")
assert_true(is.data.frame(full$diagnostics$magic1d_trace[[1L]]) &&
              nrow(full$diagnostics$magic1d_trace[[1L]]) > 0L,
            "full diagnostics should keep magic trace")
assert_true(is.null(compact$grid),
            "compact diagnostics should omit GCV grid data frame")
assert_true(is.null(compact$diagnostics$magic1d_trace),
            "compact diagnostics should omit magic trace data frame")
assert_true(abs(log(full$selected_sp / compact$selected_sp)) < 1e-10,
            "compact selected lambda should match full")
assert_true(abs(full$edf - compact$edf) < 1e-8,
            "compact EDF should match full")
rel <- fastkpc_kpc_tprs_rel_l2(full$residuals, compact$residuals)
assert_true(is.finite(rel) && rel < 1e-10,
            "compact residuals should match full")

cat("PASS kpcTprsResidualCPP compact diagnostics\n")
