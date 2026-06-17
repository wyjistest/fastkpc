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
assert_true(identical(bridge$sp_source, "mgcv"),
            "GCVBridge sp source must be mgcv")
assert_true(identical(bridge$gcv_source, "mgcv"),
            "GCVBridge gcv source must be mgcv")
assert_true(identical(bridge$solve_source, "fastkpc-fixed-sp"),
            "GCVBridge solve source must be fastkpc fixed-sp")
assert_true(isFALSE(bridge$is_self_contained_gcv),
            "GCVBridge must not claim self-contained GCV")
assert_true(max(abs(bridge$residuals - stats::residuals(legacy))) < 1e-5,
            "GCV bridge residuals should match direct legacy fit")
assert_true(max(abs(log(bridge$sp) - log(legacy$sp))) < 1e-8,
            "GCV bridge selected sp should match direct legacy fit")

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

cat("PASS mgcv extract GCV bridge\n")
