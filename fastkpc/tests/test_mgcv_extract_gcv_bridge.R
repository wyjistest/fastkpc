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
