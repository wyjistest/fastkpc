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
  method = "GCV.Cp",
  target = 1L,
  S = c(2L, 3L)
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
