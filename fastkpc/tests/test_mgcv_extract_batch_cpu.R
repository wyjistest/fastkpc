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
assert_true(!identical(batch$sp[[1]], batch$sp[[2]]),
            "targets must not share one selected smoothing parameter")
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
