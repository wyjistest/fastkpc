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
