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
