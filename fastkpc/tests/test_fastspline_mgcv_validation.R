source("fastkpc/R/fastspline_validation.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

if (!requireNamespace("mgcv", quietly = TRUE)) {
  stop("mgcv is required for this validation", call. = FALSE)
}

result <- validate_fastspline_against_mgcv(seed = 61, n = 160)

assert_true(result$one_d$residual_correlation >= 0.97,
            "one_d residual correlation should be close to mgcv")
assert_true(result$one_d$relative_residual_l2 <= 0.35,
            "one_d relative residual L2 should be bounded")
assert_true(result$two_d$residual_correlation >= 0.85,
            "two_d residual correlation should be close enough to mgcv")
assert_true(result$two_d$relative_residual_l2 <= 0.60,
            "two_d relative residual L2 should be bounded")
assert_true(is.finite(result$one_d$dcov_pvalue_abs_diff),
            "one_d dCov p-value difference should be finite")
assert_true(is.finite(result$two_d$dcov_pvalue_abs_diff),
            "two_d dCov p-value difference should be finite")

assert_true(is.list(result$graph), "graph section should exist")
assert_true(is.logical(result$graph$available), "graph availability should be recorded")
if (!result$graph$available) {
  assert_true(grepl("pcalg", result$graph$reason_if_unavailable, fixed = TRUE),
              "unavailable graph comparison should record pcalg reason")
} else {
  assert_true(is.list(result$graph$diff$adjacency), "graph diff should include adjacency")
  assert_true(is.list(result$graph$diff$pMax), "graph diff should include pMax")
  assert_true(is.list(result$graph$diff$sepsets), "graph diff should include sepsets")
  assert_true(is.list(result$graph$diff$n_edgetests), "graph diff should include n_edgetests")
}

cat("test_fastspline_mgcv_validation.R: PASS\n")
