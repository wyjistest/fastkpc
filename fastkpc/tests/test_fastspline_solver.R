source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

self <- fastspline_solver_selftest()

check_case <- function(case, label, rss_ratio) {
  assert_true(is.finite(case$fastspline_rss), paste(label, "fastSpline RSS should be finite"))
  assert_true(is.finite(case$linear_rss), paste(label, "linear RSS should be finite"))
  assert_true(case$fastspline_rss < rss_ratio * case$linear_rss,
              paste(label, "fastSpline RSS should improve over linear RSS"))
  assert_true(abs(case$residual_mean) < 1e-8,
              paste(label, "residual mean should be close to zero"))
  assert_true(is.finite(case$selected_lambda),
              paste(label, "selected lambda should be finite"))
  assert_true(case$selected_lambda >= 1e-4 && case$selected_lambda <= 1e4,
              paste(label, "selected lambda should be inside default grid"))
  assert_true(is.finite(case$edf), paste(label, "edf should be finite"))
  assert_true(case$edf >= 1 && case$edf <= case$design_cols,
              paste(label, "edf should be inside design rank bounds"))
}

check_case(self$one_d, "one_d", 0.75)
check_case(self$two_d, "two_d", 0.80)
check_case(self$three_d, "three_d", 0.85)

constant <- self$constant
assert_true(constant$finite_residuals, "constant conditioning residuals should be finite")
assert_true(abs(constant$residual_mean) < 1e-8,
            "constant conditioning residual mean should be close to zero")
assert_true(is.finite(constant$selected_lambda),
            "constant conditioning selected lambda should be finite")

cat("test_fastspline_solver.R: PASS\n")
