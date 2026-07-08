source("fastkpc/R/mgcv_residual_oracle_trace.R")
if (!file.exists("fastkpc/R/mgcv_residual_setup_shadow.R")) {
  stop("mgcv residual setup shadow implementation is missing", call. = FALSE)
}
source("fastkpc/R/mgcv_residual_setup_shadow.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra", "Rcpp")[
  !vapply(c("mgcv", "RSpectra", "Rcpp"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP guarded C++ envelope shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(11437)
n <- 72L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.05),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.05),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.15 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.07),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.11)
)
cases <- data.frame(
  case_id = c("envelope_s1", "envelope_s2", "envelope_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 5L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)

oracle_dir <- tempfile("fastkpc-mgcv-envelope-oracle-")
invisible(fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
))

out_dir <- tempfile("fastkpc-mgcv-envelope-shadow-")
artifact <- fastkpc_run_mgcv_residual_setup_shadow(
  data = data,
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1,
  solver = "cpp_guarded",
  condition_threshold = 1e300,
  native_s_size_limit = 2L
)

required_summary <- c(
  "native_s_size_limit", "outside_envelope_fallback_count",
  "fallback_count", "cpp_guarded_count"
)
missing_summary <- setdiff(required_summary, names(artifact$summary))
assert_true(length(missing_summary) == 0L,
            paste("guarded envelope summary missing",
                  missing_summary[[1L]]))

assert_true(artifact$summary$native_s_size_limit[[1L]] == 2L,
            "summary should record native S-size envelope")
assert_true(artifact$summary$fallback_count[[1L]] == 2L,
            "only the |S|>2 case should fallback both targets")
assert_true(artifact$summary$outside_envelope_fallback_count[[1L]] == 2L,
            "outside-envelope fallback count should cover both |S|>2 targets")
assert_true(artifact$summary$high_condition_fallback_count[[1L]] == 0L,
            "high condition fallback should not be used in this test")
assert_true(artifact$summary$cpp_guarded_count[[1L]] == 4L,
            "|S|<=2 cases should use native C++ guarded solve")

inside <- artifact$cases[artifact$cases$S_size <= 2L, , drop = FALSE]
outside <- artifact$cases[artifact$cases$S_size > 2L, , drop = FALSE]
assert_true(all(!inside$fallback_used_x) && all(!inside$fallback_used_y),
            "|S|<=2 targets should stay on native C++")
assert_true(all(inside$solve_source_x == "fastkpc-native-cpp-fixed-sp") &&
              all(inside$solve_source_y == "fastkpc-native-cpp-fixed-sp"),
            "|S|<=2 solve source should be native C++")
assert_true(all(outside$fallback_used_x) && all(outside$fallback_used_y),
            "|S|>2 targets should fail closed to fallback")
assert_true(all(outside$fallback_reason_x ==
                  "outside_native_s_size_envelope") &&
              all(outside$fallback_reason_y ==
                    "outside_native_s_size_envelope"),
            "|S|>2 fallback should be attributed to the envelope")
assert_true(all(outside$solve_source_x == "fastkpc-fixed-sp") &&
              all(outside$solve_source_y == "fastkpc-fixed-sp"),
            "|S|>2 fallback should use mgcv C_magic fixed-sp solve")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "guarded envelope fallback should not flip oracle decisions")
assert_true(isTRUE(artifact$summary$pass[[1L]]),
            "guarded envelope shadow should pass")

cat("PASS guarded C++ residual envelope shadow\n")
