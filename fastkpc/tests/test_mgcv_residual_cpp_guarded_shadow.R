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
  cat("SKIP guarded C++ residual shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(11431)
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
  case_id = c("guarded_s1", "guarded_s2", "guarded_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 5L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)

oracle_dir <- tempfile("fastkpc-mgcv-guarded-oracle-")
invisible(fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
))

out_dir <- tempfile("fastkpc-mgcv-guarded-shadow-")
artifact <- fastkpc_run_mgcv_residual_setup_shadow(
  data = data,
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1,
  solver = "cpp_guarded",
  condition_threshold = 0
)

required_cases <- c(
  "fallback_used_x", "fallback_used_y",
  "fallback_reason_x", "fallback_reason_y",
  "normal_matrix_condition_x", "normal_matrix_condition_y"
)
missing_cases <- setdiff(required_cases, names(artifact$cases))
assert_true(length(missing_cases) == 0L,
            paste("guarded cases missing", missing_cases[[1L]]))

required_summary <- c(
  "fallback_count", "high_condition_fallback_count",
  "cpp_guarded_count", "max_normal_matrix_condition",
  "condition_threshold"
)
missing_summary <- setdiff(required_summary, names(artifact$summary))
assert_true(length(missing_summary) == 0L,
            paste("guarded summary missing", missing_summary[[1L]]))

assert_true(artifact$summary$case_count[[1L]] == nrow(cases),
            "guarded shadow should evaluate every oracle case")
assert_true(artifact$summary$setup_supported_count[[1L]] == nrow(cases),
            "guarded shadow should support synthetic fixed-sp cases")
assert_true(artifact$summary$fallback_count[[1L]] == nrow(cases) * 2L,
            "zero threshold should force every target through fallback")
assert_true(artifact$summary$high_condition_fallback_count[[1L]] ==
              artifact$summary$fallback_count[[1L]],
            "forced guarded fallbacks should be attributed to condition")
assert_true(artifact$summary$cpp_guarded_count[[1L]] == 0L,
            "zero threshold should not use native C++ targets")
assert_true(all(artifact$cases$fallback_used_x),
            "x targets should use guarded fallback")
assert_true(all(artifact$cases$fallback_used_y),
            "y targets should use guarded fallback")
assert_true(all(artifact$cases$fallback_reason_x ==
                  "high_normal_matrix_condition"),
            "x fallback reason should be high condition")
assert_true(all(artifact$cases$fallback_reason_y ==
                  "high_normal_matrix_condition"),
            "y fallback reason should be high condition")
assert_true(all(artifact$cases$solve_source_x == "fastkpc-fixed-sp"),
            "x fallback should use mgcv C_magic fixed-sp solve")
assert_true(all(artifact$cases$solve_source_y == "fastkpc-fixed-sp"),
            "y fallback should use mgcv C_magic fixed-sp solve")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "guarded fallback should not flip oracle decisions")
assert_true(isTRUE(artifact$summary$pass[[1L]]),
            "guarded shadow summary should pass under forced fallback")

cat("PASS guarded C++ residual shadow\n")
