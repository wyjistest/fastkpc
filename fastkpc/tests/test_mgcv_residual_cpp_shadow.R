source("fastkpc/R/mgcv_residual_oracle_trace.R")
source("fastkpc/R/mgcv_residual_cpp_shadow.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra", "Rcpp")[
  !vapply(c("mgcv", "RSpectra", "Rcpp"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP mgcv residual cpp shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(11403)
n <- 64L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.06),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.06),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.2 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12)
)
cases <- data.frame(
  case_id = c("cpp_shadow_s1", "cpp_shadow_s2", "cpp_shadow_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 5L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)
oracle_dir <- tempfile("fastkpc-mgcv-cpp-shadow-oracle-")
invisible(fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
))

out_dir <- tempfile("fastkpc-mgcv-cpp-shadow-")
artifact <- fastkpc_run_mgcv_residual_cpp_shadow(
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "cpp shadow summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "cpp shadow cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "summary.md")),
            "cpp shadow summary.md should be written")

required <- c(
  "case_id", "S_size", "cpp_supported", "authoritative",
  "residual_pair_match", "dcov_p_match", "decision_match",
  "decision_flip", "backend_family", "replay_status"
)
missing_fields <- setdiff(required, names(artifact$cases))
assert_true(length(missing_fields) == 0L,
            paste("cpp shadow cases missing", missing_fields[[1L]]))

assert_true(artifact$summary$case_count[[1L]] == nrow(cases),
            "cpp shadow should evaluate all oracle cases")
assert_true(artifact$summary$cpp_supported_count[[1L]] == nrow(cases),
            "cpp shadow should support captured setup oracle cases")
assert_true(artifact$summary$residual_pair_match_count[[1L]] == nrow(cases),
            "cpp shadow should match residual pairs")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "cpp shadow should not flip oracle decisions")
assert_true(all(!artifact$cases$authoritative),
            "cpp shadow must not be authoritative")
assert_true(all(artifact$cases$backend_family == "mgcvCapturedCppReplay"),
            "cpp shadow should identify the captured C++ replay backend")
assert_true(isTRUE(artifact$summary$pass[[1L]]),
            "cpp shadow summary should pass")

cat("PASS mgcv residual cpp shadow\n")
