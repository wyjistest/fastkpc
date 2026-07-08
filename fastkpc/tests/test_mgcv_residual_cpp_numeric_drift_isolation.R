source("fastkpc/R/mgcv_residual_oracle_trace.R")
source("fastkpc/R/mgcv_residual_setup_shadow.R")
if (!file.exists("fastkpc/R/mgcv_residual_cpp_numeric_drift_isolation.R")) {
  stop("C++ numeric drift isolation implementation is missing", call. = FALSE)
}
source("fastkpc/R/mgcv_residual_cpp_numeric_drift_isolation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra", "Rcpp")[
  !vapply(c("mgcv", "RSpectra", "Rcpp"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP C++ numeric drift isolation: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(43113)
n <- 70L
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
  case_id = c("drift_iso_s2", "drift_iso_gt2"),
  x = c(1L, 2L),
  y = c(6L, 5L),
  S = c("3,5", "3,4,5"),
  role = c("|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)

oracle_dir <- tempfile("fastkpc-drift-isolation-oracle-")
invisible(fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
))
shadow_dir <- tempfile("fastkpc-drift-isolation-shadow-")
invisible(fastkpc_run_mgcv_residual_setup_shadow(
  data = data,
  oracle_dir = oracle_dir,
  output_dir = shadow_dir,
  alpha = 0.1,
  solver = "cpp"
))

out_dir <- tempfile("fastkpc-drift-isolation-")
artifact <- fastkpc_run_mgcv_residual_cpp_numeric_drift_isolation(
  data = data,
  shadow_dir = shadow_dir,
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1,
  include_all = TRUE
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "drift isolation summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "drift isolation cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "targets.csv")),
            "drift isolation targets.csv should be written")

required_summary <- c(
  "case_count", "target_count", "cpp_matches_r_normal_target_count",
  "cmagic_matches_oracle_target_count", "decision_flip_count"
)
missing_summary <- setdiff(required_summary, names(artifact$summary))
assert_true(length(missing_summary) == 0L,
            paste("drift summary missing", missing_summary[[1L]]))
required_targets <- c(
  "case_id", "target_role", "cpp_vs_r_normal_max_abs_diff",
  "cpp_vs_cmagic_max_abs_diff", "cmagic_vs_oracle_max_abs_diff",
  "r_normal_vs_oracle_max_abs_diff", "normal_matrix_condition"
)
missing_targets <- setdiff(required_targets, names(artifact$targets))
assert_true(length(missing_targets) == 0L,
            paste("drift targets missing", missing_targets[[1L]]))

assert_true(artifact$summary$case_count[[1L]] == nrow(cases),
            "include_all should isolate every synthetic case")
assert_true(artifact$summary$target_count[[1L]] == nrow(cases) * 2L,
            "each case should produce x and y target diagnostics")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "drift isolation should preserve decisions for synthetic cases")
assert_true(max(artifact$targets$cpp_vs_r_normal_max_abs_diff) < 1e-7,
            "native C++ and R normal-equation solves should agree")

cat("PASS mgcv residual C++ numeric drift isolation\n")
