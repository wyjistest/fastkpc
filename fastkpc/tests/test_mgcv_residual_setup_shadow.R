source("fastkpc/R/mgcv_residual_oracle_trace.R")
if (!file.exists("fastkpc/R/mgcv_residual_setup_shadow.R")) {
  stop("mgcv residual setup shadow implementation is missing", call. = FALSE)
}
source("fastkpc/R/mgcv_residual_setup_shadow.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra")[
  !vapply(c("mgcv", "RSpectra"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP mgcv residual setup shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(11409)
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
  case_id = c("setup_shadow_s1", "setup_shadow_s2",
              "setup_shadow_gt2"),
  x = c(1L, 1L, 2L),
  y = c(2L, 6L, 5L),
  S = c("3", "3,5", "3,4,5"),
  role = c("|S|=1", "|S|=2", "|S|>2"),
  stringsAsFactors = FALSE
)

oracle_dir <- tempfile("fastkpc-mgcv-setup-shadow-oracle-")
invisible(fastkpc_run_mgcv_residual_oracle_trace(
  data = data,
  cases = cases,
  alpha = 0.1,
  output_dir = oracle_dir
))

out_dir <- tempfile("fastkpc-mgcv-setup-shadow-")
artifact <- fastkpc_run_mgcv_residual_setup_shadow(
  data = data,
  oracle_dir = oracle_dir,
  output_dir = out_dir,
  alpha = 0.1,
  solver = "cpp"
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "setup shadow summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "setup shadow cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "summary.md")),
            "setup shadow summary.md should be written")

required <- c(
  "case_id", "S_size", "setup_supported", "authoritative",
  "residual_pair_match", "dcov_p_match", "decision_match",
  "decision_flip", "backend_family_x", "backend_family_y",
  "setup_status"
)
missing_fields <- setdiff(required, names(artifact$cases))
assert_true(length(missing_fields) == 0L,
            paste("setup shadow cases missing", missing_fields[[1L]]))

assert_true(artifact$summary$case_count[[1L]] == nrow(cases),
            "setup shadow should evaluate all oracle cases")
assert_true(artifact$summary$setup_supported_count[[1L]] == nrow(cases),
            "setup shadow should support fixed-sp oracle cases")
assert_true(artifact$summary$residual_pair_match_count[[1L]] == nrow(cases),
            "setup shadow should match residual pairs")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "setup shadow should not flip oracle decisions")
assert_true(all(!artifact$cases$authoritative),
            "setup shadow must not be authoritative")
assert_true(all(artifact$cases$backend_family_x == "mgcvExtractCPP"),
            "x setup shadow should use mgcvExtractCPP")
assert_true(all(artifact$cases$backend_family_y == "mgcvExtractCPP"),
            "y setup shadow should use mgcvExtractCPP")
assert_true(all(artifact$cases$solve_source_x ==
                  "fastkpc-native-cpp-fixed-sp"),
            "x setup shadow should use native C++ fixed-sp solve")
assert_true(all(artifact$cases$solve_source_y ==
                  "fastkpc-native-cpp-fixed-sp"),
            "y setup shadow should use native C++ fixed-sp solve")
assert_true(isTRUE(artifact$summary$pass[[1L]]),
            "setup shadow summary should pass")

cat("PASS mgcv residual setup shadow\n")
