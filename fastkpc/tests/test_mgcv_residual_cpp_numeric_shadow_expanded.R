source("fastkpc/R/mgcv_residual_oracle_trace.R")
if (!file.exists("fastkpc/R/mgcv_residual_cpp_numeric_shadow_expanded.R")) {
  stop("expanded C++ numeric shadow implementation is missing", call. = FALSE)
}
source("fastkpc/R/mgcv_residual_cpp_numeric_shadow_expanded.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "RSpectra", "Rcpp")[
  !vapply(c("mgcv", "RSpectra", "Rcpp"), requireNamespace,
          logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP expanded C++ numeric shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

make_entry <- function(x, y, S) {
  list(x = x, y = y, S_xy = as.integer(S), S_yx = integer())
}

set.seed(43109)
n <- 68L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.05),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.05),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.15 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.07),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.11),
  x7 = z2 * z3 + stats::rnorm(n, sd = 0.11)
)
pMax <- matrix(0.5, ncol(data), ncol(data))
for (i in seq_len(ncol(data))) {
  for (j in seq_len(ncol(data))) {
    pMax[i, j] <- 0.1 + abs(i - j) * 0.0002 + (i + j) * 0.00001
  }
}
source_result <- list(
  skeleton = list(
    pMax = pMax,
    per.level.log = list(
      list(
        make_entry(1L, 2L, 3L),
        make_entry(1L, 3L, 4L),
        make_entry(2L, 4L, 5L)
      ),
      list(
        make_entry(3L, 5L, c(1L, 2L)),
        make_entry(4L, 6L, c(1L, 3L)),
        make_entry(5L, 7L, c(2L, 4L))
      ),
      list(
        make_entry(6L, 7L, c(1L, 2L, 3L)),
        make_entry(2L, 7L, c(1L, 4L, 5L))
      )
    )
  )
)

out_dir <- tempfile("fastkpc-expanded-cpp-numeric-shadow-")
artifact <- fastkpc_run_mgcv_residual_cpp_numeric_shadow_expanded(
  data = data,
  source_result = source_result,
  output_dir = out_dir,
  alpha = 0.1,
  near_alpha_count = 4L,
  per_s_size_count = 2L,
  per_level_count = 2L,
  max_cases = 10L
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "expanded shadow summary.csv should be written")
assert_true(file.exists(file.path(out_dir, "cases.csv")),
            "expanded shadow cases.csv should be written")
assert_true(file.exists(file.path(out_dir, "oracle", "summary.csv")),
            "expanded shadow oracle summary.csv should be written")
assert_true(artifact$summary$case_count[[1L]] > 4L,
            "expanded shadow should run more than tiny smoke cases")
assert_true(artifact$summary$setup_supported_count[[1L]] ==
              artifact$summary$case_count[[1L]],
            "expanded shadow should support all synthetic cases")
assert_true(artifact$summary$decision_flip_count[[1L]] == 0L,
            "expanded shadow should not flip decisions")
assert_true(all(artifact$cases$backend_family_x == "mgcvExtractCPP"),
            "expanded shadow x backend should be native C++")
assert_true(all(artifact$cases$backend_family_y == "mgcvExtractCPP"),
            "expanded shadow y backend should be native C++")
assert_true(isTRUE(artifact$summary$pass[[1L]]),
            "expanded shadow summary should pass")

cat("PASS mgcv residual C++ numeric expanded shadow\n")
