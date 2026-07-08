if (!file.exists("fastkpc/R/legacy_mgcv_residual_cpp_shadow_artifact.R")) {
  stop("legacy mgcv residual C++ shadow artifact runner is missing",
       call. = FALSE)
}
source("fastkpc/R/legacy_mgcv_residual_cpp_shadow_artifact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "RSpectra", "Rcpp")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra", "Rcpp"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy mgcv residual C++ shadow artifact: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

set.seed(11457)
n <- 56L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.07),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.07),
  x3 = z1 + stats::rnorm(n, sd = 0.04),
  x4 = z2 + 0.15 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12)
)

out_dir <- tempfile("fastkpc-legacy-mgcv-cpp-shadow-artifact-")
old_backend_env <- Sys.getenv(c(
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD"
), unset = NA_character_)
restore_env <- function(name, value) {
  if (is.na(value)) {
    Sys.unsetenv(name)
  } else {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}
on.exit({
  for (name in names(old_backend_env)) restore_env(name, old_backend_env[[name]])
}, add = TRUE)
Sys.setenv(
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND = "cpp_guarded",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT = "2",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD = "1e300"
)
artifact <- fastkpc_run_legacy_mgcv_residual_cpp_shadow_artifact(
  data = data,
  output_dir = out_dir,
  alpha = 0.08,
  max_conditioning_size = 2L,
  num_cores = 1L,
  native_s_size_limit = 2L,
  condition_threshold = 1e300,
  dcov_backend = "r"
)

assert_true(file.exists(file.path(out_dir, "summary.csv")),
            "artifact runner should write summary.csv")
assert_true(file.exists(file.path(out_dir, "result.rds")),
            "artifact runner should write result.rds")
assert_true(file.exists(file.path(out_dir, "summary.md")),
            "artifact runner should write summary.md")

required <- c(
  "artifact", "adjacency_identical", "n_edgetests_identical",
  "residual_request_count", "shadow_count", "native_count",
  "fallback_count", "error_count", "residual_mismatch_count",
  "max_abs_diff", "max_rel_l2", "native_s_size_limit",
  "condition_threshold"
)
missing_fields <- setdiff(required, names(artifact$summary))
assert_true(length(missing_fields) == 0L,
            paste("artifact summary missing", missing_fields[[1L]]))

assert_true(isTRUE(artifact$summary$adjacency_identical[[1L]]),
            "shadow artifact should preserve adjacency")
assert_true(isTRUE(artifact$summary$n_edgetests_identical[[1L]]),
            "shadow artifact should preserve n.edgetests")
assert_true(artifact$summary$residual_request_count[[1L]] > 0L,
            "shadow artifact should exercise residual requests")
assert_true(artifact$summary$shadow_count[[1L]] ==
              artifact$summary$residual_request_count[[1L]],
            "shadow artifact should cover every uncached residual request")
assert_true(artifact$summary$native_count[[1L]] > 0L,
            "shadow artifact should use native C++ residual replay")
assert_true(artifact$summary$fallback_count[[1L]] == 0L,
            "S-size 2 artifact should not fallback")
assert_true(artifact$summary$error_count[[1L]] == 0L,
            "shadow artifact should not report errors")
assert_true(artifact$summary$residual_mismatch_count[[1L]] == 0L,
            "shadow artifact should not report residual mismatches")
assert_true(artifact$summary$max_abs_diff[[1L]] <= 1e-5,
            "shadow artifact residual abs diff should be within tolerance")
assert_true(identical(
  artifact$baseline$scheduler_diagnostics$summary$legacy_mgcv_residual_backend,
  "r"
), "shadow artifact baseline should not inherit mgcv residual backend env")
assert_true(identical(
  artifact$shadow$scheduler_diagnostics$summary$legacy_mgcv_residual_backend,
  "r"
), "shadow artifact shadow run should not inherit mgcv residual backend env")

cat("PASS legacy mgcv residual C++ shadow artifact\n")
