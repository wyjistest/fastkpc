source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("mgcv", "Rcpp")[
  !vapply(c("mgcv", "Rcpp"), requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP legacy same-S fixed-sp batch provider: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_env <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE",
                      unset = NA_character_)
on.exit({
  if (is.na(old_env)) {
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE")
  } else {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE = old_env)
  }
}, add = TRUE)

set.seed(250)
n <- 58
s1 <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(s1) + stats::rnorm(n, sd = 0.04),
  x2 = cos(s1) + stats::rnorm(n, sd = 0.04),
  x3 = sin(2 * s1) + stats::rnorm(n, sd = 0.04),
  s1 = s1
)
env <- fastkpc_legacy_env()

Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE")
baseline <- fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group(
  metrics = fastkpc_legacy_runtime_zero(),
  data = data,
  targets = c(1L, 2L, 3L),
  S = 4L,
  env = env,
  condition_threshold = 1e300,
  native_s_size_limit = 2L
)

Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE = "cpp")
batched <- fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group(
  metrics = fastkpc_legacy_runtime_zero(),
  data = data,
  targets = c(1L, 2L, 3L),
  S = 4L,
  env = env,
  condition_threshold = 1e300,
  native_s_size_limit = 2L
)

assert_true(length(batched$keys) == length(baseline$keys),
            "batched provider should return the same residual key count")
assert_true(setequal(batched$keys, baseline$keys),
            "batched provider should return the same residual keys")
for (key in baseline$keys) {
  assert_true(max(abs(batched$residuals[[key]] - baseline$residuals[[key]])) < 1e-8,
              paste("batched residual should match baseline provider for", key))
}

metrics <- batched$metrics
assert_true(identical(as.integer(metrics$mgcv_cpp_same_s_setup_provider_batch_solve_enabled), 1L),
            "batch solve provider diagnostics should report enabled")
assert_true(metrics$mgcv_cpp_same_s_setup_provider_batch_solve_group_count > 0L,
            "batch solve provider should report grouped solves")
assert_true(metrics$mgcv_cpp_same_s_setup_provider_batch_solve_target_count ==
              length(batched$keys),
            "batch solve target count should match returned residual count")
assert_true(metrics$mgcv_cpp_same_s_setup_provider_batch_solve_error_count == 0L,
            "batch solve provider should not report errors")
assert_true(metrics$mgcv_cpp_same_s_setup_provider_template_count == 1L,
            "batch solve provider should use one setup template")
assert_true(metrics$mgcv_cpp_same_s_setup_provider_reuse_count ==
              length(batched$keys) - 1L,
            "batch solve provider should report same-S setup reuse")
assert_true(metrics$mgcv_cpp_backend_native_count == length(batched$keys),
            "batch solve provider should keep native backend counts")
assert_true(metrics$mgcv_fit_count == length(batched$keys),
            "batch solve provider should preserve per-target mgcv fit count")

cat("PASS legacy same-S fixed-sp batch provider\n")
