source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "RSpectra", "Rcpp")[
  !vapply(c("graph", "mgcv", "pcalg", "RSpectra", "Rcpp"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual C++ shadow: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_env <- Sys.getenv(c(
  "FASTKPC_LEGACY_PARALLEL_CORES",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE"
), unset = NA_character_)
restore_env <- function(name, value) {
  if (is.na(value)) {
    Sys.unsetenv(name)
  } else {
    do.call(Sys.setenv, stats::setNames(list(value), name))
  }
}
on.exit({
  for (name in names(old_env)) restore_env(name, old_env[[name]])
}, add = TRUE)

set.seed(11443)
n <- 66L
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

run_compatible <- function() {
  fastkpc_legacy_parallel_skeleton(
    data,
    alpha = 0.08,
    max_conditioning_size = 2,
    ic.method = "dcc.gamma",
    num_cores = 1L
  )
}

Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = "1")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW")
baseline <- run_compatible()
baseline_summary <- baseline$scheduler_diagnostics$summary

Sys.setenv(
  FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT = "2",
  FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD = "1e300"
)
shadow <- run_compatible()
summary <- shadow$scheduler_diagnostics$summary

assert_true(identical(shadow$adjacency,
                      baseline$adjacency),
            "mgcv residual C++ shadow must not change adjacency")
assert_true(identical(shadow$n.edgetests,
                      baseline$n.edgetests),
            "mgcv residual C++ shadow must not change n.edgetests")

required <- c(
  "legacy_mgcv_cpp_shadow_enabled",
  "legacy_mgcv_cpp_shadow_count",
  "legacy_mgcv_cpp_shadow_native_count",
  "legacy_mgcv_cpp_shadow_fallback_count",
  "legacy_mgcv_cpp_shadow_high_condition_fallback_count",
  "legacy_mgcv_cpp_shadow_outside_envelope_fallback_count",
  "legacy_mgcv_cpp_shadow_error_count",
  "legacy_mgcv_cpp_shadow_residual_mismatch_count",
  "legacy_mgcv_cpp_shadow_max_abs_diff",
  "legacy_mgcv_cpp_shadow_max_rel_l2",
  "legacy_mgcv_cpp_shadow_native_s_size_limit",
  "legacy_mgcv_cpp_shadow_condition_threshold"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy mgcv C++ shadow summary missing",
                  missing_fields[[1L]]))

assert_true(isTRUE(summary$legacy_mgcv_cpp_shadow_enabled),
            "mgcv residual C++ shadow should be enabled")
assert_true(summary$legacy_mgcv_cpp_shadow_count > 0L,
            "mgcv residual C++ shadow should evaluate target residuals")
assert_true(summary$legacy_mgcv_cpp_shadow_count ==
              summary$legacy_mgcv_residual_request_count,
            "shadow count should cover every residual request without cache")
assert_true(summary$legacy_mgcv_cpp_shadow_native_count > 0L,
            "native C++ shadow count should be recorded")
assert_true(summary$legacy_mgcv_cpp_shadow_fallback_count == 0L,
            "S-size limit 2 should not fallback in max conditioning size 2 run")
assert_true(summary$legacy_mgcv_cpp_shadow_error_count == 0L,
            "mgcv residual C++ shadow should not error")
assert_true(summary$legacy_mgcv_cpp_shadow_residual_mismatch_count == 0L,
            "mgcv residual C++ shadow should match authoritative residuals")
assert_true(summary$legacy_mgcv_cpp_shadow_max_abs_diff <= 1e-5,
            "mgcv residual C++ shadow max abs diff should be within tolerance")
assert_true(summary$legacy_mgcv_cpp_shadow_max_rel_l2 <= 1e-5,
            "mgcv residual C++ shadow max relative L2 should be within tolerance")
assert_true(summary$legacy_mgcv_cpp_shadow_native_s_size_limit == 2,
            "summary should record native S-size shadow envelope")
assert_true(summary$legacy_mgcv_cpp_shadow_condition_threshold > 1e100,
            "summary should record condition threshold")
assert_true(summary$legacy_mgcv_residual_request_count ==
              baseline_summary$legacy_mgcv_residual_request_count,
            "shadow should not change residual request count")

cat("PASS precision compatible legacy mgcv residual C++ shadow\n")
