source("fastkpc/R/legacy_runner.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

missing <- c("graph", "mgcv", "pcalg", "Rcpp")[
  !vapply(c("graph", "mgcv", "pcalg", "Rcpp"),
          requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing) > 0L) {
  cat("SKIP precision compatible legacy mgcv residual C++ backend: missing",
      paste(missing, collapse = ","), "\n")
  quit(save = "no", status = 0)
}

old_env <- Sys.getenv(c(
  "FASTKPC_LEGACY_PARALLEL_CORES",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL",
  "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW"
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

set.seed(21891)
n <- 48L
z1 <- stats::runif(n, -2, 2)
z2 <- stats::rnorm(n)
z3 <- stats::runif(n, -1, 1)
data <- cbind(
  x1 = sin(z1) + stats::rnorm(n, sd = 0.09),
  x2 = cos(z1) + stats::rnorm(n, sd = 0.09),
  x3 = z1 + stats::rnorm(n, sd = 0.05),
  x4 = z2 + 0.2 * z1,
  x5 = z3 + stats::rnorm(n, sd = 0.08),
  x6 = z1 * z3 + stats::rnorm(n, sd = 0.12)
)

run_compatible <- function(num_cores = 1L) {
  fastkpc_legacy_parallel_skeleton(
    data,
    alpha = 0.08,
    max_conditioning_size = 2L,
    ic.method = "dcc.gamma",
    num_cores = num_cores
  )
}

Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = "1")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW")
Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND")
baseline <- run_compatible()
baseline_summary <- baseline$scheduler_diagnostics$summary

Sys.setenv(
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND = "cpp_guarded",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT = "2",
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD = "1e300"
)
backend <- run_compatible()
summary <- backend$scheduler_diagnostics$summary

assert_true(identical(backend$adjacency, baseline$adjacency),
            "guarded mgcv residual C++ backend must preserve adjacency")
assert_true(identical(backend$n.edgetests, baseline$n.edgetests),
            "guarded mgcv residual C++ backend must preserve n.edgetests")

required <- c(
  "legacy_mgcv_residual_backend",
  "legacy_mgcv_r_backend_count",
  "legacy_mgcv_cpp_backend_enabled",
  "legacy_mgcv_cpp_backend_count",
  "legacy_mgcv_cpp_backend_native_count",
  "legacy_mgcv_cpp_backend_fallback_count",
  "legacy_mgcv_cpp_backend_high_condition_fallback_count",
  "legacy_mgcv_cpp_backend_outside_envelope_fallback_count",
  "legacy_mgcv_cpp_backend_error_count",
  "legacy_mgcv_cpp_backend_ms",
  "legacy_mgcv_cpp_backend_input_setup_ms",
  "legacy_mgcv_cpp_backend_gam_fit_ms",
  "legacy_mgcv_cpp_backend_sp_extract_ms",
  "legacy_mgcv_cpp_backend_setup_extract_ms",
  "legacy_mgcv_cpp_backend_condition_ms",
  "legacy_mgcv_cpp_backend_native_solve_ms",
  "legacy_mgcv_cpp_backend_fallback_ms",
  "legacy_mgcv_cpp_backend_s_size_0_count",
  "legacy_mgcv_cpp_backend_s_size_1_count",
  "legacy_mgcv_cpp_backend_s_size_2_count",
  "legacy_mgcv_cpp_backend_s_size_gt2_count",
  "legacy_mgcv_cpp_backend_native_s_size_0_count",
  "legacy_mgcv_cpp_backend_native_s_size_1_count",
  "legacy_mgcv_cpp_backend_native_s_size_2_count",
  "legacy_mgcv_cpp_backend_fallback_s_size_gt2_count",
  "legacy_mgcv_cpp_backend_same_s_native_group_count",
  "legacy_mgcv_cpp_backend_same_s_native_target_count",
  "legacy_mgcv_cpp_backend_same_s_native_max_targets",
  "legacy_mgcv_cpp_backend_same_s_native_mean_targets",
  "legacy_mgcv_cpp_backend_same_s_native_reuse_opportunity_count",
  "legacy_mgcv_cpp_backend_same_s_native_setup_reuse_ratio",
  "legacy_mgcv_cpp_backend_same_s_sp_native_group_count",
  "legacy_mgcv_cpp_backend_same_s_sp_native_target_count",
  "legacy_mgcv_cpp_backend_same_s_sp_native_max_targets",
  "legacy_mgcv_cpp_backend_same_s_sp_native_mean_targets",
  "legacy_mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count",
  "legacy_mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio",
  "legacy_mgcv_cpp_backend_native_s_size_limit",
  "legacy_mgcv_cpp_backend_condition_threshold",
  "legacy_mgcv_cpp_same_s_prefill_enabled",
  "legacy_mgcv_cpp_same_s_prefill_group_count",
  "legacy_mgcv_cpp_same_s_prefill_target_count",
  "legacy_mgcv_cpp_same_s_prefill_cache_insert_count",
  "legacy_mgcv_cpp_same_s_prefill_existing_count",
  "legacy_mgcv_cpp_same_s_prefill_unused_count",
  "legacy_mgcv_cpp_same_s_prefill_ms",
  "legacy_mgcv_cpp_same_s_prefill_error_count"
)
missing_fields <- setdiff(required, names(summary))
assert_true(length(missing_fields) == 0L,
            paste("legacy mgcv C++ backend summary missing",
                  missing_fields[[1L]]))

assert_true(identical(baseline_summary$legacy_mgcv_residual_backend, "r"),
            "default compatible route should keep R mgcv residual authority")
assert_true(!isTRUE(baseline_summary$legacy_mgcv_cpp_backend_enabled),
            "default compatible route should not enable C++ residual backend")
assert_true(identical(summary$legacy_mgcv_residual_backend, "cpp_guarded"),
            "env gate should select cpp_guarded mgcv residual backend")
assert_true(isTRUE(summary$legacy_mgcv_cpp_backend_enabled),
            "cpp_guarded mgcv residual backend should report enabled")
assert_true(summary$legacy_mgcv_cpp_backend_count ==
              summary$legacy_mgcv_residual_request_count,
            "backend should cover every uncached residual request")
assert_true(summary$legacy_mgcv_cpp_backend_native_count > 0L,
            "backend should use native C++ residual replay for supported S")
assert_true(summary$legacy_mgcv_cpp_backend_fallback_count == 0L,
            "max conditioning size 2 route should stay inside native envelope")
assert_true(summary$legacy_mgcv_cpp_backend_input_setup_ms > 0,
            "backend should report input/formula setup time")
assert_true(summary$legacy_mgcv_cpp_backend_gam_fit_ms > 0,
            "backend should report mgcv gam fit time")
assert_true(summary$legacy_mgcv_cpp_backend_setup_extract_ms > 0,
            "backend should report setup extraction time")
assert_true(summary$legacy_mgcv_cpp_backend_condition_ms >= 0,
            "backend should report condition check time")
assert_true(summary$legacy_mgcv_cpp_backend_native_solve_ms > 0,
            "backend should report native fixed-sp solve time")
assert_true(summary$legacy_mgcv_cpp_backend_fallback_ms == 0,
            "native-envelope skeleton should not spend fallback time")
assert_true(summary$legacy_mgcv_cpp_backend_s_size_1_count +
              summary$legacy_mgcv_cpp_backend_s_size_2_count ==
              summary$legacy_mgcv_cpp_backend_count,
            "native-envelope skeleton should account for |S| 1 and 2 calls")
assert_true(summary$legacy_mgcv_cpp_backend_native_s_size_1_count +
              summary$legacy_mgcv_cpp_backend_native_s_size_2_count ==
              summary$legacy_mgcv_cpp_backend_native_count,
            "native |S| counters should account for native backend calls")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_target_count ==
              summary$legacy_mgcv_cpp_backend_native_count,
            "same-S native target count should match native backend calls")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_group_count > 0L,
            "same-S native grouping should report groups")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_max_targets >= 1L,
            "same-S native grouping should report max targets")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_mean_targets >= 1,
            "same-S native grouping should report mean targets")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_reuse_opportunity_count >=
              0L,
            "same-S native reuse opportunity should be nonnegative")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_native_setup_reuse_ratio >=
              0,
            "same-S native setup reuse ratio should be nonnegative")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_sp_native_target_count ==
              summary$legacy_mgcv_cpp_backend_native_count,
            "same-S+sp native target count should match native backend calls")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_sp_native_group_count >=
              summary$legacy_mgcv_cpp_backend_same_s_native_group_count,
            "same-S+sp grouping should be at least as specific as same-S grouping")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count <=
              summary$legacy_mgcv_cpp_backend_same_s_native_reuse_opportunity_count,
            "same-S+sp reuse opportunity should not exceed same-S opportunity")
assert_true(summary$legacy_mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio <=
              summary$legacy_mgcv_cpp_backend_same_s_native_setup_reuse_ratio,
            "same-S+sp reuse ratio should not exceed same-S reuse ratio")
assert_true(summary$legacy_mgcv_cpp_backend_high_condition_fallback_count == 0L,
            "high-condition fallback should not trigger with loose threshold")
assert_true(summary$legacy_mgcv_cpp_backend_error_count == 0L,
            "guarded mgcv residual C++ backend should not report errors")
assert_true(summary$legacy_mgcv_cpp_backend_native_count +
              summary$legacy_mgcv_cpp_backend_fallback_count +
              summary$legacy_mgcv_cpp_backend_error_count ==
              summary$legacy_mgcv_cpp_backend_count,
            "backend native/fallback/error counters should account for calls")
assert_true(summary$legacy_mgcv_r_backend_count == 0L,
            "R mgcv residual authority should not handle native envelope calls")
assert_true(summary$legacy_mgcv_cpp_backend_native_s_size_limit == 2,
            "summary should record backend native S-size envelope")
assert_true(summary$legacy_mgcv_cpp_backend_condition_threshold > 1e100,
            "summary should record backend condition threshold")
assert_true(summary$legacy_mgcv_residual_request_count ==
              baseline_summary$legacy_mgcv_residual_request_count,
            "backend should not change residual request count")
assert_true(!isTRUE(summary$legacy_mgcv_cpp_same_s_prefill_enabled),
            "same-S prefill should remain disabled by default")

Sys.setenv(
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
  FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL = "1"
)
prefill <- run_compatible(num_cores = 2L)
prefill_summary <- prefill$scheduler_diagnostics$summary

assert_true(identical(prefill$adjacency, baseline$adjacency),
            "same-S prefill must preserve adjacency")
assert_true(identical(prefill$n.edgetests, baseline$n.edgetests),
            "same-S prefill must preserve n.edgetests")
assert_true(isTRUE(prefill_summary$legacy_mgcv_cpp_same_s_prefill_enabled),
            "same-S prefill env gate should report enabled")
assert_true(prefill_summary$legacy_mgcv_cpp_same_s_prefill_group_count > 0L,
            "same-S prefill should report grouped conditioning sets")
assert_true(prefill_summary$legacy_mgcv_cpp_same_s_prefill_target_count > 0L,
            "same-S prefill should enumerate native target residual keys")
assert_true(
  prefill_summary$legacy_mgcv_cpp_same_s_prefill_cache_insert_count > 0L,
  "same-S prefill should insert residuals into worker-local cache"
)
assert_true(prefill_summary$legacy_mgcv_cpp_same_s_prefill_error_count == 0L,
            "same-S prefill should not report guarded backend errors")
assert_true(prefill_summary$legacy_mgcv_residual_request_count ==
              baseline_summary$legacy_mgcv_residual_request_count,
            "same-S prefill should not change residual request count")

fallback_metrics <- fastkpc_legacy_runtime_zero()
fallback <- fastkpc_legacy_run_mgcv_residual_pair(
  metrics = fallback_metrics,
  data = data,
  x = 1L,
  y = 2L,
  S = c(3L, 4L, 5L),
  env = fastkpc_legacy_env(),
  cpp_backend_enabled = TRUE,
  cpp_backend_condition_threshold = 1e300,
  cpp_backend_native_s_size_limit = 2L
)
fallback_summary <- fallback$metrics
assert_true(fallback_summary$mgcv_cpp_backend_count == 2L,
            "direct residual pair backend should cover both target residuals")
assert_true(fallback_summary$mgcv_cpp_backend_native_count == 0L,
            "outside envelope residual pair should not use native backend")
assert_true(fallback_summary$mgcv_cpp_backend_fallback_count == 2L,
            "outside envelope residual pair should fallback for both targets")
assert_true(fallback_summary$mgcv_cpp_backend_outside_envelope_fallback_count ==
              2L,
            "direct residual pair should report outside-envelope fallback")
assert_true(fallback_summary$mgcv_cpp_backend_error_count == 0L,
            "outside-envelope fallback should not report backend errors")
assert_true(fallback_summary$mgcv_r_backend_count == 2L,
            "R mgcv residual authority should handle guarded fallbacks")
assert_true(fallback_summary$mgcv_cpp_backend_s_size_gt2_count == 2L,
            "outside-envelope fallback should report |S| > 2 backend calls")
assert_true(fallback_summary$mgcv_cpp_backend_fallback_s_size_gt2_count == 2L,
            "outside-envelope fallback should report |S| > 2 fallbacks")
assert_true(fallback_summary$mgcv_cpp_backend_fallback_ms > 0,
            "outside-envelope fallback should report fallback time")
assert_true(fallback_summary$mgcv_cpp_backend_gam_fit_ms == 0,
            "outside-envelope early fallback should skip mgcv setup extraction")
assert_true(fallback_summary$mgcv_cpp_backend_native_solve_ms == 0,
            "outside-envelope fallback should not run native fixed-sp solve")
assert_true(ncol(fallback$residuals) == 2L && nrow(fallback$residuals) == n,
            "direct residual pair fallback should return two residual columns")

native_metrics <- fastkpc_legacy_runtime_zero()
native <- fastkpc_legacy_run_mgcv_residual_pair(
  metrics = native_metrics,
  data = data,
  x = 1L,
  y = 2L,
  S = c(3L, 4L),
  env = fastkpc_legacy_env(),
  cpp_backend_enabled = TRUE,
  cpp_backend_condition_threshold = 1e300,
  cpp_backend_native_s_size_limit = 2L
)
native_summary <- fastkpc_legacy_runtime_finalize_mgcv_keys(native$metrics)
assert_true(native_summary$mgcv_cpp_backend_native_count == 2L,
            "direct native residual pair should use native backend")
assert_true(native_summary$mgcv_cpp_backend_same_s_native_group_count == 1L,
            "direct native residual pair should report one same-S group")
assert_true(native_summary$mgcv_cpp_backend_same_s_native_target_count == 2L,
            "direct native residual pair should report two same-S targets")
assert_true(native_summary$mgcv_cpp_backend_same_s_native_reuse_opportunity_count ==
              1L,
            "direct native residual pair should expose one setup reuse opportunity")
assert_true(native_summary$mgcv_cpp_backend_same_s_native_setup_reuse_ratio ==
              0.5,
            "direct native residual pair should report exact setup reuse ratio")
assert_true(native_summary$mgcv_cpp_backend_same_s_sp_native_target_count == 2L,
            "direct native residual pair should report same-S+sp targets")
assert_true(native_summary$mgcv_cpp_backend_same_s_sp_native_group_count >= 1L,
            "direct native residual pair should report same-S+sp groups")
assert_true(native_summary$mgcv_cpp_backend_same_s_sp_native_group_count >=
              native_summary$mgcv_cpp_backend_same_s_native_group_count,
            "direct same-S+sp grouping should be at least as specific as same-S")
assert_true(
  native_summary$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count <=
    native_summary$mgcv_cpp_backend_same_s_native_reuse_opportunity_count,
  "direct same-S+sp reuse opportunity should not exceed same-S opportunity"
)

cat("PASS precision compatible legacy mgcv residual C++ backend\n")
