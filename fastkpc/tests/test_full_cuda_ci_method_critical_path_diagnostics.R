source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
near <- function(left, right, tolerance = 1e-7) {
  isTRUE(all.equal(
    as.numeric(left), as.numeric(right), tolerance = tolerance,
    check.attributes = FALSE
  ))
}

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP strict method critical-path diagnostics: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

set.seed(8317)
n <- 48L
p <- 4L
common <- stats::rnorm(n)
data <- sapply(seq_len(p), function(index) {
  common + 0.35 * stats::rnorm(n)
})
colnames(data) <- paste0("x", seq_len(p))

run_cuda <- function(trace_capacity) {
  if (trace_capacity > 0L) {
    Sys.setenv(
      FASTKPC_STRICT_METHOD_CRITICAL_PATH_TRACE_CAPACITY = trace_capacity
    )
  } else {
    Sys.unsetenv("FASTKPC_STRICT_METHOD_CRITICAL_PATH_TRACE_CAPACITY")
  }
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))
  set.seed(707)
  result <- precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 1L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE,
    ci_method = "dcc.perm",
    permutation_params = list(
      replicates = 20L,
      seed = 707L,
      include_observed = TRUE
    )
  )
  list(result = result, rng_state = .Random.seed)
}

on.exit(Sys.unsetenv(
  "FASTKPC_STRICT_METHOD_CRITICAL_PATH_TRACE_CAPACITY"
), add = TRUE)

baseline <- run_cuda(0L)
traced <- run_cuda(10000L)
baseline_result <- baseline$result
result <- traced$result
summary <- result$summary
trace <- result$method_critical_path

assert_true(
  identical(baseline_result$tasks$p_used, result$tasks$p_used) &&
    identical(baseline_result$adjacency, result$adjacency) &&
    identical(baseline_result$sepsets, result$sepsets) &&
    identical(baseline_result$pMax, result$pMax) &&
    identical(baseline_result$n.edgetests, result$n.edgetests) &&
    identical(baseline$rng_state, traced$rng_state),
  "critical-path tracing changed strict dcc.perm semantics"
)
assert_true(
  nrow(baseline_result$method_critical_path) == 0L &&
    baseline_result$summary$method_critical_path_trace_capacity == 0L &&
    baseline_result$summary$method_critical_path_trace_count == 0L &&
    baseline_result$summary$method_critical_path_trace_overflow_count == 0L,
  "critical-path tracing was not disabled by default"
)
assert_true(
  nrow(trace) == summary$method_execution_context_call_count &&
    nrow(trace) == summary$method_critical_path_trace_count &&
    summary$method_critical_path_trace_capacity == 10000L &&
    summary$method_critical_path_trace_overflow_count == 0L,
  "critical-path trace cardinality changed"
)
expected_names <- c(
  "level", "penalty_count", "target_count", "pair_count",
  "first_logical_sequence_id", "last_logical_sequence_id",
  "residual_cache_all_hit", "deferred_svd_submission",
  "preparation_submit_nonblocking", "preparation_ready_at_submit",
  "preparation_ready_after_permutation", "permutation_table_host_ms",
  "preparation_submit_host_ms",
  "identity_build_host_ms", "identity_validation_host_ms",
  "residual_solve_host_ms", "component_cuda_ms",
  "permutation_h2d_submit_host_ms", "pair_cuda_ms",
  "compact_d2h_cuda_ms", "method_total_host_ms",
  "overlap_upper_bound_ms", "overlap_lower_bound_ms",
  "intermediate_host_wait_count",
  "final_host_wait_count"
)
assert_true(
  identical(names(trace), expected_names) &&
    all(trace$target_count >= 2L) && all(trace$pair_count >= 1L) &&
    all(trace$first_logical_sequence_id <= trace$last_logical_sequence_id) &&
    all(trace$intermediate_host_wait_count == 0L) &&
    all(trace$final_host_wait_count == 1L),
  "critical-path trace shape or event-wait census changed"
)
assert_true(
  near(sum(trace$permutation_table_host_ms),
       summary$method_permutation_table_host_ms) &&
    near(sum(trace$preparation_submit_host_ms),
         summary$method_preparation_submit_host_ms) &&
    near(sum(trace$identity_build_host_ms),
         summary$method_request_identity_build_host_ms) &&
    near(sum(trace$identity_validation_host_ms),
         summary$method_request_identity_validation_host_ms) &&
    near(sum(trace$residual_solve_host_ms),
         summary$cuda_residual_solve_host_ms) &&
    near(sum(trace$component_cuda_ms),
         summary$cuda_dcov_component_build_ms) &&
    near(sum(trace$permutation_h2d_submit_host_ms),
         summary$method_permutation_h2d_submit_host_ms) &&
    near(sum(trace$overlap_upper_bound_ms),
         summary$method_permutation_gpu_overlap_upper_bound_ms) &&
    near(sum(trace$overlap_lower_bound_ms),
         summary$method_permutation_gpu_overlap_lower_bound_ms),
  "critical-path trace timing accounting changed"
)
assert_true(
  all(trace$overlap_upper_bound_ms <= trace$permutation_table_host_ms) &&
    all(trace$overlap_lower_bound_ms <= trace$overlap_upper_bound_ms) &&
    all(trace$overlap_lower_bound_ms <= trace$permutation_table_host_ms) &&
    all(trace$overlap_upper_bound_ms <= pmax(
      trace$residual_solve_host_ms + trace$component_cuda_ms,
      trace$overlap_lower_bound_ms
    )) &&
    isTRUE(summary$method_permutation_gpu_overlap_enabled) &&
    summary$method_preparation_submit_before_rng_count == nrow(trace) &&
    summary$method_deferred_preparation_error_count == 0L &&
    summary$method_preparation_ready_at_submit_count ==
      sum(trace$preparation_ready_at_submit) &&
    summary$method_preparation_ready_after_permutation_count ==
      sum(trace$preparation_ready_after_permutation) &&
    summary$method_component_host_wait_count == 0L &&
    summary$method_pair_host_wait_count == 0L &&
    summary$method_compact_host_wait_count == nrow(trace) &&
    summary$method_consumer_host_wait_count == 0L &&
    summary$method_intermediate_host_event_wait_count == 0L &&
    summary$method_final_result_host_event_wait_count == nrow(trace) &&
    summary$method_in_flight_peak == 1L &&
    summary$method_submit_hidden_stream_sync_count == 0L &&
    summary$method_submit_hidden_device_sync_count == 0L &&
    summary$method_submit_completion_event_wait_count == 0L,
  "critical-path overlap bound or host-wait accounting changed"
)

cat("test_full_cuda_ci_method_critical_path_diagnostics.R: PASS\n")
