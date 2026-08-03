source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!isTRUE(tryCatch(
  fastkpc_cuda_available(),
  error = function(error) FALSE
))) {
  cat("SKIP Phase 10 dataset cache epoch: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

make_data <- function(seed) {
  set.seed(seed)
  n <- 72L
  z <- stats::runif(n, -2, 2)
  cbind(
    x1 = sin(z) + stats::rnorm(n, sd = 0.08),
    x2 = cos(z) + stats::rnorm(n, sd = 0.08),
    x3 = z + stats::rnorm(n, sd = 0.08),
    x4 = z^2 + stats::rnorm(n, sd = 0.08)
  )
}

run_candidate <- function(data) {
  precision_run_skeleton_full_cuda_native(
    data = data,
    alpha = 0.1,
    max_conditioning_size = 1L,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    compatible_cuda_strict = TRUE
  )
}

data_a <- make_data(6110L)
data_b <- make_data(6111L)
invisible(full_cuda_ci_one_call_cache_control_native("configure", 4096L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 4096L
))

before <- full_cuda_ci_one_call_cache_state_native(data_a)
invisible(full_cuda_ci_one_call_cache_control_native("reset"))
empty <- full_cuda_ci_one_call_cache_state_native(data_a)
assert_true(
  identical(empty$schema_version, "full-cuda-ci-dataset-cache-state-v2") &&
    identical(empty$dataset_key, before$dataset_key) &&
    empty$cache_epoch > before$cache_epoch &&
    empty$result_cache_entries == 0L &&
    empty$result_cache_dataset_entries == 0L &&
    empty$target_cache_entries == 0L &&
    empty$target_cache_dataset_entries == 0L,
  "Phase 10 dataset cache reset did not advance an empty epoch"
)

cold <- run_candidate(data_a)
summary <- cold$summary
assert_true(
  identical(summary$dataset_key, empty$dataset_key) &&
    summary$dataset_cache_epoch_at_start == empty$cache_epoch &&
    summary$result_cache_warm_start_entries == 0L &&
    summary$result_cache_dataset_warm_start_entries == 0L &&
    summary$result_cache_preexisting_hit_count == 0L &&
    summary$target_cache_warm_start_entries == 0L &&
    summary$target_cache_dataset_warm_start_entries == 0L &&
    summary$target_cache_preexisting_hit_count == 0L &&
    summary$native_setup_cache_warm_start_entries == 0L &&
    summary$residual_cache_warm_start_entries == 0L &&
    summary$component_cache_warm_start_entries == 0L &&
    summary$unique_prepared_s_key_count > 0L &&
    summary$physical_prepared_s_key_count >=
      summary$unique_prepared_s_key_count &&
    summary$speculative_prepared_s_build_count ==
      summary$physical_prepared_s_key_count -
        summary$unique_prepared_s_key_count &&
    summary$unique_target_key_count > 0L &&
    summary$unique_residual_key_count > 0L &&
    summary$unique_target_key_count == summary$target_cache_miss_count &&
    summary$native_setup_cache_hit_count ==
      summary$native_setup_device_cache_hit_count +
        summary$native_setup_host_cache_hit_count &&
    summary$native_setup_device_rehydrate_count ==
      summary$native_setup_host_cache_hit_count &&
    summary$native_setup_host_cache_peak_entries <=
      summary$native_setup_host_cache_capacity &&
    summary$physical_target_optimization_count ==
      summary$cuda_single_penalty_target_count +
        summary$cuda_multi_penalty_target_count &&
    summary$prefill_target_optimization_count ==
      summary$prefill_single_penalty_target_count +
        summary$prefill_multi_penalty_target_count &&
    summary$prefill_unique_target_key_count ==
      summary$prefill_consumed_unique_target_key_count +
        summary$prefill_unconsumed_unique_target_key_count &&
    summary$physical_target_optimization_count ==
      summary$prefill_target_optimization_count +
        summary$frontier_physical_target_optimization_count &&
    summary$frontier_physical_target_optimization_count ==
      summary$frontier_live_target_optimization_count +
        summary$singleton_padding_target_count &&
    summary$singleton_padding_batch_count ==
      summary$singleton_padding_target_count &&
    is.data.frame(summary$prefill_batches) &&
    nrow(summary$prefill_batches) == summary$prefill_window_count &&
    summary$excess_target_optimization_count ==
      summary$physical_target_optimization_count -
        summary$unique_target_key_count &&
    summary$reused_target_state_count == 0L &&
    summary$unique_residual_key_count <= summary$physical_residual_fits &&
    summary$excess_residual_fit_count ==
      summary$physical_residual_fits - summary$unique_residual_key_count &&
    summary$excess_native_setup_build_count ==
      summary$native_setup_count - summary$physical_prepared_s_key_count &&
    summary$cuda_single_penalty_optimizer_call_count > 0L &&
    summary$cuda_multi_penalty_optimizer_call_count == 0L &&
    summary$cuda_optimizer_host_boundary_count ==
      summary$cuda_single_penalty_optimizer_call_count +
        summary$cuda_multi_penalty_optimizer_call_count &&
    summary$cuda_optimizer_kernel_launch_count > 0L &&
    summary$cuda_single_penalty_optimizer_host_ms > 0 &&
    summary$cuda_multi_penalty_optimizer_host_ms == 0 &&
    summary$cuda_residual_solve_host_ms > 0 &&
    summary$cuda_dcov_component_build_ms > 0 &&
    summary$cuda_dcov_pair_gamma_ms > 0 &&
    summary$physical_tests_evaluated > 0L &&
    summary$physical_residual_fits > 0L,
  "Phase 10 cold call did not prove an empty DatasetKey cache state"
)

populated <- full_cuda_ci_one_call_cache_state_native(data_a)
other <- full_cuda_ci_one_call_cache_state_native(data_b)
assert_true(
  populated$result_cache_dataset_entries ==
      summary$result_cache_insert_count &&
    populated$target_cache_dataset_entries ==
      summary$target_cache_insert_count &&
    other$cache_epoch == populated$cache_epoch &&
    other$result_cache_entries == populated$result_cache_entries &&
    other$target_cache_entries == populated$target_cache_entries &&
    other$result_cache_dataset_entries == 0L &&
    other$target_cache_dataset_entries == 0L &&
    !identical(other$dataset_key, populated$dataset_key),
  "Phase 10 cache state did not distinguish DatasetKey ownership"
)

replay <- run_candidate(data_a)
assert_true(
  replay$summary$result_cache_dataset_warm_start_entries ==
      populated$result_cache_dataset_entries &&
    replay$summary$result_cache_preexisting_hit_count ==
      replay$summary$logical_tests_consumed &&
    replay$summary$physical_tests_evaluated == 0L &&
    replay$summary$physical_residual_fits == 0L,
  "Phase 10 replay did not identify preexisting DatasetKey cache hits"
)

invisible(full_cuda_ci_one_call_cache_control_native("configure", 262144L))
invisible(full_cuda_ci_one_call_cache_control_native(
  "configure_target", 131072L
))
invisible(full_cuda_ci_one_call_cache_control_native("reset"))

cat("PASS Phase 10 dataset cache epoch and DatasetKey entry accounting\n")
