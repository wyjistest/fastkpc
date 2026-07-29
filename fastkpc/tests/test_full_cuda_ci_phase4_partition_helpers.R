source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase4_artifacts.R")

assert_true <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

keys <- sprintf("%064x", seq_len(37L))
parts <- lapply(0:7, function(partition_id) {
  fastkpc_full_cuda_phase4_shadow_partition(keys, partition_id, 8L)
})
observed <- unlist(lapply(parts, `[[`, "setup_keys"), use.names = FALSE)
assert_true(
  !anyDuplicated(observed) && identical(sort(observed), sort(keys)),
  "Phase 4 partitions must cover every setup exactly once"
)
assert_true(
  identical(
    fastkpc_full_cuda_phase4_shadow_partition(keys)$setup_keys, keys
  ),
  "Phase 4 unpartitioned selection must preserve canonical order"
)
invalid_partition_rejected <- tryCatch({
  fastkpc_full_cuda_phase4_shadow_partition(keys, 8L, 8L)
  FALSE
}, error = function(error) TRUE)
assert_true(
  invalid_partition_rejected,
  "Phase 4 partition ids outside the declared range must fail closed"
)

targets <- data.frame(
  fitted_same_sp_max_absolute = 0,
  fitted_same_sp_relative_l2 = 0,
  residual_same_sp_max_absolute = 0,
  residual_same_sp_relative_l2 = 0,
  fitted_oracle_max_absolute = 0,
  fitted_oracle_relative_l2 = 0,
  residual_oracle_max_absolute = 0,
  residual_oracle_relative_l2 = 0,
  planned_route = "AUGMENTED_SVD",
  executed_route = "AUGMENTED_SVD",
  solver_status = "OK_AUGMENTED_SVD",
  stringsAsFactors = FALSE
)
logical_rows <- data.frame(
  absolute_p_value_difference = 0,
  decision_flip = FALSE,
  backend_error = FALSE,
  spectra_fallback = FALSE
)
timings <- data.frame(
  target_count = 1L,
  fallback_count = 0L,
  legacy_mgcv_target_calls = 0L,
  spectral_optimizer_target_count = 1L,
  spectral_only_target_count = 1L,
  exact_replay_target_count = 0L,
  exact_replay_endpoint_risk_count = 0L,
  exact_replay_convergence_risk_count = 0L,
  exact_replay_boundary_risk_count = 0L,
  exact_replay_numerical_risk_count = 0L,
  optimizer_target_coverage_complete = TRUE,
  sp_selection_backend_executed = "cuda",
  gcv_score_backend_executed = "cuda",
  optimizer_backend_executed =
    "cuda-spectral-risk-gated-exact-replay",
  exact_replay_backend_executed =
    "cuda-dpstf2-lapack-3.12-dgesdd",
  selected_sp_returned_to_r_before_solve = FALSE,
  implicit_residual_d2h_count = 0L,
  integrated_host_ms = 1,
  cuda_gcv_score_ms = 1,
  cuda_selected_sp_solve_ms = 1,
  shadow_ms = 1,
  validation_ms = 1,
  dcov_ms = 1
)
summary <- fastkpc_full_cuda_phase4_shadow_summary(
  targets, logical_rows, timings, 1L, TRUE, 1
)
assert_true(
  isTRUE(summary$same_sp_fixed_solver_gate) &&
    isTRUE(summary$oracle_residual_gate) &&
    isTRUE(summary$downstream_decision_gate) &&
    isTRUE(summary$optimizer_coverage_gate) &&
    isTRUE(summary$backend_gate),
  "Phase 4 shadow summary must accept a clean partition"
)
logical_rows$decision_flip <- TRUE
failed <- fastkpc_full_cuda_phase4_shadow_summary(
  targets, logical_rows, timings, 1L, TRUE, 1
)
assert_true(
  !isTRUE(failed$downstream_decision_gate),
  "Phase 4 shadow summary must reject a downstream decision flip"
)

cat("PASS Phase 4 partition helpers\n")
