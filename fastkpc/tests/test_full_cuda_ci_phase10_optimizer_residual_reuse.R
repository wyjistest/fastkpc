source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_phase10_optimizer_residual_reuse.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 10 optimizer residual qualification: CUDA tests disabled\n")
  quit(save = "no", status = 0L)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 10 optimizer residual qualification: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_1.rds"
))
setup_key <-
  "05b18f09b303d481e84ec14c6a3d11da92db2fcc0f9f213aa8319015f7369ae4"
setup <- shard$prepared_s_setups[[setup_key]]
state_indices <- head(which(
  shard$target_states$prepared_s_key_sha256 == setup_key
), 2L)
states <- shard$target_states[state_indices, , drop = FALSE]
assert_true(
  !is.null(setup) && nrow(states) == 2L && ncol(setup$X) == 64L &&
    length(setup$penalty_blocks) == 7L,
  "Phase 10 optimizer residual fixture is malformed"
)

resource_before <- fixed_sp_cuda_live_owner_snapshot()
qualification <- fastkpc_full_cuda_phase10_qualify_optimizer_residual_reuse(
  setup, states, data, device_id = 0L
)
resource_after <- fixed_sp_cuda_live_owner_snapshot()
fastkpc_full_cuda_phase10_validate_optimizer_residual_qualification(
  qualification
)

parity <- qualification$residual_parity
assert_true(
  identical(
    qualification$decision,
    "STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY"
  ) &&
    parity$n == 351L && parity$target_count == 2L &&
    parity$value_count == 702 &&
    parity$bitwise_equal_target_count == 0L &&
    parity$mismatch_target_count == 2L &&
    parity$bitwise_equal_value_count == 0 &&
    parity$mismatch_value_count == 702 &&
    parity$max_abs_difference > 0.2 &&
    parity$relative_l2_difference > 0.02 &&
    parity$optimizer_status_failure_count == 0L &&
    parity$fixed_status_failure_count == 0L &&
    parity$fixed_route_status_mismatch_count == 0L &&
    parity$producer_event_wait_count == 2L &&
    parity$consumer_event_registration_count == 1L &&
    parity$compact_d2h_count == 1L && parity$compact_d2h_bytes == 64 &&
    parity$residual_d2h_count == 0L && parity$residual_d2h_bytes == 0 &&
    isTRUE(parity$target_identity_authenticated) &&
    isTRUE(parity$device_identity_authenticated) &&
    isTRUE(parity$residual_payload_device_resident) &&
    isTRUE(parity$compact_diagnostics_only_d2h) &&
    isTRUE(parity$caller_device_restored),
  "optimizer and fixed residual device parity diagnosis drifted"
)
assert_true(
  qualification$gates$exact_p_value_mismatch_count == 1L &&
    qualification$gates$legacy_eig_p_value_mismatch_count == 1L &&
    qualification$gates$final_decision_flip_count == 0L &&
    !qualification$gates$numerical_gate &&
    qualification$gates$authority_gate &&
    !qualification$exact_p_value_parity$bitwise_equal[[1L]] &&
    !qualification$legacy_eig_p_value_parity$bitwise_equal[[1L]] &&
    qualification$exact_p_value_parity$absolute_difference[[1L]] > 0 &&
    qualification$legacy_eig_p_value_parity$absolute_difference[[1L]] > 0 &&
    isTRUE(qualification$consumer$exact_residual_solve_bypassed) &&
    isTRUE(qualification$consumer$legacy_residual_solve_bypassed) &&
    qualification$consumer$exact_residual_d2h_bytes == 0 &&
    qualification$consumer$legacy_residual_d2h_bytes == 0 &&
    isTRUE(qualification$lifetime$residual_slot_leased_before_release) &&
    qualification$lifetime$residual_shadow_d2h_count == 0L &&
    isTRUE(qualification$lifetime$released) &&
    isTRUE(qualification$lifetime$stale_token_rejected) &&
    !qualification$detached_arena$attempted &&
    identical(
      qualification$detached_arena$reason,
      "not-attempted-after-numerical-parity-stop"
    ) &&
    identical(resource_after, resource_before),
  "optimizer residual consumer, authority, or lifetime gate drifted"
)

tampered <- qualification
tampered$residual_parity$mismatch_value_count <- 0
assert_error(
  fastkpc_full_cuda_phase10_validate_optimizer_residual_qualification(
    tampered
  ),
  "optimizer residual qualification is malformed",
  "optimizer residual qualification must reject accounting drift"
)

cat(
  "PASS Phase 10 optimizer residual qualification; decision=",
  qualification$decision,
  "; residual_mismatches=", parity$mismatch_value_count,
  "; exact_p_mismatches=",
  qualification$gates$exact_p_value_mismatch_count,
  "; legacy_p_mismatches=",
  qualification$gates$legacy_eig_p_value_mismatch_count,
  "\n", sep = ""
)
