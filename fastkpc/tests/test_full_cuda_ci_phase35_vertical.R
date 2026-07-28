source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/dcov_exact.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
}

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP full CUDA CI Phase 3.5 vertical structural prototype\n")
  quit(save = "no", status = 0L)
}

build_fastkpc_cuda_native(rebuild = TRUE)
assert_true(fastkpc_cuda_available(), "CUDA must be available")

vertical_source <- paste(
  readLines("fastkpc/src/cuda/full_cuda_ci_vertical.cu", warn = FALSE),
  collapse = "\n"
)
semantic_header <- paste(
  readLines("fastkpc/src/full_cuda_ci_semantic_abi.hpp", warn = FALSE),
  collapse = "\n"
)
assert_true(
  !grepl("Rf_pgamma", vertical_source, fixed = TRUE) &&
    !grepl("R::pgamma", vertical_source, fixed = TRUE) &&
    !grepl("cudaDeviceSynchronize", vertical_source, fixed = TRUE) &&
    !grepl("materialize_fixed_sp_shadow", vertical_source, fixed = TRUE) &&
    grepl("regularized_gamma_q", vertical_source, fixed = TRUE),
  "vertical prototype must keep pair statistics and gamma evaluation on CUDA"
)
assert_true(
  !grepl("DeviceResidualConsumerView", semantic_header, fixed = TRUE) &&
    !grepl("double*", semantic_header, fixed = TRUE) &&
    !grepl("cudaStream_t", semantic_header, fixed = TRUE) &&
    !grepl("cudaEvent_t", semantic_header, fixed = TRUE),
  "ephemeral residual pointers and CUDA layout must remain outside the semantic ABI"
)

shard_path <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_0.rds"
)
summary_path <- sub("[.]rds$", ".summary.json", shard_path)
shard <- readRDS(shard_path)
summary <- fastkpc_full_cuda_fixed_sp_read_json(summary_path)
authentication <- fastkpc_full_cuda_prepared_s_shard_authentication(shard)
assert_true(
  identical(authentication$payload_hash, summary$payload_hash) &&
    identical(authentication$manifest_hash, summary$manifest_hash) &&
    identical(authentication$setup_key_set_hash,
              summary$setup_key_set_hash) &&
    identical(authentication$target_key_set_hash,
              summary$target_key_set_hash) &&
    identical(fastkpc_full_cuda_census_file_hash(shard_path),
              summary$rds_file_sha256),
  "canonical Prepared-S shard must authenticate before the vertical call"
)

prepared_key <-
  "000bf94226b34186828cfa30c400753eb19ca2ff99409573df21ac06da2a72be"
logical_sequence_id <- 139040
logical_tests <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1",
  "logical_ci_tests.rds"
))
logical_row <- logical_tests[
  logical_tests$logical_sequence_id == logical_sequence_id,
  , drop = FALSE
]
assert_true(
  nrow(logical_row) == 1L && logical_row$x == 17L &&
    logical_row$y == 30L && identical(logical_row$S_key, "11|35") &&
    identical(logical_row$alpha, 0.1) &&
    identical(logical_row$reference_decision, "dependent"),
  "logical request 139040 must retain its canonical trace identity"
)

setup <- shard$prepared_s_setups[[prepared_key]]
states <- shard$target_states[
  shard$target_states$prepared_s_key_sha256 == prepared_key &
    shard$target_states$target %in% c(logical_row$x, logical_row$y),
  , drop = FALSE
]
states <- states[match(c(logical_row$x, logical_row$y), states$target),
                 , drop = FALSE]
assert_true(
  nrow(states) == 2L &&
    identical(states$residual_key_sha256,
              c(logical_row$residual_key_x, logical_row$residual_key_y)) &&
    identical(setup$sorted_S, c(11L, 35L)),
  "vertical targets must resolve to the authenticated common Prepared-S setup"
)

phase3_setups <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "fixed_sp_cuda_oracle_sp_v1",
  "setup_results.rds"
))
phase3_setup <- phase3_setups[
  phase3_setups$prepared_s_key_sha256 == prepared_key,
  , drop = FALSE
]
assert_true(
  nrow(phase3_setup) == 1L && phase3_setup$target_count == 41L &&
    phase3_setup$planned_cholesky_target_count == 41L &&
    phase3_setup$executed_cholesky_target_count == 41L &&
    phase3_setup$stable_reroute_count == 0L,
  "canonical vertical targets must inherit authenticated Phase 3 routes"
)

dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
Y <- data[, states$target, drop = FALSE]
SP <- do.call(cbind, lapply(states$selected_sp, as.numeric))
planned_route <- rep("CHOLESKY_BATCHED", 2L)
request <- fastkpc_full_cuda_phase35_vertical_request(
  expected_prepared_s_key_sha256 = prepared_key,
  target_keys = states$residual_key_sha256,
  logical_sequence_id = as.double(logical_sequence_id),
  left_target_ordinal = 1L,
  right_target_ordinal = 2L,
  alpha = 0.1,
  exercise_eviction = TRUE
)

runtime <- fixed_sp_cuda_runtime_create(0L)
handle <- NULL
on.exit({
  if (!is.null(handle)) {
    try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
  }
  if (!is.null(runtime)) {
    try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }
}, add = TRUE)
fixed_sp_cuda_runtime_reserve(
  runtime, dto$n, dto$null_dim, 2L, dto$penalty_count,
  dto$n + dto$null_dim
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)

resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
caller_device_before <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
result <- fastkpc_full_cuda_phase35_vertical_ci(
  handle, Y, SP, planned_route, states$residual_key_sha256, request
)
caller_device_after <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
compact_fields <- unlist(
  contracts$architecture_contract_v1$payload$compact_result_fields,
  use.names = FALSE
)
assert_true(
  identical(names(result), c(
    "schema_version", "request_identity_sha256",
    "prepared_s_key_sha256", "target_keys", "first_result",
    "replay_result", "first_numerical", "replay_numerical",
    "diagnostics"
  )) &&
    identical(result$schema_version,
              "full-cuda-ci-phase35-vertical-result-v1") &&
    identical(result$request_identity_sha256,
              request$request_identity_sha256) &&
    identical(result$prepared_s_key_sha256, prepared_key) &&
    identical(result$target_keys, states$residual_key_sha256) &&
    identical(names(result$first_result), compact_fields) &&
    identical(names(result$replay_result), compact_fields) &&
    identical(result$first_result, result$replay_result),
  "vertical result must expose the exact compact ABI and deterministic replay"
)

oracle_residuals <- lapply(seq_len(2L), function(index) {
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    prepared_setup = setup,
    target_state = list(
      row = states[index, , drop = FALSE],
      y = as.numeric(Y[, index])
    )
  )$residuals
})
exact_oracle <- dcov_gamma_exact(oracle_residuals[[1L]],
                                oracle_residuals[[2L]])
expected_numerical <- c(
  statistic = unname(exact_oracle$statistic),
  mean = unname(exact_oracle$estimates[[2L]]),
  variance = unname(exact_oracle$estimates[[3L]])
)
actual_numerical <- unlist(
  result$first_numerical[c("statistic", "mean", "variance")],
  use.names = TRUE
)
assert_true(
  abs(actual_numerical[["statistic"]] -
        expected_numerical[["statistic"]]) <= 1e-9 &&
    abs(actual_numerical[["mean"]] - expected_numerical[["mean"]]) <=
      1e-10 &&
    abs(actual_numerical[["variance"]] -
        expected_numerical[["variance"]]) <= 1e-10 &&
    abs(result$first_result$p_value - exact_oracle$p.value) <= 1e-10 &&
    identical(result$first_result$status, "OK") &&
    identical(result$first_result$dcov_status,
              "OK_EXACT_CUDA_GAMMA") &&
    identical(result$first_result$logical_sequence_id,
              as.double(logical_sequence_id)) &&
    result$first_result$p_value < 0.1,
  "CUDA exact statistic, moments, gamma tail, and decision must pass parity"
)

diagnostics <- result$diagnostics
component_bytes <- 8 * (dto$n^2 + dto$n + 2)
assert_true(
  identical(caller_device_after, caller_device_before) &&
  diagnostics$component_semantic_version ==
    "full-cuda-ci-exact-centered-distance-component-v1" &&
    diagnostics$n == dto$n && diagnostics$target_count == 2L &&
    diagnostics$component_build_count == 4L &&
    diagnostics$component_cache_eviction_count == 2L &&
    diagnostics$pair_evaluation_count == 2L &&
    diagnostics$deterministic_replay_count == 1L &&
    diagnostics$residual_d2h_count == 0L &&
    diagnostics$residual_d2h_bytes == 0 &&
    diagnostics$component_d2h_count == 0L &&
    diagnostics$component_d2h_bytes == 0 &&
    diagnostics$compact_result_d2h_count == 2L &&
    diagnostics$compact_result_d2h_bytes > 0 &&
    diagnostics$compact_result_d2h_bytes < 1024 &&
    diagnostics$cpu_dcov_component_count == 0L &&
    diagnostics$cpu_dcov_pair_statistic_count == 0L &&
    diagnostics$cpu_gamma_p_value_count == 0L &&
    diagnostics$consumer_event_registration_count == 1L &&
    diagnostics$component_bytes_per_target == component_bytes &&
    diagnostics$peak_component_bytes == 2 * component_bytes &&
    diagnostics$peak_live_device_bytes ==
      diagnostics$peak_component_bytes +
        diagnostics$compact_result_d2h_bytes / 2 &&
    diagnostics$device_allocation_count ==
      diagnostics$device_free_count &&
    diagnostics$device_allocation_count == 17L &&
    all(is.finite(unlist(diagnostics[c(
      "residual_solve_host_ms", "first_component_build_cuda_ms",
      "first_pair_evaluation_cuda_ms", "first_compact_d2h_cuda_ms",
      "replay_component_build_cuda_ms", "replay_pair_evaluation_cuda_ms",
      "replay_compact_d2h_cuda_ms", "teardown_host_ms", "total_host_ms"
    )], use.names = FALSE))) &&
    all(unlist(diagnostics[c(
      "residual_solve_host_ms", "first_component_build_cuda_ms",
      "first_pair_evaluation_cuda_ms", "first_compact_d2h_cuda_ms",
      "replay_component_build_cuda_ms", "replay_pair_evaluation_cuda_ms",
      "replay_compact_d2h_cuda_ms", "teardown_host_ms", "total_host_ms"
    )], use.names = FALSE) >= 0) &&
    diagnostics$total_host_ms >= diagnostics$residual_solve_host_ms &&
    all(unlist(diagnostics[c(
      "request_identity_authenticated", "prepared_identity_authenticated",
      "target_identity_authenticated", "residuals_device_resident",
      "components_device_resident", "compact_result_only_d2h",
      "eviction_result_bit_identical", "deterministic_logical_replay",
      "bounded_allocation", "leak_free_teardown",
      "caller_device_restored"
    )], use.names = FALSE)),
  "vertical structural residency, authority, eviction, and memory gates must pass"
)
live_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
assert_true(
  identical(resources_after[live_fields], resources_before[live_fields]) &&
    resources_after$total_device_allocations -
      resources_before$total_device_allocations == 17 &&
    resources_after$total_device_frees -
      resources_before$total_device_frees == 17 &&
    resources_after$total_stream_creates -
      resources_before$total_stream_creates == 1 &&
    resources_after$total_stream_destroys -
      resources_before$total_stream_destroys == 1 &&
    resources_after$total_event_creates -
      resources_before$total_event_creates == 5 &&
    resources_after$total_event_destroys -
      resources_before$total_event_destroys == 5,
  "vertical module resource ledger must return to its exact live baseline"
)
prepared_after <- fixed_sp_cuda_prepared_info(handle)
assert_true(
  identical(prepared_after$output_slot_leased, FALSE) &&
    identical(prepared_after$output_slot_state, "free"),
  "registered residual consumer completion must release the Phase 3 slot"
)

forged_request <- request
forged_request$request_identity_sha256 <- strrep("0", 64L)
forged_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
assert_error(
  fastkpc_full_cuda_phase35_vertical_ci(
    handle, Y, SP, planned_route, states$residual_key_sha256,
    forged_request
  ),
  "vertical request identity SHA-256 mismatch",
  "forged request identity must fail before CUDA work"
)
forged_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
assert_true(
  identical(forged_after, forged_before) &&
    identical(fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"),
  "pre-submission authentication failure must allocate nothing and retain the slot"
)

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL

cat(
  "PASS full CUDA CI Phase 3.5 vertical structural prototype; p=",
  format(result$first_result$p_value, digits = 17L),
  "; compact_d2h_bytes=", diagnostics$compact_result_d2h_bytes,
  "; peak_device_bytes=", diagnostics$peak_live_device_bytes, "\n",
  sep = ""
)
