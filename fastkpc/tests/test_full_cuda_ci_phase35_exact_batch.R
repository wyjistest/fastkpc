source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
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
  cat("SKIP full CUDA CI Phase 3.5E exact batch\n")
  quit(save = "no", status = 0L)
}

load_fastkpc_cuda_native()
assert_true(fastkpc_cuda_available(), "CUDA must be available")

prepared_key <-
  "000bf94226b34186828cfa30c400753eb19ca2ff99409573df21ac06da2a72be"
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_0.rds"
))
setup <- shard$prepared_s_setups[[prepared_key]]
logical_tests <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1",
  "logical_ci_tests.rds"
))
logical_rows <- logical_tests[logical_tests$S_key == "11|35", , drop = FALSE]
logical_rows <- logical_rows[
  order(logical_rows$logical_sequence_id, method = "radix"), , drop = FALSE
][seq_len(12L), , drop = FALSE]
target_keys <- sort(unique(c(
  logical_rows$residual_key_x, logical_rows$residual_key_y
)), method = "radix")
states <- shard$target_states[
  match(target_keys, shard$target_states$residual_key_sha256), , drop = FALSE
]
assert_true(
  nrow(states) == length(target_keys) && !anyNA(states$target) &&
    identical(states$residual_key_sha256, target_keys) &&
    all(states$prepared_s_key_sha256 == prepared_key) &&
    identical(setup$sorted_S, c(11L, 35L)),
  "canonical exact-batch endpoints must resolve to one authenticated setup"
)

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
Y <- data[, states$target, drop = FALSE]
SP <- do.call(cbind, lapply(states$selected_sp, as.numeric))
planned_route <- rep("CHOLESKY_BATCHED", nrow(states))
request <- fastkpc_full_cuda_phase35_exact_batch_request_from_logical(
  expected_prepared_s_key_sha256 = prepared_key,
  target_keys = target_keys,
  logical_rows = logical_rows
)
assert_true(
  request$component_capacity == length(target_keys) &&
    identical(request$logical_sequence_ids,
              as.double(logical_rows$logical_sequence_id)),
  "exact-batch request must bind every canonical endpoint and pair"
)

dto <- fastkpc_full_cuda_fixed_sp_native_dto(setup)
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
  runtime, dto$n, dto$null_dim, nrow(states), dto$penalty_count,
  dto$n + dto$null_dim
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)

resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
device_before <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
result <- fastkpc_full_cuda_phase35_exact_batch_ci(
  handle, Y, SP, planned_route, target_keys, request
)
device_after <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
fastkpc_full_cuda_phase35_validate_exact_batch_result(
  result, request, target_keys
)

residuals <- setNames(lapply(seq_len(nrow(states)), function(index) {
  fastkpc_mgcv_magic_fixed_sp_from_prepared(
    prepared_setup = setup,
    target_state = list(
      row = states[index, , drop = FALSE],
      y = as.numeric(Y[, index])
    )
  )$residuals
}), target_keys)
oracle_p <- vapply(seq_len(nrow(logical_rows)), function(index) {
  dcov_gamma_exact(
    residuals[[logical_rows$residual_key_x[[index]]]],
    residuals[[logical_rows$residual_key_y[[index]]]]
  )$p.value
}, numeric(1L))
absolute_error <- abs(result$records$p_value - oracle_p)
assert_true(
  max(absolute_error) <= 1e-10 &&
    identical(result$records$p_value >= logical_rows$alpha,
              logical_rows$reference_independent) &&
    identical(result$records$p_value >= logical_rows$alpha,
              oracle_p >= logical_rows$alpha),
  "exact CUDA batch p-values and decisions must match both authorities"
)

diagnostics <- result$diagnostics
component_bytes <- 8 * (dto$n^2 + dto$n + 2)
metadata_bytes <- 4 * length(target_keys) + 16 * nrow(logical_rows)
assert_true(
  identical(device_after, device_before) &&
    diagnostics$referenced_component_count == length(target_keys) &&
    diagnostics$component_cache_lookup_count == 2L * nrow(logical_rows) &&
    diagnostics$component_cache_miss_count == length(target_keys) &&
    diagnostics$component_cache_hit_count ==
      2L * nrow(logical_rows) - length(target_keys) &&
    diagnostics$component_bytes_per_target == component_bytes &&
    diagnostics$peak_component_bytes == length(target_keys) * component_bytes &&
    diagnostics$metadata_h2d_count == 4L &&
    diagnostics$metadata_h2d_bytes == metadata_bytes &&
    diagnostics$peak_live_device_bytes ==
      diagnostics$peak_component_bytes + metadata_bytes +
        72 * nrow(logical_rows) &&
    diagnostics$consumer_event_registration_count == 1L &&
    diagnostics$explicit_host_wait_count == 2L &&
    all(is.finite(unlist(diagnostics[c(
      "residual_solve_host_ms", "metadata_h2d_cuda_ms",
      "component_build_cuda_ms", "pair_evaluation_cuda_ms",
      "compact_d2h_cuda_ms", "dcov_host_boundary_ms",
      "teardown_host_ms", "total_host_ms"
    )], use.names = FALSE))) &&
    all(unlist(diagnostics[c(
      "residual_solve_host_ms", "metadata_h2d_cuda_ms",
      "component_build_cuda_ms", "pair_evaluation_cuda_ms",
      "compact_d2h_cuda_ms", "dcov_host_boundary_ms",
      "teardown_host_ms", "total_host_ms"
    )], use.names = FALSE) >= 0),
  "exact batch cache, residency, allocation, and timing gates must pass"
)
live_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
assert_true(
  identical(resources_after[live_fields], resources_before[live_fields]) &&
    resources_after$total_device_allocations -
      resources_before$total_device_allocations == 9 &&
    resources_after$total_device_frees -
      resources_before$total_device_frees == 9 &&
    resources_after$total_stream_creates -
      resources_before$total_stream_creates == 1 &&
    resources_after$total_stream_destroys -
      resources_before$total_stream_destroys == 1 &&
    resources_after$total_event_creates -
      resources_before$total_event_creates == 5 &&
    resources_after$total_event_destroys -
      resources_before$total_event_destroys == 5,
  "exact batch resource ledger must return to its exact live baseline"
)
assert_true(
  identical(fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"),
  "exact batch must release the Phase 3 residual output slot"
)

bad_capacity <- request
bad_capacity$component_capacity <- 1L
assert_error(
  fastkpc_full_cuda_phase35_exact_batch_ci(
    handle, Y, SP, planned_route, target_keys, bad_capacity
  ),
  "exact batch component capacity is invalid",
  "insufficient component capacity must fail closed"
)

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL
cat(
  "PASS full CUDA CI Phase 3.5E exact batch; pairs=", nrow(logical_rows),
  "; components=", length(target_keys),
  "; max_p_error=", format(max(absolute_error), digits = 6L),
  "; component_ms=",
  format(diagnostics$component_build_cuda_ms, digits = 6L),
  "; pair_ms=", format(diagnostics$pair_evaluation_cuda_ms, digits = 6L),
  "\n", sep = ""
)
