source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")

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
  cat("SKIP full CUDA CI Phase 3.5E legacy eig batch\n")
  quit(save = "no", status = 0L)
}

load_fastkpc_cuda_native()
assert_true(fastkpc_cuda_available(), "CUDA must be available")

prepared_key <-
  "014ef34cf19da2d77273b48587138b42d23a1b9a392d27711f0d7a06404ab64b"
target_keys <- sort(c(
  "cdf8b20711a23fc242d32e952f85db2483d26e091a7bbffddb9572a8ff1dfcb0",
  "b65ded9844ca0ea2f1dd819ee66baadc2bb275ed22872e6dfdd1d011cbd3b3ce"
), method = "radix")
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", "shard_34.rds"
))
setup <- shard$prepared_s_setups[[prepared_key]]
states <- shard$target_states[
  match(target_keys, shard$target_states$residual_key_sha256), , drop = FALSE
]
assert_true(
  !is.null(setup) && nrow(states) == 2L && !anyNA(states$target) &&
    identical(states$residual_key_sha256, target_keys) &&
    all(states$prepared_s_key_sha256 == prepared_key) &&
    identical(setup$sorted_S, c(13L, 32L, 35L, 39L, 45L)),
  "canonical legacy-eig fixture must resolve to authenticated shard 34"
)

logical_rows <- data.frame(
  logical_sequence_id = 239277L,
  residual_key_x =
    "cdf8b20711a23fc242d32e952f85db2483d26e091a7bbffddb9572a8ff1dfcb0",
  residual_key_y =
    "b65ded9844ca0ea2f1dd819ee66baadc2bb275ed22872e6dfdd1d011cbd3b3ce",
  alpha = 0.1,
  stringsAsFactors = FALSE
)
request <- fastkpc_full_cuda_phase35_legacy_eig_batch_request_from_logical(
  expected_prepared_s_key_sha256 = prepared_key,
  target_keys = target_keys,
  logical_rows = logical_rows
)
assert_true(
  request$component_capacity == 2L && request$num_col == 35L &&
    identical(request$left_target_ordinals, 2L) &&
    identical(request$right_target_ordinals, 1L),
  "legacy-eig request must bind the canonical fixture endpoints"
)
assert_error(
  fastkpc_full_cuda_phase35_legacy_eig_batch_request_from_logical(
    prepared_key, target_keys, logical_rows, num_col = 34L
  ),
  "canonical num_col 35L",
  "noncanonical legacy eig rank must fail closed"
)

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
Y <- data[, states$target, drop = FALSE]
SP <- do.call(cbind, lapply(states$selected_sp, as.numeric))
planned_route <- c("AUGMENTED_SVD", "CHOLESKY_BATCHED")
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
  runtime, dto$n, dto$null_dim, 2L, dto$penalty_count,
  dto$n + dto$null_dim
)
handle <- fixed_sp_cuda_prepared_create(runtime, dto)

resources_before <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
device_before <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
result <- fastkpc_full_cuda_phase35_legacy_eig_batch_ci(
  handle, Y, SP, planned_route, target_keys, request
)
device_after <- .Call(
  "C_fixed_sp_cuda_test_get_device", PACKAGE = "fastkpc_cuda"
)
resources_after <- fastkpc_full_cuda_phase35_vertical_resource_snapshot()
fastkpc_full_cuda_phase35_validate_legacy_eig_batch_result(
  result, request, target_keys
)

legacy <- c(
  p_value = 0.10432070647093661,
  statistic = 1.724800097132142,
  mean = 1.1260888036552767,
  variance = 0.21062519899877513
)
error <- c(
  p_value = abs(result$records$p_value[[1L]] - legacy[["p_value"]]),
  statistic = abs(result$numerical$statistic[[1L]] -
                    legacy[["statistic"]]),
  mean = abs(result$numerical$mean[[1L]] - legacy[["mean"]]),
  variance = abs(result$numerical$variance[[1L]] - legacy[["variance"]])
)
assert_true(
  error[["p_value"]] <= 1e-10 &&
    error[["statistic"]] <= 1e-9 &&
    error[["mean"]] <= 1e-10 && error[["variance"]] <= 1e-10 &&
    result$records$p_value[[1L]] >= 0.1,
  "legacy full-eig CUDA fixture must match numerical and decision authority"
)

diagnostics <- result$diagnostics
live_fields <- c(
  "live_device_allocations", "live_device_bytes", "live_streams",
  "live_events"
)
assert_true(
  identical(device_after, device_before) &&
    identical(resources_after[live_fields], resources_before[live_fields]) &&
    resources_after$total_device_allocations -
      resources_before$total_device_allocations == 19 &&
    resources_after$total_device_frees -
      resources_before$total_device_frees == 19 &&
    resources_after$total_stream_creates -
      resources_before$total_stream_creates == 1 &&
    resources_after$total_stream_destroys -
      resources_before$total_stream_destroys == 1 &&
    resources_after$total_event_creates -
      resources_before$total_event_creates == 10 &&
    resources_after$total_event_destroys -
      resources_before$total_event_destroys == 10 &&
    diagnostics$cuda_full_eig_count == 2L &&
    diagnostics$cuda_pair_count == 1L &&
    diagnostics$cuda_gamma_count == 1L,
  "legacy-eig resource and CUDA authority ledgers must close"
)
assert_true(
  identical(fixed_sp_cuda_prepared_info(handle)$output_slot_state, "free"),
  "legacy-eig batch must release the Phase 3 residual output slot"
)

fixed_sp_cuda_prepared_free(handle)
handle <- NULL
fixed_sp_cuda_runtime_free(runtime)
runtime <- NULL
cat(
  "PASS full CUDA CI Phase 3.5E legacy eig batch; logical_id=239277",
  "; p_error=", format(error[["p_value"]], digits = 6L),
  "; statistic_error=", format(error[["statistic"]], digits = 6L),
  "; component_ms=",
  format(diagnostics$component_build_cuda_ms, digits = 6L),
  "; pair_ms=", format(diagnostics$pair_evaluation_cuda_ms, digits = 6L),
  "\n", sep = ""
)
