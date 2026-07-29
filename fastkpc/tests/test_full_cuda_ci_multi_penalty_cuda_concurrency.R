source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 6 CUDA concurrency: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 6 CUDA concurrency: CUDA unavailable\n")
  quit(save = "no", status = 0)
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
assert_true(
  !is.null(setup) && length(state_indices) == 2L && ncol(setup$X) == 64L &&
    length(setup$penalty_blocks) == 7L,
  "Phase 6 CUDA concurrency fixture is malformed"
)
states <- shard$target_states[state_indices, , drop = FALSE]
contexts <- lapply(seq_len(nrow(states)), function(index) {
  target <- fastkpc_full_cuda_materialize_target_state(
    states[index, , drop = FALSE], data, setup$dataset_sha256
  )
  fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
})
Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
target_keys <- as.character(states$residual_key_sha256)
prepared <- fastkpc_full_cuda_phase6_prepare(setup)
setup_count <- 4L
handles <- lapply(seq_len(setup_count), function(index) {
  fastkpc_full_cuda_phase6_prepared_create(
    prepared, target_capacity = ncol(Y)
  )
})
on.exit(lapply(handles, function(handle) {
  try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle), silent = TRUE)
}), add = TRUE)

executed <- fastkpc_full_cuda_phase6_optimize_prepared_multi(
  handles = handles,
  target_batches = rep(list(Y), setup_count),
  target_keys = rep(list(target_keys), setup_count),
  concurrency = setup_count
)
tokens <- lapply(executed$setups, `[[`, "residual")
on.exit(lapply(tokens, function(token) {
  try(full_cuda_ci_multi_penalty_gcv_residual_free_native(token), silent = TRUE)
}), add = TRUE)
optimizations <- lapply(executed$setups, `[[`, "optimization")
diagnostics <- executed$diagnostics
reference <- optimizations[[1L]]

assert_true(
  identical(
    executed$schema_version,
    "full-cuda-ci-multi-penalty-gcv-cuda-multi-setup-v1"
  ) && identical(
    diagnostics$execution_strategy,
    "bounded-independent-prepared-streams-v1"
  ) && diagnostics$setup_count == setup_count &&
    diagnostics$target_count == setup_count * ncol(Y) &&
    diagnostics$requested_concurrency == setup_count &&
    diagnostics$worker_count == setup_count &&
    diagnostics$max_host_calls_in_flight == setup_count &&
    diagnostics$setup_stream_count == setup_count &&
    diagnostics$wall_host_ms > 0 && diagnostics$summed_setup_host_ms > 0 &&
    diagnostics$host_overlap_factor > 1,
  "Phase 6 bounded multi-setup scheduling diagnostics drifted"
)
assert_true(
  all(vapply(optimizations, function(value) {
    all(value$optimizer_status == 0L) &&
      value$diagnostics$device_allocation_count == 0L &&
      value$diagnostics$cuda_optimizer_kernel_launch_count == 1L &&
      identical(value$selected_log_sp, reference$selected_log_sp) &&
      identical(value$optimizer_iterations, reference$optimizer_iterations) &&
      identical(value$score_calls, reference$score_calls)
  }, logical(1L))) && all(vapply(tokens, function(token) {
    info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
    isTRUE(info$device_resident) && !isTRUE(info$released)
  }, logical(1L))),
  "Phase 6 concurrent setup results or residual tokens drifted"
)

lapply(tokens, full_cuda_ci_multi_penalty_gcv_residual_free_native)
lapply(handles, full_cuda_ci_multi_penalty_gcv_prepared_free_native)
cat("PASS Phase 6 bounded multi-setup CUDA concurrency\n")
