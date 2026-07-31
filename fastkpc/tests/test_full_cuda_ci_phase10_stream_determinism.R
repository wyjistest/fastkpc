source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 10 stream determinism: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0L)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 10 stream determinism: CUDA unavailable\n")
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
assert_true(
  !is.null(setup) && length(state_indices) == 2L && ncol(setup$X) == 64L &&
    length(setup$penalty_blocks) == 7L,
  "Phase 10 stream determinism fixture is malformed"
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

execute <- function(stream_count) {
  handles <- lapply(seq_len(stream_count), function(index) {
    fastkpc_full_cuda_phase6_prepared_create(
      prepared, target_capacity = ncol(Y)
    )
  })
  tokens <- list()
  on.exit({
    lapply(tokens, function(token) {
      try(full_cuda_ci_multi_penalty_gcv_residual_free_native(token),
          silent = TRUE)
    })
    lapply(handles, function(handle) {
      try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle),
          silent = TRUE)
    })
  }, add = TRUE)
  executed <- fastkpc_full_cuda_phase6_optimize_prepared_multi(
    handles = handles,
    target_batches = rep(list(Y), stream_count),
    target_keys = rep(list(target_keys), stream_count),
    concurrency = stream_count
  )
  tokens <- lapply(executed$setups, `[[`, "residual")
  optimizations <- lapply(executed$setups, `[[`, "optimization")
  diagnostics <- executed$diagnostics
  reference <- optimizations[[1L]]
  assert_true(
    diagnostics$setup_count == stream_count &&
      diagnostics$requested_concurrency == stream_count &&
      diagnostics$worker_count == stream_count &&
      diagnostics$max_host_calls_in_flight == stream_count &&
      diagnostics$setup_stream_count == stream_count &&
      all(vapply(optimizations, function(value) {
        all(value$optimizer_status == 0L) &&
          identical(value$selected_log_sp, reference$selected_log_sp) &&
          identical(value$optimizer_iterations,
                    reference$optimizer_iterations) &&
          identical(value$score_calls, reference$score_calls)
      }, logical(1L))),
    paste0("stream count ", stream_count,
           " changed optimizer state or scheduler accounting")
  )
  lapply(tokens, full_cuda_ci_multi_penalty_gcv_residual_free_native)
  tokens <- list()
  lapply(handles, full_cuda_ci_multi_penalty_gcv_prepared_free_native)
  handles <- list()
  list(
    stream_count = stream_count,
    selected_log_sp = reference$selected_log_sp,
    optimizer_iterations = reference$optimizer_iterations,
    score_calls = reference$score_calls,
    optimizer_status = reference$optimizer_status
  )
}

results <- lapply(c(1L, 2L, 4L), execute)
reference <- results[[1L]]
assert_true(
  all(vapply(results, function(value) {
    identical(value$selected_log_sp, reference$selected_log_sp) &&
      identical(value$optimizer_iterations,
                reference$optimizer_iterations) &&
      identical(value$score_calls, reference$score_calls) &&
      identical(value$optimizer_status, reference$optimizer_status)
  }, logical(1L))),
  "legal stream counts 1/2/4 changed canonical optimizer results"
)

evidence_path <- Sys.getenv(
  "FASTKPC_PHASE10_STREAM_EVIDENCE_RDS", unset = ""
)
if (nzchar(evidence_path)) {
  saveRDS(list(
    schema_version = "full-cuda-ci-phase10-stream-evidence-v1",
    captured_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    stream_counts = vapply(results, `[[`, integer(1L), "stream_count"),
    selected_log_sp = lapply(results, `[[`, "selected_log_sp"),
    optimizer_iterations = lapply(results, `[[`, "optimizer_iterations"),
    score_calls = lapply(results, `[[`, "score_calls"),
    optimizer_status = lapply(results, `[[`, "optimizer_status"),
    pass = TRUE
  ), evidence_path, compress = "xz")
}

cat("PASS Phase 10 stream-count determinism; counts=1,2,4\n")
