source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
output_path <- if (length(args) >= 1L) args[[1L]] else {
  "/tmp/fastkpc_formal_p13_selected_sp_objective_replay.rds"
}
shard_id <- if (length(args) >= 2L) as.integer(args[[2L]]) else 45L
prepared_key <- if (length(args) >= 3L) args[[3L]] else {
  "c8c47b11af8006e5c845c00af69822d72f56e0c718effe23b5b28c33ef5417df"
}
residual_key <- if (length(args) >= 4L) args[[4L]] else {
  "8d4a2fb574d88f3bc0784d3ae393d9572c388a4cf1685479c1886d7237a1f84b"
}

if (file.exists(output_path)) stop("output already exists", call. = FALSE)
if (is.na(shard_id) || shard_id < 0L || !nzchar(prepared_key) ||
    !nzchar(residual_key)) {
  stop("formal replay witness arguments are invalid", call. = FALSE)
}

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", paste0("shard_", shard_id, ".rds")
))
setup <- shard$prepared_s_setups[[prepared_key]]
witness_index <- match(residual_key, shard$target_states$residual_key_sha256)
if (is.null(setup) || is.na(witness_index)) {
  stop("formal replay witness is missing", call. = FALSE)
}
state_indices <- unique(c(
  witness_index,
  which(shard$target_states$prepared_s_key_sha256 == prepared_key)
))
state_indices <- head(state_indices, 2L)
if (length(state_indices) == 1L) state_indices <- rep(state_indices, 2L)
contexts <- lapply(state_indices, function(index) {
  state <- shard$target_states[index, , drop = FALSE]
  target <- fastkpc_full_cuda_materialize_target_state(
    state, data, setup$dataset_sha256
  )
  fastkpc_full_cuda_validate_materialized_target_for_prepared(setup, target)
})
Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
prepared <- fastkpc_full_cuda_phase6_prepare(setup)
reference <- fastkpc_full_cuda_phase5_optimize_cpp(
  setup, Y[, 1L], keep_transcript = TRUE
)
transcript <- reference$transcript
point_log_sp <- do.call(cbind, lapply(transcript, function(entry) {
  if (identical(entry$stage, "iteration_state")) {
    as.double(entry$current_log_sp)
  } else {
    as.double(entry$trial_log_sp)
  }
}))
point_y <- matrix(
  rep(Y[, 1L], length(transcript)),
  nrow = nrow(Y), ncol = length(transcript)
)
gpu <- fastkpc_full_cuda_phase6_evaluate_cuda(
  prepared, point_y, point_log_sp
)

receipt <- list(
  schema_version = "mgcv-formal-selected-sp-objective-replay-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  shard_id = shard_id,
  prepared_s_key_sha256 = prepared_key,
  residual_key_sha256 = residual_key,
  state_indices = state_indices,
  reference = reference,
  cpp_transcript = transcript,
  point_log_sp = point_log_sp,
  gpu = gpu,
  formal_context = list(setup = setup, Y = Y, target_index = 1L)
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
cat(sprintf(
  "shard=%d q=%d transcript=%d selected-log-sp=(%s)\n",
  shard_id, ncol(prepared$X), length(transcript),
  paste(sprintf("% .17g", reference$selected_log_sp), collapse = ",")
))
