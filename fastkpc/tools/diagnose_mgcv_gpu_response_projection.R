source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

args <- commandArgs(trailingOnly = TRUE)
shard_id <- if (length(args) >= 1L) as.integer(args[[1L]]) else 45L
prepared_key <- if (length(args) >= 2L) args[[2L]] else {
  "c8c47b11af8006e5c845c00af69822d72f56e0c718effe23b5b28c33ef5417df"
}
residual_key <- if (length(args) >= 3L) args[[3L]] else {
  "8d4a2fb574d88f3bc0784d3ae393d9572c388a4cf1685479c1886d7237a1f84b"
}
output_path <- if (length(args) >= 4L) args[[4L]] else {
  "/tmp/fastkpc_mgcv_gpu_response_projection.rds"
}

if (is.na(shard_id) || shard_id < 0L || !nzchar(prepared_key) ||
    !nzchar(residual_key)) {
  stop("response projection witness arguments are invalid", call. = FALSE)
}
if (file.exists(output_path)) stop("output already exists", call. = FALSE)

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
  stop("response projection witness is missing", call. = FALSE)
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
gpu <- full_cuda_ci_multi_penalty_gcv_response_transcript_for_test(
  Y, prepared$magic_qr_packed, prepared$magic_tau
)

qr_reference <- qr(prepared$X, LAPACK = TRUE)
cpu_y0 <- vapply(seq_len(ncol(Y)), function(target) {
  qr.qty(qr_reference, Y[, target])[seq_len(ncol(prepared$X))]
}, numeric(ncol(prepared$X)))
cpu_squared_norm <- vapply(seq_len(ncol(Y)), function(target) {
  value <- 0
  for (row in seq_len(nrow(Y))) {
    value <- value + Y[row, target] * Y[row, target]
  }
  value
}, numeric(1L))

first_mismatch <- function(expected, observed) {
  index <- which(expected != observed)[1L]
  if (is.na(index)) return(NULL)
  list(
    index = as.integer(index),
    expected = expected[[index]],
    observed = observed[[index]],
    expected_hex = sprintf("%a", expected[[index]]),
    observed_hex = sprintf("%a", observed[[index]])
  )
}

receipt <- list(
  schema_version = "mgcv-gpu-response-projection-transcript-v1",
  native_sha256 = unname(digest::digest(
    file = "fastkpc/build/fastkpc_cuda.so", algo = "sha256",
    serialize = FALSE
  )),
  shard_id = shard_id,
  prepared_s_key_sha256 = prepared_key,
  residual_key_sha256 = residual_key,
  state_indices = state_indices,
  n = nrow(Y),
  coefficient_dim = ncol(prepared$X),
  target_count = ncol(Y),
  y0_bitwise_exact = identical(unname(cpu_y0), unname(gpu$y0)),
  squared_norm_bitwise_exact = identical(
    unname(cpu_squared_norm), unname(gpu$squared_norm)
  ),
  y0_max_abs_error = max(abs(cpu_y0 - gpu$y0)),
  squared_norm_max_abs_error = max(abs(
    cpu_squared_norm - gpu$squared_norm
  )),
  y0_first_mismatch = first_mismatch(cpu_y0, gpu$y0),
  squared_norm_first_mismatch = first_mismatch(
    cpu_squared_norm, gpu$squared_norm
  ),
  cpu_y0 = cpu_y0,
  gpu_y0 = gpu$y0,
  cpu_squared_norm = cpu_squared_norm,
  gpu_squared_norm = gpu$squared_norm
)
saveRDS(receipt, output_path, compress = "xz", version = 3L)
print(receipt[c(
  "schema_version", "native_sha256", "shard_id", "n",
  "coefficient_dim", "target_count", "y0_bitwise_exact",
  "squared_norm_bitwise_exact", "y0_max_abs_error",
  "squared_norm_max_abs_error", "y0_first_mismatch",
  "squared_norm_first_mismatch"
)], digits = 17)
