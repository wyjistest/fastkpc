source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

require_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

read_integer <- function(name, default, minimum = 0L) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, unset = default)))
  require_true(
    length(value) == 1L && !is.na(value) && value >= minimum,
    paste0(name, " must be one integer >= ", minimum)
  )
  value
}

if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  stop("Phase 6 mixed-window benchmark requires CUDA", call. = FALSE)
}

shard_id <- read_integer("FASTKPC_PHASE6_WINDOW_SHARD", "0")
setup_count <- read_integer("FASTKPC_PHASE6_WINDOW_SETUPS", "32", 1L)
concurrency <- read_integer("FASTKPC_PHASE6_WINDOW_CONCURRENCY", "32", 1L)
warmup_count <- read_integer("FASTKPC_PHASE6_WINDOW_WARMUPS", "2", 1L)
repetition_count <- read_integer(
  "FASTKPC_PHASE6_WINDOW_REPETITIONS", "7", 1L
)
output_path <- Sys.getenv("FASTKPC_PHASE6_WINDOW_OUTPUT", unset = "")
require_true(
  concurrency <= setup_count && concurrency <= 64L,
  "Phase 6 mixed-window concurrency is outside the supported envelope"
)

data <- readRDS(file.path(
  "fastkpc", "artifacts", "kpc_tprs_real_zhu",
  "cancer_RD-causalDiscoveryInput.rds"
))
shard <- readRDS(file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1",
  "shards", paste0("shard_", shard_id, ".rds")
))

contexts <- list()
for (setup_key in names(shard$prepared_s_setups)) {
  setup <- shard$prepared_s_setups[[setup_key]]
  target_index <- which(
    shard$target_states$prepared_s_key_sha256 == setup_key
  )
  if (length(setup$penalty_blocks) <= 1L || length(target_index) <= 1L) {
    next
  }
  targets <- lapply(target_index, function(index) {
    fastkpc_full_cuda_materialize_target_state(
      shard$target_states[index, , drop = FALSE],
      data, setup$dataset_sha256
    )
  })
  Y <- do.call(cbind, lapply(targets, `[[`, "y"))
  prepared <- fastkpc_full_cuda_phase6_prepare(setup)
  contexts[[length(contexts) + 1L]] <- list(
    setup_key = setup_key,
    handle = fastkpc_full_cuda_phase6_prepared_create(
      prepared, target_capacity = ncol(Y)
    ),
    Y = Y,
    target_keys = as.character(
      shard$target_states$residual_key_sha256[target_index]
    ),
    coefficient_dim = ncol(setup$X),
    penalty_count = length(setup$penalty_blocks)
  )
  if (length(contexts) == setup_count) break
}
require_true(
  length(contexts) == setup_count,
  "Phase 6 mixed-window shard has too few eligible setups"
)
on.exit(lapply(contexts, function(context) {
  try(
    full_cuda_ci_multi_penalty_gcv_prepared_free_native(context$handle),
    silent = TRUE
  )
}), add = TRUE)

execute <- function() {
  started <- proc.time()[["elapsed"]]
  value <- fastkpc_full_cuda_phase6_optimize_prepared_multi(
    handles = lapply(contexts, `[[`, "handle"),
    target_batches = lapply(contexts, `[[`, "Y"),
    target_keys = lapply(contexts, `[[`, "target_keys"),
    concurrency = concurrency
  )
  wall_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  signature <- fastkpc_full_cuda_census_hash_raw(serialize(
    lapply(value$setups, function(result) list(
      selected_log_sp = result$optimization$selected_log_sp,
      score = result$optimization$score,
      optimizer_status = result$optimization$optimizer_status
    )), NULL, version = 2L
  ))
  diagnostics <- lapply(
    value$setups, function(result) result$optimization$diagnostics
  )
  lapply(value$setups, function(result) {
    full_cuda_ci_multi_penalty_gcv_residual_free_native(result$residual)
  })
  sum_field <- function(name) {
    sum(vapply(diagnostics, `[[`, numeric(1L), name))
  }
  data.frame(
    wall_ms = wall_ms,
    signature_sha256 = signature,
    qr_bidiagonal_reduction_cycles = sum_field(
      "cuda_qr_bidiagonal_reduction_cycles"
    ),
    bidiagonal_svd_cycles = sum_field("cuda_bidiagonal_svd_cycles"),
    svd_vector_postback_cycles = sum_field(
      "cuda_svd_vector_postback_cycles"
    ),
    left_vector_product_cycles = sum_field(
      "cuda_left_vector_product_cycles"
    ),
    guarded_qr_evaluation_count = sum_field(
      "cuda_guarded_qr_evaluation_count"
    ),
    stable_svd_evaluation_count = sum_field(
      "cuda_stable_svd_evaluation_count"
    ),
    stringsAsFactors = FALSE
  )
}

warmups <- lapply(seq_len(warmup_count), function(index) execute())
runs <- do.call(rbind, lapply(seq_len(repetition_count), function(index) {
  execute()
}))
require_true(
  length(unique(c(
    vapply(warmups, `[[`, character(1L), "signature_sha256"),
    runs$signature_sha256
  ))) == 1L,
  "Phase 6 mixed-window benchmark result signature drifted"
)

stage_columns <- c(
  "qr_bidiagonal_reduction_cycles", "bidiagonal_svd_cycles",
  "svd_vector_postback_cycles", "left_vector_product_cycles"
)
stage_cycles <- colSums(runs[stage_columns])
shape <- data.frame(
  coefficient_dim = vapply(contexts, `[[`, integer(1L), "coefficient_dim"),
  penalty_count = vapply(contexts, `[[`, integer(1L), "penalty_count"),
  target_count = vapply(contexts, function(context) ncol(context$Y), integer(1L))
)
evidence <- list(
  schema_version =
    "full-cuda-ci-phase6-mixed-window-benchmark-evidence-v1",
  summary = list(
    schema_version =
      "full-cuda-ci-phase6-mixed-window-benchmark-summary-v1",
    shard_id = shard_id,
    setup_count = setup_count,
    target_count = sum(shape$target_count),
    concurrency = concurrency,
    warmup_count = warmup_count,
    repetition_count = repetition_count,
    median_wall_ms = stats::median(runs$wall_ms),
    minimum_wall_ms = min(runs$wall_ms),
    maximum_wall_ms = max(runs$wall_ms),
    result_signature_sha256 = runs$signature_sha256[[1L]],
    native_binary_sha256 = fastkpc_full_cuda_census_file_hash(
      file.path("fastkpc", "build", "fastkpc_cuda.so")
    ),
    guarded_qr_evaluation_count = runs$guarded_qr_evaluation_count[[1L]],
    stable_svd_evaluation_count = runs$stable_svd_evaluation_count[[1L]],
    stage_cycle_share = as.list(stage_cycles / sum(stage_cycles)),
    pass = TRUE
  ),
  shape = shape,
  runs = runs
)

print(evidence$summary)
if (nzchar(output_path)) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(output_path, ".tmp-", Sys.getpid())
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(evidence, temporary, compress = "gzip", version = 3L)
  if (!file.rename(temporary, output_path)) {
    stop("Phase 6 mixed-window benchmark rename failed", call. = FALSE)
  }
}
cat("PASS Phase 6 mixed-window CUDA benchmark\n")
