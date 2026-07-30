source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R")

require_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}
read_integer <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, unset = default)))
  require_true(
    length(value) == 1L && !is.na(value) && value >= 1L,
    paste0(name, " must be a positive integer")
  )
  value
}

if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  stop("Phase 6 concurrency benchmark requires CUDA", call. = FALSE)
}
repetitions <- read_integer(
  "FASTKPC_PHASE6_CONCURRENCY_BENCHMARK_REPETITIONS", "2"
)
output_path <- Sys.getenv(
  "FASTKPC_PHASE6_CONCURRENCY_BENCHMARK_OUTPUT", unset = ""
)
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
require_true(
  !is.null(setup) && length(state_indices) == 2L && ncol(setup$X) == 64L &&
    length(setup$penalty_blocks) == 7L,
  "Phase 6 concurrency benchmark fixture is malformed"
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
setup_count <- 64L
handles <- lapply(seq_len(setup_count), function(index) {
  fastkpc_full_cuda_phase6_prepared_create(
    prepared, target_capacity = ncol(Y)
  )
})
on.exit(lapply(handles, function(handle) {
  try(full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle), silent = TRUE)
}), add = TRUE)

signature <- function(optimization) {
  fastkpc_full_cuda_census_hash_raw(serialize(list(
    selected_log_sp = optimization$selected_log_sp,
    score = optimization$score,
    edf = optimization$edf,
    optimizer_iterations = optimization$optimizer_iterations,
    score_calls = optimization$score_calls,
    objective_calls = optimization$objective_calls,
    step_halving_count = optimization$step_halving_count,
    boundary_probe_count = optimization$boundary_probe_count,
    optimizer_status = optimization$optimizer_status
  ), NULL, version = 2L))
}
execute <- function(concurrency) {
  started <- proc.time()[["elapsed"]]
  value <- fastkpc_full_cuda_phase6_optimize_prepared_multi(
    handles = handles,
    target_batches = rep(list(Y), setup_count),
    target_keys = rep(list(target_keys), setup_count),
    concurrency = concurrency
  )
  external_wall_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  tokens <- lapply(value$setups, `[[`, "residual")
  on.exit(lapply(tokens, function(token) {
    try(full_cuda_ci_multi_penalty_gcv_residual_free_native(token), silent = TRUE)
  }), add = TRUE)
  optimizations <- lapply(value$setups, `[[`, "optimization")
  signatures <- vapply(optimizations, signature, character(1L))
  require_true(
    length(unique(signatures)) == 1L &&
      all(vapply(tokens, function(token) {
        info <- full_cuda_ci_multi_penalty_gcv_residual_info_native(token)
        isTRUE(info$device_resident) && !isTRUE(info$released)
      }, logical(1L))) &&
      value$diagnostics$worker_count == concurrency &&
      value$diagnostics$max_host_calls_in_flight == concurrency &&
      value$diagnostics$setup_stream_count == setup_count &&
      value$diagnostics$host_overlap_factor > 0,
    "Phase 6 concurrency benchmark execution drifted"
  )
  lapply(tokens, full_cuda_ci_multi_penalty_gcv_residual_free_native)
  list(
    signature = signatures[[1L]],
    measurement = data.frame(
      concurrency = concurrency,
      external_wall_ms = external_wall_ms,
      native_wall_ms = value$diagnostics$wall_host_ms,
      summed_setup_host_ms = value$diagnostics$summed_setup_host_ms,
      host_overlap_factor = value$diagnostics$host_overlap_factor,
      max_host_calls_in_flight =
        value$diagnostics$max_host_calls_in_flight,
      stringsAsFactors = FALSE
    )
  )
}

warm <- execute(1L)
concurrency_values <- c(1L, 2L, 4L, 8L, 16L, 32L, 64L)
runs <- vector("list", length(concurrency_values) * repetitions)
ordinal <- 0L
for (repetition in seq_len(repetitions)) {
  for (concurrency in concurrency_values) {
    ordinal <- ordinal + 1L
    cat(
      "Phase 6 concurrency benchmark repetition ", repetition, "/",
      repetitions, " concurrency=", concurrency, "\n", sep = ""
    )
    flush.console()
    run <- execute(concurrency)
    run$measurement$repetition <- repetition
    runs[[ordinal]] <- run
  }
}
signatures <- vapply(runs, `[[`, character(1L), "signature")
require_true(
  length(unique(c(warm$signature, signatures))) == 1L,
  "Phase 6 concurrency levels changed optimizer results"
)
raw_runs <- do.call(rbind, lapply(runs, `[[`, "measurement"))
medians <- aggregate(
  external_wall_ms ~ concurrency, raw_runs, stats::median
)
medians <- medians[order(medians$concurrency), , drop = FALSE]
best <- medians$concurrency[[which.min(medians$external_wall_ms)]]
evidence <- list(
  schema_version =
    "full-cuda-ci-phase6-concurrency-benchmark-evidence-v1",
  summary = list(
    schema_version =
      "full-cuda-ci-phase6-concurrency-benchmark-summary-v1",
    setup_count = setup_count,
    target_count = setup_count * ncol(Y),
    coefficient_dim = ncol(setup$X),
    penalty_count = length(setup$penalty_blocks),
    repetition_count = repetitions,
    concurrency_values = concurrency_values,
    result_signature_sha256 = signatures[[1L]],
    fastest_concurrency = as.integer(best),
    fastest_median_wall_ms = medians$external_wall_ms[
      medians$concurrency == best
    ][[1L]],
    pass = TRUE
  ),
  median_wall_ms = medians,
  raw_runs = raw_runs
)
print(medians, row.names = FALSE)
if (nzchar(output_path)) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(output_path, ".tmp-", Sys.getpid())
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(evidence, temporary, compress = "gzip", version = 3L)
  if (!file.rename(temporary, output_path)) {
    stop("Phase 6 concurrency benchmark rename failed", call. = FALSE)
  }
}
lapply(handles, full_cuda_ci_multi_penalty_gcv_prepared_free_native)
cat(
  "PASS Phase 6 concurrency benchmark fastest=", best,
  " median_ms=",
  format(evidence$summary$fastest_median_wall_ms, digits = 8L), "\n",
  sep = ""
)
