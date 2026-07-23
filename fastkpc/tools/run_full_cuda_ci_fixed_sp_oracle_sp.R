source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase3_artifacts.R")

scalar_environment <- function(name, default) {
  value <- Sys.getenv(name, unset = default)
  if (length(value) != 1L || is.na(value) || !nzchar(value) ||
      grepl("[\r\n]", value)) {
    stop(name, " must be one nonempty scalar", call. = FALSE)
  }
  value
}

integer_environment <- function(name, default) {
  value <- scalar_environment(name, default)
  if (!grepl("^(0|[1-9][0-9]*)$", value) ||
      nchar(value, type = "bytes") > 10L) {
    stop(name, " must be a canonical nonnegative integer", call. = FALSE)
  }
  parsed <- suppressWarnings(as.numeric(value))
  if (!is.finite(parsed) || parsed > .Machine$integer.max) {
    stop(name, " is outside the integer range", call. = FALSE)
  }
  as.integer(parsed)
}

device_id <- integer_environment("FASTKPC_FULL_CUDA_PHASE3_DEVICE", "0")
scope <- scalar_environment("FASTKPC_FULL_CUDA_PHASE3_SCOPE", "iteration")
if (!scope %in% c("iteration", "qualification", "full")) {
  stop("FASTKPC_FULL_CUDA_PHASE3_SCOPE is invalid", call. = FALSE)
}
phase0_dir <- scalar_environment(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE0_DIR",
  file.path("fastkpc", "artifacts", "full_cuda_ci", "oracle_351x48_v1")
)
phase1_dir <- scalar_environment(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE1_DIR",
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "workload_census_351x48_v1"
  )
)
phase2_dir <- scalar_environment(
  "FASTKPC_FULL_CUDA_PHASE3_PHASE2_DIR",
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "prepared_s_contract_v1"
  )
)
data_path <- scalar_environment(
  "FASTKPC_FULL_CUDA_PHASE3_DATA",
  file.path(
    "fastkpc", "artifacts", "kpc_tprs_real_zhu",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
output_dir <- scalar_environment(
  "FASTKPC_FULL_CUDA_PHASE3_OUTPUT",
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "fixed_sp_cuda_oracle_sp_v1"
  )
)

started <- proc.time()[["elapsed"]]
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir, phase1_dir, phase2_dir, data_path
)
selected_scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, scope)
setup_keys <- as.character(
  selected_scope$setup_rows$prepared_s_key_sha256
)
target_rows <- selected_scope$target_rows
target_rows <- .fastkpc_full_cuda_phase3_oracle_descriptor_target_rows(
  catalog, target_rows
)
route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- fastkpc_full_cuda_phase3_input_identity(catalog, device_id)
capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
artifact_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  output_dir, "oracle_sp"
)
runner_command_line <- paste(
  "Rscript fastkpc/tools/run_full_cuda_ci_fixed_sp_oracle_sp.R",
  paste(commandArgs(trailingOnly = TRUE), collapse = " ")
)
completion_markers <- c(
  manifest = file.exists(artifact_paths$manifest_json),
  summary = file.exists(artifact_paths$summary_json)
)
if (any(completion_markers)) {
  completed <- tryCatch(
    fastkpc_full_cuda_phase3_publish_oracle_artifact(
      output_dir = output_dir, setup_keys = setup_keys,
      target_rows = target_rows, identity = identity,
      route_config = route_config, scope = scope,
      catalog = catalog, device_id = device_id,
      command_lines = runner_command_line
    ),
    error = function(error) error
  )
  if (!inherits(completed, "error")) {
    cat("Phase 3 oracle artifact:", completed$status, "\n")
    cat(
      "setups/targets:", completed$validation$row_summary$setup_count, "/",
      completed$validation$row_summary$target_count, "\n"
    )
    quit(save = "no", status = 0L, runLast = FALSE)
  }
  remaining_markers <- c(
    manifest = file.exists(artifact_paths$manifest_json),
    summary = file.exists(artifact_paths$summary_json)
  )
  if (any(remaining_markers)) stop(completed)
}

runtime_create <- function() {
  runtime <- fixed_sp_cuda_runtime_create(device_id)
  keep <- FALSE
  on.exit({
    if (!keep) try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, capacity$n, capacity$null_dim, capacity$target_count,
    capacity$penalty_count, capacity$augmented_rows
  )
  info <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    info, "Phase 3 oracle reserved runtime"
  )
  .fastkpc_full_cuda_phase3_validate_runtime_attestation(runtime, identity)
  if (info$workspace_reserve_count != 1L ||
      !isTRUE(info$cublas_user_workspace_installed) ||
      info$cublas_workspace_alignment < 256 ||
      info$augmented_workspace_bytes != as.double(8 * 415L * 64L)) {
    stop("Phase 3 oracle canonical runtime reserve is invalid", call. = FALSE)
  }
  keep <- TRUE
  runtime
}

runtime_destroy <- function(runtime) {
  info <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    info, "Phase 3 oracle final runtime"
  )
  fixed_sp_cuda_runtime_free(runtime)
  invisible(NULL)
}

executor <- function(context, shard_id, setup_keys, target_rows) {
  fastkpc_full_cuda_phase3_run_oracle_shard(
    context = context, shard_id = shard_id, setup_keys = setup_keys,
    target_rows = target_rows, catalog = catalog, scope = scope
  )
}

run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir,
  kind = "oracle_sp",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route_config,
  executor = executor,
  runtime_create = runtime_create,
  runtime_destroy = runtime_destroy,
  scope = scope
)
if (!identical(run$status, "complete")) {
  stop("Phase 3 oracle shard execution stopped before completion",
       call. = FALSE)
}

merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = output_dir,
  kind = "oracle_sp",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route_config,
  scope = scope
)
payload <- merged$payload
summary <- fastkpc_full_cuda_phase3_summarize_oracle_rows(
  setup_results = payload$setup_results,
  target_parity = payload$target_parity,
  resource_metrics = payload$resource_metrics,
  stage_timing = payload$stage_timing,
  fallbacks = payload$fallbacks,
  failures = payload$failures
)

hard_gate <- summary$non_ok_solver_status_count == 0L &&
  summary$nonfinite_output_count == 0L &&
  summary$cpu_fallback_count == 0L &&
  summary$unknown_fallback_count == 0L &&
  summary$approximate_backend_count == 0L &&
  summary$failure_count == 0L &&
  summary$max_fitted_abs_diff < route_config$fitted_tolerance &&
  summary$max_fitted_relative_l2 < route_config$fitted_tolerance &&
  summary$max_residual_abs_diff < route_config$residual_tolerance &&
  summary$max_residual_relative_l2 < route_config$residual_tolerance &&
  summary$per_target_allocation_count_after_warmup == 0L &&
  summary$per_target_handle_create_count == 0L &&
  summary$implicit_residual_d2h_count == 0L &&
  summary$cuda_device_synchronize_count == 0L &&
  summary$target_level_stable_sync_count == 0L &&
  summary$output_slot_live_count == 0L &&
  summary$prepared_handle_create_count == summary$setup_count &&
  summary$prepared_handle_destroy_count == summary$setup_count &&
  summary$residual_token_acquire_count == summary$setup_count &&
  summary$residual_token_release_count == summary$setup_count &&
  summary$output_slot_acquire_count == summary$setup_count &&
  summary$output_slot_release_count == summary$setup_count
if (!isTRUE(hard_gate)) {
  stop("Phase 3 oracle row-derived hard gate failed", call. = FALSE)
}

if (identical(scope, "iteration")) {
  iteration_exact <- summary$setup_count == 44L &&
    summary$target_count == 270L &&
    identical(
      as.integer(unlist(summary[c(
        "planned_cholesky_target_count", "planned_qr_target_count",
        "planned_svd_target_count"
      )], use.names = FALSE)),
      c(172L, 31L, 67L)
    ) && identical(
      as.integer(unlist(summary[c(
        "executed_cholesky_target_count", "executed_qr_target_count",
        "executed_svd_target_count"
      )], use.names = FALSE)),
      c(172L, 31L, 67L)
    ) && summary$cholesky_to_svd_count == 0L &&
    summary$qr_to_svd_count == 0L
  if (!isTRUE(iteration_exact)) {
    stop("Phase 3 oracle iteration counts are not canonical", call. = FALSE)
  }
}

publication <- fastkpc_full_cuda_phase3_publish_oracle_artifact(
  output_dir = output_dir, setup_keys = setup_keys,
  target_rows = target_rows, identity = identity,
  route_config = route_config, scope = scope,
  catalog = catalog, device_id = device_id,
  command_lines = runner_command_line
)
if (!identical(publication$status, "published")) {
  stop("Phase 3 oracle top-level publication did not complete",
       call. = FALSE)
}

elapsed_seconds <- as.double(proc.time()[["elapsed"]] - started)
cat("Phase 3 oracle shard run:", run$status, "\n")
cat("Phase 3 oracle artifact:", publication$status, "\n")
cat("setups/targets:", summary$setup_count, "/", summary$target_count, "\n")
cat(
  "planned routes:", summary$planned_cholesky_target_count, "/",
  summary$planned_qr_target_count, "/",
  summary$planned_svd_target_count, "\n"
)
cat(
  "executed routes:", summary$executed_cholesky_target_count, "/",
  summary$executed_qr_target_count, "/",
  summary$executed_svd_target_count, "\n"
)
cat(
  "reroutes:", summary$cholesky_to_svd_count, "/",
  summary$qr_to_svd_count, "\n"
)
cat(
  "fallback/non-OK:",
  summary$cpu_fallback_count + summary$unknown_fallback_count +
    summary$approximate_backend_count,
  "/", summary$non_ok_solver_status_count, "\n"
)
cat(
  "written/reused shards:", length(run$written_shard_ids), "/",
  length(run$reused_shard_ids), "\n"
)
cat("elapsed seconds:", format(elapsed_seconds, digits = 8L), "\n")
