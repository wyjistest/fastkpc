source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_shadow.R")
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
    "fastkpc", "artifacts", "full_cuda_ci",
    "fixed_sp_cuda_full_shadow_v1"
  )
)

started <- proc.time()[["elapsed"]]
catalog <- fastkpc_full_cuda_open_fixed_sp_catalog(
  phase0_dir, phase1_dir, phase2_dir, data_path
)
plan <- fastkpc_full_cuda_shadow_plan(catalog)
execution_snapshot <-
  fastkpc_full_cuda_phase3_create_shadow_execution_snapshot(
    catalog = catalog, plan = plan, scope = scope
  )
selected <- .fastkpc_full_cuda_phase3_resolve_shadow_execution_snapshot(
  execution_snapshot, expected_scope = scope
)
setup_keys <- selected$setup_keys
target_rows <- selected$target_rows
logical_tests <- selected$logical_rows
route_config <- fastkpc_full_cuda_phase3_route_config()
identity <- fastkpc_full_cuda_phase3_input_identity(catalog, device_id)
capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities

direct_paths <- fastkpc_full_cuda_phase3_artifact_paths(
  output_dir, kind = "full_shadow"
)
direct_present <- c(
  rds = file.exists(direct_paths$direct_ci_rds),
  summary = file.exists(direct_paths$direct_ci_summary_json)
)
if (xor(direct_present[["rds"]], direct_present[["summary"]])) {
  stop("existing direct-CI artifact pair is incomplete", call. = FALSE)
}
direct <- if (all(direct_present)) {
  fastkpc_full_cuda_phase3_validate_direct_ci_payload(
    output_dir, catalog
  )$payload
} else {
  fastkpc_full_cuda_shadow_write_direct_ci(
    catalog = catalog, output_dir = output_dir, plan = plan
  )
}
if (nrow(direct$rows) != 2213L || any(direct$rows$decision_flip) ||
    any(direct$rows$backend_error) || any(direct$rows$spectra_fallback)) {
  stop("Phase 3 direct-CI shadow gate failed", call. = FALSE)
}

lifecycle <- new.env(parent = emptyenv())
lifecycle$runtime_after <- NULL
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
    info, "Phase 3 shadow reserved runtime"
  )
  .fastkpc_full_cuda_phase3_validate_runtime_attestation(runtime, identity)
  if (info$workspace_reserve_count != 1L ||
      !isTRUE(info$cublas_user_workspace_installed) ||
      info$cublas_workspace_alignment < 256 ||
      info$augmented_workspace_bytes != as.double(8 * 415L * 64L)) {
    stop("Phase 3 shadow canonical runtime reserve is invalid",
         call. = FALSE)
  }
  keep <- TRUE
  runtime
}

runtime_destroy <- function(runtime) {
  on.exit(try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE), add = TRUE)
  info <- fixed_sp_cuda_runtime_info(runtime)
  fastkpc_full_cuda_fixed_sp_phase3c_validate_runtime_info(
    info, "Phase 3 shadow final runtime"
  )
  lifecycle$runtime_after <- info
  invisible(NULL)
}

executor <- function(context, shard_id, setup_keys, target_rows) {
  fastkpc_full_cuda_phase3_run_shadow_shard(
    context = context,
    shard_id = shard_id,
    setup_keys = setup_keys,
    target_rows = target_rows,
    catalog = catalog,
    execution_snapshot = execution_snapshot
  )
}

run <- fastkpc_full_cuda_phase3_run_shards(
  output_dir = output_dir,
  kind = "full_shadow",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route_config,
  executor = executor,
  runtime_create = runtime_create,
  runtime_destroy = runtime_destroy,
  scope = scope,
  canonical_setup_shards = TRUE,
  execution_snapshot = execution_snapshot
)
if (!identical(run$status, "complete")) {
  stop("Phase 3 shadow shard execution stopped before completion",
       call. = FALSE)
}

merged <- fastkpc_full_cuda_phase3_merge_shards(
  output_dir = output_dir,
  kind = "full_shadow",
  setup_keys = setup_keys,
  target_rows = target_rows,
  identity = identity,
  route_config = route_config,
  scope = scope,
  canonical_setup_shards = TRUE,
  execution_snapshot = execution_snapshot
)
payload <- merged$payload
summary <- fastkpc_full_cuda_phase3_validate_shadow_payload(
  payload = payload,
  expected_setup_keys = setup_keys,
  expected_target_rows = target_rows,
  expected_logical_tests = logical_tests,
  require_logical_authority = TRUE,
  expected_setup_rows = selected$setup_authority
)
rows <- payload$logical_ci_parity
rows <- rows[order(rows$logical_sequence_id, method = "radix"), , drop = FALSE]
rownames(rows) <- NULL
expected_logical <- logical_tests[order(
  logical_tests$logical_sequence_id, method = "radix"
), , drop = FALSE]
rownames(expected_logical) <- NULL
fastkpc_full_cuda_shadow_validate_conditional_rows(
  rows,
  expected_logical_tests = expected_logical
)

resources <- payload$resource_metrics
runtime <- fastkpc_full_cuda_shadow_runtime_counters(resources)
resource_gate <- nrow(resources) == length(setup_keys) &&
  sum(resources$target_count) == nrow(target_rows) &&
  sum(resources$prepared_handle_create_count) == length(setup_keys) &&
  sum(resources$prepared_handle_destroy_count) == length(setup_keys) &&
  sum(resources$residual_token_acquire_count) == length(setup_keys) &&
  sum(resources$residual_token_release_count) == length(setup_keys) &&
  sum(resources$output_slot_acquire_count) == length(setup_keys) &&
  sum(resources$output_slot_release_count) == length(setup_keys) &&
  runtime$shadow_materialize_call_count == length(setup_keys) &&
  runtime$shadow_materialize_target_count == nrow(target_rows) &&
  runtime$implicit_residual_d2h_count == 0L &&
  all(resources$shadow_materialize_call_count == 1L) &&
  identical(
    as.integer(resources$shadow_materialize_target_count),
    as.integer(resources$target_count)
  ) &&
  all(resources$implicit_residual_d2h_count == 0L) &&
  sum(resources$cpu_fallback_count) == 0L &&
  sum(resources$unknown_fallback_count) == 0L &&
  sum(resources$approximate_backend_count) == 0L &&
  sum(resources$cuda_device_synchronize_count) == 0L &&
  all(!resources$output_slot_leased_after_release)
if (!isTRUE(resource_gate)) {
  stop("Phase 3 shadow aggregate resource gate failed", call. = FALSE)
}

scope_gate <- switch(
  scope,
  iteration = length(setup_keys) == 44L && nrow(target_rows) == 270L &&
    nrow(rows) == 44L,
  qualification = length(setup_keys) == 2061L &&
    nrow(target_rows) == 6143L && nrow(rows) == 3808L &&
    sum(rows$near_alpha) == 1478L,
  full = length(setup_keys) == 8634L &&
    nrow(target_rows) == 110617L && nrow(rows) == 238276L &&
    sum(rows$near_alpha) == 1478L
)
decision_gate <- sum(rows$decision_flip) == 0L &&
  sum(rows$backend_error) == 0L &&
  sum(rows$spectra_fallback) == 0L
if (!isTRUE(scope_gate) || !isTRUE(decision_gate)) {
  stop("Phase 3 shadow logical subset gate failed", call. = FALSE)
}

elapsed_seconds <- as.double(proc.time()[["elapsed"]] - started)
cat("Phase 3 conditional shadow shard run:", run$status, "\n")
cat("scope:", scope, "\n")
cat("setups/targets:", summary$setup_count, "/", summary$target_count, "\n")
cat(
  "conditional logical rows:", summary$logical_test_count, "/",
  nrow(logical_tests), "\n"
)
cat("conditional near-alpha:", summary$near_alpha_count, "\n")
cat(
  "flips/errors/fallbacks:", summary$decision_flip_count, "/",
  summary$backend_error_count, "/", summary$spectra_fallback_count, "\n"
)
cat(
  "explicit/implicit residual D2H targets:",
  runtime$shadow_materialize_target_count, "/",
  runtime$implicit_residual_d2h_count, "\n"
)
cat(
  "written/reused shards:", length(run$written_shard_ids), "/",
  length(run$reused_shard_ids), "\n"
)
cat("elapsed seconds:", format(elapsed_seconds, digits = 8L), "\n")
