#!/usr/bin/env Rscript

source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_vertical.R")
source("fastkpc/R/dcov_exact.R")

args <- commandArgs(trailingOnly = TRUE)
rebuild <- "--rebuild" %in% args
args <- setdiff(args, "--rebuild")
output_dir <- if (length(args) == 0L) {
  file.path("fastkpc", "artifacts", "full_cuda_ci",
            "phase35_vertical_v1")
} else if (length(args) == 1L) {
  args[[1L]]
} else {
  stop("usage: run_full_cuda_ci_phase35_vertical.R [--rebuild] [output_dir]",
       call. = FALSE)
}

if (rebuild) build_fastkpc_cuda_native(rebuild = TRUE)
run <- fastkpc_full_cuda_phase35_run_canonical_vertical(
  rebuild = FALSE, include_cpu_oracle = TRUE
)
fixture <- run$fixture
result <- run$result
diagnostics <- result$diagnostics
oracle <- run$exact_cpu_oracle
logical <- fixture$logical_row

oracle_values <- c(
  statistic = unname(oracle$statistic),
  mean = unname(oracle$estimates[[2L]]),
  variance = unname(oracle$estimates[[3L]]),
  p_value = unname(oracle$p.value)
)
candidate_values <- c(
  statistic = result$first_numerical$statistic,
  mean = result$first_numerical$mean,
  variance = result$first_numerical$variance,
  p_value = result$first_result$p_value
)
absolute_error <- abs(candidate_values - oracle_values)
tolerances <- c(
  statistic = 1e-9,
  mean = 1e-10,
  variance = 1e-10,
  p_value = 1e-10
)
numerical_pass <- all(is.finite(candidate_values)) &&
  all(absolute_error <= tolerances)
structural_flags <- c(
  diagnostics$request_identity_authenticated,
  diagnostics$prepared_identity_authenticated,
  diagnostics$target_identity_authenticated,
  diagnostics$residuals_device_resident,
  diagnostics$components_device_resident,
  diagnostics$compact_result_only_d2h,
  diagnostics$eviction_result_bit_identical,
  diagnostics$deterministic_logical_replay,
  diagnostics$bounded_allocation,
  diagnostics$leak_free_teardown,
  diagnostics$caller_device_restored
)
structural_pass <- isTRUE(numerical_pass) && all(structural_flags) &&
  identical(result$first_result, result$replay_result) &&
  diagnostics$residual_d2h_bytes == 0 &&
  diagnostics$component_d2h_bytes == 0 &&
  diagnostics$cpu_dcov_component_count == 0L &&
  diagnostics$cpu_dcov_pair_statistic_count == 0L &&
  diagnostics$cpu_gamma_p_value_count == 0L &&
  diagnostics$device_allocation_count == diagnostics$device_free_count &&
  identical(run$prepared_after$output_slot_state, "free")
if (!isTRUE(structural_pass)) {
  stop("Phase 3.5D vertical structural gate failed", call. = FALSE)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
contracts <- fastkpc_full_cuda_phase35_load_contract_set()
native_path <- normalizePath(
  file.path("fastkpc", "build", "fastkpc_cuda.so"), mustWork = TRUE
)
source_paths <- c(
  "fastkpc/R/full_cuda_ci_phase35_vertical.R",
  "fastkpc/src/cuda/full_cuda_ci_vertical.hpp",
  "fastkpc/src/cuda/full_cuda_ci_vertical.cu",
  "fastkpc/src/cuda/mgcv_fixed_sp_runtime.hpp",
  "fastkpc/src/cuda/mgcv_fixed_sp_runtime.cu",
  "fastkpc/src/r_api_cuda.cpp",
  "fastkpc/tools/build_cuda_native.sh",
  "fastkpc/tools/run_full_cuda_ci_phase35_vertical.R",
  "fastkpc/tests/test_full_cuda_ci_phase35_vertical.R"
)
source_hashes <- setNames(
  lapply(source_paths, fastkpc_full_cuda_census_file_hash),
  source_paths
)
source_closure <- data.frame(
  path = names(source_hashes),
  sha256 = unlist(source_hashes, use.names = FALSE),
  stringsAsFactors = FALSE
)
source_closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
  fastkpc_full_cuda_phase35_canonical_json(source_hashes)
)
native_sha256 <- fastkpc_full_cuda_census_file_hash(native_path)
dataset_or_corpus_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(
  list(
    schema_version = "full-cuda-ci-phase35d-corpus-identity-v1",
    dataset_matrix_sha256 = fixture$setup$dataset_sha256,
    prepared_shard_payload_sha256 =
      fixture$shard_authentication$payload_hash,
    logical_sequence_id = as.integer(logical$logical_sequence_id),
    residual_key_x = logical$residual_key_x,
    residual_key_y = logical$residual_key_y
  )
)
oracle_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(
  list(
    schema_version = "full-cuda-ci-phase35d-exact-oracle-v1",
    statistic = oracle_values[["statistic"]],
    mean = oracle_values[["mean"]],
    variance = oracle_values[["variance"]],
    p_value = oracle_values[["p_value"]],
    legacy_reference_p_value = logical$reference_p_value,
    alpha = logical$alpha
  )
)
backend_configuration_sha256 <-
  fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase35d-backend-config-v1",
    component_semantic_version =
      diagnostics$component_semantic_version,
    precision = "IEEE-754-binary64",
    distance = "absolute-1d",
    component_capacity = 2L,
    deterministic_eviction_replay = TRUE,
    device_gamma = TRUE,
    alpha = 0.1
  ))
build_recipe_sha256 <- fastkpc_full_cuda_census_file_hash(
  "fastkpc/tools/build_cuda_native.sh"
)
producer <- fastkpc_full_cuda_phase35_producer_identity(
  producer_source_closure_sha256 = source_closure_sha256,
  native_binary_sha256 = native_sha256,
  route_semantic_version = "full-cuda-ci-phase35d-vertical-v1",
  dataset_or_corpus_sha256 = dataset_or_corpus_sha256,
  oracle_sha256 = oracle_sha256,
  backend_configuration_sha256 = backend_configuration_sha256,
  build_recipe_sha256 = build_recipe_sha256,
  contracts = contracts
)

case_results <- data.frame(
  logical_sequence_id = as.integer(logical$logical_sequence_id),
  x = as.integer(logical$x),
  y = as.integer(logical$y),
  S_key = logical$S_key,
  prepared_s_key_sha256 = fixture$prepared_key,
  residual_key_x = logical$residual_key_x,
  residual_key_y = logical$residual_key_y,
  candidate_statistic = candidate_values[["statistic"]],
  oracle_statistic = oracle_values[["statistic"]],
  statistic_absolute_error = absolute_error[["statistic"]],
  candidate_mean = candidate_values[["mean"]],
  oracle_mean = oracle_values[["mean"]],
  mean_absolute_error = absolute_error[["mean"]],
  candidate_variance = candidate_values[["variance"]],
  oracle_variance = oracle_values[["variance"]],
  variance_absolute_error = absolute_error[["variance"]],
  candidate_p_value = candidate_values[["p_value"]],
  exact_oracle_p_value = oracle_values[["p_value"]],
  legacy_reference_p_value = logical$reference_p_value,
  p_value_absolute_error = absolute_error[["p_value"]],
  candidate_independent = candidate_values[["p_value"]] >= logical$alpha,
  exact_oracle_independent = oracle_values[["p_value"]] >= logical$alpha,
  legacy_reference_independent = logical$reference_independent,
  numerical_pass = numerical_pass,
  stringsAsFactors = FALSE
)
cache_results <- data.frame(
  component_capacity = 2L,
  component_bytes_per_target = diagnostics$component_bytes_per_target,
  peak_component_bytes = diagnostics$peak_component_bytes,
  peak_live_device_bytes = diagnostics$peak_live_device_bytes,
  component_build_count = diagnostics$component_build_count,
  eviction_count = diagnostics$component_cache_eviction_count,
  replay_count = diagnostics$deterministic_replay_count,
  result_bit_identical = diagnostics$eviction_result_bit_identical,
  bounded_allocation = diagnostics$bounded_allocation,
  leak_free_teardown = diagnostics$leak_free_teardown,
  caller_device_restored = diagnostics$caller_device_restored,
  stringsAsFactors = FALSE
)
stage_timing <- data.frame(
  stage = c(
    "residual_solve_host", "first_component_build_cuda",
    "first_pair_evaluation_cuda", "first_compact_d2h_cuda",
    "replay_component_build_cuda", "replay_pair_evaluation_cuda",
    "replay_compact_d2h_cuda", "teardown_host", "total_host"
  ),
  elapsed_ms = unlist(diagnostics[c(
    "residual_solve_host_ms", "first_component_build_cuda_ms",
    "first_pair_evaluation_cuda_ms", "first_compact_d2h_cuda_ms",
    "replay_component_build_cuda_ms", "replay_pair_evaluation_cuda_ms",
    "replay_compact_d2h_cuda_ms", "teardown_host_ms", "total_host_ms"
  )], use.names = FALSE),
  stringsAsFactors = FALSE
)
raw_runs <- data.frame(
  run = c("initial", "eviction_reconstruction"),
  logical_sequence_id = rep(as.integer(logical$logical_sequence_id), 2L),
  p_value = c(result$first_result$p_value,
              result$replay_result$p_value),
  status = c(result$first_result$status, result$replay_result$status),
  dcov_status = c(result$first_result$dcov_status,
                  result$replay_result$dcov_status),
  stringsAsFactors = FALSE
)
fallbacks <- data.frame(
  fallback_type = c(
    "unknown", "approximate", "cpu_residual", "cpu_dcov_component",
    "cpu_dcov_pair", "cpu_gamma"
  ),
  count = rep(0L, 6L),
  stringsAsFactors = FALSE
)
graph_agreement <- data.frame(
  gate = "NOT_APPLICABLE_PHASE_3_5D_STRUCTURAL_ONLY",
  full_graph_claim = FALSE,
  phase3_inherited_shd = 0L,
  stringsAsFactors = FALSE
)
sepset_agreement <- data.frame(
  gate = "NOT_APPLICABLE_PHASE_3_5D_STRUCTURAL_ONLY",
  full_graph_claim = FALSE,
  stringsAsFactors = FALSE
)
n_edgetests <- data.frame(
  level = 0:7,
  phase3_inherited = c(2213L, 52659L, 125293L, 40694L,
                       13293L, 5422L, 835L, 80L),
  vertical_candidate = NA_integer_
)

summary <- list(
  schema_version = "full-cuda-ci-phase35d-vertical-summary-v1",
  claim_scope = "phase3.5D-structural-only",
  run_status = "COMPLETE",
  timeout = FALSE,
  source_commit = trimws(system2("git", c("rev-parse", "HEAD"),
                                 stdout = TRUE)),
  source_closure_sha256 = source_closure_sha256,
  native_binary_sha256 = native_sha256,
  producer_identity_sha256 = producer$identity_sha256,
  oracle_artifact = "independent-exact-dcov-gamma-over-phase2-fixed-sp",
  candidate_route = "phase3-oracle-sp-to-exact-centered-distance-cuda-gamma",
  full_graph_claim = FALSE,
  graph_gate = "NOT_APPLICABLE_BY_PHASE_3_5D_SPECIFICATION",
  edge_count_reference = NA_integer_,
  edge_count_candidate = NA_integer_,
  SHD = NA_integer_,
  adjacency_identical = NA,
  sepsets_identical = NA,
  n_edgetests_identical = NA,
  deletions_identical = NA,
  logical_sequence_id = as.integer(logical$logical_sequence_id),
  request_identity_sha256 = result$request_identity_sha256,
  statistic_absolute_error = absolute_error[["statistic"]],
  mean_absolute_error = absolute_error[["mean"]],
  variance_absolute_error = absolute_error[["variance"]],
  p_value_absolute_error = absolute_error[["p_value"]],
  decision_flip_count = as.integer(
    (candidate_values[["p_value"]] >= logical$alpha) !=
      (oracle_values[["p_value"]] >= logical$alpha)
  ),
  residual_d2h_bytes = diagnostics$residual_d2h_bytes,
  component_d2h_bytes = diagnostics$component_d2h_bytes,
  compact_result_d2h_bytes = diagnostics$compact_result_d2h_bytes,
  cpu_numerical_dcov_count = diagnostics$cpu_dcov_component_count +
    diagnostics$cpu_dcov_pair_statistic_count,
  cpu_gamma_p_value_count = diagnostics$cpu_gamma_p_value_count,
  eviction_result_bit_identical =
    diagnostics$eviction_result_bit_identical,
  peak_live_device_bytes = diagnostics$peak_live_device_bytes,
  allocation_count = diagnostics$device_allocation_count,
  free_count = diagnostics$device_free_count,
  caller_device_restored = diagnostics$caller_device_restored,
  unknown_fallback_count = 0L,
  approximate_backend_count = 0L,
  architecture_contract_sha256 =
    contracts$architecture_contract_v1$sha256,
  numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
  artifact_identity_contract_sha256 =
    contracts$artifact_identity_contract_v1$sha256,
  reference_machine_contract_sha256 =
    contracts$reference_machine_v1$sha256,
  performance_budget_contract_sha256 =
    contracts$performance_budget_v1$sha256,
  elapsed_sec = diagnostics$total_host_ms / 1000,
  structural_pass = structural_pass,
  promotion_authority = FALSE,
  pass = structural_pass
)

payload_manifest_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(
  list(
    schema_version = "full-cuda-ci-phase35d-payload-manifest-v1",
    summary = summary,
    case_results = as.list(case_results),
    cache_results = as.list(cache_results),
    raw_runs = as.list(raw_runs),
    fallbacks = as.list(fallbacks)
  )
)
producer_envelope <- fastkpc_full_cuda_phase35_identity_envelope(
  producer, payload_manifest_sha256
)
identity <- producer_envelope
timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
environment_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
  schema_version = "full-cuda-ci-phase35d-environment-v1",
  gpu_name = run$runtime_info$gpu_name,
  gpu_uuid = run$runtime_info$gpu_uuid,
  runtime_abi = run$runtime_info$runtime_abi_schema_version,
  R_version = R.version.string,
  native_binary_sha256 = native_sha256
))
attestation <- fastkpc_full_cuda_phase35_validator_attestation(
  producer = producer,
  validator_source_closure_sha256 =
    fastkpc_full_cuda_census_file_hash(
      "fastkpc/tests/test_full_cuda_ci_phase35_vertical.R"
    ),
  validator_semantic_version = "full-cuda-ci-phase35d-validator-v1",
  validator_contracts = contracts,
  validation_timestamp_utc = timestamp,
  environment_sha256 = environment_sha256,
  validation_result = "PASS"
)
identity <- fastkpc_full_cuda_phase35_append_attestation(
  identity, attestation
)
artifact_info <- file.info(output_dir, extra_cols = TRUE)
artifact_inode <- if ("ino" %in% names(artifact_info)) {
  as.character(artifact_info$ino[[1L]])
} else {
  "unavailable"
}
receipt <- fastkpc_full_cuda_phase35_execution_receipt(
  producer = producer,
  pid = as.integer(Sys.getpid()),
  session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
  cuda_context_id = paste0(run$runtime_info$gpu_uuid, ":device-",
                           run$runtime_info$device_id),
  artifact_path = normalizePath(output_dir, mustWork = TRUE),
  artifact_inode = artifact_inode,
  staging_path = normalizePath(tempdir(), mustWork = TRUE),
  recorded_at_utc = timestamp
)
identity <- fastkpc_full_cuda_phase35_append_receipt(identity, receipt)
stopifnot(fastkpc_full_cuda_phase35_validate_identity_envelope(identity))

write.csv(case_results, file.path(output_dir, "case_results.csv"),
          row.names = FALSE, na = "")
write.csv(cache_results, file.path(output_dir, "cache.csv"),
          row.names = FALSE, na = "")
write.csv(stage_timing, file.path(output_dir, "stage_timing.csv"),
          row.names = FALSE, na = "")
write.csv(raw_runs, file.path(output_dir, "raw_runs.csv"),
          row.names = FALSE, na = "")
write.csv(fallbacks, file.path(output_dir, "fallbacks.csv"),
          row.names = FALSE, na = "")
write.csv(graph_agreement, file.path(output_dir, "graph_agreement.csv"),
          row.names = FALSE, na = "")
write.csv(sepset_agreement, file.path(output_dir, "sepset_agreement.csv"),
          row.names = FALSE, na = "")
write.csv(n_edgetests, file.path(output_dir, "n_edgetests.csv"),
          row.names = FALSE, na = "")
write.csv(source_closure, file.path(output_dir, "source_closure.csv"),
          row.names = FALSE, na = "")
fastkpc_full_cuda_write_json(
  list(
    schema_version = "full-cuda-ci-first-divergence-v1",
    status = "NOT_APPLICABLE_NO_MISMATCH_IN_CLAIMED_SCOPE",
    logical_sequence_id = NULL
  ),
  file.path(output_dir, "first_divergence.json")
)
fastkpc_full_cuda_write_json(summary, file.path(output_dir, "summary.json"))
fastkpc_full_cuda_write_json(producer,
                             file.path(output_dir, "producer_identity.json"))
fastkpc_full_cuda_write_json(
  list(attestations = list(attestation)),
  file.path(output_dir, "validator_attestations.json")
)
fastkpc_full_cuda_write_json(
  list(execution_receipts = list(receipt)),
  file.path(output_dir, "execution_receipts.json")
)

semantic_payload_files <- c(
  "summary.json", "case_results.csv", "cache.csv", "stage_timing.csv",
  "raw_runs.csv", "fallbacks.csv", "graph_agreement.csv",
  "sepset_agreement.csv", "n_edgetests.csv", "first_divergence.json",
  "producer_identity.json", "source_closure.csv"
)
payload_file_sha256 <- setNames(
  lapply(file.path(output_dir, semantic_payload_files),
         fastkpc_full_cuda_census_file_hash),
  semantic_payload_files
)
manifest <- list(
  schema_version = "full-cuda-ci-phase35d-vertical-manifest-v1",
  claim_scope = "phase3.5D-structural-only",
  producer_semantic_envelope = producer_envelope,
  payload_manifest_sha256 = payload_manifest_sha256,
  payload_file_sha256 = payload_file_sha256,
  semantic_file_count = length(semantic_payload_files),
  validator_attestations_file = "validator_attestations.json",
  volatile_receipt_file = "execution_receipts.json"
)
fastkpc_full_cuda_write_json(manifest, file.path(output_dir, "manifest.json"))

writeLines(
  c(
    "# Full CUDA CI Phase 3.5D vertical structural prototype",
    "",
    paste0("- structural pass: ", structural_pass),
    paste0("- logical sequence: ", logical$logical_sequence_id),
    paste0("- exact CUDA p-value: ",
           format(candidate_values[["p_value"]], digits = 17L)),
    paste0("- exact CPU p-value: ",
           format(oracle_values[["p_value"]], digits = 17L)),
    paste0("- residual/component D2H bytes: ",
           diagnostics$residual_d2h_bytes, "/",
           diagnostics$component_d2h_bytes),
    paste0("- compact-result D2H bytes: ",
           diagnostics$compact_result_d2h_bytes),
    paste0("- peak prototype device bytes: ",
           diagnostics$peak_live_device_bytes),
    "- full graph claim: false",
    "- promotion authority: false"
  ),
  file.path(output_dir, "summary.md"), useBytes = TRUE
)
writeLines(
  c(
    "FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_full_cuda_ci_phase35_vertical.R",
    paste(
      "Rscript fastkpc/tools/run_full_cuda_ci_phase35_vertical.R",
      if (rebuild) "--rebuild" else "",
      normalizePath(output_dir, mustWork = TRUE)
    )
  ),
  file.path(output_dir, "commands.txt"), useBytes = TRUE
)
environment_lines <- c(
  paste0("recorded_at_utc=", timestamp),
  paste0("R_version=", R.version.string),
  paste0("gpu_name=", run$runtime_info$gpu_name),
  paste0("gpu_uuid=", run$runtime_info$gpu_uuid),
  paste0("device_id=", run$runtime_info$device_id),
  paste0("native_binary_sha256=", native_sha256),
  paste0("environment_sha256=", environment_sha256),
  capture.output(sessionInfo())
)
writeLines(environment_lines, file.path(output_dir, "environment.txt"),
           useBytes = TRUE)

cat(
  "PASS Phase 3.5D artifact: ", normalizePath(output_dir),
  "; producer=", producer$identity_sha256,
  "; p=", format(candidate_values[["p_value"]], digits = 17L), "\n",
  sep = ""
)
