#!/usr/bin/env Rscript

source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_phase35_feasibility.R")

options(digits = 17)

args <- commandArgs(trailingOnly = TRUE)
evidence_option <- grep("^--evidence=", args, value = TRUE)
if (length(evidence_option) > 1L) {
  stop("--evidence may be supplied at most once", call. = FALSE)
}
evidence_manifest_path <- if (length(evidence_option) == 1L) {
  sub("^--evidence=", "", evidence_option)
} else {
  "/tmp/fastkpc-phase35-prepublication-evidence.rds"
}
args <- setdiff(args, evidence_option)
output_dir <- if (length(args) == 0L) {
  file.path(
    "fastkpc", "artifacts", "full_cuda_ci", "phase35_feasibility_v1"
  )
} else if (length(args) == 1L) {
  args[[1L]]
} else {
  stop(
    "usage: run_full_cuda_ci_phase35_feasibility.R [--evidence=PATH] [output_dir]",
    call. = FALSE
  )
}

run_checked <- function(command, arguments, label) {
  output <- suppressWarnings(system2(
    command, arguments, stdout = TRUE, stderr = TRUE
  ))
  status <- attr(output, "status", exact = TRUE)
  if (is.null(status)) status <- 0L
  if (status != 0L) {
    stop(
      label, " failed with status ", status, ": ",
      paste(output, collapse = "\n"), call. = FALSE
    )
  }
  output
}

nvcc <- "/usr/local/cuda/bin/nvcc"
if (!file.exists(nvcc)) stop("qualified nvcc is missing", call. = FALSE)
build_dir <- file.path("fastkpc", "build")
dir.create(build_dir, recursive = TRUE, showWarnings = FALSE)
benchmark_specs <- list(
  eig = list(
    source = "fastkpc/tools/phase35_cuda_eig_benchmark.cu",
    binary = file.path(build_dir, "phase35_cuda_eig_benchmark"),
    libraries = c("-lcusolver", "-lcublas"),
    run_args = c("351", "100", "35")
  ),
  block = list(
    source = "fastkpc/tools/phase35_cuda_block_benchmark.cu",
    binary = file.path(build_dir, "phase35_cuda_block_benchmark"),
    libraries = c("-lcusolver", "-lcublas"),
    run_args = c("351", "62", "47", "12", "20")
  )
)

gpu_query <- run_checked(
  "nvidia-smi",
  c(
    "--query-gpu=index,name,uuid,memory.total,driver_version,pstate,temperature.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits"
  ),
  "reference GPU query"
)
compute_query <- suppressWarnings(system2(
  "nvidia-smi",
  c(
    "--query-compute-apps=pid,process_name,used_memory",
    "--format=csv,noheader,nounits"
  ),
  stdout = TRUE, stderr = TRUE
))
compute_status <- attr(compute_query, "status", exact = TRUE)
if (!is.null(compute_status) && compute_status != 0L) {
  stop("GPU compute-process query failed", call. = FALSE)
}
if (length(compute_query[nzchar(trimws(compute_query))]) != 0L) {
  stop("reference GPU is not idle before the sequential benchmarks",
       call. = FALSE)
}

compile_rows <- vector("list", length(benchmark_specs))
outputs <- list()
for (index in seq_along(benchmark_specs)) {
  name <- names(benchmark_specs)[[index]]
  spec <- benchmark_specs[[index]]
  compile_args <- c(
    "-std=c++17", "-O3", "-arch=sm_89", spec$source,
    "-o", spec$binary, spec$libraries
  )
  run_checked(nvcc, compile_args, paste0(name, " benchmark compilation"))
  output <- run_checked(
    normalizePath(spec$binary, winslash = "/", mustWork = TRUE),
    spec$run_args, paste0(name, " benchmark execution")
  )
  outputs[[name]] <- output
  compile_rows[[index]] <- data.frame(
    benchmark = name,
    source_path = spec$source,
    source_sha256 = fastkpc_full_cuda_census_file_hash(spec$source),
    binary_path = spec$binary,
    binary_sha256 = fastkpc_full_cuda_census_file_hash(spec$binary),
    compiler_path = nvcc,
    compiler_sha256 = fastkpc_full_cuda_census_file_hash(nvcc),
    compile_arguments = paste(compile_args, collapse = " "),
    run_arguments = paste(spec$run_args, collapse = " "),
    raw_output = paste(output, collapse = "\n"),
    sequential_idle_gpu_measurement = TRUE,
    stringsAsFactors = FALSE
  )
}
benchmark_builds <- do.call(rbind, compile_rows)
rownames(benchmark_builds) <- NULL
measurements <- fastkpc_full_cuda_phase35_parse_candidate_measurements(
  outputs$eig, outputs$block
)

contracts <- fastkpc_full_cuda_phase35_load_contract_set()
evidence <- fastkpc_full_cuda_phase35_load_evidence_bundle(
  evidence_manifest_path, require_current_execution_source = TRUE
)
reference <- contracts$reference_machine_v1$payload
runtime <- evidence$two_scale$runtime_info
normalized_runtime_uuid <- tolower(gsub("-", "", runtime$gpu_uuid))
normalized_contract_uuid <- tolower(gsub("-", "", reference$gpu$uuid))
if (!identical(runtime$gpu_name, reference$gpu$model) ||
    !identical(normalized_runtime_uuid, normalized_contract_uuid) ||
    runtime$device_id != reference$gpu$device_id ||
    runtime$compute_capability_major != 8L ||
    runtime$compute_capability_minor != 9L ||
    !identical(runtime$cusolver_deterministic_mode, "enabled") ||
    !identical(runtime$cublas_math_mode, "pedantic") ||
    !identical(runtime$cublas_atomics_mode, "not_allowed")) {
  stop("reference GPU or deterministic runtime contract mismatch",
       call. = FALSE)
}

cache_memory <- fastkpc_full_cuda_phase35_build_cache_memory_model(
  evidence, gpu_memory_bytes = reference$gpu$total_memory_bytes
)
performance <- fastkpc_full_cuda_phase35_build_performance_model(
  evidence, measurements, contracts
)
performance_tables <- fastkpc_full_cuda_phase35_build_performance_tables(
  performance
)
scale_results <- fastkpc_full_cuda_phase35_build_scale_results(evidence)
full_pair_decisions <-
  fastkpc_full_cuda_phase35_build_full_pair_decisions(evidence)
candidate_decisions <- fastkpc_full_cuda_phase35_build_candidate_decisions(
  evidence, measurements, performance, scale_results
)
corpus_policy <- fastkpc_full_cuda_phase35_build_corpus_policy(
  contracts, evidence
)

manifest_evidence <- evidence$manifest
setup_results_path <- file.path(
  "fastkpc", "artifacts", "full_cuda_ci", "fixed_sp_cuda_oracle_sp_v1",
  "setup_results.rds"
)
native_path <- file.path("fastkpc", "build", "fastkpc_cuda.so")
evidence_inputs <- data.frame(
  evidence_id = c(
    "prepublication-manifest", "two-scale-guarded-hybrid",
    "full-conditional-exact-screen", "full-conditional-guarded-hybrid",
    "phase3-setup-results", "executed-native-binary"
  ),
  path = c(
    evidence_manifest_path, manifest_evidence$two_scale_rds,
    manifest_evidence$full_exact_rds, manifest_evidence$full_hybrid_rds,
    setup_results_path, native_path
  ),
  sha256 = c(
    fastkpc_full_cuda_census_file_hash(evidence_manifest_path),
    manifest_evidence$two_scale_rds_sha256,
    manifest_evidence$full_exact_rds_sha256,
    manifest_evidence$full_hybrid_rds_sha256,
    fastkpc_full_cuda_census_file_hash(setup_results_path),
    manifest_evidence$native_binary_sha256
  ),
  embedded_summary_sufficient_for_on_disk_validation = c(
    TRUE, TRUE, TRUE, TRUE, TRUE, TRUE
  ),
  stringsAsFactors = FALSE
)

runner_path <- "fastkpc/tools/run_full_cuda_ci_phase35_feasibility.R"
source_paths <- sort(unique(c(
  names(manifest_evidence$execution_source_file_sha256),
  "fastkpc/R/full_cuda_ci_gate.R",
  "fastkpc/R/full_cuda_ci_workload_census.R",
  "fastkpc/R/full_cuda_ci_phase35_contracts.R",
  "fastkpc/R/full_cuda_ci_phase35_feasibility.R",
  runner_path,
  vapply(benchmark_specs, `[[`, character(1L), "source")
)), method = "radix")
source_hashes <- setNames(as.list(vapply(
  source_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)), source_paths)
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
    schema_version = "full-cuda-ci-phase35-feasibility-corpus-v1",
    development_contract_sha256 =
      contracts$development_qualification_corpus_v1$sha256,
    two_scale_sha256 = manifest_evidence$two_scale_rds_sha256,
    full_exact_sha256 = manifest_evidence$full_exact_rds_sha256,
    full_hybrid_sha256 = manifest_evidence$full_hybrid_rds_sha256
  )
)
oracle_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
  schema_version = "full-cuda-ci-phase35-feasibility-oracle-v1",
  logical_trace_file_sha256 =
    contracts$development_qualification_corpus_v1$payload$
      source_identities$oracle_logical_trace_file_sha256,
  qualification_pair_count = 3808L,
  full_conditional_pair_count = 238276L,
  allowed_decision_flip_count = 0L
))
backend_configuration_sha256 <-
  fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase35-guarded-hybrid-config-v1",
    candidate_id = evidence$full_hybrid$candidate_id,
    guard_lower_inclusive = evidence$full_hybrid$guard$lower_inclusive,
    guard_upper_inclusive = evidence$full_hybrid$guard$upper_inclusive,
    alpha = evidence$full_hybrid$guard$alpha,
    reference_component_capacity =
      cache_memory$policy$reference_component_capacity,
    minimum_component_capacity =
      cache_memory$policy$minimum_component_capacity,
    exact_outside_guard_is_internal_screen = TRUE,
    universal_p_value_parity_claim = FALSE
  ))
build_recipe_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
  schema_version = "full-cuda-ci-phase35-feasibility-build-recipe-v1",
  nvcc_sha256 = fastkpc_full_cuda_census_file_hash(nvcc),
  native_build_recipe_sha256 = fastkpc_full_cuda_census_file_hash(
    "fastkpc/tools/build_cuda_native.sh"
  ),
  benchmark_source_sha256 = as.list(benchmark_builds$source_sha256),
  benchmark_binary_sha256 = as.list(benchmark_builds$binary_sha256),
  benchmark_compile_arguments = as.list(benchmark_builds$compile_arguments)
))
producer <- fastkpc_full_cuda_phase35_producer_identity(
  producer_source_closure_sha256 = source_closure_sha256,
  native_binary_sha256 = native_sha256,
  route_semantic_version = "full-cuda-ci-phase35-guarded-feasibility-v1",
  dataset_or_corpus_sha256 = dataset_or_corpus_sha256,
  oracle_sha256 = oracle_sha256,
  backend_configuration_sha256 = backend_configuration_sha256,
  build_recipe_sha256 = build_recipe_sha256,
  contracts = contracts
)

selected <- candidate_decisions[candidate_decisions$phase8_go, , drop = FALSE]
full_scale <- scale_results[
  scale_results$scale_id == "FULL_CONDITIONAL_LEVELS_1_TO_7",
  , drop = FALSE
]
memory_value <- function(item) cache_memory$memory$bytes[
  cache_memory$memory$item == item
]
summary <- list(
  schema_version = "full-cuda-ci-phase35-feasibility-summary-v1",
  claim_scope = "phase3.5-architecture-feasibility-only",
  run_status = "COMPLETE",
  source_commit = trimws(system2(
    "git", c("rev-parse", "HEAD"), stdout = TRUE
  )),
  source_closure_sha256 = source_closure_sha256,
  native_binary_sha256 = native_sha256,
  producer_identity_sha256 = producer$identity_sha256,
  evidence_manifest_sha256 = evidence_inputs$sha256[
    evidence_inputs$evidence_id == "prepublication-manifest"
  ],
  selected_candidate_id = selected$candidate_id,
  selected_decision = selected$decision,
  phase8_go = TRUE,
  production_backend_promoted = FALSE,
  phase10_promotion_claim = FALSE,
  full_graph_claim = FALSE,
  universal_p_value_parity_claim = FALSE,
  unguarded_exact_values_are_legacy_p_values = FALSE,
  unguarded_exact_role =
    "internal-decision-screen-outside-closed-refinement-guard",
  guard_lower_inclusive = evidence$full_hybrid$guard$lower_inclusive,
  guard_upper_inclusive = evidence$full_hybrid$guard$upper_inclusive,
  alpha = evidence$full_hybrid$guard$alpha,
  qualification_pair_count = scale_results$pair_count[[1L]],
  qualification_screen_flip_count =
    scale_results$screen_decision_flip_count[[1L]],
  qualification_final_flip_count =
    scale_results$final_decision_flip_count[[1L]],
  campaign_slice_pair_count = scale_results$pair_count[[2L]],
  campaign_slice_final_flip_count =
    scale_results$final_decision_flip_count[[2L]],
  full_conditional_pair_count = full_scale$pair_count,
  full_conditional_component_count = full_scale$exact_component_count,
  full_conditional_screen_flip_count =
    full_scale$screen_decision_flip_count,
  full_conditional_guarded_pair_count = full_scale$guarded_pair_count,
  full_conditional_final_flip_count =
    full_scale$final_decision_flip_count,
  maximum_refined_p_value_absolute_error =
    full_scale$maximum_refined_p_value_absolute_error,
  measured_full_conditional_dcov_ms =
    full_scale$dcov_host_boundary_ms,
  conservative_dcov_upper_bound_ms =
    performance$model$dcov_total_upper_bound_ms,
  allocated_dcov_upper_bound_ms =
    performance$model$dcov_allocated_upper_bound_ms,
  full_campaign_used_or_reserved_ms =
    performance$model$full_campaign_used_or_reserved_ms,
  full_campaign_allocated_upper_bound_ms =
    performance$model$full_campaign_allocated_upper_bound_ms,
  reference_component_capacity =
    cache_memory$policy$reference_component_capacity,
  reference_capacity_miss_count = cache_memory$cache_curve$component_misses[
    cache_memory$cache_curve$component_capacity == 47L
  ],
  reference_capacity_eviction_count =
    cache_memory$cache_curve$component_evictions[
      cache_memory$cache_curve$component_capacity == 47L
    ],
  declared_peak_device_bytes = memory_value("declared_capacity_bound"),
  declared_device_headroom_bytes = memory_value("declared_headroom"),
  residual_d2h_bytes = sum(scale_results$residual_d2h_bytes),
  component_d2h_bytes = sum(scale_results$component_d2h_bytes),
  cpu_numerical_dcov_count =
    sum(scale_results$cpu_numerical_dcov_count),
  corpus_policy_pass = all(corpus_policy$pass),
  holdout_state = contracts$promotion_holdout_manifest_v1$payload$state,
  architecture_contract_sha256 =
    contracts$architecture_contract_v1$sha256,
  numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
  artifact_identity_contract_sha256 =
    contracts$artifact_identity_contract_v1$sha256,
  reference_machine_contract_sha256 =
    contracts$reference_machine_v1$sha256,
  performance_budget_contract_sha256 =
    contracts$performance_budget_v1$sha256,
  development_corpus_contract_sha256 =
    contracts$development_qualification_corpus_v1$sha256,
  metamorphic_contract_sha256 = contracts$metamorphic_contract_v1$sha256,
  promotion_holdout_contract_sha256 =
    contracts$promotion_holdout_manifest_v1$sha256,
  pass = TRUE
)

output_parent <- dirname(output_dir)
dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
stage_dir <- tempfile(".phase35-feasibility-stage-", tmpdir = output_parent)
dir.create(stage_dir, recursive = TRUE)
on.exit(unlink(stage_dir, recursive = TRUE, force = TRUE), add = TRUE)

write_table <- function(value, name) {
  write.csv(
    value, file.path(stage_dir, name), row.names = FALSE, na = ""
  )
}
write_table(benchmark_builds, "benchmark_builds.csv")
write_table(measurements$table, "benchmark_measurements.csv")
write_table(cache_memory$cache_curve, "cache.csv")
write_table(candidate_decisions, "candidate_decisions.csv")
write_table(corpus_policy, "corpus_policy.csv")
write_table(cache_memory$dense_level_pressure, "dense_level_pressure.csv")
write_table(evidence_inputs, "evidence_inputs.csv")
write_table(full_pair_decisions, "full_pair_decisions.csv")
write_table(cache_memory$memory, "memory_model.csv")
write_table(performance_tables$bound, "performance_bound.csv")
write_table(performance_tables$budget, "performance_budget.csv")
write_table(performance_tables$classes, "performance_classes.csv")
write_table(cache_memory$reuse_summary, "reuse_distance.csv")
write_table(scale_results, "scale_results.csv")
write_table(source_closure, "source_closure.csv")
fastkpc_full_cuda_write_json(
  summary, file.path(stage_dir, "summary.json")
)
fastkpc_full_cuda_write_json(
  producer, file.path(stage_dir, "producer_identity.json")
)

semantic_payload_files <- c(
  "benchmark_builds.csv", "benchmark_measurements.csv", "cache.csv",
  "candidate_decisions.csv", "corpus_policy.csv",
  "dense_level_pressure.csv", "evidence_inputs.csv",
  "full_pair_decisions.csv", "memory_model.csv", "performance_bound.csv",
  "performance_budget.csv", "performance_classes.csv",
  "producer_identity.json", "reuse_distance.csv", "scale_results.csv",
  "source_closure.csv", "summary.json"
)
payload_file_sha256 <- setNames(lapply(
  file.path(stage_dir, semantic_payload_files),
  fastkpc_full_cuda_census_file_hash
), semantic_payload_files)
payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
  fastkpc_full_cuda_phase35_canonical_json(payload_file_sha256)
)
producer_envelope <- fastkpc_full_cuda_phase35_identity_envelope(
  producer, payload_manifest_sha256
)
timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
environment_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
  schema_version = "full-cuda-ci-phase35-feasibility-environment-v1",
  gpu_query = as.list(gpu_query),
  compute_process_count = 0L,
  runtime_gpu_name = runtime$gpu_name,
  runtime_gpu_uuid = runtime$gpu_uuid,
  R_version = R.version.string,
  native_binary_sha256 = native_sha256
))
validator_paths <- c(
  "fastkpc/R/full_cuda_ci_workload_census.R",
  "fastkpc/R/full_cuda_ci_phase35_contracts.R",
  "fastkpc/R/full_cuda_ci_phase35_feasibility.R"
)
validator_hashes <- setNames(as.list(vapply(
  validator_paths, fastkpc_full_cuda_census_file_hash, character(1L)
)), validator_paths)
validator_closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
  fastkpc_full_cuda_phase35_canonical_json(validator_hashes)
)
attestation <- fastkpc_full_cuda_phase35_validator_attestation(
  producer = producer,
  validator_source_closure_sha256 = validator_closure_sha256,
  validator_semantic_version = "full-cuda-ci-phase35-feasibility-validator-v1",
  validator_contracts = contracts,
  validation_timestamp_utc = timestamp,
  environment_sha256 = environment_sha256,
  validation_result = "PASS"
)
stage_info <- file.info(stage_dir, extra_cols = TRUE)
stage_inode <- if ("ino" %in% names(stage_info)) {
  as.character(stage_info$ino[[1L]])
} else {
  "unavailable"
}
final_path <- file.path(
  normalizePath(output_parent, winslash = "/", mustWork = TRUE),
  basename(output_dir)
)
receipt <- fastkpc_full_cuda_phase35_execution_receipt(
  producer = producer,
  pid = as.integer(Sys.getpid()),
  session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
  cuda_context_id = paste0(runtime$gpu_uuid, ":device-", runtime$device_id),
  artifact_path = final_path,
  artifact_inode = stage_inode,
  staging_path = normalizePath(stage_dir, winslash = "/", mustWork = TRUE),
  recorded_at_utc = timestamp
)
fastkpc_full_cuda_write_json(
  list(attestations = list(attestation)),
  file.path(stage_dir, "validator_attestations.json")
)
fastkpc_full_cuda_write_json(
  list(execution_receipts = list(receipt)),
  file.path(stage_dir, "execution_receipts.json")
)
manifest <- list(
  schema_version = "full-cuda-ci-phase35-feasibility-manifest-v1",
  claim_scope = "phase3.5-architecture-feasibility-only",
  producer_semantic_envelope = producer_envelope,
  payload_manifest_sha256 = payload_manifest_sha256,
  payload_file_sha256 = payload_file_sha256,
  semantic_file_count = length(semantic_payload_files),
  validator_attestations_file = "validator_attestations.json",
  volatile_receipt_file = "execution_receipts.json"
)
fastkpc_full_cuda_write_json(manifest, file.path(stage_dir, "manifest.json"))
writeLines(
  c(
    "# Full CUDA CI Phase 3.5 architecture feasibility",
    "",
    "- result: PASS",
    paste0("- selected architecture: ", summary$selected_candidate_id),
    paste0("- full conditional final decision flips: ",
           summary$full_conditional_final_flip_count),
    paste0("- conservative dCov bound: ",
           format(summary$conservative_dcov_upper_bound_ms / 1000,
                  digits = 6L), " seconds"),
    paste0("- full campaign used or reserved: ",
           format(summary$full_campaign_used_or_reserved_ms / 1000,
                  digits = 6L), " seconds"),
    "- universal legacy p-value parity claim: false",
    "- production backend promotion claim: false",
    "- Phase 10 campaign claim: false"
  ),
  file.path(stage_dir, "summary.md"), useBytes = TRUE
)
writeLines(
  c(
    paste(c(nvcc, benchmark_builds$compile_arguments[[1L]]),
          collapse = " "),
    paste(c(benchmark_specs$eig$binary,
            benchmark_specs$eig$run_args), collapse = " "),
    paste(c(nvcc, benchmark_builds$compile_arguments[[2L]]),
          collapse = " "),
    paste(c(benchmark_specs$block$binary,
            benchmark_specs$block$run_args), collapse = " "),
    paste(
      "Rscript", runner_path,
      paste0("--evidence=", evidence_manifest_path), final_path
    )
  ),
  file.path(stage_dir, "commands.txt"), useBytes = TRUE
)
writeLines(
  c(
    paste0("recorded_at_utc=", timestamp),
    paste0("R_version=", R.version.string),
    paste0("native_binary_sha256=", native_sha256),
    paste0("environment_sha256=", environment_sha256),
    paste0("gpu_query=", paste(gpu_query, collapse = " | ")),
    "compute_process_count_before_benchmarks=0",
    capture.output(sessionInfo())
  ),
  file.path(stage_dir, "environment.txt"), useBytes = TRUE
)

fastkpc_full_cuda_phase35_validate_feasibility_artifact(
  stage_dir, verify_current_sources = TRUE
)
backup_dir <- NULL
if (dir.exists(output_dir)) {
  backup_dir <- tempfile(".phase35-feasibility-backup-", tmpdir = output_parent)
  if (!file.rename(output_dir, backup_dir)) {
    stop("cannot stage prior feasibility artifact for replacement",
         call. = FALSE)
  }
}
published <- file.rename(stage_dir, output_dir)
if (!published) {
  if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
  stop("cannot publish feasibility artifact", call. = FALSE)
}
validated <- tryCatch(
  fastkpc_full_cuda_phase35_validate_feasibility_artifact(
    output_dir, verify_current_sources = TRUE
  ),
  error = identity
)
if (inherits(validated, "error")) {
  unlink(output_dir, recursive = TRUE, force = TRUE)
  if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
  stop("published feasibility artifact failed validation: ",
       conditionMessage(validated), call. = FALSE)
}
if (!is.null(backup_dir)) unlink(backup_dir, recursive = TRUE, force = TRUE)

cat(
  "PASS Phase 3.5 feasibility artifact: ",
  normalizePath(output_dir, winslash = "/", mustWork = TRUE),
  "; producer=", producer$identity_sha256,
  "; dCov_bound_ms=",
  format(summary$conservative_dcov_upper_bound_ms, digits = 10L),
  "; full_reserved_ms=",
  format(summary$full_campaign_used_or_reserved_ms, digits = 10L), "\n",
  sep = ""
)
