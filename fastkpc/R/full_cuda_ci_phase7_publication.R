fastkpc_full_cuda_phase7_publication_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase7_artifact_kinds <- function() {
  c("oracle", "full_shadow", "backend")
}

fastkpc_full_cuda_phase7_artifact_directory_names <- function() {
  c(
    oracle = "native_setup_oracle_v1",
    full_shadow = "native_setup_full_shadow_v1",
    backend = "native_setup_backend_v1"
  )
}

fastkpc_full_cuda_phase7_backend_configuration <- function(kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase7_artifact_kinds())
  value <- list(
    schema_version = "full-cuda-ci-phase7-backend-configuration-v1",
    artifact_kind = kind,
    setup_backend = "native-cpp-mgcv-1.9-1-tprs",
    setup_semantic_version = "mgcv-1.9-1-tprs-native-setup-v1",
    supported_S_sizes = as.list(1:7),
    single_penalty_formula = "target~s(S)",
    joint_two_dimensional_formula = "target~s(S1,S2)",
    additive_formula = "target~s(S1)+...+s(Sk)",
    geometry_backend = "native-lapack-pivoted-qr-mroot-initial-sp",
    single_penalty_residual_backend =
      "cuda-spectral-risk-gated-exact-replay",
    multi_penalty_residual_backend =
      "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual",
    downstream_validation_backend =
      "legacy-dcov-gamma-cpp-spectra-component-cache",
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE,
    native_setup_authoritative = identical(kind, "backend"),
    shadow_comparison = !identical(kind, "backend"),
    skeleton_legacy_mgcv_setup_count = 0L,
    skeleton_legacy_mgcv_fit_count = 0L,
    skeleton_r_callback_count = 0L,
    unsupported_count = 0L
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase7_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase7-build-recipe-v1",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
    native_setup_target = "mgcv-1.9-1-compatible-tprs-subset",
    nvcc_path = "/usr/local/cuda/bin/nvcc",
    nvcc_version = fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    ),
    cuda_architecture = "sm_89",
    fmad = FALSE,
    fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase7_producer <- function(
    kind, catalog, evidence, source_closure, native_identity,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase7_artifact_kinds())
  backend <- fastkpc_full_cuda_phase7_backend_configuration(kind)
  build <- fastkpc_full_cuda_phase7_build_recipe()
  corpus_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase7-canonical-corpus-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    setup_count = 8634L,
    target_count = 110617L,
    logical_test_count = 240489L,
    S_size = 1:7
  ))
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase7-native-setup-", kind, "-v1"
    ),
    dataset_or_corpus_sha256 = corpus_sha256,
    oracle_sha256 =
      evidence$setup_oracle_execution_identity$identity_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(producer = producer, backend = backend, build = build)
}

fastkpc_full_cuda_phase7_artifact_summary <- function(
    kind, evidence, producer_bundle, evidence_sha256, contracts) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase7_artifact_kinds())
  source <- evidence$summary
  list(
    schema_version = "full-cuda-ci-phase7-artifact-summary-v1",
    artifact_kind = kind,
    run_status = "ok",
    timeout = FALSE,
    source_commit = evidence$execution_identity$source_commit,
    oracle_artifact = paste0(
      "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"
    ),
    candidate_route = paste0(
      "native-cpp-setup+phase4-single-penalty-cuda+",
      "phase6-multi-penalty-cuda"
    ),
    setup_count = source$setup_count,
    single_penalty_setup_count = source$single_penalty_setup_count,
    multi_penalty_setup_count = source$multi_penalty_setup_count,
    target_count = source$target_count,
    logical_test_count = source$logical_test_count,
    qualification_oracle_setup_count =
      evidence$setup_corpus$summary$oracle_setup_count,
    skeleton_native_setup_count = source$native_setup_count,
    skeleton_native_setup_unsupported_count =
      source$native_setup_unsupported_count,
    skeleton_legacy_mgcv_setup_count = source$legacy_mgcv_setup_count,
    skeleton_legacy_mgcv_fit_count = source$legacy_mgcv_fit_count,
    skeleton_legacy_mgcv_target_call_count =
      source$legacy_mgcv_target_call_count,
    skeleton_r_callback_count = source$r_callback_count,
    cpu_residual_numerical_solve_count =
      source$cpu_residual_numerical_solve_count,
    residual_numerical_fallback_count =
      source$residual_numerical_fallback_count,
    downstream_legacy_dcov_decision_flip_count =
      source$downstream_legacy_dcov_decision_flip_count,
    downstream_legacy_dcov_backend_error_count =
      source$downstream_legacy_dcov_backend_error_count,
    downstream_legacy_dcov_fallback_count =
      source$downstream_legacy_dcov_fallback_count,
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    backend_fallback_error_count =
      source$residual_numerical_fallback_count +
        source$downstream_legacy_dcov_backend_error_count +
        source$downstream_legacy_dcov_fallback_count,
    maximum_single_penalty_same_sp_fitted_error =
      source$maximum_single_penalty_same_sp_fitted_error,
    maximum_single_penalty_same_sp_residual_error =
      source$maximum_single_penalty_same_sp_residual_error,
    maximum_single_penalty_selected_log_sp_error =
      source$maximum_single_penalty_selected_log_sp_error,
    maximum_multi_penalty_selected_log_sp_error =
      source$maximum_multi_penalty_selected_log_sp_error,
    edge_count_reference = source$edge_count_reference,
    edge_count_candidate = source$edge_count_candidate,
    SHD = source$SHD,
    adjacency_identical = source$adjacency_identical,
    sepsets_identical = source$sepsets_identical,
    n_edgetests_identical = source$n_edgetests_identical,
    deletions_identical = source$deletions_identical,
    native_setup_authoritative = identical(kind, "backend"),
    shadow_comparison = !identical(kind, "backend"),
    setup_corpus_gate = source$setup_corpus_gate,
    selected_fit_gate =
      source$single_penalty_gate && source$multi_penalty_gate,
    graph_gate = source$graph_gate,
    authority_gate = source$authority_gate,
    architecture_contract_sha256 =
      contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    reference_machine_contract_sha256 =
      contracts$reference_machine_v1$sha256,
    performance_budget_contract_sha256 =
      contracts$performance_budget_v1$sha256,
    elapsed_sec = as.numeric(
      evidence$phase4$summary$elapsed_seconds +
        evidence$phase6$parallel_elapsed_seconds
    ),
    source_evidence_sha256 = evidence_sha256,
    producer_identity_sha256 =
      producer_bundle$producer$identity_sha256,
    source_closure_sha256 =
      producer_bundle$producer$producer_source_closure_sha256,
    native_binary_sha256 =
      producer_bundle$producer$native_binary_sha256,
    pass = isTRUE(source$pass)
  )
}

fastkpc_full_cuda_phase7_validate_summary <- function(summary, kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase7_artifact_kinds())
  clean <- is.list(summary) && identical(
    summary$schema_version,
    "full-cuda-ci-phase7-artifact-summary-v1"
  ) && identical(summary$artifact_kind, kind) &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    is.character(summary$source_commit) &&
    grepl("^[0-9a-f]{40}$", summary$source_commit) &&
    identical(
      summary$oracle_artifact,
      "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"
    ) && identical(
      summary$candidate_route,
      paste0(
        "native-cpp-setup+phase4-single-penalty-cuda+",
        "phase6-multi-penalty-cuda"
      )
    ) &&
    summary$setup_count == 8634L &&
    summary$single_penalty_setup_count == 1174L &&
    summary$multi_penalty_setup_count == 7460L &&
    summary$target_count == 110617L &&
    summary$logical_test_count == 240489L &&
    summary$qualification_oracle_setup_count == 8634L &&
    summary$skeleton_native_setup_count == 8634L &&
    summary$skeleton_native_setup_unsupported_count == 0L &&
    summary$skeleton_legacy_mgcv_setup_count == 0L &&
    summary$skeleton_legacy_mgcv_fit_count == 0L &&
    summary$skeleton_legacy_mgcv_target_call_count == 0L &&
    summary$skeleton_r_callback_count == 0L &&
    summary$cpu_residual_numerical_solve_count == 0L &&
    summary$residual_numerical_fallback_count == 0L &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$downstream_legacy_dcov_backend_error_count == 0L &&
    summary$downstream_legacy_dcov_fallback_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$backend_fallback_error_count == 0L &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    identical(
      isTRUE(summary$native_setup_authoritative),
      identical(kind, "backend")
    ) && identical(
      isTRUE(summary$shadow_comparison),
      !identical(kind, "backend")
    ) && isTRUE(summary$setup_corpus_gate) &&
    isTRUE(summary$selected_fit_gate) && isTRUE(summary$graph_gate) &&
    isTRUE(summary$authority_gate) && isTRUE(summary$pass) &&
    is.numeric(summary$elapsed_sec) && length(summary$elapsed_sec) == 1L &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0 &&
    all(vapply(c(
      "source_evidence_sha256", "producer_identity_sha256",
      "source_closure_sha256", "native_binary_sha256",
      "architecture_contract_sha256", "numerical_contract_sha256",
      "artifact_identity_contract_sha256",
      "reference_machine_contract_sha256",
      "performance_budget_contract_sha256"
    ), function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase7_publication_require(
    clean, "Phase 7 artifact summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase7_write_table <- function(value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
  invisible(file.path(directory, name))
}

fastkpc_full_cuda_phase7_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "setup_results.csv",
    "evidence_inputs.csv", "source_closure.csv", "source_evidence.rds",
    "producer_identity.json", "backend_configuration.json",
    "build_recipe.json", "validator_attestations.json",
    "execution_receipts.json"
  )
}

fastkpc_full_cuda_phase7_rank_table <- function(catalog) {
  rows <- fastkpc_full_cuda_phase7_setup_scope(catalog)$setup_rows
  fields <- c(
    "prepared_s_key_sha256", "S_size", "formula_class",
    "model_matrix_nrow", "model_matrix_ncol", "model_matrix_rank",
    "model_matrix_condition", "penalty_count", "constraint_rank",
    "constraint_nullspace_dimension", "conditioning_rank",
    "conditioning_condition", "near_constant_conditioning_count"
  )
  fastkpc_full_cuda_phase7_publication_require(
    all(fields %in% names(rows)),
    "Phase 7 rank-condition source fields are incomplete"
  )
  value <- rows[, fields, drop = FALSE]
  fastkpc_full_cuda_phase7_publication_require(
    nrow(value) == 8634L &&
      !anyDuplicated(value$prepared_s_key_sha256),
    "Phase 7 rank-condition table is incomplete"
  )
  value
}

fastkpc_full_cuda_phase7_case_results <- function(evidence) {
  phase4 <- evidence$phase4$targets
  phase6 <- evidence$phase6$targets
  phase4_fitted <- pmax(
    phase4$fitted_same_sp_max_absolute,
    phase4$fitted_oracle_max_absolute
  )
  phase4_residual <- pmax(
    phase4$residual_same_sp_max_absolute,
    phase4$residual_oracle_max_absolute
  )
  single <- data.frame(
    route = "phase4-single-penalty-cuda",
    prepared_s_key_sha256 = phase4$prepared_s_key_sha256,
    residual_key_sha256 = phase4$residual_key_sha256,
    target = phase4$target,
    penalty_count = 1L,
    selected_log_sp_max_error = abs(phase4$log_sp_error),
    score_absolute_error = NA_real_,
    edf_absolute_error = NA_real_,
    fitted_max_absolute_error = phase4_fitted,
    residual_max_absolute_error = phase4_residual,
    optimizer_mismatch = FALSE,
    rank_mismatch = FALSE,
    all_finite =
      is.finite(phase4$candidate_sp) & is.finite(phase4$score) &
        is.finite(phase4$edf) & is.finite(phase4_fitted) &
        is.finite(phase4_residual),
    stringsAsFactors = FALSE
  )
  multi_optimizer_mismatch <-
    phase6$optimizer_iteration_mismatch |
      phase6$score_call_mismatch | phase6$objective_call_mismatch |
      phase6$step_halving_mismatch | phase6$boundary_probe_mismatch |
      phase6$boundary_status_mismatch | phase6$convergence_mismatch |
      phase6$hessian_state_mismatch
  multi <- data.frame(
    route = "phase6-multi-penalty-cuda",
    prepared_s_key_sha256 = phase6$prepared_s_key_sha256,
    residual_key_sha256 = phase6$residual_key_sha256,
    target = phase6$target,
    penalty_count = phase6$penalty_count,
    selected_log_sp_max_error = phase6$selected_log_sp_max_error,
    score_absolute_error = abs(phase6$score_error),
    edf_absolute_error = abs(phase6$edf_error),
    fitted_max_absolute_error = phase6$fitted_gemm_max_absolute,
    residual_max_absolute_error = phase6$residual_identity_max_absolute,
    optimizer_mismatch = multi_optimizer_mismatch,
    rank_mismatch = phase6$rank_mismatch,
    all_finite = phase6$all_finite,
    stringsAsFactors = FALSE
  )
  value <- rbind(single, multi)
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase7_near_alpha_results <- function(evidence) {
  phase4 <- evidence$phase4$logical_rows
  phase6 <- evidence$phase6$logical_rows
  fields <- intersect(names(phase4), names(phase6))
  select <- function(rows, route) {
    value <- rows[rows$near_alpha, fields, drop = FALSE]
    value <- cbind(route = rep.int(route, nrow(value)), value)
    rownames(value) <- NULL
    value
  }
  value <- rbind(
    select(phase4, "phase4-single-penalty-cuda"),
    select(phase6, "phase6-multi-penalty-cuda")
  )
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase7_stage_timing <- function(evidence) {
  phase4 <- evidence$phase4$timings
  phase6 <- evidence$phase6$timings
  single <- data.frame(
    route = "phase4-single-penalty-cuda",
    prepared_s_key_sha256 = phase4$prepared_s_key_sha256,
    target_count = phase4$target_count,
    logical_test_count = phase4$logical_test_count,
    setup_and_prepare_ms = phase4$native_setup_ms,
    execution_ms = phase4$integrated_host_ms,
    validation_ms = phase4$validation_ms,
    downstream_dcov_ms = phase4$dcov_ms,
    native_setup_count = phase4$native_setup_count,
    native_setup_unsupported_count =
      phase4$native_setup_unsupported_count,
    legacy_mgcv_setup_count = phase4$legacy_mgcv_setup_count,
    legacy_mgcv_target_call_count = phase4$legacy_mgcv_target_calls,
    r_callback_count = phase4$r_callback_count,
    residual_fallback_count = phase4$fallback_count,
    stringsAsFactors = FALSE
  )
  multi <- data.frame(
    route = "phase6-multi-penalty-cuda",
    prepared_s_key_sha256 = phase6$prepared_s_key_sha256,
    target_count = phase6$target_count,
    logical_test_count = phase6$logical_test_count,
    setup_and_prepare_ms = phase6$setup_ms,
    execution_ms = phase6$optimizer_ms,
    validation_ms = phase6$validation_ms,
    downstream_dcov_ms = phase6$dcov_ms,
    native_setup_count = phase6$native_setup_count,
    native_setup_unsupported_count =
      phase6$native_setup_unsupported_count,
    legacy_mgcv_setup_count = phase6$legacy_mgcv_setup_count,
    legacy_mgcv_target_call_count = phase6$legacy_mgcv_target_calls,
    r_callback_count = phase6$r_callback_count,
    residual_fallback_count = phase6$fallback_count,
    stringsAsFactors = FALSE
  )
  value <- rbind(single, multi)
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase7_hash_runs <- function(
    hashes, stage, route) {
  values <- as.character(unlist(hashes, use.names = FALSE))
  files <- names(hashes)
  if (is.null(files)) files <- as.character(seq_along(values))
  data.frame(
    stage = stage,
    route = route,
    evidence_file = basename(files),
    sha256 = values,
    pass = TRUE,
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase7_raw_runs <- function(evidence) {
  value <- rbind(
    fastkpc_full_cuda_phase7_hash_runs(
      evidence$setup_corpus$shard_file_sha256,
      "setup-oracle", "native-setup-oracle-shard"
    ),
    fastkpc_full_cuda_phase7_hash_runs(
      evidence$phase4$partition_file_sha256,
      "single-penalty", "native-setup-phase4-partition"
    ),
    fastkpc_full_cuda_phase7_hash_runs(
      evidence$phase6$partition_file_sha256,
      "multi-penalty", "native-setup-phase6-partition"
    )
  )
  value <- cbind(run = seq_len(nrow(value)), value)
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase7_evidence_inputs <- function(evidence) {
  scalar <- data.frame(
    input_kind = c("setup_corpus", "phase5_oracle"),
    input_file = c("source_evidence.rds", "source_evidence.rds"),
    sha256 = c(
      evidence$evidence_file_sha256$setup_corpus,
      evidence$evidence_file_sha256$phase5
    ),
    stringsAsFactors = FALSE
  )
  partitions <- rbind(
    fastkpc_full_cuda_phase7_hash_runs(
      evidence$evidence_file_sha256$phase4_partitions,
      "phase4_partition", "phase4"
    )[, c("stage", "evidence_file", "sha256")],
    fastkpc_full_cuda_phase7_hash_runs(
      evidence$evidence_file_sha256$phase6_partitions,
      "phase6_partition", "phase6"
    )[, c("stage", "evidence_file", "sha256")]
  )
  names(partitions)[1:2] <- c("input_kind", "input_file")
  value <- rbind(scalar, partitions)
  rownames(value) <- NULL
  value
}

fastkpc_full_cuda_phase7_cache_table <- function(evidence) {
  setup_misses <- c(
    evidence$summary$setup_count,
    evidence$summary$single_penalty_setup_count,
    evidence$summary$multi_penalty_setup_count
  )
  requests <- c(
    evidence$summary$target_count,
    evidence$summary$single_penalty_target_count,
    evidence$summary$multi_penalty_target_count
  )
  data.frame(
    cache = c(
      "native-setup-target-reuse",
      "single-penalty-prepared-device",
      "multi-penalty-prepared-device"
    ),
    ownership = c(
      "phase7-native-cpp-setup", "phase4-persistent-runtime",
      "phase6-persistent-runtime"
    ),
    requests = requests,
    hits = requests - setup_misses,
    misses = setup_misses,
    evictions = 0L,
    semantic_eviction_effect = "none",
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase7_write_standard_payload <- function(
    directory, catalog, evidence, summary) {
  comparison <- evidence$phase6$mixed_graph$comparison
  fastkpc_full_cuda_phase7_write_table(
    comparison$graph_agreement, directory, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    comparison$sepset_agreement, directory, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    comparison$n_edgetests, directory, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    comparison$candidate_deletions, directory, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence,
    file.path(directory, "first_divergence.json")
  )
  fallbacks <- data.frame(
    fallback_class = c(
      "unknown", "approximate", "unsupported-native-setup",
      "legacy-mgcv-in-skeleton", "residual-numerical"
    ),
    supported_scope = "none",
    count = 0L,
    accepted_for_phase7 = FALSE,
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase7_write_table(
    fallbacks, directory, "fallbacks.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_stage_timing(evidence),
    directory, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_raw_runs(evidence),
    directory, "raw_runs.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_case_results(evidence),
    directory, "case_results.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_near_alpha_results(evidence),
    directory, "near_alpha_results.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_rank_table(catalog),
    directory, "rank_condition_results.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_cache_table(evidence),
    directory, "cache.csv"
  )
  fastkpc_full_cuda_phase7_write_table(
    fastkpc_full_cuda_phase7_evidence_inputs(evidence),
    directory, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_write_summary_md(
    summary, file.path(directory, "summary.md"),
    paste0("Full CUDA CI Phase 7 ", summary$artifact_kind)
  )
  invisible(directory)
}

fastkpc_full_cuda_phase7_validate_standard_payload <- function(
    artifact_dir, evidence) {
  read_table <- function(name) {
    utils::read.csv(
      file.path(artifact_dir, name),
      stringsAsFactors = FALSE, check.names = FALSE
    )
  }
  comparison <- evidence$phase6$mixed_graph$comparison
  graph <- read_table("graph_agreement.csv")
  sepsets <- read_table("sepset_agreement.csv")
  n_edgetests <- read_table("n_edgetests.csv")
  deletions <- read_table("deletion_trace.csv")
  first <- jsonlite::read_json(
    file.path(artifact_dir, "first_divergence.json"),
    simplifyVector = TRUE
  )
  fallbacks <- read_table("fallbacks.csv")
  stage_timing <- read_table("stage_timing.csv")
  raw_runs <- read_table("raw_runs.csv")
  case_results <- read_table("case_results.csv")
  near_alpha <- read_table("near_alpha_results.csv")
  rank_condition <- read_table("rank_condition_results.csv")
  cache <- read_table("cache.csv")
  setup_results <- read_table("setup_results.csv")
  evidence_inputs <- read_table("evidence_inputs.csv")
  exact_n_edgetests <- c(2213L, 52659L, 125293L, 40694L,
                         13293L, 5422L, 835L, 80L)
  graph_gate <- nrow(graph) == 1L &&
    graph$edge_count_reference[[1L]] == 110L &&
    graph$edge_count_candidate[[1L]] == 110L &&
    graph$SHD[[1L]] == 0L &&
    isTRUE(as.logical(graph$adjacency_identical[[1L]])) &&
    nrow(sepsets) == nrow(comparison$sepset_agreement) &&
    nrow(sepsets) > 0L && all(as.logical(sepsets$identical)) &&
    nrow(n_edgetests) == 8L &&
    identical(as.integer(n_edgetests$reference), exact_n_edgetests) &&
    identical(as.integer(n_edgetests$candidate), exact_n_edgetests) &&
    all(as.logical(n_edgetests$identical)) &&
    nrow(deletions) == nrow(comparison$candidate_deletions) &&
    identical(
      as.integer(deletions$canonical_deletion_id),
      as.integer(comparison$candidate_deletions$canonical_deletion_id)
    ) && !isTRUE(first$first_divergence_found) &&
    nrow(fallbacks) == 5L && sum(fallbacks$count) == 0L &&
    !any(as.logical(fallbacks$accepted_for_phase7))
  fastkpc_full_cuda_phase7_publication_require(
    graph_gate, "Phase 7 standard graph payload is malformed"
  )

  expected_routes <- c(
    "phase4-single-penalty-cuda", "phase6-multi-penalty-cuda"
  )
  stage_routes <- table(stage_timing$route)
  timing_fields <- c(
    "setup_and_prepare_ms", "execution_ms", "validation_ms",
    "downstream_dcov_ms"
  )
  stage_gate <- nrow(stage_timing) == 8634L &&
    !anyDuplicated(stage_timing$prepared_s_key_sha256) &&
    identical(names(stage_routes), expected_routes) &&
    identical(as.integer(stage_routes), c(1174L, 7460L)) &&
    sum(stage_timing$target_count) == 110617L &&
    sum(stage_timing$logical_test_count) == 238276L &&
    sum(stage_timing$native_setup_count) == 8634L &&
    sum(stage_timing$native_setup_unsupported_count) == 0L &&
    sum(stage_timing$legacy_mgcv_setup_count) == 0L &&
    sum(stage_timing$legacy_mgcv_target_call_count) == 0L &&
    sum(stage_timing$r_callback_count) == 0L &&
    sum(stage_timing$residual_fallback_count) == 0L &&
    all(vapply(timing_fields, function(field) {
      all(is.finite(stage_timing[[field]])) &&
        all(stage_timing[[field]] >= 0)
    }, logical(1L)))
  fastkpc_full_cuda_phase7_publication_require(
    stage_gate, "Phase 7 standard stage timing payload is malformed"
  )

  case_routes <- table(case_results$route)
  single <- case_results$route == "phase4-single-penalty-cuda"
  multi <- case_results$route == "phase6-multi-penalty-cuda"
  case_gate <- nrow(case_results) == 110617L &&
    !anyDuplicated(case_results$residual_key_sha256) &&
    identical(names(case_routes), expected_routes) &&
    identical(as.integer(case_routes), c(44941L, 65676L)) &&
    all(case_results$penalty_count[single] == 1L) &&
    all(case_results$penalty_count[multi] >= 3L) &&
    all(is.finite(case_results$selected_log_sp_max_error)) &&
    max(case_results$selected_log_sp_max_error) <= 1e-6 &&
    all(is.na(case_results$score_absolute_error[single])) &&
    all(is.na(case_results$edf_absolute_error[single])) &&
    all(is.finite(case_results$score_absolute_error[multi])) &&
    all(is.finite(case_results$edf_absolute_error[multi])) &&
    max(case_results$fitted_max_absolute_error) <= 1e-7 &&
    max(case_results$residual_max_absolute_error) <= 1e-7 &&
    !any(as.logical(case_results$optimizer_mismatch)) &&
    !any(as.logical(case_results$rank_mismatch)) &&
    all(as.logical(case_results$all_finite))
  fastkpc_full_cuda_phase7_publication_require(
    case_gate, "Phase 7 standard case-results payload is malformed"
  )

  expected_near_alpha <-
    sum(evidence$phase4$logical_rows$near_alpha) +
      sum(evidence$phase6$logical_rows$near_alpha)
  setup_error_fields <- grep(
    "error$", names(setup_results), value = TRUE
  )
  supplemental_gate <- nrow(near_alpha) == expected_near_alpha &&
    nrow(near_alpha) > 0L && all(as.logical(near_alpha$near_alpha)) &&
    !any(as.logical(near_alpha$decision_flip)) &&
    nrow(rank_condition) == 8634L &&
    !anyDuplicated(rank_condition$prepared_s_key_sha256) &&
    identical(
      as.integer(table(rank_condition$S_size)),
      c(48L, 1126L, 4064L, 2152L, 955L, 245L, 44L)
    ) && all(rank_condition$model_matrix_rank > 0L) &&
    all(rank_condition$conditioning_rank > 0L) &&
    nrow(setup_results) == 8634L &&
    !anyDuplicated(setup_results$prepared_s_key_sha256) &&
    length(setup_error_fields) > 0L &&
    all(vapply(setup_error_fields, function(field) {
      all(is.finite(setup_results[[field]])) &&
        max(setup_results[[field]]) == 0
    }, logical(1L)))
  fastkpc_full_cuda_phase7_publication_require(
    supplemental_gate,
    "Phase 7 standard numerical supplemental payload is malformed"
  )

  cache_gate <- nrow(cache) == 3L &&
    identical(as.numeric(cache$requests), c(110617, 44941, 65676)) &&
    identical(as.numeric(cache$misses), c(8634, 1174, 7460)) &&
    identical(cache$hits + cache$misses, cache$requests) &&
    sum(cache$evictions) == 0L &&
    all(cache$semantic_eviction_effect == "none")
  run_gate <- nrow(raw_runs) == 96L &&
    identical(
      as.integer(table(raw_runs$stage)[c(
        "setup-oracle", "single-penalty", "multi-penalty"
      )]),
      c(64L, 16L, 16L)
    ) && all(as.logical(raw_runs$pass)) &&
    all(grepl("^[0-9a-f]{64}$", raw_runs$sha256)) &&
    nrow(evidence_inputs) == 34L &&
    all(grepl("^[0-9a-f]{64}$", evidence_inputs$sha256)) &&
    file.info(file.path(artifact_dir, "summary.md"))$size[[1L]] > 0L
  fastkpc_full_cuda_phase7_publication_require(
    cache_gate && run_gate,
    "Phase 7 standard cache or raw-run payload is malformed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase7_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase7_validate_artifact <- function(
    artifact_dir, expected_kind = NULL, catalog = NULL,
    verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase7_publication_require(
    is.list(manifest) &&
      identical(names(manifest), fastkpc_full_cuda_phase7_manifest_fields()) &&
      identical(
        manifest$schema_version,
        "full-cuda-ci-phase7-native-setup-artifact-manifest-v1"
      ) && identical(
        manifest$claim_scope,
        "phase7-native-setup-complete-canonical-envelope"
      ),
    "Phase 7 artifact manifest schema mismatch"
  )
  kind <- as.character(manifest$artifact_kind)
  fastkpc_full_cuda_phase7_publication_require(
    kind %in% fastkpc_full_cuda_phase7_artifact_kinds() &&
      (is.null(expected_kind) || identical(kind, expected_kind)),
    "Phase 7 artifact kind mismatch"
  )
  required_files <- fastkpc_full_cuda_phase7_required_files()
  fastkpc_full_cuda_phase7_publication_require(
    all(file.exists(file.path(artifact_dir, required_files))) &&
      identical(
        sort(list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
             method = "radix"),
        sort(required_files, method = "radix")
      ),
    "Phase 7 artifact standard file set is incomplete"
  )
  payload_hashes <- manifest$payload_file_sha256
  expected_semantic_files <- sort(setdiff(
    required_files,
    c(
      "manifest.json", "validator_attestations.json",
      "execution_receipts.json", "environment.txt", "commands.txt"
    )
  ), method = "radix")
  fastkpc_full_cuda_phase7_publication_require(
    is.list(payload_hashes) && length(payload_hashes) > 0L &&
      !is.null(names(payload_hashes)) && !anyDuplicated(names(payload_hashes)) &&
      identical(sort(names(payload_hashes), method = "radix"),
                expected_semantic_files),
    "Phase 7 payload manifest is malformed"
  )
  actual_hashes <- setNames(lapply(names(payload_hashes), function(name) {
    path <- file.path(artifact_dir, name)
    fastkpc_full_cuda_phase7_publication_require(
      file.exists(path) && !dir.exists(path),
      paste0("Phase 7 payload file is missing: ", name)
    )
    fastkpc_full_cuda_census_file_hash(path)
  }), names(payload_hashes))
  mismatch <- names(payload_hashes)[vapply(names(payload_hashes), function(name) {
    !identical(payload_hashes[[name]], actual_hashes[[name]])
  }, logical(1L))]
  fastkpc_full_cuda_phase7_publication_require(
    length(mismatch) == 0L,
    paste0(
      "Phase 7 payload file hash mismatch: ",
      if (length(mismatch) == 0L) "NONE" else mismatch[[1L]]
    )
  )
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase7_publication_require(
    identical(payload_manifest_sha256, manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(actual_hashes),
    "Phase 7 payload manifest hash mismatch"
  )
  envelope <- manifest$producer_semantic_envelope
  fastkpc_full_cuda_phase35_validate_identity_envelope(envelope)
  fastkpc_full_cuda_phase7_publication_require(
    identical(envelope$payload_manifest_sha256, payload_manifest_sha256),
    "Phase 7 producer envelope payload mismatch"
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"),
    simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase7_publication_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(envelope$producer)
    ),
    "Phase 7 producer identity file does not match the envelope"
  )
  source_closure <- utils::read.csv(
    file.path(artifact_dir, "source_closure.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  closure_hashes <- setNames(
    as.list(as.character(source_closure$sha256)),
    as.character(source_closure$path)
  )
  closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
  )
  fastkpc_full_cuda_phase7_publication_require(
    nrow(source_closure) > 0L && !anyDuplicated(source_closure$path) &&
      identical(
        closure_sha256, producer$producer_source_closure_sha256
      ),
    "Phase 7 producer source closure identity mismatch"
  )
  if (isTRUE(verify_current_sources)) {
    current_hashes <- setNames(as.list(vapply(
      source_closure$path,
      fastkpc_full_cuda_census_file_hash,
      character(1L)
    )), source_closure$path)
    fastkpc_full_cuda_phase7_publication_require(
      identical(
        fastkpc_full_cuda_phase35_canonical_json(current_hashes),
        fastkpc_full_cuda_phase35_canonical_json(closure_hashes)
      ),
      "Phase 7 artifact source closure is stale"
    )
  }
  backend <- jsonlite::read_json(
    file.path(artifact_dir, "backend_configuration.json"),
    simplifyVector = FALSE
  )
  build <- jsonlite::read_json(
    file.path(artifact_dir, "build_recipe.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase7_publication_require(
    identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(backend)
      ),
      producer$backend_configuration_sha256
    ) && identical(
      fastkpc_full_cuda_phase35_sha256_utf8(
        fastkpc_full_cuda_phase35_canonical_json(build)
      ),
      producer$build_recipe_sha256
    ),
    "Phase 7 backend or build identity mismatch"
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(artifact_dir, manifest$environment_file)
  )
  fastkpc_full_cuda_phase7_publication_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 7 environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  fastkpc_full_cuda_phase7_publication_require(
    is.list(attestations) && length(attestations) > 0L,
    "Phase 7 validator attestation is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase7_publication_require(
      identical(
        attestation$attested_producer_sha256, producer$identity_sha256
      ) && identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 7 validator attestation does not attest this producer"
    )
  }
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase7_publication_require(
    is.list(receipts) && length(receipts) > 0L,
    "Phase 7 execution receipt is missing"
  )
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase7_publication_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 7 execution receipt producer mismatch"
    )
  }
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase7_validate_summary(summary, kind)
  fastkpc_full_cuda_phase7_publication_require(
    identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(
        summary$source_closure_sha256,
        producer$producer_source_closure_sha256
      ) && identical(
        summary$native_binary_sha256, producer$native_binary_sha256
      ) && identical(
        summary$source_evidence_sha256,
        actual_hashes$source_evidence.rds
      ) && identical(
        summary$architecture_contract_sha256,
        producer$contract_snapshots$architecture_contract_v1$sha256
      ) && identical(
        summary$numerical_contract_sha256,
        producer$contract_snapshots$numerical_contract_v1$sha256
      ) && identical(
        summary$artifact_identity_contract_sha256,
        producer$contract_snapshots$artifact_identity_contract_v1$sha256
      ) && identical(
        summary$reference_machine_contract_sha256,
        producer$contract_snapshots$reference_machine_v1$sha256
      ) && identical(
        summary$performance_budget_contract_sha256,
        producer$contract_snapshots$performance_budget_v1$sha256
      ),
    "Phase 7 summary identity mismatch"
  )
  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  fastkpc_full_cuda_phase7_publication_require(
    identical(summary$source_commit, evidence$execution_identity$source_commit),
    "Phase 7 summary source commit mismatch"
  )
  fastkpc_full_cuda_phase7_validate_standard_payload(
    artifact_dir, evidence
  )
  if (!is.null(catalog)) {
    fastkpc_full_cuda_phase7_validate_full_evidence(
      evidence, catalog,
      verify_current_identity = isTRUE(verify_current_sources)
    )
  }
  list(
    kind = kind, manifest = manifest, summary = summary,
    producer = producer, source_closure = source_closure,
    evidence = evidence
  )
}

fastkpc_full_cuda_phase7_publish_one <- function(
    kind, output_dir, catalog, evidence, evidence_path,
    evidence_sha256, contracts, source_closure, native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase7_artifact_kinds())
  producer_bundle <- fastkpc_full_cuda_phase7_producer(
    kind, catalog, evidence, source_closure, native_identity, contracts
  )
  output_parent <- dirname(output_dir)
  dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
  stage_dir <- tempfile(
    paste0(".phase7-", kind, "-stage-"), tmpdir = output_parent
  )
  dir.create(stage_dir, recursive = TRUE)
  stage_active <- TRUE
  on.exit({
    if (stage_active && dir.exists(stage_dir)) {
      unlink(stage_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  summary <- fastkpc_full_cuda_phase7_artifact_summary(
    kind, evidence, producer_bundle, evidence_sha256, contracts
  )
  fastkpc_full_cuda_phase7_validate_summary(summary, kind)
  fastkpc_full_cuda_phase7_publication_require(
    file.copy(
      evidence_path, file.path(stage_dir, "source_evidence.rds"),
      overwrite = TRUE, copy.mode = FALSE, copy.date = FALSE
    ),
    "Phase 7 source evidence copy failed"
  )
  fastkpc_full_cuda_phase7_write_table(
    evidence$setup_corpus$rows, stage_dir, "setup_results.csv"
  )
  fastkpc_full_cuda_phase7_write_standard_payload(
    stage_dir, catalog, evidence, summary
  )
  fastkpc_full_cuda_phase7_write_table(
    source_closure$table, stage_dir, "source_closure.csv"
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage_dir, "summary.json"))
  fastkpc_full_cuda_write_json(
    producer_bundle$producer,
    file.path(stage_dir, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$backend$value,
    file.path(stage_dir, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$build$value, file.path(stage_dir, "build_recipe.json")
  )
  writeLines(
    c(
      paste0("source evidence SHA-256: ", evidence_sha256),
      paste0("producer source closure SHA-256: ", source_closure$sha256),
      paste0("native binary SHA-256: ", native_identity$sha256),
      "bash fastkpc/tools/run_full_cuda_ci_native_setup_gate.sh",
      "Rscript fastkpc/tools/run_full_cuda_ci_native_setup_artifacts.R"
    ),
    file.path(stage_dir, "commands.txt"), useBytes = TRUE
  )
  environment_lines <- c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase7_setup_execution=native-cpp",
    "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"
  )
  writeLines(
    environment_lines, file.path(stage_dir, "environment.txt"),
    useBytes = TRUE
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage_dir, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage_dir, semantic_files),
    fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer_bundle$producer, payload_manifest_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage_dir, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer_bundle$producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase7-validator-v1",
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
    producer = producer_bundle$producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-devices-0-1-phase7-native-setup",
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
    schema_version =
      "full-cuda-ci-phase7-native-setup-artifact-manifest-v1",
    artifact_kind = kind,
    claim_scope = "phase7-native-setup-complete-canonical-envelope",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_manifest_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage_dir, "manifest.json"))
  fastkpc_full_cuda_phase7_validate_artifact(
    stage_dir, expected_kind = kind, catalog = catalog,
    verify_current_sources = TRUE
  )
  backup_dir <- NULL
  if (dir.exists(output_dir)) {
    backup_dir <- tempfile(
      paste0(".phase7-", kind, "-backup-"), tmpdir = output_parent
    )
    fastkpc_full_cuda_phase7_publication_require(
      file.rename(output_dir, backup_dir),
      "Phase 7 prior artifact could not be staged for replacement"
    )
  }
  if (!file.rename(stage_dir, output_dir)) {
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop("Phase 7 artifact publication failed", call. = FALSE)
  }
  stage_active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase7_validate_artifact(
      output_dir, expected_kind = kind, catalog = catalog,
      verify_current_sources = TRUE
    ),
    error = identity
  )
  if (inherits(validated, "error")) {
    failed_dir <- tempfile(
      paste0(".phase7-", kind, "-failed-"), tmpdir = output_parent
    )
    file.rename(output_dir, failed_dir)
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup_dir)) unlink(backup_dir, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase7_publish_artifacts <- function(
    catalog, evidence_path,
    output_root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  evidence_path <- normalizePath(
    evidence_path, winslash = "/", mustWork = TRUE
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase7_validate_full_evidence(
    evidence, catalog, verify_current_identity = TRUE
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase7_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(evidence_path)
  directories <- fastkpc_full_cuda_phase7_artifact_directory_names()
  results <- lapply(
    fastkpc_full_cuda_phase7_artifact_kinds(), function(kind) {
      fastkpc_full_cuda_phase7_publish_one(
        kind = kind,
        output_dir = file.path(output_root, directories[[kind]]),
        catalog = catalog, evidence = evidence,
        evidence_path = evidence_path,
        evidence_sha256 = evidence_sha256,
        contracts = contracts, source_closure = source_closure,
        native_identity = native_identity
      )
    }
  )
  names(results) <- fastkpc_full_cuda_phase7_artifact_kinds()
  results
}
