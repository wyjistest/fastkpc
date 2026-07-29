fastkpc_full_cuda_phase5_publication_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase5_artifact_kinds <- function() {
  c("oracle", "full_shadow", "backend")
}

fastkpc_full_cuda_phase5_artifact_directory_names <- function() {
  c(
    oracle = "multi_penalty_cpp_oracle_v1",
    full_shadow = "multi_penalty_cpp_full_shadow_v1",
    backend = "multi_penalty_cpp_backend_v1"
  )
}

fastkpc_full_cuda_phase5_publication_input_identity <- function(catalog) {
  files <- c(
    canonical_data = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    phase0_manifest = file.path(catalog$phase0_dir, "manifest.json"),
    phase1_manifest = file.path(catalog$phase1_dir, "manifest.json"),
    phase2_manifest = file.path(catalog$phase2_dir, "manifest.json"),
    inherited_phase4_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1", "manifest.json"
    ),
    inherited_phase4_logical_results = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1", "logical_ci_results.rds"
    )
  )
  hashes <- setNames(as.list(vapply(
    files, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(files))
  dataset <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase5-canonical-corpus-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    input_file_sha256 = hashes$canonical_data,
    phase1_manifest_sha256 = hashes$phase1_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    setup_count = 7460L,
    target_count = 65676L,
    logical_test_count = 60324L,
    S_size = 3:7
  ))
  oracle <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase5-oracle-authority-v1",
    phase0_manifest_sha256 = hashes$phase0_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    inherited_phase4_manifest_sha256 =
      hashes$inherited_phase4_manifest,
    inherited_phase4_logical_results_sha256 =
      hashes$inherited_phase4_logical_results,
    mgcv_semantics_version = "mgcv-gam-gcv-cp-v1",
    allowed_residual_decision_flip_count = 0L,
    allowed_dcov_decision_flip_count = 0L,
    required_SHD = 0L
  ))
  list(files = files, hashes = hashes, dataset_sha256 = dataset,
       oracle_sha256 = oracle)
}

fastkpc_full_cuda_phase5_backend_configuration <- function(kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase5_artifact_kinds())
  value <- list(
    schema_version = "full-cuda-ci-phase5-backend-configuration-v1",
    artifact_kind = kind,
    formula_route = "target~s(S1)+...+s(Sk)",
    family = "gaussian",
    link = "identity",
    sp_selection_backend = "cpp-magic-multi-penalty",
    gcv_score_backend = "cpp-analytic-magic-gH",
    residual_backend =
      "cpp-pivoted-qr-augmented-lapack-dgesdd-svd",
    penalty_root_backend = "lapack-dpstrf-aggregate-penalty",
    rank_tolerance = "sqrt(.Machine$double.eps)",
    convergence_tolerance = "1e-7",
    max_step_halving = 25L,
    max_iterations = 400L,
    max_newton_component = "5",
    boundary_probe_step = "2",
    max_boundary_probes = 5L,
    downstream_validation_backend =
      "legacy-dcov-gamma-cpp-spectra-component-cache",
    precision = "float64",
    normal_equations_used = FALSE,
    candidate_legacy_mgcv_target_calls = 0L,
    fallback_count = 0L,
    shadow_only = TRUE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase5_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase5-build-recipe-v1",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    eigen_source = "RcppEigen",
    penalty_root_lapack = "dpstrf",
    optimizer_hessian_eigensolver = "Eigen-SelfAdjointEigenSolver"
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase5_producer <- function(
    kind, catalog, source_closure, native_identity,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  input <- fastkpc_full_cuda_phase5_publication_input_identity(catalog)
  backend <- fastkpc_full_cuda_phase5_backend_configuration(kind)
  build <- fastkpc_full_cuda_phase5_build_recipe()
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase5-multi-penalty-cpp-", kind, "-v1"
    ),
    dataset_or_corpus_sha256 = input$dataset_sha256,
    oracle_sha256 = input$oracle_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(
    producer = producer, input = input, backend = backend, build = build
  )
}

fastkpc_full_cuda_phase5_validate_merged_evidence <- function(
    evidence, catalog, verify_current_identity = TRUE) {
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  summary <- evidence$summary
  graph <- evidence$mixed_graph$summary
  clean <- is.list(evidence) && identical(
    evidence$schema_version,
    "full-cuda-ci-multi-penalty-cpp-full-shadow-merged-v1"
  ) && is.list(summary) && isTRUE(summary$pass) &&
    summary$setup_count == 7460L && summary$target_count == 65676L &&
    summary$logical_test_count == 60324L &&
    summary$penalty_count_min == 3L && summary$penalty_count_max == 7L &&
    summary$optimizer_iteration_mismatch_count == 0L &&
    summary$score_call_mismatch_count == 0L &&
    summary$convergence_mismatch_count == 0L &&
    summary$hessian_state_mismatch_count == 0L &&
    summary$rank_mismatch_count == 0L && summary$fallback_count == 0L &&
    summary$candidate_legacy_mgcv_target_calls == 0L &&
    summary$validation_legacy_mgcv_fixed_sp_calls == 65676L &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$near_alpha_decision_flip_count == 0L &&
    isTRUE(summary$stable_rank_path_gate) &&
    isTRUE(summary$numerical_gate) && isTRUE(summary$optimizer_gate) &&
    isTRUE(summary$downstream_decision_gate) &&
    isTRUE(summary$backend_gate) && isTRUE(graph$pass) &&
    graph$direct_legacy_logical_test_count == 2213L &&
    graph$phase4_cuda_logical_test_count == 177952L &&
    graph$phase5_cpp_logical_test_count == 60324L &&
    graph$explicit_legacy_fallback_count == 0L &&
    graph$unknown_fallback_count == 0L &&
    graph$edge_count_reference == 110L &&
    graph$edge_count_candidate == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) && isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical) &&
    is.data.frame(evidence$targets) && nrow(evidence$targets) == 65676L &&
    is.data.frame(evidence$logical_rows) &&
    nrow(evidence$logical_rows) == 60324L &&
    is.data.frame(evidence$timings) && nrow(evidence$timings) == 7460L &&
    !anyDuplicated(evidence$targets$residual_key_sha256) &&
    !anyDuplicated(evidence$logical_rows$logical_sequence_id) &&
    identical(
      evidence$targets$residual_key_sha256,
      scope$target_rows$residual_key_sha256
    ) && !any(evidence$logical_rows$decision_flip) &&
    !any(evidence$logical_rows$backend_error) &&
    !any(evidence$logical_rows$spectra_fallback) &&
    all(evidence$targets$fallback_reason == "NONE") &&
    all(evidence$targets$rank_path ==
          "pivoted-qr-augmented-lapack-dgesdd-svd") &&
    !any(evidence$targets$normal_equations_used) &&
    all(evidence$targets$all_finite)
  fastkpc_full_cuda_phase5_publication_require(
    clean, "Phase 5 merged evidence gate failed"
  )
  if (isTRUE(verify_current_identity)) {
    current <- fastkpc_full_cuda_phase5_execution_identity(catalog)
    fastkpc_full_cuda_phase5_publication_require(
      identical(
        fastkpc_full_cuda_phase35_canonical_json(current),
        fastkpc_full_cuda_phase35_canonical_json(
          evidence$execution_identity
        )
      ),
      "Phase 5 merged evidence does not match current producer identity"
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase5_common_summary <- function(
    kind, evidence, producer_bundle, contracts, source_closure,
    native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase5_artifact_kinds())
  source <- evidence$summary
  graph <- evidence$mixed_graph$summary
  summary <- list(
    schema_version = paste0(
      "full-cuda-ci-phase5-multi-penalty-cpp-", kind, "-summary-v1"
    ),
    artifact_kind = kind,
    claim_scope = "phase5-additive-multi-penalty-S-size-greater-than-2",
    run_status = "ok",
    timeout = FALSE,
    source_commit = fastkpc_full_cuda_source_commit(),
    source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    producer_identity_sha256 =
      producer_bundle$producer$identity_sha256,
    oracle_artifact =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
    inherited_phase4_artifact = paste0(
      "fastkpc/artifacts/full_cuda_ci/",
      "single_penalty_cuda_gcv_full_shadow_v1"
    ),
    candidate_route = paste0(
      "cpp-magic-multi-penalty+",
      "cpp-pivoted-qr-augmented-lapack-dgesdd-svd"
    ),
    edge_count_reference = as.integer(graph$edge_count_reference),
    edge_count_candidate = as.integer(graph$edge_count_candidate),
    SHD = as.integer(graph$SHD),
    adjacency_identical = isTRUE(graph$adjacency_identical),
    sepsets_identical = isTRUE(graph$sepsets_identical),
    n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
    deletions_identical = isTRUE(graph$deletions_identical),
    direct_legacy_logical_test_count =
      as.integer(graph$direct_legacy_logical_test_count),
    phase4_cuda_logical_test_count =
      as.integer(graph$phase4_cuda_logical_test_count),
    phase5_cpp_logical_test_count =
      as.integer(graph$phase5_cpp_logical_test_count),
    explicit_legacy_fallback_count =
      as.integer(graph$explicit_legacy_fallback_count),
    unknown_fallback_count = as.integer(graph$unknown_fallback_count),
    approximate_backend_count = 0L,
    multi_penalty_setup_count = as.integer(source$setup_count),
    multi_penalty_target_count = as.integer(source$target_count),
    penalty_count_min = as.integer(source$penalty_count_min),
    penalty_count_max = as.integer(source$penalty_count_max),
    candidate_legacy_mgcv_target_calls =
      as.integer(source$candidate_legacy_mgcv_target_calls),
    validation_legacy_mgcv_fixed_sp_calls =
      as.integer(source$validation_legacy_mgcv_fixed_sp_calls),
    fallback_count = as.integer(source$fallback_count),
    max_selected_log_sp_error =
      as.numeric(source$max_selected_log_sp_error),
    max_score_absolute_error =
      as.numeric(source$max_score_absolute_error),
    max_edf_absolute_error = as.numeric(source$max_edf_absolute_error),
    max_fitted_absolute_error =
      as.numeric(source$max_fitted_absolute_error),
    max_fitted_relative_l2 = as.numeric(source$max_fitted_relative_l2),
    max_residual_absolute_error =
      as.numeric(source$max_residual_absolute_error),
    max_residual_relative_l2 =
      as.numeric(source$max_residual_relative_l2),
    max_absolute_p_value_difference =
      as.numeric(source$max_absolute_p_value_difference),
    optimizer_iteration_mismatch_count =
      as.integer(source$optimizer_iteration_mismatch_count),
    score_call_mismatch_count =
      as.integer(source$score_call_mismatch_count),
    convergence_mismatch_count =
      as.integer(source$convergence_mismatch_count),
    hessian_state_mismatch_count =
      as.integer(source$hessian_state_mismatch_count),
    rank_mismatch_count = as.integer(source$rank_mismatch_count),
    downstream_legacy_dcov_decision_flip_count = as.integer(
      source$downstream_legacy_dcov_decision_flip_count
    ),
    near_alpha_decision_flip_count =
      as.integer(source$near_alpha_decision_flip_count),
    transcript_preserved_count =
      as.integer(source$transcript_preserved_count),
    dcov_component_request_count =
      as.integer(source$dcov_component_request_count),
    dcov_component_cache_hit_count =
      as.integer(source$dcov_component_cache_hit_count),
    dcov_component_cache_miss_count =
      as.integer(source$dcov_component_cache_miss_count),
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
    elapsed_sec = as.numeric(evidence$parallel_elapsed_seconds),
    numerical_gate = isTRUE(source$numerical_gate),
    optimizer_gate = isTRUE(source$optimizer_gate),
    downstream_decision_gate = isTRUE(source$downstream_decision_gate),
    backend_gate = isTRUE(source$backend_gate),
    phase5_only = TRUE,
    shadow_only = TRUE,
    production_backend_promoted = FALSE,
    phase10_promotion_claim = FALSE,
    pass = TRUE
  )
  summary
}

fastkpc_full_cuda_phase5_write_table <- function(value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase5_rank_table <- function(catalog) {
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  fields <- intersect(c(
    "prepared_s_key_sha256", "S_size", "formula_class", "penalty_count",
    "coefficient_dim", "penalty_rank", "penalty_nullity",
    "model_matrix_rank", "model_matrix_condition", "conditioning_rank",
    "conditioning_condition", "condition_bucket", "planned_route"
  ), names(scope$setup_rows))
  value <- scope$setup_rows[, fields, drop = FALSE]
  fastkpc_full_cuda_phase5_publication_require(
    nrow(value) == 7460L && "prepared_s_key_sha256" %in% names(value),
    "Phase 5 rank-condition table is incomplete"
  )
  value
}

fastkpc_full_cuda_phase5_write_graph <- function(evidence, directory) {
  comparison <- evidence$mixed_graph$comparison
  fastkpc_full_cuda_phase5_write_table(
    comparison$graph_agreement, directory, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase5_write_table(
    comparison$sepset_agreement, directory, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase5_write_table(
    comparison$n_edgetests, directory, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase5_write_table(
    comparison$candidate_deletions, directory, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence,
    file.path(directory, "first_divergence.json")
  )
  fallbacks <- data.frame(
    fallback_class = c("unknown", "approximate", "explicit_legacy_oracle"),
    supported_scope = c("none", "none", "none"),
    count = c(0L, 0L, 0L),
    accepted_for_phase5 = c(FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase5_write_table(
    fallbacks, directory, "fallbacks.csv"
  )
}

fastkpc_full_cuda_phase5_write_payload <- function(
    kind, directory, catalog, evidence, source_evidence_path) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase5_artifact_kinds())
  fastkpc_full_cuda_phase5_write_graph(evidence, directory)
  fastkpc_full_cuda_phase5_write_table(
    fastkpc_full_cuda_phase5_rank_table(catalog), directory,
    "rank_condition_results.csv"
  )
  near_alpha <- evidence$logical_rows[
    evidence$logical_rows$near_alpha, , drop = FALSE
  ]
  fastkpc_full_cuda_phase5_write_table(
    near_alpha, directory, "near_alpha_results.csv"
  )
  cache <- data.frame(
    cache = c("PreparedS", "legacy-dCov-component"),
    ownership = c("phase2-canonical-setup", "phase5-shadow-setup"),
    requests = c(
      nrow(evidence$targets),
      evidence$summary$dcov_component_request_count
    ),
    hits = c(0L, evidence$summary$dcov_component_cache_hit_count),
    misses = c(
      nrow(evidence$targets),
      evidence$summary$dcov_component_cache_miss_count
    ),
    semantic_eviction_effect = c("none", "none"),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase5_write_table(cache, directory, "cache.csv")
  fastkpc_full_cuda_phase5_write_table(
    evidence$targets, directory, "case_results.csv"
  )
  fastkpc_full_cuda_phase5_write_table(
    evidence$timings, directory, "stage_timing.csv"
  )
  partition_hashes <- data.frame(
    partition_file = names(evidence$partition_file_sha256),
    sha256 = unlist(evidence$partition_file_sha256, use.names = FALSE),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase5_write_table(
    partition_hashes, directory, "partition_hashes.csv"
  )
  raw_runs <- data.frame(
    run = seq_len(nrow(partition_hashes)),
    route = "cpp-multi-penalty-shadow-partition",
    partition_file = partition_hashes$partition_file,
    partition_sha256 = partition_hashes$sha256,
    pass = TRUE,
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase5_write_table(
    raw_runs, directory, "raw_runs.csv"
  )
  if (kind %in% c("full_shadow", "backend")) {
    saveRDS(
      evidence$logical_rows,
      file.path(directory, "logical_ci_results.rds"), version = 3L
    )
  }
  if (kind %in% c("oracle", "backend")) {
    saveRDS(
      evidence$transcripts,
      file.path(directory, "optimizer_transcripts.rds"), version = 3L
    )
  }
  copied <- file.copy(
    source_evidence_path, file.path(directory, "source_evidence.rds"),
    overwrite = FALSE, copy.mode = TRUE, copy.date = FALSE
  )
  fastkpc_full_cuda_phase5_publication_require(
    copied, "Phase 5 source evidence copy failed"
  )
}

fastkpc_full_cuda_phase5_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase5_validate_artifact <- function(
    artifact_dir, expected_kind = NULL, verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  kind <- as.character(manifest$artifact_kind)
  if (!is.null(expected_kind)) {
    expected_kind <- match.arg(
      expected_kind, fastkpc_full_cuda_phase5_artifact_kinds()
    )
  }
  manifest_clean <- identical(
    names(manifest), fastkpc_full_cuda_phase5_manifest_fields()
  ) && identical(
    manifest$schema_version,
    "full-cuda-ci-phase5-multi-penalty-cpp-artifact-manifest-v1"
  ) && kind %in% fastkpc_full_cuda_phase5_artifact_kinds() &&
    (is.null(expected_kind) || identical(kind, expected_kind)) &&
    identical(
      manifest$claim_scope,
      "phase5-additive-multi-penalty-S-size-greater-than-2"
    )
  fastkpc_full_cuda_phase5_publication_require(
    manifest_clean, "Phase 5 artifact manifest schema mismatch"
  )
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase5_publication_require(
    is.list(payload_hashes) && length(payload_hashes) > 0L &&
      length(payload_hashes) == manifest$semantic_file_count,
    "Phase 5 payload manifest is malformed"
  )
  for (name in names(payload_hashes)) {
    path <- file.path(artifact_dir, name)
    fastkpc_full_cuda_phase5_publication_require(
      file.exists(path) && !dir.exists(path) && identical(
        fastkpc_full_cuda_census_file_hash(path), payload_hashes[[name]]
      ),
      paste0("Phase 5 payload file hash mismatch: ", name)
    )
  }
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  fastkpc_full_cuda_phase5_publication_require(
    identical(
      payload_manifest_sha256, manifest$payload_manifest_sha256
    ),
    "Phase 5 payload manifest hash mismatch"
  )
  envelope <- manifest$producer_semantic_envelope
  fastkpc_full_cuda_phase35_validate_identity_envelope(envelope)
  fastkpc_full_cuda_phase5_publication_require(
    identical(envelope$payload_manifest_sha256,
              payload_manifest_sha256),
    "Phase 5 producer envelope payload mismatch"
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"),
    simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase5_publication_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(envelope$producer)
    ),
    "Phase 5 producer identity file does not match the envelope"
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
  fastkpc_full_cuda_phase5_publication_require(
    nrow(source_closure) > 0L && !anyDuplicated(source_closure$path) &&
      identical(
        closure_sha256, producer$producer_source_closure_sha256
      ),
    "Phase 5 producer source closure identity mismatch"
  )
  environment_path <- file.path(artifact_dir, manifest$environment_file)
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(environment_path)
  fastkpc_full_cuda_phase5_publication_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 5 environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  fastkpc_full_cuda_phase5_publication_require(
    is.list(attestations) && length(attestations) > 0L,
    "Phase 5 validator attestation is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase5_publication_require(
      identical(
        attestation$attested_producer_sha256, producer$identity_sha256
      ) && identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 5 validator attestation does not attest this producer"
    )
  }
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase5_publication_require(
    is.list(receipts) && length(receipts) > 0L,
    "Phase 5 execution receipt is missing"
  )
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase5_publication_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 5 execution receipt producer mismatch"
    )
  }
  required_files <- c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "source_closure.csv",
    "source_evidence.rds", "evidence_inputs.csv", "partition_hashes.csv",
    "producer_identity.json", "backend_configuration.json",
    "build_recipe.json", "validator_attestations.json",
    "execution_receipts.json"
  )
  fastkpc_full_cuda_phase5_publication_require(
    all(file.exists(file.path(artifact_dir, required_files))),
    "Phase 5 artifact standard file set is incomplete"
  )
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  common_gate <- identical(summary$artifact_kind, kind) &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    summary$direct_legacy_logical_test_count == 2213L &&
    summary$phase4_cuda_logical_test_count == 177952L &&
    summary$phase5_cpp_logical_test_count == 60324L &&
    summary$explicit_legacy_fallback_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$multi_penalty_setup_count == 7460L &&
    summary$multi_penalty_target_count == 65676L &&
    summary$candidate_legacy_mgcv_target_calls == 0L &&
    summary$fallback_count == 0L &&
    summary$optimizer_iteration_mismatch_count == 0L &&
    summary$score_call_mismatch_count == 0L &&
    summary$convergence_mismatch_count == 0L &&
    summary$hessian_state_mismatch_count == 0L &&
    summary$rank_mismatch_count == 0L &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$near_alpha_decision_flip_count == 0L &&
    summary$max_selected_log_sp_error <= 1e-6 &&
    summary$max_score_absolute_error <= 1e-8 &&
    summary$max_edf_absolute_error <= 1e-8 &&
    summary$max_fitted_absolute_error <= 1e-7 &&
    summary$max_fitted_relative_l2 <= 1e-8 &&
    summary$max_residual_absolute_error <= 1e-7 &&
    summary$max_residual_relative_l2 <= 1e-8 &&
    summary$max_absolute_p_value_difference <= 1e-10 &&
    isTRUE(summary$numerical_gate) && isTRUE(summary$optimizer_gate) &&
    isTRUE(summary$downstream_decision_gate) &&
    isTRUE(summary$backend_gate) &&
    identical(summary$source_closure_sha256,
              producer$producer_source_closure_sha256) &&
    identical(summary$native_binary_sha256,
              producer$native_binary_sha256) &&
    identical(summary$producer_identity_sha256, producer$identity_sha256) &&
    isTRUE(summary$phase5_only) && isTRUE(summary$shadow_only) &&
    !isTRUE(summary$production_backend_promoted) &&
    !isTRUE(summary$phase10_promotion_claim) && isTRUE(summary$pass)
  fastkpc_full_cuda_phase5_publication_require(
    common_gate, "Phase 5 artifact common gate failed"
  )
  cases <- utils::read.csv(
    file.path(artifact_dir, "case_results.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  graph <- utils::read.csv(
    file.path(artifact_dir, "graph_agreement.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  n_edgetests <- utils::read.csv(
    file.path(artifact_dir, "n_edgetests.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  first <- jsonlite::read_json(
    file.path(artifact_dir, "first_divergence.json"),
    simplifyVector = TRUE
  )
  payload_gate <- nrow(cases) == 65676L &&
    all(cases$fallback_reason == "NONE") &&
    all(cases$rank_path ==
          "pivoted-qr-augmented-lapack-dgesdd-svd") &&
    !any(cases$normal_equations_used) && all(cases$all_finite) &&
    nrow(graph) == 1L && graph$SHD[[1L]] == 0L &&
    isTRUE(graph$adjacency_identical[[1L]]) && identical(
      as.integer(n_edgetests$reference),
      c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
    ) && identical(
      as.integer(n_edgetests$candidate),
      c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
    ) && all(n_edgetests$identical) &&
    !isTRUE(first$first_divergence_found)
  fastkpc_full_cuda_phase5_publication_require(
    payload_gate, "Phase 5 artifact payload gate failed"
  )
  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  if (kind %in% c("full_shadow", "backend")) {
    logical <- readRDS(file.path(artifact_dir, "logical_ci_results.rds"))
    fastkpc_full_cuda_phase5_publication_require(
      nrow(logical) == 60324L && !any(logical$decision_flip) &&
        !any(logical$backend_error) && !any(logical$spectra_fallback),
      paste0("Phase 5 ", kind, " logical payload gate failed")
    )
  }
  if (kind %in% c("oracle", "backend")) {
    transcripts <- readRDS(
      file.path(artifact_dir, "optimizer_transcripts.rds")
    )
    fastkpc_full_cuda_phase5_publication_require(
      is.list(transcripts) && length(transcripts) ==
        summary$transcript_preserved_count && length(transcripts) > 0L,
      paste0("Phase 5 ", kind, " transcript gate failed")
    )
  }
  if (isTRUE(verify_current_sources)) {
    current_source <- fastkpc_full_cuda_phase5_evidence_source_closure()
    current_native <- fastkpc_full_cuda_phase5_native_identity()
    fastkpc_full_cuda_phase5_publication_require(
      identical(current_source$sha256, closure_sha256) &&
        identical(current_source$table, source_closure) &&
        identical(current_native$sha256, producer$native_binary_sha256),
      "Phase 5 artifact does not match current sources/binary"
    )
    implementation <- paste(c(
      readLines("fastkpc/src/mgcv_multi_penalty_cpp.cpp", warn = FALSE),
      readLines("fastkpc/R/full_cuda_ci_multi_penalty_cpp.R", warn = FALSE)
    ), collapse = "\n")
    fastkpc_full_cuda_phase5_publication_require(
      !grepl("[0-9a-f]{64}", implementation),
      "Phase 5 implementation contains target/setup-key-specific routing"
    )
  }
  list(
    artifact_dir = artifact_dir,
    kind = kind,
    manifest = manifest,
    summary = summary,
    producer = producer,
    source_closure = source_closure,
    evidence = evidence
  )
}

fastkpc_full_cuda_phase5_publish_one <- function(
    kind, output_dir, catalog, evidence, source_evidence_path,
    evidence_sha256, contracts, source_closure, native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase5_artifact_kinds())
  producer_bundle <- fastkpc_full_cuda_phase5_producer(
    kind, catalog, source_closure, native_identity, contracts
  )
  output_parent <- dirname(output_dir)
  dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
  stage_dir <- tempfile(
    paste0(".phase5-", kind, "-stage-"), tmpdir = output_parent
  )
  dir.create(stage_dir, recursive = TRUE)
  stage_active <- TRUE
  on.exit({
    if (stage_active && dir.exists(stage_dir)) {
      unlink(stage_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  summary <- fastkpc_full_cuda_phase5_common_summary(
    kind, evidence, producer_bundle, contracts, source_closure,
    native_identity
  )
  fastkpc_full_cuda_phase5_write_payload(
    kind, stage_dir, catalog, evidence, source_evidence_path
  )
  fastkpc_full_cuda_phase5_write_table(
    source_closure$table, stage_dir, "source_closure.csv"
  )
  evidence_inputs <- data.frame(
    evidence_id = c("merged_phase5", "inherited_phase4_logical"),
    sha256 = c(
      evidence_sha256,
      evidence$mixed_graph$inherited_phase4_logical_results_sha256
    ),
    semantic_role = c(
      "objective-residual-dcov-and-graph-authority",
      "accepted-single-penalty-logical-authority"
    ),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase5_write_table(
    evidence_inputs, stage_dir, "evidence_inputs.csv"
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
  fastkpc_full_cuda_write_summary_md(
    summary, file.path(stage_dir, "summary.md"),
    paste0("Full CUDA CI Phase 5 ", kind)
  )
  writeLines(
    c(
      paste0("source evidence SHA-256: ", evidence_sha256),
      paste0("producer source closure SHA-256: ", source_closure$sha256),
      paste0("native binary SHA-256: ", native_identity$sha256),
      "see fastkpc/tools/run_full_cuda_ci_multi_penalty_cpp_artifacts.R"
    ),
    file.path(stage_dir, "commands.txt"), useBytes = TRUE
  )
  environment_lines <- c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase5_execution=cpp-shadow",
    "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1"
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
  validator_paths <- c(
    "fastkpc/R/full_cuda_ci_gate.R",
    "fastkpc/R/full_cuda_ci_workload_census.R",
    "fastkpc/R/full_cuda_ci_phase35_contracts.R",
    "fastkpc/R/full_cuda_ci_phase5_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase5_publication.R"
  )
  validator_hashes <- setNames(as.list(vapply(
    validator_paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), validator_paths)
  validator_closure_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(validator_hashes)
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer_bundle$producer,
    validator_source_closure_sha256 = validator_closure_sha256,
    validator_semantic_version =
      "full-cuda-ci-phase5-multi-penalty-cpp-validator-v1",
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
    cuda_context_id = "not-applicable-phase5-cpp-shadow",
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
      "full-cuda-ci-phase5-multi-penalty-cpp-artifact-manifest-v1",
    artifact_kind = kind,
    claim_scope = "phase5-additive-multi-penalty-S-size-greater-than-2",
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
  fastkpc_full_cuda_phase5_validate_artifact(
    stage_dir, expected_kind = kind, verify_current_sources = TRUE
  )
  backup_dir <- NULL
  if (dir.exists(output_dir)) {
    backup_dir <- tempfile(
      paste0(".phase5-", kind, "-backup-"), tmpdir = output_parent
    )
    fastkpc_full_cuda_phase5_publication_require(
      file.rename(output_dir, backup_dir),
      "Phase 5 prior artifact could not be staged for replacement"
    )
  }
  published <- file.rename(stage_dir, output_dir)
  if (!published) {
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop("Phase 5 artifact publication failed", call. = FALSE)
  }
  stage_active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase5_validate_artifact(
      output_dir, expected_kind = kind, verify_current_sources = TRUE
    ),
    error = identity
  )
  if (inherits(validated, "error")) {
    failed_dir <- tempfile(
      paste0(".phase5-", kind, "-failed-"), tmpdir = output_parent
    )
    file.rename(output_dir, failed_dir)
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup_dir)) unlink(backup_dir, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase5_publish_artifacts <- function(
    catalog, evidence_path,
    output_root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  evidence_path <- normalizePath(
    evidence_path, winslash = "/", mustWork = TRUE
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase5_validate_merged_evidence(
    evidence, catalog, verify_current_identity = TRUE
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase5_evidence_source_closure()
  native_identity <- fastkpc_full_cuda_phase5_native_identity()
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(evidence_path)
  directories <- fastkpc_full_cuda_phase5_artifact_directory_names()
  results <- lapply(
    fastkpc_full_cuda_phase5_artifact_kinds(), function(kind) {
      fastkpc_full_cuda_phase5_publish_one(
        kind = kind,
        output_dir = file.path(output_root, directories[[kind]]),
        catalog = catalog,
        evidence = evidence,
        source_evidence_path = evidence_path,
        evidence_sha256 = evidence_sha256,
        contracts = contracts,
        source_closure = source_closure,
        native_identity = native_identity
      )
    }
  )
  names(results) <- fastkpc_full_cuda_phase5_artifact_kinds()
  results
}
