fastkpc_full_cuda_phase4_publication_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase4_artifact_kinds <- function() {
  c("oracle", "full_shadow", "backend")
}

fastkpc_full_cuda_phase4_artifact_directory_names <- function() {
  c(
    oracle = "single_penalty_cuda_gcv_oracle_v1",
    full_shadow = "single_penalty_cuda_gcv_full_shadow_v1",
    backend = "single_penalty_cuda_gcv_backend_v1"
  )
}

fastkpc_full_cuda_phase4_publication_source_paths <- function() {
  native <- list.files(
    "fastkpc/src", recursive = TRUE, full.names = TRUE,
    include.dirs = FALSE
  )
  native <- native[grepl(
    "\\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx|cuh|inc)$", native
  )]
  phase4 <- c(
    "fastkpc/R/cuda_native.R",
    "fastkpc/R/full_cuda_ci_gate.R",
    "fastkpc/R/full_cuda_ci_oracle_contract.R",
    "fastkpc/R/full_cuda_ci_workload_census.R",
    "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R",
    "fastkpc/R/full_cuda_ci_phase3_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase35_contracts.R",
    "fastkpc/R/full_cuda_ci_single_penalty_gcv.R",
    "fastkpc/R/full_cuda_ci_phase4_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase4_backend.R",
    "fastkpc/R/full_cuda_ci_phase4_publication.R",
    "fastkpc/tools/build_cuda_native.sh",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_oracle.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_shadow.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_merge.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_partitions.sh",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_backend.R",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_backend.sh",
    "fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_artifacts.R"
  )
  paths <- sort(unique(c(native, phase4)), method = "radix")
  fastkpc_full_cuda_phase4_publication_require(
    length(paths) > 0L && all(file.exists(paths) & !dir.exists(paths)),
    "Phase 4 publication source closure contains a missing file"
  )
  paths
}

fastkpc_full_cuda_phase4_publication_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase4_publication_source_paths()
  hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), paths)
  list(
    table = data.frame(
      path = names(hashes),
      sha256 = unlist(hashes, use.names = FALSE),
      stringsAsFactors = FALSE
    ),
    hashes = hashes,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(hashes)
    )
  )
}

fastkpc_full_cuda_phase4_publication_native_identity <- function() {
  load_fastkpc_cuda_native()
  dll <- getLoadedDLLs()[["fastkpc_cuda"]]
  fastkpc_full_cuda_phase4_publication_require(
    !is.null(dll), "Phase 4 publication native DLL is not loaded"
  )
  path <- normalizePath(dll[["path"]], winslash = "/", mustWork = TRUE)
  list(
    path = path,
    sha256 = fastkpc_full_cuda_census_file_hash(path)
  )
}

fastkpc_full_cuda_phase4_publication_input_identity <- function(catalog) {
  files <- c(
    canonical_data = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    phase0_manifest = file.path(catalog$phase0_dir, "manifest.json"),
    phase1_manifest = file.path(catalog$phase1_dir, "manifest.json"),
    phase2_manifest = file.path(catalog$phase2_dir, "manifest.json")
  )
  hashes <- setNames(as.list(vapply(
    files, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(files))
  dataset <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase4-canonical-corpus-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    input_file_sha256 = hashes$canonical_data,
    phase1_manifest_sha256 = hashes$phase1_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    setup_count = 1174L,
    target_count = 44941L,
    S_size = c(1L, 2L)
  ))
  oracle <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase4-oracle-authority-v1",
    phase0_manifest_sha256 = hashes$phase0_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    mgcv_semantics_version = "mgcv-gam-gcv-cp-v1",
    allowed_residual_decision_flip_count = 0L,
    allowed_dcov_decision_flip_count = 0L,
    required_SHD = 0L
  ))
  list(files = files, hashes = hashes, dataset_sha256 = dataset,
       oracle_sha256 = oracle)
}

fastkpc_full_cuda_phase4_publication_backend_configuration <- function(kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase4_artifact_kinds())
  value <- list(
    schema_version = "full-cuda-ci-phase4-backend-configuration-v1",
    artifact_kind = kind,
    sp_selection_backend = "cuda",
    gcv_score_backend = "cuda",
    optimizer_backend = "cuda-spectral-risk-gated-exact-replay",
    exact_replay_backend = "cuda-dpstf2-lapack-3.12-dgesdd",
    exact_replay_origin = "initial_sp",
    spectral_precision_floor_rule =
      "reported_rms_gradient<=0.5*convergence_tolerance",
    convergence_tolerance = 1e-7,
    execution_strategy = "cuda-cross-setup-fused-exact-replay",
    concurrency = 1L,
    physical_gpu_index = 0L,
    precision = "float64",
    fmad = FALSE,
    cpu_score_count = 0L,
    cpu_optimizer_count = 0L,
    fallback_count = 0L,
    legacy_mgcv_target_calls = 0L
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase4_publication_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase4-build-recipe-v1",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    nvcc_path = "/usr/local/cuda/bin/nvcc",
    nvcc_version = fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    cuda_architecture = "sm_89",
    phase4_fmad = FALSE,
    phase4_fast_math = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase4_publication_producer <- function(
    kind, catalog, source_closure, native_identity,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  input <- fastkpc_full_cuda_phase4_publication_input_identity(catalog)
  backend <- fastkpc_full_cuda_phase4_publication_backend_configuration(kind)
  build <- fastkpc_full_cuda_phase4_publication_build_recipe()
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase4-single-penalty-gcv-", kind, "-v1"
    ),
    dataset_or_corpus_sha256 = input$dataset_sha256,
    oracle_sha256 = input$oracle_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(
    producer = producer,
    input = input,
    backend = backend,
    build = build
  )
}

fastkpc_full_cuda_phase4_read_gpu_samples <- function(path) {
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  samples <- utils::read.csv(
    path, header = FALSE, stringsAsFactors = FALSE, strip.white = TRUE,
    check.names = FALSE
  )
  fastkpc_full_cuda_phase4_publication_require(
    ncol(samples) == 7L && nrow(samples) > 0L,
    "Phase 4 GPU sample schema is malformed"
  )
  names(samples) <- c(
    "timestamp", "physical_gpu_index", "gpu_uuid", "gpu_name",
    "memory_used_mib", "utilization_percent", "power_w"
  )
  samples$physical_gpu_index <- as.integer(samples$physical_gpu_index)
  samples$memory_used_mib <- as.numeric(samples$memory_used_mib)
  samples$utilization_percent <- as.numeric(samples$utilization_percent)
  samples$power_w <- as.numeric(samples$power_w)
  clean <- !anyNA(samples) &&
    identical(unique(samples$physical_gpu_index), 0L) &&
    length(unique(samples$gpu_uuid)) == 1L &&
    length(unique(samples$gpu_name)) == 1L &&
    max(samples$utilization_percent) > 0 &&
    max(samples$memory_used_mib) > min(samples$memory_used_mib) &&
    all(samples$memory_used_mib >= 0) &&
    all(samples$utilization_percent >= 0 &
          samples$utilization_percent <= 100) &&
    all(samples$power_w > 0)
  fastkpc_full_cuda_phase4_publication_require(
    clean, "Phase 4 GPU samples do not prove physical GPU 0 execution"
  )
  samples
}

fastkpc_full_cuda_phase4_validate_publication_inputs <- function(
    oracle, shadow, backend) {
  oracle_clean <- is.list(oracle) && identical(
    oracle$schema_version,
    "full-cuda-ci-single-penalty-gcv-oracle-evidence-v1"
  ) && oracle$summary$setup_count == 1174L &&
    oracle$summary$target_count == 44941L &&
    isTRUE(oracle$summary$objective_curve_gate) &&
    isTRUE(oracle$summary$optimizer_objective_gate) &&
    isTRUE(oracle$summary$optimizer_coverage_gate) &&
    isTRUE(oracle$summary$backend_gate) &&
    isTRUE(oracle$summary$transcript_gate)
  shadow_clean <- is.list(shadow) && identical(
    shadow$schema_version,
    "full-cuda-ci-single-penalty-gcv-full-shadow-merged-v1"
  ) && shadow$summary$setup_count == 1174L &&
    shadow$summary$target_count == 44941L &&
    shadow$summary$logical_test_count == 177952L &&
    isTRUE(shadow$summary$same_sp_fixed_solver_gate) &&
    isTRUE(shadow$summary$oracle_residual_gate) &&
    isTRUE(shadow$summary$downstream_decision_gate) &&
    isTRUE(shadow$summary$optimizer_coverage_gate) &&
    isTRUE(shadow$summary$backend_gate) &&
    isTRUE(shadow$mixed_graph$summary$pass)
  backend_error <- tryCatch(
    fastkpc_full_cuda_phase4_validate_backend_evidence(backend),
    error = identity
  )
  fastkpc_full_cuda_phase4_publication_require(
    oracle_clean, "Phase 4 oracle publication input failed its gates"
  )
  fastkpc_full_cuda_phase4_publication_require(
    shadow_clean, "Phase 4 shadow publication input failed its gates"
  )
  fastkpc_full_cuda_phase4_publication_require(
    !inherits(backend_error, "error"),
    if (inherits(backend_error, "error")) conditionMessage(backend_error) else
      "Phase 4 backend publication input failed its gates"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase4_publication_common_summary <- function(
    kind, oracle, shadow, backend, producer_bundle, contracts,
    source_closure, native_identity, gpu_samples) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase4_artifact_kinds())
  graph <- shadow$mixed_graph$summary
  elapsed_sec <- switch(
    kind,
    oracle = as.numeric(oracle$summary$elapsed_seconds),
    full_shadow = as.numeric(shadow$summary$elapsed_seconds),
    backend = as.numeric(backend$summary$candidate_median_ms) / 1000
  )
  summary <- list(
    schema_version = paste0(
      "full-cuda-ci-phase4-single-penalty-gcv-", kind, "-summary-v1"
    ),
    artifact_kind = kind,
    claim_scope = "phase4-single-penalty-S-size-at-most-2",
    run_status = "ok",
    timeout = FALSE,
    source_commit = fastkpc_full_cuda_source_commit(),
    source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    producer_identity_sha256 =
      producer_bundle$producer$identity_sha256,
    oracle_artifact =
      "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
    candidate_route =
      "cuda-spectral-risk-gated-exact-replay+cuda-fixed-sp",
    edge_count_reference = as.integer(graph$edge_count_reference),
    edge_count_candidate = as.integer(graph$edge_count_candidate),
    SHD = as.integer(graph$SHD),
    adjacency_identical = isTRUE(graph$adjacency_identical),
    sepsets_identical = isTRUE(graph$sepsets_identical),
    n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
    deletions_identical = isTRUE(graph$deletions_identical),
    phase4_cuda_logical_test_count =
      as.integer(graph$phase4_cuda_logical_test_count),
    direct_legacy_logical_test_count =
      as.integer(graph$direct_legacy_logical_test_count),
    explicit_legacy_fallback_count =
      as.integer(graph$explicit_legacy_fallback_count),
    fallback_min_S_size = as.integer(graph$fallback_min_S_size),
    fallback_max_S_size = as.integer(graph$fallback_max_S_size),
    unknown_fallback_count = 0L,
    approximate_backend_count = 0L,
    legacy_mgcv_target_calls = 0L,
    cpu_score_count = 0L,
    cpu_optimizer_count = 0L,
    single_penalty_setup_count = 1174L,
    single_penalty_target_count = 44941L,
    downstream_legacy_dcov_decision_flip_count =
      as.integer(shadow$summary$downstream_legacy_dcov_decision_flip_count),
    max_residual_oracle_relative_l2 =
      as.numeric(shadow$summary$max_residual_oracle_relative_l2),
    exact_replay_target_count =
      as.integer(backend$summary$exact_replay_target_count),
    numerical_risk_count = as.integer(backend$summary$numerical_risk_count),
    physical_gpu_index = 0L,
    physical_gpu_uuid = unique(gpu_samples$gpu_uuid),
    physical_gpu_name = unique(gpu_samples$gpu_name),
    maximum_gpu_utilization_percent =
      max(gpu_samples$utilization_percent),
    maximum_gpu_memory_used_mib = max(gpu_samples$memory_used_mib),
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
    elapsed_sec = elapsed_sec,
    phase4_only = TRUE,
    production_backend_promoted = FALSE,
    phase10_promotion_claim = FALSE,
    pass = TRUE
  )
  if (identical(kind, "oracle")) {
    summary$objective_curve_gate <-
      isTRUE(oracle$summary$objective_curve_gate)
    summary$optimizer_objective_gate <-
      isTRUE(oracle$summary$optimizer_objective_gate)
    summary$optimizer_coverage_gate <-
      isTRUE(oracle$summary$optimizer_coverage_gate)
    summary$transcript_gate <- isTRUE(oracle$summary$transcript_gate)
    summary$max_rss_absolute_error <-
      as.numeric(oracle$summary$max_rss_absolute_error)
    summary$max_rss_relative_error <-
      as.numeric(oracle$summary$max_rss_relative_error)
    summary$max_edf_absolute_error <-
      as.numeric(oracle$summary$max_edf_absolute_error)
    summary$max_score_absolute_error <-
      as.numeric(oracle$summary$max_score_absolute_error)
    summary$max_score_relative_error <-
      as.numeric(oracle$summary$max_score_relative_error)
    summary$transcript_required_count <-
      as.integer(oracle$summary$transcript_required_count)
    summary$transcript_preserved_count <-
      as.integer(oracle$summary$transcript_preserved_count)
  } else if (identical(kind, "full_shadow")) {
    summary$same_sp_fixed_solver_gate <-
      isTRUE(shadow$summary$same_sp_fixed_solver_gate)
    summary$oracle_residual_gate <-
      isTRUE(shadow$summary$oracle_residual_gate)
    summary$downstream_decision_gate <-
      isTRUE(shadow$summary$downstream_decision_gate)
    summary$optimizer_coverage_gate <-
      isTRUE(shadow$summary$optimizer_coverage_gate)
    summary$logical_test_count <- as.integer(shadow$summary$logical_test_count)
    summary$max_absolute_p_value_difference <-
      as.numeric(shadow$summary$max_absolute_p_value_difference)
    summary$partition_count <- as.integer(shadow$partition_count)
  } else {
    summary$repetition_count <- as.integer(backend$summary$repetition_count)
    summary$candidate_median_ms <-
      as.numeric(backend$summary$candidate_median_ms)
    summary$baseline_median_ms <-
      as.numeric(backend$summary$baseline_median_ms)
    summary$candidate_to_baseline_ratio <-
      as.numeric(backend$summary$candidate_to_baseline_ratio)
    summary$all_candidate_targets_identical <-
      isTRUE(backend$summary$all_candidate_targets_identical)
    summary$all_candidate_counters_identical <-
      isTRUE(backend$summary$all_candidate_counters_identical)
    summary$absolute_performance_gate <-
      isTRUE(backend$summary$absolute_performance_gate)
    summary$relative_performance_gate <-
      isTRUE(backend$summary$relative_performance_gate)
    summary$backend_gate <- isTRUE(backend$summary$backend_gate)
  }
  summary
}

fastkpc_full_cuda_phase4_publication_rank_table <- function(catalog) {
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  fields <- intersect(c(
    "prepared_s_key_sha256", "S_size", "formula_class", "penalty_count",
    "coefficient_dim", "penalty_rank", "penalty_nullity",
    "model_matrix_rank", "model_matrix_condition", "conditioning_rank",
    "conditioning_condition", "positive_eigen_min", "positive_eigen_max",
    "positive_eigen_condition", "condition_bucket", "planned_route"
  ), names(scope$setup_rows))
  value <- scope$setup_rows[, fields, drop = FALSE]
  fastkpc_full_cuda_phase4_publication_require(
    nrow(value) == 1174L && "prepared_s_key_sha256" %in% names(value),
    "Phase 4 rank-condition table is incomplete"
  )
  value
}

fastkpc_full_cuda_phase4_publication_write_table <- function(
    value, stage_dir, name) {
  utils::write.csv(
    value, file.path(stage_dir, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase4_publication_write_graph <- function(
    shadow, stage_dir) {
  comparison <- shadow$mixed_graph$comparison
  fastkpc_full_cuda_phase4_publication_write_table(
    comparison$graph_agreement, stage_dir, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    comparison$sepset_agreement, stage_dir, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    comparison$n_edgetests, stage_dir, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    comparison$candidate_deletions, stage_dir, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence,
    file.path(stage_dir, "first_divergence.json")
  )
  fallbacks <- data.frame(
    fallback_class = c("unknown", "approximate", "explicit_legacy_oracle"),
    supported_scope = c("none", "none", "S_size_greater_than_2"),
    count = c(
      0L, 0L,
      as.integer(shadow$mixed_graph$summary$explicit_legacy_fallback_count)
    ),
    accepted_for_phase4 = c(FALSE, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    fallbacks, stage_dir, "fallbacks.csv"
  )
}

fastkpc_full_cuda_phase4_publication_write_kind_payload <- function(
    kind, stage_dir, catalog, oracle, shadow, backend, gpu_samples,
    source_evidence_path) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase4_artifact_kinds())
  fastkpc_full_cuda_phase4_publication_write_graph(shadow, stage_dir)
  rank_table <- fastkpc_full_cuda_phase4_publication_rank_table(catalog)
  fastkpc_full_cuda_phase4_publication_write_table(
    rank_table, stage_dir, "rank_condition_results.csv"
  )
  near_alpha <- shadow$logical_rows[
    shadow$logical_rows$near_alpha, , drop = FALSE
  ]
  fastkpc_full_cuda_phase4_publication_write_table(
    near_alpha, stage_dir, "near_alpha_results.csv"
  )
  cache <- data.frame(
    cache = c("PreparedS", "selected-sp-fixed-solve"),
    ownership = c("phase2-canonical-setup", "phase3-persistent-runtime"),
    capacity_policy = c(
      "complete-phase4-corpus-publication-input",
      "canonical-capacity-contract"
    ),
    semantic_eviction_effect = c("none", "none"),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    cache, stage_dir, "cache.csv"
  )

  if (identical(kind, "oracle")) {
    case_results <- oracle$targets
    stage_timing <- oracle$timings
    raw_runs <- data.frame(
      run = 1L,
      route = "cuda-objective-and-optimizer-scan",
      elapsed_sec = oracle$summary$elapsed_seconds,
      pass = TRUE,
      stringsAsFactors = FALSE
    )
    fastkpc_full_cuda_phase4_publication_write_table(
      oracle$curves, stage_dir, "curve_results.csv"
    )
    saveRDS(
      oracle$transcripts,
      file.path(stage_dir, "optimizer_transcripts.rds"), version = 3L
    )
  } else if (identical(kind, "full_shadow")) {
    case_results <- shadow$targets
    stage_timing <- shadow$timings
    raw_runs <- data.frame(
      run = seq_along(shadow$partition_file_sha256),
      route = "cuda-gcv-residual-shadow-partition",
      elapsed_sec = NA_real_,
      pass = TRUE,
      stringsAsFactors = FALSE
    )
    saveRDS(
      shadow$logical_rows,
      file.path(stage_dir, "logical_ci_results.rds"), version = 3L
    )
    partition_hashes <- data.frame(
      partition_file = names(shadow$partition_file_sha256),
      sha256 = unlist(shadow$partition_file_sha256, use.names = FALSE),
      stringsAsFactors = FALSE
    )
    fastkpc_full_cuda_phase4_publication_write_table(
      partition_hashes, stage_dir, "partition_hashes.csv"
    )
  } else {
    case_results <- backend$candidate_targets
    stage_timing <- rbind(
      backend$candidate_measurements, backend$baseline_measurements
    )
    raw_runs <- stage_timing
    fastkpc_full_cuda_phase4_publication_write_table(
      backend$baseline_targets, stage_dir, "baseline_case_results.csv"
    )
    counter_signature <- data.frame(
      prepared_s_key_sha256 = rownames(backend$counter_signature),
      backend$counter_signature,
      row.names = NULL,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    fastkpc_full_cuda_phase4_publication_write_table(
      counter_signature, stage_dir, "counter_signature.csv"
    )
    fastkpc_full_cuda_phase4_publication_write_table(
      backend$request_identity, stage_dir, "request_identity.csv"
    )
    fastkpc_full_cuda_phase4_publication_write_table(
      gpu_samples, stage_dir, "gpu_samples.csv"
    )
  }
  fastkpc_full_cuda_phase4_publication_write_table(
    case_results, stage_dir, "case_results.csv"
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    stage_timing, stage_dir, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    raw_runs, stage_dir, "raw_runs.csv"
  )
  copied <- file.copy(
    source_evidence_path, file.path(stage_dir, "source_evidence.rds"),
    overwrite = FALSE, copy.mode = TRUE, copy.date = FALSE
  )
  fastkpc_full_cuda_phase4_publication_require(
    copied, "Phase 4 source evidence copy failed"
  )
}

fastkpc_full_cuda_phase4_publication_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase4_validate_artifact <- function(
    artifact_dir, expected_kind = NULL, verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  manifest_path <- file.path(artifact_dir, "manifest.json")
  fastkpc_full_cuda_phase4_publication_require(
    file.exists(manifest_path) && !dir.exists(manifest_path),
    "Phase 4 artifact manifest is missing"
  )
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  kind <- manifest$artifact_kind
  if (!is.null(expected_kind)) {
    expected_kind <- match.arg(
      expected_kind, fastkpc_full_cuda_phase4_artifact_kinds()
    )
  }
  manifest_clean <- is.list(manifest) && identical(
    names(manifest), fastkpc_full_cuda_phase4_publication_manifest_fields()
  ) && identical(
    manifest$schema_version,
    "full-cuda-ci-phase4-single-penalty-gcv-artifact-manifest-v1"
  ) && kind %in% fastkpc_full_cuda_phase4_artifact_kinds() &&
    (is.null(expected_kind) || identical(kind, expected_kind)) &&
    identical(
      manifest$claim_scope, "phase4-single-penalty-S-size-at-most-2"
    ) && identical(
      manifest$validator_attestations_file,
      "validator_attestations.json"
    ) && identical(
      manifest$volatile_receipt_file, "execution_receipts.json"
    ) && identical(manifest$environment_file, "environment.txt")
  fastkpc_full_cuda_phase4_publication_require(
    manifest_clean, "Phase 4 artifact manifest schema mismatch"
  )
  envelope <- manifest$producer_semantic_envelope
  fastkpc_full_cuda_phase4_publication_require(
    isTRUE(fastkpc_full_cuda_phase35_validate_identity_envelope(envelope)),
    "Phase 4 producer semantic envelope is invalid"
  )
  fastkpc_full_cuda_phase4_publication_require(
    identical(
      manifest$payload_manifest_sha256,
      envelope$payload_manifest_sha256
    ) && identical(
      manifest$payload_manifest_sha256,
      manifest$producer_semantic_envelope$payload_manifest_sha256
    ),
    "Phase 4 payload manifest identity mismatch"
  )
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase4_publication_require(
    is.list(payload_hashes) && length(payload_hashes) > 0L &&
      !is.null(names(payload_hashes)) && !anyDuplicated(names(payload_hashes)) &&
      as.integer(manifest$semantic_file_count) == length(payload_hashes),
    "Phase 4 payload hash table is malformed"
  )
  for (name in names(payload_hashes)) {
    path <- file.path(artifact_dir, name)
    actual <- if (file.exists(path) && !dir.exists(path)) {
      fastkpc_full_cuda_census_file_hash(path)
    } else {
      NA_character_
    }
    fastkpc_full_cuda_phase4_publication_require(
      identical(actual, payload_hashes[[name]]),
      paste0("Phase 4 payload file hash mismatch: ", name)
    )
  }
  expected_payload_hash <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  fastkpc_full_cuda_phase4_publication_require(
    identical(expected_payload_hash, manifest$payload_manifest_sha256),
    "Phase 4 aggregate payload hash mismatch"
  )

  producer_path <- file.path(artifact_dir, "producer_identity.json")
  producer <- jsonlite::read_json(producer_path, simplifyVector = FALSE)
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase4_publication_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(envelope$producer)
    ),
    "Phase 4 producer identity file does not match the envelope"
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
  fastkpc_full_cuda_phase4_publication_require(
    nrow(source_closure) > 0L && !anyDuplicated(source_closure$path) &&
      identical(
        closure_sha256, producer$producer_source_closure_sha256
      ),
    "Phase 4 producer source closure identity mismatch"
  )

  environment_path <- file.path(artifact_dir, manifest$environment_file)
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(environment_path)
  fastkpc_full_cuda_phase4_publication_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 4 environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  fastkpc_full_cuda_phase4_publication_require(
    is.list(attestations) && length(attestations) > 0L,
    "Phase 4 validator attestation is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase4_publication_require(
      identical(
        attestation$attested_producer_sha256, producer$identity_sha256
      ) && identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 4 validator attestation does not attest this producer"
    )
  }
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase4_publication_require(
    is.list(receipts) && length(receipts) > 0L,
    "Phase 4 execution receipt is missing"
  )
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase4_publication_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 4 execution receipt producer mismatch"
    )
  }

  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  common_summary_gate <- identical(summary$artifact_kind, kind) &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    summary$phase4_cuda_logical_test_count == 177952L &&
    summary$explicit_legacy_fallback_count == 60324L &&
    summary$fallback_min_S_size >= 3L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$legacy_mgcv_target_calls == 0L &&
    summary$cpu_score_count == 0L && summary$cpu_optimizer_count == 0L &&
    summary$single_penalty_setup_count == 1174L &&
    summary$single_penalty_target_count == 44941L &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$max_residual_oracle_relative_l2 <= 1e-8 &&
    summary$physical_gpu_index == 0L &&
    summary$maximum_gpu_utilization_percent > 0 &&
    identical(summary$source_closure_sha256,
              producer$producer_source_closure_sha256) &&
    identical(summary$native_binary_sha256,
              producer$native_binary_sha256) &&
    identical(summary$producer_identity_sha256, producer$identity_sha256) &&
    isTRUE(summary$phase4_only) &&
    !isTRUE(summary$production_backend_promoted) &&
    !isTRUE(summary$phase10_promotion_claim) && isTRUE(summary$pass)
  fastkpc_full_cuda_phase4_publication_require(
    common_summary_gate, "Phase 4 artifact common gate failed"
  )

  required_files <- c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "source_closure.csv",
    "source_evidence.rds", "evidence_inputs.csv",
    "producer_identity.json", "backend_configuration.json",
    "build_recipe.json", "validator_attestations.json",
    "execution_receipts.json"
  )
  fastkpc_full_cuda_phase4_publication_require(
    all(file.exists(file.path(artifact_dir, required_files))),
    "Phase 4 artifact standard file set is incomplete"
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
    file.path(artifact_dir, "first_divergence.json"), simplifyVector = TRUE
  )
  graph_gate <- nrow(graph) == 1L && graph$SHD[[1L]] == 0L &&
    isTRUE(graph$adjacency_identical[[1L]]) &&
    identical(
      as.integer(n_edgetests$reference),
      c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
    ) && identical(
      as.integer(n_edgetests$candidate),
      c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)
    ) && all(n_edgetests$identical) &&
    !isTRUE(first$first_divergence_found)
  fastkpc_full_cuda_phase4_publication_require(
    graph_gate, "Phase 4 artifact graph payload gate failed"
  )

  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  case_results <- utils::read.csv(
    file.path(artifact_dir, "case_results.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  if (identical(kind, "oracle")) {
    kind_gate <- nrow(case_results) == 44941L &&
      isTRUE(summary$objective_curve_gate) &&
      isTRUE(summary$optimizer_objective_gate) &&
      isTRUE(summary$optimizer_coverage_gate) &&
      isTRUE(summary$transcript_gate) && identical(
        evidence$schema_version,
        "full-cuda-ci-single-penalty-gcv-oracle-evidence-v1"
      ) && nrow(utils::read.csv(
        file.path(artifact_dir, "curve_results.csv")
      )) == 1174L && file.exists(
        file.path(artifact_dir, "optimizer_transcripts.rds")
      )
  } else if (identical(kind, "full_shadow")) {
    logical_rows <- readRDS(file.path(artifact_dir, "logical_ci_results.rds"))
    kind_gate <- nrow(case_results) == 44941L &&
      nrow(logical_rows) == 177952L &&
      !any(logical_rows$decision_flip) &&
      isTRUE(summary$oracle_residual_gate) &&
      isTRUE(summary$downstream_decision_gate) && identical(
        evidence$schema_version,
        "full-cuda-ci-single-penalty-gcv-full-shadow-merged-v1"
      ) && nrow(utils::read.csv(
        file.path(artifact_dir, "partition_hashes.csv")
      )) == 16L
  } else {
    gpu <- utils::read.csv(
      file.path(artifact_dir, "gpu_samples.csv"),
      stringsAsFactors = FALSE, check.names = FALSE
    )
    backend_validation <- tryCatch(
      fastkpc_full_cuda_phase4_validate_backend_evidence(evidence),
      error = identity
    )
    kind_gate <- nrow(case_results) == 44941L &&
      nrow(gpu) > 0L && identical(unique(gpu$physical_gpu_index), 0L) &&
      max(gpu$utilization_percent) > 0 && summary$repetition_count >= 5L &&
      summary$candidate_median_ms <= 25000 &&
      summary$candidate_median_ms < summary$baseline_median_ms &&
      isTRUE(summary$all_candidate_targets_identical) &&
      isTRUE(summary$all_candidate_counters_identical) &&
      isTRUE(summary$absolute_performance_gate) &&
      isTRUE(summary$relative_performance_gate) &&
      isTRUE(summary$backend_gate) && identical(
        evidence$schema_version,
        "full-cuda-ci-phase4-backend-evidence-v1"
      ) && !inherits(backend_validation, "error")
  }
  fastkpc_full_cuda_phase4_publication_require(
    kind_gate, paste0("Phase 4 ", kind, " artifact gate failed")
  )

  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase4_publication_source_closure()
    native <- fastkpc_full_cuda_phase4_publication_native_identity()
    current_gate <- identical(current$sha256, closure_sha256) &&
      identical(current$table, source_closure) &&
      identical(native$sha256, producer$native_binary_sha256)
    fastkpc_full_cuda_phase4_publication_require(
      current_gate, "Phase 4 artifact does not match current sources/binary"
    )
    implementation_paths <- c(
      "fastkpc/src/cuda/mgcv_single_penalty_gcv.cu",
      "fastkpc/src/r_api_cuda.cpp",
      "fastkpc/R/full_cuda_ci_single_penalty_gcv.R"
    )
    implementation <- paste(unlist(lapply(
      implementation_paths, readLines, warn = FALSE
    )), collapse = "\n")
    regression_keys <- c(
      "0df3599d3b6c6ea6e3c99f5b80fbbb51b70797494699116f33d88fa3eca808d1",
      "45b718aaf9dab33661f5ed6ca512bcfafca0371036381be0f3045d05d3a8bebb",
      "dbd7c0cce091b2d0337397f83aa9504ff0e1c9e69eaa2a1666306b92f343d5c8"
    )
    fastkpc_full_cuda_phase4_publication_require(
      !any(vapply(
        regression_keys, grepl, logical(1L), x = implementation,
        fixed = TRUE
      )),
      "Phase 4 implementation contains setup-key-specific routing"
    )
  }
  list(
    artifact_dir = artifact_dir,
    kind = kind,
    manifest = manifest,
    summary = summary,
    producer = producer,
    source_closure = source_closure
  )
}

fastkpc_full_cuda_phase4_publish_one_artifact <- function(
    kind, output_dir, catalog, oracle, shadow, backend, gpu_samples,
    evidence_paths, evidence_hashes,
    contracts, source_closure, native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase4_artifact_kinds())
  producer_bundle <- fastkpc_full_cuda_phase4_publication_producer(
    kind, catalog, source_closure, native_identity, contracts
  )
  output_parent <- dirname(output_dir)
  dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
  stage_dir <- tempfile(
    paste0(".phase4-", kind, "-stage-"), tmpdir = output_parent
  )
  dir.create(stage_dir, recursive = TRUE)
  stage_active <- TRUE
  on.exit({
    if (stage_active && dir.exists(stage_dir)) {
      unlink(stage_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)

  summary <- fastkpc_full_cuda_phase4_publication_common_summary(
    kind, oracle, shadow, backend, producer_bundle, contracts,
    source_closure, native_identity, gpu_samples
  )
  source_evidence_path <- evidence_paths[[kind]]
  fastkpc_full_cuda_phase4_publication_write_kind_payload(
    kind, stage_dir, catalog, oracle, shadow, backend, gpu_samples,
    source_evidence_path
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    source_closure$table, stage_dir, "source_closure.csv"
  )
  evidence_inputs <- data.frame(
    evidence_id = names(evidence_hashes),
    sha256 = unlist(evidence_hashes, use.names = FALSE),
    semantic_role = c(
      "objective-and-optimizer-authority",
      "residual-dcov-and-graph-authority",
      "determinism-performance-and-backend-authority",
      "physical-gpu-zero-execution-samples"
    ),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase4_publication_write_table(
    evidence_inputs, stage_dir, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_write_json(
    summary, file.path(stage_dir, "summary.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$producer,
    file.path(stage_dir, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$backend$value,
    file.path(stage_dir, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$build$value,
    file.path(stage_dir, "build_recipe.json")
  )
  fastkpc_full_cuda_write_summary_md(
    summary, file.path(stage_dir, "summary.md"),
    paste0("Full CUDA CI Phase 4 ", kind)
  )
  writeLines(
    c(
      paste0("source evidence SHA-256: ", evidence_hashes[[kind]]),
      paste0("producer source closure SHA-256: ", source_closure$sha256),
      paste0("native binary SHA-256: ", native_identity$sha256),
      "see fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_artifacts.R"
    ),
    file.path(stage_dir, "commands.txt"), useBytes = TRUE
  )
  environment_lines <- c(
    fastkpc_full_cuda_environment_lines(),
    paste0("physical_gpu_index=0"),
    paste0("physical_gpu_uuid=", unique(gpu_samples$gpu_uuid)),
    paste0("physical_gpu_name=", unique(gpu_samples$gpu_name)),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256)
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
    "fastkpc/R/full_cuda_ci_phase4_backend.R",
    "fastkpc/R/full_cuda_ci_phase4_publication.R"
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
      "full-cuda-ci-phase4-single-penalty-gcv-validator-v1",
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
    cuda_context_id = paste0(
      unique(gpu_samples$gpu_uuid), ":physical-0:logical-0"
    ),
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
      "full-cuda-ci-phase4-single-penalty-gcv-artifact-manifest-v1",
    artifact_kind = kind,
    claim_scope = "phase4-single-penalty-S-size-at-most-2",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_manifest_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(
    manifest, file.path(stage_dir, "manifest.json")
  )
  fastkpc_full_cuda_phase4_validate_artifact(
    stage_dir, expected_kind = kind, verify_current_sources = TRUE
  )

  backup_dir <- NULL
  if (dir.exists(output_dir)) {
    backup_dir <- tempfile(
      paste0(".phase4-", kind, "-backup-"), tmpdir = output_parent
    )
    fastkpc_full_cuda_phase4_publication_require(
      file.rename(output_dir, backup_dir),
      "Phase 4 prior artifact could not be staged for replacement"
    )
  }
  published <- file.rename(stage_dir, output_dir)
  if (!published) {
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop("Phase 4 artifact publication failed", call. = FALSE)
  }
  stage_active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase4_validate_artifact(
      output_dir, expected_kind = kind, verify_current_sources = TRUE
    ),
    error = identity
  )
  if (inherits(validated, "error")) {
    failed_dir <- tempfile(
      paste0(".phase4-", kind, "-failed-"), tmpdir = output_parent
    )
    file.rename(output_dir, failed_dir)
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup_dir)) unlink(backup_dir, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase4_publish_artifacts <- function(
    catalog, oracle_path, shadow_path, backend_path, gpu_samples_path,
    output_root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  evidence_paths <- c(
    oracle = normalizePath(oracle_path, winslash = "/", mustWork = TRUE),
    full_shadow = normalizePath(
      shadow_path, winslash = "/", mustWork = TRUE
    ),
    backend = normalizePath(backend_path, winslash = "/", mustWork = TRUE)
  )
  gpu_samples_path <- normalizePath(
    gpu_samples_path, winslash = "/", mustWork = TRUE
  )
  oracle <- readRDS(evidence_paths[["oracle"]])
  shadow <- readRDS(evidence_paths[["full_shadow"]])
  backend <- readRDS(evidence_paths[["backend"]])
  gpu_samples <- fastkpc_full_cuda_phase4_read_gpu_samples(gpu_samples_path)
  fastkpc_full_cuda_phase4_validate_publication_inputs(
    oracle, shadow, backend
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase4_publication_source_closure()
  native_identity <- fastkpc_full_cuda_phase4_publication_native_identity()
  evidence_hashes <- c(
    setNames(as.list(vapply(
      evidence_paths, fastkpc_full_cuda_census_file_hash, character(1L)
    )), names(evidence_paths)),
    gpu_samples = list(fastkpc_full_cuda_census_file_hash(gpu_samples_path))
  )
  directories <- fastkpc_full_cuda_phase4_artifact_directory_names()
  results <- lapply(fastkpc_full_cuda_phase4_artifact_kinds(), function(kind) {
    fastkpc_full_cuda_phase4_publish_one_artifact(
      kind = kind,
      output_dir = file.path(output_root, directories[[kind]]),
      catalog = catalog,
      oracle = oracle,
      shadow = shadow,
      backend = backend,
      gpu_samples = gpu_samples,
      evidence_paths = evidence_paths,
      evidence_hashes = evidence_hashes,
      contracts = contracts,
      source_closure = source_closure,
      native_identity = native_identity
    )
  })
  names(results) <- fastkpc_full_cuda_phase4_artifact_kinds()
  results
}
