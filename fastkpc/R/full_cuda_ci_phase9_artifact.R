fastkpc_full_cuda_phase9_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase9_artifact_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "one_call_full_cuda_351x48_default_inf_v2")
}

fastkpc_full_cuda_phase9_required_files <- function() {
  c(
    "manifest.json", "summary.json", "summary.md", "commands.txt",
    "environment.txt", "graph_agreement.csv", "sepset_agreement.csv",
    "n_edgetests.csv", "deletion_trace.csv", "first_divergence.json",
    "fallbacks.csv", "stage_timing.csv", "raw_runs.csv",
    "case_results.csv", "near_alpha_results.csv",
    "rank_condition_results.csv", "cache.csv", "evidence_inputs.csv",
    "source_closure.csv", "source_evidence.rds", "producer_identity.json",
    "backend_configuration.json", "build_recipe.json",
    "validator_attestations.json", "execution_receipts.json",
    "adjacency.rds", "sepsets.rds", "pmax.rds",
    "logical_ci_trace.rds"
  )
}

fastkpc_full_cuda_phase9_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase8_publication_source_paths(),
    "fastkpc/R/fast_kpc.R",
    "fastkpc/R/full_cuda_ci_phase9_artifact.R",
    "fastkpc/src/full_cuda_ci_one_call.cpp",
    "fastkpc/src/full_cuda_ci_one_call.hpp",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.cpp",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.hpp",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_extended.hpp",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_small64.cu",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_ext192.cu",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_ext384.cu",
    "fastkpc/src/cuda/mgcv_multi_penalty_gcv_ext559.cu",
    "fastkpc/tests/test_full_cuda_ci_one_call.R",
    "fastkpc/tests/test_full_cuda_ci_default_inf_production.R",
    "fastkpc/tests/test_full_cuda_ci_one_call_artifact.R",
    "fastkpc/tools/run_full_cuda_ci_default_inf_oracle.R",
    "fastkpc/tools/run_full_cuda_ci_phase9.R"
  )), method = "radix")
  fastkpc_full_cuda_phase9_require(
    all(file.exists(paths) & !dir.exists(paths)) && !anyDuplicated(paths),
    "Phase 9 source closure is incomplete"
  )
  paths
}

fastkpc_full_cuda_phase9_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase9_source_paths()
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

fastkpc_full_cuda_phase9_backend_configuration <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase9-backend-configuration-v2",
    route = "compatible.cuda",
    native_entrypoint = "compatible-cuda-full-skeleton-native-v1",
    native_call_count = 1L,
    scheduler = "reachable-edge-round-canonical-replay-v1",
    setup_backend = "phase7-native-cpp",
    single_penalty_backend = "phase4-live-cuda-gcv",
    multi_penalty_backend = "phase6-live-cuda-gcv",
    residual_backend = "phase3-device-resident-fixed-sp",
    dcov_backend = "phase8-exact-screen-guarded-full-eig-cuda",
    prepared_setup_cache_capacity = 64L,
    component_capacity = 47L,
    guard_lower_inclusive = "0.05",
    guard_upper_inclusive = "0.15",
    alpha = "0.1",
    index = 1L,
    num_col = 35L,
    requested_max_conditioning_size = "Inf",
    resolved_max_conditioning_size = 46L,
    natural_stop_level = 8L,
    multi_penalty_capacity_buckets = "64/7|80/8|192/21|384/42|559/62",
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE,
    explicit_route_only = TRUE,
    phase10_performance_claim = FALSE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_phase35_sha256_utf8(
      fastkpc_full_cuda_phase35_canonical_json(value)
    )
  )
}

fastkpc_full_cuda_phase9_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase9-build-recipe-v2",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
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

fastkpc_full_cuda_phase9_label_result <- function(result, labels) {
  labels <- as.character(labels)
  dimnames(result$adjacency) <- list(labels, labels)
  dimnames(result$pMax) <- list(labels, labels)
  names(result$sepsets) <- labels
  result$sepsets <- lapply(result$sepsets, function(row) {
    names(row) <- labels
    row
  })
  result
}

fastkpc_full_cuda_phase9_result_from_raw <- function(raw) {
  if (is.list(raw) && fastkpc_full_cuda_is_skeleton(raw$result)) {
    return(raw$result)
  }
  if (fastkpc_full_cuda_is_skeleton(raw)) return(raw)
  stop("Phase 9 evidence does not contain a candidate skeleton", call. = FALSE)
}

fastkpc_full_cuda_phase9_elapsed <- function(raw, result) {
  timing <- raw$timing
  elapsed <- if (!is.null(timing) && "elapsed" %in% names(timing)) {
    as.numeric(timing[["elapsed"]])
  } else {
    as.numeric(result$summary$elapsed_sec)
  }
  fastkpc_full_cuda_phase9_require(
    length(elapsed) == 1L && is.finite(elapsed) && elapsed > 0,
    "Phase 9 evidence elapsed time is missing"
  )
  elapsed
}

fastkpc_full_cuda_phase9_load_logical <- function(path) {
  fastkpc_full_cuda_phase9_require(
    file.exists(path), "Phase 9 canonical logical trace is missing"
  )
  logical <- readRDS(path)
  fastkpc_full_cuda_phase9_require(
    is.data.frame(logical) && nrow(logical) > 0L,
    "Phase 9 canonical logical trace is malformed"
  )
  if (!"S_size" %in% names(logical)) {
    logical$S_size <- as.integer(logical$level)
  }
  if (!"reference_p_value" %in% names(logical)) {
    logical$reference_p_value <- as.numeric(logical$p_value)
  }
  if (!"reference_independent" %in% names(logical)) {
    logical$reference_independent <- as.logical(
      logical$reference_p_value >= 0.1
    )
  }
  logical
}

fastkpc_full_cuda_phase9_validate_evidence <- function(
    raw,
    data_path = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    ),
    oracle_dir = fastkpc_full_cuda_default_kpcalg_oracle_dir(),
    logical_path = file.path(oracle_dir, "logical_ci_trace.rds")) {
  fastkpc_full_cuda_phase9_require(
    file.exists(data_path) && file.exists(logical_path),
    "Phase 9 canonical inputs are missing"
  )
  data <- readRDS(data_path)
  fastkpc_full_cuda_phase9_require(
    is.matrix(data) && identical(dim(data), c(351L, 48L)) &&
      !is.null(colnames(data)),
    "Phase 9 canonical data identity is malformed"
  )
  result <- fastkpc_full_cuda_phase9_label_result(
    fastkpc_full_cuda_phase9_result_from_raw(raw), colnames(data)
  )
  elapsed <- fastkpc_full_cuda_phase9_elapsed(raw, result)
  result$summary$elapsed_sec <- elapsed
  oracle <- fastkpc_load_full_cuda_ci_oracle(oracle_dir)
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(oracle, result)
  logical <- fastkpc_full_cuda_phase9_load_logical(logical_path)
  tasks <- result$tasks
  contract <- fastkpc_full_cuda_default_kpcalg_contract()
  expected_n_edgetests <- contract$n_edgetests
  structural_fields_exact <- c(
    logical_id = identical(
      as.integer(tasks$canonical_test_order_id),
      as.integer(logical$logical_sequence_id)
    ),
    level = identical(as.integer(tasks$level), as.integer(logical$level)),
    task_index = identical(
      as.integer(tasks$task_index), as.integer(logical$source_task_index)
    ),
    x = identical(as.integer(tasks$x), as.integer(logical$x)),
    y = identical(as.integer(tasks$y), as.integer(logical$y)),
    S_key = identical(as.character(tasks$S_key), as.character(logical$S_key)),
    S_size = identical(
      as.integer(tasks$conditioning_size), as.integer(logical$S_size)
    ),
    deletion = identical(
      as.logical(tasks$native_edge_deleted), as.logical(logical$deletes_edge)
    )
  )
  candidate_independent <- tasks$p_used >= 0.1
  decision_flip <- candidate_independent != logical$reference_independent
  zero_fields <- c(
    "r_callback_count", "legacy_mgcv_fit_count",
    "legacy_mgcv_setup_count", "cpu_residual_solve_count",
    "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
    "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count",
    "cpu_spectra_count", "residual_d2h_bytes",
    "unknown_fallback_count", "approximate_backend_count"
  )
  zero_values <- vapply(zero_fields, function(field) {
    value <- result$summary[[field]]
    if (is.null(value)) return(NA_real_)
    as.numeric(value)
  }, numeric(1L))
  hard_gate <- isTRUE(comparison$summary$pass) &&
    identical(as.integer(result$n.edgetests), expected_n_edgetests) &&
    nrow(tasks) == contract$logical_test_count &&
    all(structural_fields_exact) &&
    !any(decision_flip) && all(is.finite(tasks$p_used)) &&
    all(!is.na(zero_values) & zero_values == 0) &&
    identical(result$summary$run_status, "ok") &&
    identical(result$summary$entrypoint,
              "compatible-cuda-full-skeleton-native-v1") &&
    identical(result$summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(result$summary$compatible_cuda_strict) &&
    as.integer(result$summary$native_call_count) == 1L &&
    identical(result$summary$max_conditioning_size_requested,
              contract$requested_max_conditioning_size) &&
    as.integer(result$summary$max_conditioning_size_resolved) ==
      contract$resolved_max_conditioning_size &&
    max(as.integer(result$levels$level)) == contract$natural_stop_level &&
    as.integer(result$summary$logical_tests_consumed) ==
      contract$logical_test_count &&
    as.integer(result$summary$physical_tests_evaluated) ==
      contract$physical_test_count &&
    as.integer(result$summary$guarded_pair_count) ==
      contract$guarded_pair_count &&
    as.integer(result$summary$cuda_dcov_pair_count) ==
      contract$physical_test_count &&
    as.integer(result$summary$cuda_gamma_pvalue_count) ==
      contract$physical_test_count &&
    result$summary$cuda_single_penalty_target_count > 0L &&
    result$summary$cuda_multi_penalty_target_count > 0L &&
    result$summary$native_setup_cache_request_count ==
      result$summary$native_setup_cache_hit_count +
        result$summary$native_setup_cache_miss_count
  fastkpc_full_cuda_phase9_require(
    hard_gate, "Phase 9 one-call full canonical evidence gate failed"
  )
  captured_at <- if (!is.null(raw$started_utc)) {
    as.character(raw$started_utc)
  } else if (!is.null(raw$captured_at_utc)) {
    as.character(raw$captured_at_utc)
  } else {
    format(Sys.time(), tz = "UTC", usetz = TRUE)
  }
  list(
    schema_version = "full-cuda-ci-phase9-one-call-evidence-v2",
    captured_at_utc = captured_at,
    validated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    source_commit = fastkpc_full_cuda_source_commit(),
    data_path = data_path,
    data_sha256 = fastkpc_full_cuda_census_file_hash(data_path),
    oracle_dir = oracle_dir,
    oracle_manifest_sha256 = fastkpc_full_cuda_census_file_hash(
      file.path(oracle_dir, "manifest.json")
    ),
    logical_path = logical_path,
    logical_sha256 = fastkpc_full_cuda_census_file_hash(logical_path),
    elapsed_sec = elapsed,
    result = result,
    comparison_summary = comparison$summary,
    first_divergence = comparison$first_divergence,
    structural_fields_exact = structural_fields_exact,
    decision_flip_count = sum(decision_flip),
    authority_zero_values = zero_values,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase9_capture <- function(
    data_path = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    )) {
  data <- readRDS(data_path)
  started <- Sys.time()
  timing <- system.time(result <- fastkpc_compatible_cuda_skeleton(
    data = data,
    alpha = 0.1,
    labels = colnames(data),
    options = list(
      route = "full_cuda",
      compatible_cuda_strict = TRUE,
      max_conditioning_size = Inf,
      index = 1,
      numCol = 35L,
      trace_level = "logical"
    )
  ))
  list(
    started_utc = format(started, tz = "UTC", usetz = TRUE),
    timing = timing,
    result = result
  )
}

fastkpc_full_cuda_phase9_producer <- function(
    evidence, source_closure, native_identity, contracts) {
  backend <- fastkpc_full_cuda_phase9_backend_configuration()
  build <- fastkpc_full_cuda_phase9_build_recipe()
  corpus_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase9-canonical-corpus-v2",
    dataset_sha256 = evidence$data_sha256,
    logical_sha256 = evidence$logical_sha256,
    logical_test_count =
      fastkpc_full_cuda_default_kpcalg_contract()$logical_test_count
  ))
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version =
      "full-cuda-ci-one-call-compatible-skeleton-default-inf-v2",
    dataset_or_corpus_sha256 = corpus_sha256,
    oracle_sha256 = evidence$oracle_manifest_sha256,
    backend_configuration_sha256 = backend$sha256,
    build_recipe_sha256 = build$sha256,
    contracts = contracts
  )
  list(producer = producer, backend = backend, build = build)
}

fastkpc_full_cuda_phase9_summary <- function(
    evidence, producer_bundle, evidence_sha256, contracts) {
  source <- evidence$result$summary
  comparison <- evidence$comparison_summary
  list(
    schema_version = "full-cuda-ci-phase9-one-call-summary-v2",
    run_status = "ok",
    timeout = FALSE,
    source_commit = evidence$source_commit,
    oracle_artifact = evidence$oracle_dir,
    candidate_route = "compatible.cuda/full_cuda-explicit",
    edge_count_reference = comparison$edge_count_reference,
    edge_count_candidate = comparison$edge_count_candidate,
    SHD = comparison$SHD,
    adjacency_identical = comparison$adjacency_identical,
    sepsets_identical = comparison$sepsets_identical,
    n_edgetests_identical = comparison$n_edgetests_identical,
    deletions_identical = comparison$deletions_identical,
    logical_ci_trace_identical = comparison$logical_ci_trace_identical,
    max_conditioning_size_requested =
      source$max_conditioning_size_requested,
    max_conditioning_size_resolved =
      source$max_conditioning_size_resolved,
    natural_stop_level = max(as.integer(evidence$result$levels$level)),
    logical_tests_consumed = source$logical_tests_consumed,
    physical_tests_evaluated = source$physical_tests_evaluated,
    guarded_pair_count = source$guarded_pair_count,
    cuda_single_penalty_target_count = source$cuda_single_penalty_target_count,
    cuda_multi_penalty_target_count = source$cuda_multi_penalty_target_count,
    cuda_residual_batch_count = source$cuda_residual_batch_count,
    logical_residual_requests = source$logical_residual_requests,
    physical_residual_fits = source$physical_residual_fits,
    native_setup_count = source$native_setup_count,
    native_setup_cache_capacity = source$native_setup_cache_capacity,
    native_setup_cache_request_count =
      source$native_setup_cache_request_count,
    native_setup_cache_hit_count = source$native_setup_cache_hit_count,
    native_setup_cache_miss_count = source$native_setup_cache_miss_count,
    native_setup_cache_eviction_count =
      source$native_setup_cache_eviction_count,
    component_cache_capacity = source$component_cache_capacity,
    component_cache_request_count = source$component_cache_request_count,
    component_cache_hit_count = source$component_cache_hit_count,
    component_cache_miss_count = source$component_cache_miss_count,
    component_cache_eviction_count = source$component_cache_eviction_count,
    r_callback_count = source$r_callback_count,
    legacy_mgcv_fit_count = source$legacy_mgcv_fit_count,
    legacy_mgcv_setup_count = source$legacy_mgcv_setup_count,
    cpu_residual_solve_count = source$cpu_residual_solve_count,
    cpu_dcov_component_count = source$cpu_dcov_component_count,
    cpu_dcov_eigen_or_lowrank_count = source$cpu_dcov_eigen_or_lowrank_count,
    cpu_dcov_pair_stat_count = source$cpu_dcov_pair_stat_count,
    cpu_gamma_pvalue_count = source$cpu_gamma_pvalue_count,
    cpu_spectra_count = source$cpu_spectra_count,
    cuda_dcov_component_count = source$cuda_dcov_component_count,
    cuda_dcov_pair_count = source$cuda_dcov_pair_count,
    cuda_gamma_pvalue_count = source$cuda_gamma_pvalue_count,
    residual_d2h_bytes = source$residual_d2h_bytes,
    component_d2h_bytes = source$component_d2h_bytes,
    unknown_fallback_count = source$unknown_fallback_count,
    approximate_backend_count = source$approximate_backend_count,
    architecture_contract_sha256 = contracts$architecture_contract_v1$sha256,
    numerical_contract_sha256 = contracts$numerical_contract_v1$sha256,
    artifact_identity_contract_sha256 =
      contracts$artifact_identity_contract_v1$sha256,
    reference_machine_contract_sha256 =
      contracts$reference_machine_v1$sha256,
    performance_budget_contract_sha256 =
      contracts$performance_budget_v1$sha256,
    elapsed_sec = evidence$elapsed_sec,
    phase9_correctness_gate = TRUE,
    phase10_performance_gate = evidence$elapsed_sec <= 120,
    source_evidence_sha256 = evidence_sha256,
    producer_identity_sha256 = producer_bundle$producer$identity_sha256,
    source_closure_sha256 =
      producer_bundle$producer$producer_source_closure_sha256,
    native_binary_sha256 = producer_bundle$producer$native_binary_sha256,
    pass = TRUE
  )
}

fastkpc_full_cuda_phase9_cases <- function(evidence) {
  tasks <- evidence$result$tasks
  logical <- fastkpc_full_cuda_phase9_load_logical(evidence$logical_path)
  data.frame(
    logical_sequence_id = as.integer(tasks$canonical_test_order_id),
    level = as.integer(tasks$level),
    source_task_index = as.integer(tasks$task_index),
    x = as.integer(tasks$x),
    y = as.integer(tasks$y),
    S_key = as.character(tasks$S_key),
    reference_p_value = as.numeric(logical$reference_p_value),
    candidate_p_value = as.numeric(tasks$p_used),
    absolute_p_value_error = abs(
      as.numeric(tasks$p_used) - as.numeric(logical$reference_p_value)
    ),
    reference_independent = as.logical(logical$reference_independent),
    candidate_independent = as.logical(tasks$p_used >= 0.1),
    decision_flip = as.logical(
      (tasks$p_used >= 0.1) != logical$reference_independent
    ),
    deletes_edge = as.logical(tasks$native_edge_deleted),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase9_write_table <- function(value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase9_cache_table <- function(summary) {
  data.frame(
    cache = c("native-prepared-setup", "cuda-dcov-component"),
    capacity = c(
      summary$native_setup_cache_capacity,
      summary$component_cache_capacity
    ),
    requests = c(
      summary$native_setup_cache_request_count,
      summary$component_cache_request_count
    ),
    hits = c(
      summary$native_setup_cache_hit_count,
      summary$component_cache_hit_count
    ),
    misses = c(
      summary$native_setup_cache_miss_count,
      summary$component_cache_miss_count
    ),
    evictions = c(
      summary$native_setup_cache_eviction_count,
      summary$component_cache_eviction_count
    ),
    bounded = TRUE,
    semantic_eviction_effect = "none",
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase9_evidence_inputs <- function(evidence) {
  paths <- c(
    canonical_dataset = evidence$data_path,
    oracle_manifest = file.path(evidence$oracle_dir, "manifest.json"),
    canonical_logical = evidence$logical_path,
    phase8_backend_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci", "dcov_cuda_backend_v1",
      "manifest.json"
    )
  )
  fastkpc_full_cuda_phase9_require(
    all(file.exists(paths)), "Phase 9 evidence lineage input is missing"
  )
  data.frame(
    input_kind = names(paths),
    input_file = unname(paths),
    sha256 = unname(vapply(
      paths, fastkpc_full_cuda_census_file_hash, character(1L)
    )),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase9_validate_summary <- function(summary) {
  expected_numerical_zero <- c(
    "r_callback_count", "legacy_mgcv_fit_count",
    "legacy_mgcv_setup_count", "cpu_residual_solve_count",
    "cpu_dcov_component_count", "cpu_dcov_eigen_or_lowrank_count",
    "cpu_dcov_pair_stat_count", "cpu_gamma_pvalue_count",
    "cpu_spectra_count", "residual_d2h_bytes", "component_d2h_bytes",
    "unknown_fallback_count", "approximate_backend_count"
  )
  hash_fields <- c(
    "architecture_contract_sha256", "numerical_contract_sha256",
    "artifact_identity_contract_sha256",
    "reference_machine_contract_sha256",
    "performance_budget_contract_sha256", "source_evidence_sha256",
    "producer_identity_sha256", "source_closure_sha256",
    "native_binary_sha256"
  )
  clean <- is.list(summary) &&
    identical(summary$schema_version,
              "full-cuda-ci-phase9-one-call-summary-v2") &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    isTRUE(summary$logical_ci_trace_identical) &&
    identical(summary$max_conditioning_size_requested,
              fastkpc_full_cuda_default_kpcalg_contract()[[
                "requested_max_conditioning_size"
              ]]) &&
    summary$max_conditioning_size_resolved == 46L &&
    summary$natural_stop_level == 8L &&
    summary$logical_tests_consumed == 240498L &&
    summary$physical_tests_evaluated == 241686L &&
    summary$guarded_pair_count == 1188L &&
    summary$cuda_dcov_pair_count == 241686L &&
    summary$cuda_gamma_pvalue_count == 241686L &&
    summary$cuda_single_penalty_target_count > 0L &&
    summary$cuda_multi_penalty_target_count > 0L &&
    summary$native_setup_cache_request_count ==
      summary$native_setup_cache_hit_count +
        summary$native_setup_cache_miss_count &&
    all(vapply(expected_numerical_zero, function(field) {
      identical(as.numeric(summary[[field]]), 0)
    }, logical(1L))) &&
    is.finite(summary$elapsed_sec) && summary$elapsed_sec > 0 &&
    isTRUE(summary$phase9_correctness_gate) && isTRUE(summary$pass) &&
    all(vapply(hash_fields, function(field) {
      is.character(summary[[field]]) && length(summary[[field]]) == 1L &&
        grepl("^[0-9a-f]{64}$", summary[[field]])
    }, logical(1L)))
  fastkpc_full_cuda_phase9_require(
    clean, "Phase 9 artifact summary gate failed"
  )
  invisible(TRUE)
}

fastkpc_full_cuda_phase9_publish <- function(
    raw,
    output_dir = fastkpc_full_cuda_phase9_artifact_dir(),
    data_path = file.path(
      "fastkpc", "artifacts", "kpc_tprs_real_zhu",
      "cancer_RD-causalDiscoveryInput.rds"
    )) {
  evidence <- fastkpc_full_cuda_phase9_validate_evidence(
    raw, data_path = data_path
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase9_source_closure()
  native_identity <- fastkpc_full_cuda_phase7_native_identity()
  producer_bundle <- fastkpc_full_cuda_phase9_producer(
    evidence, source_closure, native_identity, contracts
  )
  parent <- dirname(output_dir)
  dir.create(parent, recursive = TRUE, showWarnings = FALSE)
  stage <- tempfile(".phase9-stage-", tmpdir = parent)
  dir.create(stage, recursive = TRUE)
  active <- TRUE
  on.exit({
    if (active && dir.exists(stage)) unlink(stage, recursive = TRUE, force = TRUE)
  }, add = TRUE)

  saveRDS(evidence, file.path(stage, "source_evidence.rds"), compress = "xz")
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "source_evidence.rds")
  )
  summary <- fastkpc_full_cuda_phase9_summary(
    evidence, producer_bundle, evidence_sha256, contracts
  )
  fastkpc_full_cuda_phase9_validate_summary(summary)
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    fastkpc_load_full_cuda_ci_oracle(evidence$oracle_dir), evidence$result
  )
  cases <- fastkpc_full_cuda_phase9_cases(evidence)
  near <- cases[
    cases$reference_p_value >= 0.05 & cases$reference_p_value <= 0.15,
    , drop = FALSE
  ]
  fastkpc_full_cuda_phase9_write_table(
    comparison$graph_agreement, stage, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    comparison$sepset_agreement, stage, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    comparison$n_edgetests, stage, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    comparison$candidate_deletions, stage, "deletion_trace.csv"
  )
  fastkpc_full_cuda_write_json(
    comparison$first_divergence, file.path(stage, "first_divergence.json")
  )
  fastkpc_full_cuda_phase9_write_table(data.frame(
    fallback_class = c("unknown", "approximate", "cpu-numerical"),
    count = 0L,
    accepted_for_phase9 = FALSE,
    stringsAsFactors = FALSE
  ), stage, "fallbacks.csv")
  fastkpc_full_cuda_phase9_write_table(
    evidence$result$levels, stage, "stage_timing.csv"
  )
  fastkpc_full_cuda_phase9_write_table(data.frame(
    run = 1L,
    route = "compatible.cuda/full_cuda-explicit",
    elapsed_sec = evidence$elapsed_sec,
    correctness_pass = TRUE,
    phase10_performance_pass = evidence$elapsed_sec <= 120,
    stringsAsFactors = FALSE
  ), stage, "raw_runs.csv")
  fastkpc_full_cuda_phase9_write_table(cases, stage, "case_results.csv")
  fastkpc_full_cuda_phase9_write_table(
    near, stage, "near_alpha_results.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    evidence$result$levels, stage, "rank_condition_results.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    fastkpc_full_cuda_phase9_cache_table(evidence$result$summary),
    stage, "cache.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    fastkpc_full_cuda_phase9_evidence_inputs(evidence),
    stage, "evidence_inputs.csv"
  )
  fastkpc_full_cuda_phase9_write_table(
    source_closure$table, stage, "source_closure.csv"
  )
  saveRDS(comparison$candidate_adjacency, file.path(stage, "adjacency.rds"))
  saveRDS(evidence$result$sepsets, file.path(stage, "sepsets.rds"))
  saveRDS(evidence$result$pMax, file.path(stage, "pmax.rds"))
  saveRDS(
    comparison$candidate_logical, file.path(stage, "logical_ci_trace.rds")
  )
  fastkpc_full_cuda_write_json(summary, file.path(stage, "summary.json"))
  fastkpc_full_cuda_write_summary_md(
    summary, file.path(stage, "summary.md"),
    "Full CUDA CI Phase 9 one-call skeleton"
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$producer, file.path(stage, "producer_identity.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$backend$value,
    file.path(stage, "backend_configuration.json")
  )
  fastkpc_full_cuda_write_json(
    producer_bundle$build$value, file.path(stage, "build_recipe.json")
  )
  writeLines(c(
    "bash fastkpc/tools/build_cuda_native.sh",
    "Rscript fastkpc/tools/run_full_cuda_ci_phase9.R",
    paste0("source_evidence_sha256=", evidence_sha256),
    paste0("native_binary_sha256=", native_identity$sha256)
  ), file.path(stage, "commands.txt"), useBytes = TRUE)
  writeLines(c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase9_route=compatible.cuda/full_cuda-explicit",
    "phase9_native_call_count=1",
    "CUDA_VISIBLE_DEVICES=0", "OPENBLAS_NUM_THREADS=1",
    "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1"
  ), file.path(stage, "environment.txt"), useBytes = TRUE)

  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic_files <- sort(setdiff(
    list.files(stage, all.files = FALSE, no.. = TRUE), excluded
  ), method = "radix")
  payload_hashes <- setNames(lapply(
    file.path(stage, semantic_files), fastkpc_full_cuda_census_file_hash
  ), semantic_files)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  envelope <- fastkpc_full_cuda_phase35_identity_envelope(
    producer_bundle$producer, payload_sha256
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(stage, "environment.txt")
  )
  timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  attestation <- fastkpc_full_cuda_phase35_validator_attestation(
    producer = producer_bundle$producer,
    validator_source_closure_sha256 = source_closure$sha256,
    validator_semantic_version = "full-cuda-ci-phase9-validator-v2",
    validator_contracts = contracts,
    validation_timestamp_utc = timestamp,
    environment_sha256 = environment_sha256,
    validation_result = "PASS"
  )
  info <- file.info(stage, extra_cols = TRUE)
  inode <- if ("ino" %in% names(info)) as.character(info$ino[[1L]]) else
    "unavailable"
  final_path <- file.path(
    normalizePath(parent, winslash = "/", mustWork = TRUE),
    basename(output_dir)
  )
  receipt <- fastkpc_full_cuda_phase35_execution_receipt(
    producer = producer_bundle$producer,
    pid = as.integer(Sys.getpid()),
    session_id = paste0("R-", Sys.getpid(), "-", format(Sys.time(), "%s")),
    cuda_context_id = "cuda-device-0-phase9-one-call",
    artifact_path = final_path,
    artifact_inode = inode,
    staging_path = normalizePath(stage, winslash = "/", mustWork = TRUE),
    recorded_at_utc = timestamp
  )
  fastkpc_full_cuda_write_json(
    list(attestations = list(attestation)),
    file.path(stage, "validator_attestations.json")
  )
  fastkpc_full_cuda_write_json(
    list(execution_receipts = list(receipt)),
    file.path(stage, "execution_receipts.json")
  )
  manifest <- list(
    schema_version = "full-cuda-ci-phase9-artifact-manifest-v2",
    artifact_kind = "one_call_full_cuda_351x48_default_inf",
    claim_scope = "phase9-default-kpcalg-one-call-correctness-and-authority",
    producer_semantic_envelope = envelope,
    payload_manifest_sha256 = payload_sha256,
    payload_file_sha256 = payload_hashes,
    semantic_file_count = length(payload_hashes),
    validator_attestations_file = "validator_attestations.json",
    volatile_receipt_file = "execution_receipts.json",
    environment_file = "environment.txt",
    environment_file_sha256 = environment_sha256
  )
  fastkpc_full_cuda_write_json(manifest, file.path(stage, "manifest.json"))
  fastkpc_full_cuda_phase9_validate_artifact(
    stage, verify_current_sources = TRUE
  )

  backup <- NULL
  if (dir.exists(output_dir)) {
    backup <- tempfile(".phase9-backup-", tmpdir = parent)
    fastkpc_full_cuda_phase9_require(
      file.rename(output_dir, backup),
      "Phase 9 prior artifact could not be staged"
    )
  }
  if (!file.rename(stage, output_dir)) {
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop("Phase 9 artifact publication failed", call. = FALSE)
  }
  active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase9_validate_artifact(
      output_dir, verify_current_sources = TRUE
    ),
    error = identity
  )
  if (inherits(validated, "error")) {
    failed <- tempfile(".phase9-failed-", tmpdir = parent)
    file.rename(output_dir, failed)
    if (!is.null(backup)) file.rename(backup, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup)) unlink(backup, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase9_validate_artifact <- function(
    artifact_dir = fastkpc_full_cuda_phase9_artifact_dir(),
    verify_current_sources = FALSE) {
  artifact_dir <- normalizePath(
    artifact_dir, winslash = "/", mustWork = TRUE
  )
  required <- sort(fastkpc_full_cuda_phase9_required_files(), method = "radix")
  actual <- sort(
    list.files(artifact_dir, all.files = FALSE, no.. = TRUE),
    method = "radix"
  )
  fastkpc_full_cuda_phase9_require(
    identical(actual, required),
    "Phase 9 artifact standard file set is incomplete"
  )
  manifest <- jsonlite::read_json(
    file.path(artifact_dir, "manifest.json"), simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase9_require(
    identical(manifest$schema_version,
              "full-cuda-ci-phase9-artifact-manifest-v2") &&
      identical(manifest$artifact_kind,
                "one_call_full_cuda_351x48_default_inf") &&
      identical(manifest$claim_scope,
                "phase9-default-kpcalg-one-call-correctness-and-authority"),
    "Phase 9 artifact manifest schema mismatch"
  )
  excluded <- c(
    "manifest.json", "validator_attestations.json",
    "execution_receipts.json", "environment.txt", "commands.txt"
  )
  semantic <- sort(setdiff(required, excluded), method = "radix")
  actual_hashes <- setNames(lapply(
    file.path(artifact_dir, semantic), fastkpc_full_cuda_census_file_hash
  ), semantic)
  payload_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(actual_hashes)
  )
  fastkpc_full_cuda_phase9_require(
    identical(actual_hashes, manifest$payload_file_sha256) &&
      identical(payload_sha256, manifest$payload_manifest_sha256) &&
      as.integer(manifest$semantic_file_count) == length(semantic),
    "Phase 9 artifact payload identity mismatch"
  )
  fastkpc_full_cuda_phase35_validate_identity_envelope(
    manifest$producer_semantic_envelope
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"),
    simplifyVector = FALSE
  )
  fastkpc_full_cuda_phase9_require(
    identical(producer, manifest$producer_semantic_envelope$producer),
    "Phase 9 producer identity drifted"
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
  fastkpc_full_cuda_phase9_require(
    identical(closure_sha256, producer$producer_source_closure_sha256),
    "Phase 9 source closure identity mismatch"
  )
  if (isTRUE(verify_current_sources)) {
    current <- fastkpc_full_cuda_phase9_source_closure()
    fastkpc_full_cuda_phase9_require(
      identical(current$sha256, closure_sha256) &&
        identical(current$hashes, closure_hashes),
      "Phase 9 current source closure drifted"
    )
    fastkpc_full_cuda_phase9_require(
      identical(
        fastkpc_full_cuda_phase7_native_identity()$sha256,
        producer$native_binary_sha256
      ),
      "Phase 9 current native binary drifted"
    )
  }
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  fastkpc_full_cuda_phase9_validate_summary(summary)
  fastkpc_full_cuda_phase9_require(
    identical(summary$source_evidence_sha256,
              fastkpc_full_cuda_census_file_hash(file.path(
                artifact_dir, "source_evidence.rds"
              ))) &&
      identical(summary$producer_identity_sha256, producer$identity_sha256) &&
      identical(summary$source_closure_sha256,
                producer$producer_source_closure_sha256) &&
      identical(summary$native_binary_sha256, producer$native_binary_sha256),
    "Phase 9 summary identity linkage failed"
  )
  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  fastkpc_full_cuda_phase9_validate_evidence(evidence)
  read_table <- function(name) utils::read.csv(
    file.path(artifact_dir, name), stringsAsFactors = FALSE,
    check.names = FALSE
  )
  graph <- read_table("graph_agreement.csv")
  sepsets <- read_table("sepset_agreement.csv")
  tests <- read_table("n_edgetests.csv")
  deletions <- read_table("deletion_trace.csv")
  cases <- read_table("case_results.csv")
  cache <- read_table("cache.csv")
  fallbacks <- read_table("fallbacks.csv")
  contract <- fastkpc_full_cuda_default_kpcalg_contract()
  expected_n <- contract$n_edgetests
  payload_gate <- nrow(graph) == 1L && graph$SHD[[1L]] == 0L &&
    graph$edge_count_reference[[1L]] == 110L &&
    graph$edge_count_candidate[[1L]] == 110L &&
    isTRUE(as.logical(graph$adjacency_identical[[1L]])) &&
    all(as.logical(sepsets$identical)) &&
    nrow(tests) == length(expected_n) &&
    identical(as.integer(tests$reference), expected_n) &&
    identical(as.integer(tests$candidate), expected_n) &&
    all(as.logical(tests$identical)) && nrow(deletions) == 1018L &&
    nrow(cases) == contract$logical_test_count &&
    identical(as.integer(cases$logical_sequence_id),
              seq_len(contract$logical_test_count)) &&
    !any(as.logical(cases$decision_flip)) &&
    all(is.finite(cases$candidate_p_value)) &&
    nrow(cache) == 2L && all(as.logical(cache$bounded)) &&
    all(cache$requests == cache$hits + cache$misses) &&
    sum(fallbacks$count) == 0L
  fastkpc_full_cuda_phase9_require(
    payload_gate, "Phase 9 artifact payload gate failed"
  )
  inputs <- read_table("evidence_inputs.csv")
  fastkpc_full_cuda_phase9_require(
    all(file.exists(inputs$input_file)) &&
      identical(
        unname(vapply(
          inputs$input_file,
          fastkpc_full_cuda_census_file_hash,
          character(1L)
        )),
        as.character(inputs$sha256)
      ),
    "Phase 9 evidence input identity drifted"
  )
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(
    file.path(artifact_dir, manifest$environment_file)
  )
  fastkpc_full_cuda_phase9_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 9 environment receipt drifted"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase9_require(
    length(attestations) == 1L && length(receipts) == 1L,
    "Phase 9 attestation or receipt is missing"
  )
  .fastkpc_full_cuda_phase35_validate_attestation(attestations[[1L]])
  .fastkpc_full_cuda_phase35_validate_receipt(receipts[[1L]])
  fastkpc_full_cuda_phase9_require(
    identical(attestations[[1L]]$attested_producer_sha256,
              producer$identity_sha256) &&
      identical(receipts[[1L]]$producer_sha256, producer$identity_sha256),
    "Phase 9 attestation linkage failed"
  )
  list(
    manifest = manifest,
    producer = producer,
    summary = summary,
    evidence = evidence,
    source_closure = source_closure
  )
}
