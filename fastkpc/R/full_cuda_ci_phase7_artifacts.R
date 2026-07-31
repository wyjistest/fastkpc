fastkpc_full_cuda_phase7_artifact_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase7_validate_setup_corpus <- function(
    evidence, catalog, verify_current_identity = TRUE) {
  summary <- evidence$summary
  rows <- evidence$rows
  error_fields <- c(
    "x_max_absolute_error",
    "model_space_projector_max_absolute_error",
    "constraint_max_absolute_error",
    "penalty_operator_max_absolute_error",
    "shift_max_absolute_error", "penalty_scale_max_absolute_error",
    "qr_q_max_absolute_error", "qr_packed_max_absolute_error",
    "qr_tau_max_absolute_error", "qr_r_max_absolute_error",
    "mroot_max_absolute_error", "penalty_matrix_max_absolute_error",
    "initial_sp_max_absolute_error"
  )
  mismatch_fields <- c(
    "rank_mismatch_count", "null_space_mismatch_count",
    "basis_dimension_mismatch_count", "qr_pivot_mismatch_count"
  )
  scope <- fastkpc_full_cuda_phase7_setup_scope(catalog)
  expected_keys <- as.character(scope$setup_rows$prepared_s_key_sha256)
  clean <- is.list(evidence) && identical(
    evidence$schema_version,
    "full-cuda-ci-native-setup-corpus-evidence-v1"
  ) && is.list(summary) && isTRUE(summary$pass) &&
    summary$setup_count == 8634L &&
    identical(summary$counts_by_S_size,
              c(48L, 1126L, 4064L, 2152L, 955L, 245L, 44L)) &&
    summary$native_setup_count == 8634L &&
    summary$oracle_setup_count == 8634L &&
    summary$legacy_mgcv_setup_count == 8634L &&
    summary$r_callback_count == 0L && summary$unsupported_count == 0L &&
    summary$fixed_sp_operator_equivalence_count == 8634L &&
    is.data.frame(rows) && nrow(rows) == 8634L &&
    !anyDuplicated(rows$prepared_s_key_sha256) &&
    identical(rows$prepared_s_key_sha256, expected_keys) &&
    all(rows$fixed_sp_operator_equivalent) &&
    all(vapply(error_fields, function(field) {
      field %in% names(rows) && all(is.finite(rows[[field]])) &&
        max(rows[[field]]) == 0
    }, logical(1L))) &&
    all(vapply(mismatch_fields, function(field) {
      field %in% names(rows) && sum(rows[[field]]) == 0L
    }, logical(1L))) &&
    sum(rows$unsupported_count) == 0L &&
    sum(rows$r_callback_count) == 0L
  fastkpc_full_cuda_phase7_artifact_require(
    clean, "Phase 7 setup corpus evidence is malformed"
  )
  if (isTRUE(verify_current_identity)) {
    current <- fastkpc_full_cuda_phase7_execution_identity(
      catalog, "native-setup-oracle"
    )
    fastkpc_full_cuda_phase7_artifact_require(
      identical(
        fastkpc_full_cuda_phase35_canonical_json(
          evidence$execution_identity
        ),
        fastkpc_full_cuda_phase35_canonical_json(current)
      ),
      "Phase 7 setup corpus execution identity is stale"
    )
  }
  invisible(evidence)
}

fastkpc_full_cuda_phase7_validate_native_partition_identity <- function(
    parts, catalog, label) {
  fastkpc_full_cuda_phase7_artifact_require(
    length(parts) > 0L && all(vapply(parts, is.list, logical(1L))),
    paste0("Phase 7 ", label, " partitions are missing")
  )
  expected <- fastkpc_full_cuda_phase7_execution_identity(
    catalog, "native-setup-backend"
  )
  expected_json <- fastkpc_full_cuda_phase35_canonical_json(expected)
  identity_json <- vapply(parts, function(value) {
    fastkpc_full_cuda_phase35_canonical_json(
      value$phase7_execution_identity
    )
  }, character(1L))
  timing_clean <- all(vapply(parts, function(value) {
    timings <- value$timings
    is.data.frame(timings) && nrow(timings) > 0L &&
      all(c(
        "native_setup_count", "native_setup_unsupported_count",
        "legacy_mgcv_setup_count", "r_callback_count"
      ) %in% names(timings)) &&
      all(timings$native_setup_count == 1L) &&
      sum(timings$native_setup_unsupported_count) == 0L &&
      sum(timings$legacy_mgcv_setup_count) == 0L &&
      sum(timings$r_callback_count) == 0L
  }, logical(1L)))
  fastkpc_full_cuda_phase7_artifact_require(
    timing_clean && length(unique(identity_json)) == 1L &&
      identical(identity_json[[1L]], expected_json),
    paste0("Phase 7 ", label, " native execution identity is malformed")
  )
  expected
}

fastkpc_full_cuda_phase7_graph_summary <- function(phase6) {
  graph <- phase6$mixed_graph$summary
  list(
    direct_logical_test_count =
      as.integer(graph$direct_legacy_logical_test_count),
    single_penalty_cuda_logical_test_count =
      as.integer(graph$phase4_cuda_logical_test_count),
    multi_penalty_cuda_logical_test_count =
      as.integer(graph$phase6_cuda_logical_test_count),
    edge_count_reference = as.integer(graph$edge_count_reference),
    edge_count_candidate = as.integer(graph$edge_count_candidate),
    SHD = as.integer(graph$SHD),
    adjacency_identical = isTRUE(graph$adjacency_identical),
    sepsets_identical = isTRUE(graph$sepsets_identical),
    n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
    deletions_identical = isTRUE(graph$deletions_identical),
    pass = isTRUE(graph$pass)
  )
}

fastkpc_full_cuda_phase7_build_full_evidence <- function(
    catalog, setup_corpus_path, phase4_partition_paths,
    phase6_partition_paths, phase5_evidence_path) {
  setup_corpus_path <- normalizePath(
    setup_corpus_path, winslash = "/", mustWork = TRUE
  )
  phase5_evidence_path <- normalizePath(
    phase5_evidence_path, winslash = "/", mustWork = TRUE
  )
  phase4_partition_paths <- as.character(phase4_partition_paths)
  phase6_partition_paths <- as.character(phase6_partition_paths)
  fastkpc_full_cuda_phase7_artifact_require(
    length(phase4_partition_paths) > 0L &&
      length(phase6_partition_paths) > 0L &&
      all(file.exists(phase4_partition_paths) &
            !dir.exists(phase4_partition_paths)) &&
      all(file.exists(phase6_partition_paths) &
            !dir.exists(phase6_partition_paths)),
    "Phase 7 native backend partition paths are incomplete"
  )
  setup_corpus <- readRDS(setup_corpus_path)
  fastkpc_full_cuda_phase7_validate_setup_corpus(
    setup_corpus, catalog, verify_current_identity = TRUE
  )
  phase4_parts <- lapply(phase4_partition_paths, readRDS)
  phase6_parts <- lapply(phase6_partition_paths, readRDS)
  backend_identity <-
    fastkpc_full_cuda_phase7_validate_native_partition_identity(
      phase4_parts, catalog, "Phase 4"
    )
  phase6_identity <-
    fastkpc_full_cuda_phase7_validate_native_partition_identity(
      phase6_parts, catalog, "Phase 6"
    )
  fastkpc_full_cuda_phase7_artifact_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(backend_identity),
      fastkpc_full_cuda_phase35_canonical_json(phase6_identity)
    ),
    "Phase 7 Phase 4/6 execution identities differ"
  )

  phase4 <- fastkpc_full_cuda_phase4_merge_full_shadow(
    catalog, phase4_partition_paths
  )
  phase6 <- fastkpc_full_cuda_phase6_merge_partitions(
    catalog = catalog,
    evidence_paths = phase6_partition_paths,
    phase5_evidence_path = phase5_evidence_path,
    phase4_logical_rows = phase4$logical_rows
  )
  phase4_timings <- phase4$timings
  phase6_timings <- phase6$timings
  graph <- fastkpc_full_cuda_phase7_graph_summary(phase6)
  native_setup_count <- sum(phase4_timings$native_setup_count) +
    sum(phase6_timings$native_setup_count)
  unsupported_count <-
    sum(phase4_timings$native_setup_unsupported_count) +
      sum(phase6_timings$native_setup_unsupported_count)
  legacy_setup_count <- sum(phase4_timings$legacy_mgcv_setup_count) +
    sum(phase6_timings$legacy_mgcv_setup_count)
  r_callback_count <- sum(phase4_timings$r_callback_count) +
    sum(phase6_timings$r_callback_count)
  decision_flip_count <- sum(phase4$logical_rows$decision_flip) +
    sum(phase6$logical_rows$decision_flip)
  backend_error_count <- sum(phase4$logical_rows$backend_error) +
    sum(phase6$logical_rows$backend_error)
  dcov_fallback_count <- sum(phase4$logical_rows$spectra_fallback) +
    sum(phase6$logical_rows$spectra_fallback)
  setup_keys <- c(
    as.character(phase4_timings$prepared_s_key_sha256),
    as.character(phase6_timings$prepared_s_key_sha256)
  )
  expected_setup_keys <- as.character(
    fastkpc_full_cuda_phase7_setup_scope(catalog)$setup_rows[[
      "prepared_s_key_sha256"
    ]]
  )
  phase4_gate <-
    isTRUE(phase4$summary$same_sp_fixed_solver_gate) &&
      isTRUE(phase4$summary$oracle_residual_gate) &&
      isTRUE(phase4$summary$downstream_decision_gate) &&
      isTRUE(phase4$summary$optimizer_coverage_gate) &&
      isTRUE(phase4$summary$backend_gate)
  summary <- list(
    schema_version = "full-cuda-ci-native-setup-full-summary-v1",
    run_status = "ok",
    setup_count = length(setup_keys),
    single_penalty_setup_count = nrow(phase4_timings),
    multi_penalty_setup_count = nrow(phase6_timings),
    target_count = nrow(phase4$targets) + nrow(phase6$targets),
    single_penalty_target_count = nrow(phase4$targets),
    multi_penalty_target_count = nrow(phase6$targets),
    logical_test_count = 2213L + nrow(phase4$logical_rows) +
      nrow(phase6$logical_rows),
    native_setup_count = native_setup_count,
    native_setup_unsupported_count = unsupported_count,
    legacy_mgcv_setup_count = legacy_setup_count,
    legacy_mgcv_fit_count = 0L,
    legacy_mgcv_target_call_count =
      sum(phase4_timings$legacy_mgcv_target_calls) +
        sum(phase6_timings$legacy_mgcv_target_calls),
    r_callback_count = r_callback_count,
    cpu_residual_numerical_solve_count = 0L,
    residual_numerical_fallback_count =
      sum(phase4_timings$fallback_count) +
        sum(phase6_timings$fallback_count),
    downstream_legacy_dcov_decision_flip_count = decision_flip_count,
    downstream_legacy_dcov_backend_error_count = backend_error_count,
    downstream_legacy_dcov_fallback_count = dcov_fallback_count,
    maximum_single_penalty_same_sp_fitted_error =
      max(phase4$targets$fitted_same_sp_max_absolute),
    maximum_single_penalty_same_sp_residual_error =
      max(phase4$targets$residual_same_sp_max_absolute),
    maximum_single_penalty_oracle_fitted_error =
      max(phase4$targets$fitted_oracle_max_absolute),
    maximum_single_penalty_oracle_residual_error =
      max(phase4$targets$residual_oracle_max_absolute),
    maximum_single_penalty_selected_log_sp_error =
      max(abs(phase4$targets$log_sp_error)),
    maximum_multi_penalty_selected_log_sp_error =
      max(phase6$targets$selected_log_sp_max_error),
    maximum_multi_penalty_fitted_gemm_error =
      max(phase6$targets$fitted_gemm_max_absolute),
    maximum_multi_penalty_residual_identity_error =
      max(phase6$targets$residual_identity_max_absolute),
    setup_corpus_gate = isTRUE(setup_corpus$summary$pass),
    single_penalty_gate = phase4_gate,
    multi_penalty_gate = isTRUE(phase6$summary$pass),
    graph_gate = isTRUE(graph$pass),
    edge_count_reference = graph$edge_count_reference,
    edge_count_candidate = graph$edge_count_candidate,
    SHD = graph$SHD,
    adjacency_identical = graph$adjacency_identical,
    sepsets_identical = graph$sepsets_identical,
    n_edgetests_identical = graph$n_edgetests_identical,
    deletions_identical = graph$deletions_identical
  )
  summary$authority_gate <-
    length(setup_keys) == 8634L && !anyDuplicated(setup_keys) &&
      identical(sort(setup_keys, method = "radix"), expected_setup_keys) &&
      summary$target_count == 110617L &&
      summary$logical_test_count == 240489L &&
      native_setup_count == 8634L && unsupported_count == 0L &&
      legacy_setup_count == 0L && summary$legacy_mgcv_fit_count == 0L &&
      summary$legacy_mgcv_target_call_count == 0L &&
      r_callback_count == 0L &&
      summary$residual_numerical_fallback_count == 0L
  summary$pass <- isTRUE(summary$setup_corpus_gate) &&
    isTRUE(summary$single_penalty_gate) &&
    isTRUE(summary$multi_penalty_gate) &&
    isTRUE(summary$graph_gate) && isTRUE(summary$authority_gate) &&
    decision_flip_count == 0L && backend_error_count == 0L &&
    dcov_fallback_count == 0L &&
    graph$edge_count_reference == 110L &&
    graph$edge_count_candidate == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) &&
    isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical)
  fastkpc_full_cuda_phase7_artifact_require(
    summary$pass, "Phase 7 full native setup evidence gate failed"
  )
  list(
    schema_version = "full-cuda-ci-native-setup-full-evidence-v1",
    summary = summary,
    graph = graph,
    execution_identity = backend_identity,
    setup_oracle_execution_identity = setup_corpus$execution_identity,
    evidence_file_sha256 = list(
      setup_corpus = fastkpc_full_cuda_census_file_hash(setup_corpus_path),
      phase5 = fastkpc_full_cuda_census_file_hash(phase5_evidence_path),
      phase4_partitions = setNames(as.list(vapply(
        phase4_partition_paths,
        fastkpc_full_cuda_census_file_hash,
        character(1L)
      )), basename(phase4_partition_paths)),
      phase6_partitions = setNames(as.list(vapply(
        phase6_partition_paths,
        fastkpc_full_cuda_census_file_hash,
        character(1L)
      )), basename(phase6_partition_paths))
    ),
    setup_corpus = setup_corpus,
    phase4 = phase4,
    phase6 = phase6
  )
}

fastkpc_full_cuda_phase7_validate_full_evidence <- function(
    evidence, catalog, verify_current_identity = TRUE) {
  summary <- evidence$summary
  graph <- evidence$graph
  phase4 <- evidence$phase4
  phase6 <- evidence$phase6
  phase4_gate <- is.list(phase4) &&
    isTRUE(phase4$summary$same_sp_fixed_solver_gate) &&
    isTRUE(phase4$summary$oracle_residual_gate) &&
    isTRUE(phase4$summary$downstream_decision_gate) &&
    isTRUE(phase4$summary$optimizer_coverage_gate) &&
    isTRUE(phase4$summary$backend_gate)
  phase6_gate <- is.list(phase6) && isTRUE(phase6$summary$pass) &&
    isTRUE(phase6$summary$numerical_gate) &&
    isTRUE(phase6$summary$optimizer_gate) &&
    isTRUE(phase6$summary$authority_gate) &&
    isTRUE(phase6$summary$backend_gate)
  nested_graph <- if (is.list(phase6)) {
    fastkpc_full_cuda_phase7_graph_summary(phase6)
  } else {
    NULL
  }
  setup_keys <- if (is.list(phase4) && is.list(phase6)) c(
    as.character(phase4$timings$prepared_s_key_sha256),
    as.character(phase6$timings$prepared_s_key_sha256)
  ) else character()
  expected_setup_keys <- as.character(
    fastkpc_full_cuda_phase7_setup_scope(catalog)$setup_rows[[
      "prepared_s_key_sha256"
    ]]
  )
  nested_clean <- phase4_gate && phase6_gate &&
    identical(
      fastkpc_full_cuda_phase35_canonical_json(graph),
      fastkpc_full_cuda_phase35_canonical_json(nested_graph)
    ) && nrow(phase4$timings) == 1174L &&
    nrow(phase6$timings) == 7460L &&
    nrow(phase4$targets) == 44941L &&
    nrow(phase6$targets) == 65676L &&
    nrow(phase4$logical_rows) == 177952L &&
    nrow(phase6$logical_rows) == 60324L &&
    length(setup_keys) == 8634L && !anyDuplicated(setup_keys) &&
    identical(sort(setup_keys, method = "radix"), expected_setup_keys) &&
    all(phase4$timings$native_setup_count == 1L) &&
    all(phase6$timings$native_setup_count == 1L) &&
    sum(phase4$timings$native_setup_unsupported_count) == 0L &&
    sum(phase6$timings$native_setup_unsupported_count) == 0L &&
    sum(phase4$timings$legacy_mgcv_setup_count) == 0L &&
    sum(phase6$timings$legacy_mgcv_setup_count) == 0L &&
    sum(phase4$timings$r_callback_count) == 0L &&
    sum(phase6$timings$r_callback_count) == 0L &&
    sum(phase4$logical_rows$decision_flip) == 0L &&
    sum(phase6$logical_rows$decision_flip) == 0L &&
    sum(phase4$logical_rows$backend_error) == 0L &&
    sum(phase6$logical_rows$backend_error) == 0L &&
    sum(phase4$logical_rows$spectra_fallback) == 0L &&
    sum(phase6$logical_rows$spectra_fallback) == 0L &&
    identical(
      summary$maximum_single_penalty_same_sp_fitted_error,
      max(phase4$targets$fitted_same_sp_max_absolute)
    ) && identical(
      summary$maximum_single_penalty_same_sp_residual_error,
      max(phase4$targets$residual_same_sp_max_absolute)
    ) && identical(
      summary$maximum_single_penalty_selected_log_sp_error,
      max(abs(phase4$targets$log_sp_error))
    ) && identical(
      summary$maximum_multi_penalty_selected_log_sp_error,
      max(phase6$targets$selected_log_sp_max_error)
    )
  clean <- is.list(evidence) && identical(
    evidence$schema_version,
    "full-cuda-ci-native-setup-full-evidence-v1"
  ) && is.list(summary) && isTRUE(summary$pass) &&
    summary$setup_count == 8634L &&
    summary$single_penalty_setup_count == 1174L &&
    summary$multi_penalty_setup_count == 7460L &&
    summary$target_count == 110617L &&
    summary$logical_test_count == 240489L &&
    summary$native_setup_count == 8634L &&
    summary$native_setup_unsupported_count == 0L &&
    summary$legacy_mgcv_setup_count == 0L &&
    summary$legacy_mgcv_fit_count == 0L &&
    summary$legacy_mgcv_target_call_count == 0L &&
    summary$r_callback_count == 0L &&
    summary$cpu_residual_numerical_solve_count == 0L &&
    summary$residual_numerical_fallback_count == 0L &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$downstream_legacy_dcov_backend_error_count == 0L &&
    summary$downstream_legacy_dcov_fallback_count == 0L &&
    isTRUE(summary$authority_gate) && isTRUE(summary$graph_gate) &&
    graph$edge_count_reference == 110L &&
    graph$edge_count_candidate == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) &&
    isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical) && isTRUE(graph$pass) &&
    isTRUE(nested_clean)
  fastkpc_full_cuda_phase7_artifact_require(
    clean, "Phase 7 full evidence is malformed"
  )
  fastkpc_full_cuda_phase7_validate_setup_corpus(
    evidence$setup_corpus, catalog,
    verify_current_identity = verify_current_identity
  )
  if (isTRUE(verify_current_identity)) {
    current <- fastkpc_full_cuda_phase7_execution_identity(
      catalog, "native-setup-backend"
    )
    fastkpc_full_cuda_phase7_artifact_require(
      identical(
        fastkpc_full_cuda_phase35_canonical_json(
          evidence$execution_identity
        ),
        fastkpc_full_cuda_phase35_canonical_json(current)
      ),
      "Phase 7 full evidence execution identity is stale"
    )
  }
  invisible(evidence)
}
