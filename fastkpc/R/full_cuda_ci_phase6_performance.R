fastkpc_full_cuda_phase6_performance_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase6_performance_object_hash <- function(value) {
  fastkpc_full_cuda_census_hash_raw(serialize(value, NULL, version = 2L))
}

fastkpc_full_cuda_phase6_performance_paths <- function() {
  c(
    phase4_backend_summary = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_backend_v1", "summary.json"
    ),
    phase4_full_shadow_evidence = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1", "source_evidence.rds"
    ),
    legacy_mgcv_trace_summary = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "trace_source_351x48_v1", "summary.csv"
    )
  )
}

fastkpc_full_cuda_phase6_build_performance_evidence <- function(
    merged_evidence, merged_evidence_path = NULL,
    paths = fastkpc_full_cuda_phase6_performance_paths()) {
  paths <- stats::setNames(as.character(paths), names(paths))
  fastkpc_full_cuda_phase6_performance_require(
    identical(names(paths), names(fastkpc_full_cuda_phase6_performance_paths())) &&
      all(file.exists(paths) & !dir.exists(paths)),
    "Phase 6 performance input paths are incomplete"
  )
  fastkpc_full_cuda_phase6_performance_require(
    is.list(merged_evidence) && identical(
      merged_evidence$schema_version,
      "full-cuda-ci-multi-penalty-cuda-full-shadow-merged-v1"
    ) && isTRUE(merged_evidence$summary$pass) &&
      isTRUE(merged_evidence$mixed_graph$summary$pass),
    "Phase 6 performance merged evidence is malformed"
  )
  phase4_backend <- jsonlite::read_json(
    paths[["phase4_backend_summary"]], simplifyVector = TRUE
  )
  phase4_shadow <- readRDS(paths[["phase4_full_shadow_evidence"]])
  baseline <- utils::read.csv(
    paths[["legacy_mgcv_trace_summary"]],
    stringsAsFactors = FALSE, check.names = FALSE
  )
  expected_n_edgetests <-
    "2213|52659|125293|40694|13293|5422|835|80"
  inherited_gate <-
    identical(phase4_backend$artifact_kind, "backend") &&
    isTRUE(phase4_backend$pass) &&
    phase4_backend$single_penalty_setup_count == 1174L &&
    phase4_backend$single_penalty_target_count == 44941L &&
    phase4_backend$repetition_count >= 5L &&
    isTRUE(phase4_backend$all_candidate_targets_identical) &&
    isTRUE(phase4_backend$backend_gate) &&
    is.finite(phase4_backend$candidate_median_ms) &&
    phase4_backend$candidate_median_ms > 0 &&
    identical(
      phase4_shadow$schema_version,
      "full-cuda-ci-single-penalty-gcv-full-shadow-merged-v1"
    ) &&
    phase4_shadow$summary$setup_count == 1174L &&
    phase4_shadow$summary$target_count == 44941L &&
    isTRUE(phase4_shadow$summary$same_sp_fixed_solver_gate) &&
    isTRUE(phase4_shadow$summary$oracle_residual_gate) &&
    isTRUE(phase4_shadow$summary$downstream_decision_gate) &&
    isTRUE(phase4_shadow$summary$optimizer_coverage_gate) &&
    isTRUE(phase4_shadow$summary$backend_gate) &&
    is.finite(phase4_shadow$summary$summed_cuda_selected_sp_solve_ms) &&
    phase4_shadow$summary$summed_cuda_selected_sp_solve_ms > 0
  baseline_gate <- nrow(baseline) == 1L &&
    identical(baseline$run_status[[1L]], "ok") &&
    identical(
      baseline$compatible_cuda_route[[1L]],
      "legacy-mgcv-provider-native-legacy-dcov"
    ) && identical(baseline$mgcv_residual_backend[[1L]], "r") &&
    isTRUE(baseline$residual_provider_parallel_enabled[[1L]]) &&
    baseline$residual_provider_parallel_cores[[1L]] == 20L &&
    baseline$edge_count[[1L]] == 110L &&
    baseline$shd[[1L]] == 0L &&
    isTRUE(baseline$adjacency_identical[[1L]]) &&
    isTRUE(baseline$n_edgetests_identical[[1L]]) &&
    identical(baseline$facade_n_edgetests[[1L]], expected_n_edgetests) &&
    is.finite(baseline$residual_provider_total_ms[[1L]]) &&
    baseline$residual_provider_total_ms[[1L]] > 0
  graph <- merged_evidence$mixed_graph$summary
  candidate_gate <- merged_evidence$summary$setup_count == 7460L &&
    merged_evidence$summary$target_count == 65676L &&
    merged_evidence$summary$legacy_mgcv_target_calls == 0L &&
    merged_evidence$summary$cpu_multi_penalty_solve_count == 0L &&
    merged_evidence$summary$fallback_count == 0L &&
    isTRUE(merged_evidence$summary$authority_gate) &&
    isTRUE(merged_evidence$summary$concurrency_gate) &&
    isTRUE(merged_evidence$summary$observed_concurrent_execution) &&
    graph$edge_count_candidate == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) && isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical)
  fastkpc_full_cuda_phase6_performance_require(
    inherited_gate && baseline_gate && candidate_gate,
    "Phase 6 performance input authority gate failed"
  )
  windows <- merged_evidence$timings[
    merged_evidence$timings$scheduler_window_leader, , drop = FALSE
  ]
  stages <- data.frame(
    stage = c(
      "phase4_cuda_gcv_optimizer",
      "phase4_cuda_selected_fit",
      "phase6_cuda_prepared_setup_upload",
      "phase6_cuda_optimizer_gemm_residual"
    ),
    wall_ms = c(
      as.numeric(phase4_backend$candidate_median_ms),
      as.numeric(
        phase4_shadow$summary$summed_cuda_selected_sp_solve_ms
      ),
      sum(merged_evidence$timings$setup_ms),
      sum(windows$optimizer_window_wall_ms)
    ),
    source = c(
      "accepted-phase4-five-run-median",
      "accepted-phase4-full-shadow-stage-sum",
      "phase6-full-corpus-stage-sum",
      "phase6-full-corpus-bounded-window-stage-sum"
    ),
    stringsAsFactors = FALSE
  )
  candidate_wall_ms <- sum(stages$wall_ms)
  baseline_wall_ms <- as.numeric(
    baseline$residual_provider_total_ms[[1L]]
  )
  ratio <- candidate_wall_ms / baseline_wall_ms
  source_hashes <- setNames(as.list(vapply(
    paths, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(paths))
  merged_sha256 <- if (is.null(merged_evidence_path)) {
    fastkpc_full_cuda_census_hash_raw(
      serialize(merged_evidence, NULL, version = 2L)
    )
  } else {
    fastkpc_full_cuda_census_file_hash(merged_evidence_path)
  }
  summary <- list(
    schema_version = "full-cuda-ci-phase6-performance-summary-v1",
    measurement_protocol =
      "authenticated-full-trace-residual-stage-wall-v1",
    candidate_route =
      "phase4-and-phase6-full-cuda-residual-authority",
    baseline_route = "legacy-mgcv-provider-native-legacy-dcov-20-core",
    canonical_logical_test_count = 240489L,
    direct_logical_test_count = 2213L,
    conditional_logical_test_count = 238276L,
    unique_residual_target_count = 110617L,
    phase4_target_count = 44941L,
    phase6_target_count = 65676L,
    candidate_residual_wall_ms = candidate_wall_ms,
    baseline_residual_wall_ms = baseline_wall_ms,
    candidate_to_baseline_ratio = ratio,
    candidate_graph_gate = candidate_gate,
    baseline_graph_gate = baseline_gate,
    inherited_phase4_gate = inherited_gate,
    relative_performance_gate = candidate_wall_ms < baseline_wall_ms,
    phase10_performance_claim = FALSE,
    pass = inherited_gate && baseline_gate && candidate_gate &&
      candidate_wall_ms < baseline_wall_ms
  )
  fastkpc_full_cuda_phase6_performance_require(
    isTRUE(summary$pass), "Phase 6 full residual performance gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase6-performance-evidence-v1",
    summary = summary,
    merged_evidence_sha256 = merged_sha256,
    input_file_sha256 = source_hashes,
    candidate_stage_timing = stages,
    baseline_measurement = data.frame(
      route = summary$baseline_route,
      residual_provider_request_count =
        as.integer(baseline$residual_provider_request_count[[1L]]),
      residual_provider_parallel_cores =
        as.integer(baseline$residual_provider_parallel_cores[[1L]]),
      residual_provider_total_ms = baseline_wall_ms,
      edge_count = as.integer(baseline$edge_count[[1L]]),
      SHD = as.integer(baseline$shd[[1L]]),
      n_edgetests = baseline$facade_n_edgetests[[1L]],
      stringsAsFactors = FALSE
    ),
    candidate_graph_summary = graph
  )
}

fastkpc_full_cuda_phase6_validate_performance_evidence <- function(
    value, merged_evidence, merged_evidence_path = NULL,
    verify_current_inputs = TRUE) {
  clean <- is.list(value) && identical(
    value$schema_version,
    "full-cuda-ci-phase6-performance-evidence-v1"
  ) && is.list(value$summary) && identical(
    value$summary$schema_version,
    "full-cuda-ci-phase6-performance-summary-v1"
  ) && value$summary$canonical_logical_test_count == 240489L &&
    value$summary$unique_residual_target_count == 110617L &&
    value$summary$phase4_target_count == 44941L &&
    value$summary$phase6_target_count == 65676L &&
    isTRUE(value$summary$candidate_graph_gate) &&
    isTRUE(value$summary$baseline_graph_gate) &&
    isTRUE(value$summary$inherited_phase4_gate) &&
    isTRUE(value$summary$relative_performance_gate) &&
    !isTRUE(value$summary$phase10_performance_claim) &&
    value$summary$candidate_residual_wall_ms <
      value$summary$baseline_residual_wall_ms &&
    value$summary$candidate_to_baseline_ratio < 1 &&
    isTRUE(value$summary$pass) &&
    is.data.frame(value$candidate_stage_timing) &&
    nrow(value$candidate_stage_timing) == 4L &&
    is.data.frame(value$baseline_measurement) &&
    nrow(value$baseline_measurement) == 1L
  fastkpc_full_cuda_phase6_performance_require(
    clean, "Phase 6 performance evidence validation failed"
  )
  if (isTRUE(verify_current_inputs)) {
    recomputed <- fastkpc_full_cuda_phase6_build_performance_evidence(
      merged_evidence = merged_evidence,
      merged_evidence_path = merged_evidence_path
    )
    fastkpc_full_cuda_phase6_performance_require(
      identical(
        fastkpc_full_cuda_phase6_performance_object_hash(value),
        fastkpc_full_cuda_phase6_performance_object_hash(recomputed)
      ),
      "Phase 6 performance evidence does not match current inputs"
    )
  }
  invisible(value)
}
