fastkpc_full_cuda_phase6_publication_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase6_artifact_kinds <- function() {
  c("oracle", "full_shadow", "backend")
}

fastkpc_full_cuda_phase6_artifact_directory_names <- function() {
  c(
    oracle = "multi_penalty_cuda_oracle_v1",
    full_shadow = "multi_penalty_cuda_full_shadow_v1",
    backend = "full_residual_cuda_backend_v1"
  )
}

fastkpc_full_cuda_phase6_publication_input_identity <- function(catalog) {
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
    ),
    inherited_phase4_backend_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_backend_v1", "manifest.json"
    ),
    phase5_oracle_manifest = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "multi_penalty_cpp_full_shadow_v1", "manifest.json"
    ),
    phase5_oracle_evidence = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "multi_penalty_cpp_full_shadow_v1", "source_evidence.rds"
    ),
    legacy_mgcv_trace_summary = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "trace_source_351x48_v1", "summary.csv"
    )
  )
  hashes <- setNames(as.list(vapply(
    files, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(files))
  dataset <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase6-canonical-corpus-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    input_file_sha256 = hashes$canonical_data,
    phase1_manifest_sha256 = hashes$phase1_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    phase5_oracle_evidence_sha256 = hashes$phase5_oracle_evidence,
    setup_count = 7460L,
    target_count = 65676L,
    logical_test_count = 60324L,
    S_size = 3:7
  ))
  oracle <- fastkpc_full_cuda_census_named_metadata_hash(list(
    schema_version = "full-cuda-ci-phase6-oracle-authority-v1",
    phase0_manifest_sha256 = hashes$phase0_manifest,
    phase2_manifest_sha256 = hashes$phase2_manifest,
    inherited_phase4_manifest_sha256 =
      hashes$inherited_phase4_manifest,
    inherited_phase4_logical_results_sha256 =
      hashes$inherited_phase4_logical_results,
    phase5_oracle_manifest_sha256 = hashes$phase5_oracle_manifest,
    phase5_oracle_evidence_sha256 = hashes$phase5_oracle_evidence,
    mgcv_semantics_version = "mgcv-gam-gcv-cp-v1",
    allowed_residual_decision_flip_count = 0L,
    allowed_dcov_decision_flip_count = 0L,
    required_SHD = 0L
  ))
  list(files = files, hashes = hashes, dataset_sha256 = dataset,
       oracle_sha256 = oracle)
}

fastkpc_full_cuda_phase6_backend_configuration <- function(kind) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase6_artifact_kinds())
  value <- list(
    schema_version = "full-cuda-ci-phase6-backend-configuration-v1",
    artifact_kind = kind,
    formula_route = "target~s(S1)+...+s(Sk)",
    family = "gaussian",
    link = "identity",
    sp_selection_backend = "cuda-magic-multi-penalty",
    gcv_score_backend = "cuda-magic-double-deterministic",
    residual_backend =
      "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual",
    guarded_qr_backend = "cuda-householder-qr-r-transpose-inverse",
    guarded_qr_minimum_diagonal_ratio = "1e-6",
    guarded_qr_condition_upper_bound = "1e7",
    stable_solver_backend =
      "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd",
    execution_strategy = "bounded-independent-prepared-streams-v1",
    maximum_concurrent_setups = 32L,
    target_optimizer_state = "independent-per-target",
    residual_placement = "device-resident-token",
    rank_tolerance = "sqrt(.Machine$double.eps)",
    convergence_tolerance = "1e-7",
    max_step_halving = 25L,
    max_iterations = 400L,
    max_newton_component = "5",
    boundary_probe_step = "2",
    max_boundary_probes = 5L,
    stability_replay_backend = "device-all-svd-full-trajectory",
    stability_replay_long_trajectory_minimum_iterations = 101L,
    stability_replay_boundary_minimum_iterations = 25L,
    stability_replay_boundary_requires_accepted_probe = TRUE,
    stability_replay_boundary_minimum_step_halving_per_iteration = "2.25",
    stability_replay_high_condition_minimum_iterations = 16L,
    stability_replay_high_condition_threshold = "16777216",
    stability_replay_high_condition_requires_accepted_probe = TRUE,
    stability_replay_high_condition_minimum_step_halving_per_iteration =
      "0.75",
    stability_replay_high_condition_extrapolation_fraction = "0.25",
    stability_replay_high_condition_maximum_extrapolation = "4e-6",
    stability_replay_selection_log_sp_spread = "5e-8",
    stability_replay_maximum_inward_shift = "1e-6",
    stability_replay_inward_shift_scope = "long-trajectory-only",
    terminal_boundary_confirmation_backend =
      "device-stable-svd-plus-direct-qr-residual-delta-identity",
    terminal_boundary_confirmation_condition_threshold = "33554432",
    terminal_boundary_confirmation_score_tie_relative_tolerance = "1e-10",
    terminal_boundary_confirmation_delta_identity_tolerance =
      "512*lapack_epsilon*q*(1+abs(current_score))",
    terminal_boundary_confirmation_acceptance_modes =
      "reinforced-strong-delta-or-endpoint-identity-verified-tie",
    downstream_validation_backend =
      "legacy-dcov-gamma-cpp-spectra-component-cache",
    precision = "float64",
    fmad = FALSE,
    fast_math = FALSE,
    normal_equations_used = FALSE,
    legacy_mgcv_target_calls = 0L,
    cpu_multi_penalty_solve_count = 0L,
    fallback_count = 0L,
    shadow_only = TRUE
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase6_build_recipe <- function() {
  value <- list(
    schema_version = "full-cuda-ci-phase6-build-recipe-v1",
    build_script_sha256 = fastkpc_full_cuda_census_file_hash(
      "fastkpc/tools/build_cuda_native.sh"
    ),
    CXX17 = fastkpc_full_cuda_command_output(
      "R", c("CMD", "config", "CXX17")
    ),
    R_version = R.version.string,
    mgcv_version = as.character(utils::packageVersion("mgcv")),
    nvcc_path = "/usr/local/cuda/bin/nvcc",
    nvcc_version = fastkpc_full_cuda_command_output(
      "/usr/local/cuda/bin/nvcc", "--version"
    ),
    cuda_architecture = "sm_89",
    fmad = FALSE,
    fast_math = FALSE,
    stable_solver = "device-guarded-qr-lapack-3.12-dgesdd"
  )
  list(
    value = value,
    sha256 = fastkpc_full_cuda_census_named_metadata_hash(value)
  )
}

fastkpc_full_cuda_phase6_producer <- function(
    kind, catalog, source_closure, native_identity,
    contracts = fastkpc_full_cuda_phase35_load_contract_set()) {
  input <- fastkpc_full_cuda_phase6_publication_input_identity(catalog)
  backend <- fastkpc_full_cuda_phase6_backend_configuration(kind)
  build <- fastkpc_full_cuda_phase6_build_recipe()
  producer <- fastkpc_full_cuda_phase35_producer_identity(
    producer_source_closure_sha256 = source_closure$sha256,
    native_binary_sha256 = native_identity$sha256,
    route_semantic_version = paste0(
      "full-cuda-ci-phase6-multi-penalty-cuda-", kind, "-v1"
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

fastkpc_full_cuda_phase6_validate_merged_evidence <- function(
    evidence, catalog, verify_current_identity = TRUE) {
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  summary <- evidence$summary
  graph <- evidence$mixed_graph$summary
  recomputed_summary <- tryCatch(
    fastkpc_full_cuda_phase6_shadow_summary(
      evidence$targets, evidence$logical_rows, evidence$timings,
      7460L, TRUE, evidence$parallel_elapsed_seconds
    ),
    error = function(error) NULL
  )
  clean <- is.list(evidence) && identical(
    evidence$schema_version,
    "full-cuda-ci-multi-penalty-cuda-full-shadow-merged-v1"
  ) && is.list(summary) && is.list(recomputed_summary) && identical(
    fastkpc_full_cuda_phase35_canonical_json(summary),
    fastkpc_full_cuda_phase35_canonical_json(recomputed_summary)
  ) && isTRUE(summary$pass) &&
    summary$setup_count == 7460L && summary$target_count == 65676L &&
    summary$logical_test_count == 60324L &&
    summary$penalty_count_min == 3L && summary$penalty_count_max == 7L &&
    summary$optimizer_iteration_mismatch_count == 0L &&
    summary$score_call_mismatch_count == 0L &&
    summary$objective_call_mismatch_count == 0L &&
    summary$step_halving_mismatch_count == 0L &&
    summary$boundary_probe_mismatch_count == 0L &&
    summary$boundary_status_mismatch_count == 0L &&
    summary$convergence_mismatch_count == 0L &&
    summary$hessian_state_mismatch_count == 0L &&
    summary$rank_mismatch_count == 0L && summary$fallback_count == 0L &&
    summary$cuda_optimizer_error_count == 0L &&
    summary$legacy_mgcv_target_calls == 0L &&
    summary$cpu_multi_penalty_solve_count == 0L &&
    summary$setup_upload_count == 7460L &&
    summary$workspace_grow_count == 0L &&
    summary$solve_device_allocation_count == 0L &&
    summary$cublas_gemm_count == 7460L &&
    summary$residual_kernel_count == 7460L &&
    summary$cuda_complete_evaluation_count +
      summary$cuda_score_only_evaluation_count +
      summary$cuda_selected_evaluation_reuse_count ==
        summary$cuda_objective_call_count &&
    summary$cuda_selected_evaluation_reuse_count ==
      sum(evidence$targets$boundary_accepted_count == 0L) &&
    summary$cuda_guarded_qr_evaluation_count +
      summary$cuda_stable_svd_evaluation_count ==
        summary$cuda_complete_evaluation_count +
          summary$cuda_score_only_evaluation_count &&
    summary$cuda_stability_replay_kernel_launch_count == 7460L &&
    summary$cuda_stability_merge_kernel_launch_count == 7460L &&
    summary$cuda_stability_replay_target_count > 0L &&
    summary$cuda_stability_replay_selected_count > 0L &&
    summary$cuda_stability_replay_selected_count <=
      summary$cuda_stability_replay_target_count &&
    summary$cuda_stability_replay_error_count == 0L &&
    summary$cuda_stability_replay_extrapolation_target_count > 0L &&
    summary$cuda_stability_replay_extrapolation_target_count <=
      summary$cuda_stability_replay_selected_count &&
    summary$cuda_stability_replay_discarded_complete_evaluation_count +
      summary$cuda_stability_replay_discarded_score_only_evaluation_count ==
        summary$cuda_stability_replay_discarded_guarded_qr_evaluation_count +
          summary$cuda_stability_replay_discarded_stable_svd_evaluation_count &&
    summary$cuda_stability_replay_max_log_sp_spread > 5e-8 &&
    summary$cuda_stability_replay_max_extrapolation > 0 &&
    summary$cuda_stability_replay_max_extrapolation <= 4e-6 &&
    summary$cuda_terminal_boundary_confirmation_count > 0L &&
    summary$cuda_terminal_boundary_confirmation_accepted_count > 0L &&
    summary$cuda_terminal_boundary_confirmation_rejected_count > 0L &&
    summary[[
      "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] > 0L &&
    summary[[
      "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] > 0L &&
    summary$cuda_terminal_boundary_confirmation_accepted_count +
      summary$cuda_terminal_boundary_confirmation_rejected_count ==
        summary$cuda_terminal_boundary_confirmation_count &&
    summary[[
      "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] + summary[[
      "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] == summary$cuda_terminal_boundary_confirmation_accepted_count &&
    summary$cuda_terminal_boundary_confirmation_complete_evaluation_count ==
      2L * summary$cuda_terminal_boundary_confirmation_count &&
    summary$cuda_terminal_boundary_confirmation_stable_svd_evaluation_count ==
      summary$cuda_terminal_boundary_confirmation_complete_evaluation_count &&
    summary$cuda_terminal_boundary_confirmation_cycles > 0 &&
    summary$cuda_physical_evaluation_count ==
      summary$cuda_physical_factorization_count &&
    summary$cuda_guarded_qr_evaluation_count > 0L &&
    summary$cuda_stable_svd_evaluation_count > 0L &&
    summary$configured_concurrency >= 1L &&
    summary$configured_concurrency <= 32L &&
    summary$maximum_host_calls_in_flight > 1L &&
    summary$maximum_setup_stream_count > 1L &&
    isTRUE(summary$observed_concurrent_execution) &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$near_alpha_decision_flip_count == 0L &&
    isTRUE(summary$numerical_gate) && isTRUE(summary$optimizer_gate) &&
    isTRUE(summary$stability_replay_gate) &&
    isTRUE(summary$terminal_boundary_confirmation_gate) &&
    isTRUE(summary$physical_evaluation_accounting_gate) &&
    isTRUE(summary$authority_gate) && isTRUE(summary$concurrency_gate) &&
    isTRUE(summary$downstream_decision_gate) &&
    isTRUE(summary$backend_gate) && isTRUE(graph$pass) &&
    graph$direct_legacy_logical_test_count == 2213L &&
    graph$phase4_cuda_logical_test_count == 177952L &&
    graph$phase6_cuda_logical_test_count == 60324L &&
    graph$legacy_mgcv_target_call_count == 0L &&
    graph$cpu_residual_numerical_solve_count == 0L &&
    graph$residual_numerical_fallback_count == 0L &&
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
          "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd") &&
    all(evidence$targets$optimizer_backend_executed ==
          "cuda-magic-multi-penalty") &&
    all(evidence$targets$residual_backend_executed ==
          "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual") &&
    !any(evidence$targets$normal_equations_used) &&
    all(evidence$targets$all_finite)
  fastkpc_full_cuda_phase6_publication_require(
    clean, "Phase 6 merged evidence gate failed"
  )
  if (isTRUE(verify_current_identity)) {
    current <- fastkpc_full_cuda_phase6_execution_identity(catalog)
    fastkpc_full_cuda_phase6_publication_require(
      identical(
        fastkpc_full_cuda_phase35_canonical_json(current),
        fastkpc_full_cuda_phase35_canonical_json(
          evidence$execution_identity
        )
      ),
      "Phase 6 merged evidence does not match current producer identity"
    )
  }
  invisible(TRUE)
}

fastkpc_full_cuda_phase6_common_summary <- function(
    kind, evidence, producer_bundle, contracts, source_closure,
    native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase6_artifact_kinds())
  source <- evidence$summary
  graph <- evidence$mixed_graph$summary
  performance <- if (identical(kind, "backend")) {
    evidence$performance$summary
  } else {
    NULL
  }
  performance_required <- identical(kind, "backend")
  performance_pass <- !performance_required || (
    is.list(performance) && isTRUE(performance$pass)
  )
  summary <- list(
    schema_version = paste0(
      "full-cuda-ci-phase6-multi-penalty-cuda-", kind, "-summary-v1"
    ),
    artifact_kind = kind,
    claim_scope = "phase6-additive-multi-penalty-S-size-greater-than-2",
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
      "phase4-cuda-single-penalty+",
      "phase6-cuda-multi-penalty-device-residual"
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
    phase6_cuda_logical_test_count =
      as.integer(graph$phase6_cuda_logical_test_count),
    explicit_legacy_fallback_count =
      as.integer(graph$explicit_legacy_fallback_count),
    unknown_fallback_count = as.integer(graph$unknown_fallback_count),
    approximate_backend_count = 0L,
    multi_penalty_setup_count = as.integer(source$setup_count),
    multi_penalty_target_count = as.integer(source$target_count),
    penalty_count_min = as.integer(source$penalty_count_min),
    penalty_count_max = as.integer(source$penalty_count_max),
    legacy_mgcv_target_calls =
      as.integer(source$legacy_mgcv_target_calls),
    cpu_multi_penalty_solve_count =
      as.integer(source$cpu_multi_penalty_solve_count),
    residual_numerical_fallback_count =
      as.integer(graph$residual_numerical_fallback_count),
    fallback_count = as.integer(source$fallback_count),
    max_selected_log_sp_error =
      as.numeric(source$max_selected_log_sp_error),
    max_score_absolute_error =
      as.numeric(source$max_score_absolute_error),
    max_edf_absolute_error = as.numeric(source$max_edf_absolute_error),
    max_coefficient_shadow_absolute_error =
      as.numeric(source$max_coefficient_shadow_absolute_error),
    max_fitted_gemm_absolute_error =
      as.numeric(source$max_fitted_gemm_absolute_error),
    max_residual_identity_absolute_error =
      as.numeric(source$max_residual_identity_absolute_error),
    max_absolute_p_value_difference =
      as.numeric(source$max_absolute_p_value_difference),
    optimizer_iteration_mismatch_count =
      as.integer(source$optimizer_iteration_mismatch_count),
    score_call_mismatch_count =
      as.integer(source$score_call_mismatch_count),
    objective_call_mismatch_count =
      as.integer(source$objective_call_mismatch_count),
    step_halving_mismatch_count =
      as.integer(source$step_halving_mismatch_count),
    boundary_probe_mismatch_count =
      as.integer(source$boundary_probe_mismatch_count),
    boundary_status_mismatch_count =
      as.integer(source$boundary_status_mismatch_count),
    convergence_mismatch_count =
      as.integer(source$convergence_mismatch_count),
    hessian_state_mismatch_count =
      as.integer(source$hessian_state_mismatch_count),
    rank_mismatch_count = as.integer(source$rank_mismatch_count),
    cuda_optimizer_error_count =
      as.integer(source$cuda_optimizer_error_count),
    cuda_objective_call_count =
      as.integer(source$cuda_objective_call_count),
    cuda_complete_evaluation_count =
      as.integer(source$cuda_complete_evaluation_count),
    cuda_score_only_evaluation_count =
      as.integer(source$cuda_score_only_evaluation_count),
    cuda_selected_evaluation_reuse_count =
      as.integer(source$cuda_selected_evaluation_reuse_count),
    cuda_guarded_qr_evaluation_count =
      as.integer(source$cuda_guarded_qr_evaluation_count),
    cuda_stable_svd_evaluation_count =
      as.integer(source$cuda_stable_svd_evaluation_count),
    cuda_stability_replay_kernel_launch_count =
      as.integer(source$cuda_stability_replay_kernel_launch_count),
    cuda_stability_merge_kernel_launch_count =
      as.integer(source$cuda_stability_merge_kernel_launch_count),
    cuda_stability_replay_target_count =
      as.integer(source$cuda_stability_replay_target_count),
    cuda_stability_replay_selected_count =
      as.integer(source$cuda_stability_replay_selected_count),
    cuda_stability_replay_error_count =
      as.integer(source$cuda_stability_replay_error_count),
    cuda_stability_replay_extrapolation_target_count = as.integer(
      source$cuda_stability_replay_extrapolation_target_count
    ),
    cuda_stability_replay_discarded_complete_evaluation_count = as.integer(
      source$cuda_stability_replay_discarded_complete_evaluation_count
    ),
    cuda_stability_replay_discarded_score_only_evaluation_count = as.integer(
      source$cuda_stability_replay_discarded_score_only_evaluation_count
    ),
    cuda_stability_replay_discarded_guarded_qr_evaluation_count = as.integer(
      source$cuda_stability_replay_discarded_guarded_qr_evaluation_count
    ),
    cuda_stability_replay_discarded_stable_svd_evaluation_count = as.integer(
      source$cuda_stability_replay_discarded_stable_svd_evaluation_count
    ),
    cuda_stability_replay_discarded_cycles =
      as.numeric(source$cuda_stability_replay_discarded_cycles),
    cuda_stability_replay_max_log_sp_spread =
      as.numeric(source$cuda_stability_replay_max_log_sp_spread),
    cuda_stability_replay_max_extrapolation =
      as.numeric(source$cuda_stability_replay_max_extrapolation),
    cuda_terminal_boundary_confirmation_count =
      as.integer(source$cuda_terminal_boundary_confirmation_count),
    cuda_terminal_boundary_confirmation_accepted_count = as.integer(
      source$cuda_terminal_boundary_confirmation_accepted_count
    ),
    cuda_terminal_boundary_confirmation_rejected_count = as.integer(
      source$cuda_terminal_boundary_confirmation_rejected_count
    ),
    cuda_terminal_boundary_confirmation_strong_delta_accepted_count =
      as.integer(source[[
        "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
      ]]),
    cuda_terminal_boundary_confirmation_identity_tie_accepted_count =
      as.integer(source[[
        "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
      ]]),
    cuda_terminal_boundary_confirmation_complete_evaluation_count = as.integer(
      source$cuda_terminal_boundary_confirmation_complete_evaluation_count
    ),
    cuda_terminal_boundary_confirmation_stable_svd_evaluation_count =
      as.integer(
        source$cuda_terminal_boundary_confirmation_stable_svd_evaluation_count
      ),
    cuda_terminal_boundary_confirmation_cycles =
      as.numeric(source$cuda_terminal_boundary_confirmation_cycles),
    cuda_terminal_boundary_confirmation_max_identity_disagreement = as.numeric(
      source$cuda_terminal_boundary_confirmation_max_identity_disagreement
    ),
    cuda_terminal_boundary_confirmation_max_identity_ratio = as.numeric(
      source$cuda_terminal_boundary_confirmation_max_identity_ratio
    ),
    cuda_terminal_boundary_confirmation_max_delta_disagreement = as.numeric(
      source$cuda_terminal_boundary_confirmation_max_delta_disagreement
    ),
    cuda_terminal_boundary_confirmation_max_delta_ratio = as.numeric(
      source$cuda_terminal_boundary_confirmation_max_delta_ratio
    ),
    cuda_physical_evaluation_count =
      as.integer(source$cuda_physical_evaluation_count),
    cuda_physical_factorization_count =
      as.integer(source$cuda_physical_factorization_count),
    downstream_legacy_dcov_decision_flip_count = as.integer(
      source$downstream_legacy_dcov_decision_flip_count
    ),
    near_alpha_decision_flip_count =
      as.integer(source$near_alpha_decision_flip_count),
    configured_concurrency = as.integer(source$configured_concurrency),
    scheduler_window_count = as.integer(source$scheduler_window_count),
    concurrent_scheduler_window_count =
      as.integer(source$concurrent_scheduler_window_count),
    maximum_host_calls_in_flight =
      as.integer(source$maximum_host_calls_in_flight),
    maximum_setup_stream_count =
      as.integer(source$maximum_setup_stream_count),
    maximum_host_overlap_factor =
      as.numeric(source$maximum_host_overlap_factor),
    observed_concurrent_execution =
      isTRUE(source$observed_concurrent_execution),
    setup_upload_count = as.integer(source$setup_upload_count),
    workspace_grow_count = as.integer(source$workspace_grow_count),
    solve_device_allocation_count =
      as.integer(source$solve_device_allocation_count),
    cublas_gemm_count = as.integer(source$cublas_gemm_count),
    residual_kernel_count = as.integer(source$residual_kernel_count),
    validation_residual_shadow_d2h_count =
      as.integer(source$validation_residual_shadow_d2h_count),
    dcov_component_request_count =
      as.integer(source$dcov_component_request_count),
    dcov_component_cache_hit_count =
      as.integer(source$dcov_component_cache_hit_count),
    dcov_component_cache_miss_count =
      as.integer(source$dcov_component_cache_miss_count),
    performance_evidence_present = is.list(performance),
    performance_baseline_route = if (is.list(performance)) {
      as.character(performance$baseline_route)
    } else {
      NA_character_
    },
    candidate_residual_wall_ms = if (is.list(performance)) {
      as.numeric(performance$candidate_residual_wall_ms)
    } else {
      NA_real_
    },
    baseline_residual_wall_ms = if (is.list(performance)) {
      as.numeric(performance$baseline_residual_wall_ms)
    } else {
      NA_real_
    },
    candidate_to_baseline_ratio = if (is.list(performance)) {
      as.numeric(performance$candidate_to_baseline_ratio)
    } else {
      NA_real_
    },
    performance_gate = performance_pass,
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
    stability_replay_gate = isTRUE(source$stability_replay_gate),
    terminal_boundary_confirmation_gate =
      isTRUE(source$terminal_boundary_confirmation_gate),
    physical_evaluation_accounting_gate =
      isTRUE(source$physical_evaluation_accounting_gate),
    authority_gate = isTRUE(source$authority_gate),
    concurrency_gate = isTRUE(source$concurrency_gate),
    downstream_decision_gate = isTRUE(source$downstream_decision_gate),
    backend_gate = isTRUE(source$backend_gate),
    phase6_only = TRUE,
    shadow_only = TRUE,
    full_residual_cuda_authoritative = identical(kind, "backend"),
    production_backend_promoted = FALSE,
    phase10_promotion_claim = FALSE,
    pass = performance_pass
  )
  summary
}

fastkpc_full_cuda_phase6_write_table <- function(value, directory, name) {
  utils::write.csv(
    value, file.path(directory, name), row.names = FALSE, na = ""
  )
}

fastkpc_full_cuda_phase6_rank_table <- function(catalog) {
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  fields <- intersect(c(
    "prepared_s_key_sha256", "S_size", "formula_class", "penalty_count",
    "coefficient_dim", "penalty_rank", "penalty_nullity",
    "model_matrix_rank", "model_matrix_condition", "conditioning_rank",
    "conditioning_condition", "condition_bucket", "planned_route"
  ), names(scope$setup_rows))
  value <- scope$setup_rows[, fields, drop = FALSE]
  fastkpc_full_cuda_phase6_publication_require(
    nrow(value) == 7460L && "prepared_s_key_sha256" %in% names(value),
    "Phase 6 rank-condition table is incomplete"
  )
  value
}

fastkpc_full_cuda_phase6_write_graph <- function(evidence, directory) {
  comparison <- evidence$mixed_graph$comparison
  fastkpc_full_cuda_phase6_write_table(
    comparison$graph_agreement, directory, "graph_agreement.csv"
  )
  fastkpc_full_cuda_phase6_write_table(
    comparison$sepset_agreement, directory, "sepset_agreement.csv"
  )
  fastkpc_full_cuda_phase6_write_table(
    comparison$n_edgetests, directory, "n_edgetests.csv"
  )
  fastkpc_full_cuda_phase6_write_table(
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
    accepted_for_phase6 = c(FALSE, FALSE, FALSE),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase6_write_table(
    fallbacks, directory, "fallbacks.csv"
  )
}

fastkpc_full_cuda_phase6_write_payload <- function(
    kind, directory, catalog, evidence, source_evidence_path) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase6_artifact_kinds())
  fastkpc_full_cuda_phase6_write_graph(evidence, directory)
  fastkpc_full_cuda_phase6_write_table(
    fastkpc_full_cuda_phase6_rank_table(catalog), directory,
    "rank_condition_results.csv"
  )
  near_alpha <- evidence$logical_rows[
    evidence$logical_rows$near_alpha, , drop = FALSE
  ]
  fastkpc_full_cuda_phase6_write_table(
    near_alpha, directory, "near_alpha_results.csv"
  )
  cache <- data.frame(
    cache = c(
      "PreparedS-device-workspace", "CUDA-residual-token",
      "legacy-dCov-component"
    ),
    ownership = c(
      "phase6-persistent-setup", "phase6-device-residual",
      "phase6-shadow-validation"
    ),
    requests = c(
      nrow(evidence$timings),
      nrow(evidence$timings),
      evidence$summary$dcov_component_request_count
    ),
    hits = c(
      0L, 0L, evidence$summary$dcov_component_cache_hit_count
    ),
    misses = c(
      nrow(evidence$timings),
      nrow(evidence$timings),
      evidence$summary$dcov_component_cache_miss_count
    ),
    semantic_eviction_effect = c("none", "none", "none"),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase6_write_table(cache, directory, "cache.csv")
  fastkpc_full_cuda_phase6_write_table(
    evidence$targets, directory, "case_results.csv"
  )
  fastkpc_full_cuda_phase6_write_table(
    evidence$timings, directory, "stage_timing.csv"
  )
  partition_hashes <- data.frame(
    partition_file = names(evidence$partition_file_sha256),
    sha256 = unlist(evidence$partition_file_sha256, use.names = FALSE),
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase6_write_table(
    partition_hashes, directory, "partition_hashes.csv"
  )
  raw_runs <- data.frame(
    run = seq_len(nrow(partition_hashes)),
    route = "cuda-multi-penalty-bounded-stream-shadow-partition",
    partition_file = partition_hashes$partition_file,
    partition_sha256 = partition_hashes$sha256,
    pass = TRUE,
    stringsAsFactors = FALSE
  )
  fastkpc_full_cuda_phase6_write_table(
    raw_runs, directory, "raw_runs.csv"
  )
  if (kind %in% c("full_shadow", "backend")) {
    saveRDS(
      evidence$logical_rows,
      file.path(directory, "logical_ci_results.rds"), version = 3L
    )
  }
  if (identical(kind, "backend")) {
    saveRDS(
      evidence$performance,
      file.path(directory, "performance_evidence.rds"), version = 3L
    )
  }
  copied <- file.copy(
    source_evidence_path, file.path(directory, "source_evidence.rds"),
    overwrite = FALSE, copy.mode = TRUE, copy.date = FALSE
  )
  fastkpc_full_cuda_phase6_publication_require(
    copied, "Phase 6 source evidence copy failed"
  )
}

fastkpc_full_cuda_phase6_manifest_fields <- function() {
  c(
    "schema_version", "artifact_kind", "claim_scope",
    "producer_semantic_envelope", "payload_manifest_sha256",
    "payload_file_sha256", "semantic_file_count",
    "validator_attestations_file", "volatile_receipt_file",
    "environment_file", "environment_file_sha256"
  )
}

fastkpc_full_cuda_phase6_validate_artifact <- function(
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
      expected_kind, fastkpc_full_cuda_phase6_artifact_kinds()
    )
  }
  manifest_clean <- identical(
    names(manifest), fastkpc_full_cuda_phase6_manifest_fields()
  ) && identical(
    manifest$schema_version,
    "full-cuda-ci-phase6-multi-penalty-cuda-artifact-manifest-v1"
  ) && kind %in% fastkpc_full_cuda_phase6_artifact_kinds() &&
    (is.null(expected_kind) || identical(kind, expected_kind)) &&
    identical(
      manifest$claim_scope,
      "phase6-additive-multi-penalty-S-size-greater-than-2"
    )
  fastkpc_full_cuda_phase6_publication_require(
    manifest_clean, "Phase 6 artifact manifest schema mismatch"
  )
  payload_hashes <- manifest$payload_file_sha256
  fastkpc_full_cuda_phase6_publication_require(
    is.list(payload_hashes) && length(payload_hashes) > 0L &&
      length(payload_hashes) == manifest$semantic_file_count,
    "Phase 6 payload manifest is malformed"
  )
  for (name in names(payload_hashes)) {
    path <- file.path(artifact_dir, name)
    fastkpc_full_cuda_phase6_publication_require(
      file.exists(path) && !dir.exists(path) && identical(
        fastkpc_full_cuda_census_file_hash(path), payload_hashes[[name]]
      ),
      paste0("Phase 6 payload file hash mismatch: ", name)
    )
  }
  payload_manifest_sha256 <- fastkpc_full_cuda_phase35_sha256_utf8(
    fastkpc_full_cuda_phase35_canonical_json(payload_hashes)
  )
  fastkpc_full_cuda_phase6_publication_require(
    identical(
      payload_manifest_sha256, manifest$payload_manifest_sha256
    ),
    "Phase 6 payload manifest hash mismatch"
  )
  envelope <- manifest$producer_semantic_envelope
  fastkpc_full_cuda_phase35_validate_identity_envelope(envelope)
  fastkpc_full_cuda_phase6_publication_require(
    identical(envelope$payload_manifest_sha256,
              payload_manifest_sha256),
    "Phase 6 producer envelope payload mismatch"
  )
  producer <- jsonlite::read_json(
    file.path(artifact_dir, "producer_identity.json"),
    simplifyVector = FALSE
  )
  .fastkpc_full_cuda_phase35_validate_producer(producer)
  fastkpc_full_cuda_phase6_publication_require(
    identical(
      fastkpc_full_cuda_phase35_canonical_json(producer),
      fastkpc_full_cuda_phase35_canonical_json(envelope$producer)
    ),
    "Phase 6 producer identity file does not match the envelope"
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
  fastkpc_full_cuda_phase6_publication_require(
    nrow(source_closure) > 0L && !anyDuplicated(source_closure$path) &&
      identical(
        closure_sha256, producer$producer_source_closure_sha256
      ),
    "Phase 6 producer source closure identity mismatch"
  )
  environment_path <- file.path(artifact_dir, manifest$environment_file)
  environment_sha256 <- fastkpc_full_cuda_census_file_hash(environment_path)
  fastkpc_full_cuda_phase6_publication_require(
    identical(environment_sha256, manifest$environment_file_sha256),
    "Phase 6 environment evidence hash mismatch"
  )
  attestations <- jsonlite::read_json(
    file.path(artifact_dir, manifest$validator_attestations_file),
    simplifyVector = FALSE
  )$attestations
  fastkpc_full_cuda_phase6_publication_require(
    is.list(attestations) && length(attestations) > 0L,
    "Phase 6 validator attestation is missing"
  )
  for (attestation in attestations) {
    .fastkpc_full_cuda_phase35_validate_attestation(attestation)
    fastkpc_full_cuda_phase6_publication_require(
      identical(
        attestation$attested_producer_sha256, producer$identity_sha256
      ) && identical(attestation$environment_sha256, environment_sha256) &&
        identical(attestation$validation_result, "PASS"),
      "Phase 6 validator attestation does not attest this producer"
    )
  }
  receipts <- jsonlite::read_json(
    file.path(artifact_dir, manifest$volatile_receipt_file),
    simplifyVector = FALSE
  )$execution_receipts
  fastkpc_full_cuda_phase6_publication_require(
    is.list(receipts) && length(receipts) > 0L,
    "Phase 6 execution receipt is missing"
  )
  for (receipt in receipts) {
    .fastkpc_full_cuda_phase35_validate_receipt(receipt)
    fastkpc_full_cuda_phase6_publication_require(
      identical(receipt$producer_sha256, producer$identity_sha256),
      "Phase 6 execution receipt producer mismatch"
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
  if (identical(kind, "backend")) {
    required_files <- c(required_files, "performance_evidence.rds")
  }
  fastkpc_full_cuda_phase6_publication_require(
    all(file.exists(file.path(artifact_dir, required_files))),
    "Phase 6 artifact standard file set is incomplete"
  )
  summary <- jsonlite::read_json(
    file.path(artifact_dir, "summary.json"), simplifyVector = TRUE
  )
  stage_timing <- utils::read.csv(
    file.path(artifact_dir, "stage_timing.csv"),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  telemetry_sum_fields <- c(
    "cuda_stability_replay_kernel_launch_count",
    "cuda_stability_merge_kernel_launch_count",
    "cuda_stability_replay_target_count",
    "cuda_stability_replay_selected_count",
    "cuda_stability_replay_error_count",
    "cuda_stability_replay_extrapolation_target_count",
    "cuda_stability_replay_discarded_complete_evaluation_count",
    "cuda_stability_replay_discarded_score_only_evaluation_count",
    "cuda_stability_replay_discarded_guarded_qr_evaluation_count",
    "cuda_stability_replay_discarded_stable_svd_evaluation_count",
    "cuda_stability_replay_discarded_cycles",
    "cuda_terminal_boundary_confirmation_count",
    "cuda_terminal_boundary_confirmation_accepted_count",
    "cuda_terminal_boundary_confirmation_rejected_count",
    "cuda_terminal_boundary_confirmation_strong_delta_accepted_count",
    "cuda_terminal_boundary_confirmation_identity_tie_accepted_count",
    "cuda_terminal_boundary_confirmation_complete_evaluation_count",
    "cuda_terminal_boundary_confirmation_stable_svd_evaluation_count",
    "cuda_terminal_boundary_confirmation_cycles"
  )
  telemetry_max_fields <- c(
    "cuda_stability_replay_max_log_sp_spread",
    "cuda_stability_replay_max_extrapolation",
    "cuda_terminal_boundary_confirmation_max_identity_disagreement",
    "cuda_terminal_boundary_confirmation_max_identity_ratio",
    "cuda_terminal_boundary_confirmation_max_delta_disagreement",
    "cuda_terminal_boundary_confirmation_max_delta_ratio"
  )
  telemetry_required_fields <- c(
    "target_count", "cuda_complete_evaluation_count",
    "cuda_score_only_evaluation_count", "cuda_guarded_qr_evaluation_count",
    "cuda_stable_svd_evaluation_count", telemetry_sum_fields,
    telemetry_max_fields
  )
  telemetry_gate <- nrow(stage_timing) == 7460L &&
    all(telemetry_required_fields %in% names(stage_timing))
  if (isTRUE(telemetry_gate)) {
    replay_evaluations <-
      stage_timing$cuda_stability_replay_discarded_complete_evaluation_count +
        stage_timing[[
          "cuda_stability_replay_discarded_score_only_evaluation_count"
        ]]
    replay_factorizations <-
      stage_timing[[
        "cuda_stability_replay_discarded_guarded_qr_evaluation_count"
      ]] + stage_timing[[
        "cuda_stability_replay_discarded_stable_svd_evaluation_count"
      ]]
    confirmation_evaluations <- stage_timing[[
      "cuda_terminal_boundary_confirmation_complete_evaluation_count"
    ]]
    confirmation_factorizations <- stage_timing[[
      "cuda_terminal_boundary_confirmation_stable_svd_evaluation_count"
    ]]
    physical_evaluations <-
      stage_timing$cuda_complete_evaluation_count +
        stage_timing$cuda_score_only_evaluation_count +
        replay_evaluations + confirmation_evaluations
    physical_factorizations <-
      stage_timing$cuda_guarded_qr_evaluation_count +
        stage_timing$cuda_stable_svd_evaluation_count +
        replay_factorizations + confirmation_factorizations
    telemetry_gate <-
      all(stage_timing$cuda_stability_replay_kernel_launch_count == 1L) &&
      all(stage_timing$cuda_stability_merge_kernel_launch_count == 1L) &&
      all(stage_timing$cuda_stability_replay_target_count >= 0L) &&
      all(stage_timing$cuda_stability_replay_target_count <=
            stage_timing$target_count) &&
      all(stage_timing$cuda_stability_replay_selected_count >= 0L) &&
      all(stage_timing$cuda_stability_replay_selected_count <=
            stage_timing$cuda_stability_replay_target_count) &&
      sum(stage_timing$cuda_stability_replay_error_count) == 0L &&
      all(
        stage_timing$cuda_stability_replay_extrapolation_target_count >= 0L &
          stage_timing$cuda_stability_replay_extrapolation_target_count <=
            stage_timing$cuda_stability_replay_selected_count
      ) &&
      all(replay_evaluations == replay_factorizations) &&
      all(
        (stage_timing$cuda_stability_replay_target_count == 0L &
           replay_evaluations == 0L) |
          (stage_timing$cuda_stability_replay_target_count > 0L &
             replay_evaluations > 0L)
      ) &&
      all(
        stage_timing$cuda_stability_replay_selected_count == 0L |
          stage_timing$cuda_stability_replay_max_log_sp_spread > 5e-8
      ) &&
      all(
        (stage_timing$cuda_stability_replay_extrapolation_target_count == 0L &
          stage_timing$cuda_stability_replay_max_extrapolation == 0) |
          (stage_timing$cuda_stability_replay_extrapolation_target_count > 0L &
            stage_timing$cuda_stability_replay_max_extrapolation > 0 &
            stage_timing$cuda_stability_replay_max_extrapolation <= 4e-6)
      ) &&
      all(
        stage_timing$cuda_terminal_boundary_confirmation_accepted_count +
          stage_timing$cuda_terminal_boundary_confirmation_rejected_count ==
          stage_timing$cuda_terminal_boundary_confirmation_count
      ) &&
      all(
        stage_timing[[
          "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
        ]] + stage_timing[[
          "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
        ]] ==
          stage_timing$cuda_terminal_boundary_confirmation_accepted_count
      ) &&
      all(confirmation_evaluations ==
            2L * stage_timing$cuda_terminal_boundary_confirmation_count) &&
      all(confirmation_factorizations == confirmation_evaluations) &&
      all(
        stage_timing$cuda_terminal_boundary_confirmation_count == 0L |
          stage_timing$cuda_terminal_boundary_confirmation_cycles > 0
      ) &&
      all(physical_evaluations == physical_factorizations) &&
      all(vapply(
        c(telemetry_sum_fields, telemetry_max_fields), function(field) {
          all(is.finite(stage_timing[[field]])) &&
            all(stage_timing[[field]] >= 0)
        }, logical(1L)
      )) &&
      all(vapply(telemetry_sum_fields, function(field) {
        isTRUE(all.equal(
          as.numeric(sum(stage_timing[[field]])),
          as.numeric(summary[[field]]), tolerance = 1e-12
        ))
      }, logical(1L))) &&
      all(vapply(telemetry_max_fields, function(field) {
        isTRUE(all.equal(
          as.numeric(max(stage_timing[[field]])),
          as.numeric(summary[[field]]), tolerance = 1e-12
        ))
      }, logical(1L))) &&
      sum(physical_evaluations) == summary$cuda_physical_evaluation_count &&
      sum(physical_factorizations) ==
        summary$cuda_physical_factorization_count
  }
  common_gate <- telemetry_gate &&
    identical(summary$artifact_kind, kind) &&
    identical(summary$run_status, "ok") && !isTRUE(summary$timeout) &&
    summary$edge_count_reference == 110L &&
    summary$edge_count_candidate == 110L && summary$SHD == 0L &&
    isTRUE(summary$adjacency_identical) &&
    isTRUE(summary$sepsets_identical) &&
    isTRUE(summary$n_edgetests_identical) &&
    isTRUE(summary$deletions_identical) &&
    summary$direct_legacy_logical_test_count == 2213L &&
    summary$phase4_cuda_logical_test_count == 177952L &&
    summary$phase6_cuda_logical_test_count == 60324L &&
    summary$explicit_legacy_fallback_count == 0L &&
    summary$unknown_fallback_count == 0L &&
    summary$approximate_backend_count == 0L &&
    summary$multi_penalty_setup_count == 7460L &&
    summary$multi_penalty_target_count == 65676L &&
    summary$legacy_mgcv_target_calls == 0L &&
    summary$cpu_multi_penalty_solve_count == 0L &&
    summary$residual_numerical_fallback_count == 0L &&
    summary$fallback_count == 0L &&
    summary$optimizer_iteration_mismatch_count == 0L &&
    summary$score_call_mismatch_count == 0L &&
    summary$objective_call_mismatch_count == 0L &&
    summary$step_halving_mismatch_count == 0L &&
    summary$boundary_probe_mismatch_count == 0L &&
    summary$boundary_status_mismatch_count == 0L &&
    summary$convergence_mismatch_count == 0L &&
    summary$hessian_state_mismatch_count == 0L &&
    summary$rank_mismatch_count == 0L &&
    summary$cuda_optimizer_error_count == 0L &&
    summary$cuda_objective_call_count ==
      summary$cuda_complete_evaluation_count +
        summary$cuda_score_only_evaluation_count +
        summary$cuda_selected_evaluation_reuse_count &&
    summary$cuda_guarded_qr_evaluation_count > 0L &&
    summary$cuda_stable_svd_evaluation_count > 0L &&
    summary$cuda_stability_replay_kernel_launch_count == 7460L &&
    summary$cuda_stability_merge_kernel_launch_count == 7460L &&
    summary$cuda_stability_replay_target_count > 0L &&
    summary$cuda_stability_replay_selected_count > 0L &&
    summary$cuda_stability_replay_selected_count <=
      summary$cuda_stability_replay_target_count &&
    summary$cuda_stability_replay_error_count == 0L &&
    summary$cuda_stability_replay_extrapolation_target_count > 0L &&
    summary$cuda_stability_replay_extrapolation_target_count <=
      summary$cuda_stability_replay_selected_count &&
    summary$cuda_stability_replay_discarded_complete_evaluation_count +
      summary$cuda_stability_replay_discarded_score_only_evaluation_count ==
        summary$cuda_stability_replay_discarded_guarded_qr_evaluation_count +
          summary$cuda_stability_replay_discarded_stable_svd_evaluation_count &&
    summary$cuda_stability_replay_max_log_sp_spread > 5e-8 &&
    summary$cuda_stability_replay_max_extrapolation > 0 &&
    summary$cuda_stability_replay_max_extrapolation <= 4e-6 &&
    summary$cuda_terminal_boundary_confirmation_count > 0L &&
    summary$cuda_terminal_boundary_confirmation_accepted_count > 0L &&
    summary$cuda_terminal_boundary_confirmation_rejected_count > 0L &&
    summary[[
      "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] > 0L &&
    summary[[
      "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] > 0L &&
    summary$cuda_terminal_boundary_confirmation_accepted_count +
      summary$cuda_terminal_boundary_confirmation_rejected_count ==
        summary$cuda_terminal_boundary_confirmation_count &&
    summary[[
      "cuda_terminal_boundary_confirmation_strong_delta_accepted_count"
    ]] + summary[[
      "cuda_terminal_boundary_confirmation_identity_tie_accepted_count"
    ]] == summary$cuda_terminal_boundary_confirmation_accepted_count &&
    summary$cuda_terminal_boundary_confirmation_complete_evaluation_count ==
      2L * summary$cuda_terminal_boundary_confirmation_count &&
    summary$cuda_terminal_boundary_confirmation_stable_svd_evaluation_count ==
      summary$cuda_terminal_boundary_confirmation_complete_evaluation_count &&
    summary$cuda_terminal_boundary_confirmation_cycles > 0 &&
    summary$cuda_physical_evaluation_count ==
      summary$cuda_physical_factorization_count &&
    summary$downstream_legacy_dcov_decision_flip_count == 0L &&
    summary$near_alpha_decision_flip_count == 0L &&
    summary$max_selected_log_sp_error <= 1e-6 &&
    summary$max_score_absolute_error <= 1e-8 &&
    summary$max_edf_absolute_error <= 1e-8 &&
    summary$max_coefficient_shadow_absolute_error <= 1e-12 &&
    summary$max_fitted_gemm_absolute_error <= 1e-8 &&
    summary$max_residual_identity_absolute_error <= 1e-12 &&
    summary$max_absolute_p_value_difference <= 1e-10 &&
    summary$configured_concurrency >= 1L &&
    summary$configured_concurrency <= 32L &&
    summary$maximum_host_calls_in_flight > 1L &&
    summary$maximum_setup_stream_count > 1L &&
    isTRUE(summary$observed_concurrent_execution) &&
    isTRUE(summary$numerical_gate) && isTRUE(summary$optimizer_gate) &&
    isTRUE(summary$stability_replay_gate) &&
    isTRUE(summary$terminal_boundary_confirmation_gate) &&
    isTRUE(summary$physical_evaluation_accounting_gate) &&
    isTRUE(summary$authority_gate) && isTRUE(summary$concurrency_gate) &&
    isTRUE(summary$downstream_decision_gate) &&
    isTRUE(summary$backend_gate) &&
    identical(summary$source_closure_sha256,
              producer$producer_source_closure_sha256) &&
    identical(summary$native_binary_sha256,
              producer$native_binary_sha256) &&
    identical(summary$producer_identity_sha256, producer$identity_sha256) &&
    isTRUE(summary$phase6_only) && isTRUE(summary$shadow_only) &&
    identical(
      isTRUE(summary$full_residual_cuda_authoritative),
      identical(kind, "backend")
    ) && isTRUE(summary$performance_gate) &&
    (!identical(kind, "backend") || (
      isTRUE(summary$performance_evidence_present) &&
        is.finite(summary$candidate_residual_wall_ms) &&
        is.finite(summary$baseline_residual_wall_ms) &&
        summary$candidate_residual_wall_ms <
          summary$baseline_residual_wall_ms &&
        summary$candidate_to_baseline_ratio < 1
    )) &&
    !isTRUE(summary$production_backend_promoted) &&
    !isTRUE(summary$phase10_promotion_claim) && isTRUE(summary$pass)
  fastkpc_full_cuda_phase6_publication_require(
    common_gate, "Phase 6 artifact common gate failed"
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
          "cuda-dpstf2-guarded-qr-lapack-3.12-dgesdd") &&
    all(cases$optimizer_backend_executed ==
          "cuda-magic-multi-penalty") &&
    all(cases$residual_backend_executed ==
          "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual") &&
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
  fastkpc_full_cuda_phase6_publication_require(
    payload_gate, "Phase 6 artifact payload gate failed"
  )
  evidence <- readRDS(file.path(artifact_dir, "source_evidence.rds"))
  if (kind %in% c("full_shadow", "backend")) {
    logical <- readRDS(file.path(artifact_dir, "logical_ci_results.rds"))
    fastkpc_full_cuda_phase6_publication_require(
      nrow(logical) == 60324L && !any(logical$decision_flip) &&
        !any(logical$backend_error) && !any(logical$spectra_fallback),
      paste0("Phase 6 ", kind, " logical payload gate failed")
    )
  }
  if (identical(kind, "backend")) {
    performance <- readRDS(
      file.path(artifact_dir, "performance_evidence.rds")
    )
    fastkpc_full_cuda_phase6_publication_require(
      is.list(performance) && isTRUE(performance$summary$pass),
      "Phase 6 backend performance payload gate failed"
    )
    fastkpc_full_cuda_phase6_validate_performance_evidence(
      performance, evidence,
      merged_evidence_path = file.path(artifact_dir, "source_evidence.rds"),
      verify_current_inputs = verify_current_sources
    )
  }
  if (isTRUE(verify_current_sources)) {
    current_source <- fastkpc_full_cuda_phase6_evidence_source_closure()
    current_native <- fastkpc_full_cuda_phase6_native_identity()
    fastkpc_full_cuda_phase6_publication_require(
      identical(current_source$sha256, closure_sha256) &&
        identical(current_source$table, source_closure) &&
        identical(current_native$sha256, producer$native_binary_sha256),
      "Phase 6 artifact does not match current sources/binary"
    )
    implementation <- paste(c(
      readLines(
        "fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu", warn = FALSE
      ),
      readLines(
        "fastkpc/src/cuda/mgcv_multi_penalty_gcv.hpp", warn = FALSE
      ),
      readLines("fastkpc/R/full_cuda_ci_multi_penalty_cuda.R", warn = FALSE)
    ), collapse = "\n")
    fastkpc_full_cuda_phase6_publication_require(
      !grepl("[0-9a-f]{64}", implementation),
      "Phase 6 implementation contains target/setup-key-specific routing"
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

fastkpc_full_cuda_phase6_publish_one <- function(
    kind, output_dir, catalog, evidence, source_evidence_path,
    evidence_sha256, contracts, source_closure, native_identity) {
  kind <- match.arg(kind, fastkpc_full_cuda_phase6_artifact_kinds())
  producer_bundle <- fastkpc_full_cuda_phase6_producer(
    kind, catalog, source_closure, native_identity, contracts
  )
  output_parent <- dirname(output_dir)
  dir.create(output_parent, recursive = TRUE, showWarnings = FALSE)
  stage_dir <- tempfile(
    paste0(".phase6-", kind, "-stage-"), tmpdir = output_parent
  )
  dir.create(stage_dir, recursive = TRUE)
  stage_active <- TRUE
  on.exit({
    if (stage_active && dir.exists(stage_dir)) {
      unlink(stage_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  summary <- fastkpc_full_cuda_phase6_common_summary(
    kind, evidence, producer_bundle, contracts, source_closure,
    native_identity
  )
  fastkpc_full_cuda_phase6_write_payload(
    kind, stage_dir, catalog, evidence, source_evidence_path
  )
  fastkpc_full_cuda_phase6_write_table(
    source_closure$table, stage_dir, "source_closure.csv"
  )
  evidence_inputs <- data.frame(
    evidence_id = c(
      "merged_phase6", "accepted_phase5_oracle",
      "inherited_phase4_logical"
    ),
    sha256 = c(
      evidence_sha256,
      evidence$phase5_evidence_sha256,
      evidence$mixed_graph$inherited_phase4_logical_results_sha256
    ),
    semantic_role = c(
      "cuda-objective-residual-dcov-and-graph-authority",
      "accepted-cpp-multi-penalty-numerical-oracle",
      "accepted-single-penalty-logical-authority"
    ),
    stringsAsFactors = FALSE
  )
  if (identical(kind, "backend")) {
    evidence_inputs <- rbind(
      evidence_inputs,
      data.frame(
        evidence_id = "phase6_full_residual_performance",
        sha256 = evidence$performance_file_sha256,
        semantic_role = "same-trace-relative-performance-authority",
        stringsAsFactors = FALSE
      )
    )
  }
  fastkpc_full_cuda_phase6_write_table(
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
    paste0("Full CUDA CI Phase 6 ", kind)
  )
  writeLines(
    c(
      paste0("source evidence SHA-256: ", evidence_sha256),
      paste0("producer source closure SHA-256: ", source_closure$sha256),
      paste0("native binary SHA-256: ", native_identity$sha256),
      "see fastkpc/tools/run_full_cuda_ci_multi_penalty_cuda_artifacts.R"
    ),
    file.path(stage_dir, "commands.txt"), useBytes = TRUE
  )
  environment_lines <- c(
    fastkpc_full_cuda_environment_lines(),
    paste0("native_binary_path=", native_identity$path),
    paste0("native_binary_sha256=", native_identity$sha256),
    "phase6_execution=cuda-bounded-stream-shadow",
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
    "fastkpc/R/full_cuda_ci_phase6_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase6_performance.R",
    "fastkpc/R/full_cuda_ci_phase6_publication.R"
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
      "full-cuda-ci-phase6-multi-penalty-cuda-validator-v1",
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
    cuda_context_id = "cuda-device-0-phase6-bounded-streams",
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
      "full-cuda-ci-phase6-multi-penalty-cuda-artifact-manifest-v1",
    artifact_kind = kind,
    claim_scope = "phase6-additive-multi-penalty-S-size-greater-than-2",
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
  fastkpc_full_cuda_phase6_validate_artifact(
    stage_dir, expected_kind = kind, verify_current_sources = TRUE
  )
  backup_dir <- NULL
  if (dir.exists(output_dir)) {
    backup_dir <- tempfile(
      paste0(".phase6-", kind, "-backup-"), tmpdir = output_parent
    )
    fastkpc_full_cuda_phase6_publication_require(
      file.rename(output_dir, backup_dir),
      "Phase 6 prior artifact could not be staged for replacement"
    )
  }
  published <- file.rename(stage_dir, output_dir)
  if (!published) {
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop("Phase 6 artifact publication failed", call. = FALSE)
  }
  stage_active <- FALSE
  validated <- tryCatch(
    fastkpc_full_cuda_phase6_validate_artifact(
      output_dir, expected_kind = kind, verify_current_sources = TRUE
    ),
    error = identity
  )
  if (inherits(validated, "error")) {
    failed_dir <- tempfile(
      paste0(".phase6-", kind, "-failed-"), tmpdir = output_parent
    )
    file.rename(output_dir, failed_dir)
    if (!is.null(backup_dir)) file.rename(backup_dir, output_dir)
    stop(conditionMessage(validated), call. = FALSE)
  }
  if (!is.null(backup_dir)) unlink(backup_dir, recursive = TRUE, force = TRUE)
  validated
}

fastkpc_full_cuda_phase6_publish_artifacts <- function(
    catalog, evidence_path, performance_evidence_path,
    output_root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  evidence_path <- normalizePath(
    evidence_path, winslash = "/", mustWork = TRUE
  )
  evidence <- readRDS(evidence_path)
  fastkpc_full_cuda_phase6_validate_merged_evidence(
    evidence, catalog, verify_current_identity = TRUE
  )
  performance_evidence_path <- normalizePath(
    performance_evidence_path, winslash = "/", mustWork = TRUE
  )
  performance <- readRDS(performance_evidence_path)
  fastkpc_full_cuda_phase6_validate_performance_evidence(
    performance, evidence, merged_evidence_path = evidence_path,
    verify_current_inputs = TRUE
  )
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  source_closure <- fastkpc_full_cuda_phase6_evidence_source_closure()
  native_identity <- fastkpc_full_cuda_phase6_native_identity()
  evidence_sha256 <- fastkpc_full_cuda_census_file_hash(evidence_path)
  directories <- fastkpc_full_cuda_phase6_artifact_directory_names()
  results <- lapply(
    fastkpc_full_cuda_phase6_artifact_kinds(), function(kind) {
      artifact_evidence <- evidence
      if (identical(kind, "backend")) {
        artifact_evidence$performance <- performance
        artifact_evidence$performance_file_sha256 <-
          fastkpc_full_cuda_census_file_hash(performance_evidence_path)
      }
      fastkpc_full_cuda_phase6_publish_one(
        kind = kind,
        output_dir = file.path(output_root, directories[[kind]]),
        catalog = catalog,
        evidence = artifact_evidence,
        source_evidence_path = evidence_path,
        evidence_sha256 = evidence_sha256,
        contracts = contracts,
        source_closure = source_closure,
        native_identity = native_identity
      )
    }
  )
  names(results) <- fastkpc_full_cuda_phase6_artifact_kinds()
  results
}
