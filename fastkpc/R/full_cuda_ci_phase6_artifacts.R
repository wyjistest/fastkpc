fastkpc_full_cuda_phase6_artifact_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase6_condition_bucket <- function(rank, dimension,
                                                       condition) {
  if (rank < dimension || !is.finite(condition)) {
    return("rank-deficient-or-unauthenticated-condition")
  }
  if (condition < 1e8) return("finite-full-rank-lt-1e8")
  if (condition < 1e12) return("finite-full-rank-1e8-to-lt-1e12")
  "finite-full-rank-ge-1e12"
}

fastkpc_full_cuda_phase6_evidence_source_paths <- function() {
  paths <- sort(unique(c(
    fastkpc_full_cuda_phase5_evidence_source_paths(),
    "fastkpc/R/full_cuda_ci_multi_penalty_cuda.R",
    "fastkpc/R/full_cuda_ci_phase6_artifacts.R",
    "fastkpc/tools/run_full_cuda_ci_multi_penalty_cuda_partition.R",
    "fastkpc/tools/merge_full_cuda_ci_multi_penalty_cuda_partitions.R"
  )), method = "radix")
  fastkpc_full_cuda_phase6_artifact_require(
    length(paths) > 0L && all(file.exists(paths) & !dir.exists(paths)),
    "Phase 6 evidence source closure contains a missing file"
  )
  paths
}

fastkpc_full_cuda_phase6_evidence_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase6_evidence_source_paths()
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

fastkpc_full_cuda_phase6_native_identity <- function() {
  fastkpc_full_cuda_phase5_native_identity()
}

fastkpc_full_cuda_phase6_execution_identity <- function(
    catalog, phase5_evidence_path = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "multi_penalty_cpp_full_shadow_v1", "source_evidence.rds"
    )) {
  phase5_evidence_path <- normalizePath(
    phase5_evidence_path, winslash = "/", mustWork = TRUE
  )
  native <- fastkpc_full_cuda_phase5_native_identity()
  source <- fastkpc_full_cuda_phase6_evidence_source_closure()
  value <- list(
    schema_version = "full-cuda-ci-phase6-execution-identity-v1",
    source_commit = fastkpc_full_cuda_source_commit(),
    producer_source_closure_sha256 = source$sha256,
    native_binary_sha256 = native$sha256,
    route_semantic_version =
      "full-cuda-ci-phase6-multi-penalty-cuda-shadow-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    phase5_evidence_sha256 =
      fastkpc_full_cuda_census_file_hash(phase5_evidence_path)
  )
  value$identity_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(value)
  value
}

fastkpc_full_cuda_phase6_shadow_summary <- function(
    targets, logical_rows, timings, setup_count, run_dcov,
    elapsed_seconds) {
  numerical <- fastkpc_full_cuda_phase35_load_contract(
    "numerical_contract_v1"
  )
  tolerance <- numerical$payload$tolerances
  number <- function(value) as.numeric(value)
  numerical_gate <-
    max(targets$selected_log_sp_max_error) <= 1e-6 &&
    max(abs(targets$score_error)) <=
      number(tolerance$gcv_cp_score$absolute) &&
    max(abs(targets$edf_error)) <= number(tolerance$edf$absolute) &&
    max(targets$coefficient_shadow_max_absolute) <= 1e-12 &&
    max(targets$fitted_gemm_max_absolute) <= 1e-8 &&
    max(targets$residual_identity_max_absolute) <= 1e-12
  optimizer_gate <-
    sum(targets$optimizer_iteration_mismatch) == 0L &&
    sum(targets$score_call_mismatch) == 0L &&
    sum(targets$objective_call_mismatch) == 0L &&
    sum(targets$step_halving_mismatch) == 0L &&
    sum(targets$boundary_probe_mismatch) == 0L &&
    sum(targets$boundary_status_mismatch) == 0L &&
    sum(targets$convergence_mismatch) == 0L &&
    sum(targets$hessian_state_mismatch) == 0L &&
    sum(targets$rank_mismatch) == 0L &&
    sum(targets$optimizer_status != 0L) == 0L &&
    all(targets$fully_converged) && all(targets$all_finite)
  replay_discarded_evaluations <-
    timings$cuda_stability_replay_discarded_complete_evaluation_count +
      timings$cuda_stability_replay_discarded_score_only_evaluation_count
  replay_discarded_factorizations <-
    timings$cuda_stability_replay_discarded_guarded_qr_evaluation_count +
      timings$cuda_stability_replay_discarded_stable_svd_evaluation_count
  confirmation_evaluations <-
    timings$cuda_terminal_boundary_confirmation_complete_evaluation_count
  confirmation_factorizations <-
    timings$cuda_terminal_boundary_confirmation_stable_svd_evaluation_count
  physical_evaluations <-
    timings$cuda_complete_evaluation_count +
      timings$cuda_score_only_evaluation_count +
      replay_discarded_evaluations + confirmation_evaluations
  physical_factorizations <-
    timings$cuda_guarded_qr_evaluation_count +
      timings$cuda_stable_svd_evaluation_count +
      replay_discarded_factorizations + confirmation_factorizations
  stability_replay_gate <-
    all(timings$cuda_stability_replay_kernel_launch_count == 1L) &&
    all(timings$cuda_stability_merge_kernel_launch_count == 1L) &&
    all(timings$cuda_stability_replay_target_count >= 0L) &&
    all(timings$cuda_stability_replay_target_count <=
          timings$target_count) &&
    all(timings$cuda_stability_replay_selected_count >= 0L) &&
    all(timings$cuda_stability_replay_selected_count <=
          timings$cuda_stability_replay_target_count) &&
    sum(timings$cuda_stability_replay_error_count) == 0L &&
    all(replay_discarded_evaluations == replay_discarded_factorizations) &&
    all(
      (timings$cuda_stability_replay_target_count == 0L &
         replay_discarded_evaluations == 0L) |
        (timings$cuda_stability_replay_target_count > 0L &
           replay_discarded_evaluations > 0L)
    ) &&
    all(is.finite(timings$cuda_stability_replay_max_log_sp_spread)) &&
    all(timings$cuda_stability_replay_max_log_sp_spread >= 0) &&
    all(
      timings$cuda_stability_replay_selected_count == 0L |
        timings$cuda_stability_replay_max_log_sp_spread > 1e-7
    )
  terminal_boundary_confirmation_gate <-
    all(timings$cuda_terminal_boundary_confirmation_count >= 0L) &&
    all(timings$cuda_terminal_boundary_confirmation_accepted_count >= 0L) &&
    all(timings$cuda_terminal_boundary_confirmation_rejected_count >= 0L) &&
    all(
      timings$cuda_terminal_boundary_confirmation_accepted_count +
        timings$cuda_terminal_boundary_confirmation_rejected_count ==
        timings$cuda_terminal_boundary_confirmation_count
    ) &&
    all(confirmation_evaluations ==
          2L * timings$cuda_terminal_boundary_confirmation_count) &&
    all(confirmation_factorizations == confirmation_evaluations) &&
    all(is.finite(timings$cuda_terminal_boundary_confirmation_cycles)) &&
    all(timings$cuda_terminal_boundary_confirmation_cycles >= 0) &&
    all(is.finite(
      timings$cuda_terminal_boundary_confirmation_max_identity_disagreement
    )) &&
    all(
      timings$cuda_terminal_boundary_confirmation_max_identity_disagreement >=
        0
    ) &&
    all(is.finite(
      timings$cuda_terminal_boundary_confirmation_max_identity_ratio
    )) &&
    all(timings$cuda_terminal_boundary_confirmation_max_identity_ratio >= 0) &&
    all(
      timings$cuda_terminal_boundary_confirmation_count == 0L |
        timings$cuda_terminal_boundary_confirmation_cycles > 0
    )
  physical_evaluation_accounting_gate <-
    all(physical_evaluations == physical_factorizations)
  authority_gate <-
    sum(timings$legacy_mgcv_target_calls) == 0L &&
    sum(timings$cpu_multi_penalty_solve_count) == 0L &&
    sum(timings$fallback_count) == 0L &&
    sum(timings$cuda_error_count) == 0L &&
    all(timings$setup_upload_count == 1L) &&
    all(timings$setup_device_allocation_count == 1L) &&
    all(timings$workspace_grow_count == 0L) &&
    all(timings$solve_device_allocation_count == 0L) &&
    all(timings$cublas_gemm_count == 1L) &&
    all(timings$residual_kernel_count == 1L) &&
    all(timings$cuda_complete_evaluation_count +
          timings$cuda_score_only_evaluation_count +
          timings$cuda_selected_evaluation_reuse_count ==
          timings$cuda_objective_count) &&
    all(timings$cuda_guarded_qr_evaluation_count +
          timings$cuda_stable_svd_evaluation_count ==
          timings$cuda_complete_evaluation_count +
            timings$cuda_score_only_evaluation_count) &&
    sum(timings$cuda_selected_evaluation_reuse_count) ==
      sum(targets$boundary_accepted_count == 0L) &&
    stability_replay_gate && terminal_boundary_confirmation_gate &&
    physical_evaluation_accounting_gate
  windows <- timings[timings$scheduler_window_leader, , drop = FALSE]
  concurrent_windows <- windows$scheduler_window_setup_count > 1L
  concurrency_gate <-
    nrow(windows) > 0L &&
    all(timings$execution_strategy ==
          "bounded-independent-prepared-streams-v1") &&
    all(timings$scheduler_window_setup_count >= 1L) &&
    all(timings$scheduler_window_setup_count <=
          timings$configured_concurrency) &&
    all(windows$requested_concurrency == windows$worker_count) &&
    all(windows$worker_count == pmin(
      windows$configured_concurrency,
      windows$scheduler_window_setup_count
    )) &&
    all(windows$max_host_calls_in_flight == windows$worker_count) &&
    all(windows$setup_stream_count == windows$scheduler_window_setup_count) &&
    all(windows$host_overlap_factor > 0) &&
    (!any(concurrent_windows) || any(
      concurrent_windows & windows$max_host_calls_in_flight > 1L &
        windows$host_overlap_factor > 1
    ))
  downstream_gate <- !isTRUE(run_dcov) || (
    sum(logical_rows$decision_flip) == 0L &&
      sum(logical_rows$backend_error) == 0L &&
      sum(logical_rows$spectra_fallback) == 0L &&
      max(logical_rows$absolute_p_value_difference) <=
        number(tolerance$p_value$absolute)
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cuda-shadow-summary-v1",
    setup_count = as.integer(setup_count),
    target_count = nrow(targets),
    logical_test_count = nrow(logical_rows),
    penalty_count_min = min(targets$penalty_count),
    penalty_count_max = max(targets$penalty_count),
    coefficient_dim_max = max(targets$free_dim),
    max_selected_log_sp_error = max(targets$selected_log_sp_max_error),
    max_score_absolute_error = max(abs(targets$score_error)),
    max_edf_absolute_error = max(abs(targets$edf_error)),
    max_coefficient_shadow_absolute_error =
      max(targets$coefficient_shadow_max_absolute),
    max_fitted_gemm_absolute_error =
      max(targets$fitted_gemm_max_absolute),
    max_residual_identity_absolute_error =
      max(targets$residual_identity_max_absolute),
    optimizer_iteration_mismatch_count =
      sum(targets$optimizer_iteration_mismatch),
    score_call_mismatch_count = sum(targets$score_call_mismatch),
    objective_call_mismatch_count = sum(targets$objective_call_mismatch),
    step_halving_mismatch_count = sum(targets$step_halving_mismatch),
    boundary_probe_mismatch_count = sum(targets$boundary_probe_mismatch),
    boundary_status_mismatch_count = sum(targets$boundary_status_mismatch),
    convergence_mismatch_count = sum(targets$convergence_mismatch),
    hessian_state_mismatch_count = sum(targets$hessian_state_mismatch),
    rank_mismatch_count = sum(targets$rank_mismatch),
    cuda_optimizer_error_count = sum(targets$optimizer_status != 0L),
    cuda_objective_call_count = sum(targets$objective_calls),
    cuda_score_call_count = sum(targets$score_calls),
    cuda_step_halving_count = sum(targets$step_halving_count),
    cuda_boundary_probe_count = sum(targets$boundary_probe_count),
    cuda_boundary_accepted_count = sum(targets$boundary_accepted_count),
    cuda_penalty_factor_augmentation_cycles =
      sum(timings$cuda_penalty_factor_augmentation_cycles),
    cuda_qr_svd_cycles = sum(timings$cuda_qr_svd_cycles),
    cuda_qr_bidiagonal_reduction_cycles =
      sum(timings$cuda_qr_bidiagonal_reduction_cycles),
    cuda_bidiagonal_svd_cycles =
      sum(timings$cuda_bidiagonal_svd_cycles),
    cuda_svd_vector_postback_cycles =
      sum(timings$cuda_svd_vector_postback_cycles),
    cuda_left_vector_product_cycles =
      sum(timings$cuda_left_vector_product_cycles),
    cuda_score_construction_cycles =
      sum(timings$cuda_score_construction_cycles),
    cuda_derivative_hessian_cycles =
      sum(timings$cuda_derivative_hessian_cycles),
    cuda_complete_evaluation_count =
      sum(timings$cuda_complete_evaluation_count),
    cuda_score_only_evaluation_count =
      sum(timings$cuda_score_only_evaluation_count),
    cuda_selected_evaluation_reuse_count =
      sum(timings$cuda_selected_evaluation_reuse_count),
    cuda_guarded_qr_evaluation_count =
      sum(timings$cuda_guarded_qr_evaluation_count),
    cuda_stable_svd_evaluation_count =
      sum(timings$cuda_stable_svd_evaluation_count),
    cuda_stability_replay_kernel_launch_count =
      sum(timings$cuda_stability_replay_kernel_launch_count),
    cuda_stability_merge_kernel_launch_count =
      sum(timings$cuda_stability_merge_kernel_launch_count),
    cuda_stability_replay_target_count =
      sum(timings$cuda_stability_replay_target_count),
    cuda_stability_replay_selected_count =
      sum(timings$cuda_stability_replay_selected_count),
    cuda_stability_replay_error_count =
      sum(timings$cuda_stability_replay_error_count),
    cuda_stability_replay_discarded_complete_evaluation_count = sum(
      timings$cuda_stability_replay_discarded_complete_evaluation_count
    ),
    cuda_stability_replay_discarded_score_only_evaluation_count = sum(
      timings$cuda_stability_replay_discarded_score_only_evaluation_count
    ),
    cuda_stability_replay_discarded_guarded_qr_evaluation_count = sum(
      timings$cuda_stability_replay_discarded_guarded_qr_evaluation_count
    ),
    cuda_stability_replay_discarded_stable_svd_evaluation_count = sum(
      timings$cuda_stability_replay_discarded_stable_svd_evaluation_count
    ),
    cuda_stability_replay_discarded_cycles =
      sum(timings$cuda_stability_replay_discarded_cycles),
    cuda_stability_replay_max_log_sp_spread =
      max(timings$cuda_stability_replay_max_log_sp_spread),
    cuda_terminal_boundary_confirmation_count =
      sum(timings$cuda_terminal_boundary_confirmation_count),
    cuda_terminal_boundary_confirmation_accepted_count =
      sum(timings$cuda_terminal_boundary_confirmation_accepted_count),
    cuda_terminal_boundary_confirmation_rejected_count =
      sum(timings$cuda_terminal_boundary_confirmation_rejected_count),
    cuda_terminal_boundary_confirmation_complete_evaluation_count = sum(
      timings$cuda_terminal_boundary_confirmation_complete_evaluation_count
    ),
    cuda_terminal_boundary_confirmation_stable_svd_evaluation_count = sum(
      timings$cuda_terminal_boundary_confirmation_stable_svd_evaluation_count
    ),
    cuda_terminal_boundary_confirmation_cycles =
      sum(timings$cuda_terminal_boundary_confirmation_cycles),
    cuda_terminal_boundary_confirmation_max_identity_disagreement = max(
      timings$cuda_terminal_boundary_confirmation_max_identity_disagreement
    ),
    cuda_terminal_boundary_confirmation_max_identity_ratio =
      max(timings$cuda_terminal_boundary_confirmation_max_identity_ratio),
    cuda_physical_evaluation_count = sum(physical_evaluations),
    cuda_physical_factorization_count = sum(physical_factorizations),
    legacy_mgcv_target_calls = sum(timings$legacy_mgcv_target_calls),
    cpu_multi_penalty_solve_count =
      sum(timings$cpu_multi_penalty_solve_count),
    setup_upload_count = sum(timings$setup_upload_count),
    workspace_grow_count = sum(timings$workspace_grow_count),
    solve_device_allocation_count =
      sum(timings$solve_device_allocation_count),
    cublas_gemm_count = sum(timings$cublas_gemm_count),
    residual_kernel_count = sum(timings$residual_kernel_count),
    configured_concurrency = max(timings$configured_concurrency),
    scheduler_window_count = nrow(windows),
    concurrent_scheduler_window_count = sum(concurrent_windows),
    observed_concurrent_execution = any(
      concurrent_windows & windows$max_host_calls_in_flight > 1L &
        windows$host_overlap_factor > 1
    ),
    maximum_host_calls_in_flight =
      max(windows$max_host_calls_in_flight),
    maximum_setup_stream_count = max(windows$setup_stream_count),
    maximum_host_overlap_factor = max(windows$host_overlap_factor),
    summed_optimizer_window_wall_ms =
      sum(windows$optimizer_window_wall_ms),
    summed_optimizer_setup_host_ms =
      sum(windows$summed_setup_host_ms),
    validation_residual_shadow_d2h_count =
      sum(timings$validation_residual_shadow_d2h_count),
    dcov_component_request_count =
      sum(timings$dcov_component_request_count),
    dcov_component_cache_hit_count =
      sum(timings$dcov_component_cache_hit_count),
    dcov_component_cache_miss_count =
      sum(timings$dcov_component_cache_miss_count),
    dcov_spectra_decomposition_count =
      sum(timings$dcov_spectra_decomposition_count),
    fallback_count = sum(timings$fallback_count),
    max_absolute_p_value_difference = if (isTRUE(run_dcov)) {
      max(logical_rows$absolute_p_value_difference)
    } else {
      NA_real_
    },
    downstream_legacy_dcov_decision_flip_count = if (isTRUE(run_dcov)) {
      sum(logical_rows$decision_flip)
    } else {
      NA_integer_
    },
    near_alpha_decision_flip_count = if (isTRUE(run_dcov)) {
      sum(logical_rows$decision_flip & logical_rows$near_alpha)
    } else {
      NA_integer_
    },
    summed_setup_ms = sum(timings$setup_ms),
    summed_optimizer_ms = sum(timings$optimizer_ms),
    summed_validation_ms = sum(timings$validation_ms),
    summed_dcov_ms = sum(timings$dcov_ms),
    elapsed_seconds = as.numeric(elapsed_seconds),
    numerical_gate = numerical_gate,
    optimizer_gate = optimizer_gate,
    stability_replay_gate = stability_replay_gate,
    terminal_boundary_confirmation_gate =
      terminal_boundary_confirmation_gate,
    physical_evaluation_accounting_gate =
      physical_evaluation_accounting_gate,
    authority_gate = authority_gate,
    concurrency_gate = concurrency_gate,
    downstream_decision_gate = downstream_gate,
    backend_gate = authority_gate && concurrency_gate && optimizer_gate &&
      all(targets$optimizer_backend_executed ==
            "cuda-magic-multi-penalty") &&
      all(targets$residual_backend_executed ==
            "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual") &&
      all(targets$fallback_reason == "NONE") &&
      !any(targets$normal_equations_used),
    pass = numerical_gate && optimizer_gate && authority_gate &&
      concurrency_gate &&
      downstream_gate,
    numerical_contract_sha256 = numerical$sha256
  )
}

fastkpc_full_cuda_phase6_scan_partition <- function(
    catalog, phase5_evidence, phase5_evidence_path,
    max_setups = NULL, run_dcov = TRUE,
    partition_id = NULL, partition_count = NULL,
    concurrency = 32L, progress = interactive()) {
  concurrency <- as.integer(concurrency)
  fastkpc_full_cuda_phase6_artifact_require(
    length(concurrency) == 1L && !is.na(concurrency) &&
      concurrency >= 1L && concurrency <= 32L,
    "Phase 6 concurrency must be one integer in [1, 32]"
  )
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  all_setup_keys <- as.character(scope$setup_rows$prepared_s_key_sha256)
  if (!is.null(max_setups)) {
    max_setups <- as.integer(max_setups)
    fastkpc_full_cuda_phase6_artifact_require(
      length(max_setups) == 1L && !is.na(max_setups) && max_setups > 0L,
      "Phase 6 max_setups must be positive"
    )
    all_setup_keys <- head(all_setup_keys, max_setups)
  }
  setup_scope_index <- match(
    all_setup_keys, as.character(scope$setup_rows$prepared_s_key_sha256)
  )
  partition <- fastkpc_full_cuda_phase5_shadow_partition(
    all_setup_keys, scope$shard_id[setup_scope_index],
    partition_id, partition_count
  )
  all_setup_keys <- partition$setup_keys
  selected_setup <- scope$setup_rows$prepared_s_key_sha256 %in%
    all_setup_keys
  scope$setup_rows <- scope$setup_rows[selected_setup, , drop = FALSE]
  scope$setup_rank <- scope$setup_rank[selected_setup]
  scope$shard_id <- scope$shard_id[selected_setup]
  scope$shard_ids <- sort(unique(scope$shard_id))
  scope$target_rows <- scope$target_rows[
    scope$target_rows$prepared_s_key_sha256 %in% all_setup_keys,
    , drop = FALSE
  ]
  phase5_targets <- phase5_evidence$targets
  fastkpc_full_cuda_phase6_artifact_require(
    identical(
      phase5_evidence$schema_version,
      "full-cuda-ci-multi-penalty-cpp-full-shadow-merged-v1"
    ) && nrow(phase5_targets) == 65676L &&
      !anyDuplicated(phase5_targets$residual_key_sha256),
    "Phase 6 immutable Phase 5 oracle is malformed"
  )
  logical_tests <- data.frame()
  if (isTRUE(run_dcov)) {
    plan <- fastkpc_full_cuda_shadow_plan(catalog)
    logical_tests <- plan$conditional_tests[
      plan$conditional_tests$prepared_s_key_x %in% all_setup_keys,
      , drop = FALSE
    ]
    fastkpc_full_cuda_phase6_artifact_require(
      all(logical_tests$prepared_s_key_y %in% all_setup_keys) &&
        all(logical_tests$prepared_s_key_x ==
              logical_tests$prepared_s_key_y) &&
        all(logical_tests$level > 2L),
      "Phase 6 logical setup ownership is malformed"
    )
  }
  preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
    inputs = catalog$inputs,
    shard_count = catalog$catalog_contract$shard_count,
    expected_source_commit = catalog$phase2_manifest$source_commit
  )
  target_parts <- vector("list", length(all_setup_keys))
  logical_parts <- if (isTRUE(run_dcov)) {
    vector("list", length(all_setup_keys))
  } else {
    list()
  }
  timing_parts <- vector("list", length(all_setup_keys))
  setup_ordinal <- 0L
  started <- proc.time()[["elapsed"]]
  scheduler_window_id <- 0L
  for (shard_id in scope$shard_ids) {
    shard <- fastkpc_full_cuda_phase5_read_shard(
      catalog, scope, shard_id, preparation = preparation
    )
    shard_setup_keys <- names(shard$setups)
    window_starts <- seq.int(1L, length(shard_setup_keys), by = concurrency)
    for (window_start in window_starts) {
      scheduler_window_id <- scheduler_window_id + 1L
      window_end <- min(
        length(shard_setup_keys), window_start + concurrency - 1L
      )
      window_keys <- shard_setup_keys[window_start:window_end]
      contexts <- vector("list", length(window_keys))
      handles <- vector("list", length(window_keys))
      tokens <- vector("list", length(window_keys))
      tryCatch({
        for (window_index in seq_along(window_keys)) {
          setup_key <- window_keys[[window_index]]
          batch <- fastkpc_full_cuda_phase5_batch_from_shard(
            catalog, shard, setup_key
          )
          target_count <- length(batch$target_keys)
          expected_index <- match(
            batch$target_keys, phase5_targets$residual_key_sha256
          )
          fastkpc_full_cuda_phase6_artifact_require(
            !anyNA(expected_index),
            "Phase 6 target is missing from the immutable Phase 5 oracle"
          )
          setup_started <- proc.time()[["elapsed"]]
          cuda_setup <- fastkpc_full_cuda_phase6_prepare(batch$setup)
          handles[[window_index]] <-
            fastkpc_full_cuda_phase6_prepared_create(
              cuda_setup, target_capacity = target_count
            )
          contexts[[window_index]] <- list(
            setup_key = setup_key,
            batch = batch,
            target_count = target_count,
            expected = phase5_targets[expected_index, , drop = FALSE],
            cuda_setup = cuda_setup,
            setup_ms = 1000 *
              (proc.time()[["elapsed"]] - setup_started),
            before = full_cuda_ci_multi_penalty_gcv_prepared_info_native(
              handles[[window_index]]
            )
          )
        }
        window_concurrency <- min(concurrency, length(window_keys))
        optimizer_started <- proc.time()[["elapsed"]]
        executed <- fastkpc_full_cuda_phase6_optimize_prepared_multi(
          handles = handles,
          target_batches = lapply(contexts, function(value) value$batch$Y),
          target_keys = lapply(
            contexts, function(value) value$batch$target_keys
          ),
          concurrency = window_concurrency
        )
        optimizer_window_wall_ms <- 1000 *
          (proc.time()[["elapsed"]] - optimizer_started)

        for (window_index in seq_along(window_keys)) {
          setup_ordinal <- setup_ordinal + 1L
          context <- contexts[[window_index]]
          setup_key <- context$setup_key
          batch <- context$batch
          target_count <- context$target_count
          expected <- context$expected
          cuda_setup <- context$cuda_setup
          setup_ms <- context$setup_ms
          before <- context$before
          result <- executed$setups[[window_index]]
          candidate <- result$optimization
          tokens[[window_index]] <- result$residual
          optimizer_ms <- candidate$diagnostics$total_host_ms
          validation_started <- proc.time()[["elapsed"]]
          shadow <- full_cuda_ci_multi_penalty_gcv_residual_shadow_native(
            tokens[[window_index]]
          )
          validation_ms <- 1000 *
            (proc.time()[["elapsed"]] - validation_started)
          after <- full_cuda_ci_multi_penalty_gcv_prepared_info_native(
            handles[[window_index]]
          )
      candidate_residuals <- shadow$residuals
      colnames(candidate_residuals) <- batch$target_keys
      fitted_cpu <- cuda_setup$X %*% shadow$coefficients
      residual_identity <- batch$Y - shadow$fitted
      target_parts[[setup_ordinal]] <- do.call(rbind, lapply(
        seq_len(target_count), function(target_index) {
          state <- batch$states[target_index, , drop = FALSE]
          convergence <- fastkpc_full_cuda_phase5_oracle_convergence(state)
          oracle_log_sp <- log(batch$oracle_sp[target_index, ])
          selected_log_sp <- candidate$selected_log_sp[, target_index]
          boundary_status <- paste(
            candidate$boundary_status[, target_index], collapse = ","
          )
          condition_bucket <- fastkpc_full_cuda_phase6_condition_bucket(
            candidate$numerical_rank[[target_index]],
            ncol(cuda_setup$X), candidate$condition[[target_index]]
          )
          data.frame(
            prepared_s_key_sha256 = setup_key,
            residual_key_sha256 = batch$target_keys[[target_index]],
            target = batch$target_ids[[target_index]],
            penalty_count = nrow(candidate$selected_log_sp),
            oracle_log_sp =
              fastkpc_full_cuda_phase5_encode_vector(oracle_log_sp),
            selected_log_sp =
              fastkpc_full_cuda_phase5_encode_vector(selected_log_sp),
            selected_log_sp_max_error =
              max(abs(selected_log_sp - oracle_log_sp)),
            oracle_score = as.numeric(
              batch$states$GCV_Cp_score[[target_index]]
            ),
            candidate_score = candidate$score[[target_index]],
            score_error = candidate$score[[target_index]] - as.numeric(
              batch$states$GCV_Cp_score[[target_index]]
            ),
            oracle_edf = as.numeric(batch$states$EDF[[target_index]]),
            candidate_edf = candidate$edf[[target_index]],
            edf_error = candidate$edf[[target_index]] -
              as.numeric(batch$states$EDF[[target_index]]),
            oracle_optimizer_iterations =
              convergence$optimizer_iterations,
            optimizer_iterations =
              candidate$optimizer_iterations[[target_index]],
            optimizer_iteration_mismatch =
              candidate$optimizer_iterations[[target_index]] !=
                convergence$optimizer_iterations,
            oracle_score_calls = convergence$score_calls,
            score_calls = candidate$score_calls[[target_index]],
            score_call_mismatch =
              candidate$score_calls[[target_index]] !=
                convergence$score_calls,
            oracle_objective_calls = expected$objective_calls[[target_index]],
            objective_calls = candidate$objective_calls[[target_index]],
            objective_call_mismatch =
              candidate$objective_calls[[target_index]] !=
                expected$objective_calls[[target_index]],
            oracle_step_halving_count =
              expected$step_halving_count[[target_index]],
            step_halving_count =
              candidate$step_halving_count[[target_index]],
            step_halving_mismatch =
              candidate$step_halving_count[[target_index]] !=
                expected$step_halving_count[[target_index]],
            oracle_boundary_probe_count =
              expected$boundary_probe_count[[target_index]],
            boundary_probe_count =
              candidate$boundary_probe_count[[target_index]],
            boundary_probe_mismatch =
              candidate$boundary_probe_count[[target_index]] !=
                expected$boundary_probe_count[[target_index]],
            boundary_accepted_count =
              candidate$boundary_accepted_count[[target_index]],
            oracle_boundary_status =
              expected$boundary_status[[target_index]],
            boundary_status = boundary_status,
            boundary_status_mismatch = boundary_status !=
              expected$boundary_status[[target_index]],
            oracle_fully_converged = convergence$fully_converged,
            fully_converged = candidate$fully_converged[[target_index]],
            convergence_mismatch =
              candidate$fully_converged[[target_index]] !=
                convergence$fully_converged,
            oracle_hessian_positive_definite =
              convergence$hessian_positive_definite,
            hessian_positive_definite =
              candidate$hessian_positive_definite[[target_index]],
            hessian_state_mismatch =
              candidate$hessian_positive_definite[[target_index]] !=
                convergence$hessian_positive_definite,
            oracle_rms_gradient = convergence$rms_gradient,
            rms_gradient = candidate$rms_gradient[[target_index]],
            rms_gradient_error = candidate$rms_gradient[[target_index]] -
              convergence$rms_gradient,
            oracle_rank = convergence$rank,
            numerical_rank = candidate$numerical_rank[[target_index]],
            free_dim = ncol(cuda_setup$X),
            rank_mismatch =
              candidate$numerical_rank[[target_index]] !=
                convergence$rank ||
              ncol(cuda_setup$X) != convergence$full_rank,
            rank_path = candidate$rank_path,
            selected_fit_refinement_path = candidate$rank_path,
            condition = candidate$condition[[target_index]],
            condition_bucket = condition_bucket,
            normal_equations_used =
              candidate$diagnostics$normal_equations_used,
            coefficient_shadow_max_absolute = max(abs(
              shadow$coefficients[, target_index] -
                candidate$coefficients[, target_index]
            )),
            fitted_gemm_max_absolute = max(abs(
              shadow$fitted[, target_index] -
                fitted_cpu[, target_index]
            )),
            residual_identity_max_absolute = max(abs(
              shadow$residuals[, target_index] -
                residual_identity[, target_index]
            )),
            all_finite = all(is.finite(c(
              candidate$score[[target_index]],
              candidate$edf[[target_index]],
              shadow$fitted[, target_index],
              shadow$residuals[, target_index], selected_log_sp
            ))),
            convergence_code =
              candidate$convergence_code[[target_index]],
            optimizer_status = candidate$optimizer_status[[target_index]],
            fallback_reason = "NONE",
            optimizer_backend_executed = "cuda-magic-multi-penalty",
            residual_backend_executed =
              "cuda-persistent-guarded-qr-dgesdd-gemm-device-residual",
            stringsAsFactors = FALSE
          )
        }
      ))

      setup_logical <- if (isTRUE(run_dcov)) {
        logical_tests[
          logical_tests$prepared_s_key_x == setup_key,
          , drop = FALSE
        ]
      } else {
        data.frame()
      }
      dcov_ms <- 0
      dcov_diagnostics <- list(
        component_request_count = 0L,
        component_cache_hit_count = 0L,
        component_cache_miss_count = 0L,
        lowrank_spectra_count = 0L
      )
      if (isTRUE(run_dcov) && nrow(setup_logical) > 0L) {
        dcov_started <- proc.time()[["elapsed"]]
        dcov <- fastkpc_full_cuda_phase5_compute_setup_rows(
          logical_tests = setup_logical,
          setup_key = setup_key,
          shard_id = shard_id,
          target_keys = batch$target_keys,
          residuals = candidate_residuals
        )
        logical_parts[[setup_ordinal]] <- dcov$rows
        dcov_diagnostics <- dcov$diagnostics
        dcov_ms <- 1000 * (proc.time()[["elapsed"]] - dcov_started)
      } else if (isTRUE(run_dcov)) {
        logical_parts[[setup_ordinal]] <- NULL
      }
      timing_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        target_count = target_count,
        logical_test_count = nrow(setup_logical),
        setup_ms = setup_ms,
        optimizer_ms = optimizer_ms,
        validation_ms = validation_ms,
        dcov_ms = dcov_ms,
        scheduler_window_id = scheduler_window_id,
        partition_id = partition$partition_id,
        scheduler_window_leader = window_index == 1L,
        scheduler_window_setup_count = length(window_keys),
        configured_concurrency = concurrency,
        execution_strategy = executed$diagnostics$execution_strategy,
        requested_concurrency =
          executed$diagnostics$requested_concurrency,
        worker_count = executed$diagnostics$worker_count,
        max_host_calls_in_flight =
          executed$diagnostics$max_host_calls_in_flight,
        setup_stream_count = executed$diagnostics$setup_stream_count,
        summed_setup_host_ms =
          executed$diagnostics$summed_setup_host_ms,
        optimizer_window_wall_ms = if (window_index == 1L) {
          optimizer_window_wall_ms
        } else {
          0
        },
        host_overlap_factor = executed$diagnostics$host_overlap_factor,
        setup_upload_count = before$setup_upload_count,
        setup_h2d_bytes = before$setup_h2d_bytes,
        setup_device_allocation_count = before$device_allocation_count,
        workspace_grow_count = after$workspace_grow_count,
        solve_device_allocation_count =
          candidate$diagnostics$device_allocation_count,
        cublas_gemm_count = after$cublas_gemm_count,
        residual_kernel_count = after$residual_kernel_count,
        validation_residual_shadow_d2h_count =
          after$residual_shadow_d2h_count,
        validation_residual_shadow_d2h_bytes =
          after$residual_shadow_d2h_bytes,
        cuda_objective_count =
          candidate$diagnostics$cuda_optimizer_objective_count,
        cuda_penalty_factor_augmentation_cycles =
          candidate$diagnostics$cuda_penalty_factor_augmentation_cycles,
        cuda_qr_svd_cycles = candidate$diagnostics$cuda_qr_svd_cycles,
        cuda_qr_bidiagonal_reduction_cycles =
          candidate$diagnostics$cuda_qr_bidiagonal_reduction_cycles,
        cuda_bidiagonal_svd_cycles =
          candidate$diagnostics$cuda_bidiagonal_svd_cycles,
        cuda_svd_vector_postback_cycles =
          candidate$diagnostics$cuda_svd_vector_postback_cycles,
        cuda_left_vector_product_cycles =
          candidate$diagnostics$cuda_left_vector_product_cycles,
        cuda_score_construction_cycles =
          candidate$diagnostics$cuda_score_construction_cycles,
        cuda_derivative_hessian_cycles =
          candidate$diagnostics$cuda_derivative_hessian_cycles,
        cuda_complete_evaluation_count =
          candidate$diagnostics$cuda_complete_evaluation_count,
        cuda_score_only_evaluation_count =
          candidate$diagnostics$cuda_score_only_evaluation_count,
        cuda_selected_evaluation_reuse_count =
          candidate$diagnostics$cuda_selected_evaluation_reuse_count,
        cuda_guarded_qr_evaluation_count =
          candidate$diagnostics$cuda_guarded_qr_evaluation_count,
        cuda_stable_svd_evaluation_count =
          candidate$diagnostics$cuda_stable_svd_evaluation_count,
        cuda_stability_replay_kernel_launch_count =
          candidate$diagnostics$cuda_stability_replay_kernel_launch_count,
        cuda_stability_merge_kernel_launch_count =
          candidate$diagnostics$cuda_stability_merge_kernel_launch_count,
        cuda_stability_replay_target_count =
          candidate$diagnostics$cuda_stability_replay_target_count,
        cuda_stability_replay_selected_count =
          candidate$diagnostics$cuda_stability_replay_selected_count,
        cuda_stability_replay_error_count =
          candidate$diagnostics$cuda_stability_replay_error_count,
        cuda_stability_replay_discarded_complete_evaluation_count =
          candidate$diagnostics[[
            "cuda_stability_replay_discarded_complete_evaluation_count"
          ]],
        cuda_stability_replay_discarded_score_only_evaluation_count =
          candidate$diagnostics[[
            "cuda_stability_replay_discarded_score_only_evaluation_count"
          ]],
        cuda_stability_replay_discarded_guarded_qr_evaluation_count =
          candidate$diagnostics[[
            "cuda_stability_replay_discarded_guarded_qr_evaluation_count"
          ]],
        cuda_stability_replay_discarded_stable_svd_evaluation_count =
          candidate$diagnostics[[
            "cuda_stability_replay_discarded_stable_svd_evaluation_count"
          ]],
        cuda_stability_replay_discarded_cycles =
          candidate$diagnostics$cuda_stability_replay_discarded_cycles,
        cuda_stability_replay_max_log_sp_spread =
          candidate$diagnostics$cuda_stability_replay_max_log_sp_spread,
        cuda_terminal_boundary_confirmation_count = candidate$diagnostics[[
          "cuda_terminal_boundary_confirmation_count"
        ]],
        cuda_terminal_boundary_confirmation_accepted_count =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_accepted_count"
          ]],
        cuda_terminal_boundary_confirmation_rejected_count =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_rejected_count"
          ]],
        cuda_terminal_boundary_confirmation_complete_evaluation_count =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_complete_evaluation_count"
          ]],
        cuda_terminal_boundary_confirmation_stable_svd_evaluation_count =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_stable_svd_evaluation_count"
          ]],
        cuda_terminal_boundary_confirmation_cycles =
          candidate$diagnostics$cuda_terminal_boundary_confirmation_cycles,
        cuda_terminal_boundary_confirmation_max_identity_disagreement =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_max_identity_disagreement"
          ]],
        cuda_terminal_boundary_confirmation_max_identity_ratio =
          candidate$diagnostics[[
            "cuda_terminal_boundary_confirmation_max_identity_ratio"
          ]],
        cuda_error_count = candidate$diagnostics$cuda_error_count,
        legacy_mgcv_target_calls = 0L,
        cpu_multi_penalty_solve_count = 0L,
        fallback_count = candidate$diagnostics$fallback_count,
        dcov_component_request_count =
          as.integer(dcov_diagnostics$component_request_count),
        dcov_component_cache_hit_count =
          as.integer(dcov_diagnostics$component_cache_hit_count),
        dcov_component_cache_miss_count =
          as.integer(dcov_diagnostics$component_cache_miss_count),
        dcov_spectra_decomposition_count =
          as.integer(dcov_diagnostics$lowrank_spectra_count),
        stringsAsFactors = FALSE
      )
      if (isTRUE(progress) && setup_ordinal %% 25L == 0L) {
        cat(
          "Phase 6 CUDA shadow setups:", setup_ordinal, "/",
          length(all_setup_keys), "\n"
        )
        flush.console()
      }
        }
      }, finally = {
        for (token in tokens) {
          if (!is.null(token)) try(
            full_cuda_ci_multi_penalty_gcv_residual_free_native(token),
            silent = TRUE
          )
        }
        for (handle in handles) {
          if (!is.null(handle)) try(
            full_cuda_ci_multi_penalty_gcv_prepared_free_native(handle),
            silent = TRUE
          )
        }
      })
    }
    rm(shard)
    gc(FALSE)
  }
  targets <- do.call(rbind, target_parts)
  timings <- do.call(rbind, timing_parts)
  rownames(targets) <- rownames(timings) <- NULL
  logical_rows <- if (isTRUE(run_dcov)) {
    nonempty <- Filter(Negate(is.null), logical_parts)
    value <- if (length(nonempty) == 0L) data.frame() else do.call(
      rbind, nonempty
    )
    if (nrow(value) > 0L) {
      value <- value[order(value$logical_sequence_id), , drop = FALSE]
      rownames(value) <- NULL
    }
    value
  } else {
    data.frame()
  }
  summary <- fastkpc_full_cuda_phase6_shadow_summary(
    targets, logical_rows, timings, length(all_setup_keys), run_dcov,
    proc.time()[["elapsed"]] - started
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cuda-shadow-partition-evidence-v1",
    partition = list(
      schema_version =
        "full-cuda-ci-multi-penalty-cuda-shadow-partition-v1",
      partition_id = partition$partition_id,
      partition_count = partition$partition_count,
      assignment_strategy = partition$assignment_strategy,
      configured_concurrency = concurrency,
      execution_strategy = "bounded-independent-prepared-streams-v1",
      shard_ids = partition$shard_ids,
      setup_keys = all_setup_keys
    ),
    summary = summary,
    execution_identity = fastkpc_full_cuda_phase6_execution_identity(
      catalog, phase5_evidence_path
    ),
    targets = targets,
    logical_rows = logical_rows,
    timings = timings
  )
}

fastkpc_full_cuda_phase6_mixed_graph_replay <- function(
    catalog, phase6_logical_rows,
    phase4_artifact_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1"
    )) {
  inherited <- fastkpc_full_cuda_phase5_mixed_graph_replay(
    catalog = catalog,
    phase5_logical_rows = phase6_logical_rows,
    phase4_artifact_dir = phase4_artifact_dir
  )
  graph <- inherited$summary
  summary <- list(
    direct_legacy_logical_test_count =
      as.integer(graph$direct_legacy_logical_test_count),
    phase4_cuda_logical_test_count =
      as.integer(graph$phase4_cuda_logical_test_count),
    phase6_cuda_logical_test_count = 60324L,
    legacy_mgcv_target_call_count = 0L,
    cpu_residual_numerical_solve_count = 0L,
    residual_numerical_fallback_count = 0L,
    explicit_legacy_fallback_count =
      as.integer(graph$explicit_legacy_fallback_count),
    unknown_fallback_count = as.integer(graph$unknown_fallback_count),
    edge_count_reference = as.integer(graph$edge_count_reference),
    edge_count_candidate = as.integer(graph$edge_count_candidate),
    SHD = as.integer(graph$SHD),
    adjacency_identical = isTRUE(graph$adjacency_identical),
    sepsets_identical = isTRUE(graph$sepsets_identical),
    n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
    deletions_identical = isTRUE(graph$deletions_identical),
    pass = isTRUE(graph$pass)
  )
  fastkpc_full_cuda_phase6_artifact_require(
    summary$direct_legacy_logical_test_count == 2213L &&
      summary$phase4_cuda_logical_test_count == 177952L &&
      summary$phase6_cuda_logical_test_count == 60324L &&
      summary$edge_count_reference == 110L &&
      summary$edge_count_candidate == 110L && summary$SHD == 0L &&
      isTRUE(summary$adjacency_identical) &&
      isTRUE(summary$sepsets_identical) &&
      isTRUE(summary$n_edgetests_identical) &&
      isTRUE(summary$deletions_identical) && isTRUE(summary$pass),
    "Phase 6 mixed full-residual graph gate failed"
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cuda-mixed-graph-v1",
    summary = summary,
    inherited_phase4_logical_results_sha256 =
      inherited$inherited_phase4_logical_results_sha256,
    replay = inherited$replay,
    comparison = inherited$comparison
  )
}

fastkpc_full_cuda_phase6_merge_partitions <- function(
    catalog, evidence_paths, phase5_evidence_path,
    phase4_artifact_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1"
    )) {
  evidence_paths <- as.character(evidence_paths)
  phase5_evidence_path <- normalizePath(
    phase5_evidence_path, winslash = "/", mustWork = TRUE
  )
  fastkpc_full_cuda_phase6_artifact_require(
    length(evidence_paths) > 0L && !anyNA(evidence_paths) &&
      !anyDuplicated(evidence_paths) &&
      all(file.exists(evidence_paths) & !dir.exists(evidence_paths)),
    "Phase 6 partition paths are malformed"
  )
  parts <- lapply(evidence_paths, readRDS)
  partition_count <- unique(vapply(
    parts, function(value) as.integer(value$partition$partition_count),
    integer(1L)
  ))
  partition_ids <- vapply(
    parts, function(value) as.integer(value$partition$partition_id),
    integer(1L)
  )
  configured_concurrency <- unique(vapply(
    parts, function(value) {
      as.integer(value$partition$configured_concurrency)
    }, integer(1L)
  ))
  clean <- length(partition_count) == 1L &&
    partition_count == length(parts) &&
    identical(sort(partition_ids), seq.int(0L, partition_count - 1L)) &&
    length(configured_concurrency) == 1L &&
    configured_concurrency >= 1L && configured_concurrency <= 32L &&
    all(vapply(parts, function(value) {
      is.list(value) && identical(
        value$schema_version,
        "full-cuda-ci-multi-penalty-cuda-shadow-partition-evidence-v1"
      ) && identical(
        value$partition$schema_version,
        "full-cuda-ci-multi-penalty-cuda-shadow-partition-v1"
      ) && identical(
        value$partition$execution_strategy,
        "bounded-independent-prepared-streams-v1"
      ) && (
        identical(
          value$partition$assignment_strategy,
          "authenticated-shard-modulo-v1"
        ) || (
          partition_count == 1L && length(parts) == 1L && identical(
            value$partition$assignment_strategy,
            "all-authenticated-shards-v1"
          )
        )
      ) && is.integer(value$partition$shard_ids) &&
        length(value$partition$shard_ids) > 0L && identical(
          value$partition$shard_ids,
          sort(unique(value$partition$shard_ids))
        ) && identical(
          value$execution_identity$schema_version,
          "full-cuda-ci-phase6-execution-identity-v1"
        ) && isTRUE(value$summary$pass) &&
        isTRUE(value$summary$numerical_gate) &&
        isTRUE(value$summary$optimizer_gate) &&
        isTRUE(value$summary$authority_gate) &&
        isTRUE(value$summary$concurrency_gate) &&
        isTRUE(value$summary$downstream_decision_gate) &&
        isTRUE(value$summary$backend_gate)
    }, logical(1L)))
  fastkpc_full_cuda_phase6_artifact_require(
    clean, "Phase 6 partition evidence is incomplete"
  )
  execution_json <- vapply(parts, function(value) {
    fastkpc_full_cuda_phase35_canonical_json(value$execution_identity)
  }, character(1L))
  current_identity <- fastkpc_full_cuda_phase6_execution_identity(
    catalog, phase5_evidence_path
  )
  fastkpc_full_cuda_phase6_artifact_require(
    length(unique(execution_json)) == 1L && identical(
      execution_json[[1L]],
      fastkpc_full_cuda_phase35_canonical_json(current_identity)
    ),
    "Phase 6 partition producer execution identities differ"
  )
  order_index <- order(partition_ids)
  parts <- parts[order_index]
  evidence_paths <- evidence_paths[order_index]
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  expected_setup_keys <- as.character(
    scope$setup_rows$prepared_s_key_sha256
  )
  assignment_clean <- all(vapply(parts, function(value) {
    partition_id <- as.integer(value$partition$partition_id)
    all_shards <- identical(
      value$partition$assignment_strategy,
      "all-authenticated-shards-v1"
    )
    selected <- if (all_shards) {
      rep.int(TRUE, length(scope$shard_id))
    } else {
      scope$shard_id %% partition_count == partition_id
    }
    identical(
      as.integer(value$partition$shard_ids),
      sort(unique(as.integer(scope$shard_id[selected])))
    ) && identical(
      as.character(value$partition$setup_keys),
      expected_setup_keys[selected]
    )
  }, logical(1L)))
  fastkpc_full_cuda_phase6_artifact_require(
    assignment_clean,
    "Phase 6 authenticated-shard partition assignment drifted"
  )
  observed_setup_keys <- unlist(lapply(
    parts, function(value) value$partition$setup_keys
  ), use.names = FALSE)
  fastkpc_full_cuda_phase6_artifact_require(
    length(observed_setup_keys) == 7460L &&
      !anyDuplicated(observed_setup_keys) && identical(
        sort(observed_setup_keys, method = "radix"), expected_setup_keys
      ),
    "Phase 6 partitions do not exactly cover the setup corpus"
  )

  targets <- do.call(rbind, lapply(parts, `[[`, "targets"))
  timings <- do.call(rbind, lapply(parts, `[[`, "timings"))
  logical_rows <- do.call(rbind, lapply(parts, `[[`, "logical_rows"))
  expected_target_keys <- as.character(
    scope$target_rows$residual_key_sha256
  )
  target_match <- match(expected_target_keys, targets$residual_key_sha256)
  setup_match <- match(expected_setup_keys, timings$prepared_s_key_sha256)
  targets <- targets[target_match, , drop = FALSE]
  timings <- timings[setup_match, , drop = FALSE]
  logical_rows <- logical_rows[order(
    logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  rownames(targets) <- rownames(timings) <- rownames(logical_rows) <- NULL
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  expected_logical <- plan$conditional_tests[
    plan$conditional_tests$level > 2L, , drop = FALSE
  ]
  expected_logical <- expected_logical[order(
    expected_logical$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  merge_clean <- nrow(targets) == 65676L && !anyNA(target_match) &&
    !anyDuplicated(targets$residual_key_sha256) && identical(
      targets$residual_key_sha256, expected_target_keys
    ) && identical(
      targets$prepared_s_key_sha256,
      scope$target_rows$prepared_s_key_sha256
    ) && nrow(timings) == 7460L && !anyNA(setup_match) &&
    !anyDuplicated(timings$prepared_s_key_sha256) && identical(
      timings$prepared_s_key_sha256, expected_setup_keys
    ) && all(timings$partition_id %in% partition_ids) &&
    nrow(logical_rows) == 60324L &&
    !anyDuplicated(logical_rows$logical_sequence_id) && identical(
      logical_rows$logical_sequence_id,
      expected_logical$logical_sequence_id
    ) && identical(
      logical_rows$residual_key_x, expected_logical$residual_key_x
    ) && identical(
      logical_rows$residual_key_y, expected_logical$residual_key_y
    )
  fastkpc_full_cuda_phase6_artifact_require(
    merge_clean, "Phase 6 merged partition lineage is incomplete"
  )
  summary <- fastkpc_full_cuda_phase6_shadow_summary(
    targets, logical_rows, timings, 7460L, TRUE,
    max(vapply(parts, function(value) {
      as.numeric(value$summary$elapsed_seconds)
    }, numeric(1L)))
  )
  fastkpc_full_cuda_phase6_artifact_require(
    isTRUE(summary$pass) && isTRUE(summary$observed_concurrent_execution),
    "Phase 6 merged numerical/concurrency gate failed"
  )
  mixed_graph <- fastkpc_full_cuda_phase6_mixed_graph_replay(
    catalog, logical_rows, phase4_artifact_dir = phase4_artifact_dir
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cuda-full-shadow-merged-v1",
    summary = summary,
    execution_identity = current_identity,
    mixed_graph = mixed_graph,
    partition_count = partition_count,
    configured_concurrency = configured_concurrency,
    phase5_evidence_sha256 =
      fastkpc_full_cuda_census_file_hash(phase5_evidence_path),
    partition_file_sha256 = setNames(
      as.list(vapply(
        evidence_paths, fastkpc_full_cuda_census_file_hash, character(1L)
      )), basename(evidence_paths)
    ),
    parallel_elapsed_seconds = max(vapply(parts, function(value) {
      as.numeric(value$summary$elapsed_seconds)
    }, numeric(1L))),
    summed_partition_elapsed_seconds = sum(vapply(parts, function(value) {
      as.numeric(value$summary$elapsed_seconds)
    }, numeric(1L))),
    targets = targets,
    logical_rows = logical_rows,
    timings = timings
  )
}
