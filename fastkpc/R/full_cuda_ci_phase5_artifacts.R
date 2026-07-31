fastkpc_full_cuda_phase5_artifact_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase5_evidence_source_paths <- function() {
  native <- list.files(
    "fastkpc/src", recursive = TRUE, full.names = TRUE,
    include.dirs = FALSE
  )
  native <- native[grepl(
    "\\.(c|cc|cpp|cxx|cu|h|hh|hpp|hxx|cuh|inc)$", native
  )]
  phase5 <- c(
    "fastkpc/R/cuda_native.R",
    "fastkpc/R/full_cuda_ci_gate.R",
    "fastkpc/R/full_cuda_ci_oracle_contract.R",
    "fastkpc/R/full_cuda_ci_workload_census.R",
    "fastkpc/R/full_cuda_ci_prepared_s_contract.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_runtime.R",
    "fastkpc/R/full_cuda_ci_fixed_sp_shadow.R",
    "fastkpc/R/full_cuda_ci_phase3_artifacts.R",
    "fastkpc/R/full_cuda_ci_phase35_contracts.R",
    "fastkpc/R/full_cuda_ci_multi_penalty_cpp.R",
    "fastkpc/R/full_cuda_ci_phase5_artifacts.R",
    "fastkpc/tools/build_cuda_native.sh",
    "fastkpc/tools/run_full_cuda_ci_multi_penalty_cpp_partition.R",
    "fastkpc/tools/merge_full_cuda_ci_multi_penalty_cpp_partitions.R"
  )
  paths <- sort(unique(c(native, phase5)), method = "radix")
  fastkpc_full_cuda_phase5_artifact_require(
    length(paths) > 0L && all(file.exists(paths) & !dir.exists(paths)),
    "Phase 5 evidence source closure contains a missing file"
  )
  paths
}

fastkpc_full_cuda_phase5_evidence_source_closure <- function() {
  paths <- fastkpc_full_cuda_phase5_evidence_source_paths()
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

fastkpc_full_cuda_phase5_native_identity <- function() {
  load_fastkpc_cuda_native()
  dll <- getLoadedDLLs()[["fastkpc_cuda"]]
  fastkpc_full_cuda_phase5_artifact_require(
    !is.null(dll), "Phase 5 native DLL is not loaded"
  )
  path <- normalizePath(dll[["path"]], winslash = "/", mustWork = TRUE)
  list(path = path, sha256 = fastkpc_full_cuda_census_file_hash(path))
}

fastkpc_full_cuda_phase5_execution_identity <- function(catalog) {
  source <- fastkpc_full_cuda_phase5_evidence_source_closure()
  native <- fastkpc_full_cuda_phase5_native_identity()
  contracts <- fastkpc_full_cuda_phase35_load_contract_set()
  phase4_path <- file.path(
    "fastkpc", "artifacts", "full_cuda_ci",
    "single_penalty_cuda_gcv_full_shadow_v1", "logical_ci_results.rds"
  )
  inputs <- c(
    phase0_manifest = file.path(catalog$phase0_dir, "manifest.json"),
    phase1_manifest = file.path(catalog$phase1_dir, "manifest.json"),
    phase2_manifest = file.path(catalog$phase2_dir, "manifest.json"),
    inherited_phase4_logical_results = phase4_path
  )
  input_hashes <- setNames(as.list(vapply(
    inputs, fastkpc_full_cuda_census_file_hash, character(1L)
  )), names(inputs))
  contract_hashes <- setNames(lapply(
    contracts, `[[`, "sha256"
  ), names(contracts))
  value <- list(
    schema_version = "full-cuda-ci-phase5-execution-identity-v1",
    source_commit = fastkpc_full_cuda_source_commit(),
    producer_source_closure_sha256 = source$sha256,
    native_binary_sha256 = native$sha256,
    route_semantic_version =
      "full-cuda-ci-phase5-multi-penalty-cpp-shadow-v1",
    dataset_sha256 = catalog$inputs$dataset_sha256,
    input_file_sha256 = input_hashes,
    contract_sha256 = contract_hashes
  )
  value$identity_sha256 <- fastkpc_full_cuda_census_named_metadata_hash(
    value
  )
  value
}

fastkpc_full_cuda_phase5_multi_penalty_scope <- function(catalog) {
  scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
  selected_setup <- scope$setup_rows$penalty_count > 1L
  setup_rows <- scope$setup_rows[selected_setup, , drop = FALSE]
  setup_keys <- as.character(setup_rows$prepared_s_key_sha256)
  target_rows <- scope$target_rows[
    scope$target_rows$prepared_s_key_sha256 %in% setup_keys,
    , drop = FALSE
  ]
  setup_rank <- match(
    setup_keys, as.character(catalog$setup_index$prepared_s_key_sha256)
  )
  shard_count <- as.integer(catalog$catalog_contract$shard_count)
  shard_id <- as.integer((setup_rank - 1L) %% shard_count)
  clean <- nrow(setup_rows) == 7460L && nrow(target_rows) == 65676L &&
    !anyNA(setup_rank) && !anyDuplicated(setup_keys) &&
    !anyDuplicated(target_rows$residual_key_sha256) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    all(setup_rows$S_size > 2L) && all(setup_rows$penalty_count > 1L) &&
    all(shard_id >= 0L & shard_id < shard_count)
  fastkpc_full_cuda_phase5_artifact_require(
    clean, "Phase 5 canonical multi-penalty scope is malformed"
  )
  list(
    setup_rows = setup_rows,
    target_rows = target_rows,
    setup_rank = as.integer(setup_rank),
    shard_id = shard_id,
    shard_ids = sort(unique(shard_id))
  )
}

fastkpc_full_cuda_phase5_read_shard <- function(
    catalog, scope, shard_id, preparation = NULL,
    include_oracle_setups = TRUE) {
  shard_id <- as.integer(shard_id)
  fastkpc_full_cuda_phase5_artifact_require(
    is.logical(include_oracle_setups) &&
      length(include_oracle_setups) == 1L && !is.na(include_oracle_setups),
    "Phase 5 oracle-setup inclusion flag is malformed"
  )
  selected <- scope$shard_id == shard_id
  setup_keys <- as.character(
    scope$setup_rows$prepared_s_key_sha256[selected]
  )
  if (is.null(preparation)) {
    preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
      inputs = catalog$inputs,
      shard_count = catalog$catalog_contract$shard_count,
      expected_source_commit = catalog$phase2_manifest$source_commit
    )
  }
  completed <- .fastkpc_full_cuda_prepared_s_read_one_shard(
    shard_dir = file.path(catalog$phase2_dir, "shards"),
    shard_id = shard_id,
    inputs = catalog$inputs,
    preparation = preparation
  )
  payload <- completed$payload
  setup_match <- match(setup_keys, payload$ordered_setup_keys)
  fastkpc_full_cuda_phase5_artifact_require(
    !anyNA(setup_match), "Phase 5 shard is missing a selected setup"
  )
  setups <- if (isTRUE(include_oracle_setups)) {
    value <- payload$prepared_s_setups[setup_match]
    names(value) <- setup_keys
    value
  } else {
    NULL
  }
  target_rows <- scope$target_rows[
    scope$target_rows$prepared_s_key_sha256 %in% setup_keys,
    , drop = FALSE
  ]
  state_match <- match(
    as.character(target_rows$residual_key_sha256),
    as.character(payload$target_states$residual_key_sha256)
  )
  fastkpc_full_cuda_phase5_artifact_require(
    !anyNA(state_match), "Phase 5 shard is missing a selected target"
  )
  states <- payload$target_states[state_match, , drop = FALSE]
  fastkpc_full_cuda_phase5_artifact_require(
    identical(
      as.character(states$residual_key_sha256),
      as.character(target_rows$residual_key_sha256)
    ) && identical(
      as.character(states$prepared_s_key_sha256),
      as.character(target_rows$prepared_s_key_sha256)
    ),
    "Phase 5 selected target lineage is inconsistent"
  )
  list(
    shard_id = shard_id,
    setup_keys = setup_keys,
    setup_rows = scope$setup_rows[selected, , drop = FALSE],
    setups = setups,
    target_states = states,
    target_rows = target_rows
  )
}

fastkpc_full_cuda_phase5_batch_from_shard <- function(
    catalog, shard, setup_key, setup_override = NULL) {
  setup <- if (is.null(setup_override)) {
    shard$setups[[setup_key]]
  } else {
    setup_override
  }
  indices <- which(
    shard$target_states$prepared_s_key_sha256 == setup_key
  )
  states <- shard$target_states[indices, , drop = FALSE]
  metadata <- shard$target_rows[indices, , drop = FALSE]
  fastkpc_full_cuda_phase5_artifact_require(
    length(setup$penalty_blocks) > 1L && length(setup$sorted_S) > 2L &&
      nrow(states) > 0L && identical(
        as.character(states$residual_key_sha256),
        as.character(metadata$residual_key_sha256)
      ),
    "Phase 5 shard batch identity is malformed"
  )
  targets <- lapply(seq_len(nrow(states)), function(index) {
    fastkpc_full_cuda_materialize_target_state(
      states[index, , drop = FALSE], catalog$inputs$data,
      catalog$inputs$dataset_sha256
    )
  })
  contexts <- lapply(targets, function(target) {
    fastkpc_full_cuda_validate_materialized_target_for_prepared(
      setup, target
    )
  })
  Y <- do.call(cbind, lapply(contexts, `[[`, "y"))
  storage.mode(Y) <- "double"
  target_keys <- as.character(states$residual_key_sha256)
  colnames(Y) <- target_keys
  oracle_sp <- do.call(rbind, lapply(contexts, `[[`, "sp"))
  storage.mode(oracle_sp) <- "double"
  colnames(oracle_sp) <- setup$penalty_sp_labels
  rownames(oracle_sp) <- target_keys
  fastkpc_full_cuda_phase5_artifact_require(
    identical(target_keys, sort(target_keys, method = "radix")) &&
      nrow(oracle_sp) == ncol(Y) &&
      ncol(oracle_sp) == length(setup$penalty_blocks) &&
      all(is.finite(oracle_sp) & oracle_sp > 0),
    "Phase 5 materialized target batch is malformed"
  )
  list(
    setup = setup,
    states = states,
    metadata = metadata,
    targets = targets,
    contexts = contexts,
    Y = Y,
    oracle_sp = oracle_sp,
    target_keys = target_keys,
    target_ids = as.integer(states$target)
  )
}

fastkpc_full_cuda_phase5_oracle_convergence <- function(state_row) {
  value <- state_row$convergence_fields[[1L]]$mgcv.conv$value
  list(
    fully_converged = isTRUE(value$fully.converged),
    hessian_positive_definite = isTRUE(value$hess.pos.def),
    optimizer_iterations = as.integer(value$iter),
    score_calls = as.integer(value$score.calls),
    rms_gradient = as.numeric(value$rms.grad),
    rank = as.integer(value$rank),
    full_rank = as.integer(value$full.rank)
  )
}

fastkpc_full_cuda_phase5_encode_vector <- function(value) {
  paste(formatC(as.numeric(value), digits = 17L, format = "g"),
        collapse = ",")
}

fastkpc_full_cuda_phase5_vector_errors <- function(candidate, reference) {
  candidate <- as.numeric(candidate)
  reference <- as.numeric(reference)
  difference <- candidate - reference
  c(
    max_absolute = max(abs(difference)),
    relative_l2 = sqrt(sum(difference * difference)) /
      max(sqrt(sum(reference * reference)), 1e-300)
  )
}

fastkpc_full_cuda_phase5_transcript_keys <- function(catalog, scope) {
  path <- file.path(catalog$phase2_dir, "qualification_target_keys.rds")
  fastkpc_full_cuda_phase5_artifact_require(
    file.exists(path), "Phase 5 qualification target keys are missing"
  )
  qualification <- readRDS(path)
  qualification <- qualification[
    qualification$residual_key_sha256 %in%
      scope$target_rows$residual_key_sha256,
    , drop = FALSE
  ]
  qualification <- qualification[order(
    qualification$penalty_count, qualification$condition_bucket,
    -qualification$optimizer_iterations,
    qualification$residual_key_sha256,
    method = "radix"
  ), , drop = FALSE]
  group <- interaction(
    qualification$penalty_count, qualification$condition_bucket,
    drop = TRUE, lex.order = TRUE
  )
  representative <- unlist(lapply(
    split(seq_len(nrow(qualification)), group), head, 3L
  ), use.names = FALSE)
  mandatory <- which(
    qualification$rank_deficient |
      qualification$mgcv_warning |
      qualification$mgcv_nonconverged |
      qualification$nonfinite_metadata
  )
  near_alpha <- head(which(qualification$near_alpha), 50L)
  sort(unique(qualification$residual_key_sha256[c(
    representative, mandatory, near_alpha
  )]), method = "radix")
}

fastkpc_full_cuda_phase5_shadow_partition <- function(
    setup_keys, shard_ids, partition_id = NULL, partition_count = NULL) {
  setup_keys <- as.character(setup_keys)
  shard_ids <- as.integer(shard_ids)
  fastkpc_full_cuda_phase5_artifact_require(
    length(setup_keys) > 0L && length(shard_ids) == length(setup_keys) &&
      !anyNA(shard_ids) && all(shard_ids >= 0L),
    "Phase 5 shadow partition inputs are malformed"
  )
  partitioned <- !is.null(partition_id) || !is.null(partition_count)
  if (!partitioned) {
    return(list(
      partition_id = 0L, partition_count = 1L,
      assignment_strategy = "all-authenticated-shards-v1",
      shard_ids = sort(unique(shard_ids)), setup_keys = setup_keys
    ))
  }
  partition_id <- as.integer(partition_id)
  partition_count <- as.integer(partition_count)
  fastkpc_full_cuda_phase5_artifact_require(
    length(partition_id) == 1L && !is.na(partition_id) &&
      length(partition_count) == 1L && !is.na(partition_count) &&
      partition_count >= 1L && partition_count <= length(setup_keys) &&
      partition_id >= 0L && partition_id < partition_count,
    "Phase 5 shadow partition is malformed"
  )
  selected <- shard_ids %% partition_count == partition_id
  fastkpc_full_cuda_phase5_artifact_require(
    any(selected), "Phase 5 shadow partition has no authenticated shard"
  )
  list(
    partition_id = partition_id,
    partition_count = partition_count,
    assignment_strategy = "authenticated-shard-modulo-v1",
    shard_ids = sort(unique(shard_ids[selected])),
    setup_keys = setup_keys[selected]
  )
}

fastkpc_full_cuda_phase5_compute_setup_rows <- function(
    logical_tests, setup_key, shard_id, target_keys, residuals) {
  required <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
    "reference_p_value", "alpha", "reference_decision",
    "absolute_log_distance_from_alpha"
  )
  target_keys <- as.character(target_keys)
  clean <- is.data.frame(logical_tests) && nrow(logical_tests) > 0L &&
    length(setdiff(required, names(logical_tests))) == 0L &&
    !anyDuplicated(logical_tests$logical_sequence_id) &&
    all(logical_tests$level > 2L) &&
    identical(target_keys, sort(target_keys, method = "radix")) &&
    !anyDuplicated(target_keys) && is.matrix(residuals) &&
    typeof(residuals) == "double" && ncol(residuals) == length(target_keys) &&
    identical(colnames(residuals), target_keys) && all(is.finite(residuals))
  fastkpc_full_cuda_phase5_artifact_require(
    clean, "Phase 5 component-cache dCov inputs are malformed"
  )
  left <- match(logical_tests$residual_key_x, target_keys)
  right <- match(logical_tests$residual_key_y, target_keys)
  fastkpc_full_cuda_phase5_artifact_require(
    !anyNA(left) && !anyNA(right),
    "Phase 5 component-cache endpoint is missing"
  )
  oracle <- fastkpc_full_cuda_fixed_sp_with_legacy_dcov_backend(function() {
    legacy_dcov_gamma_cpp_component_cache_batch_native(
      residuals, left, right, numCol = 35L, index = 1
    )
  })
  diagnostics <- oracle$diagnostics
  expected_components <- length(unique(c(left, right)))
  diagnostic_clean <- is.list(diagnostics) &&
    diagnostics$n == nrow(residuals) &&
    diagnostics$pair_count == nrow(logical_tests) &&
    diagnostics$component_count == expected_components &&
    diagnostics$component_request_count == 2L * nrow(logical_tests) &&
    diagnostics$component_cache_miss_count == expected_components &&
    diagnostics$component_cache_hit_count ==
      2L * nrow(logical_tests) - expected_components &&
    identical(diagnostics$lowrank_mode, "spectra") &&
    diagnostics$lowrank_spectra_count == expected_components &&
    diagnostics$lowrank_spectra_converged_count == expected_components &&
    diagnostics$lowrank_spectra_failed_count == 0L &&
    diagnostics$lowrank_spectra_fallback_full_eig_count == 0L
  fastkpc_full_cuda_phase5_artifact_require(
    diagnostic_clean, "Phase 5 component-cache dCov diagnostics drifted"
  )
  p_value <- as.numeric(oracle$p.value)
  alpha <- as.numeric(logical_tests$alpha)
  reference <- as.numeric(logical_tests$reference_p_value)
  candidate_decision <- ifelse(
    p_value > alpha, "independent", "dependent"
  )
  log_distance <- as.numeric(
    logical_tests$absolute_log_distance_from_alpha
  )
  near_alpha <- is.finite(log_distance) & log_distance <= log(2)
  rows <- data.frame(
    logical_sequence_id = as.integer(logical_tests$logical_sequence_id),
    source_sequence_id = as.integer(logical_tests$source_sequence_id),
    source_task_index = as.integer(logical_tests$source_task_index),
    level = as.integer(logical_tests$level),
    x = as.integer(logical_tests$x),
    y = as.integer(logical_tests$y),
    S_key = as.character(logical_tests$S_key),
    residual_key_x = as.character(logical_tests$residual_key_x),
    residual_key_y = as.character(logical_tests$residual_key_y),
    prepared_s_key_sha256 = rep.int(setup_key, nrow(logical_tests)),
    shard_id = rep.int(as.integer(shard_id), nrow(logical_tests)),
    reference_p_value = reference,
    candidate_p_value = p_value,
    absolute_p_value_difference = abs(p_value - reference),
    alpha = alpha,
    reference_decision = as.character(logical_tests$reference_decision),
    candidate_decision = candidate_decision,
    decision_flip =
      candidate_decision != as.character(logical_tests$reference_decision),
    near_alpha = near_alpha,
    near_alpha_bucket = vapply(
      log_distance,
      fastkpc_full_cuda_census_near_alpha_bucket, character(1L)
    ),
    backend = rep.int("cpp-component-cache", nrow(logical_tests)),
    backend_version = rep.int(
      fastkpc_full_cuda_shadow_dcov_backend_version(), nrow(logical_tests)
    ),
    low_rank_backend = rep.int("spectra", nrow(logical_tests)),
    backend_error = rep.int(FALSE, nrow(logical_tests)),
    spectra_fallback = rep.int(FALSE, nrow(logical_tests)),
    stringsAsFactors = FALSE
  )
  rownames(rows) <- NULL
  list(rows = rows, diagnostics = diagnostics)
}

fastkpc_full_cuda_phase5_shadow_summary <- function(
    targets, logical_rows, timings, setup_count, run_dcov,
    elapsed_seconds, transcript_count) {
  numerical <- fastkpc_full_cuda_phase35_load_contract(
    "numerical_contract_v1"
  )
  tolerance <- numerical$payload$tolerances
  number <- function(value) as.numeric(value)
  numerical_gate <-
    max(abs(targets$score_error)) <=
      number(tolerance$gcv_cp_score$absolute) &&
    max(abs(targets$edf_error)) <= number(tolerance$edf$absolute) &&
    max(targets$fitted_max_absolute) <=
      number(tolerance$fitted$max_absolute) &&
    max(targets$fitted_relative_l2) <=
      number(tolerance$fitted$relative_l2) &&
    max(targets$residual_max_absolute) <=
      number(tolerance$residual$max_absolute) &&
    max(targets$residual_relative_l2) <=
      number(tolerance$residual$relative_l2)
  optimizer_gate <-
    sum(targets$optimizer_iteration_mismatch) == 0L &&
    sum(targets$score_call_mismatch) == 0L &&
    sum(targets$convergence_mismatch) == 0L &&
    sum(targets$hessian_state_mismatch) == 0L &&
    sum(targets$rank_mismatch) == 0L &&
    sum(targets$fallback_reason != "NONE") == 0L &&
    all(targets$fully_converged) && all(targets$all_finite)
  downstream_gate <- !isTRUE(run_dcov) || (
    sum(logical_rows$decision_flip) == 0L &&
      sum(logical_rows$backend_error) == 0L &&
      sum(logical_rows$spectra_fallback) == 0L &&
      max(logical_rows$absolute_p_value_difference) <=
        number(tolerance$p_value$absolute)
  )
  list(
    schema_version = "full-cuda-ci-multi-penalty-cpp-shadow-summary-v1",
    setup_count = as.integer(setup_count),
    target_count = nrow(targets),
    logical_test_count = nrow(logical_rows),
    penalty_count_min = min(targets$penalty_count),
    penalty_count_max = max(targets$penalty_count),
    max_selected_log_sp_error = max(targets$selected_log_sp_max_error),
    selected_log_sp_diagnostic_exceedance_count = sum(
      targets$selected_log_sp_max_error >
        number(tolerance$selected_log_sp$absolute_diagnostic)
    ),
    max_score_absolute_error = max(abs(targets$score_error)),
    max_edf_absolute_error = max(abs(targets$edf_error)),
    max_fitted_absolute_error = max(targets$fitted_max_absolute),
    max_fitted_relative_l2 = max(targets$fitted_relative_l2),
    max_residual_absolute_error = max(targets$residual_max_absolute),
    max_residual_relative_l2 = max(targets$residual_relative_l2),
    optimizer_iteration_mismatch_count =
      sum(targets$optimizer_iteration_mismatch),
    score_call_mismatch_count = sum(targets$score_call_mismatch),
    convergence_mismatch_count = sum(targets$convergence_mismatch),
    hessian_state_mismatch_count = sum(targets$hessian_state_mismatch),
    rank_mismatch_count = sum(targets$rank_mismatch),
    indefinite_hessian_target_count =
      sum(!targets$hessian_positive_definite),
    numerical_rank_deficient_target_count =
      sum(targets$numerical_rank < targets$free_dim),
    fallback_count = sum(targets$fallback_reason != "NONE"),
    candidate_legacy_mgcv_target_calls = 0L,
    validation_legacy_mgcv_fixed_sp_calls = nrow(targets),
    cpp_objective_call_count = sum(targets$objective_calls),
    cpp_score_call_count = sum(targets$score_calls),
    cpp_step_halving_count = sum(targets$step_halving_count),
    cpp_boundary_probe_count = sum(targets$boundary_probe_count),
    transcript_preserved_count = as.integer(transcript_count),
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
    summed_optimizer_ms = sum(timings$optimizer_ms),
    summed_validation_ms = sum(timings$validation_ms),
    summed_dcov_ms = sum(timings$dcov_ms),
    dcov_component_request_count =
      sum(timings$dcov_component_request_count),
    dcov_component_cache_hit_count =
      sum(timings$dcov_component_cache_hit_count),
    dcov_component_cache_miss_count =
      sum(timings$dcov_component_cache_miss_count),
    dcov_spectra_decomposition_count =
      sum(timings$dcov_spectra_decomposition_count),
    elapsed_seconds = as.numeric(elapsed_seconds),
    stable_rank_path_gate = all(
      targets$rank_path == "pivoted-qr-augmented-lapack-dgesdd-svd"
    ) &&
      all(targets$selected_fit_refinement_path ==
            "pivoted-qr-augmented-lapack-dgesdd-svd") &&
      !any(targets$normal_equations_used),
    numerical_gate = numerical_gate,
    optimizer_gate = optimizer_gate,
    downstream_decision_gate = downstream_gate,
    backend_gate = optimizer_gate &&
      all(targets$optimizer_backend_executed == "cpp-magic-multi-penalty") &&
      all(targets$residual_backend_executed ==
            "cpp-pivoted-qr-augmented-lapack-dgesdd-svd") &&
      all(targets$selected_fit_refinement_path ==
            "pivoted-qr-augmented-lapack-dgesdd-svd") &&
      all(targets$fallback_reason == "NONE"),
    pass = numerical_gate && optimizer_gate && downstream_gate &&
      all(targets$rank_path ==
            "pivoted-qr-augmented-lapack-dgesdd-svd") &&
      all(targets$selected_fit_refinement_path ==
            "pivoted-qr-augmented-lapack-dgesdd-svd") &&
      !any(targets$normal_equations_used),
    numerical_contract_sha256 = numerical$sha256
  )
}

fastkpc_full_cuda_phase5_scan_partition <- function(
    catalog, max_setups = NULL, run_dcov = TRUE,
    partition_id = NULL, partition_count = NULL,
    preserve_transcripts = TRUE, progress = interactive()) {
  scope <- fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
  all_setup_keys <- as.character(
    scope$setup_rows$prepared_s_key_sha256
  )
  if (!is.null(max_setups)) {
    max_setups <- as.integer(max_setups)
    fastkpc_full_cuda_phase5_artifact_require(
      length(max_setups) == 1L && !is.na(max_setups) && max_setups > 0L,
      "Phase 5 max_setups must be positive"
    )
    all_setup_keys <- head(all_setup_keys, max_setups)
  }
  setup_scope_index <- match(
    all_setup_keys, as.character(scope$setup_rows$prepared_s_key_sha256)
  )
  fastkpc_full_cuda_phase5_artifact_require(
    !anyNA(setup_scope_index),
    "Phase 5 partition setup is missing from the canonical scope"
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
  diagnostic_keys <- if (isTRUE(preserve_transcripts)) {
    fastkpc_full_cuda_phase5_transcript_keys(
      catalog, fastkpc_full_cuda_phase5_multi_penalty_scope(catalog)
    )
  } else {
    character()
  }
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  logical_tests <- plan$conditional_tests[
    plan$conditional_tests$prepared_s_key_x %in% all_setup_keys,
    , drop = FALSE
  ]
  fastkpc_full_cuda_phase5_artifact_require(
    all(logical_tests$prepared_s_key_y %in% all_setup_keys) &&
      all(logical_tests$prepared_s_key_x ==
            logical_tests$prepared_s_key_y) &&
      all(logical_tests$level > 2L),
    "Phase 5 logical setup ownership is malformed"
  )

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
  transcripts <- list()
  setup_ordinal <- 0L
  started <- proc.time()[["elapsed"]]
  for (shard_id in scope$shard_ids) {
    shard <- fastkpc_full_cuda_phase5_read_shard(
      catalog, scope, shard_id, preparation = preparation
    )
    for (setup_key in names(shard$setups)) {
      setup_ordinal <- setup_ordinal + 1L
      batch <- fastkpc_full_cuda_phase5_batch_from_shard(
        catalog, shard, setup_key
      )
      target_count <- length(batch$target_keys)
      candidate_results <- vector("list", target_count)
      oracle_results <- vector("list", target_count)
      target_rows <- vector("list", target_count)
      optimizer_started <- proc.time()[["elapsed"]]
      for (target_index in seq_len(target_count)) {
        state <- batch$states[target_index, , drop = FALSE]
        convergence <- fastkpc_full_cuda_phase5_oracle_convergence(state)
        key <- batch$target_keys[[target_index]]
        keep_transcript <- isTRUE(preserve_transcripts) && (
          key %in% diagnostic_keys ||
            !convergence$hessian_positive_definite ||
            convergence$rank < convergence$full_rank ||
            convergence$optimizer_iterations >= 50L
        )
        candidate <- fastkpc_full_cuda_phase5_optimize_cpp(
          batch$setup, batch$Y[, target_index],
          keep_transcript = keep_transcript
        )
        candidate_results[[target_index]] <- candidate
        if (keep_transcript) {
          transcripts[[key]] <- list(
            prepared_s_key_sha256 = setup_key,
            residual_key_sha256 = key,
            target = batch$target_ids[[target_index]],
            transcript = candidate$transcript
          )
        }
        target_rows[[target_index]] <- list(
          convergence = convergence,
          candidate = candidate
        )
      }
      optimizer_ms <- 1000 *
        (proc.time()[["elapsed"]] - optimizer_started)

      validation_started <- proc.time()[["elapsed"]]
      for (target_index in seq_len(target_count)) {
        oracle_results[[target_index]] <-
          fastkpc_mgcv_magic_fixed_sp_from_prepared(
            batch$setup, batch$targets[[target_index]]
          )
      }
      validation_ms <- 1000 *
        (proc.time()[["elapsed"]] - validation_started)
      candidate_residuals <- do.call(cbind, lapply(
        candidate_results, `[[`, "residuals"
      ))
      storage.mode(candidate_residuals) <- "double"
      colnames(candidate_residuals) <- batch$target_keys

      target_parts[[setup_ordinal]] <- do.call(rbind, lapply(
        seq_len(target_count), function(target_index) {
          key <- batch$target_keys[[target_index]]
          candidate <- candidate_results[[target_index]]
          oracle <- oracle_results[[target_index]]
          convergence <- target_rows[[target_index]]$convergence
          oracle_log_sp <- log(batch$oracle_sp[target_index, ])
          selected_log_sp <- as.numeric(candidate$selected_log_sp)
          fitted_error <- fastkpc_full_cuda_phase5_vector_errors(
            candidate$fitted, oracle$fitted
          )
          residual_error <- fastkpc_full_cuda_phase5_vector_errors(
            candidate$residuals, oracle$residuals
          )
          data.frame(
            prepared_s_key_sha256 = setup_key,
            residual_key_sha256 = key,
            target = batch$target_ids[[target_index]],
            penalty_count = candidate$penalty_count,
            oracle_log_sp =
              fastkpc_full_cuda_phase5_encode_vector(oracle_log_sp),
            selected_log_sp =
              fastkpc_full_cuda_phase5_encode_vector(selected_log_sp),
            selected_log_sp_max_error =
              max(abs(selected_log_sp - oracle_log_sp)),
            oracle_score = as.numeric(
              batch$states$GCV_Cp_score[[target_index]]
            ),
            candidate_score = candidate$score,
            score_error = candidate$score - as.numeric(
              batch$states$GCV_Cp_score[[target_index]]
            ),
            oracle_edf = as.numeric(batch$states$EDF[[target_index]]),
            candidate_edf = candidate$edf,
            edf_error = candidate$edf -
              as.numeric(batch$states$EDF[[target_index]]),
            oracle_optimizer_iterations =
              convergence$optimizer_iterations,
            optimizer_iterations = candidate$optimizer_iterations,
            optimizer_iteration_mismatch =
              candidate$optimizer_iterations !=
                convergence$optimizer_iterations,
            oracle_score_calls = convergence$score_calls,
            score_calls = candidate$score_calls,
            score_call_mismatch =
              candidate$score_calls != convergence$score_calls,
            objective_calls = candidate$objective_calls,
            step_halving_count = candidate$step_halving_count,
            boundary_probe_count = candidate$boundary_probe_count,
            boundary_status = paste(candidate$boundary_status,
                                    collapse = ","),
            oracle_fully_converged = convergence$fully_converged,
            fully_converged = candidate$fully_converged,
            convergence_mismatch =
              candidate$fully_converged !=
                convergence$fully_converged,
            oracle_hessian_positive_definite =
              convergence$hessian_positive_definite,
            hessian_positive_definite =
              candidate$hessian_positive_definite,
            hessian_state_mismatch =
              candidate$hessian_positive_definite !=
                convergence$hessian_positive_definite,
            oracle_rms_gradient = convergence$rms_gradient,
            rms_gradient = candidate$rms_gradient,
            oracle_rank = convergence$rank,
            numerical_rank = candidate$numerical_rank,
            free_dim = candidate$free_dim,
            rank_mismatch = candidate$numerical_rank != convergence$rank ||
              candidate$free_dim != convergence$full_rank,
            rank_path = candidate$rank_path,
            selected_fit_refinement_path =
              candidate$selected_fit_refinement_path,
            condition = candidate$condition,
            condition_bucket = candidate$condition_bucket,
            normal_equations_used = candidate$normal_equations_used,
            fitted_max_absolute = fitted_error[["max_absolute"]],
            fitted_relative_l2 = fitted_error[["relative_l2"]],
            residual_max_absolute = residual_error[["max_absolute"]],
            residual_relative_l2 = residual_error[["relative_l2"]],
            all_finite = all(is.finite(c(
              candidate$score, candidate$edf, candidate$fitted,
              candidate$residuals, candidate$selected_log_sp
            ))),
            convergence_code = candidate$convergence_code,
            fallback_reason = candidate$fallback_reason,
            optimizer_backend_executed = "cpp-magic-multi-penalty",
            residual_backend_executed =
              "cpp-pivoted-qr-augmented-lapack-dgesdd-svd",
            transcript_preserved = !is.null(transcripts[[key]]),
            stringsAsFactors = FALSE
          )
        }
      ))

      setup_logical <- logical_tests[
        logical_tests$prepared_s_key_x == setup_key,
        , drop = FALSE
      ]
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
        dcov_ms <- 1000 *
          (proc.time()[["elapsed"]] - dcov_started)
      } else if (isTRUE(run_dcov)) {
        logical_parts[[setup_ordinal]] <- NULL
      }
      timing_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        target_count = target_count,
        logical_test_count = nrow(setup_logical),
        optimizer_ms = optimizer_ms,
        validation_ms = validation_ms,
        dcov_ms = dcov_ms,
        dcov_component_request_count =
          as.integer(dcov_diagnostics$component_request_count),
        dcov_component_cache_hit_count =
          as.integer(dcov_diagnostics$component_cache_hit_count),
        dcov_component_cache_miss_count =
          as.integer(dcov_diagnostics$component_cache_miss_count),
        dcov_spectra_decomposition_count =
          as.integer(dcov_diagnostics$lowrank_spectra_count),
        candidate_legacy_mgcv_target_calls = 0L,
        validation_legacy_mgcv_fixed_sp_calls = target_count,
        cpp_target_count = target_count,
        fallback_count = 0L,
        stringsAsFactors = FALSE
      )
      if (isTRUE(progress) && setup_ordinal %% 25L == 0L) {
        cat(
          "Phase 5 C++ shadow setups:", setup_ordinal, "/",
          length(all_setup_keys), "\n"
        )
        flush.console()
      }
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
  transcript_keys <- names(transcripts)
  if (is.null(transcript_keys)) transcript_keys <- character()
  transcript_keys <- sort(transcript_keys, method = "radix")
  transcripts <- transcripts[transcript_keys]
  summary <- fastkpc_full_cuda_phase5_shadow_summary(
    targets, logical_rows, timings, length(all_setup_keys), run_dcov,
    proc.time()[["elapsed"]] - started, length(transcripts)
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cpp-shadow-partition-evidence-v1",
    partition = list(
      schema_version =
        "full-cuda-ci-multi-penalty-cpp-shadow-partition-v1",
      partition_id = partition$partition_id,
      partition_count = partition$partition_count,
      assignment_strategy = partition$assignment_strategy,
      shard_ids = partition$shard_ids,
      setup_keys = all_setup_keys
    ),
    summary = summary,
    execution_identity =
      fastkpc_full_cuda_phase5_execution_identity(catalog),
    targets = targets,
    logical_rows = logical_rows,
    timings = timings,
    transcripts = transcripts
  )
}

fastkpc_full_cuda_phase5_mixed_graph_replay <- function(
    catalog, phase5_logical_rows,
    phase4_artifact_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1"
    ), phase4_logical_rows = NULL) {
  required <- c(
    "fastkpc_full_cuda_replay_logical_ci",
    "fastkpc_full_cuda_compare_candidate_skeleton",
    ".fastkpc_full_cuda_phase3_shadow_phase0_authority",
    ".fastkpc_full_cuda_phase3_load_shadow_phase0_oracle"
  )
  fastkpc_full_cuda_phase5_artifact_require(
    all(vapply(required, exists, logical(1L), mode = "function",
               inherits = TRUE)),
    "Phase 5 mixed graph replay dependencies are unavailable"
  )
  if (is.null(phase4_logical_rows) && exists(
        "fastkpc_full_cuda_phase4_validate_artifact", mode = "function",
        inherits = TRUE
      )) {
    phase4_validation <- fastkpc_full_cuda_phase4_validate_artifact(
      phase4_artifact_dir, expected_kind = "full_shadow"
    )
    fastkpc_full_cuda_phase5_artifact_require(
      isTRUE(phase4_validation$summary$pass),
      "Phase 5 inherited Phase 4 artifact validation failed"
    )
  }
  phase4_path <- file.path(phase4_artifact_dir, "logical_ci_results.rds")
  if (is.null(phase4_logical_rows)) {
    fastkpc_full_cuda_phase5_artifact_require(
      file.exists(phase4_path),
      "Phase 5 inherited Phase 4 logical results are missing"
    )
    phase4_rows <- readRDS(phase4_path)
    phase4_identity <- fastkpc_full_cuda_census_file_hash(phase4_path)
  } else {
    fastkpc_full_cuda_phase5_artifact_require(
      is.data.frame(phase4_logical_rows),
      "Phase 5 supplied Phase 4 logical results are malformed"
    )
    phase4_rows <- phase4_logical_rows
    phase4_identity <- fastkpc_full_cuda_census_frame_hash(phase4_rows)
  }
  phase4_rows <- phase4_rows[order(
    phase4_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  phase5_rows <- phase5_logical_rows[order(
    phase5_logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  expected_phase4 <- plan$conditional_tests[
    plan$conditional_tests$level %in% c(1L, 2L), , drop = FALSE
  ]
  expected_phase5 <- plan$conditional_tests[
    plan$conditional_tests$level > 2L, , drop = FALSE
  ]
  expected_phase4 <- expected_phase4[order(
    expected_phase4$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  expected_phase5 <- expected_phase5[order(
    expected_phase5$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  lineage <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
    "reference_p_value", "alpha", "reference_decision"
  )
  clean <- is.data.frame(phase4_rows) && is.data.frame(phase5_rows) &&
    nrow(phase4_rows) == 177952L && nrow(expected_phase4) == 177952L &&
    nrow(phase5_rows) == 60324L && nrow(expected_phase5) == 60324L &&
    !anyDuplicated(phase4_rows$logical_sequence_id) &&
    !anyDuplicated(phase5_rows$logical_sequence_id) &&
    all(vapply(lineage, function(field) {
      identical(phase4_rows[[field]], expected_phase4[[field]]) &&
        identical(phase5_rows[[field]], expected_phase5[[field]])
    }, logical(1L))) &&
    !any(phase4_rows$decision_flip) && !any(phase5_rows$decision_flip) &&
    !any(phase4_rows$backend_error) && !any(phase5_rows$backend_error) &&
    !any(phase4_rows$spectra_fallback) &&
    !any(phase5_rows$spectra_fallback)
  fastkpc_full_cuda_phase5_artifact_require(
    clean, "Phase 5 mixed graph logical authority is malformed"
  )

  authority <- catalog$inputs$logical_tests
  candidate_p_value <- as.numeric(authority$reference_p_value)
  phase4_match <- match(
    phase4_rows$logical_sequence_id, authority$logical_sequence_id
  )
  phase5_match <- match(
    phase5_rows$logical_sequence_id, authority$logical_sequence_id
  )
  direct <- authority$level == 0L
  coverage <- sort(c(
    authority$logical_sequence_id[direct],
    phase4_rows$logical_sequence_id,
    phase5_rows$logical_sequence_id
  ))
  fastkpc_full_cuda_phase5_artifact_require(
    nrow(authority) == 240489L && sum(direct) == 2213L &&
      !anyNA(phase4_match) && !anyNA(phase5_match) &&
      identical(coverage, seq_len(nrow(authority))) &&
      all(is.finite(candidate_p_value)),
    "Phase 5 mixed graph canonical coverage is malformed"
  )
  candidate_p_value[phase4_match] <- phase4_rows$candidate_p_value
  candidate_p_value[phase5_match] <- phase5_rows$candidate_p_value
  replay <- fastkpc_full_cuda_replay_logical_ci(
    logical_tests = authority,
    candidate_p_value = candidate_p_value,
    labels = colnames(catalog$inputs$data),
    expected_logical_contract =
      fastkpc_full_cuda_shadow_logical_contract(authority)
  )
  phase0_authority <- .fastkpc_full_cuda_phase3_shadow_phase0_authority(
    catalog, catalog$phase0_dir
  )
  phase0 <- .fastkpc_full_cuda_phase3_load_shadow_phase0_oracle(
    catalog, phase0_authority
  )
  comparison <- fastkpc_full_cuda_compare_candidate_skeleton(
    phase0, replay$skeleton
  )
  graph <- comparison$summary
  pass <- isTRUE(graph$pass) && graph$edge_count_candidate == 110L &&
    graph$edge_count_reference == 110L && graph$SHD == 0L &&
    isTRUE(graph$adjacency_identical) && isTRUE(graph$sepsets_identical) &&
    isTRUE(graph$n_edgetests_identical) &&
    isTRUE(graph$deletions_identical)
  fastkpc_full_cuda_phase5_artifact_require(
    pass, "Phase 5 mixed graph gate failed"
  )
  list(
    schema_version = "full-cuda-ci-multi-penalty-cpp-mixed-graph-v1",
    summary = list(
      direct_legacy_logical_test_count = 2213L,
      phase4_cuda_logical_test_count = 177952L,
      phase5_cpp_logical_test_count = 60324L,
      explicit_legacy_fallback_count = 0L,
      unknown_fallback_count = 0L,
      edge_count_reference = as.integer(graph$edge_count_reference),
      edge_count_candidate = as.integer(graph$edge_count_candidate),
      SHD = as.integer(graph$SHD),
      adjacency_identical = isTRUE(graph$adjacency_identical),
      sepsets_identical = isTRUE(graph$sepsets_identical),
      n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
      deletions_identical = isTRUE(graph$deletions_identical),
      pass = pass
    ),
    inherited_phase4_logical_results_sha256 =
      phase4_identity,
    replay = replay,
    comparison = comparison
  )
}

fastkpc_full_cuda_phase5_merge_partitions <- function(
    catalog, evidence_paths,
    phase4_artifact_dir = file.path(
      "fastkpc", "artifacts", "full_cuda_ci",
      "single_penalty_cuda_gcv_full_shadow_v1"
    )) {
  evidence_paths <- as.character(evidence_paths)
  fastkpc_full_cuda_phase5_artifact_require(
    length(evidence_paths) > 0L && !anyNA(evidence_paths) &&
      !anyDuplicated(evidence_paths) &&
      all(file.exists(evidence_paths) & !dir.exists(evidence_paths)),
    "Phase 5 partition paths are malformed"
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
  clean <- length(partition_count) == 1L &&
    partition_count == length(parts) &&
    identical(sort(partition_ids), seq.int(0L, partition_count - 1L)) &&
    all(vapply(parts, function(value) {
      is.list(value) && identical(
        value$schema_version,
        "full-cuda-ci-multi-penalty-cpp-shadow-partition-evidence-v1"
      ) && identical(
        value$partition$schema_version,
        "full-cuda-ci-multi-penalty-cpp-shadow-partition-v1"
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
        length(value$partition$shard_ids) > 0L &&
        identical(
          value$partition$shard_ids,
          sort(unique(value$partition$shard_ids))
      ) && identical(
        value$execution_identity$schema_version,
        "full-cuda-ci-phase5-execution-identity-v1"
      ) && isTRUE(value$summary$pass) &&
        isTRUE(value$summary$numerical_gate) &&
        isTRUE(value$summary$optimizer_gate) &&
        isTRUE(value$summary$downstream_decision_gate) &&
        isTRUE(value$summary$backend_gate)
    }, logical(1L)))
  fastkpc_full_cuda_phase5_artifact_require(
    clean, "Phase 5 partition evidence is incomplete"
  )
  execution_json <- vapply(parts, function(value) {
    fastkpc_full_cuda_phase35_canonical_json(value$execution_identity)
  }, character(1L))
  current_identity <- fastkpc_full_cuda_phase5_execution_identity(catalog)
  fastkpc_full_cuda_phase5_artifact_require(
    length(unique(execution_json)) == 1L && identical(
      execution_json[[1L]],
      fastkpc_full_cuda_phase35_canonical_json(current_identity)
    ),
    "Phase 5 partition producer execution identities differ"
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
  fastkpc_full_cuda_phase5_artifact_require(
    assignment_clean,
    "Phase 5 authenticated-shard partition assignment drifted"
  )
  observed_setup_keys <- unlist(lapply(
    parts, function(value) value$partition$setup_keys
  ), use.names = FALSE)
  fastkpc_full_cuda_phase5_artifact_require(
    length(observed_setup_keys) == 7460L &&
      !anyDuplicated(observed_setup_keys) && identical(
        sort(observed_setup_keys, method = "radix"), expected_setup_keys
      ),
    "Phase 5 partitions do not exactly cover the setup corpus"
  )

  targets <- do.call(rbind, lapply(parts, `[[`, "targets"))
  timings <- do.call(rbind, lapply(parts, `[[`, "timings"))
  logical_rows <- do.call(rbind, lapply(parts, `[[`, "logical_rows"))
  transcripts <- do.call(c, lapply(parts, `[[`, "transcripts"))
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
    !anyDuplicated(targets$residual_key_sha256) &&
    identical(targets$residual_key_sha256, expected_target_keys) &&
    identical(
      targets$prepared_s_key_sha256,
      scope$target_rows$prepared_s_key_sha256
    ) && nrow(timings) == 7460L && !anyNA(setup_match) &&
    !anyDuplicated(timings$prepared_s_key_sha256) &&
    identical(timings$prepared_s_key_sha256, expected_setup_keys) &&
    nrow(logical_rows) == 60324L &&
    !anyDuplicated(logical_rows$logical_sequence_id) &&
    identical(
      logical_rows$logical_sequence_id,
      expected_logical$logical_sequence_id
    ) && identical(
      logical_rows$residual_key_x, expected_logical$residual_key_x
    ) && identical(
      logical_rows$residual_key_y, expected_logical$residual_key_y
    ) && !anyDuplicated(names(transcripts))
  fastkpc_full_cuda_phase5_artifact_require(
    merge_clean, "Phase 5 merged partition lineage is incomplete"
  )
  summary <- fastkpc_full_cuda_phase5_shadow_summary(
    targets, logical_rows, timings, 7460L, TRUE,
    max(vapply(parts, function(value) {
      as.numeric(value$summary$elapsed_seconds)
    }, numeric(1L))), length(transcripts)
  )
  fastkpc_full_cuda_phase5_artifact_require(
    isTRUE(summary$pass), "Phase 5 merged numerical gate failed"
  )
  mixed_graph <- fastkpc_full_cuda_phase5_mixed_graph_replay(
    catalog, logical_rows, phase4_artifact_dir = phase4_artifact_dir
  )
  list(
    schema_version =
      "full-cuda-ci-multi-penalty-cpp-full-shadow-merged-v1",
    summary = summary,
    execution_identity = current_identity,
    mixed_graph = mixed_graph,
    partition_count = partition_count,
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
    timings = timings,
    transcripts = transcripts
  )
}
