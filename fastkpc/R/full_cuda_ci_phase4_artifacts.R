fastkpc_full_cuda_phase4_artifact_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase4_single_penalty_scope <- function(catalog) {
  scope <- fastkpc_full_cuda_fixed_sp_scope(catalog, "full")
  selected_setup <- scope$setup_rows$penalty_count == 1L
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
  clean <- nrow(setup_rows) == 1174L && nrow(target_rows) == 44941L &&
    !anyNA(setup_rank) && !anyDuplicated(setup_keys) &&
    !anyDuplicated(target_rows$residual_key_sha256) &&
    identical(setup_keys, sort(setup_keys, method = "radix")) &&
    all(setup_rows$S_size <= 2L) && all(shard_id >= 0L) &&
    all(shard_id < shard_count)
  fastkpc_full_cuda_phase4_artifact_require(
    clean, "Phase 4 canonical single-penalty scope is malformed"
  )
  list(
    setup_rows = setup_rows,
    target_rows = target_rows,
    setup_rank = as.integer(setup_rank),
    shard_id = shard_id,
    shard_ids = sort(unique(shard_id))
  )
}

fastkpc_full_cuda_phase4_read_shard <- function(
    catalog, scope, shard_id, preparation = NULL,
    include_oracle_setups = TRUE) {
  shard_id <- as.integer(shard_id)
  fastkpc_full_cuda_phase4_artifact_require(
    is.logical(include_oracle_setups) &&
      length(include_oracle_setups) == 1L && !is.na(include_oracle_setups),
    "Phase 4 oracle-setup inclusion flag is malformed"
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
  fastkpc_full_cuda_phase4_artifact_require(
    !anyNA(setup_match), "Phase 4 shard is missing a selected setup"
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
  fastkpc_full_cuda_phase4_artifact_require(
    !anyNA(state_match), "Phase 4 shard is missing a selected target"
  )
  states <- payload$target_states[state_match, , drop = FALSE]
  fastkpc_full_cuda_phase4_artifact_require(
    identical(
      as.character(states$residual_key_sha256),
      as.character(target_rows$residual_key_sha256)
    ) && identical(
      as.character(states$prepared_s_key_sha256),
      as.character(target_rows$prepared_s_key_sha256)
    ),
    "Phase 4 selected target lineage is inconsistent"
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

fastkpc_full_cuda_phase4_batch_from_shard <- function(
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
  fastkpc_full_cuda_phase4_artifact_require(
    length(setup$penalty_blocks) == 1L && nrow(states) > 0L &&
      identical(
        as.character(states$residual_key_sha256),
        as.character(metadata$residual_key_sha256)
      ),
    "Phase 4 shard batch identity is malformed"
  )
  materialized <- lapply(seq_len(nrow(states)), function(index) {
    target <- fastkpc_full_cuda_materialize_target_state(
      states[index, , drop = FALSE], catalog$inputs$data,
      catalog$inputs$dataset_sha256
    )
    fastkpc_full_cuda_validate_materialized_target_for_prepared(
      setup, target
    )
  })
  Y <- do.call(cbind, lapply(materialized, `[[`, "y"))
  storage.mode(Y) <- "double"
  target_keys <- as.character(states$residual_key_sha256)
  colnames(Y) <- target_keys
  list(
    setup = setup,
    states = states,
    metadata = metadata,
    Y = Y,
    oracle_sp = vapply(materialized, function(value) {
      as.numeric(value$sp[[1L]])
    }, numeric(1L)),
    target_keys = target_keys,
    target_ids = as.integer(states$target),
    planned_route = as.character(metadata$planned_route)
  )
}

fastkpc_full_cuda_phase4_oracle_convergence <- function(states) {
  values <- lapply(states$convergence_fields, function(value) {
    mgcv <- value$mgcv.conv$value
    list(
      fully = isTRUE(mgcv$fully.converged),
      hessian_positive = isTRUE(mgcv$hess.pos.def),
      iteration = as.integer(mgcv$iter),
      score_calls = as.integer(mgcv$score.calls)
    )
  })
  data.frame(
    fully_converged = vapply(values, `[[`, logical(1L), "fully"),
    hessian_positive_definite = vapply(
      values, `[[`, logical(1L), "hessian_positive"
    ),
    iteration_count = vapply(values, `[[`, integer(1L), "iteration"),
    score_call_count = vapply(
      values, `[[`, integer(1L), "score_calls"
    ),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase4_reference_grid <- function(
    spectral, Y, log_sp_grid) {
  exact <- fastkpc_full_cuda_phase4_exact_reference_grid(
    spectral, Y, log_sp_grid
  )
  projected_rhs <- crossprod(spectral$X, Y)
  exact$spectral_projection <- spectral$rhs_transform %*% projected_rhs
  exact
}

fastkpc_full_cuda_phase4_max_error <- function(candidate, reference) {
  difference <- abs(candidate - reference)
  c(
    absolute = max(difference),
    relative = max(difference / pmax(abs(reference), 1e-300))
  )
}

fastkpc_full_cuda_phase4_transcript_keys <- function(catalog, scope) {
  available <- as.character(scope$target_rows$residual_key_sha256)
  scoped <- unique(c(
    as.character(catalog$scopes$iteration$target_rows$residual_key_sha256),
    as.character(catalog$scopes$qualification$target_rows$residual_key_sha256),
    as.character(catalog$inputs$risk_cases$residual_key_sha256)
  ))
  logical <- catalog$inputs$logical_tests
  near_alpha <- character()
  if (is.data.frame(logical) && all(c(
    "absolute_log_distance_from_alpha", "residual_key_x", "residual_key_y"
  ) %in% names(logical))) {
    threshold <- stats::quantile(
      logical$absolute_log_distance_from_alpha, 0.01,
      names = FALSE, na.rm = TRUE
    )
    selected <- logical$absolute_log_distance_from_alpha <= threshold
    near_alpha <- c(
      as.character(logical$residual_key_x[selected]),
      as.character(logical$residual_key_y[selected])
    )
  }
  sort(intersect(unique(c(scoped, near_alpha)), available), method = "radix")
}

fastkpc_full_cuda_phase4_scan_optimizer <- function(
    catalog, dense_grid_size = 161L, preserve_transcripts = TRUE,
    progress = interactive()) {
  dense_grid_size <- as.integer(dense_grid_size)
  fastkpc_full_cuda_phase4_artifact_require(
    dense_grid_size >= 33L,
    "Phase 4 dense grid must contain at least 33 points"
  )
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  required_transcript_keys <- if (isTRUE(preserve_transcripts)) {
    fastkpc_full_cuda_phase4_transcript_keys(catalog, scope)
  } else {
    character()
  }
  target_parts <- vector("list", nrow(scope$setup_rows))
  curve_parts <- vector("list", nrow(scope$setup_rows))
  timing_parts <- vector("list", nrow(scope$setup_rows))
  transcript_parts <- list()
  setup_ordinal <- 0L
  started <- proc.time()[["elapsed"]]
  preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
    inputs = catalog$inputs,
    shard_count = catalog$catalog_contract$shard_count,
    expected_source_commit = catalog$phase2_manifest$source_commit
  )
  for (shard_id in scope$shard_ids) {
    shard <- fastkpc_full_cuda_phase4_read_shard(
      catalog, scope, shard_id, preparation = preparation
    )
    for (setup_key in names(shard$setups)) {
      setup_ordinal <- setup_ordinal + 1L
      batch <- fastkpc_full_cuda_phase4_batch_from_shard(
        catalog, shard, setup_key
      )
      spectral <- fastkpc_full_cuda_phase4_spectral_prepare(batch$setup)
      spectral$X <- batch$setup$X
      log_grid <- sort(unique(c(
        seq(-40, 40, length.out = dense_grid_size),
        log(batch$oracle_sp)
      )))
      reference <- fastkpc_full_cuda_phase4_reference_grid(
        spectral, batch$Y, log_grid
      )
      host_begin <- proc.time()[["elapsed"]]
      cuda <- fastkpc_full_cuda_phase4_cuda_batch(
        batch$setup, batch$Y, target_ids = batch$target_ids,
        sp_grid = exp(log_grid), materialize_grid = TRUE,
        keep_transcript = FALSE
      )
      host_ms <- 1000 * (proc.time()[["elapsed"]] - host_begin)
      oracle_convergence <-
        fastkpc_full_cuda_phase4_oracle_convergence(batch$states)
      target_count <- ncol(batch$Y)
      oracle_score <- as.numeric(batch$states$GCV_Cp_score)
      oracle_edf <- as.numeric(batch$states$EDF)
      at_oracle <- lapply(seq_len(target_count), function(target) {
        fastkpc_full_cuda_phase4_objective(
          log(batch$oracle_sp[[target]]), spectral$eigenvalues,
          reference$spectral_projection[, target]^2,
          sum(batch$Y[, target]^2), spectral$n
        )
      })
      target_rows <- data.frame(
        prepared_s_key_sha256 = setup_key,
        residual_key_sha256 = batch$target_keys,
        target = batch$target_ids,
        oracle_sp = batch$oracle_sp,
        candidate_sp = cuda$targets$sp,
        oracle_score = oracle_score,
        candidate_score = cuda$targets$score,
        oracle_edf = oracle_edf,
        candidate_edf = cuda$targets$edf,
        oracle_point_score = vapply(
          at_oracle, `[[`, numeric(1L), "score"
        ),
        oracle_point_edf = vapply(at_oracle, `[[`, numeric(1L), "edf"),
        oracle_iteration_count = oracle_convergence$iteration_count,
        candidate_iteration_count = cuda$targets$iteration_count,
        oracle_score_call_count = oracle_convergence$score_call_count,
        candidate_score_call_count = cuda$targets$score_call_count,
        oracle_fully_converged = oracle_convergence$fully_converged,
        candidate_fully_converged = cuda$targets$fully_converged,
        oracle_hessian_positive_definite =
          oracle_convergence$hessian_positive_definite,
        candidate_hessian_positive_definite =
          cuda$targets$hessian_positive_definite,
        termination_reason = cuda$targets$termination_reason,
        boundary_status = cuda$targets$boundary_status,
        stringsAsFactors = FALSE
      )
      target_rows$log_sp_error <-
        log(target_rows$candidate_sp) - log(target_rows$oracle_sp)
      target_rows$score_error <-
        target_rows$candidate_score - target_rows$oracle_score
      target_rows$edf_error <-
        target_rows$candidate_edf - target_rows$oracle_edf
      target_rows$oracle_point_score_error <-
        target_rows$oracle_point_score - target_rows$oracle_score
      target_rows$oracle_point_edf_error <-
        target_rows$oracle_point_edf - target_rows$oracle_edf
      target_parts[[setup_ordinal]] <- target_rows

      rss_error <- fastkpc_full_cuda_phase4_max_error(
        cuda$grid$rss, reference$rss
      )
      edf_error <- fastkpc_full_cuda_phase4_max_error(
        cuda$grid$edf, reference$edf
      )
      score_error <- fastkpc_full_cuda_phase4_max_error(
        cuda$grid$score, reference$score
      )
      curve_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        target_count = target_count,
        grid_count = length(log_grid),
        rss_max_absolute = rss_error[["absolute"]],
        rss_max_relative = rss_error[["relative"]],
        edf_max_absolute = edf_error[["absolute"]],
        score_max_absolute = score_error[["absolute"]],
        score_max_relative = score_error[["relative"]],
        stringsAsFactors = FALSE
      )
      diagnostics <- cuda$diagnostics
      timing_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        host_ms = as.numeric(host_ms),
        cuda_ms = as.numeric(
          diagnostics$upload_cuda_ms + diagnostics$projection_cuda_ms +
            diagnostics$grid_cuda_ms + diagnostics$optimizer_cuda_ms +
            diagnostics$d2h_cuda_ms
        ),
        fallback_count = as.integer(diagnostics$fallback_count),
        legacy_mgcv_target_calls =
          as.integer(diagnostics$legacy_mgcv_target_calls),
        spectral_optimizer_target_count =
          as.integer(diagnostics$spectral_optimizer_target_count),
        spectral_only_target_count =
          as.integer(diagnostics$spectral_only_target_count),
        exact_replay_target_count =
          as.integer(diagnostics$exact_replay_target_count),
        exact_replay_endpoint_risk_count =
          as.integer(diagnostics$exact_replay_endpoint_risk_count),
        exact_replay_convergence_risk_count =
          as.integer(diagnostics$exact_replay_convergence_risk_count),
        exact_replay_boundary_risk_count =
          as.integer(diagnostics$exact_replay_boundary_risk_count),
        exact_replay_numerical_risk_count =
          as.integer(diagnostics$exact_replay_numerical_risk_count),
        optimizer_target_coverage_complete =
          isTRUE(diagnostics$optimizer_target_coverage_complete),
        sp_selection_backend_executed =
          as.character(diagnostics$sp_selection_backend_executed),
        gcv_score_backend_executed =
          as.character(diagnostics$gcv_score_backend_executed),
        optimizer_backend_executed =
          as.character(diagnostics$optimizer_backend_executed),
        exact_replay_backend_executed =
          as.character(diagnostics$exact_replay_backend_executed),
        stringsAsFactors = FALSE
      )

      exceptional <- target_rows$residual_key_sha256[
        !target_rows$oracle_fully_converged |
          !target_rows$candidate_fully_converged |
          target_rows$boundary_status != "finite_refinement"
      ]
      preserve <- intersect(
        batch$target_keys,
        unique(c(required_transcript_keys, exceptional))
      )
      if (isTRUE(preserve_transcripts) && length(preserve) > 0L) {
        transcript_cuda <- fastkpc_full_cuda_phase4_cuda_batch(
          batch$setup, batch$Y, target_ids = batch$target_ids,
          keep_transcript = TRUE
        )
        selected_transcripts <- match(preserve, batch$target_keys)
        names(selected_transcripts) <- preserve
        for (key in preserve) {
          index <- selected_transcripts[[key]]
          transcript <- transcript_cuda$transcripts[[index]]
          transcript$prepared_s_key_sha256 <- setup_key
          transcript$residual_key_sha256 <- key
          transcript$target <- batch$target_ids[[index]]
          transcript_parts[[key]] <- transcript
        }
      }
      if (isTRUE(progress) && setup_ordinal %% 100L == 0L) {
        cat("Phase 4 optimizer setups:", setup_ordinal, "/ 1174\n")
        flush.console()
      }
    }
    rm(shard)
    gc(FALSE)
  }
  targets <- do.call(rbind, target_parts)
  curves <- do.call(rbind, curve_parts)
  timings <- do.call(rbind, timing_parts)
  rownames(targets) <- rownames(curves) <- rownames(timings) <- NULL
  transcript_keys <- sort(names(transcript_parts), method = "radix")
  transcripts <- transcript_parts[transcript_keys]
  required_exceptional <- sort(unique(targets$residual_key_sha256[
    !targets$oracle_fully_converged |
      !targets$candidate_fully_converged |
      targets$boundary_status != "finite_refinement"
  ]), method = "radix")
  required_all <- sort(unique(c(
    required_transcript_keys, required_exceptional
  )), method = "radix")
  numerical <- fastkpc_full_cuda_phase35_load_contract(
    "numerical_contract_v1"
  )
  tolerance <- numerical$payload$tolerances
  numeric_tolerance <- function(value) as.numeric(value)
  summary <- list(
    schema_version = "full-cuda-ci-single-penalty-gcv-oracle-summary-v1",
    setup_count = nrow(curves),
    target_count = nrow(targets),
    dense_curve_cell_count = sum(curves$target_count * curves$grid_count),
    max_rss_absolute_error = max(curves$rss_max_absolute),
    max_rss_relative_error = max(curves$rss_max_relative),
    max_edf_absolute_error = max(curves$edf_max_absolute),
    max_score_absolute_error = max(curves$score_max_absolute),
    max_score_relative_error = max(curves$score_max_relative),
    max_oracle_point_score_error =
      max(abs(targets$oracle_point_score_error)),
    max_oracle_point_edf_error =
      max(abs(targets$oracle_point_edf_error)),
    max_selected_log_sp_error = max(abs(targets$log_sp_error)),
    selected_log_sp_diagnostic_exceedance_count = sum(
      abs(targets$log_sp_error) >
        numeric_tolerance(tolerance$selected_log_sp$absolute_diagnostic)
    ),
    max_selected_score_error = max(abs(targets$score_error)),
    optimizer_iteration_mismatch_count = sum(
      targets$oracle_iteration_count != targets$candidate_iteration_count
    ),
    optimizer_hessian_state_mismatch_count = sum(
      targets$oracle_hessian_positive_definite !=
        targets$candidate_hessian_positive_definite
    ),
    oracle_non_fully_converged_count = sum(
      !targets$oracle_fully_converged
    ),
    candidate_non_fully_converged_count = sum(
      !targets$candidate_fully_converged
    ),
    boundary_target_count = sum(
      targets$boundary_status != "finite_refinement"
    ),
    flat_objective_target_count = sum(
      targets$termination_reason == "flat_objective"
    ),
    fallback_count = sum(timings$fallback_count),
    legacy_mgcv_target_calls = sum(timings$legacy_mgcv_target_calls),
    spectral_optimizer_target_count =
      sum(timings$spectral_optimizer_target_count),
    spectral_only_target_count = sum(timings$spectral_only_target_count),
    exact_replay_target_count = sum(timings$exact_replay_target_count),
    exact_replay_endpoint_risk_count =
      sum(timings$exact_replay_endpoint_risk_count),
    exact_replay_convergence_risk_count =
      sum(timings$exact_replay_convergence_risk_count),
    exact_replay_boundary_risk_count =
      sum(timings$exact_replay_boundary_risk_count),
    exact_replay_numerical_risk_count =
      sum(timings$exact_replay_numerical_risk_count),
    optimizer_backend_executed =
      unique(timings$optimizer_backend_executed),
    exact_replay_backend_executed =
      unique(timings$exact_replay_backend_executed),
    transcript_required_count = length(required_all),
    transcript_preserved_count = length(transcripts),
    transcript_missing_count = length(setdiff(required_all, names(transcripts))),
    summed_cuda_ms = sum(timings$cuda_ms),
    summed_host_ms = sum(timings$host_ms),
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    objective_curve_gate =
      max(curves$rss_max_absolute) <=
        numeric_tolerance(tolerance$rss$absolute) &&
      max(curves$rss_max_relative) <=
        numeric_tolerance(tolerance$rss$relative) &&
      max(curves$edf_max_absolute) <=
        numeric_tolerance(tolerance$edf$absolute) &&
      max(curves$score_max_absolute) <=
        numeric_tolerance(tolerance$gcv_cp_score$absolute) &&
      max(curves$score_max_relative) <=
        numeric_tolerance(tolerance$gcv_cp_score$relative),
    optimizer_objective_gate =
      max(abs(targets$score_error)) <=
        numeric_tolerance(tolerance$gcv_cp_score$absolute),
    optimizer_coverage_gate =
      all(timings$optimizer_target_coverage_complete) &&
      sum(timings$spectral_optimizer_target_count) == nrow(targets) &&
      sum(timings$spectral_only_target_count) +
        sum(timings$exact_replay_target_count) == nrow(targets),
    backend_gate = sum(timings$fallback_count) == 0L &&
      sum(timings$legacy_mgcv_target_calls) == 0L &&
      all(timings$sp_selection_backend_executed == "cuda") &&
      all(timings$gcv_score_backend_executed == "cuda") &&
      all(timings$optimizer_backend_executed ==
            "cuda-spectral-risk-gated-exact-replay") &&
      all(timings$exact_replay_backend_executed ==
            "cuda-dpstf2-lapack-3.12-dgesdd") &&
      all(timings$optimizer_target_coverage_complete) &&
      sum(timings$spectral_only_target_count) +
        sum(timings$exact_replay_target_count) == nrow(targets),
    transcript_gate = length(setdiff(required_all, names(transcripts))) == 0L,
    numerical_contract_sha256 = numerical$sha256
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-gcv-oracle-evidence-v1",
    summary = summary,
    targets = targets,
    curves = curves,
    timings = timings,
    transcripts = transcripts,
    required_transcript_keys = required_all
  )
}

fastkpc_full_cuda_phase4_reference_fit <- function(
    prepared_setup, Y, sp) {
  Y <- as.matrix(Y)
  storage.mode(Y) <- "double"
  sp <- as.numeric(sp)
  target_count <- ncol(Y)
  fastkpc_full_cuda_phase4_artifact_require(
    nrow(Y) == nrow(prepared_setup$X) && length(sp) == target_count &&
      target_count > 0L && all(is.finite(Y)) &&
      all(is.finite(sp) & sp > 0),
    "Phase 4 fixed-sp reference inputs are malformed"
  )
  penalty_count <- length(prepared_setup$penalty_blocks)
  fastkpc_full_cuda_phase4_artifact_require(
    penalty_count == 1L,
    "Phase 4 fixed-sp reference requires one penalty"
  )
  coefficients <- vapply(seq_len(target_count), function(target) {
    minimal <- list(
      G = list(
        L = matrix(numeric(), nrow = penalty_count, ncol = 0L),
        lsp0 = log(sp[[target]])
      ),
      X = prepared_setup$X,
      y = as.numeric(Y[, target]),
      S = prepared_setup$penalty_blocks,
      off = prepared_setup$penalty_offsets,
      rank = prepared_setup$mgcv_penalty_rank_metadata,
      H = prepared_setup$H,
      C = prepared_setup$constraint,
      w = prepared_setup$weights,
      sp = sp[[target]]
    )
    as.numeric(fastkpc_mgcv_magic_kernel_fixed_sp_coefficients(
      minimal, sp = sp[[target]]
    ))
  }, numeric(ncol(prepared_setup$X)))
  fitted <- prepared_setup$X %*% coefficients
  residuals <- Y - fitted
  fastkpc_full_cuda_phase4_artifact_require(
    all(is.finite(coefficients)) && all(is.finite(fitted)) &&
      all(is.finite(residuals)),
    "Phase 4 fixed-sp reference produced non-finite output"
  )
  list(
    coefficients = coefficients,
    fitted = fitted,
    residuals = residuals,
    rss = colSums(residuals * residuals)
  )
}

fastkpc_full_cuda_phase4_column_errors <- function(candidate, reference) {
  candidate <- as.matrix(candidate)
  reference <- as.matrix(reference)
  fastkpc_full_cuda_phase4_artifact_require(
    identical(dim(candidate), dim(reference)) &&
      all(is.finite(candidate)) && all(is.finite(reference)),
    "Phase 4 vector comparison inputs are malformed"
  )
  difference <- candidate - reference
  data.frame(
    max_absolute = apply(abs(difference), 2L, max),
    relative_l2 = sqrt(colSums(difference * difference)) /
      pmax(sqrt(colSums(reference * reference)), 1e-300),
    stringsAsFactors = FALSE
  )
}

fastkpc_full_cuda_phase4_execute_integrated_setup <- function(
    runtime, batch) {
  dto <- if (identical(
    batch$setup$schema_version, "full-cuda-ci-native-runtime-setup-v1"
  )) {
    fastkpc_full_cuda_phase7_fixed_sp_dto(batch$setup)
  } else {
    fastkpc_full_cuda_fixed_sp_native_dto(batch$setup)
  }
  native <- list(
    Y = batch$Y,
    planned_route = batch$planned_route,
    target_keys = batch$target_keys,
    target_count = ncol(batch$Y)
  )
  handle <- NULL
  token <- NULL
  token_released <- FALSE
  token_freed <- FALSE
  handle_freed <- FALSE
  on.exit({
    if (!is.null(token) && !token_released) {
      try(fixed_sp_cuda_residual_release(token), silent = TRUE)
    }
    if (!is.null(token) && !token_freed) {
      try(fixed_sp_cuda_residual_free(token), silent = TRUE)
    }
    if (!is.null(handle) && !handle_freed) {
      try(fixed_sp_cuda_prepared_free(handle), silent = TRUE)
    }
  }, add = TRUE)
  handle <- fixed_sp_cuda_prepared_create(runtime, dto)
  started <- proc.time()[["elapsed"]]
  fit <- fastkpc_full_cuda_phase4_select_and_solve_cuda(
    prepared_handle = handle,
    prepared_setup = batch$setup,
    Y = native$Y,
    planned_route = native$planned_route,
    target_keys = native$target_keys,
    target_ids = batch$target_ids,
    outputs = c("fitted", "residuals", "rss")
  )
  integrated_host_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  token <- fit$residual_token
  info_before_shadow <- fixed_sp_cuda_residual_info(token)
  shadow_started <- proc.time()[["elapsed"]]
  shadow <- fixed_sp_cuda_materialize_shadow(
    token, outputs = c("fitted", "residuals", "rss")
  )
  colnames(shadow$fitted) <- batch$target_keys
  colnames(shadow$residuals) <- batch$target_keys
  names(shadow$rss) <- batch$target_keys
  shadow_ms <- 1000 * (proc.time()[["elapsed"]] - shadow_started)
  info_after_shadow <- fixed_sp_cuda_residual_info(token)
  fixed_sp_cuda_residual_release(token)
  token_released <- TRUE
  fixed_sp_cuda_residual_free(token)
  token_freed <- TRUE
  token <- NULL
  fixed_sp_cuda_prepared_free(handle)
  handle_freed <- TRUE
  handle <- NULL
  list(
    dto = dto,
    targets = fit$targets,
    diagnostics = fit$diagnostics,
    info_before_shadow = info_before_shadow,
    info_after_shadow = info_after_shadow,
    shadow = shadow,
    integrated_host_ms = integrated_host_ms,
    shadow_ms = shadow_ms
  )
}

fastkpc_full_cuda_phase4_shadow_partition <- function(
    setup_keys, partition_id = NULL, partition_count = NULL) {
  setup_keys <- as.character(setup_keys)
  partitioned <- !is.null(partition_id) || !is.null(partition_count)
  if (!partitioned) {
    return(list(
      partition_id = 0L, partition_count = 1L, setup_keys = setup_keys
    ))
  }
  partition_id <- as.integer(partition_id)
  partition_count <- as.integer(partition_count)
  fastkpc_full_cuda_phase4_artifact_require(
    length(partition_id) == 1L && !is.na(partition_id) &&
      length(partition_count) == 1L && !is.na(partition_count) &&
      partition_count >= 1L && partition_count <= length(setup_keys) &&
      partition_id >= 0L && partition_id < partition_count,
    "Phase 4 shadow partition is malformed"
  )
  selected <- (seq_along(setup_keys) - 1L) %% partition_count == partition_id
  list(
    partition_id = partition_id,
    partition_count = partition_count,
    setup_keys = setup_keys[selected]
  )
}

fastkpc_full_cuda_phase4_shadow_summary <- function(
    targets, logical_rows, timings, setup_count, run_dcov,
    elapsed_seconds) {
  numerical <- fastkpc_full_cuda_phase35_load_contract(
    "numerical_contract_v1"
  )
  tolerance <- numerical$payload$tolerances
  number <- function(value) as.numeric(value)
  same_sp_gate <-
    max(targets$fitted_same_sp_max_absolute) <=
      number(tolerance$fitted$max_absolute) &&
    max(targets$fitted_same_sp_relative_l2) <=
      number(tolerance$fitted$relative_l2) &&
    max(targets$residual_same_sp_max_absolute) <=
      number(tolerance$residual$max_absolute) &&
    max(targets$residual_same_sp_relative_l2) <=
      number(tolerance$residual$relative_l2)
  oracle_residual_gate <-
    max(targets$fitted_oracle_max_absolute) <=
      number(tolerance$fitted$max_absolute) &&
    max(targets$fitted_oracle_relative_l2) <=
      number(tolerance$fitted$relative_l2) &&
    max(targets$residual_oracle_max_absolute) <=
      number(tolerance$residual$max_absolute) &&
    max(targets$residual_oracle_relative_l2) <=
      number(tolerance$residual$relative_l2)
  downstream_gate <- !isTRUE(run_dcov) || (
    sum(logical_rows$decision_flip) == 0L &&
      sum(logical_rows$backend_error) == 0L &&
      sum(logical_rows$spectra_fallback) == 0L
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-gcv-shadow-summary-v1",
    setup_count = as.integer(setup_count),
    target_count = nrow(targets),
    logical_test_count = nrow(logical_rows),
    max_fitted_same_sp_absolute =
      max(targets$fitted_same_sp_max_absolute),
    max_fitted_same_sp_relative_l2 =
      max(targets$fitted_same_sp_relative_l2),
    max_residual_same_sp_absolute =
      max(targets$residual_same_sp_max_absolute),
    max_residual_same_sp_relative_l2 =
      max(targets$residual_same_sp_relative_l2),
    max_fitted_oracle_absolute = max(targets$fitted_oracle_max_absolute),
    max_fitted_oracle_relative_l2 =
      max(targets$fitted_oracle_relative_l2),
    max_residual_oracle_absolute =
      max(targets$residual_oracle_max_absolute),
    max_residual_oracle_relative_l2 =
      max(targets$residual_oracle_relative_l2),
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
    downstream_legacy_dcov_backend_error_count = if (isTRUE(run_dcov)) {
      sum(logical_rows$backend_error)
    } else {
      NA_integer_
    },
    downstream_legacy_dcov_fallback_count = if (isTRUE(run_dcov)) {
      sum(logical_rows$spectra_fallback)
    } else {
      NA_integer_
    },
    solver_reroute_count = sum(targets$planned_route !=
                                 targets$executed_route),
    solver_failure_count = sum(!grepl("^OK_", targets$solver_status)),
    fallback_count = sum(timings$fallback_count),
    legacy_mgcv_target_calls = sum(timings$legacy_mgcv_target_calls),
    spectral_optimizer_target_count =
      sum(timings$spectral_optimizer_target_count),
    spectral_only_target_count = sum(timings$spectral_only_target_count),
    exact_replay_target_count = sum(timings$exact_replay_target_count),
    exact_replay_endpoint_risk_count =
      sum(timings$exact_replay_endpoint_risk_count),
    exact_replay_convergence_risk_count =
      sum(timings$exact_replay_convergence_risk_count),
    exact_replay_boundary_risk_count =
      sum(timings$exact_replay_boundary_risk_count),
    exact_replay_numerical_risk_count =
      sum(timings$exact_replay_numerical_risk_count),
    optimizer_backend_executed =
      unique(timings$optimizer_backend_executed),
    exact_replay_backend_executed =
      unique(timings$exact_replay_backend_executed),
    validation_mgcv_fixed_sp_call_count = 2L * nrow(targets),
    selected_sp_r_roundtrip_count = sum(
      timings$selected_sp_returned_to_r_before_solve
    ),
    implicit_residual_d2h_count =
      sum(timings$implicit_residual_d2h_count),
    summed_integrated_host_ms = sum(timings$integrated_host_ms),
    summed_cuda_gcv_score_ms = sum(timings$cuda_gcv_score_ms),
    summed_cuda_selected_sp_solve_ms =
      sum(timings$cuda_selected_sp_solve_ms),
    summed_shadow_ms = sum(timings$shadow_ms),
    summed_validation_ms = sum(timings$validation_ms),
    summed_dcov_ms = sum(timings$dcov_ms),
    elapsed_seconds = as.numeric(elapsed_seconds),
    same_sp_fixed_solver_gate = same_sp_gate,
    oracle_residual_gate = oracle_residual_gate,
    downstream_decision_gate = downstream_gate,
    optimizer_coverage_gate =
      all(timings$optimizer_target_coverage_complete) &&
      sum(timings$spectral_optimizer_target_count) == nrow(targets) &&
      sum(timings$spectral_only_target_count) +
        sum(timings$exact_replay_target_count) == nrow(targets),
    backend_gate = sum(timings$fallback_count) == 0L &&
      sum(timings$legacy_mgcv_target_calls) == 0L &&
      all(timings$sp_selection_backend_executed == "cuda") &&
      all(timings$gcv_score_backend_executed == "cuda") &&
      all(timings$optimizer_backend_executed ==
            "cuda-spectral-risk-gated-exact-replay") &&
      all(timings$exact_replay_backend_executed ==
            "cuda-dpstf2-lapack-3.12-dgesdd") &&
      all(timings$optimizer_target_coverage_complete) &&
      sum(timings$spectral_only_target_count) +
        sum(timings$exact_replay_target_count) == nrow(targets) &&
      sum(timings$selected_sp_returned_to_r_before_solve) == 0L &&
      sum(timings$implicit_residual_d2h_count) == 0L &&
      sum(!grepl("^OK_", targets$solver_status)) == 0L,
    numerical_contract_sha256 = numerical$sha256
  )
}

fastkpc_full_cuda_phase4_scan_full_shadow <- function(
    catalog, max_setups = NULL, run_dcov = TRUE,
    partition_id = NULL, partition_count = NULL,
    progress = interactive(), setup_builder = NULL) {
  fastkpc_full_cuda_phase4_artifact_require(
    is.null(setup_builder) || is.function(setup_builder),
    "Phase 4 setup builder must be NULL or a function"
  )
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  all_setup_keys <- as.character(
    scope$setup_rows$prepared_s_key_sha256
  )
  if (!is.null(max_setups)) {
    max_setups <- as.integer(max_setups)
    fastkpc_full_cuda_phase4_artifact_require(
      length(max_setups) == 1L && !is.na(max_setups) && max_setups > 0L,
      "Phase 4 max_setups must be positive"
    )
    all_setup_keys <- head(all_setup_keys, max_setups)
  }
  partition <- fastkpc_full_cuda_phase4_shadow_partition(
    all_setup_keys, partition_id, partition_count
  )
  all_setup_keys <- partition$setup_keys
  selected_setup <- scope$setup_rows$prepared_s_key_sha256 %in% all_setup_keys
  scope$setup_rows <- scope$setup_rows[selected_setup, , drop = FALSE]
  scope$setup_rank <- scope$setup_rank[selected_setup]
  scope$shard_id <- scope$shard_id[selected_setup]
  scope$shard_ids <- sort(unique(scope$shard_id))
  scope$target_rows <- scope$target_rows[
    scope$target_rows$prepared_s_key_sha256 %in% all_setup_keys,
    , drop = FALSE
  ]
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  logical_tests <- plan$conditional_tests[
    plan$conditional_tests$prepared_s_key_x %in% all_setup_keys,
    , drop = FALSE
  ]
  fastkpc_full_cuda_phase4_artifact_require(
    all(logical_tests$prepared_s_key_y %in% all_setup_keys) &&
      all(logical_tests$prepared_s_key_x ==
            logical_tests$prepared_s_key_y),
    "Phase 4 logical setup ownership is malformed"
  )

  capacity <- fastkpc_full_cuda_fixed_sp_contract()$canonical_capacities
  runtime <- fixed_sp_cuda_runtime_create(0L)
  runtime_freed <- FALSE
  on.exit({
    if (!runtime_freed) try(fixed_sp_cuda_runtime_free(runtime), silent = TRUE)
  }, add = TRUE)
  fixed_sp_cuda_runtime_reserve(
    runtime, capacity$n, capacity$null_dim, capacity$target_count,
    capacity$penalty_count, capacity$augmented_rows
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
  setup_ordinal <- 0L
  started <- proc.time()[["elapsed"]]
  for (shard_id in scope$shard_ids) {
    shard <- fastkpc_full_cuda_phase4_read_shard(
      catalog, scope, shard_id, preparation = preparation,
      include_oracle_setups = is.null(setup_builder)
    )
    for (setup_key in shard$setup_keys) {
      setup_ordinal <- setup_ordinal + 1L
      setup_started <- proc.time()[["elapsed"]]
      setup_override <- NULL
      if (is.function(setup_builder)) {
        setup_index <- match(
          setup_key, shard$setup_rows$prepared_s_key_sha256
        )
        state_index <- which(
          shard$target_states$prepared_s_key_sha256 == setup_key
        )
        fastkpc_full_cuda_phase4_artifact_require(
          !is.na(setup_index) && length(state_index) > 0L,
          "Phase 4 native setup lineage is incomplete"
        )
        setup_override <- setup_builder(
          catalog = catalog,
          setup_row = shard$setup_rows[setup_index, , drop = FALSE],
          target_states = shard$target_states[state_index, , drop = FALSE],
          setup_key = setup_key
        )
      }
      native_setup_ms <-
        1000 * (proc.time()[["elapsed"]] - setup_started)
      batch <- fastkpc_full_cuda_phase4_batch_from_shard(
        catalog, shard, setup_key, setup_override = setup_override
      )
      executed <- fastkpc_full_cuda_phase4_execute_integrated_setup(
        runtime, batch
      )
      candidate_sp <- as.numeric(executed$targets$sp)
      validation_started <- proc.time()[["elapsed"]]
      candidate_reference <- fastkpc_full_cuda_phase4_reference_fit(
        batch$setup, batch$Y, candidate_sp
      )
      oracle_reference <- fastkpc_full_cuda_phase4_reference_fit(
        batch$setup, batch$Y, batch$oracle_sp
      )
      validation_ms <-
        1000 * (proc.time()[["elapsed"]] - validation_started)
      candidate_fitted_error <- fastkpc_full_cuda_phase4_column_errors(
        executed$shadow$fitted, candidate_reference$fitted
      )
      candidate_residual_error <- fastkpc_full_cuda_phase4_column_errors(
        executed$shadow$residuals, candidate_reference$residuals
      )
      oracle_fitted_error <- fastkpc_full_cuda_phase4_column_errors(
        executed$shadow$fitted, oracle_reference$fitted
      )
      oracle_residual_error <- fastkpc_full_cuda_phase4_column_errors(
        executed$shadow$residuals, oracle_reference$residuals
      )
      info <- executed$info_after_shadow
      target_count <- length(batch$target_keys)
      target_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        residual_key_sha256 = batch$target_keys,
        target = batch$target_ids,
        oracle_sp = batch$oracle_sp,
        candidate_sp = candidate_sp,
        log_sp_error = log(candidate_sp) - log(batch$oracle_sp),
        score = as.numeric(executed$targets$score),
        edf = as.numeric(executed$targets$edf),
        iteration_count = as.integer(executed$targets$iteration_count),
        fully_converged = as.logical(executed$targets$fully_converged),
        termination_reason =
          as.character(executed$targets$termination_reason),
        planned_route = as.character(info$planned_route),
        executed_route = as.character(info$executed_route),
        reroute_reason = as.character(info$reroute_reason),
        solver_status = as.character(info$solver_status),
        fitted_same_sp_max_absolute =
          candidate_fitted_error$max_absolute,
        fitted_same_sp_relative_l2 = candidate_fitted_error$relative_l2,
        residual_same_sp_max_absolute =
          candidate_residual_error$max_absolute,
        residual_same_sp_relative_l2 =
          candidate_residual_error$relative_l2,
        fitted_oracle_max_absolute = oracle_fitted_error$max_absolute,
        fitted_oracle_relative_l2 = oracle_fitted_error$relative_l2,
        residual_oracle_max_absolute = oracle_residual_error$max_absolute,
        residual_oracle_relative_l2 = oracle_residual_error$relative_l2,
        stringsAsFactors = FALSE
      )
      setup_logical <- logical_tests[
        logical_tests$prepared_s_key_x == setup_key,
        , drop = FALSE
      ]
      dcov_ms <- 0
      if (isTRUE(run_dcov)) {
        dcov_started <- proc.time()[["elapsed"]]
        rows <- fastkpc_full_cuda_shadow_compute_setup_rows(
          logical_tests = setup_logical,
          setup_key = setup_key,
          shard_id = shard_id,
          target_keys = batch$target_keys,
          residuals = executed$shadow$residuals
        )
        route_rows <- data.frame(
          prepared_s_key_sha256 = rep.int(setup_key, target_count),
          residual_key_sha256 = batch$target_keys,
          planned_route = as.character(info$planned_route),
          executed_route = as.character(info$executed_route),
          reroute_reason = as.character(info$reroute_reason),
          solver_status = as.character(info$solver_status),
          stringsAsFactors = FALSE
        )
        logical_parts[[setup_ordinal]] <-
          fastkpc_full_cuda_shadow_attach_target_routes(
            rows, route_rows, setup_logical
          )
        dcov_ms <- 1000 * (proc.time()[["elapsed"]] - dcov_started)
      }
      diagnostics <- executed$diagnostics
      native_diagnostics <- batch$setup$native_setup_diagnostics
      native_count <- if (is.null(native_diagnostics)) 0L else
        as.integer(native_diagnostics$native_setup_count)
      native_unsupported <- if (is.null(native_diagnostics)) 0L else
        as.integer(native_diagnostics$unsupported_count)
      legacy_setup_count <- if (is.null(native_diagnostics)) 0L else
        as.integer(native_diagnostics$legacy_mgcv_setup_count)
      r_callback_count <- if (is.null(native_diagnostics)) 0L else
        as.integer(native_diagnostics$r_callback_count)
      timing_parts[[setup_ordinal]] <- data.frame(
        prepared_s_key_sha256 = setup_key,
        target_count = target_count,
        logical_test_count = nrow(setup_logical),
        native_setup_ms = native_setup_ms,
        native_setup_count = native_count,
        native_setup_unsupported_count = native_unsupported,
        legacy_mgcv_setup_count = legacy_setup_count,
        r_callback_count = r_callback_count,
        integrated_host_ms = executed$integrated_host_ms,
        cuda_gcv_score_ms = diagnostics$cuda_gcv_score_ms,
        cuda_selected_sp_solve_ms =
          diagnostics$cuda_selected_sp_solve_ms,
        shadow_ms = executed$shadow_ms,
        validation_ms = validation_ms,
        dcov_ms = dcov_ms,
        fallback_count = as.integer(diagnostics$fallback_count),
        legacy_mgcv_target_calls =
          as.integer(diagnostics$legacy_mgcv_target_calls),
        spectral_optimizer_target_count =
          as.integer(diagnostics$spectral_optimizer_target_count),
        spectral_only_target_count =
          as.integer(diagnostics$spectral_only_target_count),
        exact_replay_target_count =
          as.integer(diagnostics$exact_replay_target_count),
        exact_replay_endpoint_risk_count =
          as.integer(diagnostics$exact_replay_endpoint_risk_count),
        exact_replay_convergence_risk_count =
          as.integer(diagnostics$exact_replay_convergence_risk_count),
        exact_replay_boundary_risk_count =
          as.integer(diagnostics$exact_replay_boundary_risk_count),
        exact_replay_numerical_risk_count =
          as.integer(diagnostics$exact_replay_numerical_risk_count),
        optimizer_target_coverage_complete =
          isTRUE(diagnostics$optimizer_target_coverage_complete),
        sp_selection_backend_executed =
          as.character(diagnostics$sp_selection_backend_executed),
        gcv_score_backend_executed =
          as.character(diagnostics$gcv_score_backend_executed),
        optimizer_backend_executed =
          as.character(diagnostics$optimizer_backend_executed),
        exact_replay_backend_executed =
          as.character(diagnostics$exact_replay_backend_executed),
        selected_sp_returned_to_r_before_solve =
          isTRUE(diagnostics$selected_sp_returned_to_r_before_solve),
        implicit_residual_d2h_count =
          as.integer(executed$info_before_shadow$implicit_residual_d2h_count),
        stringsAsFactors = FALSE
      )
      if (isTRUE(progress) && setup_ordinal %% 50L == 0L) {
        cat(
          "Phase 4 full shadow setups:", setup_ordinal, "/",
          length(all_setup_keys), "\n"
        )
        flush.console()
      }
    }
    rm(shard)
    gc(FALSE)
  }
  fixed_sp_cuda_runtime_free(runtime)
  runtime_freed <- TRUE
  targets <- do.call(rbind, target_parts)
  timings <- do.call(rbind, timing_parts)
  rownames(targets) <- rownames(timings) <- NULL
  logical_rows <- if (isTRUE(run_dcov)) {
    value <- do.call(rbind, logical_parts)
    value <- value[order(value$logical_sequence_id), , drop = FALSE]
    rownames(value) <- NULL
    value
  } else {
    data.frame()
  }
  summary <- fastkpc_full_cuda_phase4_shadow_summary(
    targets, logical_rows, timings, length(all_setup_keys), run_dcov,
    proc.time()[["elapsed"]] - started
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-gcv-shadow-evidence-v1",
    partition = list(
      schema_version =
        "full-cuda-ci-single-penalty-gcv-shadow-partition-v1",
      partition_id = partition$partition_id,
      partition_count = partition$partition_count,
      setup_keys = all_setup_keys
    ),
    summary = summary,
    targets = targets,
    logical_rows = logical_rows,
    timings = timings
  )
}

fastkpc_full_cuda_phase4_mixed_graph_replay <- function(
    catalog, phase4_logical_rows) {
  required <- c(
    "fastkpc_full_cuda_replay_logical_ci",
    "fastkpc_full_cuda_compare_candidate_skeleton",
    ".fastkpc_full_cuda_phase3_shadow_phase0_authority",
    ".fastkpc_full_cuda_phase3_load_shadow_phase0_oracle"
  )
  fastkpc_full_cuda_phase4_artifact_require(
    all(vapply(required, exists, logical(1L), mode = "function",
               inherits = TRUE)),
    "Phase 4 mixed graph replay dependencies are unavailable"
  )
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  setup_keys <- as.character(scope$setup_rows$prepared_s_key_sha256)
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  expected_phase4 <- plan$conditional_tests[
    plan$conditional_tests$prepared_s_key_x %in% setup_keys,
    , drop = FALSE
  ]
  expected_phase4 <- expected_phase4[order(
    expected_phase4$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  rows <- phase4_logical_rows[order(
    phase4_logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  lineage_fields <- c(
    "logical_sequence_id", "source_sequence_id", "source_task_index",
    "level", "x", "y", "S_key", "residual_key_x", "residual_key_y",
    "reference_p_value", "alpha",
    "reference_decision"
  )
  clean <- is.data.frame(rows) && nrow(rows) == 177952L &&
    nrow(expected_phase4) == 177952L &&
    all(expected_phase4$level %in% c(1L, 2L)) &&
    !anyDuplicated(rows$logical_sequence_id) &&
    identical(
      rows$prepared_s_key_sha256, expected_phase4$prepared_s_key_x
    ) &&
    all(vapply(lineage_fields, function(field) {
      identical(rows[[field]], expected_phase4[[field]])
    }, logical(1L))) &&
    !any(rows$decision_flip) && !any(rows$backend_error) &&
    !any(rows$spectra_fallback)
  fastkpc_full_cuda_phase4_artifact_require(
    clean, "Phase 4 mixed graph logical authority is malformed"
  )

  authority <- catalog$inputs$logical_tests
  candidate_p_value <- as.numeric(authority$reference_p_value)
  phase4_match <- match(
    rows$logical_sequence_id, authority$logical_sequence_id
  )
  fastkpc_full_cuda_phase4_artifact_require(
    nrow(authority) == 240489L && !anyNA(phase4_match) &&
      identical(authority$logical_sequence_id, seq_len(nrow(authority))) &&
      all(is.finite(candidate_p_value)),
    "Phase 4 mixed graph canonical logical authority is malformed"
  )
  candidate_p_value[phase4_match] <- rows$candidate_p_value
  fallback <- authority$level > 2L
  direct <- authority$level == 0L
  fastkpc_full_cuda_phase4_artifact_require(
    sum(direct) == 2213L && sum(fallback) == 60324L &&
      sum(!direct & !fallback) == 177952L &&
      all(authority$level[fallback] > 2L),
    "Phase 4 mixed graph fallback envelope is malformed"
  )
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
  fastkpc_full_cuda_phase4_artifact_require(
    pass, "Phase 4 mixed graph gate failed"
  )
  list(
    schema_version = "full-cuda-ci-single-penalty-gcv-mixed-graph-v1",
    summary = list(
      phase4_cuda_logical_test_count = 177952L,
      direct_legacy_logical_test_count = 2213L,
      explicit_legacy_fallback_count = 60324L,
      fallback_min_S_size = min(authority$level[fallback]),
      fallback_max_S_size = max(authority$level[fallback]),
      edge_count_reference = as.integer(graph$edge_count_reference),
      edge_count_candidate = as.integer(graph$edge_count_candidate),
      SHD = as.integer(graph$SHD),
      adjacency_identical = isTRUE(graph$adjacency_identical),
      sepsets_identical = isTRUE(graph$sepsets_identical),
      n_edgetests_identical = isTRUE(graph$n_edgetests_identical),
      deletions_identical = isTRUE(graph$deletions_identical),
      pass = pass
    ),
    replay = replay,
    comparison = comparison
  )
}

fastkpc_full_cuda_phase4_merge_full_shadow <- function(
    catalog, evidence_paths) {
  evidence_paths <- as.character(evidence_paths)
  fastkpc_full_cuda_phase4_artifact_require(
    length(evidence_paths) > 0L && !anyNA(evidence_paths) &&
      !anyDuplicated(evidence_paths) &&
      all(file.exists(evidence_paths) & !dir.exists(evidence_paths)),
    "Phase 4 shadow partition paths are malformed"
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
  part_clean <- length(partition_count) == 1L &&
    partition_count == length(parts) &&
    identical(sort(partition_ids), seq.int(0L, partition_count - 1L)) &&
    all(vapply(parts, function(value) {
      is.list(value) && identical(
        value$schema_version,
        "full-cuda-ci-single-penalty-gcv-shadow-evidence-v1"
      ) && identical(
        value$partition$schema_version,
        "full-cuda-ci-single-penalty-gcv-shadow-partition-v1"
      ) && isTRUE(value$summary$same_sp_fixed_solver_gate) &&
        isTRUE(value$summary$oracle_residual_gate) &&
        isTRUE(value$summary$downstream_decision_gate) &&
        isTRUE(value$summary$optimizer_coverage_gate) &&
        isTRUE(value$summary$backend_gate) &&
        nrow(value$logical_rows) > 0L
    }, logical(1L)))
  fastkpc_full_cuda_phase4_artifact_require(
    part_clean, "Phase 4 shadow partition evidence is incomplete"
  )
  parts <- parts[order(partition_ids)]
  evidence_paths <- evidence_paths[order(partition_ids)]
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  expected_setup_keys <- as.character(
    scope$setup_rows$prepared_s_key_sha256
  )
  observed_setup_keys <- unlist(lapply(
    parts, function(value) value$partition$setup_keys
  ), use.names = FALSE)
  fastkpc_full_cuda_phase4_artifact_require(
    length(observed_setup_keys) == 1174L &&
      !anyDuplicated(observed_setup_keys) &&
      identical(
        sort(observed_setup_keys, method = "radix"), expected_setup_keys
      ),
    "Phase 4 shadow partitions do not exactly cover the setup corpus"
  )

  targets <- do.call(rbind, lapply(parts, `[[`, "targets"))
  timings <- do.call(rbind, lapply(parts, `[[`, "timings"))
  logical_rows <- do.call(rbind, lapply(parts, `[[`, "logical_rows"))
  expected_target_keys <- as.character(scope$target_rows$residual_key_sha256)
  target_match <- match(expected_target_keys, targets$residual_key_sha256)
  setup_match <- match(expected_setup_keys, timings$prepared_s_key_sha256)
  logical_rows <- logical_rows[order(
    logical_rows$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  targets <- targets[target_match, , drop = FALSE]
  timings <- timings[setup_match, , drop = FALSE]
  rownames(targets) <- rownames(timings) <- rownames(logical_rows) <- NULL
  plan <- fastkpc_full_cuda_shadow_plan(catalog)
  expected_logical <- plan$conditional_tests[
    plan$conditional_tests$prepared_s_key_x %in% expected_setup_keys,
    , drop = FALSE
  ]
  expected_logical <- expected_logical[order(
    expected_logical$logical_sequence_id, method = "radix"
  ), , drop = FALSE]
  merge_clean <- nrow(targets) == 44941L && !anyNA(target_match) &&
    !anyDuplicated(targets$residual_key_sha256) &&
    identical(targets$residual_key_sha256, expected_target_keys) &&
    identical(
      targets$prepared_s_key_sha256,
      scope$target_rows$prepared_s_key_sha256
    ) && nrow(timings) == 1174L && !anyNA(setup_match) &&
    !anyDuplicated(timings$prepared_s_key_sha256) &&
    identical(timings$prepared_s_key_sha256, expected_setup_keys) &&
    nrow(logical_rows) == 177952L &&
    !anyDuplicated(logical_rows$logical_sequence_id) &&
    identical(
      logical_rows$logical_sequence_id,
      expected_logical$logical_sequence_id
    ) && identical(
      logical_rows$prepared_s_key_sha256,
      expected_logical$prepared_s_key_x
    ) && identical(
      logical_rows$residual_key_x, expected_logical$residual_key_x
    ) && identical(
      logical_rows$residual_key_y, expected_logical$residual_key_y
    )
  fastkpc_full_cuda_phase4_artifact_require(
    merge_clean, "Phase 4 merged shadow lineage is incomplete"
  )
  summary <- fastkpc_full_cuda_phase4_shadow_summary(
    targets, logical_rows, timings, 1174L, TRUE,
    max(vapply(parts, function(value) {
      as.numeric(value$summary$elapsed_seconds)
    }, numeric(1L)))
  )
  fastkpc_full_cuda_phase4_artifact_require(
    isTRUE(summary$same_sp_fixed_solver_gate) &&
      isTRUE(summary$oracle_residual_gate) &&
      isTRUE(summary$downstream_decision_gate) &&
      isTRUE(summary$optimizer_coverage_gate) &&
      isTRUE(summary$backend_gate),
    "Phase 4 merged shadow numerical or backend gate failed"
  )
  mixed_graph <- fastkpc_full_cuda_phase4_mixed_graph_replay(
    catalog, logical_rows
  )
  list(
    schema_version =
      "full-cuda-ci-single-penalty-gcv-full-shadow-merged-v1",
    summary = summary,
    mixed_graph = mixed_graph,
    partition_count = partition_count,
    partition_file_sha256 = setNames(
      as.list(vapply(
        evidence_paths, fastkpc_full_cuda_census_file_hash, character(1L)
      )), basename(evidence_paths)
    ),
    targets = targets,
    logical_rows = logical_rows,
    timings = timings
  )
}
