fastkpc_full_cuda_phase4_backend_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
}

fastkpc_full_cuda_phase4_backend_object_hash <- function(value) {
  fastkpc_full_cuda_census_hash_raw(serialize(value, NULL, version = 2L))
}

fastkpc_full_cuda_phase4_backend_build_requests <- function(
    catalog, progress = interactive()) {
  scope <- fastkpc_full_cuda_phase4_single_penalty_scope(catalog)
  setup_keys <- as.character(scope$setup_rows$prepared_s_key_sha256)
  requests <- vector("list", length(setup_keys))
  target_counts <- integer(length(setup_keys))
  target_id_hashes <- character(length(setup_keys))
  names(requests) <- names(target_counts) <- names(target_id_hashes) <- setup_keys
  preparation <- .fastkpc_full_cuda_prepared_s_prepare_shard_read(
    inputs = catalog$inputs,
    shard_count = catalog$catalog_contract$shard_count,
    expected_source_commit = catalog$phase2_manifest$source_commit
  )
  built <- 0L
  started <- proc.time()[["elapsed"]]
  for (shard_id in scope$shard_ids) {
    shard <- fastkpc_full_cuda_phase4_read_shard(
      catalog, scope, shard_id, preparation = preparation
    )
    for (setup_key in names(shard$setups)) {
      setup_position <- match(setup_key, setup_keys)
      fastkpc_full_cuda_phase4_backend_require(
        !is.na(setup_position) && is.null(requests[[setup_position]]),
        "Phase 4 backend request setup ownership is malformed"
      )
      batch <- fastkpc_full_cuda_phase4_batch_from_shard(
        catalog, shard, setup_key
      )
      request <- fastkpc_full_cuda_phase4_cuda_native_input(
        prepared_setup = batch$setup,
        Y = batch$Y,
        target_ids = batch$target_ids
      )$native
      requests[[setup_position]] <- request
      target_counts[[setup_position]] <- length(request$target_ids)
      target_id_hashes[[setup_position]] <-
        fastkpc_full_cuda_census_named_metadata_hash(list(
          prepared_s_key_sha256 = setup_key,
          target_ids = request$target_ids
        ))
      built <- built + 1L
      if (isTRUE(progress) && built %% 100L == 0L) {
        cat("Phase 4 backend requests:", built, "/ 1174\n")
        flush.console()
      }
    }
    rm(shard)
    gc(FALSE)
  }
  request_identity <- data.frame(
    prepared_s_key_sha256 = setup_keys,
    target_count = unname(target_counts),
    target_ids_sha256 = unname(target_id_hashes),
    stringsAsFactors = FALSE
  )
  clean <- built == 1174L && length(requests) == 1174L &&
    !any(vapply(requests, is.null, logical(1L))) &&
    sum(target_counts) == 44941L && all(target_counts > 0L) &&
    !anyNA(target_id_hashes) &&
    all(grepl("^[0-9a-f]{64}$", target_id_hashes)) &&
    identical(names(requests), setup_keys)
  fastkpc_full_cuda_phase4_backend_require(
    clean, "Phase 4 full backend request corpus is incomplete"
  )
  list(
    schema_version = "full-cuda-ci-phase4-backend-requests-v1",
    setup_keys = setup_keys,
    target_counts = unname(target_counts),
    request_identity = request_identity,
    request_identity_sha256 =
      fastkpc_full_cuda_phase4_backend_object_hash(request_identity),
    request_payload_sha256 =
      fastkpc_full_cuda_phase4_backend_object_hash(requests),
    build_wall_ms = 1000 * (proc.time()[["elapsed"]] - started),
    requests = requests
  )
}

fastkpc_full_cuda_phase4_validate_backend_requests <- function(value) {
  required <- c(
    "schema_version", "setup_keys", "target_counts", "request_identity",
    "request_identity_sha256", "request_payload_sha256", "build_wall_ms",
    "requests"
  )
  clean <- is.list(value) && identical(names(value), required) &&
    identical(
      value$schema_version, "full-cuda-ci-phase4-backend-requests-v1"
    ) && length(value$setup_keys) == 1174L &&
    length(value$target_counts) == 1174L &&
    sum(value$target_counts) == 44941L &&
    nrow(value$request_identity) == 1174L &&
    length(value$requests) == 1174L &&
    identical(names(value$requests), value$setup_keys) &&
    identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$request_identity),
      value$request_identity_sha256
    ) && identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$requests),
      value$request_payload_sha256
    )
  fastkpc_full_cuda_phase4_backend_require(
    clean, "Phase 4 backend request cache validation failed"
  )
  invisible(value)
}

fastkpc_full_cuda_phase4_backend_cpu_request <- function(request) {
  X <- as.matrix(request$X)
  Y <- as.matrix(request$Y)
  projected_rhs <- crossprod(X, Y)
  spectral_projection <- request$rhs_transform %*% projected_rhs
  y_squared_norm <- colSums(Y * Y)
  spectral <- list(
    eigenvalues = as.numeric(request$eigenvalues),
    initial_sp = as.numeric(request$initial_sp),
    n = nrow(Y)
  )
  optimized <- lapply(seq_len(ncol(Y)), function(target) {
    fastkpc_full_cuda_phase4_magic1_optimize(
      spectral = spectral,
      squared_projection = spectral_projection[, target]^2,
      y_squared_norm = y_squared_norm[[target]],
      keep_transcript = FALSE
    )
  })
  fields <- list(
    target_position = seq_len(ncol(Y)),
    sp = vapply(optimized, `[[`, numeric(1L), "sp"),
    log_sp = vapply(optimized, `[[`, numeric(1L), "log_sp"),
    score = vapply(optimized, `[[`, numeric(1L), "score"),
    edf = vapply(optimized, `[[`, numeric(1L), "edf"),
    rss = vapply(optimized, `[[`, numeric(1L), "rss"),
    gradient = vapply(optimized, `[[`, numeric(1L), "gradient"),
    hessian = vapply(optimized, `[[`, numeric(1L), "hessian"),
    reported_rms_gradient = vapply(
      optimized, `[[`, numeric(1L), "reported_rms_gradient"
    ),
    pre_boundary_log_sp = vapply(
      optimized, `[[`, numeric(1L), "pre_boundary_log_sp"
    ),
    iteration_count = vapply(
      optimized, `[[`, integer(1L), "iteration_count"
    ),
    score_call_count = vapply(
      optimized, `[[`, integer(1L), "score_call_count"
    ),
    actual_objective_call_count = vapply(
      optimized, `[[`, integer(1L), "actual_objective_call_count"
    ),
    fully_converged = vapply(
      optimized, `[[`, logical(1L), "fully_converged"
    ),
    hessian_positive_definite = vapply(
      optimized, `[[`, logical(1L), "hessian_positive_definite"
    ),
    boundary_probe_count = vapply(
      optimized, `[[`, integer(1L), "boundary_probe_count"
    ),
    boundary_accepted_count = vapply(
      optimized, `[[`, integer(1L), "boundary_accepted_count"
    ),
    termination_reason = vapply(
      optimized, `[[`, character(1L), "termination_reason"
    ),
    boundary_status = vapply(
      optimized, `[[`, character(1L), "boundary_status"
    )
  )
  as.data.frame(fields, stringsAsFactors = FALSE)
}

fastkpc_full_cuda_phase4_backend_flatten_targets <- function(
    setup_results, requests) {
  setup_keys <- names(requests)
  fastkpc_full_cuda_phase4_backend_require(
    is.list(setup_results) && length(setup_results) == length(requests) &&
      identical(names(setup_results), setup_keys),
    "Phase 4 backend setup result ordering is malformed"
  )
  parts <- lapply(seq_along(setup_results), function(index) {
    value <- setup_results[[index]]
    targets <- if (is.data.frame(value)) value else value$targets
    request <- requests[[index]]
    fastkpc_full_cuda_phase4_backend_require(
      is.data.frame(targets) && nrow(targets) == length(request$target_ids),
      "Phase 4 backend target result geometry is malformed"
    )
    targets$prepared_s_key_sha256 <- setup_keys[[index]]
    targets$target <- as.integer(request$target_ids)
    targets
  })
  targets <- do.call(rbind, parts)
  rownames(targets) <- NULL
  fastkpc_full_cuda_phase4_backend_require(
    nrow(targets) == 44941L &&
      !anyDuplicated(paste(targets$prepared_s_key_sha256, targets$target)),
    "Phase 4 backend target coverage is incomplete"
  )
  targets
}

fastkpc_full_cuda_phase4_backend_counter_fields <- function() {
  c(
    "spectral_optimizer_target_count", "spectral_only_target_count",
    "exact_replay_target_count", "exact_replay_endpoint_risk_count",
    "exact_replay_convergence_risk_count",
    "exact_replay_boundary_risk_count",
    "exact_replay_numerical_risk_count", "cpu_score_count",
    "cpu_optimizer_count", "fallback_count", "legacy_mgcv_target_calls"
  )
}

fastkpc_full_cuda_phase4_backend_candidate_once <- function(requests) {
  started <- proc.time()[["elapsed"]]
  native <- full_cuda_ci_single_penalty_gcv_multi_cuda(
    requests, concurrency = 1L
  )
  external_wall_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  targets <- fastkpc_full_cuda_phase4_backend_flatten_targets(
    native$setups, requests
  )
  counter_fields <- fastkpc_full_cuda_phase4_backend_counter_fields()
  counter_signature <- do.call(rbind, lapply(native$setups, function(value) {
    unlist(value$diagnostics[counter_fields], use.names = TRUE)
  }))
  rownames(counter_signature) <- names(requests)
  totals <- colSums(counter_signature)
  backend_gate <- totals[["spectral_optimizer_target_count"]] == 44941 &&
    totals[["spectral_only_target_count"]] +
      totals[["exact_replay_target_count"]] == 44941 &&
    totals[["cpu_score_count"]] == 0 &&
    totals[["cpu_optimizer_count"]] == 0 &&
    totals[["fallback_count"]] == 0 &&
    totals[["legacy_mgcv_target_calls"]] == 0 &&
    identical(
      native$diagnostics$execution_strategy,
      "cuda-cross-setup-fused-exact-replay"
    ) && native$diagnostics$device_id == 0L &&
    native$diagnostics$fused_exact_replay_target_count ==
      totals[["exact_replay_target_count"]] &&
    all(vapply(native$setups, function(value) {
      diagnostics <- value$diagnostics
      isTRUE(diagnostics$optimizer_target_coverage_complete) &&
        identical(diagnostics$sp_selection_backend_executed, "cuda") &&
        identical(diagnostics$gcv_score_backend_executed, "cuda") &&
        identical(
          diagnostics$optimizer_backend_executed,
          "cuda-spectral-risk-gated-exact-replay"
        ) && identical(
          diagnostics$exact_replay_backend_executed,
          "cuda-dpstf2-lapack-3.12-dgesdd"
        )
    }, logical(1L)))
  fastkpc_full_cuda_phase4_backend_require(
    backend_gate, "Phase 4 candidate backend authority gate failed"
  )
  list(
    targets = targets,
    targets_sha256 = fastkpc_full_cuda_phase4_backend_object_hash(targets),
    counter_signature = counter_signature,
    counter_signature_sha256 =
      fastkpc_full_cuda_phase4_backend_object_hash(counter_signature),
    totals = totals,
    measurement = data.frame(
      route = "cuda-spectral-risk-gated-exact-replay",
      external_wall_ms = external_wall_ms,
      native_wall_ms = as.numeric(native$diagnostics$wall_host_ms),
      spectral_setup_wall_ms =
        as.numeric(native$diagnostics$spectral_setup_wall_ms),
      fused_exact_cuda_ms =
        as.numeric(native$diagnostics$fused_exact_replay_cuda_ms),
      fused_exact_host_ms =
        as.numeric(native$diagnostics$fused_exact_replay_host_ms),
      exact_replay_target_count =
        as.integer(totals[["exact_replay_target_count"]]),
      numerical_risk_count =
        as.integer(totals[["exact_replay_numerical_risk_count"]]),
      cpu_score_count = as.integer(totals[["cpu_score_count"]]),
      cpu_optimizer_count = as.integer(totals[["cpu_optimizer_count"]]),
      fallback_count = as.integer(totals[["fallback_count"]]),
      legacy_mgcv_target_calls =
        as.integer(totals[["legacy_mgcv_target_calls"]]),
      backend_gate = backend_gate,
      stringsAsFactors = FALSE
    )
  )
}

fastkpc_full_cuda_phase4_backend_cpu_once <- function(requests) {
  started <- proc.time()[["elapsed"]]
  setup_results <- lapply(
    requests, fastkpc_full_cuda_phase4_backend_cpu_request
  )
  external_wall_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  targets <- fastkpc_full_cuda_phase4_backend_flatten_targets(
    setup_results, requests
  )
  list(
    targets = targets,
    targets_sha256 = fastkpc_full_cuda_phase4_backend_object_hash(targets),
    measurement = data.frame(
      route = "r-cpu-spectral",
      external_wall_ms = external_wall_ms,
      native_wall_ms = NA_real_,
      spectral_setup_wall_ms = NA_real_,
      fused_exact_cuda_ms = NA_real_,
      fused_exact_host_ms = NA_real_,
      exact_replay_target_count = 0L,
      numerical_risk_count = 0L,
      cpu_score_count = nrow(targets),
      cpu_optimizer_count = nrow(targets),
      fallback_count = 0L,
      legacy_mgcv_target_calls = 0L,
      backend_gate = TRUE,
      stringsAsFactors = FALSE
    )
  )
}

fastkpc_full_cuda_phase4_run_backend_benchmark <- function(
    catalog, repetitions = 5L, progress = interactive(),
    request_corpus = NULL) {
  repetitions <- as.integer(repetitions)
  fastkpc_full_cuda_phase4_backend_require(
    length(repetitions) == 1L && !is.na(repetitions) && repetitions >= 5L,
    "Phase 4 backend benchmark requires at least five repetitions"
  )
  corpus <- if (is.null(request_corpus)) {
    fastkpc_full_cuda_phase4_backend_build_requests(
      catalog, progress = progress
    )
  } else {
    fastkpc_full_cuda_phase4_validate_backend_requests(request_corpus)
    request_corpus
  }
  requests <- corpus$requests
  if (isTRUE(progress)) {
    cat("Phase 4 backend warm-up: CUDA\n")
    flush.console()
  }
  warm_candidate <- fastkpc_full_cuda_phase4_backend_candidate_once(requests)
  if (isTRUE(progress)) {
    cat("Phase 4 backend warm-up: R CPU spectral\n")
    flush.console()
  }
  warm_baseline <- fastkpc_full_cuda_phase4_backend_cpu_once(requests)

  candidate_runs <- vector("list", repetitions)
  baseline_runs <- vector("list", repetitions)
  for (repetition in seq_len(repetitions)) {
    if (isTRUE(progress)) {
      cat("Phase 4 backend repetition", repetition, "/", repetitions, "\n")
      flush.console()
    }
    baseline_runs[[repetition]] <-
      fastkpc_full_cuda_phase4_backend_cpu_once(requests)
    candidate_runs[[repetition]] <-
      fastkpc_full_cuda_phase4_backend_candidate_once(requests)
  }
  candidate_measurements <- do.call(rbind, lapply(
    candidate_runs, `[[`, "measurement"
  ))
  baseline_measurements <- do.call(rbind, lapply(
    baseline_runs, `[[`, "measurement"
  ))
  candidate_measurements$repetition <- seq_len(repetitions)
  baseline_measurements$repetition <- seq_len(repetitions)
  candidate_hashes <- vapply(
    candidate_runs, `[[`, character(1L), "targets_sha256"
  )
  counter_hashes <- vapply(
    candidate_runs, `[[`, character(1L), "counter_signature_sha256"
  )
  baseline_hashes <- vapply(
    baseline_runs, `[[`, character(1L), "targets_sha256"
  )
  candidate_median_ms <- stats::median(candidate_measurements$external_wall_ms)
  baseline_median_ms <- stats::median(baseline_measurements$external_wall_ms)
  summary <- list(
    schema_version = "full-cuda-ci-phase4-backend-summary-v1",
    setup_count = length(requests),
    target_count = sum(corpus$target_counts),
    request_identity_sha256 = corpus$request_identity_sha256,
    request_payload_sha256 = corpus$request_payload_sha256,
    request_build_wall_ms = corpus$build_wall_ms,
    repetition_count = repetitions,
    warm_candidate_external_ms =
      warm_candidate$measurement$external_wall_ms[[1L]],
    warm_baseline_external_ms =
      warm_baseline$measurement$external_wall_ms[[1L]],
    candidate_median_ms = candidate_median_ms,
    candidate_minimum_ms = min(candidate_measurements$external_wall_ms),
    candidate_maximum_ms = max(candidate_measurements$external_wall_ms),
    candidate_mad_ms = stats::mad(candidate_measurements$external_wall_ms),
    baseline_median_ms = baseline_median_ms,
    baseline_minimum_ms = min(baseline_measurements$external_wall_ms),
    baseline_maximum_ms = max(baseline_measurements$external_wall_ms),
    baseline_mad_ms = stats::mad(baseline_measurements$external_wall_ms),
    candidate_to_baseline_ratio = candidate_median_ms / baseline_median_ms,
    exact_replay_target_count =
      candidate_runs[[1L]]$measurement$exact_replay_target_count[[1L]],
    numerical_risk_count =
      candidate_runs[[1L]]$measurement$numerical_risk_count[[1L]],
    all_candidate_targets_identical = length(unique(candidate_hashes)) == 1L,
    all_candidate_counters_identical = length(unique(counter_hashes)) == 1L,
    all_baseline_targets_identical = length(unique(baseline_hashes)) == 1L,
    backend_gate = all(candidate_measurements$backend_gate),
    absolute_performance_gate = candidate_median_ms <= 25000,
    relative_performance_gate = candidate_median_ms < baseline_median_ms,
    pass = all(candidate_measurements$backend_gate) &&
      length(unique(candidate_hashes)) == 1L &&
      length(unique(counter_hashes)) == 1L &&
      length(unique(baseline_hashes)) == 1L &&
      candidate_median_ms <= 25000 && candidate_median_ms < baseline_median_ms
  )
  fastkpc_full_cuda_phase4_backend_require(
    isTRUE(summary$pass), "Phase 4 backend benchmark gate failed"
  )
  list(
    schema_version = "full-cuda-ci-phase4-backend-evidence-v1",
    summary = summary,
    request_identity = corpus$request_identity,
    warm_candidate = warm_candidate$measurement,
    warm_baseline = warm_baseline$measurement,
    candidate_measurements = candidate_measurements,
    baseline_measurements = baseline_measurements,
    candidate_target_hashes = candidate_hashes,
    baseline_target_hashes = baseline_hashes,
    counter_signature_hashes = counter_hashes,
    candidate_targets = candidate_runs[[1L]]$targets,
    baseline_targets = baseline_runs[[1L]]$targets,
    counter_signature = candidate_runs[[1L]]$counter_signature
  )
}

fastkpc_full_cuda_phase4_validate_backend_evidence <- function(value) {
  required <- c(
    "schema_version", "summary", "request_identity", "warm_candidate",
    "warm_baseline", "candidate_measurements", "baseline_measurements",
    "candidate_target_hashes", "baseline_target_hashes",
    "counter_signature_hashes", "candidate_targets", "baseline_targets",
    "counter_signature"
  )
  fastkpc_full_cuda_phase4_backend_require(
    is.list(value) && identical(names(value), required) && identical(
      value$schema_version, "full-cuda-ci-phase4-backend-evidence-v1"
    ),
    "Phase 4 backend evidence schema is malformed"
  )
  summary <- value$summary
  repetitions <- as.integer(summary$repetition_count)
  clean <- identical(
    summary$schema_version, "full-cuda-ci-phase4-backend-summary-v1"
  ) && summary$setup_count == 1174L && summary$target_count == 44941L &&
    repetitions >= 5L && nrow(value$request_identity) == 1174L &&
    nrow(value$candidate_measurements) == repetitions &&
    nrow(value$baseline_measurements) == repetitions &&
    nrow(value$candidate_targets) == 44941L &&
    nrow(value$baseline_targets) == 44941L &&
    nrow(value$counter_signature) == 1174L &&
    identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$request_identity),
      summary$request_identity_sha256
    ) && identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$candidate_targets),
      value$candidate_target_hashes[[1L]]
    ) && identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$baseline_targets),
      value$baseline_target_hashes[[1L]]
    ) && identical(
      fastkpc_full_cuda_phase4_backend_object_hash(value$counter_signature),
      value$counter_signature_hashes[[1L]]
    ) && length(unique(value$candidate_target_hashes)) == 1L &&
    length(unique(value$baseline_target_hashes)) == 1L &&
    length(unique(value$counter_signature_hashes)) == 1L &&
    isTRUE(summary$backend_gate) &&
    isTRUE(summary$absolute_performance_gate) &&
    isTRUE(summary$relative_performance_gate) && isTRUE(summary$pass)
  fastkpc_full_cuda_phase4_backend_require(
    clean, "Phase 4 backend evidence validation failed"
  )
  invisible(value)
}
