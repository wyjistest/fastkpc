fastkpc_full_cuda_phase10_compute_require <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

fastkpc_full_cuda_phase10_compute_staging_dir <- function(
    root = file.path("fastkpc", "artifacts", "full_cuda_ci")) {
  file.path(root, "phase10_promotion_staging_v2", "compute-runs")
}

fastkpc_full_cuda_phase10_compute_run_path <- function(
    repetition,
    staging_dir = fastkpc_full_cuda_phase10_compute_staging_dir()) {
  fastkpc_full_cuda_phase10_compute_require(
    length(repetition) == 1L && !is.na(repetition) &&
      repetition >= 1L && repetition <= 5L,
    "Phase 10 compute-warm repetition is outside [1, 5]"
  )
  file.path(staging_dir, sprintf("candidate_compute_warm-%02d.rds",
                                 as.integer(repetition)))
}

fastkpc_full_cuda_phase10_compute_prewarm_data <- function() {
  z <- seq(-2, 2, length.out = 72L)
  data <- cbind(
    prewarm_a = sin(z) + 0.01 * sin(17 * z),
    prewarm_b = cos(z) + 0.01 * cos(13 * z),
    prewarm_c = z + 0.01 * sin(11 * z),
    prewarm_d = z^2 + 0.01 * cos(19 * z)
  )
  storage.mode(data) <- "double"
  data
}

fastkpc_full_cuda_phase10_compute_candidate_call <- function(
    data, max_conditioning_size = 7L) {
  fastkpc_compatible_cuda_skeleton(
    data = data,
    alpha = 0.1,
    labels = colnames(data),
    options = list(
      route = "full_cuda",
      compatible_cuda_strict = TRUE,
      max_conditioning_size = as.integer(max_conditioning_size),
      index = 1,
      numCol = 35L,
      trace_level = "logical"
    )
  )
}

fastkpc_full_cuda_phase10_compute_cache_state <- function(data) {
  state <- full_cuda_ci_one_call_cache_state_native(data)
  fastkpc_full_cuda_phase10_compute_require(
    is.list(state) && identical(
      state$schema_version, "full-cuda-ci-dataset-cache-state-v2"
    ) && grepl("^[0-9a-f]{64}$", state$dataset_key) &&
      is.finite(state$cache_epoch) && state$cache_epoch >= 1 &&
      state$result_cache_entries >= 0L &&
      state$result_cache_dataset_entries >= 0L &&
      state$result_cache_dataset_entries <= state$result_cache_entries &&
      state$target_cache_entries >= 0L &&
      state$target_cache_dataset_entries >= 0L &&
      state$target_cache_dataset_entries <= state$target_cache_entries,
    "Phase 10 dataset cache state is malformed"
  )
  state
}

fastkpc_full_cuda_phase10_compute_validate_generic_result <- function(result) {
  zero_fields <- fastkpc_full_cuda_phase10_campaign_authority_zero_fields()
  summary <- result$summary
  zero_values <- vapply(zero_fields, function(field) {
    value <- summary[[field]]
    if (is.null(value)) NA_real_ else as.numeric(value)
  }, numeric(1L))
  clean <- fastkpc_full_cuda_is_skeleton(result) &&
    identical(summary$run_status, "ok") &&
    identical(summary$entrypoint,
              "compatible-cuda-full-skeleton-native-v1") &&
    identical(summary$compatible_cuda_route, "compatible.cuda") &&
    isTRUE(summary$compatible_cuda_strict) &&
    isTRUE(summary$authority_gate_pass) &&
    as.integer(summary$native_call_count) == 1L &&
    as.integer(summary$logical_tests_consumed) > 0L &&
    as.integer(summary$physical_tests_evaluated) > 0L &&
    as.integer(summary$physical_residual_fits) > 0L &&
    as.integer(summary$native_setup_count) > 0L &&
    as.integer(summary$cuda_dcov_component_count) > 0L &&
    as.integer(summary$cuda_dcov_pair_count) > 0L &&
    as.integer(summary$cuda_gamma_pvalue_count) > 0L &&
    all(!is.na(zero_values) & zero_values == 0) &&
    all(is.finite(result$tasks$p_used))
  fastkpc_full_cuda_phase10_compute_require(
    clean, "Phase 10 generic CUDA prewarm result is invalid"
  )
  invisible(result)
}

fastkpc_full_cuda_phase10_compute_required_positive_fields <- function() {
  contract <- fastkpc_full_cuda_phase10_load_performance_budget_v2()
  unname(unlist(
    contract$payload$boundaries$fresh_data_compute_warm$
      required_positive_counters,
    use.names = FALSE
  ))
}

fastkpc_full_cuda_phase10_compute_validate_cache_precondition <- function(
    proof) {
  required <- c(
    "dataset_key", "prewarm_dataset_key", "cache_epoch_before_reset",
    "cache_epoch_after_reset", "result_cache_entries_before",
    "result_cache_entries_after_reset",
    "result_cache_dataset_entries_before",
    "result_cache_dataset_entries_after_reset",
    "target_cache_entries_before", "target_cache_entries_after_reset",
    "target_cache_dataset_entries_before",
    "target_cache_dataset_entries_after_reset",
    "preexisting_result_cache_hit_count",
    "preexisting_target_cache_hit_count"
  )
  clean <- is.list(proof) && identical(names(proof), required) &&
    grepl("^[0-9a-f]{64}$", proof$dataset_key) &&
    grepl("^[0-9a-f]{64}$", proof$prewarm_dataset_key) &&
    !identical(proof$dataset_key, proof$prewarm_dataset_key) &&
    is.finite(proof$cache_epoch_before_reset) &&
    is.finite(proof$cache_epoch_after_reset) &&
    proof$cache_epoch_after_reset > proof$cache_epoch_before_reset &&
    proof$result_cache_entries_before > 0L &&
    proof$target_cache_entries_before > 0L &&
    proof$result_cache_dataset_entries_before == 0L &&
    proof$target_cache_dataset_entries_before == 0L &&
    proof$result_cache_entries_after_reset == 0L &&
    proof$result_cache_dataset_entries_after_reset == 0L &&
    proof$target_cache_entries_after_reset == 0L &&
    proof$target_cache_dataset_entries_after_reset == 0L &&
    proof$preexisting_result_cache_hit_count == 0L &&
    proof$preexisting_target_cache_hit_count == 0L
  fastkpc_full_cuda_phase10_compute_require(
    clean, "Phase 10 compute-warm cache precondition failed"
  )
  invisible(proof)
}

fastkpc_full_cuda_phase10_compute_validate_measured_result <- function(
    result, cache_precondition, formal_canonical = TRUE) {
  if (isTRUE(formal_canonical)) {
    fastkpc_full_cuda_phase10_validate_candidate_result(
      result, boundary = "cold"
    )
  } else {
    fastkpc_full_cuda_phase10_compute_validate_generic_result(result)
  }
  summary <- result$summary
  positive <- fastkpc_full_cuda_phase10_compute_required_positive_fields()
  positive_values <- vapply(positive, function(field) {
    value <- summary[[field]]
    if (is.null(value)) NA_real_ else as.numeric(value)
  }, numeric(1L))
  if (!isTRUE(formal_canonical)) {
    positive_values[c(
      "cuda_single_penalty_target_count",
      "cuda_multi_penalty_target_count"
    )] <- 1
  }
  clean <- identical(summary$dataset_key, cache_precondition$dataset_key) &&
    summary$dataset_cache_epoch_at_start ==
      cache_precondition$cache_epoch_after_reset &&
    summary$result_cache_warm_start_entries == 0L &&
    summary$result_cache_dataset_warm_start_entries == 0L &&
    summary$result_cache_hit_count == 0L &&
    summary$result_cache_preexisting_hit_count == 0L &&
    summary$target_cache_warm_start_entries == 0L &&
    summary$target_cache_dataset_warm_start_entries == 0L &&
    summary$target_cache_preexisting_hit_count == 0L &&
    summary$native_setup_cache_warm_start_entries == 0L &&
    summary$residual_cache_warm_start_entries == 0L &&
    summary$component_cache_warm_start_entries == 0L &&
    all(!is.na(positive_values) & positive_values > 0)
  fastkpc_full_cuda_phase10_compute_require(
    clean,
    "Phase 10 compute-warm measured call was replayed or skipped CUDA work"
  )
  invisible(result)
}

fastkpc_full_cuda_phase10_capture_compute_warm <- function(
    data, repetition = 1L, max_conditioning_size = 7L,
    formal_canonical = TRUE, capture_machine = formal_canonical) {
  fastkpc_full_cuda_phase10_compute_run_path(repetition)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fastkpc_full_cuda_phase10_compute_require(
    nrow(data) > 35L && ncol(data) >= 2L && ncol(data) <= 64L &&
      all(is.finite(data)) && !is.null(colnames(data)),
    "Phase 10 compute-warm measured data is malformed"
  )
  if (isTRUE(formal_canonical)) {
    fastkpc_full_cuda_phase10_compute_require(
      identical(dim(data), c(351L, 48L)) &&
        as.integer(max_conditioning_size) == 7L,
      "Phase 10 formal compute-warm boundary requires canonical 351x48 data"
    )
  }
  fastkpc_full_cuda_phase10_load_performance_budget_v2()
  invisible(full_cuda_ci_one_call_cache_control_native(
    "configure", 262144L
  ))
  invisible(full_cuda_ci_one_call_cache_control_native(
    "configure_target", 131072L
  ))
  invisible(full_cuda_ci_one_call_cache_control_native("reset"))

  measured_initial <- fastkpc_full_cuda_phase10_compute_cache_state(data)
  prewarm_data <- fastkpc_full_cuda_phase10_compute_prewarm_data()
  prewarm <- fastkpc_full_cuda_phase10_campaign_timed_call(
    fastkpc_full_cuda_phase10_compute_candidate_call(
      prewarm_data, max_conditioning_size = 1L
    )
  )
  fastkpc_full_cuda_phase10_compute_validate_generic_result(prewarm$value)
  prewarm_state <- fastkpc_full_cuda_phase10_compute_cache_state(prewarm_data)
  before_reset <- fastkpc_full_cuda_phase10_compute_cache_state(data)
  fastkpc_full_cuda_phase10_compute_require(
    identical(measured_initial$dataset_key, before_reset$dataset_key) &&
      identical(prewarm$value$summary$dataset_key,
                prewarm_state$dataset_key) &&
      !identical(before_reset$dataset_key, prewarm_state$dataset_key) &&
      before_reset$result_cache_dataset_entries == 0L &&
      before_reset$target_cache_dataset_entries == 0L &&
      before_reset$result_cache_entries > 0L &&
      before_reset$target_cache_entries > 0L,
    "Phase 10 compute-warm prewarm contaminated the measured DatasetKey"
  )

  reset_receipt <- full_cuda_ci_one_call_cache_control_native("reset")
  after_reset <- fastkpc_full_cuda_phase10_compute_cache_state(data)
  cache_precondition <- list(
    dataset_key = after_reset$dataset_key,
    prewarm_dataset_key = prewarm_state$dataset_key,
    cache_epoch_before_reset = before_reset$cache_epoch,
    cache_epoch_after_reset = after_reset$cache_epoch,
    result_cache_entries_before = before_reset$result_cache_entries,
    result_cache_entries_after_reset = after_reset$result_cache_entries,
    result_cache_dataset_entries_before =
      before_reset$result_cache_dataset_entries,
    result_cache_dataset_entries_after_reset =
      after_reset$result_cache_dataset_entries,
    target_cache_entries_before = before_reset$target_cache_entries,
    target_cache_entries_after_reset = after_reset$target_cache_entries,
    target_cache_dataset_entries_before =
      before_reset$target_cache_dataset_entries,
    target_cache_dataset_entries_after_reset =
      after_reset$target_cache_dataset_entries,
    preexisting_result_cache_hit_count = 0L,
    preexisting_target_cache_hit_count = 0L
  )
  fastkpc_full_cuda_phase10_compute_require(
    reset_receipt$cache_epoch == after_reset$cache_epoch,
    "Phase 10 compute-warm reset receipt epoch drifted"
  )
  fastkpc_full_cuda_phase10_compute_validate_cache_precondition(
    cache_precondition
  )

  resource_before <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  machine_before <- if (isTRUE(capture_machine)) {
    value <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
      require_idle_gpu = FALSE
    )
    fastkpc_full_cuda_phase10_compute_require(
      nrow(value$compute_processes) == 0L ||
        all(value$compute_processes$pid == Sys.getpid()),
      "Phase 10 compute-warm found a concurrent external GPU process"
    )
    value
  } else NULL
  captured <- fastkpc_full_cuda_phase10_campaign_timed_call(
    fastkpc_full_cuda_phase10_compute_candidate_call(
      data, max_conditioning_size = max_conditioning_size
    )
  )
  fastkpc_full_cuda_phase10_compute_validate_measured_result(
    captured$value, cache_precondition,
    formal_canonical = formal_canonical
  )
  cache_precondition$preexisting_result_cache_hit_count <-
    as.integer(captured$value$summary$result_cache_preexisting_hit_count)
  cache_precondition$preexisting_target_cache_hit_count <-
    as.integer(captured$value$summary$target_cache_preexisting_hit_count)
  fastkpc_full_cuda_phase10_compute_validate_cache_precondition(
    cache_precondition
  )
  after_measurement <- fastkpc_full_cuda_phase10_compute_cache_state(data)
  resource_after <- fastkpc_full_cuda_phase10_campaign_resource_snapshot()
  machine_after <- if (isTRUE(capture_machine)) {
    value <- fastkpc_full_cuda_phase10_campaign_machine_snapshot(
      require_idle_gpu = FALSE
    )
    fastkpc_full_cuda_phase10_compute_require(
      nrow(value$compute_processes) == 0L ||
        all(value$compute_processes$pid == Sys.getpid()),
      "Phase 10 compute-warm found a concurrent external GPU process"
    )
    value
  } else NULL
  fastkpc_full_cuda_phase10_compute_require(
    after_measurement$cache_epoch == after_reset$cache_epoch &&
      after_measurement$result_cache_dataset_entries > 0L &&
      after_measurement$target_cache_dataset_entries > 0L &&
      all(fastkpc_full_cuda_phase10_campaign_active_resources(
        resource_before
      ) == 0) &&
      all(fastkpc_full_cuda_phase10_campaign_active_resources(
        resource_after
      ) == 0),
    "Phase 10 compute-warm cache publication or resource cleanup failed"
  )
  if (isTRUE(capture_machine)) {
    fastkpc_full_cuda_phase10_compute_require(
      identical(
        fastkpc_full_cuda_phase10_campaign_machine_identity(machine_before),
        fastkpc_full_cuda_phase10_campaign_machine_identity(machine_after)
      ),
      "Phase 10 compute-warm machine identity drifted"
    )
  }
  evidence <- list(
    schema_version = "full-cuda-ci-phase10-compute-warm-evidence-v2",
    run_key = sprintf("candidate_compute_warm-%02d", as.integer(repetition)),
    mode = "candidate_compute_warm",
    repetition = as.integer(repetition),
    formal_canonical = isTRUE(formal_canonical),
    cache_precondition = cache_precondition,
    prewarm = list(
      policy = "different-DatasetKey-and-noncanonical",
      n = nrow(prewarm_data), p = ncol(prewarm_data),
      elapsed_sec = prewarm$elapsed_sec,
      dataset_key = prewarm$value$summary$dataset_key,
      physical_tests_evaluated =
        prewarm$value$summary$physical_tests_evaluated,
      pass = TRUE
    ),
    started_utc = captured$started_utc,
    ended_utc = captured$ended_utc,
    timing = captured$timing,
    elapsed_sec = captured$elapsed_sec,
    result = captured$value,
    cache_after_measurement = after_measurement,
    machine_before = machine_before,
    machine_after = machine_after,
    resource_before = resource_before,
    resource_after = resource_after,
    pass = TRUE
  )
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
  evidence
}

fastkpc_full_cuda_phase10_validate_compute_warm_evidence <- function(evidence) {
  fastkpc_full_cuda_phase10_compute_require(
    is.list(evidence) && identical(
      evidence$schema_version,
      "full-cuda-ci-phase10-compute-warm-evidence-v2"
    ) && identical(evidence$mode, "candidate_compute_warm") &&
      identical(
        evidence$run_key,
        sprintf("candidate_compute_warm-%02d", evidence$repetition)
      ) && evidence$repetition >= 1L && evidence$repetition <= 5L &&
      is.finite(evidence$elapsed_sec) && evidence$elapsed_sec > 0 &&
      isTRUE(evidence$pass),
    "Phase 10 compute-warm evidence schema is malformed"
  )
  fastkpc_full_cuda_phase10_compute_validate_cache_precondition(
    evidence$cache_precondition
  )
  fastkpc_full_cuda_phase10_compute_require(
    is.list(evidence$prewarm) &&
      identical(evidence$prewarm$policy,
                "different-DatasetKey-and-noncanonical") &&
      evidence$prewarm$n != 351L && evidence$prewarm$p != 48L &&
      identical(evidence$prewarm$dataset_key,
                evidence$cache_precondition$prewarm_dataset_key) &&
      evidence$prewarm$physical_tests_evaluated > 0L &&
      isTRUE(evidence$prewarm$pass),
    "Phase 10 compute-warm prewarm evidence is malformed"
  )
  fastkpc_full_cuda_phase10_compute_validate_measured_result(
    evidence$result, evidence$cache_precondition,
    formal_canonical = isTRUE(evidence$formal_canonical)
  )
  after <- evidence$cache_after_measurement
  fastkpc_full_cuda_phase10_compute_require(
    identical(after$dataset_key, evidence$cache_precondition$dataset_key) &&
      after$cache_epoch ==
        evidence$cache_precondition$cache_epoch_after_reset &&
      after$result_cache_dataset_entries > 0L &&
      after$target_cache_dataset_entries > 0L &&
      all(fastkpc_full_cuda_phase10_campaign_active_resources(
        evidence$resource_before
      ) == 0) &&
      all(fastkpc_full_cuda_phase10_campaign_active_resources(
        evidence$resource_after
      ) == 0),
    "Phase 10 compute-warm postcondition or resource evidence is invalid"
  )
  if (!is.null(evidence$machine_before) || !is.null(evidence$machine_after)) {
    fastkpc_full_cuda_phase10_compute_require(
      !is.null(evidence$machine_before) && !is.null(evidence$machine_after) &&
        identical(
          fastkpc_full_cuda_phase10_campaign_machine_identity(
            evidence$machine_before
          ),
          fastkpc_full_cuda_phase10_campaign_machine_identity(
            evidence$machine_after
          )
        ),
      "Phase 10 compute-warm machine evidence is invalid"
    )
  }
  invisible(evidence)
}

fastkpc_full_cuda_phase10_write_compute_warm_evidence <- function(
    evidence, path) {
  fastkpc_full_cuda_phase10_validate_compute_warm_evidence(evidence)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(".phase10-compute-warm-", tmpdir = dirname(path))
  on.exit(unlink(temporary, force = TRUE), add = TRUE)
  saveRDS(evidence, temporary, compress = "xz")
  fastkpc_full_cuda_phase10_compute_require(
    file.rename(temporary, path),
    "Phase 10 compute-warm evidence could not be committed atomically"
  )
  invisible(path)
}
