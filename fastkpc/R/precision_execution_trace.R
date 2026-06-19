fastkpc_trace_git_sha <- function() {
  sha <- tryCatch(
    system2("git", c("rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
    error = function(e) NA_character_
  )
  if (length(sha) == 0L) NA_character_ else sha[1L]
}

fastkpc_precision_trace_row <- function(
  run_id,
  scenario_id = NA_character_,
  dataset_hash = NA_character_,
  conditioning_level = NA_integer_,
  canonical_test_order_id = NA_integer_,
  setup_fingerprint = NA_character_,
  target_id = NA_character_,
  x = NA_integer_,
  y = NA_integer_,
  S_key = "",
  conditioning_target_side = NA_character_,
  backend_requested,
  backend_used,
  backend_planned = backend_requested,
  backend_executed = backend_used,
  verifier_backend = NA_character_,
  verifier_planned = verifier_backend,
  verifier_executed = NA_character_,
  compatibility_action = NA_character_,
  fallback_reason = "",
  CUDA_device = NA_character_,
  git_sha = fastkpc_trace_git_sha(),
  primary_p = NA_real_,
  verifier_p = NA_real_,
  p_used = NA_real_,
  p_raw = NA_real_,
  p_was_nonfinite = FALSE,
  nonfinite_action = "",
  p_source_used = NA_character_,
  primary_residual_backend_executed = NA_character_,
  primary_ci_backend_executed = NA_character_,
  primary_p_raw = NA_real_,
  primary_p_used = NA_real_,
  near_alpha_triggered = FALSE,
  verifier_residual_backend_executed = NA_character_,
  verifier_ci_backend_executed = NA_character_,
  verifier_p_raw = NA_real_,
  verifier_p_used = NA_real_,
  cache_hit = FALSE,
  fallback_triggered = FALSE,
  attempt_count = NA_integer_,
  attempt_backend_sequence = "",
  attempt_status_sequence = "",
  ci_randomness_id = "",
  permutation_seed_effective = NA_integer_,
  permutation_plan_spec_hash = "",
  permutation_plan_hash = "",
  permutation_replicates = NA_integer_,
  precision_execution_status = "control-plane-only",
  decision_before_verify = NA,
  decision_after_verify = NA,
  mgcv_setup_cpu_ms = NA_real_,
  setup_cache_lookup_ms = NA_real_,
  host_to_device_ms = NA_real_,
  spectral_prepare_ms = NA_real_,
  gcv_score_ms = NA_real_,
  linear_solve_ms = NA_real_,
  residual_materialize_ms = NA_real_,
  device_to_host_ms = NA_real_,
  ci_test_ms = NA_real_,
  canonical_replay_ms = NA_real_,
  total_ms = NA_real_
) {
  data.frame(
    run_id = as.character(run_id),
    scenario_id = as.character(scenario_id),
    dataset_hash = as.character(dataset_hash),
    conditioning_level = as.integer(conditioning_level),
    canonical_test_order_id = as.integer(canonical_test_order_id),
    setup_fingerprint = as.character(setup_fingerprint),
    target_id = as.character(target_id),
    x = as.integer(x),
    y = as.integer(y),
    S_key = as.character(S_key),
    conditioning_target_side = as.character(conditioning_target_side),
    backend_requested = as.character(backend_requested),
    backend_used = as.character(backend_used),
    backend_planned = as.character(backend_planned),
    backend_executed = as.character(backend_executed),
    verifier_backend = as.character(verifier_backend),
    verifier_planned = as.character(verifier_planned),
    verifier_executed = as.character(verifier_executed),
    compatibility_action = as.character(compatibility_action),
    fallback_reason = as.character(fallback_reason),
    precision_execution_status = as.character(precision_execution_status),
    CUDA_device = as.character(CUDA_device),
    git_sha = as.character(git_sha),
    primary_p = as.numeric(primary_p),
    verifier_p = as.numeric(verifier_p),
    p_used = as.numeric(p_used),
    p_raw = as.numeric(p_raw),
    p_was_nonfinite = as.logical(p_was_nonfinite),
    nonfinite_action = as.character(nonfinite_action),
    p_source_used = as.character(p_source_used),
    primary_residual_backend_executed =
      as.character(primary_residual_backend_executed),
    primary_ci_backend_executed = as.character(primary_ci_backend_executed),
    primary_p_raw = as.numeric(primary_p_raw),
    primary_p_used = as.numeric(primary_p_used),
    near_alpha_triggered = as.logical(near_alpha_triggered),
    verifier_residual_backend_executed =
      as.character(verifier_residual_backend_executed),
    verifier_ci_backend_executed = as.character(verifier_ci_backend_executed),
    verifier_p_raw = as.numeric(verifier_p_raw),
    verifier_p_used = as.numeric(verifier_p_used),
    cache_hit = as.logical(cache_hit),
    fallback_triggered = as.logical(fallback_triggered),
    attempt_count = as.integer(attempt_count),
    attempt_backend_sequence = as.character(attempt_backend_sequence),
    attempt_status_sequence = as.character(attempt_status_sequence),
    ci_randomness_id = as.character(ci_randomness_id),
    permutation_seed_effective = as.integer(permutation_seed_effective),
    permutation_plan_spec_hash = as.character(permutation_plan_spec_hash),
    permutation_plan_hash = as.character(permutation_plan_hash),
    permutation_replicates = as.integer(permutation_replicates),
    decision_before_verify = as.logical(decision_before_verify),
    decision_after_verify = as.logical(decision_after_verify),
    mgcv_setup_cpu_ms = as.numeric(mgcv_setup_cpu_ms),
    setup_cache_lookup_ms = as.numeric(setup_cache_lookup_ms),
    host_to_device_ms = as.numeric(host_to_device_ms),
    spectral_prepare_ms = as.numeric(spectral_prepare_ms),
    gcv_score_ms = as.numeric(gcv_score_ms),
    linear_solve_ms = as.numeric(linear_solve_ms),
    residual_materialize_ms = as.numeric(residual_materialize_ms),
    device_to_host_ms = as.numeric(device_to_host_ms),
    ci_test_ms = as.numeric(ci_test_ms),
    canonical_replay_ms = as.numeric(canonical_replay_ms),
    total_ms = as.numeric(total_ms),
    stringsAsFactors = FALSE
  )
}

fastkpc_precision_trace_from_result <- function(result, route, run_id,
                                                scenario_id = "fastkpc-run",
                                                elapsed_total_sec = NA_real_) {
  tests <- result$skeleton$scheduler_diagnostics$summary$tests_replayed %||% 1L
  if (!is.finite(tests) || tests <= 0L) tests <- 1L
  ids <- seq_len(as.integer(tests))
  do.call(rbind, lapply(ids, function(id) {
    fastkpc_precision_trace_row(
      run_id = run_id,
      scenario_id = scenario_id,
      dataset_hash = result$data_info$data_hash %||% NA_character_,
      conditioning_level = NA_integer_,
      canonical_test_order_id = id,
      setup_fingerprint = route$setup_fingerprint %||% NA_character_,
      target_id = NA_character_,
      backend_requested = route$primary_backend,
      backend_used = result$config$backend_used %||%
        result$config$backend_executed %||% route$primary_backend,
      backend_planned = result$config$backend_planned %||% route$primary_backend,
      backend_executed = result$config$backend_executed %||%
        result$config$backend_used %||% route$primary_backend,
      verifier_backend = result$config$verifier_planned %||%
        route$verifier_backend %||% NA_character_,
      verifier_planned = result$config$verifier_planned %||%
        route$verifier_backend %||% NA_character_,
      verifier_executed = result$config$verifier_executed %||% NA_character_,
      compatibility_action = route$compatibility_action %||% "",
      fallback_reason = route$fallback_reason %||% "",
      p_source_used = "not-recorded",
      precision_execution_status = result$config$precision_execution_status %||%
        "control-plane-only",
      total_ms = if (id == 1L) as.numeric(elapsed_total_sec) * 1000 else NA_real_
    )
  }))
}

fastkpc_write_precision_trace <- function(
  trace,
  output_dir = file.path("fastkpc", "artifacts", "precision_trace")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  csv_path <- file.path(output_dir, "precision_execution_trace.csv")
  utils::write.csv(trace, csv_path, row.names = FALSE, na = "")
  list(csv_path = csv_path, trace = trace)
}
