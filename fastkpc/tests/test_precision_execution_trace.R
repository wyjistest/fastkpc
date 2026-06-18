source("fastkpc/R/precision_execution_trace.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

row <- fastkpc_precision_trace_row(
  run_id = "run-1",
  scenario_id = "unit",
  dataset_hash = "hash",
  conditioning_level = 1L,
  canonical_test_order_id = 3L,
  setup_fingerprint = "setup",
  target_id = "x1",
  backend_requested = "mgcvExtractGPUGCV",
  backend_used = "legacy-mgcv",
  verifier_backend = "legacy-mgcv",
  compatibility_action = "fallback",
  fallback_reason = "unsupported mgcv version",
  p_source_used = "legacy-mgcv",
  mgcv_setup_cpu_ms = 1,
  linear_solve_ms = 2,
  total_ms = 5
)

required <- c(
  "run_id", "scenario_id", "dataset_hash", "conditioning_level",
  "canonical_test_order_id", "setup_fingerprint", "target_id",
  "x", "y", "S_key", "conditioning_target_side",
  "backend_requested", "backend_used", "verifier_backend",
  "compatibility_action", "fallback_reason", "CUDA_device",
  "git_sha", "primary_p", "verifier_p", "p_used", "p_raw",
  "p_was_nonfinite", "nonfinite_action", "p_source_used",
  "primary_residual_backend_executed", "primary_ci_backend_executed",
  "primary_p_raw", "primary_p_used", "near_alpha_triggered",
  "verifier_residual_backend_executed", "verifier_ci_backend_executed",
  "verifier_p_raw", "verifier_p_used", "fallback_triggered",
  "attempt_count", "attempt_backend_sequence", "attempt_status_sequence",
  "ci_randomness_id", "permutation_seed_effective",
  "permutation_plan_spec_hash", "permutation_plan_hash",
  "permutation_replicates",
  "decision_before_verify", "decision_after_verify",
  "mgcv_setup_cpu_ms", "setup_cache_lookup_ms", "host_to_device_ms",
  "spectral_prepare_ms", "gcv_score_ms", "linear_solve_ms",
  "residual_materialize_ms", "device_to_host_ms", "ci_test_ms",
  "canonical_replay_ms", "total_ms"
)
assert_true(all(required %in% names(row)), "trace row missing required fields")
assert_true(row$total_ms >= 5, "trace should preserve total timing")

cat("PASS precision execution trace\n")
