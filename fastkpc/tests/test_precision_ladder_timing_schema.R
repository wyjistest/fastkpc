fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/precision_ladder_timing.R")

row <- fastkpc_precision_ladder_timing_row(
  backend = "mgcvExtractGPUFixedSP",
  mode = "fixed-sp",
  solve_source = "cuda-fixed-sp",
  native_gpu_solve_used = TRUE,
  true_batched_kernel = FALSE,
  targets_per_setup = 4L,
  mgcv_setup_cpu_ms = 3,
  setup_cache_lookup_ms = 1,
  host_to_device_ms = 2,
  linear_solve_ms = 10,
  residual_materialize_ms = 2,
  device_to_host_ms = 1,
  ci_test_ms = 5,
  canonical_replay_ms = 1
)

required <- c(
  "backend", "mode", "solve_source", "native_gpu_solve_used",
  "true_batched_kernel", "targets_per_setup", "setup_reuse_count",
  "mgcv_setup_cpu_ms", "setup_cache_lookup_ms", "setup_cache_hit",
  "host_to_device_ms", "spectral_prepare_ms", "gcv_score_ms",
  "linear_solve_ms", "residual_materialize_ms", "device_to_host_ms",
  "ci_test_ms", "canonical_replay_ms", "total_ms",
  "gcv_grid_points", "gcv_grid_boundary_hit", "condition_estimate",
  "fallback_reason", "setup_fingerprint", "target_fingerprint",
  "timing_accounting_note"
)
missing <- setdiff(required, names(row))
assert_true(length(missing) == 0L,
            paste("missing timing fields:", paste(missing, collapse = ", ")))
assert_true(row$total_ms >= 25, "total_ms should include known component time")
assert_true(row$true_batched_kernel == FALSE,
            "same-setup bridge should not claim true batched kernel")
assert_true(row$targets_per_setup == 4L, "targets_per_setup should be preserved")
assert_true(identical(
  fastkpc_classify_timing_bottleneck(row),
  "linear_solve_dominated"
), "linear solve should be classified as bottleneck")

cat("PASS precision ladder timing schema\n")
