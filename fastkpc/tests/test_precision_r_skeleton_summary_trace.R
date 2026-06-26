source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name) {
  force(p_value)
  force(backend_name)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    list(
      p.value = p_value,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0(backend_name, ":S:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0, total_ms = 0)
    )
  }
}

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)
executors <- list(
  `direct-ci` = make_spy(0.001, "direct-ci-spy"),
  fastSplineCPU = make_spy(0.001, "fastSplineCPU-spy"),
  mgcvExtractCPUGCVBridge = make_spy(0.001, "mgcvExtractCPU-spy"),
  `legacy-mgcv` = make_spy(0.001, "legacy-mgcv-spy")
)

set.seed(62510)
data <- matrix(stats::rnorm(50 * 4), 50, 4)

summary_result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_executors = executors
)
summary_diag <- summary_result$skeleton$scheduler_diagnostics$summary
assert_true(identical(summary_diag$trace_level, "summary"),
            "R precision skeleton should default auto trace to summary")
assert_true(summary_diag$trace_append_ms == 0,
            "summary trace should not construct per-test trace rows")
assert_true(!is.data.frame(summary_result$skeleton$precision_trace),
            "summary skeleton should not materialize per-test trace rows")
assert_true(!is.data.frame(summary_result$diagnostics$precision_trace),
            "summary diagnostics should not materialize fallback trace rows")

full_result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_executors = executors,
  precision_trace_level = "full"
)
full_diag <- full_result$skeleton$scheduler_diagnostics$summary
assert_true(identical(full_diag$trace_level, "full"),
            "full trace should remain opt-in")
assert_true(nrow(full_result$diagnostics$precision_trace) > 0L,
            "full trace should materialize per-test rows")

cat("PASS precision R skeleton summary trace\n")
