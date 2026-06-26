source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env,
                     p_value_nonempty = p_value) {
  force(p_value)
  force(p_value_nonempty)
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    list(
      p.value = if (length(S) == 0L) p_value else p_value_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1187)
data <- matrix(rnorm(64 * 3), 64, 3)

direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
verifier_calls <- new.env(parent = emptyenv())
verifier_calls$count <- 0L
legacy_calls <- new.env(parent = emptyenv())
legacy_calls$count <- 0L

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_trace_level = "full",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCPU = make_spy(0.049, "fastSplineCPU-spy", primary_calls),
    mgcvExtractCPUGCVBridge = make_spy(NA_real_, "mgcvExtractCPU-spy",
                                       verifier_calls),
    `legacy-mgcv` = make_spy(0.051, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps
)

trace <- result$diagnostics$precision_trace
fallback_rows <- trace[trace$fallback_triggered, , drop = FALSE]
assert_true(nrow(fallback_rows) > 0L,
            "non-finite verifier should produce fallback trace rows")
assert_true(any(fallback_rows$attempt_count == 3L),
            "attempt_count should include primary, verifier, and legacy fallback")
assert_true("attempt_backend_sequence" %in% names(fallback_rows),
            "trace should expose attempt backend sequence")
assert_true("attempt_status_sequence" %in% names(fallback_rows),
            "trace should expose attempt status sequence")
assert_true(any(grepl("fastSplineCPU-spy>mgcvExtractCPU-spy>legacy-mgcv-spy",
                      fallback_rows$attempt_backend_sequence, fixed = TRUE)),
            "attempt backend sequence should preserve all executed attempts")
assert_true(any(grepl("ok>ok>ok", fallback_rows$attempt_status_sequence,
                      fixed = TRUE)),
            "attempt status sequence should preserve all attempt statuses")

cat("PASS precision attempt ledger\n")
