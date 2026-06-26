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
      setup_fingerprint = paste0("spy:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

caps_canary <- list(
  R_version = "4.4.1",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

warnings_seen <- character()
withCallingHandlers(
  result <- fast_kpc(
    matrix(rnorm(50 * 3), 50, 3),
    alpha = 0.05,
    max_conditioning_size = 1,
    engine = "cuda",
    precision = "compatible",
    graph_stage = "skeleton",
    allow_canary_mgcv_extract = TRUE,
    runtime_capabilities = caps_canary,
    precision_trace_level = "full",
    precision_executors = list(
      `direct-ci` = make_spy(0.001, "direct-ci-spy"),
      mgcvExtractGPUGCV = make_spy(0.001, "mgcvExtractGPU-spy"),
      mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy"),
      `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy")
    )
  ),
  warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)

assert_true(length(warnings_seen) == 1L,
            "canary precision route should warn exactly once per fast_kpc run")
assert_true(grepl("mgcvExtractGPU compatibility envelope mismatch",
                  warnings_seen[[1L]], fixed = TRUE),
            "canary warning should expose compatibility envelope mismatch")
assert_true(grepl("action warn-and-run", warnings_seen[[1L]], fixed = TRUE),
            "canary warning should expose warn-and-run action")
assert_true(result$config$compatibility_status == "canary",
            "config should preserve canary compatibility status")
assert_true(result$config$compatibility_action == "warn-and-run",
            "config should preserve canary compatibility action")
assert_true(grepl("warn-and-run", result$config$fallback_reason, fixed = TRUE),
            "config should preserve canary warning reason")

trace <- result$diagnostics$precision_trace
assert_true(any(grepl("warn-and-run", trace$fallback_reason, fixed = TRUE)),
            "trace should preserve canary warning reason")

cat("PASS precision canary warning\n")
