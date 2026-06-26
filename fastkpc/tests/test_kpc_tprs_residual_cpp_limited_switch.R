source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env, fail_nonempty = FALSE,
                     p_value_nonempty = p_value) {
  force(p_value)
  force(backend_name)
  force(calls_env)
  force(fail_nonempty)
  force(p_value_nonempty)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    if (length(S) > 0L && isTRUE(fail_nonempty)) {
      stop(paste0(backend_name, " injected failure"), call. = FALSE)
    }
    list(
      p.value = if (length(S) == 0L) p_value else p_value_nonempty,
      residual_backend_executed = backend_name,
      ci_backend_executed = "spy-ci",
      setup_fingerprint = paste0(backend_name, ":S:", paste(S, collapse = "|")),
      p_source_used = paste0(role, ":", backend_name, "+spy-ci"),
      timings = list(ci_test_ms = 0)
    )
  }
}

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  kpcTprsResidualCPP_supported = TRUE,
  kpcTprsResidualCPP_backend_version = "kpcTprsResidualCPP-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(62320)
data <- matrix(stats::rnorm(60 * 4), 60, 4)

kpc_calls <- new.env(parent = emptyenv())
kpc_calls$count <- 0L
fallback_calls <- new.env(parent = emptyenv())
fallback_calls$count <- 0L
result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_trace_level = "full",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy",
                           new.env(parent = emptyenv())),
    kpcTprsResidualCPP = make_spy(0.001, "kpcTprsResidualCPP", kpc_calls,
                                  p_value_nonempty = 0.001),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                       fallback_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy",
                             new.env(parent = emptyenv()))
  )
)

trace <- result$diagnostics$precision_trace
conditional <- trace[nzchar(trace$S_key), , drop = FALSE]
assert_true(nrow(conditional) > 0L, "limited switch test needs conditional rows")
assert_true(all(conditional$backend_requested == "kpcTprsResidualCPP"),
            "compatible route should request kpcTprsResidualCPP when opted in")
assert_true(all(conditional$primary_residual_backend_executed ==
                  "kpcTprsResidualCPP"),
            "conditional rows should execute kpcTprsResidualCPP")
assert_true(kpc_calls$count > 0L, "kpcTprsResidualCPP executor should run")
assert_true(fallback_calls$count == 0L,
            "mgcv fallback should not run when kpcTprsResidualCPP succeeds")

kpc_fail_calls <- new.env(parent = emptyenv())
kpc_fail_calls$count <- 0L
fallback_calls$count <- 0L
fallback <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_trace_level = "full",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy",
                           new.env(parent = emptyenv())),
    kpcTprsResidualCPP = make_spy(0.001, "kpcTprsResidualCPP", kpc_fail_calls,
                                  fail_nonempty = TRUE),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                       fallback_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy",
                             new.env(parent = emptyenv()))
  )
)
fallback_trace <- fallback$diagnostics$precision_trace
fallback_conditional <- fallback_trace[nzchar(fallback_trace$S_key), ,
                                       drop = FALSE]
assert_true(any(fallback_conditional$fallback_triggered),
            "candidate failure should trigger fallback")
assert_true(any(grepl("kpcTprsResidualCPP injected failure",
                      fallback_conditional$fallback_reason, fixed = TRUE)),
            "fallback reason should record candidate failure")
assert_true(any(fallback_conditional$primary_residual_backend_executed ==
                  "mgcvExtractCPU-spy"),
            "mgcv fallback should remain available")

cat("PASS kpcTprsResidualCPP limited switch\n")
