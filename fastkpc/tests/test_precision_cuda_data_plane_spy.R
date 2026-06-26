source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(p_value, backend_name, calls_env,
                     p_value_nonempty = p_value, fail_nonempty = FALSE) {
  force(p_value)
  force(p_value_nonempty)
  force(backend_name)
  force(calls_env)
  force(fail_nonempty)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
    if (isTRUE(fail_nonempty) && length(S) > 0L) {
      stop(paste("forced", backend_name, "failure"), call. = FALSE)
    }
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
  cuda_available = TRUE,
  cuda_device_capability = "8.9",
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(1301)
data <- matrix(rnorm(56 * 3), 56, 3)

direct_calls <- new.env(parent = emptyenv())
direct_calls$count <- 0L
direct_calls$rows <- list()
gpu_calls <- new.env(parent = emptyenv())
gpu_calls$count <- 0L
gpu_calls$rows <- list()
cpu_calls <- new.env(parent = emptyenv())
cpu_calls$count <- 0L
cpu_calls$rows <- list()
legacy_calls <- new.env(parent = emptyenv())
legacy_calls$count <- 0L
legacy_calls$rows <- list()

compatible <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    mgcvExtractGPUGCV = make_spy(0.001, "mgcvExtractGPU-spy",
                                 gpu_calls, p_value_nonempty = 0.051),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                       cpu_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

assert_true(gpu_calls$count > 0L,
            "CUDA compatible precision should execute mgcvExtractGPUGCV")
assert_true(cpu_calls$count == 0L,
            "CUDA compatible precision should not call CPU bridge when GPU succeeds")
assert_true(legacy_calls$count == 0L,
            "CUDA compatible precision should not call legacy when GPU succeeds")
assert_true(compatible$config$precision_execution_status == "data-plane-executed",
            "CUDA compatible precision should mark data-plane execution")
assert_true(compatible$config$backend_planned == "mgcvExtractGPUGCV",
            "CUDA compatible route should plan mgcvExtractGPUGCV")
assert_true(compatible$config$backend_executed == "mgcvExtractGPU-spy",
            "backend_executed should come from GPU executor receipt")
assert_true(!isTRUE(compatible$skeleton$adjacency[1, 2]),
            "compatible GPU p=0.051 should delete edge at alpha=0.05")

trace <- compatible$diagnostics$precision_trace
assert_true(any(trace$backend_planned == "mgcvExtractGPUGCV" &
                  trace$backend_executed == "mgcvExtractGPU-spy" &
                  trace$p_used == 0.051),
            "trace should record GPU planned/executed backend and p-value")

direct_calls$count <- 0L
direct_calls$rows <- list()
gpu_calls$count <- 0L
gpu_calls$rows <- list()
cpu_calls$count <- 0L
cpu_calls$rows <- list()
legacy_calls$count <- 0L
legacy_calls$rows <- list()

primary_calls <- new.env(parent = emptyenv())
primary_calls$count <- 0L
primary_calls$rows <- list()

hybrid <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "hybrid",
  tau = log(2),
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    fastSplineCUDA = make_spy(0.001, "fastSplineCUDA-spy",
                              primary_calls, p_value_nonempty = 0.049),
    mgcvExtractGPUGCV = make_spy(0.001, "mgcvExtractGPU-spy",
                                 gpu_calls, p_value_nonempty = 0.051),
    mgcvExtractCPUGCVBridge = make_spy(0.9, "mgcvExtractCPU-spy",
                                       cpu_calls),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

assert_true(primary_calls$count > 0L,
            "CUDA hybrid precision should execute fastSplineCUDA primary")
assert_true(gpu_calls$count > 0L,
            "CUDA hybrid near-alpha should execute mgcvExtractGPUGCV verifier")
assert_true(cpu_calls$count == 0L && legacy_calls$count == 0L,
            "CUDA hybrid should not fall back when GPU verifier succeeds")
verified <- hybrid$diagnostics$precision_trace[
  hybrid$diagnostics$precision_trace$near_alpha_triggered,
  , drop = FALSE
]
assert_true(any(verified$backend_planned == "fastSplineCUDA" &
                  verified$verifier_planned == "mgcvExtractGPUGCV" &
                  verified$verifier_executed == "mgcvExtractGPU-spy" &
                  verified$primary_p_raw == 0.049 &
                  verified$verifier_p_raw == 0.051 &
                  verified$p_used == 0.051 &
                  verified$decision_before_verify == FALSE &
                  verified$decision_after_verify == TRUE),
            "CUDA hybrid trace should show GPU verifier changing the decision")

direct_calls$count <- 0L
direct_calls$rows <- list()
gpu_calls$count <- 0L
gpu_calls$rows <- list()
cpu_calls$count <- 0L
cpu_calls$rows <- list()
legacy_calls$count <- 0L
legacy_calls$rows <- list()

fallback <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = list(
    `direct-ci` = make_spy(0.001, "direct-ci-spy", direct_calls),
    mgcvExtractGPUGCV = make_spy(0.001, "mgcvExtractGPU-spy",
                                 gpu_calls, fail_nonempty = TRUE),
    mgcvExtractCPUGCVBridge = make_spy(0.001, "mgcvExtractCPU-spy",
                                       cpu_calls, p_value_nonempty = 0.051),
    `legacy-mgcv` = make_spy(0.9, "legacy-mgcv-spy", legacy_calls)
  ),
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

trace <- fallback$diagnostics$precision_trace
fallback_rows <- trace[trace$fallback_triggered, , drop = FALSE]
assert_true(gpu_calls$count > 0L && cpu_calls$count > 0L,
            "GPU failure should fall back to CPU bridge")
assert_true(legacy_calls$count == 0L,
            "legacy should not run when CPU bridge fallback succeeds")
assert_true(any(fallback_rows$backend_executed == "mgcvExtractCPU-spy" &
                  fallback_rows$p_used == 0.051),
            "fallback decision should use CPU bridge p-value")
assert_true(any(grepl("mgcvExtractGPUGCV>mgcvExtractCPU-spy",
                      fallback_rows$attempt_backend_sequence, fixed = TRUE)),
            "attempt ledger should preserve GPU then CPU bridge attempts")

cat("PASS precision CUDA data plane spy\n")
