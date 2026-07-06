source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

make_spy <- function(backend_name, calls_env) {
  force(backend_name)
  force(calls_env)
  function(data, x, y, S, ci_method, index, legacy_index, hsic_params,
           permutation_params, route, role) {
    calls_env$count <- calls_env$count + 1L
    calls_env$max_s <- max(calls_env$max_s, length(S))
    calls_env$rows[[length(calls_env$rows) + 1L]] <- list(
      x = x, y = y, S = S, role = role, backend = backend_name
    )
    p_value <- 0.001
    if (length(S) == 1L && calls_env$delete_s1 < 1L) {
      p_value <- 0.051
      calls_env$delete_s1 <- calls_env$delete_s1 + 1L
    } else if (length(S) == 2L && calls_env$delete_s2 < 1L) {
      p_value <- 0.051
      calls_env$delete_s2 <- calls_env$delete_s2 + 1L
    }
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

calls <- new.env(parent = emptyenv())
calls$count <- 0L
calls$max_s <- 0L
calls$delete_s1 <- 0L
calls$delete_s2 <- 0L
calls$rows <- list()

executors <- list(
  `direct-ci` = make_spy("direct-ci-spy", calls),
  fastSplineCUDA = make_spy("fastSplineCUDA-spy", calls),
  fastSplineCPU = make_spy("fastSplineCPU-spy", calls),
  mgcvExtractCPUGCVBridge = make_spy("mgcvExtractCPU-spy", calls),
  `legacy-mgcv` = make_spy("legacy-mgcv-spy", calls)
)

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = NA_character_,
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

set.seed(8701)
data <- matrix(stats::rnorm(80 * 6), 80, 6)

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 3,
  engine = "cuda",
  precision = "compatible",
  graph_stage = "skeleton",
  precision_executors = executors,
  runtime_capabilities = caps,
  precision_trace_level = "summary"
)

assert_true(identical(result$skeleton$scheduler, "r-precision"),
            "compatible unbounded skeleton should use R precision route")
assert_true(result$config$precision_execution_status == "data-plane-executed",
            "compatible unbounded skeleton should execute precision data plane")
assert_true(calls$count > 0L,
            "compatible unbounded skeleton should call precision executors")
assert_true(calls$max_s >= 3L,
            "compatible unbounded skeleton should support |S| > 2")

cat("PASS precision compatible unbounded R skeleton route\n")
