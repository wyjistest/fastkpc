source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP precision compatible CPU data plane smoke: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

set.seed(994)
n <- 45
s <- seq(-1, 1, length.out = n)
data <- cbind(
  x = sin(2 * s) + rnorm(n, sd = 0.05),
  y = cos(2 * s) + rnorm(n, sd = 0.05),
  z = s + rnorm(n, sd = 0.02)
)

caps <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = TRUE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data,
  alpha = 0.05,
  max_conditioning_size = 1,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps,
  precision_trace_level = "full"
)

assert_true(result$config$backend_planned == "mgcvExtractCPUGCVBridge",
            "supported compatible route should plan mgcvExtractCPUGCVBridge")
assert_true(result$config$backend_executed == "mgcvExtractCPU",
            "default compatible CPU vertical slice should execute mgcvExtractCPU")
assert_true(result$config$precision_execution_status == "data-plane-executed",
            "compatible CPU vertical slice should disclose data-plane execution")

trace <- result$diagnostics$precision_trace
assert_true(nrow(trace) > 0L, "compatible CPU trace should contain rows")
assert_true(any(trace$backend_executed == "mgcvExtractCPU"),
            "trace should contain mgcvExtractCPU receipt rows")
assert_true(any(is.finite(trace$p_used)),
            "trace should contain real p_used values")
assert_true(!any(trace$p_source_used == "not-recorded"),
            "data-plane trace should not use not-recorded p sources")

cat("PASS precision compatible CPU data plane smoke\n")
