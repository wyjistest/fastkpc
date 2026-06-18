source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(1104)
data <- matrix(rnorm(90 * 4), 90, 4)
caps_cpu_only <- list(
  R_version = "4.5.0",
  mgcv_version = "1.9-4",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data,
  alpha = 1.1,
  max_conditioning_size = 2,
  engine = "cpu",
  precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps_cpu_only
)

trace <- result$diagnostics$precision_trace
two <- trace[grepl("\\|", trace$S_key), , drop = FALSE]
assert_true(nrow(two) > 0L,
            "compatible CPU smoke should execute at least one |S|=2 test")
assert_true(any(two$backend_planned == "mgcvExtractCPUGCVBridge"),
            "compatible |S|=2 should plan CPU mgcvExtract")
assert_true(any(two$backend_executed == "mgcvExtractCPU"),
            "compatible |S|=2 should execute mgcvExtractCPU")
assert_true(any(grepl("\\|", two$S_key)),
            "trace should record joint two-variable S_key")

cat("PASS precision CPU |S|=2 mgcv smoke\n")
