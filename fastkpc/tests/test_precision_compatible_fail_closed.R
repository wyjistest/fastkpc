source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(102)
data <- matrix(rnorm(70 * 5), 70, 5)
caps <- list(
  R_version = "unsupported-R",
  mgcv_version = "unsupported-mgcv",
  cuda_available = FALSE,
  mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
  spectral_gcv_version = "single-penalty-spectral-gcv-v1",
  setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
)

result <- fast_kpc(
  data, alpha = 0.2, max_conditioning_size = 1,
  engine = "cpu", precision = "compatible",
  graph_stage = "skeleton",
  runtime_capabilities = caps
)

assert_true(result$config$precision == "compatible",
            "config should record compatible mode")
assert_true(result$config$compatibility_action == "fallback",
            "unsupported compatible mode must fall back")
assert_true(grepl("unsupported", result$config$fallback_reason, fixed = TRUE),
            "fallback reason should be public")
assert_true(result$config$backend_used != "fastSplineCUDA",
            "compatible fallback must not silently use fastSplineCUDA")

cat("PASS precision compatible fail closed\n")
