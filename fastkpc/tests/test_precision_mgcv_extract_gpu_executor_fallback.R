source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!requireNamespace("mgcv", quietly = TRUE)) {
  cat("SKIP precision mgcvExtractGPU executor fallback: mgcv unavailable\n")
  quit(save = "no", status = 0)
}

fastkpc_cuda_available <- function() FALSE

set.seed(1302)
n <- 48
s1 <- stats::runif(n, -2, 2)
x <- sin(s1) + stats::rnorm(n, sd = 0.05)
y <- cos(s1) + stats::rnorm(n, sd = 0.05)
z <- s1
data <- cbind(x, y, z)

route <- fastkpc_precision_group_route(
  precision = "compatible",
  alpha = 0.05,
  tau = log(2),
  S = 3L,
  runtime_capabilities = list(
    R_version = "4.5.0",
    mgcv_version = "1.9-4",
    cuda_available = TRUE,
    cuda_device_capability = "8.9",
    mgcvExtractGPU_backend_version = "mgcvExtractGPU-v1",
    spectral_gcv_version = "single-penalty-spectral-gcv-v1",
    setup_fingerprint_schema_version = "mgcvExtractGPU-setup-v1"
  ),
  execution_engine = "cuda"
)
assert_true(route$primary_backend == "mgcvExtractGPUGCV",
            "test requires GPU compatible route")

direct_gpu <- tryCatch(
  fastkpc_execute_ci_mgcv_extract_gpu(
    data = data, x = 1L, y = 2L, S = 3L,
    ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
    hsic_params = list(), permutation_params = list(),
    route = route, role = "primary"
  ),
  error = function(e) e
)
assert_true(inherits(direct_gpu, "error"),
            "mgcvExtractGPUGCV executor should not silently CPU-fallback")

resolved <- fastkpc_precision_resolve_test(
  data = data, x = 1L, y = 2L, S = 3L, route = route,
  precision = "compatible", alpha = 0.05, tau = log(2),
  ci_method = "dcc.gamma", index = 1, legacy_index = TRUE,
  hsic_params = list(), permutation_params = list(),
  precision_executors = fastkpc_default_precision_executors(),
  canonical_test_order_id = 1L,
  execution_engine = "cuda"
)

assert_true(resolved$fallback_triggered,
            "GPU executor failure should trigger explicit fallback")
assert_true(resolved$receipt$residual_backend_executed == "mgcvExtractCPU",
            "fallback should execute CPU mgcvExtract bridge")
assert_true(grepl("mgcvExtractGPUGCV>mgcvExtractCPU",
                  resolved$attempt_backend_sequence, fixed = TRUE),
            "attempt ledger should record GPU attempt then CPU bridge")
assert_true(is.finite(resolved$pval),
            "fallback should produce a finite p-value")

cat("PASS precision mgcvExtractGPU executor fallback\n")
