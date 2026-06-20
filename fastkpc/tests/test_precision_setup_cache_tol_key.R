source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

set.seed(6201)
data <- matrix(rnorm(48 * 4), 48, 4)
context <- fastkpc_precision_create_execution_context(
  data = data,
  residual_cache = TRUE,
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
fastkpc_precision_init_cache_stats(context)

sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 5L))

first <- fastkpc_prepare_gpu_setup_state(
  data = data,
  S = 3L,
  template_target = 1L,
  sp_grid = sp_grid,
  context = context,
  tol = sqrt(.Machine$double.eps)
)
second_same_tol <- fastkpc_prepare_gpu_setup_state(
  data = data,
  S = 3L,
  template_target = 2L,
  sp_grid = sp_grid,
  context = context,
  tol = sqrt(.Machine$double.eps)
)
third_different_tol <- fastkpc_prepare_gpu_setup_state(
  data = data,
  S = 3L,
  template_target = 4L,
  sp_grid = sp_grid,
  context = context,
  tol = 1e-6
)

stats <- fastkpc_precision_cache_stats(context, "mgcvExtractGPU")
assert_true(!isTRUE(first$cache_hit), "first setup preparation should miss")
assert_true(isTRUE(second_same_tol$cache_hit),
            "same tol should reuse prepared setup")
assert_true(!isTRUE(third_different_tol$cache_hit),
            "different tol should not reuse prepared setup")
assert_true(stats$setup_cache_misses == 2L,
            "different tol should create a second setup cache miss")
assert_true(stats$setup_cache_hits == 1L,
            "only the same-tol lookup should hit setup cache")

cat("PASS precision setup cache tol key\n")
