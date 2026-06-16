source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(141)
n <- 96
z <- seq(-2.5, 2.5, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + rnorm(n, sd = 0.08),
  x3 = cos(0.5 * z) + rnorm(n, sd = 0.08),
  x4 = z^2 + rnorm(n, sd = 0.08)
)

params <- list(knots = 7, lambda_count = 13, ridge = 1e-8)

cpu_orientation <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cpu",
  orientation_batch_size = 1L,
  orientation_diagnostics = TRUE,
  graph_stage = "wanpdag",
  fastspline_params = params,
  seed = 141
)

cuda_orientation <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  graph_stage = "wanpdag",
  fastspline_params = params,
  seed = 141
)

assert_true(inherits(cuda_orientation, "fastkpc_result"),
            "orientation device call should return fastkpc_result")
assert_true(cuda_orientation$config$orientation_residual_device_requested == "cuda",
            "config should record requested orientation residual device")
assert_true(cuda_orientation$config$orientation_residual_device_used == "cuda",
            "config should record used orientation residual device")
assert_true(cuda_orientation$config$orientation_batch_size == 0L,
            "config should record requested orientation batch size")
assert_true(isTRUE(cuda_orientation$config$orientation_diagnostics),
            "config should record orientation diagnostics flag")
assert_true(cuda_orientation$orientation$residual_device == "cuda",
            "orientation result should record cuda residual device")
assert_true(cuda_orientation$orientation$residual_device_requested == "cuda",
            "orientation result should record requested residual device")
assert_true(is.character(cuda_orientation$orientation$residual_device_reason),
            "orientation result should record residual device reason")
assert_true(!is.null(cuda_orientation$orientation$diagnostics),
            "orientation diagnostics should be present")
assert_true(cuda_orientation$orientation$diagnostics$regrvonps_cuda_calls > 0L,
            "CUDA orientation diagnostics should record CUDA regrVonPS calls")
assert_true(cuda_orientation$orientation$diagnostics$orientation_dcov_pairs > 0L,
            "CUDA orientation diagnostics should record dCov pairs")
assert_true(identical(cuda_orientation$orientation$pdag,
                      cpu_orientation$orientation$pdag),
            "CUDA orientation pdag should match CPU orientation pdag")

fallback <- fast_kpc(
  data,
  alpha = 0.18,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "linear",
  residual_device = "cpu",
  orientation_residual_device = "cuda",
  orientation_diagnostics = TRUE,
  graph_stage = "wanpdag",
  seed = 141
)

assert_true(fallback$config$orientation_residual_device_used == "cpu",
            "linear CUDA orientation request should resolve to CPU")
assert_true(grepl("linear orientation residual CUDA device is not implemented",
                  fallback$config$orientation_residual_device_reason,
                  fixed = TRUE),
            "linear orientation fallback should explain why CPU was used")

cat("test_fastkpc_orientation_device_public_api.R: PASS\n")
