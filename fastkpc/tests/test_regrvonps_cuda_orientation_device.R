source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(143)
n <- 120
z <- seq(-pi, pi, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + 0.04 * cos(17 * z),
  x3 = cos(z) + 0.04 * sin(19 * z),
  x4 = sin(2 * z) + z / 3
)

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

cpu_orientation <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.14,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cpu",
  orientation_batch_size = 1L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE
)

cuda_orientation <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.14,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 1L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE
)

assert_true(identical(cuda_orientation$orientation$pdag,
                      cpu_orientation$orientation$pdag),
            "CUDA regrVonPS orientation should preserve pdag")
assert_true(identical(cuda_orientation$orientation$counts,
                      cpu_orientation$orientation$counts),
            "CUDA regrVonPS orientation should preserve orientation counts")

diag <- cuda_orientation$orientation$diagnostics
assert_true(diag$regrvonps_calls > 0L,
            "fixture should exercise generalized regrVonPS calls")
assert_true(diag$regrvonps_calls == diag$regrvonps_cuda_calls,
            "CUDA orientation should account for every regrVonPS call as CUDA")
assert_true(diag$regrvonps_cpu_calls == 0L,
            "CUDA orientation should avoid CPU regrVonPS calls on fastSpline CUDA path")
assert_true(diag$orientation_dcov_pairs >= diag$regrvonps_calls,
            "CUDA orientation should record dCov pair work")

cat("test_regrvonps_cuda_orientation_device.R: PASS\n")
