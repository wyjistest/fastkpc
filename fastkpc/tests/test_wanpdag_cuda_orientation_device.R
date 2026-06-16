source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(142)
n <- 96
z <- seq(-2.5, 2.5, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + 0.05 * cos(13 * z),
  x3 = cos(0.7 * z) + 0.03 * sin(11 * z),
  x4 = z^2 + 0.04 * sin(7 * z)
)

params <- list(knots = 7, lambda_count = 13, ridge = 1e-8)

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

cpu_orientation <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cpu",
  orientation_batch_size = 1L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_orientation <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params
)

assert_true(identical(cuda_orientation$orientation$residual_device, "cuda"),
            "WAN-PDAG CUDA orientation should record cuda residual device")
assert_true(identical(cuda_orientation$orientation$residual_device_requested, "cuda"),
            "WAN-PDAG CUDA orientation should record requested residual device")
assert_true(identical(cuda_orientation$orientation$pdag,
                      cpu_orientation$orientation$pdag),
            "WAN-PDAG CUDA orientation pdag should match CPU orientation")
assert_true(length(cuda_orientation$orientation$events) ==
              length(cpu_orientation$orientation$events),
            "WAN-PDAG CUDA orientation should preserve event count")
for (i in seq_along(cuda_orientation$orientation$events)) {
  cpu_event <- cpu_orientation$orientation$events[[i]]
  cuda_event <- cuda_orientation$orientation$events[[i]]
  assert_true(identical(cuda_event$phase, cpu_event$phase),
              "WAN-PDAG CUDA orientation should preserve event phase order")
  assert_true(identical(cuda_event$rule, cpu_event$rule),
              "WAN-PDAG CUDA orientation should preserve event rule order")
  assert_true(identical(cuda_event$accepted, cpu_event$accepted),
              "WAN-PDAG CUDA orientation should preserve event acceptance")
  assert_true(identical(cuda_event$S, cpu_event$S),
              "WAN-PDAG CUDA orientation should preserve event conditioning sets")
  if (is.finite(cpu_event$p.value) && is.finite(cuda_event$p.value)) {
    assert_true(abs(cuda_event$p.value - cpu_event$p.value) < 1e-7,
                "WAN-PDAG CUDA orientation finite event p-values should match")
  }
}
assert_true(cuda_orientation$orientation$diagnostics$regrvonps_cuda_calls > 0L,
            "WAN-PDAG CUDA orientation should use CUDA regrVonPS calls")
assert_true(cuda_orientation$orientation$diagnostics$orientation_dcov_batches > 0L,
            "WAN-PDAG CUDA orientation should record dCov batches")
assert_true(cuda_orientation$orientation$diagnostics$orientation_cuda_residual_fits > 0L,
            "WAN-PDAG CUDA orientation should record CUDA residual fits")

cuda_one <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 1L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params
)

assert_true(identical(cuda_one$orientation$pdag,
                      cuda_orientation$orientation$pdag),
            "orientation_batch_size=1 pdag should match automatic batching")

cat("test_wanpdag_cuda_orientation_device.R: PASS\n")
