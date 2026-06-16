source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(306)
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
load_fastkpc_cuda_native(rebuild = TRUE)

cpu <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE,
  fastspline_params = params,
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1)
)

cuda <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params,
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1)
)

assert_true(cuda$skeleton$ci_backend == "cuda-hsic",
            "WAN-PDAG CUDA HSIC skeleton should use cuda-hsic")
assert_true(cuda$orientation$ci_method == "hsic.gamma",
            "WAN-PDAG CUDA HSIC orientation should record HSIC gamma")
assert_true(cuda$orientation$ci_backend == "cuda-hsic",
            "WAN-PDAG CUDA HSIC orientation should use cuda-hsic")
assert_true(cuda$orientation$ci_backend_reason == "",
            "successful WAN-PDAG CUDA HSIC orientation should not record fallback reason")

diag <- cuda$orientation$ci_diagnostics
assert_true(diag$regrvonps_hsic_gamma_cuda_tests > 0L,
            "WAN-PDAG CUDA HSIC orientation should record CUDA HSIC tests")
assert_true(diag$regrvonps_hsic_cuda_batches > 0L,
            "WAN-PDAG CUDA HSIC orientation should record CUDA HSIC batches")
assert_true(diag$regrvonps_hsic_cuda_pairs >=
              diag$regrvonps_hsic_gamma_cuda_tests,
            "WAN-PDAG CUDA HSIC orientation should record CUDA HSIC pairs")

assert_true(identical(cpu$skeleton$adjacency, cuda$skeleton$adjacency),
            "CPU and CUDA HSIC WAN-PDAG skeleton adjacency should match")
assert_true(identical(cpu$orientation$pdag, cuda$orientation$pdag),
            "CPU and CUDA HSIC WAN-PDAG pdag should match")

perm_a <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params,
  ci_method = "hsic.perm",
  hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, seed = 440L,
                            include_observed = TRUE)
)
perm_b <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.16,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0L,
  orientation_diagnostics = TRUE,
  residual_cache = TRUE,
  fastspline_params = params,
  ci_method = "hsic.perm",
  hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, seed = 440L,
                            include_observed = TRUE)
)

assert_true(perm_a$orientation$ci_backend == "cuda-hsic",
            "fixed-seed WAN-PDAG CUDA HSIC permutation orientation should use cuda-hsic")
assert_true(identical(perm_a$orientation$pdag, perm_b$orientation$pdag),
            "fixed-seed WAN-PDAG CUDA HSIC permutation pdag should repeat")

cat("test_hsic_cuda_wanpdag_orientation.R: PASS\n")
