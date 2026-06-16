source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(215)
n <- 54
z <- seq(-2, 2, length.out = n)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.08),
  x2 = z + rnorm(n, sd = 0.08),
  x3 = cos(z) + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)

build_fastkpc_native(rebuild = TRUE)

cpu <- fast_kpc_wanpdag_cpp(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(cpu$skeleton$ci_method == "hsic.gamma",
            "WAN-PDAG CPU skeleton should record HSIC gamma")
assert_true(cpu$orientation$ci_method == "hsic.gamma",
            "WAN-PDAG CPU orientation should record HSIC gamma")

load_fastkpc_cuda_native(rebuild = TRUE)
cuda <- fast_kpc_wanpdag_cuda(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  orientation_residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(cuda$skeleton$ci_method == "hsic.gamma",
            "WAN-PDAG CUDA skeleton should record HSIC gamma")
assert_true(cuda$skeleton$ci_backend == "cuda-hsic",
            "WAN-PDAG CUDA HSIC skeleton should use cuda-hsic")
assert_true(cuda$skeleton$ci_diagnostics$ci_hsic_gamma_cuda_tests > 0,
            "WAN-PDAG CUDA HSIC skeleton should record CUDA HSIC tests")
assert_true(cuda$skeleton$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "WAN-PDAG CUDA HSIC skeleton should record CUDA HSIC batches")
assert_true(identical(cpu$skeleton$adjacency, cuda$skeleton$adjacency),
            "WAN-PDAG CPU and CUDA HSIC skeleton adjacency should match")
assert_true(max_abs_diff(cpu$skeleton$pMax, cuda$skeleton$pMax) < 1e-7,
            "WAN-PDAG CPU and CUDA HSIC skeleton pMax should be close")

perm_a <- fast_kpc_wanpdag_cpp(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 15L, seed = 44L,
                            include_observed = TRUE)
)
perm_b <- fast_kpc_wanpdag_cpp(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 15L, seed = 44L,
                            include_observed = TRUE)
)

assert_true(identical(perm_a$orientation$pdag, perm_b$orientation$pdag),
            "WAN-PDAG HSIC permutation fixed seed pdag should repeat")
assert_true(length(perm_a$orientation$events) == length(perm_b$orientation$events),
            "WAN-PDAG HSIC permutation fixed seed event count should repeat")

cat("test_hsic_wanpdag_pipeline.R: PASS\n")
