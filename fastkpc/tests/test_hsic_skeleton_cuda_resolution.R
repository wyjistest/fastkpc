source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(214)
n <- 48
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.08),
  x2 = cos(z) + rnorm(n, sd = 0.08),
  x3 = z^2 + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)

build_fastkpc_native(rebuild = TRUE)
load_fastkpc_cuda_native(rebuild = TRUE)

cpu <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

cuda <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(cuda$ci_method == "hsic.gamma",
            "CUDA skeleton should record requested HSIC gamma method")
assert_true(cuda$ci_backend == "cuda-hsic",
            "CUDA HSIC gamma should use real cuda-hsic backend")
assert_true(cuda$ci_backend_reason == "",
            "successful CUDA HSIC gamma should not record CPU fallback reason")
assert_true(cuda$ci_diagnostics$ci_hsic_gamma_cuda_tests > 0,
            "CUDA HSIC gamma should record CUDA test count")
assert_true(cuda$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "CUDA HSIC gamma should record CUDA batch count")
assert_true(identical(cpu$adjacency, cuda$adjacency),
            "CPU and CUDA HSIC gamma adjacency should match")
assert_true(max_abs_diff(cpu$pMax, cuda$pMax) < 1e-7,
            "CPU and CUDA HSIC gamma pMax should be close")

perm_a <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, seed = 303L,
                            include_observed = TRUE)
)
perm_b <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, seed = 303L,
                            include_observed = TRUE)
)

assert_true(perm_a$ci_method == "hsic.perm",
            "CUDA skeleton should record requested HSIC permutation method")
assert_true(perm_a$ci_backend == "cuda-hsic",
            "fixed-seed CUDA HSIC permutation should use cuda-hsic backend")
assert_true(perm_a$ci_diagnostics$ci_hsic_perm_cuda_tests > 0,
            "fixed-seed CUDA HSIC permutation should record CUDA test count")
assert_true(max_abs_diff(perm_a$pMax, perm_b$pMax) < 1e-12,
            "CUDA HSIC permutation fixed seed should repeat")

perm_fallback <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, include_observed = TRUE)
)

assert_true(perm_fallback$ci_backend == "native-cpu",
            "seedless CUDA HSIC permutation should fall back to native CPU")
assert_true(perm_fallback$ci_backend_reason ==
              "CUDA HSIC permutation requires explicit seed in this stage",
            "seedless CUDA HSIC permutation should record fallback reason")
assert_true(perm_fallback$ci_diagnostics$ci_hsic_cuda_fallback_tests > 0,
            "seedless CUDA HSIC permutation should record fallback tests")

cat("test_hsic_skeleton_cuda_resolution.R: PASS\n")
