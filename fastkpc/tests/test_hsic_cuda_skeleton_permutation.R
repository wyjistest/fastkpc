source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(305)
n <- 58
z <- runif(n, -1.5, 1.5)
data <- cbind(
  x1 = z + rnorm(n, sd = 0.06),
  x2 = z^2 + rnorm(n, sd = 0.06),
  x3 = sin(z) + rnorm(n, sd = 0.06),
  x4 = rnorm(n)
)

load_fastkpc_cuda_native(rebuild = TRUE)
a <- fast_skeleton_cuda_backend(
  data, alpha = 0.25, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 4,
  scheduler = "legacy",
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 24L, seed = 900L,
                            include_observed = TRUE)
)
b <- fast_skeleton_cuda_backend(
  data, alpha = 0.25, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 4,
  scheduler = "legacy",
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 24L, seed = 900L,
                            include_observed = TRUE)
)

assert_true(a$ci_method == "hsic.perm",
            "CUDA skeleton should record HSIC permutation")
assert_true(a$ci_backend == "cuda-hsic",
            "fixed-seed CUDA HSIC permutation should use cuda-hsic backend")
assert_true(a$ci_backend_reason == "",
            "successful CUDA HSIC permutation should not record fallback reason")
assert_true(a$ci_diagnostics$ci_hsic_perm_cuda_tests > 0,
            "CUDA HSIC permutation diagnostics should record CUDA tests")
assert_true(a$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "CUDA HSIC permutation diagnostics should record CUDA batches")
assert_true(a$ci_diagnostics$ci_hsic_permutation_replicates > 0,
            "CUDA HSIC permutation should record replicate work")
assert_true(identical(a$adjacency, b$adjacency),
            "fixed-seed CUDA HSIC permutation adjacency should repeat")
assert_true(max_abs_diff(a$pMax, b$pMax) < 1e-12,
            "fixed-seed CUDA HSIC permutation pMax should repeat")

fallback <- fast_skeleton_cuda_backend(
  data, alpha = 0.25, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 4,
  scheduler = "legacy",
  ci_method = "hsic.perm", hsic_params = list(sig = 1),
  permutation_params = list(replicates = 24L, include_observed = TRUE)
)

assert_true(fallback$ci_backend == "native-cpu",
            "seedless CUDA HSIC permutation should use native CPU fallback")
assert_true(fallback$ci_backend_reason ==
              "CUDA HSIC permutation requires explicit seed in this stage",
            "seedless CUDA HSIC permutation should record explicit fallback reason")
assert_true(fallback$ci_diagnostics$ci_hsic_cuda_fallback_tests > 0,
            "seedless CUDA HSIC permutation should record fallback tests")

cat("test_hsic_cuda_skeleton_permutation.R: PASS\n")
