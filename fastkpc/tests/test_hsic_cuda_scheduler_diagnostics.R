source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(304)
n <- 72
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.07),
  x2 = z + rnorm(n, sd = 0.07),
  x3 = z^2 + rnorm(n, sd = 0.07),
  x4 = rnorm(n),
  x5 = cos(z) + rnorm(n, sd = 0.07)
)

load_fastkpc_cuda_native(rebuild = TRUE)
cuda <- fast_skeleton_cuda_backend(
  data, alpha = 0.18, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 5, residual_batch_size = 3,
  scheduler = "layer",
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(cuda$scheduler == "layer",
            "CUDA HSIC skeleton should honor layer scheduler")
assert_true(cuda$ci_backend == "cuda-hsic",
            "layer scheduler HSIC gamma should use cuda-hsic backend")
assert_true(cuda$ci_diagnostics$ci_hsic_gamma_cuda_tests > 0,
            "layer scheduler should record CUDA HSIC gamma tests")
assert_true(cuda$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "layer scheduler should record CUDA HSIC batches")
assert_true(cuda$ci_diagnostics$ci_hsic_cuda_pairs >=
              cuda$ci_diagnostics$ci_hsic_gamma_cuda_tests,
            "CUDA HSIC pair counter should cover gamma tests")

batches <- cuda$scheduler_diagnostics$batches
assert_true(nrow(batches) > 0, "scheduler diagnostics should include batches")
assert_true(all(batches$kind == "hsic" | batches$kind == "residual"),
            "scheduler batches should label HSIC CI batches explicitly")
assert_true(any(batches$kind == "hsic"),
            "scheduler diagnostics should include HSIC CI batch rows")
assert_true(cuda$scheduler_diagnostics$summary$dcov_batches == 0,
            "HSIC scheduler path should not increment dCov batch counters")

cat("test_hsic_cuda_scheduler_diagnostics.R: PASS\n")
