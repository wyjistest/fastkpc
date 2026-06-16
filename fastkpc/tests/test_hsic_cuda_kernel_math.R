source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(301)
n <- 64
x <- seq(-2, 2, length.out = n)
y <- sin(x) + rnorm(n, sd = 0.05)

cpu <- fast_hsic_gamma_cpp(x, y, sig = 1)
load_fastkpc_cuda_native(rebuild = TRUE)
gpu <- fast_hsic_gamma_cuda(x, y, sig = 1)

assert_true(gpu$backend == "cuda-hsic", "GPU HSIC gamma should report cuda-hsic")
assert_true(is.finite(gpu$statistic), "GPU HSIC statistic should be finite")
assert_true(abs(gpu$statistic - cpu$statistic) < 1e-8,
            "GPU HSIC statistic should match CPU dense HSIC")
assert_true(abs(gpu$p.value - cpu$p.value) < 1e-7,
            "GPU HSIC p-value should match CPU dense HSIC")
assert_true(gpu$diagnostics$n == n, "GPU HSIC diagnostics should record n")
assert_true(gpu$diagnostics$kernel == "rbf",
            "GPU HSIC diagnostics should record kernel")
assert_true(gpu$diagnostics$bytes_allocated > 0,
            "GPU HSIC diagnostics should record memory")

cat("test_hsic_cuda_kernel_math.R: PASS\n")
