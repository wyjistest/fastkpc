source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(302)
n <- 56
x <- seq(-1.5, 1.5, length.out = n)
y <- x^2 + rnorm(n, sd = 0.03)

load_fastkpc_cuda_native(rebuild = TRUE)
a <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 500L,
                         include_observed = TRUE)
b <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 500L,
                         include_observed = TRUE)
c <- fast_hsic_perm_cuda(x, y, sig = 1, replicates = 40L, seed = 501L,
                         include_observed = TRUE)

assert_true(a$backend == "cuda-hsic", "GPU HSIC permutation should report cuda-hsic")
assert_true(is.finite(a$p.value), "GPU HSIC permutation p-value should be finite")
assert_true(a$p.value >= 0 && a$p.value <= 1,
            "GPU HSIC permutation p-value should be in [0, 1]")
assert_true(length(a$replicates) == 40L,
            "GPU HSIC permutation should return requested replicates")
assert_true(max(abs(as.numeric(a$replicates) - as.numeric(b$replicates))) < 1e-12,
            "GPU HSIC permutation fixed seed should repeat within tolerance")
assert_true(a$p.value == b$p.value,
            "GPU HSIC permutation fixed seed p-value should repeat exactly")
assert_true(max(abs(as.numeric(a$replicates) - as.numeric(c$replicates))) > 1e-15,
            "GPU HSIC permutation different seed should change replicate order")
assert_true(a$diagnostics$seed == 500L,
            "GPU HSIC permutation diagnostics should record seed")

cat("test_hsic_cuda_permutation.R: PASS\n")
