source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

compare_sepsets_exact <- function(a, b) {
  for (i in seq_along(a)) {
    for (j in seq_along(a[[i]])) {
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) {
        return(FALSE)
      }
    }
  }
  TRUE
}

set.seed(31)
n <- 90
z1 <- rnorm(n)
z2 <- rnorm(n)
data <- cbind(
  x1 = z1 + rnorm(n, sd = 0.2),
  x2 = z1 - z2 + rnorm(n, sd = 0.2),
  x3 = z2 + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)

alpha <- 0.2
max_ord <- 2

cuda_plain <- fast_skeleton_cuda(data, alpha = alpha, max_conditioning_size = max_ord)
cuda_uncached <- fast_skeleton_cuda_cached(data, alpha = alpha,
                                           max_conditioning_size = max_ord,
                                           residual_cache = FALSE)
cpu_uncached <- fast_skeleton_cpp_cached(data, alpha = alpha,
                                         max_conditioning_size = max_ord,
                                         residual_cache = FALSE)
cuda_cached <- fast_skeleton_cuda_cached(data, alpha = alpha,
                                         max_conditioning_size = max_ord,
                                         residual_cache = TRUE,
                                         batch_size = 0)
cuda_cached_one <- fast_skeleton_cuda_cached(data, alpha = alpha,
                                             max_conditioning_size = max_ord,
                                             residual_cache = TRUE,
                                             batch_size = 1)

assert_true(identical(cuda_uncached$adjacency, cuda_plain$adjacency),
            "CUDA cached wrapper with cache disabled should match fast_skeleton_cuda adjacency")
assert_true(max(abs(cuda_uncached$pMax - cuda_plain$pMax)) < 1e-8,
            "CUDA cached wrapper with cache disabled should match fast_skeleton_cuda pMax")

assert_true(identical(cuda_cached$adjacency, cpu_uncached$adjacency),
            "cached CUDA adjacency should match uncached CPU")
assert_true(max(abs(cuda_cached$pMax - cpu_uncached$pMax)) < 1e-8,
            "cached CUDA pMax should match uncached CPU")
assert_true(compare_sepsets_exact(cuda_cached$sepsets, cpu_uncached$sepsets),
            "cached CUDA sepsets should match uncached CPU")
assert_true(identical(cuda_cached$n.edgetests, cpu_uncached$n.edgetests),
            "cached CUDA n.edgetests should match uncached CPU")

assert_true(cuda_cached$backend == "cuda", "cached CUDA run should report backend cuda")
assert_true(cuda_cached$residual_cache$enabled, "cached CUDA run should report cache enabled")
assert_true(cuda_cached$residual_cache$hits > 0, "cached CUDA run should report cache hits")
assert_true(cuda_cached$residual_cache$computations < cuda_cached$residual_cache$requests,
            "cached CUDA run should compute fewer residuals than requests")

assert_true(identical(cuda_cached_one$adjacency, cuda_cached$adjacency),
            "batch_size=1 cached CUDA adjacency should match auto")
assert_true(max(abs(cuda_cached_one$pMax - cuda_cached$pMax)) < 1e-8,
            "batch_size=1 cached CUDA pMax should match auto")
assert_true(compare_sepsets_exact(cuda_cached_one$sepsets, cuda_cached$sepsets),
            "batch_size=1 cached CUDA sepsets should match auto")
assert_true(identical(cuda_cached_one$n.edgetests, cuda_cached$n.edgetests),
            "batch_size=1 cached CUDA n.edgetests should match auto")

cat("test_cuda_residual_cache.R: PASS\n")
