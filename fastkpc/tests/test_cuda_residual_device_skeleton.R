source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set_key <- function(values) paste(sort(as.integer(values)), collapse = ",")

compare_sepsets_exact <- function(a, b) {
  for (i in seq_along(a)) {
    for (j in seq_along(a[[i]])) {
      if (!identical(set_key(a[[i]][[j]]), set_key(b[[i]][[j]]))) return(FALSE)
    }
  }
  TRUE
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(103)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.12),
  x2 = cos(z1) + rnorm(n, sd = 0.12),
  x3 = sin(z2) + rnorm(n, sd = 0.12),
  x4 = z1 * z2 + rnorm(n, sd = 0.12),
  x5 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

cuda_cpu_residual <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cpu",
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_cuda_residual <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  residual_cache = TRUE,
  batch_size = 0,
  fastspline_params = params
)

assert_true(cuda_cuda_residual$backend == "cuda",
            "skeleton backend should remain cuda")
assert_true(cuda_cuda_residual$residual_backend == "fastSpline",
            "residual_backend should be fastSpline")
assert_true(cuda_cuda_residual$residual_device == "cuda",
            "residual_device should be cuda")
assert_true(cuda_cuda_residual$residual_device_requested == "cuda",
            "requested residual device should be recorded")
assert_true(cuda_cuda_residual$residual_cache$residual_device == "cuda",
            "cache residual device should be cuda")
assert_true(cuda_cuda_residual$residual_cache$hits > 0,
            "CUDA residual cache should still have hits")
assert_true(cuda_cuda_residual$residual_cache$computations <
              cuda_cuda_residual$residual_cache$requests,
            "cache computations should be lower than requests")

assert_true(identical(cuda_cuda_residual$adjacency, cuda_cpu_residual$adjacency),
            "CUDA residual adjacency should match CPU residual")
assert_true(compare_sepsets_exact(cuda_cuda_residual$sepsets,
                                  cuda_cpu_residual$sepsets),
            "CUDA residual sepsets should match CPU residual")
assert_true(identical(cuda_cuda_residual$n.edgetests,
                      cuda_cpu_residual$n.edgetests),
            "CUDA residual n.edgetests should match CPU residual")
assert_true(max_abs_diff(cuda_cuda_residual$pMax, cuda_cpu_residual$pMax) < 1e-7,
            "CUDA residual pMax should match CPU residual")

linear_cuda_requested <- fast_skeleton_cuda_backend(
  data, alpha, 1L,
  residual_backend = "linear",
  residual_device = "cuda",
  residual_cache = TRUE
)
assert_true(linear_cuda_requested$residual_device == "cpu",
            "linear residual_device cuda request should resolve to cpu")
assert_true(grepl("linear residual CUDA device is not implemented",
                  linear_cuda_requested$residual_device_reason, fixed = TRUE),
            "linear cuda request should record reason")

cat("test_cuda_residual_device_skeleton.R: PASS\n")
