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

build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(307)
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

cpu_device <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cpu",
  scheduler = "layer",
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_auto_batch <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 0,
  residual_cache = TRUE,
  fastspline_params = params
)

cuda_one_batch <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 1,
  residual_cache = TRUE,
  fastspline_params = params
)

assert_true(identical(cuda_auto_batch$adjacency, cpu_device$adjacency),
            "CUDA residual prefetch adjacency should match CPU residual device")
assert_true(compare_sepsets_exact(cuda_auto_batch$sepsets, cpu_device$sepsets),
            "CUDA residual prefetch sepsets should match CPU residual device")
assert_true(identical(cuda_auto_batch$n.edgetests, cpu_device$n.edgetests),
            "CUDA residual prefetch n.edgetests should match CPU residual device")
assert_true(max(abs(cuda_auto_batch$pMax - cpu_device$pMax)) < 1e-7,
            "CUDA residual prefetch pMax should match CPU residual device")

assert_true(identical(cuda_one_batch$adjacency, cuda_auto_batch$adjacency),
            "residual_batch_size=1 adjacency should match auto")
assert_true(max(abs(cuda_one_batch$pMax - cuda_auto_batch$pMax)) < 1e-8,
            "residual_batch_size=1 pMax should match auto")
assert_true(cuda_auto_batch$scheduler == "layer",
            "scheduler should be layer")
assert_true(cuda_auto_batch$scheduler_diagnostics$summary$unique_residual_requests > 0,
            "scheduler should record unique residual requests")
assert_true(cuda_auto_batch$scheduler_diagnostics$summary$residual_batches > 0,
            "scheduler should record residual batches")
assert_true(cuda_auto_batch$residual_cache$computations <=
              cuda_auto_batch$residual_cache$requests,
            "cache computations should not exceed task-level requests")

cat("test_cuda_layer_scheduler_residual_prefetch.R: PASS\n")
