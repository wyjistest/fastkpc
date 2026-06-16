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

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

set.seed(51)
n <- 120
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.2),
  x2 = cos(z1) + rnorm(n, sd = 0.2),
  x3 = sin(z2) + rnorm(n, sd = 0.2),
  x4 = z1 * z2 + rnorm(n, sd = 0.2),
  x5 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2

cuda_linear_backend <- fast_skeleton_cuda_backend(
  data, alpha, max_ord, residual_backend = "linear", residual_cache = TRUE
)
cuda_linear_cached <- fast_skeleton_cuda_cached(
  data, alpha, max_ord, residual_cache = TRUE
)

assert_true(identical(cuda_linear_backend$adjacency, cuda_linear_cached$adjacency),
            "CUDA linear backend adjacency should match cached CUDA wrapper")
assert_true(max_abs_diff(cuda_linear_backend$pMax, cuda_linear_cached$pMax) < 1e-8,
            "CUDA linear backend pMax should match cached CUDA wrapper")

cpu_fastspline <- fast_skeleton_cpp_backend(
  data, alpha, max_ord, residual_backend = "fastSpline", residual_cache = TRUE
)
cuda_fastspline <- fast_skeleton_cuda_backend(
  data, alpha, max_ord, residual_backend = "fastSpline", residual_cache = TRUE,
  batch_size = 0
)
cuda_fastspline_one <- fast_skeleton_cuda_backend(
  data, alpha, max_ord, residual_backend = "fastSpline", residual_cache = TRUE,
  batch_size = 1
)

assert_true(cuda_fastspline$backend == "cuda", "CUDA fastSpline backend should report cuda")
assert_true(cuda_fastspline$residual_backend == "fastSpline",
            "CUDA fastSpline should record residual backend")
assert_true(is.character(cuda_fastspline$residual_backend_params),
            "CUDA fastSpline should include backend params")
assert_true(cuda_fastspline$residual_cache$hits > 0,
            "CUDA fastSpline cache should have hits")
assert_true(cuda_fastspline$residual_cache$computations <
              cuda_fastspline$residual_cache$requests,
            "CUDA fastSpline cache computations should be lower than requests")

assert_true(identical(cuda_fastspline$adjacency, cpu_fastspline$adjacency),
            "CUDA fastSpline adjacency should match CPU fastSpline")
assert_true(compare_sepsets_exact(cuda_fastspline$sepsets, cpu_fastspline$sepsets),
            "CUDA fastSpline sepsets should match CPU fastSpline")
assert_true(identical(cuda_fastspline$n.edgetests, cpu_fastspline$n.edgetests),
            "CUDA fastSpline n.edgetests should match CPU fastSpline")
assert_true(max_abs_diff(cuda_fastspline$pMax, cpu_fastspline$pMax) < 1e-8,
            "CUDA fastSpline pMax should match CPU fastSpline")

assert_true(identical(cuda_fastspline_one$adjacency, cuda_fastspline$adjacency),
            "batch_size=1 CUDA fastSpline adjacency should match auto")
assert_true(compare_sepsets_exact(cuda_fastspline_one$sepsets, cuda_fastspline$sepsets),
            "batch_size=1 CUDA fastSpline sepsets should match auto")
assert_true(identical(cuda_fastspline_one$n.edgetests, cuda_fastspline$n.edgetests),
            "batch_size=1 CUDA fastSpline n.edgetests should match auto")
assert_true(max_abs_diff(cuda_fastspline_one$pMax, cuda_fastspline$pMax) < 1e-8,
            "batch_size=1 CUDA fastSpline pMax should match auto")

cat("test_skeleton_fastspline_cuda.R: PASS\n")
