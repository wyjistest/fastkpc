source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(81)
n <- 120
z <- seq(-pi, pi, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + 0.03 * cos(19 * z),
  x3 = cos(0.5 * z),
  x4 = z^2 + 0.02 * sin(11 * z)
)

build_fastkpc_native(rebuild = TRUE)
build_fastkpc_cuda_native(rebuild = TRUE)

cpu <- fast_kpc_wanpdag_cpp(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE
)

cuda <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE,
  batch_size = 0
)

cuda_one <- fast_kpc_wanpdag_cuda(
  data,
  alpha = 0.12,
  max_conditioning_size = 1L,
  residual_backend = "fastSpline",
  residual_cache = TRUE,
  batch_size = 1
)

assert_true(identical(cuda$skeleton$backend, "cuda"),
            "CUDA pipeline skeleton backend should report cuda")
assert_true(identical(cuda$orientation$residual_backend, "fastSpline"),
            "CUDA pipeline orientation backend should report fastSpline")
assert_true(identical(cuda$orientation$pdag, cpu$orientation$pdag),
            "CUDA pipeline pdag should match CPU pipeline pdag")
assert_true(identical(cuda_one$orientation$pdag, cuda$orientation$pdag),
            "CUDA batch_size=1 pdag should match auto batch pdag")
assert_true(cuda$orientation$residual_cache$requests >=
              cuda$orientation$residual_cache$computations,
            "CUDA orientation cache stats should be recorded")

cat("test_wanpdag_cuda_pipeline.R: PASS\n")
