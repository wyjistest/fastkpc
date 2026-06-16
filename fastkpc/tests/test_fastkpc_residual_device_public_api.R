source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) max(abs(as.numeric(a) - as.numeric(b)))

set.seed(104)
n <- 100
z <- seq(-2, 2, length.out = n)
data <- cbind(
  x1 = z,
  x2 = sin(z) + rnorm(n, sd = 0.1),
  x3 = cos(z) + rnorm(n, sd = 0.1),
  x4 = z^2 + rnorm(n, sd = 0.1)
)

cpu_residual <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cpu",
  graph_stage = "wanpdag",
  seed = 104
)

cuda_residual <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  graph_stage = "wanpdag",
  seed = 104
)

assert_true(inherits(cuda_residual, "fastkpc_result"),
            "result should have fastkpc_result class")
assert_true(cuda_residual$config$residual_device_requested == "cuda",
            "config should record requested residual device")
assert_true(cuda_residual$config$residual_device_used == "cuda",
            "config should record used residual device")
assert_true(cuda_residual$skeleton$residual_device == "cuda",
            "skeleton should record cuda residual device")
assert_true(cuda_residual$skeleton$residual_cache$residual_device == "cuda",
            "cache should record cuda residual device")
assert_true(isTRUE(cuda_residual$diagnostics$cuda_residual_available),
            "diagnostics should report cuda residual availability")

assert_true(identical(cuda_residual$skeleton$adjacency,
                      cpu_residual$skeleton$adjacency),
            "public CUDA residual skeleton should match CPU residual")
assert_true(max_abs_diff(cuda_residual$skeleton$pMax,
                         cpu_residual$skeleton$pMax) < 1e-7,
            "public CUDA residual pMax should match CPU residual")
assert_true(identical(cuda_residual$orientation$pdag,
                      cpu_residual$orientation$pdag),
            "public CUDA residual pdag should match CPU residual")

default_result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  residual_backend = "fastSpline",
  graph_stage = "skeleton"
)
assert_true(default_result$config$residual_device_requested == "auto",
            "default residual_device should be auto")
assert_true(default_result$config$residual_device_used == "cpu",
            "CPU engine should use CPU residual device")

cat("test_fastkpc_residual_device_public_api.R: PASS\n")
