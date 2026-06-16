source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(216)
n <- 48
z <- runif(n, -2, 2)
data <- data.frame(
  a = sin(z) + rnorm(n, sd = 0.08),
  b = cos(z) + rnorm(n, sd = 0.08),
  c = z^2 + rnorm(n, sd = 0.08),
  d = rnorm(n)
)

cpu <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cpu",
  graph_stage = "skeleton",
  residual_backend = "linear",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1)
)

assert_true(inherits(cpu, "fastkpc_result"),
            "fast_kpc HSIC CPU should return fastkpc_result")
assert_true(cpu$config$ci_method == "hsic.gamma",
            "config should record HSIC gamma")
assert_true(cpu$skeleton$ci_method == "hsic.gamma",
            "skeleton should record HSIC gamma")
assert_true(cpu$diagnostics$ci_method_available,
            "diagnostics should record CI method availability")

cuda <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  graph_stage = "skeleton",
  residual_backend = "linear",
  residual_device = "cuda",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1)
)

assert_true(cuda$config$ci_method == "hsic.gamma",
            "CUDA config should record HSIC gamma")
assert_true(cuda$skeleton$ci_backend == "cuda-hsic",
            "CUDA HSIC public API should record cuda-hsic CI backend")
assert_true(cuda$skeleton$ci_backend_reason == "",
            "CUDA HSIC public API should not record fallback reason on success")
assert_true(cuda$skeleton$ci_diagnostics$ci_hsic_gamma_cuda_tests > 0,
            "CUDA HSIC public API should expose CUDA HSIC test count")
assert_true(cuda$skeleton$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "CUDA HSIC public API should expose CUDA HSIC batch count")

default <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cpu",
  graph_stage = "skeleton",
  residual_backend = "linear"
)
assert_true(default$config$ci_method == "dcc.gamma",
            "default public API CI method should remain dcc.gamma")
assert_true(default$skeleton$ci_method == "dcc.gamma",
            "default skeleton CI method should remain dcc.gamma")

cat("test_fastkpc_ci_method_public_api.R: PASS\n")
