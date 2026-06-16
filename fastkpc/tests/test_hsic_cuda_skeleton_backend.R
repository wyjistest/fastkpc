source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

max_abs_diff <- function(a, b) {
  max(abs(as.numeric(a) - as.numeric(b)))
}

set.seed(303)
n <- 60
z <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + rnorm(n, sd = 0.08),
  x2 = cos(z) + rnorm(n, sd = 0.08),
  x3 = z^2 + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)

cpu <- fast_skeleton_cpp_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_cache = TRUE,
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

load_fastkpc_cuda_native(rebuild = TRUE)
cuda <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  scheduler = "legacy",
  ci_method = "hsic.gamma", hsic_params = list(sig = 1)
)

assert_true(cuda$ci_method == "hsic.gamma",
            "CUDA skeleton should record HSIC gamma")
assert_true(cuda$ci_backend == "cuda-hsic",
            "CUDA skeleton HSIC gamma should use cuda-hsic backend")
assert_true(cuda$ci_backend_reason == "",
            "successful CUDA HSIC gamma should not record fallback reason")
assert_true(cuda$ci_diagnostics$ci_hsic_gamma_cuda_tests > 0,
            "CUDA HSIC gamma diagnostics should record CUDA tests")
assert_true(cuda$ci_diagnostics$ci_hsic_cuda_batches > 0,
            "CUDA HSIC gamma diagnostics should record CUDA batches")
assert_true(identical(cpu$adjacency, cuda$adjacency),
            "CPU and CUDA HSIC gamma adjacency should match")
assert_true(max_abs_diff(cpu$pMax, cuda$pMax) < 1e-7,
            "CPU and CUDA HSIC gamma pMax should be close")

dcc <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  scheduler = "legacy",
  ci_method = "dcc.gamma"
)
assert_true(dcc$ci_method == "dcc.gamma",
            "dCov CUDA skeleton should still record dcc.gamma")
assert_true(dcc$ci_backend %in% c("cuda", "cuda-dcov"),
            "dCov CUDA skeleton should keep CUDA dCov backend")

limit_fallback <- fast_skeleton_cuda_backend(
  data, alpha = 0.2, max_conditioning_size = 1,
  residual_backend = "linear", residual_device = "cuda",
  residual_cache = TRUE, batch_size = 8,
  scheduler = "legacy",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1, cuda_max_n = 10L,
                     cuda_memory_fallback = TRUE)
)
assert_true(limit_fallback$ci_backend == "native-cpu",
            "CUDA HSIC n limit should fall back to native CPU")
assert_true(grepl("n exceeds configured max_n",
                  limit_fallback$ci_backend_reason, fixed = TRUE),
            "CUDA HSIC n limit fallback should record max_n reason")
assert_true(limit_fallback$ci_diagnostics$ci_hsic_cuda_fallback_tests > 0,
            "CUDA HSIC n limit fallback should record fallback tests")

limit_error <- tryCatch(
  fast_skeleton_cuda_backend(
    data, alpha = 0.2, max_conditioning_size = 1,
    residual_backend = "linear", residual_device = "cuda",
    residual_cache = TRUE, batch_size = 8,
    scheduler = "legacy",
    ci_method = "hsic.gamma",
    hsic_params = list(sig = 1, cuda_max_n = 10L,
                       cuda_memory_fallback = FALSE)
  ),
  error = conditionMessage
)
assert_true(is.character(limit_error) &&
              grepl("n exceeds configured max_n", limit_error, fixed = TRUE),
            "CUDA HSIC n limit with fallback disabled should raise an error")

cat("test_hsic_cuda_skeleton_backend.R: PASS\n")
