source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(307)
n <- 56
z <- runif(n, -2, 2)
data <- data.frame(
  a = sin(z) + rnorm(n, sd = 0.07),
  b = z + rnorm(n, sd = 0.07),
  c = z^2 + rnorm(n, sd = 0.07),
  d = rnorm(n)
)

result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  graph_stage = "skeleton",
  residual_backend = "linear",
  residual_device = "cuda",
  scheduler = "legacy",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1)
)

assert_true(result$config$ci_method == "hsic.gamma",
            "fast_kpc config should record HSIC gamma")
assert_true(result$config$ci_backend == "cuda-hsic",
            "fast_kpc config should record cuda-hsic backend")
assert_true(result$config$ci_backend_requested == "cuda",
            "fast_kpc config should record requested CUDA CI backend")
assert_true(isTRUE(result$config$cuda_hsic_requested),
            "fast_kpc config should record CUDA HSIC requested")
assert_true(isTRUE(result$config$cuda_hsic_used),
            "fast_kpc config should record CUDA HSIC used")
assert_true(result$skeleton$ci_backend == "cuda-hsic",
            "fast_kpc skeleton should expose cuda-hsic backend")
assert_true(isTRUE(result$diagnostics$cuda_hsic_available),
            "fast_kpc diagnostics should record CUDA HSIC availability")
assert_true(identical(result$diagnostics$cuda_hsic_reason, ""),
            "fast_kpc diagnostics should not record CUDA HSIC reason on success")

printed <- paste(capture.output(print(result)), collapse = "\n")
assert_true(grepl("ci_method: hsic.gamma", printed, fixed = TRUE),
            "print.fastkpc_result should include ci_method")
assert_true(grepl("ci_backend: cuda-hsic", printed, fixed = TRUE),
            "print.fastkpc_result should include ci_backend")

fallback <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cuda",
  graph_stage = "skeleton",
  residual_backend = "linear",
  residual_device = "cuda",
  scheduler = "legacy",
  ci_method = "hsic.perm",
  hsic_params = list(sig = 1),
  permutation_params = list(replicates = 20L, include_observed = TRUE)
)

assert_true(fallback$config$ci_backend == "native-cpu",
            "seedless CUDA HSIC permutation should record native-cpu fallback")
assert_true(fallback$config$ci_backend_reason ==
              "CUDA HSIC permutation requires explicit seed in this stage",
            "seedless CUDA HSIC permutation should record fallback reason")
assert_true(isTRUE(fallback$config$cuda_hsic_requested),
            "seedless CUDA HSIC permutation should still record CUDA HSIC requested")
assert_true(!isTRUE(fallback$config$cuda_hsic_used),
            "seedless CUDA HSIC permutation should record CUDA HSIC unused")

default <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1L,
  engine = "cpu",
  graph_stage = "skeleton",
  residual_backend = "linear"
)
assert_true(default$config$ci_method == "dcc.gamma",
            "default fast_kpc CI method should remain dcc.gamma")

cat("test_hsic_cuda_public_api.R: PASS\n")
