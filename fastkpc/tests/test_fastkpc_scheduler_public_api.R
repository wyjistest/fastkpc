source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(409)
n <- 100
z <- seq(-pi, pi, length.out = n)
data <- cbind(
  x1 = z + rnorm(n, sd = 0.05),
  x2 = sin(z) + rnorm(n, sd = 0.08),
  x3 = cos(z) + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)
alpha <- 0.2
max_ord <- 1
params <- list(knots = 7, lambda_count = 13, ridge = 1e-8)

cuda_layer <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 0,
  graph_stage = "skeleton",
  fastspline_params = params
)

cuda_legacy <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "legacy",
  graph_stage = "skeleton",
  fastspline_params = params
)

cuda_auto <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "auto",
  graph_stage = "skeleton",
  fastspline_params = params
)

assert_true(inherits(cuda_layer, "fastkpc_result"),
            "layer result should be fastkpc_result")
assert_true(cuda_layer$config$scheduler_requested == "layer",
            "scheduler_requested should be layer")
assert_true(cuda_layer$config$scheduler_used == "layer",
            "scheduler_used should be layer")
assert_true(cuda_layer$config$residual_batch_size == 0L,
            "residual_batch_size should be recorded")
assert_true(cuda_layer$skeleton$scheduler == "layer",
            "skeleton scheduler should be layer")
assert_true(is.list(cuda_layer$skeleton$scheduler_diagnostics),
            "skeleton scheduler diagnostics should be present")
assert_true(cuda_layer$metrics$tasks_planned >= cuda_layer$metrics$tests_replayed,
            "scheduler metrics should be included")

assert_true(cuda_auto$config$scheduler_requested == "auto",
            "auto scheduler request should be recorded")
assert_true(cuda_auto$config$scheduler_used == "layer",
            "auto scheduler should resolve to layer on CUDA")
assert_true(identical(cuda_layer$skeleton$adjacency, cuda_legacy$skeleton$adjacency),
            "layer adjacency should match legacy")
assert_true(max(abs(cuda_layer$skeleton$pMax - cuda_legacy$skeleton$pMax)) < 1e-7,
            "layer pMax should match legacy")

cpu_auto <- fast_kpc(
  data,
  alpha = alpha,
  max_conditioning_size = max_ord,
  engine = "cpu",
  residual_backend = "linear",
  scheduler = "auto",
  graph_stage = "skeleton"
)
assert_true(cpu_auto$config$scheduler_used == "legacy",
            "CPU auto scheduler should resolve to legacy")

err <- tryCatch(
  fast_kpc(data, alpha = alpha, max_conditioning_size = max_ord,
           engine = "cpu", residual_backend = "linear",
           scheduler = "layer", graph_stage = "skeleton"),
  error = conditionMessage
)
assert_true(grepl("layer scheduler is only implemented for CUDA", err, fixed = TRUE),
            "explicit CPU layer scheduler should error clearly")

cat("test_fastkpc_scheduler_public_api.R: PASS\n")
