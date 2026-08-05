source("fastkpc/R/fast_kpc.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

set.seed(1221)
n <- 48L
z <- runif(n, -2, 2)
data <- cbind(
  x1 = z + rnorm(n, sd = 0.05),
  x2 = sin(z) + rnorm(n, sd = 0.05),
  x3 = z^2 + rnorm(n, sd = 0.05),
  x4 = cos(z) + rnorm(n, sd = 0.05),
  x5 = z^3 + rnorm(n, sd = 0.05)
)
expected_backend <- c(
  `dcc.perm` = "cuda-dcov",
  `hsic.gamma` = "cuda-hsic",
  `hsic.perm` = "cuda-hsic"
)

load_fastkpc_cuda_native(rebuild = FALSE)

for (method in names(expected_backend)) {
  result <- fast_kpc(
    data,
    alpha = 0.2,
    max_conditioning_size = 2L,
    engine = "cuda",
    graph_stage = "skeleton",
    residual_backend = "linear",
    residual_device = "cpu",
    scheduler = "layer",
    ci_method = method,
    hsic_params = list(sig = 1),
    permutation_params = list(
      replicates = 19L,
      seed = 910L,
      include_observed = TRUE
    )
  )

  assert_true(result$config$ci_backend == expected_backend[[method]],
              paste(method, "did not execute on its CUDA backend"))
  assert_true(length(result$skeleton$n.edgetests) == 3L &&
                result$skeleton$n.edgetests[[3L]] > 0L,
              paste(method, "did not execute conditioning level 2"))
}

cat("test_ci_methods_cuda_conditioning_levels.R: PASS\n")
