source("fastkpc/R/cuda_native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

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

set.seed(503)
n <- 130
z1 <- runif(n, -2, 2)
z2 <- runif(n, -2, 2)
data <- cbind(
  x1 = sin(z1) + rnorm(n, sd = 0.10),
  x2 = cos(z1) + rnorm(n, sd = 0.10),
  x3 = z1 * z2 + rnorm(n, sd = 0.10),
  x4 = sin(z2) + rnorm(n, sd = 0.10),
  x5 = cos(z2) + rnorm(n, sd = 0.10),
  x6 = rnorm(n)
)
alpha <- 0.2
max_ord <- 2
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

true_batch <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 0,
  residual_cache = TRUE,
  fastspline_params = params
)

one_at_a_time <- fast_skeleton_cuda_backend(
  data, alpha, max_ord,
  residual_backend = "fastSpline",
  residual_device = "cuda",
  scheduler = "layer",
  residual_batch_size = 1,
  residual_cache = TRUE,
  fastspline_params = params
)

assert_true(identical(true_batch$adjacency, one_at_a_time$adjacency),
            "true-batch residual prefetch adjacency should match one-at-a-time")
assert_true(compare_sepsets_exact(true_batch$sepsets, one_at_a_time$sepsets),
            "true-batch residual prefetch sepsets should match one-at-a-time")
assert_true(identical(true_batch$n.edgetests, one_at_a_time$n.edgetests),
            "true-batch residual prefetch n.edgetests should match one-at-a-time")
assert_true(max(abs(true_batch$pMax - one_at_a_time$pMax)) < 1e-7,
            "true-batch residual prefetch pMax should match one-at-a-time")

summary <- true_batch$scheduler_diagnostics$summary %||% list()
assert_true(as.integer(summary$cuda_residual_true_batched_groups %||% 0L) > 0L,
            "scheduler should report true-batched residual groups")
assert_true(as.integer(summary$cuda_residual_true_batched_fits %||% 0L) > 0L,
            "scheduler should report true-batched residual fits")
assert_true(identical(as.integer(summary$cuda_residual_single_fit_calls %||% -1L), 0L),
            "automatic residual batching should not use single-fit CUDA calls")
assert_true(as.integer(summary$residual_workspace_reuse_count %||% 0L) > 0L,
            "scheduler should reuse one fastSpline CUDA residual workspace")
assert_true(as.integer(summary$residual_solver_handle_create_count %||% 0L) <= 1L,
            "fastSpline CUDA residual workspace should amortize solver handles")

cat("test_cuda_layer_scheduler_true_residual_batch.R: PASS\n")
