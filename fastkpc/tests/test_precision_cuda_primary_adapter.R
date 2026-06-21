source("fastkpc/R/fast_kpc.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

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

set.seed(621)
n <- 110
z <- stats::runif(n, -2, 2)
data <- cbind(
  x1 = sin(z) + stats::rnorm(n, sd = 0.12),
  x2 = cos(z) + stats::rnorm(n, sd = 0.12),
  x3 = z^2 + stats::rnorm(n, sd = 0.12),
  x4 = stats::rnorm(n),
  x5 = 0.5 * z + stats::rnorm(n, sd = 0.12)
)
params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)

fast_cuda <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "fast",
  graph_stage = "skeleton",
  fastspline_params = params,
  seed = 621
)

precision_primary <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "fast",
  graph_stage = "skeleton",
  fastspline_params = params,
  precision_executors = fastkpc_default_precision_executors(),
  seed = 621
)

assert_true(identical(precision_primary$skeleton$adjacency,
                      fast_cuda$skeleton$adjacency),
            "batched precision primary should match fast CUDA adjacency")
assert_true(compare_sepsets_exact(precision_primary$skeleton$sepsets,
                                  fast_cuda$skeleton$sepsets),
            "batched precision primary should match fast CUDA sepsets")
assert_true(max(abs(precision_primary$skeleton$pMax -
                      fast_cuda$skeleton$pMax)) < 1e-8,
            "batched precision primary pMax should match fast CUDA pMax")
assert_true(identical(precision_primary$skeleton$n.edgetests,
                      fast_cuda$skeleton$n.edgetests),
            "batched precision primary should match fast CUDA n.edgetests")
assert_true(identical(precision_primary$skeleton$scheduler, "layer"),
            "default CUDA precision primary should use layer scheduler")
summary <- precision_primary$skeleton$scheduler_diagnostics$summary
assert_true(as.integer(summary$dcov_batches %||% 0L) > 0L,
            "batched precision primary should use CUDA dCov batches")
assert_true(as.integer(summary$cuda_residual_true_batched_groups %||% 0L) > 0L,
            "batched precision primary should use true-batched residual groups")
assert_true(!identical(precision_primary$skeleton$scheduler, "r-precision"),
            "default CUDA precision primary should not use scalar R precision loop")

scalar_executors <- fastkpc_default_precision_executors()
scalar_executors$fastSplineCUDA <- function(...) {
  fastkpc_execute_ci_fast_spline_cuda(...)
}

scalar_hybrid <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  fastspline_params = params,
  precision_executors = scalar_executors,
  seed = 621
)

batched_hybrid <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 1,
  engine = "cuda",
  precision = "hybrid",
  tau = Inf,
  graph_stage = "skeleton",
  fastspline_params = params,
  seed = 621
)

assert_true(identical(batched_hybrid$skeleton$adjacency,
                      scalar_hybrid$skeleton$adjacency),
            "batched hybrid should match scalar precision hybrid adjacency")
assert_true(compare_sepsets_exact(batched_hybrid$skeleton$sepsets,
                                  scalar_hybrid$skeleton$sepsets),
            "batched hybrid should match scalar precision hybrid sepsets")
assert_true(max(abs(batched_hybrid$skeleton$pMax -
                      scalar_hybrid$skeleton$pMax)) < 1e-7,
            "batched hybrid pMax should match scalar precision hybrid pMax")
assert_true(identical(batched_hybrid$skeleton$n.edgetests,
                      scalar_hybrid$skeleton$n.edgetests),
            "batched hybrid n.edgetests should match scalar precision hybrid")
assert_true(identical(batched_hybrid$skeleton$scheduler, "layer-precision"),
            "default CUDA hybrid should use batched precision layer scheduler")
hybrid_trace <- batched_hybrid$skeleton$precision_trace
assert_true(nrow(hybrid_trace) == sum(batched_hybrid$skeleton$n.edgetests),
            "batched hybrid trace should preserve canonical replay rows")
assert_true(any(hybrid_trace$near_alpha_triggered),
            "batched hybrid should still execute sparse verifier rows")

cat("PASS precision CUDA primary adapter\n")
