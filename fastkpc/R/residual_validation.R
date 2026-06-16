.fastkpc_residual_cache_scenario <- function(seed = 31, n = 90) {
  set.seed(seed)
  z1 <- stats::rnorm(n)
  z2 <- stats::rnorm(n)
  cbind(
    x1 = z1 + stats::rnorm(n, sd = 0.2),
    x2 = z1 - z2 + stats::rnorm(n, sd = 0.2),
    x3 = z2 + stats::rnorm(n, sd = 0.2),
    x4 = z1 * z2 + stats::rnorm(n, sd = 0.2),
    x5 = stats::rnorm(n)
  )
}

validate_cpu_residual_cache <- function(seed = 31, n = 90, alpha = 0.2,
                                        max_conditioning_size = 2) {
  source("fastkpc/R/native.R")
  source("fastkpc/R/diff_report.R")

  data <- .fastkpc_residual_cache_scenario(seed = seed, n = n)
  uncached <- fast_skeleton_cpp_cached(
    data, alpha = alpha, max_conditioning_size = max_conditioning_size,
    residual_cache = FALSE)
  cached <- fast_skeleton_cpp_cached(
    data, alpha = alpha, max_conditioning_size = max_conditioning_size,
    residual_cache = TRUE)
  diff <- summarize_graph_diff(uncached, cached)

  list(
    diff = diff,
    max_abs_pmax_diff = max(abs(cached$pMax - uncached$pMax)),
    adjacency_identical = identical(cached$adjacency, uncached$adjacency),
    sepsets_identical = diff$sepsets$differing_count == 0,
    n_edgetests_identical = identical(cached$n.edgetests, uncached$n.edgetests),
    cache_stats = cached$residual_cache
  )
}

validate_cuda_residual_cache <- function(seed = 31, n = 90, alpha = 0.2,
                                         max_conditioning_size = 2,
                                         batch_size = 0) {
  source("fastkpc/R/native.R")
  source("fastkpc/R/cuda_native.R")
  source("fastkpc/R/diff_report.R")

  data <- .fastkpc_residual_cache_scenario(seed = seed, n = n)
  uncached <- fast_skeleton_cpp_cached(
    data, alpha = alpha, max_conditioning_size = max_conditioning_size,
    residual_cache = FALSE)
  cached <- fast_skeleton_cuda_cached(
    data, alpha = alpha, max_conditioning_size = max_conditioning_size,
    batch_size = batch_size, residual_cache = TRUE)
  diff <- summarize_graph_diff(uncached, cached)

  list(
    diff = diff,
    max_abs_pmax_diff = max(abs(cached$pMax - uncached$pMax)),
    adjacency_identical = identical(cached$adjacency, uncached$adjacency),
    sepsets_identical = diff$sepsets$differing_count == 0,
    n_edgetests_identical = identical(cached$n.edgetests, uncached$n.edgetests),
    cache_stats = cached$residual_cache
  )
}
