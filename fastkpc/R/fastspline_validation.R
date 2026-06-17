source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/diff_report.R")
source("fastkpc/R/dcov_exact.R")

fastkpc_mgcv_residual <- function(y, S) {
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    stop("mgcv is required for this validation", call. = FALSE)
  }
  env <- fastkpc_legacy_env()
  residuals <- env$regrXonS(matrix(as.numeric(y), ncol = 1), as.matrix(S))
  as.numeric(residuals[, 1])
}

fastkpc_fastspline_residual <- function(y, S, fastspline_params = list()) {
  fastspline_residual(y, as.matrix(S), fastspline_params = fastspline_params)
}

residual_validation_metrics <- function(y, S) {
  mgcv_res <- fastkpc_mgcv_residual(y, S)
  fast <- fastkpc_fastspline_residual(y, S)
  fast_res <- as.numeric(fast$residual)
  p_mgcv <- dcov_gamma_exact(mgcv_res, seq_along(mgcv_res))$p.value
  p_fast <- dcov_gamma_exact(fast_res, seq_along(fast_res))$p.value
  denom <- sqrt(sum(mgcv_res^2))
  if (denom == 0) denom <- 1
  list(
    residual_correlation = unname(stats::cor(mgcv_res, fast_res)),
    relative_residual_l2 = sqrt(sum((mgcv_res - fast_res)^2)) / denom,
    fastspline_rss = sum(fast_res^2),
    mgcv_rss = sum(mgcv_res^2),
    dcov_pvalue_abs_diff = abs(p_fast - p_mgcv),
    selected_lambda = fast$selected_lambda,
    edf = fast$edf
  )
}

fastspline_validation_scenario <- function(seed = 61, n = 160) {
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  list(
    one_y = sin(z1) + stats::rnorm(n, sd = 0.15),
    one_S = matrix(z1, ncol = 1),
    two_y = sin(z1) + cos(z2) + stats::rnorm(n, sd = 0.15),
    two_S = cbind(z1, z2),
    graph_data = cbind(
      x1 = sin(z1) + stats::rnorm(n, sd = 0.2),
      x2 = cos(z1) + stats::rnorm(n, sd = 0.2),
      x3 = sin(z2) + stats::rnorm(n, sd = 0.2),
      x4 = z1 * z2 + stats::rnorm(n, sd = 0.2),
      x5 = stats::rnorm(n)
    )
  )
}

legacy_graph_result <- function(data, alpha, max_conditioning_size) {
  legacy <- fastkpc_legacy_skeleton(
    data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    ic.method = "dcc.gamma",
    numCol = max(1L, floor(nrow(data) / 10))
  )
  amat <- methods::as(legacy@graph, "matrix")
  storage.mode(amat) <- "logical"
  list(
    adjacency = amat,
    sepsets = legacy@sepset,
    pMax = legacy@pMax,
    n.edgetests = legacy@n.edgetests
  )
}

validate_fastspline_against_mgcv <- function(seed = 61, n = 160) {
  scenario <- fastspline_validation_scenario(seed = seed, n = n)
  one_d <- residual_validation_metrics(scenario$one_y, scenario$one_S)
  two_d <- residual_validation_metrics(scenario$two_y, scenario$two_S)

  graph <- if (!requireNamespace("pcalg", quietly = TRUE)) {
    list(
      available = FALSE,
      reason_if_unavailable = "pcalg is not installed",
      diff = NULL,
      max_abs_pmax_diff = NA_real_,
      adjacency_added_count = NA_integer_,
      adjacency_removed_count = NA_integer_
    )
  } else {
    legacy <- legacy_graph_result(scenario$graph_data, alpha = 0.2,
                                  max_conditioning_size = 1)
    fast <- fast_skeleton_cpp_backend(
      scenario$graph_data, alpha = 0.2, max_conditioning_size = 1,
      residual_backend = "fastSpline", residual_cache = TRUE
    )
    diff <- summarize_graph_diff(legacy, fast)
    list(
      available = TRUE,
      reason_if_unavailable = "",
      diff = diff,
      max_abs_pmax_diff = diff$pMax$max_abs_diff,
      adjacency_added_count = length(diff$adjacency$added_edges),
      adjacency_removed_count = length(diff$adjacency$removed_edges)
    )
  }

  list(one_d = one_d, two_d = two_d, graph = graph)
}

compare_fastspline_linear_graph <- function(seed = 51, n = 120,
                                            alpha = 0.2,
                                            max_conditioning_size = 2) {
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z1) + stats::rnorm(n, sd = 0.2),
    x2 = cos(z1) + stats::rnorm(n, sd = 0.2),
    x3 = sin(z2) + stats::rnorm(n, sd = 0.2),
    x4 = z1 * z2 + stats::rnorm(n, sd = 0.2),
    x5 = stats::rnorm(n)
  )
  linear <- fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                                      residual_backend = "linear",
                                      residual_cache = TRUE)
  fast <- fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                                    residual_backend = "fastSpline",
                                    residual_cache = TRUE)
  summarize_graph_diff(linear, fast)
}

compare_fastspline_cpu_cuda_graph <- function(seed = 51, n = 120,
                                              alpha = 0.2,
                                              max_conditioning_size = 2) {
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z1) + stats::rnorm(n, sd = 0.2),
    x2 = cos(z1) + stats::rnorm(n, sd = 0.2),
    x3 = sin(z2) + stats::rnorm(n, sd = 0.2),
    x4 = z1 * z2 + stats::rnorm(n, sd = 0.2),
    x5 = stats::rnorm(n)
  )
  cpu <- fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                                   residual_backend = "fastSpline",
                                   residual_cache = TRUE)
  cuda <- fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                                     residual_backend = "fastSpline",
                                     residual_cache = TRUE)
  list(
    diff = summarize_graph_diff(cpu, cuda),
    max_abs_pmax_diff = max(abs(cpu$pMax - cuda$pMax)),
    adjacency_identical = identical(cpu$adjacency, cuda$adjacency),
    sepsets_identical = identical(cpu$sepsets, cuda$sepsets),
    n_edgetests_identical = identical(cpu$n.edgetests, cuda$n.edgetests),
    cache_stats = cuda$residual_cache
  )
}

benchmark_fastspline_backends <- function(seed = 71, n = 180,
                                          alpha = 0.2,
                                          max_conditioning_size = 2) {
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z1) + stats::rnorm(n, sd = 0.2),
    x2 = cos(z1) + stats::rnorm(n, sd = 0.2),
    x3 = sin(z2) + stats::rnorm(n, sd = 0.2),
    x4 = z1 * z2 + stats::rnorm(n, sd = 0.2),
    x5 = stats::rnorm(n)
  )

  timed <- function(expr) {
    start <- proc.time()[["elapsed"]]
    value <- force(expr)
    elapsed <- proc.time()[["elapsed"]] - start
    list(value = value, elapsed = max(elapsed, .Machine$double.eps))
  }

  linear_cpu <- timed(fast_skeleton_cpp_backend(
    data, alpha, max_conditioning_size, residual_backend = "linear",
    residual_cache = TRUE
  ))
  fastspline_cpu <- timed(fast_skeleton_cpp_backend(
    data, alpha, max_conditioning_size, residual_backend = "fastSpline",
    residual_cache = TRUE
  ))
  fastspline_cuda <- timed(fast_skeleton_cuda_backend(
    data, alpha, max_conditioning_size, residual_backend = "fastSpline",
    residual_cache = TRUE
  ))

  timings <- data.frame(
    backend = c("linear", "fastSpline", "fastSpline"),
    engine = c("cpu", "cpu", "cuda"),
    elapsed_sec = c(linear_cpu$elapsed, fastspline_cpu$elapsed,
                    fastspline_cuda$elapsed)
  )
  cache <- data.frame(
    backend = c("linear", "fastSpline", "fastSpline"),
    engine = c("cpu", "cpu", "cuda"),
    requests = c(linear_cpu$value$residual_cache$requests,
                 fastspline_cpu$value$residual_cache$requests,
                 fastspline_cuda$value$residual_cache$requests),
    hits = c(linear_cpu$value$residual_cache$hits,
             fastspline_cpu$value$residual_cache$hits,
             fastspline_cuda$value$residual_cache$hits),
    computations = c(linear_cpu$value$residual_cache$computations,
                     fastspline_cpu$value$residual_cache$computations,
                     fastspline_cuda$value$residual_cache$computations)
  )

  fastspline_cpu_vs_cuda <- list(
    diff = summarize_graph_diff(fastspline_cpu$value, fastspline_cuda$value),
    max_abs_pmax_diff = max(abs(fastspline_cpu$value$pMax - fastspline_cuda$value$pMax)),
    adjacency_identical = identical(fastspline_cpu$value$adjacency,
                                    fastspline_cuda$value$adjacency),
    sepsets_identical = identical(fastspline_cpu$value$sepsets,
                                  fastspline_cuda$value$sepsets),
    n_edgetests_identical = identical(fastspline_cpu$value$n.edgetests,
                                      fastspline_cuda$value$n.edgetests)
  )

  list(
    timings = timings,
    cache = cache,
    graph = list(
      linear_vs_fastspline_cpu = summarize_graph_diff(linear_cpu$value,
                                                      fastspline_cpu$value),
      fastspline_cpu_vs_cuda = fastspline_cpu_vs_cuda$diff
    ),
    fastspline_cpu_vs_cuda = fastspline_cpu_vs_cuda
  )
}
