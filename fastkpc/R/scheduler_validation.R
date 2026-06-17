source("fastkpc/R/fast_kpc.R")

fastkpc_scheduler_benchmark_data <- function(seed, n, p) {
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  noise <- function(sd = 0.10) stats::rnorm(n, sd = sd)
  base <- list(
    sin(z1) + noise(),
    cos(z1) + noise(),
    z1 * z2 + noise(),
    sin(z2) + noise(),
    cos(z2) + noise(),
    z1 + noise(),
    z2 + noise(),
    stats::rnorm(n)
  )
  data <- do.call(cbind, base[seq_len(min(p, length(base)))])
  if (p > length(base)) {
    extra <- replicate(p - length(base), stats::rnorm(n), simplify = FALSE)
    data <- cbind(data, do.call(cbind, extra))
  }
  colnames(data) <- paste0("x", seq_len(ncol(data)))
  storage.mode(data) <- "double"
  data
}

benchmark_layer_scheduler <- function(seed = 407,
                                      n = 500,
                                      p = 8,
                                      alpha = 0.2,
                                      max_conditioning_size = 2,
                                      residual_backend = "fastSpline",
                                      residual_device = "cuda",
                                      schedulers = c("legacy", "layer"),
                                      batch_sizes = c(1L, 0L),
                                      residual_batch_sizes = c(1L, 0L),
                                      fastspline_params = list(knots = 8,
                                                               lambda_count = 17,
                                                               ridge = 1e-8)) {
  schedulers <- match.arg(schedulers, c("legacy", "layer", "auto"),
                          several.ok = TRUE)
  data <- fastkpc_scheduler_benchmark_data(seed = seed, n = n, p = p)
  rows <- list()
  results <- list()
  grid <- expand.grid(
    scheduler = schedulers,
    batch_size = as.integer(batch_sizes),
    residual_batch_size = as.integer(residual_batch_sizes),
    stringsAsFactors = FALSE
  )
  for (i in seq_len(nrow(grid))) {
    row <- grid[i, ]
    elapsed <- system.time({
      result <- fast_kpc(
        data,
        alpha = alpha,
        max_conditioning_size = max_conditioning_size,
        engine = "cuda",
        residual_backend = residual_backend,
        residual_device = residual_device,
        scheduler = row$scheduler,
        batch_size = row$batch_size,
        residual_batch_size = row$residual_batch_size,
        graph_stage = "skeleton",
        fastspline_params = fastspline_params,
        benchmark = TRUE
      )
    })[["elapsed"]]
    scheduler_summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
    cache <- result$skeleton$residual_cache %||% list()
    rows[[length(rows) + 1L]] <- data.frame(
      scheduler = row$scheduler,
      batch_size = as.integer(row$batch_size),
      residual_batch_size = as.integer(row$residual_batch_size),
      elapsed_sec = as.numeric(elapsed),
      skeleton_edges = as.integer(result$metrics$skeleton_edge_count),
      tasks_planned = as.integer(scheduler_summary$tasks_planned %||% 0L),
      tasks_evaluated = as.integer(scheduler_summary$tasks_evaluated %||% 0L),
      tests_replayed = as.integer(scheduler_summary$tests_replayed %||% 0L),
      tasks_ignored_after_delete =
        as.integer(scheduler_summary$tasks_ignored_after_delete %||% 0L),
      dcov_batches = as.integer(scheduler_summary$dcov_batches %||% 0L),
      unique_residual_requests =
        as.integer(scheduler_summary$unique_residual_requests %||% 0L),
      residual_batches = as.integer(scheduler_summary$residual_batches %||% 0L),
      cuda_residual_batch_groups =
        as.integer(scheduler_summary$cuda_residual_batch_groups %||% 0L),
      cuda_residual_true_batched_groups =
        as.integer(scheduler_summary$cuda_residual_true_batched_groups %||% 0L),
      cuda_residual_true_batched_fits =
        as.integer(scheduler_summary$cuda_residual_true_batched_fits %||% 0L),
      cuda_residual_single_fit_calls =
        as.integer(scheduler_summary$cuda_residual_single_fit_calls %||% 0L),
      cuda_residual_cpu_fallback_fits =
        as.integer(scheduler_summary$cuda_residual_cpu_fallback_fits %||% 0L),
      cuda_residual_unique_designs =
        as.integer(scheduler_summary$cuda_residual_unique_designs %||% 0L),
      cuda_residual_duplicate_design_fits =
        as.integer(scheduler_summary$cuda_residual_duplicate_design_fits %||% 0L),
      cuda_residual_max_fits_per_design =
        as.integer(scheduler_summary$cuda_residual_max_fits_per_design %||% 0L),
      residual_cache_requests = as.integer(cache$requests %||% 0L),
      residual_cache_computations = as.integer(cache$computations %||% 0L),
      stringsAsFactors = FALSE
    )
    key <- paste(row$scheduler, row$batch_size, row$residual_batch_size, sep = "\r")
    results[[key]] <- result
  }
  runs <- do.call(rbind, rows)
  baseline <- results[[1L]]
  graph_equal <- all(vapply(results, function(result) {
    identical(result$skeleton$adjacency, baseline$skeleton$adjacency) &&
      max(abs(result$skeleton$pMax - baseline$skeleton$pMax)) < 1e-7
  }, logical(1)))
  list(
    runs = runs,
    summary = aggregate(elapsed_sec ~ scheduler, runs, mean),
    graph_equal = graph_equal
  )
}
