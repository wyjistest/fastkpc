source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

fastkpc_cuda_residual_fixture <- function(seed, n) {
  set.seed(seed)
  z1 <- runif(n, -2, 2)
  z2 <- runif(n, -2, 2)
  z3 <- runif(n, -2, 2)
  y <- sin(z1) + cos(z2) + 0.25 * z3 + rnorm(n, sd = 0.08)
  list(
    y = y,
    cases = list(
      empty = matrix(numeric(0), nrow = n, ncol = 0),
      one = cbind(z1 = z1),
      two = cbind(z1 = z1, z2 = z2),
      three = cbind(z1 = z1, z2 = z2, z3 = z3)
    )
  )
}

fastkpc_residual_values <- function(fit) {
  fit$residuals %||% fit$residual
}

fastkpc_elapsed_value <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed = max(elapsed, .Machine$double.eps))
}

validate_cuda_fastspline_residuals <- function(seed = 108, n = 96) {
  build_fastkpc_native()
  build_fastkpc_cuda_native()
  fixture <- fastkpc_cuda_residual_fixture(seed, n)
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  rows <- list()
  raw <- list()

  for (case in names(fixture$cases)) {
    S <- fixture$cases[[case]]
    result <- tryCatch({
      cpu <- fastspline_residual(fixture$y, S, fastspline_params = params)
      cuda <- fastspline_residual_cuda(fixture$y, S, fastspline_params = params,
                                       fallback = FALSE)
      residual_diff <- max(abs(as.numeric(fastkpc_residual_values(cpu)) -
                                 as.numeric(cuda$residuals)))
      fitted_diff <- max(abs(as.numeric(cpu$fitted) - as.numeric(cuda$fitted)))
      rel_rss <- abs(cuda$rss - cpu$rss) / max(1, abs(cpu$rss))
      raw[[case]] <- list(cpu = cpu, cuda = cuda)
      list(status = "ok", error_message = "", max_abs_residual_diff = residual_diff,
           max_abs_fitted_diff = fitted_diff, relative_rss_diff = rel_rss,
           cpu_rss = cpu$rss, cuda_rss = cuda$rss,
           cpu_lambda = cpu$selected_lambda,
           cuda_lambda = cuda$selected_lambda)
    }, error = function(e) {
      list(status = "error", error_message = conditionMessage(e),
           max_abs_residual_diff = NA_real_, max_abs_fitted_diff = NA_real_,
           relative_rss_diff = NA_real_, cpu_rss = NA_real_,
           cuda_rss = NA_real_, cpu_lambda = NA_real_,
           cuda_lambda = NA_real_)
    })
    rows[[length(rows) + 1L]] <- c(list(case = case), result)
  }

  list(
    config = list(seed = seed, n = n, fastspline_params = params),
    cases = do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE)),
    raw = raw
  )
}

benchmark_cuda_fastspline_residuals <- function(seed = 109, n = 160,
                                                repeats = 3) {
  build_fastkpc_native()
  build_fastkpc_cuda_native()
  fixture <- fastkpc_cuda_residual_fixture(seed, n)
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  rows <- list()

  for (case in names(fixture$cases)) {
    S <- fixture$cases[[case]]
    for (repeat_id in seq_len(repeats)) {
      cpu <- tryCatch({
        timed <- fastkpc_elapsed_value(
          fastspline_residual(fixture$y, S, fastspline_params = params)
        )
        list(device = "cpu", "repeat" = repeat_id, elapsed_sec = timed$elapsed,
             residual_backend = "fastSpline", residual_device = "cpu",
             status = "ok", error_message = "")
      }, error = function(e) {
        list(device = "cpu", "repeat" = repeat_id, elapsed_sec = NA_real_,
             residual_backend = "fastSpline", residual_device = "cpu",
             status = "error", error_message = conditionMessage(e))
      })
      cuda <- tryCatch({
        timed <- fastkpc_elapsed_value(
          fastspline_residual_cuda(fixture$y, S, fastspline_params = params,
                                   fallback = FALSE)
        )
        list(device = "cuda", "repeat" = repeat_id, elapsed_sec = timed$elapsed,
             residual_backend = "fastSpline", residual_device = "cuda",
             status = "ok", error_message = "")
      }, error = function(e) {
        list(device = "cuda", "repeat" = repeat_id, elapsed_sec = NA_real_,
             residual_backend = "fastSpline", residual_device = "cuda",
             status = "error", error_message = conditionMessage(e))
      })
      rows[[length(rows) + 1L]] <- c(list(case = case), cpu)
      rows[[length(rows) + 1L]] <- c(list(case = case), cuda)
    }
  }

  timings <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  names(timings)[names(timings) == "repeat."] <- "repeat"
  list(
    config = list(seed = seed, n = n, repeats = repeats,
                  fastspline_params = params),
    timings = timings,
    summary = aggregate(elapsed_sec ~ case + residual_device, timings,
                        function(x) mean(x, na.rm = TRUE))
  )
}

validate_cuda_fastspline_residual_batch <- function(seed = 510,
                                                    n = 128,
                                                    p = 6,
                                                    fastspline_params = list(knots = 8,
                                                                             lambda_count = 17,
                                                                             ridge = 1e-8)) {
  build_fastkpc_native()
  build_fastkpc_cuda_native()
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    sin(z1) + stats::rnorm(n, sd = 0.08),
    cos(z1) + stats::rnorm(n, sd = 0.08),
    z1 * z2 + stats::rnorm(n, sd = 0.08),
    sin(z2) + stats::rnorm(n, sd = 0.08),
    cos(z2) + stats::rnorm(n, sd = 0.08),
    stats::rnorm(n)
  )
  if (ncol(data) > p) data <- data[, seq_len(p), drop = FALSE]
  targets <- seq_len(min(5L, ncol(data) - 1L))
  conditioning_sets <- rep(list(ncol(data)), length(targets))
  batch <- fastspline_residual_batch_cuda(
    data, targets, conditioning_sets,
    fastspline_params = fastspline_params,
    fallback = FALSE
  )
  rows <- vector("list", length(targets))
  for (k in seq_along(targets)) {
    S <- data[, conditioning_sets[[k]], drop = FALSE]
    cpu <- fastspline_residual(data[, targets[[k]]], S,
                               fastspline_params = fastspline_params)
    residual_diff <- max(abs(as.numeric(batch$residuals[, k]) -
                               as.numeric(fastkpc_residual_values(cpu))))
    fitted_diff <- max(abs(as.numeric(batch$fitted[, k]) -
                             as.numeric(cpu$fitted)))
    rel_rss <- abs(batch$rss[[k]] - cpu$rss) / max(1, abs(cpu$rss))
    rows[[k]] <- data.frame(
      fit = k,
      target = targets[[k]],
      conditioning_size = length(conditioning_sets[[k]]),
      status = if (residual_diff < 1e-7 && fitted_diff < 1e-7 &&
                   rel_rss < 1e-8) "ok" else "diff",
      max_abs_residual_diff = residual_diff,
      max_abs_fitted_diff = fitted_diff,
      relative_rss_diff = rel_rss,
      true_batched = isTRUE(batch$diagnostics[[k]]$true_batched),
      stringsAsFactors = FALSE
    )
  }
  list(
    config = list(seed = seed, n = n, p = p,
                  fastspline_params = fastspline_params),
    cases = do.call(rbind, rows),
    batch_diagnostics = batch$batch_diagnostics,
    raw = batch
  )
}

benchmark_cuda_fastspline_residual_batch <- function(seed = 511,
                                                     n = 160,
                                                     repeats = 3,
                                                     fastspline_params = list(knots = 8,
                                                                              lambda_count = 17,
                                                                              ridge = 1e-8)) {
  build_fastkpc_native()
  build_fastkpc_cuda_native()
  set.seed(seed)
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    sin(z1) + stats::rnorm(n, sd = 0.08),
    cos(z1) + stats::rnorm(n, sd = 0.08),
    z1 * z2 + stats::rnorm(n, sd = 0.08),
    sin(z2) + stats::rnorm(n, sd = 0.08),
    cos(z2) + stats::rnorm(n, sd = 0.08),
    stats::rnorm(n)
  )
  targets <- 1:5
  conditioning_sets <- rep(list(6L), length(targets))
  rows <- list()
  true_batch_result <- NULL
  for (repeat_id in seq_len(repeats)) {
    single <- tryCatch({
      timed <- fastkpc_elapsed_value({
        fits <- vector("list", length(targets))
        for (k in seq_along(targets)) {
          fits[[k]] <- fastspline_residual_cuda(
            data[, targets[[k]]],
            data[, conditioning_sets[[k]], drop = FALSE],
            fastspline_params = fastspline_params,
            fallback = FALSE
          )
        }
        fits
      })
      list(mode = "single_loop", "repeat" = repeat_id,
           elapsed_sec = timed$elapsed, status = "ok", error_message = "")
    }, error = function(e) {
      list(mode = "single_loop", "repeat" = repeat_id,
           elapsed_sec = NA_real_, status = "error",
           error_message = conditionMessage(e))
    })
    batch <- tryCatch({
      timed <- fastkpc_elapsed_value(
        fastspline_residual_batch_cuda(
          data, targets, conditioning_sets,
          fastspline_params = fastspline_params,
          fallback = FALSE
        )
      )
      true_batch_result <- timed$value
      list(mode = "true_batch", "repeat" = repeat_id,
           elapsed_sec = timed$elapsed, status = "ok", error_message = "")
    }, error = function(e) {
      list(mode = "true_batch", "repeat" = repeat_id,
           elapsed_sec = NA_real_, status = "error",
           error_message = conditionMessage(e))
    })
    rows[[length(rows) + 1L]] <- single
    rows[[length(rows) + 1L]] <- batch
  }
  timings <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  list(
    config = list(seed = seed, n = n, repeats = repeats,
                  fastspline_params = fastspline_params),
    timings = timings,
    summary = aggregate(elapsed_sec ~ mode, timings,
                        function(x) mean(x, na.rm = TRUE)),
    batch_diagnostics = true_batch_result$batch_diagnostics %||% list()
  )
}
