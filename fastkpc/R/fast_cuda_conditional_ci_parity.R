if (!exists("dcov_gamma_exact", mode = "function")) {
  source("fastkpc/R/dcov_exact.R")
}
if (!exists("fastspline_residual", mode = "function")) {
  source("fastkpc/R/native.R")
}
if (!exists("fastspline_residual_cuda", mode = "function")) {
  source("fastkpc/R/cuda_native.R")
}

if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

fastkpc_fast_cuda_rel_l2 <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  denom <- sqrt(sum(b^2))
  if (!is.finite(denom) || denom <= .Machine$double.eps) {
    return(sqrt(sum((a - b)^2)))
  }
  sqrt(sum((a - b)^2)) / denom
}

fastkpc_fast_cuda_log_p_diff <- function(a, b) {
  a <- as.numeric(a)[1L]
  b <- as.numeric(b)[1L]
  if (!is.finite(a) || !is.finite(b) || a <= 0 || b <= 0) return(NA_real_)
  abs(log(a) - log(b))
}

fastkpc_fast_cuda_s_key <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("")
  paste(S, collapse = "|")
}

fastkpc_fast_cuda_conditional_ci_scenarios <- function() {
  set.seed(8401)
  n <- 96
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  z3 <- stats::runif(n, -2, 2)
  base <- cbind(
    x1 = sin(z1) + stats::rnorm(n, sd = 0.08),
    x2 = cos(z1) + 0.2 * z2 + stats::rnorm(n, sd = 0.08),
    s1 = z1,
    s2 = z2,
    s3 = z3,
    noise = stats::rnorm(n)
  )
  shifted <- base
  shifted[, 3L] <- shifted[, 3L] + 10
  shifted[, 4L] <- shifted[, 4L] - 4
  scaled <- base
  scaled[, 1L] <- scaled[, 1L] * 3
  scaled[, 2L] <- scaled[, 2L] * 0.5
  scaled[, 3L] <- scaled[, 3L] * 2

  set.seed(8402)
  nt <- 90
  tied_s <- sample(seq(-2, 2, length.out = 9L), nt, replace = TRUE)
  tied <- cbind(
    x1 = sin(tied_s) + stats::rnorm(nt, sd = 0.12),
    x2 = cos(tied_s) + stats::rnorm(nt, sd = 0.12),
    s1 = tied_s,
    s2 = stats::rnorm(nt),
    noise = stats::rnorm(nt)
  )

  list(
    list(scenario_id = "one_s", data = base, x = 1L, y = 2L, S = 3L),
    list(scenario_id = "two_s", data = base, x = 1L, y = 2L, S = c(3L, 4L)),
    list(scenario_id = "translation", data = shifted, x = 1L, y = 2L,
         S = c(3L, 4L)),
    list(scenario_id = "scale", data = scaled, x = 1L, y = 2L,
         S = c(3L, 5L)),
    list(scenario_id = "ties", data = tied, x = 1L, y = 2L,
         S = c(3L, 4L))
  )
}

fastkpc_fast_cuda_residual_values <- function(fit) {
  if (!is.null(fit$residuals)) return(as.numeric(fit$residuals))
  as.numeric(fit$residual)
}

fastkpc_fast_cuda_conditional_ci_case <- function(
    case,
    alpha = 0.05,
    index = 1,
    legacy_index = TRUE,
    fastspline_params = list(knots = 8, lambda_count = 17, ridge = 1e-8)) {
  data <- as.matrix(case$data)
  storage.mode(data) <- "double"
  x <- as.integer(case$x)
  y <- as.integer(case$y)
  S <- as.integer(case$S)
  S_data <- data[, S, drop = FALSE]

  fit_one <- function(target, device) {
    if (identical(device, "cpu")) {
      fastspline_residual(data[, target], S_data,
                          fastspline_params = fastspline_params)
    } else {
      fastspline_residual_cuda(data[, target], S_data,
                               fastspline_params = fastspline_params,
                               fallback = FALSE)
    }
  }

  cpu_x <- fit_one(x, "cpu")
  cpu_y <- fit_one(y, "cpu")
  cuda_x <- fit_one(x, "cuda")
  cuda_y <- fit_one(y, "cuda")

  cpu_rx <- fastkpc_fast_cuda_residual_values(cpu_x)
  cpu_ry <- fastkpc_fast_cuda_residual_values(cpu_y)
  cuda_rx <- fastkpc_fast_cuda_residual_values(cuda_x)
  cuda_ry <- fastkpc_fast_cuda_residual_values(cuda_y)

  cpu_ci <- dcov_gamma_exact(cpu_rx, cpu_ry, index = index,
                             legacy_index = legacy_index)
  cuda_ci <- fast_dcov_batch_cuda(matrix(cuda_rx, ncol = 1L),
                                  matrix(cuda_ry, ncol = 1L),
                                  index = index,
                                  legacy_index = legacy_index)
  cpu_on_cuda_residual <- dcov_gamma_exact(cuda_rx, cuda_ry, index = index,
                                           legacy_index = legacy_index)
  cuda_on_cpu_residual <- fast_dcov_batch_cuda(matrix(cpu_rx, ncol = 1L),
                                               matrix(cpu_ry, ncol = 1L),
                                               index = index,
                                               legacy_index = legacy_index)

  cpu_p <- as.numeric(cpu_ci$p.value)
  cuda_p <- as.numeric(cuda_ci$p.value[[1L]])
  cpu_delete <- cpu_p > alpha
  cuda_delete <- cuda_p > alpha

  data.frame(
    scenario_id = as.character(case$scenario_id),
    n = as.integer(nrow(data)),
    p = as.integer(ncol(data)),
    x = x,
    y = y,
    S_key = fastkpc_fast_cuda_s_key(S),
    conditioning_size = as.integer(length(S)),
    index = as.numeric(index),
    legacy_index = isTRUE(legacy_index),
    alpha = as.numeric(alpha),
    residual_rel_l2_x = fastkpc_fast_cuda_rel_l2(cuda_rx, cpu_rx),
    residual_rel_l2_y = fastkpc_fast_cuda_rel_l2(cuda_ry, cpu_ry),
    fitted_rel_l2_x = fastkpc_fast_cuda_rel_l2(cuda_x$fitted, cpu_x$fitted),
    fitted_rel_l2_y = fastkpc_fast_cuda_rel_l2(cuda_y$fitted, cpu_y$fitted),
    lambda_abs_diff_x = abs(as.numeric(cuda_x$selected_lambda) -
                              as.numeric(cpu_x$selected_lambda)),
    lambda_abs_diff_y = abs(as.numeric(cuda_y$selected_lambda) -
                              as.numeric(cpu_y$selected_lambda)),
    cpu_p = cpu_p,
    cuda_p = cuda_p,
    p_abs_diff = abs(cpu_p - cuda_p),
    log_p_diff = fastkpc_fast_cuda_log_p_diff(cpu_p, cuda_p),
    cpu_statistic = as.numeric(cpu_ci$statistic),
    cuda_nV2 = as.numeric(cuda_ci$nV2[[1L]]),
    stat_abs_diff = abs(as.numeric(cpu_ci$statistic) -
                          as.numeric(cuda_ci$nV2[[1L]])),
    cpu_mean = as.numeric(cpu_ci$estimates[2L]),
    cuda_mean = as.numeric(cuda_ci$mean[[1L]]),
    mean_abs_diff = abs(as.numeric(cpu_ci$estimates[2L]) -
                          as.numeric(cuda_ci$mean[[1L]])),
    cpu_variance = as.numeric(cpu_ci$estimates[3L]),
    cuda_variance = as.numeric(cuda_ci$variance[[1L]]),
    variance_abs_diff = abs(as.numeric(cpu_ci$estimates[3L]) -
                              as.numeric(cuda_ci$variance[[1L]])),
    cpu_on_cuda_residual_p = as.numeric(cpu_on_cuda_residual$p.value),
    cuda_on_cpu_residual_p = as.numeric(cuda_on_cpu_residual$p.value[[1L]]),
    residual_only_p_abs_diff =
      abs(cpu_p - as.numeric(cpu_on_cuda_residual$p.value)),
    dcov_only_p_abs_diff =
      abs(cpu_p - as.numeric(cuda_on_cpu_residual$p.value[[1L]])),
    cpu_delete = isTRUE(cpu_delete),
    cuda_delete = isTRUE(cuda_delete),
    decision_flip = !identical(isTRUE(cpu_delete), isTRUE(cuda_delete)),
    cuda_residual_device_x = as.character(cuda_x$residual_device %||% "cuda"),
    cuda_residual_device_y = as.character(cuda_y$residual_device %||% "cuda"),
    fallback_used_x = isTRUE(cuda_x$fallback_used),
    fallback_used_y = isTRUE(cuda_y$fallback_used),
    stringsAsFactors = FALSE
  )
}

fastkpc_fast_cuda_conditional_ci_summary <- function(rows) {
  max_or_na <- function(x) {
    x <- as.numeric(x)
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    max(x)
  }
  data.frame(
    rows = as.integer(nrow(rows)),
    max_residual_rel_l2 =
      max_or_na(c(rows$residual_rel_l2_x, rows$residual_rel_l2_y)),
    max_fitted_rel_l2 =
      max_or_na(c(rows$fitted_rel_l2_x, rows$fitted_rel_l2_y)),
    max_p_abs_diff = max_or_na(rows$p_abs_diff),
    max_log_p_diff = max_or_na(rows$log_p_diff),
    max_stat_abs_diff = max_or_na(rows$stat_abs_diff),
    decision_flips = as.integer(sum(rows$decision_flip %in% TRUE)),
    fallback_count = as.integer(sum(rows$fallback_used_x %in% TRUE) +
                                  sum(rows$fallback_used_y %in% TRUE)),
    stringsAsFactors = FALSE
  )
}

fastkpc_run_fast_cuda_conditional_ci_parity <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_conditional_ci_parity"),
    alpha = 0.05,
    index = 1,
    legacy_index = TRUE,
    scenarios = NULL,
    fastspline_params = list(knots = 8, lambda_count = 17, ridge = 1e-8)) {
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for fast CUDA conditional CI parity", call. = FALSE)
  }
  if (is.null(scenarios)) {
    scenarios <- fastkpc_fast_cuda_conditional_ci_scenarios()
  }
  rows <- do.call(rbind, lapply(
    scenarios,
    fastkpc_fast_cuda_conditional_ci_case,
    alpha = alpha,
    index = index,
    legacy_index = legacy_index,
    fastspline_params = fastspline_params
  ))
  summary <- fastkpc_fast_cuda_conditional_ci_summary(rows)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    csv = file.path(output_dir, "fast_cuda_conditional_ci_parity.csv"),
    summary_csv = file.path(output_dir,
                            "fast_cuda_conditional_ci_parity_summary.csv"),
    summary_md = file.path(output_dir,
                           "fast_cuda_conditional_ci_parity_summary.md")
  )
  utils::write.csv(rows, paths$csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  md <- c(
    "# Fast CUDA Conditional CI Parity",
    "",
    paste0("- CSV: `", basename(paths$csv), "`"),
    paste0("- Rows: ", summary$rows[[1L]]),
    paste0("- Max residual rel-L2: ", signif(summary$max_residual_rel_l2[[1L]], 8)),
    paste0("- Max p abs diff: ", signif(summary$max_p_abs_diff[[1L]], 8)),
    paste0("- Decision flips: ", summary$decision_flips[[1L]]),
    paste0("- Fallback count: ", summary$fallback_count[[1L]])
  )
  writeLines(md, paths$summary_md)

  list(rows = rows, summary = summary, paths = paths, path = paths$csv)
}
