if (!exists("dcov_gamma_exact", mode = "function")) {
  source("fastkpc/R/dcov_exact.R")
}
if (!exists("fast_dcov_batch_cuda", mode = "function")) {
  source("fastkpc/R/cuda_native.R")
}

fastkpc_dcc_gamma_log_p_diff <- function(a, b) {
  a <- as.numeric(a)[1L]
  b <- as.numeric(b)[1L]
  if (!is.finite(a) || !is.finite(b) || a <= 0 || b <= 0) return(NA_real_)
  abs(log(a) - log(b))
}

fastkpc_dcc_gamma_parity_bool <- function(value) {
  if (isTRUE(value)) return(TRUE)
  if (identical(value, FALSE)) return(FALSE)
  NA
}

fastkpc_dcc_gamma_error_message <- function(expr) {
  tryCatch({
    force(expr)
    ""
  }, error = function(e) conditionMessage(e))
}

fastkpc_dcc_gamma_near_alpha_pair <- function(n = 80, alpha = 0.05,
                                              seed = 7301) {
  set.seed(seed)
  best <- NULL
  for (rho in seq(0, 0.5, length.out = 51L)) {
    x <- stats::rnorm(n)
    y <- rho * x + stats::rnorm(n)
    p <- tryCatch(dcov_gamma_exact(x, y)$p.value, error = function(e) NA_real_)
    if (!is.finite(p)) next
    distance <- abs(log(p) - log(alpha))
    if (is.null(best) || distance < best$distance) {
      best <- list(x = x, y = y, p = p, distance = distance)
    }
  }
  if (is.null(best)) {
    stop("failed to generate a finite near-alpha dcc.gamma pair", call. = FALSE)
  }
  list(x = matrix(best$x, ncol = 1L), y = matrix(best$y, ncol = 1L))
}

fastkpc_dcc_gamma_cuda_parity_scenarios <- function(alpha = 0.05) {
  set.seed(7201)
  ordinary_x <- matrix(stats::rnorm(90 * 4), 90, 4)
  ordinary_y <- ordinary_x * rep(seq(0.05, 0.25, length.out = 4L), each = 90) +
    matrix(stats::rnorm(90 * 4), 90, 4)

  set.seed(7202)
  ties_x <- matrix(sample(seq(-2, 2), 70 * 3, replace = TRUE), 70, 3)
  ties_y <- ties_x * rep(c(0.0, 0.15, -0.25), each = 70) +
    matrix(sample(seq(-1, 1), 70 * 3, replace = TRUE), 70, 3)

  set.seed(7203)
  small_x <- matrix(stats::rnorm(8 * 2), 8, 2)
  small_y <- small_x * rep(c(0.1, 0.35), each = 8) +
    matrix(stats::rnorm(8 * 2), 8, 2)

  set.seed(7204)
  semantic_x <- matrix(stats::rnorm(85 * 3), 85, 3)
  semantic_y <- matrix(stats::rnorm(85 * 3), 85, 3)

  set.seed(7205)
  near_constant_x <- matrix(1 + stats::rnorm(75) * 1e-8, 75, 1)
  near_constant_y <- matrix(stats::rnorm(75), 75, 1)

  near_alpha <- fastkpc_dcc_gamma_near_alpha_pair(alpha = alpha)

  invalid_n_x <- matrix(seq_len(5L), 5, 1)
  invalid_n_y <- matrix(seq_len(5L), 5, 1)

  nonfinite_x <- matrix(stats::rnorm(40), 40, 1)
  nonfinite_y <- matrix(stats::rnorm(40), 40, 1)
  nonfinite_x[3L, 1L] <- Inf

  list(
    list(scenario_id = "ordinary_batch", x = ordinary_x, y = ordinary_y,
         index = 1, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "ties_batch", x = ties_x, y = ties_y,
         index = 1, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "small_n_valid", x = small_x, y = small_y,
         index = 1, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "semantic_index_1_5", x = semantic_x, y = semantic_y,
         index = 1.5, legacy_index = FALSE, expect_error = FALSE),
    list(scenario_id = "legacy_index_ignored", x = semantic_x, y = semantic_y,
         index = 1.5, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "near_alpha", x = near_alpha$x, y = near_alpha$y,
         index = 1, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "near_constant", x = near_constant_x, y = near_constant_y,
         index = 1, legacy_index = TRUE, expect_error = FALSE),
    list(scenario_id = "invalid_n", x = invalid_n_x, y = invalid_n_y,
         index = 1, legacy_index = TRUE, expect_error = TRUE),
    list(scenario_id = "nonfinite", x = nonfinite_x, y = nonfinite_y,
         index = 1, legacy_index = TRUE, expect_error = TRUE)
  )
}

fastkpc_dcc_gamma_cuda_parity_case <- function(case, alpha = 0.05) {
  x <- if (is.matrix(case$x)) case$x else matrix(as.numeric(case$x), ncol = 1L)
  y <- if (is.matrix(case$y)) case$y else matrix(as.numeric(case$y), ncol = 1L)
  storage.mode(x) <- "double"
  storage.mode(y) <- "double"
  batch <- ncol(x)

  cuda_error <- fastkpc_dcc_gamma_error_message(
    cuda <- fast_dcov_batch_cuda(
      x, y, index = case$index, legacy_index = case$legacy_index
    )
  )
  cuda_ok <- identical(cuda_error, "")

  rows <- vector("list", batch)
  for (k in seq_len(batch)) {
    cpu_error <- fastkpc_dcc_gamma_error_message(
      cpu <- dcov_gamma_exact(
        x[, k], y[, k], index = case$index,
        legacy_index = case$legacy_index
      )
    )
    cpu_ok <- identical(cpu_error, "")
    status <- if (cpu_ok && cuda_ok) {
      "ok"
    } else if (!cpu_ok && !cuda_ok) {
      "error_parity"
    } else {
      "error_mismatch"
    }

    cpu_p <- if (cpu_ok) as.numeric(cpu$p.value) else NA_real_
    cuda_p <- if (cuda_ok) as.numeric(cuda$p.value[k]) else NA_real_
    cpu_nV2 <- if (cpu_ok) as.numeric(cpu$statistic) else NA_real_
    cuda_nV2 <- if (cuda_ok) as.numeric(cuda$nV2[k]) else NA_real_
    cpu_mean <- if (cpu_ok) as.numeric(cpu$estimates[2L]) else NA_real_
    cuda_mean <- if (cuda_ok) as.numeric(cuda$mean[k]) else NA_real_
    cpu_var <- if (cpu_ok) as.numeric(cpu$estimates[3L]) else NA_real_
    cuda_var <- if (cuda_ok) as.numeric(cuda$variance[k]) else NA_real_
    cpu_delete <- if (cpu_ok) isTRUE(cpu_p > alpha) else NA
    cuda_delete <- if (cuda_ok) isTRUE(cuda_p > alpha) else NA

    rows[[k]] <- data.frame(
      scenario_id = as.character(case$scenario_id),
      column_id = as.integer(k),
      n = as.integer(nrow(x)),
      batch = as.integer(batch),
      index = as.numeric(case$index),
      legacy_index = isTRUE(case$legacy_index),
      alpha = as.numeric(alpha),
      expected_error = isTRUE(case$expect_error),
      status = status,
      error_cpu = cpu_error,
      error_cuda = cuda_error,
      cpu_p = cpu_p,
      cuda_p = cuda_p,
      p_abs_diff = abs(cpu_p - cuda_p),
      log_p_diff = fastkpc_dcc_gamma_log_p_diff(cpu_p, cuda_p),
      cpu_statistic = cpu_nV2,
      cuda_nV2 = cuda_nV2,
      stat_abs_diff = abs(cpu_nV2 - cuda_nV2),
      cpu_mean = cpu_mean,
      cuda_mean = cuda_mean,
      mean_abs_diff = abs(cpu_mean - cuda_mean),
      cpu_variance = cpu_var,
      cuda_variance = cuda_var,
      variance_abs_diff = abs(cpu_var - cuda_var),
      cpu_delete = fastkpc_dcc_gamma_parity_bool(cpu_delete),
      cuda_delete = fastkpc_dcc_gamma_parity_bool(cuda_delete),
      decision_flip = if (cpu_ok && cuda_ok) !identical(cpu_delete, cuda_delete) else NA,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

fastkpc_dcc_gamma_cuda_parity_summary <- function(rows) {
  valid <- rows[rows$status == "ok", , drop = FALSE]
  max_or_na <- function(x) {
    x <- as.numeric(x)
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    max(x)
  }
  data.frame(
    rows = as.integer(nrow(rows)),
    valid_rows = as.integer(nrow(valid)),
    error_parity_rows = as.integer(sum(rows$status == "error_parity")),
    error_mismatch_rows = as.integer(sum(rows$status == "error_mismatch")),
    max_p_abs_diff = max_or_na(valid$p_abs_diff),
    max_log_p_diff = max_or_na(valid$log_p_diff),
    max_stat_abs_diff = max_or_na(valid$stat_abs_diff),
    max_mean_abs_diff = max_or_na(valid$mean_abs_diff),
    max_variance_abs_diff = max_or_na(valid$variance_abs_diff),
    decision_flips = as.integer(sum(valid$decision_flip %in% TRUE)),
    stringsAsFactors = FALSE
  )
}

fastkpc_dcc_gamma_json_number <- function(value) {
  value <- as.numeric(value)[1L]
  if (!is.finite(value)) return("null")
  format(signif(value, 8), scientific = FALSE, trim = TRUE)
}

fastkpc_write_dcc_gamma_cuda_parity_summary <- function(summary, paths) {
  json <- c(
    "{",
    paste0('  "parity_csv": "', basename(paths$csv), '",'),
    paste0('  "rows": ', as.integer(summary$rows[[1L]]), ","),
    paste0('  "valid_rows": ', as.integer(summary$valid_rows[[1L]]), ","),
    paste0('  "error_parity_rows": ', as.integer(summary$error_parity_rows[[1L]]), ","),
    paste0('  "error_mismatch_rows": ', as.integer(summary$error_mismatch_rows[[1L]]), ","),
    paste0('  "max_p_abs_diff": ',
           fastkpc_dcc_gamma_json_number(summary$max_p_abs_diff[[1L]]), ","),
    paste0('  "max_log_p_diff": ',
           fastkpc_dcc_gamma_json_number(summary$max_log_p_diff[[1L]]), ","),
    paste0('  "decision_flips": ', as.integer(summary$decision_flips[[1L]])),
    "}"
  )
  writeLines(json, paths$summary_json)

  md <- c(
    "# dcc.gamma CUDA Parity Summary",
    "",
    paste0("- CSV: `", basename(paths$csv), "`"),
    paste0("- Valid rows: ", summary$valid_rows[[1L]], " / ", summary$rows[[1L]]),
    paste0("- Error parity rows: ", summary$error_parity_rows[[1L]]),
    paste0("- Error mismatch rows: ", summary$error_mismatch_rows[[1L]]),
    paste0("- Max p abs diff: ",
           fastkpc_dcc_gamma_json_number(summary$max_p_abs_diff[[1L]])),
    paste0("- Max log-p diff: ",
           fastkpc_dcc_gamma_json_number(summary$max_log_p_diff[[1L]])),
    paste0("- Decision flips: ", summary$decision_flips[[1L]])
  )
  writeLines(md, paths$summary_md)
}

fastkpc_run_dcc_gamma_cuda_parity <- function(
    output_dir = file.path("fastkpc", "artifacts", "dcc_gamma_cuda_parity"),
    alpha = 0.05,
    scenarios = NULL) {
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for dcc.gamma CUDA parity", call. = FALSE)
  }
  if (is.null(scenarios)) {
    scenarios <- fastkpc_dcc_gamma_cuda_parity_scenarios(alpha = alpha)
  }
  rows <- do.call(rbind, lapply(
    scenarios,
    fastkpc_dcc_gamma_cuda_parity_case,
    alpha = alpha
  ))
  summary <- fastkpc_dcc_gamma_cuda_parity_summary(rows)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    csv = file.path(output_dir, "dcc_gamma_cpu_cuda_parity.csv"),
    summary_csv = file.path(output_dir, "dcc_gamma_cpu_cuda_parity_summary.csv"),
    summary_json = file.path(output_dir, "dcc_gamma_cpu_cuda_parity_summary.json"),
    summary_md = file.path(output_dir, "dcc_gamma_cpu_cuda_parity_summary.md")
  )
  utils::write.csv(rows, paths$csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  fastkpc_write_dcc_gamma_cuda_parity_summary(summary, paths)

  list(
    rows = rows,
    summary = summary,
    paths = paths,
    path = paths$csv
  )
}
