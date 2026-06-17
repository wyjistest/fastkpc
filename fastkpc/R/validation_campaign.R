source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/validation_scenarios.R")
source("fastkpc/R/wanpdag_validation.R")
if (file.exists("fastkpc/R/mgcv_extract_validation.R")) {
  source("fastkpc/R/mgcv_extract_validation.R")
}

fastkpc_run_id <- function(scenario, seed, n, engine, residual_backend,
                           residual_device = "auto", scheduler = "auto",
                           orientation_residual_device = "auto",
                           ci_method = "dcc.gamma") {
  paste(scenario, seed, n, engine, residual_backend, residual_device, scheduler,
        orientation_residual_device, ci_method, sep = "-")
}

fastkpc_safe_run <- function(expr) {
  tryCatch(
    list(status = "ok", value = force(expr), error_message = ""),
    error = function(e) {
      list(status = "error", value = NULL, error_message = conditionMessage(e))
    }
  )
}

fastkpc_empty_df <- function(columns) {
  out <- stats::setNames(replicate(length(columns), logical(0), simplify = FALSE),
                         columns)
  as.data.frame(out, stringsAsFactors = FALSE)
}

fastkpc_bind_rows <- function(rows, columns) {
  if (length(rows) == 0L) return(fastkpc_empty_df(columns))
  rows <- lapply(rows, function(row) {
    missing <- setdiff(columns, names(row))
    for (name in missing) row[[name]] <- NA
    row[columns]
  })
  out <- do.call(rbind, lapply(rows, as.data.frame, stringsAsFactors = FALSE))
  rownames(out) <- NULL
  out
}

fastkpc_max_abs_matrix_diff <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NA_real_)
  a <- as.matrix(a)
  b <- as.matrix(b)
  if (!identical(dim(a), dim(b))) return(NA_real_)
  max(abs(as.numeric(a) - as.numeric(b)), na.rm = TRUE)
}

fastkpc_bool_identical_matrix <- function(a, b) {
  if (is.null(a) || is.null(b)) return(FALSE)
  identical(as.matrix(a), as.matrix(b))
}

fastkpc_flatten_cache <- function(result, run_id) {
  meta <- list(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    orientation_residual_device =
      result$config$orientation_residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto"
  )
  sections <- result$cache
  rows <- list()
  for (section in names(sections)) {
    cache <- sections[[section]]
    rows[[length(rows) + 1L]] <- c(meta, list(
      section = section,
      requests = as.integer(cache$requests %||% 0L),
      hits = as.integer(cache$hits %||% 0L),
      misses = as.integer(cache$misses %||% 0L),
      computations = as.integer(cache$computations %||% 0L),
      stored_vectors = as.integer(cache$stored_vectors %||% 0L),
      stored_values = as.integer(cache$stored_values %||% 0L)
    ))
  }
  fastkpc_bind_rows(rows, c("run_id", "scenario", "seed", "n", "engine",
                            "residual_backend", "residual_device",
                            "orientation_residual_device", "scheduler",
                            "section", "requests", "hits", "misses",
                            "computations", "stored_vectors", "stored_values"))
}

fastkpc_flatten_timings <- function(result, run_id) {
  timings <- result$timings
  if (nrow(timings) == 0L) {
    return(fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                              "residual_backend", "residual_device", "scheduler",
                              "stage", "elapsed_sec")))
  }
  data.frame(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    orientation_residual_device =
      result$config$orientation_residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto",
    stage = timings$stage,
    elapsed_sec = timings$elapsed_sec,
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_orientation_counts <- function(result, run_id) {
  counts <- if (is.null(result$orientation)) list() else result$orientation$counts
  data.frame(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    orientation_residual_device =
      result$config$orientation_residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto",
    collider = as.integer(counts$collider %||% 0L),
    rule1 = as.integer(counts$rule1 %||% 0L),
    rule2 = as.integer(counts$rule2 %||% 0L),
    rule3 = as.integer(counts$rule3 %||% 0L),
    generalized = as.integer(counts$generalized %||% 0L),
    regrvonps_calls = as.integer(counts$regrvonps_calls %||% 0L),
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_ci_method_diagnostics <- function(result, run_id) {
  skeleton_diag <- result$skeleton$ci_diagnostics %||% list()
  orientation_diag <- if (is.null(result$orientation)) {
    list()
  } else {
    result$orientation$ci_diagnostics %||% list()
  }
  data.frame(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    orientation_residual_device =
      result$config$orientation_residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto",
    ci_method = result$config$ci_method %||% "dcc.gamma",
    ci_backend = result$config$ci_backend %||% NA_character_,
    ci_backend_requested = result$config$ci_backend_requested %||% NA_character_,
    ci_backend_reason = result$config$ci_backend_reason %||% "",
    cuda_hsic_requested = isTRUE(result$config$cuda_hsic_requested),
    cuda_hsic_used = isTRUE(result$config$cuda_hsic_used),
    ci_dcc_gamma_tests =
      as.integer(skeleton_diag$ci_dcc_gamma_tests %||% 0L),
    ci_hsic_gamma_tests =
      as.integer(skeleton_diag$ci_hsic_gamma_tests %||% 0L),
    ci_hsic_perm_tests =
      as.integer(skeleton_diag$ci_hsic_perm_tests %||% 0L),
    ci_hsic_permutation_replicates =
      as.integer(skeleton_diag$ci_hsic_permutation_replicates %||% 0L),
    ci_hsic_gamma_cuda_tests =
      as.integer(skeleton_diag$ci_hsic_gamma_cuda_tests %||% 0L),
    ci_hsic_perm_cuda_tests =
      as.integer(skeleton_diag$ci_hsic_perm_cuda_tests %||% 0L),
    ci_hsic_cuda_batches =
      as.integer(skeleton_diag$ci_hsic_cuda_batches %||% 0L),
    ci_hsic_cuda_pairs =
      as.integer(skeleton_diag$ci_hsic_cuda_pairs %||% 0L),
    ci_hsic_cuda_fallback_tests =
      as.integer(skeleton_diag$ci_hsic_cuda_fallback_tests %||% 0L),
    ci_tests =
      as.integer((skeleton_diag$ci_dcc_gamma_tests %||% 0L) +
                   (skeleton_diag$ci_hsic_gamma_tests %||% 0L) +
                   (skeleton_diag$ci_hsic_perm_tests %||% 0L)),
    regrvonps_dcc_gamma_tests =
      as.integer(orientation_diag$regrvonps_dcc_gamma_tests %||% 0L),
    regrvonps_hsic_gamma_tests =
      as.integer(orientation_diag$regrvonps_hsic_gamma_tests %||% 0L),
    regrvonps_hsic_perm_tests =
      as.integer(orientation_diag$regrvonps_hsic_perm_tests %||% 0L),
    regrvonps_hsic_permutation_replicates =
      as.integer(orientation_diag$regrvonps_hsic_permutation_replicates %||% 0L),
    regrvonps_hsic_gamma_cuda_tests =
      as.integer(orientation_diag$regrvonps_hsic_gamma_cuda_tests %||% 0L),
    regrvonps_hsic_perm_cuda_tests =
      as.integer(orientation_diag$regrvonps_hsic_perm_cuda_tests %||% 0L),
    regrvonps_hsic_cuda_batches =
      as.integer(orientation_diag$regrvonps_hsic_cuda_batches %||% 0L),
    regrvonps_hsic_cuda_pairs =
      as.integer(orientation_diag$regrvonps_hsic_cuda_pairs %||% 0L),
    regrvonps_hsic_cuda_fallback_tests =
      as.integer(orientation_diag$regrvonps_hsic_cuda_fallback_tests %||% 0L),
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_orientation_device_diagnostics <- function(result, run_id) {
  diag <- if (is.null(result$orientation)) list() else
    result$orientation$diagnostics %||% list()
  data.frame(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto",
    orientation_residual_device_requested =
      result$config$orientation_residual_device_requested %||% "auto",
    orientation_residual_device =
      result$config$orientation_residual_device_used %||%
        diag$orientation_residual_device %||% NA_character_,
    orientation_residual_device_reason =
      result$config$orientation_residual_device_reason %||%
        diag$orientation_residual_device_reason %||% "",
    orientation_batch_size_requested =
      as.integer(result$config$orientation_batch_size %||%
                   diag$orientation_batch_size_requested %||% 0L),
    orientation_batch_size_used =
      as.integer(diag$orientation_batch_size_used %||% 0L),
    regrvonps_calls = as.integer(diag$regrvonps_calls %||% 0L),
    regrvonps_cuda_calls = as.integer(diag$regrvonps_cuda_calls %||% 0L),
    regrvonps_cpu_calls = as.integer(diag$regrvonps_cpu_calls %||% 0L),
    orientation_dcov_batches =
      as.integer(diag$orientation_dcov_batches %||% 0L),
    orientation_dcov_pairs = as.integer(diag$orientation_dcov_pairs %||% 0L),
    orientation_residual_fits =
      as.integer(diag$orientation_residual_fits %||% 0L),
    orientation_cuda_residual_fits =
      as.integer(diag$orientation_cuda_residual_fits %||% 0L),
    orientation_cpu_fallback_fits =
      as.integer(diag$orientation_cpu_fallback_fits %||% 0L),
    orientation_cache_requests =
      as.integer(diag$orientation_cache_requests %||% 0L),
    orientation_cache_hits = as.integer(diag$orientation_cache_hits %||% 0L),
    orientation_cache_computations =
      as.integer(diag$orientation_cache_computations %||% 0L),
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_scheduler_levels <- function(result, run_id) {
  level_columns <- c(
    "level", "tasks_planned", "tasks_evaluated", "tests_replayed",
    "tasks_ignored_after_delete", "deletions", "unconditional_tasks",
    "conditional_tasks", "unique_residual_requests", "dcov_batches",
    "residual_batches", "plan_elapsed_sec",
    "residual_prefetch_elapsed_sec", "ci_eval_elapsed_sec",
    "replay_elapsed_sec", "total_elapsed_sec"
  )
  diag <- result$skeleton$scheduler_diagnostics
  levels <- diag$levels %||% fastkpc_empty_df(level_columns)
  for (column in setdiff(level_columns, names(levels))) {
    levels[[column]] <- NA_real_
  }
  levels <- levels[, level_columns, drop = FALSE]
  if (nrow(levels) == 0L) {
    return(fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                              "residual_backend", "residual_device", "scheduler",
                              names(levels))))
  }
  cbind(
    data.frame(
      run_id = run_id,
      scenario = result$config$scenario %||% NA_character_,
      seed = result$config$seed %||% NA_integer_,
      n = result$data_info$n,
      engine = result$config$engine_used,
      residual_backend = result$config$residual_backend,
      residual_device = result$config$residual_device_requested %||% "auto",
      scheduler = result$config$scheduler_requested %||% "auto",
      stringsAsFactors = FALSE
    ),
    levels,
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_scheduler_batches <- function(result, run_id) {
  batch_columns <- c(
    "level", "batch_id", "kind", "start_task_id", "task_count", "n", "status",
    "groups", "true_batched_groups", "true_batched_fits", "single_fit_calls",
    "cpu_fallback_fits", "unique_designs", "duplicate_design_fits",
    "max_fits_per_design", "max_group_size", "min_group_size",
    "max_design_cols", "min_design_cols"
  )
  diag <- result$skeleton$scheduler_diagnostics
  batches <- diag$batches %||% fastkpc_empty_df(batch_columns)
  for (column in setdiff(batch_columns, names(batches))) {
    batches[[column]] <- NA_real_
  }
  batches <- batches[, batch_columns, drop = FALSE]
  if (nrow(batches) == 0L) {
    return(fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                              "residual_backend", "residual_device", "scheduler",
                              names(batches))))
  }
  cbind(
    data.frame(
      run_id = run_id,
      scenario = result$config$scenario %||% NA_character_,
      seed = result$config$seed %||% NA_integer_,
      n = result$data_info$n,
      engine = result$config$engine_used,
      residual_backend = result$config$residual_backend,
      residual_device = result$config$residual_device_requested %||% "auto",
      scheduler = result$config$scheduler_requested %||% "auto",
      stringsAsFactors = FALSE
    ),
    batches,
    stringsAsFactors = FALSE
  )
}

fastkpc_flatten_scheduler_residuals <- function(result, run_id) {
  diag <- result$skeleton$scheduler_diagnostics
  residuals <- diag$residuals %||% fastkpc_empty_df(c(
    "level", "request_id", "target", "conditioning_size", "residual_backend",
    "residual_device", "materialized", "fallback_used", "reason"
  ))
  if (nrow(residuals) == 0L) {
    return(fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                              "residual_backend", "residual_device_requested",
                              "scheduler", names(residuals))))
  }
  cbind(
    data.frame(
      run_id = run_id,
      scenario = result$config$scenario %||% NA_character_,
      seed = result$config$seed %||% NA_integer_,
      n = result$data_info$n,
      engine = result$config$engine_used,
      residual_backend_requested = result$config$residual_backend,
      residual_device_requested =
        result$config$residual_device_requested %||% "auto",
      scheduler = result$config$scheduler_requested %||% "auto",
      stringsAsFactors = FALSE
    ),
    residuals,
    stringsAsFactors = FALSE
  )
}

fastkpc_true_batch_summary <- function(result) {
  result$skeleton$scheduler_diagnostics$summary %||% list()
}

fastkpc_flatten_true_batched_residuals <- function(result, run_id) {
  summary <- fastkpc_true_batch_summary(result)
  data.frame(
    run_id = run_id,
    scenario = result$config$scenario %||% NA_character_,
    seed = result$config$seed %||% NA_integer_,
    n = result$data_info$n,
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    residual_device = result$config$residual_device_requested %||% "auto",
    scheduler = result$config$scheduler_requested %||% "auto",
    residual_batch_size = as.integer(result$config$residual_batch_size %||% 0L),
    cuda_residual_batch_groups =
      as.integer(summary$cuda_residual_batch_groups %||% 0L),
    cuda_residual_true_batched_groups =
      as.integer(summary$cuda_residual_true_batched_groups %||% 0L),
    cuda_residual_true_batched_fits =
      as.integer(summary$cuda_residual_true_batched_fits %||% 0L),
    cuda_residual_single_fit_calls =
      as.integer(summary$cuda_residual_single_fit_calls %||% 0L),
    cuda_residual_cpu_fallback_fits =
      as.integer(summary$cuda_residual_cpu_fallback_fits %||% 0L),
    cuda_residual_unique_designs =
      as.integer(summary$cuda_residual_unique_designs %||% 0L),
    cuda_residual_duplicate_design_fits =
      as.integer(summary$cuda_residual_duplicate_design_fits %||% 0L),
    cuda_residual_max_fits_per_design =
      as.integer(summary$cuda_residual_max_fits_per_design %||% 0L),
    status = "ok",
    stringsAsFactors = FALSE
  )
}

fastkpc_successful_results <- function(results) {
  Filter(function(x) identical(x$status, "ok") && inherits(x$result, "fastkpc_result"),
         results)
}

fastkpc_find_result <- function(results, scenario, seed, n, engine,
                                residual_backend, residual_device = NULL,
                                scheduler = NULL,
                                orientation_residual_device = NULL) {
  matches <- Filter(function(x) {
    identical(x$scenario, scenario) &&
      identical(as.integer(x$seed), as.integer(seed)) &&
      identical(as.integer(x$n), as.integer(n)) &&
      identical(x$engine, engine) &&
      identical(x$residual_backend, residual_backend) &&
      (is.null(residual_device) || identical(x$residual_device, residual_device)) &&
      (is.null(scheduler) || identical(x$scheduler, scheduler)) &&
      (is.null(orientation_residual_device) ||
         identical(x$orientation_residual_device,
                   orientation_residual_device)) &&
      identical(x$status, "ok")
  }, results)
  if (length(matches) == 0L) NULL else matches[[1L]]$result
}

fastkpc_campaign_orientation_device_diffs <- function(results) {
  ok <- fastkpc_successful_results(results)
  keys <- unique(vapply(ok, function(x) {
    paste(x$scenario, x$seed, x$n, x$engine, x$residual_backend,
          x$residual_device, x$scheduler, sep = "\r")
  }, character(1)))
  rows <- list()
  for (key in keys) {
    parts <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    scenario <- parts[[1L]]
    seed <- as.integer(parts[[2L]])
    n <- as.integer(parts[[3L]])
    engine <- parts[[4L]]
    residual_backend <- parts[[5L]]
    residual_device <- parts[[6L]]
    scheduler <- parts[[7L]]
    cpu <- fastkpc_find_result(ok, scenario, seed, n, engine, residual_backend,
                               residual_device, scheduler, "cpu")
    cuda <- fastkpc_find_result(ok, scenario, seed, n, engine, residual_backend,
                                residual_device, scheduler, "cuda")
    status <- if (is.null(cpu) || is.null(cuda)) "missing" else "ok"
    rows[[length(rows) + 1L]] <- list(
      scenario = scenario,
      seed = seed,
      n = n,
      engine = engine,
      residual_backend = residual_backend,
      residual_device = residual_device,
      scheduler = scheduler,
      left_orientation_residual_device = "cpu",
      right_orientation_residual_device = "cuda",
      pdag_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(fastkpc_extract_pdag(cpu),
                                                          fastkpc_extract_pdag(cuda)) else NA,
      skeleton_adjacency_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(cpu$skeleton$adjacency,
                                                          cuda$skeleton$adjacency) else NA,
      max_abs_pmax_diff =
        if (status == "ok") fastkpc_max_abs_matrix_diff(cpu$skeleton$pMax,
                                                        cuda$skeleton$pMax) else NA_real_,
      orientation_counts_identical =
        if (status == "ok") identical(cpu$orientation$counts,
                                      cuda$orientation$counts) else NA,
      status = status
    )
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "engine",
                            "residual_backend", "residual_device",
                            "scheduler", "left_orientation_residual_device",
                            "right_orientation_residual_device",
                            "pdag_identical", "skeleton_adjacency_identical",
                            "max_abs_pmax_diff",
                            "orientation_counts_identical", "status"))
}

fastkpc_campaign_pairwise_diffs <- function(results) {
  results <- fastkpc_successful_results(results)
  rows <- list()
  if (length(results) < 2L) {
    return(fastkpc_empty_df(c("comparison", "left_run_id", "right_run_id",
                              "pdag_identical", "skeleton_adjacency_identical",
                              "max_abs_pmax_diff", "max_abs_pdag_diff")))
  }
  for (i in seq_len(length(results) - 1L)) {
    for (j in (i + 1L):length(results)) {
      left <- results[[i]]
      right <- results[[j]]
      left_result <- left$result
      right_result <- right$result
      pdag_left <- fastkpc_extract_pdag(left_result)
      pdag_right <- fastkpc_extract_pdag(right_result)
      rows[[length(rows) + 1L]] <- list(
        comparison = "all_successful_pairs",
        left_run_id = left$run_id,
        right_run_id = right$run_id,
        pdag_identical = fastkpc_bool_identical_matrix(pdag_left, pdag_right),
        skeleton_adjacency_identical =
          fastkpc_bool_identical_matrix(left_result$skeleton$adjacency,
                                        right_result$skeleton$adjacency),
        max_abs_pmax_diff =
          fastkpc_max_abs_matrix_diff(left_result$skeleton$pMax,
                                      right_result$skeleton$pMax),
        max_abs_pdag_diff = fastkpc_max_abs_matrix_diff(pdag_left, pdag_right)
      )
    }
  }
  fastkpc_bind_rows(rows, c("comparison", "left_run_id", "right_run_id",
                            "pdag_identical", "skeleton_adjacency_identical",
                            "max_abs_pmax_diff", "max_abs_pdag_diff"))
}

fastkpc_campaign_cpu_cuda <- function(results) {
  ok <- fastkpc_successful_results(results)
  keys <- unique(vapply(ok, function(x) {
    paste(x$scenario, x$seed, x$n, x$residual_backend, x$residual_device,
          x$scheduler, sep = "\r")
  }, character(1)))
  rows <- list()
  for (key in keys) {
    parts <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    scenario <- parts[[1L]]
    seed <- as.integer(parts[[2L]])
    n <- as.integer(parts[[3L]])
    residual_backend <- parts[[4L]]
    residual_device <- parts[[5L]]
    scheduler <- parts[[6L]]
    cpu <- fastkpc_find_result(ok, scenario, seed, n, "cpu", residual_backend,
                               residual_device, scheduler)
    cuda <- fastkpc_find_result(ok, scenario, seed, n, "cuda", residual_backend,
                                residual_device, scheduler)
    status <- if (is.null(cpu) || is.null(cuda)) "missing" else "ok"
    rows[[length(rows) + 1L]] <- list(
      scenario = scenario,
      seed = seed,
      n = n,
      residual_backend = residual_backend,
      residual_device = residual_device,
      scheduler = scheduler,
      pdag_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(fastkpc_extract_pdag(cpu),
                                                          fastkpc_extract_pdag(cuda)) else NA,
      skeleton_adjacency_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(cpu$skeleton$adjacency,
                                                          cuda$skeleton$adjacency) else NA,
      max_abs_pmax_diff =
        if (status == "ok") fastkpc_max_abs_matrix_diff(cpu$skeleton$pMax,
                                                        cuda$skeleton$pMax) else NA_real_,
      orientation_counts_identical =
        if (status == "ok") identical(cpu$orientation$counts,
                                      cuda$orientation$counts) else NA,
      status = status
    )
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "residual_backend",
                            "residual_device", "scheduler", "pdag_identical",
                            "skeleton_adjacency_identical", "max_abs_pmax_diff",
                            "orientation_counts_identical", "status"))
}

fastkpc_campaign_linear_fastspline <- function(results) {
  ok <- fastkpc_successful_results(results)
  keys <- unique(vapply(ok, function(x) {
    paste(x$scenario, x$seed, x$n, x$engine, x$residual_device, x$scheduler,
          sep = "\r")
  }, character(1)))
  rows <- list()
  for (key in keys) {
    parts <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    scenario <- parts[[1L]]
    seed <- as.integer(parts[[2L]])
    n <- as.integer(parts[[3L]])
    engine <- parts[[4L]]
    residual_device <- parts[[5L]]
    scheduler <- parts[[6L]]
    linear <- fastkpc_find_result(ok, scenario, seed, n, engine, "linear",
                                  residual_device, scheduler)
    fastspline <- fastkpc_find_result(ok, scenario, seed, n, engine, "fastSpline",
                                      residual_device, scheduler)
    status <- if (is.null(linear) || is.null(fastspline)) "missing" else "ok"
    rows[[length(rows) + 1L]] <- list(
      scenario = scenario,
      seed = seed,
      n = n,
      engine = engine,
      residual_device = residual_device,
      scheduler = scheduler,
      pdag_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(fastkpc_extract_pdag(linear),
                                                          fastkpc_extract_pdag(fastspline)) else NA,
      skeleton_adjacency_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(linear$skeleton$adjacency,
                                                          fastspline$skeleton$adjacency) else NA,
      max_abs_pmax_diff =
        if (status == "ok") fastkpc_max_abs_matrix_diff(linear$skeleton$pMax,
                                                        fastspline$skeleton$pMax) else NA_real_,
      status = status
    )
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "engine",
                            "residual_device", "scheduler", "pdag_identical",
                            "skeleton_adjacency_identical",
                            "max_abs_pmax_diff", "status"))
}

fastkpc_campaign_residual_device_diffs <- function(results) {
  ok <- fastkpc_successful_results(results)
  keys <- unique(vapply(ok, function(x) {
    paste(x$scenario, x$seed, x$n, x$engine, x$residual_backend, x$scheduler,
          sep = "\r")
  }, character(1)))
  rows <- list()
  for (key in keys) {
    parts <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    scenario <- parts[[1L]]
    seed <- as.integer(parts[[2L]])
    n <- as.integer(parts[[3L]])
    engine <- parts[[4L]]
    residual_backend <- parts[[5L]]
    scheduler <- parts[[6L]]
    cpu <- fastkpc_find_result(ok, scenario, seed, n, engine, residual_backend,
                               "cpu", scheduler)
    cuda <- fastkpc_find_result(ok, scenario, seed, n, engine, residual_backend,
                                "cuda", scheduler)
    status <- if (is.null(cpu) || is.null(cuda)) "missing" else "ok"
    rows[[length(rows) + 1L]] <- list(
      scenario = scenario,
      seed = seed,
      n = n,
      engine = engine,
      residual_backend = residual_backend,
      scheduler = scheduler,
      pdag_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(fastkpc_extract_pdag(cpu),
                                                          fastkpc_extract_pdag(cuda)) else NA,
      skeleton_adjacency_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(cpu$skeleton$adjacency,
                                                          cuda$skeleton$adjacency) else NA,
      max_abs_pmax_diff =
        if (status == "ok") fastkpc_max_abs_matrix_diff(cpu$skeleton$pMax,
                                                        cuda$skeleton$pMax) else NA_real_,
      orientation_counts_identical =
        if (status == "ok") identical(cpu$orientation$counts,
                                      cuda$orientation$counts) else NA,
      status = status
    )
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "engine",
                            "residual_backend", "scheduler", "pdag_identical",
                            "skeleton_adjacency_identical",
                            "max_abs_pmax_diff",
                            "orientation_counts_identical", "status"))
}

fastkpc_campaign_scheduler_diffs <- function(results) {
  ok <- fastkpc_successful_results(results)
  keys <- unique(vapply(ok, function(x) {
    paste(x$scenario, x$seed, x$n, x$engine, x$residual_backend,
          x$residual_device, sep = "\r")
  }, character(1)))
  rows <- list()
  for (key in keys) {
    parts <- strsplit(key, "\r", fixed = TRUE)[[1L]]
    scenario <- parts[[1L]]
    seed <- as.integer(parts[[2L]])
    n <- as.integer(parts[[3L]])
    engine <- parts[[4L]]
    residual_backend <- parts[[5L]]
    residual_device <- parts[[6L]]
    legacy <- fastkpc_find_result(ok, scenario, seed, n, engine,
                                  residual_backend, residual_device, "legacy")
    layer <- fastkpc_find_result(ok, scenario, seed, n, engine,
                                 residual_backend, residual_device, "layer")
    status <- if (is.null(legacy) || is.null(layer)) "missing" else "ok"
    rows[[length(rows) + 1L]] <- list(
      scenario = scenario,
      seed = seed,
      n = n,
      engine = engine,
      residual_backend = residual_backend,
      residual_device = residual_device,
      left_scheduler = "legacy",
      right_scheduler = "layer",
      pdag_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(fastkpc_extract_pdag(legacy),
                                                          fastkpc_extract_pdag(layer)) else NA,
      skeleton_adjacency_identical =
        if (status == "ok") fastkpc_bool_identical_matrix(legacy$skeleton$adjacency,
                                                          layer$skeleton$adjacency) else NA,
      max_abs_pmax_diff =
        if (status == "ok") fastkpc_max_abs_matrix_diff(legacy$skeleton$pMax,
                                                        layer$skeleton$pMax) else NA_real_,
      orientation_counts_identical =
        if (status == "ok") identical(legacy$orientation$counts,
                                      layer$orientation$counts) else NA,
      status = status
    )
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "engine",
                            "residual_backend", "residual_device",
                            "left_scheduler", "right_scheduler",
                            "pdag_identical", "skeleton_adjacency_identical",
                            "max_abs_pmax_diff",
                            "orientation_counts_identical", "status"))
}

fastkpc_legacy_unavailable_reason <- function() {
  missing <- c("pcalg", "graph")[!vapply(c("pcalg", "graph"), requireNamespace,
                                         logical(1), quietly = TRUE)]
  if (length(missing) == 0L) "" else paste("missing package(s):",
                                           paste(missing, collapse = ", "))
}

fastkpc_campaign_legacy <- function(results, scenarios, alpha,
                                    max_conditioning_size, legacy) {
  ok <- fastkpc_successful_results(results)
  rows <- list()
  targets <- Filter(function(x) {
    identical(x$engine, "cpu") && identical(x$residual_backend, "fastSpline")
  }, ok)
  if (length(targets) == 0L) targets <- ok
  reason <- if (isTRUE(legacy)) fastkpc_legacy_unavailable_reason() else "legacy disabled"
  for (entry in targets) {
    base <- list(
      scenario = entry$scenario,
      seed = as.integer(entry$seed),
      n = as.integer(entry$n),
      available = FALSE,
      reason_if_unavailable = reason,
      native_engine = entry$engine,
      native_residual_backend = entry$residual_backend,
      pdag_exact = NA,
      directed_added = NA_integer_,
      directed_removed = NA_integer_,
      undirected_added = NA_integer_,
      undirected_removed = NA_integer_,
      max_abs_pdag_diff = NA_real_,
      status = if (isTRUE(legacy) && !nzchar(reason)) "pending" else "unavailable"
    )
    if (isTRUE(legacy) && !nzchar(reason)) {
      legacy_result <- fastkpc_safe_run(validate_wanpdag_against_legacy(
        seed = entry$seed,
        n = entry$n,
        alpha = alpha,
        max_conditioning_size = max_conditioning_size
      ))
      if (identical(legacy_result$status, "ok")) {
        metrics <- legacy_result$value$metrics
        base$available <- isTRUE(legacy_result$value$available)
        base$reason_if_unavailable <- legacy_result$value$reason_if_unavailable %||% ""
        base$pdag_exact <- metrics$pdag_exact
        base$directed_added <- metrics$directed_edge_added_count
        base$directed_removed <- metrics$directed_edge_removed_count
        base$undirected_added <- metrics$undirected_edge_added_count
        base$undirected_removed <- metrics$undirected_edge_removed_count
        base$max_abs_pdag_diff <- metrics$max_abs_pdag_diff
        base$status <- if (isTRUE(base$available)) "ok" else "unavailable"
      } else {
        base$reason_if_unavailable <- legacy_result$error_message
        base$status <- "error"
      }
    }
    rows[[length(rows) + 1L]] <- base
  }
  fastkpc_bind_rows(rows, c("scenario", "seed", "n", "available",
                            "reason_if_unavailable", "native_engine",
                            "native_residual_backend", "pdag_exact",
                            "directed_added", "directed_removed",
                            "undirected_added", "undirected_removed",
                            "max_abs_pdag_diff", "status"))
}

fastkpc_campaign_summary <- function(campaign) {
  runs <- campaign$runs
  cpu_cuda <- campaign$cpu_cuda
  residual_device_diffs <- campaign$residual_device_diffs %||%
    fastkpc_empty_df(c("max_abs_pmax_diff", "pdag_identical"))
  orientation_device_diffs <- campaign$orientation_device_diffs %||%
    fastkpc_empty_df(c("max_abs_pmax_diff", "pdag_identical"))
  orientation_device_diagnostics <- campaign$orientation_device_diagnostics %||%
    fastkpc_empty_df(c("orientation_dcov_batches",
                       "orientation_dcov_pairs",
                       "orientation_cuda_residual_fits",
                       "orientation_cpu_fallback_fits"))
  scheduler_diffs <- campaign$scheduler_diffs %||%
    fastkpc_empty_df(c("max_abs_pmax_diff", "pdag_identical"))
  true_batched_residuals <- campaign$true_batched_residuals %||%
    fastkpc_empty_df(c("cuda_residual_true_batched_groups",
                       "cuda_residual_true_batched_fits",
                       "cuda_residual_single_fit_calls",
                       "cuda_residual_cpu_fallback_fits",
                       "cuda_residual_unique_designs",
                       "cuda_residual_duplicate_design_fits",
                       "cuda_residual_max_fits_per_design"))
  hsic_cuda_backend_diagnostics <- campaign$hsic_cuda_backend_diagnostics %||%
    fastkpc_empty_df(c("cuda_hsic_used", "ci_hsic_cuda_batches",
                       "ci_hsic_cuda_pairs"))
  hsic_cuda_cpu_fallbacks <- campaign$hsic_cuda_cpu_fallbacks %||%
    fastkpc_empty_df(c("run_id"))
  compatibility <- campaign$compatibility %||%
    fastkpc_empty_compatibility_campaign_metrics()
  legacy <- campaign$legacy
  cpu_cuda_diffs <- cpu_cuda$max_abs_pmax_diff
  cpu_cuda_diffs <- cpu_cuda_diffs[is.finite(cpu_cuda_diffs)]
  residual_device_pmax <- residual_device_diffs$max_abs_pmax_diff
  residual_device_pmax <- residual_device_pmax[is.finite(residual_device_pmax)]
  orientation_device_pmax <- orientation_device_diffs$max_abs_pmax_diff
  orientation_device_pmax <- orientation_device_pmax[is.finite(orientation_device_pmax)]
  scheduler_pmax <- scheduler_diffs$max_abs_pmax_diff
  scheduler_pmax <- scheduler_pmax[is.finite(scheduler_pmax)]
  list(
    total_runs = nrow(runs),
    ok_runs = sum(runs$status == "ok", na.rm = TRUE),
    error_runs = sum(runs$status != "ok", na.rm = TRUE),
    cpu_cuda_rows = nrow(cpu_cuda),
    cpu_cuda_pdag_identical = sum(cpu_cuda$pdag_identical %in% TRUE, na.rm = TRUE),
    max_cpu_cuda_pmax_diff =
      if (length(cpu_cuda_diffs) == 0L) NA_real_ else max(cpu_cuda_diffs),
    residual_device_diff_rows = nrow(residual_device_diffs),
    residual_device_pdag_identical =
      sum(residual_device_diffs$pdag_identical %in% TRUE, na.rm = TRUE),
    max_residual_device_pmax_diff =
      if (length(residual_device_pmax) == 0L) NA_real_ else max(residual_device_pmax),
    orientation_device_diff_rows = nrow(orientation_device_diffs),
    orientation_device_pdag_identical =
      sum(orientation_device_diffs$pdag_identical %in% TRUE, na.rm = TRUE),
    max_orientation_device_pmax_diff =
      if (length(orientation_device_pmax) == 0L) NA_real_ else max(orientation_device_pmax),
    orientation_dcov_batches =
      sum(orientation_device_diagnostics$orientation_dcov_batches,
          na.rm = TRUE),
    orientation_dcov_pairs =
      sum(orientation_device_diagnostics$orientation_dcov_pairs,
          na.rm = TRUE),
    orientation_cuda_residual_fits =
      sum(orientation_device_diagnostics$orientation_cuda_residual_fits,
          na.rm = TRUE),
    orientation_cpu_fallback_fits =
      sum(orientation_device_diagnostics$orientation_cpu_fallback_fits,
          na.rm = TRUE),
    scheduler_diff_rows = nrow(scheduler_diffs),
    scheduler_pdag_identical =
      sum(scheduler_diffs$pdag_identical %in% TRUE, na.rm = TRUE),
    max_scheduler_pmax_diff =
      if (length(scheduler_pmax) == 0L) NA_real_ else max(scheduler_pmax),
    layer_runs = sum(runs$scheduler == "layer", na.rm = TRUE),
    legacy_runs = sum(runs$scheduler == "legacy", na.rm = TRUE),
    cuda_residual_true_batched_groups =
      sum(true_batched_residuals$cuda_residual_true_batched_groups,
          na.rm = TRUE),
    cuda_residual_true_batched_fits =
      sum(true_batched_residuals$cuda_residual_true_batched_fits,
          na.rm = TRUE),
    cuda_residual_single_fit_calls =
      sum(true_batched_residuals$cuda_residual_single_fit_calls,
          na.rm = TRUE),
    cuda_residual_cpu_fallback_fits =
      sum(true_batched_residuals$cuda_residual_cpu_fallback_fits,
          na.rm = TRUE),
    cuda_residual_unique_designs =
      sum(true_batched_residuals$cuda_residual_unique_designs,
          na.rm = TRUE),
    cuda_residual_duplicate_design_fits =
      sum(true_batched_residuals$cuda_residual_duplicate_design_fits,
          na.rm = TRUE),
    cuda_residual_max_fits_per_design = if (nrow(true_batched_residuals) == 0L) {
      0L
    } else {
      max(true_batched_residuals$cuda_residual_max_fits_per_design,
          na.rm = TRUE)
    },
    hsic_cuda_runs = sum(hsic_cuda_backend_diagnostics$cuda_hsic_used %in% TRUE,
                         na.rm = TRUE),
    hsic_cuda_batches =
      sum(hsic_cuda_backend_diagnostics$ci_hsic_cuda_batches, na.rm = TRUE),
    hsic_cuda_pairs =
      sum(hsic_cuda_backend_diagnostics$ci_hsic_cuda_pairs, na.rm = TRUE),
    hsic_cuda_cpu_fallbacks = nrow(hsic_cuda_cpu_fallbacks),
    compatibility_residual_rows = nrow(compatibility$residual),
    compatibility_ci_rows = nrow(compatibility$ci),
    compatibility_graph_rows = nrow(compatibility$graph),
    legacy_available = any(legacy$available %in% TRUE, na.rm = TRUE),
    legacy_reason_if_unavailable =
      paste(unique(legacy$reason_if_unavailable[nzchar(legacy$reason_if_unavailable)]),
            collapse = "; ")
  )
}

run_fastkpc_validation_campaign <- function(seeds = c(11, 12, 13),
                                            n_values = c(80, 140),
                                            scenarios = c("chain", "fork",
                                                          "collider",
                                                          "independent",
                                                          "additive"),
                                            engines = c("cpu", "cuda"),
                                            residual_backends = c("linear",
                                                                  "fastSpline"),
                                            residual_devices = c("auto"),
                                            orientation_residual_devices = c("auto"),
                                            schedulers = c("auto"),
                                            ci_methods = c("dcc.gamma"),
                                            residual_batch_size = 0,
                                            orientation_batch_size = 0,
                                            scheduler_diagnostics = TRUE,
                                            orientation_diagnostics = TRUE,
                                            ci_diagnostics = TRUE,
                                            fastspline_params = list(),
                                            hsic_params = list(sig = 1),
                                            permutation_params = list(replicates = 100,
                                                                      seed = NULL,
                                                                      include_observed = TRUE),
                                            alpha = 0.2,
                                            max_conditioning_size = 2,
                                            legacy = TRUE,
                                            benchmark = TRUE,
                                            output_dir = NULL) {
  unknown <- setdiff(scenarios, fastkpc_scenario_names())
  if (length(unknown) > 0L) {
    stop("Unknown fastkpc validation scenario: ", unknown[[1L]], call. = FALSE)
  }
  engines <- match.arg(engines, c("cpu", "cuda", "auto"), several.ok = TRUE)
  residual_backends <- match.arg(residual_backends, c("linear", "fastSpline"),
                                 several.ok = TRUE)
  residual_devices <- match.arg(residual_devices, c("auto", "cpu", "cuda"),
                                several.ok = TRUE)
  orientation_residual_devices <-
    match.arg(orientation_residual_devices, c("auto", "cpu", "cuda"),
              several.ok = TRUE)
  schedulers <- match.arg(schedulers, c("auto", "layer", "legacy"),
                          several.ok = TRUE)
  ci_methods <- match.arg(ci_methods, c("dcc.gamma", "hsic.gamma", "hsic.perm"),
                          several.ok = TRUE)
  grid <- expand.grid(
    seed = as.integer(seeds),
    n = as.integer(n_values),
    scenario = scenarios,
    engine = engines,
    residual_backend = residual_backends,
    residual_device = residual_devices,
    orientation_residual_device = orientation_residual_devices,
    scheduler = schedulers,
    ci_method = ci_methods,
    stringsAsFactors = FALSE
  )

  run_rows <- list()
  graph_rows <- list()
  timing_rows <- list()
  cache_rows <- list()
  orientation_rows <- list()
  scheduler_level_rows <- list()
  scheduler_batch_rows <- list()
  scheduler_residual_rows <- list()
  true_batched_rows <- list()
  orientation_device_diag_rows <- list()
  ci_method_diag_rows <- list()
  results <- list()

  for (i in seq_len(nrow(grid))) {
    row <- grid[i, ]
    run_id <- fastkpc_run_id(row$scenario, row$seed, row$n, row$engine,
                             row$residual_backend, row$residual_device,
                             row$scheduler, row$orientation_residual_device,
                             row$ci_method)
    scenario_value <- generate_fastkpc_scenario(row$scenario, row$seed, row$n)
    run <- fastkpc_safe_run(fast_kpc(
      scenario_value$data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = row$engine,
      residual_backend = row$residual_backend,
      residual_device = row$residual_device,
      orientation_residual_device = row$orientation_residual_device,
      scheduler = row$scheduler,
      residual_batch_size = residual_batch_size,
      orientation_batch_size = orientation_batch_size,
      scheduler_diagnostics = scheduler_diagnostics,
      orientation_diagnostics = orientation_diagnostics,
      ci_diagnostics = ci_diagnostics,
      fastspline_params = fastspline_params,
      hsic_params = hsic_params,
      permutation_params = permutation_params,
      ci_method = row$ci_method,
      graph_stage = "wanpdag",
      validate = FALSE,
      benchmark = benchmark,
      legacy = legacy,
      seed = row$seed
    ))
    result <- run$value
    metrics <- if (identical(run$status, "ok")) result$metrics else list()
    elapsed_total <- if (identical(run$status, "ok")) {
      total <- result$timings$elapsed_sec[result$timings$stage == "total"]
      if (length(total) == 0L) NA_real_ else total[[1L]]
    } else {
      NA_real_
    }
    scheduler_summary <- if (identical(run$status, "ok")) {
      result$skeleton$scheduler_diagnostics$summary %||% list()
    } else {
      list()
    }
    run_rows[[length(run_rows) + 1L]] <- list(
      run_id = run_id,
      scenario = row$scenario,
      seed = as.integer(row$seed),
      n = as.integer(row$n),
      p = ncol(scenario_value$data),
      engine = row$engine,
      residual_backend = row$residual_backend,
      residual_device = row$residual_device,
      orientation_residual_device = row$orientation_residual_device,
      scheduler = row$scheduler,
      ci_method = row$ci_method,
      ci_backend =
        if (identical(run$status, "ok")) result$config$ci_backend %||% NA_character_
        else NA_character_,
      ci_backend_requested =
        if (identical(run$status, "ok")) result$config$ci_backend_requested %||% NA_character_
        else NA_character_,
      ci_backend_reason =
        if (identical(run$status, "ok")) result$config$ci_backend_reason %||% ""
        else "",
      cuda_hsic_requested =
        if (identical(run$status, "ok")) isTRUE(result$config$cuda_hsic_requested)
        else FALSE,
      cuda_hsic_used =
        if (identical(run$status, "ok")) isTRUE(result$config$cuda_hsic_used)
        else FALSE,
      status = run$status,
      error_message = run$error_message,
      skeleton_edge_count = as.integer(metrics$skeleton_edge_count %||% NA_integer_),
      directed_edge_count = as.integer(metrics$directed_edge_count %||% NA_integer_),
      undirected_edge_count =
        as.integer(metrics$undirected_edge_count %||% NA_integer_),
      bidirected_edge_count =
        as.integer(metrics$bidirected_edge_count %||% NA_integer_),
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
      elapsed_total_sec = elapsed_total
    )
    graph_rows[[length(graph_rows) + 1L]] <- c(run_rows[[length(run_rows)]],
                                               list(
                                                 orientation_event_count =
                                                   as.integer(metrics$orientation_event_count %||% NA_integer_),
                                                 generalized_orientation_count =
                                                   as.integer(metrics$generalized_orientation_count %||% NA_integer_),
                                                 max_pmax = metrics$max_pmax %||% NA_real_,
                                                 min_nonzero_pmax =
                                                  metrics$min_nonzero_pmax %||% NA_real_
                                               ))
    if (identical(run$status, "ok")) {
      result$config$scenario <- row$scenario
      timing_rows[[length(timing_rows) + 1L]] <- fastkpc_flatten_timings(result, run_id)
      cache_rows[[length(cache_rows) + 1L]] <- fastkpc_flatten_cache(result, run_id)
      orientation_rows[[length(orientation_rows) + 1L]] <-
        fastkpc_flatten_orientation_counts(result, run_id)
      scheduler_level_rows[[length(scheduler_level_rows) + 1L]] <-
        fastkpc_flatten_scheduler_levels(result, run_id)
      scheduler_batch_rows[[length(scheduler_batch_rows) + 1L]] <-
        fastkpc_flatten_scheduler_batches(result, run_id)
      scheduler_residual_rows[[length(scheduler_residual_rows) + 1L]] <-
        fastkpc_flatten_scheduler_residuals(result, run_id)
      true_batched_rows[[length(true_batched_rows) + 1L]] <-
        fastkpc_flatten_true_batched_residuals(result, run_id)
      orientation_device_diag_rows[[length(orientation_device_diag_rows) + 1L]] <-
        fastkpc_flatten_orientation_device_diagnostics(result, run_id)
      ci_method_diag_rows[[length(ci_method_diag_rows) + 1L]] <-
        fastkpc_flatten_ci_method_diagnostics(result, run_id)
    }
    results[[length(results) + 1L]] <- list(
      run_id = run_id,
      scenario = row$scenario,
      seed = as.integer(row$seed),
      n = as.integer(row$n),
      engine = row$engine,
      residual_backend = row$residual_backend,
      residual_device = row$residual_device,
      orientation_residual_device = row$orientation_residual_device,
      scheduler = row$scheduler,
      ci_method = row$ci_method,
      status = run$status,
      error_message = run$error_message,
      result = result
    )
  }

  runs <- fastkpc_bind_rows(run_rows, c("run_id", "scenario", "seed", "n", "p",
                                        "engine", "residual_backend",
                                        "residual_device",
                                        "orientation_residual_device",
                                        "scheduler", "ci_method",
                                        "ci_backend", "ci_backend_requested",
                                        "ci_backend_reason",
                                        "cuda_hsic_requested",
                                        "cuda_hsic_used", "status",
                                        "error_message", "skeleton_edge_count",
                                        "directed_edge_count",
                                        "undirected_edge_count",
                                        "bidirected_edge_count",
                                        "cuda_residual_batch_groups",
                                        "cuda_residual_true_batched_groups",
                                        "cuda_residual_true_batched_fits",
                                        "cuda_residual_single_fit_calls",
                                        "cuda_residual_cpu_fallback_fits",
                                        "cuda_residual_unique_designs",
                                        "cuda_residual_duplicate_design_fits",
                                        "cuda_residual_max_fits_per_design",
                                        "elapsed_total_sec"))
  graph_metrics <- fastkpc_bind_rows(graph_rows, c(
    "run_id", "scenario", "seed", "n", "p", "engine", "residual_backend",
    "residual_device", "orientation_residual_device", "scheduler", "ci_method",
    "ci_backend", "ci_backend_requested", "ci_backend_reason",
    "cuda_hsic_requested", "cuda_hsic_used",
    "status", "error_message", "skeleton_edge_count",
    "directed_edge_count",
    "undirected_edge_count", "bidirected_edge_count",
    "cuda_residual_batch_groups", "cuda_residual_true_batched_groups",
    "cuda_residual_true_batched_fits", "cuda_residual_single_fit_calls",
    "cuda_residual_cpu_fallback_fits", "cuda_residual_unique_designs",
    "cuda_residual_duplicate_design_fits",
    "cuda_residual_max_fits_per_design", "elapsed_total_sec",
    "orientation_event_count", "generalized_orientation_count", "max_pmax",
    "min_nonzero_pmax"
  ))
  timings <- if (length(timing_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device",
                       "orientation_residual_device", "scheduler",
                       "stage", "elapsed_sec"))
  } else {
    do.call(rbind, timing_rows)
  }
  cache <- if (length(cache_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device",
                       "orientation_residual_device", "scheduler",
                       "section", "requests", "hits", "misses",
                       "computations", "stored_vectors", "stored_values"))
  } else {
    do.call(rbind, cache_rows)
  }
  orientation_counts <- if (length(orientation_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device",
                       "orientation_residual_device", "scheduler",
                       "collider", "rule1", "rule2", "rule3", "generalized",
                       "regrvonps_calls"))
  } else {
    do.call(rbind, orientation_rows)
  }
  errors <- runs[runs$status != "ok", c("run_id", "scenario", "seed", "n",
                                        "engine", "residual_backend",
                                        "residual_device",
                                        "orientation_residual_device",
                                        "scheduler", "ci_method", "status",
                                        "error_message"), drop = FALSE]
  scheduler_levels <- if (length(scheduler_level_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device", "scheduler",
                       "level", "tasks_planned", "tasks_evaluated",
                       "tests_replayed", "tasks_ignored_after_delete",
                       "deletions", "unconditional_tasks", "conditional_tasks",
                       "unique_residual_requests", "dcov_batches",
                       "residual_batches", "plan_elapsed_sec",
                       "residual_prefetch_elapsed_sec", "ci_eval_elapsed_sec",
                       "replay_elapsed_sec", "total_elapsed_sec"))
  } else {
    do.call(rbind, scheduler_level_rows)
  }
  scheduler_batches <- if (length(scheduler_batch_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device", "scheduler",
                       "level", "batch_id", "kind", "start_task_id",
                       "task_count", "n", "status", "groups",
                       "true_batched_groups", "true_batched_fits",
                       "single_fit_calls", "cpu_fallback_fits",
                       "unique_designs", "duplicate_design_fits",
                       "max_fits_per_design",
                       "max_group_size", "min_group_size",
                       "max_design_cols", "min_design_cols"))
  } else {
    do.call(rbind, scheduler_batch_rows)
  }
  scheduler_residuals <- if (length(scheduler_residual_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend_requested", "residual_device_requested",
                       "scheduler", "level", "request_id", "target",
                       "conditioning_size", "residual_backend",
                       "residual_device", "materialized", "fallback_used",
                       "reason"))
  } else {
    do.call(rbind, scheduler_residual_rows)
  }
  true_batched_residuals <- if (length(true_batched_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device", "scheduler",
                       "residual_batch_size", "cuda_residual_batch_groups",
                       "cuda_residual_true_batched_groups",
                       "cuda_residual_true_batched_fits",
                       "cuda_residual_single_fit_calls",
                       "cuda_residual_cpu_fallback_fits",
                       "cuda_residual_unique_designs",
                       "cuda_residual_duplicate_design_fits",
                       "cuda_residual_max_fits_per_design", "status"))
  } else {
    do.call(rbind, true_batched_rows)
  }
  orientation_device_diagnostics <-
    if (length(orientation_device_diag_rows) == 0L) {
      fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                         "residual_backend", "residual_device", "scheduler",
                         "orientation_residual_device_requested",
                         "orientation_residual_device",
                         "orientation_residual_device_reason",
                         "orientation_batch_size_requested",
                         "orientation_batch_size_used",
                         "regrvonps_calls", "regrvonps_cuda_calls",
                         "regrvonps_cpu_calls", "orientation_dcov_batches",
                         "orientation_dcov_pairs",
                         "orientation_residual_fits",
                         "orientation_cuda_residual_fits",
                         "orientation_cpu_fallback_fits",
                         "orientation_cache_requests",
                         "orientation_cache_hits",
                         "orientation_cache_computations"))
    } else {
      do.call(rbind, orientation_device_diag_rows)
    }
  ci_method_diagnostics <- if (length(ci_method_diag_rows) == 0L) {
    fastkpc_empty_df(c("run_id", "scenario", "seed", "n", "engine",
                       "residual_backend", "residual_device",
                       "orientation_residual_device", "scheduler", "ci_method",
                       "ci_backend", "ci_backend_requested",
                       "ci_backend_reason", "cuda_hsic_requested",
                       "cuda_hsic_used",
                       "ci_dcc_gamma_tests", "ci_hsic_gamma_tests",
                       "ci_hsic_perm_tests",
                       "ci_hsic_permutation_replicates",
                       "ci_hsic_gamma_cuda_tests",
                       "ci_hsic_perm_cuda_tests",
                       "ci_hsic_cuda_batches",
                       "ci_hsic_cuda_pairs",
                       "ci_hsic_cuda_fallback_tests", "ci_tests",
                       "regrvonps_dcc_gamma_tests",
                       "regrvonps_hsic_gamma_tests",
                       "regrvonps_hsic_perm_tests",
                       "regrvonps_hsic_permutation_replicates",
                       "regrvonps_hsic_gamma_cuda_tests",
                       "regrvonps_hsic_perm_cuda_tests",
                       "regrvonps_hsic_cuda_batches",
                       "regrvonps_hsic_cuda_pairs",
                       "regrvonps_hsic_cuda_fallback_tests"))
  } else {
    do.call(rbind, ci_method_diag_rows)
  }
  hsic_cuda_backend_diagnostics <- ci_method_diagnostics[
    ci_method_diagnostics$ci_method %in% c("hsic.gamma", "hsic.perm"),
    , drop = FALSE
  ]
  hsic_cuda_cpu_fallbacks <- hsic_cuda_backend_diagnostics[
    hsic_cuda_backend_diagnostics$cuda_hsic_requested %in% TRUE &
      hsic_cuda_backend_diagnostics$ci_backend == "native-cpu",
    , drop = FALSE
  ]
  hsic_cuda_perf <- hsic_cuda_backend_diagnostics[, intersect(
    c("run_id", "scenario", "seed", "n", "engine", "residual_backend",
      "residual_device", "orientation_residual_device", "scheduler",
      "ci_method", "ci_backend", "cuda_hsic_used",
      "ci_hsic_cuda_batches", "ci_hsic_cuda_pairs",
      "regrvonps_hsic_cuda_batches", "regrvonps_hsic_cuda_pairs"),
    names(hsic_cuda_backend_diagnostics)
  ), drop = FALSE]
  if (nrow(hsic_cuda_perf) > 0L && "run_id" %in% names(hsic_cuda_perf)) {
    hsic_cuda_perf <- merge(
      hsic_cuda_perf,
      runs[, c("run_id", "elapsed_total_sec"), drop = FALSE],
      by = "run_id",
      all.x = TRUE,
      sort = FALSE
    )
  }
  ci_method_diffs <- fastkpc_empty_df(c("scenario", "seed", "n", "engine",
                                        "residual_backend", "residual_device",
                                        "orientation_residual_device",
                                        "scheduler", "left_ci_method",
                                        "right_ci_method", "pdag_identical",
                                        "skeleton_adjacency_identical",
                                        "max_abs_pmax_diff",
                                        "orientation_counts_identical",
                                        "status"))

  campaign <- list(
    config = list(
      seeds = as.integer(seeds),
      n_values = as.integer(n_values),
      scenarios = scenarios,
      engines = engines,
      residual_backends = residual_backends,
      residual_devices = residual_devices,
      orientation_residual_devices = orientation_residual_devices,
      schedulers = schedulers,
      ci_methods = ci_methods,
      residual_batch_size = as.integer(residual_batch_size),
      orientation_batch_size = as.integer(orientation_batch_size),
      scheduler_diagnostics = isTRUE(scheduler_diagnostics),
      orientation_diagnostics = isTRUE(orientation_diagnostics),
      ci_diagnostics = isTRUE(ci_diagnostics),
      fastspline_params = fastspline_params,
      hsic_params = hsic_params,
      permutation_params = permutation_params,
      alpha = alpha,
      max_conditioning_size = as.integer(max_conditioning_size),
      legacy = isTRUE(legacy),
      benchmark = isTRUE(benchmark),
      output_dir = output_dir
    ),
    runs = runs,
    graph_metrics = graph_metrics,
    pairwise_diffs = fastkpc_campaign_pairwise_diffs(results),
    cpu_cuda = fastkpc_campaign_cpu_cuda(results),
    linear_fastspline = fastkpc_campaign_linear_fastspline(results),
    residual_device_diffs = fastkpc_campaign_residual_device_diffs(results),
    orientation_device_diffs =
      fastkpc_campaign_orientation_device_diffs(results),
    orientation_device_diagnostics = orientation_device_diagnostics,
    ci_method_diagnostics = ci_method_diagnostics,
    hsic_cuda_backend_diagnostics = hsic_cuda_backend_diagnostics,
    hsic_cuda_cpu_fallbacks = hsic_cuda_cpu_fallbacks,
    hsic_cuda_perf = hsic_cuda_perf,
    ci_method_diffs = ci_method_diffs,
    scheduler_diffs = fastkpc_campaign_scheduler_diffs(results),
    legacy = fastkpc_campaign_legacy(results, scenarios, alpha,
                                     max_conditioning_size, legacy),
    timings = timings,
    cache = cache,
    orientation_counts = orientation_counts,
    scheduler_levels = scheduler_levels,
    scheduler_batches = scheduler_batches,
    scheduler_residuals = scheduler_residuals,
    true_batched_residuals = true_batched_residuals,
    compatibility = fastkpc_empty_compatibility_campaign_metrics(),
    errors = errors,
    artifacts = list(output_dir = output_dir)
  )
  campaign$summary <- fastkpc_campaign_summary(campaign)
  campaign
}
