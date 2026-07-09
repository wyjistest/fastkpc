source("fastkpc/R/fast_kpc.R")

fastkpc_legacy_dcov_batch_restore_env <- function(old_env) {
  for (name in names(old_env)) {
    if (is.na(old_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_env[[name]]), name))
    }
  }
  invisible(TRUE)
}

fastkpc_legacy_dcov_batch_shd <- function(left, right) {
  sum(abs(as.integer(unname(left)) - as.integer(unname(right)))) / 2L
}

fastkpc_legacy_dcov_batch_summary_value <- function(summary, name, default) {
  value <- summary[[name]]
  if (is.null(value)) default else value
}

fastkpc_legacy_dcov_batch_timeout_enabled <- function(candidate_timeout_sec) {
  !is.null(candidate_timeout_sec) &&
    length(candidate_timeout_sec) > 0L &&
    !is.na(candidate_timeout_sec[[1L]])
}

fastkpc_legacy_dcov_batch_timeout_error <- function(
    timeout_sec, elapsed_sec, message = NULL) {
  if (is.null(message)) {
    message <- paste0("candidate exceeded timeout_sec=", timeout_sec)
  }
  structure(
    list(
      message = message,
      call = NULL,
      timeout_sec = as.numeric(timeout_sec),
      elapsed_sec = as.numeric(elapsed_sec)
    ),
    class = c("fastkpc_legacy_dcov_batch_timeout", "error", "condition")
  )
}

fastkpc_legacy_dcov_batch_is_time_limit <- function(error) {
  grepl("reached elapsed time limit|reached CPU time limit",
        conditionMessage(error))
}

fastkpc_legacy_dcov_batch_progress_row <- function(
    artifact_name, route, batch_mode, batch_min_size, event, status,
    elapsed_sec = NA_real_, message = NA_character_) {
  data.frame(
    artifact = artifact_name,
    route = route,
    batch_mode = if (nzchar(batch_mode)) batch_mode else "none",
    batch_min_size = as.integer(batch_min_size %||% NA_integer_),
    event = event,
    status = status,
    elapsed_sec = as.numeric(elapsed_sec),
    message = message %||% NA_character_,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    stringsAsFactors = FALSE
  )
}

fastkpc_legacy_dcov_batch_append_progress <- function(path, row) {
  utils::write.table(
    row,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file.exists(path),
    append = file.exists(path),
    qmethod = "double"
  )
  invisible(TRUE)
}

fastkpc_legacy_dcov_batch_extract_reference <- function(result) {
  candidates <- list(
    root = result,
    reference = result$reference,
    skeleton = result$skeleton,
    baseline = result$baseline,
    backend = result$backend,
    facade = result$facade
  )
  for (name in names(candidates)) {
    candidate <- candidates[[name]]
    if (is.list(candidate) &&
        !is.null(candidate$skeleton) &&
        !is.null(candidate$skeleton$adjacency) &&
        !is.null(candidate$skeleton$n.edgetests)) {
      attr(candidate, "fastkpc_reference_slot") <- name
      return(candidate)
    }
    if (is.list(candidate) &&
        !is.null(candidate$adjacency) &&
        !is.null(candidate$n.edgetests)) {
      wrapped <- list(skeleton = candidate)
      attr(wrapped, "fastkpc_reference_slot") <- name
      return(wrapped)
    }
  }
  stop("reference result does not contain a skeleton-like object",
       call. = FALSE)
}

fastkpc_legacy_dcov_batch_run_once <- function(
    data, alpha, max_conditioning_size, index, numCol, num_cores,
    batch_mode = "", batch_min_size = NA_integer_, trace_level = "summary") {
  if (nzchar(batch_mode)) {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH = batch_mode)
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
  }
  if (is.na(batch_min_size)) {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_MIN_SIZE")
  } else {
    Sys.setenv(
      FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_MIN_SIZE =
        as.character(as.integer(batch_min_size))
    )
  }
  if (is.null(num_cores) || is.na(num_cores)) {
    Sys.unsetenv("FASTKPC_LEGACY_PARALLEL_CORES")
  } else {
    Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = as.character(as.integer(num_cores)))
  }
  fastkpc_elapsed(
    fast_kpc(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = "cuda",
      precision = "compatible",
      graph_stage = "skeleton",
      ci_method = "dcc.gamma",
      precision_trace_level = trace_level,
      benchmark = TRUE
    )
  )
}

fastkpc_legacy_dcov_batch_run_once_with_timeout <- function(
    data, alpha, max_conditioning_size, index, numCol, num_cores,
    batch_mode = "", batch_min_size = NA_integer_, trace_level = "summary",
    candidate_timeout_sec = NULL) {
  if (!fastkpc_legacy_dcov_batch_timeout_enabled(candidate_timeout_sec)) {
    return(fastkpc_legacy_dcov_batch_run_once(
      data = data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      index = index,
      numCol = numCol,
      num_cores = num_cores,
      batch_mode = batch_mode,
      batch_min_size = batch_min_size,
      trace_level = trace_level
    ))
  }

  timeout_sec <- as.numeric(candidate_timeout_sec[[1L]])
  start <- proc.time()[["elapsed"]]
  if (timeout_sec <= 0) {
    stop(fastkpc_legacy_dcov_batch_timeout_error(
      timeout_sec = timeout_sec,
      elapsed_sec = 0,
      message = "candidate timed out before execution"
    ))
  }

  setTimeLimit(cpu = Inf, elapsed = timeout_sec, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  tryCatch(
    fastkpc_legacy_dcov_batch_run_once(
      data = data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      index = index,
      numCol = numCol,
      num_cores = num_cores,
      batch_mode = batch_mode,
      batch_min_size = batch_min_size,
      trace_level = trace_level
    ),
    error = function(e) {
      if (fastkpc_legacy_dcov_batch_is_time_limit(e)) {
        stop(fastkpc_legacy_dcov_batch_timeout_error(
          timeout_sec = timeout_sec,
          elapsed_sec = proc.time()[["elapsed"]] - start,
          message = conditionMessage(e)
        ))
      }
      stop(e)
    }
  )
}

fastkpc_legacy_dcov_batch_summary_row <- function(
    artifact_name, route, batch_mode, batch_min_size, run, reference,
    n, p, alpha, max_conditioning_size, index, numCol,
    reference_source, reference_result_path,
    run_status = "ok", timeout = FALSE, timeout_sec = NA_real_) {
  skeleton <- run$value$skeleton
  reference_skeleton <- reference$value$skeleton
  summary <- skeleton$scheduler_diagnostics$summary %||% list()
  reference_edge_count <- as.integer(sum(unname(reference_skeleton$adjacency)) / 2L)
  edge_count <- as.integer(sum(unname(skeleton$adjacency)) / 2L)
  n_edgetests_identical <- identical(as.integer(skeleton$n.edgetests),
                                    as.integer(reference_skeleton$n.edgetests))
  data.frame(
    artifact = artifact_name,
    route = route,
    batch_mode = if (nzchar(batch_mode)) batch_mode else "none",
    batch_min_size = as.integer(batch_min_size %||% NA_integer_),
    n = as.integer(n),
    p = as.integer(p),
    reference_source = reference_source,
    reference_result_path = reference_result_path %||% NA_character_,
    run_status = run_status,
    timeout = isTRUE(timeout),
    timeout_sec = as.numeric(timeout_sec),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    index = as.integer(index),
    numCol = as.integer(numCol),
    edge_count = edge_count,
    reference_edge_count = reference_edge_count,
    shd = as.integer(fastkpc_legacy_dcov_batch_shd(
      skeleton$adjacency, reference_skeleton$adjacency
    )),
    adjacency_identical =
      identical(unname(skeleton$adjacency), unname(reference_skeleton$adjacency)),
    n_edgetests_identical = n_edgetests_identical,
    n_edgetests_exact = n_edgetests_identical,
    n_edgetests = paste(skeleton$n.edgetests, collapse = "|"),
    reference_n_edgetests =
      paste(reference_skeleton$n.edgetests, collapse = "|"),
    elapsed_sec = as.numeric(run$elapsed),
    legacy_dcov_cpp_backend_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_backend_count", NA_integer_
      )
    ),
    legacy_dcov_cpp_backend_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_backend_ms", NA_real_
      )
    ),
    legacy_dcov_cpp_batch_backend_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_backend_count", 0L
      )
    ),
    legacy_dcov_cpp_batch_backend_pair_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_backend_pair_count", 0L
      )
    ),
    legacy_dcov_cpp_batch_backend_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_backend_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_backend_max_batch_size = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_backend_max_batch_size", 0L
      )
    ),
    legacy_dcov_cpp_batch_backend_mean_batch_size = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_backend_mean_batch_size", 0
      )
    ),
    legacy_dcov_cpp_batch_candidate_pair_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_candidate_pair_count", 0L
      )
    ),
    legacy_dcov_cpp_batch_pair_coverage_ratio = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_pair_coverage_ratio", 0
      )
    ),
    legacy_dcov_cpp_batch_skipped_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_skipped_count", 0L
      )
    ),
    legacy_dcov_cpp_batch_skipped_pair_count = as.integer(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_skipped_pair_count", 0L
      )
    ),
    legacy_dcov_cpp_batch_skipped_pair_ratio = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_skipped_pair_ratio", 0
      )
    ),
    legacy_dcov_cpp_batch_prepare_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_prepare_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_materialize_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_materialize_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_apply_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_apply_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_round_prepare_worker_max_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_round_prepare_worker_max_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_round_prepare_worker_median_ms = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary, "legacy_dcov_cpp_batch_round_prepare_worker_median_ms", 0
      )
    ),
    legacy_dcov_cpp_batch_round_prepare_worker_elapsed_imbalance = as.numeric(
      fastkpc_legacy_dcov_batch_summary_value(
        summary,
        "legacy_dcov_cpp_batch_round_prepare_worker_elapsed_imbalance",
        0
      )
    ),
    stringsAsFactors = FALSE
  )
}

fastkpc_legacy_dcov_batch_timeout_summary_row <- function(
    artifact_name, batch_mode, batch_min_size, reference,
    n, p, alpha, max_conditioning_size, index, numCol,
    reference_source, reference_result_path, timeout_sec, elapsed_sec) {
  reference_skeleton <- reference$value$skeleton
  reference_edge_count <- as.integer(sum(unname(reference_skeleton$adjacency)) / 2L)
  data.frame(
    artifact = artifact_name,
    route = "candidate",
    batch_mode = if (nzchar(batch_mode)) batch_mode else "none",
    batch_min_size = as.integer(batch_min_size %||% NA_integer_),
    n = as.integer(n),
    p = as.integer(p),
    reference_source = reference_source,
    reference_result_path = reference_result_path %||% NA_character_,
    run_status = "timeout",
    timeout = TRUE,
    timeout_sec = as.numeric(timeout_sec),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    index = as.integer(index),
    numCol = as.integer(numCol),
    edge_count = NA_integer_,
    reference_edge_count = reference_edge_count,
    shd = NA_integer_,
    adjacency_identical = NA,
    n_edgetests_identical = NA,
    n_edgetests_exact = NA,
    n_edgetests = NA_character_,
    reference_n_edgetests =
      paste(reference_skeleton$n.edgetests, collapse = "|"),
    elapsed_sec = as.numeric(elapsed_sec),
    legacy_dcov_cpp_backend_count = NA_integer_,
    legacy_dcov_cpp_backend_ms = NA_real_,
    legacy_dcov_cpp_batch_backend_count = NA_integer_,
    legacy_dcov_cpp_batch_backend_pair_count = NA_integer_,
    legacy_dcov_cpp_batch_backend_ms = NA_real_,
    legacy_dcov_cpp_batch_backend_max_batch_size = NA_integer_,
    legacy_dcov_cpp_batch_backend_mean_batch_size = NA_real_,
    legacy_dcov_cpp_batch_candidate_pair_count = NA_integer_,
    legacy_dcov_cpp_batch_pair_coverage_ratio = NA_real_,
    legacy_dcov_cpp_batch_skipped_count = NA_integer_,
    legacy_dcov_cpp_batch_skipped_pair_count = NA_integer_,
    legacy_dcov_cpp_batch_skipped_pair_ratio = NA_real_,
    legacy_dcov_cpp_batch_prepare_ms = NA_real_,
    legacy_dcov_cpp_batch_materialize_ms = NA_real_,
    legacy_dcov_cpp_batch_apply_ms = NA_real_,
    legacy_dcov_cpp_batch_round_prepare_worker_max_ms = NA_real_,
    legacy_dcov_cpp_batch_round_prepare_worker_median_ms = NA_real_,
    legacy_dcov_cpp_batch_round_prepare_worker_elapsed_imbalance = NA_real_,
    stringsAsFactors = FALSE
  )
}

fastkpc_legacy_dcov_batch_level_rows <- function(
    route, batch_mode, batch_min_size, run) {
  skeleton <- run$value$skeleton
  by_level <- skeleton$scheduler_diagnostics$legacy_runtime_by_level
  if (!is.data.frame(by_level) || nrow(by_level) == 0L) {
    return(data.frame())
  }
  data.frame(
    route = route,
    batch_mode = if (nzchar(batch_mode)) batch_mode else "none",
    batch_min_size = as.integer(batch_min_size %||% NA_integer_),
    by_level,
    stringsAsFactors = FALSE
  )
}

fastkpc_run_legacy_dcov_cpp_batch_min_size_artifact <- function(
    data = NULL,
    data_path = file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
                          "cancer_RD-causalDiscoveryInput.rds"),
    columns = NULL,
    output_dir = file.path("fastkpc", "artifacts",
                           "legacy_dcov_cpp_batch_min_size_sweep_v1"),
    artifact_name = basename(output_dir),
    alpha = 0.1,
    max_conditioning_size = NULL,
    index = 1,
    numCol = NULL,
    batch_mode = c("round", "chunk"),
    min_sizes = c(1L),
    low_rank = "spectra",
    use_residual_cache = TRUE,
    residual_affinity = "s",
    num_cores = NULL,
    trace_level = "summary",
    reference_result_path = NULL,
    candidate_timeout_sec = NULL) {
  batch_mode <- match.arg(batch_mode)
  if (fastkpc_legacy_dcov_batch_timeout_enabled(candidate_timeout_sec)) {
    candidate_timeout_sec <- as.numeric(candidate_timeout_sec[[1L]])
    if (!is.finite(candidate_timeout_sec) || candidate_timeout_sec < 0) {
      stop("candidate_timeout_sec must be NULL or a non-negative finite number",
           call. = FALSE)
    }
  }
  if (is.null(data)) {
    if (!file.exists(data_path)) {
      stop("real data fixture not found: ", data_path, call. = FALSE)
    }
    data <- readRDS(data_path)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (!is.null(columns)) {
    data <- data[, as.integer(columns), drop = FALSE]
  }
  if (ncol(data) < 4L) {
    stop("dCov batch min-size artifact data must contain at least four columns",
         call. = FALSE)
  }
  if (is.null(max_conditioning_size)) {
    max_conditioning_size <- ncol(data) - 2L
  }
  max_conditioning_size <- min(as.integer(max_conditioning_size),
                               ncol(data) - 2L)
  if (is.null(numCol)) {
    numCol <- floor(nrow(data) / 10)
  }
  min_sizes <- unique(as.integer(min_sizes))
  min_sizes <- min_sizes[is.finite(min_sizes) & min_sizes >= 1L]
  if (length(min_sizes) == 0L) {
    stop("min_sizes must contain at least one positive integer", call. = FALSE)
  }

  env_names <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH_MIN_SIZE",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
    "FASTKPC_LEGACY_PARALLEL_CORES"
  )
  old_env <- Sys.getenv(env_names, unset = NA_character_)
  on.exit(fastkpc_legacy_dcov_batch_restore_env(old_env), add = TRUE)

  Sys.setenv(
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = low_rank
  )
  if (isTRUE(use_residual_cache)) {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1")
    if (nzchar(residual_affinity)) {
      Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = residual_affinity)
    }
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    progress_csv = file.path(output_dir, "progress.csv"),
    runtime_by_level_csv = file.path(output_dir, "runtime_by_level.csv"),
    result_rds = file.path(output_dir, "result.rds"),
    summary_md = file.path(output_dir, "summary.md")
  )
  if (file.exists(paths$progress_csv)) unlink(paths$progress_csv)

  reference_source <- if (is.null(reference_result_path)) "computed" else "rds"
  fastkpc_legacy_dcov_batch_append_progress(
    paths$progress_csv,
    fastkpc_legacy_dcov_batch_progress_row(
      artifact_name = artifact_name,
      route = "reference",
      batch_mode = "",
      batch_min_size = NA_integer_,
      event = "start",
      status = reference_source
    )
  )
  reference_start <- proc.time()[["elapsed"]]
  if (is.null(reference_result_path)) {
    reference <- fastkpc_legacy_dcov_batch_run_once(
      data = data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      index = index,
      numCol = numCol,
      num_cores = num_cores,
      batch_mode = "",
      batch_min_size = NA_integer_,
      trace_level = trace_level
    )
    reference_elapsed <- as.numeric(reference$elapsed)
  } else {
    if (!file.exists(reference_result_path)) {
      stop("reference result not found: ", reference_result_path,
           call. = FALSE)
    }
    reference <- list(
      value = fastkpc_legacy_dcov_batch_extract_reference(
        readRDS(reference_result_path)
      ),
      elapsed = 0
    )
    reference_elapsed <- proc.time()[["elapsed"]] - reference_start
  }
  fastkpc_legacy_dcov_batch_append_progress(
    paths$progress_csv,
    fastkpc_legacy_dcov_batch_progress_row(
      artifact_name = artifact_name,
      route = "reference",
      batch_mode = "",
      batch_min_size = NA_integer_,
      event = "complete",
      status = reference_source,
      elapsed_sec = reference_elapsed
    )
  )

  rows <- list(fastkpc_legacy_dcov_batch_summary_row(
    artifact_name = artifact_name,
    route = "reference",
    batch_mode = "",
    batch_min_size = NA_integer_,
    run = reference,
    reference = reference,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    n = nrow(data),
    p = ncol(data),
    index = index,
    numCol = numCol,
    reference_source = reference_source,
    reference_result_path = reference_result_path
  ))
  by_level_rows <- list(fastkpc_legacy_dcov_batch_level_rows(
    route = "reference",
    batch_mode = "",
    batch_min_size = NA_integer_,
    run = reference
  ))
  candidates <- list()

  for (min_size in min_sizes) {
    fastkpc_legacy_dcov_batch_append_progress(
      paths$progress_csv,
      fastkpc_legacy_dcov_batch_progress_row(
        artifact_name = artifact_name,
        route = "candidate",
        batch_mode = batch_mode,
        batch_min_size = min_size,
        event = "start",
        status = "running"
      )
    )
    candidate <- tryCatch(
      fastkpc_legacy_dcov_batch_run_once_with_timeout(
        data = data,
        alpha = alpha,
        max_conditioning_size = max_conditioning_size,
        index = index,
        numCol = numCol,
        num_cores = num_cores,
        batch_mode = batch_mode,
        batch_min_size = min_size,
        trace_level = trace_level,
        candidate_timeout_sec = candidate_timeout_sec
      ),
      fastkpc_legacy_dcov_batch_timeout = function(e) {
        fastkpc_legacy_dcov_batch_append_progress(
          paths$progress_csv,
          fastkpc_legacy_dcov_batch_progress_row(
            artifact_name = artifact_name,
            route = "candidate",
            batch_mode = batch_mode,
            batch_min_size = min_size,
            event = "timeout",
            status = "timeout",
            elapsed_sec = e$elapsed_sec,
            message = conditionMessage(e)
          )
        )
        rows[[length(rows) + 1L]] <<-
          fastkpc_legacy_dcov_batch_timeout_summary_row(
            artifact_name = artifact_name,
            batch_mode = batch_mode,
            batch_min_size = min_size,
            reference = reference,
            alpha = alpha,
            max_conditioning_size = max_conditioning_size,
            n = nrow(data),
            p = ncol(data),
            index = index,
            numCol = numCol,
            reference_source = reference_source,
            reference_result_path = reference_result_path,
            timeout_sec = e$timeout_sec,
            elapsed_sec = e$elapsed_sec
          )
        NULL
      },
      error = function(e) {
        fastkpc_legacy_dcov_batch_append_progress(
          paths$progress_csv,
          fastkpc_legacy_dcov_batch_progress_row(
            artifact_name = artifact_name,
            route = "candidate",
            batch_mode = batch_mode,
            batch_min_size = min_size,
            event = "error",
            status = "error",
            message = conditionMessage(e)
          )
        )
        stop(e)
      }
    )
    if (is.null(candidate)) {
      break
    }
    fastkpc_legacy_dcov_batch_append_progress(
      paths$progress_csv,
      fastkpc_legacy_dcov_batch_progress_row(
        artifact_name = artifact_name,
        route = "candidate",
        batch_mode = batch_mode,
        batch_min_size = min_size,
        event = "complete",
        status = "ok",
        elapsed_sec = candidate$elapsed
      )
    )
    name <- paste0(batch_mode, "_min_", min_size)
    candidates[[name]] <- candidate
    rows[[length(rows) + 1L]] <- fastkpc_legacy_dcov_batch_summary_row(
      artifact_name = artifact_name,
      route = "candidate",
      batch_mode = batch_mode,
      batch_min_size = min_size,
      run = candidate,
      reference = reference,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      n = nrow(data),
      p = ncol(data),
      index = index,
      numCol = numCol,
      reference_source = reference_source,
      reference_result_path = reference_result_path,
      timeout_sec =
        if (fastkpc_legacy_dcov_batch_timeout_enabled(candidate_timeout_sec)) {
          candidate_timeout_sec
        } else {
          NA_real_
        }
    )
    by_level_rows[[length(by_level_rows) + 1L]] <-
      fastkpc_legacy_dcov_batch_level_rows(
        route = "candidate",
        batch_mode = batch_mode,
        batch_min_size = min_size,
        run = candidate
      )
  }

  summary <- do.call(rbind, rows)
  runtime_by_level <- do.call(rbind, by_level_rows)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(runtime_by_level, paths$runtime_by_level_csv,
                   row.names = FALSE)
  saveRDS(
    list(
      summary = summary,
      runtime_by_level = runtime_by_level,
      reference = reference$value,
      reference_source = reference_source,
      reference_result_path = reference_result_path,
      candidates = lapply(candidates, function(item) item$value),
      paths = paths
    ),
    paths$result_rds
  )

  candidate_md <- summary[summary$route == "candidate", , drop = FALSE]
  writeLines(c(
    paste0("# ", artifact_name),
    "",
    paste0("- n / p: ", nrow(data), " / ", ncol(data)),
    paste0("- alpha: ", alpha),
    paste0("- max conditioning size: ", max_conditioning_size),
    paste0("- batch mode: ", batch_mode),
    paste0("- min sizes: ", paste(min_sizes, collapse = ", ")),
    paste0("- reference source: ", reference_source),
    paste0("- reference result path: ", reference_result_path %||% NA_character_),
    paste0("- reference elapsed sec: ",
           signif(summary$elapsed_sec[[1L]], 8L)),
    "",
    "## Candidate batch coverage",
    "",
    paste(
      sprintf(
        paste0(
          "- min_size=%s: status=%s, SHD=%s, elapsed=%s sec, ",
          "batch coverage=%s, skipped=%s"
        ),
        candidate_md$batch_min_size,
        candidate_md$run_status,
        candidate_md$shd,
        signif(candidate_md$elapsed_sec, 8L),
        signif(candidate_md$legacy_dcov_cpp_batch_pair_coverage_ratio, 6L),
        signif(candidate_md$legacy_dcov_cpp_batch_skipped_pair_ratio, 6L)
      ),
      collapse = "\n"
    )
  ), paths$summary_md)

  list(
    summary = summary,
    runtime_by_level = runtime_by_level,
    reference = reference$value,
    candidates = lapply(candidates, function(item) item$value),
    paths = paths
  )
}
