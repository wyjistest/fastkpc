source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

fastkpc_legacy_cuda_lowrank_restore_env <- function(old_env) {
  for (name in names(old_env)) {
    value <- old_env[[name]]
    if (is.na(value)) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(value), name))
    }
  }
  invisible(TRUE)
}

fastkpc_legacy_cuda_lowrank_edge_count <- function(adjacency) {
  adjacency <- as.matrix(adjacency)
  as.integer(sum(adjacency[upper.tri(adjacency)] != 0))
}

fastkpc_legacy_cuda_lowrank_shd <- function(candidate, reference) {
  candidate <- as.matrix(candidate)
  reference <- as.matrix(reference)
  as.integer(sum(candidate[upper.tri(candidate)] !=
                   reference[upper.tri(reference)]))
}

fastkpc_legacy_cuda_lowrank_pmax_diff <- function(candidate, reference) {
  if (is.null(candidate) || is.null(reference)) return(NA_real_)
  candidate <- as.matrix(candidate)
  reference <- as.matrix(reference)
  if (!identical(dim(candidate), dim(reference))) return(NA_real_)
  max(abs(candidate - reference), na.rm = TRUE)
}

fastkpc_legacy_cuda_lowrank_summary_value <- function(summary, name,
                                                       default = NA) {
  value <- summary[[name]]
  if (is.null(value)) default else value
}

fastkpc_legacy_cuda_lowrank_timeout_enabled <- function(candidate_timeout_sec) {
  !is.null(candidate_timeout_sec) &&
    length(candidate_timeout_sec) > 0L &&
    !is.na(candidate_timeout_sec[[1L]])
}

fastkpc_legacy_cuda_lowrank_timeout_error <- function(
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
    class = c("fastkpc_legacy_cuda_lowrank_timeout", "error", "condition")
  )
}

fastkpc_legacy_cuda_lowrank_is_time_limit <- function(error) {
  grepl("reached elapsed time limit|reached CPU time limit",
        conditionMessage(error))
}

fastkpc_legacy_cuda_lowrank_run_with_timeout <- function(
    fun, candidate_timeout_sec = NULL) {
  if (!fastkpc_legacy_cuda_lowrank_timeout_enabled(candidate_timeout_sec)) {
    return(fun())
  }
  timeout_sec <- as.numeric(candidate_timeout_sec[[1L]])
  start <- proc.time()[["elapsed"]]
  if (timeout_sec <= 0) {
    stop(fastkpc_legacy_cuda_lowrank_timeout_error(
      timeout_sec = timeout_sec,
      elapsed_sec = 0,
      message = "candidate timed out before execution"
    ))
  }

  if (identical(.Platform$OS.type, "unix")) {
    job <- parallel::mcparallel(
      tryCatch(
        list(ok = TRUE, value = fun()),
        error = function(e) {
          list(ok = FALSE, message = conditionMessage(e), class = class(e))
        }
      ),
      mc.set.seed = FALSE,
      silent = TRUE
    )
    cleanup_job <- TRUE
    on.exit({
      if (isTRUE(cleanup_job)) {
        try(tools::pskill(job$pid, 15L), silent = TRUE)
        try(parallel::mccollect(job, wait = FALSE), silent = TRUE)
      }
    }, add = TRUE)
    collected <- NULL
    repeat {
      collected <- parallel::mccollect(job, wait = FALSE)
      if (!is.null(collected) && length(collected) > 0L &&
          !is.null(collected[[1L]])) {
        break
      }
      elapsed <- proc.time()[["elapsed"]] - start
      if (elapsed >= timeout_sec) {
        cleanup_job <- FALSE
        try(tools::pskill(job$pid, 15L), silent = TRUE)
        Sys.sleep(0.2)
        try(tools::pskill(job$pid, 9L), silent = TRUE)
        try(parallel::mccollect(job, wait = FALSE), silent = TRUE)
        stop(fastkpc_legacy_cuda_lowrank_timeout_error(
          timeout_sec = timeout_sec,
          elapsed_sec = elapsed,
          message = paste0("candidate exceeded timeout_sec=", timeout_sec)
        ))
      }
      Sys.sleep(min(0.1, max(0.01, timeout_sec - elapsed)))
    }
    cleanup_job <- FALSE
    payload <- collected[[1L]]
    if (is.list(payload) && isTRUE(payload$ok)) {
      return(payload$value)
    }
    if (is.list(payload) && identical(payload$ok, FALSE)) {
      stop(simpleError(payload$message %||% "candidate failed"))
    }
    return(payload)
  }

  setTimeLimit(cpu = Inf, elapsed = timeout_sec, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  tryCatch(
    fun(),
    error = function(e) {
      if (fastkpc_legacy_cuda_lowrank_is_time_limit(e)) {
        stop(fastkpc_legacy_cuda_lowrank_timeout_error(
          timeout_sec = timeout_sec,
          elapsed_sec = proc.time()[["elapsed"]] - start,
          message = conditionMessage(e)
        ))
      }
      stop(e)
    }
  )
}

fastkpc_legacy_cuda_lowrank_progress_row <- function(
    artifact_name, route, event, status, elapsed_sec = NA_real_,
    message = NA_character_) {
  data.frame(
    artifact = artifact_name,
    route = route,
    event = event,
    status = status,
    elapsed_sec = as.numeric(elapsed_sec),
    message = message %||% NA_character_,
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %z"),
    stringsAsFactors = FALSE
  )
}

fastkpc_legacy_cuda_lowrank_append_progress <- function(path, row) {
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

fastkpc_legacy_cuda_lowrank_extract_reference <- function(result) {
  candidates <- list(
    skeleton = result$skeleton,
    reference = result$reference,
    baseline = result$baseline,
    candidate = result$candidate
  )
  for (name in names(candidates)) {
    candidate <- candidates[[name]]
    if (is.list(candidate) &&
        !is.null(candidate$adjacency) &&
        !is.null(candidate$n.edgetests)) {
      attr(candidate, "fastkpc_reference_slot") <- name
      return(candidate)
    }
  }
  if (is.list(result) &&
      !is.null(result$adjacency) &&
      !is.null(result$n.edgetests)) {
    attr(result, "fastkpc_reference_slot") <- "root"
    return(result)
  }
  stop("reference result does not contain a skeleton-like object",
       call. = FALSE)
}

fastkpc_run_legacy_dcov_cuda_lowrank_backend_real_subset_artifact <- function(
    output_dir = "fastkpc/artifacts/legacy_dcov_cuda_lowrank_backend_real_subset_v1",
    artifact_name = "legacy_dcov_cuda_lowrank_backend_real_subset_v1",
    data_path = "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
    columns = c(1L, 2L, 3L, 4L, 5L),
    alpha = 0.1,
    max_conditioning_size = 3L,
    rebuild_cuda = FALSE,
    parallel_cores = 1L,
    reference_result_path = NULL,
    expected_edge_count = NULL,
    expected_n_edgetests = NULL,
    candidate_timeout_sec = NULL) {
  if (fastkpc_legacy_cuda_lowrank_timeout_enabled(candidate_timeout_sec)) {
    candidate_timeout_sec <- as.numeric(candidate_timeout_sec[[1L]])
    if (!is.finite(candidate_timeout_sec) || candidate_timeout_sec < 0) {
      stop("candidate_timeout_sec must be NULL or a non-negative finite number",
           call. = FALSE)
    }
  }
  if (!file.exists(data_path)) {
    stop("real data fixture not found: ", data_path, call. = FALSE)
  }
  build_fastkpc_cuda_native(rebuild = isTRUE(rebuild_cuda))
  if (!fastkpc_cuda_available()) {
    stop("CUDA unavailable for legacy dCov CUDA lowrank backend artifact",
         call. = FALSE)
  }

  real_data <- readRDS(data_path)
  if (is.null(columns)) {
    data <- as.matrix(real_data)
  } else {
    data <- as.matrix(real_data[, as.integer(columns), drop = FALSE])
  }
  storage.mode(data) <- "double"
  n <- nrow(data)
  p <- ncol(data)
  columns_label <- if (is.null(columns)) {
    "all"
  } else {
    paste(as.integer(columns), collapse = ",")
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    progress_csv = file.path(output_dir, "progress.csv"),
    legacy_progress_csv = file.path(output_dir, "legacy_progress.csv"),
    summary_md = file.path(output_dir, "summary.md"),
    result_rds = file.path(output_dir, "result.rds")
  )
  if (file.exists(paths$progress_csv)) unlink(paths$progress_csv)
  if (file.exists(paths$legacy_progress_csv)) unlink(paths$legacy_progress_csv)

  tracked_env <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
    "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW",
    "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
    "FASTKPC_LEGACY_PARALLEL_CORES",
    "FASTKPC_LEGACY_PROGRESS_CSV"
  )
  old_env <- stats::setNames(
    lapply(tracked_env, Sys.getenv, unset = NA_character_),
    tracked_env
  )
  on.exit(fastkpc_legacy_cuda_lowrank_restore_env(old_env), add = TRUE)

  run_route <- function(lowrank_mode) {
    if (is.null(parallel_cores) || is.na(parallel_cores)) {
      parallel_value <- "1"
    } else {
      parallel_value <- as.character(as.integer(parallel_cores))
    }
    Sys.setenv(
      FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
      FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = lowrank_mode,
      FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
      FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
      FASTKPC_LEGACY_PARALLEL_CORES = parallel_value,
      FASTKPC_LEGACY_PROGRESS_CSV = paths$legacy_progress_csv
    )
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH")
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW")
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS")
    start <- proc.time()[["elapsed"]]
    result <- fast_kpc(
      data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      engine = "cuda",
      precision = "compatible",
      graph_stage = "skeleton",
      ci_method = "dcc.gamma",
      precision_trace_level = "summary",
      benchmark = TRUE
    )
    list(
      result = result,
      elapsed_sec = proc.time()[["elapsed"]] - start,
      summary = result$skeleton$scheduler_diagnostics$summary
    )
  }

  fastkpc_legacy_cuda_lowrank_append_progress(
    paths$progress_csv,
    fastkpc_legacy_cuda_lowrank_progress_row(
      artifact_name, "reference", "start",
      if (is.null(reference_result_path)) "computed" else "rds"
    )
  )
  if (is.null(reference_result_path)) {
    baseline <- run_route("spectra")
    reference_slot <- "computed"
  } else {
    if (!file.exists(reference_result_path)) {
      stop("reference result not found: ", reference_result_path,
           call. = FALSE)
    }
    reference <- fastkpc_legacy_cuda_lowrank_extract_reference(
      readRDS(reference_result_path)
    )
    reference_slot <- attr(reference, "fastkpc_reference_slot", exact = TRUE)
    baseline <- list(
      result = list(skeleton = reference),
      elapsed_sec = 0,
      summary = reference$summary %||%
        reference$scheduler_diagnostics$summary %||%
        list()
    )
  }
  fastkpc_legacy_cuda_lowrank_append_progress(
    paths$progress_csv,
    fastkpc_legacy_cuda_lowrank_progress_row(
      artifact_name, "reference", "complete",
      if (is.null(reference_result_path)) "computed" else "rds",
      elapsed_sec = baseline$elapsed_sec
    )
  )

  fastkpc_legacy_cuda_lowrank_append_progress(
    paths$progress_csv,
    fastkpc_legacy_cuda_lowrank_progress_row(
      artifact_name, "candidate", "start", "running"
    )
  )
  timeout_error <- NULL
  candidate <- tryCatch(
    fastkpc_legacy_cuda_lowrank_run_with_timeout(
      function() run_route("cuda_spectra"),
      candidate_timeout_sec = candidate_timeout_sec
    ),
    fastkpc_legacy_cuda_lowrank_timeout = function(e) {
      timeout_error <<- e
      NULL
    }
  )

  baseline_skeleton <- baseline$result$skeleton
  if (is.null(candidate)) {
    fastkpc_legacy_cuda_lowrank_append_progress(
      paths$progress_csv,
      fastkpc_legacy_cuda_lowrank_progress_row(
        artifact_name, "candidate", "timeout", "timeout",
        elapsed_sec = timeout_error$elapsed_sec,
        message = conditionMessage(timeout_error)
      )
    )
    row <- data.frame(
      artifact_name = artifact_name,
      data_path = data_path,
      reference_result_path = reference_result_path %||% NA_character_,
      reference_result_slot = reference_slot,
      columns = columns_label,
      n = as.integer(n),
      p = as.integer(p),
      alpha = as.numeric(alpha),
      max_conditioning_size = as.integer(max_conditioning_size),
      parallel_cores = as.integer(parallel_cores %||% NA_integer_),
      run_status = "timeout",
      timeout = TRUE,
      timeout_sec = as.numeric(timeout_error$timeout_sec),
      expected_edge_count = expected_edge_count %||% NA_integer_,
      expected_n_edgetests = if (is.null(expected_n_edgetests)) {
        NA_character_
      } else {
        paste(as.integer(expected_n_edgetests), collapse = ",")
      },
      baseline_elapsed_sec = as.numeric(baseline$elapsed_sec),
      candidate_elapsed_sec = as.numeric(timeout_error$elapsed_sec),
      baseline_edge_count =
        fastkpc_legacy_cuda_lowrank_edge_count(baseline_skeleton$adjacency),
      candidate_edge_count = NA_integer_,
      adjacency_identical = NA,
      shd = NA_integer_,
      n_edgetests_exact = NA,
      baseline_n_edgetests =
        paste(as.integer(baseline_skeleton$n.edgetests), collapse = ","),
      candidate_n_edgetests = NA_character_,
      pmax_max_abs_diff = NA_real_,
      baseline_legacy_dcov_gamma_count =
        as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
          baseline$summary, "legacy_dcov_gamma_count", 0L)),
      baseline_legacy_dcov_cpp_backend_count =
        as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
          baseline$summary, "legacy_dcov_cpp_backend_count", 0L)),
      candidate_legacy_dcov_gamma_count = NA_integer_,
      candidate_legacy_dcov_cpp_backend_count = NA_integer_,
      candidate_legacy_dcov_r_backend_count = NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_enabled = NA,
      candidate_legacy_dcov_cuda_lowrank_backend_count = NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_ms = NA_real_,
      candidate_legacy_dcov_cuda_lowrank_backend_error_count = NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_fallback_count = NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_cpu_fallback_count =
        NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_converged_count =
        NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_spectra_matvec_count =
        NA_integer_,
      candidate_legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max =
        NA_real_,
      candidate_legacy_dcov_cuda_lowrank_backend_matrix_bytes = NA_real_,
      candidate_legacy_dcov_cuda_lowrank_backend_workspace_realloc_count =
        NA_integer_,
      stringsAsFactors = FALSE
    )
    utils::write.csv(row, paths$summary_csv, row.names = FALSE)
    saveRDS(
      list(summary = row, baseline = baseline, candidate = NULL,
           paths = paths),
      paths$result_rds
    )
    writeLines(c(
      paste0("# ", artifact_name),
      "",
      paste0("- n / p: ", row$n[[1L]], " / ", row$p[[1L]]),
      paste0("- run status: ", row$run_status[[1L]]),
      paste0("- timeout sec: ", row$timeout_sec[[1L]]),
      paste0("- baseline edge count: ", row$baseline_edge_count[[1L]]),
      paste0("- elapsed sec: ", signif(row$candidate_elapsed_sec[[1L]], 8L)),
      paste0("- legacy progress: ", paths$legacy_progress_csv)
    ), paths$summary_md)
    return(list(summary = row, paths = paths, baseline = baseline,
                candidate = NULL, output_dir = output_dir))
  }
  fastkpc_legacy_cuda_lowrank_append_progress(
    paths$progress_csv,
    fastkpc_legacy_cuda_lowrank_progress_row(
      artifact_name, "candidate", "complete", "ok",
      elapsed_sec = candidate$elapsed_sec
    )
  )

  candidate_skeleton <- candidate$result$skeleton
  baseline_summary <- baseline$summary
  candidate_summary <- candidate$summary

  adjacency_identical <- identical(candidate_skeleton$adjacency,
                                   baseline_skeleton$adjacency)
  n_edgetests_exact <- identical(candidate_skeleton$n.edgetests,
                                 baseline_skeleton$n.edgetests)
  shd <- fastkpc_legacy_cuda_lowrank_shd(
    candidate_skeleton$adjacency,
    baseline_skeleton$adjacency
  )

  row <- data.frame(
    artifact_name = artifact_name,
    data_path = data_path,
    reference_result_path = reference_result_path %||% NA_character_,
    reference_result_slot = reference_slot,
    columns = columns_label,
    n = as.integer(n),
    p = as.integer(p),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    parallel_cores = as.integer(parallel_cores %||% NA_integer_),
    run_status = "ok",
    timeout = FALSE,
    timeout_sec =
      if (fastkpc_legacy_cuda_lowrank_timeout_enabled(candidate_timeout_sec)) {
        as.numeric(candidate_timeout_sec)
      } else {
        NA_real_
      },
    expected_edge_count = expected_edge_count %||% NA_integer_,
    expected_n_edgetests = if (is.null(expected_n_edgetests)) {
      NA_character_
    } else {
      paste(as.integer(expected_n_edgetests), collapse = ",")
    },
    baseline_elapsed_sec = as.numeric(baseline$elapsed_sec),
    candidate_elapsed_sec = as.numeric(candidate$elapsed_sec),
    baseline_edge_count =
      fastkpc_legacy_cuda_lowrank_edge_count(baseline_skeleton$adjacency),
    candidate_edge_count =
      fastkpc_legacy_cuda_lowrank_edge_count(candidate_skeleton$adjacency),
    adjacency_identical = adjacency_identical,
    shd = as.integer(shd),
    n_edgetests_exact = n_edgetests_exact,
    baseline_n_edgetests =
      paste(as.integer(baseline_skeleton$n.edgetests), collapse = ","),
    candidate_n_edgetests =
      paste(as.integer(candidate_skeleton$n.edgetests), collapse = ","),
    pmax_max_abs_diff = fastkpc_legacy_cuda_lowrank_pmax_diff(
      candidate_skeleton$pMax, baseline_skeleton$pMax
    ),
    baseline_legacy_dcov_gamma_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        baseline_summary, "legacy_dcov_gamma_count", 0L)),
    baseline_legacy_dcov_cpp_backend_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        baseline_summary, "legacy_dcov_cpp_backend_count", 0L)),
    candidate_legacy_dcov_gamma_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_gamma_count", 0L)),
    candidate_legacy_dcov_cpp_backend_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_cpp_backend_count", 0L)),
    candidate_legacy_dcov_r_backend_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_r_backend_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_enabled =
      isTRUE(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_cuda_lowrank_backend_enabled", FALSE)),
    candidate_legacy_dcov_cuda_lowrank_backend_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_cuda_lowrank_backend_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_ms =
      as.numeric(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_cuda_lowrank_backend_ms", 0)),
    candidate_legacy_dcov_cuda_lowrank_backend_error_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary, "legacy_dcov_cuda_lowrank_backend_error_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_fallback_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_fallback_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_cpu_fallback_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_cpu_fallback_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_converged_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_converged_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_spectra_matvec_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_spectra_matvec_count", 0L)),
    candidate_legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max =
      as.numeric(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max",
        0)),
    candidate_legacy_dcov_cuda_lowrank_backend_matrix_bytes =
      as.numeric(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_matrix_bytes", 0)),
    candidate_legacy_dcov_cuda_lowrank_backend_workspace_realloc_count =
      as.integer(fastkpc_legacy_cuda_lowrank_summary_value(
        candidate_summary,
        "legacy_dcov_cuda_lowrank_backend_workspace_realloc_count", 0L)),
    stringsAsFactors = FALSE
  )

  utils::write.csv(row, paths$summary_csv, row.names = FALSE)
  saveRDS(
    list(summary = row, baseline = baseline, candidate = candidate,
         paths = paths),
    paths$result_rds
  )
  writeLines(c(
    paste0("# ", artifact_name),
    "",
    paste0("- n / p: ", row$n[[1L]], " / ", row$p[[1L]]),
    paste0("- edge count: ", row$candidate_edge_count[[1L]], " / ",
           row$baseline_edge_count[[1L]]),
    paste0("- SHD: ", row$shd[[1L]]),
    paste0("- n.edgetests exact: ", row$n_edgetests_exact[[1L]]),
    paste0("- CUDA lowrank backend calls: ",
           row$candidate_legacy_dcov_cuda_lowrank_backend_count[[1L]]),
    paste0("- CUDA lowrank backend errors: ",
           row$candidate_legacy_dcov_cuda_lowrank_backend_error_count[[1L]]),
    paste0("- CUDA lowrank backend fallbacks: ",
           row$candidate_legacy_dcov_cuda_lowrank_backend_fallback_count[[1L]]),
    paste0("- matrix H2D during compute max ms: ",
           row$candidate_legacy_dcov_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max[[1L]]),
    paste0("- candidate elapsed sec: ",
           signif(row$candidate_elapsed_sec[[1L]], 8L)),
    paste0("- baseline elapsed sec: ",
           signif(row$baseline_elapsed_sec[[1L]], 8L)),
    paste0("- legacy progress: ", paths$legacy_progress_csv)
  ), paths$summary_md)

  list(summary = row, paths = paths, baseline = baseline,
       candidate = candidate, output_dir = output_dir)
}

fastkpc_run_legacy_dcov_cuda_lowrank_backend_hot12_artifact <- function(
    output_dir = "fastkpc/artifacts/legacy_dcov_cuda_lowrank_backend_hot12_v1",
    artifact_name = "legacy_dcov_cuda_lowrank_backend_hot12_v1",
    data_path = "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
    alpha = 0.1,
    max_conditioning_size = 3L,
    rebuild_cuda = FALSE,
    parallel_cores = 1L,
    reference_result_path = NULL,
    candidate_timeout_sec = NULL) {
  fastkpc_run_legacy_dcov_cuda_lowrank_backend_real_subset_artifact(
    output_dir = output_dir,
    artifact_name = artifact_name,
    data_path = data_path,
    columns = c(1L, 2L, 3L, 4L, 5L, 6L, 9L, 12L, 15L, 16L, 17L, 18L),
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    rebuild_cuda = rebuild_cuda,
    parallel_cores = parallel_cores,
    reference_result_path = reference_result_path,
    candidate_timeout_sec = candidate_timeout_sec
  )
}

fastkpc_run_legacy_dcov_cuda_lowrank_backend_full_artifact <- function(
    output_dir = "fastkpc/artifacts/legacy_dcov_cuda_lowrank_backend_full_351x48_v1",
    artifact_name = "legacy_dcov_cuda_lowrank_backend_full_351x48_v1",
    data_path = "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
    reference_result_path =
      paste0("fastkpc/artifacts/legacy_mgcv_residual_cache_s_affinity_v1/",
             "compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds"),
    alpha = 0.1,
    max_conditioning_size = 46L,
    rebuild_cuda = FALSE,
    parallel_cores = 20L,
    candidate_timeout_sec = NULL,
    expected_edge_count = 110L,
    expected_n_edgetests =
      c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L)) {
  fastkpc_run_legacy_dcov_cuda_lowrank_backend_real_subset_artifact(
    output_dir = output_dir,
    artifact_name = artifact_name,
    data_path = data_path,
    columns = NULL,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    rebuild_cuda = rebuild_cuda,
    parallel_cores = parallel_cores,
    reference_result_path = reference_result_path,
    expected_edge_count = expected_edge_count,
    expected_n_edgetests = expected_n_edgetests,
    candidate_timeout_sec = candidate_timeout_sec
  )
}
