source("fastkpc/R/fast_kpc.R")

fastkpc_compatible_cuda_restore_env <- function(old_env) {
  for (name in names(old_env)) {
    if (is.na(old_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_env[[name]]), name))
    }
  }
  invisible(TRUE)
}

fastkpc_compatible_cuda_compare_sepsets <- function(left, right) {
  if (length(left) != length(right)) return(FALSE)
  for (i in seq_along(left)) {
    if (length(left[[i]]) != length(right[[i]])) return(FALSE)
    for (j in seq_along(left[[i]])) {
      lhs <- sort(as.integer(left[[i]][[j]]))
      rhs <- sort(as.integer(right[[i]][[j]]))
      if (!identical(lhs, rhs)) return(FALSE)
    }
  }
  TRUE
}

fastkpc_compatible_cuda_pmax_max_abs_diff <- function(left, right) {
  left <- unname(left)
  right <- unname(right)
  finite <- is.finite(left) & is.finite(right)
  if (!any(finite)) return(0)
  max(abs(left[finite] - right[finite]))
}

fastkpc_compatible_cuda_skeleton_shd <- function(left, right) {
  left <- unname(left)
  right <- unname(right)
  sum(abs(as.integer(left) - as.integer(right))) / 2L
}

fastkpc_compatible_cuda_timeout_enabled <- function(candidate_timeout_sec) {
  !is.null(candidate_timeout_sec) &&
    length(candidate_timeout_sec) > 0L &&
    !is.na(candidate_timeout_sec[[1L]])
}

fastkpc_compatible_cuda_timeout_error <- function(
    timeout_sec, elapsed_sec, message = NULL) {
  if (is.null(message)) {
    message <- paste0("facade candidate exceeded timeout_sec=", timeout_sec)
  }
  structure(
    list(
      message = message,
      call = NULL,
      timeout_sec = as.numeric(timeout_sec),
      elapsed_sec = as.numeric(elapsed_sec)
    ),
    class = c("fastkpc_compatible_cuda_timeout", "error", "condition")
  )
}

fastkpc_compatible_cuda_is_time_limit <- function(error) {
  grepl("reached elapsed time limit|reached CPU time limit",
        conditionMessage(error))
}

fastkpc_compatible_cuda_run_with_timeout <- function(
    fun, candidate_timeout_sec = NULL) {
  if (!fastkpc_compatible_cuda_timeout_enabled(candidate_timeout_sec)) {
    return(fun())
  }
  timeout_sec <- as.numeric(candidate_timeout_sec[[1L]])
  start <- proc.time()[["elapsed"]]
  if (timeout_sec <= 0) {
    stop(fastkpc_compatible_cuda_timeout_error(
      timeout_sec = timeout_sec,
      elapsed_sec = 0,
      message = "facade candidate timed out before execution"
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
        stop(fastkpc_compatible_cuda_timeout_error(
          timeout_sec = timeout_sec,
          elapsed_sec = elapsed,
          message = paste0("facade candidate exceeded timeout_sec=", timeout_sec)
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
      stop(simpleError(payload$message %||% "facade candidate failed"))
    }
    return(payload)
  }

  setTimeLimit(cpu = Inf, elapsed = timeout_sec, transient = TRUE)
  on.exit(setTimeLimit(cpu = Inf, elapsed = Inf, transient = FALSE), add = TRUE)
  tryCatch(
    fun(),
    error = function(e) {
      if (fastkpc_compatible_cuda_is_time_limit(e)) {
        stop(fastkpc_compatible_cuda_timeout_error(
          timeout_sec = timeout_sec,
          elapsed_sec = proc.time()[["elapsed"]] - start,
          message = conditionMessage(e)
        ))
      }
      stop(e)
    }
  )
}

fastkpc_compatible_cuda_progress_row <- function(
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

fastkpc_compatible_cuda_append_progress <- function(path, row) {
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

fastkpc_compatible_cuda_share <- function(value, total) {
  value <- as.numeric(value[[1L]])
  total <- as.numeric(total[[1L]])
  if (!is.finite(value) || !is.finite(total) || total <= 0) {
    return(NA_real_)
  }
  value / total
}

fastkpc_compatible_cuda_native_dcov_stage_rows <- function(row) {
  stage <- c(
    "materialize", "call_wall", "input", "distance", "lowrank",
    "lowrank_eig", "lowrank_select", "lowrank_center",
    "lowrank_unaccounted", "statistic", "moment", "pgamma", "accounted",
    "scalar_total", "wrapper_overhead", "batch_overhead"
  )
  elapsed_ms <- c(
    row$legacy_dcov_native_batch_materialize_ms[[1L]],
    row$legacy_dcov_native_batch_call_ms[[1L]],
    row$legacy_dcov_native_batch_input_ms[[1L]],
    row$legacy_dcov_native_batch_distance_ms[[1L]],
    row$legacy_dcov_native_batch_lowrank_ms[[1L]],
    row$legacy_dcov_native_batch_lowrank_eig_ms[[1L]],
    row$legacy_dcov_native_batch_lowrank_select_ms[[1L]],
    row$legacy_dcov_native_batch_lowrank_center_ms[[1L]],
    row$legacy_dcov_native_batch_lowrank_unaccounted_ms[[1L]],
    row$legacy_dcov_native_batch_statistic_ms[[1L]],
    row$legacy_dcov_native_batch_moment_ms[[1L]],
    row$legacy_dcov_native_batch_pgamma_ms[[1L]],
    row$legacy_dcov_native_batch_accounted_ms[[1L]],
    row$legacy_dcov_native_batch_scalar_total_ms[[1L]],
    row$legacy_dcov_native_batch_wrapper_overhead_ms[[1L]],
    row$legacy_dcov_native_batch_overhead_ms[[1L]]
  )
  scalar_total <- row$legacy_dcov_native_batch_scalar_total_ms[[1L]]
  batch_call <- row$legacy_dcov_native_batch_call_ms[[1L]]
  data.frame(
    artifact = row$artifact[[1L]],
    route = row$route[[1L]],
    batch_mode = row$legacy_dcov_native_batch_mode[[1L]],
    direct_input = row$legacy_dcov_native_batch_direct_input_enabled[[1L]],
    stage = stage,
    elapsed_ms = as.numeric(elapsed_ms),
    share_of_scalar_total = vapply(
      elapsed_ms,
      fastkpc_compatible_cuda_share,
      numeric(1),
      total = scalar_total
    ),
    share_of_batch_call = vapply(
      elapsed_ms,
      fastkpc_compatible_cuda_share,
      numeric(1),
      total = batch_call
    ),
    stringsAsFactors = FALSE
  )
}

fastkpc_compatible_cuda_extract_reference <- function(result) {
  candidates <- list(
    reference = result$reference,
    skeleton = result$skeleton,
    baseline = result$baseline,
    backend = result$backend,
    facade = result$facade
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

fastkpc_compatible_cuda_timeout_summary_row <- function(
    artifact_name, data, columns, alpha, max_conditioning_size, index, numCol,
    dcov_batch, low_rank, mgcv_residual_backend,
    mgcv_residual_backend_native_s_size_limit,
    mgcv_residual_backend_condition_threshold, reference,
    reference_source, reference_result_path, reference_slot,
    expected_edge_count, expected_n_edgetests, timeout_sec, elapsed_sec,
    reference_elapsed_sec) {
  reference_edge_count <- as.integer(sum(unname(reference$adjacency)) / 2L)
  data.frame(
    artifact = artifact_name,
    route = "facade",
    n = nrow(data),
    p = ncol(data),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    columns = if (is.null(columns)) "all" else paste(columns, collapse = "|"),
    index = as.integer(index),
    numCol = as.integer(numCol),
    dcov_batch = dcov_batch,
    low_rank = low_rank,
    mgcv_residual_backend = mgcv_residual_backend,
    mgcv_residual_backend_native_s_size_limit =
      mgcv_residual_backend_native_s_size_limit %||% NA_real_,
    mgcv_residual_backend_condition_threshold =
      mgcv_residual_backend_condition_threshold %||% NA_real_,
    reference_source = reference_source,
    reference_result_path = reference_result_path %||% NA_character_,
    reference_result_slot = reference_slot,
    run_status = "timeout",
    timeout = TRUE,
    timeout_sec = as.numeric(timeout_sec),
    edge_count = NA_integer_,
    reference_edge_count = reference_edge_count,
    expected_edge_count = expected_edge_count %||% NA_integer_,
    expected_edge_count_match = NA,
    shd = NA_integer_,
    adjacency_identical = NA,
    sepsets_identical = NA,
    n_edgetests_identical = NA,
    n_edgetests_exact = NA,
    facade_n_edgetests = NA_character_,
    reference_n_edgetests = paste(reference$n.edgetests, collapse = "|"),
    expected_n_edgetests = if (is.null(expected_n_edgetests)) {
      NA_character_
    } else {
      paste(expected_n_edgetests, collapse = "|")
    },
    pmax_max_abs_diff = NA_real_,
    residual_provider_request_count = NA_integer_,
    reference_residual_provider_request_count = NA_integer_,
    residual_provider_call_ms = NA_real_,
    residual_provider_matrix_copy_ms = NA_real_,
    residual_provider_total_ms = NA_real_,
    residual_provider_parallel_enabled = NA,
    residual_provider_parallel_cores = NA_integer_,
    residual_provider_parallel_level_count = NA_integer_,
    residual_provider_parallel_request_count = NA_integer_,
    legacy_dcov_native_count = NA_integer_,
    reference_legacy_dcov_native_count = NA_integer_,
    legacy_dcov_native_batch_enabled = NA,
    legacy_dcov_native_batch_mode = NA_character_,
    legacy_dcov_native_batch_count = NA_integer_,
    legacy_dcov_native_batch_pair_count = NA_integer_,
    legacy_dcov_native_batch_parallel_enabled = NA,
    legacy_dcov_native_batch_parallel_threads = NA_integer_,
    legacy_dcov_native_batch_direct_input_enabled = NA,
    legacy_dcov_native_batch_column_materialize_count = NA_integer_,
    legacy_dcov_native_batch_materialize_ms = NA_real_,
    legacy_dcov_native_batch_call_ms = NA_real_,
    legacy_dcov_native_batch_input_ms = NA_real_,
    legacy_dcov_native_batch_distance_ms = NA_real_,
    legacy_dcov_native_batch_lowrank_ms = NA_real_,
    legacy_dcov_native_batch_lowrank_eig_ms = NA_real_,
    legacy_dcov_native_batch_lowrank_select_ms = NA_real_,
    legacy_dcov_native_batch_lowrank_center_ms = NA_real_,
    legacy_dcov_native_batch_lowrank_unaccounted_ms = NA_real_,
    legacy_dcov_native_lowrank_mode = NA_character_,
    legacy_dcov_native_lowrank_full_eig_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_converged_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_failed_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_fallback_full_eig_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_iterations = NA_integer_,
    legacy_dcov_native_lowrank_spectra_nconv = NA_integer_,
    legacy_dcov_native_lowrank_spectra_ncv = NA_integer_,
    legacy_dcov_native_lowrank_spectra_tol = NA_real_,
    legacy_dcov_native_lowrank_spectra_matvec_count = NA_integer_,
    legacy_dcov_native_lowrank_spectra_matvec_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_enabled = NA,
    legacy_dcov_native_cuda_lowrank_backend_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_error_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_fallback_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_converged_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_enabled = NA,
    legacy_dcov_native_cuda_lowrank_component_cache_scope = NA_character_,
    legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_lookup_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_hit_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_miss_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_entry_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_eviction_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_component_distance_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_component_moment_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_device_matrix_reuse_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_device_workspace_reuse_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_workspace_realloc_count = NA_integer_,
    legacy_dcov_native_cuda_lowrank_backend_matrix_bytes = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_workspace_bytes = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_workspace_alloc_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_h2d_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_kernel_ms = NA_real_,
    legacy_dcov_native_cuda_lowrank_backend_d2h_ms = NA_real_,
    legacy_dcov_native_batch_statistic_ms = NA_real_,
    legacy_dcov_native_batch_moment_ms = NA_real_,
    legacy_dcov_native_batch_pgamma_ms = NA_real_,
    legacy_dcov_native_batch_accounted_ms = NA_real_,
    legacy_dcov_native_batch_scalar_total_ms = NA_real_,
    legacy_dcov_native_batch_wrapper_overhead_ms = NA_real_,
    legacy_dcov_native_batch_overhead_ms = NA_real_,
    compatible_cuda_facade = NA,
    compatible_cuda_route = NA_character_,
    compatible_cuda_residual_authority = NA_character_,
    compatible_cuda_ci_authority = NA_character_,
    residual_provider_response_backend = NA_character_,
    residual_provider_mgcv_backend = NA_character_,
    residual_provider_mgcv_cpp_backend_enabled = NA,
    residual_provider_mgcv_cpp_backend_count = NA_integer_,
    residual_provider_mgcv_cpp_backend_native_count = NA_integer_,
    residual_provider_mgcv_cpp_backend_fallback_count = NA_integer_,
    residual_provider_mgcv_cpp_backend_error_count = NA_integer_,
    residual_provider_mgcv_cpp_backend_ms = NA_real_,
    residual_provider_mgcv_cpp_backend_native_solve_ms = NA_real_,
    elapsed_sec = as.numeric(elapsed_sec),
    reference_elapsed_sec = as.numeric(reference_elapsed_sec),
    stringsAsFactors = FALSE
  )
}

fastkpc_run_compatible_cuda_skeleton_artifact <- function(
    data = NULL,
    data_path = file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
                          "cancer_RD-causalDiscoveryInput.rds"),
    columns = NULL,
    output_dir = file.path("fastkpc", "artifacts",
                           "compatible_cuda_skeleton_full_351x48_v1"),
    artifact_name = basename(output_dir),
    alpha = 0.1,
    max_conditioning_size = NULL,
    index = 1,
    numCol = NULL,
    trace_level = "summary",
    dcov_batch = "level",
    low_rank = "spectra",
    mgcv_residual_backend = c("env", "r", "cpp_guarded"),
    mgcv_residual_backend_native_s_size_limit = NULL,
    mgcv_residual_backend_condition_threshold = NULL,
    reference_result_path = NULL,
    expected_edge_count = NULL,
    expected_n_edgetests = NULL,
    candidate_timeout_sec = NULL) {
  mgcv_residual_backend <- match.arg(mgcv_residual_backend)
  low_rank <- as.character(low_rank %||% "")
  if (length(low_rank) != 1L || is.na(low_rank)) {
    stop("low_rank must be a single character value", call. = FALSE)
  }
  supported_low_rank <- c("", "full_eig", "spectra", "selected",
                          "selected_eigs", "cuda_spectra")
  if (!(low_rank %in% supported_low_rank)) {
    stop(
      "unsupported low_rank for native compatible CUDA skeleton artifact: ",
      low_rank,
      ". Supported values are: ",
      paste(supported_low_rank[nzchar(supported_low_rank)], collapse = ", "),
      " or empty string to unset the lowrank env.",
      call. = FALSE
    )
  }
  if (fastkpc_compatible_cuda_timeout_enabled(candidate_timeout_sec)) {
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
    stop("compatible CUDA skeleton artifact data must contain at least four columns",
         call. = FALSE)
  }
  labels <- colnames(data)
  if (is.null(labels)) {
    labels <- paste0("V", seq_len(ncol(data)))
    colnames(data) <- labels
  }
  if (is.null(max_conditioning_size)) {
    max_conditioning_size <- ncol(data) - 2L
  }
  max_conditioning_size <- min(as.integer(max_conditioning_size),
                               ncol(data) - 2L)
  if (is.null(numCol)) {
    numCol <- floor(nrow(data) / 10)
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    progress_csv = file.path(output_dir, "progress.csv"),
    native_progress_csv = file.path(output_dir, "native_progress.csv"),
    native_dcov_stage_csv = file.path(output_dir,
                                      "native_dcov_stage_timing.csv"),
    native_lowrank_cache_progress_csv = file.path(
      output_dir,
      "native_lowrank_component_cache_progress.csv"
    ),
    result_rds = file.path(output_dir, "result.rds"),
    summary_md = file.path(output_dir, "summary.md")
  )
  if (file.exists(paths$progress_csv)) unlink(paths$progress_csv)
  if (file.exists(paths$native_progress_csv)) unlink(paths$native_progress_csv)
  if (file.exists(paths$native_lowrank_cache_progress_csv)) {
    unlink(paths$native_lowrank_cache_progress_csv)
  }

  env_names <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD",
    "FASTKPC_NATIVE_LEGACY_PROGRESS_CSV",
    "FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_PROGRESS_CSV"
  )
  old_env <- Sys.getenv(env_names, unset = NA_character_)
  on.exit(fastkpc_compatible_cuda_restore_env(old_env), add = TRUE)
  if (nzchar(low_rank)) {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = low_rank)
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  }

  provider_counts <- new.env(parent = emptyenv())
  provider_counts$level_calls <- 0L
  provider_counts$request_count <- 0L

  reference_source <- if (is.null(reference_result_path)) "computed" else "rds"
  reference_slot <- "computed"
  fastkpc_compatible_cuda_append_progress(
    paths$progress_csv,
    fastkpc_compatible_cuda_progress_row(
      artifact_name = artifact_name,
      route = "reference",
      event = "start",
      status = reference_source
    )
  )
  if (is.null(reference_result_path)) {
    reference_timed <- fastkpc_elapsed(
      precision_run_skeleton_residual_provider_legacy_dcov_native(
        data = data,
        alpha = alpha,
        max_conditioning_size = max_conditioning_size,
        residual_provider = fastkpc_legacy_mgcv_residual_provider(
          data = data,
          counter_env = provider_counts,
          backend = "r"
        ),
        index = index,
        numCol = numCol,
        trace_level = trace_level
      )
    )
    reference <- reference_timed$value
  } else {
    if (!file.exists(reference_result_path)) {
      stop("reference result not found: ", reference_result_path,
           call. = FALSE)
    }
    loaded_reference <- readRDS(reference_result_path)
    reference <- fastkpc_compatible_cuda_extract_reference(loaded_reference)
    reference_slot <- attr(reference, "fastkpc_reference_slot", exact = TRUE)
    reference_timed <- list(
      value = reference,
      elapsed = 0
    )
  }
  fastkpc_compatible_cuda_append_progress(
    paths$progress_csv,
    fastkpc_compatible_cuda_progress_row(
      artifact_name = artifact_name,
      route = "reference",
      event = "complete",
      status = reference_source,
      elapsed_sec = reference_timed$elapsed
    )
  )

  fastkpc_compatible_cuda_append_progress(
    paths$progress_csv,
    fastkpc_compatible_cuda_progress_row(
      artifact_name = artifact_name,
      route = "facade",
      event = "start",
      status = "running"
    )
  )
  Sys.setenv(FASTKPC_NATIVE_LEGACY_PROGRESS_CSV =
               paths$native_progress_csv)
  Sys.setenv(
    FASTKPC_NATIVE_CUDA_LOWRANK_COMPONENT_CACHE_PROGRESS_CSV =
      paths$native_lowrank_cache_progress_csv
  )
  timeout_error <- NULL
  facade_timed <- tryCatch(
    fastkpc_compatible_cuda_run_with_timeout(
      function() {
        fastkpc_elapsed(
          fastkpc_compatible_cuda_skeleton(
            data = data,
            alpha = alpha,
            labels = labels,
            options = list(
              max_conditioning_size = max_conditioning_size,
              index = index,
              numCol = numCol,
              trace_level = trace_level,
              dcov_batch = dcov_batch,
              mgcv_residual_backend = mgcv_residual_backend,
              mgcv_residual_backend_native_s_size_limit =
                mgcv_residual_backend_native_s_size_limit,
              mgcv_residual_backend_condition_threshold =
                mgcv_residual_backend_condition_threshold
            )
          )
        )
      },
      candidate_timeout_sec = candidate_timeout_sec
    ),
    fastkpc_compatible_cuda_timeout = function(e) {
      timeout_error <<- e
      NULL
    },
    error = function(e) {
      fastkpc_compatible_cuda_append_progress(
        paths$progress_csv,
        fastkpc_compatible_cuda_progress_row(
          artifact_name = artifact_name,
          route = "facade",
          event = "error",
          status = "error",
          message = conditionMessage(e)
        )
      )
      stop(e)
    }
  )
  if (is.null(facade_timed)) {
    fastkpc_compatible_cuda_append_progress(
      paths$progress_csv,
      fastkpc_compatible_cuda_progress_row(
        artifact_name = artifact_name,
        route = "facade",
        event = "timeout",
        status = "timeout",
        elapsed_sec = timeout_error$elapsed_sec,
        message = conditionMessage(timeout_error)
      )
    )
    row <- fastkpc_compatible_cuda_timeout_summary_row(
      artifact_name = artifact_name,
      data = data,
      columns = columns,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      index = index,
      numCol = numCol,
      dcov_batch = dcov_batch,
      low_rank = low_rank,
      mgcv_residual_backend = mgcv_residual_backend,
      mgcv_residual_backend_native_s_size_limit =
        mgcv_residual_backend_native_s_size_limit,
      mgcv_residual_backend_condition_threshold =
        mgcv_residual_backend_condition_threshold,
      reference = reference,
      reference_source = reference_source,
      reference_result_path = reference_result_path,
      reference_slot = reference_slot,
      expected_edge_count = expected_edge_count,
      expected_n_edgetests = expected_n_edgetests,
      timeout_sec = timeout_error$timeout_sec,
      elapsed_sec = timeout_error$elapsed_sec,
      reference_elapsed_sec = reference_timed$elapsed
    )
    utils::write.csv(row, paths$summary_csv, row.names = FALSE)
    utils::write.csv(
      fastkpc_compatible_cuda_native_dcov_stage_rows(row),
      paths$native_dcov_stage_csv,
      row.names = FALSE
    )
    saveRDS(
      list(
        summary = row,
        facade = NULL,
        reference = reference,
        paths = paths
      ),
      paths$result_rds
    )
    writeLines(c(
      paste0("# ", artifact_name),
      "",
      paste0("- route: ", row$route[[1L]]),
      paste0("- run status: ", row$run_status[[1L]]),
      paste0("- timeout sec: ", row$timeout_sec[[1L]]),
      paste0("- reference source: ", row$reference_source[[1L]]),
      paste0("- reference result path: ",
             row$reference_result_path[[1L]]),
      paste0("- elapsed sec: ", signif(row$elapsed_sec[[1L]], 8L)),
      paste0("- reference elapsed sec: ",
             signif(row$reference_elapsed_sec[[1L]], 8L)),
      paste0("- native progress: ", paths$native_progress_csv),
      paste0("- native dCov stage timing: ",
             paths$native_dcov_stage_csv),
      paste0("- native lowrank component cache progress: ",
             paths$native_lowrank_cache_progress_csv)
    ), paths$summary_md)
    return(list(
      summary = row,
      facade = NULL,
      reference = reference,
      paths = paths,
      output_dir = output_dir
    ))
  }
  fastkpc_compatible_cuda_append_progress(
    paths$progress_csv,
    fastkpc_compatible_cuda_progress_row(
      artifact_name = artifact_name,
      route = "facade",
      event = "complete",
      status = "ok",
      elapsed_sec = facade_timed$elapsed
    )
  )

  facade <- facade_timed$value
  facade_summary <- facade$summary %||% list()
  reference_summary <- reference$summary %||%
    reference$scheduler_diagnostics$summary %||%
    list()

  edge_count <- as.integer(sum(unname(facade$adjacency)) / 2L)
  reference_edge_count <- as.integer(sum(unname(reference$adjacency)) / 2L)
  shd <- as.integer(
    fastkpc_compatible_cuda_skeleton_shd(
      facade$adjacency,
      reference$adjacency
    )
  )
  n_edgetests_identical <- identical(as.integer(facade$n.edgetests),
                                    as.integer(reference$n.edgetests))
  if (is.null(expected_n_edgetests)) {
    n_edgetests_exact <- n_edgetests_identical
  } else {
    n_edgetests_exact <- identical(as.integer(facade$n.edgetests),
                                   as.integer(expected_n_edgetests))
  }
  expected_edge_count_match <- if (is.null(expected_edge_count)) {
    NA
  } else {
    identical(edge_count, as.integer(expected_edge_count))
  }

  row <- data.frame(
    artifact = artifact_name,
    route = "facade",
    n = nrow(data),
    p = ncol(data),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    columns = if (is.null(columns)) "all" else paste(columns, collapse = "|"),
    index = as.integer(index),
    numCol = as.integer(numCol),
    dcov_batch = dcov_batch,
    low_rank = low_rank,
    mgcv_residual_backend = mgcv_residual_backend,
    mgcv_residual_backend_native_s_size_limit =
      mgcv_residual_backend_native_s_size_limit %||% NA_real_,
    mgcv_residual_backend_condition_threshold =
      mgcv_residual_backend_condition_threshold %||% NA_real_,
    reference_source = reference_source,
    reference_result_path = reference_result_path %||% NA_character_,
    reference_result_slot = reference_slot,
    run_status = "ok",
    timeout = FALSE,
    timeout_sec =
      if (fastkpc_compatible_cuda_timeout_enabled(candidate_timeout_sec)) {
        as.numeric(candidate_timeout_sec)
      } else {
        NA_real_
      },
    edge_count = edge_count,
    reference_edge_count = reference_edge_count,
    expected_edge_count = expected_edge_count %||% NA_integer_,
    expected_edge_count_match = expected_edge_count_match,
    shd = shd,
    adjacency_identical =
      identical(unname(facade$adjacency), unname(reference$adjacency)),
    sepsets_identical =
      fastkpc_compatible_cuda_compare_sepsets(facade$sepsets,
                                              reference$sepsets),
    n_edgetests_identical = n_edgetests_identical,
    n_edgetests_exact = n_edgetests_exact,
    facade_n_edgetests = paste(facade$n.edgetests, collapse = "|"),
    reference_n_edgetests = paste(reference$n.edgetests, collapse = "|"),
    expected_n_edgetests = if (is.null(expected_n_edgetests)) {
      NA_character_
    } else {
      paste(expected_n_edgetests, collapse = "|")
    },
    pmax_max_abs_diff =
      fastkpc_compatible_cuda_pmax_max_abs_diff(facade$pMax,
                                                reference$pMax),
    residual_provider_request_count =
      as.integer(facade_summary$residual_provider_request_count %||% NA_integer_),
    reference_residual_provider_request_count =
      as.integer(provider_counts$request_count),
    residual_provider_call_ms =
      as.numeric(facade_summary$residual_provider_call_ms %||% NA_real_),
    residual_provider_matrix_copy_ms =
      as.numeric(facade_summary$residual_provider_matrix_copy_ms %||% NA_real_),
    residual_provider_total_ms =
      as.numeric(facade_summary$residual_provider_total_ms %||% NA_real_),
    residual_provider_parallel_enabled =
      isTRUE(facade_summary$residual_provider_parallel_enabled),
    residual_provider_parallel_cores =
      as.integer(facade_summary$residual_provider_parallel_cores %||%
                   NA_integer_),
    residual_provider_parallel_level_count =
      as.integer(facade_summary$residual_provider_parallel_level_count %||%
                   NA_integer_),
    residual_provider_parallel_request_count =
      as.integer(facade_summary$residual_provider_parallel_request_count %||%
                   NA_integer_),
    legacy_dcov_native_count =
      as.integer(facade_summary$legacy_dcov_native_count %||% NA_integer_),
    reference_legacy_dcov_native_count =
      as.integer(reference_summary$legacy_dcov_native_count %||% NA_integer_),
    legacy_dcov_native_batch_enabled =
      isTRUE(facade_summary$legacy_dcov_native_batch_enabled),
    legacy_dcov_native_batch_mode =
      facade_summary$legacy_dcov_native_batch_mode %||% NA_character_,
    legacy_dcov_native_batch_count =
      as.integer(facade_summary$legacy_dcov_native_batch_count %||% NA_integer_),
    legacy_dcov_native_batch_pair_count =
      as.integer(facade_summary$legacy_dcov_native_batch_pair_count %||%
                   NA_integer_),
    legacy_dcov_native_batch_parallel_enabled =
      isTRUE(facade_summary$legacy_dcov_native_batch_parallel_enabled),
    legacy_dcov_native_batch_parallel_threads =
      as.integer(facade_summary$legacy_dcov_native_batch_parallel_threads %||%
                   NA_integer_),
    legacy_dcov_native_batch_direct_input_enabled =
      isTRUE(facade_summary$legacy_dcov_native_batch_direct_input_enabled),
    legacy_dcov_native_batch_column_materialize_count =
      as.integer(facade_summary$legacy_dcov_native_batch_column_materialize_count %||%
                   NA_integer_),
    legacy_dcov_native_batch_materialize_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_materialize_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_call_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_call_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_input_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_input_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_distance_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_distance_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_lowrank_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_lowrank_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_lowrank_eig_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_lowrank_eig_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_lowrank_select_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_lowrank_select_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_lowrank_center_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_lowrank_center_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_lowrank_unaccounted_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_lowrank_unaccounted_ms %||%
                   NA_real_),
    legacy_dcov_native_lowrank_mode =
      facade_summary$legacy_dcov_native_lowrank_mode %||% NA_character_,
    legacy_dcov_native_lowrank_full_eig_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_full_eig_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_converged_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_converged_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_failed_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_failed_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_fallback_full_eig_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_fallback_full_eig_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_iterations =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_iterations %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_nconv =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_nconv %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_ncv =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_ncv %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_tol =
      as.numeric(facade_summary$legacy_dcov_native_lowrank_spectra_tol %||%
                   NA_real_),
    legacy_dcov_native_lowrank_spectra_matvec_count =
      as.integer(facade_summary$legacy_dcov_native_lowrank_spectra_matvec_count %||%
                   NA_integer_),
    legacy_dcov_native_lowrank_spectra_matvec_ms =
      as.numeric(facade_summary$legacy_dcov_native_lowrank_spectra_matvec_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_enabled =
      isTRUE(facade_summary$legacy_dcov_native_cuda_lowrank_backend_enabled),
    legacy_dcov_native_cuda_lowrank_backend_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_error_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_error_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_fallback_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_fallback_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_converged_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_converged_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_enabled =
      isTRUE(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_enabled),
    legacy_dcov_native_cuda_lowrank_component_cache_scope =
      as.character(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_scope %||%
                     NA_character_),
    legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_level_max_entries %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_lookup_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_lookup_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_hit_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_hit_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_miss_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_miss_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_entry_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_entry_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_cross_batch_hit_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_eviction_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_eviction_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_cache_level_entry_count_max %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_batch_substrate_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_component_batch_substrate_pair_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_component_distance_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_component_distance_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_component_lowrank_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_component_moment_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_component_moment_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_component_unaccounted_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_spectra_matvec_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_kernel_launch_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_device_matrix_reuse_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_device_matrix_reuse_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_device_workspace_reuse_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_device_workspace_reuse_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_workspace_realloc_count =
      as.integer(facade_summary$legacy_dcov_native_cuda_lowrank_backend_workspace_realloc_count %||%
                   NA_integer_),
    legacy_dcov_native_cuda_lowrank_backend_matrix_bytes =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_matrix_bytes %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_workspace_bytes =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_workspace_bytes %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_matrix_h2d_ms_during_compute_max %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_workspace_alloc_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_workspace_alloc_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_h2d_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_h2d_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_kernel_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_kernel_ms %||%
                   NA_real_),
    legacy_dcov_native_cuda_lowrank_backend_d2h_ms =
      as.numeric(facade_summary$legacy_dcov_native_cuda_lowrank_backend_d2h_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_statistic_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_statistic_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_moment_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_moment_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_pgamma_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_pgamma_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_accounted_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_accounted_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_scalar_total_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_scalar_total_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_wrapper_overhead_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_wrapper_overhead_ms %||%
                   NA_real_),
    legacy_dcov_native_batch_overhead_ms =
      as.numeric(facade_summary$legacy_dcov_native_batch_overhead_ms %||%
                   NA_real_),
    compatible_cuda_facade =
      isTRUE(facade_summary$compatible_cuda_facade),
    compatible_cuda_route =
      facade_summary$compatible_cuda_route %||% NA_character_,
    compatible_cuda_residual_authority =
      facade_summary$compatible_cuda_residual_authority %||% NA_character_,
    compatible_cuda_ci_authority =
      facade_summary$compatible_cuda_ci_authority %||% NA_character_,
    residual_provider_response_backend =
      facade_summary$residual_provider_response_backend %||% NA_character_,
    residual_provider_mgcv_backend =
      facade_summary$residual_provider_mgcv_backend %||% NA_character_,
    residual_provider_mgcv_cpp_backend_enabled =
      isTRUE(facade_summary$residual_provider_mgcv_cpp_backend_enabled),
    residual_provider_mgcv_cpp_backend_count =
      as.integer(facade_summary$residual_provider_mgcv_cpp_backend_count %||%
                   NA_integer_),
    residual_provider_mgcv_cpp_backend_native_count =
      as.integer(facade_summary$residual_provider_mgcv_cpp_backend_native_count %||%
                   NA_integer_),
    residual_provider_mgcv_cpp_backend_fallback_count =
      as.integer(facade_summary$residual_provider_mgcv_cpp_backend_fallback_count %||%
                   NA_integer_),
    residual_provider_mgcv_cpp_backend_error_count =
      as.integer(facade_summary$residual_provider_mgcv_cpp_backend_error_count %||%
                   NA_integer_),
    residual_provider_mgcv_cpp_backend_ms =
      as.numeric(facade_summary$residual_provider_mgcv_cpp_backend_ms %||%
                   NA_real_),
    residual_provider_mgcv_cpp_backend_native_solve_ms =
      as.numeric(facade_summary$residual_provider_mgcv_cpp_backend_native_solve_ms %||%
                   NA_real_),
    elapsed_sec = as.numeric(facade_timed$elapsed),
    reference_elapsed_sec = as.numeric(reference_timed$elapsed),
    stringsAsFactors = FALSE
  )

  utils::write.csv(row, paths$summary_csv, row.names = FALSE)
  utils::write.csv(
    fastkpc_compatible_cuda_native_dcov_stage_rows(row),
    paths$native_dcov_stage_csv,
    row.names = FALSE
  )
  saveRDS(
    list(
      summary = row,
      facade = facade,
      reference = reference,
      paths = paths
    ),
    paths$result_rds
  )
  writeLines(c(
    paste0("# ", artifact_name),
    "",
    paste0("- route: ", row$route[[1L]]),
    paste0("- reference source: ", row$reference_source[[1L]]),
    paste0("- reference result path: ",
           row$reference_result_path[[1L]]),
    paste0("- n / p: ", row$n[[1L]], " / ", row$p[[1L]]),
    paste0("- alpha: ", row$alpha[[1L]]),
    paste0("- max conditioning size: ",
           row$max_conditioning_size[[1L]]),
    paste0("- edge count: ", row$edge_count[[1L]], " / ",
           row$reference_edge_count[[1L]]),
    paste0("- SHD: ", row$shd[[1L]]),
    paste0("- n.edgetests exact: ", row$n_edgetests_exact[[1L]]),
    paste0("- facade n.edgetests: ",
           row$facade_n_edgetests[[1L]]),
    paste0("- reference n.edgetests: ",
           row$reference_n_edgetests[[1L]]),
    paste0("- pMax max abs diff: ",
           signif(row$pmax_max_abs_diff[[1L]], 8L)),
    paste0("- residual provider requests: ",
           row$residual_provider_request_count[[1L]]),
    paste0("- native legacy dCov calls: ",
           row$legacy_dcov_native_count[[1L]]),
    paste0("- native dCov batch enabled: ",
           row$legacy_dcov_native_batch_enabled[[1L]]),
    paste0("- compatible CUDA route: ",
           row$compatible_cuda_route[[1L]]),
    paste0("- mgcv residual backend: ",
           row$mgcv_residual_backend[[1L]]),
    paste0("- residual provider backend: ",
           row$residual_provider_response_backend[[1L]]),
    paste0("- provider C++ residual backend count: ",
           row$residual_provider_mgcv_cpp_backend_count[[1L]]),
    paste0("- elapsed sec: ", signif(row$elapsed_sec[[1L]], 8L)),
    paste0("- reference elapsed sec: ",
           signif(row$reference_elapsed_sec[[1L]], 8L)),
    paste0("- native progress: ", paths$native_progress_csv),
    paste0("- native dCov stage timing: ", paths$native_dcov_stage_csv),
    paste0("- native lowrank component cache progress: ",
           paths$native_lowrank_cache_progress_csv)
  ), paths$summary_md)

  list(
    summary = row,
    facade = facade,
    reference = reference,
    paths = paths,
    output_dir = output_dir
  )
}
