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
    expected_n_edgetests = NULL) {
  mgcv_residual_backend <- match.arg(mgcv_residual_backend)
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

  env_names <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD"
  )
  old_env <- Sys.getenv(env_names, unset = NA_character_)
  on.exit(fastkpc_compatible_cuda_restore_env(old_env), add = TRUE)
  if (nzchar(low_rank)) {
    Sys.setenv(FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = low_rank)
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  }
  if (!identical(mgcv_residual_backend, "env")) {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND =
                 mgcv_residual_backend)
  }
  if (!is.null(mgcv_residual_backend_native_s_size_limit)) {
    Sys.setenv(
      FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT =
        as.character(mgcv_residual_backend_native_s_size_limit)
    )
  }
  if (!is.null(mgcv_residual_backend_condition_threshold)) {
    Sys.setenv(
      FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD =
        as.character(mgcv_residual_backend_condition_threshold)
    )
  }

  provider_counts <- new.env(parent = emptyenv())
  provider_counts$level_calls <- 0L
  provider_counts$request_count <- 0L

  reference_source <- "computed"
  reference_slot <- "computed"
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
    reference_source <- "rds"
    reference_timed <- list(value = reference, elapsed = 0)
  }
  facade_timed <- fastkpc_elapsed(
    fastkpc_compatible_cuda_skeleton(
      data = data,
      alpha = alpha,
      labels = labels,
      options = list(
        max_conditioning_size = max_conditioning_size,
        index = index,
        numCol = numCol,
        trace_level = trace_level,
        dcov_batch = dcov_batch
      )
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
    legacy_dcov_native_count =
      as.integer(facade_summary$legacy_dcov_native_count %||% NA_integer_),
    reference_legacy_dcov_native_count =
      as.integer(reference_summary$legacy_dcov_native_count %||% NA_integer_),
    legacy_dcov_native_batch_enabled =
      isTRUE(facade_summary$legacy_dcov_native_batch_enabled),
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

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    result_rds = file.path(output_dir, "result.rds"),
    summary_md = file.path(output_dir, "summary.md")
  )
  utils::write.csv(row, paths$summary_csv, row.names = FALSE)
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
           signif(row$reference_elapsed_sec[[1L]], 8L))
  ), paths$summary_md)

  list(
    summary = row,
    facade = facade,
    reference = reference,
    paths = paths,
    output_dir = output_dir
  )
}
