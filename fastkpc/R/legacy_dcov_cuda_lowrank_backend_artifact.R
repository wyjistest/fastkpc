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

fastkpc_run_legacy_dcov_cuda_lowrank_backend_real_subset_artifact <- function(
    output_dir = "fastkpc/artifacts/legacy_dcov_cuda_lowrank_backend_real_subset_v1",
    artifact_name = "legacy_dcov_cuda_lowrank_backend_real_subset_v1",
    data_path = "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
    columns = c(1L, 2L, 3L, 4L, 5L),
    alpha = 0.1,
    max_conditioning_size = 3L,
    rebuild_cuda = FALSE) {
  if (!file.exists(data_path)) {
    stop("real data fixture not found: ", data_path, call. = FALSE)
  }
  build_fastkpc_cuda_native(rebuild = isTRUE(rebuild_cuda))
  if (!fastkpc_cuda_available()) {
    stop("CUDA unavailable for legacy dCov CUDA lowrank backend artifact",
         call. = FALSE)
  }

  real_data <- readRDS(data_path)
  data <- as.matrix(real_data[, columns, drop = FALSE])
  storage.mode(data) <- "double"
  n <- nrow(data)
  p <- ncol(data)

  tracked_env <- c(
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH",
    "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW",
    "FASTKPC_LEGACY_DCOV_GAMMA_CUDA_LOW_RANK_SHADOW_MAX_CALLS",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
    "FASTKPC_LEGACY_PARALLEL_CORES"
  )
  old_env <- stats::setNames(
    lapply(tracked_env, Sys.getenv, unset = NA_character_),
    tracked_env
  )
  on.exit(fastkpc_legacy_cuda_lowrank_restore_env(old_env), add = TRUE)

  run_route <- function(lowrank_mode) {
    Sys.setenv(
      FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
      FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = lowrank_mode,
      FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1",
      FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = "s",
      FASTKPC_LEGACY_PARALLEL_CORES = "1"
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

  baseline <- run_route("spectra")
  candidate <- run_route("cuda_spectra")

  baseline_skeleton <- baseline$result$skeleton
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
    columns = paste(as.integer(columns), collapse = ","),
    n = as.integer(n),
    p = as.integer(p),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
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

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    summary_csv = file.path(output_dir, "summary.csv"),
    summary_md = file.path(output_dir, "summary.md"),
    result_rds = file.path(output_dir, "result.rds")
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
           signif(row$baseline_elapsed_sec[[1L]], 8L))
  ), paths$summary_md)

  list(summary = row, paths = paths, baseline = baseline,
       candidate = candidate, output_dir = output_dir)
}
