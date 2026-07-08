source("fastkpc/R/legacy_runner.R")

fastkpc_legacy_mgcv_shadow_restore_env <- function(old_env) {
  for (name in names(old_env)) {
    if (is.na(old_env[[name]])) {
      Sys.unsetenv(name)
    } else {
      do.call(Sys.setenv, stats::setNames(list(old_env[[name]]), name))
    }
  }
  invisible(TRUE)
}

fastkpc_legacy_mgcv_shadow_set_common_env <- function(
    num_cores, dcov_backend, low_rank, residual_cache, affinity) {
  Sys.setenv(FASTKPC_LEGACY_PARALLEL_CORES = as.character(num_cores))
  if (identical(dcov_backend, "cpp")) {
    Sys.setenv(
      FASTKPC_LEGACY_DCOV_GAMMA_BACKEND = "cpp",
      FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK = low_rank
    )
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND")
    Sys.unsetenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK")
  }
  if (isTRUE(residual_cache)) {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE = "1")
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE")
  }
  if (nzchar(affinity)) {
    Sys.setenv(FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY = affinity)
  } else {
    Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY")
  }
  invisible(TRUE)
}

fastkpc_run_legacy_mgcv_residual_cpp_shadow_artifact <- function(
    data = NULL,
    data_path = file.path("fastkpc", "artifacts", "kpc_tprs_real_zhu",
                          "cancer_RD-causalDiscoveryInput.rds"),
    columns = NULL,
    output_dir = file.path("fastkpc", "artifacts",
                           "legacy_mgcv_residual_cpp_shadow_real_subset_v1"),
    artifact_name = basename(output_dir),
    alpha = 0.1,
    max_conditioning_size = 2L,
    num_cores = 1L,
    native_s_size_limit = 2L,
    condition_threshold = 1e12,
    dcov_backend = c("cpp", "r"),
    low_rank = "spectra",
    residual_cache = FALSE,
    affinity = "") {
  dcov_backend <- match.arg(dcov_backend)
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
    stop("shadow artifact data must contain at least four columns",
         call. = FALSE)
  }
  max_conditioning_size <- min(
    as.integer(max_conditioning_size),
    ncol(data) - 2L
  )
  num_cores <- max(1L, as.integer(num_cores))

  env_names <- c(
    "FASTKPC_LEGACY_PARALLEL_CORES",
    "FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
    "FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT",
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD"
  )
  old_env <- Sys.getenv(env_names, unset = NA_character_)
  on.exit(fastkpc_legacy_mgcv_shadow_restore_env(old_env), add = TRUE)

  run_once <- function(enable_shadow) {
    fastkpc_legacy_mgcv_shadow_set_common_env(
      num_cores = num_cores,
      dcov_backend = dcov_backend,
      low_rank = low_rank,
      residual_cache = residual_cache,
      affinity = affinity
    )
    if (isTRUE(enable_shadow)) {
      Sys.setenv(
        FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW = "1",
        FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT =
          as.character(native_s_size_limit),
        FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD =
          as.character(condition_threshold)
      )
    } else {
      Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW")
      Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT")
      Sys.unsetenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD")
    }
    fastkpc_legacy_parallel_skeleton(
      data = data,
      alpha = alpha,
      max_conditioning_size = max_conditioning_size,
      ic.method = "dcc.gamma",
      num_cores = num_cores
    )
  }

  baseline <- run_once(FALSE)
  shadow <- run_once(TRUE)
  summary <- shadow$scheduler_diagnostics$summary
  row <- data.frame(
    artifact = artifact_name,
    n = nrow(data),
    p = ncol(data),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    num_cores = as.integer(num_cores),
    dcov_backend = dcov_backend,
    residual_cache = isTRUE(residual_cache),
    affinity = affinity,
    adjacency_identical = identical(shadow$adjacency, baseline$adjacency),
    n_edgetests_identical =
      identical(shadow$n.edgetests, baseline$n.edgetests),
    baseline_n_edgetests = paste(baseline$n.edgetests, collapse = "|"),
    shadow_n_edgetests = paste(shadow$n.edgetests, collapse = "|"),
    baseline_edge_count = sum(baseline$adjacency) / 2,
    shadow_edge_count = sum(shadow$adjacency) / 2,
    residual_request_count =
      as.integer(summary$legacy_mgcv_residual_request_count),
    shadow_count = as.integer(summary$legacy_mgcv_cpp_shadow_count),
    native_count = as.integer(summary$legacy_mgcv_cpp_shadow_native_count),
    fallback_count =
      as.integer(summary$legacy_mgcv_cpp_shadow_fallback_count),
    high_condition_fallback_count =
      as.integer(summary$legacy_mgcv_cpp_shadow_high_condition_fallback_count),
    outside_envelope_fallback_count =
      as.integer(summary$legacy_mgcv_cpp_shadow_outside_envelope_fallback_count),
    error_count = as.integer(summary$legacy_mgcv_cpp_shadow_error_count),
    residual_mismatch_count =
      as.integer(summary$legacy_mgcv_cpp_shadow_residual_mismatch_count),
    max_abs_diff = as.numeric(summary$legacy_mgcv_cpp_shadow_max_abs_diff),
    max_rel_l2 = as.numeric(summary$legacy_mgcv_cpp_shadow_max_rel_l2),
    native_s_size_limit =
      as.numeric(summary$legacy_mgcv_cpp_shadow_native_s_size_limit),
    condition_threshold =
      as.numeric(summary$legacy_mgcv_cpp_shadow_condition_threshold),
    elapsed_ms = as.numeric(summary$legacy_parallel_elapsed_ms),
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
    list(baseline = baseline, shadow = shadow, summary = row),
    paths$result_rds
  )
  writeLines(c(
    paste0("# ", artifact_name),
    "",
    paste0("- n / p: ", row$n[[1L]], " / ", row$p[[1L]]),
    paste0("- alpha: ", row$alpha[[1L]]),
    paste0("- max conditioning size: ",
           row$max_conditioning_size[[1L]]),
    paste0("- adjacency identical: ", row$adjacency_identical[[1L]]),
    paste0("- n.edgetests identical: ",
           row$n_edgetests_identical[[1L]]),
    paste0("- baseline n.edgetests: ",
           row$baseline_n_edgetests[[1L]]),
    paste0("- shadow n.edgetests: ", row$shadow_n_edgetests[[1L]]),
    paste0("- residual requests: ", row$residual_request_count[[1L]]),
    paste0("- shadow targets: ", row$shadow_count[[1L]]),
    paste0("- native targets: ", row$native_count[[1L]]),
    paste0("- fallback targets: ", row$fallback_count[[1L]]),
    paste0("- errors: ", row$error_count[[1L]]),
    paste0("- residual mismatches: ",
           row$residual_mismatch_count[[1L]]),
    paste0("- max abs diff: ", signif(row$max_abs_diff[[1L]], 8L)),
    paste0("- max rel l2: ", signif(row$max_rel_l2[[1L]], 8L))
  ), paths$summary_md)

  list(
    summary = row,
    baseline = baseline,
    shadow = shadow,
    paths = paths,
    output_dir = output_dir
  )
}
