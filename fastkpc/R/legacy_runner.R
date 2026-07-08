fastkpc_legacy_env <- function(path = "kpcalg/R") {
  if (!dir.exists(path)) {
    stop("Cannot find legacy kpcalg R directory: ", path, call. = FALSE)
  }

  env <- new.env(parent = globalenv())

  if (requireNamespace("energy", quietly = TRUE)) env$dcov.test <- energy::dcov.test
  if (requireNamespace("RSpectra", quietly = TRUE)) env$eigs <- RSpectra::eigs
  if (requireNamespace("kernlab", quietly = TRUE)) {
    env$inchol <- kernlab::inchol
    env$rbfdot <- kernlab::rbfdot
  }
  if (requireNamespace("mgcv", quietly = TRUE)) env$gam <- mgcv::gam
  if (requireNamespace("graph", quietly = TRUE)) env$numEdges <- graph::numEdges
  if (requireNamespace("pcalg", quietly = TRUE)) env$triple2numb <- pcalg::triple2numb
  env$as <- methods::as
  env$combn <- utils::combn
  env$makeCluster <- parallel::makeCluster
  env$clusterEvalQ <- parallel::clusterEvalQ
  env$parLapply <- parallel::parLapply
  env$stopCluster <- parallel::stopCluster

  files <- c(
    "dcovgamma.R",
    "frmladditivesmooth.R",
    "frmlfullsmooth.R",
    "hsicgamma.R",
    "hsicperm.R",
    "hsicclust.R",
    "hsictest.R",
    "regrXonS.R",
    "kernelCItest.R",
    "regrvonps.R",
    "udag2wanpdag.R",
    "kpc.R"
  )

  for (file in file.path(path, files)) {
    if (file.exists(file)) sys.source(file, envir = env)
  }

  env
}

fastkpc_require_legacy_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop("Missing legacy kpcalg dependency: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_legacy_packages_for_method <- function(ic.method, conditional = FALSE) {
  packages <- character()
  if (ic.method == "dcc.gamma") packages <- c(packages, "RSpectra")
  if (ic.method == "dcc.perm") packages <- c(packages, "energy")
  if (ic.method %in% c("hsic.gamma", "hsic.perm", "hsic.clust")) {
    packages <- c(packages, "kernlab")
  }
  if (ic.method == "hsic.clust") packages <- c(packages, "parallel")
  if (conditional && ic.method != "hsic.clust") packages <- c(packages, "mgcv")
  unique(packages)
}

fastkpc_legacy_dcov_gamma <- function(x, y, index = 1, numCol = NULL,
                                      env = fastkpc_legacy_env()) {
  fastkpc_require_legacy_packages("RSpectra")
  if (is.null(numCol)) numCol <- floor(length(x) / 10)
  env$dcov.gamma(x = x, y = y, index = index, numCol = numCol)
}

fastkpc_legacy_dcov_gamma_timed <- function(x, y, index = 1, numCol = NULL,
                                            env = fastkpc_legacy_env()) {
  fastkpc_require_legacy_packages("RSpectra")
  total_start <- proc.time()[["elapsed"]]
  input_start <- total_start
  n <- length(x)
  m <- length(y)
  if (is.null(numCol)) numCol <- floor(n / 10)
  if (index < 0 || index > 2) {
    warning("index must be in [0,2), using default index=1")
    index <- 1
  }
  if (n != m) stop("Sample sizes must agree", call. = FALSE)
  if (!(all(is.finite(c(x, y))))) {
    stop("Data contains missing or infinite values", call. = FALSE)
  }
  input_ms <- (proc.time()[["elapsed"]] - input_start) * 1000

  h_start <- proc.time()[["elapsed"]]
  H <- diag(n) - matrix(1 / n, n, n)
  h_ms <- (proc.time()[["elapsed"]] - h_start) * 1000

  distance_start <- proc.time()[["elapsed"]]
  matx <- as.matrix(stats::dist(x))
  maty <- as.matrix(stats::dist(y))
  distance_ms <- (proc.time()[["elapsed"]] - distance_start) * 1000

  lowrank_start <- proc.time()[["elapsed"]]
  P <- env$eigs(matx, numCol)
  Q <- env$eigs(maty, numCol)
  Ux <- P$vectors
  Sx <- diag(P$values)
  Uy <- Q$vectors
  Sy <- diag(Q$values)
  lowrank_ms <- (proc.time()[["elapsed"]] - lowrank_start) * 1000

  statistic_start <- proc.time()[["elapsed"]]
  nV2 <- sum(diag(
    (H %*% Ux) %*% Sx %*%
      ((t(Ux) %*% H) %*% (H %*% Uy)) %*%
      Sy %*% (t(Uy) %*% H)
  )) / n
  statistic_ms <- (proc.time()[["elapsed"]] - statistic_start) * 1000

  moment_start <- proc.time()[["elapsed"]]
  nV2Mean <- mean(matx) * mean(maty)
  nV2Variance <- 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) /
    (n - 3) *
    sum(diag(
      (H %*% Ux) %*% Sx %*%
        ((t(Ux) %*% H) %*% (H %*% Ux)) %*%
        Sx %*% (t(Ux) %*% H)
    )) *
    sum(diag(
      (H %*% Uy) %*% Sy %*%
        ((t(Uy) %*% H) %*% (H %*% Uy)) %*%
        Sy %*% (t(Uy) %*% H)
    )) / n^4 * n^2
  alpha <- (nV2Mean)^2 / nV2Variance
  beta <- nV2Variance / nV2Mean
  moment_ms <- (proc.time()[["elapsed"]] - moment_start) * 1000

  pgamma_start <- proc.time()[["elapsed"]]
  pval <- 1 - stats::pgamma(q = nV2, shape = alpha, rate = 1 / beta)
  dCov <- sqrt(nV2 / n)
  pgamma_ms <- (proc.time()[["elapsed"]] - pgamma_start) * 1000

  output_start <- proc.time()[["elapsed"]]
  names(dCov) <- "dCov"
  names(nV2) <- "nV^2"
  names(nV2Mean) <- "nV^2 mean"
  names(nV2Variance) <- "nV^2 variance"
  dataname <- paste("index 1, Gamma approximation", sep = "")
  result <- list(
    method = paste("dCov test of independence", sep = ""),
    statistic = nV2,
    estimate = dCov,
    estimates = c(nV2, nV2Mean, nV2Variance),
    p.value = pval,
    replicates = NULL,
    data.name = dataname
  )
  class(result) <- "htest"
  output_ms <- (proc.time()[["elapsed"]] - output_start) * 1000
  total_ms <- (proc.time()[["elapsed"]] - total_start) * 1000
  accounted_ms <- input_ms + h_ms + distance_ms + lowrank_ms +
    statistic_ms + moment_ms + pgamma_ms + output_ms
  list(
    result = result,
    diagnostics = list(
      n = as.integer(n),
      numCol = as.integer(numCol),
      index = as.numeric(index),
      input_ms = input_ms,
      h_ms = h_ms,
      distance_ms = distance_ms,
      lowrank_ms = lowrank_ms,
      statistic_ms = statistic_ms,
      moment_ms = moment_ms,
      pgamma_ms = pgamma_ms,
      output_ms = output_ms,
      accounted_ms = accounted_ms,
      unaccounted_ms = max(0, total_ms - accounted_ms),
      total_ms = total_ms
    )
  )
}

fastkpc_legacy_dcov_gamma_oracle_case <- function(data, x, y, S = integer(),
                                                  alpha, index = 1,
                                                  numCol = floor(nrow(data) / 10),
                                                  env = fastkpc_legacy_env(),
                                                  case_id = "case") {
  data <- as.matrix(data)
  S <- as.integer(S)
  residual_start <- proc.time()[["elapsed"]]
  if (length(S) == 0L) {
    rx <- as.numeric(data[, x])
    ry <- as.numeric(data[, y])
  } else {
    residuals <- env$regrXonS(data[, c(x, y)], data[, S])
    rx <- as.numeric(residuals[, 1L])
    ry <- as.numeric(residuals[, 2L])
  }
  residual_ms <- (proc.time()[["elapsed"]] - residual_start) * 1000
  timed <- fastkpc_legacy_dcov_gamma_timed(
    rx, ry, index = index, numCol = numCol, env = env
  )
  diag <- timed$diagnostics
  S_key <- if (length(S) == 0L) "" else paste(S, collapse = "|")
  meta <- data.frame(
    case_id = as.character(case_id),
    x = as.integer(x),
    y = as.integer(y),
    S_key = S_key,
    S_size = length(S),
    n = nrow(data),
    numCol = as.integer(numCol),
    index = as.numeric(index),
    alpha = as.numeric(alpha),
    p.value = as.numeric(timed$result$p.value),
    delete_edge = as.numeric(timed$result$p.value) >= as.numeric(alpha),
    nV2 = as.numeric(timed$result$estimates[[1L]]),
    nV2Mean = as.numeric(timed$result$estimates[[2L]]),
    nV2Variance = as.numeric(timed$result$estimates[[3L]]),
    residual_ms = residual_ms,
    dcov_total_ms = as.numeric(diag$total_ms),
    dcov_distance_ms = as.numeric(diag$distance_ms),
    dcov_lowrank_ms = as.numeric(diag$lowrank_ms),
    dcov_statistic_ms = as.numeric(diag$statistic_ms),
    dcov_moment_ms = as.numeric(diag$moment_ms),
    dcov_pgamma_ms = as.numeric(diag$pgamma_ms),
    stringsAsFactors = FALSE
  )
  residual_frame <- data.frame(
    case_id = as.character(case_id),
    row = seq_len(nrow(data)),
    rx = rx,
    ry = ry
  )
  list(
    meta = meta,
    residuals = residual_frame,
    result = timed$result,
    diagnostics = diag
  )
}

fastkpc_legacy_kernel_ci <- function(data, x, y, S = integer(),
                                     ic.method = "dcc.gamma",
                                     index = 1,
                                     numCol = floor(nrow(data) / 10),
                                     env = fastkpc_legacy_env(),
                                     ...) {
  fastkpc_require_legacy_packages(
    fastkpc_legacy_packages_for_method(ic.method, conditional = length(S) > 0)
  )
  suffStat <- list(
    data = as.matrix(data),
    ic.method = ic.method,
    index = index,
    numCol = numCol
  )
  env$kernelCItest(x = x, y = y, S = S, suffStat = suffStat, ...)
}

fastkpc_legacy_skeleton <- function(data, alpha, max_conditioning_size,
                                    method = "stable",
                                    ic.method = "dcc.gamma",
                                    index = 1,
                                    numCol = floor(nrow(data) / 10),
                                    env = fastkpc_legacy_env(),
                                    ...) {
  if (!requireNamespace("pcalg", quietly = TRUE)) {
    stop("pcalg is required for legacy skeleton baselines", call. = FALSE)
  }
  fastkpc_require_legacy_packages(
    fastkpc_legacy_packages_for_method(ic.method, conditional = max_conditioning_size > 0)
  )
  data <- as.matrix(data)
  labels <- colnames(data)
  if (is.null(labels)) labels <- paste0("V", seq_len(ncol(data)))
  suffStat <- list(
    data = data,
    ic.method = ic.method,
    index = index,
    numCol = numCol
  )
  pcalg::skeleton(
    suffStat = suffStat,
    indepTest = env$kernelCItest,
    alpha = alpha,
    labels = labels,
    m.max = max_conditioning_size,
    method = method,
    ...
  )
}

fastkpc_legacy_parallel_cores <- function(num_cores = NULL) {
  if (!is.null(num_cores)) {
    out <- as.integer(num_cores)
  } else {
    env_value <- Sys.getenv("FASTKPC_LEGACY_PARALLEL_CORES", "")
    out <- suppressWarnings(as.integer(env_value))
  }
  if (length(out) != 1L || is.na(out) || out < 1L) {
    out <- parallel::detectCores()
  }
  max(1L, as.integer(out))
}

fastkpc_legacy_sepsets <- function(p) {
  replicate(p, replicate(p, integer(), simplify = FALSE), simplify = FALSE)
}

fastkpc_legacy_combinations <- function(values, choose) {
  values <- as.integer(values)
  choose <- as.integer(choose)
  if (choose == 0L) return(list(integer()))
  if (length(values) < choose) return(list())
  lapply(utils::combn(seq_along(values), choose, simplify = FALSE), function(idx) {
    as.integer(values[idx])
  })
}

fastkpc_legacy_runtime_zero <- function() {
  list(
    ci_total_ms = 0,
    residual_ms = 0,
    dcov_gamma_ms = 0,
    dcov_input_ms = 0,
    dcov_h_ms = 0,
    dcov_distance_ms = 0,
    dcov_lowrank_ms = 0,
    dcov_statistic_ms = 0,
    dcov_moment_ms = 0,
    dcov_pgamma_ms = 0,
    dcov_output_ms = 0,
    dcov_unaccounted_ms = 0,
    dcov_r_backend_count = 0L,
    dcov_cpp_backend_count = 0L,
    dcov_cpp_backend_ms = 0,
    dcov_cpp_backend_error_count = 0L,
    dcov_cpp_backend_fallback_count = 0L,
    dcov_cpp_backend_max_p_diff = 0,
    dcov_cpp_backend_decision_flip_count = 0L,
    dcov_cpp_shadow_ms = 0,
    dcov_cpp_shadow_count = 0L,
    dcov_cpp_shadow_error_count = 0L,
    dcov_cpp_shadow_decision_flip_count = 0L,
    dcov_cpp_shadow_near_alpha_count = 0L,
    dcov_cpp_shadow_max_p_diff = 0,
    dcov_cpp_shadow_max_nV2_diff = 0,
    dcov_cpp_shadow_max_mean_diff = 0,
    dcov_cpp_shadow_max_variance_diff = 0,
    dcov_cpp_input_ms = 0,
    dcov_cpp_distance_ms = 0,
    dcov_cpp_lowrank_ms = 0,
    dcov_cpp_lowrank_eig_ms = 0,
    dcov_cpp_lowrank_select_ms = 0,
    dcov_cpp_lowrank_center_ms = 0,
    dcov_cpp_lowrank_unaccounted_ms = 0,
    dcov_cpp_lowrank_full_eig_count = 0L,
    dcov_cpp_lowrank_spectra_count = 0L,
    dcov_cpp_lowrank_spectra_converged_count = 0L,
    dcov_cpp_lowrank_spectra_failed_count = 0L,
    dcov_cpp_lowrank_spectra_fallback_full_eig_count = 0L,
    dcov_cpp_lowrank_spectra_iterations = 0L,
    dcov_cpp_lowrank_spectra_nconv = 0L,
    dcov_cpp_lowrank_spectra_ncv = 0L,
    dcov_cpp_lowrank_spectra_tol = 0,
    dcov_cpp_statistic_ms = 0,
    dcov_cpp_moment_ms = 0,
    dcov_cpp_pgamma_ms = 0,
    dcov_cpp_accounted_ms = 0,
    dcov_cpp_unaccounted_ms = 0,
    dcov_cpp_overhead_ms = 0,
    mgcv_residual_request_count = 0L,
    mgcv_cache_hit_count = 0L,
    mgcv_cache_miss_count = 0L,
    mgcv_residual_cache_hit_key_count = 0L,
    mgcv_residual_cache_miss_key_count = 0L,
    mgcv_residual_cache_miss_s_group_count = 0L,
    mgcv_residual_cache_miss_s_total_targets = 0L,
    mgcv_residual_cache_miss_s_max_targets = 0L,
    mgcv_residual_cache_miss_s_mean_targets = 0,
    mgcv_residual_cache_miss_s_reuse_opportunity_count = 0L,
    mgcv_residual_cache_miss_s_reuse_ratio = 0,
    mgcv_unique_residual_key_count = 0L,
    mgcv_duplicate_residual_key_count = 0L,
    mgcv_unique_target_s_count = 0L,
    mgcv_unique_s_count = 0L,
    mgcv_same_s_group_count = 0L,
    mgcv_same_s_total_targets = 0L,
    mgcv_same_s_max_targets = 0L,
    mgcv_same_s_mean_targets = 0,
    mgcv_same_s_reuse_opportunity_count = 0L,
    mgcv_residual_cache_insert_count = 0L,
    mgcv_residual_cache_entries = 0L,
    mgcv_residual_cache_hit_ms = 0,
    mgcv_residual_cache_store_ms = 0,
    mgcv_fit_avoided_count = 0L,
    mgcv_residual_affinity_enabled = 0L,
    mgcv_residual_affinity_group_count = 0L,
    mgcv_residual_affinity_task_count = 0L,
    mgcv_residual_affinity_worker_count = 0L,
    mgcv_residual_affinity_max_group_size = 0L,
    mgcv_residual_affinity_mean_group_size = 0,
    mgcv_residual_affinity_load_imbalance = 0,
    mgcv_residual_affinity_split_group_count = 0L,
    mgcv_residual_affinity_split_group_tasks = 0L,
    mgcv_residual_affinity_split_group_pieces = 0L,
    mgcv_residual_cache_theoretical_hit_count = 0L,
    mgcv_residual_cache_realized_hit_count = 0L,
    mgcv_residual_cache_lost_duplicate_count = 0L,
    mgcv_residual_cache_lost_cross_worker_count = 0L,
    mgcv_residual_cache_lost_split_s_group_count = 0L,
    mgcv_residual_cache_lost_cross_level_count = 0L,
    mgcv_residual_owner_enabled = 0L,
    mgcv_residual_owner_key_count = 0L,
    mgcv_residual_owner_task_count = 0L,
    mgcv_residual_owner_both_local_count = 0L,
    mgcv_residual_owner_one_local_count = 0L,
    mgcv_residual_owner_none_local_count = 0L,
    mgcv_residual_owner_conflict_count = 0L,
    mgcv_residual_owner_predicted_hit_count = 0L,
    mgcv_residual_owner_realized_hit_count = 0L,
    mgcv_residual_owner_lost_duplicate_count = 0L,
    mgcv_residual_owner_load_imbalance = 0,
    mgcv_residual_owner_spill_count = 0L,
    mgcv_residual_owner_schedule_build_ms = 0,
    mgcv_residual_owner_key_enum_ms = 0,
    mgcv_residual_owner_key_map_build_ms = 0,
    mgcv_residual_owner_task_score_ms = 0,
    mgcv_residual_owner_greedy_assign_ms = 0,
    mgcv_residual_owner_chunk_sort_ms = 0,
    mgcv_residual_owner_chunk_materialize_ms = 0,
    mgcv_residual_owner_worker_max_ms = 0,
    mgcv_residual_owner_worker_median_ms = 0,
    mgcv_residual_owner_worker_elapsed_imbalance = 0,
    mgcv_residual_owner_worker_task_max = 0L,
    mgcv_residual_owner_worker_task_median = 0,
    mgcv_residual_owner_worker_fit_max = 0L,
    mgcv_residual_owner_worker_fit_median = 0,
    mgcv_residual_owner_worker_cache_hit_max = 0L,
    mgcv_residual_owner_worker_cache_hit_median = 0,
    mgcv_residual_owner_worker_residual_ms_max = 0,
    mgcv_residual_owner_worker_residual_ms_median = 0,
    mgcv_prefetch_enabled = 0L,
    mgcv_prefetch_level_count = 0L,
    mgcv_prefetch_key_count = 0L,
    mgcv_prefetch_fit_count = 0L,
    mgcv_prefetch_fit_ms = 0,
    mgcv_prefetch_collect_ms = 0,
    mgcv_prefetch_matrix_build_ms = 0,
    mgcv_prefetch_payload_bytes = 0,
    mgcv_prefetch_max_level_payload_bytes = 0,
    mgcv_prefetch_lookup_ms = 0,
    mgcv_prefetch_ci_phase_ms = 0,
    mgcv_prefetch_error_count = 0L,
    mgcv_prefetch_consumed_key_count = 0L,
    mgcv_prefetch_unused_key_count = 0L,
    mgcv_key_build_ms = 0,
    mgcv_cache_lookup_ms = 0,
    mgcv_formula_build_ms = 0,
    mgcv_data_subset_ms = 0,
    mgcv_fit_call_ms = 0,
    mgcv_residual_extract_ms = 0,
    mgcv_result_store_ms = 0,
    mgcv_unaccounted_ms = 0,
    mgcv_s_size_0_count = 0L,
    mgcv_s_size_1_count = 0L,
    mgcv_s_size_2_count = 0L,
    mgcv_s_size_gt2_count = 0L,
    mgcv_r_backend_count = 0L,
    mgcv_cpp_backend_enabled = 0L,
    mgcv_cpp_backend_count = 0L,
    mgcv_cpp_backend_native_count = 0L,
    mgcv_cpp_backend_fallback_count = 0L,
    mgcv_cpp_backend_high_condition_fallback_count = 0L,
    mgcv_cpp_backend_outside_envelope_fallback_count = 0L,
    mgcv_cpp_backend_error_count = 0L,
    mgcv_cpp_backend_ms = 0,
    mgcv_cpp_backend_input_setup_ms = 0,
    mgcv_cpp_backend_gam_fit_ms = 0,
    mgcv_cpp_backend_sp_extract_ms = 0,
    mgcv_cpp_backend_setup_extract_ms = 0,
    mgcv_cpp_backend_condition_ms = 0,
    mgcv_cpp_backend_native_solve_ms = 0,
    mgcv_cpp_backend_fallback_ms = 0,
    mgcv_cpp_backend_s_size_0_count = 0L,
    mgcv_cpp_backend_s_size_1_count = 0L,
    mgcv_cpp_backend_s_size_2_count = 0L,
    mgcv_cpp_backend_s_size_gt2_count = 0L,
    mgcv_cpp_backend_native_s_size_0_count = 0L,
    mgcv_cpp_backend_native_s_size_1_count = 0L,
    mgcv_cpp_backend_native_s_size_2_count = 0L,
    mgcv_cpp_backend_native_s_size_gt2_count = 0L,
    mgcv_cpp_backend_fallback_s_size_0_count = 0L,
    mgcv_cpp_backend_fallback_s_size_1_count = 0L,
    mgcv_cpp_backend_fallback_s_size_2_count = 0L,
    mgcv_cpp_backend_fallback_s_size_gt2_count = 0L,
    mgcv_cpp_backend_same_s_native_group_count = 0L,
    mgcv_cpp_backend_same_s_native_target_count = 0L,
    mgcv_cpp_backend_same_s_native_max_targets = 0L,
    mgcv_cpp_backend_same_s_native_mean_targets = 0,
    mgcv_cpp_backend_same_s_native_reuse_opportunity_count = 0L,
    mgcv_cpp_backend_same_s_native_setup_reuse_ratio = 0,
    mgcv_cpp_backend_same_s_sp_native_group_count = 0L,
    mgcv_cpp_backend_same_s_sp_native_target_count = 0L,
    mgcv_cpp_backend_same_s_sp_native_max_targets = 0L,
    mgcv_cpp_backend_same_s_sp_native_mean_targets = 0,
    mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count = 0L,
    mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio = 0,
    mgcv_cpp_backend_same_s_setup_native_group_count = 0L,
    mgcv_cpp_backend_same_s_setup_native_target_count = 0L,
    mgcv_cpp_backend_same_s_setup_native_max_targets = 0L,
    mgcv_cpp_backend_same_s_setup_native_mean_targets = 0,
    mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count = 0L,
    mgcv_cpp_backend_same_s_setup_native_reuse_ratio = 0,
    mgcv_cpp_backend_same_s_setup_input_potential_saved_ms = 0,
    mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms = 0,
    mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms = 0,
    mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms = 0,
    mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms = 0,
    mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio = 0,
    mgcv_cpp_backend_same_s_gam_fit_preserved_ms = 0,
    mgcv_cpp_same_s_prefill_enabled = 0L,
    mgcv_cpp_same_s_prefill_group_count = 0L,
    mgcv_cpp_same_s_prefill_target_count = 0L,
    mgcv_cpp_same_s_prefill_cache_insert_count = 0L,
    mgcv_cpp_same_s_prefill_existing_count = 0L,
    mgcv_cpp_same_s_prefill_unused_count = 0L,
    mgcv_cpp_same_s_prefill_ms = 0,
    mgcv_cpp_same_s_prefill_error_count = 0L,
    mgcv_cpp_same_s_setup_provider_enabled = 0L,
    mgcv_cpp_same_s_setup_provider_group_count = 0L,
    mgcv_cpp_same_s_setup_provider_target_count = 0L,
    mgcv_cpp_same_s_setup_provider_template_count = 0L,
    mgcv_cpp_same_s_setup_provider_reuse_count = 0L,
    mgcv_cpp_same_s_setup_provider_setup_ms = 0,
    mgcv_cpp_same_s_setup_provider_error_count = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_enabled = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_count = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_group_count = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_target_count = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count = 0L,
    mgcv_cpp_same_s_setup_provider_chunk_ms = 0,
    mgcv_cpp_same_s_setup_provider_chunk_error_count = 0L,
    mgcv_cpp_backend_native_s_size_limit = 0,
    mgcv_cpp_backend_condition_threshold = 0,
    mgcv_cpp_shadow_enabled = 0L,
    mgcv_cpp_shadow_count = 0L,
    mgcv_cpp_shadow_native_count = 0L,
    mgcv_cpp_shadow_fallback_count = 0L,
    mgcv_cpp_shadow_high_condition_fallback_count = 0L,
    mgcv_cpp_shadow_outside_envelope_fallback_count = 0L,
    mgcv_cpp_shadow_error_count = 0L,
    mgcv_cpp_shadow_residual_mismatch_count = 0L,
    mgcv_cpp_shadow_ms = 0,
    mgcv_cpp_shadow_max_abs_diff = 0,
    mgcv_cpp_shadow_max_rel_l2 = 0,
    mgcv_cpp_shadow_native_s_size_limit = 0,
    mgcv_cpp_shadow_condition_threshold = 0,
    mgcv_residual_keys = character(),
    mgcv_s_keys = character(),
    mgcv_residual_cache_hit_keys = character(),
    mgcv_residual_cache_miss_keys = character(),
    mgcv_cpp_backend_native_residual_keys = character(),
    mgcv_cpp_backend_native_s_keys = character(),
    mgcv_cpp_backend_native_s_sp_keys = character(),
    mgcv_cpp_backend_native_s_setup_keys = character(),
    mgcv_cpp_backend_native_input_setup_ms = numeric(),
    mgcv_cpp_backend_native_gam_fit_ms = numeric(),
    mgcv_cpp_backend_native_sp_extract_ms = numeric(),
    mgcv_cpp_backend_native_setup_extract_ms = numeric(),
    mgcv_cpp_backend_native_condition_ms = numeric(),
    mgcv_cpp_backend_native_solve_call_ms = numeric(),
    direct_ci_count = 0L,
    conditional_ci_count = 0L,
    mgcv_fit_count = 0L,
    dcov_gamma_count = 0L,
    fake_level0_test_count = 0L
  )
}

fastkpc_legacy_runtime_add <- function(a, b) {
  if (is.null(a)) a <- fastkpc_legacy_runtime_zero()
  if (is.null(b)) b <- fastkpc_legacy_runtime_zero()
  list(
    ci_total_ms = as.numeric(a$ci_total_ms) + as.numeric(b$ci_total_ms),
    residual_ms = as.numeric(a$residual_ms) + as.numeric(b$residual_ms),
    dcov_gamma_ms =
      as.numeric(a$dcov_gamma_ms) + as.numeric(b$dcov_gamma_ms),
    dcov_input_ms =
      as.numeric(a$dcov_input_ms) + as.numeric(b$dcov_input_ms),
    dcov_h_ms =
      as.numeric(a$dcov_h_ms) + as.numeric(b$dcov_h_ms),
    dcov_distance_ms =
      as.numeric(a$dcov_distance_ms) + as.numeric(b$dcov_distance_ms),
    dcov_lowrank_ms =
      as.numeric(a$dcov_lowrank_ms) + as.numeric(b$dcov_lowrank_ms),
    dcov_statistic_ms =
      as.numeric(a$dcov_statistic_ms) + as.numeric(b$dcov_statistic_ms),
    dcov_moment_ms =
      as.numeric(a$dcov_moment_ms) + as.numeric(b$dcov_moment_ms),
    dcov_pgamma_ms =
      as.numeric(a$dcov_pgamma_ms) + as.numeric(b$dcov_pgamma_ms),
    dcov_output_ms =
      as.numeric(a$dcov_output_ms) + as.numeric(b$dcov_output_ms),
    dcov_unaccounted_ms =
      as.numeric(a$dcov_unaccounted_ms) +
        as.numeric(b$dcov_unaccounted_ms),
    dcov_r_backend_count =
      as.integer(a$dcov_r_backend_count) +
        as.integer(b$dcov_r_backend_count),
    dcov_cpp_backend_count =
      as.integer(a$dcov_cpp_backend_count) +
        as.integer(b$dcov_cpp_backend_count),
    dcov_cpp_backend_ms =
      as.numeric(a$dcov_cpp_backend_ms) + as.numeric(b$dcov_cpp_backend_ms),
    dcov_cpp_backend_error_count =
      as.integer(a$dcov_cpp_backend_error_count) +
        as.integer(b$dcov_cpp_backend_error_count),
    dcov_cpp_backend_fallback_count =
      as.integer(a$dcov_cpp_backend_fallback_count) +
        as.integer(b$dcov_cpp_backend_fallback_count),
    dcov_cpp_backend_max_p_diff =
      max(as.numeric(a$dcov_cpp_backend_max_p_diff),
          as.numeric(b$dcov_cpp_backend_max_p_diff)),
    dcov_cpp_backend_decision_flip_count =
      as.integer(a$dcov_cpp_backend_decision_flip_count) +
        as.integer(b$dcov_cpp_backend_decision_flip_count),
    dcov_cpp_shadow_ms =
      as.numeric(a$dcov_cpp_shadow_ms) +
        as.numeric(b$dcov_cpp_shadow_ms),
    dcov_cpp_shadow_count =
      as.integer(a$dcov_cpp_shadow_count) +
        as.integer(b$dcov_cpp_shadow_count),
    dcov_cpp_shadow_error_count =
      as.integer(a$dcov_cpp_shadow_error_count) +
        as.integer(b$dcov_cpp_shadow_error_count),
    dcov_cpp_shadow_decision_flip_count =
      as.integer(a$dcov_cpp_shadow_decision_flip_count) +
        as.integer(b$dcov_cpp_shadow_decision_flip_count),
    dcov_cpp_shadow_near_alpha_count =
      as.integer(a$dcov_cpp_shadow_near_alpha_count) +
        as.integer(b$dcov_cpp_shadow_near_alpha_count),
    dcov_cpp_shadow_max_p_diff =
      max(as.numeric(a$dcov_cpp_shadow_max_p_diff),
          as.numeric(b$dcov_cpp_shadow_max_p_diff)),
    dcov_cpp_shadow_max_nV2_diff =
      max(as.numeric(a$dcov_cpp_shadow_max_nV2_diff),
          as.numeric(b$dcov_cpp_shadow_max_nV2_diff)),
    dcov_cpp_shadow_max_mean_diff =
      max(as.numeric(a$dcov_cpp_shadow_max_mean_diff),
          as.numeric(b$dcov_cpp_shadow_max_mean_diff)),
    dcov_cpp_shadow_max_variance_diff =
      max(as.numeric(a$dcov_cpp_shadow_max_variance_diff),
          as.numeric(b$dcov_cpp_shadow_max_variance_diff)),
    dcov_cpp_input_ms =
      as.numeric(a$dcov_cpp_input_ms) + as.numeric(b$dcov_cpp_input_ms),
    dcov_cpp_distance_ms =
      as.numeric(a$dcov_cpp_distance_ms) + as.numeric(b$dcov_cpp_distance_ms),
    dcov_cpp_lowrank_ms =
      as.numeric(a$dcov_cpp_lowrank_ms) + as.numeric(b$dcov_cpp_lowrank_ms),
    dcov_cpp_lowrank_eig_ms =
      as.numeric(a$dcov_cpp_lowrank_eig_ms) +
        as.numeric(b$dcov_cpp_lowrank_eig_ms),
    dcov_cpp_lowrank_select_ms =
      as.numeric(a$dcov_cpp_lowrank_select_ms) +
        as.numeric(b$dcov_cpp_lowrank_select_ms),
    dcov_cpp_lowrank_center_ms =
      as.numeric(a$dcov_cpp_lowrank_center_ms) +
        as.numeric(b$dcov_cpp_lowrank_center_ms),
    dcov_cpp_lowrank_unaccounted_ms =
      as.numeric(a$dcov_cpp_lowrank_unaccounted_ms) +
        as.numeric(b$dcov_cpp_lowrank_unaccounted_ms),
    dcov_cpp_lowrank_full_eig_count =
      as.integer(a$dcov_cpp_lowrank_full_eig_count) +
        as.integer(b$dcov_cpp_lowrank_full_eig_count),
    dcov_cpp_lowrank_spectra_count =
      as.integer(a$dcov_cpp_lowrank_spectra_count) +
        as.integer(b$dcov_cpp_lowrank_spectra_count),
    dcov_cpp_lowrank_spectra_converged_count =
      as.integer(a$dcov_cpp_lowrank_spectra_converged_count) +
        as.integer(b$dcov_cpp_lowrank_spectra_converged_count),
    dcov_cpp_lowrank_spectra_failed_count =
      as.integer(a$dcov_cpp_lowrank_spectra_failed_count) +
        as.integer(b$dcov_cpp_lowrank_spectra_failed_count),
    dcov_cpp_lowrank_spectra_fallback_full_eig_count =
      as.integer(a$dcov_cpp_lowrank_spectra_fallback_full_eig_count) +
        as.integer(b$dcov_cpp_lowrank_spectra_fallback_full_eig_count),
    dcov_cpp_lowrank_spectra_iterations =
      as.integer(a$dcov_cpp_lowrank_spectra_iterations) +
        as.integer(b$dcov_cpp_lowrank_spectra_iterations),
    dcov_cpp_lowrank_spectra_nconv =
      as.integer(a$dcov_cpp_lowrank_spectra_nconv) +
        as.integer(b$dcov_cpp_lowrank_spectra_nconv),
    dcov_cpp_lowrank_spectra_ncv =
      max(as.integer(a$dcov_cpp_lowrank_spectra_ncv),
          as.integer(b$dcov_cpp_lowrank_spectra_ncv)),
    dcov_cpp_lowrank_spectra_tol =
      max(as.numeric(a$dcov_cpp_lowrank_spectra_tol),
          as.numeric(b$dcov_cpp_lowrank_spectra_tol)),
    dcov_cpp_statistic_ms =
      as.numeric(a$dcov_cpp_statistic_ms) +
        as.numeric(b$dcov_cpp_statistic_ms),
    dcov_cpp_moment_ms =
      as.numeric(a$dcov_cpp_moment_ms) + as.numeric(b$dcov_cpp_moment_ms),
    dcov_cpp_pgamma_ms =
      as.numeric(a$dcov_cpp_pgamma_ms) + as.numeric(b$dcov_cpp_pgamma_ms),
    dcov_cpp_accounted_ms =
      as.numeric(a$dcov_cpp_accounted_ms) +
        as.numeric(b$dcov_cpp_accounted_ms),
    dcov_cpp_unaccounted_ms =
      as.numeric(a$dcov_cpp_unaccounted_ms) +
        as.numeric(b$dcov_cpp_unaccounted_ms),
    dcov_cpp_overhead_ms =
      as.numeric(a$dcov_cpp_overhead_ms) + as.numeric(b$dcov_cpp_overhead_ms),
    mgcv_residual_request_count =
      as.integer(a$mgcv_residual_request_count) +
        as.integer(b$mgcv_residual_request_count),
    mgcv_cache_hit_count =
      as.integer(a$mgcv_cache_hit_count) +
        as.integer(b$mgcv_cache_hit_count),
    mgcv_cache_miss_count =
      as.integer(a$mgcv_cache_miss_count) +
        as.integer(b$mgcv_cache_miss_count),
    mgcv_residual_cache_hit_key_count =
      as.integer(a$mgcv_residual_cache_hit_key_count) +
        as.integer(b$mgcv_residual_cache_hit_key_count),
    mgcv_residual_cache_miss_key_count =
      as.integer(a$mgcv_residual_cache_miss_key_count) +
        as.integer(b$mgcv_residual_cache_miss_key_count),
    mgcv_residual_cache_miss_s_group_count =
      as.integer(a$mgcv_residual_cache_miss_s_group_count) +
        as.integer(b$mgcv_residual_cache_miss_s_group_count),
    mgcv_residual_cache_miss_s_total_targets =
      as.integer(a$mgcv_residual_cache_miss_s_total_targets) +
        as.integer(b$mgcv_residual_cache_miss_s_total_targets),
    mgcv_residual_cache_miss_s_max_targets =
      max(as.integer(a$mgcv_residual_cache_miss_s_max_targets),
          as.integer(b$mgcv_residual_cache_miss_s_max_targets)),
    mgcv_residual_cache_miss_s_mean_targets = {
      total_groups <- as.integer(a$mgcv_residual_cache_miss_s_group_count) +
        as.integer(b$mgcv_residual_cache_miss_s_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_residual_cache_miss_s_mean_targets) *
           as.integer(a$mgcv_residual_cache_miss_s_group_count) +
           as.numeric(b$mgcv_residual_cache_miss_s_mean_targets) *
           as.integer(b$mgcv_residual_cache_miss_s_group_count)) /
          total_groups
      } else {
        0
      }
    },
    mgcv_residual_cache_miss_s_reuse_opportunity_count =
      as.integer(a$mgcv_residual_cache_miss_s_reuse_opportunity_count) +
        as.integer(b$mgcv_residual_cache_miss_s_reuse_opportunity_count),
    mgcv_residual_cache_miss_s_reuse_ratio = {
      total_targets <- as.integer(a$mgcv_residual_cache_miss_s_total_targets) +
        as.integer(b$mgcv_residual_cache_miss_s_total_targets)
      if (total_targets > 0L) {
        (as.integer(a$mgcv_residual_cache_miss_s_reuse_opportunity_count) +
           as.integer(b$mgcv_residual_cache_miss_s_reuse_opportunity_count)) /
          total_targets
      } else {
        0
      }
    },
    mgcv_unique_residual_key_count =
      as.integer(a$mgcv_unique_residual_key_count) +
        as.integer(b$mgcv_unique_residual_key_count),
    mgcv_duplicate_residual_key_count =
      as.integer(a$mgcv_duplicate_residual_key_count) +
        as.integer(b$mgcv_duplicate_residual_key_count),
    mgcv_unique_target_s_count =
      as.integer(a$mgcv_unique_target_s_count) +
        as.integer(b$mgcv_unique_target_s_count),
    mgcv_unique_s_count =
      as.integer(a$mgcv_unique_s_count) + as.integer(b$mgcv_unique_s_count),
    mgcv_same_s_group_count =
      as.integer(a$mgcv_same_s_group_count) +
        as.integer(b$mgcv_same_s_group_count),
    mgcv_same_s_total_targets =
      as.integer(a$mgcv_same_s_total_targets) +
        as.integer(b$mgcv_same_s_total_targets),
    mgcv_same_s_max_targets =
      max(as.integer(a$mgcv_same_s_max_targets),
          as.integer(b$mgcv_same_s_max_targets)),
    mgcv_same_s_mean_targets = {
      total_groups <- as.integer(a$mgcv_same_s_group_count) +
        as.integer(b$mgcv_same_s_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_same_s_mean_targets) *
           as.integer(a$mgcv_same_s_group_count) +
           as.numeric(b$mgcv_same_s_mean_targets) *
           as.integer(b$mgcv_same_s_group_count)) / total_groups
      } else {
        0
      }
    },
    mgcv_same_s_reuse_opportunity_count =
      as.integer(a$mgcv_same_s_reuse_opportunity_count) +
        as.integer(b$mgcv_same_s_reuse_opportunity_count),
    mgcv_residual_cache_insert_count =
      as.integer(a$mgcv_residual_cache_insert_count) +
        as.integer(b$mgcv_residual_cache_insert_count),
    mgcv_residual_cache_entries =
      as.integer(a$mgcv_residual_cache_entries) +
        as.integer(b$mgcv_residual_cache_entries),
    mgcv_residual_cache_hit_ms =
      as.numeric(a$mgcv_residual_cache_hit_ms) +
        as.numeric(b$mgcv_residual_cache_hit_ms),
    mgcv_residual_cache_store_ms =
      as.numeric(a$mgcv_residual_cache_store_ms) +
        as.numeric(b$mgcv_residual_cache_store_ms),
    mgcv_fit_avoided_count =
      as.integer(a$mgcv_fit_avoided_count) +
        as.integer(b$mgcv_fit_avoided_count),
    mgcv_residual_affinity_enabled =
      max(as.integer(a$mgcv_residual_affinity_enabled),
          as.integer(b$mgcv_residual_affinity_enabled)),
    mgcv_residual_affinity_group_count =
      as.integer(a$mgcv_residual_affinity_group_count) +
        as.integer(b$mgcv_residual_affinity_group_count),
    mgcv_residual_affinity_task_count =
      as.integer(a$mgcv_residual_affinity_task_count) +
        as.integer(b$mgcv_residual_affinity_task_count),
    mgcv_residual_affinity_worker_count =
      max(as.integer(a$mgcv_residual_affinity_worker_count),
          as.integer(b$mgcv_residual_affinity_worker_count)),
    mgcv_residual_affinity_max_group_size =
      max(as.integer(a$mgcv_residual_affinity_max_group_size),
          as.integer(b$mgcv_residual_affinity_max_group_size)),
    mgcv_residual_affinity_mean_group_size = {
      total_groups <- as.integer(a$mgcv_residual_affinity_group_count) +
        as.integer(b$mgcv_residual_affinity_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_residual_affinity_mean_group_size) *
           as.integer(a$mgcv_residual_affinity_group_count) +
           as.numeric(b$mgcv_residual_affinity_mean_group_size) *
           as.integer(b$mgcv_residual_affinity_group_count)) / total_groups
      } else {
        0
      }
    },
    mgcv_residual_affinity_load_imbalance =
      max(as.numeric(a$mgcv_residual_affinity_load_imbalance),
          as.numeric(b$mgcv_residual_affinity_load_imbalance)),
    mgcv_residual_affinity_split_group_count =
      as.integer(a$mgcv_residual_affinity_split_group_count) +
        as.integer(b$mgcv_residual_affinity_split_group_count),
    mgcv_residual_affinity_split_group_tasks =
      as.integer(a$mgcv_residual_affinity_split_group_tasks) +
        as.integer(b$mgcv_residual_affinity_split_group_tasks),
    mgcv_residual_affinity_split_group_pieces =
      as.integer(a$mgcv_residual_affinity_split_group_pieces) +
        as.integer(b$mgcv_residual_affinity_split_group_pieces),
    mgcv_residual_cache_theoretical_hit_count =
      as.integer(a$mgcv_residual_cache_theoretical_hit_count) +
        as.integer(b$mgcv_residual_cache_theoretical_hit_count),
    mgcv_residual_cache_realized_hit_count =
      as.integer(a$mgcv_residual_cache_realized_hit_count) +
        as.integer(b$mgcv_residual_cache_realized_hit_count),
    mgcv_residual_cache_lost_duplicate_count =
      as.integer(a$mgcv_residual_cache_lost_duplicate_count) +
        as.integer(b$mgcv_residual_cache_lost_duplicate_count),
    mgcv_residual_cache_lost_cross_worker_count =
      as.integer(a$mgcv_residual_cache_lost_cross_worker_count) +
        as.integer(b$mgcv_residual_cache_lost_cross_worker_count),
    mgcv_residual_cache_lost_split_s_group_count =
      as.integer(a$mgcv_residual_cache_lost_split_s_group_count) +
        as.integer(b$mgcv_residual_cache_lost_split_s_group_count),
    mgcv_residual_cache_lost_cross_level_count =
      as.integer(a$mgcv_residual_cache_lost_cross_level_count) +
        as.integer(b$mgcv_residual_cache_lost_cross_level_count),
    mgcv_residual_owner_enabled =
      max(as.integer(a$mgcv_residual_owner_enabled),
          as.integer(b$mgcv_residual_owner_enabled)),
    mgcv_residual_owner_key_count =
      as.integer(a$mgcv_residual_owner_key_count) +
        as.integer(b$mgcv_residual_owner_key_count),
    mgcv_residual_owner_task_count =
      as.integer(a$mgcv_residual_owner_task_count) +
        as.integer(b$mgcv_residual_owner_task_count),
    mgcv_residual_owner_both_local_count =
      as.integer(a$mgcv_residual_owner_both_local_count) +
        as.integer(b$mgcv_residual_owner_both_local_count),
    mgcv_residual_owner_one_local_count =
      as.integer(a$mgcv_residual_owner_one_local_count) +
        as.integer(b$mgcv_residual_owner_one_local_count),
    mgcv_residual_owner_none_local_count =
      as.integer(a$mgcv_residual_owner_none_local_count) +
        as.integer(b$mgcv_residual_owner_none_local_count),
    mgcv_residual_owner_conflict_count =
      as.integer(a$mgcv_residual_owner_conflict_count) +
        as.integer(b$mgcv_residual_owner_conflict_count),
    mgcv_residual_owner_predicted_hit_count =
      as.integer(a$mgcv_residual_owner_predicted_hit_count) +
        as.integer(b$mgcv_residual_owner_predicted_hit_count),
    mgcv_residual_owner_realized_hit_count =
      as.integer(a$mgcv_residual_owner_realized_hit_count) +
        as.integer(b$mgcv_residual_owner_realized_hit_count),
    mgcv_residual_owner_lost_duplicate_count =
      as.integer(a$mgcv_residual_owner_lost_duplicate_count) +
        as.integer(b$mgcv_residual_owner_lost_duplicate_count),
    mgcv_residual_owner_load_imbalance =
      max(as.numeric(a$mgcv_residual_owner_load_imbalance),
          as.numeric(b$mgcv_residual_owner_load_imbalance)),
    mgcv_residual_owner_spill_count =
      as.integer(a$mgcv_residual_owner_spill_count) +
        as.integer(b$mgcv_residual_owner_spill_count),
    mgcv_residual_owner_schedule_build_ms =
      as.numeric(a$mgcv_residual_owner_schedule_build_ms) +
        as.numeric(b$mgcv_residual_owner_schedule_build_ms),
    mgcv_residual_owner_key_enum_ms =
      as.numeric(a$mgcv_residual_owner_key_enum_ms) +
        as.numeric(b$mgcv_residual_owner_key_enum_ms),
    mgcv_residual_owner_key_map_build_ms =
      as.numeric(a$mgcv_residual_owner_key_map_build_ms) +
        as.numeric(b$mgcv_residual_owner_key_map_build_ms),
    mgcv_residual_owner_task_score_ms =
      as.numeric(a$mgcv_residual_owner_task_score_ms) +
        as.numeric(b$mgcv_residual_owner_task_score_ms),
    mgcv_residual_owner_greedy_assign_ms =
      as.numeric(a$mgcv_residual_owner_greedy_assign_ms) +
        as.numeric(b$mgcv_residual_owner_greedy_assign_ms),
    mgcv_residual_owner_chunk_sort_ms =
      as.numeric(a$mgcv_residual_owner_chunk_sort_ms) +
        as.numeric(b$mgcv_residual_owner_chunk_sort_ms),
    mgcv_residual_owner_chunk_materialize_ms =
      as.numeric(a$mgcv_residual_owner_chunk_materialize_ms) +
        as.numeric(b$mgcv_residual_owner_chunk_materialize_ms),
    mgcv_residual_owner_worker_max_ms =
      max(as.numeric(a$mgcv_residual_owner_worker_max_ms),
          as.numeric(b$mgcv_residual_owner_worker_max_ms)),
    mgcv_residual_owner_worker_median_ms =
      max(as.numeric(a$mgcv_residual_owner_worker_median_ms),
          as.numeric(b$mgcv_residual_owner_worker_median_ms)),
    mgcv_residual_owner_worker_elapsed_imbalance =
      max(as.numeric(a$mgcv_residual_owner_worker_elapsed_imbalance),
          as.numeric(b$mgcv_residual_owner_worker_elapsed_imbalance)),
    mgcv_residual_owner_worker_task_max =
      max(as.integer(a$mgcv_residual_owner_worker_task_max),
          as.integer(b$mgcv_residual_owner_worker_task_max)),
    mgcv_residual_owner_worker_task_median =
      max(as.numeric(a$mgcv_residual_owner_worker_task_median),
          as.numeric(b$mgcv_residual_owner_worker_task_median)),
    mgcv_residual_owner_worker_fit_max =
      max(as.integer(a$mgcv_residual_owner_worker_fit_max),
          as.integer(b$mgcv_residual_owner_worker_fit_max)),
    mgcv_residual_owner_worker_fit_median =
      max(as.numeric(a$mgcv_residual_owner_worker_fit_median),
          as.numeric(b$mgcv_residual_owner_worker_fit_median)),
    mgcv_residual_owner_worker_cache_hit_max =
      max(as.integer(a$mgcv_residual_owner_worker_cache_hit_max),
          as.integer(b$mgcv_residual_owner_worker_cache_hit_max)),
    mgcv_residual_owner_worker_cache_hit_median =
      max(as.numeric(a$mgcv_residual_owner_worker_cache_hit_median),
          as.numeric(b$mgcv_residual_owner_worker_cache_hit_median)),
    mgcv_residual_owner_worker_residual_ms_max =
      max(as.numeric(a$mgcv_residual_owner_worker_residual_ms_max),
          as.numeric(b$mgcv_residual_owner_worker_residual_ms_max)),
    mgcv_residual_owner_worker_residual_ms_median =
      max(as.numeric(a$mgcv_residual_owner_worker_residual_ms_median),
          as.numeric(b$mgcv_residual_owner_worker_residual_ms_median)),
    mgcv_prefetch_enabled =
      max(as.integer(a$mgcv_prefetch_enabled),
          as.integer(b$mgcv_prefetch_enabled)),
    mgcv_prefetch_level_count =
      as.integer(a$mgcv_prefetch_level_count) +
        as.integer(b$mgcv_prefetch_level_count),
    mgcv_prefetch_key_count =
      as.integer(a$mgcv_prefetch_key_count) +
        as.integer(b$mgcv_prefetch_key_count),
    mgcv_prefetch_fit_count =
      as.integer(a$mgcv_prefetch_fit_count) +
        as.integer(b$mgcv_prefetch_fit_count),
    mgcv_prefetch_fit_ms =
      as.numeric(a$mgcv_prefetch_fit_ms) +
        as.numeric(b$mgcv_prefetch_fit_ms),
    mgcv_prefetch_collect_ms =
      as.numeric(a$mgcv_prefetch_collect_ms) +
        as.numeric(b$mgcv_prefetch_collect_ms),
    mgcv_prefetch_matrix_build_ms =
      as.numeric(a$mgcv_prefetch_matrix_build_ms) +
        as.numeric(b$mgcv_prefetch_matrix_build_ms),
    mgcv_prefetch_payload_bytes =
      as.numeric(a$mgcv_prefetch_payload_bytes) +
        as.numeric(b$mgcv_prefetch_payload_bytes),
    mgcv_prefetch_max_level_payload_bytes =
      max(as.numeric(a$mgcv_prefetch_max_level_payload_bytes),
          as.numeric(b$mgcv_prefetch_max_level_payload_bytes)),
    mgcv_prefetch_lookup_ms =
      as.numeric(a$mgcv_prefetch_lookup_ms) +
        as.numeric(b$mgcv_prefetch_lookup_ms),
    mgcv_prefetch_ci_phase_ms =
      as.numeric(a$mgcv_prefetch_ci_phase_ms) +
        as.numeric(b$mgcv_prefetch_ci_phase_ms),
    mgcv_prefetch_error_count =
      as.integer(a$mgcv_prefetch_error_count) +
        as.integer(b$mgcv_prefetch_error_count),
    mgcv_prefetch_consumed_key_count =
      as.integer(a$mgcv_prefetch_consumed_key_count) +
        as.integer(b$mgcv_prefetch_consumed_key_count),
    mgcv_prefetch_unused_key_count =
      as.integer(a$mgcv_prefetch_unused_key_count) +
        as.integer(b$mgcv_prefetch_unused_key_count),
    mgcv_key_build_ms =
      as.numeric(a$mgcv_key_build_ms) + as.numeric(b$mgcv_key_build_ms),
    mgcv_cache_lookup_ms =
      as.numeric(a$mgcv_cache_lookup_ms) + as.numeric(b$mgcv_cache_lookup_ms),
    mgcv_formula_build_ms =
      as.numeric(a$mgcv_formula_build_ms) +
        as.numeric(b$mgcv_formula_build_ms),
    mgcv_data_subset_ms =
      as.numeric(a$mgcv_data_subset_ms) +
        as.numeric(b$mgcv_data_subset_ms),
    mgcv_fit_call_ms =
      as.numeric(a$mgcv_fit_call_ms) + as.numeric(b$mgcv_fit_call_ms),
    mgcv_residual_extract_ms =
      as.numeric(a$mgcv_residual_extract_ms) +
        as.numeric(b$mgcv_residual_extract_ms),
    mgcv_result_store_ms =
      as.numeric(a$mgcv_result_store_ms) +
        as.numeric(b$mgcv_result_store_ms),
    mgcv_unaccounted_ms =
      as.numeric(a$mgcv_unaccounted_ms) +
        as.numeric(b$mgcv_unaccounted_ms),
    mgcv_s_size_0_count =
      as.integer(a$mgcv_s_size_0_count) + as.integer(b$mgcv_s_size_0_count),
    mgcv_s_size_1_count =
      as.integer(a$mgcv_s_size_1_count) + as.integer(b$mgcv_s_size_1_count),
    mgcv_s_size_2_count =
      as.integer(a$mgcv_s_size_2_count) + as.integer(b$mgcv_s_size_2_count),
    mgcv_s_size_gt2_count =
      as.integer(a$mgcv_s_size_gt2_count) +
        as.integer(b$mgcv_s_size_gt2_count),
    mgcv_r_backend_count =
      as.integer(a$mgcv_r_backend_count) +
        as.integer(b$mgcv_r_backend_count),
    mgcv_cpp_backend_enabled =
      max(as.integer(a$mgcv_cpp_backend_enabled),
          as.integer(b$mgcv_cpp_backend_enabled)),
    mgcv_cpp_backend_count =
      as.integer(a$mgcv_cpp_backend_count) +
        as.integer(b$mgcv_cpp_backend_count),
    mgcv_cpp_backend_native_count =
      as.integer(a$mgcv_cpp_backend_native_count) +
        as.integer(b$mgcv_cpp_backend_native_count),
    mgcv_cpp_backend_fallback_count =
      as.integer(a$mgcv_cpp_backend_fallback_count) +
        as.integer(b$mgcv_cpp_backend_fallback_count),
    mgcv_cpp_backend_high_condition_fallback_count =
      as.integer(a$mgcv_cpp_backend_high_condition_fallback_count) +
        as.integer(b$mgcv_cpp_backend_high_condition_fallback_count),
    mgcv_cpp_backend_outside_envelope_fallback_count =
      as.integer(a$mgcv_cpp_backend_outside_envelope_fallback_count) +
        as.integer(b$mgcv_cpp_backend_outside_envelope_fallback_count),
    mgcv_cpp_backend_error_count =
      as.integer(a$mgcv_cpp_backend_error_count) +
        as.integer(b$mgcv_cpp_backend_error_count),
    mgcv_cpp_backend_ms =
      as.numeric(a$mgcv_cpp_backend_ms) +
        as.numeric(b$mgcv_cpp_backend_ms),
    mgcv_cpp_backend_input_setup_ms =
      as.numeric(a$mgcv_cpp_backend_input_setup_ms) +
        as.numeric(b$mgcv_cpp_backend_input_setup_ms),
    mgcv_cpp_backend_gam_fit_ms =
      as.numeric(a$mgcv_cpp_backend_gam_fit_ms) +
        as.numeric(b$mgcv_cpp_backend_gam_fit_ms),
    mgcv_cpp_backend_sp_extract_ms =
      as.numeric(a$mgcv_cpp_backend_sp_extract_ms) +
        as.numeric(b$mgcv_cpp_backend_sp_extract_ms),
    mgcv_cpp_backend_setup_extract_ms =
      as.numeric(a$mgcv_cpp_backend_setup_extract_ms) +
        as.numeric(b$mgcv_cpp_backend_setup_extract_ms),
    mgcv_cpp_backend_condition_ms =
      as.numeric(a$mgcv_cpp_backend_condition_ms) +
        as.numeric(b$mgcv_cpp_backend_condition_ms),
    mgcv_cpp_backend_native_solve_ms =
      as.numeric(a$mgcv_cpp_backend_native_solve_ms) +
        as.numeric(b$mgcv_cpp_backend_native_solve_ms),
    mgcv_cpp_backend_fallback_ms =
      as.numeric(a$mgcv_cpp_backend_fallback_ms) +
        as.numeric(b$mgcv_cpp_backend_fallback_ms),
    mgcv_cpp_backend_s_size_0_count =
      as.integer(a$mgcv_cpp_backend_s_size_0_count) +
        as.integer(b$mgcv_cpp_backend_s_size_0_count),
    mgcv_cpp_backend_s_size_1_count =
      as.integer(a$mgcv_cpp_backend_s_size_1_count) +
        as.integer(b$mgcv_cpp_backend_s_size_1_count),
    mgcv_cpp_backend_s_size_2_count =
      as.integer(a$mgcv_cpp_backend_s_size_2_count) +
        as.integer(b$mgcv_cpp_backend_s_size_2_count),
    mgcv_cpp_backend_s_size_gt2_count =
      as.integer(a$mgcv_cpp_backend_s_size_gt2_count) +
        as.integer(b$mgcv_cpp_backend_s_size_gt2_count),
    mgcv_cpp_backend_native_s_size_0_count =
      as.integer(a$mgcv_cpp_backend_native_s_size_0_count) +
        as.integer(b$mgcv_cpp_backend_native_s_size_0_count),
    mgcv_cpp_backend_native_s_size_1_count =
      as.integer(a$mgcv_cpp_backend_native_s_size_1_count) +
        as.integer(b$mgcv_cpp_backend_native_s_size_1_count),
    mgcv_cpp_backend_native_s_size_2_count =
      as.integer(a$mgcv_cpp_backend_native_s_size_2_count) +
        as.integer(b$mgcv_cpp_backend_native_s_size_2_count),
    mgcv_cpp_backend_native_s_size_gt2_count =
      as.integer(a$mgcv_cpp_backend_native_s_size_gt2_count) +
        as.integer(b$mgcv_cpp_backend_native_s_size_gt2_count),
    mgcv_cpp_backend_fallback_s_size_0_count =
      as.integer(a$mgcv_cpp_backend_fallback_s_size_0_count) +
        as.integer(b$mgcv_cpp_backend_fallback_s_size_0_count),
    mgcv_cpp_backend_fallback_s_size_1_count =
      as.integer(a$mgcv_cpp_backend_fallback_s_size_1_count) +
        as.integer(b$mgcv_cpp_backend_fallback_s_size_1_count),
    mgcv_cpp_backend_fallback_s_size_2_count =
      as.integer(a$mgcv_cpp_backend_fallback_s_size_2_count) +
        as.integer(b$mgcv_cpp_backend_fallback_s_size_2_count),
    mgcv_cpp_backend_fallback_s_size_gt2_count =
      as.integer(a$mgcv_cpp_backend_fallback_s_size_gt2_count) +
        as.integer(b$mgcv_cpp_backend_fallback_s_size_gt2_count),
    mgcv_cpp_backend_same_s_native_group_count =
      as.integer(a$mgcv_cpp_backend_same_s_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_native_group_count),
    mgcv_cpp_backend_same_s_native_target_count =
      as.integer(a$mgcv_cpp_backend_same_s_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_native_target_count),
    mgcv_cpp_backend_same_s_native_max_targets =
      max(as.integer(a$mgcv_cpp_backend_same_s_native_max_targets),
          as.integer(b$mgcv_cpp_backend_same_s_native_max_targets)),
    mgcv_cpp_backend_same_s_native_mean_targets = {
      total_groups <- as.integer(a$mgcv_cpp_backend_same_s_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_native_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_cpp_backend_same_s_native_mean_targets) *
           as.integer(a$mgcv_cpp_backend_same_s_native_group_count) +
           as.numeric(b$mgcv_cpp_backend_same_s_native_mean_targets) *
             as.integer(b$mgcv_cpp_backend_same_s_native_group_count)) /
          total_groups
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_native_reuse_opportunity_count =
      as.integer(a$mgcv_cpp_backend_same_s_native_reuse_opportunity_count) +
        as.integer(b$mgcv_cpp_backend_same_s_native_reuse_opportunity_count),
    mgcv_cpp_backend_same_s_native_setup_reuse_ratio = {
      total_targets <- as.integer(a$mgcv_cpp_backend_same_s_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_native_target_count)
      if (total_targets > 0L) {
        (as.integer(a$mgcv_cpp_backend_same_s_native_reuse_opportunity_count) +
           as.integer(b$mgcv_cpp_backend_same_s_native_reuse_opportunity_count)) /
          total_targets
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_sp_native_group_count =
      as.integer(a$mgcv_cpp_backend_same_s_sp_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_sp_native_group_count),
    mgcv_cpp_backend_same_s_sp_native_target_count =
      as.integer(a$mgcv_cpp_backend_same_s_sp_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_sp_native_target_count),
    mgcv_cpp_backend_same_s_sp_native_max_targets =
      max(as.integer(a$mgcv_cpp_backend_same_s_sp_native_max_targets),
          as.integer(b$mgcv_cpp_backend_same_s_sp_native_max_targets)),
    mgcv_cpp_backend_same_s_sp_native_mean_targets = {
      total_groups <-
        as.integer(a$mgcv_cpp_backend_same_s_sp_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_sp_native_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_cpp_backend_same_s_sp_native_mean_targets) *
           as.integer(a$mgcv_cpp_backend_same_s_sp_native_group_count) +
           as.numeric(b$mgcv_cpp_backend_same_s_sp_native_mean_targets) *
             as.integer(b$mgcv_cpp_backend_same_s_sp_native_group_count)) /
          total_groups
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count =
      as.integer(a$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count) +
        as.integer(b$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count),
    mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio = {
      total_targets <-
        as.integer(a$mgcv_cpp_backend_same_s_sp_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_sp_native_target_count)
      if (total_targets > 0L) {
        (as.integer(a$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count) +
           as.integer(b$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count)) /
          total_targets
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_setup_native_group_count =
      as.integer(a$mgcv_cpp_backend_same_s_setup_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_setup_native_group_count),
    mgcv_cpp_backend_same_s_setup_native_target_count =
      as.integer(a$mgcv_cpp_backend_same_s_setup_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_setup_native_target_count),
    mgcv_cpp_backend_same_s_setup_native_max_targets =
      max(as.integer(a$mgcv_cpp_backend_same_s_setup_native_max_targets),
          as.integer(b$mgcv_cpp_backend_same_s_setup_native_max_targets)),
    mgcv_cpp_backend_same_s_setup_native_mean_targets = {
      total_groups <-
        as.integer(a$mgcv_cpp_backend_same_s_setup_native_group_count) +
        as.integer(b$mgcv_cpp_backend_same_s_setup_native_group_count)
      if (total_groups > 0L) {
        (as.numeric(a$mgcv_cpp_backend_same_s_setup_native_mean_targets) *
           as.integer(a$mgcv_cpp_backend_same_s_setup_native_group_count) +
           as.numeric(b$mgcv_cpp_backend_same_s_setup_native_mean_targets) *
             as.integer(b$mgcv_cpp_backend_same_s_setup_native_group_count)) /
          total_groups
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count =
      as.integer(a$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count) +
        as.integer(b$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count),
    mgcv_cpp_backend_same_s_setup_native_reuse_ratio = {
      total_targets <-
        as.integer(a$mgcv_cpp_backend_same_s_setup_native_target_count) +
        as.integer(b$mgcv_cpp_backend_same_s_setup_native_target_count)
      if (total_targets > 0L) {
        (as.integer(a$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count) +
           as.integer(b$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count)) /
          total_targets
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_setup_input_potential_saved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_setup_input_potential_saved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_setup_input_potential_saved_ms),
    mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms),
    mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms),
    mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms),
    mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms),
    mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio = {
      total_solve <- as.numeric(a$mgcv_cpp_backend_native_solve_ms) +
        as.numeric(b$mgcv_cpp_backend_native_solve_ms)
      if (total_solve > 0) {
        (as.numeric(a$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms) +
           as.numeric(b$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms)) /
          total_solve
      } else {
        0
      }
    },
    mgcv_cpp_backend_same_s_gam_fit_preserved_ms =
      as.numeric(a$mgcv_cpp_backend_same_s_gam_fit_preserved_ms) +
        as.numeric(b$mgcv_cpp_backend_same_s_gam_fit_preserved_ms),
    mgcv_cpp_same_s_prefill_enabled =
      max(as.integer(a$mgcv_cpp_same_s_prefill_enabled),
          as.integer(b$mgcv_cpp_same_s_prefill_enabled)),
    mgcv_cpp_same_s_prefill_group_count =
      as.integer(a$mgcv_cpp_same_s_prefill_group_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_group_count),
    mgcv_cpp_same_s_prefill_target_count =
      as.integer(a$mgcv_cpp_same_s_prefill_target_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_target_count),
    mgcv_cpp_same_s_prefill_cache_insert_count =
      as.integer(a$mgcv_cpp_same_s_prefill_cache_insert_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_cache_insert_count),
    mgcv_cpp_same_s_prefill_existing_count =
      as.integer(a$mgcv_cpp_same_s_prefill_existing_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_existing_count),
    mgcv_cpp_same_s_prefill_unused_count =
      as.integer(a$mgcv_cpp_same_s_prefill_unused_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_unused_count),
    mgcv_cpp_same_s_prefill_ms =
      as.numeric(a$mgcv_cpp_same_s_prefill_ms) +
        as.numeric(b$mgcv_cpp_same_s_prefill_ms),
    mgcv_cpp_same_s_prefill_error_count =
      as.integer(a$mgcv_cpp_same_s_prefill_error_count) +
        as.integer(b$mgcv_cpp_same_s_prefill_error_count),
    mgcv_cpp_same_s_setup_provider_enabled =
      max(as.integer(a$mgcv_cpp_same_s_setup_provider_enabled),
          as.integer(b$mgcv_cpp_same_s_setup_provider_enabled)),
    mgcv_cpp_same_s_setup_provider_group_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_group_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_group_count),
    mgcv_cpp_same_s_setup_provider_target_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_target_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_target_count),
    mgcv_cpp_same_s_setup_provider_template_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_template_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_template_count),
    mgcv_cpp_same_s_setup_provider_reuse_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_reuse_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_reuse_count),
    mgcv_cpp_same_s_setup_provider_setup_ms =
      as.numeric(a$mgcv_cpp_same_s_setup_provider_setup_ms) +
        as.numeric(b$mgcv_cpp_same_s_setup_provider_setup_ms),
    mgcv_cpp_same_s_setup_provider_error_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_error_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_error_count),
    mgcv_cpp_same_s_setup_provider_chunk_enabled =
      max(as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_enabled),
          as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_enabled)),
    mgcv_cpp_same_s_setup_provider_chunk_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_count),
    mgcv_cpp_same_s_setup_provider_chunk_group_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_group_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_group_count),
    mgcv_cpp_same_s_setup_provider_chunk_target_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_target_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_target_count),
    mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count),
    mgcv_cpp_same_s_setup_provider_chunk_ms =
      as.numeric(a$mgcv_cpp_same_s_setup_provider_chunk_ms) +
        as.numeric(b$mgcv_cpp_same_s_setup_provider_chunk_ms),
    mgcv_cpp_same_s_setup_provider_chunk_error_count =
      as.integer(a$mgcv_cpp_same_s_setup_provider_chunk_error_count) +
        as.integer(b$mgcv_cpp_same_s_setup_provider_chunk_error_count),
    mgcv_cpp_backend_native_s_size_limit =
      max(as.numeric(a$mgcv_cpp_backend_native_s_size_limit),
          as.numeric(b$mgcv_cpp_backend_native_s_size_limit)),
    mgcv_cpp_backend_condition_threshold =
      max(as.numeric(a$mgcv_cpp_backend_condition_threshold),
          as.numeric(b$mgcv_cpp_backend_condition_threshold)),
    mgcv_cpp_shadow_enabled =
      max(as.integer(a$mgcv_cpp_shadow_enabled),
          as.integer(b$mgcv_cpp_shadow_enabled)),
    mgcv_cpp_shadow_count =
      as.integer(a$mgcv_cpp_shadow_count) +
        as.integer(b$mgcv_cpp_shadow_count),
    mgcv_cpp_shadow_native_count =
      as.integer(a$mgcv_cpp_shadow_native_count) +
        as.integer(b$mgcv_cpp_shadow_native_count),
    mgcv_cpp_shadow_fallback_count =
      as.integer(a$mgcv_cpp_shadow_fallback_count) +
        as.integer(b$mgcv_cpp_shadow_fallback_count),
    mgcv_cpp_shadow_high_condition_fallback_count =
      as.integer(a$mgcv_cpp_shadow_high_condition_fallback_count) +
        as.integer(b$mgcv_cpp_shadow_high_condition_fallback_count),
    mgcv_cpp_shadow_outside_envelope_fallback_count =
      as.integer(a$mgcv_cpp_shadow_outside_envelope_fallback_count) +
        as.integer(b$mgcv_cpp_shadow_outside_envelope_fallback_count),
    mgcv_cpp_shadow_error_count =
      as.integer(a$mgcv_cpp_shadow_error_count) +
        as.integer(b$mgcv_cpp_shadow_error_count),
    mgcv_cpp_shadow_residual_mismatch_count =
      as.integer(a$mgcv_cpp_shadow_residual_mismatch_count) +
        as.integer(b$mgcv_cpp_shadow_residual_mismatch_count),
    mgcv_cpp_shadow_ms =
      as.numeric(a$mgcv_cpp_shadow_ms) +
        as.numeric(b$mgcv_cpp_shadow_ms),
    mgcv_cpp_shadow_max_abs_diff =
      max(as.numeric(a$mgcv_cpp_shadow_max_abs_diff),
          as.numeric(b$mgcv_cpp_shadow_max_abs_diff)),
    mgcv_cpp_shadow_max_rel_l2 =
      max(as.numeric(a$mgcv_cpp_shadow_max_rel_l2),
          as.numeric(b$mgcv_cpp_shadow_max_rel_l2)),
    mgcv_cpp_shadow_native_s_size_limit =
      max(as.numeric(a$mgcv_cpp_shadow_native_s_size_limit),
          as.numeric(b$mgcv_cpp_shadow_native_s_size_limit)),
    mgcv_cpp_shadow_condition_threshold =
      max(as.numeric(a$mgcv_cpp_shadow_condition_threshold),
          as.numeric(b$mgcv_cpp_shadow_condition_threshold)),
    mgcv_residual_keys = c(a$mgcv_residual_keys, b$mgcv_residual_keys),
    mgcv_s_keys = c(a$mgcv_s_keys, b$mgcv_s_keys),
    mgcv_residual_cache_hit_keys =
      c(a$mgcv_residual_cache_hit_keys, b$mgcv_residual_cache_hit_keys),
    mgcv_residual_cache_miss_keys =
      c(a$mgcv_residual_cache_miss_keys, b$mgcv_residual_cache_miss_keys),
    mgcv_cpp_backend_native_residual_keys =
      c(a$mgcv_cpp_backend_native_residual_keys,
        b$mgcv_cpp_backend_native_residual_keys),
    mgcv_cpp_backend_native_s_keys =
      c(a$mgcv_cpp_backend_native_s_keys, b$mgcv_cpp_backend_native_s_keys),
    mgcv_cpp_backend_native_s_sp_keys =
      c(a$mgcv_cpp_backend_native_s_sp_keys,
        b$mgcv_cpp_backend_native_s_sp_keys),
    mgcv_cpp_backend_native_s_setup_keys =
      c(a$mgcv_cpp_backend_native_s_setup_keys,
        b$mgcv_cpp_backend_native_s_setup_keys),
    mgcv_cpp_backend_native_input_setup_ms =
      c(a$mgcv_cpp_backend_native_input_setup_ms,
        b$mgcv_cpp_backend_native_input_setup_ms),
    mgcv_cpp_backend_native_gam_fit_ms =
      c(a$mgcv_cpp_backend_native_gam_fit_ms,
        b$mgcv_cpp_backend_native_gam_fit_ms),
    mgcv_cpp_backend_native_sp_extract_ms =
      c(a$mgcv_cpp_backend_native_sp_extract_ms,
        b$mgcv_cpp_backend_native_sp_extract_ms),
    mgcv_cpp_backend_native_setup_extract_ms =
      c(a$mgcv_cpp_backend_native_setup_extract_ms,
        b$mgcv_cpp_backend_native_setup_extract_ms),
    mgcv_cpp_backend_native_condition_ms =
      c(a$mgcv_cpp_backend_native_condition_ms,
        b$mgcv_cpp_backend_native_condition_ms),
    mgcv_cpp_backend_native_solve_call_ms =
      c(a$mgcv_cpp_backend_native_solve_call_ms,
        b$mgcv_cpp_backend_native_solve_call_ms),
    direct_ci_count =
      as.integer(a$direct_ci_count) + as.integer(b$direct_ci_count),
    conditional_ci_count =
      as.integer(a$conditional_ci_count) +
        as.integer(b$conditional_ci_count),
    mgcv_fit_count =
      as.integer(a$mgcv_fit_count) + as.integer(b$mgcv_fit_count),
    dcov_gamma_count =
      as.integer(a$dcov_gamma_count) + as.integer(b$dcov_gamma_count),
    fake_level0_test_count =
      as.integer(a$fake_level0_test_count) +
        as.integer(b$fake_level0_test_count)
  )
}

fastkpc_legacy_runtime_frame <- function(level_metrics, n_edgetests) {
  level_count <- length(n_edgetests)
  if (level_count == 0L) {
    return(data.frame(
      level = integer(), recorded_tests = integer(), ci_calls = integer(),
      direct_ci_calls = integer(), conditional_ci_calls = integer(),
      fake_level0_tests = integer(), residual_ms = numeric(),
      dcov_gamma_ms = numeric(), ci_total_ms = numeric(),
      dcov_distance_ms = numeric(), dcov_lowrank_ms = numeric(),
      dcov_statistic_ms = numeric(), dcov_moment_ms = numeric(),
      dcov_pgamma_ms = numeric(),
      dcov_r_backend_count = integer(), dcov_cpp_backend_count = integer(),
      dcov_cpp_backend_ms = numeric(),
      dcov_cpp_backend_error_count = integer(),
      dcov_cpp_backend_fallback_count = integer(),
      dcov_cpp_backend_decision_flip_count = integer(),
      dcov_cpp_shadow_ms = numeric(), dcov_cpp_shadow_count = integer(),
      dcov_cpp_shadow_decision_flip_count = integer(),
      dcov_cpp_shadow_error_count = integer(),
      dcov_cpp_distance_ms = numeric(), dcov_cpp_lowrank_ms = numeric(),
      dcov_cpp_lowrank_eig_ms = numeric(),
      dcov_cpp_lowrank_select_ms = numeric(),
      dcov_cpp_lowrank_center_ms = numeric(),
      dcov_cpp_lowrank_unaccounted_ms = numeric(),
      dcov_cpp_lowrank_full_eig_count = integer(),
      dcov_cpp_lowrank_spectra_count = integer(),
      dcov_cpp_lowrank_spectra_converged_count = integer(),
      dcov_cpp_lowrank_spectra_failed_count = integer(),
      dcov_cpp_lowrank_spectra_fallback_full_eig_count = integer(),
      dcov_cpp_lowrank_spectra_iterations = integer(),
      dcov_cpp_lowrank_spectra_nconv = integer(),
      dcov_cpp_lowrank_spectra_ncv = integer(),
      dcov_cpp_lowrank_spectra_tol = numeric(),
      dcov_cpp_statistic_ms = numeric(), dcov_cpp_moment_ms = numeric(),
      dcov_cpp_pgamma_ms = numeric(), dcov_cpp_overhead_ms = numeric(),
      mgcv_residual_request_count = integer(),
      mgcv_cache_hit_count = integer(), mgcv_cache_miss_count = integer(),
      mgcv_residual_cache_hit_key_count = integer(),
      mgcv_residual_cache_miss_key_count = integer(),
      mgcv_residual_cache_miss_s_group_count = integer(),
      mgcv_residual_cache_miss_s_total_targets = integer(),
      mgcv_residual_cache_miss_s_max_targets = integer(),
      mgcv_residual_cache_miss_s_mean_targets = numeric(),
      mgcv_residual_cache_miss_s_reuse_opportunity_count = integer(),
      mgcv_residual_cache_miss_s_reuse_ratio = numeric(),
      mgcv_unique_residual_key_count = integer(),
      mgcv_duplicate_residual_key_count = integer(),
      mgcv_unique_target_s_count = integer(), mgcv_unique_s_count = integer(),
      mgcv_same_s_group_count = integer(),
      mgcv_same_s_total_targets = integer(),
      mgcv_same_s_max_targets = integer(),
      mgcv_same_s_mean_targets = numeric(),
      mgcv_same_s_reuse_opportunity_count = integer(),
      mgcv_residual_cache_insert_count = integer(),
      mgcv_residual_cache_entries = integer(),
      mgcv_residual_cache_hit_ms = numeric(),
      mgcv_residual_cache_store_ms = numeric(),
      mgcv_fit_avoided_count = integer(),
      mgcv_residual_affinity_enabled = integer(),
      mgcv_residual_affinity_group_count = integer(),
      mgcv_residual_affinity_task_count = integer(),
      mgcv_residual_affinity_worker_count = integer(),
      mgcv_residual_affinity_max_group_size = integer(),
      mgcv_residual_affinity_mean_group_size = numeric(),
      mgcv_residual_affinity_load_imbalance = numeric(),
      mgcv_residual_affinity_split_group_count = integer(),
      mgcv_residual_affinity_split_group_tasks = integer(),
      mgcv_residual_affinity_split_group_pieces = integer(),
      mgcv_residual_cache_theoretical_hit_count = integer(),
      mgcv_residual_cache_realized_hit_count = integer(),
      mgcv_residual_cache_lost_duplicate_count = integer(),
      mgcv_residual_cache_lost_cross_worker_count = integer(),
      mgcv_residual_cache_lost_split_s_group_count = integer(),
      mgcv_residual_cache_lost_cross_level_count = integer(),
      mgcv_residual_owner_enabled = integer(),
      mgcv_residual_owner_key_count = integer(),
      mgcv_residual_owner_task_count = integer(),
      mgcv_residual_owner_both_local_count = integer(),
      mgcv_residual_owner_one_local_count = integer(),
      mgcv_residual_owner_none_local_count = integer(),
      mgcv_residual_owner_conflict_count = integer(),
      mgcv_residual_owner_predicted_hit_count = integer(),
      mgcv_residual_owner_realized_hit_count = integer(),
      mgcv_residual_owner_lost_duplicate_count = integer(),
      mgcv_residual_owner_load_imbalance = numeric(),
      mgcv_residual_owner_spill_count = integer(),
      mgcv_residual_owner_schedule_build_ms = numeric(),
      mgcv_residual_owner_key_enum_ms = numeric(),
      mgcv_residual_owner_key_map_build_ms = numeric(),
      mgcv_residual_owner_task_score_ms = numeric(),
      mgcv_residual_owner_greedy_assign_ms = numeric(),
      mgcv_residual_owner_chunk_sort_ms = numeric(),
      mgcv_residual_owner_chunk_materialize_ms = numeric(),
      mgcv_residual_owner_worker_max_ms = numeric(),
      mgcv_residual_owner_worker_median_ms = numeric(),
      mgcv_residual_owner_worker_elapsed_imbalance = numeric(),
      mgcv_residual_owner_worker_task_max = integer(),
      mgcv_residual_owner_worker_task_median = numeric(),
      mgcv_residual_owner_worker_fit_max = integer(),
      mgcv_residual_owner_worker_fit_median = numeric(),
      mgcv_residual_owner_worker_cache_hit_max = integer(),
      mgcv_residual_owner_worker_cache_hit_median = numeric(),
      mgcv_residual_owner_worker_residual_ms_max = numeric(),
      mgcv_residual_owner_worker_residual_ms_median = numeric(),
      mgcv_prefetch_enabled = integer(),
      mgcv_prefetch_level_count = integer(),
      mgcv_prefetch_key_count = integer(),
      mgcv_prefetch_fit_count = integer(),
      mgcv_prefetch_fit_ms = numeric(),
      mgcv_prefetch_collect_ms = numeric(),
      mgcv_prefetch_matrix_build_ms = numeric(),
      mgcv_prefetch_payload_bytes = numeric(),
      mgcv_prefetch_max_level_payload_bytes = numeric(),
      mgcv_prefetch_lookup_ms = numeric(),
      mgcv_prefetch_ci_phase_ms = numeric(),
      mgcv_prefetch_error_count = integer(),
      mgcv_prefetch_consumed_key_count = integer(),
      mgcv_prefetch_unused_key_count = integer(),
      mgcv_key_build_ms = numeric(), mgcv_cache_lookup_ms = numeric(),
      mgcv_formula_build_ms = numeric(), mgcv_data_subset_ms = numeric(),
      mgcv_fit_call_ms = numeric(), mgcv_residual_extract_ms = numeric(),
      mgcv_result_store_ms = numeric(), mgcv_unaccounted_ms = numeric(),
      mgcv_s_size_0_count = integer(), mgcv_s_size_1_count = integer(),
      mgcv_s_size_2_count = integer(), mgcv_s_size_gt2_count = integer(),
      mgcv_cpp_backend_input_setup_ms = numeric(),
      mgcv_cpp_backend_gam_fit_ms = numeric(),
      mgcv_cpp_backend_sp_extract_ms = numeric(),
      mgcv_cpp_backend_setup_extract_ms = numeric(),
      mgcv_cpp_backend_condition_ms = numeric(),
      mgcv_cpp_backend_native_solve_ms = numeric(),
      mgcv_cpp_backend_fallback_ms = numeric(),
      mgcv_cpp_backend_s_size_0_count = integer(),
      mgcv_cpp_backend_s_size_1_count = integer(),
      mgcv_cpp_backend_s_size_2_count = integer(),
      mgcv_cpp_backend_s_size_gt2_count = integer(),
      mgcv_cpp_backend_native_s_size_0_count = integer(),
      mgcv_cpp_backend_native_s_size_1_count = integer(),
      mgcv_cpp_backend_native_s_size_2_count = integer(),
      mgcv_cpp_backend_native_s_size_gt2_count = integer(),
      mgcv_cpp_backend_fallback_s_size_0_count = integer(),
      mgcv_cpp_backend_fallback_s_size_1_count = integer(),
      mgcv_cpp_backend_fallback_s_size_2_count = integer(),
      mgcv_cpp_backend_fallback_s_size_gt2_count = integer(),
      mgcv_cpp_backend_same_s_native_group_count = integer(),
      mgcv_cpp_backend_same_s_native_target_count = integer(),
      mgcv_cpp_backend_same_s_native_max_targets = integer(),
      mgcv_cpp_backend_same_s_native_mean_targets = numeric(),
      mgcv_cpp_backend_same_s_native_reuse_opportunity_count = integer(),
      mgcv_cpp_backend_same_s_native_setup_reuse_ratio = numeric(),
      mgcv_cpp_backend_same_s_sp_native_group_count = integer(),
      mgcv_cpp_backend_same_s_sp_native_target_count = integer(),
      mgcv_cpp_backend_same_s_sp_native_max_targets = integer(),
      mgcv_cpp_backend_same_s_sp_native_mean_targets = numeric(),
      mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count = integer(),
      mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio = numeric(),
      mgcv_cpp_backend_same_s_setup_native_group_count = integer(),
      mgcv_cpp_backend_same_s_setup_native_target_count = integer(),
      mgcv_cpp_backend_same_s_setup_native_max_targets = integer(),
      mgcv_cpp_backend_same_s_setup_native_mean_targets = numeric(),
      mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count = integer(),
      mgcv_cpp_backend_same_s_setup_native_reuse_ratio = numeric(),
      mgcv_cpp_backend_same_s_setup_input_potential_saved_ms = numeric(),
      mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms = numeric(),
      mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms = numeric(),
      mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms = numeric(),
      mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms = numeric(),
      mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio = numeric(),
      mgcv_cpp_backend_same_s_gam_fit_preserved_ms = numeric(),
      mgcv_cpp_same_s_prefill_enabled = integer(),
      mgcv_cpp_same_s_prefill_group_count = integer(),
      mgcv_cpp_same_s_prefill_target_count = integer(),
      mgcv_cpp_same_s_prefill_cache_insert_count = integer(),
      mgcv_cpp_same_s_prefill_existing_count = integer(),
      mgcv_cpp_same_s_prefill_unused_count = integer(),
      mgcv_cpp_same_s_prefill_ms = numeric(),
      mgcv_cpp_same_s_prefill_error_count = integer(),
      mgcv_cpp_same_s_setup_provider_enabled = integer(),
      mgcv_cpp_same_s_setup_provider_group_count = integer(),
      mgcv_cpp_same_s_setup_provider_target_count = integer(),
      mgcv_cpp_same_s_setup_provider_template_count = integer(),
      mgcv_cpp_same_s_setup_provider_reuse_count = integer(),
      mgcv_cpp_same_s_setup_provider_setup_ms = numeric(),
      mgcv_cpp_same_s_setup_provider_error_count = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_enabled = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_count = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_group_count = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_target_count = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count = integer(),
      mgcv_cpp_same_s_setup_provider_chunk_ms = numeric(),
      mgcv_cpp_same_s_setup_provider_chunk_error_count = integer(),
      mgcv_fit_count = integer(), dcov_gamma_count = integer()
    ))
  }
  rows <- lapply(seq_len(level_count), function(i) {
    metrics <- level_metrics[[i]]
    if (is.null(metrics)) metrics <- fastkpc_legacy_runtime_zero()
    data.frame(
      level = i - 1L,
      recorded_tests = as.integer(n_edgetests[[i]]),
      ci_calls = as.integer(metrics$direct_ci_count) +
        as.integer(metrics$conditional_ci_count),
      direct_ci_calls = as.integer(metrics$direct_ci_count),
      conditional_ci_calls = as.integer(metrics$conditional_ci_count),
      fake_level0_tests = as.integer(metrics$fake_level0_test_count),
      residual_ms = as.numeric(metrics$residual_ms),
      dcov_gamma_ms = as.numeric(metrics$dcov_gamma_ms),
      ci_total_ms = as.numeric(metrics$ci_total_ms),
      dcov_distance_ms = as.numeric(metrics$dcov_distance_ms),
      dcov_lowrank_ms = as.numeric(metrics$dcov_lowrank_ms),
      dcov_statistic_ms = as.numeric(metrics$dcov_statistic_ms),
      dcov_moment_ms = as.numeric(metrics$dcov_moment_ms),
      dcov_pgamma_ms = as.numeric(metrics$dcov_pgamma_ms),
      dcov_r_backend_count = as.integer(metrics$dcov_r_backend_count),
      dcov_cpp_backend_count = as.integer(metrics$dcov_cpp_backend_count),
      dcov_cpp_backend_ms = as.numeric(metrics$dcov_cpp_backend_ms),
      dcov_cpp_backend_error_count =
        as.integer(metrics$dcov_cpp_backend_error_count),
      dcov_cpp_backend_fallback_count =
        as.integer(metrics$dcov_cpp_backend_fallback_count),
      dcov_cpp_backend_decision_flip_count =
        as.integer(metrics$dcov_cpp_backend_decision_flip_count),
      dcov_cpp_shadow_ms = as.numeric(metrics$dcov_cpp_shadow_ms),
      dcov_cpp_shadow_count = as.integer(metrics$dcov_cpp_shadow_count),
      dcov_cpp_shadow_decision_flip_count =
        as.integer(metrics$dcov_cpp_shadow_decision_flip_count),
      dcov_cpp_shadow_error_count =
        as.integer(metrics$dcov_cpp_shadow_error_count),
      dcov_cpp_distance_ms = as.numeric(metrics$dcov_cpp_distance_ms),
      dcov_cpp_lowrank_ms = as.numeric(metrics$dcov_cpp_lowrank_ms),
      dcov_cpp_lowrank_eig_ms =
        as.numeric(metrics$dcov_cpp_lowrank_eig_ms),
      dcov_cpp_lowrank_select_ms =
        as.numeric(metrics$dcov_cpp_lowrank_select_ms),
      dcov_cpp_lowrank_center_ms =
        as.numeric(metrics$dcov_cpp_lowrank_center_ms),
      dcov_cpp_lowrank_unaccounted_ms =
        as.numeric(metrics$dcov_cpp_lowrank_unaccounted_ms),
      dcov_cpp_lowrank_full_eig_count =
        as.integer(metrics$dcov_cpp_lowrank_full_eig_count),
      dcov_cpp_lowrank_spectra_count =
        as.integer(metrics$dcov_cpp_lowrank_spectra_count),
      dcov_cpp_lowrank_spectra_converged_count =
        as.integer(metrics$dcov_cpp_lowrank_spectra_converged_count),
      dcov_cpp_lowrank_spectra_failed_count =
        as.integer(metrics$dcov_cpp_lowrank_spectra_failed_count),
      dcov_cpp_lowrank_spectra_fallback_full_eig_count =
        as.integer(metrics$dcov_cpp_lowrank_spectra_fallback_full_eig_count),
      dcov_cpp_lowrank_spectra_iterations =
        as.integer(metrics$dcov_cpp_lowrank_spectra_iterations),
      dcov_cpp_lowrank_spectra_nconv =
        as.integer(metrics$dcov_cpp_lowrank_spectra_nconv),
      dcov_cpp_lowrank_spectra_ncv =
        as.integer(metrics$dcov_cpp_lowrank_spectra_ncv),
      dcov_cpp_lowrank_spectra_tol =
        as.numeric(metrics$dcov_cpp_lowrank_spectra_tol),
      dcov_cpp_statistic_ms = as.numeric(metrics$dcov_cpp_statistic_ms),
      dcov_cpp_moment_ms = as.numeric(metrics$dcov_cpp_moment_ms),
      dcov_cpp_pgamma_ms = as.numeric(metrics$dcov_cpp_pgamma_ms),
      dcov_cpp_overhead_ms = as.numeric(metrics$dcov_cpp_overhead_ms),
      mgcv_residual_request_count =
        as.integer(metrics$mgcv_residual_request_count),
      mgcv_cache_hit_count = as.integer(metrics$mgcv_cache_hit_count),
      mgcv_cache_miss_count = as.integer(metrics$mgcv_cache_miss_count),
      mgcv_residual_cache_hit_key_count =
        as.integer(metrics$mgcv_residual_cache_hit_key_count),
      mgcv_residual_cache_miss_key_count =
        as.integer(metrics$mgcv_residual_cache_miss_key_count),
      mgcv_residual_cache_miss_s_group_count =
        as.integer(metrics$mgcv_residual_cache_miss_s_group_count),
      mgcv_residual_cache_miss_s_total_targets =
        as.integer(metrics$mgcv_residual_cache_miss_s_total_targets),
      mgcv_residual_cache_miss_s_max_targets =
        as.integer(metrics$mgcv_residual_cache_miss_s_max_targets),
      mgcv_residual_cache_miss_s_mean_targets =
        as.numeric(metrics$mgcv_residual_cache_miss_s_mean_targets),
      mgcv_residual_cache_miss_s_reuse_opportunity_count =
        as.integer(metrics$mgcv_residual_cache_miss_s_reuse_opportunity_count),
      mgcv_residual_cache_miss_s_reuse_ratio =
        as.numeric(metrics$mgcv_residual_cache_miss_s_reuse_ratio),
      mgcv_unique_residual_key_count =
        as.integer(metrics$mgcv_unique_residual_key_count),
      mgcv_duplicate_residual_key_count =
        as.integer(metrics$mgcv_duplicate_residual_key_count),
      mgcv_unique_target_s_count =
        as.integer(metrics$mgcv_unique_target_s_count),
      mgcv_unique_s_count = as.integer(metrics$mgcv_unique_s_count),
      mgcv_same_s_group_count =
        as.integer(metrics$mgcv_same_s_group_count),
      mgcv_same_s_total_targets =
        as.integer(metrics$mgcv_same_s_total_targets),
      mgcv_same_s_max_targets =
        as.integer(metrics$mgcv_same_s_max_targets),
      mgcv_same_s_mean_targets =
        as.numeric(metrics$mgcv_same_s_mean_targets),
      mgcv_same_s_reuse_opportunity_count =
        as.integer(metrics$mgcv_same_s_reuse_opportunity_count),
      mgcv_residual_cache_insert_count =
        as.integer(metrics$mgcv_residual_cache_insert_count),
      mgcv_residual_cache_entries =
        as.integer(metrics$mgcv_residual_cache_entries),
      mgcv_residual_cache_hit_ms =
        as.numeric(metrics$mgcv_residual_cache_hit_ms),
      mgcv_residual_cache_store_ms =
        as.numeric(metrics$mgcv_residual_cache_store_ms),
      mgcv_fit_avoided_count = as.integer(metrics$mgcv_fit_avoided_count),
      mgcv_residual_affinity_enabled =
        as.integer(metrics$mgcv_residual_affinity_enabled),
      mgcv_residual_affinity_group_count =
        as.integer(metrics$mgcv_residual_affinity_group_count),
      mgcv_residual_affinity_task_count =
        as.integer(metrics$mgcv_residual_affinity_task_count),
      mgcv_residual_affinity_worker_count =
        as.integer(metrics$mgcv_residual_affinity_worker_count),
      mgcv_residual_affinity_max_group_size =
        as.integer(metrics$mgcv_residual_affinity_max_group_size),
      mgcv_residual_affinity_mean_group_size =
        as.numeric(metrics$mgcv_residual_affinity_mean_group_size),
      mgcv_residual_affinity_load_imbalance =
        as.numeric(metrics$mgcv_residual_affinity_load_imbalance),
      mgcv_residual_affinity_split_group_count =
        as.integer(metrics$mgcv_residual_affinity_split_group_count),
      mgcv_residual_affinity_split_group_tasks =
        as.integer(metrics$mgcv_residual_affinity_split_group_tasks),
      mgcv_residual_affinity_split_group_pieces =
        as.integer(metrics$mgcv_residual_affinity_split_group_pieces),
      mgcv_residual_cache_theoretical_hit_count =
        as.integer(metrics$mgcv_residual_cache_theoretical_hit_count),
      mgcv_residual_cache_realized_hit_count =
        as.integer(metrics$mgcv_residual_cache_realized_hit_count),
      mgcv_residual_cache_lost_duplicate_count =
        as.integer(metrics$mgcv_residual_cache_lost_duplicate_count),
      mgcv_residual_cache_lost_cross_worker_count =
        as.integer(metrics$mgcv_residual_cache_lost_cross_worker_count),
      mgcv_residual_cache_lost_split_s_group_count =
        as.integer(metrics$mgcv_residual_cache_lost_split_s_group_count),
      mgcv_residual_cache_lost_cross_level_count =
        as.integer(metrics$mgcv_residual_cache_lost_cross_level_count),
      mgcv_residual_owner_enabled =
        as.integer(metrics$mgcv_residual_owner_enabled),
      mgcv_residual_owner_key_count =
        as.integer(metrics$mgcv_residual_owner_key_count),
      mgcv_residual_owner_task_count =
        as.integer(metrics$mgcv_residual_owner_task_count),
      mgcv_residual_owner_both_local_count =
        as.integer(metrics$mgcv_residual_owner_both_local_count),
      mgcv_residual_owner_one_local_count =
        as.integer(metrics$mgcv_residual_owner_one_local_count),
      mgcv_residual_owner_none_local_count =
        as.integer(metrics$mgcv_residual_owner_none_local_count),
      mgcv_residual_owner_conflict_count =
        as.integer(metrics$mgcv_residual_owner_conflict_count),
      mgcv_residual_owner_predicted_hit_count =
        as.integer(metrics$mgcv_residual_owner_predicted_hit_count),
      mgcv_residual_owner_realized_hit_count =
        as.integer(metrics$mgcv_residual_owner_realized_hit_count),
      mgcv_residual_owner_lost_duplicate_count =
        as.integer(metrics$mgcv_residual_owner_lost_duplicate_count),
      mgcv_residual_owner_load_imbalance =
        as.numeric(metrics$mgcv_residual_owner_load_imbalance),
      mgcv_residual_owner_spill_count =
        as.integer(metrics$mgcv_residual_owner_spill_count),
      mgcv_residual_owner_schedule_build_ms =
        as.numeric(metrics$mgcv_residual_owner_schedule_build_ms),
      mgcv_residual_owner_key_enum_ms =
        as.numeric(metrics$mgcv_residual_owner_key_enum_ms),
      mgcv_residual_owner_key_map_build_ms =
        as.numeric(metrics$mgcv_residual_owner_key_map_build_ms),
      mgcv_residual_owner_task_score_ms =
        as.numeric(metrics$mgcv_residual_owner_task_score_ms),
      mgcv_residual_owner_greedy_assign_ms =
        as.numeric(metrics$mgcv_residual_owner_greedy_assign_ms),
      mgcv_residual_owner_chunk_sort_ms =
        as.numeric(metrics$mgcv_residual_owner_chunk_sort_ms),
      mgcv_residual_owner_chunk_materialize_ms =
        as.numeric(metrics$mgcv_residual_owner_chunk_materialize_ms),
      mgcv_residual_owner_worker_max_ms =
        as.numeric(metrics$mgcv_residual_owner_worker_max_ms),
      mgcv_residual_owner_worker_median_ms =
        as.numeric(metrics$mgcv_residual_owner_worker_median_ms),
      mgcv_residual_owner_worker_elapsed_imbalance =
        as.numeric(metrics$mgcv_residual_owner_worker_elapsed_imbalance),
      mgcv_residual_owner_worker_task_max =
        as.integer(metrics$mgcv_residual_owner_worker_task_max),
      mgcv_residual_owner_worker_task_median =
        as.numeric(metrics$mgcv_residual_owner_worker_task_median),
      mgcv_residual_owner_worker_fit_max =
        as.integer(metrics$mgcv_residual_owner_worker_fit_max),
      mgcv_residual_owner_worker_fit_median =
        as.numeric(metrics$mgcv_residual_owner_worker_fit_median),
      mgcv_residual_owner_worker_cache_hit_max =
        as.integer(metrics$mgcv_residual_owner_worker_cache_hit_max),
      mgcv_residual_owner_worker_cache_hit_median =
        as.numeric(metrics$mgcv_residual_owner_worker_cache_hit_median),
      mgcv_residual_owner_worker_residual_ms_max =
        as.numeric(metrics$mgcv_residual_owner_worker_residual_ms_max),
      mgcv_residual_owner_worker_residual_ms_median =
        as.numeric(metrics$mgcv_residual_owner_worker_residual_ms_median),
      mgcv_prefetch_enabled = as.integer(metrics$mgcv_prefetch_enabled),
      mgcv_prefetch_level_count =
        as.integer(metrics$mgcv_prefetch_level_count),
      mgcv_prefetch_key_count =
        as.integer(metrics$mgcv_prefetch_key_count),
      mgcv_prefetch_fit_count =
        as.integer(metrics$mgcv_prefetch_fit_count),
      mgcv_prefetch_fit_ms = as.numeric(metrics$mgcv_prefetch_fit_ms),
      mgcv_prefetch_collect_ms =
        as.numeric(metrics$mgcv_prefetch_collect_ms),
      mgcv_prefetch_matrix_build_ms =
        as.numeric(metrics$mgcv_prefetch_matrix_build_ms),
      mgcv_prefetch_payload_bytes =
        as.numeric(metrics$mgcv_prefetch_payload_bytes),
      mgcv_prefetch_max_level_payload_bytes =
        as.numeric(metrics$mgcv_prefetch_max_level_payload_bytes),
      mgcv_prefetch_lookup_ms =
        as.numeric(metrics$mgcv_prefetch_lookup_ms),
      mgcv_prefetch_ci_phase_ms =
        as.numeric(metrics$mgcv_prefetch_ci_phase_ms),
      mgcv_prefetch_error_count =
        as.integer(metrics$mgcv_prefetch_error_count),
      mgcv_prefetch_consumed_key_count =
        as.integer(metrics$mgcv_prefetch_consumed_key_count),
      mgcv_prefetch_unused_key_count =
        as.integer(metrics$mgcv_prefetch_unused_key_count),
      mgcv_key_build_ms = as.numeric(metrics$mgcv_key_build_ms),
      mgcv_cache_lookup_ms = as.numeric(metrics$mgcv_cache_lookup_ms),
      mgcv_formula_build_ms = as.numeric(metrics$mgcv_formula_build_ms),
      mgcv_data_subset_ms = as.numeric(metrics$mgcv_data_subset_ms),
      mgcv_fit_call_ms = as.numeric(metrics$mgcv_fit_call_ms),
      mgcv_residual_extract_ms =
        as.numeric(metrics$mgcv_residual_extract_ms),
      mgcv_result_store_ms = as.numeric(metrics$mgcv_result_store_ms),
      mgcv_unaccounted_ms = as.numeric(metrics$mgcv_unaccounted_ms),
      mgcv_s_size_0_count = as.integer(metrics$mgcv_s_size_0_count),
      mgcv_s_size_1_count = as.integer(metrics$mgcv_s_size_1_count),
      mgcv_s_size_2_count = as.integer(metrics$mgcv_s_size_2_count),
      mgcv_s_size_gt2_count = as.integer(metrics$mgcv_s_size_gt2_count),
      mgcv_cpp_backend_input_setup_ms =
        as.numeric(metrics$mgcv_cpp_backend_input_setup_ms),
      mgcv_cpp_backend_gam_fit_ms =
        as.numeric(metrics$mgcv_cpp_backend_gam_fit_ms),
      mgcv_cpp_backend_sp_extract_ms =
        as.numeric(metrics$mgcv_cpp_backend_sp_extract_ms),
      mgcv_cpp_backend_setup_extract_ms =
        as.numeric(metrics$mgcv_cpp_backend_setup_extract_ms),
      mgcv_cpp_backend_condition_ms =
        as.numeric(metrics$mgcv_cpp_backend_condition_ms),
      mgcv_cpp_backend_native_solve_ms =
        as.numeric(metrics$mgcv_cpp_backend_native_solve_ms),
      mgcv_cpp_backend_fallback_ms =
        as.numeric(metrics$mgcv_cpp_backend_fallback_ms),
      mgcv_cpp_backend_s_size_0_count =
        as.integer(metrics$mgcv_cpp_backend_s_size_0_count),
      mgcv_cpp_backend_s_size_1_count =
        as.integer(metrics$mgcv_cpp_backend_s_size_1_count),
      mgcv_cpp_backend_s_size_2_count =
        as.integer(metrics$mgcv_cpp_backend_s_size_2_count),
      mgcv_cpp_backend_s_size_gt2_count =
        as.integer(metrics$mgcv_cpp_backend_s_size_gt2_count),
      mgcv_cpp_backend_native_s_size_0_count =
        as.integer(metrics$mgcv_cpp_backend_native_s_size_0_count),
      mgcv_cpp_backend_native_s_size_1_count =
        as.integer(metrics$mgcv_cpp_backend_native_s_size_1_count),
      mgcv_cpp_backend_native_s_size_2_count =
        as.integer(metrics$mgcv_cpp_backend_native_s_size_2_count),
      mgcv_cpp_backend_native_s_size_gt2_count =
        as.integer(metrics$mgcv_cpp_backend_native_s_size_gt2_count),
      mgcv_cpp_backend_fallback_s_size_0_count =
        as.integer(metrics$mgcv_cpp_backend_fallback_s_size_0_count),
      mgcv_cpp_backend_fallback_s_size_1_count =
        as.integer(metrics$mgcv_cpp_backend_fallback_s_size_1_count),
      mgcv_cpp_backend_fallback_s_size_2_count =
        as.integer(metrics$mgcv_cpp_backend_fallback_s_size_2_count),
      mgcv_cpp_backend_fallback_s_size_gt2_count =
        as.integer(metrics$mgcv_cpp_backend_fallback_s_size_gt2_count),
      mgcv_cpp_backend_same_s_native_group_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_native_group_count),
      mgcv_cpp_backend_same_s_native_target_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_native_target_count),
      mgcv_cpp_backend_same_s_native_max_targets =
        as.integer(metrics$mgcv_cpp_backend_same_s_native_max_targets),
      mgcv_cpp_backend_same_s_native_mean_targets =
        as.numeric(metrics$mgcv_cpp_backend_same_s_native_mean_targets),
      mgcv_cpp_backend_same_s_native_reuse_opportunity_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_native_reuse_opportunity_count),
      mgcv_cpp_backend_same_s_native_setup_reuse_ratio =
        as.numeric(metrics$mgcv_cpp_backend_same_s_native_setup_reuse_ratio),
      mgcv_cpp_backend_same_s_sp_native_group_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_sp_native_group_count),
      mgcv_cpp_backend_same_s_sp_native_target_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_sp_native_target_count),
      mgcv_cpp_backend_same_s_sp_native_max_targets =
        as.integer(metrics$mgcv_cpp_backend_same_s_sp_native_max_targets),
      mgcv_cpp_backend_same_s_sp_native_mean_targets =
        as.numeric(metrics$mgcv_cpp_backend_same_s_sp_native_mean_targets),
      mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count),
      mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio =
        as.numeric(metrics$mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio),
      mgcv_cpp_backend_same_s_setup_native_group_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_setup_native_group_count),
      mgcv_cpp_backend_same_s_setup_native_target_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_setup_native_target_count),
      mgcv_cpp_backend_same_s_setup_native_max_targets =
        as.integer(metrics$mgcv_cpp_backend_same_s_setup_native_max_targets),
      mgcv_cpp_backend_same_s_setup_native_mean_targets =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_native_mean_targets),
      mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count =
        as.integer(metrics$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count),
      mgcv_cpp_backend_same_s_setup_native_reuse_ratio =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_native_reuse_ratio),
      mgcv_cpp_backend_same_s_setup_input_potential_saved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_input_potential_saved_ms),
      mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms),
      mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms),
      mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms),
      mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms),
      mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio =
        as.numeric(metrics$mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio),
      mgcv_cpp_backend_same_s_gam_fit_preserved_ms =
        as.numeric(metrics$mgcv_cpp_backend_same_s_gam_fit_preserved_ms),
      mgcv_cpp_same_s_prefill_enabled =
        as.integer(metrics$mgcv_cpp_same_s_prefill_enabled),
      mgcv_cpp_same_s_prefill_group_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_group_count),
      mgcv_cpp_same_s_prefill_target_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_target_count),
      mgcv_cpp_same_s_prefill_cache_insert_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_cache_insert_count),
      mgcv_cpp_same_s_prefill_existing_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_existing_count),
      mgcv_cpp_same_s_prefill_unused_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_unused_count),
      mgcv_cpp_same_s_prefill_ms =
        as.numeric(metrics$mgcv_cpp_same_s_prefill_ms),
      mgcv_cpp_same_s_prefill_error_count =
        as.integer(metrics$mgcv_cpp_same_s_prefill_error_count),
      mgcv_cpp_same_s_setup_provider_enabled =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_enabled),
      mgcv_cpp_same_s_setup_provider_group_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_group_count),
      mgcv_cpp_same_s_setup_provider_target_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_target_count),
      mgcv_cpp_same_s_setup_provider_template_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_template_count),
      mgcv_cpp_same_s_setup_provider_reuse_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_reuse_count),
      mgcv_cpp_same_s_setup_provider_setup_ms =
        as.numeric(metrics$mgcv_cpp_same_s_setup_provider_setup_ms),
      mgcv_cpp_same_s_setup_provider_error_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_error_count),
      mgcv_cpp_same_s_setup_provider_chunk_enabled =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_enabled),
      mgcv_cpp_same_s_setup_provider_chunk_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_count),
      mgcv_cpp_same_s_setup_provider_chunk_group_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_group_count),
      mgcv_cpp_same_s_setup_provider_chunk_target_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_target_count),
      mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count),
      mgcv_cpp_same_s_setup_provider_chunk_ms =
        as.numeric(metrics$mgcv_cpp_same_s_setup_provider_chunk_ms),
      mgcv_cpp_same_s_setup_provider_chunk_error_count =
        as.integer(metrics$mgcv_cpp_same_s_setup_provider_chunk_error_count),
      mgcv_fit_count = as.integer(metrics$mgcv_fit_count),
      dcov_gamma_count = as.integer(metrics$dcov_gamma_count)
    )
  })
  do.call(rbind, rows)
}

fastkpc_legacy_mgcv_prefetch_potential_frame <- function(runtime_by_level, n) {
  if (!is.data.frame(runtime_by_level) || nrow(runtime_by_level) == 0L) {
    return(data.frame(
      level = integer(),
      request_count = integer(),
      current_fit_count = integer(),
      current_hit_count = integer(),
      unique_target_s_count = integer(),
      theoretical_fit_count = integer(),
      theoretical_hit_count = integer(),
      lost_duplicate_count = integer(),
      fit_reduction_potential = integer(),
      residual_payload_bytes = numeric(),
      unique_s_count = integer(),
      task_count = integer()
    ))
  }
  column_or_zero <- function(name) {
    if (name %in% names(runtime_by_level)) {
      runtime_by_level[[name]]
    } else {
      rep(0L, nrow(runtime_by_level))
    }
  }
  request_count <- as.integer(column_or_zero("mgcv_residual_request_count"))
  current_fit_count <- as.integer(column_or_zero("mgcv_fit_count"))
  current_hit_count <- as.integer(column_or_zero("mgcv_cache_hit_count"))
  unique_target_s_count <- as.integer(column_or_zero("mgcv_unique_target_s_count"))
  theoretical_hit_count <- as.integer(pmax(0L, request_count - unique_target_s_count))
  fit_reduction_potential <- as.integer(pmax(
    0L,
    current_fit_count - unique_target_s_count
  ))
  data.frame(
    level = as.integer(column_or_zero("level")),
    request_count = request_count,
    current_fit_count = current_fit_count,
    current_hit_count = current_hit_count,
    unique_target_s_count = unique_target_s_count,
    theoretical_fit_count = unique_target_s_count,
    theoretical_hit_count = theoretical_hit_count,
    lost_duplicate_count = as.integer(column_or_zero(
      "mgcv_residual_cache_lost_duplicate_count"
    )),
    fit_reduction_potential = fit_reduction_potential,
    residual_payload_bytes = as.numeric(unique_target_s_count) *
      as.numeric(n) * 8,
    unique_s_count = as.integer(column_or_zero("mgcv_unique_s_count")),
    task_count = as.integer(column_or_zero("recorded_tests"))
  )
}

fastkpc_legacy_mgcv_prefetch_runtime_frame <- function(runtime_by_level) {
  if (!is.data.frame(runtime_by_level) || nrow(runtime_by_level) == 0L) {
    return(data.frame(
      level = integer(),
      task_count = integer(),
      residual_request_count = integer(),
      unique_key_count = integer(),
      fit_count = integer(),
      payload_bytes = numeric(),
      prefetch_fit_ms = numeric(),
      prefetch_collect_ms = numeric(),
      matrix_build_ms = numeric(),
      ci_phase_ms = numeric(),
      elapsed_ms = numeric()
    ))
  }
  column_or_zero <- function(name) {
    if (name %in% names(runtime_by_level)) {
      runtime_by_level[[name]]
    } else {
      rep(0L, nrow(runtime_by_level))
    }
  }
  data.frame(
    level = as.integer(column_or_zero("level")),
    task_count = as.integer(column_or_zero("recorded_tests")),
    residual_request_count =
      as.integer(column_or_zero("mgcv_residual_request_count")),
    unique_key_count = as.integer(column_or_zero("mgcv_prefetch_key_count")),
    fit_count = as.integer(column_or_zero("mgcv_prefetch_fit_count")),
    payload_bytes = as.numeric(column_or_zero("mgcv_prefetch_payload_bytes")),
    prefetch_fit_ms = as.numeric(column_or_zero("mgcv_prefetch_fit_ms")),
    prefetch_collect_ms =
      as.numeric(column_or_zero("mgcv_prefetch_collect_ms")),
    matrix_build_ms =
      as.numeric(column_or_zero("mgcv_prefetch_matrix_build_ms")),
    ci_phase_ms = as.numeric(column_or_zero("mgcv_prefetch_ci_phase_ms")),
    elapsed_ms = as.numeric(column_or_zero("mgcv_prefetch_collect_ms")) +
      as.numeric(column_or_zero("mgcv_prefetch_matrix_build_ms")) +
      as.numeric(column_or_zero("mgcv_prefetch_ci_phase_ms"))
  )
}

fastkpc_legacy_runtime_add_dcov <- function(metrics, diagnostics) {
  metrics$dcov_gamma_ms <- metrics$dcov_gamma_ms +
    as.numeric(diagnostics$total_ms)
  metrics$dcov_input_ms <- metrics$dcov_input_ms +
    as.numeric(diagnostics$input_ms)
  metrics$dcov_h_ms <- metrics$dcov_h_ms +
    as.numeric(diagnostics$h_ms)
  metrics$dcov_distance_ms <- metrics$dcov_distance_ms +
    as.numeric(diagnostics$distance_ms)
  metrics$dcov_lowrank_ms <- metrics$dcov_lowrank_ms +
    as.numeric(diagnostics$lowrank_ms)
  metrics$dcov_statistic_ms <- metrics$dcov_statistic_ms +
    as.numeric(diagnostics$statistic_ms)
  metrics$dcov_moment_ms <- metrics$dcov_moment_ms +
    as.numeric(diagnostics$moment_ms)
  metrics$dcov_pgamma_ms <- metrics$dcov_pgamma_ms +
    as.numeric(diagnostics$pgamma_ms)
  metrics$dcov_output_ms <- metrics$dcov_output_ms +
    as.numeric(diagnostics$output_ms)
  metrics$dcov_unaccounted_ms <- metrics$dcov_unaccounted_ms +
    as.numeric(diagnostics$unaccounted_ms)
  metrics
}

fastkpc_legacy_mgcv_s_key <- function(S) {
  S <- as.integer(S)
  if (length(S) == 0L) return("")
  paste(S, collapse = "|")
}

fastkpc_legacy_mgcv_residual_key <- function(target, S) {
  paste0(as.integer(target), ":", fastkpc_legacy_mgcv_s_key(S))
}

fastkpc_legacy_mgcv_sp_key <- function(sp) {
  sp <- as.numeric(sp)
  if (length(sp) == 0L) return("")
  paste(formatC(signif(sp, digits = 14L), digits = 14L, format = "fg"),
        collapse = "|")
}

fastkpc_legacy_mgcv_setup_structure_key <- function(setup) {
  if (!exists("fastkpc_hash_object", mode = "function")) {
    source("fastkpc/R/mgcv_extract_oracle.R")
  }
  penalty_keys <- if (length(setup$S) > 0L) {
    vapply(setup$S, fastkpc_hash_object, character(1L))
  } else {
    character()
  }
  paste(
    fastkpc_hash_object(round(as.numeric(setup$X), digits = 14L)),
    paste(penalty_keys, collapse = "|"),
    fastkpc_hash_object(setup$C),
    paste(as.integer(setup$rank), collapse = "|"),
    sep = "||"
  )
}

fastkpc_legacy_group_reuse_potential_ms <- function(group_keys, values) {
  group_keys <- as.character(group_keys)
  values <- as.numeric(values)
  len <- min(length(group_keys), length(values))
  if (len == 0L) return(0)
  group_keys <- group_keys[seq_len(len)]
  values <- values[seq_len(len)]
  keep <- nzchar(group_keys) & is.finite(values) & values >= 0
  if (!any(keep)) return(0)
  groups <- split(values[keep], group_keys[keep], drop = TRUE)
  sum(vapply(groups, function(x) {
    if (length(x) <= 1L) return(0)
    max(0, sum(x) - min(x))
  }, numeric(1L)))
}

fastkpc_legacy_mgcv_residual_cache_enabled <- function() {
  identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE", unset = ""), "1")
}

fastkpc_legacy_mgcv_residual_affinity_mode <- function() {
  tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY", unset = ""))
}

fastkpc_legacy_mgcv_residual_prefetch_mode <- function() {
  tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH", unset = ""))
}

fastkpc_legacy_mgcv_residual_same_s_prefill_enabled <- function() {
  identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL",
                       unset = ""), "1")
}

fastkpc_legacy_mgcv_residual_same_s_setup_enabled <- function() {
  identical(fastkpc_legacy_mgcv_residual_same_s_setup_mode(), "prefill")
}

fastkpc_legacy_mgcv_residual_same_s_setup_mode <- function() {
  raw <- tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP",
                            unset = ""))
  if (raw %in% c("1", "true", "yes", "on")) {
    "prefill"
  } else if (identical(raw, "consumed")) {
    "consumed"
  } else if (raw %in% c("chunk", "miss_chunk", "batched")) {
    "chunk"
  } else {
    "off"
  }
}

fastkpc_legacy_mgcv_residual_cpp_shadow_enabled <- function() {
  identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW",
                       unset = ""), "1")
}

fastkpc_legacy_mgcv_residual_cpp_shadow_condition_threshold <- function() {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    "FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_CONDITION_THRESHOLD",
    unset = "1e12"
  )))
  if (length(value) != 1L || !is.finite(value) || value < 0) 1e12 else value
}

fastkpc_legacy_mgcv_residual_cpp_shadow_native_s_size_limit <- function() {
  raw <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CPP_SHADOW_NATIVE_S_SIZE_LIMIT",
                    unset = "Inf")
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || is.na(value) || value < 0) Inf else value
}

fastkpc_legacy_mgcv_residual_backend <- function() {
  raw <- tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND",
                            unset = "r"))
  if (raw %in% c("", "legacy", "r")) {
    "r"
  } else if (identical(raw, "cpp_guarded")) {
    "cpp_guarded"
  } else {
    "r"
  }
}

fastkpc_legacy_mgcv_residual_backend_condition_threshold <- function() {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD",
    unset = "1e12"
  )))
  if (length(value) != 1L || !is.finite(value) || value < 0) 1e12 else value
}

fastkpc_legacy_mgcv_residual_backend_native_s_size_limit <- function() {
  raw <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
                    unset = "Inf")
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || is.na(value) || value < 0) Inf else value
}

fastkpc_legacy_prepare_mgcv_cpp_shadow <- function(enabled) {
  if (!isTRUE(enabled)) return(invisible(FALSE))
  if (!exists("fastkpc_mgcv_extract_setup", mode = "function")) {
    source("fastkpc/R/mgcv_extract_oracle.R")
  }
  if (!exists("fastkpc_mgcv_solve_setup_fixed_sp_cpp", mode = "function")) {
    source("fastkpc/R/mgcv_extract_oracle.R")
  }
  if (exists("build_fastkpc_native", mode = "function")) {
    build_fastkpc_native()
  } else if (file.exists("fastkpc/R/native.R")) {
    source("fastkpc/R/native.R")
    build_fastkpc_native()
  }
  invisible(
    exists("fastkpc_mgcv_extract_setup", mode = "function") &&
      exists("fastkpc_mgcv_solve_setup_fixed_sp_cpp", mode = "function")
  )
}

fastkpc_legacy_mgcv_parse_residual_key <- function(key) {
  key <- as.character(key)
  target <- as.integer(sub(":.*$", "", key))
  s_key <- sub("^[^:]*:", "", key)
  S <- if (nzchar(s_key)) {
    as.integer(strsplit(s_key, "|", fixed = TRUE)[[1L]])
  } else {
    integer()
  }
  list(target = target, S = S)
}

fastkpc_legacy_mgcv_empty_prefetch_cache <- function(n) {
  list(
    residuals = matrix(numeric(), nrow = as.integer(n), ncol = 0L),
    key_index = new.env(parent = emptyenv()),
    keys = character()
  )
}

fastkpc_legacy_mgcv_residual_cpp_guarded_compute <- function(
    target_data, s_data, env, condition_threshold, native_s_size_limit) {
  target_data <- as.matrix(target_data)
  s_data <- as.matrix(s_data)
  s_size <- ncol(s_data)
  timings <- list(
    input_setup_ms = 0,
    gam_fit_ms = 0,
    sp_extract_ms = 0,
    setup_extract_ms = 0,
    condition_ms = 0,
    native_solve_ms = 0,
    fallback_ms = 0
  )

  fallback <- function(reason, error = FALSE, message = "") {
    fallback_start <- proc.time()[["elapsed"]]
    residual <- as.numeric(env$regrXonS(target_data, s_data)[, 1L])
    timings$fallback_ms <<- timings$fallback_ms +
      (proc.time()[["elapsed"]] - fallback_start) * 1000
    list(
      residual = residual,
      fallback_used = TRUE,
      fallback_reason = reason,
      error = isTRUE(error),
      message = as.character(message),
      s_size = as.integer(s_size),
      sp_key = "",
      setup_structure_key = "",
      timings = timings
    )
  }

  if (is.finite(native_s_size_limit) && s_size > native_s_size_limit) {
    return(fallback("outside_native_s_size_envelope"))
  }

  tryCatch({
    if (!exists("fastkpc_mgcv_extract_setup", mode = "function")) {
      source("fastkpc/R/mgcv_extract_oracle.R")
    }
    input_start <- proc.time()[["elapsed"]]
    fit_data <- data.frame(cbind(target_data, s_data))
    colnames(fit_data) <- paste0("x", seq_len(ncol(fit_data)))
    pred <- if (s_size > 0L) seq.int(2L, 1L + s_size) else integer()
    formula <- if (s_size > 2L) {
      env$frml.additive.smooth(1L, pred)
    } else {
      env$frml.full.smooth(1L, pred)
    }
    timings$input_setup_ms <- timings$input_setup_ms +
      (proc.time()[["elapsed"]] - input_start) * 1000

    gam_start <- proc.time()[["elapsed"]]
    fit <- mgcv::gam(formula, data = fit_data)
    timings$gam_fit_ms <- timings$gam_fit_ms +
      (proc.time()[["elapsed"]] - gam_start) * 1000

    sp_start <- proc.time()[["elapsed"]]
    sp <- fastkpc_mgcv_selected_sp(fit, fallback = fit$sp)
    if (length(sp) == 0L || any(!is.finite(sp)) || any(sp <= 0)) {
      stop("mgcv fit did not produce fixed positive sp", call. = FALSE)
    }
    method <- if (!is.null(fit$method)) {
      as.character(fit$method[[1L]])
    } else {
      ""
    }
    if (!nzchar(method)) method <- "GCV.Cp"
    timings$sp_extract_ms <- timings$sp_extract_ms +
      (proc.time()[["elapsed"]] - sp_start) * 1000

    setup_start <- proc.time()[["elapsed"]]
    setup <- fastkpc_mgcv_extract_setup(
      formula = formula,
      data = fit_data,
      sp = sp,
      method = method,
      target = 1L,
      S = pred
    )
    timings$setup_extract_ms <- timings$setup_extract_ms +
      (proc.time()[["elapsed"]] - setup_start) * 1000

    condition_start <- proc.time()[["elapsed"]]
    normal_condition <- fastkpc_mgcv_fixed_sp_normal_matrix_condition(
      setup, sp = setup$sp
    )
    timings$condition_ms <- timings$condition_ms +
      (proc.time()[["elapsed"]] - condition_start) * 1000
    if (is.finite(normal_condition) &&
        normal_condition > condition_threshold) {
      return(fallback("high_normal_matrix_condition"))
    }

    solve_start <- proc.time()[["elapsed"]]
    solution <- fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)
    timings$native_solve_ms <- timings$native_solve_ms +
      (proc.time()[["elapsed"]] - solve_start) * 1000
    list(
      residual = as.numeric(solution$residuals),
      fallback_used = FALSE,
      fallback_reason = "",
      error = FALSE,
      message = "",
      s_size = as.integer(s_size),
      sp_key = fastkpc_legacy_mgcv_sp_key(sp),
      setup_structure_key = fastkpc_legacy_mgcv_setup_structure_key(setup),
      timings = timings
    )
  }, error = function(e) {
    fallback("native_error", error = TRUE, message = conditionMessage(e))
  })
}

fastkpc_legacy_mgcv_residual_cpp_backend_target <- function(
    metrics, target_data, s_data, env, condition_threshold,
    native_s_size_limit, target = NA_integer_, S = integer()) {
  backend_start <- proc.time()[["elapsed"]]
  metrics$mgcv_cpp_backend_enabled <- 1L
  metrics$mgcv_cpp_backend_condition_threshold <-
    as.numeric(condition_threshold)
  metrics$mgcv_cpp_backend_native_s_size_limit <-
    as.numeric(native_s_size_limit)
  result <- fastkpc_legacy_mgcv_residual_cpp_guarded_compute(
    target_data = target_data,
    s_data = s_data,
    env = env,
    condition_threshold = condition_threshold,
    native_s_size_limit = native_s_size_limit
  )
  metrics$mgcv_cpp_backend_ms <- metrics$mgcv_cpp_backend_ms +
    (proc.time()[["elapsed"]] - backend_start) * 1000
  metrics$mgcv_cpp_backend_count <- metrics$mgcv_cpp_backend_count + 1L
  timings <- result$timings
  if (is.null(timings)) timings <- list()
  metric_time <- function(name) {
    value <- timings[[name]]
    if (is.null(value) || !is.finite(value)) 0 else as.numeric(value)
  }
  input_setup_ms <- metric_time("input_setup_ms")
  gam_fit_ms <- metric_time("gam_fit_ms")
  sp_extract_ms <- metric_time("sp_extract_ms")
  setup_extract_ms <- metric_time("setup_extract_ms")
  condition_ms <- metric_time("condition_ms")
  native_solve_ms <- metric_time("native_solve_ms")
  fallback_ms <- metric_time("fallback_ms")
  metrics$mgcv_cpp_backend_input_setup_ms <-
    metrics$mgcv_cpp_backend_input_setup_ms + input_setup_ms
  metrics$mgcv_cpp_backend_gam_fit_ms <-
    metrics$mgcv_cpp_backend_gam_fit_ms + gam_fit_ms
  metrics$mgcv_cpp_backend_sp_extract_ms <-
    metrics$mgcv_cpp_backend_sp_extract_ms + sp_extract_ms
  metrics$mgcv_cpp_backend_setup_extract_ms <-
    metrics$mgcv_cpp_backend_setup_extract_ms + setup_extract_ms
  metrics$mgcv_cpp_backend_condition_ms <-
    metrics$mgcv_cpp_backend_condition_ms + condition_ms
  metrics$mgcv_cpp_backend_native_solve_ms <-
    metrics$mgcv_cpp_backend_native_solve_ms + native_solve_ms
  metrics$mgcv_cpp_backend_fallback_ms <-
    metrics$mgcv_cpp_backend_fallback_ms + fallback_ms
  s_size <- as.integer(result$s_size)
  if (length(s_size) != 1L || is.na(s_size)) s_size <- ncol(as.matrix(s_data))
  if (s_size == 0L) {
    metrics$mgcv_cpp_backend_s_size_0_count <-
      metrics$mgcv_cpp_backend_s_size_0_count + 1L
  } else if (s_size == 1L) {
    metrics$mgcv_cpp_backend_s_size_1_count <-
      metrics$mgcv_cpp_backend_s_size_1_count + 1L
  } else if (s_size == 2L) {
    metrics$mgcv_cpp_backend_s_size_2_count <-
      metrics$mgcv_cpp_backend_s_size_2_count + 1L
  } else {
    metrics$mgcv_cpp_backend_s_size_gt2_count <-
      metrics$mgcv_cpp_backend_s_size_gt2_count + 1L
  }
  if (isTRUE(result$error)) {
    metrics$mgcv_cpp_backend_error_count <-
      metrics$mgcv_cpp_backend_error_count + 1L
  }
  if (isTRUE(result$fallback_used)) {
    metrics$mgcv_cpp_backend_fallback_count <-
      metrics$mgcv_cpp_backend_fallback_count + 1L
    metrics$mgcv_r_backend_count <- metrics$mgcv_r_backend_count + 1L
    if (s_size == 0L) {
      metrics$mgcv_cpp_backend_fallback_s_size_0_count <-
        metrics$mgcv_cpp_backend_fallback_s_size_0_count + 1L
    } else if (s_size == 1L) {
      metrics$mgcv_cpp_backend_fallback_s_size_1_count <-
        metrics$mgcv_cpp_backend_fallback_s_size_1_count + 1L
    } else if (s_size == 2L) {
      metrics$mgcv_cpp_backend_fallback_s_size_2_count <-
        metrics$mgcv_cpp_backend_fallback_s_size_2_count + 1L
    } else {
      metrics$mgcv_cpp_backend_fallback_s_size_gt2_count <-
        metrics$mgcv_cpp_backend_fallback_s_size_gt2_count + 1L
    }
    if (identical(result$fallback_reason, "high_normal_matrix_condition")) {
      metrics$mgcv_cpp_backend_high_condition_fallback_count <-
        metrics$mgcv_cpp_backend_high_condition_fallback_count + 1L
    } else if (identical(result$fallback_reason,
                         "outside_native_s_size_envelope")) {
      metrics$mgcv_cpp_backend_outside_envelope_fallback_count <-
        metrics$mgcv_cpp_backend_outside_envelope_fallback_count + 1L
    }
  } else {
    metrics$mgcv_cpp_backend_native_count <-
      metrics$mgcv_cpp_backend_native_count + 1L
    if (!is.na(as.integer(target))) {
      s_key <- fastkpc_legacy_mgcv_s_key(S)
      metrics$mgcv_cpp_backend_native_residual_keys <- c(
        metrics$mgcv_cpp_backend_native_residual_keys,
        fastkpc_legacy_mgcv_residual_key(target, S)
      )
      metrics$mgcv_cpp_backend_native_s_keys <- c(
        metrics$mgcv_cpp_backend_native_s_keys,
        s_key
      )
      metrics$mgcv_cpp_backend_native_s_sp_keys <- c(
        metrics$mgcv_cpp_backend_native_s_sp_keys,
        paste0(s_key, "||sp=", as.character(result$sp_key))
      )
      metrics$mgcv_cpp_backend_native_s_setup_keys <- c(
        metrics$mgcv_cpp_backend_native_s_setup_keys,
        paste0(s_key, "||setup=", as.character(result$setup_structure_key))
      )
      metrics$mgcv_cpp_backend_native_input_setup_ms <- c(
        metrics$mgcv_cpp_backend_native_input_setup_ms,
        input_setup_ms
      )
      metrics$mgcv_cpp_backend_native_gam_fit_ms <- c(
        metrics$mgcv_cpp_backend_native_gam_fit_ms,
        gam_fit_ms
      )
      metrics$mgcv_cpp_backend_native_sp_extract_ms <- c(
        metrics$mgcv_cpp_backend_native_sp_extract_ms,
        sp_extract_ms
      )
      metrics$mgcv_cpp_backend_native_setup_extract_ms <- c(
        metrics$mgcv_cpp_backend_native_setup_extract_ms,
        setup_extract_ms
      )
      metrics$mgcv_cpp_backend_native_condition_ms <- c(
        metrics$mgcv_cpp_backend_native_condition_ms,
        condition_ms
      )
      metrics$mgcv_cpp_backend_native_solve_call_ms <- c(
        metrics$mgcv_cpp_backend_native_solve_call_ms,
        native_solve_ms
      )
    }
    if (s_size == 0L) {
      metrics$mgcv_cpp_backend_native_s_size_0_count <-
        metrics$mgcv_cpp_backend_native_s_size_0_count + 1L
    } else if (s_size == 1L) {
      metrics$mgcv_cpp_backend_native_s_size_1_count <-
        metrics$mgcv_cpp_backend_native_s_size_1_count + 1L
    } else if (s_size == 2L) {
      metrics$mgcv_cpp_backend_native_s_size_2_count <-
        metrics$mgcv_cpp_backend_native_s_size_2_count + 1L
    } else {
      metrics$mgcv_cpp_backend_native_s_size_gt2_count <-
        metrics$mgcv_cpp_backend_native_s_size_gt2_count + 1L
    }
  }
  list(residual = as.numeric(result$residual), metrics = metrics)
}

fastkpc_legacy_mgcv_residual_cpp_shadow_target <- function(
    metrics, target_data, s_data, legacy_residual, env,
    condition_threshold, native_s_size_limit, residual_tol = 1e-5) {
  shadow_start <- proc.time()[["elapsed"]]
  metrics$mgcv_cpp_shadow_enabled <- 1L
  metrics$mgcv_cpp_shadow_condition_threshold <-
    as.numeric(condition_threshold)
  metrics$mgcv_cpp_shadow_native_s_size_limit <-
    as.numeric(native_s_size_limit)
  result <- tryCatch({
    if (!exists("fastkpc_mgcv_extract_setup", mode = "function")) {
      source("fastkpc/R/mgcv_extract_oracle.R")
    }
    s_data <- as.matrix(s_data)
    target_data <- as.matrix(target_data)
    s_size <- ncol(s_data)
    fit_data <- data.frame(cbind(target_data, s_data))
    colnames(fit_data) <- paste0("x", seq_len(ncol(fit_data)))
    pred <- if (s_size > 0L) seq.int(2L, 1L + s_size) else integer()
    formula <- if (s_size > 2L) {
      env$frml.additive.smooth(1L, pred)
    } else {
      env$frml.full.smooth(1L, pred)
    }
    fit <- mgcv::gam(formula, data = fit_data)
    sp <- fastkpc_mgcv_selected_sp(fit, fallback = fit$sp)
    if (length(sp) == 0L || any(!is.finite(sp)) || any(sp <= 0)) {
      stop("mgcv fit did not produce fixed positive sp", call. = FALSE)
    }
    method <- if (!is.null(fit$method)) {
      as.character(fit$method[[1L]])
    } else {
      ""
    }
    if (!nzchar(method)) method <- "GCV.Cp"
    setup <- fastkpc_mgcv_extract_setup(
      formula = formula,
      data = fit_data,
      sp = sp,
      method = method,
      target = 1L,
      S = pred
    )
    normal_condition <- fastkpc_mgcv_fixed_sp_normal_matrix_condition(
      setup, sp = setup$sp
    )
    fallback_reason <- ""
    fallback_used <- FALSE
    solution <- if (is.finite(native_s_size_limit) &&
                    s_size > native_s_size_limit) {
      fallback_used <- TRUE
      fallback_reason <- "outside_native_s_size_envelope"
      fastkpc_mgcv_solve_setup_fixed_sp(setup)
    } else if (is.finite(normal_condition) &&
               normal_condition > condition_threshold) {
      fallback_used <- TRUE
      fallback_reason <- "high_normal_matrix_condition"
      fastkpc_mgcv_solve_setup_fixed_sp(setup)
    } else {
      fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)
    }
    shadow_residual <- as.numeric(solution$residuals)
    legacy_residual <- as.numeric(legacy_residual)
    max_abs_diff <- max(abs(shadow_residual - legacy_residual))
    rel_l2 <- fastkpc_relative_l2_diff(shadow_residual, legacy_residual)
    list(
      fallback_used = fallback_used,
      fallback_reason = fallback_reason,
      max_abs_diff = max_abs_diff,
      rel_l2 = rel_l2,
      mismatch = isTRUE(max_abs_diff > residual_tol || rel_l2 > residual_tol)
    )
  }, error = function(e) {
    structure(list(message = conditionMessage(e)),
              class = "fastkpc_mgcv_cpp_shadow_error")
  })
  metrics$mgcv_cpp_shadow_ms <- metrics$mgcv_cpp_shadow_ms +
    (proc.time()[["elapsed"]] - shadow_start) * 1000
  metrics$mgcv_cpp_shadow_count <- metrics$mgcv_cpp_shadow_count + 1L
  if (inherits(result, "fastkpc_mgcv_cpp_shadow_error")) {
    metrics$mgcv_cpp_shadow_error_count <-
      metrics$mgcv_cpp_shadow_error_count + 1L
    return(metrics)
  }
  if (isTRUE(result$fallback_used)) {
    metrics$mgcv_cpp_shadow_fallback_count <-
      metrics$mgcv_cpp_shadow_fallback_count + 1L
    if (identical(result$fallback_reason, "high_normal_matrix_condition")) {
      metrics$mgcv_cpp_shadow_high_condition_fallback_count <-
        metrics$mgcv_cpp_shadow_high_condition_fallback_count + 1L
    } else if (identical(result$fallback_reason,
                         "outside_native_s_size_envelope")) {
      metrics$mgcv_cpp_shadow_outside_envelope_fallback_count <-
        metrics$mgcv_cpp_shadow_outside_envelope_fallback_count + 1L
    }
  } else {
    metrics$mgcv_cpp_shadow_native_count <-
      metrics$mgcv_cpp_shadow_native_count + 1L
  }
  metrics$mgcv_cpp_shadow_max_abs_diff <- max(
    metrics$mgcv_cpp_shadow_max_abs_diff, as.numeric(result$max_abs_diff)
  )
  metrics$mgcv_cpp_shadow_max_rel_l2 <- max(
    metrics$mgcv_cpp_shadow_max_rel_l2, as.numeric(result$rel_l2)
  )
  if (isTRUE(result$mismatch)) {
    metrics$mgcv_cpp_shadow_residual_mismatch_count <-
      metrics$mgcv_cpp_shadow_residual_mismatch_count + 1L
  }
  metrics
}

fastkpc_legacy_mgcv_residual_prefetch_level <- function(keys, data, env,
                                                        workers) {
  metrics <- fastkpc_legacy_runtime_zero()
  metrics$mgcv_prefetch_enabled <- 1L
  metrics$mgcv_prefetch_level_count <- 1L
  keys <- unique(as.character(keys))
  keys <- keys[nzchar(keys)]
  key_count <- length(keys)
  n <- nrow(data)
  metrics$mgcv_prefetch_key_count <- as.integer(key_count)
  metrics$mgcv_prefetch_payload_bytes <- as.numeric(n) *
    as.numeric(key_count) * 8
  metrics$mgcv_prefetch_max_level_payload_bytes <-
    metrics$mgcv_prefetch_payload_bytes
  if (key_count == 0L) {
    return(list(
      cache = fastkpc_legacy_mgcv_empty_prefetch_cache(n),
      metrics = metrics
    ))
  }

  workers <- max(1L, min(as.integer(workers), key_count))
  worker_id <- rep(seq_len(workers), length.out = key_count)
  key_chunks <- split(keys, worker_id)
  compute_chunk <- function(chunk_keys) {
    chunk_metrics <- fastkpc_legacy_runtime_zero()
    chunk_matrix <- matrix(NA_real_, nrow = n, ncol = length(chunk_keys))
    success <- rep(FALSE, length(chunk_keys))
    for (idx in seq_along(chunk_keys)) {
      parsed <- fastkpc_legacy_mgcv_parse_residual_key(chunk_keys[[idx]])
      data_subset_start <- proc.time()[["elapsed"]]
      x_data <- data[, parsed$target, drop = FALSE]
      s_data <- data[, parsed$S, drop = FALSE]
      chunk_metrics$mgcv_data_subset_ms <-
        chunk_metrics$mgcv_data_subset_ms +
          (proc.time()[["elapsed"]] - data_subset_start) * 1000

      fit_start <- proc.time()[["elapsed"]]
      fit <- tryCatch(
        env$regrXonS(x_data, s_data),
        error = function(e) structure(
          list(message = conditionMessage(e)),
          class = "fastkpc_mgcv_prefetch_error"
        )
      )
      fit_elapsed <- (proc.time()[["elapsed"]] - fit_start) * 1000
      chunk_metrics$mgcv_prefetch_fit_ms <-
        chunk_metrics$mgcv_prefetch_fit_ms + fit_elapsed
      chunk_metrics$mgcv_fit_call_ms <-
        chunk_metrics$mgcv_fit_call_ms + fit_elapsed
      if (inherits(fit, "fastkpc_mgcv_prefetch_error")) {
        chunk_metrics$mgcv_prefetch_error_count <-
          chunk_metrics$mgcv_prefetch_error_count + 1L
        next
      }

      extract_start <- proc.time()[["elapsed"]]
      chunk_matrix[, idx] <- as.numeric(fit[, 1L])
      chunk_metrics$mgcv_residual_extract_ms <-
        chunk_metrics$mgcv_residual_extract_ms +
          (proc.time()[["elapsed"]] - extract_start) * 1000
      chunk_metrics$mgcv_prefetch_fit_count <-
        chunk_metrics$mgcv_prefetch_fit_count + 1L
      chunk_metrics$mgcv_fit_count <- chunk_metrics$mgcv_fit_count + 1L
      chunk_metrics$mgcv_residual_cache_insert_count <-
        chunk_metrics$mgcv_residual_cache_insert_count + 1L
      chunk_metrics$mgcv_residual_cache_entries <-
        chunk_metrics$mgcv_residual_cache_entries + 1L
      success[[idx]] <- TRUE
    }
    list(
      keys = chunk_keys[success],
      residuals = chunk_matrix[, success, drop = FALSE],
      metrics = chunk_metrics
    )
  }

  collect_start <- proc.time()[["elapsed"]]
  chunks <- if (.Platform$OS.type == "unix" && workers > 1L) {
    parallel::mclapply(
      key_chunks, compute_chunk, mc.cores = workers, mc.set.seed = FALSE,
      mc.cleanup = TRUE, mc.allow.recursive = FALSE, mc.preschedule = TRUE
    )
  } else {
    lapply(key_chunks, compute_chunk)
  }
  metrics$mgcv_prefetch_collect_ms <-
    (proc.time()[["elapsed"]] - collect_start) * 1000

  for (chunk in chunks) {
    metrics <- fastkpc_legacy_runtime_add(metrics, chunk$metrics)
  }

  matrix_start <- proc.time()[["elapsed"]]
  success_keys <- unlist(lapply(chunks, `[[`, "keys"), use.names = FALSE)
  residual_chunks <- lapply(chunks, `[[`, "residuals")
  residual_chunks <- residual_chunks[vapply(
    residual_chunks, ncol, integer(1L)
  ) > 0L]
  residuals <- if (length(residual_chunks) > 0L) {
    do.call(cbind, residual_chunks)
  } else {
    matrix(numeric(), nrow = n, ncol = 0L)
  }
  key_index <- new.env(parent = emptyenv())
  if (length(success_keys) > 0L) {
    for (idx in seq_along(success_keys)) {
      assign(success_keys[[idx]], as.integer(idx), envir = key_index)
    }
  }
  metrics$mgcv_prefetch_matrix_build_ms <-
    (proc.time()[["elapsed"]] - matrix_start) * 1000
  metrics$mgcv_prefetch_payload_bytes <- as.numeric(n) *
    as.numeric(ncol(residuals)) * 8
  metrics$mgcv_prefetch_max_level_payload_bytes <-
    metrics$mgcv_prefetch_payload_bytes
  metrics$mgcv_prefetch_key_count <- as.integer(ncol(residuals))
  metrics$residual_ms <- metrics$mgcv_data_subset_ms +
    metrics$mgcv_fit_call_ms + metrics$mgcv_residual_extract_ms +
    metrics$mgcv_prefetch_matrix_build_ms

  list(
    cache = list(
      residuals = residuals,
      key_index = key_index,
      keys = success_keys
    ),
    metrics = metrics
  )
}

fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group <- function(
    metrics, data, targets, S, env, condition_threshold,
    native_s_size_limit) {
  metrics$mgcv_cpp_same_s_setup_provider_enabled <- 1L
  targets <- as.integer(unique(targets))
  targets <- targets[!is.na(targets)]
  if (length(targets) == 0L) {
    return(list(keys = character(), residuals = list(), metrics = metrics))
  }

  S <- as.integer(S)
  s_data <- as.matrix(data[, S, drop = FALSE])
  s_size <- ncol(s_data)
  if (is.finite(native_s_size_limit) && s_size > native_s_size_limit) {
    return(list(keys = character(), residuals = list(), metrics = metrics))
  }

  if (!exists("fastkpc_mgcv_extract_setup", mode = "function")) {
    source("fastkpc/R/mgcv_extract_oracle.R")
  }
  if (!exists("fastkpc_mgcv_extract_retarget_setup", mode = "function")) {
    source("fastkpc/R/mgcv_extract_oracle.R")
  }

  group_start <- proc.time()[["elapsed"]]
  metrics$mgcv_cpp_same_s_setup_provider_group_count <-
    metrics$mgcv_cpp_same_s_setup_provider_group_count + 1L
  metrics$mgcv_cpp_same_s_setup_provider_target_count <-
    metrics$mgcv_cpp_same_s_setup_provider_target_count + length(targets)

  shared_input_start <- proc.time()[["elapsed"]]
  pred <- if (s_size > 0L) seq.int(2L, 1L + s_size) else integer()
  formula <- if (s_size > 2L) {
    env$frml.additive.smooth(1L, pred)
  } else {
    env$frml.full.smooth(1L, pred)
  }
  shared_input_ms <- (proc.time()[["elapsed"]] - shared_input_start) * 1000
  templates <- new.env(parent = emptyenv())
  template_structure_keys <- new.env(parent = emptyenv())
  residuals <- list()
  residual_keys <- character()

  add_s_size_count <- function(native = FALSE, fallback = FALSE) {
    if (s_size == 0L) {
      metrics$mgcv_cpp_backend_s_size_0_count <<-
        metrics$mgcv_cpp_backend_s_size_0_count + 1L
      if (isTRUE(native)) {
        metrics$mgcv_cpp_backend_native_s_size_0_count <<-
          metrics$mgcv_cpp_backend_native_s_size_0_count + 1L
      }
      if (isTRUE(fallback)) {
        metrics$mgcv_cpp_backend_fallback_s_size_0_count <<-
          metrics$mgcv_cpp_backend_fallback_s_size_0_count + 1L
      }
    } else if (s_size == 1L) {
      metrics$mgcv_cpp_backend_s_size_1_count <<-
        metrics$mgcv_cpp_backend_s_size_1_count + 1L
      if (isTRUE(native)) {
        metrics$mgcv_cpp_backend_native_s_size_1_count <<-
          metrics$mgcv_cpp_backend_native_s_size_1_count + 1L
      }
      if (isTRUE(fallback)) {
        metrics$mgcv_cpp_backend_fallback_s_size_1_count <<-
          metrics$mgcv_cpp_backend_fallback_s_size_1_count + 1L
      }
    } else if (s_size == 2L) {
      metrics$mgcv_cpp_backend_s_size_2_count <<-
        metrics$mgcv_cpp_backend_s_size_2_count + 1L
      if (isTRUE(native)) {
        metrics$mgcv_cpp_backend_native_s_size_2_count <<-
          metrics$mgcv_cpp_backend_native_s_size_2_count + 1L
      }
      if (isTRUE(fallback)) {
        metrics$mgcv_cpp_backend_fallback_s_size_2_count <<-
          metrics$mgcv_cpp_backend_fallback_s_size_2_count + 1L
      }
    } else {
      metrics$mgcv_cpp_backend_s_size_gt2_count <<-
        metrics$mgcv_cpp_backend_s_size_gt2_count + 1L
      if (isTRUE(native)) {
        metrics$mgcv_cpp_backend_native_s_size_gt2_count <<-
          metrics$mgcv_cpp_backend_native_s_size_gt2_count + 1L
      }
      if (isTRUE(fallback)) {
        metrics$mgcv_cpp_backend_fallback_s_size_gt2_count <<-
          metrics$mgcv_cpp_backend_fallback_s_size_gt2_count + 1L
      }
    }
    invisible(TRUE)
  }

  run_fallback <- function(target_data, reason, error = FALSE) {
    fallback_start <- proc.time()[["elapsed"]]
    residual <- as.numeric(env$regrXonS(target_data, s_data)[, 1L])
    fallback_ms <- (proc.time()[["elapsed"]] - fallback_start) * 1000
    metrics$mgcv_cpp_backend_fallback_ms <<-
      metrics$mgcv_cpp_backend_fallback_ms + fallback_ms
    metrics$mgcv_cpp_backend_fallback_count <<-
      metrics$mgcv_cpp_backend_fallback_count + 1L
    metrics$mgcv_r_backend_count <<- metrics$mgcv_r_backend_count + 1L
    if (isTRUE(error)) {
      metrics$mgcv_cpp_backend_error_count <<-
        metrics$mgcv_cpp_backend_error_count + 1L
      metrics$mgcv_cpp_same_s_setup_provider_error_count <<-
        metrics$mgcv_cpp_same_s_setup_provider_error_count + 1L
      metrics$mgcv_cpp_same_s_prefill_error_count <<-
        metrics$mgcv_cpp_same_s_prefill_error_count + 1L
    }
    if (identical(reason, "high_normal_matrix_condition")) {
      metrics$mgcv_cpp_backend_high_condition_fallback_count <<-
        metrics$mgcv_cpp_backend_high_condition_fallback_count + 1L
    } else if (identical(reason, "outside_native_s_size_envelope")) {
      metrics$mgcv_cpp_backend_outside_envelope_fallback_count <<-
        metrics$mgcv_cpp_backend_outside_envelope_fallback_count + 1L
    }
    add_s_size_count(fallback = TRUE)
    residual
  }

  for (target in targets) {
    key <- fastkpc_legacy_mgcv_residual_key(target, S)
    backend_start <- proc.time()[["elapsed"]]
    metrics$mgcv_cpp_backend_enabled <- 1L
    metrics$mgcv_cpp_backend_condition_threshold <-
      as.numeric(condition_threshold)
    metrics$mgcv_cpp_backend_native_s_size_limit <-
      as.numeric(native_s_size_limit)
    metrics$mgcv_cpp_backend_count <- metrics$mgcv_cpp_backend_count + 1L
    target_data <- data[, target, drop = FALSE]
    input_setup_ms <- 0
    gam_fit_ms <- 0
    sp_extract_ms <- 0
    setup_extract_ms <- 0
    condition_ms <- 0
    native_solve_ms <- 0
    residual <- tryCatch({
      input_start <- proc.time()[["elapsed"]]
      fit_data <- data.frame(cbind(target_data, s_data))
      colnames(fit_data) <- paste0("x", seq_len(ncol(fit_data)))
      input_setup_ms <- shared_input_ms +
        (proc.time()[["elapsed"]] - input_start) * 1000
      shared_input_ms <- 0

      gam_start <- proc.time()[["elapsed"]]
      fit <- mgcv::gam(formula, data = fit_data)
      gam_fit_ms <- (proc.time()[["elapsed"]] - gam_start) * 1000

      sp_start <- proc.time()[["elapsed"]]
      sp <- fastkpc_mgcv_selected_sp(fit, fallback = fit$sp)
      if (length(sp) == 0L || any(!is.finite(sp)) || any(sp <= 0)) {
        stop("mgcv fit did not produce fixed positive sp", call. = FALSE)
      }
      method <- if (!is.null(fit$method)) {
        as.character(fit$method[[1L]])
      } else {
        ""
      }
      if (!nzchar(method)) method <- "GCV.Cp"
      sp_extract_ms <- (proc.time()[["elapsed"]] - sp_start) * 1000

      template_key <- paste0("method=", method)
      if (exists(template_key, envir = templates, inherits = FALSE)) {
        template <- get(template_key, envir = templates, inherits = FALSE)
        setup_structure_key <- get(
          template_key, envir = template_structure_keys, inherits = FALSE
        )
        metrics$mgcv_cpp_same_s_setup_provider_reuse_count <-
          metrics$mgcv_cpp_same_s_setup_provider_reuse_count + 1L
      } else {
        setup_start <- proc.time()[["elapsed"]]
        template <- fastkpc_mgcv_extract_setup(
          formula = formula,
          data = fit_data,
          sp = sp,
          method = method,
          target = 1L,
          S = pred
        )
        setup_extract_ms <- (proc.time()[["elapsed"]] - setup_start) * 1000
        setup_structure_key <- fastkpc_legacy_mgcv_setup_structure_key(template)
        assign(template_key, template, envir = templates)
        assign(template_key, setup_structure_key,
               envir = template_structure_keys)
        metrics$mgcv_cpp_same_s_setup_provider_template_count <-
          metrics$mgcv_cpp_same_s_setup_provider_template_count + 1L
        metrics$mgcv_cpp_same_s_setup_provider_setup_ms <-
          metrics$mgcv_cpp_same_s_setup_provider_setup_ms + setup_extract_ms
      }

      setup <- fastkpc_mgcv_extract_retarget_setup(
        setup = template,
        y = as.numeric(target_data[, 1L]),
        sp = sp,
        target = target
      )

      condition_start <- proc.time()[["elapsed"]]
      normal_condition <- fastkpc_mgcv_fixed_sp_normal_matrix_condition(
        setup, sp = setup$sp
      )
      condition_ms <- (proc.time()[["elapsed"]] - condition_start) * 1000
      if (is.finite(normal_condition) &&
          normal_condition > condition_threshold) {
        run_fallback(target_data, "high_normal_matrix_condition")
      } else {
        solve_start <- proc.time()[["elapsed"]]
        solution <- fastkpc_mgcv_solve_setup_fixed_sp_cpp(setup)
        native_solve_ms <- (proc.time()[["elapsed"]] - solve_start) * 1000
        s_key <- fastkpc_legacy_mgcv_s_key(S)
        metrics$mgcv_cpp_backend_native_count <-
          metrics$mgcv_cpp_backend_native_count + 1L
        metrics$mgcv_cpp_backend_native_residual_keys <- c(
          metrics$mgcv_cpp_backend_native_residual_keys,
          key
        )
        metrics$mgcv_cpp_backend_native_s_keys <- c(
          metrics$mgcv_cpp_backend_native_s_keys,
          s_key
        )
        metrics$mgcv_cpp_backend_native_s_sp_keys <- c(
          metrics$mgcv_cpp_backend_native_s_sp_keys,
          paste0(s_key, "||sp=", fastkpc_legacy_mgcv_sp_key(sp))
        )
        metrics$mgcv_cpp_backend_native_s_setup_keys <- c(
          metrics$mgcv_cpp_backend_native_s_setup_keys,
          paste0(s_key, "||setup=", setup_structure_key)
        )
        metrics$mgcv_cpp_backend_native_input_setup_ms <- c(
          metrics$mgcv_cpp_backend_native_input_setup_ms,
          input_setup_ms
        )
        metrics$mgcv_cpp_backend_native_gam_fit_ms <- c(
          metrics$mgcv_cpp_backend_native_gam_fit_ms,
          gam_fit_ms
        )
        metrics$mgcv_cpp_backend_native_sp_extract_ms <- c(
          metrics$mgcv_cpp_backend_native_sp_extract_ms,
          sp_extract_ms
        )
        metrics$mgcv_cpp_backend_native_setup_extract_ms <- c(
          metrics$mgcv_cpp_backend_native_setup_extract_ms,
          setup_extract_ms
        )
        metrics$mgcv_cpp_backend_native_condition_ms <- c(
          metrics$mgcv_cpp_backend_native_condition_ms,
          condition_ms
        )
        metrics$mgcv_cpp_backend_native_solve_call_ms <- c(
          metrics$mgcv_cpp_backend_native_solve_call_ms,
          native_solve_ms
        )
        add_s_size_count(native = TRUE)
        as.numeric(solution$residuals)
      }
    }, error = function(e) {
      run_fallback(target_data, "native_error", error = TRUE)
    })

    backend_ms <- (proc.time()[["elapsed"]] - backend_start) * 1000
    metrics$mgcv_cpp_backend_ms <- metrics$mgcv_cpp_backend_ms + backend_ms
    metrics$mgcv_cpp_backend_input_setup_ms <-
      metrics$mgcv_cpp_backend_input_setup_ms + input_setup_ms
    metrics$mgcv_cpp_backend_gam_fit_ms <-
      metrics$mgcv_cpp_backend_gam_fit_ms + gam_fit_ms
    metrics$mgcv_cpp_backend_sp_extract_ms <-
      metrics$mgcv_cpp_backend_sp_extract_ms + sp_extract_ms
    metrics$mgcv_cpp_backend_setup_extract_ms <-
      metrics$mgcv_cpp_backend_setup_extract_ms + setup_extract_ms
    metrics$mgcv_cpp_backend_condition_ms <-
      metrics$mgcv_cpp_backend_condition_ms + condition_ms
    metrics$mgcv_cpp_backend_native_solve_ms <-
      metrics$mgcv_cpp_backend_native_solve_ms + native_solve_ms
    metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms + backend_ms
    metrics$mgcv_fit_count <- metrics$mgcv_fit_count + 1L
    residual_keys <- c(residual_keys, key)
    residuals[[key]] <- as.numeric(residual)
  }
  metrics$residual_ms <- metrics$residual_ms +
    (proc.time()[["elapsed"]] - group_start) * 1000
  list(keys = residual_keys, residuals = residuals, metrics = metrics)
}

fastkpc_legacy_mgcv_cpp_same_s_prefill_chunk <- function(
    chunk, ind, G_l, seq_p, ord, data, env, cache_env,
    condition_threshold, native_s_size_limit) {
  metrics <- fastkpc_legacy_runtime_zero()
  prefill_start <- proc.time()[["elapsed"]]
  metrics$mgcv_cpp_same_s_prefill_enabled <- 1L
  if (is.null(cache_env) || length(chunk) == 0L || ord <= 0L) {
    return(list(keys = character(), metrics = metrics))
  }
  if (is.finite(native_s_size_limit) && ord > native_s_size_limit) {
    return(list(keys = character(), metrics = metrics))
  }

  key_seen <- new.env(parent = emptyenv())
  key_target <- integer()
  key_s <- list()
  key_s_key <- character()
  existing_count <- 0L
  add_key <- function(target, S) {
    key <- fastkpc_legacy_mgcv_residual_key(target, S)
    if (exists(key, envir = key_seen, inherits = FALSE)) return(invisible())
    assign(key, TRUE, envir = key_seen)
    if (exists(key, envir = cache_env, inherits = FALSE)) {
      existing_count <<- existing_count + 1L
      return(invisible())
    }
    key_target[[length(key_target) + 1L]] <<- as.integer(target)
    key_s[[length(key_s) + 1L]] <<- as.integer(S)
    key_s_key[[length(key_s_key) + 1L]] <<- fastkpc_legacy_mgcv_s_key(S)
    invisible()
  }
  add_keys_for_xy <- function(x, y) {
    nbrsBool <- G_l[[x]]
    nbrsBool[y] <- FALSE
    nbrs <- seq_p[nbrsBool]
    if (length(nbrs) < ord) return(invisible())
    for (S in fastkpc_legacy_combinations(nbrs, ord)) {
      add_key(x, S)
      add_key(y, S)
    }
    invisible()
  }

  for (i in as.integer(chunk)) {
    x <- ind[i, 1L]
    y <- ind[i, 2L]
    add_keys_for_xy(x, y)
    add_keys_for_xy(y, x)
  }

  keys <- names(as.list(key_seen, all.names = TRUE))
  target_count <- length(keys)
  compute_count <- length(key_target)
  metrics$mgcv_cpp_same_s_prefill_existing_count <- existing_count
  metrics$mgcv_cpp_same_s_prefill_target_count <- as.integer(target_count)
  if (compute_count == 0L) {
    metrics$mgcv_cpp_same_s_prefill_ms <-
      (proc.time()[["elapsed"]] - prefill_start) * 1000
    metrics$residual_ms <- metrics$mgcv_cpp_same_s_prefill_ms
    return(list(keys = character(), metrics = metrics))
  }

  groups <- split(seq_len(compute_count), key_s_key)
  metrics$mgcv_cpp_same_s_prefill_group_count <- length(groups)
  inserted_keys <- character()
  setup_provider_enabled <-
    fastkpc_legacy_mgcv_residual_same_s_setup_enabled()
  for (group_idx in groups) {
    S <- key_s[[group_idx[[1L]]]]
    data_subset_start <- proc.time()[["elapsed"]]
    s_data <- data[, S, drop = FALSE]
    metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
      (proc.time()[["elapsed"]] - data_subset_start) * 1000
    if (isTRUE(setup_provider_enabled) &&
        !(is.finite(native_s_size_limit) && length(S) > native_s_size_limit)) {
      provider <- fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group(
        metrics = metrics,
        data = data,
        targets = vapply(group_idx, function(idx) key_target[[idx]],
                         integer(1L)),
        S = S,
        env = env,
        condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit
      )
      metrics <- provider$metrics
      for (key in provider$keys) {
        if (exists(key, envir = cache_env, inherits = FALSE)) {
          metrics$mgcv_cpp_same_s_prefill_existing_count <-
            metrics$mgcv_cpp_same_s_prefill_existing_count + 1L
          next
        }
        store_start <- proc.time()[["elapsed"]]
        assign(key, as.numeric(provider$residuals[[key]]), envir = cache_env)
        store_elapsed <- (proc.time()[["elapsed"]] - store_start) * 1000
        metrics$mgcv_residual_cache_store_ms <-
          metrics$mgcv_residual_cache_store_ms + store_elapsed
        metrics$mgcv_residual_cache_insert_count <-
          metrics$mgcv_residual_cache_insert_count + 1L
        metrics$mgcv_residual_cache_entries <-
          metrics$mgcv_residual_cache_entries + 1L
        metrics$mgcv_cpp_same_s_prefill_cache_insert_count <-
          metrics$mgcv_cpp_same_s_prefill_cache_insert_count + 1L
        inserted_keys <- c(inserted_keys, key)
      }
      next
    }
    for (idx in group_idx) {
      target <- key_target[[idx]]
      key <- fastkpc_legacy_mgcv_residual_key(target, S)
      if (exists(key, envir = cache_env, inherits = FALSE)) {
        metrics$mgcv_cpp_same_s_prefill_existing_count <-
          metrics$mgcv_cpp_same_s_prefill_existing_count + 1L
        next
      }

      data_subset_start <- proc.time()[["elapsed"]]
      x_data <- data[, target, drop = FALSE]
      metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
        (proc.time()[["elapsed"]] - data_subset_start) * 1000

      before_error <- as.integer(metrics$mgcv_cpp_backend_error_count)
      fit_start <- proc.time()[["elapsed"]]
      backend <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = x_data,
        s_data = s_data,
        env = env,
        condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit,
        target = target,
        S = S
      )
      metrics <- backend$metrics
      metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms +
        (proc.time()[["elapsed"]] - fit_start) * 1000
      metrics$mgcv_fit_count <- metrics$mgcv_fit_count + 1L
      metrics$mgcv_cpp_same_s_prefill_error_count <-
        metrics$mgcv_cpp_same_s_prefill_error_count +
          max(0L, as.integer(metrics$mgcv_cpp_backend_error_count) -
                before_error)

      store_start <- proc.time()[["elapsed"]]
      assign(key, as.numeric(backend$residual), envir = cache_env)
      store_elapsed <- (proc.time()[["elapsed"]] - store_start) * 1000
      metrics$mgcv_residual_cache_store_ms <-
        metrics$mgcv_residual_cache_store_ms + store_elapsed
      metrics$mgcv_residual_cache_insert_count <-
        metrics$mgcv_residual_cache_insert_count + 1L
      metrics$mgcv_residual_cache_entries <-
        metrics$mgcv_residual_cache_entries + 1L
      metrics$mgcv_cpp_same_s_prefill_cache_insert_count <-
        metrics$mgcv_cpp_same_s_prefill_cache_insert_count + 1L
      inserted_keys <- c(inserted_keys, key)
    }
  }
  metrics$mgcv_cpp_same_s_prefill_ms <-
    (proc.time()[["elapsed"]] - prefill_start) * 1000
  metrics$residual_ms <- metrics$mgcv_cpp_same_s_prefill_ms
  list(keys = unique(inserted_keys), metrics = metrics)
}

fastkpc_legacy_mgcv_cpp_same_s_batch_misses <- function(
    tests, data, env, cache_env, condition_threshold, native_s_size_limit) {
  metrics <- fastkpc_legacy_runtime_zero()
  batch_start <- proc.time()[["elapsed"]]
  metrics$mgcv_cpp_same_s_setup_provider_chunk_enabled <- 1L
  if (is.null(cache_env) || length(tests) == 0L) {
    return(list(keys = character(), metrics = metrics))
  }

  key_seen <- new.env(parent = emptyenv())
  key_target <- integer()
  key_s <- list()
  key_s_key <- character()
  add_key <- function(target, S) {
    key <- fastkpc_legacy_mgcv_residual_key(target, S)
    if (exists(key, envir = key_seen, inherits = FALSE)) return(invisible())
    assign(key, TRUE, envir = key_seen)
    if (exists(key, envir = cache_env, inherits = FALSE)) return(invisible())
    key_target[[length(key_target) + 1L]] <<- as.integer(target)
    key_s[[length(key_s) + 1L]] <<- as.integer(S)
    key_s_key[[length(key_s_key) + 1L]] <<- fastkpc_legacy_mgcv_s_key(S)
    invisible()
  }

  for (test in tests) {
    S <- as.integer(test$S)
    if (length(S) == 0L) next
    add_key(test$x, S)
    add_key(test$y, S)
  }

  key_count <- length(key_target)
  if (key_count == 0L) {
    metrics$mgcv_cpp_same_s_setup_provider_chunk_ms <-
      (proc.time()[["elapsed"]] - batch_start) * 1000
    return(list(keys = character(), metrics = metrics))
  }

  groups <- split(seq_len(key_count), key_s_key)
  groups <- groups[lengths(groups) > 1L]
  if (length(groups) == 0L) {
    metrics$mgcv_cpp_same_s_setup_provider_chunk_ms <-
      (proc.time()[["elapsed"]] - batch_start) * 1000
    return(list(keys = character(), metrics = metrics))
  }

  metrics$mgcv_cpp_same_s_setup_provider_chunk_count <- 1L
  metrics$mgcv_cpp_same_s_setup_provider_chunk_group_count <- length(groups)
  metrics$mgcv_cpp_same_s_setup_provider_chunk_target_count <-
    as.integer(sum(lengths(groups)))
  inserted_keys <- character()
  for (group_idx in groups) {
    S <- key_s[[group_idx[[1L]]]]
    before_error <- as.integer(metrics$mgcv_cpp_same_s_setup_provider_error_count)
    provider <- fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group(
      metrics = metrics,
      data = data,
      targets = vapply(group_idx, function(idx) key_target[[idx]],
                       integer(1L)),
      S = S,
      env = env,
      condition_threshold = condition_threshold,
      native_s_size_limit = native_s_size_limit
    )
    metrics <- provider$metrics
    metrics$mgcv_cpp_same_s_setup_provider_chunk_error_count <-
      metrics$mgcv_cpp_same_s_setup_provider_chunk_error_count +
        max(0L, as.integer(metrics$mgcv_cpp_same_s_setup_provider_error_count) -
              before_error)
    for (key in provider$keys) {
      if (exists(key, envir = cache_env, inherits = FALSE)) next
      store_start <- proc.time()[["elapsed"]]
      assign(key, as.numeric(provider$residuals[[key]]), envir = cache_env)
      metrics$mgcv_residual_cache_store_ms <-
        metrics$mgcv_residual_cache_store_ms +
          (proc.time()[["elapsed"]] - store_start) * 1000
      metrics$mgcv_residual_cache_insert_count <-
        metrics$mgcv_residual_cache_insert_count + 1L
      metrics$mgcv_residual_cache_entries <-
        metrics$mgcv_residual_cache_entries + 1L
      metrics$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count <-
        metrics$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count + 1L
      inserted_keys <- c(inserted_keys, key)
    }
  }

  chunk_ms <- (proc.time()[["elapsed"]] - batch_start) * 1000
  metrics$mgcv_cpp_same_s_setup_provider_chunk_ms <- chunk_ms
  metrics$residual_ms <- max(as.numeric(metrics$residual_ms), chunk_ms)
  list(keys = unique(inserted_keys), metrics = metrics)
}

fastkpc_legacy_mgcv_residual_affinity_chunks <- function(edge_indices,
                                                         group_keys,
                                                         workers,
                                                         task_weights = NULL) {
  workers <- as.integer(workers)
  edge_indices <- as.integer(edge_indices)
  if (is.null(task_weights)) {
    task_weights <- rep(1, length(edge_indices))
  }
  task_weights <- as.numeric(task_weights)
  task_weights[!is.finite(task_weights) | task_weights <= 0] <- 1
  weights_by_edge <- stats::setNames(task_weights, as.character(edge_indices))
  groups <- split(as.integer(edge_indices), as.character(group_keys),
                  drop = TRUE)
  group_sizes <- lengths(groups, use.names = TRUE)
  target_load <- max(1, sum(task_weights) / workers)
  pieces <- list()
  piece_names <- character()
  piece_weights <- numeric()
  piece_group_names <- character()
  for (group_name in names(groups)) {
    group_edges <- groups[[group_name]]
    current_edges <- integer()
    current_weight <- 0
    piece_idx <- 1L
    for (edge in group_edges) {
      edge_weight <- weights_by_edge[[as.character(edge)]]
      if (length(current_edges) > 0L &&
          current_weight + edge_weight > target_load) {
        pieces[[length(pieces) + 1L]] <- current_edges
        piece_names[[length(piece_names) + 1L]] <-
          paste0(group_name, "#", piece_idx)
        piece_weights[[length(piece_weights) + 1L]] <- current_weight
        piece_group_names[[length(piece_group_names) + 1L]] <- group_name
        current_edges <- integer()
        current_weight <- 0
        piece_idx <- piece_idx + 1L
      }
      current_edges <- c(current_edges, as.integer(edge))
      current_weight <- current_weight + edge_weight
    }
    if (length(current_edges) > 0L) {
      pieces[[length(pieces) + 1L]] <- current_edges
      piece_names[[length(piece_names) + 1L]] <-
        paste0(group_name, "#", piece_idx)
      piece_weights[[length(piece_weights) + 1L]] <- current_weight
      piece_group_names[[length(piece_group_names) + 1L]] <- group_name
    }
  }
  pieces_per_group <- table(piece_group_names)
  split_group_names <- names(pieces_per_group[pieces_per_group > 1L])
  names(piece_weights) <- piece_names
  order_idx <- order(-piece_weights, names(piece_weights))
  chunks <- vector("list", workers)
  loads <- numeric(workers)
  for (piece_idx in order_idx) {
    worker <- which.min(loads)
    piece_edges <- pieces[[piece_idx]]
    chunks[[worker]] <- c(chunks[[worker]], piece_edges)
    loads[[worker]] <- loads[[worker]] + piece_weights[[piece_idx]]
  }
  chunks <- lapply(chunks, sort)
  mean_load <- mean(loads)
  list(
    chunks = chunks[lengths(chunks) > 0L],
    group_count = as.integer(length(groups)),
    task_count = as.integer(length(edge_indices)),
    worker_count = as.integer(workers),
    max_group_size = if (length(group_sizes) > 0L) {
      as.integer(max(group_sizes))
    } else {
      0L
    },
    mean_group_size = if (length(group_sizes) > 0L) {
      as.numeric(mean(group_sizes))
    } else {
      0
    },
    split_group_count = as.integer(length(split_group_names)),
    split_group_tasks = if (length(split_group_names) > 0L) {
      as.integer(sum(group_sizes[split_group_names]))
    } else {
      0L
    },
    split_group_pieces = if (length(split_group_names) > 0L) {
      as.integer(sum(pieces_per_group[split_group_names]))
    } else {
      0L
    },
    split_group_keys = split_group_names,
    load_imbalance = if (mean_load > 0) {
      as.numeric(max(loads) / mean_load)
    } else {
      0
    }
  )
}

fastkpc_legacy_mgcv_residual_key_worker_loss <- function(keys_by_worker) {
  if (length(keys_by_worker) == 0L) return(0L)
  all_keys <- unlist(keys_by_worker, use.names = FALSE)
  if (length(all_keys) == 0L) return(0L)
  worker_unique_count <- sum(vapply(
    keys_by_worker,
    function(keys) length(unique(keys)),
    integer(1L)
  ))
  as.integer(max(0L, worker_unique_count - length(unique(all_keys))))
}

fastkpc_legacy_mgcv_residual_owner_chunks <- function(edge_indices,
                                                       group_keys,
                                                       edge_keys,
                                                       workers,
                                                       task_weights = NULL) {
  workers <- as.integer(workers)
  edge_indices <- as.integer(edge_indices)
  if (is.null(task_weights)) {
    task_weights <- rep(1, length(edge_indices))
  }
  task_weights <- as.numeric(task_weights)
  task_weights[!is.finite(task_weights) | task_weights <= 0] <- 1
  names(task_weights) <- as.character(edge_indices)
  key_map_start <- proc.time()[["elapsed"]]
  all_keys <- unlist(edge_keys, use.names = FALSE)
  key_weights <- table(all_keys)
  key_map_ms <- (proc.time()[["elapsed"]] - key_map_start) * 1000
  key_owner <- integer()
  chunks <- vector("list", workers)
  loads <- numeric(workers)
  target_load <- max(1, sum(task_weights) / workers)
  max_load <- target_load * 1.35
  both_local <- 0L
  one_local <- 0L
  none_local <- 0L
  conflict <- 0L
  predicted_hits <- 0L
  spill <- 0L
  task_score_ms <- 0
  greedy_assign_start <- proc.time()[["elapsed"]]

  order_idx <- order(-task_weights, as.integer(names(task_weights)))
  for (edge in as.integer(names(task_weights)[order_idx])) {
    edge_name <- as.character(edge)
    keys <- unique(edge_keys[[edge_name]])
    keys <- keys[nzchar(keys)]
    owners <- key_owner[keys]
    owners <- owners[!is.na(owners) & owners > 0L]
    if (length(unique(owners)) > 1L) conflict <- conflict + 1L

    score_start <- proc.time()[["elapsed"]]
    local_score <- numeric(workers)
    if (length(keys) > 0L && length(key_owner) > 0L) {
      for (worker in seq_len(workers)) {
        local_keys <- keys[key_owner[keys] == worker]
        local_keys <- local_keys[!is.na(local_keys)]
        if (length(local_keys) > 0L) {
          local_score[[worker]] <-
            sum(log1p(as.numeric(key_weights[local_keys])))
        }
      }
    }
    task_score_ms <- task_score_ms +
      (proc.time()[["elapsed"]] - score_start) * 1000

    edge_weight <- task_weights[[edge_name]]
    feasible <- (loads + edge_weight) <= max_load
    candidate_workers <- if (any(feasible)) {
      which(feasible)
    } else {
      seq_len(workers)
    }
    if (any(local_score > 0)) {
      score <- local_score - (loads + edge_weight) / target_load
      best_local_worker <- which.max(score)
      candidate_scores <- score[candidate_workers]
      worker <- candidate_workers[[which.max(candidate_scores)]]
      if (!identical(as.integer(worker), as.integer(best_local_worker))) {
        spill <- spill + 1L
      }
    } else {
      candidate_loads <- loads[candidate_workers]
      worker <- candidate_workers[[which.min(candidate_loads)]]
    }

    local_count <- 0L
    if (length(keys) > 0L && length(key_owner) > 0L) {
      local_count <- sum(key_owner[keys] == worker, na.rm = TRUE)
    }
    if (length(keys) == 0L || local_count == 0L) {
      none_local <- none_local + 1L
    } else if (local_count >= length(keys)) {
      both_local <- both_local + 1L
    } else {
      one_local <- one_local + 1L
    }
    predicted_hits <- predicted_hits + as.integer(local_count)

    unowned <- keys[is.na(key_owner[keys]) | key_owner[keys] == 0L]
    if (length(unowned) > 0L) {
      key_owner[unowned] <- as.integer(worker)
    }
    chunks[[worker]] <- c(chunks[[worker]], edge)
    loads[[worker]] <- loads[[worker]] + edge_weight
  }
  greedy_assign_ms <- (proc.time()[["elapsed"]] - greedy_assign_start) *
    1000 - task_score_ms

  chunk_sort_start <- proc.time()[["elapsed"]]
  chunks <- lapply(chunks, sort)
  chunk_sort_ms <- (proc.time()[["elapsed"]] - chunk_sort_start) * 1000
  materialize_start <- proc.time()[["elapsed"]]
  worker_by_edge <- integer(length(edge_indices))
  names(worker_by_edge) <- as.character(edge_indices)
  for (worker in seq_along(chunks)) {
    worker_by_edge[as.character(chunks[[worker]])] <- as.integer(worker)
  }

  group_worker_pairs <- unique(data.frame(
    group = as.character(group_keys),
    worker = as.integer(worker_by_edge[as.character(edge_indices)]),
    stringsAsFactors = FALSE
  ))
  workers_per_group <- table(group_worker_pairs$group)
  split_group_names <- names(workers_per_group[workers_per_group > 1L])
  group_sizes <- table(as.character(group_keys))
  mean_load <- mean(loads)
  chunk_materialize_ms <- (proc.time()[["elapsed"]] - materialize_start) *
    1000

  list(
    chunks = chunks[lengths(chunks) > 0L],
    worker_by_edge = worker_by_edge,
    group_count = as.integer(length(group_sizes)),
    task_count = as.integer(length(edge_indices)),
    worker_count = as.integer(workers),
    max_group_size = if (length(group_sizes) > 0L) {
      as.integer(max(group_sizes))
    } else {
      0L
    },
    mean_group_size = if (length(group_sizes) > 0L) {
      as.numeric(mean(group_sizes))
    } else {
      0
    },
    split_group_count = as.integer(length(split_group_names)),
    split_group_tasks = if (length(split_group_names) > 0L) {
      as.integer(sum(group_sizes[split_group_names]))
    } else {
      0L
    },
    split_group_pieces = if (length(split_group_names) > 0L) {
      as.integer(sum(workers_per_group[split_group_names]))
    } else {
      0L
    },
    split_group_keys = split_group_names,
    load_imbalance = if (mean_load > 0) {
      as.numeric(max(loads) / mean_load)
    } else {
      0
    },
    owner_key_count = as.integer(length(key_owner)),
    owner_task_count = as.integer(length(edge_indices)),
    both_local_count = as.integer(both_local),
    one_local_count = as.integer(one_local),
    none_local_count = as.integer(none_local),
    conflict_count = as.integer(conflict),
    predicted_hit_count = as.integer(predicted_hits),
    owner_load_imbalance = if (mean_load > 0) {
      as.numeric(max(loads) / mean_load)
    } else {
      0
    },
    spill_count = as.integer(spill),
    key_map_build_ms = as.numeric(key_map_ms),
    task_score_ms = as.numeric(task_score_ms),
    greedy_assign_ms = as.numeric(max(0, greedy_assign_ms)),
    chunk_sort_ms = as.numeric(chunk_sort_ms),
    chunk_materialize_ms = as.numeric(chunk_materialize_ms)
  )
}

fastkpc_legacy_run_mgcv_residual_pair <- function(
    metrics, data, x, y, S, env, cache_env = NULL, cache_enabled = FALSE,
    prefetch_cache = NULL, prefetch_enabled = FALSE,
    cpp_backend_enabled = FALSE,
    same_s_setup_consumed_enabled = FALSE,
    cpp_backend_condition_threshold = 1e12,
    cpp_backend_native_s_size_limit = Inf,
    cpp_shadow_enabled = FALSE,
    cpp_shadow_condition_threshold = 1e12,
    cpp_shadow_native_s_size_limit = Inf) {
  residual_start <- proc.time()[["elapsed"]]
  S_int <- as.integer(S)

  key_start <- residual_start
  target_keys <- c(
    fastkpc_legacy_mgcv_residual_key(x, S_int),
    fastkpc_legacy_mgcv_residual_key(y, S_int)
  )
  s_key <- fastkpc_legacy_mgcv_s_key(S_int)
  metrics$mgcv_residual_keys <- target_keys
  metrics$mgcv_s_keys <- rep(s_key, 2L)
  metrics$mgcv_residual_request_count <- 2L
  if (length(S_int) == 0L) {
    metrics$mgcv_s_size_0_count <- 2L
  } else if (length(S_int) == 1L) {
    metrics$mgcv_s_size_1_count <- 2L
  } else if (length(S_int) == 2L) {
    metrics$mgcv_s_size_2_count <- 2L
  } else {
    metrics$mgcv_s_size_gt2_count <- 2L
  }
  metrics$mgcv_key_build_ms <-
    metrics$mgcv_key_build_ms +
      (proc.time()[["elapsed"]] - key_start) * 1000

  if (isTRUE(prefetch_enabled) && !is.null(prefetch_cache) &&
      !is.null(prefetch_cache$key_index) && !is.null(prefetch_cache$residuals)) {
    lookup_start <- proc.time()[["elapsed"]]
    col_idx <- vapply(target_keys, function(key) {
      if (exists(key, envir = prefetch_cache$key_index, inherits = FALSE)) {
        as.integer(get(key, envir = prefetch_cache$key_index, inherits = FALSE))
      } else {
        NA_integer_
      }
    }, integer(1L))
    lookup_elapsed <- (proc.time()[["elapsed"]] - lookup_start) * 1000
    metrics$mgcv_cache_lookup_ms <- metrics$mgcv_cache_lookup_ms +
      lookup_elapsed
    metrics$mgcv_prefetch_lookup_ms <- metrics$mgcv_prefetch_lookup_ms +
      lookup_elapsed

    if (all(!is.na(col_idx))) {
      extract_start <- proc.time()[["elapsed"]]
      residuals <- cbind(
        prefetch_cache$residuals[, col_idx[[1L]]],
        prefetch_cache$residuals[, col_idx[[2L]]]
      )
      extract_elapsed <- (proc.time()[["elapsed"]] - extract_start) * 1000
      metrics$mgcv_residual_extract_ms <-
        metrics$mgcv_residual_extract_ms + extract_elapsed
      metrics$mgcv_prefetch_ci_phase_ms <-
        metrics$mgcv_prefetch_ci_phase_ms + lookup_elapsed + extract_elapsed
      metrics$mgcv_cache_hit_count <- metrics$mgcv_cache_hit_count + 2L
      metrics$mgcv_fit_avoided_count <- metrics$mgcv_fit_avoided_count + 2L
      metrics$mgcv_residual_cache_hit_keys <-
        c(metrics$mgcv_residual_cache_hit_keys, target_keys)
      metrics$mgcv_residual_cache_hit_ms <-
        metrics$mgcv_residual_cache_hit_ms + lookup_elapsed
      metrics$residual_ms <- (proc.time()[["elapsed"]] - residual_start) * 1000
      metrics$mgcv_unaccounted_ms <- max(
        0,
        metrics$residual_ms - metrics$mgcv_key_build_ms -
          metrics$mgcv_cache_lookup_ms -
          metrics$mgcv_formula_build_ms -
          metrics$mgcv_data_subset_ms -
          metrics$mgcv_fit_call_ms -
          metrics$mgcv_residual_extract_ms -
          metrics$mgcv_result_store_ms -
          metrics$mgcv_residual_cache_store_ms
      )
      return(list(residuals = residuals, metrics = metrics))
    }
    metrics$mgcv_prefetch_error_count <-
      metrics$mgcv_prefetch_error_count + sum(is.na(col_idx))
  }

  if (!isTRUE(cache_enabled)) {
    data_subset_start <- proc.time()[["elapsed"]]
    xy_data <- data[, c(x, y)]
    s_data <- data[, S_int, drop = FALSE]
    metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
      (proc.time()[["elapsed"]] - data_subset_start) * 1000

    fit_start <- proc.time()[["elapsed"]]
    if (isTRUE(cpp_backend_enabled)) {
      backend_x <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = xy_data[, 1L, drop = FALSE],
        s_data = s_data,
        env = env,
        condition_threshold = cpp_backend_condition_threshold,
        native_s_size_limit = cpp_backend_native_s_size_limit,
        target = x,
        S = S_int
      )
      metrics <- backend_x$metrics
      backend_y <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = xy_data[, 2L, drop = FALSE],
        s_data = s_data,
        env = env,
        condition_threshold = cpp_backend_condition_threshold,
        native_s_size_limit = cpp_backend_native_s_size_limit,
        target = y,
        S = S_int
      )
      metrics <- backend_y$metrics
      residuals <- cbind(backend_x$residual, backend_y$residual)
    } else {
      residuals <- env$regrXonS(xy_data, s_data)
      metrics$mgcv_r_backend_count <- metrics$mgcv_r_backend_count + 2L
    }
    metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms +
      (proc.time()[["elapsed"]] - fit_start) * 1000
    metrics$mgcv_fit_count <- 2L
    metrics$mgcv_cache_miss_count <- 2L

    residual_extract_start <- proc.time()[["elapsed"]]
    residuals <- as.matrix(residuals)
    metrics$mgcv_residual_extract_ms <- metrics$mgcv_residual_extract_ms +
      (proc.time()[["elapsed"]] - residual_extract_start) * 1000
    if (!isTRUE(cpp_backend_enabled) && isTRUE(cpp_shadow_enabled)) {
      metrics <- fastkpc_legacy_mgcv_residual_cpp_shadow_target(
        metrics = metrics,
        target_data = xy_data[, 1L, drop = FALSE],
        s_data = s_data,
        legacy_residual = residuals[, 1L],
        env = env,
        condition_threshold = cpp_shadow_condition_threshold,
        native_s_size_limit = cpp_shadow_native_s_size_limit
      )
      metrics <- fastkpc_legacy_mgcv_residual_cpp_shadow_target(
        metrics = metrics,
        target_data = xy_data[, 2L, drop = FALSE],
        s_data = s_data,
        legacy_residual = residuals[, 2L],
        env = env,
        condition_threshold = cpp_shadow_condition_threshold,
        native_s_size_limit = cpp_shadow_native_s_size_limit
      )
    }
    metrics$residual_ms <- (proc.time()[["elapsed"]] - residual_start) * 1000
    metrics$mgcv_unaccounted_ms <- max(
      0,
      metrics$residual_ms - metrics$mgcv_key_build_ms -
        metrics$mgcv_cache_lookup_ms -
        metrics$mgcv_formula_build_ms -
        metrics$mgcv_data_subset_ms -
        metrics$mgcv_fit_call_ms -
        metrics$mgcv_residual_extract_ms -
        metrics$mgcv_result_store_ms -
        metrics$mgcv_cpp_shadow_ms
    )
    return(list(residuals = residuals, metrics = metrics))
  }

  if (is.null(cache_env)) cache_env <- new.env(parent = emptyenv())
  s_data <- NULL
  residual_cols <- vector("list", 2L)
  targets <- c(as.integer(x), as.integer(y))
  miss_indices <- integer()
  for (idx in seq_along(targets)) {
    key <- target_keys[[idx]]
    lookup_start <- proc.time()[["elapsed"]]
    hit <- exists(key, envir = cache_env, inherits = FALSE)
    if (isTRUE(hit)) {
      residual_cols[[idx]] <- get(key, envir = cache_env, inherits = FALSE)
      hit_elapsed <- (proc.time()[["elapsed"]] - lookup_start) * 1000
      metrics$mgcv_cache_hit_count <- metrics$mgcv_cache_hit_count + 1L
      metrics$mgcv_fit_avoided_count <- metrics$mgcv_fit_avoided_count + 1L
      metrics$mgcv_residual_cache_hit_keys <-
        c(metrics$mgcv_residual_cache_hit_keys, key)
      metrics$mgcv_residual_cache_hit_ms <-
        metrics$mgcv_residual_cache_hit_ms + hit_elapsed
      metrics$mgcv_cache_lookup_ms <- metrics$mgcv_cache_lookup_ms +
        hit_elapsed
      next
    }
    metrics$mgcv_cache_lookup_ms <- metrics$mgcv_cache_lookup_ms +
      (proc.time()[["elapsed"]] - lookup_start) * 1000
    metrics$mgcv_cache_miss_count <- metrics$mgcv_cache_miss_count + 1L
    metrics$mgcv_residual_cache_miss_keys <-
      c(metrics$mgcv_residual_cache_miss_keys, key)
    miss_indices <- c(miss_indices, idx)
  }

  if (isTRUE(same_s_setup_consumed_enabled) &&
      isTRUE(cpp_backend_enabled) &&
      length(miss_indices) > 1L &&
      !(is.finite(cpp_backend_native_s_size_limit) &&
          length(S_int) > cpp_backend_native_s_size_limit)) {
    provider <- fastkpc_legacy_mgcv_cpp_same_s_setup_provider_group(
      metrics = metrics,
      data = data,
      targets = targets[miss_indices],
      S = S_int,
      env = env,
      condition_threshold = cpp_backend_condition_threshold,
      native_s_size_limit = cpp_backend_native_s_size_limit
    )
    metrics <- provider$metrics
    if (length(provider$keys) > 0L) {
      for (key in provider$keys) {
        idx <- match(key, target_keys)
        if (length(idx) != 1L || is.na(idx)) next
        target_residual <- as.numeric(provider$residuals[[key]])
        store_start <- proc.time()[["elapsed"]]
        assign(key, target_residual, envir = cache_env)
        metrics$mgcv_residual_cache_store_ms <-
          metrics$mgcv_residual_cache_store_ms +
            (proc.time()[["elapsed"]] - store_start) * 1000
        metrics$mgcv_residual_cache_insert_count <-
          metrics$mgcv_residual_cache_insert_count + 1L
        metrics$mgcv_residual_cache_entries <-
          metrics$mgcv_residual_cache_entries + 1L
        residual_cols[[idx]] <- target_residual
      }
      miss_indices <- miss_indices[
        !(target_keys[miss_indices] %in% provider$keys)
      ]
    }
  }

  for (idx in miss_indices) {
    key <- target_keys[[idx]]
    data_subset_start <- proc.time()[["elapsed"]]
    x_data <- data[, targets[[idx]], drop = FALSE]
    if (is.null(s_data)) s_data <- data[, S_int, drop = FALSE]
    metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
      (proc.time()[["elapsed"]] - data_subset_start) * 1000

    fit_start <- proc.time()[["elapsed"]]
    if (isTRUE(cpp_backend_enabled)) {
      backend <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = x_data,
        s_data = s_data,
        env = env,
        condition_threshold = cpp_backend_condition_threshold,
        native_s_size_limit = cpp_backend_native_s_size_limit,
        target = targets[[idx]],
        S = S_int
      )
      metrics <- backend$metrics
      target_residual <- backend$residual
    } else {
      target_residual <- as.numeric(env$regrXonS(x_data, s_data)[, 1L])
      metrics$mgcv_r_backend_count <- metrics$mgcv_r_backend_count + 1L
    }
    metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms +
      (proc.time()[["elapsed"]] - fit_start) * 1000
    metrics$mgcv_fit_count <- metrics$mgcv_fit_count + 1L
    if (!isTRUE(cpp_backend_enabled) && isTRUE(cpp_shadow_enabled)) {
      metrics <- fastkpc_legacy_mgcv_residual_cpp_shadow_target(
        metrics = metrics,
        target_data = x_data,
        s_data = s_data,
        legacy_residual = target_residual,
        env = env,
        condition_threshold = cpp_shadow_condition_threshold,
        native_s_size_limit = cpp_shadow_native_s_size_limit
      )
    }

    store_start <- proc.time()[["elapsed"]]
    assign(key, target_residual, envir = cache_env)
    metrics$mgcv_residual_cache_store_ms <-
      metrics$mgcv_residual_cache_store_ms +
        (proc.time()[["elapsed"]] - store_start) * 1000
    metrics$mgcv_residual_cache_insert_count <-
      metrics$mgcv_residual_cache_insert_count + 1L
    metrics$mgcv_residual_cache_entries <-
      metrics$mgcv_residual_cache_entries + 1L
    residual_cols[[idx]] <- target_residual
  }

  residual_extract_start <- proc.time()[["elapsed"]]
  residuals <- cbind(residual_cols[[1L]], residual_cols[[2L]])
  metrics$mgcv_residual_extract_ms <- metrics$mgcv_residual_extract_ms +
    (proc.time()[["elapsed"]] - residual_extract_start) * 1000
  metrics$residual_ms <- (proc.time()[["elapsed"]] - residual_start) * 1000
  metrics$mgcv_unaccounted_ms <- max(
    0,
    metrics$residual_ms - metrics$mgcv_key_build_ms -
      metrics$mgcv_cache_lookup_ms -
      metrics$mgcv_formula_build_ms -
      metrics$mgcv_data_subset_ms -
      metrics$mgcv_fit_call_ms -
      metrics$mgcv_residual_extract_ms -
      metrics$mgcv_result_store_ms -
      metrics$mgcv_residual_cache_store_ms -
      metrics$mgcv_cpp_shadow_ms
  )
  list(residuals = residuals, metrics = metrics)
}

fastkpc_legacy_runtime_finalize_mgcv_keys <- function(metrics) {
  residual_keys <- metrics$mgcv_residual_keys
  s_keys <- metrics$mgcv_s_keys
  hit_keys <- metrics$mgcv_residual_cache_hit_keys
  miss_keys <- metrics$mgcv_residual_cache_miss_keys
  if (length(hit_keys) > 0L) {
    metrics$mgcv_residual_cache_hit_key_count <- length(hit_keys)
  }
  if (length(miss_keys) > 0L) {
    metrics$mgcv_residual_cache_miss_key_count <- length(miss_keys)
    miss_s_keys <- sub("^[^:]*:", "", as.character(miss_keys))
    miss_s_keys <- miss_s_keys[nzchar(miss_s_keys)]
    if (length(miss_s_keys) > 0L) {
      miss_per_s <- table(miss_s_keys)
      metrics$mgcv_residual_cache_miss_s_group_count <-
        length(miss_per_s)
      metrics$mgcv_residual_cache_miss_s_total_targets <-
        sum(as.integer(miss_per_s))
      metrics$mgcv_residual_cache_miss_s_max_targets <-
        max(as.integer(miss_per_s))
      metrics$mgcv_residual_cache_miss_s_mean_targets <-
        mean(as.integer(miss_per_s))
      metrics$mgcv_residual_cache_miss_s_reuse_opportunity_count <-
        metrics$mgcv_residual_cache_miss_s_total_targets -
          metrics$mgcv_residual_cache_miss_s_group_count
      metrics$mgcv_residual_cache_miss_s_reuse_ratio <-
        if (metrics$mgcv_residual_cache_miss_s_total_targets > 0L) {
          metrics$mgcv_residual_cache_miss_s_reuse_opportunity_count /
            metrics$mgcv_residual_cache_miss_s_total_targets
        } else {
          0
        }
    }
  }
  if (length(residual_keys) > 0L) {
    unique_residual <- unique(residual_keys)
    metrics$mgcv_unique_residual_key_count <- length(unique_residual)
    metrics$mgcv_duplicate_residual_key_count <-
      length(residual_keys) - length(unique_residual)
    metrics$mgcv_unique_target_s_count <- length(unique_residual)
  }
  if (length(s_keys) > 0L) {
    per_s <- table(s_keys)
    metrics$mgcv_unique_s_count <- length(per_s)
    metrics$mgcv_same_s_group_count <- length(per_s)
    metrics$mgcv_same_s_total_targets <- sum(as.integer(per_s))
    metrics$mgcv_same_s_max_targets <- max(as.integer(per_s))
    metrics$mgcv_same_s_mean_targets <- mean(as.integer(per_s))
    metrics$mgcv_same_s_reuse_opportunity_count <-
      metrics$mgcv_same_s_total_targets - metrics$mgcv_same_s_group_count
  }
  native_s_keys <- metrics$mgcv_cpp_backend_native_s_keys
  native_residual_keys <- metrics$mgcv_cpp_backend_native_residual_keys
  native_s_sp_keys <- metrics$mgcv_cpp_backend_native_s_sp_keys
  native_s_setup_keys <- metrics$mgcv_cpp_backend_native_s_setup_keys
  if (length(native_s_keys) > 0L) {
    native_per_s <- table(native_s_keys)
    metrics$mgcv_cpp_backend_same_s_native_group_count <-
      length(native_per_s)
    metrics$mgcv_cpp_backend_same_s_native_target_count <-
      length(native_s_keys)
    metrics$mgcv_cpp_backend_same_s_native_max_targets <-
      max(as.integer(native_per_s))
    metrics$mgcv_cpp_backend_same_s_native_mean_targets <-
      mean(as.integer(native_per_s))
    metrics$mgcv_cpp_backend_same_s_native_reuse_opportunity_count <-
      length(native_s_keys) - length(native_per_s)
    metrics$mgcv_cpp_backend_same_s_native_setup_reuse_ratio <-
      if (length(native_s_keys) > 0L) {
        metrics$mgcv_cpp_backend_same_s_native_reuse_opportunity_count /
          length(native_s_keys)
      } else {
        0
      }
  }
  if (length(native_s_sp_keys) > 0L) {
    native_per_s_sp <- table(native_s_sp_keys)
    metrics$mgcv_cpp_backend_same_s_sp_native_group_count <-
      length(native_per_s_sp)
    metrics$mgcv_cpp_backend_same_s_sp_native_target_count <-
      length(native_s_sp_keys)
    metrics$mgcv_cpp_backend_same_s_sp_native_max_targets <-
      max(as.integer(native_per_s_sp))
    metrics$mgcv_cpp_backend_same_s_sp_native_mean_targets <-
      mean(as.integer(native_per_s_sp))
    metrics$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count <-
      length(native_s_sp_keys) - length(native_per_s_sp)
    metrics$mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio <-
      if (length(native_s_sp_keys) > 0L) {
        metrics$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count /
          length(native_s_sp_keys)
      } else {
        0
      }
  }
  if (length(native_s_setup_keys) > 0L) {
    native_per_s_setup <- table(native_s_setup_keys)
    metrics$mgcv_cpp_backend_same_s_setup_native_group_count <-
      length(native_per_s_setup)
    metrics$mgcv_cpp_backend_same_s_setup_native_target_count <-
      length(native_s_setup_keys)
    metrics$mgcv_cpp_backend_same_s_setup_native_max_targets <-
      max(as.integer(native_per_s_setup))
    metrics$mgcv_cpp_backend_same_s_setup_native_mean_targets <-
      mean(as.integer(native_per_s_setup))
    metrics$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count <-
      length(native_s_setup_keys) - length(native_per_s_setup)
    metrics$mgcv_cpp_backend_same_s_setup_native_reuse_ratio <-
      if (length(native_s_setup_keys) > 0L) {
        metrics$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count /
          length(native_s_setup_keys)
      } else {
        0
      }
  }
  if (length(native_s_setup_keys) > 0L) {
    input_saved <- fastkpc_legacy_group_reuse_potential_ms(
      native_s_setup_keys,
      metrics$mgcv_cpp_backend_native_input_setup_ms
    )
    setup_saved <- fastkpc_legacy_group_reuse_potential_ms(
      native_s_setup_keys,
      metrics$mgcv_cpp_backend_native_setup_extract_ms
    )
    condition_saved <- fastkpc_legacy_group_reuse_potential_ms(
      native_s_setup_keys,
      metrics$mgcv_cpp_backend_native_condition_ms
    )
    metrics$mgcv_cpp_backend_same_s_setup_input_potential_saved_ms <-
      input_saved
    metrics$mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms <-
      setup_saved
    metrics$mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms <-
      condition_saved
    metrics$mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms <-
      input_saved + setup_saved + condition_saved
  }
  if (length(native_s_sp_keys) > 0L) {
    solve_saved <- fastkpc_legacy_group_reuse_potential_ms(
      native_s_sp_keys,
      metrics$mgcv_cpp_backend_native_solve_call_ms
    )
    total_solve <- sum(as.numeric(metrics$mgcv_cpp_backend_native_solve_call_ms),
                       na.rm = TRUE)
    metrics$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms <-
      solve_saved
    metrics$mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio <-
      if (total_solve > 0) solve_saved / total_solve else 0
  }
  metrics$mgcv_cpp_backend_same_s_gam_fit_preserved_ms <-
    as.numeric(metrics$mgcv_cpp_backend_gam_fit_ms)
  metrics$mgcv_residual_keys <- character()
  metrics$mgcv_s_keys <- character()
  metrics$mgcv_residual_cache_hit_keys <- character()
  metrics$mgcv_residual_cache_miss_keys <- character()
  metrics$mgcv_cpp_backend_native_residual_keys <- character()
  metrics$mgcv_cpp_backend_native_s_keys <- character()
  metrics$mgcv_cpp_backend_native_s_sp_keys <- character()
  metrics$mgcv_cpp_backend_native_s_setup_keys <- character()
  metrics$mgcv_cpp_backend_native_input_setup_ms <- numeric()
  metrics$mgcv_cpp_backend_native_gam_fit_ms <- numeric()
  metrics$mgcv_cpp_backend_native_sp_extract_ms <- numeric()
  metrics$mgcv_cpp_backend_native_setup_extract_ms <- numeric()
  metrics$mgcv_cpp_backend_native_condition_ms <- numeric()
  metrics$mgcv_cpp_backend_native_solve_call_ms <- numeric()
  metrics
}

fastkpc_legacy_dcov_cpp_shadow_enabled <- function() {
  identical(Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_CPP_SHADOW", unset = ""), "1")
}

fastkpc_legacy_dcov_backend <- function() {
  backend <- tolower(Sys.getenv("FASTKPC_LEGACY_DCOV_GAMMA_BACKEND",
                                unset = "r"))
  if (backend %in% c("cpp", "c++")) "cpp" else "r"
}

fastkpc_legacy_prepare_dcov_cpp <- function(enabled) {
  if (!isTRUE(enabled)) return(invisible(FALSE))
  if (!exists("fastkpc_legacy_dcov_gamma_cpp_oracle", mode = "function")) {
    try(source("fastkpc/R/native.R"), silent = TRUE)
  }
  if (exists("build_fastkpc_native", mode = "function")) {
    build_fastkpc_native()
  }
  invisible(exists("fastkpc_legacy_dcov_gamma_cpp_oracle", mode = "function"))
}

fastkpc_legacy_runtime_add_dcov_cpp_diagnostics <- function(
    metrics, cpp_diag, wrapper_ms = NULL) {
  cpp_diag_value <- function(name) {
    value <- cpp_diag[[name]]
    if (is.null(value)) 0 else as.numeric(value)
  }
  cpp_total_ms <- cpp_diag_value("total_ms")
  metrics$dcov_cpp_input_ms <- metrics$dcov_cpp_input_ms +
    cpp_diag_value("input_ms")
  metrics$dcov_cpp_distance_ms <- metrics$dcov_cpp_distance_ms +
    cpp_diag_value("distance_ms")
  metrics$dcov_cpp_lowrank_ms <- metrics$dcov_cpp_lowrank_ms +
    cpp_diag_value("lowrank_ms")
  metrics$dcov_cpp_lowrank_eig_ms <- metrics$dcov_cpp_lowrank_eig_ms +
    cpp_diag_value("lowrank_eig_ms")
  metrics$dcov_cpp_lowrank_select_ms <- metrics$dcov_cpp_lowrank_select_ms +
    cpp_diag_value("lowrank_select_ms")
  metrics$dcov_cpp_lowrank_center_ms <- metrics$dcov_cpp_lowrank_center_ms +
    cpp_diag_value("lowrank_center_ms")
  metrics$dcov_cpp_lowrank_unaccounted_ms <-
    metrics$dcov_cpp_lowrank_unaccounted_ms +
      cpp_diag_value("lowrank_unaccounted_ms")
  metrics$dcov_cpp_lowrank_full_eig_count <-
    metrics$dcov_cpp_lowrank_full_eig_count +
      as.integer(cpp_diag_value("lowrank_full_eig_count"))
  metrics$dcov_cpp_lowrank_spectra_count <-
    metrics$dcov_cpp_lowrank_spectra_count +
      as.integer(cpp_diag_value("lowrank_spectra_count"))
  metrics$dcov_cpp_lowrank_spectra_converged_count <-
    metrics$dcov_cpp_lowrank_spectra_converged_count +
      as.integer(cpp_diag_value("lowrank_spectra_converged_count"))
  metrics$dcov_cpp_lowrank_spectra_failed_count <-
    metrics$dcov_cpp_lowrank_spectra_failed_count +
      as.integer(cpp_diag_value("lowrank_spectra_failed_count"))
  metrics$dcov_cpp_lowrank_spectra_fallback_full_eig_count <-
    metrics$dcov_cpp_lowrank_spectra_fallback_full_eig_count +
      as.integer(cpp_diag_value("lowrank_spectra_fallback_full_eig_count"))
  metrics$dcov_cpp_lowrank_spectra_iterations <-
    metrics$dcov_cpp_lowrank_spectra_iterations +
      as.integer(cpp_diag_value("lowrank_spectra_iterations"))
  metrics$dcov_cpp_lowrank_spectra_nconv <-
    metrics$dcov_cpp_lowrank_spectra_nconv +
      as.integer(cpp_diag_value("lowrank_spectra_nconv"))
  metrics$dcov_cpp_lowrank_spectra_ncv <-
    max(metrics$dcov_cpp_lowrank_spectra_ncv,
        as.integer(cpp_diag_value("lowrank_spectra_ncv")))
  metrics$dcov_cpp_lowrank_spectra_tol <-
    max(metrics$dcov_cpp_lowrank_spectra_tol,
        cpp_diag_value("lowrank_spectra_tol"))
  metrics$dcov_cpp_statistic_ms <- metrics$dcov_cpp_statistic_ms +
    cpp_diag_value("statistic_ms")
  metrics$dcov_cpp_moment_ms <- metrics$dcov_cpp_moment_ms +
    cpp_diag_value("moment_ms")
  metrics$dcov_cpp_pgamma_ms <- metrics$dcov_cpp_pgamma_ms +
    cpp_diag_value("pgamma_ms")
  metrics$dcov_cpp_accounted_ms <- metrics$dcov_cpp_accounted_ms +
    cpp_diag_value("accounted_ms")
  metrics$dcov_cpp_unaccounted_ms <- metrics$dcov_cpp_unaccounted_ms +
    cpp_diag_value("unaccounted_ms")
  if (!is.null(wrapper_ms)) {
    metrics$dcov_cpp_overhead_ms <- metrics$dcov_cpp_overhead_ms +
      max(0, as.numeric(wrapper_ms) - cpp_total_ms)
  }
  metrics
}

fastkpc_legacy_runtime_add_dcov_cpp_shadow <- function(
    metrics, x, y, legacy_result, alpha, index, numCol) {
  shadow_start <- proc.time()[["elapsed"]]
  cpp <- tryCatch(
    fastkpc_legacy_dcov_gamma_cpp_oracle(
      x = x, y = y, index = index, numCol = numCol
    ),
    error = function(e) structure(list(message = conditionMessage(e)),
                                  class = "fastkpc_dcov_cpp_shadow_error")
  )
  shadow_elapsed_ms <- (proc.time()[["elapsed"]] - shadow_start) * 1000
  metrics$dcov_cpp_shadow_ms <- metrics$dcov_cpp_shadow_ms + shadow_elapsed_ms
  if (inherits(cpp, "fastkpc_dcov_cpp_shadow_error")) {
    metrics$dcov_cpp_shadow_error_count <-
      metrics$dcov_cpp_shadow_error_count + 1L
    return(metrics)
  }

  cpp_diag <- cpp$diagnostics
  metrics <- fastkpc_legacy_runtime_add_dcov_cpp_diagnostics(
    metrics, cpp_diag, wrapper_ms = shadow_elapsed_ms
  )

  legacy_estimates <- as.numeric(legacy_result$estimates)
  legacy_p <- as.numeric(legacy_result$p.value)
  cpp_p <- as.numeric(cpp$p.value)
  p_diff <- abs(cpp_p - legacy_p)
  nV2_diff <- abs(as.numeric(cpp$nV2) - legacy_estimates[[1L]])
  mean_diff <- abs(as.numeric(cpp$mean) - legacy_estimates[[2L]])
  variance_diff <- abs(as.numeric(cpp$variance) - legacy_estimates[[3L]])
  metrics$dcov_cpp_shadow_count <- metrics$dcov_cpp_shadow_count + 1L
  metrics$dcov_cpp_shadow_max_p_diff <- max(
    metrics$dcov_cpp_shadow_max_p_diff, p_diff)
  metrics$dcov_cpp_shadow_max_nV2_diff <- max(
    metrics$dcov_cpp_shadow_max_nV2_diff, nV2_diff)
  metrics$dcov_cpp_shadow_max_mean_diff <- max(
    metrics$dcov_cpp_shadow_max_mean_diff, mean_diff)
  metrics$dcov_cpp_shadow_max_variance_diff <- max(
    metrics$dcov_cpp_shadow_max_variance_diff, variance_diff)
  if (!identical(cpp_p >= alpha, legacy_p >= alpha)) {
    metrics$dcov_cpp_shadow_decision_flip_count <-
      metrics$dcov_cpp_shadow_decision_flip_count + 1L
  }
  if (abs(legacy_p - alpha) <= 1e-6 || abs(cpp_p - alpha) <= 1e-6) {
    metrics$dcov_cpp_shadow_near_alpha_count <-
      metrics$dcov_cpp_shadow_near_alpha_count + 1L
  }
  metrics
}

fastkpc_legacy_run_dcov_cpp_backend <- function(
    metrics, x, y, index, numCol, env) {
  backend_start <- proc.time()[["elapsed"]]
  cpp <- tryCatch(
    fastkpc_legacy_dcov_gamma_cpp_oracle(
      x = x, y = y, index = index, numCol = numCol
    ),
    error = function(e) structure(list(message = conditionMessage(e)),
                                  class = "fastkpc_dcov_cpp_backend_error")
  )
  backend_elapsed_ms <- (proc.time()[["elapsed"]] - backend_start) * 1000
  metrics$dcov_cpp_backend_ms <-
    metrics$dcov_cpp_backend_ms + backend_elapsed_ms

  if (!inherits(cpp, "fastkpc_dcov_cpp_backend_error")) {
    metrics$dcov_cpp_backend_count <- metrics$dcov_cpp_backend_count + 1L
    metrics <- fastkpc_legacy_runtime_add_dcov_cpp_diagnostics(
      metrics, cpp$diagnostics, wrapper_ms = backend_elapsed_ms
    )
    return(list(result = cpp, metrics = metrics, used_cpp = TRUE))
  }

  metrics$dcov_cpp_backend_error_count <-
    metrics$dcov_cpp_backend_error_count + 1L
  metrics$dcov_cpp_backend_fallback_count <-
    metrics$dcov_cpp_backend_fallback_count + 1L
  timed <- fastkpc_legacy_dcov_gamma_timed(
    x = x, y = y, index = index, numCol = numCol, env = env
  )
  metrics <- fastkpc_legacy_runtime_add_dcov(metrics, timed$diagnostics)
  metrics$dcov_r_backend_count <- metrics$dcov_r_backend_count + 1L
  list(result = timed$result, metrics = metrics, used_cpp = FALSE)
}

fastkpc_legacy_parallel_skeleton <- function(data, alpha, max_conditioning_size,
                                             ic.method = "dcc.gamma",
                                             index = 1,
                                             numCol = floor(nrow(data) / 10),
                                             labels = NULL,
                                             num_cores = NULL,
                                             env = fastkpc_legacy_env(),
                                             na_delete = TRUE) {
  if (!requireNamespace("graph", quietly = TRUE)) {
    stop("graph is required for legacy parallel skeleton compatibility",
         call. = FALSE)
  }
  fastkpc_require_legacy_packages(
    fastkpc_legacy_packages_for_method(
      ic.method, conditional = isTRUE(max_conditioning_size > 0)
    )
  )
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  p <- ncol(data)
  if (is.null(labels)) {
    labels <- colnames(data)
    if (is.null(labels)) labels <- as.character(seq_len(p))
  }
  colnames(data) <- labels
  seq_p <- seq_len(p)
  max_level <- if (is.infinite(max_conditioning_size)) {
    p - 2L
  } else {
    min(as.integer(max_conditioning_size), p - 2L)
  }
  num_cores <- fastkpc_legacy_parallel_cores(num_cores)
  dcov_backend <- if (identical(ic.method, "dcc.gamma")) {
    fastkpc_legacy_dcov_backend()
  } else {
    "r"
  }
  dcov_cpp_shadow_enabled <- identical(ic.method, "dcc.gamma") &&
    identical(dcov_backend, "r") &&
    fastkpc_legacy_dcov_cpp_shadow_enabled()
  dcov_cpp_required <- identical(dcov_backend, "cpp") ||
    isTRUE(dcov_cpp_shadow_enabled)
  fastkpc_legacy_prepare_dcov_cpp(dcov_cpp_required)
  mgcv_residual_cache_enabled <- identical(ic.method, "dcc.gamma") &&
    fastkpc_legacy_mgcv_residual_cache_enabled()
  mgcv_residual_affinity_mode <-
    fastkpc_legacy_mgcv_residual_affinity_mode()
  mgcv_residual_prefetch_mode <-
    fastkpc_legacy_mgcv_residual_prefetch_mode()
  mgcv_residual_prefetch_requested <- identical(
    mgcv_residual_prefetch_mode, "level"
  )
  mgcv_residual_prefetch_enabled <- identical(ic.method, "dcc.gamma") &&
    isTRUE(mgcv_residual_cache_enabled) &&
    isTRUE(mgcv_residual_prefetch_requested) &&
    identical(.Platform$OS.type, "unix")
  mgcv_residual_backend <- if (identical(ic.method, "dcc.gamma")) {
    fastkpc_legacy_mgcv_residual_backend()
  } else {
    "r"
  }
  mgcv_residual_cpp_backend_enabled <-
    identical(mgcv_residual_backend, "cpp_guarded")
  mgcv_residual_same_s_prefill_enabled <- identical(ic.method, "dcc.gamma") &&
    isTRUE(mgcv_residual_cache_enabled) &&
    isTRUE(mgcv_residual_cpp_backend_enabled) &&
    fastkpc_legacy_mgcv_residual_same_s_prefill_enabled()
  mgcv_residual_same_s_setup_mode <-
    fastkpc_legacy_mgcv_residual_same_s_setup_mode()
  mgcv_residual_same_s_setup_consumed_enabled <-
    identical(ic.method, "dcc.gamma") &&
    isTRUE(mgcv_residual_cache_enabled) &&
    isTRUE(mgcv_residual_cpp_backend_enabled) &&
    identical(mgcv_residual_same_s_setup_mode, "consumed")
  mgcv_residual_same_s_setup_chunk_enabled <-
    identical(ic.method, "dcc.gamma") &&
    isTRUE(mgcv_residual_cache_enabled) &&
    isTRUE(mgcv_residual_cpp_backend_enabled) &&
    identical(mgcv_residual_same_s_setup_mode, "chunk")
  mgcv_residual_cpp_backend_condition_threshold <-
    fastkpc_legacy_mgcv_residual_backend_condition_threshold()
  mgcv_residual_cpp_backend_native_s_size_limit <-
    fastkpc_legacy_mgcv_residual_backend_native_s_size_limit()
  mgcv_residual_cpp_shadow_enabled <- identical(ic.method, "dcc.gamma") &&
    fastkpc_legacy_mgcv_residual_cpp_shadow_enabled()
  mgcv_residual_cpp_shadow_condition_threshold <-
    fastkpc_legacy_mgcv_residual_cpp_shadow_condition_threshold()
  mgcv_residual_cpp_shadow_native_s_size_limit <-
    fastkpc_legacy_mgcv_residual_cpp_shadow_native_s_size_limit()
  fastkpc_legacy_prepare_mgcv_cpp_shadow(
    isTRUE(mgcv_residual_cpp_backend_enabled) ||
      isTRUE(mgcv_residual_cpp_shadow_enabled)
  )

  G <- matrix(TRUE, nrow = p, ncol = p)
  diag(G) <- FALSE
  sepset <- fastkpc_legacy_sepsets(p)
  pMax <- matrix(-Inf, nrow = p, ncol = p)
  diag(pMax) <- 1
  n_edgetests <- numeric()
  n_edges <- numeric()
  level_logs <- list()
  level_metrics <- list()
  done <- FALSE
  ord <- 0L
  suffStat <- list(data = data, ic.method = ic.method, index = index,
                   numCol = numCol)
  total_start <- proc.time()[["elapsed"]]
  mgcv_residual_cache_env <- NULL
  mgcv_residual_prefetch_cache <- NULL
  mgcv_residual_prefetch_level_enabled <- FALSE

  run_legacy_ci <- function(x, y, S) {
    metrics <- fastkpc_legacy_runtime_zero()
    ci_start <- proc.time()[["elapsed"]]
    value <- tryCatch({
      if (identical(ic.method, "dcc.gamma")) {
        if (length(S) == 0L) {
          metrics$direct_ci_count <- 1L
          if (identical(dcov_backend, "cpp")) {
            backend <- fastkpc_legacy_run_dcov_cpp_backend(
              metrics = metrics, x = data[, x], y = data[, y],
              index = index, numCol = numCol, env = env
            )
            metrics <- backend$metrics
            result <- backend$result
          } else {
            timed <- fastkpc_legacy_dcov_gamma_timed(
              x = data[, x], y = data[, y], index = index, numCol = numCol,
              env = env
            )
            metrics <- fastkpc_legacy_runtime_add_dcov(
              metrics, timed$diagnostics
            )
            metrics$dcov_r_backend_count <- 1L
            if (isTRUE(dcov_cpp_shadow_enabled)) {
              metrics <- fastkpc_legacy_runtime_add_dcov_cpp_shadow(
                metrics = metrics, x = data[, x], y = data[, y],
                legacy_result = timed$result, alpha = alpha,
                index = index, numCol = numCol
              )
            }
            result <- timed$result
          }
          metrics$dcov_gamma_count <- 1L
          result$p.value
        } else {
          metrics$conditional_ci_count <- 1L
          residual <- fastkpc_legacy_run_mgcv_residual_pair(
            metrics = metrics, data = data, x = x, y = y, S = S, env = env,
            cache_env = mgcv_residual_cache_env,
            cache_enabled = mgcv_residual_cache_enabled,
            prefetch_cache = mgcv_residual_prefetch_cache,
            prefetch_enabled = mgcv_residual_prefetch_level_enabled,
            cpp_backend_enabled = mgcv_residual_cpp_backend_enabled,
            same_s_setup_consumed_enabled =
              mgcv_residual_same_s_setup_consumed_enabled,
            cpp_backend_condition_threshold =
              mgcv_residual_cpp_backend_condition_threshold,
            cpp_backend_native_s_size_limit =
              mgcv_residual_cpp_backend_native_s_size_limit,
            cpp_shadow_enabled = mgcv_residual_cpp_shadow_enabled,
            cpp_shadow_condition_threshold =
              mgcv_residual_cpp_shadow_condition_threshold,
            cpp_shadow_native_s_size_limit =
              mgcv_residual_cpp_shadow_native_s_size_limit
          )
          metrics <- residual$metrics
          residuals <- residual$residuals
          if (identical(dcov_backend, "cpp")) {
            backend <- fastkpc_legacy_run_dcov_cpp_backend(
              metrics = metrics, x = residuals[, 1L], y = residuals[, 2L],
              index = index, numCol = numCol, env = env
            )
            metrics <- backend$metrics
            result <- backend$result
          } else {
            timed <- fastkpc_legacy_dcov_gamma_timed(
              x = residuals[, 1L], y = residuals[, 2L],
              index = index, numCol = numCol, env = env
            )
            metrics <- fastkpc_legacy_runtime_add_dcov(
              metrics, timed$diagnostics
            )
            metrics$dcov_r_backend_count <- 1L
            if (isTRUE(dcov_cpp_shadow_enabled)) {
              metrics <- fastkpc_legacy_runtime_add_dcov_cpp_shadow(
                metrics = metrics, x = residuals[, 1L], y = residuals[, 2L],
                legacy_result = timed$result, alpha = alpha,
                index = index, numCol = numCol
              )
            }
            result <- timed$result
          }
          metrics$dcov_gamma_count <- 1L
          result$p.value
        }
      } else {
        env$kernelCItest(x, y, S, suffStat)
      }
    }, error = function(e) {
      structure(NA_real_, fastkpc_error = conditionMessage(e))
    })
    metrics$ci_total_ms <- (proc.time()[["elapsed"]] - ci_start) * 1000
    list(p.value = value, metrics = metrics)
  }

  while (!done && any(G) && ord <= max_level) {
    ord1 <- ord + 1L
    n_edgetests[ord1] <- 0
    n_edges[ord1] <- 0
    level_log <- list()
    done <- TRUE
    ind <- which(G, arr.ind = TRUE)
    ind <- ind[order(ind[, 1L]), , drop = FALSE]
    ind <- ind[ind[, 1L] < ind[, 2L], , drop = FALSE]
    remaining_edge_tests <- nrow(ind)
    if (remaining_edge_tests == 0L) break
    G_l <- split(G, gl(p, p))
    mgcv_residual_cache_env <- if (isTRUE(mgcv_residual_cache_enabled)) {
      new.env(parent = emptyenv())
    } else {
      NULL
    }
    edge_indices <- seq_len(remaining_edge_tests)
    workers <- min(num_cores, length(edge_indices))
    mgcv_residual_prefetch_cache <- NULL
    mgcv_residual_prefetch_level_enabled <- FALSE
    prefetch_metrics <- fastkpc_legacy_runtime_zero()
    if (isTRUE(mgcv_residual_prefetch_enabled) && workers > 1L &&
        ord > 0L) {
      level_prefetch_keys_for_xy <- function(x, y) {
        nbrsBool <- G_l[[x]]
        nbrsBool[y] <- FALSE
        nbrs <- seq_p[nbrsBool]
        if (length(nbrs) < ord) return(character())
        keys <- character()
        for (S in fastkpc_legacy_combinations(nbrs, ord)) {
          keys <- c(
            keys,
            fastkpc_legacy_mgcv_residual_key(x, S),
            fastkpc_legacy_mgcv_residual_key(y, S)
          )
        }
        keys
      }
      prefetch_keys <- unlist(lapply(edge_indices, function(i) {
        x <- ind[i, 1L]
        y <- ind[i, 2L]
        c(level_prefetch_keys_for_xy(x, y),
          level_prefetch_keys_for_xy(y, x))
      }), use.names = FALSE)
      prefetched <- fastkpc_legacy_mgcv_residual_prefetch_level(
        prefetch_keys, data = data, env = env, workers = workers
      )
      mgcv_residual_prefetch_cache <- prefetched$cache
      mgcv_residual_prefetch_level_enabled <- TRUE
      prefetch_metrics <- prefetched$metrics
    }

    edge_test_xy <- function(x, y) {
      G_xy <- TRUE
      num_tests_xy <- 0L
      pMax_xy <- pMax[x, y]
      sepset_xy <- NULL
      done_xy <- TRUE
      metrics <- fastkpc_legacy_runtime_zero()
      nbrsBool <- G_l[[x]]
      nbrsBool[y] <- FALSE
      nbrs <- seq_p[nbrsBool]
      length_nbrs <- length(nbrs)
      if (length_nbrs >= ord) {
        if (length_nbrs > ord) done_xy <- FALSE
        for (S in fastkpc_legacy_combinations(nbrs, ord)) {
          num_tests_xy <- num_tests_xy + 1L
          ci <- run_legacy_ci(x, y, S)
          metrics <- fastkpc_legacy_runtime_add(metrics, ci$metrics)
          pval <- ci$p.value
          if (length(pval) == 0L ||
              !is.numeric(pval) || is.na(pval[1L])) {
            pval <- as.numeric(na_delete)
          } else {
            pval <- as.numeric(pval[1L])
          }
          if (pMax_xy < pval) pMax_xy <- pval
          if (pval >= alpha) {
            G_xy <- FALSE
            sepset_xy <- as.integer(S)
            break
          }
        }
      }
      list(G_xy, sepset_xy, num_tests_xy, pMax_xy, done_xy, metrics)
    }

    edge_test <- function(i) {
      x <- ind[i, 1L]
      y <- ind[i, 2L]
      num_tests_i <- 0L
      G_i <- TRUE
      pMax_xy <- pMax[x, y]
      pMax_yx <- pMax[y, x]
      sepset_xy <- NULL
      sepset_yx <- NULL
      done_i <- TRUE
      metrics_i <- fastkpc_legacy_runtime_zero()

      res_x <- edge_test_xy(x, y)
      G_i <- res_x[[1L]]
      sepset_xy <- res_x[[2L]]
      num_tests_i <- num_tests_i + res_x[[3L]]
      pMax_xy <- res_x[[4L]]
      done_i <- done_i & res_x[[5L]]
      metrics_i <- fastkpc_legacy_runtime_add(metrics_i, res_x[[6L]])

      if (G_i) {
        if (ord == 0L) {
          num_tests_i <- num_tests_i + 1L
          metrics_i$fake_level0_test_count <-
            metrics_i$fake_level0_test_count + 1L
        } else {
          res_y <- edge_test_xy(y, x)
          G_i <- res_y[[1L]]
          sepset_yx <- res_y[[2L]]
          num_tests_i <- num_tests_i + res_y[[3L]]
          pMax_yx <- res_y[[4L]]
          done_i <- done_i & res_y[[5L]]
          metrics_i <- fastkpc_legacy_runtime_add(metrics_i, res_y[[6L]])
        }
      }

      list(i, G_i, sepset_xy, sepset_yx, num_tests_i, pMax_xy, pMax_yx,
           done_i, metrics_i)
    }

    normalize_ci_pvalue <- function(pval) {
      if (length(pval) == 0L || !is.numeric(pval) || is.na(pval[1L])) {
        as.numeric(na_delete)
      } else {
        as.numeric(pval[1L])
      }
    }

    chunk_state_direction <- function(a, b) {
      nbrsBool <- G_l[[a]]
      nbrsBool[b] <- FALSE
      nbrs <- seq_p[nbrsBool]
      length_nbrs <- length(nbrs)
      list(
        x = as.integer(a),
        y = as.integer(b),
        combos = if (length_nbrs >= ord) {
          fastkpc_legacy_combinations(nbrs, ord)
        } else {
          list()
        },
        pos = 1L,
        done_flag = !isTRUE(length_nbrs > ord)
      )
    }

    chunk_state_init <- function(i) {
      x <- ind[i, 1L]
      y <- ind[i, 2L]
      list(
        i = as.integer(i),
        directions = list(
          chunk_state_direction(x, y),
          chunk_state_direction(y, x)
        ),
        direction = 1L,
        G_i = TRUE,
        sepset_xy = NULL,
        sepset_yx = NULL,
        num_tests = 0L,
        pMax_xy = pMax[x, y],
        pMax_yx = pMax[y, x],
        done_i = TRUE,
        metrics = fastkpc_legacy_runtime_zero()
      )
    }

    chunk_state_next_test <- function(state) {
      repeat {
        if (!isTRUE(state$G_i) || state$direction > 2L) {
          return(list(state = state, test = NULL))
        }
        direction <- state$directions[[state$direction]]
        if (length(direction$combos) >= direction$pos) {
          return(list(
            state = state,
            test = list(
              state_i = state$i,
              direction = state$direction,
              x = direction$x,
              y = direction$y,
              S = direction$combos[[direction$pos]]
            )
          ))
        }
        state$done_i <- state$done_i & direction$done_flag
        state$direction <- state$direction + 1L
      }
    }

    chunk_state_apply_pvalue <- function(state, test, pval) {
      direction_idx <- as.integer(test$direction)
      direction <- state$directions[[direction_idx]]
      state$num_tests <- state$num_tests + 1L
      if (direction_idx == 1L) {
        if (state$pMax_xy < pval) state$pMax_xy <- pval
      } else {
        if (state$pMax_yx < pval) state$pMax_yx <- pval
      }
      if (pval >= alpha) {
        state$G_i <- FALSE
        if (direction_idx == 1L) {
          state$sepset_xy <- as.integer(test$S)
        } else {
          state$sepset_yx <- as.integer(test$S)
        }
        state$done_i <- state$done_i & direction$done_flag
        state$direction <- 3L
        return(state)
      }
      direction$pos <- direction$pos + 1L
      state$directions[[direction_idx]] <- direction
      if (direction$pos > length(direction$combos)) {
        state$done_i <- state$done_i & direction$done_flag
        state$direction <- state$direction + 1L
      }
      state
    }

    chunk_state_result <- function(state) {
      list(
        state$i, state$G_i, state$sepset_xy, state$sepset_yx,
        state$num_tests, state$pMax_xy, state$pMax_yx, state$done_i,
        state$metrics
      )
    }

    edge_test_chunk_same_s <- function(chunk) {
      states <- lapply(as.integer(chunk), chunk_state_init)
      names(states) <- as.character(as.integer(chunk))
      batch_metrics <- fastkpc_legacy_runtime_zero()
      repeat {
        current_tests <- list()
        for (state_name in names(states)) {
          next_state <- chunk_state_next_test(states[[state_name]])
          states[[state_name]] <- next_state$state
          if (!is.null(next_state$test)) {
            current_tests[[length(current_tests) + 1L]] <- next_state$test
          }
        }
        if (length(current_tests) == 0L) break

        batch <- fastkpc_legacy_mgcv_cpp_same_s_batch_misses(
          tests = current_tests,
          data = data,
          env = env,
          cache_env = mgcv_residual_cache_env,
          condition_threshold = mgcv_residual_cpp_backend_condition_threshold,
          native_s_size_limit = mgcv_residual_cpp_backend_native_s_size_limit
        )
        batch_metrics <- fastkpc_legacy_runtime_add(
          batch_metrics, batch$metrics
        )

        for (test in current_tests) {
          ci <- run_legacy_ci(test$x, test$y, test$S)
          state_name <- as.character(test$state_i)
          states[[state_name]]$metrics <- fastkpc_legacy_runtime_add(
            states[[state_name]]$metrics, ci$metrics
          )
          pval <- normalize_ci_pvalue(ci$p.value)
          states[[state_name]] <- chunk_state_apply_pvalue(
            states[[state_name]], test, pval
          )
        }
      }
      list(
        results = lapply(states, chunk_state_result),
        metrics = batch_metrics
      )
    }

    affinity_enabled <- .Platform$OS.type == "unix" && workers > 1L &&
      ord > 0L && isTRUE(mgcv_residual_cache_enabled) &&
      mgcv_residual_affinity_mode %in% c("s", "target_s")
    affinity_metrics <- fastkpc_legacy_runtime_zero()
    affinity_worker_by_edge <- integer()
    affinity_group_by_edge <- character()
    affinity_split_group_keys <- character()
    res <- if (isTRUE(affinity_enabled)) {
      affinity_schedule_start <- proc.time()[["elapsed"]]
      edge_s_key_for_xy <- function(x, y) {
        nbrsBool <- G_l[[x]]
        nbrsBool[y] <- FALSE
        nbrs <- seq_p[nbrsBool]
        if (length(nbrs) < ord) return(character())
        fastkpc_legacy_mgcv_s_key(nbrs[seq_len(ord)])
      }
      edge_work_for_xy <- function(x, y) {
        nbrsBool <- G_l[[x]]
        nbrsBool[y] <- FALSE
        nbrs <- seq_p[nbrsBool]
        if (length(nbrs) < ord) return(0)
        as.numeric(choose(length(nbrs), ord))
      }
      edge_owner_keys_for_xy <- function(x, y) {
        nbrsBool <- G_l[[x]]
        nbrsBool[y] <- FALSE
        nbrs <- seq_p[nbrsBool]
        if (length(nbrs) < ord) return(character())
        keys <- character()
        for (S in fastkpc_legacy_combinations(nbrs, ord)) {
          keys <- c(
            keys,
            fastkpc_legacy_mgcv_residual_key(x, S),
            fastkpc_legacy_mgcv_residual_key(y, S)
          )
        }
        unique(keys)
      }
      edge_affinity_key <- function(i) {
        x <- ind[i, 1L]
        y <- ind[i, 2L]
        s_keys <- unique(c(edge_s_key_for_xy(x, y),
                           edge_s_key_for_xy(y, x)))
        if (length(s_keys) == 0L) return(paste0("edge:", i))
        paste(sort(s_keys), collapse = "||")
      }
      edge_affinity_weight <- function(i) {
        x <- ind[i, 1L]
        y <- ind[i, 2L]
        max(1, edge_work_for_xy(x, y) + edge_work_for_xy(y, x))
      }
      edge_owner_keys <- function(i) {
        x <- ind[i, 1L]
        y <- ind[i, 2L]
        unique(c(edge_owner_keys_for_xy(x, y),
                 edge_owner_keys_for_xy(y, x)))
      }
      group_keys <- vapply(edge_indices, edge_affinity_key, character(1L))
      names(group_keys) <- as.character(edge_indices)
      task_weights <- vapply(edge_indices, edge_affinity_weight, numeric(1L))
      if (identical(mgcv_residual_affinity_mode, "target_s")) {
        owner_key_enum_start <- proc.time()[["elapsed"]]
        owner_keys <- lapply(edge_indices, edge_owner_keys)
        names(owner_keys) <- as.character(edge_indices)
        affinity_metrics$mgcv_residual_owner_key_enum_ms <-
          (proc.time()[["elapsed"]] - owner_key_enum_start) * 1000
        schedule <- fastkpc_legacy_mgcv_residual_owner_chunks(
          edge_indices, group_keys, owner_keys, workers,
          task_weights = task_weights
        )
        affinity_worker_by_edge <- schedule$worker_by_edge
        affinity_metrics$mgcv_residual_owner_enabled <- 1L
        affinity_metrics$mgcv_residual_owner_key_count <-
          schedule$owner_key_count
        affinity_metrics$mgcv_residual_owner_task_count <-
          schedule$owner_task_count
        affinity_metrics$mgcv_residual_owner_both_local_count <-
          schedule$both_local_count
        affinity_metrics$mgcv_residual_owner_one_local_count <-
          schedule$one_local_count
        affinity_metrics$mgcv_residual_owner_none_local_count <-
          schedule$none_local_count
        affinity_metrics$mgcv_residual_owner_conflict_count <-
          schedule$conflict_count
        affinity_metrics$mgcv_residual_owner_predicted_hit_count <-
          schedule$predicted_hit_count
        affinity_metrics$mgcv_residual_owner_load_imbalance <-
          schedule$owner_load_imbalance
        affinity_metrics$mgcv_residual_owner_spill_count <-
          schedule$spill_count
        affinity_metrics$mgcv_residual_owner_key_map_build_ms <-
          schedule$key_map_build_ms
        affinity_metrics$mgcv_residual_owner_task_score_ms <-
          schedule$task_score_ms
        affinity_metrics$mgcv_residual_owner_greedy_assign_ms <-
          schedule$greedy_assign_ms
        affinity_metrics$mgcv_residual_owner_chunk_sort_ms <-
          schedule$chunk_sort_ms
        affinity_metrics$mgcv_residual_owner_chunk_materialize_ms <-
          schedule$chunk_materialize_ms
      } else {
        schedule <- fastkpc_legacy_mgcv_residual_affinity_chunks(
          edge_indices, group_keys, workers, task_weights = task_weights
        )
        affinity_worker_by_edge <- integer(length(edge_indices))
        names(affinity_worker_by_edge) <- as.character(edge_indices)
        for (worker_id in seq_along(schedule$chunks)) {
          affinity_worker_by_edge[as.character(schedule$chunks[[worker_id]])] <-
            as.integer(worker_id)
        }
      }
      affinity_group_by_edge <- group_keys
      affinity_split_group_keys <- schedule$split_group_keys
      affinity_metrics$mgcv_residual_affinity_enabled <- 1L
      affinity_metrics$mgcv_residual_affinity_group_count <-
        schedule$group_count
      affinity_metrics$mgcv_residual_affinity_task_count <-
        schedule$task_count
      affinity_metrics$mgcv_residual_affinity_worker_count <-
        schedule$worker_count
      affinity_metrics$mgcv_residual_affinity_max_group_size <-
        schedule$max_group_size
      affinity_metrics$mgcv_residual_affinity_mean_group_size <-
        schedule$mean_group_size
      affinity_metrics$mgcv_residual_affinity_load_imbalance <-
        schedule$load_imbalance
      affinity_metrics$mgcv_residual_affinity_split_group_count <-
        schedule$split_group_count
      affinity_metrics$mgcv_residual_affinity_split_group_tasks <-
        schedule$split_group_tasks
      affinity_metrics$mgcv_residual_affinity_split_group_pieces <-
        schedule$split_group_pieces
      if (identical(mgcv_residual_affinity_mode, "target_s")) {
        affinity_metrics$mgcv_residual_owner_schedule_build_ms <-
          (proc.time()[["elapsed"]] - affinity_schedule_start) * 1000
      }
      same_s_prefill_chunk_enabled <-
        isTRUE(mgcv_residual_same_s_prefill_enabled) &&
          !isTRUE(mgcv_residual_prefetch_level_enabled) &&
          ord > 0L
      same_s_setup_chunk_enabled <-
        isTRUE(mgcv_residual_same_s_setup_chunk_enabled) &&
          !isTRUE(mgcv_residual_prefetch_level_enabled) &&
          ord > 0L
      res_chunks <- parallel::mclapply(
        schedule$chunks, function(chunk) {
          worker_start <- proc.time()[["elapsed"]]
          prefill <- if (isTRUE(same_s_prefill_chunk_enabled)) {
            fastkpc_legacy_mgcv_cpp_same_s_prefill_chunk(
              chunk = chunk,
              ind = ind,
              G_l = G_l,
              seq_p = seq_p,
              ord = ord,
              data = data,
              env = env,
              cache_env = mgcv_residual_cache_env,
              condition_threshold =
                mgcv_residual_cpp_backend_condition_threshold,
              native_s_size_limit =
                mgcv_residual_cpp_backend_native_s_size_limit
            )
          } else {
            list(keys = character(), metrics = fastkpc_legacy_runtime_zero())
          }
          chunk_batch <- if (isTRUE(same_s_setup_chunk_enabled)) {
            edge_test_chunk_same_s(chunk)
          } else {
            list(
              results = lapply(chunk, edge_test),
              metrics = fastkpc_legacy_runtime_zero()
            )
          }
          chunk_results <- chunk_batch$results
          if (length(prefill$keys) > 0L) {
            used_keys <- unique(unlist(lapply(
              chunk_results,
              function(item) item[[9L]]$mgcv_residual_keys
            ), use.names = FALSE))
            prefill$metrics$mgcv_cpp_same_s_prefill_unused_count <-
              as.integer(sum(!(prefill$keys %in% used_keys)))
          }
          worker_metrics <- fastkpc_legacy_runtime_add(
            prefill$metrics, chunk_batch$metrics
          )
          list(
            results = chunk_results,
            prefill_metrics = worker_metrics,
            elapsed_ms = (proc.time()[["elapsed"]] - worker_start) * 1000
          )
        },
        mc.cores = length(schedule$chunks), mc.set.seed = FALSE,
        mc.cleanup = TRUE, mc.allow.recursive = FALSE,
        mc.preschedule = TRUE
      )
      prefill_worker_metrics <- lapply(res_chunks, `[[`, "prefill_metrics")
      if (length(prefill_worker_metrics) > 0L) {
        affinity_metrics <- Reduce(
          fastkpc_legacy_runtime_add,
          prefill_worker_metrics,
          init = affinity_metrics
        )
      }
      if (identical(mgcv_residual_affinity_mode, "target_s")) {
        worker_elapsed_ms <- vapply(
          res_chunks,
          function(chunk) as.numeric(chunk$elapsed_ms),
          numeric(1L)
        )
        worker_results <- lapply(res_chunks, `[[`, "results")
        worker_metrics <- lapply(worker_results, function(chunk_results) {
          Reduce(
            fastkpc_legacy_runtime_add,
            lapply(chunk_results, function(item) item[[9L]]),
            init = fastkpc_legacy_runtime_zero()
          )
        })
        worker_task_counts <- lengths(worker_results)
        worker_fit_counts <- vapply(
          worker_metrics,
          function(metrics) as.integer(metrics$mgcv_fit_count),
          integer(1L)
        )
        worker_cache_hits <- vapply(
          worker_metrics,
          function(metrics) as.integer(metrics$mgcv_cache_hit_count),
          integer(1L)
        )
        worker_residual_ms <- vapply(
          worker_metrics,
          function(metrics) as.numeric(metrics$residual_ms),
          numeric(1L)
        )
        affinity_metrics$mgcv_residual_owner_worker_max_ms <-
          max(worker_elapsed_ms)
        affinity_metrics$mgcv_residual_owner_worker_median_ms <-
          stats::median(worker_elapsed_ms)
        affinity_metrics$mgcv_residual_owner_worker_elapsed_imbalance <-
          if (stats::median(worker_elapsed_ms) > 0) {
            max(worker_elapsed_ms) / stats::median(worker_elapsed_ms)
          } else {
            0
          }
        affinity_metrics$mgcv_residual_owner_worker_task_max <-
          max(worker_task_counts)
        affinity_metrics$mgcv_residual_owner_worker_task_median <-
          stats::median(worker_task_counts)
        affinity_metrics$mgcv_residual_owner_worker_fit_max <-
          max(worker_fit_counts)
        affinity_metrics$mgcv_residual_owner_worker_fit_median <-
          stats::median(worker_fit_counts)
        affinity_metrics$mgcv_residual_owner_worker_cache_hit_max <-
          max(worker_cache_hits)
        affinity_metrics$mgcv_residual_owner_worker_cache_hit_median <-
          stats::median(worker_cache_hits)
        affinity_metrics$mgcv_residual_owner_worker_residual_ms_max <-
          max(worker_residual_ms)
        affinity_metrics$mgcv_residual_owner_worker_residual_ms_median <-
          stats::median(worker_residual_ms)
      }
      res <- unlist(lapply(res_chunks, `[[`, "results"), recursive = FALSE)
      res[order(vapply(res, function(item) as.integer(item[[1L]]),
                       integer(1L)))]
    } else if (.Platform$OS.type == "unix" && workers > 1L) {
      parallel::mclapply(
        edge_indices, edge_test, mc.cores = workers, mc.set.seed = FALSE,
        mc.cleanup = TRUE, mc.allow.recursive = FALSE,
        mc.preschedule = isTRUE(mgcv_residual_cache_enabled)
      )
    } else {
      lapply(edge_indices, edge_test)
    }

    metrics_level <- fastkpc_legacy_runtime_zero()
    metrics_level <- fastkpc_legacy_runtime_add(
      metrics_level, prefetch_metrics
    )
    metrics_level <- fastkpc_legacy_runtime_add(
      metrics_level, affinity_metrics
    )
    mgcv_residual_key_chunks <- list()
    mgcv_s_key_chunks <- list()
    mgcv_residual_keys_by_worker <- vector("list", workers)
    mgcv_split_residual_keys_by_worker <- vector("list", workers)
    for (p_obj in res) {
      i <- p_obj[[1L]]
      x <- ind[i, 1L]
      y <- ind[i, 2L]
      n_edgetests[ord1] <- n_edgetests[ord1] + p_obj[[5L]]
      pMax[x, y] <- p_obj[[6L]]
      pMax[y, x] <- p_obj[[7L]]
      G[x, y] <- G[y, x] <- p_obj[[2L]]
      if (!isTRUE(p_obj[[2L]])) {
        if (!is.null(p_obj[[3L]])) sepset[[x]][[y]] <- p_obj[[3L]]
        if (!is.null(p_obj[[4L]])) sepset[[y]][[x]] <- p_obj[[4L]]
        level_log[[length(level_log) + 1L]] <- list(
          x = x, y = y, S_xy = p_obj[[3L]], S_yx = p_obj[[4L]]
        )
      }
      done <- done & p_obj[[8L]]
      edge_metrics <- p_obj[[9L]]
      if (length(edge_metrics$mgcv_residual_keys) > 0L) {
        if (isTRUE(affinity_enabled)) {
          worker_id <- affinity_worker_by_edge[[as.character(i)]]
          if (length(worker_id) == 1L && !is.na(worker_id) &&
              worker_id > 0L) {
            mgcv_residual_keys_by_worker[[worker_id]] <- c(
              mgcv_residual_keys_by_worker[[worker_id]],
              edge_metrics$mgcv_residual_keys
            )
            group_key <- affinity_group_by_edge[[as.character(i)]]
            if (length(group_key) == 1L &&
                group_key %in% affinity_split_group_keys) {
              mgcv_split_residual_keys_by_worker[[worker_id]] <- c(
                mgcv_split_residual_keys_by_worker[[worker_id]],
                edge_metrics$mgcv_residual_keys
              )
            }
          }
        }
        mgcv_residual_key_chunks[[length(mgcv_residual_key_chunks) + 1L]] <-
          edge_metrics$mgcv_residual_keys
        edge_metrics$mgcv_residual_keys <- character()
      }
      if (length(edge_metrics$mgcv_s_keys) > 0L) {
        mgcv_s_key_chunks[[length(mgcv_s_key_chunks) + 1L]] <-
          edge_metrics$mgcv_s_keys
        edge_metrics$mgcv_s_keys <- character()
      }
      metrics_level <- fastkpc_legacy_runtime_add(metrics_level, edge_metrics)
    }
    if (length(mgcv_residual_key_chunks) > 0L) {
      metrics_level$mgcv_residual_keys <-
        unlist(mgcv_residual_key_chunks, use.names = FALSE)
    }
    if (length(mgcv_s_key_chunks) > 0L) {
      metrics_level$mgcv_s_keys <- unlist(mgcv_s_key_chunks, use.names = FALSE)
    }
    prefetch_consumed_keys <- if (isTRUE(mgcv_residual_prefetch_level_enabled)) {
      unique(metrics_level$mgcv_residual_keys)
    } else {
      character()
    }
    metrics_level <- fastkpc_legacy_runtime_finalize_mgcv_keys(metrics_level)
    if (isTRUE(mgcv_residual_prefetch_level_enabled)) {
      prefetched_keys <- mgcv_residual_prefetch_cache$keys
      consumed_prefetched_keys <- intersect(prefetch_consumed_keys,
                                           prefetched_keys)
      metrics_level$mgcv_prefetch_consumed_key_count <-
        as.integer(length(consumed_prefetched_keys))
      metrics_level$mgcv_prefetch_unused_key_count <- as.integer(max(
        0L,
        length(prefetched_keys) - length(consumed_prefetched_keys)
      ))
    }
    metrics_level$mgcv_residual_cache_theoretical_hit_count <-
      as.integer(metrics_level$mgcv_duplicate_residual_key_count)
    metrics_level$mgcv_residual_cache_realized_hit_count <-
      as.integer(metrics_level$mgcv_cache_hit_count)
    metrics_level$mgcv_residual_cache_lost_duplicate_count <- as.integer(max(
      0L,
      metrics_level$mgcv_residual_cache_theoretical_hit_count -
        metrics_level$mgcv_residual_cache_realized_hit_count
    ))
    if (isTRUE(affinity_enabled)) {
      metrics_level$mgcv_residual_cache_lost_cross_worker_count <-
        fastkpc_legacy_mgcv_residual_key_worker_loss(
          mgcv_residual_keys_by_worker
        )
      metrics_level$mgcv_residual_cache_lost_split_s_group_count <-
        fastkpc_legacy_mgcv_residual_key_worker_loss(
          mgcv_split_residual_keys_by_worker
        )
    } else {
      metrics_level$mgcv_residual_cache_lost_cross_worker_count <-
        metrics_level$mgcv_residual_cache_lost_duplicate_count
      metrics_level$mgcv_residual_cache_lost_split_s_group_count <- 0L
    }
    metrics_level$mgcv_residual_cache_lost_cross_level_count <- 0L
    if (identical(mgcv_residual_affinity_mode, "target_s")) {
      metrics_level$mgcv_residual_owner_realized_hit_count <-
        as.integer(metrics_level$mgcv_residual_cache_realized_hit_count)
      metrics_level$mgcv_residual_owner_lost_duplicate_count <-
        as.integer(metrics_level$mgcv_residual_cache_lost_duplicate_count)
    }

    n_edges[ord1] <- sum(G) / 2
    level_logs[[ord1]] <- level_log
    level_metrics[[ord1]] <- metrics_level
    if (ord > 0L && isTRUE(n_edges[ord1] == n_edges[ord])) break
    ord <- ord + 1L
  }

  if (p > 1L) {
    for (i in seq_len(p - 1L)) {
      for (j in seq.int(i + 1L, p)) {
        pMax[i, j] <- pMax[j, i] <- max(pMax[i, j], pMax[j, i])
      }
    }
  }
  colnames(G) <- rownames(G) <- labels
  dimnames(pMax) <- list(labels, labels)
  elapsed_ms <- (proc.time()[["elapsed"]] - total_start) * 1000
  total_tests <- sum(n_edgetests)
  runtime_total <- Reduce(
    fastkpc_legacy_runtime_add, level_metrics,
    init = fastkpc_legacy_runtime_zero()
  )
  runtime_by_level <- fastkpc_legacy_runtime_frame(level_metrics, n_edgetests)
  prefetch_potential_by_level <-
    fastkpc_legacy_mgcv_prefetch_potential_frame(runtime_by_level, nrow(data))
  prefetch_runtime_by_level <-
    fastkpc_legacy_mgcv_prefetch_runtime_frame(runtime_by_level)
  prefetch_sum <- function(name) {
    if (name %in% names(prefetch_potential_by_level)) {
      sum(prefetch_potential_by_level[[name]])
    } else {
      0
    }
  }
  prefetch_max <- function(name) {
    if (name %in% names(prefetch_potential_by_level) &&
        nrow(prefetch_potential_by_level) > 0L) {
      max(prefetch_potential_by_level[[name]])
    } else {
      0
    }
  }
  cache <- list(requests = 0L, hits = 0L, computations = 0L)

  list(
    adjacency = G,
    sepsets = sepset,
    pMax = pMax,
    n.edgetests = as.integer(n_edgetests),
    max.ord = as.integer(ord - 1L),
    n.edges = n_edges,
    per.level.log = level_logs,
    backend = "cpu",
    residual_device = "cpu",
    residual_device_reason = "compatible legacy parallel skeleton executes legacy mgcv on CPU",
    residual_backend = "legacy-mgcv",
    verifier_backend = NA_character_,
    residual_backend_params = "",
    residual_cache = cache,
    ci_method = ic.method,
    ci_backend = if (identical(ic.method, "dcc.gamma")) {
      "legacy-dcov.gamma"
    } else {
      "native-cpu"
    },
    ci_backend_reason = "legacy-compatible parallel skeleton",
    ci_diagnostics = list(
      ci_dcc_gamma_tests = if (identical(ic.method, "dcc.gamma")) {
        as.integer(total_tests)
      } else {
        0L
      },
      ci_hsic_gamma_tests = if (identical(ic.method, "hsic.gamma")) {
        as.integer(total_tests)
      } else {
        0L
      },
      ci_hsic_perm_tests = if (identical(ic.method, "hsic.perm")) {
        as.integer(total_tests)
      } else {
        0L
      },
      ci_hsic_permutation_replicates = 0L,
      ci_hsic_gamma_cuda_tests = 0L,
      ci_hsic_perm_cuda_tests = 0L,
      ci_hsic_cuda_batches = 0L,
      ci_hsic_cuda_pairs = 0L,
      ci_hsic_cuda_fallback_tests = 0L,
      ci_hsic_cuda_memory_bytes = 0,
      ci_hsic_cuda_max_n = 0L,
      ci_hsic_cuda_max_batch_pairs = 0L
    ),
    scheduler = "legacy-parallel",
    scheduler_diagnostics = list(
      legacy_runtime_by_level = runtime_by_level,
      legacy_mgcv_prefetch_potential_by_level = prefetch_potential_by_level,
      legacy_mgcv_prefetch_by_level = prefetch_runtime_by_level,
      summary = list(
        tasks_planned = as.integer(total_tests),
        tasks_evaluated = as.integer(total_tests),
        tests_replayed = as.integer(total_tests),
        trace_level = "summary",
        tasks_ignored_after_delete = 0L,
        unique_residual_requests = 0L,
        residual_cache_requests = 0L,
        residual_cache_hits = 0L,
        residual_cache_misses = 0L,
        residual_cache_computations = 0L,
        residual_cache_full_hit_events = 0L,
        residual_cache_partial_hit_events = 0L,
        residual_cache_full_miss_events = 0L,
        residual_cache_target_computations = 0L,
        residual_cache_cuda_batch_calls = 0L,
        residual_cache_cuda_single_target_calls = 0L,
        residual_cache_cuda_solve_calls = 0L,
        residual_cache_cuda_api_calls = 0L,
        residual_cache_cuda_batch_api_calls = 0L,
        residual_cache_cuda_single_target_api_calls = 0L,
        residual_cache_cuda_target_solves = 0L,
        dcov_batches = 0L,
        residual_batches = 0L,
        legacy_parallel_workers = as.integer(num_cores),
        legacy_parallel_worker_count = as.integer(num_cores),
        legacy_parallel_elapsed_ms = elapsed_ms,
        legacy_scheduler_elapsed_ms = elapsed_ms,
        legacy_ci_total_ms = as.numeric(runtime_total$ci_total_ms),
        legacy_residual_total_ms =
          as.numeric(runtime_total$residual_ms),
        legacy_dcov_gamma_ms =
          as.numeric(runtime_total$dcov_gamma_ms),
        legacy_dcov_input_ms =
          as.numeric(runtime_total$dcov_input_ms),
        legacy_dcov_h_ms =
          as.numeric(runtime_total$dcov_h_ms),
        legacy_dcov_distance_ms =
          as.numeric(runtime_total$dcov_distance_ms),
        legacy_dcov_lowrank_ms =
          as.numeric(runtime_total$dcov_lowrank_ms),
        legacy_dcov_statistic_ms =
          as.numeric(runtime_total$dcov_statistic_ms),
        legacy_dcov_moment_ms =
          as.numeric(runtime_total$dcov_moment_ms),
        legacy_dcov_pgamma_ms =
          as.numeric(runtime_total$dcov_pgamma_ms),
        legacy_dcov_output_ms =
          as.numeric(runtime_total$dcov_output_ms),
        legacy_dcov_unaccounted_ms =
          as.numeric(runtime_total$dcov_unaccounted_ms),
        legacy_dcov_backend = dcov_backend,
        legacy_dcov_r_backend_count =
          as.integer(runtime_total$dcov_r_backend_count),
        legacy_dcov_cpp_backend_count =
          as.integer(runtime_total$dcov_cpp_backend_count),
        legacy_dcov_cpp_backend_ms =
          as.numeric(runtime_total$dcov_cpp_backend_ms),
        legacy_dcov_cpp_backend_error_count =
          as.integer(runtime_total$dcov_cpp_backend_error_count),
        legacy_dcov_cpp_backend_fallback_count =
          as.integer(runtime_total$dcov_cpp_backend_fallback_count),
        legacy_dcov_cpp_backend_max_p_diff =
          as.numeric(runtime_total$dcov_cpp_backend_max_p_diff),
        legacy_dcov_cpp_backend_decision_flip_count =
          as.integer(runtime_total$dcov_cpp_backend_decision_flip_count),
        legacy_dcov_cpp_shadow_count =
          as.integer(runtime_total$dcov_cpp_shadow_count),
        legacy_dcov_cpp_shadow_ms =
          as.numeric(runtime_total$dcov_cpp_shadow_ms),
        legacy_dcov_cpp_shadow_max_p_diff =
          as.numeric(runtime_total$dcov_cpp_shadow_max_p_diff),
        legacy_dcov_cpp_shadow_max_nV2_diff =
          as.numeric(runtime_total$dcov_cpp_shadow_max_nV2_diff),
        legacy_dcov_cpp_shadow_max_mean_diff =
          as.numeric(runtime_total$dcov_cpp_shadow_max_mean_diff),
        legacy_dcov_cpp_shadow_max_variance_diff =
          as.numeric(runtime_total$dcov_cpp_shadow_max_variance_diff),
        legacy_dcov_cpp_shadow_decision_flip_count =
          as.integer(runtime_total$dcov_cpp_shadow_decision_flip_count),
        legacy_dcov_cpp_shadow_error_count =
          as.integer(runtime_total$dcov_cpp_shadow_error_count),
        legacy_dcov_cpp_shadow_near_alpha_count =
          as.integer(runtime_total$dcov_cpp_shadow_near_alpha_count),
        legacy_dcov_cpp_input_ms =
          as.numeric(runtime_total$dcov_cpp_input_ms),
        legacy_dcov_cpp_distance_ms =
          as.numeric(runtime_total$dcov_cpp_distance_ms),
        legacy_dcov_cpp_lowrank_ms =
          as.numeric(runtime_total$dcov_cpp_lowrank_ms),
        legacy_dcov_cpp_lowrank_eig_ms =
          as.numeric(runtime_total$dcov_cpp_lowrank_eig_ms),
        legacy_dcov_cpp_lowrank_select_ms =
          as.numeric(runtime_total$dcov_cpp_lowrank_select_ms),
        legacy_dcov_cpp_lowrank_center_ms =
          as.numeric(runtime_total$dcov_cpp_lowrank_center_ms),
        legacy_dcov_cpp_lowrank_unaccounted_ms =
          as.numeric(runtime_total$dcov_cpp_lowrank_unaccounted_ms),
        legacy_dcov_cpp_lowrank_full_eig_count =
          as.integer(runtime_total$dcov_cpp_lowrank_full_eig_count),
        legacy_dcov_cpp_lowrank_spectra_count =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_count),
        legacy_dcov_cpp_lowrank_spectra_converged_count =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_converged_count),
        legacy_dcov_cpp_lowrank_spectra_failed_count =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_failed_count),
        legacy_dcov_cpp_lowrank_spectra_fallback_full_eig_count =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_fallback_full_eig_count),
        legacy_dcov_cpp_lowrank_spectra_iterations =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_iterations),
        legacy_dcov_cpp_lowrank_spectra_nconv =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_nconv),
        legacy_dcov_cpp_lowrank_spectra_ncv =
          as.integer(runtime_total$dcov_cpp_lowrank_spectra_ncv),
        legacy_dcov_cpp_lowrank_spectra_tol =
          as.numeric(runtime_total$dcov_cpp_lowrank_spectra_tol),
        legacy_dcov_cpp_statistic_ms =
          as.numeric(runtime_total$dcov_cpp_statistic_ms),
        legacy_dcov_cpp_moment_ms =
          as.numeric(runtime_total$dcov_cpp_moment_ms),
        legacy_dcov_cpp_pgamma_ms =
          as.numeric(runtime_total$dcov_cpp_pgamma_ms),
        legacy_dcov_cpp_accounted_ms =
          as.numeric(runtime_total$dcov_cpp_accounted_ms),
        legacy_dcov_cpp_unaccounted_ms =
          as.numeric(runtime_total$dcov_cpp_unaccounted_ms),
        legacy_dcov_cpp_overhead_ms =
          as.numeric(runtime_total$dcov_cpp_overhead_ms),
        legacy_direct_ci_count =
          as.integer(runtime_total$direct_ci_count),
        legacy_conditional_ci_count =
          as.integer(runtime_total$conditional_ci_count),
        legacy_mgcv_residual_backend = mgcv_residual_backend,
        legacy_mgcv_r_backend_count =
          as.integer(runtime_total$mgcv_r_backend_count),
        legacy_mgcv_cpp_backend_enabled =
          isTRUE(as.integer(runtime_total$mgcv_cpp_backend_enabled) > 0L),
        legacy_mgcv_cpp_backend_count =
          as.integer(runtime_total$mgcv_cpp_backend_count),
        legacy_mgcv_cpp_backend_native_count =
          as.integer(runtime_total$mgcv_cpp_backend_native_count),
        legacy_mgcv_cpp_backend_fallback_count =
          as.integer(runtime_total$mgcv_cpp_backend_fallback_count),
        legacy_mgcv_cpp_backend_high_condition_fallback_count =
          as.integer(runtime_total$mgcv_cpp_backend_high_condition_fallback_count),
        legacy_mgcv_cpp_backend_outside_envelope_fallback_count =
          as.integer(runtime_total$mgcv_cpp_backend_outside_envelope_fallback_count),
        legacy_mgcv_cpp_backend_error_count =
          as.integer(runtime_total$mgcv_cpp_backend_error_count),
        legacy_mgcv_cpp_backend_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_ms),
        legacy_mgcv_cpp_backend_input_setup_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_input_setup_ms),
        legacy_mgcv_cpp_backend_gam_fit_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_gam_fit_ms),
        legacy_mgcv_cpp_backend_sp_extract_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_sp_extract_ms),
        legacy_mgcv_cpp_backend_setup_extract_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_setup_extract_ms),
        legacy_mgcv_cpp_backend_condition_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_condition_ms),
        legacy_mgcv_cpp_backend_native_solve_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_native_solve_ms),
        legacy_mgcv_cpp_backend_fallback_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_fallback_ms),
        legacy_mgcv_cpp_backend_s_size_0_count =
          as.integer(runtime_total$mgcv_cpp_backend_s_size_0_count),
        legacy_mgcv_cpp_backend_s_size_1_count =
          as.integer(runtime_total$mgcv_cpp_backend_s_size_1_count),
        legacy_mgcv_cpp_backend_s_size_2_count =
          as.integer(runtime_total$mgcv_cpp_backend_s_size_2_count),
        legacy_mgcv_cpp_backend_s_size_gt2_count =
          as.integer(runtime_total$mgcv_cpp_backend_s_size_gt2_count),
        legacy_mgcv_cpp_backend_native_s_size_0_count =
          as.integer(runtime_total$mgcv_cpp_backend_native_s_size_0_count),
        legacy_mgcv_cpp_backend_native_s_size_1_count =
          as.integer(runtime_total$mgcv_cpp_backend_native_s_size_1_count),
        legacy_mgcv_cpp_backend_native_s_size_2_count =
          as.integer(runtime_total$mgcv_cpp_backend_native_s_size_2_count),
        legacy_mgcv_cpp_backend_native_s_size_gt2_count =
          as.integer(runtime_total$mgcv_cpp_backend_native_s_size_gt2_count),
        legacy_mgcv_cpp_backend_fallback_s_size_0_count =
          as.integer(runtime_total$mgcv_cpp_backend_fallback_s_size_0_count),
        legacy_mgcv_cpp_backend_fallback_s_size_1_count =
          as.integer(runtime_total$mgcv_cpp_backend_fallback_s_size_1_count),
        legacy_mgcv_cpp_backend_fallback_s_size_2_count =
          as.integer(runtime_total$mgcv_cpp_backend_fallback_s_size_2_count),
        legacy_mgcv_cpp_backend_fallback_s_size_gt2_count =
          as.integer(runtime_total$mgcv_cpp_backend_fallback_s_size_gt2_count),
        legacy_mgcv_cpp_backend_same_s_native_group_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_native_group_count),
        legacy_mgcv_cpp_backend_same_s_native_target_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_native_target_count),
        legacy_mgcv_cpp_backend_same_s_native_max_targets =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_native_max_targets),
        legacy_mgcv_cpp_backend_same_s_native_mean_targets =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_native_mean_targets),
        legacy_mgcv_cpp_backend_same_s_native_reuse_opportunity_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_native_reuse_opportunity_count),
        legacy_mgcv_cpp_backend_same_s_native_setup_reuse_ratio =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_native_setup_reuse_ratio),
        legacy_mgcv_cpp_backend_same_s_sp_native_group_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_sp_native_group_count),
        legacy_mgcv_cpp_backend_same_s_sp_native_target_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_sp_native_target_count),
        legacy_mgcv_cpp_backend_same_s_sp_native_max_targets =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_sp_native_max_targets),
        legacy_mgcv_cpp_backend_same_s_sp_native_mean_targets =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_sp_native_mean_targets),
        legacy_mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_sp_native_reuse_opportunity_count),
        legacy_mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_sp_native_setup_reuse_ratio),
        legacy_mgcv_cpp_backend_same_s_setup_native_group_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_setup_native_group_count),
        legacy_mgcv_cpp_backend_same_s_setup_native_target_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_setup_native_target_count),
        legacy_mgcv_cpp_backend_same_s_setup_native_max_targets =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_setup_native_max_targets),
        legacy_mgcv_cpp_backend_same_s_setup_native_mean_targets =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_native_mean_targets),
        legacy_mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count =
          as.integer(runtime_total$mgcv_cpp_backend_same_s_setup_native_reuse_opportunity_count),
        legacy_mgcv_cpp_backend_same_s_setup_native_reuse_ratio =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_native_reuse_ratio),
        legacy_mgcv_cpp_backend_same_s_setup_input_potential_saved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_input_potential_saved_ms),
        legacy_mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_extract_potential_saved_ms),
        legacy_mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_condition_potential_saved_ms),
        legacy_mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_setup_structure_potential_saved_ms),
        legacy_mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_sp_native_solve_potential_saved_ms),
        legacy_mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_sp_native_solve_reuse_ratio),
        legacy_mgcv_cpp_backend_same_s_gam_fit_preserved_ms =
          as.numeric(runtime_total$mgcv_cpp_backend_same_s_gam_fit_preserved_ms),
        legacy_mgcv_cpp_same_s_prefill_enabled =
          isTRUE(as.integer(runtime_total$mgcv_cpp_same_s_prefill_enabled) > 0L),
        legacy_mgcv_cpp_same_s_prefill_group_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_group_count),
        legacy_mgcv_cpp_same_s_prefill_target_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_target_count),
        legacy_mgcv_cpp_same_s_prefill_cache_insert_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_cache_insert_count),
        legacy_mgcv_cpp_same_s_prefill_existing_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_existing_count),
        legacy_mgcv_cpp_same_s_prefill_unused_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_unused_count),
        legacy_mgcv_cpp_same_s_prefill_ms =
          as.numeric(runtime_total$mgcv_cpp_same_s_prefill_ms),
        legacy_mgcv_cpp_same_s_prefill_error_count =
          as.integer(runtime_total$mgcv_cpp_same_s_prefill_error_count),
        legacy_mgcv_cpp_same_s_setup_provider_enabled =
          isTRUE(as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_enabled) > 0L),
        legacy_mgcv_cpp_same_s_setup_provider_group_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_group_count),
        legacy_mgcv_cpp_same_s_setup_provider_target_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_target_count),
        legacy_mgcv_cpp_same_s_setup_provider_template_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_template_count),
        legacy_mgcv_cpp_same_s_setup_provider_reuse_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_reuse_count),
        legacy_mgcv_cpp_same_s_setup_provider_setup_ms =
          as.numeric(runtime_total$mgcv_cpp_same_s_setup_provider_setup_ms),
        legacy_mgcv_cpp_same_s_setup_provider_error_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_error_count),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_enabled =
          isTRUE(as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_enabled) > 0L),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_count),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_group_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_group_count),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_target_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_target_count),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_cache_insert_count),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_ms =
          as.numeric(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_ms),
        legacy_mgcv_cpp_same_s_setup_provider_chunk_error_count =
          as.integer(runtime_total$mgcv_cpp_same_s_setup_provider_chunk_error_count),
        legacy_mgcv_cpp_backend_native_s_size_limit =
          as.numeric(runtime_total$mgcv_cpp_backend_native_s_size_limit),
        legacy_mgcv_cpp_backend_condition_threshold =
          as.numeric(runtime_total$mgcv_cpp_backend_condition_threshold),
        legacy_mgcv_fit_count =
          as.integer(runtime_total$mgcv_fit_count),
        legacy_mgcv_residual_request_count =
          as.integer(runtime_total$mgcv_residual_request_count),
        legacy_mgcv_cache_hit_count =
          as.integer(runtime_total$mgcv_cache_hit_count),
        legacy_mgcv_cache_miss_count =
          as.integer(runtime_total$mgcv_cache_miss_count),
        legacy_mgcv_residual_cache_hit_key_count =
          as.integer(runtime_total$mgcv_residual_cache_hit_key_count),
        legacy_mgcv_residual_cache_miss_key_count =
          as.integer(runtime_total$mgcv_residual_cache_miss_key_count),
        legacy_mgcv_residual_cache_miss_s_group_count =
          as.integer(runtime_total$mgcv_residual_cache_miss_s_group_count),
        legacy_mgcv_residual_cache_miss_s_total_targets =
          as.integer(runtime_total$mgcv_residual_cache_miss_s_total_targets),
        legacy_mgcv_residual_cache_miss_s_max_targets =
          as.integer(runtime_total$mgcv_residual_cache_miss_s_max_targets),
        legacy_mgcv_residual_cache_miss_s_mean_targets =
          as.numeric(runtime_total$mgcv_residual_cache_miss_s_mean_targets),
        legacy_mgcv_residual_cache_miss_s_reuse_opportunity_count =
          as.integer(runtime_total$mgcv_residual_cache_miss_s_reuse_opportunity_count),
        legacy_mgcv_residual_cache_miss_s_reuse_ratio =
          as.numeric(runtime_total$mgcv_residual_cache_miss_s_reuse_ratio),
        legacy_mgcv_residual_cache_hit_count =
          as.integer(runtime_total$mgcv_cache_hit_count),
        legacy_mgcv_residual_cache_miss_count =
          as.integer(runtime_total$mgcv_cache_miss_count),
        legacy_mgcv_residual_cache_insert_count =
          as.integer(runtime_total$mgcv_residual_cache_insert_count),
        legacy_mgcv_residual_cache_entries =
          as.integer(runtime_total$mgcv_residual_cache_entries),
        legacy_mgcv_residual_cache_hit_ms =
          as.numeric(runtime_total$mgcv_residual_cache_hit_ms),
        legacy_mgcv_residual_cache_lookup_ms =
          as.numeric(runtime_total$mgcv_cache_lookup_ms),
        legacy_mgcv_residual_cache_store_ms =
          as.numeric(runtime_total$mgcv_residual_cache_store_ms),
        legacy_mgcv_fit_avoided_count =
          as.integer(runtime_total$mgcv_fit_avoided_count),
        legacy_mgcv_residual_affinity_enabled =
          isTRUE(as.integer(runtime_total$mgcv_residual_affinity_enabled) > 0L),
        legacy_mgcv_residual_affinity_group_count =
          as.integer(runtime_total$mgcv_residual_affinity_group_count),
        legacy_mgcv_residual_affinity_task_count =
          as.integer(runtime_total$mgcv_residual_affinity_task_count),
        legacy_mgcv_residual_affinity_worker_count =
          as.integer(runtime_total$mgcv_residual_affinity_worker_count),
        legacy_mgcv_residual_affinity_max_group_size =
          as.integer(runtime_total$mgcv_residual_affinity_max_group_size),
        legacy_mgcv_residual_affinity_mean_group_size =
          as.numeric(runtime_total$mgcv_residual_affinity_mean_group_size),
        legacy_mgcv_residual_affinity_load_imbalance =
          as.numeric(runtime_total$mgcv_residual_affinity_load_imbalance),
        legacy_mgcv_residual_affinity_split_group_count =
          as.integer(runtime_total$mgcv_residual_affinity_split_group_count),
        legacy_mgcv_residual_affinity_split_group_tasks =
          as.integer(runtime_total$mgcv_residual_affinity_split_group_tasks),
        legacy_mgcv_residual_affinity_split_group_pieces =
          as.integer(runtime_total$mgcv_residual_affinity_split_group_pieces),
        legacy_mgcv_residual_cache_theoretical_hit_count =
          as.integer(runtime_total$mgcv_residual_cache_theoretical_hit_count),
        legacy_mgcv_residual_cache_realized_hit_count =
          as.integer(runtime_total$mgcv_residual_cache_realized_hit_count),
        legacy_mgcv_residual_cache_lost_duplicate_count =
          as.integer(runtime_total$mgcv_residual_cache_lost_duplicate_count),
        legacy_mgcv_residual_cache_lost_cross_worker_count =
          as.integer(runtime_total$mgcv_residual_cache_lost_cross_worker_count),
        legacy_mgcv_residual_cache_lost_split_s_group_count =
          as.integer(runtime_total$mgcv_residual_cache_lost_split_s_group_count),
        legacy_mgcv_residual_cache_lost_cross_level_count =
          as.integer(runtime_total$mgcv_residual_cache_lost_cross_level_count),
        legacy_mgcv_residual_cache_cross_worker_loss_estimate =
          as.integer(runtime_total$mgcv_residual_cache_lost_cross_worker_count),
        legacy_mgcv_residual_owner_enabled =
          isTRUE(as.integer(runtime_total$mgcv_residual_owner_enabled) > 0L),
        legacy_mgcv_residual_owner_key_count =
          as.integer(runtime_total$mgcv_residual_owner_key_count),
        legacy_mgcv_residual_owner_task_count =
          as.integer(runtime_total$mgcv_residual_owner_task_count),
        legacy_mgcv_residual_owner_both_local_count =
          as.integer(runtime_total$mgcv_residual_owner_both_local_count),
        legacy_mgcv_residual_owner_one_local_count =
          as.integer(runtime_total$mgcv_residual_owner_one_local_count),
        legacy_mgcv_residual_owner_none_local_count =
          as.integer(runtime_total$mgcv_residual_owner_none_local_count),
        legacy_mgcv_residual_owner_conflict_count =
          as.integer(runtime_total$mgcv_residual_owner_conflict_count),
        legacy_mgcv_residual_owner_predicted_hit_count =
          as.integer(runtime_total$mgcv_residual_owner_predicted_hit_count),
        legacy_mgcv_residual_owner_realized_hit_count =
          as.integer(runtime_total$mgcv_residual_owner_realized_hit_count),
        legacy_mgcv_residual_owner_lost_duplicate_count =
          as.integer(runtime_total$mgcv_residual_owner_lost_duplicate_count),
        legacy_mgcv_residual_owner_load_imbalance =
          as.numeric(runtime_total$mgcv_residual_owner_load_imbalance),
        legacy_mgcv_residual_owner_spill_count =
          as.integer(runtime_total$mgcv_residual_owner_spill_count),
        legacy_mgcv_residual_owner_schedule_build_ms =
          as.numeric(runtime_total$mgcv_residual_owner_schedule_build_ms),
        legacy_mgcv_residual_owner_key_enum_ms =
          as.numeric(runtime_total$mgcv_residual_owner_key_enum_ms),
        legacy_mgcv_residual_owner_key_map_build_ms =
          as.numeric(runtime_total$mgcv_residual_owner_key_map_build_ms),
        legacy_mgcv_residual_owner_task_score_ms =
          as.numeric(runtime_total$mgcv_residual_owner_task_score_ms),
        legacy_mgcv_residual_owner_greedy_assign_ms =
          as.numeric(runtime_total$mgcv_residual_owner_greedy_assign_ms),
        legacy_mgcv_residual_owner_chunk_sort_ms =
          as.numeric(runtime_total$mgcv_residual_owner_chunk_sort_ms),
        legacy_mgcv_residual_owner_chunk_materialize_ms =
          as.numeric(runtime_total$mgcv_residual_owner_chunk_materialize_ms),
        legacy_mgcv_residual_owner_worker_max_ms =
          as.numeric(runtime_total$mgcv_residual_owner_worker_max_ms),
        legacy_mgcv_residual_owner_worker_median_ms =
          as.numeric(runtime_total$mgcv_residual_owner_worker_median_ms),
        legacy_mgcv_residual_owner_worker_elapsed_imbalance =
          as.numeric(runtime_total$mgcv_residual_owner_worker_elapsed_imbalance),
        legacy_mgcv_residual_owner_worker_task_max =
          as.integer(runtime_total$mgcv_residual_owner_worker_task_max),
        legacy_mgcv_residual_owner_worker_task_median =
          as.numeric(runtime_total$mgcv_residual_owner_worker_task_median),
        legacy_mgcv_residual_owner_worker_fit_max =
          as.integer(runtime_total$mgcv_residual_owner_worker_fit_max),
        legacy_mgcv_residual_owner_worker_fit_median =
          as.numeric(runtime_total$mgcv_residual_owner_worker_fit_median),
        legacy_mgcv_residual_owner_worker_cache_hit_max =
          as.integer(runtime_total$mgcv_residual_owner_worker_cache_hit_max),
        legacy_mgcv_residual_owner_worker_cache_hit_median =
          as.numeric(runtime_total$mgcv_residual_owner_worker_cache_hit_median),
        legacy_mgcv_residual_owner_worker_residual_ms_max =
          as.numeric(runtime_total$mgcv_residual_owner_worker_residual_ms_max),
        legacy_mgcv_residual_owner_worker_residual_ms_median =
          as.numeric(runtime_total$mgcv_residual_owner_worker_residual_ms_median),
        legacy_mgcv_fit_avoided_estimated_ms = {
          fit_count <- as.integer(runtime_total$mgcv_fit_count)
          if (fit_count > 0L) {
            as.numeric(runtime_total$mgcv_fit_call_ms) /
              fit_count * as.integer(runtime_total$mgcv_fit_avoided_count)
          } else {
            0
          }
        },
        legacy_mgcv_unique_residual_key_count =
          as.integer(runtime_total$mgcv_unique_residual_key_count),
        legacy_mgcv_duplicate_residual_key_count =
          as.integer(runtime_total$mgcv_duplicate_residual_key_count),
        legacy_mgcv_unique_target_s_count =
          as.integer(runtime_total$mgcv_unique_target_s_count),
        legacy_mgcv_unique_s_count =
          as.integer(runtime_total$mgcv_unique_s_count),
        legacy_mgcv_same_s_group_count =
          as.integer(runtime_total$mgcv_same_s_group_count),
        legacy_mgcv_same_s_total_targets =
          as.integer(runtime_total$mgcv_same_s_total_targets),
        legacy_mgcv_same_s_max_targets =
          as.integer(runtime_total$mgcv_same_s_max_targets),
        legacy_mgcv_same_s_mean_targets =
          as.numeric(runtime_total$mgcv_same_s_mean_targets),
        legacy_mgcv_same_s_reuse_opportunity_count =
          as.integer(runtime_total$mgcv_same_s_reuse_opportunity_count),
        legacy_mgcv_level_prefetch_request_count =
          as.integer(prefetch_sum("request_count")),
        legacy_mgcv_level_prefetch_unique_key_count =
          as.integer(prefetch_sum("unique_target_s_count")),
        legacy_mgcv_level_prefetch_theoretical_hit_count =
          as.integer(prefetch_sum("theoretical_hit_count")),
        legacy_mgcv_level_prefetch_current_hit_count =
          as.integer(prefetch_sum("current_hit_count")),
        legacy_mgcv_level_prefetch_fit_reduction_potential =
          as.integer(prefetch_sum("fit_reduction_potential")),
        legacy_mgcv_level_prefetch_payload_bytes =
          as.numeric(prefetch_sum("residual_payload_bytes")),
        legacy_mgcv_level_prefetch_max_level_payload_bytes =
          as.numeric(prefetch_max("residual_payload_bytes")),
        legacy_mgcv_level_prefetch_max_level_unique_keys =
          as.integer(prefetch_max("unique_target_s_count")),
        legacy_mgcv_prefetch_enabled =
          isTRUE(as.integer(runtime_total$mgcv_prefetch_enabled) > 0L),
        legacy_mgcv_prefetch_level_count =
          as.integer(runtime_total$mgcv_prefetch_level_count),
        legacy_mgcv_prefetch_key_count =
          as.integer(runtime_total$mgcv_prefetch_key_count),
        legacy_mgcv_prefetch_fit_count =
          as.integer(runtime_total$mgcv_prefetch_fit_count),
        legacy_mgcv_prefetch_fit_ms =
          as.numeric(runtime_total$mgcv_prefetch_fit_ms),
        legacy_mgcv_prefetch_collect_ms =
          as.numeric(runtime_total$mgcv_prefetch_collect_ms),
        legacy_mgcv_prefetch_matrix_build_ms =
          as.numeric(runtime_total$mgcv_prefetch_matrix_build_ms),
        legacy_mgcv_prefetch_payload_bytes =
          as.numeric(runtime_total$mgcv_prefetch_payload_bytes),
        legacy_mgcv_prefetch_max_level_payload_bytes =
          as.numeric(runtime_total$mgcv_prefetch_max_level_payload_bytes),
        legacy_mgcv_prefetch_lookup_ms =
          as.numeric(runtime_total$mgcv_prefetch_lookup_ms),
        legacy_mgcv_prefetch_ci_phase_ms =
          as.numeric(runtime_total$mgcv_prefetch_ci_phase_ms),
        legacy_mgcv_prefetch_error_count =
          as.integer(runtime_total$mgcv_prefetch_error_count),
        legacy_mgcv_prefetch_consumed_key_count =
          as.integer(runtime_total$mgcv_prefetch_consumed_key_count),
        legacy_mgcv_prefetch_unused_key_count =
          as.integer(runtime_total$mgcv_prefetch_unused_key_count),
        legacy_mgcv_key_build_ms =
          as.numeric(runtime_total$mgcv_key_build_ms),
        legacy_mgcv_cache_lookup_ms =
          as.numeric(runtime_total$mgcv_cache_lookup_ms),
        legacy_mgcv_formula_build_ms =
          as.numeric(runtime_total$mgcv_formula_build_ms),
        legacy_mgcv_data_subset_ms =
          as.numeric(runtime_total$mgcv_data_subset_ms),
        legacy_mgcv_fit_call_ms =
          as.numeric(runtime_total$mgcv_fit_call_ms),
        legacy_mgcv_residual_extract_ms =
          as.numeric(runtime_total$mgcv_residual_extract_ms),
        legacy_mgcv_result_store_ms =
          as.numeric(runtime_total$mgcv_result_store_ms),
        legacy_mgcv_unaccounted_ms =
          as.numeric(runtime_total$mgcv_unaccounted_ms),
        legacy_mgcv_s_size_0_count =
          as.integer(runtime_total$mgcv_s_size_0_count),
        legacy_mgcv_s_size_1_count =
          as.integer(runtime_total$mgcv_s_size_1_count),
        legacy_mgcv_s_size_2_count =
          as.integer(runtime_total$mgcv_s_size_2_count),
        legacy_mgcv_s_size_gt2_count =
          as.integer(runtime_total$mgcv_s_size_gt2_count),
        legacy_mgcv_cpp_shadow_enabled =
          isTRUE(as.integer(runtime_total$mgcv_cpp_shadow_enabled) > 0L),
        legacy_mgcv_cpp_shadow_count =
          as.integer(runtime_total$mgcv_cpp_shadow_count),
        legacy_mgcv_cpp_shadow_native_count =
          as.integer(runtime_total$mgcv_cpp_shadow_native_count),
        legacy_mgcv_cpp_shadow_fallback_count =
          as.integer(runtime_total$mgcv_cpp_shadow_fallback_count),
        legacy_mgcv_cpp_shadow_high_condition_fallback_count =
          as.integer(runtime_total$mgcv_cpp_shadow_high_condition_fallback_count),
        legacy_mgcv_cpp_shadow_outside_envelope_fallback_count =
          as.integer(runtime_total$mgcv_cpp_shadow_outside_envelope_fallback_count),
        legacy_mgcv_cpp_shadow_error_count =
          as.integer(runtime_total$mgcv_cpp_shadow_error_count),
        legacy_mgcv_cpp_shadow_residual_mismatch_count =
          as.integer(runtime_total$mgcv_cpp_shadow_residual_mismatch_count),
        legacy_mgcv_cpp_shadow_ms =
          as.numeric(runtime_total$mgcv_cpp_shadow_ms),
        legacy_mgcv_cpp_shadow_max_abs_diff =
          as.numeric(runtime_total$mgcv_cpp_shadow_max_abs_diff),
        legacy_mgcv_cpp_shadow_max_rel_l2 =
          as.numeric(runtime_total$mgcv_cpp_shadow_max_rel_l2),
        legacy_mgcv_cpp_shadow_native_s_size_limit =
          as.numeric(runtime_total$mgcv_cpp_shadow_native_s_size_limit),
        legacy_mgcv_cpp_shadow_condition_threshold =
          as.numeric(runtime_total$mgcv_cpp_shadow_condition_threshold),
        legacy_dcov_gamma_count =
          as.integer(runtime_total$dcov_gamma_count),
        legacy_fake_level0_test_count =
          as.integer(runtime_total$fake_level0_test_count)
      )
    ),
    precision_trace = NULL,
    precision_receipt = list(
      residual_backend_executed = "legacy-mgcv",
      ci_backend_executed = if (identical(ic.method, "dcc.gamma")) {
        "legacy-dcov.gamma"
      } else {
        "native-cpu"
      },
      timings = list(total_ms = elapsed_ms)
    )
  )
}

fastkpc_fixed_scenario <- function(seed = 4, n = 80) {
  set.seed(seed)
  z <- stats::runif(n)
  data <- cbind(
    x1 = z + stats::rnorm(n, sd = 0.2),
    x2 = z^2 + stats::rnorm(n, sd = 0.2),
    x3 = z,
    x4 = stats::rnorm(n)
  )
  list(
    data = data,
    alpha = 0.2,
    max_conditioning_size = 1L,
    description = "Fixed four-variable nonlinear scenario used by fastkpc MVP tests"
  )
}
