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
    mgcv_residual_keys = character(),
    mgcv_s_keys = character(),
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
    mgcv_residual_keys = c(a$mgcv_residual_keys, b$mgcv_residual_keys),
    mgcv_s_keys = c(a$mgcv_s_keys, b$mgcv_s_keys),
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
      mgcv_key_build_ms = numeric(), mgcv_cache_lookup_ms = numeric(),
      mgcv_formula_build_ms = numeric(), mgcv_data_subset_ms = numeric(),
      mgcv_fit_call_ms = numeric(), mgcv_residual_extract_ms = numeric(),
      mgcv_result_store_ms = numeric(), mgcv_unaccounted_ms = numeric(),
      mgcv_s_size_0_count = integer(), mgcv_s_size_1_count = integer(),
      mgcv_s_size_2_count = integer(), mgcv_s_size_gt2_count = integer(),
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
      mgcv_fit_count = as.integer(metrics$mgcv_fit_count),
      dcov_gamma_count = as.integer(metrics$dcov_gamma_count)
    )
  })
  do.call(rbind, rows)
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

fastkpc_legacy_mgcv_residual_cache_enabled <- function() {
  identical(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE", unset = ""), "1")
}

fastkpc_legacy_mgcv_residual_affinity_mode <- function() {
  tolower(Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY", unset = ""))
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
    }
  }
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
    load_imbalance = if (mean_load > 0) {
      as.numeric(max(loads) / mean_load)
    } else {
      0
    }
  )
}

fastkpc_legacy_run_mgcv_residual_pair <- function(
    metrics, data, x, y, S, env, cache_env = NULL, cache_enabled = FALSE) {
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

  if (!isTRUE(cache_enabled)) {
    data_subset_start <- proc.time()[["elapsed"]]
    xy_data <- data[, c(x, y)]
    s_data <- data[, S_int, drop = FALSE]
    metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
      (proc.time()[["elapsed"]] - data_subset_start) * 1000

    fit_start <- proc.time()[["elapsed"]]
    residuals <- env$regrXonS(xy_data, s_data)
    metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms +
      (proc.time()[["elapsed"]] - fit_start) * 1000
    metrics$mgcv_fit_count <- 2L
    metrics$mgcv_cache_miss_count <- 2L

    residual_extract_start <- proc.time()[["elapsed"]]
    residuals <- as.matrix(residuals)
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
        metrics$mgcv_result_store_ms
    )
    return(list(residuals = residuals, metrics = metrics))
  }

  if (is.null(cache_env)) cache_env <- new.env(parent = emptyenv())
  s_data <- NULL
  residual_cols <- vector("list", 2L)
  targets <- c(as.integer(x), as.integer(y))
  for (idx in seq_along(targets)) {
    key <- target_keys[[idx]]
    lookup_start <- proc.time()[["elapsed"]]
    hit <- exists(key, envir = cache_env, inherits = FALSE)
    if (isTRUE(hit)) {
      residual_cols[[idx]] <- get(key, envir = cache_env, inherits = FALSE)
      hit_elapsed <- (proc.time()[["elapsed"]] - lookup_start) * 1000
      metrics$mgcv_cache_hit_count <- metrics$mgcv_cache_hit_count + 1L
      metrics$mgcv_fit_avoided_count <- metrics$mgcv_fit_avoided_count + 1L
      metrics$mgcv_residual_cache_hit_ms <-
        metrics$mgcv_residual_cache_hit_ms + hit_elapsed
      metrics$mgcv_cache_lookup_ms <- metrics$mgcv_cache_lookup_ms +
        hit_elapsed
      next
    }
    metrics$mgcv_cache_lookup_ms <- metrics$mgcv_cache_lookup_ms +
      (proc.time()[["elapsed"]] - lookup_start) * 1000
    metrics$mgcv_cache_miss_count <- metrics$mgcv_cache_miss_count + 1L

    data_subset_start <- proc.time()[["elapsed"]]
    x_data <- data[, targets[[idx]], drop = FALSE]
    if (is.null(s_data)) s_data <- data[, S_int, drop = FALSE]
    metrics$mgcv_data_subset_ms <- metrics$mgcv_data_subset_ms +
      (proc.time()[["elapsed"]] - data_subset_start) * 1000

    fit_start <- proc.time()[["elapsed"]]
    target_residual <- as.numeric(env$regrXonS(x_data, s_data)[, 1L])
    metrics$mgcv_fit_call_ms <- metrics$mgcv_fit_call_ms +
      (proc.time()[["elapsed"]] - fit_start) * 1000
    metrics$mgcv_fit_count <- metrics$mgcv_fit_count + 1L

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
      metrics$mgcv_residual_cache_store_ms
  )
  list(residuals = residuals, metrics = metrics)
}

fastkpc_legacy_runtime_finalize_mgcv_keys <- function(metrics) {
  residual_keys <- metrics$mgcv_residual_keys
  s_keys <- metrics$mgcv_s_keys
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
  metrics$mgcv_residual_keys <- character()
  metrics$mgcv_s_keys <- character()
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
            cache_enabled = mgcv_residual_cache_enabled
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

    edge_indices <- seq_len(remaining_edge_tests)
    workers <- min(num_cores, length(edge_indices))
    affinity_enabled <- .Platform$OS.type == "unix" && workers > 1L &&
      ord > 0L && isTRUE(mgcv_residual_cache_enabled) &&
      identical(mgcv_residual_affinity_mode, "s")
    affinity_metrics <- fastkpc_legacy_runtime_zero()
    res <- if (isTRUE(affinity_enabled)) {
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
      group_keys <- vapply(edge_indices, edge_affinity_key, character(1L))
      task_weights <- vapply(edge_indices, edge_affinity_weight, numeric(1L))
      schedule <- fastkpc_legacy_mgcv_residual_affinity_chunks(
        edge_indices, group_keys, workers, task_weights = task_weights
      )
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
      res_chunks <- parallel::mclapply(
        schedule$chunks, function(chunk) lapply(chunk, edge_test),
        mc.cores = length(schedule$chunks), mc.set.seed = FALSE,
        mc.cleanup = TRUE, mc.allow.recursive = FALSE,
        mc.preschedule = TRUE
      )
      res <- unlist(res_chunks, recursive = FALSE)
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
      metrics_level, affinity_metrics
    )
    mgcv_residual_key_chunks <- list()
    mgcv_s_key_chunks <- list()
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
    metrics_level <- fastkpc_legacy_runtime_finalize_mgcv_keys(metrics_level)

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
        legacy_mgcv_fit_count =
          as.integer(runtime_total$mgcv_fit_count),
        legacy_mgcv_residual_request_count =
          as.integer(runtime_total$mgcv_residual_request_count),
        legacy_mgcv_cache_hit_count =
          as.integer(runtime_total$mgcv_cache_hit_count),
        legacy_mgcv_cache_miss_count =
          as.integer(runtime_total$mgcv_cache_miss_count),
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
        legacy_mgcv_residual_cache_theoretical_hit_count = {
          requests <- as.integer(runtime_total$mgcv_residual_request_count)
          unique_keys <- as.integer(runtime_total$mgcv_unique_residual_key_count)
          as.integer(max(0L, requests - unique_keys))
        },
        legacy_mgcv_residual_cache_realized_hit_count =
          as.integer(runtime_total$mgcv_cache_hit_count),
        legacy_mgcv_residual_cache_cross_worker_loss_estimate = {
          requests <- as.integer(runtime_total$mgcv_residual_request_count)
          unique_keys <- as.integer(runtime_total$mgcv_unique_residual_key_count)
          theoretical <- as.integer(max(0L, requests - unique_keys))
          realized <- as.integer(runtime_total$mgcv_cache_hit_count)
          as.integer(max(0L, theoretical - realized))
        },
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
