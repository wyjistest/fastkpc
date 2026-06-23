.fastkpc_source_dir <- function() {
  if (file.exists("fastkpc/src/rcpp_exports.cpp")) return(normalizePath("fastkpc"))
  stop("Cannot find fastkpc/src/rcpp_exports.cpp from current working directory",
       call. = FALSE)
}

build_fastkpc_native <- function(rebuild = FALSE, verbose = FALSE) {
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp is required to build fastkpc native code", call. = FALSE)
  }
  root <- .fastkpc_source_dir()
  if (rebuild || !exists("fast_dcov_exact_cpp_export", mode = "function")) {
    if (rebuild) {
      unlink(file.path(root, "src", "*.o"))
    }
    Rcpp::sourceCpp(
      file.path(root, "src", "rcpp_exports.cpp"),
      rebuild = rebuild,
      verbose = verbose
    )
  }
  invisible(TRUE)
}

fast_dcov_exact_cpp <- function(x, y, index = 1, legacy_index = TRUE) {
  build_fastkpc_native()
  fast_dcov_exact_cpp_export(
    as.numeric(x),
    as.numeric(y),
    as.numeric(index),
    isTRUE(legacy_index)
  )
}

fastkpc_mgcv_extract_gpu_spectral_score_batch_cpp <- function(
    eigenvectors, inv_chol, eigenvalues, y, Xty_null, sp_grid,
    tol = sqrt(.Machine$double.eps)) {
  build_fastkpc_native()
  y <- as.matrix(y)
  Xty_null <- as.matrix(Xty_null)
  storage.mode(y) <- "double"
  storage.mode(Xty_null) <- "double"
  mgcv_extract_gpu_spectral_score_batch_export(
    as.matrix(eigenvectors),
    as.matrix(inv_chol),
    as.numeric(eigenvalues),
    y,
    Xty_null,
    as.numeric(sp_grid),
    as.numeric(tol)
  )
}

kpc_tprs_residual_cpp_setup <- function(S, k = NA_integer_,
                                        tol = sqrt(.Machine$double.eps)) {
  build_fastkpc_native()
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  kpc_tprs_residual_cpp_setup_export(
    S,
    as.integer(if (is.na(k)) 0L else k),
    as.numeric(tol)
  )
}

fast_hsic_gamma_cpp <- function(x, y, sig = 1) {
  build_fastkpc_native()
  fast_hsic_gamma_cpp_export(
    as.numeric(x),
    as.numeric(y),
    as.numeric(sig)
  )
}

fast_hsic_perm_cpp <- function(x, y, sig = 1, replicates = 100L,
                               seed = NULL, include_observed = TRUE) {
  build_fastkpc_native()
  if (is.null(seed)) {
    seed <- NULL
  } else {
    seed <- as.integer(seed)
  }
  fast_hsic_perm_cpp_export(
    as.numeric(x),
    as.numeric(y),
    as.numeric(sig),
    as.integer(replicates),
    seed,
    isTRUE(include_observed)
  )
}

fast_skeleton_cpp <- function(data, alpha, max_conditioning_size,
                              index = 1, legacy_index = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_skeleton_cpp_export(
    data,
    as.numeric(alpha),
    as.integer(max_conditioning_size),
    as.numeric(index),
    isTRUE(legacy_index)
  )
}

fast_skeleton_cpp_cached <- function(data, alpha, max_conditioning_size,
                                     index = 1, legacy_index = TRUE,
                                     residual_cache = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_skeleton_cpp_cached_export(
    data,
    as.numeric(alpha),
    as.integer(max_conditioning_size),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache)
  )
}

fast_skeleton_cpp_backend <- function(data, alpha, max_conditioning_size,
                                      residual_backend = "linear",
                                      residual_cache = TRUE,
                                      index = 1,
                                      legacy_index = TRUE,
                                      fastspline_params = list(),
                                      ci_method = "dcc.gamma",
                                      hsic_params = list(),
                                      permutation_params = list(),
                                      ci_diagnostics = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_skeleton_cpp_backend_export(
    data,
    as.numeric(alpha),
    as.integer(max_conditioning_size),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache),
    as.character(residual_backend),
    fastspline_params,
    as.character(ci_method),
    hsic_params,
    permutation_params,
    isTRUE(ci_diagnostics)
  )
}

fast_orient_wanpdag_cpp <- function(skeleton_result, data,
                                    residual_backend = "fastSpline",
                                    residual_cache = TRUE,
                                    alpha = 0.2,
                                    index = 1,
                                    legacy_index = TRUE,
                                    orient_collider = TRUE,
                                    solve_confl = FALSE,
                                    rules = c(TRUE, TRUE, TRUE),
                                    fastspline_params = list(),
                                    ci_method = "dcc.gamma",
                                    hsic_params = list(),
                                    permutation_params = list(),
                                    ci_diagnostics = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_orient_wanpdag_cpp_export(
    data,
    as.matrix(skeleton_result$adjacency),
    skeleton_result$sepsets,
    as.numeric(alpha),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache),
    as.character(residual_backend),
    fastspline_params,
    isTRUE(orient_collider),
    isTRUE(solve_confl),
    as.logical(rules),
    as.character(ci_method),
    hsic_params,
    permutation_params,
    isTRUE(ci_diagnostics)
  )
}

fast_kpc_wanpdag_cpp <- function(data, alpha, max_conditioning_size,
                                 residual_backend = "fastSpline",
                                 residual_cache = TRUE,
                                 index = 1,
                                 legacy_index = TRUE,
                                 orient_collider = TRUE,
                                 solve_confl = FALSE,
                                 rules = c(TRUE, TRUE, TRUE),
                                 fastspline_params = list(),
                                 ci_method = "dcc.gamma",
                                 hsic_params = list(),
                                 permutation_params = list(),
                                 ci_diagnostics = TRUE) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_kpc_wanpdag_cpp_export(
    data,
    as.numeric(alpha),
    as.integer(max_conditioning_size),
    as.numeric(index),
    isTRUE(legacy_index),
    isTRUE(residual_cache),
    as.character(residual_backend),
    fastspline_params,
    isTRUE(orient_collider),
    isTRUE(solve_confl),
    as.logical(rules),
    as.character(ci_method),
    hsic_params,
    permutation_params,
    isTRUE(ci_diagnostics)
  )
}

fast_residual_cache_selftest <- function(data) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fast_residual_cache_selftest_export(data)
}

fastspline_basis_selftest <- function(data) {
  build_fastkpc_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  fastspline_basis_selftest_export(data)
}

fastspline_solver_selftest <- function() {
  build_fastkpc_native()
  fastspline_solver_selftest_export()
}

orientation_matrix_selftest <- function() {
  build_fastkpc_native()
  orientation_matrix_selftest_export()
}

orientation_rules_selftest <- function() {
  build_fastkpc_native()
  orientation_rules_selftest_export()
}

regrvonps_native_selftest <- function() {
  build_fastkpc_native()
  regrvonps_native_selftest_export()
}

wanpdag_engine_core_selftest <- function() {
  build_fastkpc_native()
  wanpdag_engine_core_selftest_export()
}

list_residual_backends <- function() {
  build_fastkpc_native()
  as.character(list_residual_backends_export())
}

fast_residual_backend_selftest <- function() {
  build_fastkpc_native()
  fast_residual_backend_selftest_export()
}

fast_residual_backend_unknown_selftest <- function() {
  build_fastkpc_native()
  fast_residual_backend_unknown_selftest_export()
}

fastspline_residual <- function(y, S, fastspline_params = list()) {
  build_fastkpc_native()
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  fastspline_residual_export(as.numeric(y), S, fastspline_params)
}
