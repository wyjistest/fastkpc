.fastkpc_cuda_root <- function() {
  if (file.exists("fastkpc/src/r_api_cuda.cpp")) return(normalizePath("fastkpc"))
  stop("Cannot find fastkpc/src/r_api_cuda.cpp from current working directory",
       call. = FALSE)
}

.fastkpc_cuda_so <- function() {
  file.path(.fastkpc_cuda_root(), "build", "fastkpc_cuda.so")
}

build_fastkpc_cuda_native <- function(rebuild = FALSE) {
  root <- .fastkpc_cuda_root()
  so <- .fastkpc_cuda_so()
  if (rebuild && is.loaded("C_fastkpc_cuda_available")) {
    try(dyn.unload(so), silent = TRUE)
  }
  if (rebuild || !file.exists(so)) {
    script <- file.path(root, "tools", "build_cuda_native.sh")
    if (!file.exists(script)) {
      stop("Cannot find CUDA build script: ", script, call. = FALSE)
    }
    status <- system2("bash", script)
    if (!identical(status, 0L)) {
      stop("CUDA native build failed with status ", status, call. = FALSE)
    }
  }
  normalizePath(so)
}

load_fastkpc_cuda_native <- function(rebuild = FALSE) {
  if (!requireNamespace("Rcpp", quietly = TRUE)) {
    stop("Rcpp is required to load fastkpc CUDA native code", call. = FALSE)
  }
  so <- build_fastkpc_cuda_native(rebuild = rebuild)
  loaded <- vapply(getLoadedDLLs(), function(dll) normalizePath(dll[["path"]],
                                                               mustWork = FALSE),
                   character(1))
  if (!normalizePath(so, mustWork = FALSE) %in% loaded) {
    dyn.load(so)
  }
  invisible(so)
}

fastkpc_cuda_available <- function() {
  load_fastkpc_cuda_native()
  isTRUE(.Call("C_fastkpc_cuda_available", PACKAGE = "fastkpc_cuda"))
}

fastkpc_cuda_device_info <- function() {
  load_fastkpc_cuda_native()
  .Call("C_fastkpc_cuda_device_info", PACKAGE = "fastkpc_cuda")
}

fast_dcov_batch_cuda <- function(x, y, index = 1, legacy_index = TRUE) {
  load_fastkpc_cuda_native()
  x <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1)
  y <- if (is.matrix(y)) y else matrix(as.numeric(y), ncol = 1)
  storage.mode(x) <- "double"
  storage.mode(y) <- "double"
  .Call("C_fast_dcov_batch_cuda", x, y, as.numeric(index), isTRUE(legacy_index),
        PACKAGE = "fastkpc_cuda")
}

fast_hsic_gamma_cuda <- function(x, y, sig = 1) {
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_gamma_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), PACKAGE = "fastkpc_cuda")
}

fast_hsic_perm_cuda <- function(x, y, sig = 1, replicates = 100L,
                                seed, include_observed = TRUE) {
  if (missing(seed) || is.null(seed)) {
    stop("CUDA HSIC permutation requires explicit seed in this stage",
         call. = FALSE)
  }
  load_fastkpc_cuda_native()
  .Call("C_fast_hsic_perm_cuda", as.numeric(x), as.numeric(y),
        as.numeric(sig), as.integer(replicates), as.integer(seed),
        isTRUE(include_observed), PACKAGE = "fastkpc_cuda")
}

fastspline_residual_cuda <- function(y, S, fastspline_params = list(),
                                     fallback = TRUE) {
  load_fastkpc_cuda_native()
  S <- as.matrix(S)
  storage.mode(S) <- "double"
  .Call("C_fastspline_residual_cuda", as.numeric(y), S, fastspline_params,
        isTRUE(fallback), PACKAGE = "fastkpc_cuda")
}

fastspline_residual_batch_cuda <- function(data, targets, conditioning_sets,
                                           fastspline_params = list(),
                                           fallback = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fastspline_residual_batch_cuda", data, as.integer(targets),
        conditioning_sets, fastspline_params, isTRUE(fallback),
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda <- function(data, alpha, max_conditioning_size,
                               index = 1, legacy_index = TRUE,
                               batch_size = 0) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda_cached <- function(data, alpha, max_conditioning_size,
                                      index = 1, legacy_index = TRUE,
                                      batch_size = 0,
                                      residual_cache = TRUE) {
  load_fastkpc_cuda_native()
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_cached", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache),
        PACKAGE = "fastkpc_cuda")
}

fast_skeleton_cuda_backend <- function(data, alpha, max_conditioning_size,
                                       residual_backend = "linear",
                                       residual_device = c("auto", "cpu", "cuda"),
                                       residual_cache = TRUE,
                                       index = 1,
                                       legacy_index = TRUE,
                                       batch_size = 0,
                                       residual_batch_size = 0,
                                       scheduler = c("auto", "layer", "legacy"),
                                       scheduler_diagnostics = TRUE,
                                       fastspline_params = list(),
                                       cuda_residual_fallback = TRUE,
                                       ci_method = "dcc.gamma",
                                       hsic_params = list(),
                                       permutation_params = list(),
                                       ci_diagnostics = TRUE) {
  load_fastkpc_cuda_native()
  residual_device <- match.arg(residual_device)
  scheduler <- match.arg(scheduler)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_skeleton_cuda_backend", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache), as.character(residual_backend),
        as.character(residual_device), as.integer(residual_batch_size),
        as.character(scheduler), isTRUE(scheduler_diagnostics),
        fastspline_params,
        isTRUE(cuda_residual_fallback), as.character(ci_method),
        hsic_params, permutation_params, isTRUE(ci_diagnostics),
        PACKAGE = "fastkpc_cuda")
}

fast_kpc_wanpdag_cuda <- function(data, alpha, max_conditioning_size,
                                  residual_backend = "fastSpline",
                                  residual_device = c("auto", "cpu", "cuda"),
                                  orientation_residual_device = c("auto", "cpu", "cuda"),
                                  residual_cache = TRUE,
                                  index = 1,
                                  legacy_index = TRUE,
                                  batch_size = 0,
                                  residual_batch_size = 0,
                                  orientation_batch_size = 0,
                                  scheduler = c("auto", "layer", "legacy"),
                                  scheduler_diagnostics = TRUE,
                                  orientation_diagnostics = TRUE,
                                  orient_collider = TRUE,
                                  solve_confl = FALSE,
                                  rules = c(TRUE, TRUE, TRUE),
                                  fastspline_params = list(),
                                  cuda_residual_fallback = TRUE,
                                  ci_method = "dcc.gamma",
                                  hsic_params = list(),
                                  permutation_params = list(),
                                  ci_diagnostics = TRUE) {
  load_fastkpc_cuda_native()
  residual_device <- match.arg(residual_device)
  orientation_residual_device <- match.arg(orientation_residual_device)
  scheduler <- match.arg(scheduler)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_fast_kpc_wanpdag_cuda", data, as.numeric(alpha),
        as.integer(max_conditioning_size), as.numeric(index),
        isTRUE(legacy_index), as.integer(batch_size),
        isTRUE(residual_cache), as.character(residual_backend),
        as.character(residual_device),
        as.character(orientation_residual_device),
        as.integer(residual_batch_size),
        as.integer(orientation_batch_size),
        as.character(scheduler), isTRUE(scheduler_diagnostics),
        isTRUE(orientation_diagnostics),
        fastspline_params,
        isTRUE(cuda_residual_fallback), isTRUE(orient_collider),
        isTRUE(solve_confl), as.logical(rules), as.character(ci_method),
        hsic_params, permutation_params, isTRUE(ci_diagnostics),
        PACKAGE = "fastkpc_cuda")
}
