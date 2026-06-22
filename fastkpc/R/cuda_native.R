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

mgcv_extract_gpu_solve_handle_fixed_sp_cuda <- function(handle) {
  load_fastkpc_cuda_native()
  X <- as.matrix(handle$X)
  y <- as.numeric(handle$y)
  Z <- as.matrix(handle$Z)
  XtX_null <- as.matrix(handle$XtX_null)
  penalty_null <- as.matrix(handle$penalty_null)
  Xty_null <- as.numeric(handle$Xty_null)
  storage.mode(X) <- "double"
  storage.mode(y) <- "double"
  storage.mode(Z) <- "double"
  storage.mode(XtX_null) <- "double"
  storage.mode(penalty_null) <- "double"
  storage.mode(Xty_null) <- "double"
  .Call("C_mgcv_extract_gpu_solve_handle_fixed_sp",
        X, y, Z, XtX_null, penalty_null, Xty_null,
        PACKAGE = "fastkpc_cuda")
}

mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda <- function(handles) {
  load_fastkpc_cuda_native()
  if (!is.list(handles) || length(handles) == 0L) {
    stop("handles must be a non-empty list", call. = FALSE)
  }
  first <- handles[[1L]]
  X <- as.matrix(first$X)
  Z <- as.matrix(first$Z)
  XtX_null <- as.matrix(first$XtX_null)
  Y <- do.call(cbind, lapply(handles, function(handle) as.numeric(handle$y)))
  Xty_null <- do.call(cbind, lapply(handles, function(handle) as.numeric(handle$Xty_null)))
  penalty_null_list <- lapply(handles, function(handle) {
    penalty <- as.matrix(handle$penalty_null)
    storage.mode(penalty) <- "double"
    penalty
  })
  storage.mode(X) <- "double"
  storage.mode(Y) <- "double"
  storage.mode(Z) <- "double"
  storage.mode(XtX_null) <- "double"
  storage.mode(Xty_null) <- "double"
  .Call("C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp",
        X, Y, Z, XtX_null, penalty_null_list, Xty_null,
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

precision_replay_layer_native <- function(adjacency, edge_x, edge_y, x, y,
                                          conditioning_sets, p_values, alpha,
                                          pmax = NULL, trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  adjacency <- as.matrix(adjacency)
  storage.mode(adjacency) <- "integer"
  if (is.null(pmax)) {
    pmax <- matrix(-Inf, nrow(adjacency), ncol(adjacency))
    diag(pmax) <- 1
  }
  pmax <- as.matrix(pmax)
  storage.mode(pmax) <- "double"
  .Call("C_precision_replay_layer_native",
        adjacency,
        pmax,
        as.integer(edge_x),
        as.integer(edge_y),
        as.integer(x),
        as.integer(y),
        conditioning_sets,
        as.numeric(p_values),
        as.numeric(alpha),
        as.character(trace_level),
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
