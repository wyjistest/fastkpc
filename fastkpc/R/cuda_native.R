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

precision_make_layer_plan_native <- function(adjacency, level) {
  load_fastkpc_cuda_native()
  adjacency <- as.matrix(adjacency)
  storage.mode(adjacency) <- "integer"
  .Call("C_precision_make_layer_plan_native",
        adjacency,
        as.integer(level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_ptable_native <- function(
    p = 6L, alpha = 0.05, max_conditioning_size = 2L,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  .Call("C_precision_run_skeleton_ptable_native",
        as.integer(p),
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_provider_native <- function(
    p, alpha, max_conditioning_size, provider,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(provider)) {
    stop("provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  .Call("C_precision_run_skeleton_provider_native",
        as.integer(p),
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        provider,
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_dcov0_native <- function(
    data, alpha, index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_dcov0_native",
        data,
        as.numeric(alpha),
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_exact_ci_native <- function(
    data, alpha, max_conditioning_size, index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_exact_ci_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_residual_provider_native <- function(
    data, alpha, max_conditioning_size, residual_provider,
    index = 1, legacy_index = TRUE,
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(residual_provider)) {
    stop("residual_provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_residual_provider_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        residual_provider,
        as.numeric(index),
        isTRUE(legacy_index),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

precision_run_skeleton_residual_provider_legacy_dcov_native <- function(
    data, alpha, max_conditioning_size, residual_provider,
    index = 1, numCol = floor(nrow(as.matrix(data)) / 10),
    trace_level = c("summary", "full", "none")) {
  load_fastkpc_cuda_native()
  if (!is.function(residual_provider)) {
    stop("residual_provider must be a function", call. = FALSE)
  }
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  .Call("C_precision_run_skeleton_residual_provider_legacy_dcov_native",
        data,
        as.numeric(alpha),
        as.integer(max_conditioning_size),
        residual_provider,
        as.numeric(index),
        as.integer(numCol),
        as.character(trace_level),
        PACKAGE = "fastkpc_cuda")
}

fastkpc_native_legacy_mgcv_residual_backend <- function() {
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

fastkpc_native_legacy_mgcv_backend_condition_threshold <- function() {
  value <- suppressWarnings(as.numeric(Sys.getenv(
    "FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD",
    unset = "1e12"
  )))
  if (length(value) != 1L || !is.finite(value) || value < 0) 1e12 else value
}

fastkpc_native_legacy_mgcv_backend_native_s_size_limit <- function() {
  raw <- Sys.getenv("FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT",
                    unset = "Inf")
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || is.na(value) || value < 0) Inf else value
}

fastkpc_native_legacy_mgcv_provider_cores <- function() {
  raw <- Sys.getenv("FASTKPC_NATIVE_LEGACY_MGCV_PROVIDER_CORES", unset = "1")
  value <- suppressWarnings(as.integer(raw))
  if (length(value) != 1L || is.na(value) || value < 1L) 1L else value
}

fastkpc_native_prepare_legacy_mgcv_cpp_backend <- function() {
  if (!exists("fastkpc_legacy_mgcv_residual_cpp_backend_target",
              mode = "function")) {
    source("fastkpc/R/legacy_runner.R")
  }
  if (exists("fastkpc_legacy_prepare_mgcv_cpp_shadow",
             mode = "function")) {
    fastkpc_legacy_prepare_mgcv_cpp_shadow(TRUE)
  }
  required <- c(
    "fastkpc_legacy_runtime_zero",
    "fastkpc_legacy_mgcv_residual_cpp_backend_target",
    "fastkpc_legacy_env"
  )
  missing <- required[!vapply(required, exists, logical(1), mode = "function")]
  if (length(missing) > 0L) {
    stop("legacy mgcv C++ residual backend missing helper: ",
         paste(missing, collapse = ","), call. = FALSE)
  }
  invisible(TRUE)
}

fastkpc_native_counter_value <- function(counter_env, name, default) {
  if (is.null(counter_env)) return(default)
  value <- counter_env[[name]]
  if (is.null(value)) default else value
}

fastkpc_native_provider_backend_label <- function(backend) {
  if (identical(backend, "cpp_guarded")) {
    "legacy-mgcv-cpp-guarded-level-batch"
  } else {
    "legacy-mgcv-regrXonS-level-batch"
  }
}

fastkpc_legacy_mgcv_residual_provider_matrix <- function(
    data, requests, counter_env = NULL,
    backend = fastkpc_native_legacy_mgcv_residual_backend(),
    condition_threshold =
      fastkpc_native_legacy_mgcv_backend_condition_threshold(),
    native_s_size_limit =
      fastkpc_native_legacy_mgcv_backend_native_s_size_limit()) {
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  required <- c("request_index", "target", "conditioning_sets",
                "S_key", "conditioning_size")
  missing_fields <- setdiff(required, names(requests))
  if (length(missing_fields) > 0L) {
    stop("residual provider request table missing fields: ",
         paste(missing_fields, collapse = ","), call. = FALSE)
  }
  if (!is.null(counter_env)) {
    level_calls <- if (is.null(counter_env$level_calls)) {
      0L
    } else {
      counter_env$level_calls
    }
    request_count <- if (is.null(counter_env$request_count)) {
      0L
    } else {
      counter_env$request_count
    }
    counter_env$level_calls <- level_calls + 1L
    counter_env$request_count <- request_count + nrow(requests)
    counter_env$mgcv_backend <- backend
  }

  if (identical(backend, "cpp_guarded")) {
    fastkpc_native_prepare_legacy_mgcv_cpp_backend()
    if (is.null(counter_env)) {
      metrics <- fastkpc_legacy_runtime_zero()
      legacy_env <- fastkpc_legacy_env()
    } else {
      if (is.null(counter_env$mgcv_cpp_metrics)) {
        counter_env$mgcv_cpp_metrics <- fastkpc_legacy_runtime_zero()
      }
      if (is.null(counter_env$mgcv_legacy_env)) {
        counter_env$mgcv_legacy_env <- fastkpc_legacy_env()
      }
      metrics <- counter_env$mgcv_cpp_metrics
      legacy_env <- counter_env$mgcv_legacy_env
    }
  }

  out <- matrix(NA_real_, nrow(data), nrow(requests))
  provider_cores_requested <- fastkpc_native_legacy_mgcv_provider_cores()
  provider_cores_used <- min(provider_cores_requested, nrow(requests))
  provider_parallel_enabled <- identical(backend, "r") &&
    identical(.Platform$OS.type, "unix") &&
    provider_cores_used > 1L
  if (!is.null(counter_env)) {
    parallel_cores_seen <- counter_env$provider_parallel_cores
    if (is.null(parallel_cores_seen)) parallel_cores_seen <- 0L
    parallel_level_count <- counter_env$provider_parallel_level_count
    if (is.null(parallel_level_count)) parallel_level_count <- 0L
    parallel_request_count <- counter_env$provider_parallel_request_count
    if (is.null(parallel_request_count)) parallel_request_count <- 0L
    counter_env$provider_parallel_enabled <-
      isTRUE(counter_env$provider_parallel_enabled) ||
      isTRUE(provider_parallel_enabled)
    counter_env$provider_parallel_cores <- max(
      as.integer(parallel_cores_seen),
      if (isTRUE(provider_parallel_enabled)) provider_cores_used else 0L
    )
    counter_env$provider_parallel_level_count <-
      as.integer(parallel_level_count) +
      as.integer(isTRUE(provider_parallel_enabled))
    counter_env$provider_parallel_request_count <-
      as.integer(parallel_request_count) +
      if (isTRUE(provider_parallel_enabled)) nrow(requests) else 0L
  }
  if (isTRUE(provider_parallel_enabled)) {
    residual_list <- parallel::mclapply(
      seq_len(nrow(requests)),
      function(i) {
        S <- as.integer(requests$conditioning_sets[[i]])
        if (length(S) == 0L) {
          stop("legacy mgcv residual provider received unconditional request",
               call. = FALSE)
        }
        target <- as.integer(requests$target[[i]])
        fastkpc_legacy_mgcv_residual(
          data = data,
          target = target,
          S = S
        )
      },
      mc.cores = provider_cores_used,
      mc.preschedule = TRUE
    )
    for (i in seq_along(residual_list)) {
      out[, i] <- residual_list[[i]]
    }
    return(out)
  }
  for (i in seq_len(nrow(requests))) {
    S <- as.integer(requests$conditioning_sets[[i]])
    if (length(S) == 0L) {
      stop("legacy mgcv residual provider received unconditional request",
           call. = FALSE)
    }
    target <- as.integer(requests$target[[i]])
    if (identical(backend, "cpp_guarded")) {
      backend_result <- fastkpc_legacy_mgcv_residual_cpp_backend_target(
        metrics = metrics,
        target_data = data[, target, drop = FALSE],
        s_data = data[, S, drop = FALSE],
        env = legacy_env,
        condition_threshold = condition_threshold,
        native_s_size_limit = native_s_size_limit,
        target = target,
        S = S
      )
      metrics <- backend_result$metrics
      if (!is.null(counter_env)) counter_env$mgcv_cpp_metrics <- metrics
      out[, i] <- backend_result$residual
    } else {
      out[, i] <- fastkpc_legacy_mgcv_residual(
        data = data,
        target = target,
        S = S
      )
    }
  }
  out
}

fastkpc_legacy_mgcv_residual_provider <- function(
    data, counter_env = NULL,
    backend = fastkpc_native_legacy_mgcv_residual_backend()) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  force(counter_env)
  function(requests, level) {
    fastkpc_legacy_mgcv_residual_provider_matrix(
      data = data,
      requests = requests,
      counter_env = counter_env,
      backend = backend
    )
  }
}

fastkpc_legacy_mgcv_residual_batch_provider <- function(data,
                                                        counter_env = NULL,
                                                        backend =
                                                          fastkpc_native_legacy_mgcv_residual_backend()) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  force(counter_env)
  function(requests, level) {
    residuals <- fastkpc_legacy_mgcv_residual_provider_matrix(
      data = data,
      requests = requests,
      counter_env = counter_env,
      backend = backend
    )
    list(
      residuals = residuals,
      contract = "level-residual-matrix-v1",
      backend = fastkpc_native_provider_backend_label(backend),
      mgcv_backend = backend,
      level = as.integer(level),
      request_count = as.integer(nrow(requests)),
      n = as.integer(nrow(data))
    )
  }
}

fastkpc_native_attach_mgcv_provider_summary <- function(result, counter_env,
                                                       backend) {
  backend <- match.arg(backend, c("r", "cpp_guarded"))
  result$summary$residual_provider_mgcv_backend <- backend
  enabled <- identical(backend, "cpp_guarded")
  result$summary$residual_provider_mgcv_cpp_backend_enabled <- enabled
  metrics <- fastkpc_native_counter_value(counter_env, "mgcv_cpp_metrics", NULL)
  metric_value <- function(name, default = 0) {
    if (is.null(metrics) || is.null(metrics[[name]])) return(default)
    metrics[[name]]
  }
  result$summary$residual_provider_mgcv_cpp_backend_count <-
    as.integer(metric_value("mgcv_cpp_backend_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_native_count <-
    as.integer(metric_value("mgcv_cpp_backend_native_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_error_count <-
    as.integer(metric_value("mgcv_cpp_backend_error_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_high_condition_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_high_condition_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_outside_envelope_fallback_count <-
    as.integer(metric_value("mgcv_cpp_backend_outside_envelope_fallback_count", 0L))
  result$summary$residual_provider_mgcv_cpp_backend_ms <-
    as.numeric(metric_value("mgcv_cpp_backend_ms", 0))
  result$summary$residual_provider_mgcv_cpp_backend_native_solve_ms <-
    as.numeric(metric_value("mgcv_cpp_backend_native_solve_ms", 0))
  result$summary$residual_provider_parallel_enabled <-
    isTRUE(fastkpc_native_counter_value(
      counter_env, "provider_parallel_enabled", FALSE
    ))
  result$summary$residual_provider_parallel_cores <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_cores", 0L
    ))
  result$summary$residual_provider_parallel_level_count <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_level_count", 0L
    ))
  result$summary$residual_provider_parallel_request_count <-
    as.integer(fastkpc_native_counter_value(
      counter_env, "provider_parallel_request_count", 0L
    ))
  result
}

precision_run_skeleton_legacy_mgcv_legacy_dcov_native <- function(
    data, alpha, max_conditioning_size,
    index = 1, numCol = floor(nrow(as.matrix(data)) / 10),
    trace_level = c("summary", "full", "none"),
    dcov_batch = c("env", "none", "level", "canonical")) {
  dcov_batch <- match.arg(dcov_batch)
  old_dcov_batch <- Sys.getenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH",
                               unset = NA_character_)
  if (dcov_batch %in% c("level", "canonical")) {
    Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = dcov_batch)
    on.exit({
      if (is.na(old_dcov_batch)) {
        Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
      } else {
        Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_dcov_batch)
      }
    }, add = TRUE)
  } else if (identical(dcov_batch, "none")) {
    Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
    on.exit({
      if (is.na(old_dcov_batch)) {
        Sys.unsetenv("FASTKPC_NATIVE_LEGACY_DCOV_BATCH")
      } else {
        Sys.setenv(FASTKPC_NATIVE_LEGACY_DCOV_BATCH = old_dcov_batch)
      }
    }, add = TRUE)
  }
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  provider_backend <- fastkpc_native_legacy_mgcv_residual_backend()
  provider_counter <- new.env(parent = emptyenv())
  result <- precision_run_skeleton_residual_provider_legacy_dcov_native(
    data = data,
    alpha = alpha,
    max_conditioning_size = max_conditioning_size,
    residual_provider = fastkpc_legacy_mgcv_residual_batch_provider(
      data,
      counter_env = provider_counter,
      backend = provider_backend
    ),
    index = index,
    numCol = numCol,
    trace_level = trace_level
  )
  result <- fastkpc_native_attach_mgcv_provider_summary(
    result = result,
    counter_env = provider_counter,
    backend = provider_backend
  )
  result$summary$entrypoint <- "legacy-mgcv-legacy-dcov-native"
  result$summary$residual_provider_hidden <- TRUE
  result
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
