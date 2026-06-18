source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/precision_backend_resolver.R")
source("fastkpc/R/precision_execution_trace.R")
source("fastkpc/R/precision_data_plane.R")

`%||%` <- function(x, y) if (is.null(x)) y else x

fastkpc_zero_cache <- function() {
  list(requests = 0L, hits = 0L, computations = 0L)
}

fastkpc_elapsed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed = max(elapsed, .Machine$double.eps))
}

fastkpc_normalize_data <- function(data, labels = NULL) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (is.null(labels)) {
    labels <- colnames(data)
    if (is.null(labels)) labels <- paste0("V", seq_len(ncol(data)))
  }
  if (length(labels) != ncol(data)) {
    stop("labels length must match number of columns", call. = FALSE)
  }
  colnames(data) <- labels
  list(
    data = data,
    info = list(
      n = nrow(data),
      p = ncol(data),
      labels = labels,
      has_missing = anyNA(data),
      all_finite = all(is.finite(data)),
      storage_mode = storage.mode(data)
    )
  )
}

fastkpc_resolve_engine <- function(engine) {
  engine <- match.arg(engine, c("auto", "cuda", "cpu"))
  if (engine != "auto") return(engine)
  cuda_ok <- tryCatch(isTRUE(fastkpc_cuda_available()), error = function(e) FALSE)
  if (cuda_ok) "cuda" else "cpu"
}

fastkpc_cuda_residual_status <- function() {
  cuda_ok <- tryCatch(isTRUE(fastkpc_cuda_available()), error = function(e) FALSE)
  wrapper_ok <- exists("fastspline_residual_cuda", mode = "function")
  reason <- ""
  if (!cuda_ok) {
    reason <- "CUDA unavailable"
  } else if (!wrapper_ok) {
    reason <- "fastspline_residual_cuda wrapper unavailable"
  }
  list(available = isTRUE(cuda_ok && wrapper_ok), reason = reason)
}

fastkpc_cuda_hsic_status <- function() {
  cuda_ok <- tryCatch(isTRUE(fastkpc_cuda_available()), error = function(e) FALSE)
  gamma_ok <- exists("fast_hsic_gamma_cuda", mode = "function")
  perm_ok <- exists("fast_hsic_perm_cuda", mode = "function")
  reason <- ""
  if (!cuda_ok) {
    reason <- "CUDA unavailable"
  } else if (!gamma_ok || !perm_ok) {
    reason <- "CUDA HSIC wrappers unavailable"
  }
  list(available = isTRUE(cuda_ok && gamma_ok && perm_ok), reason = reason)
}

fastkpc_resolve_orientation_device <- function(engine_used, graph_stage,
                                               residual_backend,
                                               orientation_residual_device) {
  requested <- match.arg(orientation_residual_device, c("auto", "cpu", "cuda"))
  if (graph_stage == "skeleton") {
    return(list(used = "none", reason = "graph_stage skeleton ignores orientation residual device"))
  }
  if (engine_used != "cuda") {
    reason <- if (requested == "cuda") {
      "CPU engine resolves orientation residual device to cpu"
    } else {
      "CPU engine uses CPU orientation residuals"
    }
    return(list(used = "cpu", reason = reason))
  }
  if (residual_backend == "linear") {
    reason <- if (requested == "cuda") {
      "linear orientation residual CUDA device is not implemented"
    } else {
      "linear orientation residuals use CPU"
    }
    return(list(used = "cpu", reason = reason))
  }
  if (requested == "cpu") {
    return(list(used = "cpu", reason = "orientation residual device requested cpu"))
  }
  list(used = "cuda", reason = "")
}

fastkpc_pdag_edge_counts <- function(pdag) {
  if (is.null(pdag)) {
    return(list(directed = 0L, undirected = 0L, bidirected = 0L))
  }
  pdag <- as.matrix(pdag)
  p <- ncol(pdag)
  directed <- 0L
  undirected <- 0L
  bidirected <- 0L
  if (p < 2L) return(list(directed = directed, undirected = undirected,
                          bidirected = bidirected))
  for (i in seq_len(p - 1L)) {
    for (j in (i + 1L):p) {
      if (pdag[i, j] == 2L && pdag[j, i] == 2L) {
        bidirected <- bidirected + 1L
      } else if (pdag[i, j] == 1L && pdag[j, i] == 1L) {
        undirected <- undirected + 1L
      } else if ((pdag[i, j] == 1L && pdag[j, i] == 0L) ||
                 (pdag[j, i] == 1L && pdag[i, j] == 0L)) {
        directed <- directed + 1L
      }
    }
  }
  list(directed = directed, undirected = undirected, bidirected = bidirected)
}

fastkpc_skeleton_edge_count <- function(adjacency) {
  adjacency <- as.matrix(adjacency)
  sum(adjacency[upper.tri(adjacency)])
}

fastkpc_min_nonzero <- function(values) {
  values <- as.numeric(values)
  values <- values[is.finite(values) & values > 0]
  if (length(values) == 0L) NA_real_ else min(values)
}

fastkpc_graph_metrics <- function(result) {
  skeleton <- result$skeleton
  orientation <- result$orientation
  pdag <- if (is.null(orientation)) NULL else orientation$pdag
  edge_counts <- fastkpc_pdag_edge_counts(pdag)
  pmax <- if (is.null(skeleton$pMax)) numeric() else as.numeric(skeleton$pMax)
  scheduler_summary <- skeleton$scheduler_diagnostics$summary %||% list()
  list(
    skeleton_edge_count = as.integer(fastkpc_skeleton_edge_count(skeleton$adjacency)),
    directed_edge_count = edge_counts$directed,
    undirected_edge_count = edge_counts$undirected,
    bidirected_edge_count = edge_counts$bidirected,
    orientation_event_count = if (is.null(orientation)) 0L else length(orientation$events),
    generalized_orientation_count =
      if (is.null(orientation)) 0L else as.integer(orientation$counts$generalized %||% 0L),
    max_pmax = if (length(pmax) == 0L) NA_real_ else max(pmax, na.rm = TRUE),
    min_nonzero_pmax = fastkpc_min_nonzero(pmax),
    tasks_planned = as.integer(scheduler_summary$tasks_planned %||% 0L),
    tasks_evaluated = as.integer(scheduler_summary$tasks_evaluated %||% 0L),
    tests_replayed = as.integer(scheduler_summary$tests_replayed %||% 0L),
    tasks_ignored_after_delete =
      as.integer(scheduler_summary$tasks_ignored_after_delete %||% 0L),
    unique_residual_requests =
      as.integer(scheduler_summary$unique_residual_requests %||% 0L),
    dcov_batches = as.integer(scheduler_summary$dcov_batches %||% 0L),
    residual_batches = as.integer(scheduler_summary$residual_batches %||% 0L),
    cuda_residual_batch_groups =
      as.integer(scheduler_summary$cuda_residual_batch_groups %||% 0L),
    cuda_residual_true_batched_groups =
      as.integer(scheduler_summary$cuda_residual_true_batched_groups %||% 0L),
    cuda_residual_true_batched_fits =
      as.integer(scheduler_summary$cuda_residual_true_batched_fits %||% 0L),
    cuda_residual_single_fit_calls =
      as.integer(scheduler_summary$cuda_residual_single_fit_calls %||% 0L),
    cuda_residual_cpu_fallback_fits =
      as.integer(scheduler_summary$cuda_residual_cpu_fallback_fits %||% 0L),
    cuda_residual_unique_designs =
      as.integer(scheduler_summary$cuda_residual_unique_designs %||% 0L),
    cuda_residual_duplicate_design_fits =
      as.integer(scheduler_summary$cuda_residual_duplicate_design_fits %||% 0L),
    cuda_residual_max_fits_per_design =
      as.integer(scheduler_summary$cuda_residual_max_fits_per_design %||% 0L)
  )
}

fastkpc_cache_sections <- function(raw) {
  skeleton_cache <- raw$skeleton$residual_cache %||% fastkpc_zero_cache()
  orientation_cache <- if (is.null(raw$orientation)) {
    fastkpc_zero_cache()
  } else {
    raw$orientation$residual_cache %||% fastkpc_zero_cache()
  }
  list(skeleton = skeleton_cache, orientation = orientation_cache)
}

fastkpc_timing_table <- function(graph_stage, elapsed_total_sec) {
  if (graph_stage == "wanpdag") {
    data.frame(
      stage = c("skeleton", "orientation", "total"),
      elapsed_sec = c(elapsed_total_sec / 2, elapsed_total_sec / 2,
                      elapsed_total_sec)
    )
  } else {
    data.frame(stage = c("skeleton", "total"),
               elapsed_sec = c(elapsed_total_sec, elapsed_total_sec))
  }
}

as_fastkpc_result <- function(raw, config, data_info, elapsed_total_sec,
                              validation = list(enabled = FALSE),
                              benchmark = list(enabled = FALSE)) {
  cuda_residual_status <- fastkpc_cuda_residual_status()
  cuda_hsic_status <- fastkpc_cuda_hsic_status()
  result <- list(
    config = config,
    data_info = data_info,
    engine = config$engine_used,
    skeleton = raw$skeleton,
    orientation = raw$orientation %||% NULL,
    metrics = NULL,
    timings = fastkpc_timing_table(config$graph_stage, elapsed_total_sec),
    cache = fastkpc_cache_sections(raw),
    validation = validation,
    benchmark = benchmark,
    diagnostics = list(
      cuda_available = tryCatch(isTRUE(fastkpc_cuda_available()), error = function(e) FALSE),
      cuda_residual_available = cuda_residual_status$available,
      cuda_residual_reason =
        raw$skeleton$residual_device_reason %||% cuda_residual_status$reason,
      cuda_hsic_available = cuda_hsic_status$available,
      cuda_hsic_reason = cuda_hsic_status$reason,
      pcalg_available = requireNamespace("pcalg", quietly = TRUE),
      graph_available = requireNamespace("graph", quietly = TRUE),
      ci_method_available = !is.null(raw$skeleton$ci_method),
      R_version = as.character(getRversion())
    )
  )
  result$metrics <- fastkpc_graph_metrics(result)
  class(result) <- c("fastkpc_result", "list")
  validate_fastkpc_result(result)
  result
}

validate_fastkpc_result <- function(result) {
  if (is.null(result$config)) stop("fastkpc_result missing config", call. = FALSE)
  required_config <- c("residual_device_requested", "residual_device_used",
                       "orientation_residual_device_requested",
                       "orientation_residual_device_used",
                       "orientation_residual_device_reason",
                       "cuda_residual_fallback", "scheduler_requested",
                       "scheduler_used", "residual_batch_size",
                       "orientation_batch_size", "scheduler_diagnostics",
                       "orientation_diagnostics", "ci_method",
                       "ci_method_requested", "ci_backend",
                       "ci_backend_requested", "ci_backend_reason",
                       "cuda_hsic_requested", "cuda_hsic_used",
                       "hsic_params", "permutation_params")
  missing_config <- setdiff(required_config, names(result$config))
  if (length(missing_config) > 0L) {
    stop("fastkpc_result missing config field: ", missing_config[[1L]],
         call. = FALSE)
  }
  if (is.null(result$skeleton)) stop("fastkpc_result missing skeleton", call. = FALSE)
  if (!is.data.frame(result$timings) ||
      !all(c("stage", "elapsed_sec") %in% names(result$timings))) {
    stop("fastkpc_result timing table invalid", call. = FALSE)
  }
  if (!is.null(result$orientation)) {
    pdag <- result$orientation$pdag
    p <- result$data_info$p
    if (is.null(pdag) || !identical(dim(pdag), c(p, p))) {
      stop("fastkpc_result pdag dimension mismatch", call. = FALSE)
    }
  }
  TRUE
}

fastkpc_result_summary <- function(result) {
  validate_fastkpc_result(result)
  list(
    engine = result$config$engine_used,
    residual_backend = result$config$residual_backend,
    ci_method = result$config$ci_method,
    ci_backend = result$config$ci_backend,
    residual_device = result$config$residual_device_used,
    scheduler = result$config$scheduler_used,
    graph_stage = result$config$graph_stage,
    n = result$data_info$n,
    p = result$data_info$p,
    metrics = result$metrics,
    timings = result$timings,
    cache = result$cache
  )
}

fastkpc_extract_pdag <- function(result) {
  validate_fastkpc_result(result)
  if (is.null(result$orientation)) NULL else result$orientation$pdag
}

fastkpc_extract_skeleton <- function(result) {
  validate_fastkpc_result(result)
  result$skeleton
}

print.fastkpc_result <- function(x, ...) {
  validate_fastkpc_result(x)
  cat("fastkpc_result\n")
  cat("  engine: ", x$config$engine_used, "\n", sep = "")
  cat("  residual_backend: ", x$config$residual_backend, "\n", sep = "")
  cat("  ci_method: ", x$config$ci_method, "\n", sep = "")
  cat("  ci_backend: ", x$config$ci_backend, "\n", sep = "")
  cat("  residual_device: ", x$config$residual_device_used, "\n", sep = "")
  cat("  scheduler: ", x$config$scheduler_used, "\n", sep = "")
  cat("  graph_stage: ", x$config$graph_stage, "\n", sep = "")
  cat("  n: ", x$data_info$n, ", p: ", x$data_info$p, "\n", sep = "")
  cat("  skeleton_edges: ", x$metrics$skeleton_edge_count, "\n", sep = "")
  cat("  directed_edges: ", x$metrics$directed_edge_count, "\n", sep = "")
  invisible(x)
}

summary.fastkpc_result <- function(object, ...) {
  fastkpc_result_summary(object)
}

fast_kpc <- function(data,
                     alpha = 0.2,
                     max_conditioning_size = 2,
                     engine = c("auto", "cuda", "cpu"),
                     precision = c("legacy", "fast", "compatible", "hybrid"),
                     tau = log(2),
                     ci_method = c("dcc.gamma", "hsic.gamma", "hsic.perm"),
                     residual_backend = c("fastSpline", "linear"),
                     residual_device = c("auto", "cpu", "cuda"),
                     orientation_residual_device = c("auto", "cpu", "cuda"),
                     scheduler = c("auto", "layer", "legacy"),
                     graph_stage = c("wanpdag", "skeleton"),
                     residual_cache = TRUE,
                     index = 1,
                     legacy_index = TRUE,
                     batch_size = 0,
                     residual_batch_size = 0,
                     orientation_batch_size = 0,
                     scheduler_diagnostics = TRUE,
                     orientation_diagnostics = TRUE,
                     orient_collider = TRUE,
                     solve_confl = FALSE,
                     rules = c(TRUE, TRUE, TRUE),
                     fastspline_params = list(),
                     hsic_params = list(sig = 1),
                     permutation_params = list(replicates = 100,
                                               seed = NULL,
                                               include_observed = TRUE),
                     ci_diagnostics = TRUE,
                     precision_diagnostics = TRUE,
                     precision_executors = NULL,
                     runtime_capabilities = NULL,
                     allow_canary_mgcv_extract = FALSE,
                     cuda_residual_fallback = TRUE,
                     validate = FALSE,
                     benchmark = FALSE,
                     legacy = FALSE,
                     labels = NULL,
                     seed = NULL) {
  engine_requested <- match.arg(engine)
  engine_used <- fastkpc_resolve_engine(engine_requested)
  precision_requested <- match.arg(precision)
  ci_method <- match.arg(ci_method)
  residual_backend <- match.arg(residual_backend)
  if (precision_requested %in% c("fast", "compatible", "hybrid")) {
    residual_backend <- "fastSpline"
  }
  residual_device <- match.arg(residual_device)
  orientation_residual_device <- match.arg(orientation_residual_device)
  scheduler_requested <- match.arg(scheduler)
  graph_stage <- match.arg(graph_stage)
  if (engine_used == "cpu" && scheduler_requested == "layer") {
    stop("layer scheduler is only implemented for CUDA skeleton execution",
         call. = FALSE)
  }
  scheduler_used <- if (engine_used == "cuda") {
    if (scheduler_requested == "auto") "layer" else scheduler_requested
  } else {
    "legacy"
  }
  if (length(rules) != 3L || anyNA(rules)) {
    stop("rules must be a logical vector of length 3 without NA", call. = FALSE)
  }
  orientation_device <- fastkpc_resolve_orientation_device(
    engine_used, graph_stage, residual_backend, orientation_residual_device)
  ci_backend_requested <- if (engine_used == "cuda") "cuda" else "cpu"
  cuda_hsic_requested <- engine_used == "cuda" &&
    ci_method %in% c("hsic.gamma", "hsic.perm")
  if (!is.null(seed)) set.seed(seed)
  normalized <- fastkpc_normalize_data(data, labels = labels)
  matrix_data <- normalized$data
  data_hash <- paste(nrow(matrix_data), ncol(matrix_data),
                     signif(sum(matrix_data), 8), sep = ":")
  normalized$info$data_hash <- data_hash
  if (is.null(runtime_capabilities)) {
    runtime_capabilities <- fastkpc_precision_runtime_capabilities()
  }
  precision_executors_requested <- !is.null(precision_executors)
  if (is.null(precision_executors)) {
    precision_executors <- fastkpc_default_precision_executors()
  }
  precision_route <- if (identical(precision_requested, "legacy")) {
    list(
      precision = "legacy",
      primary_backend = if (identical(residual_backend, "fastSpline")) {
        if (identical(engine_used, "cuda")) "fastSplineCUDA" else "fastSplineCPU"
      } else {
        "linear"
      },
      verifier_backend = NA_character_,
      compatibility_status = "legacy",
      compatibility_action = "run-existing",
      compatibility_claim = "existing-fastkpc",
      near_alpha_policy = list(alpha = alpha, tau = tau),
      canonical_replay_required = FALSE,
      fallback_backend = NA_character_,
      fallback_reason = "",
      supported_checks = character(),
      unsupported_checks = character(),
      setup_fingerprint = paste("fastkpc", max_conditioning_size, sep = ":"),
      runtime_capabilities = runtime_capabilities
    )
  } else {
    fastkpc_resolve_backend_request(
      precision = precision_requested,
      alpha = alpha,
      tau = tau,
      S = seq_len(min(2L, max(1L, as.integer(max_conditioning_size)))),
      formula_class = if (max_conditioning_size <= 2L) "full-smooth" else "additive-smooth",
      penalty_count = if (max_conditioning_size <= 2L) 1L else
        max(1L, as.integer(max_conditioning_size)),
      family = "gaussian",
      link = "identity",
      setup_fingerprint = paste("fastkpc", max_conditioning_size, sep = ":"),
      runtime_capabilities = runtime_capabilities,
      fallback_backend = "legacy-mgcv",
      allow_canary = allow_canary_mgcv_extract
    )
  }
  if (identical(engine_used, "cpu") &&
      identical(precision_route$primary_backend, "fastSplineCUDA")) {
    precision_route$primary_backend <- "fastSplineCPU"
  }
  backend_requested <- precision_route$primary_backend
  backend_planned <- backend_requested
  verifier_planned <- precision_route$verifier_backend %||% NA_character_
  if (precision_requested == "compatible" &&
      identical(precision_route$compatibility_action, "fallback")) {
    backend_planned <- precision_route$fallback_backend %||% "legacy-mgcv"
  }
  backend_executed <- if (identical(residual_backend, "fastSpline")) {
    if (identical(engine_used, "cuda")) "fastSplineCUDA" else "fastSplineCPU"
  } else {
    residual_backend
  }
  verifier_executed <- NA_character_
  backend_used <- backend_executed
  precision_execution_status <- if (precision_requested %in%
                                    c("compatible", "hybrid")) {
    "control-plane-only"
  } else {
    "executed"
  }

  config <- list(
    alpha = alpha,
    tau = tau,
    precision_requested = precision_requested,
    precision = precision_requested,
    precision_route = precision_route,
    backend_requested = backend_requested,
    backend_planned = backend_planned,
    backend_executed = backend_executed,
    backend_used = backend_used,
    verifier_backend = verifier_planned,
    verifier_planned = verifier_planned,
    verifier_executed = verifier_executed,
    compatibility_status = precision_route$compatibility_status %||% "",
    compatibility_action = precision_route$compatibility_action %||% "",
    compatibility_claim = precision_route$compatibility_claim %||% "",
    fallback_reason = precision_route$fallback_reason %||% "",
    precision_execution_status = precision_execution_status,
    canonical_replay_required =
      isTRUE(precision_route$canonical_replay_required),
    precision_diagnostics = isTRUE(precision_diagnostics),
    max_conditioning_size = as.integer(max_conditioning_size),
    engine_requested = engine_requested,
    engine_used = engine_used,
    residual_backend = residual_backend,
    ci_method_requested = ci_method,
    ci_method = ci_method,
    ci_backend_requested = ci_backend_requested,
    ci_backend = if (engine_used == "cuda") ci_backend_requested else "native-cpu",
    ci_backend_reason = "",
    cuda_hsic_requested = isTRUE(cuda_hsic_requested),
    cuda_hsic_used = FALSE,
    hsic_params = hsic_params,
    permutation_params = permutation_params,
    ci_diagnostics = isTRUE(ci_diagnostics),
    residual_device_requested = residual_device,
    residual_device_used = if (engine_used == "cuda") residual_device else "cpu",
    orientation_residual_device_requested = orientation_residual_device,
    orientation_residual_device_used = orientation_device$used,
    orientation_residual_device_reason = orientation_device$reason,
    scheduler_requested = scheduler_requested,
    scheduler_used = scheduler_used,
    graph_stage = graph_stage,
    residual_cache = isTRUE(residual_cache),
    index = index,
    legacy_index = isTRUE(legacy_index),
    batch_size = as.integer(batch_size),
    residual_batch_size = as.integer(residual_batch_size),
    orientation_batch_size = as.integer(orientation_batch_size),
    scheduler_diagnostics = isTRUE(scheduler_diagnostics),
    orientation_diagnostics = isTRUE(orientation_diagnostics),
    orient_collider = isTRUE(orient_collider),
    solve_confl = isTRUE(solve_confl),
    rules = as.logical(rules),
    fastspline_params = fastspline_params,
    cuda_residual_fallback = isTRUE(cuda_residual_fallback),
    validate = isTRUE(validate),
    benchmark = isTRUE(benchmark),
    legacy = isTRUE(legacy),
    seed = seed
  )

  use_precision_r_skeleton <- graph_stage == "skeleton" &&
    identical(engine_used, "cpu") &&
    as.integer(max_conditioning_size) <= 1L &&
    precision_requested %in% c("fast", "compatible", "hybrid") &&
    (isTRUE(precision_executors_requested) ||
       precision_requested %in% c("compatible", "hybrid"))

  timed <- fastkpc_elapsed({
    if (graph_stage == "wanpdag") {
      if (engine_used == "cuda") {
        fast_kpc_wanpdag_cuda(
          matrix_data, alpha, max_conditioning_size,
          residual_backend = residual_backend,
          residual_device = residual_device,
          orientation_residual_device = orientation_residual_device,
          residual_cache = residual_cache,
          index = index,
          legacy_index = legacy_index,
          batch_size = batch_size,
          residual_batch_size = residual_batch_size,
          orientation_batch_size = orientation_batch_size,
          scheduler = scheduler_requested,
          scheduler_diagnostics = scheduler_diagnostics,
          orientation_diagnostics = orientation_diagnostics,
          orient_collider = orient_collider,
          solve_confl = solve_confl,
          rules = rules,
          fastspline_params = fastspline_params,
          cuda_residual_fallback = cuda_residual_fallback,
          ci_method = ci_method,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          ci_diagnostics = ci_diagnostics
        )
      } else {
        fast_kpc_wanpdag_cpp(
          matrix_data, alpha, max_conditioning_size,
          residual_backend = residual_backend,
          residual_cache = residual_cache,
          index = index,
          legacy_index = legacy_index,
          orient_collider = orient_collider,
          solve_confl = solve_confl,
          rules = rules,
          fastspline_params = fastspline_params,
          ci_method = ci_method,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          ci_diagnostics = ci_diagnostics
        )
      }
    } else if (isTRUE(use_precision_r_skeleton)) {
      skeleton <- fastkpc_r_skeleton_precision(
        matrix_data, alpha, max_conditioning_size,
        precision = precision_requested,
        tau = tau,
        ci_method = ci_method,
        index = index,
        legacy_index = legacy_index,
        hsic_params = hsic_params,
        permutation_params = permutation_params,
        precision_executors = precision_executors,
        runtime_capabilities = runtime_capabilities,
        allow_canary = allow_canary_mgcv_extract
      )
      list(skeleton = skeleton, orientation = NULL)
    } else {
      skeleton <- if (engine_used == "cuda") {
        fast_skeleton_cuda_backend(
          matrix_data, alpha, max_conditioning_size,
          residual_backend = residual_backend,
          residual_device = residual_device,
          residual_cache = residual_cache,
          index = index,
          legacy_index = legacy_index,
          batch_size = batch_size,
          residual_batch_size = residual_batch_size,
          scheduler = scheduler_requested,
          scheduler_diagnostics = scheduler_diagnostics,
          fastspline_params = fastspline_params,
          cuda_residual_fallback = cuda_residual_fallback,
          ci_method = ci_method,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          ci_diagnostics = ci_diagnostics
        )
      } else {
        fast_skeleton_cpp_backend(
          matrix_data, alpha, max_conditioning_size,
          residual_backend = residual_backend,
          residual_cache = residual_cache,
          index = index,
          legacy_index = legacy_index,
          fastspline_params = fastspline_params,
          ci_method = ci_method,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          ci_diagnostics = ci_diagnostics
        )
      }
      list(skeleton = skeleton, orientation = NULL)
    }
  })

  config$residual_device_used <-
    if (engine_used == "cuda") {
      timed$value$skeleton$residual_device %||% residual_device
    } else {
      "cpu"
    }
  config$scheduler_used <- timed$value$skeleton$scheduler %||% scheduler_used
  config$ci_method <- timed$value$skeleton$ci_method %||% ci_method
  config$ci_backend <- timed$value$skeleton$ci_backend %||% config$ci_backend
  config$ci_backend_reason <-
    timed$value$skeleton$ci_backend_reason %||% config$ci_backend_reason
  if (!is.null(timed$value$skeleton$precision_receipt)) {
    receipt <- timed$value$skeleton$precision_receipt
    config$backend_executed <- timed$value$skeleton$residual_backend %||%
      receipt$residual_backend_executed %||% config$backend_executed
    config$backend_used <- config$backend_executed
    config$verifier_executed <- timed$value$skeleton$verifier_backend %||%
      config$verifier_executed
    config$precision_execution_status <- "data-plane-executed"
  }
  config$cuda_hsic_used <- isTRUE(config$cuda_hsic_requested) &&
    identical(config$ci_backend, "cuda-hsic")
  if (!is.null(timed$value$orientation)) {
    config$orientation_residual_device_used <-
      timed$value$orientation$residual_device %||% config$orientation_residual_device_used
    config$orientation_residual_device_reason <-
      timed$value$orientation$residual_device_reason %||%
        config$orientation_residual_device_reason
    if (identical(timed$value$orientation$ci_backend, "cuda-hsic")) {
      config$cuda_hsic_used <- TRUE
    }
  }
  validation <- list(enabled = isTRUE(validate), legacy_requested = isTRUE(legacy))
  benchmark_section <- list(enabled = isTRUE(benchmark))
  result <- as_fastkpc_result(
    timed$value,
    config = config,
    data_info = normalized$info,
    elapsed_total_sec = timed$elapsed,
    validation = validation,
    benchmark = benchmark_section
  )
  if (isTRUE(precision_diagnostics)) {
    result$diagnostics$precision_trace <-
      timed$value$skeleton$precision_trace %||%
      fastkpc_precision_trace_from_result(
        result = result,
        route = precision_route,
        run_id = paste0("fastkpc-", format(Sys.time(), "%Y%m%d%H%M%S")),
        scenario_id = "fast_kpc",
        elapsed_total_sec = timed$elapsed
      )
  }
  validate_fastkpc_result(result)
  result
}
