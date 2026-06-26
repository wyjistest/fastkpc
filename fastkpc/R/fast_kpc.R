source("fastkpc/R/native.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/kpc_tprs_residual_cpp.R")
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

fastkpc_is_default_precision_executors <- function(precision_executors) {
  identical(precision_executors, fastkpc_default_precision_executors())
}

fastkpc_batched_precision_combinations <- function(values, choose) {
  values <- as.integer(values)
  if (choose == 0L) return(list(integer()))
  if (length(values) < choose) return(list())
  lapply(utils::combn(values, choose, simplify = FALSE), as.integer)
}

fastkpc_batched_precision_make_layer_plan <- function(adjacency, level) {
  p <- ncol(adjacency)
  tasks <- list()
  append_task <- function(edge_x, edge_y, x, y, S, side) {
    tasks[[length(tasks) + 1L]] <<- list(
      task_id = length(tasks) + 1L,
      edge_x = as.integer(edge_x),
      edge_y = as.integer(edge_y),
      x = as.integer(x),
      y = as.integer(y),
      S = as.integer(S),
      S_key = fastkpc_precision_S_key(S),
      conditioning_target_side = side
    )
  }
  for (x in seq_len(p - 1L)) {
    for (y in seq.int(x + 1L, p)) {
      if (!isTRUE(adjacency[x, y])) next
      nx <- which(adjacency[, x] & seq_len(p) != y)
      for (S in fastkpc_batched_precision_combinations(nx, level)) {
        append_task(x, y, x, y, S, "x")
      }
      ny <- which(adjacency[, y] & seq_len(p) != x)
      for (S in fastkpc_batched_precision_combinations(ny, level)) {
        append_task(x, y, y, x, S, "y")
      }
    }
  }
  tasks
}

fastkpc_batched_precision_residual_key <- function(target, S) {
  paste(as.integer(target), fastkpc_precision_S_key(S), sep = "|")
}

fastkpc_batched_precision_primary_pvalues <- function(data, tasks, ci_method,
                                                      index, legacy_index,
                                                      batch_size,
                                                      fastspline_params,
                                                      cuda_residual_fallback) {
  if (!identical(ci_method, "dcc.gamma")) {
    stop("batched CUDA precision primary currently supports dcc.gamma",
         call. = FALSE)
  }
  n <- nrow(data)
  task_count <- length(tasks)
  if (task_count == 0L) {
    return(list(pvalues = numeric(), diagnostics = list(
      residual_batches = 0L, dcov_batches = 0L,
      unique_residual_requests = 0L, residual_batch_diagnostics = NULL
    )))
  }

  residuals <- new.env(parent = emptyenv())
  residual_requests <- list()
  seen <- new.env(parent = emptyenv())
  for (task in tasks) {
    if (length(task$S) == 0L) next
    for (target in c(task$x, task$y)) {
      key <- fastkpc_batched_precision_residual_key(target, task$S)
      if (exists(key, envir = seen, inherits = FALSE)) next
      assign(key, TRUE, envir = seen)
      residual_requests[[length(residual_requests) + 1L]] <- list(
        target = as.integer(target),
        S = as.integer(task$S),
        key = key
      )
    }
  }

  residual_batches <- 0L
  residual_batch_diag <- NULL
  if (length(residual_requests) > 0L) {
    batch <- fastspline_residual_batch_cuda(
      data = data,
      targets = vapply(residual_requests, `[[`, integer(1L), "target"),
      conditioning_sets = lapply(residual_requests, `[[`, "S"),
      fastspline_params = fastspline_params,
      fallback = cuda_residual_fallback
    )
    residual_batches <- 1L
    residual_batch_diag <- batch$batch_diagnostics
    for (i in seq_along(residual_requests)) {
      assign(residual_requests[[i]]$key, as.numeric(batch$residuals[, i]),
             envir = residuals)
    }
  }

  xmat <- matrix(NA_real_, n, task_count)
  ymat <- matrix(NA_real_, n, task_count)
  for (i in seq_along(tasks)) {
    task <- tasks[[i]]
    if (length(task$S) == 0L) {
      xmat[, i] <- data[, task$x]
      ymat[, i] <- data[, task$y]
    } else {
      xmat[, i] <- get(
        fastkpc_batched_precision_residual_key(task$x, task$S),
        envir = residuals, inherits = FALSE
      )
      ymat[, i] <- get(
        fastkpc_batched_precision_residual_key(task$y, task$S),
        envir = residuals, inherits = FALSE
      )
    }
  }

  actual_batch_size <- as.integer(batch_size)
  if (is.na(actual_batch_size) || actual_batch_size <= 0L) {
    actual_batch_size <- min(task_count, 512L)
  }
  pvalues <- numeric(task_count)
  dcov_batches <- 0L
  starts <- seq.int(1L, task_count, by = actual_batch_size)
  for (start in starts) {
    end <- min(task_count, start + actual_batch_size - 1L)
    cols <- seq.int(start, end)
    ci <- fast_dcov_batch_cuda(xmat[, cols, drop = FALSE],
                               ymat[, cols, drop = FALSE],
                               index = index,
                               legacy_index = legacy_index)
    pvalues[cols] <- as.numeric(ci$p.value)
    dcov_batches <- dcov_batches + 1L
  }

  list(
    pvalues = pvalues,
    diagnostics = list(
      residual_batches = residual_batches,
      dcov_batches = dcov_batches,
      unique_residual_requests = length(residual_requests),
      residual_batch_diagnostics = residual_batch_diag
    )
  )
}

fastkpc_parse_precision_S_key <- function(S_key) {
  if (is.null(S_key) || !nzchar(as.character(S_key)[1L])) return(integer())
  as.integer(strsplit(as.character(S_key)[1L], "\\|")[[1L]])
}

fastkpc_batched_precision_execute_verifier <- function(data, row, alpha, tau,
                                                       ci_method, index,
                                                       legacy_index,
                                                       hsic_params,
                                                       permutation_params,
                                                       precision_executors,
                                                       runtime_capabilities,
                                                       allow_canary,
                                                       execution_context) {
  S <- fastkpc_parse_precision_S_key(row$S_key)
  route <- fastkpc_precision_group_route(
    precision = "hybrid", alpha = alpha, tau = tau, S = S,
    runtime_capabilities = runtime_capabilities,
    allow_canary = allow_canary,
    execution_engine = "cuda"
  )
  route$execution_context <- execution_context
  verifier_backend <- fastkpc_nonempty_backend(route$verifier_backend,
                                               "mgcvExtractGPUGCV")
  route$primary_backend <- verifier_backend
  randomness <- fastkpc_precision_ci_randomness(
    ci_method = ci_method,
    permutation_params = permutation_params,
    canonical_test_order_id = row$canonical_test_order_id
  )
  effective_permutation_params <- fastkpc_precision_effective_permutation_params(
    ci_method = ci_method,
    permutation_params = permutation_params,
    randomness = randomness
  )
  exec <- fastkpc_execute_route_with_fallback(
    data = data,
    x = as.integer(row$x),
    y = as.integer(row$y),
    S = S,
    route = route,
    role = "verifier",
    ci_method = ci_method,
    index = index,
    legacy_index = legacy_index,
    hsic_params = hsic_params,
    permutation_params = effective_permutation_params,
    precision_executors = precision_executors,
    fallback_backend = fastkpc_precision_fallback_backends(verifier_backend, route)
  )
  info <- fastkpc_resolve_ci_decision(exec$receipt$p.value, alpha = alpha,
                                      na_delete = TRUE)
  list(exec = exec, info = info, randomness = randomness)
}

fastkpc_batched_precision_execute_verifier_batch <- function(
    data, tasks, task_indices, alpha, ci_method, index, legacy_index,
    hsic_params, permutation_params, runtime_capabilities, allow_canary,
    execution_context) {
  if (!identical(ci_method, "dcc.gamma")) {
    stop("batched verifier currently supports dcc.gamma", call. = FALSE)
  }
  if (length(task_indices) == 0L) {
    return(list(results = list(), metrics = list(
      total_ms = 0, mgcv_setup_ms = 0, spectral_prepare_ms = 0,
      gcv_score_ms = 0, cuda_solve_ms = 0, ci_test_ms = 0,
      cache_full_hit_count = 0L, cache_partial_hit_count = 0L,
      cache_miss_count = 0L, fallback_count = 0L,
      verifier_batch_groups = 0L, verifier_ci_batches = 0L
    )))
  }
  total_start <- proc.time()[["elapsed"]]
  sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 17L))
  grouped <- split(task_indices, vapply(task_indices, function(i) {
    tasks[[i]]$S_key
  }, character(1L)))
  results <- vector("list", length(tasks))
  metrics <- list(
    total_ms = 0,
    mgcv_setup_ms = 0,
    spectral_prepare_ms = 0,
    gcv_score_ms = 0,
    cuda_solve_ms = 0,
    ci_test_ms = 0,
    cache_full_hit_count = 0L,
    cache_partial_hit_count = 0L,
    cache_miss_count = 0L,
    fallback_count = 0L,
    verifier_batch_groups = 0L,
    verifier_ci_batches = 0L
  )

  for (group_indices in grouped) {
    representative <- tasks[[group_indices[[1L]]]]
    S <- representative$S
    route <- fastkpc_precision_group_route(
      precision = "hybrid", alpha = alpha, tau = Inf, S = S,
      runtime_capabilities = runtime_capabilities,
      allow_canary = allow_canary,
      execution_engine = "cuda"
    )
    verifier_backend <- fastkpc_nonempty_backend(route$verifier_backend,
                                                 "mgcvExtractGPUGCV")
    if (!identical(verifier_backend, "mgcvExtractGPUGCV")) {
      stop("batched verifier only supports mgcvExtractGPUGCV", call. = FALSE)
    }
    cuda_ok <- exists("fastkpc_cuda_available", mode = "function") &&
      isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
    solve_ok <- exists(
      "mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda",
      mode = "function"
    )
    dcov_ok <- exists("fast_dcov_batch_cuda", mode = "function")
    if (!cuda_ok || !solve_ok || !dcov_ok) {
      stop("batched verifier requires native CUDA mgcvExtractGPU and dCov",
           call. = FALSE)
    }
    group_tasks <- tasks[group_indices]
    unique_targets <- sort(unique(as.integer(unlist(lapply(group_tasks, function(task) {
      c(task$x, task$y)
    }), use.names = FALSE))))
    residual_batch <- fastkpc_mgcv_extract_gpu_cached_entries_for_targets(
      data = data,
      targets = unique_targets,
      S = S,
      sp_grid = sp_grid,
      context = execution_context
    )
    entries <- residual_batch$entries
    xmat <- matrix(NA_real_, nrow(data), length(group_indices))
    ymat <- matrix(NA_real_, nrow(data), length(group_indices))
    for (j in seq_along(group_indices)) {
      task <- tasks[[group_indices[[j]]]]
      xmat[, j] <- entries[[as.character(task$x)]]$residuals
      ymat[, j] <- entries[[as.character(task$y)]]$residuals
    }
    ci_start <- proc.time()[["elapsed"]]
    ci <- fast_dcov_batch_cuda(xmat, ymat, index = index,
                               legacy_index = legacy_index)
    ci_ms <- (proc.time()[["elapsed"]] - ci_start) * 1000
    timings <- residual_batch$timings %||% list()
    timings$ci_test_ms <- ci_ms
    timings$total_ms <- fastkpc_sum_timing(list(
      timings$residualization_total_ms %||% NA_real_,
      ci_ms
    ))

    metrics$mgcv_setup_ms <- fastkpc_batched_precision_add_ms(
      metrics$mgcv_setup_ms, timings$mgcv_setup_cpu_ms
    )
    metrics$spectral_prepare_ms <- fastkpc_batched_precision_add_ms(
      metrics$spectral_prepare_ms, timings$spectral_prepare_ms
    )
    metrics$gcv_score_ms <- fastkpc_batched_precision_add_ms(
      metrics$gcv_score_ms, timings$gcv_score_ms
    )
    metrics$cuda_solve_ms <- fastkpc_batched_precision_add_ms(
      metrics$cuda_solve_ms, timings$linear_solve_ms
    )
    metrics$ci_test_ms <- fastkpc_batched_precision_add_ms(
      metrics$ci_test_ms, timings$ci_test_ms
    )
    metrics$verifier_batch_groups <- metrics$verifier_batch_groups + 1L
    metrics$verifier_ci_batches <- metrics$verifier_ci_batches + 1L

    for (j in seq_along(group_indices)) {
      task_index <- group_indices[[j]]
      task <- tasks[[task_index]]
      x_entry <- entries[[as.character(task$x)]]
      y_entry <- entries[[as.character(task$y)]]
      setup_x <- fastkpc_setup_fingerprint_value(x_entry$setup_fingerprint)
      setup_y <- fastkpc_setup_fingerprint_value(y_entry$setup_fingerprint)
      if (is.na(setup_x) || is.na(setup_y) || !identical(setup_x, setup_y)) {
        stop("batched mgcvExtractGPU verifier setup fingerprint mismatch",
             call. = FALSE)
      }
      p_value <- as.numeric(ci$p.value[[j]])
      cache_hit_x <- isTRUE(residual_batch$cache_hit[[as.character(task$x)]])
      cache_hit_y <- isTRUE(residual_batch$cache_hit[[as.character(task$y)]])
      cache_hit_any <- cache_hit_x || cache_hit_y
      cache_hit_all <- cache_hit_x && cache_hit_y
      cache_partial_hit <- xor(cache_hit_x, cache_hit_y)
      if (isTRUE(cache_hit_all)) {
        metrics$cache_full_hit_count <- metrics$cache_full_hit_count + 1L
      } else if (isTRUE(cache_partial_hit)) {
        metrics$cache_partial_hit_count <-
          metrics$cache_partial_hit_count + 1L
      } else {
        metrics$cache_miss_count <- metrics$cache_miss_count + 1L
      }
      results[[task_index]] <- list(
        receipt = list(
          p.value = p_value,
          residual_backend_executed = "mgcvExtractGPU",
          ci_backend_executed = "cuda-dcov",
          setup_fingerprint = setup_x,
          setup_fingerprint_x = setup_x,
          setup_fingerprint_y = setup_y,
          shared_setup_fingerprint = setup_x,
          p_source_used = "verifier:mgcvExtractGPU+cuda-dcov",
          sp = c(x = x_entry$sp, y = y_entry$sp),
          score = c(x = x_entry$score, y = y_entry$score),
          edf = c(x = x_entry$edf, y = y_entry$edf),
          selected_grid_index = c(x = x_entry$selected_grid_index,
                                  y = y_entry$selected_grid_index),
          gcv_grid_points = c(x = x_entry$gcv_grid_points,
                              y = y_entry$gcv_grid_points),
          used_device = "cuda",
          used_device_x = x_entry$used_device,
          used_device_y = y_entry$used_device,
          native_gpu_solve_used_x = isTRUE(x_entry$native_gpu_solve_used),
          native_gpu_solve_used_y = isTRUE(y_entry$native_gpu_solve_used),
          sp_selection_backend_executed_x =
            x_entry$sp_selection_backend_executed,
          sp_selection_backend_executed_y =
            y_entry$sp_selection_backend_executed,
          gcv_score_backend_executed_x = x_entry$gcv_score_backend_executed,
          gcv_score_backend_executed_y = y_entry$gcv_score_backend_executed,
          selected_solve_backend_executed_x =
            x_entry$selected_solve_backend_executed,
          selected_solve_backend_executed_y =
            y_entry$selected_solve_backend_executed,
          same_setup_pair_batch_used = FALSE,
          same_setup_target_batch_used =
            as.integer(residual_batch$target_computations %||% 0L) > 1L,
          cache_hit = cache_hit_any,
          cache_hit_x = cache_hit_x,
          cache_hit_y = cache_hit_y,
          cache_hit_any = cache_hit_any,
          cache_hit_all = cache_hit_all,
          cache_partial_hit = cache_partial_hit,
          cache_service_mode = "level-batched-verifier",
          residualization_compute_ms =
            timings$residualization_compute_ms %||% NA_real_,
          cache_lookup_ms = NA_real_,
          cuda_single_target_calls = 0L,
          cuda_solve_calls = as.integer(residual_batch$cuda_solve_calls %||% 0L),
          timings = timings
        )
      )
      results[[task_index]]$info <- fastkpc_resolve_ci_decision(
        p_value, alpha = alpha, na_delete = TRUE
      )
      results[[task_index]]$randomness <- fastkpc_precision_ci_randomness(
        ci_method = ci_method,
        permutation_params = permutation_params,
        canonical_test_order_id = NA_integer_
      )
      results[[task_index]]$fallback_triggered <- FALSE
      results[[task_index]]$fallback_reason <- ""
      results[[task_index]]$attempts <- list(list(
        backend_planned = "mgcvExtractGPUGCV",
        backend_attempted = "mgcvExtractGPUGCV",
        backend_executed = "mgcvExtractGPU",
        attempt_status = "ok",
        error_class = "",
        error_message = "",
        fallback_triggered = FALSE,
        fallback_reason = "",
        elapsed_ms = timings$total_ms %||% NA_real_,
        receipt = results[[task_index]]$receipt
      ))
      results[[task_index]]$attempt_count <- 1L
      results[[task_index]]$attempt_backend_sequence <- "mgcvExtractGPU"
      results[[task_index]]$attempt_status_sequence <- "ok"
    }
  }
  metrics$total_ms <- (proc.time()[["elapsed"]] - total_start) * 1000
  list(results = results, metrics = metrics)
}

fastkpc_batched_precision_add_ms <- function(total, value) {
  value <- as.numeric(value %||% NA_real_)[1L]
  if (is.finite(value)) total + value else total
}

fastkpc_batched_precision_resolve_pvalues <- function(pvalues, alpha) {
  p_raw <- as.numeric(pvalues)
  nonfinite <- !is.finite(p_raw)
  p_used <- p_raw
  p_used[nonfinite] <- 1.0
  delete_edge <- p_used >= as.numeric(alpha)
  list(
    p_raw = p_raw,
    p_used = p_used,
    delete_edge = delete_edge,
    p_was_nonfinite = nonfinite,
    nonfinite_action = ifelse(nonfinite, "na-delete-use-1", "")
  )
}

fastkpc_batched_precision_near_alpha_indices <- function(tasks, decision,
                                                         alpha, tau,
                                                         verifier_policy) {
  if (!identical(verifier_policy, "near-alpha")) return(integer())
  has_s <- vapply(tasks, function(task) length(task$S) > 0L, logical(1L))
  p_for_trigger <- pmax(as.numeric(decision$p_raw), .Machine$double.xmin)
  alpha <- max(as.numeric(alpha), .Machine$double.xmin)
  near <- abs(log(p_for_trigger / alpha)) <= as.numeric(tau)
  which(has_s & (decision$p_was_nonfinite | near))
}

fastkpc_batched_precision_replay_level <- function(data, tasks, pvalues,
                                                   state, level, alpha, tau,
                                                   ci_method, index,
                                                   legacy_index, hsic_params,
                                                   permutation_params,
                                                   precision_executors,
                                                   runtime_capabilities,
                                                   allow_canary,
                                                   execution_context,
                                                   verifier_policy,
                                                   trace_level) {
  verifier_policy <- match.arg(verifier_policy, c("none", "near-alpha"))
  trace_level <- match.arg(trace_level, c("summary", "full", "none"))
  collect_trace <- identical(trace_level, "full")
  rows <- list()
  verifier_backends <- character()
  ci_backends <- c("cuda-dcov")
  verifier_count <- 0L
  decision_changes <- 0L
  primary_decision <- fastkpc_batched_precision_resolve_pvalues(
    pvalues, alpha = alpha
  )
  final_p <- primary_decision$p_used
  final_raw_p <- primary_decision$p_raw
  final_delete <- primary_decision$delete_edge
  near_alpha_indices <- fastkpc_batched_precision_near_alpha_indices(
    tasks = tasks,
    decision = primary_decision,
    alpha = alpha,
    tau = tau,
    verifier_policy = verifier_policy
  )
  near_alpha_flags <- logical(length(tasks))
  near_alpha_flags[near_alpha_indices] <- TRUE
  verified <- logical(length(tasks))
  verifier_receipts <- vector("list", length(tasks))
  verifier_infos <- vector("list", length(tasks))
  verifier_randomness <- vector("list", length(tasks))
  p_source_vec <- rep("primary:fastSplineCUDA+cuda-dcov", length(tasks))
  fallback_triggered_vec <- logical(length(tasks))
  fallback_reason_vec <- rep("", length(tasks))
  attempt_count_vec <- rep(1L, length(tasks))
  attempt_backend_sequence_vec <- rep("fastSplineCUDA", length(tasks))
  attempt_status_sequence_vec <- rep("ok", length(tasks))
  verified_but_unreplayed <- 0L
  verifier_metrics <- list(
    total_ms = 0,
    mgcv_setup_ms = 0,
    spectral_prepare_ms = 0,
    gcv_score_ms = 0,
    cuda_solve_ms = 0,
    ci_test_ms = 0,
    cache_full_hit_count = 0L,
    cache_partial_hit_count = 0L,
    cache_miss_count = 0L,
    fallback_count = 0L,
    verifier_batch_groups = 0L,
    verifier_ci_batches = 0L
  )

  if (length(tasks) == 0L) {
    return(list(
      state = state,
      trace_rows = rows,
      level_log = list(),
      verifier_backends = verifier_backends,
      ci_backends = ci_backends,
      verifier_count = verifier_count,
      decision_changes = decision_changes,
      verifier_metrics = verifier_metrics,
      native_summary = list(
        tasks_planned = 0L,
        tests_replayed = 0L,
        tasks_ignored_after_delete = 0L,
        deletions = 0L
      ),
      verified_but_unreplayed = 0L
    ))
  }

  state$test_id <- state$test_id + length(tasks)

  record_verifier_result <- function(i, verifier_receipt, verifier_info,
                                     randomness, attempts,
                                     fallback_triggered = FALSE,
                                     fallback_reason = "") {
    verified[[i]] <<- TRUE
    verifier_receipts[[i]] <<- verifier_receipt
    verifier_infos[[i]] <<- verifier_info
    verifier_randomness[[i]] <<- randomness
    final_p[[i]] <<- verifier_info$p_used
    final_raw_p[[i]] <<- verifier_info$p_raw
    final_delete[[i]] <<- verifier_info$delete_edge
    p_source_vec[[i]] <<- verifier_receipt$p_source_used
    fallback_triggered_vec[[i]] <<- isTRUE(fallback_triggered)
    fallback_reason_vec[[i]] <<- fallback_reason %||% ""
    attempt_count_vec[[i]] <<- length(attempts)
    attempt_backend_sequence_vec[[i]] <<- paste(vapply(attempts, function(attempt) {
      fastkpc_nonempty_backend(
        attempt$backend_executed,
        fastkpc_nonempty_backend(attempt$backend_attempted, "")
      )
    }, character(1L)), collapse = ">")
    attempt_status_sequence_vec[[i]] <<- paste(vapply(attempts, function(attempt) {
      as.character((attempt$attempt_status %||% "")[1L])
    }, character(1L)), collapse = ">")
    verifier_backends <<- c(verifier_backends,
                            verifier_receipt$residual_backend_executed)
    ci_backends <<- c(ci_backends, verifier_receipt$ci_backend_executed)
    if (!identical(primary_decision$delete_edge[[i]], verifier_info$delete_edge)) {
      decision_changes <<- decision_changes + 1L
    }
  }

  use_batched_verifier <- length(near_alpha_indices) > 0L &&
    identical(ci_method, "dcc.gamma") &&
    fastkpc_is_default_precision_executors(precision_executors)
  scalar_verifier_indices <- near_alpha_indices
  if (isTRUE(use_batched_verifier)) {
    batch <- tryCatch(
      fastkpc_batched_precision_execute_verifier_batch(
        data = data,
        tasks = tasks,
        task_indices = near_alpha_indices,
        alpha = alpha,
        ci_method = ci_method,
        index = index,
        legacy_index = legacy_index,
        hsic_params = hsic_params,
        permutation_params = permutation_params,
        runtime_capabilities = runtime_capabilities,
        allow_canary = allow_canary,
        execution_context = execution_context
      ),
      error = function(e) e
    )
    if (!inherits(batch, "error")) {
      for (i in near_alpha_indices) {
        item <- batch$results[[i]]
        if (is.null(item)) next
        canonical_test_order_id <- state$test_id - length(tasks) + i
        randomness <- fastkpc_precision_ci_randomness(
          ci_method = ci_method,
          permutation_params = permutation_params,
          canonical_test_order_id = canonical_test_order_id
        )
        record_verifier_result(
          i = i,
          verifier_receipt = item$receipt,
          verifier_info = item$info,
          randomness = randomness,
          attempts = item$attempts,
          fallback_triggered = item$fallback_triggered,
          fallback_reason = item$fallback_reason
        )
        verifier_count <- verifier_count + 1L
      }
      for (name in names(batch$metrics)) {
        if (!name %in% names(verifier_metrics)) next
        verifier_metrics[[name]] <- verifier_metrics[[name]] +
          batch$metrics[[name]]
      }
      scalar_verifier_indices <- near_alpha_indices[!verified[near_alpha_indices]]
    }
  }

  for (i in scalar_verifier_indices) {
    task <- tasks[[i]]
    verifier_receipt <- NULL
    verifier_info <- NULL
    verifier_exec <- NULL
    p_source <- "primary:fastSplineCUDA+cuda-dcov"
    fallback_triggered <- FALSE
    fallback_reason <- ""
    attempt_count <- 1L
    attempt_backend_sequence <- "fastSplineCUDA"
    attempt_status_sequence <- "ok"
    canonical_test_order_id <- state$test_id - length(tasks) + i
    randomness <- fastkpc_precision_ci_randomness(
      ci_method = ci_method,
      permutation_params = permutation_params,
      canonical_test_order_id = canonical_test_order_id
    )

    verifier_count <- verifier_count + 1L
    verifier <- fastkpc_batched_precision_execute_verifier(
      data = data,
      row = data.frame(
        canonical_test_order_id = canonical_test_order_id,
        x = task$x, y = task$y, S_key = task$S_key,
        stringsAsFactors = FALSE
      ),
      alpha = alpha, tau = tau, ci_method = ci_method,
      index = index, legacy_index = legacy_index,
      hsic_params = hsic_params, permutation_params = permutation_params,
      precision_executors = precision_executors,
      runtime_capabilities = runtime_capabilities,
      allow_canary = allow_canary,
      execution_context = execution_context
    )
    verifier_exec <- verifier$exec
    verifier_receipt <- verifier_exec$receipt
    verifier_info <- verifier$info
    fallback_triggered <- isTRUE(verifier_exec$fallback_triggered)
    fallback_reason <- verifier_exec$fallback_reason %||% ""
    attempts <- verifier_exec$attempts %||% list(verifier_exec)
    record_verifier_result(
      i = i,
      verifier_receipt = verifier_receipt,
      verifier_info = verifier_info,
      randomness = verifier$randomness,
      attempts = attempts,
      fallback_triggered = fallback_triggered,
      fallback_reason = fallback_reason
    )
    timings <- verifier_receipt$timings %||% list()
    verifier_metrics$total_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$total_ms, timings$total_ms
    )
    verifier_metrics$mgcv_setup_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$mgcv_setup_ms, timings$mgcv_setup_cpu_ms
    )
    verifier_metrics$spectral_prepare_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$spectral_prepare_ms, timings$spectral_prepare_ms
    )
    verifier_metrics$gcv_score_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$gcv_score_ms, timings$gcv_score_ms
    )
    verifier_metrics$cuda_solve_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$cuda_solve_ms, timings$linear_solve_ms
    )
    verifier_metrics$ci_test_ms <- fastkpc_batched_precision_add_ms(
      verifier_metrics$ci_test_ms, timings$ci_test_ms
    )
    if (isTRUE(verifier_receipt$cache_hit_all)) {
      verifier_metrics$cache_full_hit_count <-
        verifier_metrics$cache_full_hit_count + 1L
    } else if (isTRUE(verifier_receipt$cache_partial_hit)) {
      verifier_metrics$cache_partial_hit_count <-
        verifier_metrics$cache_partial_hit_count + 1L
    } else if (identical(verifier_receipt$cache_service_mode, "full-miss")) {
      verifier_metrics$cache_miss_count <-
        verifier_metrics$cache_miss_count + 1L
    }
    if (isTRUE(fallback_triggered) || attempt_count > 1L) {
      verifier_metrics$fallback_count <- verifier_metrics$fallback_count + 1L
    }
  }

  if (isTRUE(collect_trace)) {
    for (i in seq_along(tasks)) {
      task <- tasks[[i]]
      canonical_test_order_id <- state$test_id - length(tasks) + i
      has_verifier <- isTRUE(verified[[i]])
      verifier_receipt <- verifier_receipts[[i]]
      verifier_info <- verifier_infos[[i]]
      randomness <- verifier_randomness[[i]]
      if (is.null(randomness)) {
        randomness <- fastkpc_precision_ci_randomness(
          ci_method = ci_method,
          permutation_params = permutation_params,
          canonical_test_order_id = canonical_test_order_id
        )
      }
      rows[[length(rows) + 1L]] <- fastkpc_precision_trace_row(
        run_id = "fastkpc-batched-hybrid",
        scenario_id = "fast_kpc",
        conditioning_level = level,
        canonical_test_order_id = canonical_test_order_id,
        setup_fingerprint = if (length(task$S) == 0L) {
          "direct-ci:S:"
        } else {
          paste0("fastSplineCUDA:S:", task$S_key)
        },
        target_id = paste(task$x, task$y, sep = "|"),
        x = task$x,
        y = task$y,
        S_key = task$S_key,
        conditioning_target_side = task$conditioning_target_side,
        backend_requested = "fastSplineCUDA",
        backend_used = "fastSplineCUDA",
        backend_planned = "fastSplineCUDA",
        backend_executed = "fastSplineCUDA",
        verifier_backend = if (identical(verifier_policy, "near-alpha")) {
          "mgcvExtractGPUGCV"
        } else {
          NA_character_
        },
        verifier_planned = if (identical(verifier_policy, "near-alpha")) {
          "mgcvExtractGPUGCV"
        } else {
          NA_character_
        },
        verifier_executed = if (is.null(verifier_receipt)) {
          NA_character_
        } else {
          verifier_receipt$residual_backend_executed
        },
        compatibility_action = "run-batched-primary",
        fallback_reason = fallback_reason_vec[[i]],
        primary_p = primary_decision$p_used[[i]],
        verifier_p = if (is.null(verifier_info)) NA_real_ else verifier_info$p_used,
        p_used = final_p[[i]],
        p_raw = final_raw_p[[i]],
        p_was_nonfinite = if (is.null(verifier_info)) {
          primary_decision$p_was_nonfinite[[i]]
        } else {
          verifier_info$p_was_nonfinite
        },
        nonfinite_action = if (is.null(verifier_info)) {
          primary_decision$nonfinite_action[[i]]
        } else {
          verifier_info$nonfinite_action
        },
        p_source_used = p_source_vec[[i]],
        primary_residual_backend_executed =
          if (length(task$S) == 0L) "direct-ci" else "fastSplineCUDA",
        primary_ci_backend_executed = "cuda-dcov",
        primary_p_raw = primary_decision$p_raw[[i]],
        primary_p_used = primary_decision$p_used[[i]],
        near_alpha_triggered = near_alpha_flags[[i]],
        verifier_residual_backend_executed =
          if (is.null(verifier_receipt)) NA_character_ else
            verifier_receipt$residual_backend_executed,
        verifier_ci_backend_executed =
          if (is.null(verifier_receipt)) NA_character_ else
            verifier_receipt$ci_backend_executed,
        verifier_p_raw = if (is.null(verifier_info)) NA_real_ else verifier_info$p_raw,
        verifier_p_used = if (is.null(verifier_info)) NA_real_ else verifier_info$p_used,
        fallback_triggered = fallback_triggered_vec[[i]],
        attempt_count = attempt_count_vec[[i]],
        attempt_backend_sequence = attempt_backend_sequence_vec[[i]],
        attempt_status_sequence = attempt_status_sequence_vec[[i]],
        ci_randomness_id = randomness$ci_randomness_id,
        permutation_seed_effective = randomness$permutation_seed_effective,
        permutation_plan_spec_hash = randomness$permutation_plan_spec_hash,
        permutation_plan_hash = randomness$permutation_plan_hash,
        permutation_replicates = randomness$permutation_replicates,
        precision_execution_status = "batched-primary-data-plane",
        decision_before_verify = primary_decision$delete_edge[[i]],
        decision_after_verify = final_delete[[i]],
        mgcv_setup_cpu_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$mgcv_setup_cpu_ms %||% NA_real_,
        spectral_prepare_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$spectral_prepare_ms %||% NA_real_,
        gcv_score_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$gcv_score_ms %||% NA_real_,
        linear_solve_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$linear_solve_ms %||% NA_real_,
        residual_materialize_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$residual_materialize_ms %||% NA_real_,
        ci_test_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$ci_test_ms %||% NA_real_,
        total_ms =
          if (is.null(verifier_receipt)) NA_real_ else
            verifier_receipt$timings$total_ms %||% NA_real_
      )
    }
  }

  native <- precision_replay_layer_native(
    adjacency = state$adjacency,
    edge_x = vapply(tasks, `[[`, integer(1L), "edge_x"),
    edge_y = vapply(tasks, `[[`, integer(1L), "edge_y"),
    x = vapply(tasks, `[[`, integer(1L), "x"),
    y = vapply(tasks, `[[`, integer(1L), "y"),
    conditioning_sets = lapply(tasks, `[[`, "S"),
    p_values = final_p,
    alpha = alpha,
    pmax = state$pmax,
    trace_level = trace_level
  )
  state$adjacency <- native$adjacency
  state$pmax <- native$pMax
  for (entry in native$per.level.log) {
    x <- as.integer(entry$x)
    y <- as.integer(entry$y)
    state$sepsets[[x]][[y]] <- as.integer(entry$S)
    state$sepsets[[y]][[x]] <- as.integer(entry$S)
  }
  state$n_edge_tests[[level + 1L]] <- as.integer(native$summary$tests_replayed)
  ignored_task_index <- as.integer(native$ignored_task_index %||% integer())
  deleted_task_index <- as.integer(native$deleted_task_index %||% integer())
  if (length(ignored_task_index) > 0L) {
    verified_but_unreplayed <- sum(verified[ignored_task_index] %in% TRUE)
  }
  if (length(rows) > 0L) {
    replay_rows <- native$replay_rows
    if (nrow(replay_rows) > 0L) {
      ignored <- replay_rows$edge_already_deleted %in% TRUE
      if (any(ignored)) {
        ignored_idx <- replay_rows$task_index[ignored]
        for (idx in ignored_idx) {
          rows[[idx]]$decision_after_verify <- FALSE
        }
      }
      deleted <- replay_rows$edge_deleted %in% TRUE
      if (any(deleted)) {
        deleted_idx <- replay_rows$task_index[deleted]
        for (idx in deleted_idx) {
          rows[[idx]]$decision_after_verify <- TRUE
        }
      }
    }
    if (length(deleted_task_index) > 0L) {
      for (idx in deleted_task_index) {
        rows[[idx]]$decision_after_verify <- TRUE
      }
    }
    if (length(ignored_task_index) > 0L) {
      for (idx in ignored_task_index) {
        rows[[idx]]$decision_after_verify <- FALSE
      }
    }
    if (length(ignored_task_index) > 0L) {
      rows <- rows[setdiff(seq_along(rows), ignored_task_index)]
    }
  }
  list(
    state = state,
    trace_rows = rows,
    level_log = native$per.level.log,
    verifier_backends = verifier_backends,
    ci_backends = ci_backends,
    verifier_count = verifier_count,
    decision_changes = decision_changes,
    verifier_metrics = verifier_metrics,
    native_summary = native$summary,
    verified_but_unreplayed = verified_but_unreplayed
  )
}

fastkpc_batched_cuda_precision_skeleton <- function(data, alpha,
                                                    max_conditioning_size,
                                                    tau, ci_method, index,
                                                    legacy_index, batch_size,
                                                    fastspline_params,
                                                    cuda_residual_fallback,
                                                    hsic_params,
                                                    permutation_params,
                                                    precision_executors,
                                                    runtime_capabilities,
                                                    allow_canary,
                                                    residual_cache = TRUE,
                                                    verifier_policy =
                                                      c("none", "near-alpha"),
                                                    trace_level =
                                                      c("summary", "full", "none")) {
  verifier_policy <- match.arg(verifier_policy)
  trace_level <- match.arg(trace_level)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  if (!identical(ci_method, "dcc.gamma")) {
    stop("batched CUDA precision currently supports dcc.gamma",
         call. = FALSE)
  }
  execution_context <- fastkpc_precision_create_execution_context(
    data = data,
    residual_cache = residual_cache,
    runtime_capabilities = runtime_capabilities,
    execution_engine = "cuda"
  )
  fastkpc_precision_init_cache_stats(execution_context)
  p <- ncol(data)
  state <- list(
    p = p,
    adjacency = {
      adj <- matrix(TRUE, p, p)
      diag(adj) <- FALSE
      adj
    },
    pmax = {
      mat <- matrix(-Inf, p, p)
      diag(mat) <- 1
      mat
    },
    sepsets = fastkpc_precision_sepsets(p),
    n_edge_tests = integer(max_conditioning_size + 1L),
    test_id = 0L
  )
  trace_rows <- list()
  level_logs <- vector("list", max_conditioning_size + 1L)
  verifier_backends <- character()
  ci_backends <- c("cuda-dcov")
  total_tasks <- 0L
  total_dcov_batches <- 0L
  total_residual_batches <- 0L
  total_unique_residual_requests <- 0L
  total_verifier_count <- 0L
  total_decision_changes <- 0L
  total_native_replayed <- 0L
  total_native_ignored <- 0L
  total_native_deletions <- 0L
  total_verified_but_unreplayed <- 0L
  verifier_metrics <- list(
    total_ms = 0,
    mgcv_setup_ms = 0,
    spectral_prepare_ms = 0,
    gcv_score_ms = 0,
    cuda_solve_ms = 0,
    ci_test_ms = 0,
    cache_full_hit_count = 0L,
    cache_partial_hit_count = 0L,
    cache_miss_count = 0L,
    fallback_count = 0L,
    verifier_batch_groups = 0L,
    verifier_ci_batches = 0L
  )

  for (level in seq.int(0L, as.integer(max_conditioning_size))) {
    snapshot <- state$adjacency
    tasks <- fastkpc_batched_precision_make_layer_plan(snapshot, level)
    total_tasks <- total_tasks + length(tasks)
    primary <- fastkpc_batched_precision_primary_pvalues(
      data = data,
      tasks = tasks,
      ci_method = ci_method,
      index = index,
      legacy_index = legacy_index,
      batch_size = batch_size,
      fastspline_params = fastspline_params,
      cuda_residual_fallback = cuda_residual_fallback
    )
    total_dcov_batches <- total_dcov_batches +
      as.integer(primary$diagnostics$dcov_batches %||% 0L)
    total_residual_batches <- total_residual_batches +
      as.integer(primary$diagnostics$residual_batches %||% 0L)
    total_unique_residual_requests <- total_unique_residual_requests +
      as.integer(primary$diagnostics$unique_residual_requests %||% 0L)
    replay <- fastkpc_batched_precision_replay_level(
      data = data,
      tasks = tasks,
      pvalues = primary$pvalues,
      state = state,
      level = level,
      alpha = alpha,
      tau = tau,
      ci_method = ci_method,
      index = index,
      legacy_index = legacy_index,
      hsic_params = hsic_params,
      permutation_params = permutation_params,
      precision_executors = precision_executors,
      runtime_capabilities = runtime_capabilities,
      allow_canary = allow_canary,
      execution_context = execution_context,
      verifier_policy = verifier_policy,
      trace_level = trace_level
    )
    state <- replay$state
    trace_rows <- c(trace_rows, replay$trace_rows)
    level_logs[[level + 1L]] <- replay$level_log
    verifier_backends <- c(verifier_backends, replay$verifier_backends)
    ci_backends <- c(ci_backends, replay$ci_backends)
    total_verifier_count <- total_verifier_count + replay$verifier_count
    total_decision_changes <- total_decision_changes + replay$decision_changes
    total_native_replayed <- total_native_replayed +
      as.integer(replay$native_summary$tests_replayed %||% 0L)
    total_native_ignored <- total_native_ignored +
      as.integer(replay$native_summary$tasks_ignored_after_delete %||% 0L)
    total_native_deletions <- total_native_deletions +
      as.integer(replay$native_summary$deletions %||% 0L)
    total_verified_but_unreplayed <- total_verified_but_unreplayed +
      as.integer(replay$verified_but_unreplayed %||% 0L)
    for (name in names(verifier_metrics)) {
      verifier_metrics[[name]] <- verifier_metrics[[name]] +
        replay$verifier_metrics[[name]]
    }
  }

  trace <- if (!identical(trace_level, "full")) {
    NULL
  } else if (length(trace_rows) == 0L) {
    fastkpc_precision_trace_row(
      run_id = "fastkpc-batched-hybrid",
      backend_requested = NA_character_,
      backend_used = NA_character_
    )[0, , drop = FALSE]
  } else {
    do.call(rbind, trace_rows)
  }
  if (is.data.frame(trace)) {
    trace$edge_deleted <- trace$decision_after_verify
    trace$sepset_recorded <- ifelse(trace$edge_deleted, trace$S_key, "")
  }

  verifier_backend <- if (length(verifier_backends) == 0L) {
    NA_character_
  } else {
    paste(unique(verifier_backends), collapse = "+")
  }
  cache <- fastkpc_precision_cache_stats(execution_context, "mgcvExtractGPU")
  list(
    adjacency = state$adjacency,
    sepsets = state$sepsets,
    pMax = state$pmax,
    n.edgetests = as.integer(state$n_edge_tests),
    per.level.log = level_logs,
    backend = "cuda",
    residual_backend = "fastSplineCUDA",
    verifier_backend = verifier_backend,
    residual_backend_params = "",
    residual_device = "cuda",
    residual_device_requested = "auto",
    residual_device_reason = "",
    residual_cache = cache,
    ci_method = ci_method,
    ci_backend = if (length(unique(ci_backends)) == 1L) {
      unique(ci_backends)
    } else {
      paste(unique(ci_backends), collapse = "+")
    },
    ci_backend_reason = "",
    ci_diagnostics = list(
      ci_dcc_gamma_tests = state$test_id,
      ci_hsic_gamma_tests = 0L,
      ci_hsic_perm_tests = 0L,
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
    scheduler = "layer-precision",
    scheduler_requested = "layer",
    scheduler_diagnostics = list(
      summary = list(
        scheduler = "layer-precision",
        scheduler_requested = "layer",
        tasks_planned = total_tasks,
        tasks_evaluated = total_tasks,
        tests_replayed = total_native_replayed,
        tasks_ignored_after_delete = total_native_ignored,
        native_replay_deletions = total_native_deletions,
        verified_but_unreplayed = total_verified_but_unreplayed,
        unique_residual_requests = total_unique_residual_requests,
        dcov_batches = total_dcov_batches,
        residual_batches = total_residual_batches,
        cuda_residual_true_batched_groups =
          as.integer(total_residual_batches > 0L),
        precision_verifier_tests = total_verifier_count,
        precision_verifier_decision_changes = total_decision_changes,
        verifier_total_ms = verifier_metrics$total_ms,
        verifier_mgcv_setup_ms = verifier_metrics$mgcv_setup_ms,
        verifier_spectral_prepare_ms = verifier_metrics$spectral_prepare_ms,
        verifier_gcv_score_ms = verifier_metrics$gcv_score_ms,
        verifier_cuda_solve_ms = verifier_metrics$cuda_solve_ms,
        verifier_ci_test_ms = verifier_metrics$ci_test_ms,
        verifier_cache_full_hit_count =
          as.integer(verifier_metrics$cache_full_hit_count),
        verifier_cache_partial_hit_count =
          as.integer(verifier_metrics$cache_partial_hit_count),
        verifier_cache_miss_count =
          as.integer(verifier_metrics$cache_miss_count),
        verifier_fallback_count =
          as.integer(verifier_metrics$fallback_count),
        verifier_batch_groups =
          as.integer(verifier_metrics$verifier_batch_groups),
        verifier_ci_batches =
          as.integer(verifier_metrics$verifier_ci_batches),
        verifier_ms_per_verified_test =
          if (total_verifier_count > 0L) {
            verifier_metrics$total_ms / total_verifier_count
          } else {
            NA_real_
          },
        verifier_policy = verifier_policy,
        trace_level = trace_level,
        residual_cache_setup_cache_hits = cache$setup_cache_hits,
        residual_cache_spectral_cache_hits = cache$spectral_cache_hits
      )
    ),
    precision_trace = trace,
    precision_receipt = list(
      residual_backend_executed = "fastSplineCUDA",
      ci_backend_executed = "cuda-dcov",
      p.value = NA_real_,
      setup_fingerprint = "batched-primary",
      p_source_used = "primary:fastSplineCUDA+cuda-dcov"
    )
  )
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
                     precision_trace_level = c("auto", "summary", "full", "none"),
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
  precision_trace_level <- match.arg(precision_trace_level)
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
      allow_canary = allow_canary_mgcv_extract,
      execution_engine = engine_used
    )
  }
  if (identical(engine_used, "cpu") &&
      identical(precision_route$primary_backend, "fastSplineCUDA")) {
    precision_route$primary_backend <- "fastSplineCPU"
  }
  if (identical(precision_route$compatibility_status, "canary") &&
      identical(precision_route$compatibility_action, "warn-and-run")) {
    warning(precision_route$fallback_reason, call. = FALSE)
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
    precision_trace_level = precision_trace_level,
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
    as.integer(max_conditioning_size) <= 2L &&
    precision_requested %in% c("fast", "compatible", "hybrid") &&
    (isTRUE(precision_executors_requested) ||
       precision_requested %in% c("compatible", "hybrid"))
  use_batched_precision_primary <- isTRUE(use_precision_r_skeleton) &&
    identical(precision_requested, "fast") &&
    identical(engine_used, "cuda") &&
    identical(ci_method, "dcc.gamma") &&
    fastkpc_is_default_precision_executors(precision_executors)
  use_batched_precision_hybrid <- isTRUE(use_precision_r_skeleton) &&
    identical(precision_requested, "hybrid") &&
    identical(engine_used, "cuda") &&
    identical(ci_method, "dcc.gamma") &&
    fastkpc_is_default_precision_executors(precision_executors)
  use_batched_precision_layer <- isTRUE(use_batched_precision_primary) ||
    isTRUE(use_batched_precision_hybrid)
  if (isTRUE(use_batched_precision_layer)) {
    use_precision_r_skeleton <- FALSE
  }
  if ((isTRUE(use_precision_r_skeleton) ||
       isTRUE(use_batched_precision_hybrid)) &&
      !isTRUE(normalized$info$all_finite)) {
    stop(
      "fast_kpc precision data plane requires finite input; shared row masks are not yet supported",
      call. = FALSE
    )
  }

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
      r_precision_trace_level <- if (!isTRUE(precision_diagnostics)) {
        "none"
      } else if (identical(precision_trace_level, "auto")) {
        "summary"
      } else {
        precision_trace_level
      }
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
        allow_canary = allow_canary_mgcv_extract,
        residual_cache = residual_cache,
        execution_engine = engine_used,
        trace_level = r_precision_trace_level
      )
      list(skeleton = skeleton, orientation = NULL)
    } else if (isTRUE(use_batched_precision_layer)) {
      batched_precision_trace_level <- if (!isTRUE(precision_diagnostics)) {
        "none"
      } else if (identical(precision_trace_level, "auto")) {
        "summary"
      } else {
        precision_trace_level
      }
      skeleton <- fastkpc_batched_cuda_precision_skeleton(
        matrix_data, alpha, max_conditioning_size,
        tau = tau,
        ci_method = ci_method,
        index = index,
        legacy_index = legacy_index,
        batch_size = batch_size,
        fastspline_params = fastspline_params,
        cuda_residual_fallback = cuda_residual_fallback,
        hsic_params = hsic_params,
        permutation_params = permutation_params,
        precision_executors = precision_executors,
        runtime_capabilities = runtime_capabilities,
        allow_canary = allow_canary_mgcv_extract,
        residual_cache = residual_cache,
        verifier_policy = if (isTRUE(use_batched_precision_hybrid)) {
          "near-alpha"
        } else {
          "none"
        },
        trace_level = batched_precision_trace_level
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
  if (isTRUE(use_batched_precision_primary)) {
    config$precision_execution_status <- "batched-primary-data-plane"
    config$backend_executed <- timed$value$skeleton$residual_backend %||%
      config$backend_executed
    config$backend_used <- config$backend_executed
    config$ci_backend <- timed$value$skeleton$ci_backend %||% config$ci_backend
  }
  if (isTRUE(use_batched_precision_hybrid)) {
    config$precision_execution_status <- "batched-primary-data-plane"
    config$backend_executed <- "fastSplineCUDA"
    config$backend_used <- config$backend_executed
    config$verifier_executed <- timed$value$skeleton$verifier_backend %||%
      config$verifier_executed
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
