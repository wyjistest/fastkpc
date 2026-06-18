source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/dcov_exact.R")

fastkpc_default_precision_executors <- function() {
  list(
    `direct-ci` = fastkpc_execute_ci_direct,
    fastSplineCPU = fastkpc_execute_ci_fast_spline_cpu,
    mgcvExtractGPUGCV = fastkpc_execute_ci_mgcv_extract_cpu,
    `legacy-mgcv` = fastkpc_execute_ci_mgcv_extract_cpu
  )
}

fastkpc_precision_S_key <- function(S) {
  if (length(S) == 0L) return("")
  paste(as.integer(S), collapse = "|")
}

fastkpc_precision_sepsets <- function(p) {
  replicate(p, replicate(p, integer(), simplify = FALSE), simplify = FALSE)
}

fastkpc_precision_combinations <- function(values, choose) {
  values <- as.integer(values)
  if (choose == 0L) return(list(integer()))
  if (length(values) < choose) return(list())
  lapply(combn(seq_along(values), choose, simplify = FALSE), function(idx) {
    as.integer(values[idx])
  })
}

fastkpc_precision_neighbors <- function(adjacency, vertex, excluded) {
  out <- which(adjacency[, vertex] & seq_len(nrow(adjacency)) != excluded)
  as.integer(out)
}

fastkpc_precision_group_route <- function(precision, alpha, tau, S,
                                          runtime_capabilities,
                                          allow_canary = FALSE) {
  if (length(S) == 0L) {
    return(list(
      precision = precision,
      primary_backend = "direct-ci",
      verifier_backend = NA_character_,
      compatibility_status = "direct",
      compatibility_action = "run-direct-ci",
      compatibility_claim = "no-residualization",
      canonical_replay_required = precision %in% c("compatible", "hybrid"),
      fallback_backend = NA_character_,
      fallback_reason = "",
      setup_fingerprint = "direct-ci:S:",
      runtime_capabilities = runtime_capabilities
    ))
  }
  formula_class <- fastkpc_regrxons_formula_class(S)
  penalty_count <- if (length(S) == 0L) 0L else 1L
  fastkpc_resolve_backend_request(
    precision = precision,
    alpha = alpha,
    tau = tau,
    S = S,
    formula_class = formula_class,
    penalty_count = penalty_count,
    family = "gaussian",
    link = "identity",
    setup_fingerprint = paste0("S:", fastkpc_precision_S_key(S)),
    runtime_capabilities = runtime_capabilities,
    fallback_backend = "legacy-mgcv",
    allow_canary = allow_canary
  )
}

fastkpc_execute_ci_direct <- function(data, x, y, S, ci_method,
                                      index, legacy_index,
                                      hsic_params,
                                      permutation_params, route,
                                      role = "primary") {
  start <- proc.time()[["elapsed"]]
  ci <- fastkpc_precision_ci_from_residuals(
    data[, x], data[, y], ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "direct-ci",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = route$setup_fingerprint %||% "direct-ci:S:",
    p_source_used = paste0(role, ":direct-ci+native-cpu"),
    timings = list(ci_test_ms = elapsed)
  )
}

fastkpc_execute_ci_fast_spline_cpu <- function(data, x, y, S, ci_method,
                                               index, legacy_index,
                                               hsic_params,
                                               permutation_params, route,
                                               role = "primary") {
  start <- proc.time()[["elapsed"]]
  if (length(S) == 0L) {
    rx <- data[, x]
    ry <- data[, y]
  } else {
    S_data <- data[, S, drop = FALSE]
    rx <- fastspline_residual(data[, x], S_data)$residuals
    ry <- fastspline_residual(data[, y], S_data)$residuals
  }
  ci <- fastkpc_precision_ci_from_residuals(
    rx, ry, ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "fastSplineCPU",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = route$setup_fingerprint %||%
      paste0("fastSpline:S:", fastkpc_precision_S_key(S)),
    p_source_used = paste0(role, ":fastSplineCPU+native-cpu"),
    timings = list(ci_test_ms = elapsed)
  )
}

fastkpc_mgcv_batch_residuals_for_pair <- function(data, x, y, S) {
  if (length(S) == 0L) {
    return(list(
      residuals = cbind(data[, x], data[, y]),
      setup_fingerprint = paste0("direct:", x, "-", y)
    ))
  }
  if (length(S) > 1L) {
    stop("compatible CPU vertical slice supports |S| <= 1", call. = FALSE)
  }
  S_data <- data.frame(s1 = data[, S[1L]])
  Y <- cbind(x = data[, x], y = data[, y])
  batch <- fastkpc_mgcv_extract_batch(
    Y = Y,
    S_data = S_data,
    S = S,
    target_ids = c(x, y),
    formula_class = "full-smooth"
  )
  list(
    residuals = batch$residuals,
    setup_fingerprint = batch$setup_fingerprint$fingerprint
  )
}

fastkpc_execute_ci_mgcv_extract_cpu <- function(data, x, y, S, ci_method,
                                                index, legacy_index,
                                                hsic_params,
                                                permutation_params, route,
                                                role = "primary") {
  start <- proc.time()[["elapsed"]]
  batch <- fastkpc_mgcv_batch_residuals_for_pair(data, x, y, S)
  ci <- fastkpc_precision_ci_from_residuals(
    batch$residuals[, 1L], batch$residuals[, 2L],
    ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "mgcvExtractCPU",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = batch$setup_fingerprint,
    p_source_used = paste0(role, ":mgcvExtractCPU+native-cpu"),
    timings = list(ci_test_ms = elapsed)
  )
}

fastkpc_precision_ci_from_residuals <- function(rx, ry, ci_method,
                                                index, legacy_index,
                                                hsic_params,
                                                permutation_params) {
  if (identical(ci_method, "dcc.gamma")) {
    return(dcov_gamma_exact(rx, ry, index = index,
                            legacy_index = legacy_index))
  }
  if (identical(ci_method, "hsic.gamma")) {
    sig <- hsic_params$sig %||% 1
    return(fast_hsic_gamma_cpp(rx, ry, sig = sig))
  }
  if (identical(ci_method, "hsic.perm")) {
    sig <- hsic_params$sig %||% 1
    return(fast_hsic_perm_cpp(
      rx, ry, sig = sig,
      replicates = permutation_params$replicates %||% 100L,
      seed = permutation_params$seed %||% NULL,
      include_observed = permutation_params$include_observed %||% TRUE
    ))
  }
  stop("Unknown ci_method: ", ci_method, call. = FALSE)
}

fastkpc_precision_execute_ci <- function(data, x, y, S, route, role,
                                         ci_method, index, legacy_index,
                                         hsic_params, permutation_params,
                                         precision_executors) {
  backend <- route$primary_backend
  if (identical(backend, "fastSplineCUDA") &&
      is.null(precision_executors[[backend]]) &&
      !is.null(precision_executors$fastSplineCPU)) {
    backend <- "fastSplineCPU"
  }
  executor <- precision_executors[[backend]]
  if (is.null(executor)) {
    stop("No precision executor registered for backend: ", backend,
         call. = FALSE)
  }
  executor(
    data = data, x = x, y = y, S = S, ci_method = ci_method,
    index = index, legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params, route = route, role = role
  )
}

fastkpc_r_skeleton_precision <- function(data, alpha, max_conditioning_size,
                                         precision, tau, ci_method, index,
                                         legacy_index, hsic_params,
                                         permutation_params,
                                         precision_executors,
                                         runtime_capabilities,
                                         allow_canary = FALSE) {
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  p <- ncol(data)
  adjacency <- matrix(TRUE, p, p)
  diag(adjacency) <- FALSE
  pmax <- matrix(-Inf, p, p)
  diag(pmax) <- 1
  sepsets <- fastkpc_precision_sepsets(p)
  n_edge_tests <- integer(max_conditioning_size + 1L)
  trace_rows <- list()
  level_logs <- vector("list", max_conditioning_size + 1L)
  last_receipt <- NULL
  test_id <- 0L
  executed_backends <- character()
  ci_backends <- character()

  for (ord in seq.int(0L, as.integer(max_conditioning_size))) {
    snapshot <- adjacency
    delete_edges <- matrix(FALSE, p, p)
    level_log <- list()
    for (x in seq_len(p - 1L)) {
      for (y in seq.int(x + 1L, p)) {
        if (!snapshot[x, y]) next
        edge_done <- FALSE
        nx <- fastkpc_precision_neighbors(snapshot, x, y)
        for (S in fastkpc_precision_combinations(nx, ord)) {
          test_id <- test_id + 1L
          route <- fastkpc_precision_group_route(
            precision = precision, alpha = alpha, tau = tau, S = S,
            runtime_capabilities = runtime_capabilities,
            allow_canary = allow_canary
          )
          receipt <- fastkpc_precision_execute_ci(
            data = data, x = x, y = y, S = S, route = route, role = "primary",
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors
          )
          last_receipt <- receipt
          if (!identical(route$primary_backend, "direct-ci")) {
            executed_backends <- c(executed_backends,
                                   receipt$residual_backend_executed)
          }
          ci_backends <- c(ci_backends, receipt$ci_backend_executed)
          pval <- as.numeric(receipt$p.value)
          if (!is.finite(pval)) pval <- 1.0
          if (pval > pmax[x, y]) {
            pmax[x, y] <- pval
            pmax[y, x] <- pval
          }
          deleted <- pval >= alpha
          if (deleted) {
            delete_edges[x, y] <- TRUE
            delete_edges[y, x] <- TRUE
            sepsets[[x]][[y]] <- as.integer(S)
            sepsets[[y]][[x]] <- as.integer(S)
            level_log[[length(level_log) + 1L]] <- list(
              x = x, y = y, S = as.integer(S), p.value = pval
            )
            edge_done <- TRUE
          }
          trace_rows[[length(trace_rows) + 1L]] <- fastkpc_precision_trace_row(
            run_id = "fastkpc-r-skeleton",
            scenario_id = "fast_kpc",
            conditioning_level = ord,
            canonical_test_order_id = test_id,
            setup_fingerprint = receipt$setup_fingerprint %||%
              route$setup_fingerprint,
            target_id = paste(x, y, sep = "|"),
            backend_requested = route$primary_backend,
            backend_used = receipt$residual_backend_executed,
            backend_planned = route$primary_backend,
            backend_executed = receipt$residual_backend_executed,
            verifier_backend = route$verifier_backend %||% NA_character_,
            verifier_planned = route$verifier_backend %||% NA_character_,
            verifier_executed = NA_character_,
            compatibility_action = route$compatibility_action %||% "",
            fallback_reason = route$fallback_reason %||% "",
            primary_p = pval,
            p_used = pval,
            p_source_used = receipt$p_source_used,
            precision_execution_status = "data-plane-executed",
            decision_before_verify = pval >= alpha,
            decision_after_verify = pval >= alpha,
            ci_test_ms = receipt$timings$ci_test_ms %||% NA_real_
          )
          n_edge_tests[[ord + 1L]] <- n_edge_tests[[ord + 1L]] + 1L
          if (edge_done) break
        }
        if (edge_done) next
        ny <- fastkpc_precision_neighbors(snapshot, y, x)
        for (S in fastkpc_precision_combinations(ny, ord)) {
          test_id <- test_id + 1L
          route <- fastkpc_precision_group_route(
            precision = precision, alpha = alpha, tau = tau, S = S,
            runtime_capabilities = runtime_capabilities,
            allow_canary = allow_canary
          )
          receipt <- fastkpc_precision_execute_ci(
            data = data, x = y, y = x, S = S, route = route, role = "primary",
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors
          )
          last_receipt <- receipt
          if (!identical(route$primary_backend, "direct-ci")) {
            executed_backends <- c(executed_backends,
                                   receipt$residual_backend_executed)
          }
          ci_backends <- c(ci_backends, receipt$ci_backend_executed)
          pval <- as.numeric(receipt$p.value)
          if (!is.finite(pval)) pval <- 1.0
          if (pval > pmax[x, y]) {
            pmax[x, y] <- pval
            pmax[y, x] <- pval
          }
          deleted <- pval >= alpha
          if (deleted) {
            delete_edges[x, y] <- TRUE
            delete_edges[y, x] <- TRUE
            sepsets[[x]][[y]] <- as.integer(S)
            sepsets[[y]][[x]] <- as.integer(S)
            level_log[[length(level_log) + 1L]] <- list(
              x = x, y = y, S = as.integer(S), p.value = pval
            )
            edge_done <- TRUE
          }
          trace_rows[[length(trace_rows) + 1L]] <- fastkpc_precision_trace_row(
            run_id = "fastkpc-r-skeleton",
            scenario_id = "fast_kpc",
            conditioning_level = ord,
            canonical_test_order_id = test_id,
            setup_fingerprint = receipt$setup_fingerprint %||%
              route$setup_fingerprint,
            target_id = paste(y, x, sep = "|"),
            backend_requested = route$primary_backend,
            backend_used = receipt$residual_backend_executed,
            backend_planned = route$primary_backend,
            backend_executed = receipt$residual_backend_executed,
            verifier_backend = route$verifier_backend %||% NA_character_,
            verifier_planned = route$verifier_backend %||% NA_character_,
            verifier_executed = NA_character_,
            compatibility_action = route$compatibility_action %||% "",
            fallback_reason = route$fallback_reason %||% "",
            primary_p = pval,
            p_used = pval,
            p_source_used = receipt$p_source_used,
            precision_execution_status = "data-plane-executed",
            decision_before_verify = pval >= alpha,
            decision_after_verify = pval >= alpha,
            ci_test_ms = receipt$timings$ci_test_ms %||% NA_real_
          )
          n_edge_tests[[ord + 1L]] <- n_edge_tests[[ord + 1L]] + 1L
          if (edge_done) break
        }
      }
    }
    adjacency[delete_edges] <- FALSE
    level_logs[[ord + 1L]] <- level_log
  }

  trace <- if (length(trace_rows) == 0L) {
    fastkpc_precision_trace_row(
      run_id = "fastkpc-r-skeleton",
      backend_requested = NA_character_,
      backend_used = NA_character_
    )[0, , drop = FALSE]
  } else {
    do.call(rbind, trace_rows)
  }
  trace$edge_deleted <- trace$decision_after_verify
  trace$sepset_recorded <- ""
  deleted_idx <- which(trace$edge_deleted)
  if (length(deleted_idx) > 0L) {
    trace$sepset_recorded[deleted_idx] <- vapply(deleted_idx, function(i) {
      fp <- trace$setup_fingerprint[i]
      if (grepl("spy:", fp, fixed = TRUE)) sub("^spy:", "", fp) else ""
    }, character(1))
  }

  backend_candidates <- unique(executed_backends)
  if (length(backend_candidates) == 0L) backend_candidates <- "direct-ci"
  backend <- if (length(backend_candidates) == 1L) {
    backend_candidates
  } else {
    paste(backend_candidates, collapse = "+")
  }
  ci_backend <- if (length(unique(ci_backends)) == 1L) {
    unique(ci_backends)
  } else {
    paste(unique(ci_backends), collapse = "+")
  }
  list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pmax,
    n.edgetests = as.integer(n_edge_tests),
    per.level.log = level_logs,
    backend = "cpu",
    residual_backend = backend,
    residual_backend_params = "",
    residual_cache = list(
      enabled = FALSE,
      requests = 0L,
      hits = 0L,
      misses = 0L,
      computations = 0L,
      stored_vectors = 0L,
      stored_values = 0L,
      backend_name = backend
    ),
    ci_method = ci_method,
    ci_backend = ci_backend,
    ci_backend_reason = "",
    ci_diagnostics = list(
      ci_dcc_gamma_tests = if (identical(ci_method, "dcc.gamma")) test_id else 0L,
      ci_hsic_gamma_tests = if (identical(ci_method, "hsic.gamma")) test_id else 0L,
      ci_hsic_perm_tests = if (identical(ci_method, "hsic.perm")) test_id else 0L,
      ci_hsic_permutation_replicates = if (identical(ci_method, "hsic.perm")) {
        test_id * as.integer(permutation_params$replicates %||% 100L)
      } else {
        0L
      },
      ci_hsic_gamma_cuda_tests = 0L,
      ci_hsic_perm_cuda_tests = 0L,
      ci_hsic_cuda_batches = 0L,
      ci_hsic_cuda_pairs = 0L,
      ci_hsic_cuda_fallback_tests = 0L,
      ci_hsic_cuda_memory_bytes = 0,
      ci_hsic_cuda_max_n = 0L,
      ci_hsic_cuda_max_batch_pairs = 0L
    ),
    scheduler = "r-precision",
    scheduler_diagnostics = list(
      summary = list(
        tasks_planned = test_id,
        tasks_evaluated = test_id,
        tests_replayed = test_id,
        tasks_ignored_after_delete = 0L,
        unique_residual_requests = test_id,
        dcov_batches = 0L,
        residual_batches = 0L
      )
    ),
    precision_trace = trace,
    precision_receipt = last_receipt
  )
}
