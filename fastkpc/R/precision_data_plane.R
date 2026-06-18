source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/dcov_exact.R")

fastkpc_default_precision_executors <- function() {
  list(
    `direct-ci` = fastkpc_execute_ci_direct,
    fastSplineCPU = fastkpc_execute_ci_fast_spline_cpu,
    mgcvExtractGPUGCV = fastkpc_execute_ci_mgcv_extract_cpu,
    `legacy-mgcv` = fastkpc_execute_ci_legacy_mgcv
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

fastkpc_precision_normalize_p <- function(p_raw, na_delete = TRUE) {
  p_raw <- as.numeric(p_raw)[1L]
  p_was_nonfinite <- !is.finite(p_raw)
  if (p_was_nonfinite) {
    if (isTRUE(na_delete)) {
      return(list(
        p_raw = p_raw,
        p_used = 1.0,
        p_was_nonfinite = TRUE,
        nonfinite_action = "na-delete-use-1"
      ))
    }
    return(list(
      p_raw = p_raw,
      p_used = 0.0,
      p_was_nonfinite = TRUE,
      nonfinite_action = "na-keep-use-0"
    ))
  }
  list(
    p_raw = p_raw,
    p_used = p_raw,
    p_was_nonfinite = FALSE,
    nonfinite_action = ""
  )
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

fastkpc_legacy_mgcv_residual <- function(data, target, S) {
  if (length(S) == 0L) return(as.numeric(data[, target]))
  if (length(S) > 1L) {
    stop("legacy mgcv CPU vertical slice supports |S| <= 1", call. = FALSE)
  }
  fastkpc_require_mgcv()
  local_data <- data.frame(
    .target = as.numeric(data[, target]),
    s1 = as.numeric(data[, S[1L]])
  )
  form <- stats::as.formula(".target ~ s(s1, bs = 'tp')",
                            env = asNamespace("mgcv"))
  fit <- mgcv::gam(
    formula = form,
    data = local_data,
    family = stats::gaussian(),
    method = "GCV.Cp"
  )
  as.numeric(stats::residuals(fit))
}

fastkpc_execute_ci_legacy_mgcv <- function(data, x, y, S, ci_method,
                                           index, legacy_index,
                                           hsic_params,
                                           permutation_params, route,
                                           role = "primary") {
  start <- proc.time()[["elapsed"]]
  rx <- fastkpc_legacy_mgcv_residual(data, x, S)
  ry <- fastkpc_legacy_mgcv_residual(data, y, S)
  ci <- fastkpc_precision_ci_from_residuals(
    rx, ry, ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "legacy-mgcv",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = route$setup_fingerprint %||%
      paste0("legacy-mgcv:S:", fastkpc_precision_S_key(S)),
    p_source_used = paste0(role, ":legacy-mgcv+native-cpu"),
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

fastkpc_execute_backend_attempt <- function(data, x, y, S, route, role,
                                            backend,
                                            ci_method, index, legacy_index,
                                            hsic_params, permutation_params,
                                            precision_executors) {
  start <- proc.time()[["elapsed"]]
  attempt_route <- route
  attempt_route$primary_backend <- backend
  value <- tryCatch({
    receipt <- fastkpc_precision_execute_ci(
      data = data, x = x, y = y, S = S, route = attempt_route, role = role,
      ci_method = ci_method, index = index, legacy_index = legacy_index,
      hsic_params = hsic_params, permutation_params = permutation_params,
      precision_executors = precision_executors
    )
    list(status = "ok", receipt = receipt, error = NULL)
  }, error = function(e) {
    list(status = "error", receipt = NULL, error = e)
  })
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    backend_planned = route$primary_backend %||% backend,
    backend_attempted = backend,
    backend_executed = value$receipt$residual_backend_executed %||% NA_character_,
    attempt_status = value$status,
    error_class = if (is.null(value$error)) "" else class(value$error)[1L],
    error_message = if (is.null(value$error)) "" else conditionMessage(value$error),
    fallback_triggered = FALSE,
    fallback_reason = "",
    elapsed_ms = elapsed,
    receipt = value$receipt
  )
}

fastkpc_nonempty_backend <- function(value, fallback) {
  if (is.null(value) || length(value) == 0L || is.na(value[1L]) ||
      !nzchar(as.character(value[1L]))) {
    return(fallback)
  }
  as.character(value[1L])
}

fastkpc_execute_route_with_fallback <- function(data, x, y, S, route, role,
                                                ci_method, index, legacy_index,
                                                hsic_params,
                                                permutation_params,
                                                precision_executors,
                                                fallback_backend = "legacy-mgcv") {
  primary_backend <- fastkpc_nonempty_backend(route$primary_backend, "")
  primary <- fastkpc_execute_backend_attempt(
    data = data, x = x, y = y, S = S, route = route, role = role,
    backend = primary_backend, ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params,
    precision_executors = precision_executors
  )
  attempts <- list(primary)
  if (identical(primary$attempt_status, "ok")) {
    primary$attempts <- attempts
    primary$attempt_count <- length(attempts)
    return(primary)
  }
  fallback_backend <- fastkpc_nonempty_backend(fallback_backend, "")
  if (!nzchar(fallback_backend)) {
    stop(primary$error_message, call. = FALSE)
  }
  fallback_route <- route
  fallback_route$primary_backend <- fallback_backend
  fallback <- fastkpc_execute_backend_attempt(
    data = data, x = x, y = y, S = S, route = fallback_route, role = role,
    backend = fallback_backend, ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params,
    precision_executors = precision_executors
  )
  fallback$fallback_triggered <- TRUE
  fallback$fallback_reason <- primary$error_message
  attempts[[length(attempts) + 1L]] <- fallback
  if (!identical(fallback$attempt_status, "ok")) {
    stop(paste0("backend ", primary_backend, " failed: ",
                primary$error_message, "; fallback ", fallback_backend,
                " failed: ", fallback$error_message), call. = FALSE)
  }
  fallback$attempts <- attempts
  fallback$attempt_count <- length(attempts)
  fallback
}

fastkpc_precision_primary_backend <- function(route, precision) {
  if (identical(precision, "hybrid")) {
    return("fastSplineCPU")
  }
  fastkpc_nonempty_backend(route$primary_backend, "fastSplineCPU")
}

fastkpc_precision_resolve_test <- function(data, x, y, S, route, precision,
                                           alpha, tau, ci_method, index,
                                           legacy_index, hsic_params,
                                           permutation_params,
                                           precision_executors,
                                           na_delete = TRUE) {
  primary_route <- route
  primary_route$primary_backend <- fastkpc_precision_primary_backend(route, precision)
  primary_fallback <- NA_character_
  if (identical(precision, "compatible")) {
    route_fallback <- fastkpc_nonempty_backend(route$fallback_backend,
                                               "legacy-mgcv")
    if (!identical(primary_route$primary_backend, route_fallback)) {
      primary_fallback <- route_fallback
    }
  }
  primary_exec <- fastkpc_execute_route_with_fallback(
    data = data, x = x, y = y, S = S, route = primary_route, role = "primary",
    ci_method = ci_method, index = index, legacy_index = legacy_index,
    hsic_params = hsic_params, permutation_params = permutation_params,
    precision_executors = precision_executors,
    fallback_backend = primary_fallback
  )
  primary_receipt <- primary_exec$receipt
  primary_info <- fastkpc_precision_normalize_p(primary_receipt$p.value,
                                                na_delete)

  verifier_exec <- NULL
  verifier_info <- NULL
  near_alpha <- FALSE
  if (identical(precision, "hybrid") && length(S) > 0L) {
    near_alpha <- primary_info$p_was_nonfinite ||
      fastkpc_near_alpha_trigger(primary_info$p_raw, alpha, tau)
  }

  chosen_receipt <- primary_receipt
  chosen_info <- primary_info
  p_source <- primary_receipt$p_source_used
  fallback_triggered <- isTRUE(primary_exec$fallback_triggered)
  fallback_reason <- primary_exec$fallback_reason %||% ""
  attempt_count <- primary_exec$attempt_count %||% 1L

  if (isTRUE(near_alpha)) {
    verifier_backend <- fastkpc_nonempty_backend(route$verifier_backend,
                                                 "mgcvExtractGPUGCV")
    verifier_route <- route
    verifier_route$primary_backend <- verifier_backend
    verifier_exec <- fastkpc_execute_route_with_fallback(
      data = data, x = x, y = y, S = S, route = verifier_route,
      role = "verifier", ci_method = ci_method, index = index,
      legacy_index = legacy_index, hsic_params = hsic_params,
      permutation_params = permutation_params,
      precision_executors = precision_executors,
      fallback_backend = fastkpc_nonempty_backend(route$fallback_backend,
                                                  "legacy-mgcv")
    )
    verifier_fallback_backend <- fastkpc_nonempty_backend(route$fallback_backend,
                                                          "legacy-mgcv")
    verifier_info <- fastkpc_precision_normalize_p(
      verifier_exec$receipt$p.value, na_delete
    )
    if (isTRUE(verifier_info$p_was_nonfinite) &&
        !identical(verifier_exec$receipt$residual_backend_executed,
                   verifier_fallback_backend)) {
      legacy_route <- route
      legacy_route$primary_backend <- verifier_fallback_backend
      legacy_exec <- fastkpc_execute_route_with_fallback(
        data = data, x = x, y = y, S = S, route = legacy_route,
        role = "verifier", ci_method = ci_method, index = index,
        legacy_index = legacy_index, hsic_params = hsic_params,
        permutation_params = permutation_params,
        precision_executors = precision_executors,
        fallback_backend = NA_character_
      )
      verifier_exec <- legacy_exec
      verifier_info <- fastkpc_precision_normalize_p(
        verifier_exec$receipt$p.value, na_delete
      )
      verifier_exec$fallback_triggered <- TRUE
      verifier_exec$fallback_reason <- "verifier returned non-finite p-value"
    }
    chosen_receipt <- verifier_exec$receipt
    chosen_info <- verifier_info
    p_source <- verifier_exec$receipt$p_source_used
    fallback_triggered <- isTRUE(verifier_exec$fallback_triggered)
    fallback_reason <- verifier_exec$fallback_reason
    attempt_count <- attempt_count + (verifier_exec$attempt_count %||% 1L)
  }

  list(
    pval = chosen_info$p_used,
    p_raw = chosen_info$p_raw,
    p_info = chosen_info,
    receipt = chosen_receipt,
    primary_receipt = primary_receipt,
    primary_info = primary_info,
    verifier_receipt = if (is.null(verifier_exec)) NULL else verifier_exec$receipt,
    verifier_info = verifier_info,
    near_alpha_triggered = near_alpha,
    p_source_used = p_source,
    fallback_triggered = fallback_triggered,
    fallback_reason = fallback_reason,
    attempt_count = attempt_count,
    decision_before_verify = primary_info$p_used >= alpha,
    decision_after_verify = chosen_info$p_used >= alpha
  )
}

fastkpc_precision_trace_for_test <- function(resolved, route, run_id,
                                             conditioning_level,
                                             canonical_test_order_id,
                                             x, y, S,
                                             conditioning_target_side) {
  receipt <- resolved$receipt
  primary_receipt <- resolved$primary_receipt
  verifier_receipt <- resolved$verifier_receipt
  verifier_info <- resolved$verifier_info
  verifier_executed <- if (is.null(verifier_receipt)) {
    NA_character_
  } else {
    verifier_receipt$residual_backend_executed
  }
  verifier_ci <- if (is.null(verifier_receipt)) {
    NA_character_
  } else {
    verifier_receipt$ci_backend_executed
  }
  verifier_p_raw <- if (is.null(verifier_info)) NA_real_ else verifier_info$p_raw
  verifier_p_used <- if (is.null(verifier_info)) NA_real_ else verifier_info$p_used
  fastkpc_precision_trace_row(
    run_id = run_id,
    scenario_id = "fast_kpc",
    conditioning_level = conditioning_level,
    canonical_test_order_id = canonical_test_order_id,
    setup_fingerprint = receipt$setup_fingerprint %||%
      route$setup_fingerprint,
    target_id = paste(x, y, sep = "|"),
    x = x,
    y = y,
    S_key = fastkpc_precision_S_key(S),
    conditioning_target_side = conditioning_target_side,
    backend_requested = route$primary_backend,
    backend_used = receipt$residual_backend_executed,
    backend_planned = route$primary_backend,
    backend_executed = receipt$residual_backend_executed,
    verifier_backend = route$verifier_backend %||% NA_character_,
    verifier_planned = route$verifier_backend %||% NA_character_,
    verifier_executed = verifier_executed,
    compatibility_action = route$compatibility_action %||% "",
    fallback_reason = resolved$fallback_reason %||%
      route$fallback_reason %||% "",
    primary_p = resolved$primary_info$p_used,
    verifier_p = verifier_p_used,
    p_used = resolved$pval,
    p_raw = resolved$p_raw,
    p_was_nonfinite = resolved$p_info$p_was_nonfinite,
    nonfinite_action = resolved$p_info$nonfinite_action,
    p_source_used = resolved$p_source_used,
    primary_residual_backend_executed =
      primary_receipt$residual_backend_executed,
    primary_ci_backend_executed = primary_receipt$ci_backend_executed,
    primary_p_raw = resolved$primary_info$p_raw,
    primary_p_used = resolved$primary_info$p_used,
    near_alpha_triggered = resolved$near_alpha_triggered,
    verifier_residual_backend_executed = verifier_executed,
    verifier_ci_backend_executed = verifier_ci,
    verifier_p_raw = verifier_p_raw,
    verifier_p_used = verifier_p_used,
    fallback_triggered = resolved$fallback_triggered,
    attempt_count = resolved$attempt_count,
    precision_execution_status = "data-plane-executed",
    decision_before_verify = resolved$decision_before_verify,
    decision_after_verify = resolved$decision_after_verify,
    ci_test_ms = receipt$timings$ci_test_ms %||% NA_real_
  )
}

fastkpc_r_skeleton_precision <- function(data, alpha, max_conditioning_size,
                                         precision, tau, ci_method, index,
                                         legacy_index, hsic_params,
                                         permutation_params,
                                         precision_executors,
                                         runtime_capabilities,
                                         allow_canary = FALSE,
                                         na_delete = TRUE) {
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
  verifier_backends <- character()
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
          resolved <- fastkpc_precision_resolve_test(
            data = data, x = x, y = y, S = S, route = route,
            precision = precision, alpha = alpha, tau = tau,
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors,
            na_delete = na_delete
          )
          receipt <- resolved$receipt
          last_receipt <- receipt
          primary_receipt <- resolved$primary_receipt
          if (!identical(route$primary_backend, "direct-ci")) {
            executed_backends <- c(executed_backends,
                                   primary_receipt$residual_backend_executed)
          }
          if (!is.null(resolved$verifier_receipt)) {
            verifier_backends <- c(verifier_backends,
                                   resolved$verifier_receipt$residual_backend_executed)
          }
          ci_backends <- c(ci_backends, primary_receipt$ci_backend_executed)
          if (!is.null(resolved$verifier_receipt)) {
            ci_backends <- c(ci_backends,
                             resolved$verifier_receipt$ci_backend_executed)
          }
          pval <- resolved$pval
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
          trace_rows[[length(trace_rows) + 1L]] <- fastkpc_precision_trace_for_test(
            resolved = resolved,
            route = route,
            run_id = "fastkpc-r-skeleton",
            conditioning_level = ord,
            canonical_test_order_id = test_id,
            x = x,
            y = y,
            S = S,
            conditioning_target_side = "x"
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
          resolved <- fastkpc_precision_resolve_test(
            data = data, x = y, y = x, S = S, route = route,
            precision = precision, alpha = alpha, tau = tau,
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors,
            na_delete = na_delete
          )
          receipt <- resolved$receipt
          last_receipt <- receipt
          primary_receipt <- resolved$primary_receipt
          if (!identical(route$primary_backend, "direct-ci")) {
            executed_backends <- c(executed_backends,
                                   primary_receipt$residual_backend_executed)
          }
          if (!is.null(resolved$verifier_receipt)) {
            verifier_backends <- c(verifier_backends,
                                   resolved$verifier_receipt$residual_backend_executed)
          }
          ci_backends <- c(ci_backends, primary_receipt$ci_backend_executed)
          if (!is.null(resolved$verifier_receipt)) {
            ci_backends <- c(ci_backends,
                             resolved$verifier_receipt$ci_backend_executed)
          }
          pval <- resolved$pval
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
          trace_rows[[length(trace_rows) + 1L]] <- fastkpc_precision_trace_for_test(
            resolved = resolved,
            route = route,
            run_id = "fastkpc-r-skeleton",
            conditioning_level = ord,
            canonical_test_order_id = test_id,
            x = y,
            y = x,
            S = S,
            conditioning_target_side = "y"
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
  trace$sepset_recorded <- ifelse(trace$edge_deleted, trace$S_key, "")

  backend_candidates <- unique(executed_backends)
  if (length(backend_candidates) == 0L) backend_candidates <- "direct-ci"
  backend <- if (length(backend_candidates) == 1L) {
    backend_candidates
  } else {
    paste(backend_candidates, collapse = "+")
  }
  verifier_backend_candidates <- unique(verifier_backends)
  verifier_backend <- if (length(verifier_backend_candidates) == 0L) {
    NA_character_
  } else if (length(verifier_backend_candidates) == 1L) {
    verifier_backend_candidates
  } else {
    paste(verifier_backend_candidates, collapse = "+")
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
    verifier_backend = verifier_backend,
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
