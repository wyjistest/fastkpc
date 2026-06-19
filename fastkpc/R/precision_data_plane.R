source("fastkpc/R/mgcv_extract_oracle.R")
source("fastkpc/R/dcov_exact.R")

fastkpc_default_precision_executors <- function() {
  list(
    `direct-ci` = fastkpc_execute_ci_direct,
    fastSplineCPU = fastkpc_execute_ci_fast_spline_cpu,
    fastSplineCUDA = fastkpc_execute_ci_fast_spline_cuda,
    mgcvExtractCPUGCVBridge = fastkpc_execute_ci_mgcv_extract_cpu,
    mgcvExtractGPUGCV = fastkpc_execute_ci_mgcv_extract_gpu,
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

fastkpc_precision_ci_randomness <- function(ci_method, permutation_params,
                                            canonical_test_order_id) {
  replicates <- as.integer(permutation_params$replicates %||% 100L)
  base_seed <- as.double((permutation_params$seed %||% 0L)[1L])
  if (!is.finite(base_seed) || is.na(base_seed)) base_seed <- 0
  effective_seed <- if (identical(ci_method, "hsic.perm")) {
    test_id <- as.double(canonical_test_order_id[1L])
    if (!is.finite(test_id) || is.na(test_id)) test_id <- 0
    seed <- as.integer((base_seed + 1000003 * test_id) %%
                         .Machine$integer.max)
    if (length(seed) != 1L || is.na(seed) || seed < 0L) {
      stop("failed to derive finite permutation seed", call. = FALSE)
    }
    seed
  } else {
    NA_integer_
  }
  plan_spec_hash <- if (identical(ci_method, "hsic.perm")) {
    fastkpc_hash_object(list(
      ci_method = ci_method,
      base_seed = as.integer(base_seed %% .Machine$integer.max),
      effective_seed = effective_seed,
      canonical_test_order_id = as.integer(canonical_test_order_id),
      replicates = replicates,
      include_observed = permutation_params$include_observed %||% TRUE
    ))
  } else {
    ""
  }
  list(
    ci_randomness_id = if (identical(ci_method, "hsic.perm")) {
      paste("hsic.perm", as.integer(canonical_test_order_id),
            effective_seed, replicates, sep = ":")
    } else {
      ""
    },
    permutation_seed_effective = effective_seed,
    permutation_plan_spec_hash = plan_spec_hash,
    permutation_plan_hash = plan_spec_hash,
    permutation_replicates = if (identical(ci_method, "hsic.perm")) {
      replicates
    } else {
      NA_integer_
    }
  )
}

fastkpc_precision_effective_permutation_params <- function(ci_method,
                                                           permutation_params,
                                                           randomness) {
  if (!identical(ci_method, "hsic.perm")) return(permutation_params)
  out <- permutation_params
  out$seed <- randomness$permutation_seed_effective
  out
}

fastkpc_precision_neighbors <- function(adjacency, vertex, excluded) {
  out <- which(adjacency[, vertex] & seq_len(nrow(adjacency)) != excluded)
  as.integer(out)
}

fastkpc_resolve_ci_decision <- function(p_raw, alpha, na_delete = TRUE) {
  p_raw <- as.numeric(p_raw)[1L]
  p_was_nonfinite <- !is.finite(p_raw)
  if (p_was_nonfinite) {
    if (isTRUE(na_delete)) {
      return(list(
        p_raw = p_raw,
        p_used = 1.0,
        independent = TRUE,
        delete_edge = TRUE,
        p_was_nonfinite = TRUE,
        nonfinite_action = "na-delete-use-1",
        boundary_rule = "p_used >= alpha"
      ))
    }
    return(list(
      p_raw = p_raw,
      p_used = 0.0,
      independent = FALSE,
      delete_edge = FALSE,
      p_was_nonfinite = TRUE,
      nonfinite_action = "na-keep-use-0",
      boundary_rule = "p_used >= alpha"
    ))
  }
  p_used <- p_raw
  delete_edge <- p_used >= as.numeric(alpha)
  list(
    p_raw = p_raw,
    p_used = p_used,
    independent = delete_edge,
    delete_edge = delete_edge,
    p_was_nonfinite = FALSE,
    nonfinite_action = "",
    boundary_rule = "p_used >= alpha"
  )
}

fastkpc_precision_normalize_p <- function(p_raw, na_delete = TRUE,
                                          alpha = NA_real_) {
  alpha <- if (is.finite(alpha)) alpha else Inf
  fastkpc_resolve_ci_decision(p_raw, alpha = alpha, na_delete = na_delete)
}

fastkpc_precision_group_route <- function(precision, alpha, tau, S,
                                          runtime_capabilities,
                                          allow_canary = FALSE,
                                          execution_engine = "cpu") {
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
    allow_canary = allow_canary,
    execution_engine = execution_engine
  )
}

fastkpc_precision_create_execution_context <- function(data, residual_cache,
                                                       runtime_capabilities,
                                                       execution_engine) {
  data <- as.matrix(data)
  data_hash <- fastkpc_hash_object(list(
    n = nrow(data),
    p = ncol(data),
    values = round(as.numeric(data), digits = 14)
  ))
  list(
    residual_cache_enabled = isTRUE(residual_cache),
    data_hash = data_hash,
    runtime_capabilities = runtime_capabilities,
    execution_engine = execution_engine,
    residual_cache = new.env(parent = emptyenv()),
    residual_cache_stats = new.env(parent = emptyenv())
  )
}

fastkpc_precision_init_cache_stats <- function(context) {
  if (is.null(context) || is.null(context$residual_cache_stats)) {
    return(invisible(NULL))
  }
  stats <- context$residual_cache_stats
  stats$requests <- 0L
  stats$hits <- 0L
  stats$misses <- 0L
  stats$computations <- 0L
  stats$stored_vectors <- 0L
  stats$stored_values <- 0L
  stats$setup_cache_hits <- 0L
  stats$spectral_cache_hits <- 0L
  invisible(NULL)
}

fastkpc_precision_residual_cache_key <- function(context, target, S, backend,
                                                 setup_fingerprint,
                                                 sp_grid = NULL) {
  runtime <- context$runtime_capabilities %||% list()
  fastkpc_hash_object(list(
    data_hash = context$data_hash %||% NA_character_,
    target = as.integer(target),
    S = as.integer(S),
    backend = as.character(backend),
    setup_fingerprint = as.character(setup_fingerprint),
    formula_class = fastkpc_regrxons_formula_class(S),
    mgcv_version = runtime$mgcv_version %||% NA_character_,
    setup_schema =
      runtime$setup_fingerprint_schema_version %||% NA_character_,
    spectral_gcv_version = runtime$spectral_gcv_version %||% NA_character_,
    sp_grid = round(as.numeric(sp_grid %||% numeric()), digits = 14)
  ))
}

fastkpc_precision_cache_stats <- function(context, backend_name) {
  enabled <- !is.null(context) && isTRUE(context$residual_cache_enabled)
  stats <- if (is.null(context)) NULL else context$residual_cache_stats
  list(
    enabled = enabled,
    requests = as.integer(stats$requests %||% 0L),
    hits = as.integer(stats$hits %||% 0L),
    misses = as.integer(stats$misses %||% 0L),
    computations = as.integer(stats$computations %||% 0L),
    stored_vectors = as.integer(stats$stored_vectors %||% 0L),
    stored_values = as.integer(stats$stored_values %||% 0L),
    setup_cache_hits = as.integer(stats$setup_cache_hits %||% 0L),
    spectral_cache_hits = as.integer(stats$spectral_cache_hits %||% 0L),
    backend_name = backend_name
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

fastkpc_execute_ci_fast_spline_cuda <- function(data, x, y, S, ci_method,
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
    rx <- fastspline_residual_cuda(data[, x], S_data, fallback = FALSE)$residuals
    ry <- fastspline_residual_cuda(data[, y], S_data, fallback = FALSE)$residuals
  }
  ci <- fastkpc_precision_ci_from_residuals(
    rx, ry, ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "fastSplineCUDA",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = route$setup_fingerprint %||%
      paste0("fastSplineCUDA:S:", fastkpc_precision_S_key(S)),
    p_source_used = paste0(role, ":fastSplineCUDA+native-cpu"),
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
  if (length(S) > 2L) {
    stop("compatible CPU precision slice supports |S| <= 2", call. = FALSE)
  }
  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
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

fastkpc_setup_fingerprint_value <- function(value) {
  if (is.null(value)) return(NA_character_)
  if (is.list(value)) {
    if (!is.null(value$fingerprint)) {
      value <- value$fingerprint
    } else {
      return(NA_character_)
    }
  }
  if (length(value) == 0L) return(NA_character_)
  as.character(value[1L])
}

fastkpc_sum_timing <- function(values) {
  values <- unlist(values, recursive = TRUE, use.names = FALSE)
  if (length(values) == 0L) return(NA_real_)
  values <- suppressWarnings(as.numeric(values))
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(NA_real_)
  sum(values, na.rm = TRUE)
}

fastkpc_mgcv_extract_gpu_gcv_for_target <- function(data, target, S,
                                                    sp_grid = NULL) {
  start <- proc.time()[["elapsed"]]
  if (length(S) == 0L) {
    elapsed <- (proc.time()[["elapsed"]] - start) * 1000
    setup_fingerprint <- paste0("direct:", target)
    return(list(
      residuals = as.numeric(data[, target]),
      fitted = rep(0, nrow(data)),
      setup_fingerprint = setup_fingerprint,
      setup_fingerprint_full = list(fingerprint = setup_fingerprint),
      sp = NA_real_,
      score = NA_real_,
      edf = NA_real_,
      grid = data.frame(),
      fit = list(
        used_device = "none",
        native_gpu_solve_used = FALSE,
        setup_fingerprint = list(fingerprint = setup_fingerprint)
      ),
      timings = list(residualization_total_ms = elapsed)
    ))
  }
  if (length(S) > 2L) {
    stop("mgcvExtractGPU precision slice supports |S| <= 2", call. = FALSE)
  }
  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
  local_data <- data.frame(.target = as.numeric(data[, target]), S_data)
  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    formula_class = fastkpc_regrxons_formula_class(S)
  )
  form <- stats::as.formula(paste(".target ~", rhs))
  if (is.null(sp_grid)) {
    sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 17L))
  }
  fit <- fastkpc_mgcv_extract_gpu_gcv(
    formula = form,
    data = local_data,
    setup_sp = 1,
    sp_grid = sp_grid,
    method = "GCV.Cp",
    target = target,
    S = S,
    bs = "tp",
    device = "cuda",
    allow_cpu_fallback = FALSE,
    gcv_strategy = "spectral"
  )
  elapsed <- (proc.time()[["elapsed"]] - start) * 1000
  list(
    residuals = fit$residuals,
    fitted = fit$fitted,
    setup_fingerprint = fastkpc_setup_fingerprint_value(fit$setup_fingerprint),
    setup_fingerprint_full = fit$setup_fingerprint,
    sp = fit$sp,
    score = fit$score,
    edf = fit$edf,
    grid = fit$grid,
    fit = fit,
    timings = list(residualization_total_ms = elapsed)
  )
}

fastkpc_mgcv_extract_gpu_gcv_for_pair <- function(data, x, y, S,
                                                  sp_grid = NULL) {
  total_start <- proc.time()[["elapsed"]]
  if (length(S) == 0L) {
    setup_fingerprint <- paste0("direct:", x, "-", y)
    residuals <- cbind(as.numeric(data[, x]), as.numeric(data[, y]))
    fitted <- matrix(0, nrow(data), 2L)
    colnames(residuals) <- c("x", "y")
    colnames(fitted) <- c("x", "y")
    elapsed <- (proc.time()[["elapsed"]] - total_start) * 1000
    return(list(
      residuals = residuals,
      fitted = fitted,
      setup_fingerprint = setup_fingerprint,
      setup_fingerprint_x = setup_fingerprint,
      setup_fingerprint_y = setup_fingerprint,
      shared_setup_fingerprint = setup_fingerprint,
      sp = c(x = NA_real_, y = NA_real_),
      score = c(x = NA_real_, y = NA_real_),
      edf = c(x = NA_real_, y = NA_real_),
      selected_grid_index = c(x = NA_integer_, y = NA_integer_),
      gcv_grid_points = c(x = 0L, y = 0L),
      grid = list(x = data.frame(), y = data.frame()),
      fit = list(
        used_device = "none",
        native_gpu_solve_used = FALSE,
        used_device_x = "none",
        used_device_y = "none",
        native_gpu_solve_used_x = FALSE,
        native_gpu_solve_used_y = FALSE,
        shared_setup_fingerprint = setup_fingerprint,
        same_setup_pair_batch_used = FALSE
      ),
      timings = list(residualization_total_ms = elapsed)
    ))
  }
  if (length(S) > 2L) {
    stop("mgcvExtractGPU precision slice supports |S| <= 2", call. = FALSE)
  }
  native_cuda_available <- exists("fastkpc_cuda_available", mode = "function") &&
    isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
  native_batch_available <- exists(
    "mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda",
    mode = "function"
  )
  if (!native_cuda_available || !native_batch_available) {
    stop("mgcvExtractGPU GCV native CUDA same-setup batch solve is unavailable",
         call. = FALSE)
  }
  if (is.null(sp_grid)) {
    sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 17L))
  }
  sp_grid <- fastkpc_validate_fixed_positive_sp(sp_grid)

  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
  Y <- cbind(x = as.numeric(data[, x]), y = as.numeric(data[, y]))
  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    formula_class = fastkpc_regrxons_formula_class(S)
  )
  form <- stats::as.formula(paste(".target ~", rhs))

  setup_start <- proc.time()[["elapsed"]]
  template_setup <- fastkpc_mgcv_extract_setup(
    formula = form,
    data = data.frame(.target = Y[, 1L], S_data),
    sp = 1,
    method = "GCV.Cp",
    target = x,
    S = S,
    bs = "tp"
  )
  setup_ms <- (proc.time()[["elapsed"]] - setup_start) * 1000
  if (length(template_setup$S) != 1L) {
    stop("mgcvExtractGPU GCV currently supports single-penalty setups only",
         call. = FALSE)
  }

  spectral_start <- proc.time()[["elapsed"]]
  base_handle <- fastkpc_mgcv_extract_gpu_setup_handle(
    setup = template_setup,
    sp = 1,
    device_resident = TRUE
  )
  spectral <- fastkpc_mgcv_extract_gpu_spectral_prepare(base_handle)
  spectral_ms <- (proc.time()[["elapsed"]] - spectral_start) * 1000

  gcv_start <- proc.time()[["elapsed"]]
  target_ids <- c(x = as.integer(x), y = as.integer(y))
  score_one <- function(j) {
    yj <- Y[, j]
    Xty_null <- as.numeric(crossprod(base_handle$X_null, yj))
    scored <- fastkpc_mgcv_extract_gpu_spectral_score_grid(
      spectral = spectral,
      y = yj,
      Xty_null = Xty_null,
      sp_grid = sp_grid
    )
    grid <- scored$grid
    valid <- is.finite(grid$rss) & is.finite(grid$edf) & is.finite(grid$gcv)
    if (!any(valid)) {
      stop("No valid GCV candidate in sp_grid", call. = FALSE)
    }
    idx <- which(grid$gcv == min(grid$gcv[valid]))[1L]
    list(
      grid = grid,
      selected_grid_index = idx,
      sp = as.numeric(sp_grid[idx]),
      score = as.numeric(grid$gcv[idx]),
      edf = as.numeric(grid$edf[idx]),
      y = yj,
      Xty_null = Xty_null
    )
  }
  scored <- lapply(seq_len(ncol(Y)), score_one)
  gcv_ms <- (proc.time()[["elapsed"]] - gcv_start) * 1000
  selected_sp <- vapply(scored, `[[`, numeric(1), "sp")

  make_handle <- function(j) {
    handle <- base_handle
    handle$y <- scored[[j]]$y
    handle$sp <- selected_sp[j]
    handle$Xty_null <- scored[[j]]$Xty_null
    penalty <- fastkpc_assemble_penalty(
      p = ncol(template_setup$X),
      S = template_setup$S,
      off = template_setup$off,
      sp = selected_sp[j],
      H = template_setup$H
    )
    handle$penalty <- penalty
    handle$penalty_null <- crossprod(handle$Z, penalty %*% handle$Z)
    handle$setup_fingerprint <- template_setup$setup_fingerprint
    handle
  }
  handles <- lapply(seq_along(selected_sp), make_handle)

  solve_start <- proc.time()[["elapsed"]]
  native <- mgcv_extract_gpu_solve_same_setup_batch_fixed_sp_cuda(handles)
  solve_ms <- (proc.time()[["elapsed"]] - solve_start) * 1000

  materialize_start <- proc.time()[["elapsed"]]
  residuals <- as.matrix(native$residuals)
  fitted <- as.matrix(native$fitted)
  colnames(residuals) <- c("x", "y")
  colnames(fitted) <- c("x", "y")
  coefficients <- lapply(seq_along(selected_sp), function(j) {
    native$coefficients[, j]
  })
  theta <- lapply(seq_along(selected_sp), function(j) native$theta[, j])
  materialize_ms <- (proc.time()[["elapsed"]] - materialize_start) * 1000

  shared_setup <- template_setup$setup_fingerprint$fingerprint
  elapsed <- (proc.time()[["elapsed"]] - total_start) * 1000
  list(
    residuals = residuals,
    fitted = fitted,
    setup_fingerprint = shared_setup,
    setup_fingerprint_x = shared_setup,
    setup_fingerprint_y = shared_setup,
    shared_setup_fingerprint = shared_setup,
    sp = c(x = selected_sp[1L], y = selected_sp[2L]),
    score = c(x = scored[[1L]]$score, y = scored[[2L]]$score),
    edf = c(x = scored[[1L]]$edf, y = scored[[2L]]$edf),
    selected_grid_index = c(
      x = as.integer(scored[[1L]]$selected_grid_index),
      y = as.integer(scored[[2L]]$selected_grid_index)
    ),
    gcv_grid_points = c(x = nrow(scored[[1L]]$grid),
                        y = nrow(scored[[2L]]$grid)),
    grid = list(x = scored[[1L]]$grid, y = scored[[2L]]$grid),
    coefficients = coefficients,
    theta = theta,
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = "cuda",
      used_device_y = "cuda",
      native_gpu_solve_used_x = TRUE,
      native_gpu_solve_used_y = TRUE,
      setup_fingerprint = template_setup$setup_fingerprint,
      setup_fingerprint_x = shared_setup,
      setup_fingerprint_y = shared_setup,
      shared_setup_fingerprint = shared_setup,
      sp_selection_backend_executed_x = "r-cpu-spectral",
      sp_selection_backend_executed_y = "r-cpu-spectral",
      gcv_score_backend_executed_x = "r-cpu-spectral",
      gcv_score_backend_executed_y = "r-cpu-spectral",
      selected_solve_backend_executed_x = "cuda",
      selected_solve_backend_executed_y = "cuda",
      same_setup_pair_batch_used = TRUE,
      true_batched_kernel = FALSE,
      batch_stage = "native-same-setup-repeated-cuda-solve"
    ),
    timings = list(
      mgcv_setup_cpu_ms = setup_ms,
      host_to_device_ms = NA_real_,
      spectral_prepare_ms = spectral_ms,
      gcv_score_ms = gcv_ms,
      linear_solve_ms = solve_ms,
      residual_materialize_ms = materialize_ms,
      device_to_host_ms = NA_real_,
      residualization_total_ms = elapsed
    )
  )
}

fastkpc_entries_from_gpu_pair <- function(pair, x, y) {
  targets <- c(as.integer(x), as.integer(y))
  lapply(seq_along(targets), function(j) {
    list(
      target = targets[[j]],
      residuals = as.numeric(pair$residuals[, j]),
      fitted = as.numeric(pair$fitted[, j]),
      setup_fingerprint = pair$shared_setup_fingerprint %||%
        pair$setup_fingerprint,
      sp = as.numeric(pair$sp[j]),
      score = as.numeric(pair$score[j]),
      edf = as.numeric(pair$edf[j]),
      selected_grid_index = as.integer(pair$selected_grid_index[j]),
      gcv_grid_points = as.integer(pair$gcv_grid_points[j]),
      grid = pair$grid[[j]],
      used_device = pair$fit[[paste0("used_device_", if (j == 1L) "x" else "y")]] %||%
        pair$fit$used_device,
      native_gpu_solve_used =
        isTRUE(pair$fit[[paste0("native_gpu_solve_used_",
                                if (j == 1L) "x" else "y")]]) ||
        isTRUE(pair$fit$native_gpu_solve_used),
      sp_selection_backend_executed =
        pair$fit[[paste0("sp_selection_backend_executed_",
                         if (j == 1L) "x" else "y")]] %||% NA_character_,
      gcv_score_backend_executed =
        pair$fit[[paste0("gcv_score_backend_executed_",
                         if (j == 1L) "x" else "y")]] %||% NA_character_,
      selected_solve_backend_executed =
        pair$fit[[paste0("selected_solve_backend_executed_",
                         if (j == 1L) "x" else "y")]] %||% NA_character_,
      timings = pair$timings
    )
  })
}

fastkpc_pair_from_cached_gpu_residuals <- function(x_entry, y_entry, x, y) {
  shared_setup <- x_entry$setup_fingerprint
  if (is.na(shared_setup) ||
      !identical(as.character(shared_setup),
                 as.character(y_entry$setup_fingerprint))) {
    stop("cached mgcvExtractGPU residual setup fingerprint mismatch",
         call. = FALSE)
  }
  residuals <- cbind(x = x_entry$residuals, y = y_entry$residuals)
  fitted <- cbind(x = x_entry$fitted, y = y_entry$fitted)
  list(
    residuals = residuals,
    fitted = fitted,
    setup_fingerprint = shared_setup,
    setup_fingerprint_x = shared_setup,
    setup_fingerprint_y = shared_setup,
    shared_setup_fingerprint = shared_setup,
    sp = c(x = x_entry$sp, y = y_entry$sp),
    score = c(x = x_entry$score, y = y_entry$score),
    edf = c(x = x_entry$edf, y = y_entry$edf),
    selected_grid_index = c(
      x = as.integer(x_entry$selected_grid_index),
      y = as.integer(y_entry$selected_grid_index)
    ),
    gcv_grid_points = c(
      x = as.integer(x_entry$gcv_grid_points),
      y = as.integer(y_entry$gcv_grid_points)
    ),
    grid = list(x = x_entry$grid, y = y_entry$grid),
    fit = list(
      used_device = "cuda",
      native_gpu_solve_used = TRUE,
      used_device_x = x_entry$used_device,
      used_device_y = y_entry$used_device,
      native_gpu_solve_used_x = isTRUE(x_entry$native_gpu_solve_used),
      native_gpu_solve_used_y = isTRUE(y_entry$native_gpu_solve_used),
      setup_fingerprint_x = shared_setup,
      setup_fingerprint_y = shared_setup,
      shared_setup_fingerprint = shared_setup,
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
      true_batched_kernel = FALSE,
      batch_stage = "run-scoped-residual-cache-hit"
    ),
    timings = list(
      mgcv_setup_cpu_ms = NA_real_,
      host_to_device_ms = NA_real_,
      spectral_prepare_ms = NA_real_,
      gcv_score_ms = NA_real_,
      linear_solve_ms = NA_real_,
      residual_materialize_ms = NA_real_,
      device_to_host_ms = NA_real_,
      residualization_total_ms = 0
    )
  )
}

fastkpc_execute_ci_mgcv_extract_gpu <- function(data, x, y, S, ci_method,
                                                index, legacy_index,
                                                hsic_params,
                                                permutation_params, route,
                                                role = "primary") {
  total_start <- proc.time()[["elapsed"]]
  context <- route$execution_context %||% NULL
  sp_grid <- NULL
  if (is.null(sp_grid)) {
    sp_grid <- exp(seq(log(1e-4), log(1e4), length.out = 17L))
  }
  setup_key <- paste0("S:", fastkpc_precision_S_key(S))
  cache_enabled <- !is.null(context) && isTRUE(context$residual_cache_enabled)
  cached_entries <- list()
  cache_hits <- 0L
  cache_keys <- character()
  if (isTRUE(cache_enabled)) {
    for (target in c(x, y)) {
      key <- fastkpc_precision_residual_cache_key(
        context = context,
        target = target,
        S = S,
        backend = "mgcvExtractGPU",
        setup_fingerprint = setup_key,
        sp_grid = sp_grid
      )
      cache_keys <- c(cache_keys, key)
      context$residual_cache_stats$requests <-
        as.integer(context$residual_cache_stats$requests %||% 0L) + 1L
      if (exists(key, envir = context$residual_cache, inherits = FALSE)) {
        cache_hits <- cache_hits + 1L
        context$residual_cache_stats$hits <-
          as.integer(context$residual_cache_stats$hits %||% 0L) + 1L
        cached_entries[[as.character(target)]] <-
          get(key, envir = context$residual_cache, inherits = FALSE)
      } else {
        context$residual_cache_stats$misses <-
          as.integer(context$residual_cache_stats$misses %||% 0L) + 1L
      }
    }
  }

  served_from_cache <- isTRUE(cache_enabled) && length(cached_entries) == 2L
  if (isTRUE(served_from_cache)) {
    pair <- fastkpc_pair_from_cached_gpu_residuals(
      cached_entries[[as.character(x)]],
      cached_entries[[as.character(y)]],
      x = x,
      y = y
    )
  } else {
    pair <- fastkpc_mgcv_extract_gpu_gcv_for_pair(data, x, y, S,
                                                  sp_grid = sp_grid)
    if (isTRUE(cache_enabled)) {
      context$residual_cache_stats$computations <-
        as.integer(context$residual_cache_stats$computations %||% 0L) + 1L
      entries <- fastkpc_entries_from_gpu_pair(pair, x = x, y = y)
      for (j in seq_along(entries)) {
        key <- cache_keys[[j]]
        if (!exists(key, envir = context$residual_cache, inherits = FALSE)) {
          assign(key, entries[[j]], envir = context$residual_cache)
          context$residual_cache_stats$stored_vectors <-
            as.integer(context$residual_cache_stats$stored_vectors %||% 0L) + 1L
          context$residual_cache_stats$stored_values <-
            as.integer(context$residual_cache_stats$stored_values %||% 0L) +
              length(entries[[j]]$residuals)
        }
      }
    }
  }

  if (!identical(pair$fit$used_device_x, "cuda") ||
      !identical(pair$fit$used_device_y, "cuda") ||
      !isTRUE(pair$fit$native_gpu_solve_used_x) ||
      !isTRUE(pair$fit$native_gpu_solve_used_y)) {
    stop("mgcvExtractGPU executor did not receive native CUDA target fits",
         call. = FALSE)
  }

  setup_x <- fastkpc_setup_fingerprint_value(pair$setup_fingerprint_x)
  setup_y <- fastkpc_setup_fingerprint_value(pair$setup_fingerprint_y)
  if (is.na(setup_x) || is.na(setup_y) || !identical(setup_x, setup_y)) {
    stop("mgcvExtractGPU x/y setup fingerprint mismatch", call. = FALSE)
  }
  shared_setup <- setup_x

  ci_start <- proc.time()[["elapsed"]]
  ci <- fastkpc_precision_ci_from_residuals(
    pair$residuals[, 1L], pair$residuals[, 2L],
    ci_method = ci_method, index = index,
    legacy_index = legacy_index, hsic_params = hsic_params,
    permutation_params = permutation_params
  )
  ci_elapsed <- (proc.time()[["elapsed"]] - ci_start) * 1000
  total_elapsed <- (proc.time()[["elapsed"]] - total_start) * 1000
  list(
    p.value = ci$p.value,
    residual_backend_executed = "mgcvExtractGPU",
    ci_backend_executed = "native-cpu",
    setup_fingerprint = shared_setup,
    setup_fingerprint_x = setup_x,
    setup_fingerprint_y = setup_y,
    shared_setup_fingerprint = shared_setup,
    p_source_used = paste0(role, ":mgcvExtractGPU+native-cpu"),
    sp = as.numeric(pair$sp),
    score = as.numeric(pair$score),
    edf = as.numeric(pair$edf),
    selected_grid_index = as.integer(pair$selected_grid_index),
    gcv_grid_points = as.integer(pair$gcv_grid_points),
    used_device = pair$fit$used_device,
    used_device_x = pair$fit$used_device_x,
    used_device_y = pair$fit$used_device_y,
    native_gpu_solve_used_x = isTRUE(pair$fit$native_gpu_solve_used_x),
    native_gpu_solve_used_y = isTRUE(pair$fit$native_gpu_solve_used_y),
    sp_selection_backend_executed_x =
      pair$fit$sp_selection_backend_executed_x %||% NA_character_,
    sp_selection_backend_executed_y =
      pair$fit$sp_selection_backend_executed_y %||% NA_character_,
    gcv_score_backend_executed_x =
      pair$fit$gcv_score_backend_executed_x %||% NA_character_,
    gcv_score_backend_executed_y =
      pair$fit$gcv_score_backend_executed_y %||% NA_character_,
    selected_solve_backend_executed_x =
      pair$fit$selected_solve_backend_executed_x %||% NA_character_,
    selected_solve_backend_executed_y =
      pair$fit$selected_solve_backend_executed_y %||% NA_character_,
    same_setup_pair_batch_used = isTRUE(pair$fit$same_setup_pair_batch_used),
    cache_hit = isTRUE(served_from_cache),
    cache_hit_x = isTRUE(served_from_cache),
    cache_hit_y = isTRUE(served_from_cache),
    timings = list(
      mgcv_setup_cpu_ms = pair$timings$mgcv_setup_cpu_ms %||% NA_real_,
      host_to_device_ms = pair$timings$host_to_device_ms %||% NA_real_,
      spectral_prepare_ms = pair$timings$spectral_prepare_ms %||% NA_real_,
      gcv_score_ms = pair$timings$gcv_score_ms %||% NA_real_,
      linear_solve_ms = pair$timings$linear_solve_ms %||% NA_real_,
      residual_materialize_ms =
        pair$timings$residual_materialize_ms %||% NA_real_,
      device_to_host_ms = pair$timings$device_to_host_ms %||% NA_real_,
      residualization_total_ms =
        pair$timings$residualization_total_ms %||% NA_real_,
      ci_test_ms = ci_elapsed,
      total_ms = total_elapsed
    )
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
  if (length(S) > 2L) {
    stop("legacy mgcv CPU precision slice supports |S| <= 2", call. = FALSE)
  }
  fastkpc_require_mgcv()
  S_data <- as.data.frame(data[, S, drop = FALSE])
  colnames(S_data) <- paste0("s", seq_along(S))
  local_data <- data.frame(.target = as.numeric(data[, target]), S_data)
  rhs <- fastkpc_mgcv_regrxons_rhs(
    S_data = S_data,
    S = S,
    formula_class = fastkpc_regrxons_formula_class(S)
  )
  form <- stats::as.formula(paste(".target ~", rhs),
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

fastkpc_precision_fallback_backends <- function(primary_backend, route) {
  primary_backend <- fastkpc_nonempty_backend(primary_backend, "")
  fallback_backend <- fastkpc_nonempty_backend(route$fallback_backend,
                                               "legacy-mgcv")
  out <- character()
  if (identical(primary_backend, "mgcvExtractGPUGCV")) {
    out <- c(out, "mgcvExtractCPUGCVBridge", fallback_backend)
  } else if (identical(primary_backend, "mgcvExtractCPUGCVBridge")) {
    out <- c(out, fallback_backend)
  } else if (!identical(primary_backend, fallback_backend)) {
    out <- c(out, fallback_backend)
  }
  out <- out[nzchar(out) & !is.na(out)]
  unique(out[out != primary_backend])
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
  fallback_chain <- as.character(fallback_backend)
  fallback_chain <- fallback_chain[!is.na(fallback_chain) & nzchar(fallback_chain)]
  fallback_chain <- unique(fallback_chain[fallback_chain != primary_backend])
  if (length(fallback_chain) == 0L) {
    stop(primary$error_message, call. = FALSE)
  }
  errors <- paste0("backend ", primary_backend, " failed: ",
                   primary$error_message)
  for (backend in fallback_chain) {
    fallback_route <- route
    fallback_route$primary_backend <- backend
    fallback <- fastkpc_execute_backend_attempt(
      data = data, x = x, y = y, S = S, route = fallback_route, role = role,
      backend = backend, ci_method = ci_method, index = index,
      legacy_index = legacy_index, hsic_params = hsic_params,
      permutation_params = permutation_params,
      precision_executors = precision_executors
    )
    fallback$fallback_triggered <- TRUE
    fallback$fallback_reason <- errors
    attempts[[length(attempts) + 1L]] <- fallback
    if (identical(fallback$attempt_status, "ok")) {
      fallback$attempts <- attempts
      fallback$attempt_count <- length(attempts)
      return(fallback)
    }
    errors <- paste0(errors, "; fallback ", backend, " failed: ",
                     fallback$error_message)
  }
  stop(errors, call. = FALSE)
}

fastkpc_precision_primary_backend <- function(route, precision,
                                              execution_engine = "cpu") {
  fallback <- if (identical(execution_engine, "cuda")) {
    "fastSplineCUDA"
  } else {
    "fastSplineCPU"
  }
  fastkpc_nonempty_backend(route$primary_backend, fallback)
}

fastkpc_precision_resolve_test <- function(data, x, y, S, route, precision,
                                           alpha, tau, ci_method, index,
                                           legacy_index, hsic_params,
                                           permutation_params,
                                           precision_executors,
                                           na_delete = TRUE,
                                           canonical_test_order_id = NA_integer_,
                                           execution_engine = "cpu") {
  randomness <- fastkpc_precision_ci_randomness(
    ci_method = ci_method,
    permutation_params = permutation_params,
    canonical_test_order_id = canonical_test_order_id
  )
  effective_permutation_params <- fastkpc_precision_effective_permutation_params(
    ci_method = ci_method,
    permutation_params = permutation_params,
    randomness = randomness
  )
  primary_route <- route
  primary_route$primary_backend <- fastkpc_precision_primary_backend(
    route, precision, execution_engine = execution_engine
  )
  primary_fallback <- character()
  if (identical(precision, "compatible")) {
    primary_fallback <- fastkpc_precision_fallback_backends(
      primary_route$primary_backend, route
    )
  }
  primary_exec <- fastkpc_execute_route_with_fallback(
    data = data, x = x, y = y, S = S, route = primary_route, role = "primary",
    ci_method = ci_method, index = index, legacy_index = legacy_index,
    hsic_params = hsic_params,
    permutation_params = effective_permutation_params,
    precision_executors = precision_executors,
    fallback_backend = primary_fallback
  )
  primary_receipt <- primary_exec$receipt
  attempts <- primary_exec$attempts %||% list(primary_exec)
  primary_info <- fastkpc_resolve_ci_decision(primary_receipt$p.value,
                                              alpha = alpha,
                                              na_delete = na_delete)

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

  if (isTRUE(near_alpha)) {
    verifier_backend <- fastkpc_nonempty_backend(route$verifier_backend,
                                                 "mgcvExtractGPUGCV")
    verifier_route <- route
    verifier_route$primary_backend <- verifier_backend
    verifier_exec <- fastkpc_execute_route_with_fallback(
      data = data, x = x, y = y, S = S, route = verifier_route,
      role = "verifier", ci_method = ci_method, index = index,
      legacy_index = legacy_index, hsic_params = hsic_params,
      permutation_params = effective_permutation_params,
      precision_executors = precision_executors,
      fallback_backend = fastkpc_precision_fallback_backends(
        verifier_backend, route
      )
    )
    attempts <- c(attempts, verifier_exec$attempts %||% list(verifier_exec))
    verifier_fallback_backend <- fastkpc_nonempty_backend(route$fallback_backend,
                                                          "legacy-mgcv")
    verifier_info <- fastkpc_resolve_ci_decision(
      verifier_exec$receipt$p.value, alpha = alpha, na_delete = na_delete
    )
    if (isTRUE(verifier_info$p_was_nonfinite) &&
        !identical(verifier_exec$receipt$residual_backend_executed,
                   verifier_fallback_backend)) {
      attempted <- vapply(attempts, function(attempt) {
        as.character((attempt$backend_attempted %||% "")[1L])
      }, character(1L))
      nonfinite_chain <- fastkpc_precision_fallback_backends(
        verifier_backend, route
      )
      nonfinite_chain <- setdiff(nonfinite_chain, attempted)
      if (length(nonfinite_chain) > 0L) {
        nonfinite_route <- route
        nonfinite_route$primary_backend <- nonfinite_chain[[1L]]
        legacy_exec <- fastkpc_execute_route_with_fallback(
          data = data, x = x, y = y, S = S, route = nonfinite_route,
          role = "verifier", ci_method = ci_method, index = index,
          legacy_index = legacy_index, hsic_params = hsic_params,
          permutation_params = effective_permutation_params,
          precision_executors = precision_executors,
          fallback_backend = nonfinite_chain[-1L]
        )
        attempts <- c(attempts, legacy_exec$attempts %||% list(legacy_exec))
        verifier_exec <- legacy_exec
        verifier_info <- fastkpc_resolve_ci_decision(
          verifier_exec$receipt$p.value, alpha = alpha, na_delete = na_delete
        )
        verifier_exec$fallback_triggered <- TRUE
        verifier_exec$fallback_reason <- "verifier returned non-finite p-value"
      }
    }
    chosen_receipt <- verifier_exec$receipt
    chosen_info <- verifier_info
    p_source <- verifier_exec$receipt$p_source_used
    fallback_triggered <- isTRUE(verifier_exec$fallback_triggered)
    fallback_reason <- verifier_exec$fallback_reason
  }

  attempt_backend_sequence <- paste(vapply(attempts, function(attempt) {
    fastkpc_nonempty_backend(
      attempt$backend_executed,
      fastkpc_nonempty_backend(attempt$backend_attempted, "")
    )
  }, character(1L)), collapse = ">")
  attempt_status_sequence <- paste(vapply(attempts, function(attempt) {
    as.character((attempt$attempt_status %||% "")[1L])
  }, character(1L)), collapse = ">")

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
    attempts = attempts,
    attempt_count = length(attempts),
    attempt_backend_sequence = attempt_backend_sequence,
    attempt_status_sequence = attempt_status_sequence,
    ci_randomness = randomness,
    decision_before_verify = primary_info$delete_edge,
    decision_after_verify = chosen_info$delete_edge
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
  fallback_reason <- resolved$fallback_reason %||% ""
  if (!nzchar(fallback_reason)) {
    fallback_reason <- route$fallback_reason %||% ""
  }
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
    fallback_reason = fallback_reason,
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
    cache_hit = isTRUE(receipt$cache_hit),
    fallback_triggered = resolved$fallback_triggered,
    attempt_count = resolved$attempt_count,
    attempt_backend_sequence = resolved$attempt_backend_sequence,
    attempt_status_sequence = resolved$attempt_status_sequence,
    ci_randomness_id = resolved$ci_randomness$ci_randomness_id,
    permutation_seed_effective =
      resolved$ci_randomness$permutation_seed_effective,
    permutation_plan_spec_hash =
      resolved$ci_randomness$permutation_plan_spec_hash,
    permutation_plan_hash = resolved$ci_randomness$permutation_plan_hash,
    permutation_replicates = resolved$ci_randomness$permutation_replicates,
    precision_execution_status = "data-plane-executed",
    decision_before_verify = resolved$decision_before_verify,
    decision_after_verify = resolved$decision_after_verify,
    mgcv_setup_cpu_ms = receipt$timings$mgcv_setup_cpu_ms %||% NA_real_,
    host_to_device_ms = receipt$timings$host_to_device_ms %||% NA_real_,
    spectral_prepare_ms = receipt$timings$spectral_prepare_ms %||% NA_real_,
    gcv_score_ms = receipt$timings$gcv_score_ms %||% NA_real_,
    linear_solve_ms = receipt$timings$linear_solve_ms %||% NA_real_,
    device_to_host_ms = receipt$timings$device_to_host_ms %||% NA_real_,
    ci_test_ms = receipt$timings$ci_test_ms %||% NA_real_,
    total_ms = receipt$timings$total_ms %||% NA_real_
  )
}

fastkpc_r_skeleton_precision <- function(data, alpha, max_conditioning_size,
                                         precision, tau, ci_method, index,
                                         legacy_index, hsic_params,
                                         permutation_params,
                                         precision_executors,
                                         runtime_capabilities,
                                         allow_canary = FALSE,
                                         residual_cache = TRUE,
                                         na_delete = TRUE,
                                         execution_engine = c("cpu", "cuda")) {
  execution_engine <- match.arg(execution_engine)
  data <- as.matrix(data)
  storage.mode(data) <- "double"
  execution_context <- fastkpc_precision_create_execution_context(
    data = data,
    residual_cache = residual_cache,
    runtime_capabilities = runtime_capabilities,
    execution_engine = execution_engine
  )
  fastkpc_precision_init_cache_stats(execution_context)
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
            allow_canary = allow_canary,
            execution_engine = execution_engine
          )
          route$execution_context <- execution_context
          resolved <- fastkpc_precision_resolve_test(
            data = data, x = x, y = y, S = S, route = route,
            precision = precision, alpha = alpha, tau = tau,
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors,
            na_delete = na_delete,
            canonical_test_order_id = test_id,
            execution_engine = execution_engine
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
          deleted <- resolved$p_info$delete_edge
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
            allow_canary = allow_canary,
            execution_engine = execution_engine
          )
          route$execution_context <- execution_context
          resolved <- fastkpc_precision_resolve_test(
            data = data, x = y, y = x, S = S, route = route,
            precision = precision, alpha = alpha, tau = tau,
            ci_method = ci_method, index = index, legacy_index = legacy_index,
            hsic_params = hsic_params,
            permutation_params = permutation_params,
            precision_executors = precision_executors,
            na_delete = na_delete,
            canonical_test_order_id = test_id,
            execution_engine = execution_engine
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
          deleted <- resolved$p_info$delete_edge
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
  cache <- fastkpc_precision_cache_stats(execution_context, backend)
  list(
    adjacency = adjacency,
    sepsets = sepsets,
    pMax = pmax,
    n.edgetests = as.integer(n_edge_tests),
    per.level.log = level_logs,
    backend = execution_engine,
    residual_backend = backend,
    verifier_backend = verifier_backend,
    residual_backend_params = "",
    residual_cache = cache,
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
        unique_residual_requests = cache$stored_vectors,
        residual_cache_requests = cache$requests,
        residual_cache_hits = cache$hits,
        residual_cache_misses = cache$misses,
        residual_cache_computations = cache$computations,
        residual_cache_stored_vectors = cache$stored_vectors,
        residual_cache_stored_values = cache$stored_values,
        dcov_batches = 0L,
        residual_batches = 0L
      )
    ),
    precision_trace = trace,
    precision_receipt = last_receipt
  )
}
