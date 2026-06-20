source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")
source("fastkpc/R/validation_campaign.R")

fastkpc_precision_e2e_default_scenarios <- function() {
  list(
    list(
      scenario_id = "chain-nonlinear-p5",
      n = 120L,
      seed = 6202L,
      generator = function(n, seed) {
        set.seed(seed)
        z1 <- stats::rnorm(n)
        z2 <- sin(z1) + stats::rnorm(n, sd = 0.2)
        z3 <- z2^2 + stats::rnorm(n, sd = 0.2)
        z4 <- 0.5 * z1 + cos(z3) + stats::rnorm(n, sd = 0.25)
        z5 <- stats::rnorm(n)
        cbind(z1 = z1, z2 = z2, z3 = z3, z4 = z4, z5 = z5)
      }
    ),
    list(
      scenario_id = "fork-additive-p6",
      n = 160L,
      seed = 6203L,
      generator = function(n, seed) {
        set.seed(seed)
        root <- stats::rnorm(n)
        z2 <- sin(root) + stats::rnorm(n, sd = 0.2)
        z3 <- cos(root) + stats::rnorm(n, sd = 0.2)
        z4 <- z2 + z3 + stats::rnorm(n, sd = 0.25)
        z5 <- z2^2 + stats::rnorm(n, sd = 0.25)
        z6 <- stats::rnorm(n)
        cbind(root = root, z2 = z2, z3 = z3, z4 = z4, z5 = z5, z6 = z6)
      }
    ),
    list(
      scenario_id = "scale-nonlinear-p8",
      n = 300L,
      seed = 6204L,
      generator = function(n, seed) {
        set.seed(seed)
        z1 <- stats::rnorm(n)
        z2 <- sin(z1) + stats::rnorm(n, sd = 0.2)
        z3 <- cos(z1) + stats::rnorm(n, sd = 0.2)
        z4 <- z2 * z3 + stats::rnorm(n, sd = 0.25)
        z5 <- z2^2 + stats::rnorm(n, sd = 0.25)
        z6 <- z4 + sin(z5) + stats::rnorm(n, sd = 0.3)
        z7 <- stats::rnorm(n)
        z8 <- 0.4 * z6 + stats::rnorm(n, sd = 0.35)
        cbind(z1 = z1, z2 = z2, z3 = z3, z4 = z4, z5 = z5,
              z6 = z6, z7 = z7, z8 = z8)
      }
    )
  )
}

fastkpc_precision_e2e_real_data_scenario <- function(path) {
  if (is.null(path) || !nzchar(path)) return(NULL)
  if (!file.exists(path)) {
    stop("FASTKPC precision E2E real data path does not exist: ", path,
         call. = FALSE)
  }
  ext <- tolower(tools::file_ext(path))
  raw <- if (identical(ext, "rds")) {
    readRDS(path)
  } else {
    utils::read.csv(path, check.names = FALSE)
  }
  data <- as.data.frame(raw)
  numeric <- vapply(data, is.numeric, logical(1L))
  data <- data[, numeric, drop = FALSE]
  if (ncol(data) < 3L) {
    stop("real benchmark data must contain at least three numeric columns",
         call. = FALSE)
  }
  matrix_data <- as.matrix(data)
  keep <- stats::complete.cases(matrix_data) &
    apply(matrix_data, 1L, function(row) all(is.finite(row)))
  matrix_data <- matrix_data[keep, , drop = FALSE]
  if (nrow(matrix_data) < 20L) {
    stop("real benchmark data has fewer than 20 finite complete rows",
         call. = FALSE)
  }
  list(
    scenario_id = paste0("real-", tools::file_path_sans_ext(basename(path))),
    n = nrow(matrix_data),
    seed = 0L,
    generator = function(n, seed) {
      matrix_data[seq_len(min(n, nrow(matrix_data))), , drop = FALSE]
    }
  )
}

fastkpc_precision_e2e_append_real_scenario <- function(scenarios,
                                                       real_data_path) {
  real <- fastkpc_precision_e2e_real_data_scenario(real_data_path)
  if (is.null(real)) scenarios else c(scenarios, list(real))
}

fastkpc_precision_e2e_mode_config <- function(mode) {
  mode <- match.arg(mode, c("legacy_mgcv", "fast_cuda",
                           "primary_only_cuda", "compatible_cuda",
                           "hybrid_cuda", "fast_cpu", "compatible_cpu",
                           "hybrid_cpu"))
  switch(
    mode,
    legacy_mgcv = list(engine = "cpu", precision = "legacy-mgcv",
                       label = "legacy kpcalg + mgcv"),
    fast_cuda = list(engine = "cuda", precision = "fast",
                     label = "fastSplineCUDA legacy scheduler"),
    primary_only_cuda = list(engine = "cuda", precision = "fast",
                             label = "fastSplineCUDA precision scheduler primary-only"),
    compatible_cuda = list(engine = "cuda", precision = "compatible",
                           label = "mgcvExtractGPU compatible"),
    hybrid_cuda = list(engine = "cuda", precision = "hybrid",
                       label = "fastSplineCUDA + mgcvExtractGPU hybrid"),
    fast_cpu = list(engine = "cpu", precision = "fast",
                    label = "fastSplineCPU precision scheduler"),
    compatible_cpu = list(engine = "cpu", precision = "compatible",
                          label = "mgcvExtractCPU compatible"),
    hybrid_cpu = list(engine = "cpu", precision = "hybrid",
                      label = "fastSplineCPU + mgcvExtractCPU hybrid")
  )
}

fastkpc_precision_e2e_cuda_available <- function() {
  exists("fastkpc_cuda_available", mode = "function") &&
    isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
}

fastkpc_precision_e2e_skip_reason <- function(mode, run_native_cuda) {
  cfg <- fastkpc_precision_e2e_mode_config(mode)
  if (identical(cfg$engine, "cuda") && !isTRUE(run_native_cuda)) {
    return("native CUDA benchmark disabled; set FASTKPC_RUN_CUDA_TESTS=1")
  }
  if (identical(cfg$engine, "cuda") &&
      !fastkpc_precision_e2e_cuda_available()) {
    return("CUDA runtime unavailable")
  }
  if (identical(mode, "legacy_mgcv")) {
    missing <- c("pcalg", "mgcv", "graph")[
      !vapply(c("pcalg", "mgcv", "graph"), requireNamespace,
              logical(1), quietly = TRUE)
    ]
    if (length(missing) > 0L) {
      return(paste("missing legacy dependency:", paste(missing, collapse = ", ")))
    }
  }
  ""
}

fastkpc_precision_e2e_run_id <- function(scenario_id, mode, repeat_id) {
  paste(scenario_id, mode, paste0("r", repeat_id), sep = "-")
}

fastkpc_precision_e2e_safe_run <- function(expr) {
  tryCatch(
    list(status = "ok", value = force(expr), error_message = ""),
    error = function(e) {
      list(status = "error", value = NULL, error_message = conditionMessage(e))
    }
  )
}

fastkpc_precision_e2e_timed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed = max(elapsed, 0))
}

fastkpc_precision_e2e_legacy_to_result <- function(legacy, data, config,
                                                   elapsed_sec) {
  graph <- methods::as(legacy@graph, "matrix")
  adjacency <- matrix(as.integer(graph != 0), nrow = nrow(graph))
  colnames(adjacency) <- rownames(adjacency) <- colnames(data)
  sepset <- legacy@sepset
  list(
    config = config,
    data_info = list(
      n = nrow(data),
      p = ncol(data),
      labels = colnames(data),
      data_hash = paste(nrow(data), ncol(data), signif(sum(data), 8), sep = ":")
    ),
    skeleton = list(
      adjacency = adjacency,
      pMax = as.matrix(legacy@pMax),
      sepsets = sepset,
      scheduler_diagnostics = list(summary = list(
        tests_replayed = as.integer(sum(legacy@n.edgetests, na.rm = TRUE)),
        n_edgetests = as.integer(sum(legacy@n.edgetests, na.rm = TRUE))
      )),
      residual_cache = fastkpc_zero_cache(),
      ci_backend = "legacy-kpcalg",
      residual_backend = "legacy-mgcv"
    ),
    orientation = NULL,
    timings = data.frame(
      stage = c("skeleton", "total"),
      elapsed_sec = c(elapsed_sec, elapsed_sec),
      stringsAsFactors = FALSE
    ),
    diagnostics = list(precision_trace = fastkpc_empty_df(
      c("run_id", "scenario_id", "mgcv_setup_cpu_ms", "spectral_prepare_ms",
        "gcv_score_ms", "linear_solve_ms", "ci_test_ms", "total_ms")
    ))
  )
}

fastkpc_precision_e2e_run_mode <- function(data, scenario_id, mode, repeat_id,
                                           alpha, max_conditioning_size,
                                           ci_method, tau,
                                           hsic_params,
                                           permutation_params,
                                           run_native_cuda,
                                           seed) {
  cfg <- fastkpc_precision_e2e_mode_config(mode)
  run_id <- fastkpc_precision_e2e_run_id(scenario_id, mode, repeat_id)
  skip_reason <- fastkpc_precision_e2e_skip_reason(mode, run_native_cuda)
  if (nzchar(skip_reason)) {
    return(list(
      run_id = run_id,
      mode = mode,
      status = "skipped",
      error_message = skip_reason,
      result = NULL,
      wall_time_sec = NA_real_
    ))
  }
  run <- fastkpc_precision_e2e_safe_run({
    if (identical(mode, "legacy_mgcv")) {
      timed <- fastkpc_precision_e2e_timed(
        fastkpc_legacy_skeleton(
          data = data,
          alpha = alpha,
          max_conditioning_size = max_conditioning_size,
          ic.method = ci_method
        )
      )
      config <- list(
        mode = mode,
        engine_used = "cpu",
        precision = "legacy-mgcv",
        precision_requested = "legacy-mgcv",
        backend_used = "legacy-mgcv",
        backend_executed = "legacy-mgcv",
        verifier_executed = NA_character_,
        ci_method = ci_method,
        ci_backend = "legacy-kpcalg",
        graph_stage = "skeleton"
      )
      fastkpc_precision_e2e_legacy_to_result(
        timed$value, data, config = config, elapsed_sec = timed$elapsed
      )
    } else {
      explicit_precision_executors <- if (identical(mode, "primary_only_cuda")) {
        fastkpc_default_precision_executors()
      } else {
        NULL
      }
      fast_kpc(
        data,
        alpha = alpha,
        max_conditioning_size = max_conditioning_size,
        engine = cfg$engine,
        precision = cfg$precision,
        tau = tau,
        ci_method = ci_method,
        graph_stage = "skeleton",
        residual_cache = TRUE,
        hsic_params = hsic_params,
        permutation_params = permutation_params,
        precision_executors = explicit_precision_executors,
        allow_canary_mgcv_extract = TRUE,
        benchmark = TRUE,
        seed = seed
      )
    }
  })
  result <- run$value
  wall <- if (identical(run$status, "ok")) {
    total <- result$timings$elapsed_sec[result$timings$stage == "total"]
    if (length(total) == 0L) NA_real_ else as.numeric(total[[1L]])
  } else {
    NA_real_
  }
  list(
    run_id = run_id,
    mode = mode,
    status = run$status,
    error_message = run$error_message,
    result = result,
    wall_time_sec = wall
  )
}

fastkpc_precision_e2e_trace <- function(result) {
  if (is.null(result)) return(NULL)
  trace <- result$diagnostics$precision_trace %||% NULL
  if (is.null(trace)) return(NULL)
  if (!is.data.frame(trace)) return(NULL)
  trace
}

fastkpc_precision_e2e_sum <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) == 0L) NA_real_ else sum(x)
}

fastkpc_precision_e2e_stage_row <- function(entry, scenario_id, repeat_id, n, p) {
  trace <- fastkpc_precision_e2e_trace(entry$result)
  cfg <- if (is.null(entry$result)) list() else entry$result$config
  timings <- if (is.null(entry$result)) NULL else entry$result$timings
  skeleton_sec <- if (!is.null(timings) && "skeleton" %in% timings$stage) {
    as.numeric(timings$elapsed_sec[timings$stage == "skeleton"][[1L]])
  } else {
    NA_real_
  }
  data.frame(
    run_id = entry$run_id,
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = entry$mode,
    status = entry$status,
    engine = cfg$engine_used %||%
      fastkpc_precision_e2e_mode_config(entry$mode)$engine,
    precision = cfg$precision_requested %||%
      fastkpc_precision_e2e_mode_config(entry$mode)$precision,
    n = as.integer(n),
    p = as.integer(p),
    skeleton_time_sec = skeleton_sec,
    wall_time_sec = entry$wall_time_sec,
    mgcv_setup_cpu_ms =
      fastkpc_precision_e2e_sum(trace$mgcv_setup_cpu_ms),
    spectral_prepare_ms =
      fastkpc_precision_e2e_sum(trace$spectral_prepare_ms),
    gcv_score_ms = fastkpc_precision_e2e_sum(trace$gcv_score_ms),
    cuda_solve_ms = fastkpc_precision_e2e_sum(trace$linear_solve_ms),
    residual_materialize_ms =
      fastkpc_precision_e2e_sum(trace$residual_materialize_ms),
    ci_test_ms = fastkpc_precision_e2e_sum(trace$ci_test_ms),
    precision_trace_total_ms =
      fastkpc_precision_e2e_sum(trace$total_ms),
    scheduler_replay_overhead_ms = NA_real_,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

fastkpc_precision_e2e_cache_row <- function(entry, scenario_id, repeat_id, n, p) {
  cache <- if (is.null(entry$result)) list() else
    entry$result$skeleton$residual_cache %||% list()
  data.frame(
    run_id = entry$run_id,
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = entry$mode,
    status = entry$status,
    n = as.integer(n),
    p = as.integer(p),
    residual_requests = as.integer(cache$requests %||% 0L),
    residual_hits = as.integer(cache$hits %||% 0L),
    residual_misses = as.integer(cache$misses %||% 0L),
    residual_hit_rate = if ((cache$requests %||% 0L) > 0L) {
      as.numeric(cache$hits %||% 0L) / as.numeric(cache$requests)
    } else {
      NA_real_
    },
    setup_cache_requests = as.integer(cache$setup_cache_requests %||% 0L),
    setup_cache_hits = as.integer(cache$setup_cache_hits %||% 0L),
    setup_cache_hit_rate = if ((cache$setup_cache_requests %||% 0L) > 0L) {
      as.numeric(cache$setup_cache_hits %||% 0L) /
        as.numeric(cache$setup_cache_requests)
    } else {
      NA_real_
    },
    spectral_cache_requests =
      as.integer(cache$spectral_cache_requests %||% 0L),
    spectral_cache_hits = as.integer(cache$spectral_cache_hits %||% 0L),
    spectral_cache_hit_rate =
      if ((cache$spectral_cache_requests %||% 0L) > 0L) {
        as.numeric(cache$spectral_cache_hits %||% 0L) /
          as.numeric(cache$spectral_cache_requests)
      } else {
        NA_real_
      },
    cuda_batch_calls = as.integer(cache$cuda_batch_calls %||% 0L),
    cuda_single_target_calls =
      as.integer(cache$cuda_single_target_calls %||% 0L),
    cuda_solve_calls = as.integer(cache$cuda_solve_calls %||% 0L),
    target_computations = as.integer(cache$target_computations %||% 0L),
    prepared_cache_peak_bytes =
      as.numeric(cache$prepared_cache_peak_bytes %||% 0),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

fastkpc_precision_e2e_first_sepset_key <- function(sepsets) {
  if (is.null(sepsets)) return("")
  for (i in seq_along(sepsets)) {
    row <- sepsets[[i]]
    for (j in seq_along(row)) {
      value <- as.integer(row[[j]])
      if (length(value) > 0L) return(paste(i, j, paste(value, collapse = "|"), sep = ":"))
    }
  }
  ""
}

fastkpc_precision_e2e_skeleton_shd <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NA_integer_)
  a <- as.matrix(a)
  b <- as.matrix(b)
  if (!identical(dim(a), dim(b))) return(NA_integer_)
  as.integer(sum(abs(a[upper.tri(a)] - b[upper.tri(b)])))
}

fastkpc_precision_e2e_graph_agreement <- function(entries, scenario_id, repeat_id) {
  ok <- Filter(function(x) identical(x$status, "ok") && !is.null(x$result),
               entries)
  if (length(ok) == 0L) {
    return(fastkpc_empty_df(c(
      "scenario_id", "repeat", "baseline_mode", "mode", "status",
      "adjacency_identical", "skeleton_shd", "first_sepset_identical",
      "pmax_max_abs_diff", "baseline_wall_time_sec", "mode_wall_time_sec",
      "speedup_vs_baseline"
    )))
  }
  baseline <- NULL
  for (candidate in c("legacy_mgcv", "compatible_cuda", "compatible_cpu",
                      "fast_cpu", "fast_cuda")) {
    match <- Filter(function(x) identical(x$mode, candidate), ok)
    if (length(match) > 0L) {
      baseline <- match[[1L]]
      break
    }
  }
  if (is.null(baseline)) baseline <- ok[[1L]]
  rows <- lapply(ok, function(entry) {
    b <- baseline$result$skeleton
    r <- entry$result$skeleton
    pmax_diff <- fastkpc_max_abs_matrix_diff(b$pMax, r$pMax)
    data.frame(
      scenario_id = scenario_id,
      `repeat` = as.integer(repeat_id),
      baseline_mode = baseline$mode,
      mode = entry$mode,
      status = "ok",
      adjacency_identical =
        isTRUE(fastkpc_precision_e2e_skeleton_shd(b$adjacency,
                                                  r$adjacency) == 0L),
      skeleton_shd = fastkpc_precision_e2e_skeleton_shd(b$adjacency,
                                                        r$adjacency),
      first_sepset_identical =
        identical(fastkpc_precision_e2e_first_sepset_key(b$sepsets),
                  fastkpc_precision_e2e_first_sepset_key(r$sepsets)),
      pmax_max_abs_diff = pmax_diff,
      baseline_wall_time_sec = baseline$wall_time_sec,
      mode_wall_time_sec = entry$wall_time_sec,
      speedup_vs_baseline = if (is.finite(entry$wall_time_sec) &&
                                 entry$wall_time_sec > 0) {
        baseline$wall_time_sec / entry$wall_time_sec
      } else {
        NA_real_
      },
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_precision_e2e_run_row <- function(entry, scenario_id, repeat_id, n, p,
                                          alpha, max_conditioning_size,
                                          execution_order,
                                          warmup_enabled) {
  cfg <- if (is.null(entry$result)) list() else entry$result$config
  trace <- fastkpc_precision_e2e_trace(entry$result)
  cache <- if (is.null(entry$result)) list() else
    entry$result$skeleton$residual_cache %||% list()
  verifier_tests <- if (is.null(trace) || !"near_alpha_triggered" %in% names(trace)) {
    0L
  } else {
    sum(trace$near_alpha_triggered %in% TRUE, na.rm = TRUE)
  }
  trace_tests <- if (is.null(trace)) 0L else nrow(trace)
  data.frame(
    run_id = entry$run_id,
    scenario_id = scenario_id,
    `repeat` = as.integer(repeat_id),
    mode = entry$mode,
    execution_order = as.integer(execution_order),
    warmup_enabled = isTRUE(warmup_enabled),
    engine = cfg$engine_used %||%
      fastkpc_precision_e2e_mode_config(entry$mode)$engine,
    precision = cfg$precision_requested %||%
      fastkpc_precision_e2e_mode_config(entry$mode)$precision,
    status = entry$status,
    error_message = entry$error_message,
    n = as.integer(n),
    p = as.integer(p),
    alpha = as.numeric(alpha),
    max_conditioning_size = as.integer(max_conditioning_size),
    wall_time_sec = entry$wall_time_sec,
    backend_used = cfg$backend_used %||% cfg$backend_executed %||% NA_character_,
    verifier_executed = cfg$verifier_executed %||% NA_character_,
    ci_backend = cfg$ci_backend %||% NA_character_,
    residual_backend = if (is.null(entry$result)) NA_character_ else
      entry$result$skeleton$residual_backend %||% cfg$backend_used %||% NA_character_,
    n_edgetests = if (is.null(entry$result)) NA_integer_ else
      as.integer(entry$result$skeleton$scheduler_diagnostics$summary$tests_replayed %||%
                   entry$result$skeleton$scheduler_diagnostics$summary$n_edgetests %||%
                   NA_integer_),
    verifier_tests = as.integer(verifier_tests),
    verifier_rate = if (trace_tests > 0L) verifier_tests / trace_tests else NA_real_,
    cache_hit_rate = if ((cache$requests %||% 0L) > 0L) {
      as.numeric(cache$hits %||% 0L) / as.numeric(cache$requests)
    } else {
      NA_real_
    },
    setup_cache_hit_rate = if ((cache$setup_cache_requests %||% 0L) > 0L) {
      as.numeric(cache$setup_cache_hits %||% 0L) /
        as.numeric(cache$setup_cache_requests)
    } else {
      NA_real_
    },
    spectral_cache_hit_rate =
      if ((cache$spectral_cache_requests %||% 0L) > 0L) {
        as.numeric(cache$spectral_cache_hits %||% 0L) /
          as.numeric(cache$spectral_cache_requests)
      } else {
        NA_real_
      },
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

fastkpc_precision_e2e_finite <- function(x) {
  x <- as.numeric(x)
  x[is.finite(x)]
}

fastkpc_precision_e2e_median <- function(x) {
  x <- fastkpc_precision_e2e_finite(x)
  if (length(x) == 0L) NA_real_ else stats::median(x)
}

fastkpc_precision_e2e_p90 <- function(x) {
  x <- fastkpc_precision_e2e_finite(x)
  if (length(x) == 0L) {
    NA_real_
  } else {
    as.numeric(stats::quantile(x, probs = 0.9, names = FALSE, type = 7))
  }
}

fastkpc_precision_e2e_geomean <- function(x) {
  x <- fastkpc_precision_e2e_finite(x)
  x <- x[x > 0]
  if (length(x) == 0L) NA_real_ else exp(mean(log(x)))
}

fastkpc_precision_e2e_max_finite_or_na <- function(x) {
  x <- fastkpc_precision_e2e_finite(x)
  if (length(x) == 0L) NA_real_ else max(x)
}

fastkpc_precision_e2e_stage_shares <- function(stage_timing) {
  out <- stage_timing
  setup <- rowSums(cbind(
    as.numeric(out$mgcv_setup_cpu_ms),
    as.numeric(out$spectral_prepare_ms),
    as.numeric(out$gcv_score_ms)
  ), na.rm = TRUE)
  solve <- rowSums(cbind(
    as.numeric(out$cuda_solve_ms),
    as.numeric(out$residual_materialize_ms)
  ), na.rm = TRUE)
  ci <- as.numeric(out$ci_test_ms)
  ci[!is.finite(ci)] <- 0
  total <- setup + solve + ci
  wall_ms <- as.numeric(out$wall_time_sec) * 1000
  accounted <- setup + solve + ci
  out$stage_ms_setup_spectral <- setup
  out$stage_ms_cuda_solve <- solve
  out$stage_ms_ci <- ci
  out$stage_accounted_ms <- accounted
  out$stage_accounted_share_of_wall <-
    ifelse(is.finite(wall_ms) & wall_ms > 0, accounted / wall_ms, NA_real_)
  out$stage_share_setup_spectral <- ifelse(total > 0, setup / total, NA_real_)
  out$stage_share_cuda_solve <- ifelse(total > 0, solve / total, NA_real_)
  out$stage_share_ci <- ifelse(total > 0, ci / total, NA_real_)
  out
}

fastkpc_precision_e2e_mode_summary <- function(runs, stage_timing,
                                               graph_agreement) {
  stage_timing <- fastkpc_precision_e2e_stage_shares(stage_timing)
  modes <- unique(runs$mode)
  rows <- lapply(modes, function(mode) {
    run_rows <- runs[runs$mode == mode, , drop = FALSE]
    ok <- run_rows[run_rows$status == "ok", , drop = FALSE]
    stages <- stage_timing[stage_timing$mode == mode &
                             stage_timing$status == "ok", , drop = FALSE]
    graph <- graph_agreement[graph_agreement$mode == mode &
                               graph_agreement$status == "ok", , drop = FALSE]
    data.frame(
      mode = mode,
      ok_runs = as.integer(sum(run_rows$status == "ok")),
      skipped_runs = as.integer(sum(run_rows$status == "skipped")),
      error_runs = as.integer(sum(run_rows$status == "error")),
      median_wall_time_sec =
        fastkpc_precision_e2e_median(ok$wall_time_sec),
      p90_wall_time_sec = fastkpc_precision_e2e_p90(ok$wall_time_sec),
      geomean_wall_time_sec =
        fastkpc_precision_e2e_geomean(ok$wall_time_sec),
      median_speedup_vs_legacy =
        fastkpc_precision_e2e_median(graph$speedup_vs_baseline),
      geomean_speedup_vs_legacy =
        fastkpc_precision_e2e_geomean(graph$speedup_vs_baseline),
      median_stage_share_setup_spectral =
        fastkpc_precision_e2e_median(stages$stage_share_setup_spectral),
      median_stage_share_cuda_solve =
        fastkpc_precision_e2e_median(stages$stage_share_cuda_solve),
      median_stage_share_ci =
        fastkpc_precision_e2e_median(stages$stage_share_ci),
      median_stage_accounted_share_of_wall =
        fastkpc_precision_e2e_median(stages$stage_accounted_share_of_wall),
      median_verifier_rate =
        fastkpc_precision_e2e_median(ok$verifier_rate),
      median_cache_hit_rate =
        fastkpc_precision_e2e_median(ok$cache_hit_rate),
      median_setup_cache_hit_rate =
        fastkpc_precision_e2e_median(ok$setup_cache_hit_rate),
      median_spectral_cache_hit_rate =
        fastkpc_precision_e2e_median(ok$spectral_cache_hit_rate),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_precision_e2e_entry_verifier_rate <- function(entry) {
  trace <- fastkpc_precision_e2e_trace(entry$result)
  if (is.null(trace) || nrow(trace) == 0L ||
      !"near_alpha_triggered" %in% names(trace)) {
    return(NA_real_)
  }
  sum(trace$near_alpha_triggered %in% TRUE, na.rm = TRUE) / nrow(trace)
}

fastkpc_precision_e2e_find_entry <- function(entries, mode) {
  matches <- Filter(function(x) identical(x$mode, mode) &&
                      identical(x$status, "ok") && !is.null(x$result),
                    entries)
  if (length(matches) == 0L) NULL else matches[[1L]]
}

fastkpc_precision_e2e_pair_metrics <- function(entries_by_scenario,
                                               left_mode, right_mode,
                                               comparison) {
  rows <- list()
  for (entries in entries_by_scenario) {
    left <- fastkpc_precision_e2e_find_entry(entries, left_mode)
    right <- fastkpc_precision_e2e_find_entry(entries, right_mode)
    if (is.null(left) || is.null(right)) next
    left_skel <- left$result$skeleton
    right_skel <- right$result$skeleton
    wall_ratio <- if (is.finite(left$wall_time_sec) &&
                      left$wall_time_sec > 0) {
      right$wall_time_sec / left$wall_time_sec
    } else {
      NA_real_
    }
    rows[[length(rows) + 1L]] <- data.frame(
      comparison = comparison,
      left_mode = left_mode,
      right_mode = right_mode,
      wall_time_ratio = wall_ratio,
      overhead_pct = if (is.finite(wall_ratio)) (wall_ratio - 1) * 100 else NA_real_,
      right_verifier_rate =
        fastkpc_precision_e2e_entry_verifier_rate(right),
      skeleton_shd =
        fastkpc_precision_e2e_skeleton_shd(left_skel$adjacency,
                                           right_skel$adjacency),
      first_sepset_identical =
        identical(fastkpc_precision_e2e_first_sepset_key(left_skel$sepsets),
                  fastkpc_precision_e2e_first_sepset_key(right_skel$sepsets)),
      pmax_max_abs_diff =
        fastkpc_max_abs_matrix_diff(left_skel$pMax, right_skel$pMax),
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) {
    return(fastkpc_empty_df(c(
      "comparison", "left_mode", "right_mode", "wall_time_ratio",
      "overhead_pct", "right_verifier_rate", "skeleton_shd",
      "first_sepset_identical", "pmax_max_abs_diff"
    )))
  }
  do.call(rbind, rows)
}

fastkpc_precision_e2e_comparison_summary <- function(entries_by_scenario) {
  pairs <- list(
    fastkpc_precision_e2e_pair_metrics(
      entries_by_scenario,
      left_mode = "primary_only_cuda",
      right_mode = "hybrid_cuda",
      comparison = "hybrid_cuda_vs_primary_only_cuda"
    ),
    fastkpc_precision_e2e_pair_metrics(
      entries_by_scenario,
      left_mode = "fast_cuda",
      right_mode = "primary_only_cuda",
      comparison = "primary_only_cuda_vs_fast_cuda"
    )
  )
  rows <- do.call(rbind, pairs)
  if (nrow(rows) == 0L) {
    return(data.frame(
      comparison = "hybrid_cuda_vs_primary_only_cuda",
      pair_count = 0L,
      median_wall_time_ratio = NA_real_,
      p90_wall_time_ratio = NA_real_,
      geomean_wall_time_ratio = NA_real_,
      median_overhead_pct = NA_real_,
      median_verifier_rate = NA_real_,
      median_skeleton_shd = NA_real_,
      sepset_mismatch_rate = NA_real_,
      max_pmax_abs_diff = NA_real_,
      stringsAsFactors = FALSE
    ))
  }
  comparisons <- unique(rows$comparison)
  out <- lapply(comparisons, function(comparison) {
    subset <- rows[rows$comparison == comparison, , drop = FALSE]
    data.frame(
      comparison = comparison,
      pair_count = nrow(subset),
      median_wall_time_ratio =
        fastkpc_precision_e2e_median(subset$wall_time_ratio),
      p90_wall_time_ratio =
        fastkpc_precision_e2e_p90(subset$wall_time_ratio),
      geomean_wall_time_ratio =
        fastkpc_precision_e2e_geomean(subset$wall_time_ratio),
      median_overhead_pct =
        fastkpc_precision_e2e_median(subset$overhead_pct),
      median_verifier_rate =
        fastkpc_precision_e2e_median(subset$right_verifier_rate),
      median_skeleton_shd =
        fastkpc_precision_e2e_median(subset$skeleton_shd),
      sepset_mismatch_rate =
        mean(!(subset$first_sepset_identical %in% TRUE), na.rm = TRUE),
      max_pmax_abs_diff =
        fastkpc_precision_e2e_max_finite_or_na(subset$pmax_max_abs_diff),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

fastkpc_precision_e2e_bottleneck_decision <- function(mode_summary) {
  scopes <- intersect(c("hybrid_cuda", "compatible_cuda", "primary_only_cuda"),
                      mode_summary$mode)
  if (length(scopes) == 0L) scopes <- mode_summary$mode
  rows <- lapply(scopes, function(mode) {
    row <- mode_summary[mode_summary$mode == mode, , drop = FALSE][1L, ]
    shares <- c(
      setup_spectral = as.numeric(row$median_stage_share_setup_spectral),
      cuda_solve = as.numeric(row$median_stage_share_cuda_solve),
      ci = as.numeric(row$median_stage_share_ci)
    )
    shares[!is.finite(shares)] <- NA_real_
    if (all(is.na(shares))) {
      dominant <- "unknown"
      recommendation <- "collect larger same-scheduler benchmark evidence"
      attribution_note <- "stage timings unavailable"
    } else {
      dominant <- names(which.max(replace(shares, is.na(shares), -Inf)))
      max_share <- max(shares, na.rm = TRUE)
      recommendation <- if (identical(dominant, "ci") && max_share > 0.4) {
        "connect precision scheduler CI/eval path to CUDA dCov/HSIC"
      } else if (identical(dominant, "setup_spectral") && max_share > 0.4) {
        "optimize same-S grouping and setup/spectral reuse"
      } else if (identical(dominant, "cuda_solve") && max_share > 0.4) {
        "evaluate fused/batched CUDA solve"
      } else {
        "collect larger benchmark evidence before kernel work"
      }
      attribution_note <- if (identical(mode, "primary_only_cuda")) {
        "fastSpline primary timing is aggregated as precision CI/eval"
      } else if (identical(mode, "hybrid_cuda")) {
        "hybrid includes primary CI/eval plus sparse verifier timing"
      } else {
        "mgcvExtractGPU stage timing is directly attributable"
      }
    }
    data.frame(
      analysis_scope = mode,
      dominant_phase = dominant,
      setup_spectral_share = shares[["setup_spectral"]],
      cuda_solve_share = shares[["cuda_solve"]],
      ci_share = shares[["ci"]],
      stage_accounted_share_of_wall =
        as.numeric(row$median_stage_accounted_share_of_wall),
      attribution_note = attribution_note,
      recommended_next_optimization = recommendation,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_precision_e2e_stage_totals <- function(stage_timing) {
  numeric_cols <- c(
    setup_spectral = sum(stage_timing$mgcv_setup_cpu_ms,
                         stage_timing$spectral_prepare_ms,
                         stage_timing$gcv_score_ms,
                         na.rm = TRUE),
    cuda_solve = sum(stage_timing$cuda_solve_ms,
                     stage_timing$residual_materialize_ms,
                     na.rm = TRUE),
    ci = sum(stage_timing$ci_test_ms, na.rm = TRUE)
  )
  if (all(numeric_cols <= 0 | !is.finite(numeric_cols))) {
    return(c(setup_spectral = 0, cuda_solve = 0, ci = 0))
  }
  numeric_cols
}

fastkpc_precision_e2e_recommendation <- function(stage_timing, runs) {
  totals <- fastkpc_precision_e2e_stage_totals(stage_timing)
  if (sum(totals, na.rm = TRUE) <= 0) {
    return("scheduler/replay profiling: precision stage timings were unavailable or zero")
  }
  phase <- names(which.max(totals))
  switch(
    phase,
    ci = "connect device-resident CUDA dCov/HSIC if CI dominates the next native run",
    setup_spectral = "amortize setup/spectral work before adding new CUDA kernels",
    cuda_solve = "evaluate fused/batched CUDA solve only if target batch width is high",
    "collect a larger benchmark before selecting a kernel optimization"
  )
}

fastkpc_precision_e2e_summary_recommendation <- function(bottleneck_decision) {
  if (nrow(bottleneck_decision) == 0L) {
    return("collect larger benchmark evidence before selecting optimization")
  }
  for (scope in c("hybrid_cuda", "compatible_cuda", "primary_only_cuda")) {
    row <- bottleneck_decision[
      bottleneck_decision$analysis_scope == scope, , drop = FALSE
    ]
    if (nrow(row) > 0L) return(row$recommended_next_optimization[[1L]])
  }
  bottleneck_decision$recommended_next_optimization[[1L]]
}

fastkpc_precision_e2e_json_value <- function(value) {
  if (is.logical(value)) {
    if (isTRUE(value)) return("true")
    if (identical(value, FALSE)) return("false")
    return("null")
  }
  if (is.numeric(value) || is.integer(value)) {
    value <- as.numeric(value)[1L]
    if (!is.finite(value)) return("null")
    return(format(signif(value, 8), scientific = FALSE, trim = TRUE))
  }
  value <- as.character(value %||% "")[1L]
  paste0('"', gsub('"', '\\"', value, fixed = TRUE), '"')
}

fastkpc_precision_e2e_write_summary <- function(summary, output_dir) {
  json_path <- file.path(output_dir, "summary.json")
  md_path <- file.path(output_dir, "summary.md")
  keys <- names(summary)
  json <- c(
    "{",
    paste0("  \"", keys, "\": ",
           vapply(summary, fastkpc_precision_e2e_json_value, character(1L)),
           c(rep(",", max(length(keys) - 1L, 0L)), "")),
    "}"
  )
  writeLines(json, json_path)
  md <- c(
    "# Precision End-to-End Benchmark Summary",
    "",
    paste0("- Total runs: ", summary$total_runs),
    paste0("- OK runs: ", summary$ok_runs),
    paste0("- Skipped runs: ", summary$skipped_runs),
    paste0("- Error runs: ", summary$error_runs),
    paste0("- Fastest OK mode: ", summary$fastest_mode),
    paste0("- Recommended next optimization: ",
           summary$recommended_next_optimization)
  )
  writeLines(md, md_path)
  list(summary_json = json_path, summary_md = md_path)
}

fastkpc_run_precision_end_to_end_benchmark <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "precision_end_to_end_benchmark"),
    scenarios = NULL,
    modes = c("legacy_mgcv", "fast_cuda", "primary_only_cuda",
              "compatible_cuda", "hybrid_cuda"),
    repeats = 5L,
    seed = 6202L,
    alpha = 0.05,
    max_conditioning_size = 2L,
    ci_method = "dcc.gamma",
    tau = log(2),
    hsic_params = list(sig = 1),
    permutation_params = list(replicates = 100, seed = NULL,
                              include_observed = TRUE),
    warmup = TRUE,
    randomize_mode_order = TRUE,
    real_data_path = Sys.getenv("FASTKPC_PRECISION_E2E_REAL_DATA", ""),
    run_native_cuda = identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  if (is.null(scenarios)) scenarios <- fastkpc_precision_e2e_default_scenarios()
  scenarios <- fastkpc_precision_e2e_append_real_scenario(scenarios,
                                                          real_data_path)
  modes <- match.arg(modes, c("legacy_mgcv", "fast_cuda", "primary_only_cuda",
                              "compatible_cuda",
                              "hybrid_cuda", "fast_cpu", "compatible_cpu",
                              "hybrid_cpu"), several.ok = TRUE)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  run_rows <- list()
  stage_rows <- list()
  cache_rows <- list()
  graph_rows <- list()
  entries_by_scenario <- list()

  for (scenario in scenarios) {
    scenario_id <- scenario$scenario_id %||% paste0("scenario-", length(entries_by_scenario) + 1L)
    n <- as.integer(scenario$n %||% 120L)
    scenario_seed <- as.integer(scenario$seed %||% seed)
    generator <- scenario$generator
    if (is.null(generator) || !is.function(generator)) {
      stop("Each benchmark scenario must provide a generator(n, seed)",
           call. = FALSE)
    }
    if (isTRUE(warmup)) {
      warmup_data <- as.matrix(generator(n, scenario_seed - 1L))
      storage.mode(warmup_data) <- "double"
      if (is.null(colnames(warmup_data))) {
        colnames(warmup_data) <- paste0("V", seq_len(ncol(warmup_data)))
      }
      warmup_modes <- modes
      if (isTRUE(randomize_mode_order) && length(warmup_modes) > 1L) {
        set.seed(seed + scenario_seed + 997L)
        warmup_modes <- sample(warmup_modes)
      }
      for (mode in warmup_modes) {
        invisible(fastkpc_precision_e2e_run_mode(
          data = warmup_data,
          scenario_id = scenario_id,
          mode = mode,
          repeat_id = 0L,
          alpha = alpha,
          max_conditioning_size = max_conditioning_size,
          ci_method = ci_method,
          tau = tau,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          run_native_cuda = run_native_cuda,
          seed = scenario_seed - 1L
        ))
      }
    }
    for (repeat_id in seq_len(as.integer(repeats))) {
      repeat_seed <- scenario_seed + repeat_id - 1L
      data <- as.matrix(generator(n, repeat_seed))
      storage.mode(data) <- "double"
      if (is.null(colnames(data))) {
        colnames(data) <- paste0("V", seq_len(ncol(data)))
      }
      repeat_entries <- list()
      measured_modes <- modes
      if (isTRUE(randomize_mode_order) && length(measured_modes) > 1L) {
        set.seed(seed + scenario_seed + repeat_id * 1009L)
        measured_modes <- sample(measured_modes)
      }
      for (execution_order in seq_along(measured_modes)) {
        mode <- measured_modes[[execution_order]]
        entry <- fastkpc_precision_e2e_run_mode(
          data = data,
          scenario_id = scenario_id,
          mode = mode,
          repeat_id = repeat_id,
          alpha = alpha,
          max_conditioning_size = max_conditioning_size,
          ci_method = ci_method,
          tau = tau,
          hsic_params = hsic_params,
          permutation_params = permutation_params,
          run_native_cuda = run_native_cuda,
          seed = repeat_seed
        )
        repeat_entries[[length(repeat_entries) + 1L]] <- entry
        run_rows[[length(run_rows) + 1L]] <- fastkpc_precision_e2e_run_row(
          entry, scenario_id, repeat_id, nrow(data), ncol(data), alpha,
          max_conditioning_size, execution_order = execution_order,
          warmup_enabled = warmup
        )
        stage_rows[[length(stage_rows) + 1L]] <-
          fastkpc_precision_e2e_stage_row(
            entry, scenario_id, repeat_id, nrow(data), ncol(data)
          )
        cache_rows[[length(cache_rows) + 1L]] <-
          fastkpc_precision_e2e_cache_row(
            entry, scenario_id, repeat_id, nrow(data), ncol(data)
          )
      }
      graph_rows[[length(graph_rows) + 1L]] <-
        fastkpc_precision_e2e_graph_agreement(repeat_entries, scenario_id,
                                              repeat_id)
      entries_by_scenario[[length(entries_by_scenario) + 1L]] <- repeat_entries
    }
  }

  runs <- do.call(rbind, run_rows)
  stage_timing <- do.call(rbind, stage_rows)
  cache <- do.call(rbind, cache_rows)
  graph_agreement <- do.call(rbind, graph_rows)
  mode_summary <- fastkpc_precision_e2e_mode_summary(
    runs, stage_timing, graph_agreement
  )
  comparison_summary <- fastkpc_precision_e2e_comparison_summary(
    entries_by_scenario
  )
  bottleneck_decision <- fastkpc_precision_e2e_bottleneck_decision(
    mode_summary
  )

  ok <- runs[runs$status == "ok", , drop = FALSE]
  ok_modes <- mode_summary[mode_summary$ok_runs > 0L &
                             is.finite(mode_summary$median_wall_time_sec),
                           , drop = FALSE]
  fastest_mode <- if (nrow(ok_modes) > 0L) {
    ok_modes$mode[[which.min(ok_modes$median_wall_time_sec)]]
  } else {
    ""
  }
  fastest_single_run_mode <- if (nrow(ok) > 0L) {
    ok$mode[[which.min(ok$wall_time_sec)]]
  } else {
    ""
  }
  summary <- list(
    total_runs = as.integer(nrow(runs)),
    ok_runs = as.integer(sum(runs$status == "ok")),
    skipped_runs = as.integer(sum(runs$status == "skipped")),
    error_runs = as.integer(sum(runs$status == "error")),
    fastest_mode = fastest_mode,
    fastest_single_run_mode = fastest_single_run_mode,
    native_cuda_requested = isTRUE(run_native_cuda),
    repeats = as.integer(repeats),
    warmup_enabled = isTRUE(warmup),
    randomize_mode_order = isTRUE(randomize_mode_order),
    recommended_next_optimization =
      fastkpc_precision_e2e_summary_recommendation(bottleneck_decision)
  )

  paths <- list(
    runs = file.path(output_dir, "runs.csv"),
    stage_timing = file.path(output_dir, "stage_timing.csv"),
    cache = file.path(output_dir, "cache.csv"),
    graph_agreement = file.path(output_dir, "graph_agreement.csv"),
    mode_summary = file.path(output_dir, "mode_summary.csv"),
    comparison_summary = file.path(output_dir, "comparison_summary.csv"),
    bottleneck_decision = file.path(output_dir, "bottleneck_decision.csv")
  )
  utils::write.csv(runs, paths$runs, row.names = FALSE)
  utils::write.csv(stage_timing, paths$stage_timing, row.names = FALSE)
  utils::write.csv(cache, paths$cache, row.names = FALSE)
  utils::write.csv(graph_agreement, paths$graph_agreement, row.names = FALSE)
  utils::write.csv(mode_summary, paths$mode_summary, row.names = FALSE)
  utils::write.csv(comparison_summary, paths$comparison_summary,
                   row.names = FALSE)
  utils::write.csv(bottleneck_decision, paths$bottleneck_decision,
                   row.names = FALSE)
  paths <- c(paths, fastkpc_precision_e2e_write_summary(summary, output_dir))

  list(
    runs = runs,
    stage_timing = stage_timing,
    cache = cache,
    graph_agreement = graph_agreement,
    mode_summary = mode_summary,
    comparison_summary = comparison_summary,
    bottleneck_decision = bottleneck_decision,
    summary = summary,
    paths = paths,
    output_dir = output_dir
  )
}
