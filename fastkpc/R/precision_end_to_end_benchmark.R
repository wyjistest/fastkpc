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
    )
  )
}

fastkpc_precision_e2e_mode_config <- function(mode) {
  mode <- match.arg(mode, c("legacy_mgcv", "fast_cuda", "compatible_cuda",
                           "hybrid_cuda", "fast_cpu", "compatible_cpu",
                           "hybrid_cpu"))
  switch(
    mode,
    legacy_mgcv = list(engine = "cpu", precision = "legacy-mgcv",
                       label = "legacy kpcalg + mgcv"),
    fast_cuda = list(engine = "cuda", precision = "fast",
                     label = "fastSplineCUDA"),
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
                                          alpha, max_conditioning_size) {
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
    modes = c("legacy_mgcv", "fast_cuda", "compatible_cuda", "hybrid_cuda"),
    repeats = 1L,
    seed = 6202L,
    alpha = 0.05,
    max_conditioning_size = 2L,
    ci_method = "dcc.gamma",
    tau = log(2),
    hsic_params = list(sig = 1),
    permutation_params = list(replicates = 100, seed = NULL,
                              include_observed = TRUE),
    run_native_cuda = identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  if (is.null(scenarios)) scenarios <- fastkpc_precision_e2e_default_scenarios()
  modes <- match.arg(modes, c("legacy_mgcv", "fast_cuda", "compatible_cuda",
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
    for (repeat_id in seq_len(as.integer(repeats))) {
      repeat_seed <- scenario_seed + repeat_id - 1L
      data <- as.matrix(generator(n, repeat_seed))
      storage.mode(data) <- "double"
      if (is.null(colnames(data))) {
        colnames(data) <- paste0("V", seq_len(ncol(data)))
      }
      repeat_entries <- list()
      for (mode in modes) {
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
          max_conditioning_size
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

  ok <- runs[runs$status == "ok", , drop = FALSE]
  fastest_mode <- if (nrow(ok) > 0L) {
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
    native_cuda_requested = isTRUE(run_native_cuda),
    recommended_next_optimization =
      fastkpc_precision_e2e_recommendation(stage_timing, runs)
  )

  paths <- list(
    runs = file.path(output_dir, "runs.csv"),
    stage_timing = file.path(output_dir, "stage_timing.csv"),
    cache = file.path(output_dir, "cache.csv"),
    graph_agreement = file.path(output_dir, "graph_agreement.csv")
  )
  utils::write.csv(runs, paths$runs, row.names = FALSE)
  utils::write.csv(stage_timing, paths$stage_timing, row.names = FALSE)
  utils::write.csv(cache, paths$cache, row.names = FALSE)
  utils::write.csv(graph_agreement, paths$graph_agreement, row.names = FALSE)
  paths <- c(paths, fastkpc_precision_e2e_write_summary(summary, output_dir))

  list(
    runs = runs,
    stage_timing = stage_timing,
    cache = cache,
    graph_agreement = graph_agreement,
    summary = summary,
    paths = paths,
    output_dir = output_dir
  )
}
