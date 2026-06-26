source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/legacy_runner.R")

if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

fastkpc_fast_cuda_baseline_bool_env <- function(name, default = FALSE) {
  value <- tolower(Sys.getenv(name, ""))
  if (!nzchar(value)) return(default)
  value %in% c("1", "true", "yes", "y")
}

fastkpc_fast_cuda_baseline_int_env <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.integer(value)
}

fastkpc_fast_cuda_baseline_modes <- function() {
  c("fast_cuda", "fast_cpu", "legacy_mgcv")
}

fastkpc_fast_cuda_baseline_synthetic_data <- function(n, p, seed) {
  set.seed(seed)
  roots <- matrix(stats::rnorm(n * max(2L, ceiling(p / 3L))), n)
  out <- matrix(0, n, p)
  for (j in seq_len(p)) {
    r1 <- roots[, ((j - 1L) %% ncol(roots)) + 1L]
    r2 <- roots[, (j %% ncol(roots)) + 1L]
    out[, j] <- sin(r1 * (0.5 + j / p)) + 0.35 * cos(r2) +
      0.15 * r1 * r2 + stats::rnorm(n, sd = 0.25)
  }
  colnames(out) <- paste0("V", seq_len(p))
  out
}

fastkpc_fast_cuda_baseline_real_data <- function(path, n = NULL, p = NULL) {
  if (is.null(path) || !nzchar(path)) return(NULL)
  if (!file.exists(path)) {
    stop("FASTKPC_FAST_CUDA_REAL_DATA does not exist: ", path, call. = FALSE)
  }
  ext <- tolower(tools::file_ext(path))
  raw <- if (identical(ext, "rds")) {
    readRDS(path)
  } else if (identical(ext, "rdata")) {
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    values <- mget(ls(env), envir = env)
    values[[which.max(vapply(values, function(x) {
      if (is.data.frame(x) || is.matrix(x)) prod(dim(x)) else 0
    }, numeric(1L)))]]
  } else {
    utils::read.csv(path, check.names = FALSE)
  }
  data <- as.data.frame(raw)
  numeric <- vapply(data, is.numeric, logical(1L))
  data <- data[, numeric, drop = FALSE]
  if (ncol(data) < 3L) {
    stop("real baseline data must contain at least three numeric columns",
         call. = FALSE)
  }
  matrix_data <- as.matrix(data)
  keep <- stats::complete.cases(matrix_data) &
    apply(matrix_data, 1L, function(row) all(is.finite(row)))
  matrix_data <- matrix_data[keep, , drop = FALSE]
  if (!is.null(n)) matrix_data <- matrix_data[seq_len(min(n, nrow(matrix_data))), , drop = FALSE]
  if (!is.null(p)) matrix_data <- matrix_data[, seq_len(min(p, ncol(matrix_data))), drop = FALSE]
  if (nrow(matrix_data) < 20L || ncol(matrix_data) < 3L) {
    stop("real baseline data is too small after filtering/subsetting",
         call. = FALSE)
  }
  colnames(matrix_data) <- colnames(matrix_data) %||% paste0("V", seq_len(ncol(matrix_data)))
  matrix_data
}

fastkpc_fast_cuda_baseline_scenarios <- function(full_grid = FALSE,
                                                 real_data_path = "",
                                                 real_n = 100L,
                                                 real_p = 12L) {
  if (isTRUE(full_grid)) {
    ns <- c(100L, 300L, 1000L)
    ps <- c(8L, 12L, 30L)
    levels <- c(1L, 2L, 3L)
  } else {
    ns <- c(100L, 300L)
    ps <- c(8L, 12L)
    levels <- c(1L, 2L)
  }
  scenarios <- list()
  sid <- 0L
  for (n in ns) {
    for (p in ps) {
      for (level in levels) {
        sid <- sid + 1L
        seed <- 8600L + sid
        scenarios[[length(scenarios) + 1L]] <- list(
          scenario_id = paste0("synthetic-n", n, "-p", p, "-m", level),
          source = "synthetic",
          n = as.integer(n),
          p = as.integer(p),
          max_conditioning_size = as.integer(level),
          seed = seed,
          data = fastkpc_fast_cuda_baseline_synthetic_data(n, p, seed)
        )
      }
    }
  }
  real <- fastkpc_fast_cuda_baseline_real_data(real_data_path, real_n, real_p)
  if (!is.null(real)) {
    scenarios[[length(scenarios) + 1L]] <- list(
      scenario_id = paste0("real-", tools::file_path_sans_ext(basename(real_data_path))),
      source = "real",
      n = as.integer(nrow(real)),
      p = as.integer(ncol(real)),
      max_conditioning_size = min(2L, max(1L, ncol(real) - 2L)),
      seed = 0L,
      data = real
    )
  }
  scenarios
}

fastkpc_fast_cuda_mode_config <- function(mode) {
  mode <- match.arg(mode, fastkpc_fast_cuda_baseline_modes())
  switch(
    mode,
    fast_cuda = list(engine = "cuda", precision = "fast",
                     residual_backend = "fastSpline"),
    fast_cpu = list(engine = "cpu", precision = "legacy",
                    residual_backend = "fastSpline"),
    legacy_mgcv = list(engine = "cpu", precision = "legacy-mgcv",
                       residual_backend = "legacy-mgcv")
  )
}

fastkpc_fast_cuda_legacy_skip_reason <- function(scenario,
                                                 legacy_max_n = 300L,
                                                 legacy_max_p = 12L,
                                                 legacy_max_level = 2L) {
  if (scenario$n > legacy_max_n) return("legacy skipped: n above configured limit")
  if (scenario$p > legacy_max_p) return("legacy skipped: p above configured limit")
  if (scenario$max_conditioning_size > legacy_max_level) {
    return("legacy skipped: conditioning level above configured limit")
  }
  missing <- c("pcalg", "mgcv", "graph")[
    !vapply(c("pcalg", "mgcv", "graph"), requireNamespace,
            logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0L) {
    return(paste("legacy skipped: missing dependency",
                 paste(missing, collapse = ",")))
  }
  ""
}

fastkpc_fast_cuda_skip_reason <- function(mode, scenario, run_native_cuda,
                                          legacy_max_n, legacy_max_p,
                                          legacy_max_level) {
  if (identical(mode, "fast_cuda")) {
    if (!isTRUE(run_native_cuda)) return("CUDA disabled; set FASTKPC_RUN_CUDA_TESTS=1")
    cuda_ok <- exists("fastkpc_cuda_available", mode = "function") &&
      isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))
    if (!cuda_ok) return("CUDA runtime unavailable")
  }
  if (identical(mode, "legacy_mgcv")) {
    return(fastkpc_fast_cuda_legacy_skip_reason(
      scenario, legacy_max_n, legacy_max_p, legacy_max_level
    ))
  }
  ""
}

fastkpc_fast_cuda_timed <- function(expr) {
  start <- proc.time()[["elapsed"]]
  value <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - start
  list(value = value, elapsed_sec = max(elapsed, 0))
}

fastkpc_fast_cuda_safe <- function(expr) {
  tryCatch(
    list(status = "ok", value = force(expr), error_message = ""),
    error = function(e) list(status = "error", value = NULL,
                             error_message = conditionMessage(e))
  )
}

fastkpc_fast_cuda_legacy_to_result <- function(legacy, data, elapsed_sec) {
  graph <- methods::as(legacy@graph, "matrix")
  adjacency <- matrix(as.integer(graph != 0), nrow = nrow(graph))
  colnames(adjacency) <- rownames(adjacency) <- colnames(data)
  list(
    config = list(
      engine_used = "cpu",
      precision_requested = "legacy-mgcv",
      backend_executed = "legacy-mgcv",
      backend_used = "legacy-mgcv",
      ci_backend = "legacy-kpcalg"
    ),
    skeleton = list(
      adjacency = adjacency,
      pMax = as.matrix(legacy@pMax),
      sepsets = legacy@sepset,
      n.edgetests = as.integer(legacy@n.edgetests),
      scheduler = "legacy-mgcv",
      scheduler_diagnostics = list(summary = list(
        tests_replayed = as.integer(sum(legacy@n.edgetests, na.rm = TRUE))
      )),
      residual_cache = list(requests = 0L, hits = 0L, misses = 0L,
                            computations = 0L),
      residual_backend = "legacy-mgcv",
      ci_backend = "legacy-kpcalg"
    ),
    timings = data.frame(stage = c("skeleton", "total"),
                         elapsed_sec = c(elapsed_sec, elapsed_sec),
                         stringsAsFactors = FALSE),
    diagnostics = list(precision_trace = NULL)
  )
}

fastkpc_fast_cuda_run_mode <- function(scenario, mode, repeat_id, alpha,
                                       ci_method, run_native_cuda,
                                       legacy_max_n, legacy_max_p,
                                       legacy_max_level) {
  skip <- fastkpc_fast_cuda_skip_reason(
    mode, scenario, run_native_cuda, legacy_max_n, legacy_max_p,
    legacy_max_level
  )
  run_id <- paste(scenario$scenario_id, mode, paste0("r", repeat_id), sep = "-")
  if (nzchar(skip)) {
    return(list(run_id = run_id, mode = mode, repeat_id = repeat_id,
                status = "skipped", error_message = skip, result = NULL,
                wall_sec = NA_real_))
  }
  data <- scenario$data
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  run <- fastkpc_fast_cuda_safe({
    if (identical(mode, "legacy_mgcv")) {
      timed <- fastkpc_fast_cuda_timed(
        fastkpc_legacy_skeleton(
          data,
          alpha = alpha,
          max_conditioning_size = scenario$max_conditioning_size,
          ic.method = ci_method
        )
      )
      fastkpc_fast_cuda_legacy_to_result(timed$value, data,
                                         timed$elapsed_sec)
    } else {
      cfg <- fastkpc_fast_cuda_mode_config(mode)
      fast_kpc(
        data,
        alpha = alpha,
        max_conditioning_size = scenario$max_conditioning_size,
        engine = cfg$engine,
        precision = cfg$precision,
        graph_stage = "skeleton",
        residual_backend = cfg$residual_backend,
        ci_method = ci_method,
        residual_cache = TRUE,
        fastspline_params = params,
        benchmark = TRUE,
        seed = scenario$seed + repeat_id
      )
    }
  })
  wall <- if (identical(run$status, "ok")) {
    total <- run$value$timings$elapsed_sec[run$value$timings$stage == "total"]
    if (length(total) == 0L) NA_real_ else as.numeric(total[[1L]])
  } else {
    NA_real_
  }
  list(run_id = run_id, mode = mode, repeat_id = repeat_id,
       status = run$status, error_message = run$error_message,
       result = run$value, wall_sec = wall)
}

fastkpc_fast_cuda_sepsets_identical <- function(a, b) {
  if (is.null(a) || is.null(b) || length(a) != length(b)) return(FALSE)
  for (i in seq_along(a)) {
    if (length(a[[i]]) != length(b[[i]])) return(FALSE)
    for (j in seq_along(a[[i]])) {
      left <- sort(as.integer(a[[i]][[j]]))
      right <- sort(as.integer(b[[i]][[j]]))
      if (!identical(left, right)) return(FALSE)
    }
  }
  TRUE
}

fastkpc_fast_cuda_skeleton_shd <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NA_integer_)
  a <- as.matrix(a)
  b <- as.matrix(b)
  if (!identical(dim(a), dim(b))) return(NA_integer_)
  as.integer(sum(abs(a[upper.tri(a)] - b[upper.tri(b)])))
}

fastkpc_fast_cuda_pmax_diff <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NA_real_)
  a <- as.matrix(a)
  b <- as.matrix(b)
  if (!identical(dim(a), dim(b))) return(NA_real_)
  max(abs(a - b))
}

fastkpc_fast_cuda_run_row <- function(entry, scenario, alpha) {
  result <- entry$result
  cfg <- result$config %||% list()
  skeleton <- result$skeleton %||% list()
  summary <- skeleton$scheduler_diagnostics$summary %||% list()
  cache <- skeleton$residual_cache %||% list()
  final_edges <- if (!is.null(skeleton$adjacency)) {
    as.integer(sum(skeleton$adjacency) / 2L)
  } else {
    NA_integer_
  }
  route_scheduler <- skeleton$scheduler %||% summary$scheduler %||% NA_character_
  precision_overlay_used <- identical(route_scheduler, "layer-precision") ||
    identical(route_scheduler, "r-precision")
  data.frame(
    run_id = entry$run_id,
    scenario_id = scenario$scenario_id,
    source = scenario$source,
    mode = entry$mode,
    repeat_id = as.integer(entry$repeat_id),
    status = entry$status,
    error_message = entry$error_message,
    n = as.integer(scenario$n),
    p = as.integer(scenario$p),
    max_conditioning_size = as.integer(scenario$max_conditioning_size),
    alpha = as.numeric(alpha),
    wall_ms = as.numeric(entry$wall_sec) * 1000,
    n_edgetests = as.integer(sum(skeleton$n.edgetests %||% NA_integer_,
                                 na.rm = TRUE)),
    final_edges = final_edges,
    route_scheduler = as.character(route_scheduler),
    route_data_plane = if (identical(entry$mode, "fast_cuda")) {
      "native-cuda-skeleton"
    } else if (identical(entry$mode, "fast_cpu")) {
      "native-cpu-skeleton"
    } else {
      "legacy-mgcv"
    },
    precision_overlay_used = isTRUE(precision_overlay_used),
    cuda_used = identical(entry$mode, "fast_cuda") &&
      identical(cfg$ci_backend %||% skeleton$ci_backend %||% "", "cuda-dcov"),
    mgcv_compatible_backend_used = identical(entry$mode, "legacy_mgcv"),
    cpu_fallback_count = as.integer(summary$cuda_residual_cpu_fallback_fits %||%
                                      0L),
    residual_requests = as.integer(cache$requests %||% 0L),
    residual_hits = as.integer(cache$hits %||% 0L),
    residual_misses = as.integer(cache$misses %||% 0L),
    residual_computations = as.integer(cache$computations %||% 0L),
    stringsAsFactors = FALSE
  )
}

fastkpc_fast_cuda_stage_rows <- function(entry, scenario) {
  result <- entry$result
  if (is.null(result)) {
    return(data.frame(
      run_id = entry$run_id, scenario_id = scenario$scenario_id,
      mode = entry$mode, repeat_id = as.integer(entry$repeat_id),
      stage = NA_character_, elapsed_ms = NA_real_,
      stringsAsFactors = FALSE
    ))
  }
  timings <- result$timings %||% NULL
  summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  if (is.null(timings)) {
    timings <- data.frame(stage = character(), elapsed_sec = numeric())
  }
  rows <- data.frame(
    run_id = entry$run_id,
    scenario_id = scenario$scenario_id,
    mode = entry$mode,
    repeat_id = as.integer(entry$repeat_id),
    stage = as.character(timings$stage),
    elapsed_ms = as.numeric(timings$elapsed_sec) * 1000,
    stringsAsFactors = FALSE
  )
  extra <- data.frame(
    run_id = entry$run_id,
    scenario_id = scenario$scenario_id,
    mode = entry$mode,
    repeat_id = as.integer(entry$repeat_id),
    stage = c("plan", "residual_prefetch", "ci_eval", "native_replay"),
    elapsed_ms = c(
      as.numeric(summary$plan_elapsed_sec %||% NA_real_) * 1000,
      as.numeric(summary$residual_prefetch_elapsed_sec %||% NA_real_) * 1000,
      as.numeric(summary$ci_eval_elapsed_sec %||% NA_real_) * 1000,
      as.numeric(summary$replay_elapsed_sec %||% NA_real_) * 1000
    ),
    stringsAsFactors = FALSE
  )
  rbind(rows, extra)
}

fastkpc_fast_cuda_graph_agreement_rows <- function(entries, scenario) {
  ok <- Filter(function(entry) identical(entry$status, "ok") &&
                 !is.null(entry$result), entries)
  if (length(ok) == 0L) {
    return(data.frame())
  }
  ref <- Filter(function(entry) identical(entry$mode, "fast_cpu"), ok)
  if (length(ref) == 0L) ref <- Filter(function(entry) identical(entry$mode, "legacy_mgcv"), ok)
  if (length(ref) == 0L) ref <- ok[1L]
  ref <- ref[[1L]]
  rows <- lapply(ok, function(entry) {
    a <- ref$result$skeleton
    b <- entry$result$skeleton
    data.frame(
      scenario_id = scenario$scenario_id,
      repeat_id = as.integer(entry$repeat_id),
      reference_mode = ref$mode,
      mode = entry$mode,
      adjacency_identical =
        isTRUE(fastkpc_fast_cuda_skeleton_shd(a$adjacency, b$adjacency) == 0L),
      skeleton_shd =
        fastkpc_fast_cuda_skeleton_shd(a$adjacency, b$adjacency),
      sepset_mismatch_vs_reference =
        !fastkpc_fast_cuda_sepsets_identical(a$sepsets, b$sepsets),
      pmax_max_abs_diff = fastkpc_fast_cuda_pmax_diff(a$pMax, b$pMax),
      n_edgetests_consistent =
        identical(as.integer(a$n.edgetests), as.integer(b$n.edgetests)),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_fast_cuda_speedup_summary <- function(runs) {
  ok <- runs[runs$status == "ok" & is.finite(runs$wall_ms), , drop = FALSE]
  if (nrow(ok) == 0L) return(data.frame())
  keys <- unique(ok[, c("scenario_id", "max_conditioning_size")])
  rows <- list()
  for (i in seq_len(nrow(keys))) {
    subset <- ok[ok$scenario_id == keys$scenario_id[[i]] &
                   ok$max_conditioning_size == keys$max_conditioning_size[[i]], ,
                 drop = FALSE]
    med <- tapply(subset$wall_ms, subset$mode, stats::median)
    metric <- function(name) {
      if (!name %in% names(med)) return(NA_real_)
      as.numeric(med[[name]])
    }
    fast_cuda <- metric("fast_cuda")
    fast_cpu <- metric("fast_cpu")
    legacy <- metric("legacy_mgcv")
    rows[[length(rows) + 1L]] <- data.frame(
      scenario_id = keys$scenario_id[[i]],
      max_conditioning_size = keys$max_conditioning_size[[i]],
      fast_cuda_median_ms = fast_cuda,
      fast_cpu_median_ms = fast_cpu,
      legacy_mgcv_median_ms = legacy,
      speedup_vs_fast_cpu =
        if (is.finite(fast_cuda) && fast_cuda > 0 && is.finite(fast_cpu)) {
          fast_cpu / fast_cuda
        } else {
          NA_real_
        },
      speedup_vs_legacy_mgcv =
        if (is.finite(fast_cuda) && fast_cuda > 0 && is.finite(legacy)) {
          legacy / fast_cuda
        } else {
          NA_real_
        },
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

fastkpc_fast_cuda_mode_summary <- function(runs) {
  ok <- runs[runs$status == "ok" & is.finite(runs$wall_ms), , drop = FALSE]
  if (nrow(ok) == 0L) return(data.frame())
  aggregate(
    wall_ms ~ mode,
    ok,
    function(x) c(median = stats::median(x), p90 =
                    as.numeric(stats::quantile(x, 0.9, names = FALSE)))
  )
}

fastkpc_fast_cuda_route_summary <- function(runs) {
  data.frame(
    mode = unique(runs$mode),
    route_violations = vapply(unique(runs$mode), function(mode) {
      subset <- runs[runs$mode == mode & runs$status == "ok", , drop = FALSE]
      if (nrow(subset) == 0L) return(0L)
      if (identical(mode, "fast_cuda")) {
        sum(subset$route_scheduler != "layer" |
              subset$precision_overlay_used %in% TRUE |
              subset$cpu_fallback_count > 0L)
      } else {
        0L
      }
    }, integer(1L)),
    stringsAsFactors = FALSE
  )
}

fastkpc_fast_cuda_write_summary <- function(summary, paths) {
  json <- c(
    "{",
    paste0('  "run_count": ', as.integer(summary$run_count[[1L]]), ","),
    paste0('  "ok_count": ', as.integer(summary$ok_count[[1L]]), ","),
    paste0('  "skipped_count": ', as.integer(summary$skipped_count[[1L]]), ","),
    paste0('  "error_count": ', as.integer(summary$error_count[[1L]]), ","),
    paste0('  "fast_cuda_route_violations": ',
           as.integer(summary$fast_cuda_route_violations[[1L]]), ","),
    paste0('  "fast_cuda_median_speedup_vs_fast_cpu": ',
           fastkpc_fast_cuda_json_number(
             summary$fast_cuda_median_speedup_vs_fast_cpu[[1L]]
           )),
    "}"
  )
  writeLines(json, paths$summary_json)
  md <- c(
    "# Fast CUDA Performance Baseline",
    "",
    paste0("- Runs: ", summary$run_count[[1L]]),
    paste0("- OK: ", summary$ok_count[[1L]]),
    paste0("- Skipped: ", summary$skipped_count[[1L]]),
    paste0("- Errors: ", summary$error_count[[1L]]),
    paste0("- Fast CUDA route violations: ",
           summary$fast_cuda_route_violations[[1L]]),
    paste0("- Median speedup vs fast CPU: ",
           fastkpc_fast_cuda_json_number(
             summary$fast_cuda_median_speedup_vs_fast_cpu[[1L]]
           ))
  )
  writeLines(md, paths$summary_md)
}

fastkpc_fast_cuda_json_number <- function(value) {
  value <- as.numeric(value)[1L]
  if (!is.finite(value)) return("null")
  format(signif(value, 8), scientific = FALSE, trim = TRUE)
}

fastkpc_run_fast_cuda_performance_baseline <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_performance_baseline"),
    scenarios = NULL,
    modes = fastkpc_fast_cuda_baseline_modes(),
    repeats = 5L,
    warmup = TRUE,
    alpha = 0.2,
    ci_method = "dcc.gamma",
    full_grid = FALSE,
    real_data_path = Sys.getenv("FASTKPC_FAST_CUDA_REAL_DATA", ""),
    real_n = 100L,
    real_p = 12L,
    run_native_cuda = identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1"),
    legacy_max_n = 300L,
    legacy_max_p = 12L,
    legacy_max_level = 2L) {
  modes <- match.arg(modes, fastkpc_fast_cuda_baseline_modes(),
                     several.ok = TRUE)
  if (is.null(scenarios)) {
    scenarios <- fastkpc_fast_cuda_baseline_scenarios(
      full_grid = full_grid,
      real_data_path = real_data_path,
      real_n = real_n,
      real_p = real_p
    )
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  runs <- list()
  stages <- list()
  graph <- list()
  if (isTRUE(warmup) && "fast_cuda" %in% modes && isTRUE(run_native_cuda)) {
    warm <- scenarios[[1L]]
    invisible(fastkpc_fast_cuda_run_mode(
      warm, "fast_cuda", 0L, alpha, ci_method, run_native_cuda,
      legacy_max_n, legacy_max_p, legacy_max_level
    ))
  }

  for (scenario in scenarios) {
    for (repeat_id in seq_len(as.integer(repeats))) {
      entries <- list()
      for (mode in modes) {
        entry <- fastkpc_fast_cuda_run_mode(
          scenario, mode, repeat_id, alpha, ci_method, run_native_cuda,
          legacy_max_n, legacy_max_p, legacy_max_level
        )
        entries[[length(entries) + 1L]] <- entry
        runs[[length(runs) + 1L]] <-
          fastkpc_fast_cuda_run_row(entry, scenario, alpha)
        stages[[length(stages) + 1L]] <-
          fastkpc_fast_cuda_stage_rows(entry, scenario)
      }
      graph[[length(graph) + 1L]] <-
        fastkpc_fast_cuda_graph_agreement_rows(entries, scenario)
    }
  }
  runs_df <- do.call(rbind, runs)
  stages_df <- do.call(rbind, stages)
  graph_df <- do.call(rbind, graph)
  speedup <- fastkpc_fast_cuda_speedup_summary(runs_df)
  mode_summary <- fastkpc_fast_cuda_mode_summary(runs_df)
  route_summary <- fastkpc_fast_cuda_route_summary(runs_df)
  fast_cuda_speedups <- speedup$speedup_vs_fast_cpu[
    is.finite(speedup$speedup_vs_fast_cpu)
  ]
  summary <- data.frame(
    run_count = as.integer(nrow(runs_df)),
    ok_count = as.integer(sum(runs_df$status == "ok")),
    skipped_count = as.integer(sum(runs_df$status == "skipped")),
    error_count = as.integer(sum(runs_df$status == "error")),
    fast_cuda_route_violations =
      as.integer(sum(route_summary$route_violations[
        route_summary$mode == "fast_cuda"
      ] %||% 0L)),
    fast_cuda_median_speedup_vs_fast_cpu =
      if (length(fast_cuda_speedups) == 0L) NA_real_ else
        stats::median(fast_cuda_speedups),
    stringsAsFactors = FALSE
  )

  paths <- list(
    runs_csv = file.path(output_dir, "runs.csv"),
    mode_summary_csv = file.path(output_dir, "mode_summary.csv"),
    stage_timing_csv = file.path(output_dir, "stage_timing.csv"),
    graph_agreement_csv = file.path(output_dir, "graph_agreement.csv"),
    route_summary_csv = file.path(output_dir, "route_summary.csv"),
    speedup_summary_csv = file.path(output_dir, "speedup_summary.csv"),
    summary_csv = file.path(output_dir, "fast_cuda_baseline_summary.csv"),
    summary_json = file.path(output_dir, "fast_cuda_baseline_summary.json"),
    summary_md = file.path(output_dir, "fast_cuda_baseline_summary.md")
  )
  utils::write.csv(runs_df, paths$runs_csv, row.names = FALSE)
  utils::write.csv(mode_summary, paths$mode_summary_csv, row.names = FALSE)
  utils::write.csv(stages_df, paths$stage_timing_csv, row.names = FALSE)
  utils::write.csv(graph_df, paths$graph_agreement_csv, row.names = FALSE)
  utils::write.csv(route_summary, paths$route_summary_csv, row.names = FALSE)
  utils::write.csv(speedup, paths$speedup_summary_csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  fastkpc_fast_cuda_write_summary(summary, paths)

  list(
    runs = runs_df,
    mode_summary = mode_summary,
    stage_timing = stages_df,
    graph_agreement = graph_df,
    route_summary = route_summary,
    speedup_summary = speedup,
    summary = summary,
    paths = paths
  )
}
