if (!exists("fastkpc_run_dcc_gamma_cuda_parity", mode = "function")) {
  source("fastkpc/R/dcc_gamma_cuda_parity.R")
}
if (!exists("fastkpc_run_skeleton_ptable_parity", mode = "function")) {
  source("fastkpc/R/skeleton_ptable_parity.R")
}
if (!exists("fastkpc_run_fast_cuda_conditional_ci_parity", mode = "function")) {
  source("fastkpc/R/fast_cuda_conditional_ci_parity.R")
}
if (!exists("fast_kpc", mode = "function")) {
  source("fastkpc/R/fast_kpc.R")
}

fastkpc_fast_cuda_route_guard <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_data_plane_validation",
                           "route_guard")) {
  set.seed(8501)
  data <- matrix(stats::rnorm(72 * 5), 72, 5)
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  native_fast <- fast_kpc(
    data,
    alpha = 0.2,
    max_conditioning_size = 1,
    engine = "cuda",
    precision = "fast",
    graph_stage = "skeleton",
    fastspline_params = params,
    seed = 8501
  )
  overlay <- fast_kpc(
    data,
    alpha = 0.2,
    max_conditioning_size = 1,
    engine = "cuda",
    precision = "fast",
    graph_stage = "skeleton",
    fastspline_params = params,
    precision_executors = fastkpc_default_precision_executors(),
    precision_trace_level = "summary",
    seed = 8501
  )
  row <- data.frame(
    default_scheduler = as.character(native_fast$skeleton$scheduler),
    default_precision_execution_status =
      as.character(native_fast$config$precision_execution_status),
    default_backend_executed = as.character(native_fast$config$backend_executed),
    default_ci_backend = as.character(native_fast$config$ci_backend),
    default_precision_overlay_used =
      identical(native_fast$skeleton$scheduler, "layer-precision"),
    default_precision_trace_materialized =
      is.data.frame(native_fast$skeleton$precision_trace) ||
        is.data.frame(native_fast$diagnostics$precision_trace),
    explicit_overlay_scheduler = as.character(overlay$skeleton$scheduler),
    explicit_overlay_status =
      as.character(overlay$config$precision_execution_status),
    adjacency_identical =
      identical(native_fast$skeleton$adjacency, overlay$skeleton$adjacency),
    n_edgetests_identical =
      identical(native_fast$skeleton$n.edgetests, overlay$skeleton$n.edgetests),
    pmax_max_abs_diff = max(abs(native_fast$skeleton$pMax -
                                  overlay$skeleton$pMax)),
    route_violations = as.integer(!identical(native_fast$skeleton$scheduler, "layer")) +
      as.integer(!identical(native_fast$config$precision_execution_status, "executed")) +
      as.integer(is.data.frame(native_fast$skeleton$precision_trace)) +
      as.integer(is.data.frame(native_fast$diagnostics$precision_trace)),
    stringsAsFactors = FALSE
  )
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(output_dir, "fast_cuda_route_guard.csv")
  utils::write.csv(row, path, row.names = FALSE)
  list(row = row, path = path)
}

fastkpc_fast_cuda_e2e_graph_agreement <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_data_plane_validation",
                           "e2e_graph_agreement")) {
  set.seed(8502)
  n <- 90
  z1 <- stats::runif(n, -2, 2)
  z2 <- stats::runif(n, -2, 2)
  data <- cbind(
    x1 = sin(z1) + stats::rnorm(n, sd = 0.12),
    x2 = cos(z1) + stats::rnorm(n, sd = 0.12),
    x3 = z1 * z2 + stats::rnorm(n, sd = 0.12),
    x4 = sin(z2) + stats::rnorm(n, sd = 0.12),
    x5 = stats::rnorm(n)
  )
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  fast_cpu <- fast_kpc(
    data,
    alpha = 0.2,
    max_conditioning_size = 1,
    engine = "cpu",
    precision = "legacy",
    graph_stage = "skeleton",
    residual_backend = "fastSpline",
    fastspline_params = params,
    seed = 8502
  )
  fast_cuda <- fast_kpc(
    data,
    alpha = 0.2,
    max_conditioning_size = 1,
    engine = "cuda",
    precision = "fast",
    graph_stage = "skeleton",
    fastspline_params = params,
    seed = 8502
  )
  sepsets_identical <- TRUE
  for (i in seq_along(fast_cpu$skeleton$sepsets)) {
    for (j in seq_along(fast_cpu$skeleton$sepsets[[i]])) {
      left <- sort(as.integer(fast_cpu$skeleton$sepsets[[i]][[j]]))
      right <- sort(as.integer(fast_cuda$skeleton$sepsets[[i]][[j]]))
      if (!identical(left, right)) sepsets_identical <- FALSE
    }
  }
  row <- data.frame(
    n = nrow(data),
    p = ncol(data),
    max_conditioning_size = 1L,
    cpu_scheduler = as.character(fast_cpu$skeleton$scheduler %||% "legacy"),
    cuda_scheduler = as.character(fast_cuda$skeleton$scheduler),
    adjacency_identical =
      identical(fast_cpu$skeleton$adjacency, fast_cuda$skeleton$adjacency),
    sepsets_identical = isTRUE(sepsets_identical),
    n_edgetests_identical =
      identical(fast_cpu$skeleton$n.edgetests, fast_cuda$skeleton$n.edgetests),
    pmax_max_abs_diff =
      max(abs(fast_cpu$skeleton$pMax - fast_cuda$skeleton$pMax)),
    cpu_n_edgetests = sum(fast_cpu$skeleton$n.edgetests),
    cuda_n_edgetests = sum(fast_cuda$skeleton$n.edgetests),
    stringsAsFactors = FALSE
  )
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(output_dir, "fast_cuda_e2e_graph_agreement.csv")
  utils::write.csv(row, path, row.names = FALSE)
  list(row = row, path = path)
}

fastkpc_fast_cuda_small_benchmark <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_data_plane_validation",
                           "benchmark"),
    repeats = 1L) {
  set.seed(8503)
  n <- 120
  p <- 8
  data <- matrix(stats::rnorm(n * p), n, p)
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  rows <- list()
  run_one <- function(mode, repeat_id) {
    start <- proc.time()[["elapsed"]]
    value <- switch(
      mode,
      fast_cpu = fast_kpc(
        data, alpha = 0.2, max_conditioning_size = 1,
        engine = "cpu", precision = "legacy", graph_stage = "skeleton",
        residual_backend = "fastSpline", fastspline_params = params,
        seed = 8503 + repeat_id
      ),
      fast_cuda = fast_kpc(
        data, alpha = 0.2, max_conditioning_size = 1,
        engine = "cuda", precision = "fast", graph_stage = "skeleton",
        fastspline_params = params, seed = 8503 + repeat_id
      )
    )
    elapsed <- proc.time()[["elapsed"]] - start
    data.frame(
      mode = mode,
      repeat_id = as.integer(repeat_id),
      wall_sec = as.numeric(elapsed),
      scheduler = as.character(value$skeleton$scheduler %||% "legacy"),
      n_edgetests = as.integer(sum(value$skeleton$n.edgetests)),
      edge_count = as.integer(sum(value$skeleton$adjacency) / 2L),
      stringsAsFactors = FALSE
    )
  }
  for (repeat_id in seq_len(as.integer(repeats))) {
    rows[[length(rows) + 1L]] <- run_one("fast_cpu", repeat_id)
    rows[[length(rows) + 1L]] <- run_one("fast_cuda", repeat_id)
  }
  out <- do.call(rbind, rows)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(output_dir, "fast_cuda_benchmark.csv")
  utils::write.csv(out, path, row.names = FALSE)
  list(rows = out, path = path)
}

fastkpc_fast_cuda_json_number <- function(value) {
  value <- as.numeric(value)[1L]
  if (!is.finite(value)) return("null")
  format(signif(value, 8), scientific = FALSE, trim = TRUE)
}

fastkpc_write_fast_cuda_data_plane_summary <- function(summary, paths) {
  json <- c(
    "{",
    paste0('  "dcc_gamma_max_p_abs_diff": ',
           fastkpc_fast_cuda_json_number(summary$dcc_gamma_max_p_abs_diff), ","),
    paste0('  "dcc_gamma_decision_flips": ',
           as.integer(summary$dcc_gamma_decision_flips), ","),
    paste0('  "skeleton_ptable_exact": ',
           if (isTRUE(summary$skeleton_ptable_exact)) "true" else "false", ","),
    paste0('  "route_violations": ', as.integer(summary$route_violations), ","),
    paste0('  "conditional_ci_max_p_abs_diff": ',
           fastkpc_fast_cuda_json_number(summary$conditional_ci_max_p_abs_diff), ","),
    paste0('  "conditional_ci_decision_flips": ',
           as.integer(summary$conditional_ci_decision_flips), ","),
    paste0('  "e2e_graph_exact": ',
           if (isTRUE(summary$e2e_graph_exact)) "true" else "false"),
    "}"
  )
  writeLines(json, paths$summary_json)
  md <- c(
    "# Fast CUDA Data Plane Validation",
    "",
    "## Gates",
    "",
    paste0("- dcc.gamma max p abs diff: ",
           fastkpc_fast_cuda_json_number(summary$dcc_gamma_max_p_abs_diff)),
    paste0("- dcc.gamma decision flips: ",
           summary$dcc_gamma_decision_flips),
    paste0("- skeleton p-table exact: ", summary$skeleton_ptable_exact),
    paste0("- route violations: ", summary$route_violations),
    paste0("- conditional CI max p abs diff: ",
           fastkpc_fast_cuda_json_number(summary$conditional_ci_max_p_abs_diff)),
    paste0("- conditional CI decision flips: ",
           summary$conditional_ci_decision_flips),
    paste0("- e2e graph exact: ", summary$e2e_graph_exact)
  )
  writeLines(md, paths$summary_md)
}

fastkpc_run_fast_cuda_data_plane_validation <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_data_plane_validation"),
    include_benchmark = FALSE,
    benchmark_repeats = 1L) {
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for fast CUDA data plane validation", call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dcc <- fastkpc_run_dcc_gamma_cuda_parity(
    output_dir = file.path(output_dir, "dcc_gamma")
  )
  ptable <- fastkpc_run_skeleton_ptable_parity(
    output_dir = file.path(output_dir, "skeleton_ptable")
  )
  route <- fastkpc_fast_cuda_route_guard(
    output_dir = file.path(output_dir, "route_guard")
  )
  conditional <- fastkpc_run_fast_cuda_conditional_ci_parity(
    output_dir = file.path(output_dir, "conditional_ci")
  )
  e2e <- fastkpc_fast_cuda_e2e_graph_agreement(
    output_dir = file.path(output_dir, "e2e_graph")
  )
  benchmark <- NULL
  if (isTRUE(include_benchmark)) {
    benchmark <- fastkpc_fast_cuda_small_benchmark(
      output_dir = file.path(output_dir, "benchmark"),
      repeats = benchmark_repeats
    )
  }

  summary <- data.frame(
    dcc_gamma_max_p_abs_diff = dcc$summary$max_p_abs_diff[[1L]],
    dcc_gamma_decision_flips = dcc$summary$decision_flips[[1L]],
    skeleton_ptable_exact =
      isTRUE(ptable$summary$adjacency_identical[[1L]]) &&
        isTRUE(ptable$summary$sepsets_identical[[1L]]) &&
        isTRUE(ptable$summary$n_edgetests_identical[[1L]]) &&
        ptable$summary$pmax_max_abs_diff[[1L]] < 1e-12,
    route_violations = route$row$route_violations[[1L]],
    conditional_ci_max_p_abs_diff = conditional$summary$max_p_abs_diff[[1L]],
    conditional_ci_decision_flips = conditional$summary$decision_flips[[1L]],
    conditional_ci_fallback_count = conditional$summary$fallback_count[[1L]],
    e2e_graph_exact =
      isTRUE(e2e$row$adjacency_identical[[1L]]) &&
        isTRUE(e2e$row$sepsets_identical[[1L]]) &&
        isTRUE(e2e$row$n_edgetests_identical[[1L]]) &&
        e2e$row$pmax_max_abs_diff[[1L]] < 1e-7,
    benchmark_included = isTRUE(include_benchmark),
    stringsAsFactors = FALSE
  )
  paths <- list(
    summary_csv = file.path(output_dir, "fast_cuda_data_plane_summary.csv"),
    summary_json = file.path(output_dir, "fast_cuda_data_plane_summary.json"),
    summary_md = file.path(output_dir, "fast_cuda_data_plane_summary.md")
  )
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  fastkpc_write_fast_cuda_data_plane_summary(summary[1L, , drop = FALSE],
                                             paths)

  list(
    summary = summary,
    paths = paths,
    dcc_gamma = dcc,
    skeleton_ptable = ptable,
    route_guard = route,
    conditional_ci = conditional,
    e2e_graph = e2e,
    benchmark = benchmark
  )
}
