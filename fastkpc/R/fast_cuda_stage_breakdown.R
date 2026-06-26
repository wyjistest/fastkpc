source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

fastkpc_stage_breakdown_synthetic_data <- function(n, p, seed) {
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

fastkpc_stage_breakdown_default_scenarios <- function() {
  specs <- data.frame(
    scenario_id = c("breakdown-n100-p12-m2",
                    "breakdown-n300-p12-m2",
                    "breakdown-n1000-p12-m2"),
    n = c(100L, 300L, 1000L),
    p = c(12L, 12L, 12L),
    max_conditioning_size = c(2L, 2L, 2L),
    seed = c(8701L, 8702L, 8703L),
    stringsAsFactors = FALSE
  )
  lapply(seq_len(nrow(specs)), function(i) {
    row <- specs[i, ]
    list(
      scenario_id = row$scenario_id,
      n = as.integer(row$n),
      p = as.integer(row$p),
      max_conditioning_size = as.integer(row$max_conditioning_size),
      seed = as.integer(row$seed),
      data = fastkpc_stage_breakdown_synthetic_data(
        as.integer(row$n), as.integer(row$p), as.integer(row$seed)
      )
    )
  })
}

fastkpc_stage_breakdown_seconds <- function(value) {
  value <- as.numeric(value %||% NA_real_)[1L]
  if (!is.finite(value)) return(NA_real_)
  value
}

fastkpc_stage_breakdown_rows <- function(result, scenario, repeat_id) {
  summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  total <- fastkpc_stage_breakdown_seconds(summary$total_elapsed_sec)
  row <- function(stage, elapsed_sec, group = "skeleton") {
    elapsed_ms <- elapsed_sec * 1000
    data.frame(
      scenario_id = scenario$scenario_id,
      repeat_id = as.integer(repeat_id),
      n = as.integer(scenario$n),
      p = as.integer(scenario$p),
      max_conditioning_size = as.integer(scenario$max_conditioning_size),
      stage = stage,
      stage_group = group,
      elapsed_ms = elapsed_ms,
      share_of_skeleton = if (is.finite(total) && total > 0) {
        elapsed_sec / total
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }
  rows <- list(
    row("skeleton_total", total, "skeleton"),
    row("plan", fastkpc_stage_breakdown_seconds(summary$plan_elapsed_sec),
        "control"),
    row("fastspline_residual_prefetch",
        fastkpc_stage_breakdown_seconds(summary$residual_prefetch_elapsed_sec),
        "residual"),
    row("residual_grouping",
        fastkpc_stage_breakdown_seconds(summary$residual_grouping_sec),
        "residual"),
    row("residual_host_pack",
        fastkpc_stage_breakdown_seconds(summary$residual_host_pack_sec),
        "residual"),
    row("residual_alloc",
        fastkpc_stage_breakdown_seconds(summary$residual_alloc_sec),
        "residual"),
    row("residual_h2d",
        fastkpc_stage_breakdown_seconds(summary$residual_h2d_sec),
        "residual"),
    row("residual_xtx_xty",
        fastkpc_stage_breakdown_seconds(summary$residual_xtx_xty_sec),
        "residual"),
    row("residual_pointer_setup",
        fastkpc_stage_breakdown_seconds(summary$residual_pointer_setup_sec),
        "residual"),
    row("residual_active_copy",
        fastkpc_stage_breakdown_seconds(summary$residual_active_copy_sec),
        "residual"),
    row("residual_build_system",
        fastkpc_stage_breakdown_seconds(summary$residual_build_system_sec),
        "residual"),
    row("residual_factor_solve",
        fastkpc_stage_breakdown_seconds(summary$residual_factor_solve_sec),
        "residual"),
    row("residual_summary_kernel",
        fastkpc_stage_breakdown_seconds(summary$residual_summary_sec),
        "residual"),
    row("residual_d2h",
        fastkpc_stage_breakdown_seconds(summary$residual_d2h_sec),
        "residual"),
    row("residual_host_select",
        fastkpc_stage_breakdown_seconds(summary$residual_host_select_sec),
        "residual"),
    row("residual_free",
        fastkpc_stage_breakdown_seconds(summary$residual_free_sec),
        "residual"),
    row("residual_true_batch_total",
        fastkpc_stage_breakdown_seconds(
          summary$residual_true_batch_total_sec
        ),
        "residual"),
    row("ci_eval_total",
        fastkpc_stage_breakdown_seconds(summary$ci_eval_elapsed_sec),
        "ci"),
    row("native_replay",
        fastkpc_stage_breakdown_seconds(summary$replay_elapsed_sec),
        "control"),
    row("dcov_alloc", fastkpc_stage_breakdown_seconds(summary$dcov_alloc_sec),
        "dcov"),
    row("dcov_h2d", fastkpc_stage_breakdown_seconds(summary$dcov_h2d_sec),
        "dcov"),
    row("dcov_memset", fastkpc_stage_breakdown_seconds(summary$dcov_memset_sec),
        "dcov"),
    row("dcov_rowsum_distance",
        fastkpc_stage_breakdown_seconds(summary$dcov_rowsum_sec),
        "dcov"),
    row("dcov_totals_d2h",
        fastkpc_stage_breakdown_seconds(summary$dcov_totals_d2h_sec),
        "dcov"),
    row("dcov_fused_center_reduce",
        fastkpc_stage_breakdown_seconds(summary$dcov_reduce_sec),
        "dcov"),
    row("dcov_scalars_d2h",
        fastkpc_stage_breakdown_seconds(summary$dcov_scalars_d2h_sec),
        "dcov"),
    row("dcov_host_gamma_pvalue",
        fastkpc_stage_breakdown_seconds(summary$dcov_host_scalar_sec),
        "dcov"),
    row("dcov_free", fastkpc_stage_breakdown_seconds(summary$dcov_free_sec),
        "dcov"),
    row("dcov_measured_total",
        fastkpc_stage_breakdown_seconds(summary$dcov_total_sec),
        "dcov")
  )
  do.call(rbind, rows)
}

fastkpc_stage_breakdown_run_row <- function(result, scenario, repeat_id) {
  summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  data.frame(
    scenario_id = scenario$scenario_id,
    repeat_id = as.integer(repeat_id),
    n = as.integer(scenario$n),
    p = as.integer(scenario$p),
    max_conditioning_size = as.integer(scenario$max_conditioning_size),
    scheduler = as.character(result$skeleton$scheduler),
    route_data_plane = "native-cuda-skeleton",
    precision_overlay_used =
      identical(result$skeleton$scheduler, "layer-precision") ||
        identical(result$skeleton$scheduler, "r-precision"),
    cpu_fallback_count =
      as.integer(summary$cuda_residual_cpu_fallback_fits %||% 0L),
    n_edgetests = as.integer(sum(result$skeleton$n.edgetests)),
    final_edges = as.integer(sum(result$skeleton$adjacency) / 2L),
    dcov_batches = as.integer(summary$dcov_batches %||% 0L),
    dcov_chunks = as.integer(summary$dcov_chunks %||% 0L),
    dcov_batch_size_used = as.integer(summary$dcov_batch_size_used %||% 0L),
    dcov_max_chunk_batch =
      as.integer(summary$dcov_max_chunk_batch %||% 0L),
    residual_batches = as.integer(summary$residual_batches %||% 0L),
    unique_residual_requests =
      as.integer(summary$unique_residual_requests %||% 0L),
    cuda_residual_true_batched_fits =
      as.integer(summary$cuda_residual_true_batched_fits %||% 0L),
    cuda_residual_single_fit_calls =
      as.integer(summary$cuda_residual_single_fit_calls %||% 0L),
    cuda_residual_unique_designs =
      as.integer(summary$cuda_residual_unique_designs %||% 0L),
    stringsAsFactors = FALSE
  )
}

fastkpc_stage_breakdown_summary <- function(rows) {
  keys <- unique(rows[, c("stage", "stage_group"), drop = FALSE])
  out <- lapply(seq_len(nrow(keys)), function(i) {
    stage <- keys$stage[[i]]
    group <- keys$stage_group[[i]]
    elapsed <- rows$elapsed_ms[rows$stage == stage &
                                 rows$stage_group == group]
    finite <- elapsed[is.finite(elapsed)]
    data.frame(
      stage = stage,
      stage_group = group,
      run_count = as.integer(length(elapsed)),
      finite_count = as.integer(length(finite)),
      median_ms = if (length(finite)) stats::median(finite) else NA_real_,
      p90_ms = if (length(finite)) {
        as.numeric(stats::quantile(finite, 0.9, names = FALSE))
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

fastkpc_run_fast_cuda_stage_breakdown <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_stage_breakdown"),
    scenarios = fastkpc_stage_breakdown_default_scenarios(),
    repeats = 3L,
    alpha = 0.2) {
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for fast CUDA stage breakdown", call. = FALSE)
  }
  rows <- list()
  runs <- list()
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  for (scenario in scenarios) {
    for (repeat_id in seq_len(as.integer(repeats))) {
      result <- fast_kpc(
        scenario$data,
        alpha = alpha,
        max_conditioning_size = scenario$max_conditioning_size,
        engine = "cuda",
        precision = "fast",
        graph_stage = "skeleton",
        fastspline_params = params,
        benchmark = TRUE,
        seed = scenario$seed + repeat_id
      )
      rows[[length(rows) + 1L]] <-
        fastkpc_stage_breakdown_rows(result, scenario, repeat_id)
      runs[[length(runs) + 1L]] <-
        fastkpc_stage_breakdown_run_row(result, scenario, repeat_id)
    }
  }
  breakdown <- do.call(rbind, rows)
  run_summary <- do.call(rbind, runs)
  summary <- fastkpc_stage_breakdown_summary(breakdown)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    breakdown_csv = file.path(output_dir, "fast_cuda_stage_breakdown.csv"),
    runs_csv = file.path(output_dir, "fast_cuda_stage_breakdown_runs.csv"),
    summary_csv = file.path(output_dir,
                            "fast_cuda_stage_breakdown_summary.csv"),
    summary_md = file.path(output_dir,
                           "fast_cuda_stage_breakdown_summary.md")
  )
  utils::write.csv(breakdown, paths$breakdown_csv, row.names = FALSE)
  utils::write.csv(run_summary, paths$runs_csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  md <- c(
    "# Fast CUDA Stage Breakdown",
    "",
    paste0("- Runs: ", nrow(run_summary)),
    paste0("- Route violations: ",
           sum(run_summary$scheduler != "layer" |
                 run_summary$precision_overlay_used %in% TRUE |
                 run_summary$cpu_fallback_count > 0L)),
    paste0("- CSV: `", basename(paths$breakdown_csv), "`")
  )
  writeLines(md, paths$summary_md)
  list(
    breakdown = breakdown,
    runs = run_summary,
    summary = summary,
    paths = paths
  )
}
