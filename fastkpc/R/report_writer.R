source("fastkpc/R/validation_campaign.R")

fastkpc_write_csv <- function(x, path) {
  if (!is.data.frame(x)) x <- as.data.frame(x, stringsAsFactors = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

fastkpc_markdown_table <- function(df, max_rows = 12) {
  if (!is.data.frame(df)) df <- as.data.frame(df, stringsAsFactors = FALSE)
  if (ncol(df) == 0L) return("_No columns._")
  if (nrow(df) == 0L) {
    header <- paste(names(df), collapse = " | ")
    divider <- paste(rep("---", ncol(df)), collapse = " | ")
    return(paste0("| ", header, " |\n| ", divider, " |\n"))
  }
  shown <- utils::head(df, max_rows)
  shown[] <- lapply(shown, function(col) {
    col <- as.character(col)
    col[is.na(col)] <- ""
    gsub("\n", " ", col, fixed = TRUE)
  })
  header <- paste(names(shown), collapse = " | ")
  divider <- paste(rep("---", ncol(shown)), collapse = " | ")
  rows <- apply(shown, 1L, function(row) paste(row, collapse = " | "))
  suffix <- if (nrow(df) > max_rows) {
    paste0("\n\n_Showing ", max_rows, " of ", nrow(df), " rows._")
  } else {
    ""
  }
  paste0("| ", header, " |\n| ", divider, " |\n| ",
         paste(rows, collapse = " |\n| "), " |\n", suffix)
}

fastkpc_format_config <- function(config) {
  lines <- character()
  for (name in names(config)) {
    value <- config[[name]]
    if (length(value) == 0L) value <- ""
    if (length(value) > 1L) value <- paste(value, collapse = ", ")
    lines <- c(lines, paste0("- `", name, "`: ", as.character(value)))
  }
  paste(lines, collapse = "\n")
}

fastkpc_campaign_markdown <- function(campaign) {
  summary <- campaign$summary
  summary_df <- data.frame(
    metric = names(summary),
    value = vapply(summary, function(x) paste(as.character(x), collapse = ", "),
                   character(1)),
    stringsAsFactors = FALSE
  )
  paste(
    "# fastkpc Validation Campaign",
    "",
    "## Configuration",
    fastkpc_format_config(campaign$config),
    "",
    "## Summary",
    fastkpc_markdown_table(summary_df),
    "",
    "## CPU vs CUDA",
    fastkpc_markdown_table(campaign$cpu_cuda),
    "",
    "## Linear vs fastSpline",
    fastkpc_markdown_table(campaign$linear_fastspline),
    "",
    "## Residual Device",
    fastkpc_markdown_table(campaign$residual_device_diffs %||%
                             fastkpc_empty_df(c("scenario", "seed", "n",
                                                 "engine", "residual_backend",
                                                 "status"))),
    "",
    "## Orientation Device",
    fastkpc_markdown_table(campaign$orientation_device_diffs %||%
                             fastkpc_empty_df(c("scenario", "seed", "n",
                                                 "engine", "residual_backend",
                                                 "residual_device",
                                                 "scheduler", "status"))),
    "",
    "## Orientation Device Diagnostics",
    fastkpc_markdown_table(campaign$orientation_device_diagnostics %||%
                             fastkpc_empty_df(c("run_id",
                                                 "orientation_residual_device",
                                                 "regrvonps_cuda_calls",
                                                 "orientation_dcov_batches",
                                                 "orientation_dcov_pairs"))),
    "",
    "## Scheduler",
    fastkpc_markdown_table(campaign$scheduler_diffs %||%
                             fastkpc_empty_df(c("scenario", "seed", "n",
                                                 "engine", "residual_backend",
                                                 "residual_device",
                                                 "left_scheduler",
                                                 "right_scheduler",
                                                 "status"))),
    "",
    "## CI Method",
    fastkpc_markdown_table(campaign$ci_method_diffs %||%
                             fastkpc_empty_df(c("scenario", "seed", "n",
                                                 "engine", "left_ci_method",
                                                 "right_ci_method", "status"))),
    "",
    "## CI Method Diagnostics",
    fastkpc_markdown_table(campaign$ci_method_diagnostics %||%
                             fastkpc_empty_df(c("run_id", "ci_method",
                                                 "ci_backend",
                                                 "ci_hsic_gamma_tests",
                                                 "ci_hsic_perm_tests"))),
    "",
    "## HSIC CUDA Backend",
    fastkpc_markdown_table(campaign$hsic_cuda_backend_diagnostics %||%
                             fastkpc_empty_df(c("run_id", "ci_method",
                                                 "ci_backend",
                                                 "cuda_hsic_used",
                                                 "ci_hsic_cuda_batches",
                                                 "ci_hsic_cuda_pairs"))),
    "",
    "## HSIC CUDA Fallbacks",
    fastkpc_markdown_table(campaign$hsic_cuda_cpu_fallbacks %||%
                             fastkpc_empty_df(c("run_id", "ci_method",
                                                 "ci_backend_reason",
                                                 "ci_hsic_cuda_fallback_tests"))),
    "",
    "## HSIC CUDA Performance",
    fastkpc_markdown_table(campaign$hsic_cuda_perf %||%
                             fastkpc_empty_df(c("run_id", "ci_method",
                                                 "ci_backend",
                                                 "elapsed_total_sec"))),
    "",
    "## Scheduler Levels",
    fastkpc_markdown_table(campaign$scheduler_levels %||%
                             fastkpc_empty_df(c("run_id", "level",
                                                 "tasks_planned",
                                                 "tests_replayed"))),
    "",
    "## Scheduler Batches",
    fastkpc_markdown_table(campaign$scheduler_batches %||%
                             fastkpc_empty_df(c("run_id", "level", "kind",
                                                 "task_count", "status"))),
    "",
    "## Scheduler Residuals",
    fastkpc_markdown_table(campaign$scheduler_residuals %||%
                             fastkpc_empty_df(c("run_id", "level", "target",
                                                 "residual_device",
                                                 "materialized"))),
    "",
    "## True-Batched CUDA fastSpline Residuals",
    fastkpc_markdown_table(campaign$true_batched_residuals %||%
                             fastkpc_empty_df(c("run_id", "scenario", "engine",
                                                 "residual_backend",
                                                 "residual_device",
                                                 "scheduler",
                                                 "cuda_residual_true_batched_groups",
                                                 "cuda_residual_true_batched_fits",
                                                 "cuda_residual_single_fit_calls",
                                                 "cuda_residual_cpu_fallback_fits",
                                                 "status"))),
    "",
    "## Legacy Diagnostics",
    fastkpc_markdown_table(campaign$legacy),
    "",
    "## Timings",
    fastkpc_markdown_table(campaign$timings),
    "",
    "## Cache",
    fastkpc_markdown_table(campaign$cache),
    "",
    "## Errors",
    fastkpc_markdown_table(campaign$errors),
    "",
    "## Reproduction",
    paste0("- R version: ", as.character(getRversion())),
    "- Re-run with `run_fastkpc_validation_campaign()` using the configuration above.",
    "- Full campaign object is stored in `campaign.rds`.",
    "",
    sep = "\n"
  )
}

write_fastkpc_validation_report <- function(campaign, output_dir) {
  if (missing(output_dir) || !nzchar(output_dir)) {
    stop("output_dir is required", call. = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(output_dir)) {
    stop("failed to create output_dir: ", output_dir, call. = FALSE)
  }

  artifacts <- list(
    summary_md = file.path(output_dir, "summary.md"),
    runs_csv = file.path(output_dir, "runs.csv"),
    graph_metrics_csv = file.path(output_dir, "graph_metrics.csv"),
    pairwise_diffs_csv = file.path(output_dir, "pairwise_diffs.csv"),
    cpu_cuda_csv = file.path(output_dir, "cpu_cuda.csv"),
    linear_fastspline_csv = file.path(output_dir, "linear_fastspline.csv"),
    residual_device_diffs_csv = file.path(output_dir, "residual_device_diffs.csv"),
    orientation_device_diffs_csv =
      file.path(output_dir, "orientation_device_diffs.csv"),
    orientation_device_diagnostics_csv =
      file.path(output_dir, "orientation_device_diagnostics.csv"),
    scheduler_diffs_csv = file.path(output_dir, "scheduler_diffs.csv"),
    scheduler_levels_csv = file.path(output_dir, "scheduler_levels.csv"),
    scheduler_batches_csv = file.path(output_dir, "scheduler_batches.csv"),
    scheduler_residuals_csv = file.path(output_dir, "scheduler_residuals.csv"),
    ci_method_diffs_csv = file.path(output_dir, "ci_method_diffs.csv"),
    ci_method_diagnostics_csv =
      file.path(output_dir, "ci_method_diagnostics.csv"),
    hsic_cuda_backend_diagnostics_csv =
      file.path(output_dir, "hsic_cuda_backend_diagnostics.csv"),
    hsic_cuda_cpu_fallbacks_csv =
      file.path(output_dir, "hsic_cuda_cpu_fallbacks.csv"),
    hsic_cuda_perf_csv = file.path(output_dir, "hsic_cuda_perf.csv"),
    true_batched_residuals_csv =
      file.path(output_dir, "true_batched_residuals.csv"),
    legacy_csv = file.path(output_dir, "legacy.csv"),
    timings_csv = file.path(output_dir, "timings.csv"),
    cache_csv = file.path(output_dir, "cache.csv"),
    orientation_counts_csv = file.path(output_dir, "orientation_counts.csv"),
    errors_csv = file.path(output_dir, "errors.csv"),
    campaign_rds = file.path(output_dir, "campaign.rds")
  )

  writeLines(fastkpc_campaign_markdown(campaign), artifacts$summary_md)
  fastkpc_write_csv(campaign$runs, artifacts$runs_csv)
  fastkpc_write_csv(campaign$graph_metrics, artifacts$graph_metrics_csv)
  fastkpc_write_csv(campaign$pairwise_diffs, artifacts$pairwise_diffs_csv)
  fastkpc_write_csv(campaign$cpu_cuda, artifacts$cpu_cuda_csv)
  fastkpc_write_csv(campaign$linear_fastspline, artifacts$linear_fastspline_csv)
  fastkpc_write_csv(campaign$residual_device_diffs %||%
                      fastkpc_empty_df(c("scenario", "seed", "n", "engine",
                                          "residual_backend", "pdag_identical",
                                          "skeleton_adjacency_identical",
                                          "max_abs_pmax_diff",
                                          "orientation_counts_identical",
                                          "status")),
                    artifacts$residual_device_diffs_csv)
  fastkpc_write_csv(campaign$orientation_device_diffs %||%
                      fastkpc_empty_df(c("scenario", "seed", "n", "engine",
                                          "residual_backend",
                                          "residual_device", "scheduler",
                                          "left_orientation_residual_device",
                                          "right_orientation_residual_device",
                                          "pdag_identical",
                                          "skeleton_adjacency_identical",
                                          "max_abs_pmax_diff",
                                          "orientation_counts_identical",
                                          "status")),
                    artifacts$orientation_device_diffs_csv)
  fastkpc_write_csv(campaign$orientation_device_diagnostics %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "residual_backend",
                                          "residual_device", "scheduler",
                                          "orientation_residual_device_requested",
                                          "orientation_residual_device",
                                          "orientation_residual_device_reason",
                                          "orientation_batch_size_requested",
                                          "orientation_batch_size_used",
                                          "regrvonps_calls",
                                          "regrvonps_cuda_calls",
                                          "regrvonps_cpu_calls",
                                          "orientation_dcov_batches",
                                          "orientation_dcov_pairs",
                                          "orientation_residual_fits",
                                          "orientation_cuda_residual_fits",
                                          "orientation_cpu_fallback_fits",
                                          "orientation_cache_requests",
                                          "orientation_cache_hits",
                                          "orientation_cache_computations")),
                    artifacts$orientation_device_diagnostics_csv)
  fastkpc_write_csv(campaign$scheduler_diffs %||%
                      fastkpc_empty_df(c("scenario", "seed", "n", "engine",
                                          "residual_backend", "residual_device",
                                          "left_scheduler", "right_scheduler",
                                          "pdag_identical",
                                          "skeleton_adjacency_identical",
                                          "max_abs_pmax_diff",
                                          "orientation_counts_identical",
                                          "status")),
                    artifacts$scheduler_diffs_csv)
  fastkpc_write_csv(campaign$scheduler_levels %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "residual_backend",
                                          "residual_device", "scheduler",
                                          "level", "tasks_planned",
                                          "tasks_evaluated", "tests_replayed",
                                          "tasks_ignored_after_delete",
                                          "deletions", "unconditional_tasks",
                                          "conditional_tasks",
                                          "unique_residual_requests",
                                          "dcov_batches", "residual_batches",
                                          "plan_elapsed_sec",
                                          "residual_prefetch_elapsed_sec",
                                          "ci_eval_elapsed_sec",
                                          "replay_elapsed_sec",
                                          "total_elapsed_sec")),
                    artifacts$scheduler_levels_csv)
  fastkpc_write_csv(campaign$scheduler_batches %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "residual_backend",
                                          "residual_device", "scheduler",
                                          "level", "batch_id", "kind",
                                          "start_task_id", "task_count",
                                          "status", "groups",
                                          "true_batched_groups",
                                          "true_batched_fits",
                                          "single_fit_calls",
                                          "cpu_fallback_fits",
                                          "max_group_size",
                                          "min_group_size",
                                          "max_design_cols",
                                          "min_design_cols")),
                    artifacts$scheduler_batches_csv)
  fastkpc_write_csv(campaign$scheduler_residuals %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine",
                                          "residual_backend_requested",
                                          "residual_device_requested",
                                          "scheduler", "level", "request_id",
                                          "target", "conditioning_size",
                                          "residual_backend",
                                          "residual_device", "materialized",
                                          "fallback_used", "reason")),
                    artifacts$scheduler_residuals_csv)
  fastkpc_write_csv(campaign$ci_method_diffs %||%
                      fastkpc_empty_df(c("scenario", "seed", "n", "engine",
                                          "residual_backend",
                                          "residual_device",
                                          "orientation_residual_device",
                                          "scheduler", "left_ci_method",
                                          "right_ci_method", "pdag_identical",
                                          "skeleton_adjacency_identical",
                                          "max_abs_pmax_diff",
                                          "orientation_counts_identical",
                                          "status")),
                    artifacts$ci_method_diffs_csv)
  fastkpc_write_csv(campaign$ci_method_diagnostics %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "residual_backend",
                                          "residual_device",
                                          "orientation_residual_device",
                                          "scheduler", "ci_method",
                                          "ci_backend", "ci_backend_requested",
                                          "ci_backend_reason",
                                          "cuda_hsic_requested",
                                          "cuda_hsic_used",
                                          "ci_dcc_gamma_tests",
                                          "ci_hsic_gamma_tests",
                                          "ci_hsic_perm_tests",
                                          "ci_hsic_permutation_replicates",
                                          "ci_hsic_gamma_cuda_tests",
                                          "ci_hsic_perm_cuda_tests",
                                          "ci_hsic_cuda_batches",
                                          "ci_hsic_cuda_pairs",
                                          "ci_hsic_cuda_fallback_tests",
                                          "ci_tests",
                                          "regrvonps_dcc_gamma_tests",
                                          "regrvonps_hsic_gamma_tests",
                                          "regrvonps_hsic_perm_tests",
                                          "regrvonps_hsic_permutation_replicates",
                                          "regrvonps_hsic_gamma_cuda_tests",
                                          "regrvonps_hsic_perm_cuda_tests",
                                          "regrvonps_hsic_cuda_batches",
                                          "regrvonps_hsic_cuda_pairs",
                                          "regrvonps_hsic_cuda_fallback_tests")),
                    artifacts$ci_method_diagnostics_csv)
  fastkpc_write_csv(campaign$hsic_cuda_backend_diagnostics %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "ci_method", "ci_backend",
                                          "ci_backend_requested",
                                          "cuda_hsic_requested",
                                          "cuda_hsic_used",
                                          "ci_hsic_cuda_batches",
                                          "ci_hsic_cuda_pairs")),
                    artifacts$hsic_cuda_backend_diagnostics_csv)
  fastkpc_write_csv(campaign$hsic_cuda_cpu_fallbacks %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "ci_method", "ci_backend",
                                          "ci_backend_reason",
                                          "ci_hsic_cuda_fallback_tests")),
                    artifacts$hsic_cuda_cpu_fallbacks_csv)
  fastkpc_write_csv(campaign$hsic_cuda_perf %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "ci_method", "ci_backend",
                                          "cuda_hsic_used",
                                          "ci_hsic_cuda_batches",
                                          "ci_hsic_cuda_pairs",
                                          "elapsed_total_sec")),
                    artifacts$hsic_cuda_perf_csv)
  fastkpc_write_csv(campaign$true_batched_residuals %||%
                      fastkpc_empty_df(c("run_id", "scenario", "seed", "n",
                                          "engine", "residual_backend",
                                          "residual_device", "scheduler",
                                          "residual_batch_size",
                                          "cuda_residual_batch_groups",
                                          "cuda_residual_true_batched_groups",
                                          "cuda_residual_true_batched_fits",
                                          "cuda_residual_single_fit_calls",
                                          "cuda_residual_cpu_fallback_fits",
                                          "status")),
                    artifacts$true_batched_residuals_csv)
  fastkpc_write_csv(campaign$legacy, artifacts$legacy_csv)
  fastkpc_write_csv(campaign$timings, artifacts$timings_csv)
  fastkpc_write_csv(campaign$cache, artifacts$cache_csv)
  fastkpc_write_csv(campaign$orientation_counts, artifacts$orientation_counts_csv)
  fastkpc_write_csv(campaign$errors, artifacts$errors_csv)
  saveRDS(campaign, artifacts$campaign_rds)
  artifacts
}
