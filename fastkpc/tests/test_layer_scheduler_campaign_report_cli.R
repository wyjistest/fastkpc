source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(11),
  n_values = c(60),
  scenarios = c("chain"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cuda"),
  schedulers = c("legacy", "layer"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = TRUE,
  residual_batch_size = 0,
  scheduler_diagnostics = TRUE
)

assert_true("scheduler" %in% names(campaign$runs),
            "runs should include scheduler")
assert_true(all(c("legacy", "layer") %in% campaign$runs$scheduler),
            "runs should include legacy and layer scheduler rows")
assert_true("scheduler_diffs" %in% names(campaign),
            "campaign should include scheduler_diffs")
assert_true(is.data.frame(campaign$scheduler_diffs),
            "scheduler_diffs should be a data.frame")
assert_true(nrow(campaign$scheduler_diffs) >= 1L,
            "scheduler diffs should have rows")
assert_true(all(campaign$scheduler_diffs$skeleton_adjacency_identical),
            "scheduler comparison should preserve skeleton adjacency")
assert_true(all(campaign$scheduler_diffs$max_abs_pmax_diff < 1e-7),
            "scheduler comparison pMax diff should be tiny")
assert_true(is.data.frame(campaign$scheduler_levels),
            "campaign should include scheduler_levels")
timing_cols <- c("plan_elapsed_sec", "residual_prefetch_elapsed_sec",
                 "ci_eval_elapsed_sec", "replay_elapsed_sec",
                 "total_elapsed_sec")
assert_true(all(timing_cols %in% names(campaign$scheduler_levels)),
            "scheduler_levels should include timing columns")
assert_true(all(is.finite(as.matrix(campaign$scheduler_levels[, timing_cols,
                                                              drop = FALSE]))),
            "scheduler level timing columns should be finite")
assert_true(is.data.frame(campaign$scheduler_batches),
            "campaign should include scheduler_batches")
batch_diag_cols <- c("groups", "true_batched_groups", "true_batched_fits",
                     "single_fit_calls", "cpu_fallback_fits",
                     "unique_designs", "duplicate_design_fits",
                     "max_fits_per_design",
                     "max_group_size", "min_group_size",
                     "max_design_cols", "min_design_cols")
assert_true(all(batch_diag_cols %in% names(campaign$scheduler_batches)),
            "scheduler_batches should include residual batch group diagnostics")
residual_batches <- campaign$scheduler_batches[
  campaign$scheduler_batches$kind == "residual", , drop = FALSE
]
assert_true(nrow(residual_batches) > 0L,
            "scheduler_batches should contain residual batch rows")
assert_true(any(residual_batches$true_batched_groups > 0L),
            "residual batch diagnostics should record true-batched groups")
assert_true(is.data.frame(campaign$scheduler_residuals),
            "campaign should include scheduler_residuals")

output_dir <- tempfile("fastkpc-scheduler-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
for (name in c("scheduler_diffs.csv", "scheduler_levels.csv",
               "scheduler_batches.csv", "scheduler_residuals.csv")) {
  assert_true(file.exists(file.path(output_dir, name)),
              paste("report should write", name))
}
scheduler_levels_csv <- utils::read.csv(file.path(output_dir, "scheduler_levels.csv"),
                                        check.names = FALSE)
assert_true(all(timing_cols %in% names(scheduler_levels_csv)),
            "scheduler_levels.csv should include timing columns")
scheduler_batches_csv <- utils::read.csv(file.path(output_dir, "scheduler_batches.csv"),
                                         check.names = FALSE)
assert_true(all(batch_diag_cols %in% names(scheduler_batches_csv)),
            "scheduler_batches.csv should include residual batch group diagnostics")
assert_true(file.exists(artifacts$summary_md), "summary markdown should be written")

input_csv <- tempfile(fileext = ".csv")
output_rds <- tempfile(fileext = ".rds")
set.seed(412)
n <- 50
z <- seq(-pi, pi, length.out = n)
data <- data.frame(
  x1 = z + rnorm(n, sd = 0.05),
  x2 = sin(z) + rnorm(n, sd = 0.08),
  x3 = cos(z) + rnorm(n, sd = 0.08),
  x4 = rnorm(n)
)
utils::write.csv(data, input_csv, row.names = FALSE)
status <- system2("Rscript", c("fastkpc/tools/run_fast_kpc.R",
                               "--input", input_csv,
                               "--output", output_rds,
                               "--engine", "cuda",
                               "--residual-backend", "fastSpline",
                               "--residual-device", "cuda",
                               "--scheduler", "layer",
                               "--graph-stage", "skeleton",
                               "--max-conditioning-size", "1",
                               "--residual-batch-size", "0"))
assert_true(identical(status, 0L), "run_fast_kpc.R scheduler call should pass")
cli_result <- readRDS(output_rds)
assert_true(cli_result$config$scheduler_requested == "layer",
            "single-run CLI should record scheduler")

report_dir <- tempfile("fastkpc-scheduler-cli-report-")
status <- system2("Rscript", c("fastkpc/tools/run_validation_campaign.R",
                               "--output-dir", report_dir,
                               "--seeds", "11",
                               "--n-values", "50",
                               "--scenarios", "chain",
                               "--engines", "cuda",
                               "--residual-backends", "fastSpline",
                               "--residual-devices", "cuda",
                               "--schedulers", "legacy,layer",
                               "--max-conditioning-size", "1",
                               "--legacy", "FALSE",
                               "--residual-batch-size", "0"))
assert_true(identical(status, 0L),
            "run_validation_campaign.R scheduler call should pass")
assert_true(file.exists(file.path(report_dir, "scheduler_diffs.csv")),
            "campaign CLI should write scheduler_diffs.csv")
assert_true(file.exists(file.path(report_dir, "scheduler_levels.csv")),
            "campaign CLI should write scheduler_levels.csv")

cat("test_layer_scheduler_campaign_report_cli.R: PASS\n")
