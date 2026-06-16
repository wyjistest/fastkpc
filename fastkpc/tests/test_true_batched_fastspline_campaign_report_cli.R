source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

campaign <- run_fastkpc_validation_campaign(
  seeds = c(11),
  n_values = c(60),
  scenarios = c("chain"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cuda"),
  schedulers = c("layer"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE,
  benchmark = TRUE,
  residual_batch_size = 0,
  scheduler_diagnostics = TRUE,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8)
)

assert_true("true_batched_residuals" %in% names(campaign),
            "campaign should include true_batched_residuals")
assert_true(is.data.frame(campaign$true_batched_residuals),
            "true_batched_residuals should be a data frame")
assert_true(nrow(campaign$true_batched_residuals) >= 1L,
            "true_batched_residuals should contain campaign rows")
assert_true(any(campaign$true_batched_residuals$cuda_residual_true_batched_groups > 0),
            "campaign should record true-batched residual groups")
assert_true(any(campaign$runs$cuda_residual_true_batched_groups > 0),
            "runs should include true-batch residual counters")

output_dir <- tempfile("fastkpc-true-batch-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(file.path(output_dir, "true_batched_residuals.csv")),
            "report should write true_batched_residuals.csv")
assert_true(file.exists(artifacts$summary_md), "summary markdown should be written")
summary_text <- paste(readLines(artifacts$summary_md, warn = FALSE),
                      collapse = "\n")
assert_true(grepl("True-Batched CUDA fastSpline Residuals", summary_text,
                  fixed = TRUE),
            "summary markdown should include true-batch section")

input_csv <- tempfile(fileext = ".csv")
output_rds <- tempfile(fileext = ".rds")
set.seed(512)
n <- 60
z <- seq(-pi, pi, length.out = n)
data <- data.frame(
  x1 = z + rnorm(n, sd = 0.05),
  x2 = sin(z) + rnorm(n, sd = 0.08),
  x3 = cos(z) + rnorm(n, sd = 0.08),
  x4 = z^2 + rnorm(n, sd = 0.08),
  x5 = rnorm(n)
)
utils::write.csv(data, input_csv, row.names = FALSE)
cli_output <- system2("Rscript", c("fastkpc/tools/run_fast_kpc.R",
                                   "--input", input_csv,
                                   "--output", output_rds,
                                   "--engine", "cuda",
                                   "--residual-backend", "fastSpline",
                                   "--residual-device", "cuda",
                                   "--scheduler", "layer",
                                   "--graph-stage", "skeleton",
                                   "--max-conditioning-size", "1",
                                   "--residual-batch-size", "0"),
                      stdout = TRUE, stderr = TRUE)
cli_status <- attr(cli_output, "status") %||% 0L
assert_true(identical(as.integer(cli_status), 0L),
            "run_fast_kpc.R true-batch call should pass")
assert_true(any(grepl("cuda_residual_true_batched_groups=", cli_output,
                      fixed = TRUE)),
            "run_fast_kpc.R should print true-batch counters")

report_dir <- tempfile("fastkpc-true-batch-cli-report-")
status <- system2("Rscript", c("fastkpc/tools/run_validation_campaign.R",
                               "--output-dir", report_dir,
                               "--seeds", "11",
                               "--n-values", "50",
                               "--scenarios", "chain",
                               "--engines", "cuda",
                               "--residual-backends", "fastSpline",
                               "--residual-devices", "cuda",
                               "--schedulers", "layer",
                               "--max-conditioning-size", "1",
                               "--legacy", "FALSE",
                               "--residual-batch-size", "0"))
assert_true(identical(status, 0L),
            "run_validation_campaign.R true-batch call should pass")
assert_true(file.exists(file.path(report_dir, "true_batched_residuals.csv")),
            "campaign CLI should write true_batched_residuals.csv")

cat("test_true_batched_fastspline_campaign_report_cli.R: PASS\n")
