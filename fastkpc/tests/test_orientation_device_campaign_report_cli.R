source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

output_dir <- tempfile("fastkpc-orientation-device-report-")

campaign <- run_fastkpc_validation_campaign(
  seeds = c(144),
  n_values = c(80),
  scenarios = c("chain"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cuda"),
  orientation_residual_devices = c("cpu", "cuda"),
  schedulers = c("layer"),
  residual_batch_size = 0L,
  orientation_batch_size = 1L,
  scheduler_diagnostics = TRUE,
  orientation_diagnostics = TRUE,
  legacy = FALSE,
  benchmark = TRUE,
  fastspline_params = list(knots = 7, lambda_count = 13, ridge = 1e-8),
  output_dir = output_dir
)

assert_true("orientation_device_diffs" %in% names(campaign),
            "campaign should include orientation_device_diffs")
assert_true("orientation_device_diagnostics" %in% names(campaign),
            "campaign should include orientation_device_diagnostics")
assert_true(nrow(campaign$orientation_device_diffs) > 0L,
            "orientation_device_diffs should have rows")
assert_true(any(campaign$orientation_device_diagnostics$orientation_residual_device ==
                  "cuda"),
            "orientation diagnostics should contain cuda orientation rows")
assert_true(any(campaign$orientation_device_diagnostics$orientation_dcov_batches > 0L),
            "orientation diagnostics should record dCov batches")

artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(artifacts$orientation_device_diffs_csv),
            "report should write orientation_device_diffs.csv")
assert_true(file.exists(artifacts$orientation_device_diagnostics_csv),
            "report should write orientation_device_diagnostics.csv")
summary_text <- paste(readLines(artifacts$summary_md, warn = FALSE),
                      collapse = "\n")
assert_true(grepl("Orientation Device", summary_text, fixed = TRUE),
            "summary should contain Orientation Device section")

input_file <- tempfile(fileext = ".csv")
data <- generate_fastkpc_scenario("chain", 145, 80)$data
utils::write.csv(data, input_file, row.names = FALSE)
output_file <- tempfile(fileext = ".rds")
cli_output <- system2(
  "Rscript",
  c("fastkpc/tools/run_fast_kpc.R",
    "--input", input_file,
    "--output", output_file,
    "--engine", "cuda",
    "--residual-backend", "fastSpline",
    "--residual-device", "cuda",
    "--orientation-residual-device", "cuda",
    "--orientation-batch-size", "1",
    "--scheduler", "layer",
    "--graph-stage", "wanpdag",
    "--max-conditioning-size", "1"),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(file.exists(output_file),
            "run_fast_kpc.R should write output file")
assert_true(any(grepl("orientation_residual_device=cuda", cli_output,
                      fixed = TRUE)),
            "run_fast_kpc.R should print orientation device")

cli_report_dir <- tempfile("fastkpc-orientation-device-cli-")
cli_report <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", cli_report_dir,
    "--seeds", "146",
    "--n-values", "80",
    "--scenarios", "chain",
    "--engines", "cuda",
    "--residual-backends", "fastSpline",
    "--residual-devices", "cuda",
    "--orientation-residual-devices", "cpu,cuda",
    "--orientation-batch-size", "1",
    "--schedulers", "layer",
    "--legacy", "FALSE",
    "--max-conditioning-size", "1"),
  stdout = TRUE,
  stderr = TRUE
)
assert_true(any(grepl("wrote report:", cli_report, fixed = TRUE)),
            "run_validation_campaign.R should complete")
assert_true(file.exists(file.path(cli_report_dir,
                                  "orientation_device_diffs.csv")),
            "CLI report should write orientation_device_diffs.csv")
assert_true(file.exists(file.path(cli_report_dir,
                                  "orientation_device_diagnostics.csv")),
            "CLI report should write orientation_device_diagnostics.csv")

cat("test_orientation_device_campaign_report_cli.R: PASS\n")
