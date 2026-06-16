source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(106),
  n_values = c(60),
  scenarios = c("chain"),
  engines = c("cuda"),
  residual_backends = c("fastSpline"),
  residual_devices = c("cpu", "cuda"),
  alpha = 0.2,
  max_conditioning_size = 1L,
  legacy = FALSE
)

output_dir <- tempfile("fastkpc-cuda-residual-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)
assert_true(file.exists(file.path(output_dir, "residual_device_diffs.csv")),
            "report should write residual_device_diffs.csv")
assert_true(file.exists(artifacts$summary_md), "summary.md should exist")

summary_text <- paste(readLines(artifacts$summary_md, warn = FALSE), collapse = "\n")
assert_true(grepl("## Residual Device", summary_text, fixed = TRUE),
            "summary should contain residual device section")

report_dir <- tempfile("fastkpc-cuda-residual-cli-")
status <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", report_dir,
    "--seeds", "107",
    "--n-values", "60",
    "--scenarios", "chain",
    "--engines", "cuda",
    "--residual-backends", "fastSpline",
    "--residual-devices", "cpu,cuda",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--legacy", "FALSE")
)
assert_true(identical(status, 0L), "campaign CLI should accept residual-devices")
assert_true(file.exists(file.path(report_dir, "residual_device_diffs.csv")),
            "campaign CLI should write residual_device_diffs.csv")

input <- tempfile(fileext = ".csv")
output <- tempfile(fileext = ".rds")
z <- seq(-2, 2, length.out = 70)
utils::write.csv(data.frame(x1 = z, x2 = sin(z), x3 = cos(z), x4 = z^2),
                 input, row.names = FALSE)
status_one <- system2(
  "Rscript",
  c("fastkpc/tools/run_fast_kpc.R",
    "--input", input,
    "--output", output,
    "--engine", "cuda",
    "--residual-backend", "fastSpline",
    "--residual-device", "cuda",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--graph-stage", "wanpdag")
)
assert_true(identical(status_one, 0L), "single CLI should accept residual-device")
result <- readRDS(output)
assert_true(result$config$residual_device_requested == "cuda",
            "single CLI result should record residual-device request")

cat("test_cuda_residual_device_report_cli.R: PASS\n")
