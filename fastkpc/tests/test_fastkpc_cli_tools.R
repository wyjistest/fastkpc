assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

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
    "--engine", "cpu",
    "--residual-backend", "fastSpline",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--graph-stage", "wanpdag")
)
assert_true(identical(status_one, 0L), "run_fast_kpc.R should exit 0")
assert_true(file.exists(output), "run_fast_kpc.R should write result RDS")
result <- readRDS(output)
assert_true(inherits(result, "fastkpc_result"), "CLI result should be fastkpc_result")

report_dir <- tempfile("fastkpc-cli-report-")
status_campaign <- system2(
  "Rscript",
  c("fastkpc/tools/run_validation_campaign.R",
    "--output-dir", report_dir,
    "--seeds", "61",
    "--n-values", "60",
    "--scenarios", "chain,independent",
    "--engines", "cpu",
    "--residual-backends", "linear,fastSpline",
    "--alpha", "0.2",
    "--max-conditioning-size", "1",
    "--legacy", "TRUE")
)
assert_true(identical(status_campaign, 0L), "run_validation_campaign.R should exit 0")
assert_true(file.exists(file.path(report_dir, "summary.md")),
            "campaign CLI should write summary.md")
assert_true(file.exists(file.path(report_dir, "campaign.rds")),
            "campaign CLI should write campaign.rds")

cat("test_fastkpc_cli_tools.R: PASS\n")
