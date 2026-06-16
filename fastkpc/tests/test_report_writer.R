source("fastkpc/R/validation_campaign.R")
source("fastkpc/R/report_writer.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

campaign <- run_fastkpc_validation_campaign(
  seeds = c(51),
  n_values = c(60),
  scenarios = c("chain", "independent"),
  engines = c("cpu"),
  residual_backends = c("linear", "fastSpline"),
  legacy = TRUE
)

output_dir <- tempfile("fastkpc-report-")
artifacts <- write_fastkpc_validation_report(campaign, output_dir)

required_files <- c("summary.md", "runs.csv", "graph_metrics.csv",
                    "pairwise_diffs.csv", "cpu_cuda.csv",
                    "linear_fastspline.csv", "legacy.csv", "timings.csv",
                    "cache.csv", "orientation_counts.csv", "errors.csv",
                    "campaign.rds")
for (file in required_files) {
  path <- file.path(output_dir, file)
  assert_true(file.exists(path), paste(file, "should exist"))
  assert_true(file.info(path)$size > 0, paste(file, "should be non-empty"))
}

summary_text <- paste(readLines(file.path(output_dir, "summary.md"), warn = FALSE),
                      collapse = "\n")
required_headings <- c("# fastkpc Validation Campaign", "## Configuration",
                       "## Summary", "## CPU vs CUDA", "## Linear vs fastSpline",
                       "## Legacy Diagnostics", "## Timings", "## Cache",
                       "## Errors", "## Reproduction")
for (heading in required_headings) {
  assert_true(grepl(heading, summary_text, fixed = TRUE),
              paste("summary.md missing", heading))
}

loaded <- readRDS(file.path(output_dir, "campaign.rds"))
assert_true(is.list(loaded) && is.data.frame(loaded$runs), "campaign.rds should reload")
assert_true(is.list(artifacts) && length(artifacts) >= length(required_files),
            "writer should return artifact paths")

cat("test_report_writer.R: PASS\n")
