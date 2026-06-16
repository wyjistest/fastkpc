assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

read_text <- function(path) paste(readLines(path, warn = FALSE), collapse = "\n")

readme <- read_text("fastkpc/README.md")
reports_readme <- read_text("fastkpc/reports/README.md")

required_readme_terms <- c(
  "CUDA WAN-PDAG Orientation Residuals",
  "orientation_residual_device",
  "orientation_batch_size",
  "orientation_diagnostics",
  "orientation_device_diffs.csv",
  "orientation_device_diagnostics.csv",
  "regrvonps_cuda_calls",
  "orientation_dcov_batches",
  "orientation_cuda_residual_fits"
)

for (term in required_readme_terms) {
  assert_true(grepl(term, readme, fixed = TRUE),
              paste("README should mention", term))
}

required_report_terms <- c(
  "orientation_device_diffs.csv",
  "orientation_device_diagnostics.csv",
  "orientation_residual_device",
  "orientation dCov batch/pair"
)

for (term in required_report_terms) {
  assert_true(grepl(term, reports_readme, fixed = TRUE),
              paste("reports README should mention", term))
}

cat("test_orientation_device_docs_contract.R: PASS\n")
