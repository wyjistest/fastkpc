readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports_readme <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                        collapse = "\n")

assert_contains <- function(text, pattern) {
  if (!grepl(pattern, text, fixed = TRUE)) {
    stop("missing required text: ", pattern, call. = FALSE)
  }
}

required_readme <- c(
  "CUDA Residual Device",
  "residual_device",
  "fastspline_residual_cuda",
  "fastspline_residual_batch_cuda",
  "CUDA residual kernels are opt-in",
  "linear residual CUDA device is not implemented",
  "CUDA residual fallback",
  "kpcalg::kpc() is not replaced",
  "kpcalg/R/*.R files are not modified"
)
for (pattern in required_readme) assert_contains(readme, pattern)

required_reports <- c(
  "residual_device_diffs.csv",
  "residual_device",
  "cuda-fallback-cpu"
)
for (pattern in required_reports) assert_contains(reports_readme, pattern)

cat("test_cuda_residual_docs_contract.R: PASS\n")
