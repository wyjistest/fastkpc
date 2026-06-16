readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports_readme <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                        collapse = "\n")

assert_contains <- function(text, pattern) {
  if (!grepl(pattern, text, fixed = TRUE)) {
    stop("missing required text: ", pattern, call. = FALSE)
  }
}

required_readme <- c(
  "Public fast_kpc API",
  "fast_kpc(",
  "fastkpc_result",
  "Validation Campaign",
  "run_fastkpc_validation_campaign",
  "Validation Reports",
  "write_fastkpc_validation_report",
  "Command Line Tools",
  "run_fast_kpc.R",
  "run_validation_campaign.R",
  "kpcalg::kpc() is not replaced",
  "kpcalg/R/*.R files are not modified"
)
for (pattern in required_readme) assert_contains(readme, pattern)

required_reports <- c(
  "fastkpc reports",
  "summary.md",
  "runs.csv",
  "cpu_cuda.csv",
  "legacy.csv",
  "campaign.rds"
)
for (pattern in required_reports) assert_contains(reports_readme, pattern)

cat("test_fastkpc_docs_contract.R: PASS\n")
