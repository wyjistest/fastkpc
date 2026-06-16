readme <- paste(readLines("fastkpc/README.md", warn = FALSE), collapse = "\n")
reports <- paste(readLines("fastkpc/reports/README.md", warn = FALSE),
                 collapse = "\n")

must <- c(
  "Layer-batched CUDA scheduler",
  "scheduler=\"layer\"",
  "scheduler=\"legacy\"",
  "residual_batch_size",
  "scheduler_diagnostics",
  "planned",
  "replayed",
  "CUDA orientation residual/dCov execution is opt-in",
  "WAN-PDAG graph mutation remains sequential"
)

for (needle in must) {
  if (!grepl(needle, readme, fixed = TRUE)) {
    stop("README missing: ", needle, call. = FALSE)
  }
}

for (needle in c("scheduler_diffs.csv", "scheduler_levels.csv",
                 "scheduler_batches.csv", "scheduler_residuals.csv")) {
  if (!grepl(needle, reports, fixed = TRUE)) {
    stop("reports README missing: ", needle, call. = FALSE)
  }
}

cat("test_layer_scheduler_docs_contract.R: PASS\n")
