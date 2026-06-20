args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "precision_end_to_end_benchmark")
}

source("fastkpc/R/precision_end_to_end_benchmark.R")

result <- fastkpc_run_precision_end_to_end_benchmark(output_dir = output_dir)
cat("wrote precision end-to-end benchmark artifacts:", output_dir, "\n")
print(result$summary)
