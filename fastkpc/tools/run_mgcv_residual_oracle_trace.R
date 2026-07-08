args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_oracle_v1")
}

source("fastkpc/R/mgcv_residual_oracle_trace.R")
artifact <- fastkpc_run_mgcv_residual_oracle_trace(output_dir = output_dir)
cat("wrote mgcv residual oracle trace artifact:", output_dir, "\n")
print(artifact$summary)
