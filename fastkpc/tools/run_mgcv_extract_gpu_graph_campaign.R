args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_extract_gpu_graph")
}

source("fastkpc/R/mgcv_extract_gpu_graph_campaign.R")
result <- fastkpc_run_mgcv_extract_gpu_graph_campaign(output_dir = output_dir)
cat("wrote mgcvExtractGPU graph campaign artifacts:", output_dir, "\n")
print(result$summary)
