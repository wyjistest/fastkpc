args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_cpp_shadow_v1")
}
oracle_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_oracle_v1")
}

source("fastkpc/R/mgcv_residual_cpp_shadow.R")
artifact <- fastkpc_run_mgcv_residual_cpp_shadow(
  oracle_dir = oracle_dir,
  output_dir = output_dir
)
cat("wrote mgcv residual C++ shadow artifact:", output_dir, "\n")
print(artifact$summary)
