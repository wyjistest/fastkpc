args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_replay_spec_v1")
}
oracle_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_oracle_v1")
}

source("fastkpc/R/mgcv_residual_replay_spec.R")
artifact <- fastkpc_run_mgcv_residual_replay_spec(
  oracle_dir = oracle_dir,
  output_dir = output_dir
)
cat("wrote mgcv residual replay spec artifact:", output_dir, "\n")
print(artifact$summary)
