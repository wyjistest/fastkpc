args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_setup_shadow_v1")
}
oracle_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path("fastkpc", "artifacts", "mgcv_residual_oracle_v1")
}
solver <- if (length(args) >= 3L) args[[3L]] else "mgcv_magic"
condition_threshold <- if (length(args) >= 4L) {
  as.numeric(args[[4L]])
} else {
  1e12
}

source("fastkpc/R/mgcv_residual_setup_shadow.R")
artifact <- fastkpc_run_mgcv_residual_setup_shadow(
  oracle_dir = oracle_dir,
  output_dir = output_dir,
  solver = solver,
  condition_threshold = condition_threshold
)
cat("wrote mgcv residual setup shadow artifact:", output_dir, "\n")
print(artifact$summary)
