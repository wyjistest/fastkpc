args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts",
            "mgcv_residual_cpp_numeric_drift_isolation_v1")
}
shadow_dir <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  file.path("fastkpc", "artifacts",
            "mgcv_residual_cpp_numeric_shadow_expanded_v1")
}
oracle_dir <- if (length(args) >= 3L) {
  args[[3L]]
} else {
  file.path(shadow_dir, "oracle")
}

source("fastkpc/R/mgcv_residual_cpp_numeric_drift_isolation.R")
artifact <- fastkpc_run_mgcv_residual_cpp_numeric_drift_isolation(
  shadow_dir = shadow_dir,
  oracle_dir = oracle_dir,
  output_dir = output_dir
)
cat("wrote mgcv residual C++ numeric drift isolation artifact:",
    output_dir, "\n")
print(artifact$summary)
