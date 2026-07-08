args <- commandArgs(trailingOnly = TRUE)
source("fastkpc/R/mgcv_residual_cpp_numeric_shadow_expanded.R")
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts",
            "mgcv_residual_cpp_numeric_shadow_expanded_v1")
}
source_result_path <- if (length(args) >= 2L) {
  args[[2L]]
} else {
  fastkpc_mgcv_oracle_default_result_path()
}
max_cases <- if (length(args) >= 3L) as.integer(args[[3L]]) else 48L
solver <- if (length(args) >= 4L) args[[4L]] else "cpp"
condition_threshold <- if (length(args) >= 5L) {
  as.numeric(args[[5L]])
} else {
  1e12
}

artifact <- fastkpc_run_mgcv_residual_cpp_numeric_shadow_expanded(
  output_dir = output_dir,
  source_result_path = source_result_path,
  max_cases = max_cases,
  solver = solver,
  condition_threshold = condition_threshold
)
cat("wrote expanded mgcv residual C++ numeric shadow artifact:",
    output_dir, "\n")
print(artifact$summary)
