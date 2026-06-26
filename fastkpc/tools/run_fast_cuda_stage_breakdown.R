args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  Sys.getenv(
    "FASTKPC_FAST_CUDA_STAGE_BREAKDOWN_DIR",
    file.path("fastkpc", "artifacts", "fast_cuda_stage_breakdown")
  )
}

source("fastkpc/R/fast_cuda_stage_breakdown.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.integer(value)
}

env_numeric <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.numeric(value)
}

artifact <- fastkpc_run_fast_cuda_stage_breakdown(
  output_dir = output_dir,
  repeats = env_int("FASTKPC_FAST_CUDA_STAGE_BREAKDOWN_REPEATS", 3L),
  alpha = env_numeric("FASTKPC_FAST_CUDA_STAGE_BREAKDOWN_ALPHA", 0.2)
)

cat("wrote fast CUDA stage breakdown artifacts:", output_dir, "\n")
print(artifact$summary)
