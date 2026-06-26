args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  Sys.getenv(
    "FASTKPC_FAST_CUDA_DATA_PLANE_DIR",
    file.path("fastkpc", "artifacts", "fast_cuda_data_plane_validation")
  )
}

include_benchmark <- tolower(Sys.getenv("FASTKPC_FAST_CUDA_INCLUDE_BENCHMARK",
                                        "false")) %in%
  c("1", "true", "yes", "y")
benchmark_repeats <- as.integer(Sys.getenv("FASTKPC_FAST_CUDA_BENCHMARK_REPEATS",
                                           "1"))

source("fastkpc/R/fast_cuda_data_plane_validation.R")

artifact <- fastkpc_run_fast_cuda_data_plane_validation(
  output_dir = output_dir,
  include_benchmark = include_benchmark,
  benchmark_repeats = benchmark_repeats
)
cat("wrote fast CUDA data-plane validation artifacts:", output_dir, "\n")
print(artifact$summary)
