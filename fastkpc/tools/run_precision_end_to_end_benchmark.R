args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  file.path("fastkpc", "artifacts", "precision_end_to_end_benchmark")
}

source("fastkpc/R/precision_end_to_end_benchmark.R")

fastkpc_env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.integer(value)
}

fastkpc_env_bool <- function(name, default) {
  value <- tolower(Sys.getenv(name, ""))
  if (!nzchar(value)) return(default)
  value %in% c("1", "true", "yes", "y")
}

result <- fastkpc_run_precision_end_to_end_benchmark(
  output_dir = output_dir,
  modes = fastkpc_precision_e2e_env_modes(
    c("legacy_mgcv", "fast_cuda", "primary_only_cuda",
      "compatible_cuda", "hybrid_cuda")
  ),
  repeats = fastkpc_env_int("FASTKPC_PRECISION_E2E_REPEATS", 5L),
  warmup = fastkpc_env_bool("FASTKPC_PRECISION_E2E_WARMUP", TRUE),
  randomize_mode_order =
    fastkpc_env_bool("FASTKPC_PRECISION_E2E_RANDOMIZE", TRUE),
  real_data_path = Sys.getenv("FASTKPC_PRECISION_E2E_REAL_DATA", "")
)
cat("wrote precision end-to-end benchmark artifacts:", output_dir, "\n")
print(result$summary)
