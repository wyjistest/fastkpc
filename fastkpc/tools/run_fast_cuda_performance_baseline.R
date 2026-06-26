args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  Sys.getenv(
    "FASTKPC_FAST_CUDA_BASELINE_DIR",
    file.path("fastkpc", "artifacts", "fast_cuda_performance_baseline")
  )
}

source("fastkpc/R/fast_cuda_performance_baseline.R")

env_bool <- function(name, default = FALSE) {
  value <- tolower(Sys.getenv(name, ""))
  if (!nzchar(value)) return(default)
  value %in% c("1", "true", "yes", "y")
}

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(default)
  as.integer(value)
}

env_modes <- function(default) {
  value <- Sys.getenv("FASTKPC_FAST_CUDA_BASELINE_MODES", "")
  if (!nzchar(value)) return(default)
  modes <- trimws(unlist(strsplit(value, "[,[:space:]]+")))
  modes <- modes[nzchar(modes)]
  match.arg(modes, fastkpc_fast_cuda_baseline_modes(), several.ok = TRUE)
}

artifact <- fastkpc_run_fast_cuda_performance_baseline(
  output_dir = output_dir,
  modes = env_modes(fastkpc_fast_cuda_baseline_modes()),
  repeats = env_int("FASTKPC_FAST_CUDA_BASELINE_REPEATS", 5L),
  warmup = env_bool("FASTKPC_FAST_CUDA_BASELINE_WARMUP", TRUE),
  full_grid = env_bool("FASTKPC_FAST_CUDA_FULL_GRID", FALSE),
  real_data_path = Sys.getenv("FASTKPC_FAST_CUDA_REAL_DATA", ""),
  real_n = env_int("FASTKPC_FAST_CUDA_REAL_N", 100L),
  real_p = env_int("FASTKPC_FAST_CUDA_REAL_P", 12L),
  legacy_max_n = env_int("FASTKPC_FAST_CUDA_LEGACY_MAX_N", 300L),
  legacy_max_p = env_int("FASTKPC_FAST_CUDA_LEGACY_MAX_P", 12L),
  legacy_max_level = env_int("FASTKPC_FAST_CUDA_LEGACY_MAX_LEVEL", 2L)
)

cat("wrote fast CUDA performance baseline artifacts:", output_dir, "\n")
print(artifact$summary)
