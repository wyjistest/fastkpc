args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  Sys.getenv(
    "FASTKPC_DCC_GAMMA_CUDA_PARITY_DIR",
    file.path("fastkpc", "artifacts", "dcc_gamma_cuda_parity")
  )
}

source("fastkpc/R/dcc_gamma_cuda_parity.R")

artifact <- fastkpc_run_dcc_gamma_cuda_parity(output_dir = output_dir)
cat("wrote dcc.gamma CUDA parity artifacts:", output_dir, "\n")
print(artifact$summary)
