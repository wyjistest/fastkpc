args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1L]]
} else {
  Sys.getenv(
    "FASTKPC_SKELETON_PTABLE_PARITY_DIR",
    file.path("fastkpc", "artifacts", "skeleton_ptable_parity")
  )
}

source("fastkpc/R/skeleton_ptable_parity.R")

artifact <- fastkpc_run_skeleton_ptable_parity(output_dir = output_dir)
cat("wrote skeleton p-table parity artifacts:", output_dir, "\n")
print(artifact$summary)
