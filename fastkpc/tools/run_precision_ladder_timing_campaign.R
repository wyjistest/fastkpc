source("fastkpc/R/precision_ladder_timing.R")

output_dir <- Sys.getenv(
  "FASTKPC_PRECISION_LADDER_TIMING_DIR",
  file.path("fastkpc", "artifacts", "precision_ladder_timing")
)

rows <- rbind(
  fastkpc_precision_ladder_timing_row(
    backend = "mgcvExtractGPUFixedSP",
    mode = "same-setup-native-batch",
    solve_source = "cuda-fixed-sp",
    native_gpu_solve_used = TRUE,
    true_batched_kernel = FALSE,
    targets_per_setup = 4L,
    setup_reuse_count = 1L,
    total_ms = NA_real_,
    fallback_reason = "",
    setup_fingerprint = "synthetic-setup",
    target_fingerprint = "synthetic-targets"
  ),
  fastkpc_precision_ladder_timing_row(
    backend = "legacy-mgcv",
    mode = "reference",
    solve_source = "mgcv",
    native_gpu_solve_used = FALSE,
    true_batched_kernel = FALSE,
    targets_per_setup = 1L,
    total_ms = NA_real_,
    fallback_reason = "",
    setup_fingerprint = "synthetic-setup",
    target_fingerprint = "synthetic-target"
  )
)

out <- fastkpc_write_precision_ladder_timing_report(rows, output_dir = output_dir)
cat("precision ladder timing report:", out$report_path, "\n")
cat("precision ladder timing CSV:", out$csv_path, "\n")
