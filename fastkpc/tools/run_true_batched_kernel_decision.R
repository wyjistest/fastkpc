source("fastkpc/R/true_batched_kernel_decision.R")

output_dir <- Sys.getenv(
  "FASTKPC_TRUE_BATCHED_DECISION_DIR",
  file.path("fastkpc", "artifacts", "true_batched_kernel_decision")
)

timing_csv <- Sys.getenv("FASTKPC_TIMING_CSV", "")
workload_csv <- Sys.getenv("FASTKPC_WORKLOAD_STATS_CSV", "")

if (nzchar(timing_csv) && file.exists(timing_csv) &&
    nzchar(workload_csv) && file.exists(workload_csv)) {
  timing <- utils::read.csv(timing_csv, stringsAsFactors = FALSE)
  workload <- utils::read.csv(workload_csv, stringsAsFactors = FALSE)
  decision <- fastkpc_true_batched_kernel_decision(timing, workload)
} else {
  decision <- data.frame(
    decision = "defer",
    rationale = "insufficient timing/workload evidence",
    linear_solve_fraction = NA_real_,
    targets_per_setup_p95 = NA_real_,
    unsupported_fraction = NA_real_,
    stringsAsFactors = FALSE
  )
}

out <- fastkpc_write_true_batched_kernel_decision(
  decision,
  output_dir = output_dir
)
cat("true batched kernel decision report:", out$report_path, "\n")
cat("true batched kernel decision CSV:", out$csv_path, "\n")
