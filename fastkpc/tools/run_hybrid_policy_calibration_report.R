source("fastkpc/R/hybrid_policy_calibration_report.R")

output_dir <- Sys.getenv(
  "FASTKPC_HYBRID_POLICY_DIR",
  file.path("fastkpc", "artifacts", "hybrid_policy_calibration")
)

campaign_path <- Sys.getenv("FASTKPC_HYBRID_CALIBRATION_CSV", "")
if (nzchar(campaign_path) && file.exists(campaign_path)) {
  campaign <- utils::read.csv(campaign_path, stringsAsFactors = FALSE)
} else {
  campaign <- data.frame(
    tau = c(log(1.5), log(2), log(3), log(5)),
    alpha = 0.05,
    num_tests_total = 100L,
    num_verified = c(5L, 10L, 20L, 40L),
    num_primary_decision_flips_vs_legacy = 12L,
    num_hybrid_decision_flips_vs_legacy = c(8L, 4L, 3L, 3L),
    skeleton_shd_primary = 5L,
    skeleton_shd_hybrid = c(4L, 2L, 2L, 2L),
    runtime_primary = 1,
    runtime_hybrid = c(1.1, 1.3, 1.8, 3.0),
    runtime_legacy = 10
  )
}

out <- fastkpc_write_hybrid_policy_calibration_report(
  campaign,
  output_dir = output_dir
)
cat("hybrid policy report:", out$report_path, "\n")
cat("hybrid policy summary CSV:", out$summary_csv, "\n")
