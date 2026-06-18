source("fastkpc/R/precision_ladder_report.R")

output_dir <- Sys.getenv(
  "FASTKPC_PRECISION_LADDER_SUMMARY_DIR",
  file.path("fastkpc", "artifacts", "precision_ladder_summary")
)

out <- fastkpc_write_precision_ladder_summary_report(output_dir = output_dir)
cat("precision ladder summary report:", out$report_path, "\n")
cat("precision ladder backend summary:", out$summary_csv, "\n")
