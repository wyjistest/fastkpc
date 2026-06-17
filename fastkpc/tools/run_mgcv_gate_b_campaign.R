args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) args[[1]] else file.path("fastkpc", "artifacts", "mgcv_gate_b")
source("fastkpc/R/mgcv_gate_b_campaign.R")
result <- fastkpc_run_mgcv_gate_b_campaign(output_dir = output_dir)
cat("wrote:", file.path(output_dir, "mgcv_gate_b_fixed_sp_campaign.csv"), "\n")
cat("rows:", nrow(result$fixed_sp), "\n")
cat("pass_gate_b:", sum(result$fixed_sp$pass_gate_b), "\n")
if (!all(result$fixed_sp$pass_gate_b)) {
  failing <- result$fixed_sp[!result$fixed_sp$pass_gate_b, ]
  print(utils::head(failing[, c("scenario_id", "seed", "n", "S_size", "sp_source", "warning_message")], 10))
  quit(save = "no", status = 1)
}
