args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1L) {
  args[[1]]
} else {
  file.path("fastkpc", "artifacts", "tprs_approx_decision")
}

source("fastkpc/R/mgcv_extract_validation.R")
source("fastkpc/R/tprs_approx_decision_memo.R")

attribution <- fastkpc_run_precision_ladder_attribution_campaign(
  output_dir = output_dir
)
memo <- fastkpc_write_tprs_approx_go_no_go_memo(
  attribution = attribution,
  output_dir = output_dir
)
cat("wrote tprsApproxCUDA decision memo:", memo$memo_file, "\n")
print(memo$decision_row)
