source("fastkpc/R/workload_structure_stats.R")

output_dir <- Sys.getenv(
  "FASTKPC_WORKLOAD_STATS_DIR",
  file.path("fastkpc", "artifacts", "workload_structure_stats")
)

rdata_path <- Sys.getenv("FASTKPC_WORKLOAD_RDATA", "")
default_rdata <- "/data/wenyujianData/zhuData/2025/causalDiscoveryInput.RData"
source_label <- "synthetic fallback"
if (!nzchar(rdata_path) && file.exists(default_rdata)) rdata_path <- default_rdata
if (nzchar(rdata_path) && file.exists(rdata_path)) {
  loaded <- load(rdata_path)
  source_label <- paste("local RData", rdata_path)
  data_obj <- NULL
  for (name in loaded) {
    value <- get(name)
    if (is.data.frame(value) || is.matrix(value)) {
      data_obj <- value
      break
    }
  }
  n <- if (is.null(data_obj)) 100L else nrow(data_obj)
  p <- if (is.null(data_obj)) 8L else ncol(data_obj)
} else {
  n <- 100L
  p <- 8L
}

test_plan <- data.frame(
  canonical_test_order_id = seq_len(10),
  x = c(1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
  y = c(2, 3, 3, 4, 4, 5, 5, 6, 6, 7),
  S_key = c("1", "1", "1,2", "1,2", "1,2,3", "1,2,3", "2", "2", "3", "3,4"),
  S_size = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L, 1L, 2L),
  conditioning_level = c(1L, 1L, 2L, 2L, 3L, 3L, 1L, 1L, 1L, 2L),
  near_alpha = c(TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, TRUE),
  verifier_called = c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE),
  mgcvExtractGPU_supported = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

stats <- fastkpc_workload_structure_stats(
  test_plan = test_plan,
  dataset_id = source_label,
  n = n,
  p = p,
  alpha = 0.05,
  max_conditioning_level = 3L
)
out <- fastkpc_write_workload_structure_stats(stats, output_dir = output_dir)
cat("workload source:", source_label, "\n")
cat("workload stats report:", out$report_path, "\n")
cat("workload stats CSV:", out$csv_path, "\n")
