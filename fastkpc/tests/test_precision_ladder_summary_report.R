fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/precision_ladder_report.R")

residual_metrics <- data.frame(
  backend = c("fastSplineCUDA", "mgcvExtractGPUFixedSP", "mgcvExtractGPUGCV"),
  relative_l2 = c(0.18, 0.001, 0.006),
  stringsAsFactors = FALSE
)
ci_metrics <- data.frame(
  backend_used = c("fastSplineCUDA", "mgcvExtractGPUGCV"),
  log_p_ratio = c(log(1.8), log(1.05)),
  decision_flip = c(TRUE, FALSE),
  stringsAsFactors = FALSE
)
graph_metrics <- data.frame(
  backend = c("fastSplineCUDA", "mgcvExtractGPUGCV"),
  skeleton_shd = c(4L, 1L),
  sepset_mismatch_rate = c(0.2, 0.05),
  wanpdag_orientation_mismatch = c(3L, 1L),
  stringsAsFactors = FALSE
)
gpu_graph_summary <- data.frame(
  backend = c("legacy-mgcv", "hybrid-fastSplineCUDA-mgcvExtractGPU"),
  runtime_sec = c(20, 4),
  stringsAsFactors = FALSE
)
tprs_decision <- data.frame(
  decision = "defer",
  reason = "insufficient evidence for pure GPU approximation",
  stringsAsFactors = FALSE
)

out <- fastkpc_write_precision_ladder_summary_report(
  residual_metrics = residual_metrics,
  ci_metrics = ci_metrics,
  graph_metrics = graph_metrics,
  gpu_graph_summary = gpu_graph_summary,
  tprs_decision = tprs_decision,
  output_dir = tempdir()
)

assert_true(file.exists(out$report_path), "summary report should exist")
assert_true(file.exists(out$summary_csv), "backend summary CSV should exist")

txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
required_phrases <- c(
  "mgcvExtractGPU is a compatibility bridge",
  "same-setup native batch is not a true fused/batched GPU kernel",
  "tprsApproxCUDA",
  "Decision: defer",
  "|S| = 1",
  "|S| = 2",
  "|S| > 2",
  "targets sharing setup",
  "basis dimension",
  "CI method"
)
for (phrase in required_phrases) {
  assert_true(grepl(phrase, txt, fixed = TRUE),
              paste("report missing phrase:", phrase))
}

summary <- utils::read.csv(out$summary_csv, stringsAsFactors = FALSE)
required_cols <- c(
  "backend", "role", "supported_formula_class", "supported_S",
  "residual_rel_l2_p50", "residual_rel_l2_p95", "residual_rel_l2_max",
  "log_p_drift_p50", "log_p_drift_p95", "near_alpha_flip_rate",
  "skeleton_shd", "sepset_mismatch_rate", "wanpdag_mismatch",
  "setup_time", "solve_time", "ci_time", "end_to_end_runtime",
  "speedup_vs_legacy", "recommended_use"
)
missing_cols <- setdiff(required_cols, names(summary))
assert_true(length(missing_cols) == 0L,
            paste("missing summary columns:", paste(missing_cols, collapse = ", ")))
expected_backends <- c(
  "fastSplineCUDA",
  "mgcvExtractGPUFixedSP",
  "mgcvExtractGPUGCV",
  "hybrid-fastSplineCUDA-mgcvExtractGPU",
  "legacy-mgcv"
)
assert_true(all(expected_backends %in% summary$backend),
            "summary should include all precision-ladder backends")

cat("PASS precision ladder summary report\n")
