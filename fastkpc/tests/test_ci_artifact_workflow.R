workflow <- ".github/workflows/fastkpc-cpu.yml"

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

assert_true(file.exists(workflow), "CPU workflow should exist")
text <- paste(readLines(workflow, warn = FALSE), collapse = "\n")

for (pattern in c(
  "run_mgcv_gate_b_tests.sh",
  "run_mgcv_gate_b_campaign.sh",
  "run_hybrid_compatibility_campaign.sh",
  "run_hybrid_calibration_campaign.sh",
  "run_precision_ladder_attribution_campaign.sh",
  "hybrid_calibration_summary.csv",
  "precision_ladder_attribution_summary.csv",
  "mgcv_gate_b_fixed_sp_campaign.csv",
  "FASTKPC_RUN_CUDA_TESTS"
)) {
  assert_true(grepl(pattern, text, fixed = TRUE),
              paste("workflow missing", pattern))
}

cat("PASS CI artifact workflow\n")
