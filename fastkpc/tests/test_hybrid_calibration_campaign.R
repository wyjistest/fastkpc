source("fastkpc/R/hybrid_calibration_campaign.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

out_dir <- tempfile("fastkpc-hybrid-calibration-")
dir.create(out_dir, recursive = TRUE)

campaign <- fastkpc_run_hybrid_calibration_campaign(
  output_dir = out_dir,
  seeds = 11L,
  n_values = 100L,
  p_values = c(8L, 12L),
  alpha_values = c(0.01, 0.05),
  tau_values = log(c(1.5, 2, 3)),
  max_conditioning_levels = c(1L, 2L)
)

summary_file <- file.path(out_dir, "hybrid_calibration_summary.csv")
assert_true(file.exists(summary_file), "summary CSV should exist")
assert_true(nrow(campaign$summary) == 24L, "expected calibration grid size")

required <- c(
  "scenario_id", "seed", "n", "p", "alpha", "tau",
  "backend_primary", "backend_verifier",
  "num_tests_total", "num_near_alpha", "near_alpha_rate",
  "num_verified", "verification_rate",
  "num_primary_decision_flips_vs_legacy",
  "num_hybrid_decision_flips_vs_legacy",
  "flip_reduction",
  "skeleton_shd_primary", "skeleton_shd_hybrid",
  "sepset_mismatch_primary", "sepset_mismatch_hybrid",
  "wanpdag_mismatch_primary", "wanpdag_mismatch_hybrid",
  "runtime_primary", "runtime_hybrid", "runtime_legacy",
  "speedup_vs_legacy", "recommended"
)
missing <- setdiff(required, names(campaign$summary))
assert_true(length(missing) == 0L,
            paste("missing fields:", paste(missing, collapse = ", ")))

assert_true(all(campaign$summary$num_hybrid_decision_flips_vs_legacy <=
                  campaign$summary$num_primary_decision_flips_vs_legacy),
            "hybrid should not increase decision flips in deterministic calibration")
assert_true(any(campaign$summary$flip_reduction > 0),
            "some tau values should reduce flips")
assert_true(sum(campaign$summary$recommended) >= 1L,
            "campaign should mark a recommended tau")

report_file <- file.path(out_dir, "hybrid_policy_summary.txt")
assert_true(file.exists(report_file), "policy summary report should exist")
report <- paste(readLines(report_file, warn = FALSE), collapse = "\n")
assert_true(grepl("Recommended tau", report, fixed = TRUE),
            "policy report should include recommendation")

cat("PASS hybrid calibration campaign\n")
