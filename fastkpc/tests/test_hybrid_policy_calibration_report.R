fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

source("fastkpc/R/hybrid_policy_calibration_report.R")

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

out <- fastkpc_write_hybrid_policy_calibration_report(
  campaign,
  output_dir = tempdir()
)
assert_true(file.exists(out$report_path), "hybrid policy report should exist")
assert_true(file.exists(out$summary_csv), "hybrid policy summary CSV should exist")

summary <- utils::read.csv(out$summary_csv, stringsAsFactors = FALSE)
assert_true("selected_default_tau" %in% names(summary),
            "summary should include selected_default_tau")
assert_true(any(abs(summary$selected_default_tau - log(2)) < 1e-12),
            "selection rule should choose log(2) for this campaign")

txt <- paste(readLines(out$report_path, warn = FALSE), collapse = "\n")
assert_true(grepl("default tau", txt, ignore.case = TRUE),
            "report should discuss default tau")
assert_true(grepl("canonical replay", txt, fixed = TRUE),
            "report should mention canonical replay")

cat("PASS hybrid policy calibration report\n")
