fastkpc_select_default_tau <- function(campaign) {
  required <- c(
    "tau", "num_primary_decision_flips_vs_legacy",
    "num_hybrid_decision_flips_vs_legacy", "runtime_primary", "runtime_hybrid"
  )
  missing <- setdiff(required, names(campaign))
  if (length(missing) > 0L) {
    stop("campaign missing columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  flips_primary <- as.numeric(campaign$num_primary_decision_flips_vs_legacy)
  flips_hybrid <- as.numeric(campaign$num_hybrid_decision_flips_vs_legacy)
  reduction <- flips_primary - flips_hybrid
  runtime_ok <- as.numeric(campaign$runtime_hybrid) <=
    2 * as.numeric(campaign$runtime_primary)
  eligible <- which(reduction > 0 & runtime_ok)
  if (length(eligible) == 0L) return(NA_real_)
  best <- max(reduction[eligible], na.rm = TRUE)
  near_best <- eligible[reduction[eligible] >= best - 1]
  min(as.numeric(campaign$tau[near_best]), na.rm = TRUE)
}

fastkpc_write_hybrid_policy_calibration_report <- function(
  campaign,
  output_dir = file.path("fastkpc", "artifacts", "hybrid_policy_calibration")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  selected <- fastkpc_select_default_tau(campaign)
  summary <- campaign
  summary$flip_reduction <- as.numeric(summary$num_primary_decision_flips_vs_legacy) -
    as.numeric(summary$num_hybrid_decision_flips_vs_legacy)
  summary$verification_rate <- as.numeric(summary$num_verified) /
    pmax(1, as.numeric(summary$num_tests_total))
  summary$selected_default_tau <- selected
  summary_csv <- file.path(output_dir, "hybrid_policy_calibration_summary.csv")
  report_path <- file.path(output_dir, "hybrid_policy_calibration_report.md")
  utils::write.csv(summary, summary_csv, row.names = FALSE, na = "")

  selected_text <- if (is.na(selected)) "none" else signif(selected, 8)
  lines <- c(
    "# Hybrid Policy Calibration Report",
    "",
    paste0("Recommended default tau: ", selected_text),
    "",
    "The default tau is selected from campaign evidence and preserves canonical replay.",
    "Hybrid verification may replace p-values, but canonical replay order remains fixed.",
    "",
    "## Rule",
    "",
    "Choose the smallest tau within one flip of the best eligible reduction while keeping runtime_hybrid <= 2x runtime_primary."
  )
  writeLines(lines, report_path)
  list(report_path = report_path, summary_csv = summary_csv, summary = summary)
}
