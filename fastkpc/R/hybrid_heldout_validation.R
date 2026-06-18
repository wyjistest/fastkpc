fastkpc_select_tau_lexicographic <- function(calibration, max_runtime_ratio = 2) {
  required <- c("tau", "decision_flip_rate_primary",
                "decision_flip_rate_hybrid", "skeleton_shd_primary",
                "skeleton_shd_hybrid", "verification_rate", "runtime_ratio")
  missing <- setdiff(required, names(calibration))
  if (length(missing) > 0L) {
    stop("calibration missing columns: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  graph_ok <- calibration$skeleton_shd_hybrid <= calibration$skeleton_shd_primary
  runtime_ok <- calibration$runtime_ratio <= max_runtime_ratio
  eligible <- calibration[graph_ok & runtime_ok, , drop = FALSE]
  if (nrow(eligible) == 0L) return(NA_real_)
  eligible$flip_reduction <- eligible$decision_flip_rate_primary -
    eligible$decision_flip_rate_hybrid
  eligible <- eligible[order(-eligible$flip_reduction,
                             eligible$verification_rate,
                             eligible$tau), , drop = FALSE]
  eligible$tau[1L]
}

fastkpc_validate_hybrid_tau_heldout <- function(calibration, heldout,
                                                max_runtime_ratio = 2) {
  selected <- fastkpc_select_tau_lexicographic(calibration, max_runtime_ratio)
  held <- heldout[abs(heldout$tau - selected) < 1e-12, , drop = FALSE]
  if (nrow(held) == 0L) {
    return(list(
      selected_tau = selected,
      heldout_pass = FALSE,
      recommendation = "no held-out row for selected experimental tau",
      heldout = held
    ))
  }
  graph_ok <- held$skeleton_shd_hybrid <= held$skeleton_shd_primary &&
    held$sepset_mismatch_hybrid <= held$sepset_mismatch_primary &&
    held$wanpdag_mismatch_hybrid <= held$wanpdag_mismatch_primary
  runtime_ok <- held$runtime_ratio <= max_runtime_ratio
  flip_ok <- held$decision_flip_rate_hybrid <= held$decision_flip_rate_primary
  pass <- isTRUE(graph_ok && runtime_ok && flip_ok)
  list(
    selected_tau = selected,
    heldout_pass = pass,
    recommendation = if (pass) {
      paste("experimental tau", signif(selected, 8), "validated on held-out scenarios")
    } else {
      paste("experimental tau", signif(selected, 8), "failed held-out validation")
    },
    heldout = held
  )
}

fastkpc_write_hybrid_heldout_validation_report <- function(
  result,
  output_dir = file.path("fastkpc", "artifacts", "hybrid_heldout_validation")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  report_path <- file.path(output_dir, "hybrid_heldout_validation.md")
  lines <- c(
    "# Hybrid Held-out Tau Validation",
    "",
    paste0("Selected experimental tau: ", signif(result$selected_tau, 8)),
    paste0("Held-out pass: ", isTRUE(result$heldout_pass)),
    "",
    result$recommendation
  )
  writeLines(lines, report_path)
  list(report_path = report_path, result = result)
}
