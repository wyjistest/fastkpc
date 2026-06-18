fastkpc_true_batched_kernel_decision <- function(
  timing,
  workload,
  linear_solve_fraction_threshold = 0.5,
  targets_per_setup_p95_threshold = 4,
  unsupported_fraction_threshold = 0.25
) {
  if (is.null(timing) || nrow(timing) == 0L || is.null(workload) ||
      nrow(workload) == 0L) {
    return(data.frame(
      decision = "defer",
      rationale = "insufficient timing/workload evidence",
      linear_solve_fraction = NA_real_,
      targets_per_setup_p95 = NA_real_,
      unsupported_fraction = NA_real_,
      stringsAsFactors = FALSE
    ))
  }
  row <- timing[which.max(as.numeric(timing$total_ms)), , drop = FALSE]
  total <- max(as.numeric(row$total_ms[1L]), .Machine$double.eps)
  linear_fraction <- as.numeric(row$linear_solve_ms[1L]) / total
  setup_ms <- if ("mgcv_setup_cpu_ms" %in% names(row)) as.numeric(row$mgcv_setup_cpu_ms[1L]) else 0
  ci_ms <- if ("ci_test_ms" %in% names(row)) as.numeric(row$ci_test_ms[1L]) else 0
  linear_ms <- as.numeric(row$linear_solve_ms[1L])
  targets_p95 <- max(as.numeric(workload$targets_per_setup_p95), na.rm = TRUE)
  supported <- sum(as.numeric(workload$mgcvExtractGPU_supported_tests), na.rm = TRUE)
  unsupported <- sum(as.numeric(workload$mgcvExtractGPU_unsupported_tests), na.rm = TRUE)
  unsupported_fraction <- unsupported / max(1, supported + unsupported)

  if (setup_ms >= max(linear_ms, ci_ms, na.rm = TRUE)) {
    decision <- "defer"
    rationale <- "mgcv setup dominates total timing; true batched kernel would not address main bottleneck"
  } else if (ci_ms >= max(linear_ms, setup_ms, na.rm = TRUE)) {
    decision <- "defer"
    rationale <- "CI test time dominates total timing; true batched kernel would not address main bottleneck"
  } else if (!is.finite(linear_fraction) ||
             linear_fraction < linear_solve_fraction_threshold) {
    decision <- "defer"
    rationale <- "linear_solve_ms does not dominate enough to justify true batched kernel"
  } else if (!is.finite(targets_p95) ||
             targets_p95 < targets_per_setup_p95_threshold) {
    decision <- "defer"
    rationale <- "same-setup targets_per_setup_p95 is too low to amortize true batched kernel"
  } else if (unsupported_fraction > unsupported_fraction_threshold) {
    decision <- "defer"
    rationale <- "too many verifier tests are unsupported by mgcvExtractGPU envelope"
  } else {
    decision <- "proceed"
    rationale <- "linear_solve_ms dominates and same-setup multiplicity is high enough to justify true batched kernel investigation"
  }

  data.frame(
    decision = decision,
    rationale = rationale,
    linear_solve_fraction = linear_fraction,
    targets_per_setup_p95 = targets_p95,
    unsupported_fraction = unsupported_fraction,
    stringsAsFactors = FALSE
  )
}

fastkpc_write_true_batched_kernel_decision <- function(
  decision,
  output_dir = file.path("fastkpc", "artifacts", "true_batched_kernel_decision")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  csv_path <- file.path(output_dir, "true_batched_kernel_decision.csv")
  report_path <- file.path(output_dir, "true_batched_kernel_decision.md")
  utils::write.csv(decision, csv_path, row.names = FALSE, na = "")
  lines <- c(
    "# true batched mgcvExtractGPU kernel decision",
    "",
    paste0("Decision: ", decision$decision[1L]),
    "",
    "## Rationale",
    "",
    paste0("- ", decision$rationale[1L]),
    "",
    "No fused/batched mgcvExtractGPU kernel work should start before this artifact is reviewed."
  )
  writeLines(lines, report_path)
  list(csv_path = csv_path, report_path = report_path, decision = decision)
}
