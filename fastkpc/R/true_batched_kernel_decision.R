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

fastkpc_true_batched_kernel_decision_scenario_aligned <- function(
  timing,
  workload,
  by = c("scenario_id", "dataset_id", "backend", "conditioning_level"),
  linear_solve_fraction_threshold = 0.5,
  uncached_targets_per_setup_p95_threshold = 4,
  supported_wall_time_fraction_threshold = 0.75,
  min_evidence_runs = 3L
) {
  if (is.null(timing) || is.null(workload) || nrow(timing) == 0L ||
      nrow(workload) == 0L) {
    evidence <- data.frame()
    return(list(
      decision = "insufficient-evidence",
      rationale = "missing timing or workload rows",
      evidence = evidence
    ))
  }
  by <- by[by %in% names(timing) & by %in% names(workload)]
  if (length(by) == 0L) {
    stop("no shared alignment keys between timing and workload", call. = FALSE)
  }
  evidence <- merge(timing, workload, by = by, suffixes = c("_timing", "_workload"))
  if (nrow(evidence) == 0L) {
    return(list(
      decision = "insufficient-evidence",
      rationale = "no scenario-aligned timing/workload evidence",
      evidence = evidence
    ))
  }
  total_ms <- pmax(as.numeric(evidence$total_ms), .Machine$double.eps)
  evidence$linear_solve_fraction <- as.numeric(evidence$linear_solve_ms) / total_ms
  evidence$setup_dominated <- as.numeric(evidence$mgcv_setup_cpu_ms) >=
    pmax(as.numeric(evidence$linear_solve_ms), as.numeric(evidence$ci_test_ms),
         na.rm = TRUE)
  weights <- total_ms / sum(total_ms, na.rm = TRUE)
  weighted_linear <- sum(weights * evidence$linear_solve_fraction, na.rm = TRUE)
  weighted_targets <- sum(weights * as.numeric(evidence$uncached_targets_per_setup_p95),
                          na.rm = TRUE)
  supported_fraction <- sum(weights * as.numeric(evidence$supported_wall_time_fraction),
                            na.rm = TRUE)
  evidence_runs <- sum(as.integer(evidence$evidence_runs), na.rm = TRUE)

  if (evidence_runs < min_evidence_runs) {
    decision <- "insufficient-evidence"
    rationale <- "not enough scenario-aligned evidence runs"
  } else if (any(evidence$setup_dominated %in% TRUE)) {
    decision <- "defer"
    rationale <- "mgcv setup dominated scenario remains in aligned evidence"
  } else if (weighted_linear < linear_solve_fraction_threshold) {
    decision <- "defer"
    rationale <- "weighted linear_solve_ms fraction is below threshold"
  } else if (weighted_targets < uncached_targets_per_setup_p95_threshold) {
    decision <- "defer"
    rationale <- "weighted uncached target multiplicity is below threshold"
  } else if (supported_fraction < supported_wall_time_fraction_threshold) {
    decision <- "defer"
    rationale <- "supported wall-time fraction is below threshold"
  } else {
    decision <- "proceed"
    rationale <- "scenario-aligned evidence supports true batched kernel investigation"
  }

  list(
    decision = decision,
    rationale = rationale,
    weighted_linear_solve_fraction = weighted_linear,
    weighted_uncached_targets_per_setup_p95 = weighted_targets,
    supported_wall_time_fraction = supported_fraction,
    evidence = evidence
  )
}
