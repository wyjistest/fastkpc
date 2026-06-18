fastkpc_tprs_approx_decision <- function(attribution,
                                         projection_floor_threshold = 0.10,
                                         oracle_improvement_threshold = 0.05,
                                         graph_shd_threshold = 1.0) {
  summary <- attribution$summary
  if (is.null(summary) || nrow(summary) == 0L) {
    stop("attribution summary is required", call. = FALSE)
  }
  mean_floor <- mean(summary$mean_basis_projection_floor)
  mean_oracle <- mean(summary$mean_oracle_lambda_improvement)
  native_flip <- mean(summary$native_decision_flip_rate)
  mean_shd <- mean(summary$mean_skeleton_shd)

  basis_dominates <- is.finite(mean_floor) &&
    mean_floor >= projection_floor_threshold
  optimizer_can_help <- is.finite(mean_oracle) &&
    mean_oracle >= oracle_improvement_threshold
  graph_drift_visible <- is.finite(mean_shd) && mean_shd >= graph_shd_threshold

  if (basis_dominates && !optimizer_can_help && graph_drift_visible) {
    decision <- "investigate"
    reason <- paste(
      "basis projection floor is high, oracle-lambda improvement is limited,",
      "and graph drift is visible; a thin-plate-like approximation may be justified"
    )
  } else {
    decision <- "defer"
    reason <- paste(
      "current attribution evidence does not isolate basis/penalty geometry as",
      "the dominant remaining drift source after mgcvExtractGPU"
    )
  }

  data.frame(
    decision = decision,
    mean_basis_projection_floor = mean_floor,
    mean_oracle_lambda_improvement = mean_oracle,
    native_decision_flip_rate = native_flip,
    mean_skeleton_shd = mean_shd,
    projection_floor_threshold = projection_floor_threshold,
    oracle_improvement_threshold = oracle_improvement_threshold,
    graph_shd_threshold = graph_shd_threshold,
    basis_dominates = basis_dominates,
    optimizer_can_help = optimizer_can_help,
    graph_drift_visible = graph_drift_visible,
    reason = reason,
    stringsAsFactors = FALSE
  )
}

fastkpc_write_tprs_approx_go_no_go_memo <- function(attribution, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  decision <- fastkpc_tprs_approx_decision(attribution)
  decision_file <- file.path(output_dir, "tprs_approx_cuda_decision.csv")
  memo_file <- file.path(output_dir, "tprs_approx_cuda_go_no_go_memo.md")
  utils::write.csv(decision, decision_file, row.names = FALSE)

  decision_text <- decision$decision[1]
  action <- if (identical(decision_text, "investigate")) {
    "Open a design investigation before implementation."
  } else {
    "Do not implement tprsApproxCUDA yet."
  }
  lines <- c(
    "# tprsApproxCUDA Go/No-Go Memo",
    "",
    paste0("Decision: ", decision_text),
    "",
    "## Evidence",
    "",
    paste0("- Mean basis projection floor: ",
           signif(decision$mean_basis_projection_floor[1], 6)),
    paste0("- Mean oracle-lambda improvement: ",
           signif(decision$mean_oracle_lambda_improvement[1], 6)),
    paste0("- Native CI decision flip rate: ",
           signif(decision$native_decision_flip_rate[1], 6)),
    paste0("- Mean skeleton SHD: ",
           signif(decision$mean_skeleton_shd[1], 6)),
    "",
    "## Interpretation",
    "",
    paste0("- ", decision$reason[1]),
    "- mgcvExtractGPU remains the compatibility bridge anchored by mgcv setup.",
    "- fastSplineCUDA remains the frozen approximate primary baseline.",
    "",
    "## Boundary",
    "",
    paste0("- ", action),
    "- Do not mutate fastSplineCUDA while using this memo as evidence.",
    "- Do not describe tprsApproxCUDA as mgcv-compatible; it would be a pure GPU approximation."
  )
  writeLines(lines, memo_file)

  list(
    decision = decision_text,
    decision_row = decision,
    memo_file = memo_file,
    decision_file = decision_file
  )
}
