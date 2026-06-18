fastkpc_plr_required_backends <- function() {
  c(
    "legacy-mgcv",
    "fastSplineCUDA",
    "mgcvExtractGPUFixedSP",
    "mgcvExtractGPUGCV",
    "hybrid-fastSplineCUDA-mgcvExtractGPU"
  )
}

fastkpc_plr_backend_meta <- function() {
  data.frame(
    backend = fastkpc_plr_required_backends(),
    role = c(
      "authoritative kpcalg-compatible reference",
      "frozen high-throughput approximate primary backend",
      "mgcv setup anchored fixed-sp GPU solve",
      "mgcv setup anchored single-penalty GCV bridge",
      "fastSplineCUDA primary with mgcvExtractGPU near-alpha verifier"
    ),
    supported_formula_class = c(
      "kpcalg legacy mgcv",
      "fastSpline approximation",
      "mgcv setup fixed-sp",
      "full-smooth |S| <= 2",
      "fast primary plus supported verifier"
    ),
    supported_S = c(
      "legacy",
      "approximate supported fastSpline path",
      "mgcv extracted setup",
      "|S| = 1, |S| = 2",
      "primary all supported fastSpline; verifier |S| <= 2"
    ),
    recommended_use = c(
      "reference and fallback",
      "precision = fast",
      "numerical compatibility gate",
      "precision = compatible where envelope permits",
      "precision = hybrid after calibration"
    ),
    stringsAsFactors = FALSE
  )
}

fastkpc_plr_quantile <- function(x, prob) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[is.finite(x)]
  if (length(x) == 0L) return(NA_real_)
  as.numeric(stats::quantile(x, probs = prob, names = FALSE, type = 7))
}

fastkpc_plr_first_existing <- function(df, names) {
  if (is.null(df) || !is.data.frame(df)) return(NULL)
  hit <- names[names %in% names(df)]
  if (length(hit) == 0L) return(NULL)
  hit[1L]
}

fastkpc_plr_backend_column <- function(df, fallback = "backend") {
  fastkpc_plr_first_existing(df, c("backend", "backend_used", fallback))
}

fastkpc_plr_subset_backend <- function(df, backend) {
  if (is.null(df) || !is.data.frame(df) || nrow(df) == 0L) return(df[0L, , drop = FALSE])
  backend_col <- fastkpc_plr_backend_column(df)
  if (is.null(backend_col)) return(df[0L, , drop = FALSE])
  df[as.character(df[[backend_col]]) == backend, , drop = FALSE]
}

fastkpc_plr_metric <- function(df, backend, columns, fun, default = NA_real_) {
  if (is.null(df) || !is.data.frame(df) || nrow(df) == 0L) return(default)
  col <- fastkpc_plr_first_existing(df, columns)
  if (is.null(col)) return(default)
  rows <- fastkpc_plr_subset_backend(df, backend)
  if (nrow(rows) == 0L && identical(backend, "legacy-mgcv")) rows <- df[0L, , drop = FALSE]
  if (nrow(rows) == 0L) return(default)
  fun(rows[[col]])
}

fastkpc_plr_speedup <- function(summary, backend) {
  runtime <- summary$end_to_end_runtime[summary$backend == backend]
  legacy <- summary$end_to_end_runtime[summary$backend == "legacy-mgcv"]
  if (length(runtime) == 0L || length(legacy) == 0L) return(NA_real_)
  runtime <- suppressWarnings(as.numeric(runtime[1L]))
  legacy <- suppressWarnings(as.numeric(legacy[1L]))
  if (!is.finite(runtime) || !is.finite(legacy) || runtime <= 0) return(NA_real_)
  legacy / runtime
}

fastkpc_precision_ladder_backend_summary <- function(residual_metrics = NULL,
                                                     ci_metrics = NULL,
                                                     graph_metrics = NULL,
                                                     gpu_graph_summary = NULL,
                                                     tprs_decision = NULL) {
  meta <- fastkpc_plr_backend_meta()
  out <- meta
  out$residual_rel_l2_p50 <- NA_real_
  out$residual_rel_l2_p95 <- NA_real_
  out$residual_rel_l2_max <- NA_real_
  out$log_p_drift_p50 <- NA_real_
  out$log_p_drift_p95 <- NA_real_
  out$near_alpha_flip_rate <- NA_real_
  out$skeleton_shd <- NA_real_
  out$sepset_mismatch_rate <- NA_real_
  out$wanpdag_mismatch <- NA_real_
  out$setup_time <- NA_real_
  out$solve_time <- NA_real_
  out$ci_time <- NA_real_
  out$end_to_end_runtime <- NA_real_
  out$speedup_vs_legacy <- NA_real_

  for (i in seq_len(nrow(out))) {
    backend <- out$backend[i]
    out$residual_rel_l2_p50[i] <- fastkpc_plr_metric(
      residual_metrics, backend, c("relative_l2", "residual_rel_l2"),
      function(x) fastkpc_plr_quantile(x, 0.5)
    )
    out$residual_rel_l2_p95[i] <- fastkpc_plr_metric(
      residual_metrics, backend, c("relative_l2", "residual_rel_l2"),
      function(x) fastkpc_plr_quantile(x, 0.95)
    )
    out$residual_rel_l2_max[i] <- fastkpc_plr_metric(
      residual_metrics, backend, c("relative_l2", "residual_rel_l2"),
      function(x) {
        x <- suppressWarnings(as.numeric(x))
        x <- x[is.finite(x)]
        if (length(x) == 0L) NA_real_ else max(x)
      }
    )
    out$log_p_drift_p50[i] <- fastkpc_plr_metric(
      ci_metrics, backend, c("log_p_ratio", "log_p_drift"),
      function(x) fastkpc_plr_quantile(abs(x), 0.5)
    )
    out$log_p_drift_p95[i] <- fastkpc_plr_metric(
      ci_metrics, backend, c("log_p_ratio", "log_p_drift"),
      function(x) fastkpc_plr_quantile(abs(x), 0.95)
    )
    out$near_alpha_flip_rate[i] <- fastkpc_plr_metric(
      ci_metrics, backend, c("decision_flip", "decision_flip_native"),
      function(x) mean(as.logical(x), na.rm = TRUE)
    )
    out$skeleton_shd[i] <- fastkpc_plr_metric(
      graph_metrics, backend, c("skeleton_shd"),
      function(x) fastkpc_plr_quantile(x, 0.5)
    )
    out$sepset_mismatch_rate[i] <- fastkpc_plr_metric(
      graph_metrics, backend, c("sepset_mismatch_rate"),
      function(x) fastkpc_plr_quantile(x, 0.5)
    )
    out$wanpdag_mismatch[i] <- fastkpc_plr_metric(
      graph_metrics, backend, c("wanpdag_mismatch", "wanpdag_orientation_mismatch"),
      function(x) fastkpc_plr_quantile(x, 0.5)
    )
    out$end_to_end_runtime[i] <- fastkpc_plr_metric(
      gpu_graph_summary, backend, c("runtime_sec", "runtime", "hybrid_runtime_sec"),
      function(x) fastkpc_plr_quantile(x, 0.5)
    )
  }

  for (i in seq_len(nrow(out))) {
    out$speedup_vs_legacy[i] <- fastkpc_plr_speedup(out, out$backend[i])
  }

  out
}

fastkpc_plr_markdown_table <- function(df) {
  shown <- df
  shown[] <- lapply(shown, function(x) {
    if (is.numeric(x)) x <- signif(x, 6)
    x <- as.character(x)
    x[is.na(x)] <- ""
    x
  })
  header <- paste(names(shown), collapse = " | ")
  divider <- paste(rep("---", ncol(shown)), collapse = " | ")
  rows <- if (nrow(shown) == 0L) "" else
    paste(apply(shown, 1L, paste, collapse = " | "), collapse = " |\n| ")
  paste0("| ", header, " |\n| ", divider, " |\n",
         if (nzchar(rows)) paste0("| ", rows, " |\n") else "")
}

fastkpc_write_precision_ladder_summary_report <- function(
  residual_metrics = NULL,
  ci_metrics = NULL,
  graph_metrics = NULL,
  gpu_graph_summary = NULL,
  tprs_decision = NULL,
  output_dir = file.path("fastkpc", "artifacts", "precision_ladder_summary")
) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  summary <- fastkpc_precision_ladder_backend_summary(
    residual_metrics = residual_metrics,
    ci_metrics = ci_metrics,
    graph_metrics = graph_metrics,
    gpu_graph_summary = gpu_graph_summary,
    tprs_decision = tprs_decision
  )
  summary_csv <- file.path(output_dir, "precision_ladder_backend_summary.csv")
  report_path <- file.path(output_dir, "precision_ladder_summary_report.md")
  utils::write.csv(summary, summary_csv, row.names = FALSE, na = "")

  decision <- "defer"
  if (is.data.frame(tprs_decision) && "decision" %in% names(tprs_decision) &&
      nrow(tprs_decision) > 0L) {
    decision <- as.character(tprs_decision$decision[1L])
  }

  lines <- c(
    "# fastkpc Precision Ladder Summary",
    "",
    "mgcvExtractGPU is a compatibility bridge anchored by mgcv setup.",
    "same-setup native batch is not a true fused/batched GPU kernel.",
    "",
    "## Backend Comparison",
    "",
    fastkpc_plr_markdown_table(summary),
    "",
    "## Stratification",
    "",
    "- |S| = 1",
    "- |S| = 2",
    "- |S| > 2",
    "- number of targets sharing setup",
    "- n",
    "- basis dimension / null-space dimension",
    "- CI method: dCov / HSIC",
    "- case class: well-conditioned / difficult",
    "",
    "## tprsApproxCUDA",
    "",
    paste0("Decision: ", decision),
    "",
    "tprsApproxCUDA remains deferred unless evidence reverses the decision.",
    "fastSplineCUDA remains the frozen approximate primary baseline."
  )
  writeLines(lines, report_path)
  list(report_path = report_path, summary_csv = summary_csv, summary = summary)
}
