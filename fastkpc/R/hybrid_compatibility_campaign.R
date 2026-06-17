source("fastkpc/R/hybrid_verifier.R")
source("fastkpc/R/mgcv_extract_validation.R")

fastkpc_hybrid_demo_rows <- function(alpha = 0.05) {
  data.frame(
    canonical_test_order_id = seq_len(6),
    conditioning_level = c(0L, 0L, 1L, 1L, 2L, 2L),
    x = c(1L, 1L, 1L, 2L, 2L, 3L),
    y = c(2L, 3L, 2L, 3L, 4L, 4L),
    S_key = c("", "", "3", "1", "1|3", "1|2"),
    p_legacy = c(0.90, 0.001, 0.001, 0.08, 0.20, 0.02),
    primary_p = c(0.90, 0.001, 0.051, 0.04, 0.20, 0.02),
    verifier_p = c(NA, NA, 0.001, 0.08, NA, NA),
    stringsAsFactors = FALSE
  )
}

fastkpc_run_hybrid_compatibility_campaign <- function(output_dir,
                                                      alpha = 0.05,
                                                      tau = log(3)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  rows <- fastkpc_hybrid_demo_rows(alpha)
  policy <- fastkpc_hybrid_policy(alpha = alpha, tau = tau,
                                  primary = "fastSplineCPU",
                                  verifier = "mgcvExtractCPU")
  verifier <- rows[is.finite(rows$verifier_p),
                   c("canonical_test_order_id", "verifier_p")]
  verifier$verifier_backend <- "mgcvExtractCPU"
  resolved <- fastkpc_apply_hybrid_verifier(
    rows[, c("canonical_test_order_id", "conditioning_level", "x", "y", "S_key", "primary_p")],
    verifier,
    policy
  )
  replay <- fastkpc_replay_canonical_ci_decisions(resolved, alpha = alpha, p = 4L)
  ci <- data.frame(
    canonical_test_order_id = rows$canonical_test_order_id,
    x = rows$x,
    y = rows$y,
    S_key = rows$S_key,
    conditioning_level = rows$conditioning_level,
    p_legacy = rows$p_legacy,
    p_backend = rows$primary_p,
    p_hybrid = resolved$p_used,
    decision_legacy = rows$p_legacy > alpha,
    decision_backend = rows$primary_p > alpha,
    decision_hybrid = resolved$p_used > alpha,
    decision_flip = (rows$p_legacy > alpha) != (rows$primary_p > alpha),
    hybrid_flip = (rows$p_legacy > alpha) != (resolved$p_used > alpha),
    near_alpha_triggered = resolved$near_alpha_triggered,
    p_source_used = resolved$p_source_used,
    stringsAsFactors = FALSE
  )
  residual <- data.frame(
    scenario = "demo",
    backend = c("fastSplineCPU", "mgcvExtractCPU"),
    residual_correlation = c(0.99, 1.0),
    relative_l2 = c(0.15, 1e-8),
    stringsAsFactors = FALSE
  )
  graph <- data.frame(
    scenario = "demo",
    backend = "hybrid",
    skeleton_shd = sum(ci$hybrid_flip),
    near_alpha_tests = sum(resolved$near_alpha_triggered),
    verifier_calls = sum(is.finite(resolved$verifier_p)),
    verifier_decision_changes = sum(resolved$decision_before_verify != resolved$decision_after_verify),
    stringsAsFactors = FALSE
  )
  summary <- data.frame(
    decision_flip_rate = mean(ci$decision_flip),
    near_alpha_fraction = mean(resolved$near_alpha_triggered),
    verifier_decision_changes = sum(resolved$decision_before_verify != resolved$decision_after_verify),
    stringsAsFactors = FALSE
  )
  utils::write.csv(residual, file.path(output_dir, "mgcv_residual_compatibility.csv"), row.names = FALSE)
  utils::write.csv(ci, file.path(output_dir, "mgcv_ci_compatibility.csv"), row.names = FALSE)
  utils::write.csv(graph, file.path(output_dir, "mgcv_graph_compatibility.csv"), row.names = FALSE)
  utils::write.csv(replay$diagnostics, file.path(output_dir, "hybrid_near_alpha_diagnostics.csv"), row.names = FALSE)
  utils::write.csv(summary, file.path(output_dir, "hybrid_summary.csv"), row.names = FALSE)
  list(residual = residual, ci = ci, graph = graph,
       hybrid = replay$diagnostics, summary = summary,
       output_dir = output_dir)
}
