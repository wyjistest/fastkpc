source("fastkpc/R/hybrid_verifier.R")

fastkpc_make_hybrid_golden_snapshot <- function(name, primary_p, verifier_p,
                                                p_legacy, S_key,
                                                alpha = 0.05,
                                                p = 4L) {
  ids <- seq_along(primary_p)
  test_plan <- data.frame(
    canonical_test_order_id = ids,
    conditioning_level = pmax(0L, lengths(strsplit(S_key, "\\|"))),
    x = c(1L, 1L, 2L, 2L, 3L, 1L)[ids],
    y = c(2L, 3L, 3L, 4L, 4L, 4L)[ids],
    S_key = S_key,
    p_legacy = p_legacy,
    primary_p = primary_p,
    verifier_p = verifier_p,
    stringsAsFactors = FALSE
  )
  verifier <- test_plan[is.finite(test_plan$verifier_p),
                        c("canonical_test_order_id", "verifier_p")]
  verifier$verifier_backend <- "mgcvExtractGCVBridge"
  resolved <- fastkpc_apply_hybrid_verifier(
    test_plan[, c("canonical_test_order_id", "conditioning_level",
                  "x", "y", "S_key", "primary_p")],
    verifier,
    fastkpc_hybrid_policy(alpha = alpha, tau = log(3),
                          primary = "fastSplineCUDA",
                          verifier = "mgcvExtractGCVBridge")
  )
  replay <- fastkpc_replay_canonical_ci_decisions(resolved, alpha = alpha, p = p)
  wanpdag <- replay$adjacency * 1L
  storage.mode(wanpdag) <- "integer"
  list(
    name = name,
    alpha = alpha,
    test_plan = test_plan,
    p_source_used = replay$diagnostics$p_source_used,
    edge_deletion_log = replay$diagnostics,
    sepsets = replay$sepsets,
    skeleton_adjacency = replay$adjacency,
    wanpdag_adjacency = wanpdag
  )
}

fastkpc_hybrid_golden_snapshots <- function() {
  list(
    scenario_linear_small = fastkpc_make_hybrid_golden_snapshot(
      "scenario_linear_small",
      primary_p = c(0.08, 0.001, 0.20, 0.049),
      verifier_p = c(NA, NA, NA, 0.20),
      p_legacy = c(0.08, 0.001, 0.20, 0.20),
      S_key = c("", "", "1", "1|3")
    ),
    scenario_nonlinear_additive = fastkpc_make_hybrid_golden_snapshot(
      "scenario_nonlinear_additive",
      primary_p = c(0.051, 0.04, 0.90, 0.02, 0.20),
      verifier_p = c(0.001, 0.08, NA, NA, NA),
      p_legacy = c(0.001, 0.08, 0.90, 0.02, 0.20),
      S_key = c("3", "4", "1|4", "", "1|2")
    ),
    scenario_pairwise_full_smooth = fastkpc_make_hybrid_golden_snapshot(
      "scenario_pairwise_full_smooth",
      primary_p = c(0.001, 0.052, 0.18, 0.049, 0.90),
      verifier_p = c(NA, 0.20, NA, 0.001, NA),
      p_legacy = c(0.001, 0.20, 0.18, 0.001, 0.90),
      S_key = c("", "2", "1|3", "2|4", "")
    ),
    scenario_near_alpha_flip = fastkpc_make_hybrid_golden_snapshot(
      "scenario_near_alpha_flip",
      primary_p = c(0.049, 0.051, 0.001, 0.20, 0.052, 0.02),
      verifier_p = c(0.20, 0.001, NA, NA, 0.20, NA),
      p_legacy = c(0.20, 0.001, 0.001, 0.20, 0.20, 0.02),
      S_key = c("3", "2", "", "1", "1|3", "2|3")
    )
  )
}

fastkpc_replay_golden_snapshot <- function(snapshot) {
  verifier <- snapshot$test_plan[is.finite(snapshot$test_plan$verifier_p),
                                 c("canonical_test_order_id", "verifier_p")]
  verifier$verifier_backend <- "mgcvExtractGCVBridge"
  resolved <- fastkpc_apply_hybrid_verifier(
    snapshot$test_plan[, c("canonical_test_order_id", "conditioning_level",
                           "x", "y", "S_key", "primary_p")],
    verifier,
    fastkpc_hybrid_policy(alpha = snapshot$alpha, tau = log(3),
                          primary = "fastSplineCUDA",
                          verifier = "mgcvExtractGCVBridge")
  )
  fastkpc_replay_canonical_ci_decisions(
    resolved,
    alpha = snapshot$alpha,
    p = nrow(snapshot$skeleton_adjacency)
  )
}
