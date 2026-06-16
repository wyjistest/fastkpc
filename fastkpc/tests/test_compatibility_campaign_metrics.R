source("fastkpc/R/mgcv_compat_contract.R")
source("fastkpc/R/mgcv_extract_validation.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

sample <- fastkpc_empty_compatibility_campaign_metrics()
required_residual <- c("scenario", "target", "S_key", "backend",
                       "residual_correlation", "relative_l2",
                       "max_abs_diff", "mean_diff", "sd_ratio",
                       "selected_sp", "edf", "score",
                       "setup_fingerprint", "target_fingerprint")
required_ci <- c("canonical_test_order_id", "x", "y", "S_key",
                 "conditioning_level", "p_legacy", "p_backend",
                 "log_p_ratio", "decision_legacy", "decision_backend",
                 "decision_flip", "distance_to_alpha_log",
                 "backend_used", "fallback_triggered", "verifier_backend")
required_graph <- c("scenario", "backend", "skeleton_shd",
                    "skeleton_precision", "skeleton_recall", "skeleton_f1",
                    "edge_deletion_mismatch", "sepset_mismatch_rate",
                    "first_separating_set_mismatch",
                    "wanpdag_orientation_mismatch",
                    "arrowhead_agreement", "near_alpha_tests",
                    "verifier_calls", "verifier_decision_changes")

assert_true(all(required_residual %in% names(sample$residual)),
            "residual metric columns missing")
assert_true(all(required_ci %in% names(sample$ci)),
            "CI metric columns missing")
assert_true(all(required_graph %in% names(sample$graph)),
            "graph metric columns missing")

p <- fastkpc_log_distance_to_alpha(p = 0.10, alpha = 0.05)
assert_true(abs(p - log(2)) < 1e-12, "log alpha distance must be log(p/alpha)")

row <- fastkpc_make_ci_compatibility_row(
  canonical_test_order_id = 7L,
  x = 1L,
  y = 2L,
  S = c(4L, 3L),
  conditioning_level = 2L,
  p_legacy = 0.04,
  p_backend = 0.08,
  alpha = 0.05,
  backend_used = "fastSplineCUDA",
  fallback_triggered = TRUE,
  verifier_backend = "mgcvExtractCPU"
)
assert_true(row$decision_flip, "decision flip should be TRUE")
assert_true(identical(row$S_key, "3|4"), "S key should be sorted")

cat("PASS compatibility campaign metrics\n")
