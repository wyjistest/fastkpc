source("fastkpc/R/hybrid_verifier.R")
source("fastkpc/R/precision_data_plane.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

alpha <- 0.05
eps <- .Machine$double.eps ^ 0.5
cases <- data.frame(
  label = c("below", "equal", "above", "na", "nan", "inf"),
  p_raw = c(alpha - eps, alpha, alpha + eps, NA_real_, NaN, Inf),
  delete_edge = c(FALSE, TRUE, TRUE, TRUE, TRUE, TRUE),
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(cases))) {
  decision <- fastkpc_resolve_ci_decision(cases$p_raw[i], alpha = alpha,
                                          na_delete = TRUE)
  assert_true(identical(decision$delete_edge, cases$delete_edge[i]),
              paste("decision mismatch for", cases$label[i]))
  assert_true(identical(decision$independent, cases$delete_edge[i]),
              paste("independence mismatch for", cases$label[i]))
  assert_true(decision$boundary_rule == "p_used >= alpha",
              "decision contract should disclose boundary rule")
}

na_keep <- fastkpc_resolve_ci_decision(NA_real_, alpha = alpha,
                                       na_delete = FALSE)
assert_true(!na_keep$delete_edge,
            "NAdelete=FALSE should not delete edge for NA p-value")
assert_true(na_keep$nonfinite_action == "na-keep-use-0",
            "NAdelete=FALSE should document nonfinite action")

rows <- data.frame(
  canonical_test_order_id = seq_len(nrow(cases)),
  x = rep(1L, nrow(cases)),
  y = seq_len(nrow(cases)) + 1L,
  S_key = "",
  primary_p = cases$p_raw,
  verifier_p = NA_real_,
  stringsAsFactors = FALSE
)
policy <- fastkpc_hybrid_policy(enabled = TRUE, alpha = alpha, tau = log(2))
resolved <- fastkpc_apply_hybrid_policy(rows, policy)
assert_true(identical(as.logical(resolved$decision_before_verify),
                      cases$delete_edge),
            "hybrid helper should use same alpha/nonfinite decision contract")

replay_rows <- data.frame(
  canonical_test_order_id = seq_len(nrow(cases)),
  x = rep(1L, nrow(cases)),
  y = seq_len(nrow(cases)) + 1L,
  S_key = "",
  p_used = cases$p_raw,
  stringsAsFactors = FALSE
)
replayed <- fastkpc_replay_canonical_ci_decisions(
  replay_rows, alpha = alpha, p = nrow(cases) + 1L
)$diagnostics
assert_true(identical(as.logical(replayed$edge_deleted), cases$delete_edge),
            "canonical replay should use same alpha/nonfinite decision contract")

cat("PASS precision CI decision contract\n")
