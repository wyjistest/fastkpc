source("fastkpc/R/hybrid_verifier.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_equal <- function(actual, expected, message) {
  if (!identical(actual, expected)) {
    fail(paste0(message, ": expected ", paste(expected, collapse = ","),
                " got ", paste(actual, collapse = ",")))
  }
}

policy <- fastkpc_hybrid_policy(alpha = 0.05, tau = log(3),
                                primary = "fastSplineCPU",
                                verifier = "mgcvExtractCPU")

primary <- data.frame(
  canonical_test_order_id = c(1L, 2L, 3L, 4L),
  conditioning_level = c(1L, 1L, 1L, 1L),
  x = c(1L, 1L, 1L, 2L),
  y = c(2L, 2L, 2L, 3L),
  S_key = c("3", "4", "5", "1"),
  primary_p = c(0.051, 0.90, 0.20, 0.049),
  stringsAsFactors = FALSE
)
verifier <- data.frame(
  canonical_test_order_id = c(4L, 1L),
  verifier_p = c(0.70, 0.001),
  verifier_backend = c("mgcvExtractCPU", "mgcvExtractCPU"),
  stringsAsFactors = FALSE
)

resolved <- fastkpc_apply_hybrid_verifier(primary, verifier, policy)
assert_equal(resolved$canonical_test_order_id, c(1L, 2L, 3L, 4L),
             "resolved rows must follow primary canonical order")
assert_true(resolved$near_alpha_triggered[1], "row 1 should trigger near alpha")
assert_true(resolved$p_used[1] < policy$alpha,
            "row 1 verifier should prevent deletion")
assert_true(resolved$p_used[2] > policy$alpha,
            "row 2 primary should delete edge")
assert_true(resolved$p_used[4] > policy$alpha,
            "row 4 verifier should delete edge despite primary")

replay <- fastkpc_replay_canonical_ci_decisions(
  resolved,
  alpha = policy$alpha,
  p = 5L
)
assert_true(replay$adjacency[1, 2] == FALSE && replay$adjacency[2, 1] == FALSE,
            "edge 1-2 should be deleted")
assert_equal(replay$sepsets[[1]][[2]], 4L,
             "edge 1-2 sepset must be canonical first accepted S=4")
assert_true(replay$diagnostics$edge_deleted[2],
            "row 2 should delete edge 1-2")
assert_true(replay$diagnostics$edge_already_deleted[3],
            "row 3 should be ignored after canonical deletion")
assert_true(replay$adjacency[2, 3] == FALSE && replay$adjacency[3, 2] == FALSE,
            "edge 2-3 should be deleted by verifier row")
assert_equal(replay$sepsets[[2]][[3]], 1L,
             "edge 2-3 sepset must be S=1")

cat("PASS hybrid canonical replay\n")
