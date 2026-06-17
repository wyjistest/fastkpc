source("fastkpc/R/hybrid_verifier.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

policy <- fastkpc_hybrid_policy(alpha = 0.05, tau = log(3),
                                primary = "fastSplineCPU",
                                verifier = "mgcvExtractCPU")

primary <- data.frame(
  canonical_test_order_id = c(1L, 2L, 3L),
  conditioning_level = c(1L, 1L, 1L),
  x = c(1L, 1L, 1L),
  y = c(2L, 2L, 3L),
  S_key = c("3", "4", "2"),
  primary_p = c(0.051, 0.90, 0.049),
  stringsAsFactors = FALSE
)

primary_only <- fastkpc_apply_hybrid_verifier(
  primary,
  data.frame(canonical_test_order_id = integer(), verifier_p = numeric()),
  fastkpc_hybrid_policy(enabled = FALSE, alpha = 0.05,
                        primary = "fastSplineCPU", verifier = "mgcvExtractCPU")
)
primary_graph <- fastkpc_replay_canonical_ci_decisions(primary_only, alpha = 0.05, p = 4L)

verifier <- data.frame(
  canonical_test_order_id = c(3L, 1L),
  verifier_p = c(0.20, 0.001),
  verifier_backend = c("mgcvExtractCPU", "mgcvExtractCPU"),
  stringsAsFactors = FALSE
)
hybrid <- fastkpc_apply_hybrid_verifier(primary, verifier, policy)
hybrid_graph <- fastkpc_replay_canonical_ci_decisions(hybrid, alpha = 0.05, p = 4L)

assert_true(primary_graph$adjacency[1, 2] == FALSE,
            "primary alone deletes edge 1-2")
assert_true(hybrid_graph$adjacency[1, 2] == FALSE,
            "hybrid still deletes edge 1-2 through later canonical row")
assert_true(identical(primary_graph$sepsets[[1]][[2]], 3L),
            "primary alone records first near-alpha sepset")
assert_true(identical(hybrid_graph$sepsets[[1]][[2]], 4L),
            "hybrid records later canonical sepset after verifier prevents row 1")
assert_true(primary_graph$adjacency[1, 3] == TRUE,
            "primary alone keeps edge 1-3")
assert_true(hybrid_graph$adjacency[1, 3] == FALSE,
            "hybrid verifier deletes edge 1-3")
assert_true(sum(hybrid$decision_before_verify != hybrid$decision_after_verify) == 2L,
            "hybrid should record two verifier-induced decision changes")

cat("PASS hybrid graph replay policy\n")
