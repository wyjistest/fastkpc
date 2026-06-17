source("fastkpc/R/hybrid_golden_snapshots.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

snapshots <- fastkpc_hybrid_golden_snapshots()
expected <- c(
  "scenario_linear_small",
  "scenario_nonlinear_additive",
  "scenario_pairwise_full_smooth",
  "scenario_near_alpha_flip"
)

assert_true(identical(names(snapshots), expected), "golden scenario names")

for (name in expected) {
  snapshot <- snapshots[[name]]
  assert_true(is.data.frame(snapshot$test_plan), paste(name, "test plan"))
  assert_true(all(order(snapshot$test_plan$canonical_test_order_id) ==
                    seq_len(nrow(snapshot$test_plan))),
              paste(name, "canonical order"))
  assert_true(is.matrix(snapshot$skeleton_adjacency),
              paste(name, "skeleton adjacency"))
  assert_true(identical(snapshot$skeleton_adjacency,
                        t(snapshot$skeleton_adjacency)),
              paste(name, "symmetric skeleton"))
  assert_true(is.data.frame(snapshot$edge_deletion_log),
              paste(name, "edge deletion log"))
  assert_true(all(c("p_source_used", "sepset_recorded") %in%
                    names(snapshot$edge_deletion_log)),
              paste(name, "diagnostics fields"))
  replayed <- fastkpc_replay_golden_snapshot(snapshot)
  assert_true(identical(replayed$adjacency, snapshot$skeleton_adjacency),
              paste(name, "replay adjacency"))
  assert_true(identical(replayed$diagnostics$sepset_recorded,
                        snapshot$edge_deletion_log$sepset_recorded),
              paste(name, "replay sepsets"))
}

cat("PASS hybrid golden snapshots\n")
