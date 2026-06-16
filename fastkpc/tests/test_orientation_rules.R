source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

self <- orientation_rules_selftest()

required <- c(
  "collider_orients_unshielded_triple",
  "collider_respects_sepset",
  "conflict_collider_marks_bidirected",
  "check_immor_accepts_clique_S",
  "check_immor_rejects_nonclique_S",
  "check_immor_rejects_unconnected_parent",
  "rule1_orients_chain_tail",
  "rule2_orients_directed_chain",
  "rule3_orients_double_parent_pattern",
  "fixed_point_converges",
  "rules_disabled_no_change"
)

missing <- setdiff(required, names(self))
assert_true(length(missing) == 0L,
            paste("orientation rules selftest missing fields:",
                  paste(missing, collapse = ", ")))

for (name in required) {
  assert_true(self[[name]], paste(name, "should be TRUE"))
}

cat("test_orientation_rules.R: PASS\n")
