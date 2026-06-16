source("fastkpc/R/native.R")

assert_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

self <- wanpdag_engine_core_selftest()

required <- c(
  "empty_skeleton_returns_empty_pdag",
  "collider_stage_count_correct",
  "rules_stage_count_correct",
  "generalized_stage_orients_expected_edge",
  "event_log_has_one_based_indices_in_R",
  "residual_backend_params_recorded",
  "cache_stats_recorded",
  "solve_confl_false_no_bidirected",
  "rule_flags_disable_rules"
)

missing <- setdiff(required, names(self))
assert_true(length(missing) == 0L,
            paste("WAN-PDAG engine core selftest missing fields:",
                  paste(missing, collapse = ", ")))

for (name in required) {
  assert_true(self[[name]], paste(name, "should be TRUE"))
}

cat("test_wanpdag_engine_core.R: PASS\n")
