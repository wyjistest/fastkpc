source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase10_performance_v2.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else ": no error")
  )
}

legacy_names <- fastkpc_full_cuda_phase35_contract_names()
legacy <- fastkpc_full_cuda_phase35_load_contract_set()
v2 <- fastkpc_full_cuda_phase10_load_contract_set_v2()
budget <- v2$performance_budget_v2

assert_true(
  identical(names(legacy), legacy_names) &&
    "performance_budget_v2" %in% names(v2) &&
    !"performance_budget_v1" %in% names(v2),
  "performance budget v2 must not silently mutate the historical v1 set"
)
assert_true(
  identical(budget$semantic_version,
            list(major = 2L, minor = 0L, patch = 0L)) &&
    grepl("^[0-9a-f]{64}$", budget$sha256) &&
    !identical(budget$sha256, legacy$performance_budget_v1$sha256),
  "performance budget v2 identity is malformed or aliases v1"
)
payload <- budget$payload
assert_true(
  identical(
    payload$promotion$primary_performance_boundary,
    "fresh_data_compute_warm"
  ) && identical(
    payload$boundaries$legacy_v1_evidence,
    NULL
  ) && !isTRUE(payload$boundaries$replay_warm$promotion_gate) &&
    payload$boundaries$fresh_data_compute_warm$gate$median_upper_bound_ms ==
      120000L &&
    payload$boundaries$fresh_process_cold$gate$correct_baseline_ratio_max ==
      "1.00",
  "performance budget v2 boundary authority is malformed"
)

tampered <- payload
tampered$boundaries$fresh_data_compute_warm$gate$median_upper_bound_ms <-
  120001L
assert_error(
  fastkpc_full_cuda_phase10_validate_performance_budget_v2(tampered),
  "performance budget v2 policy is invalid",
  "performance budget v2 threshold tampering must fail closed"
)
tampered <- payload
tampered$boundaries$replay_warm$promotion_gate <- TRUE
assert_error(
  fastkpc_full_cuda_phase10_validate_performance_budget_v2(tampered),
  "performance budget v2 policy is invalid",
  "replay-warm must never become a promotion gate"
)

cat("PASS Phase 10 performance budget v2 fresh-compute boundary\n")
