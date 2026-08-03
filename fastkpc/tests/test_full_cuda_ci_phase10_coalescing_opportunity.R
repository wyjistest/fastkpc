source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/full_cuda_ci_phase10_coalescing_opportunity.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    message
  )
}

path <- fastkpc_full_cuda_phase10_coalescing_default_evidence()
if (!file.exists(path)) {
  cat("SKIP Phase 10 coalescing opportunity: v5 evidence unavailable\n")
  quit(save = "no", status = 0L)
}
evidence <- readRDS(path)
load_fastkpc_cuda_native()
resource_before <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
opportunity <- fastkpc_full_cuda_phase10_coalescing_opportunity(evidence)
resource_after <- .Call(
  "C_fixed_sp_cuda_test_resource_snapshot", PACKAGE = "fastkpc_cuda"
)
fastkpc_full_cuda_phase10_validate_coalescing_opportunity(
  opportunity, evidence$result$summary
)
active_fields <- grep("_active_count$", names(resource_before), value = TRUE)

assert_true(
  opportunity$receipt_rebuild$window_count == 137L &&
    opportunity$receipt_rebuild$setup_cohort_count == 8637L &&
    opportunity$receipt_rebuild$target_optimization_count == 132908L &&
    opportunity$receipt_rebuild$consumed_target_key_count == 110617L &&
    identical(
      opportunity$decision, "STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL"
    ) &&
    opportunity$totals$value[
      opportunity$totals$metric == "ever_demanded_windows"
    ] == 137L &&
    opportunity$totals$value[
      opportunity$totals$metric == "never_demanded_windows"
    ] == 0L &&
    opportunity$totals$value[
      opportunity$totals$metric == "never_demanded_setup_cohorts"
    ] == 3L &&
    opportunity$totals$value[
      opportunity$totals$metric == "targets_in_never_demanded_cohorts"
    ] == 9L &&
    opportunity$totals$value[
      opportunity$totals$metric ==
        "unconsumed_targets_inside_demanded_cohorts"
    ] == 22282L &&
    all(opportunity$level3_checkpoint$value == c(
      83L, 83L, 5239L, 5239L, 107053L, 107053L, 0L
    )) &&
    length(active_fields) > 0L &&
    all(unlist(resource_before[active_fields], use.names = FALSE) == 0) &&
    identical(resource_before[active_fields], resource_after[active_fields]) &&
    nrow(opportunity$frontier_epochs) > 0L &&
    all(opportunity$frontier_epochs$ready_in_inactive_window_count >= 0L),
  "Phase 10 coalescing opportunity did not rebuild the v5 campaign"
)

tampered <- opportunity
tampered$setup_cohorts$target_count[[1L]] <-
  tampered$setup_cohorts$target_count[[1L]] + 1L
assert_error(
  fastkpc_full_cuda_phase10_validate_coalescing_opportunity(tampered),
  "setup-aware coalescing opportunity is malformed",
  "coalescing opportunity must reject target-count drift"
)

tampered_evidence <- evidence
tampered_evidence$result$summary$scheduler <-
  "cuda-zero-lookahead-frontier-prefill-host-v6"
assert_error(
  fastkpc_full_cuda_phase10_coalescing_opportunity(tampered_evidence),
  "requires the canonical v5 profile",
  "coalescing opportunity must reject a non-v5 scheduler"
)

cat("PASS Phase 10 setup-aware coalescing opportunity diagnostic\n")
