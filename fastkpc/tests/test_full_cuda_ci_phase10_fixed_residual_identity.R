source("fastkpc/R/full_cuda_ci_gate.R")
source("fastkpc/R/full_cuda_ci_oracle_contract.R")
source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_prepared_s_contract.R")
source("fastkpc/R/full_cuda_ci_fixed_sp_runtime.R")
source("fastkpc/R/cuda_native.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_bakeoff.R")
source("fastkpc/R/full_cuda_ci_phase10_fixed_residual_identity.R")

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

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS", unset = "0"), "1")) {
  cat("SKIP Phase 10 fixed residual identity: CUDA tests disabled\n")
  quit(save = "no", status = 0L)
}
if (!isTRUE(tryCatch(fastkpc_cuda_available(), error = function(error) FALSE))) {
  cat("SKIP Phase 10 fixed residual identity: CUDA unavailable\n")
  quit(save = "no", status = 0L)
}

fixtures <- fastkpc_full_cuda_phase10_fixed_identity_default_fixtures()
qualification <- fastkpc_full_cuda_phase10_qualify_fixed_residual_identity(
  fixtures, device_id = 0L
)
fastkpc_full_cuda_phase10_validate_fixed_residual_identity(qualification)

rows <- qualification$rows
assert_true(
  nrow(rows) == 10L &&
    setequal(unique(rows$fixture), names(fixtures)) &&
    sum(!rows$subset_sensitive) == 4L &&
    all(rows$matched_target_count >= 1L) &&
    all(rows$residual_d2h_bytes == 0) &&
    all(rows$compact_d2h_bytes > 0) &&
    all(rows$structural_gate) &&
    isTRUE(qualification$gates$structural_gate) &&
    isTRUE(qualification$gates$resource_gate),
  "fixed residual cross-batch structural qualification drifted"
)
assert_true(
  identical(qualification$decision, "CONDITIONAL_ALL_HIT_BATCH_ONLY") &&
    isTRUE(qualification$gates$cohort_gate) &&
    !isTRUE(qualification$gates$target_granular_gate) &&
    qualification$gates$residual_mismatch_value_count == 913L &&
    qualification$gates$component_mismatch_value_count == 332530L &&
    qualification$gates$exact_p_value_mismatch_count == 0L &&
    sum(rows$legacy_p_value_mismatch_count) == 2L &&
    qualification$gates$final_p_value_mismatch_count == 0L &&
    qualification$gates$final_decision_flip_count == 0L &&
    qualification$gates$metadata_mismatch_count == 3L,
  "fixed residual target-granular STOP evidence drifted"
)

repeat_rows <- rows[rows$variant == "repeat_original_cohort", , drop = FALSE]
assert_true(
  nrow(repeat_rows) == 2L &&
    all(repeat_rows$residual_mismatch_value_count == 0) &&
    all(repeat_rows$centered_component_mismatch_value_count == 0) &&
    all(repeat_rows$row_sum_mismatch_value_count == 0) &&
    all(repeat_rows$total_mismatch_value_count == 0) &&
    all(repeat_rows$self_moment_mismatch_value_count == 0) &&
    all(repeat_rows$exact_p_value_mismatch_count == 0) &&
    all(repeat_rows$legacy_p_value_mismatch_count == 0) &&
    all(repeat_rows$final_p_value_mismatch_count == 0) &&
    all(repeat_rows$planned_route_mismatch_count == 0) &&
    all(repeat_rows$executed_route_mismatch_count == 0) &&
    all(repeat_rows$solver_status_mismatch_count == 0),
  "identical fixed residual cohorts must remain bitwise exact"
)

tampered <- qualification
tampered$rows$residual_d2h_bytes[[1L]] <- 8
assert_error(
  fastkpc_full_cuda_phase10_validate_fixed_residual_identity(tampered),
  "cross-batch fixed residual qualification is malformed",
  "fixed residual identity must reject payload D2H accounting drift"
)
tampered <- qualification
tampered$rows$residual_mismatch_value_count[[1L]] <-
  tampered$rows$residual_mismatch_value_count[[1L]] + 1L
assert_error(
  fastkpc_full_cuda_phase10_validate_fixed_residual_identity(tampered),
  "cross-batch fixed residual qualification is malformed",
  "fixed residual identity must recompute mismatch accounting"
)

cat(
  "PASS Phase 10 fixed residual cross-batch qualification; decision=",
  qualification$decision,
  "; residual_mismatches=",
  qualification$gates$residual_mismatch_value_count,
  "; component_mismatches=",
  qualification$gates$component_mismatch_value_count,
  "; final_p_mismatches=",
  qualification$gates$final_p_value_mismatch_count,
  "\n", sep = ""
)
