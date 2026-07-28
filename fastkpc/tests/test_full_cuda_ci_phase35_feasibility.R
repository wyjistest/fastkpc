source("fastkpc/R/full_cuda_ci_workload_census.R")
source("fastkpc/R/full_cuda_ci_phase35_contracts.R")
source("fastkpc/R/full_cuda_ci_phase35_feasibility.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch(force(expression), error = identity)
  assert_true(
    inherits(error, "error") &&
      grepl(pattern, conditionMessage(error), fixed = TRUE),
    paste0(message, if (inherits(error, "error")) {
      paste0(": ", conditionMessage(error))
    } else {
      ": no error"
    })
  )
}

eig_output <- paste(
  "n=351 repeats=100 selected=35 info=0",
  "full_lwork=431456 partial_lwork=431456",
  "full_total_ms=744.51046 full_per_component_ms=7.4451046",
  "two_sided_partial_total_ms=1477.39954",
  "two_sided_partial_per_component_ms=14.7739954",
  "full_110617_bound_ms=823555.135",
  "partial_110617_bound_ms=1634255.05"
)
block_output <- paste(
  "n=351 block=62 batch=47 iterations=12 repeats=20",
  "sequence_ms=14.8593159 per_component_ms=0.316155657",
  "bound_110617_ms=34972.1903"
)
measurements <- fastkpc_full_cuda_phase35_parse_candidate_measurements(
  eig_output, block_output
)
assert_true(
  measurements$eig$n == 351 && measurements$eig$info == 0 &&
    measurements$block$batch == 47 &&
    measurements$global_full_eig_bound_ms > 35000 &&
    measurements$global_partial_eig_bound_ms > 35000 &&
    nrow(measurements$table) == 11L &&
    !any(measurements$table$feasibility_model_use ==
           "global-feasibility-bound"),
  "candidate measurement parser must retain diagnostic-only extrapolations"
)

assert_error(
  fastkpc_full_cuda_phase35_parse_candidate_measurements(
    sub(" info=0", " info=1", eig_output, fixed = TRUE), block_output
  ),
  "candidate benchmark dimensions or arithmetic gate failed",
  "nonzero eigensolver status must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase35_parse_candidate_measurements(
    sub(
      "full_110617_bound_ms=823555.135",
      "full_110617_bound_ms=1", eig_output, fixed = TRUE
    ),
    block_output
  ),
  "candidate benchmark dimensions or arithmetic gate failed",
  "benchmark arithmetic tampering must fail closed"
)
assert_error(
  fastkpc_full_cuda_phase35_parse_candidate_measurements(
    paste(eig_output, "unexpected=1"), block_output
  ),
  "eigensolver benchmark output field set or order mismatch",
  "unexpected benchmark fields must fail closed"
)

contracts <- fastkpc_full_cuda_phase35_load_contract_set(
  verify_source_artifacts = FALSE
)
assert_true(
  contracts$development_qualification_corpus_v1$payload$
    canonical_counts$qualification_setup_count == 2061L &&
    identical(
      contracts$numerical_contract_v1$payload$dcov_formulas$variance,
      "variance_factor*x_self_moment*y_self_moment/n^2"
    ),
  "feasibility code must consume the corrected tracked contracts"
)

cat("PASS full CUDA CI Phase 3.5 feasibility model primitives\n")
