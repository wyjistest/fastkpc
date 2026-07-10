fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)
assert_error <- function(expression, pattern, message) {
  error <- tryCatch({ force(expression); NULL }, error = identity)
  assert_true(inherits(error, "error") &&
                grepl(pattern, conditionMessage(error), fixed = TRUE),
              message)
}

source("fastkpc/R/full_cuda_ci_workload_census.R")

warning_capture <- fastkpc_full_cuda_census_capture_warnings({
  warning("first parity warning", call. = FALSE)
  warning(structure(
    list(message = "second parity warning", call = NULL),
    class = c("fastkpc_test_warning", "warning", "condition")
  ))
  42L
})
assert_true(identical(warning_capture$value, 42L) &&
              identical(
                vapply(warning_capture$warnings, `[[`, character(1L),
                       "message"),
                c("first parity warning", "second parity warning")
              ) &&
              identical(warning_capture$warnings[[2L]]$class,
                        c("fastkpc_test_warning", "warning", "condition")) &&
              is.finite(warning_capture$elapsed_ms) &&
              warning_capture$elapsed_ms >= 0,
            "warning capture must preserve class/message order and timing")

inputs <- fastkpc_full_cuda_census_load_inputs(
  oracle_dir = "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1",
  data_path = paste0(
    "fastkpc/artifacts/kpc_tprs_real_zhu/",
    "cancer_RD-causalDiscoveryInput.rds"
  )
)
cases <- fastkpc_full_cuda_census_parity_cases(inputs)

assert_true(length(cases) == 7L,
            "parity corpus must contain five canonical and two synthetic cases")
case_table <- do.call(rbind, lapply(cases, function(case) {
  data.frame(
    case_id = case$case_id,
    source_type = case$source_type,
    S_size = case$S_size,
    stringsAsFactors = FALSE
  )
}))
assert_true(all(c(1L, 2L, 3L, 7L) %in%
                  case_table$S_size[case_table$source_type == "canonical"]),
            "canonical parity cases must cover |S| 1, 2, 3, and 7")
assert_true(all(c("synthetic-rank-deficient",
                  "synthetic-near-constant") %in% case_table$case_id),
            "parity corpus must cover rank-deficient and near-constant data")

results <- do.call(rbind, lapply(
  cases,
  fastkpc_full_cuda_census_parity_case
))
required_fields <- c(
  "case_id", "source_type", "logical_sequence_id", "S_size",
  "regr_pair_residual_hash_identical",
  "regr_pair_residual_max_abs_diff",
  "pair_single_residual_hash_identical",
  "pair_single_residual_max_abs_diff", "fitted_hash_identical",
  "fitted_max_abs_diff", "selected_sp_identical",
  "pair_selected_sp_names", "single_selected_sp_names",
  "selected_sp_max_abs_diff", "GCV_Cp_identical",
  "GCV_Cp_max_abs_diff", "EDF_identical", "EDF_max_abs_diff",
  "dcov_p_value_identical", "dcov_p_value_abs_diff",
  "decision_identical", "pair_warning_count", "single_warning_count",
  "pair_fit_elapsed_ms", "single_fit_elapsed_ms", "regr_elapsed_ms",
  "pass"
)
assert_true(identical(names(results), required_fields),
            "parity results must expose the approved exact-equality evidence")
assert_true(nrow(results) == 7L && all(results$pass),
            "every canonical and synthetic parity case must pass")
assert_true(all(results$regr_pair_residual_hash_identical) &&
              all(results$pair_single_residual_hash_identical) &&
              all(results$fitted_hash_identical) &&
              all(results$regr_pair_residual_max_abs_diff == 0) &&
              all(results$pair_single_residual_max_abs_diff == 0) &&
              all(results$fitted_max_abs_diff == 0),
            "residual and fitted vectors must be bit-exact")
assert_true(all(results$selected_sp_identical) &&
              all(nzchar(results$pair_selected_sp_names)) &&
              all(nzchar(results$single_selected_sp_names)) &&
              all(results$selected_sp_max_abs_diff == 0) &&
              all(results$GCV_Cp_identical) &&
              all(results$GCV_Cp_max_abs_diff == 0) &&
              all(results$EDF_identical) &&
              all(results$EDF_max_abs_diff == 0),
            "selected sp, GCV/Cp, and EDF must be exact")
assert_true(all(results$dcov_p_value_identical) &&
              all(results$dcov_p_value_abs_diff == 0) &&
              all(results$decision_identical),
            "downstream dCov p-values and decisions must be exact")
assert_true(all(is.finite(results$pair_fit_elapsed_ms)) &&
              all(is.finite(results$single_fit_elapsed_ms)) &&
              all(is.finite(results$regr_elapsed_ms)) &&
              all(results$pair_fit_elapsed_ms >= 0) &&
              all(results$single_fit_elapsed_ms >= 0) &&
              all(results$regr_elapsed_ms >= 0),
            "parity evidence must retain fit timings")

artifact_dir <- tempfile("full-cuda-ci-parity-")
paths <- fastkpc_full_cuda_census_write_parity(
  cases = cases,
  results = results,
  output_dir = artifact_dir
)
assert_true(all(file.exists(unlist(paths, use.names = FALSE))),
            "parity writer must persist cases and exact result evidence")
stored_cases <- readRDS(paths$cases_rds)
stored_results <- utils::read.csv(paths$results_csv,
                                  stringsAsFactors = FALSE)
assert_true(length(stored_cases) == 7L && nrow(stored_results) == 7L &&
              all(stored_results$pass),
            "persisted parity artifact must retain every passing case")

failed_results <- results
failed_results$pass[[1L]] <- FALSE
assert_error(
  fastkpc_full_cuda_census_write_parity(
    cases = cases,
    results = failed_results,
    output_dir = tempfile("full-cuda-ci-parity-failed-")
  ),
  "legacy layout parity failed",
  "parity writer must reject failed evidence before metadata shards run"
)

cat("PASS full CUDA CI legacy layout parity\n")
