source("fastkpc/R/fast_cuda_stage_breakdown.R")
source("fastkpc/R/cuda_native.R")

fail <- function(message) stop(message, call. = FALSE)
assert_true <- function(value, message) if (!isTRUE(value)) fail(message)

if (!identical(Sys.getenv("FASTKPC_RUN_CUDA_TESTS"), "1")) {
  cat("SKIP fast CUDA stage breakdown: FASTKPC_RUN_CUDA_TESTS != 1\n")
  quit(save = "no", status = 0)
}

build_fastkpc_cuda_native(rebuild = FALSE)
if (!fastkpc_cuda_available()) {
  cat("SKIP fast CUDA stage breakdown: CUDA unavailable\n")
  quit(save = "no", status = 0)
}

scenario <- list(
  scenario_id = "smoke-n48-p5-m1",
  n = 48L,
  p = 5L,
  max_conditioning_size = 1L,
  seed = 9201L,
  data = fastkpc_stage_breakdown_synthetic_data(48L, 5L, 9201L)
)

artifact <- fastkpc_run_fast_cuda_stage_breakdown(
  output_dir = tempfile("fast-cuda-stage-breakdown-"),
  scenarios = list(scenario),
  repeats = 1L
)

breakdown <- artifact$breakdown
runs <- artifact$runs
summary <- artifact$summary

required_breakdown <- c(
  "scenario_id", "repeat_id", "n", "p", "max_conditioning_size",
  "stage", "stage_group", "elapsed_ms", "share_of_skeleton"
)
missing_breakdown <- setdiff(required_breakdown, names(breakdown))
assert_true(length(missing_breakdown) == 0L,
            paste("missing breakdown fields:",
                  paste(missing_breakdown, collapse = ",")))
required_stages <- c(
  "skeleton_total", "fastspline_residual_prefetch", "ci_eval_total",
  "ci_host_pack", "native_replay", "dcov_rowsum_distance",
  "dcov_fused_center_reduce", "dcov_measured_total", "residual_host_pack",
  "residual_factor_solve", "residual_factor_cholesky",
  "residual_factor_rhs_solve",
  "residual_rhs_custom_solve", "residual_rhs_cublas_solve",
  "residual_factor_inverse_solve", "residual_d2h",
  "residual_d2h_residuals", "residual_d2h_metadata",
  "residual_d2h_info",
  "residual_cache_insert", "residual_true_batch_total",
  "residual_request_collect", "residual_prefetch_missing_scan",
  "residual_prefetch_batch_input", "residual_batch_call_wall",
  "residual_diagnostic_merge", "residual_prefetch_unaccounted",
  "residual_batch_top_level_wall", "residual_result_materialize",
  "residual_fitted_materialize", "residual_batch_top_level_unaccounted",
  "ci_dcov_call_wall", "ci_pvalue_copy", "ci_diagnostic_append",
  "ci_eval_unaccounted", "dcov_result_materialize",
  "dcov_top_level_wall", "dcov_grid_limit_query",
  "dcov_chunk_dispatch", "dcov_top_level_unaccounted"
)
missing_stages <- setdiff(required_stages, unique(breakdown$stage))
assert_true(length(missing_stages) == 0L,
            paste("missing stages:", paste(missing_stages, collapse = ",")))
assert_true(all(is.finite(breakdown$elapsed_ms) | is.na(breakdown$elapsed_ms)),
            "stage elapsed_ms values should be finite or NA")
assert_true(all(breakdown$elapsed_ms[is.finite(breakdown$elapsed_ms)] >= 0),
            "stage elapsed_ms values should be nonnegative")

assert_true(nrow(runs) == 1L, "smoke run should produce one run row")
assert_true(identical(runs$scheduler[[1L]], "layer"),
            "stage breakdown should use native layer scheduler")
assert_true(!isTRUE(runs$precision_overlay_used[[1L]]),
            "stage breakdown should not use precision overlay")
assert_true(runs$cpu_fallback_count[[1L]] == 0L,
            "stage breakdown should not use CPU fallback")
assert_true(runs$dcov_batches[[1L]] > 0L,
            "stage breakdown should record dCov batches")
assert_true(runs$dcov_chunks[[1L]] > 0L,
            "stage breakdown should record dCov chunks")
assert_true(runs$dcov_workspace_reuse_count[[1L]] > 0L,
            "stage breakdown should record dCov workspace reuse")
assert_true(runs$dcov_workspace_grow_count[[1L]] > 0L,
            "stage breakdown should record dCov workspace growth")
assert_true(runs$dcov_raw_aggregate_fused_count[[1L]] > 0L,
            "stage breakdown should record dCov raw aggregate fusion")
assert_true(runs$dcov_row_product_reduce_count[[1L]] > 0L,
            "stage breakdown should record dCov row-product reduce")
assert_true(runs$dcov_pvalue_only_count[[1L]] > 0L,
            "stage breakdown should use dCov pvalue-only skeleton path")
assert_true(runs$dcov_full_result_materialize_count[[1L]] == 0L,
            "stage breakdown should avoid full dCov result materialization")
assert_true(runs$dcov_top_level_wall_ms[[1L]] > 0,
            "stage breakdown should record dCov top-level wall time")
assert_true(runs$dcov_grid_limit_query_count[[1L]] +
              runs$dcov_grid_limit_process_cache_hit_count[[1L]] >= 1L,
            "stage breakdown should record dCov grid-limit lookup accounting")
assert_true(runs$dcov_grid_limit_cache_hit_count[[1L]] > 0L,
            "stage breakdown should record cached dCov grid-limit hits")
assert_true(runs$dcov_grid_limit_process_cache_hit_count[[1L]] >= 0L,
            "stage breakdown should record process dCov grid-limit cache hits")
assert_true(runs$dcov_top_level_unaccounted_ms[[1L]] >= 0,
            "stage breakdown should record dCov top-level unaccounted time")
ci_host_pack <- breakdown$elapsed_ms[breakdown$stage == "ci_host_pack"]
assert_true(length(ci_host_pack) == 1L && is.finite(ci_host_pack[[1L]]) &&
              ci_host_pack[[1L]] > 0,
            "stage breakdown should record positive CI host pack time")
assert_true(runs$unique_residual_requests[[1L]] > 0L,
            "stage breakdown should record unique residual requests")
assert_true(runs$cuda_residual_true_batched_fits[[1L]] >= 0L,
            "stage breakdown should record true batched residual fits")
assert_true(runs$residual_factorization_count[[1L]] >= 0L,
            "stage breakdown should record factorization count")
assert_true(runs$residual_factor_cache_entries[[1L]] >= 0L,
            "stage breakdown should record factor cache entries")
assert_true(runs$residual_rhs_solve_api_calls[[1L]] >= 0L,
            "stage breakdown should record RHS solve API calls")
assert_true(runs$residual_rhs_custom_solve_count[[1L]] > 0L,
            "stage breakdown should record custom RHS solves")
assert_true(runs$residual_candidate_rhs_fused_solve_count[[1L]] > 0L,
            "stage breakdown should record fused candidate RHS solves")
assert_true(runs$residual_candidate_rhs_materialized_solve_count[[1L]] == 0L,
            "stage breakdown should avoid materialized candidate RHS solves")
assert_true(runs$residual_selected_rhs_materialized_solve_count[[1L]] > 0L,
            "stage breakdown should keep selected RHS materialization")
assert_true(runs$residual_candidate_beta_values_avoided[[1L]] > 0L,
            "stage breakdown should record avoided candidate beta values")
assert_true(runs$residual_rhs_cublas_solve_count[[1L]] == 0L,
            "stage breakdown should avoid cuBLAS RHS solves on small-p smoke run")
assert_true(runs$residual_rhs_solve_fallback_count[[1L]] == 0L,
            "stage breakdown should avoid RHS solve fallbacks on small-p smoke run")
assert_true(runs$residual_rhs_custom_solve_ms[[1L]] > 0,
            "stage breakdown should time custom RHS solves")
assert_true(runs$residual_rhs_cublas_solve_ms[[1L]] >= 0,
            "stage breakdown should time cuBLAS RHS solves")
assert_true(runs$residual_d2h_copy_count[[1L]] > 0L,
            "stage breakdown should record residual D2H copy count")
assert_true(runs$residual_d2h_metadata_coalesced_count[[1L]] > 0L,
            "stage breakdown should coalesce residual score metadata D2H")
assert_true(runs$residual_d2h_metadata_coalesced_count[[1L]] ==
              runs$residual_only_batch_count[[1L]],
            "residual score metadata should be coalesced once per batch")
assert_true(runs$residual_d2h_copy_count[[1L]] <=
              runs$residual_d2h_metadata_coalesced_count[[1L]] + 3L,
            "residual D2H copy count should avoid per-candidate tiny copies")
assert_true(runs$residual_d2h_bytes[[1L]] > 0,
            "stage breakdown should record residual D2H bytes")
assert_true(runs$residual_d2h_residual_bytes[[1L]] > 0,
            "stage breakdown should record residual D2H residual bytes")
assert_true(runs$residual_d2h_metadata_bytes[[1L]] > 0,
            "stage breakdown should record residual D2H metadata bytes")
assert_true(runs$residual_d2h_residuals_ms[[1L]] >= 0,
            "stage breakdown should time residual vector D2H")
assert_true(runs$residual_d2h_metadata_ms[[1L]] >= 0,
            "stage breakdown should time residual metadata D2H")
assert_true(runs$residual_d2h_info_ms[[1L]] >= 0,
            "stage breakdown should time residual info D2H")
assert_true(runs$residual_lambda_candidates[[1L]] >= 0L,
            "stage breakdown should record lambda candidate count")
assert_true(runs$residual_workspace_reuse_count[[1L]] >= 0L,
            "stage breakdown should record residual workspace reuse count")
assert_true(runs$residual_workspace_grow_count[[1L]] >= 0L,
            "stage breakdown should record residual workspace grow count")
assert_true(runs$residual_solver_handle_create_count[[1L]] >= 0L,
            "stage breakdown should record residual solver handle creation count")
assert_true(runs$residual_per_request_design_x_values[[1L]] == 0L,
            "stage breakdown should avoid per-request residual design X")
assert_true(runs$residual_duplicate_design_x_values_avoided[[1L]] >= 0L,
            "stage breakdown should record avoided duplicate residual design X")
assert_true(runs$residual_cache_move_insert_count[[1L]] > 0L,
            "stage breakdown should record moved residual cache inserts")
assert_true(runs$residual_cache_copy_insert_count[[1L]] == 0L,
            "stage breakdown should avoid copied residual cache inserts")
assert_true(runs$residual_algebraic_rss_count[[1L]] > 0L,
            "stage breakdown should record algebraic RSS scoring")
assert_true(runs$residual_candidate_residual_materialize_count[[1L]] == 0L,
            "stage breakdown should avoid candidate residual materialization")
assert_true(runs$residual_winning_residual_materialize_count[[1L]] > 0L,
            "stage breakdown should record winning residual materialization")
assert_true(runs$residual_batch_call_wall_ms[[1L]] > 0,
            "stage breakdown should record residual batch-call wall time")
assert_true(runs$residual_only_batch_count[[1L]] > 0L,
            "stage breakdown should use residual-only scheduler batches")
assert_true(runs$residual_full_fit_batch_count[[1L]] == 0L,
            "stage breakdown should avoid full residual fit batches")
assert_true(runs$residual_only_fit_count[[1L]] > 0L,
            "stage breakdown should record residual-only fits")
assert_true(runs$residual_full_fit_materialize_count[[1L]] == 0L,
            "stage breakdown should avoid full residual fit materialization")
assert_true(runs$residual_fitted_values_avoided[[1L]] > 0L,
            "stage breakdown should record avoided fitted residual values")
assert_true(runs$residual_batch_top_level_wall_ms[[1L]] > 0,
            "stage breakdown should record residual callee top-level wall time")
assert_true(runs$residual_result_materialize_ms[[1L]] >= 0,
            "stage breakdown should record residual result materialization time")
assert_true(runs$residual_fitted_materialize_ms[[1L]] >= 0,
            "stage breakdown should record residual fitted materialization time")
assert_true(runs$residual_batch_top_level_unaccounted_ms[[1L]] >= 0,
            "stage breakdown should record residual callee top-level unaccounted time")
assert_true(runs$residual_prefetch_unaccounted_ms[[1L]] >= 0,
            "stage breakdown should record residual prefetch unaccounted time")
assert_true(runs$ci_dcov_call_wall_ms[[1L]] > 0,
            "stage breakdown should record CI dCov call wall time")
assert_true(runs$ci_eval_unaccounted_ms[[1L]] >= 0,
            "stage breakdown should record CI eval unaccounted time")
if (runs$residual_lambda_candidates[[1L]] > 0L &&
    runs$cuda_residual_unique_designs[[1L]] > 0L) {
  factor_bound <- runs$cuda_residual_unique_designs[[1L]] *
    runs$residual_lambda_candidates[[1L]] +
    runs$unique_residual_requests[[1L]]
  assert_true(runs$residual_factorization_count[[1L]] <= factor_bound,
              "factorization count should be design/lambda scoped")
  inverse_bound <- runs$cuda_residual_unique_designs[[1L]] *
    runs$residual_lambda_candidates[[1L]]
  assert_true(runs$residual_inverse_solve_count[[1L]] <= inverse_bound,
              "inverse solve count should be design/lambda scoped")
}
assert_true(is.data.frame(artifact$reconciliation),
            "stage breakdown should return reconciliation table")
assert_true("accounted_share" %in% names(artifact$reconciliation),
            "reconciliation table should report accounted_share")
assert_true("cuda_sync_ms" %in% names(artifact$reconciliation),
            "reconciliation table should report cuda_sync_ms")

required_summary <- c(
  "stage", "stage_group", "run_count", "finite_count", "median_ms", "p90_ms"
)
missing_summary <- setdiff(required_summary, names(summary))
assert_true(length(missing_summary) == 0L,
            paste("missing summary fields:",
                  paste(missing_summary, collapse = ",")))
assert_true(file.exists(artifact$paths$breakdown_csv),
            "stage breakdown CSV should be written")
assert_true(file.exists(artifact$paths$runs_csv),
            "stage breakdown runs CSV should be written")
assert_true(file.exists(artifact$paths$summary_csv),
            "stage breakdown summary CSV should be written")
assert_true(file.exists(artifact$paths$reconciliation_csv),
            "stage breakdown reconciliation CSV should be written")
assert_true(file.exists(artifact$paths$summary_md),
            "stage breakdown Markdown summary should be written")

cat("PASS fast CUDA stage breakdown\n")
