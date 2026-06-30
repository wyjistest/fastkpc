source("fastkpc/R/fast_kpc.R")
source("fastkpc/R/cuda_native.R")

if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) if (is.null(x)) y else x
}

fastkpc_stage_breakdown_synthetic_data <- function(n, p, seed) {
  set.seed(seed)
  roots <- matrix(stats::rnorm(n * max(2L, ceiling(p / 3L))), n)
  out <- matrix(0, n, p)
  for (j in seq_len(p)) {
    r1 <- roots[, ((j - 1L) %% ncol(roots)) + 1L]
    r2 <- roots[, (j %% ncol(roots)) + 1L]
    out[, j] <- sin(r1 * (0.5 + j / p)) + 0.35 * cos(r2) +
      0.15 * r1 * r2 + stats::rnorm(n, sd = 0.25)
  }
  colnames(out) <- paste0("V", seq_len(p))
  out
}

fastkpc_stage_breakdown_default_scenarios <- function() {
  specs <- data.frame(
    scenario_id = c("breakdown-n100-p12-m2",
                    "breakdown-n300-p12-m2",
                    "breakdown-n1000-p12-m2"),
    n = c(100L, 300L, 1000L),
    p = c(12L, 12L, 12L),
    max_conditioning_size = c(2L, 2L, 2L),
    seed = c(8701L, 8702L, 8703L),
    stringsAsFactors = FALSE
  )
  lapply(seq_len(nrow(specs)), function(i) {
    row <- specs[i, ]
    list(
      scenario_id = row$scenario_id,
      n = as.integer(row$n),
      p = as.integer(row$p),
      max_conditioning_size = as.integer(row$max_conditioning_size),
      seed = as.integer(row$seed),
      data = fastkpc_stage_breakdown_synthetic_data(
        as.integer(row$n), as.integer(row$p), as.integer(row$seed)
      )
    )
  })
}

fastkpc_stage_breakdown_seconds <- function(value) {
  value <- as.numeric(value %||% NA_real_)[1L]
  if (!is.finite(value)) return(NA_real_)
  value
}

fastkpc_stage_breakdown_rows <- function(result, scenario, repeat_id) {
  summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  total <- fastkpc_stage_breakdown_seconds(summary$total_elapsed_sec)
  row <- function(stage, elapsed_sec, group = "skeleton") {
    elapsed_ms <- elapsed_sec * 1000
    data.frame(
      scenario_id = scenario$scenario_id,
      repeat_id = as.integer(repeat_id),
      n = as.integer(scenario$n),
      p = as.integer(scenario$p),
      max_conditioning_size = as.integer(scenario$max_conditioning_size),
      stage = stage,
      stage_group = group,
      elapsed_ms = elapsed_ms,
      share_of_skeleton = if (is.finite(total) && total > 0) {
        elapsed_sec / total
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }
  rows <- list(
    row("skeleton_total", total, "skeleton"),
    row("plan", fastkpc_stage_breakdown_seconds(summary$plan_elapsed_sec),
        "control"),
    row("fastspline_residual_prefetch",
        fastkpc_stage_breakdown_seconds(summary$residual_prefetch_elapsed_sec),
        "residual"),
    row("residual_request_collect",
        fastkpc_stage_breakdown_seconds(summary$residual_request_collect_sec),
        "residual"),
    row("residual_prefetch_missing_scan",
        fastkpc_stage_breakdown_seconds(
          summary$residual_prefetch_missing_scan_sec
        ),
        "residual"),
    row("residual_prefetch_batch_input",
        fastkpc_stage_breakdown_seconds(
          summary$residual_prefetch_batch_input_sec
        ),
        "residual"),
    row("residual_batch_call_wall",
        fastkpc_stage_breakdown_seconds(summary$residual_batch_call_wall_sec),
        "residual"),
    row("residual_diagnostic_merge",
        fastkpc_stage_breakdown_seconds(
          summary$residual_diagnostic_merge_sec
        ),
        "residual"),
    row("residual_prefetch_unaccounted",
        fastkpc_stage_breakdown_seconds(
          summary$residual_prefetch_unaccounted_sec
        ),
        "residual"),
    row("residual_batch_top_level_wall",
        fastkpc_stage_breakdown_seconds(
          summary$residual_batch_top_level_wall_sec
        ),
        "residual"),
    row("residual_result_materialize",
        fastkpc_stage_breakdown_seconds(
          summary$residual_result_materialize_sec
        ),
        "residual"),
    row("residual_fitted_materialize",
        fastkpc_stage_breakdown_seconds(
          summary$residual_fitted_materialize_sec
        ),
        "residual"),
    row("residual_batch_top_level_unaccounted",
        fastkpc_stage_breakdown_seconds(
          summary$residual_batch_top_level_unaccounted_sec
        ),
        "residual"),
    row("residual_grouping",
        fastkpc_stage_breakdown_seconds(summary$residual_grouping_sec),
        "residual"),
    row("residual_host_pack",
        fastkpc_stage_breakdown_seconds(summary$residual_host_pack_sec),
        "residual"),
    row("residual_alloc",
        fastkpc_stage_breakdown_seconds(summary$residual_alloc_sec),
        "residual"),
    row("residual_h2d",
        fastkpc_stage_breakdown_seconds(summary$residual_h2d_sec),
        "residual"),
    row("residual_xtx_xty",
        fastkpc_stage_breakdown_seconds(summary$residual_xtx_xty_sec),
        "residual"),
    row("residual_pointer_setup",
        fastkpc_stage_breakdown_seconds(summary$residual_pointer_setup_sec),
        "residual"),
    row("residual_active_copy",
        fastkpc_stage_breakdown_seconds(summary$residual_active_copy_sec),
        "residual"),
    row("residual_build_system",
        fastkpc_stage_breakdown_seconds(summary$residual_build_system_sec),
        "residual"),
    row("residual_factor_solve",
        fastkpc_stage_breakdown_seconds(summary$residual_factor_solve_sec),
        "residual"),
    row("residual_factor_cholesky",
        fastkpc_stage_breakdown_seconds(
          summary$residual_factor_cholesky_sec
        ),
        "residual"),
    row("residual_factor_rhs_solve",
        fastkpc_stage_breakdown_seconds(
          summary$residual_factor_rhs_solve_sec
        ),
        "residual"),
    row("residual_rhs_custom_solve",
        fastkpc_stage_breakdown_seconds(
          summary$residual_rhs_custom_solve_sec
        ),
        "residual"),
    row("residual_rhs_cublas_solve",
        fastkpc_stage_breakdown_seconds(
          summary$residual_rhs_cublas_solve_sec
        ),
        "residual"),
    row("residual_factor_inverse_solve",
        fastkpc_stage_breakdown_seconds(
          summary$residual_factor_inverse_solve_sec
        ),
        "residual"),
    row("residual_summary_kernel",
        fastkpc_stage_breakdown_seconds(summary$residual_summary_sec),
        "residual"),
    row("residual_d2h",
        fastkpc_stage_breakdown_seconds(summary$residual_d2h_sec),
        "residual"),
    row("residual_d2h_residuals",
        fastkpc_stage_breakdown_seconds(
          summary$residual_d2h_residuals_sec
        ),
        "residual"),
    row("residual_d2h_metadata",
        fastkpc_stage_breakdown_seconds(
          summary$residual_d2h_metadata_sec
        ),
        "residual"),
    row("residual_d2h_info",
        fastkpc_stage_breakdown_seconds(summary$residual_d2h_info_sec),
        "residual"),
    row("residual_host_select",
        fastkpc_stage_breakdown_seconds(summary$residual_host_select_sec),
        "residual"),
    row("residual_free",
        fastkpc_stage_breakdown_seconds(summary$residual_free_sec),
        "residual"),
    row("residual_cache_insert",
        fastkpc_stage_breakdown_seconds(summary$residual_cache_insert_sec),
        "residual"),
    row("residual_true_batch_total",
        fastkpc_stage_breakdown_seconds(
          summary$residual_true_batch_total_sec
        ),
        "residual"),
    row("ci_eval_total",
        fastkpc_stage_breakdown_seconds(summary$ci_eval_elapsed_sec),
        "ci"),
    row("ci_host_pack",
        fastkpc_stage_breakdown_seconds(summary$ci_host_pack_sec),
        "ci"),
    row("ci_dcov_call_wall",
        fastkpc_stage_breakdown_seconds(summary$ci_dcov_call_wall_sec),
        "ci"),
    row("ci_pvalue_copy",
        fastkpc_stage_breakdown_seconds(summary$ci_pvalue_copy_sec),
        "ci"),
    row("ci_diagnostic_append",
        fastkpc_stage_breakdown_seconds(summary$ci_diagnostic_append_sec),
        "ci"),
    row("ci_eval_unaccounted",
        fastkpc_stage_breakdown_seconds(summary$ci_eval_unaccounted_sec),
        "ci"),
    row("native_replay",
        fastkpc_stage_breakdown_seconds(summary$replay_elapsed_sec),
        "control"),
    row("dcov_alloc", fastkpc_stage_breakdown_seconds(summary$dcov_alloc_sec),
        "dcov"),
    row("dcov_h2d", fastkpc_stage_breakdown_seconds(summary$dcov_h2d_sec),
        "dcov"),
    row("dcov_memset", fastkpc_stage_breakdown_seconds(summary$dcov_memset_sec),
        "dcov"),
    row("dcov_rowsum_distance",
        fastkpc_stage_breakdown_seconds(summary$dcov_rowsum_sec),
        "dcov"),
    row("dcov_totals_d2h",
        fastkpc_stage_breakdown_seconds(summary$dcov_totals_d2h_sec),
        "dcov"),
    row("dcov_fused_center_reduce",
        fastkpc_stage_breakdown_seconds(summary$dcov_reduce_sec),
        "dcov"),
    row("dcov_scalars_d2h",
        fastkpc_stage_breakdown_seconds(summary$dcov_scalars_d2h_sec),
        "dcov"),
    row("dcov_host_gamma_pvalue",
        fastkpc_stage_breakdown_seconds(summary$dcov_host_scalar_sec),
        "dcov"),
    row("dcov_result_materialize",
        fastkpc_stage_breakdown_seconds(
          summary$dcov_result_materialize_sec
        ),
        "dcov"),
    row("dcov_free", fastkpc_stage_breakdown_seconds(summary$dcov_free_sec),
        "dcov"),
    row("dcov_measured_total",
        fastkpc_stage_breakdown_seconds(summary$dcov_total_sec),
        "dcov"),
    row("dcov_top_level_wall",
        fastkpc_stage_breakdown_seconds(summary$dcov_top_level_wall_sec),
        "dcov"),
    row("dcov_grid_limit_query",
        fastkpc_stage_breakdown_seconds(summary$dcov_grid_limit_query_sec),
        "dcov"),
    row("dcov_chunk_dispatch",
        fastkpc_stage_breakdown_seconds(summary$dcov_chunk_dispatch_sec),
        "dcov"),
    row("dcov_top_level_unaccounted",
        fastkpc_stage_breakdown_seconds(
          summary$dcov_top_level_unaccounted_sec
        ),
        "dcov")
  )
  do.call(rbind, rows)
}

fastkpc_stage_breakdown_run_row <- function(result, scenario, repeat_id) {
  summary <- result$skeleton$scheduler_diagnostics$summary %||% list()
  data.frame(
    scenario_id = scenario$scenario_id,
    repeat_id = as.integer(repeat_id),
    n = as.integer(scenario$n),
    p = as.integer(scenario$p),
    max_conditioning_size = as.integer(scenario$max_conditioning_size),
    scheduler = as.character(result$skeleton$scheduler),
    route_data_plane = "native-cuda-skeleton",
    precision_overlay_used =
      identical(result$skeleton$scheduler, "layer-precision") ||
        identical(result$skeleton$scheduler, "r-precision"),
    cpu_fallback_count =
      as.integer(summary$cuda_residual_cpu_fallback_fits %||% 0L),
    n_edgetests = as.integer(sum(result$skeleton$n.edgetests)),
    final_edges = as.integer(sum(result$skeleton$adjacency) / 2L),
    dcov_batches = as.integer(summary$dcov_batches %||% 0L),
    dcov_chunks = as.integer(summary$dcov_chunks %||% 0L),
    dcov_batch_size_used = as.integer(summary$dcov_batch_size_used %||% 0L),
    dcov_max_chunk_batch =
      as.integer(summary$dcov_max_chunk_batch %||% 0L),
    dcov_workspace_reuse_count =
      as.integer(summary$dcov_workspace_reuse_count %||% 0L),
    dcov_workspace_grow_count =
      as.integer(summary$dcov_workspace_grow_count %||% 0L),
    dcov_raw_aggregate_fused_count =
      as.integer(summary$dcov_raw_aggregate_fused_count %||% 0L),
    dcov_row_product_reduce_count =
      as.integer(summary$dcov_row_product_reduce_count %||% 0L),
    dcov_pvalue_only_count =
      as.integer(summary$dcov_pvalue_only_count %||% 0L),
    dcov_full_result_materialize_count =
      as.integer(summary$dcov_full_result_materialize_count %||% 0L),
    dcov_top_level_wall_ms =
      fastkpc_stage_breakdown_seconds(summary$dcov_top_level_wall_sec) * 1000,
    dcov_grid_limit_query_ms =
      fastkpc_stage_breakdown_seconds(
        summary$dcov_grid_limit_query_sec
      ) * 1000,
    dcov_chunk_dispatch_ms =
      fastkpc_stage_breakdown_seconds(summary$dcov_chunk_dispatch_sec) * 1000,
    dcov_top_level_unaccounted_ms =
      fastkpc_stage_breakdown_seconds(
        summary$dcov_top_level_unaccounted_sec
      ) * 1000,
    dcov_grid_limit_query_count =
      as.integer(summary$dcov_grid_limit_query_count %||% 0L),
    dcov_grid_limit_cache_hit_count =
      as.integer(summary$dcov_grid_limit_cache_hit_count %||% 0L),
    dcov_grid_limit_process_cache_hit_count =
      as.integer(summary$dcov_grid_limit_process_cache_hit_count %||% 0L),
    residual_batches = as.integer(summary$residual_batches %||% 0L),
    unique_residual_requests =
      as.integer(summary$unique_residual_requests %||% 0L),
    cuda_residual_true_batched_fits =
      as.integer(summary$cuda_residual_true_batched_fits %||% 0L),
    cuda_residual_single_fit_calls =
      as.integer(summary$cuda_residual_single_fit_calls %||% 0L),
    cuda_residual_unique_designs =
      as.integer(summary$cuda_residual_unique_designs %||% 0L),
    residual_factorization_count =
      as.integer(summary$residual_factorization_count %||% 0L),
    residual_rhs_solve_count =
      as.integer(summary$residual_rhs_solve_count %||% 0L),
    residual_inverse_solve_count =
      as.integer(summary$residual_inverse_solve_count %||% 0L),
    residual_rhs_solve_api_calls =
      as.integer(summary$residual_rhs_solve_api_calls %||% 0L),
    residual_rhs_target_solves =
      as.integer(summary$residual_rhs_target_solves %||% 0L),
    residual_rhs_custom_solve_count =
      as.integer(summary$residual_rhs_custom_solve_count %||% 0L),
    residual_rhs_cublas_solve_count =
      as.integer(summary$residual_rhs_cublas_solve_count %||% 0L),
    residual_rhs_solve_fallback_count =
      as.integer(summary$residual_rhs_solve_fallback_count %||% 0L),
    residual_rhs_custom_solve_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_rhs_custom_solve_sec
      ) * 1000,
    residual_rhs_cublas_solve_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_rhs_cublas_solve_sec
      ) * 1000,
    residual_candidate_rhs_fused_solve_count =
      as.integer(summary$residual_candidate_rhs_fused_solve_count %||% 0L),
    residual_candidate_rhs_materialized_solve_count =
      as.integer(
        summary$residual_candidate_rhs_materialized_solve_count %||% 0L
      ),
    residual_selected_rhs_materialized_solve_count =
      as.integer(
        summary$residual_selected_rhs_materialized_solve_count %||% 0L
      ),
    residual_candidate_beta_values_avoided =
      as.integer(summary$residual_candidate_beta_values_avoided %||% 0L),
    residual_summary_candidate_launch_count =
      as.integer(summary$residual_summary_candidate_launch_count %||% 0L),
    residual_summary_group_batched_launch_count =
      as.integer(
        summary$residual_summary_group_batched_launch_count %||% 0L
      ),
    residual_summary_group_batched_candidate_count =
      as.integer(
        summary$residual_summary_group_batched_candidate_count %||% 0L
      ),
    residual_d2h_residuals_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_d2h_residuals_sec
      ) * 1000,
    residual_d2h_metadata_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_d2h_metadata_sec
      ) * 1000,
    residual_d2h_info_ms =
      fastkpc_stage_breakdown_seconds(summary$residual_d2h_info_sec) * 1000,
    residual_d2h_copy_count =
      as.integer(summary$residual_d2h_copy_count %||% 0L),
    residual_d2h_bytes = as.numeric(summary$residual_d2h_bytes %||% 0),
    residual_d2h_residual_bytes =
      as.numeric(summary$residual_d2h_residual_bytes %||% 0),
    residual_d2h_metadata_bytes =
      as.numeric(summary$residual_d2h_metadata_bytes %||% 0),
    residual_d2h_metadata_coalesced_count =
      as.integer(summary$residual_d2h_metadata_coalesced_count %||% 0L),
    residual_d2h_metadata_coalesced_bytes =
      as.numeric(summary$residual_d2h_metadata_coalesced_bytes %||% 0),
    residual_winning_factor_reuse_count =
      as.integer(summary$residual_winning_factor_reuse_count %||% 0L),
    residual_factor_cache_hits =
      as.integer(summary$residual_factor_cache_hits %||% 0L),
    residual_factor_cache_misses =
      as.integer(summary$residual_factor_cache_misses %||% 0L),
    residual_factor_cache_entries =
      as.integer(summary$residual_factor_cache_entries %||% 0L),
    residual_factor_cache_bytes =
      as.numeric(summary$residual_factor_cache_bytes %||% 0),
    residual_lambda_candidates =
      as.integer(summary$residual_lambda_candidates %||% 0L),
    residual_workspace_reuse_count =
      as.integer(summary$residual_workspace_reuse_count %||% 0L),
    residual_workspace_grow_count =
      as.integer(summary$residual_workspace_grow_count %||% 0L),
    residual_solver_handle_create_count =
      as.integer(summary$residual_solver_handle_create_count %||% 0L),
    residual_per_request_design_x_values =
      as.integer(summary$residual_per_request_design_x_values %||% 0L),
    residual_duplicate_design_x_values_avoided =
      as.integer(summary$residual_duplicate_design_x_values_avoided %||% 0L),
    residual_cache_insert_ms =
      fastkpc_stage_breakdown_seconds(summary$residual_cache_insert_sec) * 1000,
    residual_request_collect_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_request_collect_sec
      ) * 1000,
    residual_prefetch_missing_scan_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_prefetch_missing_scan_sec
      ) * 1000,
    residual_prefetch_batch_input_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_prefetch_batch_input_sec
      ) * 1000,
    residual_batch_call_wall_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_batch_call_wall_sec
      ) * 1000,
    residual_diagnostic_merge_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_diagnostic_merge_sec
      ) * 1000,
    residual_prefetch_unaccounted_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_prefetch_unaccounted_sec
      ) * 1000,
    residual_cache_move_insert_count =
      as.integer(summary$residual_cache_move_insert_count %||% 0L),
    residual_cache_copy_insert_count =
      as.integer(summary$residual_cache_copy_insert_count %||% 0L),
    residual_algebraic_rss_count =
      as.integer(summary$residual_algebraic_rss_count %||% 0L),
    residual_candidate_residual_materialize_count =
      as.integer(summary$residual_candidate_residual_materialize_count %||% 0L),
    residual_winning_residual_materialize_count =
      as.integer(summary$residual_winning_residual_materialize_count %||% 0L),
    residual_algebraic_rss_clamp_count =
      as.integer(summary$residual_algebraic_rss_clamp_count %||% 0L),
    residual_only_batch_count =
      as.integer(summary$residual_only_batch_count %||% 0L),
    residual_full_fit_batch_count =
      as.integer(summary$residual_full_fit_batch_count %||% 0L),
    residual_only_fit_count =
      as.integer(summary$residual_only_fit_count %||% 0L),
    residual_full_fit_materialize_count =
      as.integer(summary$residual_full_fit_materialize_count %||% 0L),
    residual_fitted_values_avoided =
      as.integer(summary$residual_fitted_values_avoided %||% 0L),
    residual_batch_top_level_wall_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_batch_top_level_wall_sec
      ) * 1000,
    residual_result_materialize_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_result_materialize_sec
      ) * 1000,
    residual_fitted_materialize_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_fitted_materialize_sec
      ) * 1000,
    residual_batch_top_level_unaccounted_ms =
      fastkpc_stage_breakdown_seconds(
        summary$residual_batch_top_level_unaccounted_sec
      ) * 1000,
    ci_dcov_call_wall_ms =
      fastkpc_stage_breakdown_seconds(summary$ci_dcov_call_wall_sec) * 1000,
    ci_pvalue_copy_ms =
      fastkpc_stage_breakdown_seconds(summary$ci_pvalue_copy_sec) * 1000,
    ci_diagnostic_append_ms =
      fastkpc_stage_breakdown_seconds(summary$ci_diagnostic_append_sec) * 1000,
    ci_eval_unaccounted_ms =
      fastkpc_stage_breakdown_seconds(summary$ci_eval_unaccounted_sec) * 1000,
    stringsAsFactors = FALSE
  )
}

fastkpc_stage_breakdown_reconciliation <- function(runs, breakdown) {
  keys <- unique(runs[, c("scenario_id", "repeat_id"), drop = FALSE])
  rows <- lapply(seq_len(nrow(keys)), function(i) {
    scenario_id <- keys$scenario_id[[i]]
    repeat_id <- keys$repeat_id[[i]]
    stages <- breakdown[breakdown$scenario_id == scenario_id &
                          breakdown$repeat_id == repeat_id, , drop = FALSE]
    value <- function(stage) {
      x <- stages$elapsed_ms[stages$stage == stage]
      if (length(x) == 0L || !is.finite(x[[1L]])) return(NA_real_)
      as.numeric(x[[1L]])
    }
    skeleton_total <- value("skeleton_total")
    residual <- value("fastspline_residual_prefetch")
    ci <- value("ci_eval_total")
    replay <- value("native_replay")
    plan <- value("plan")
    sum_exclusive <- sum(c(plan, residual, ci, replay), na.rm = TRUE)
    unaccounted <- if (is.finite(skeleton_total)) {
      skeleton_total - sum_exclusive
    } else {
      NA_real_
    }
    data.frame(
      scenario_id = scenario_id,
      repeat_id = as.integer(repeat_id),
      skeleton_total_ms = skeleton_total,
      plan_exclusive_ms = plan,
      residual_prefetch_exclusive_ms = residual,
      ci_eval_exclusive_ms = ci,
      native_replay_exclusive_ms = replay,
      sum_exclusive_ms = sum_exclusive,
      unaccounted_ms = unaccounted,
      accounted_share = if (is.finite(skeleton_total) &&
                              skeleton_total > 0) {
        sum_exclusive / skeleton_total
      } else {
        NA_real_
      },
      cuda_sync_ms = NA_real_,
      cuda_measured_ms = sum(c(
        value("residual_true_batch_total"),
        value("dcov_measured_total")
      ), na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

fastkpc_stage_breakdown_summary <- function(rows) {
  keys <- unique(rows[, c("stage", "stage_group"), drop = FALSE])
  out <- lapply(seq_len(nrow(keys)), function(i) {
    stage <- keys$stage[[i]]
    group <- keys$stage_group[[i]]
    elapsed <- rows$elapsed_ms[rows$stage == stage &
                                 rows$stage_group == group]
    finite <- elapsed[is.finite(elapsed)]
    data.frame(
      stage = stage,
      stage_group = group,
      run_count = as.integer(length(elapsed)),
      finite_count = as.integer(length(finite)),
      median_ms = if (length(finite)) stats::median(finite) else NA_real_,
      p90_ms = if (length(finite)) {
        as.numeric(stats::quantile(finite, 0.9, names = FALSE))
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

fastkpc_run_fast_cuda_stage_breakdown <- function(
    output_dir = file.path("fastkpc", "artifacts",
                           "fast_cuda_stage_breakdown"),
    scenarios = fastkpc_stage_breakdown_default_scenarios(),
    repeats = 3L,
    alpha = 0.2) {
  if (!exists("fastkpc_cuda_available", mode = "function") ||
      !isTRUE(tryCatch(fastkpc_cuda_available(), error = function(e) FALSE))) {
    stop("CUDA unavailable for fast CUDA stage breakdown", call. = FALSE)
  }
  rows <- list()
  runs <- list()
  params <- list(knots = 8, lambda_count = 17, ridge = 1e-8)
  for (scenario in scenarios) {
    for (repeat_id in seq_len(as.integer(repeats))) {
      result <- fast_kpc(
        scenario$data,
        alpha = alpha,
        max_conditioning_size = scenario$max_conditioning_size,
        engine = "cuda",
        precision = "fast",
        graph_stage = "skeleton",
        fastspline_params = params,
        benchmark = TRUE,
        seed = scenario$seed + repeat_id
      )
      rows[[length(rows) + 1L]] <-
        fastkpc_stage_breakdown_rows(result, scenario, repeat_id)
      runs[[length(runs) + 1L]] <-
        fastkpc_stage_breakdown_run_row(result, scenario, repeat_id)
    }
  }
  breakdown <- do.call(rbind, rows)
  run_summary <- do.call(rbind, runs)
  summary <- fastkpc_stage_breakdown_summary(breakdown)
  reconciliation <- fastkpc_stage_breakdown_reconciliation(
    run_summary, breakdown
  )

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  paths <- list(
    breakdown_csv = file.path(output_dir, "fast_cuda_stage_breakdown.csv"),
    runs_csv = file.path(output_dir, "fast_cuda_stage_breakdown_runs.csv"),
    summary_csv = file.path(output_dir,
                            "fast_cuda_stage_breakdown_summary.csv"),
    reconciliation_csv = file.path(
      output_dir,
      "fast_cuda_stage_breakdown_reconciliation.csv"
    ),
    summary_md = file.path(output_dir,
                           "fast_cuda_stage_breakdown_summary.md")
  )
  utils::write.csv(breakdown, paths$breakdown_csv, row.names = FALSE)
  utils::write.csv(run_summary, paths$runs_csv, row.names = FALSE)
  utils::write.csv(summary, paths$summary_csv, row.names = FALSE)
  utils::write.csv(reconciliation, paths$reconciliation_csv,
                   row.names = FALSE)
  md <- c(
    "# Fast CUDA Stage Breakdown",
    "",
    paste0("- Runs: ", nrow(run_summary)),
    paste0("- Route violations: ",
           sum(run_summary$scheduler != "layer" |
                 run_summary$precision_overlay_used %in% TRUE |
                 run_summary$cpu_fallback_count > 0L)),
    paste0("- Median accounted share: ",
           format(signif(stats::median(
             reconciliation$accounted_share,
             na.rm = TRUE
           ), 4), trim = TRUE)),
    paste0("- CSV: `", basename(paths$breakdown_csv), "`")
  )
  writeLines(md, paths$summary_md)
  list(
    breakdown = breakdown,
    runs = run_summary,
    summary = summary,
    reconciliation = reconciliation,
    paths = paths
  )
}
