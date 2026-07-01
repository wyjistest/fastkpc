#ifndef FASTKPC_SKELETON_TASK_SCHEDULER_HPP
#define FASTKPC_SKELETON_TASK_SCHEDULER_HPP

#include <string>
#include <vector>

struct LayerCiTask {
  int task_id;
  int level;
  int edge_x;
  int edge_y;
  int orientation_x;
  int orientation_y;
  std::vector<int> conditioning_set;
  int edge_key;
};

struct LayerResidualRequest {
  int request_id;
  int target;
  std::vector<int> conditioning_set;
  std::string key;
};

struct LayerPlan {
  int level;
  int p;
  std::vector<int> adjacency_snapshot;
  std::vector<LayerCiTask> tasks;
  int unconditional_tasks;
  int conditional_tasks;
  int unique_residual_requests;
};

struct LayerDiagnosticsLevel {
  int level;
  int tasks_planned;
  int tasks_evaluated;
  int tests_replayed;
  int tasks_ignored_after_delete;
  int deletions;
  int unconditional_tasks;
  int conditional_tasks;
  int unique_residual_requests;
  int dcov_batches;
  int residual_batches;
  double plan_elapsed_sec;
  double residual_prefetch_elapsed_sec;
  double ci_eval_elapsed_sec;
  double replay_elapsed_sec;
  double total_elapsed_sec;
};

struct SchedulerBatchDiagnostic {
  int level;
  int batch_id;
  std::string kind;
  int start_task_id;
  int task_count;
  int n;
  std::string status;
  int groups;
  int true_batched_groups;
  int true_batched_fits;
  int single_fit_calls;
  int cpu_fallback_fits;
  int unique_designs;
  int duplicate_design_fits;
  int max_fits_per_design;
  int max_group_size;
  int min_group_size;
  int max_design_cols;
  int min_design_cols;
};

struct SchedulerResidualDiagnostic {
  int level;
  int request_id;
  int target;
  int conditioning_size;
  std::string residual_backend;
  std::string residual_device;
  bool materialized;
  bool fallback_used;
  std::string reason;
};

struct SchedulerDiagnostics {
  std::string scheduler;
  std::string scheduler_requested;
  int levels;
  int tasks_planned;
  int tasks_evaluated;
  int tests_replayed;
  int tasks_ignored_after_delete;
  int dcov_batches;
  int residual_requests;
  int unique_residual_requests;
  int residual_batches;
  int cuda_residual_batch_groups;
  int cuda_residual_true_batched_groups;
  int cuda_residual_true_batched_fits;
  int cuda_residual_single_fit_calls;
  int cuda_residual_cpu_fallback_fits;
  int cuda_residual_unique_designs;
  int cuda_residual_duplicate_design_fits;
  int cuda_residual_max_fits_per_design;
  int max_level_tasks;
  int max_level_unique_residuals;
  int dcov_batch_size_requested;
  int dcov_batch_size_used;
  int residual_batch_size_requested;
  int residual_batch_size_used;
  double plan_elapsed_sec;
  double residual_prefetch_elapsed_sec;
  double residual_request_collect_sec;
  double residual_prefetch_missing_scan_sec;
  double residual_prefetch_batch_input_sec;
  double residual_batch_call_wall_sec;
  double residual_diagnostic_merge_sec;
  double residual_prefetch_unaccounted_sec;
  double residual_grouping_sec;
  double residual_grouping_condition_key_sec;
  double residual_grouping_group_key_sec;
  double residual_grouping_design_build_sec;
  double residual_grouping_map_insert_sec;
  double residual_grouping_unaccounted_sec;
  int residual_grouping_group_count;
  int residual_grouping_design_count;
  int residual_grouping_condition_key_sort_count;
  int residual_grouping_string_key_count;
  int residual_design_cache_hit_count;
  int residual_design_cache_miss_count;
  int residual_design_cache_insert_count;
  int residual_design_cache_entries;
  double residual_design_build_total_sec;
  double residual_design_build_basis_sec;
  double residual_design_build_penalty_sec;
  double residual_design_build_x_pack_sec;
  double residual_design_build_p_pack_sec;
  double residual_design_build_alloc_sec;
  double residual_design_build_column_extract_sec;
  double residual_design_build_unaccounted_sec;
  int residual_design_build_count;
  int residual_design_build_x_values;
  int residual_design_build_p_values;
  int residual_design_build_basis_values;
  int residual_design_build_penalty_values;
  int residual_design_build_condition_cols;
  int residual_basis_cache_hit_count;
  int residual_basis_cache_miss_count;
  int residual_basis_cache_insert_count;
  int residual_basis_cache_entries;
  double residual_basis_cache_hit_sec;
  double residual_basis_cache_miss_build_sec;
  double residual_basis_build_total_sec;
  double residual_basis_build_alloc_sec;
  double residual_basis_build_near_constant_sec;
  double residual_basis_build_knots_sec;
  double residual_basis_build_min_gap_sec;
  double residual_basis_build_eval_sec;
  double residual_basis_build_normalize_sec;
  double residual_basis_build_fallback_sec;
  double residual_basis_build_unaccounted_sec;
  int residual_basis_build_count;
  int residual_basis_build_rows;
  int residual_basis_build_cols;
  int residual_basis_build_values;
  int residual_basis_build_near_constant_count;
  int residual_basis_build_fallback_row_count;
  double residual_host_pack_sec;
  double residual_alloc_sec;
  double residual_h2d_sec;
  double residual_h2d_design_sec;
  double residual_h2d_penalty_sec;
  double residual_h2d_y_sec;
  double residual_h2d_index_sec;
  double residual_h2d_lambda_sec;
  double residual_h2d_active_sec;
  int residual_h2d_copy_count;
  double residual_h2d_bytes;
  double residual_h2d_design_bytes;
  double residual_h2d_y_bytes;
  double residual_h2d_metadata_bytes;
  int residual_h2d_metadata_coalesced_count;
  double residual_h2d_metadata_coalesced_bytes;
  int residual_h2d_selected_metadata_copy_count;
  double residual_xtx_xty_sec;
  double residual_pointer_setup_sec;
  double residual_active_copy_sec;
  double residual_build_system_sec;
  double residual_factor_solve_sec;
  double residual_factor_cholesky_sec;
  double residual_factor_rhs_solve_sec;
  double residual_factor_inverse_solve_sec;
  double residual_summary_sec;
  double residual_d2h_sec;
  double residual_d2h_residuals_sec;
  double residual_d2h_metadata_sec;
  double residual_d2h_info_sec;
  int residual_d2h_copy_count;
  double residual_d2h_bytes;
  double residual_d2h_residual_bytes;
  double residual_d2h_metadata_bytes;
  int residual_d2h_metadata_coalesced_count;
  double residual_d2h_metadata_coalesced_bytes;
  double residual_host_select_sec;
  double residual_free_sec;
  double residual_true_batch_total_sec;
  int residual_factorization_count;
  int residual_rhs_solve_count;
  int residual_inverse_solve_count;
  int residual_rhs_solve_api_calls;
  int residual_rhs_target_solves;
  int residual_rhs_custom_solve_count;
  int residual_rhs_cublas_solve_count;
  int residual_rhs_solve_fallback_count;
  double residual_rhs_custom_solve_sec;
  double residual_rhs_cublas_solve_sec;
  int residual_candidate_rhs_fused_solve_count;
  int residual_candidate_rhs_materialized_solve_count;
  int residual_selected_rhs_materialized_solve_count;
  int residual_candidate_beta_values_avoided;
  int residual_summary_candidate_launch_count;
  int residual_summary_group_batched_launch_count;
  int residual_summary_group_batched_candidate_count;
  int residual_winning_factor_reuse_count;
  int residual_factor_cache_hits;
  int residual_factor_cache_misses;
  int residual_factor_cache_entries;
  double residual_factor_cache_bytes;
  int residual_lambda_candidates;
  int residual_workspace_reuse_count;
  int residual_workspace_grow_count;
  int residual_workspace_slab_grow_count;
  int residual_workspace_slab_reuse_count;
  double residual_workspace_slab_bytes;
  int residual_workspace_legacy_alloc_count;
  int residual_solver_handle_create_count;
  int residual_per_request_design_x_values;
  int residual_duplicate_design_x_values_avoided;
  double residual_cache_insert_sec;
  int residual_cache_move_insert_count;
  int residual_cache_copy_insert_count;
  int residual_algebraic_rss_count;
  int residual_candidate_residual_materialize_count;
  int residual_winning_residual_materialize_count;
  int residual_algebraic_rss_clamp_count;
  int residual_only_batch_count;
  int residual_full_fit_batch_count;
  int residual_only_fit_count;
  int residual_full_fit_materialize_count;
  int residual_fitted_values_avoided;
  double residual_result_materialize_sec;
  double residual_fitted_materialize_sec;
  double residual_batch_top_level_wall_sec;
  double residual_batch_top_level_unaccounted_sec;
  double ci_eval_elapsed_sec;
  double ci_host_pack_sec;
  double ci_dcov_call_wall_sec;
  double ci_pvalue_copy_sec;
  double ci_diagnostic_append_sec;
  double ci_eval_unaccounted_sec;
  double replay_elapsed_sec;
  double total_elapsed_sec;
  double dcov_alloc_sec;
  double dcov_h2d_sec;
  double dcov_memset_sec;
  double dcov_rowsum_sec;
  double dcov_totals_d2h_sec;
  double dcov_reduce_sec;
  double dcov_scalars_d2h_sec;
  double dcov_host_scalar_sec;
  double dcov_result_materialize_sec;
  double dcov_free_sec;
  double dcov_total_sec;
  double dcov_top_level_wall_sec;
  double dcov_grid_limit_query_sec;
  double dcov_chunk_dispatch_sec;
  double dcov_top_level_unaccounted_sec;
  int dcov_chunks;
  int dcov_max_chunk_batch;
  int dcov_workspace_reuse_count;
  int dcov_workspace_grow_count;
  int dcov_raw_aggregate_fused_count;
  int dcov_row_product_reduce_count;
  int dcov_pvalue_only_count;
  int dcov_full_result_materialize_count;
  int dcov_grid_limit_query_count;
  int dcov_grid_limit_cache_hit_count;
  int dcov_grid_limit_process_cache_hit_count;
  std::vector<LayerDiagnosticsLevel> per_level;
  std::vector<SchedulerBatchDiagnostic> batches;
  std::vector<SchedulerResidualDiagnostic> residuals;
};

LayerPlan make_layer_plan(const std::vector<int>& adjacency_snapshot,
                          int p,
                          int level);

std::vector<LayerResidualRequest> collect_unique_residual_requests(
  const LayerPlan& plan,
  int n,
  int p,
  const std::string& residual_backend,
  const std::string& residual_backend_params,
  const std::string& residual_device);

SchedulerDiagnostics make_scheduler_diagnostics(const std::string& scheduler,
                                                const std::string& scheduler_requested,
                                                int dcov_batch_size_requested,
                                                int residual_batch_size_requested);

int resolve_dcov_batch_size(int requested_batch_size,
                            int n,
                            int planned_tasks);

int resolve_residual_batch_size(int requested_residual_batch_size,
                                int unique_residual_requests);

#endif
