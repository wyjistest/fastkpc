#include "skeleton_task_scheduler.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <sstream>

namespace {

int idx(int row, int col, int p) {
  return row * p + col;
}

std::vector<int> neighbors_from_snapshot(const std::vector<int>& adjacency,
                                         int p,
                                         int vertex,
                                         int excluded) {
  std::vector<int> out;
  for (int i = 0; i < p; ++i) {
    if (i != excluded && adjacency[idx(i, vertex, p)] != 0) out.push_back(i);
  }
  return out;
}

void enumerate_combinations(const std::vector<int>& values,
                            int choose,
                            const std::function<void(const std::vector<int>&)>& visitor) {
  if (choose == 0) {
    std::vector<int> empty;
    visitor(empty);
    return;
  }
  if (static_cast<int>(values.size()) < choose) return;

  std::vector<int> current;
  std::function<void(int, int)> rec = [&](int start, int remaining) {
    if (remaining == 0) {
      visitor(current);
      return;
    }
    for (int i = start; i <= static_cast<int>(values.size()) - remaining; ++i) {
      current.push_back(values[i]);
      rec(i + 1, remaining - 1);
      current.pop_back();
    }
  };
  rec(0, choose);
}

std::string residual_key(int target,
                         std::vector<int> conditioning_set,
                         int n,
                         int p,
                         const std::string& residual_backend,
                         const std::string& residual_backend_params,
                         const std::string& residual_device) {
  std::sort(conditioning_set.begin(), conditioning_set.end());
  std::ostringstream out;
  out << target << "|" << n << "|" << p << "|" << residual_backend << "|"
      << residual_backend_params << "|" << residual_device << "|";
  for (std::size_t i = 0; i < conditioning_set.size(); ++i) {
    if (i != 0) out << ",";
    out << conditioning_set[i];
  }
  return out.str();
}

}  // namespace

LayerPlan make_layer_plan(const std::vector<int>& adjacency_snapshot,
                          int p,
                          int level) {
  LayerPlan plan;
  plan.level = level;
  plan.p = p;
  plan.adjacency_snapshot = adjacency_snapshot;
  plan.unconditional_tasks = 0;
  plan.conditional_tasks = 0;
  plan.unique_residual_requests = 0;

  int task_id = 0;
  for (int x = 0; x < p - 1; ++x) {
    for (int y = x + 1; y < p; ++y) {
      if (adjacency_snapshot[idx(x, y, p)] == 0) continue;

      const std::vector<int> nx = neighbors_from_snapshot(adjacency_snapshot, p, x, y);
      enumerate_combinations(nx, level, [&](const std::vector<int>& cond) {
        LayerCiTask task;
        task.task_id = task_id++;
        task.level = level;
        task.edge_x = x;
        task.edge_y = y;
        task.orientation_x = x;
        task.orientation_y = y;
        task.conditioning_set = cond;
        task.edge_key = idx(x, y, p);
        plan.tasks.push_back(task);
      });

      const std::vector<int> ny = neighbors_from_snapshot(adjacency_snapshot, p, y, x);
      enumerate_combinations(ny, level, [&](const std::vector<int>& cond) {
        LayerCiTask task;
        task.task_id = task_id++;
        task.level = level;
        task.edge_x = x;
        task.edge_y = y;
        task.orientation_x = y;
        task.orientation_y = x;
        task.conditioning_set = cond;
        task.edge_key = idx(x, y, p);
        plan.tasks.push_back(task);
      });
    }
  }

  for (const LayerCiTask& task : plan.tasks) {
    if (task.conditioning_set.empty()) {
      ++plan.unconditional_tasks;
    } else {
      ++plan.conditional_tasks;
    }
  }
  return plan;
}

std::vector<LayerResidualRequest> collect_unique_residual_requests(
  const LayerPlan& plan,
  int n,
  int p,
  const std::string& residual_backend,
  const std::string& residual_backend_params,
  const std::string& residual_device) {
  std::vector<LayerResidualRequest> out;
  std::map<std::string, int> seen;
  for (const LayerCiTask& task : plan.tasks) {
    if (task.conditioning_set.empty()) continue;
    const int targets[2] = {task.orientation_x, task.orientation_y};
    for (int i = 0; i < 2; ++i) {
      const std::string key = residual_key(targets[i], task.conditioning_set, n, p,
                                           residual_backend, residual_backend_params,
                                           residual_device);
      if (seen.find(key) != seen.end()) continue;
      LayerResidualRequest request;
      request.request_id = static_cast<int>(out.size());
      request.target = targets[i];
      request.conditioning_set = task.conditioning_set;
      request.key = key;
      seen[key] = request.request_id;
      out.push_back(request);
    }
  }
  return out;
}

SchedulerDiagnostics make_scheduler_diagnostics(const std::string& scheduler,
                                                const std::string& scheduler_requested,
                                                int dcov_batch_size_requested,
                                                int residual_batch_size_requested) {
  SchedulerDiagnostics out;
  out.scheduler = scheduler;
  out.scheduler_requested = scheduler_requested;
  out.levels = 0;
  out.tasks_planned = 0;
  out.tasks_evaluated = 0;
  out.tests_replayed = 0;
  out.tasks_ignored_after_delete = 0;
  out.dcov_batches = 0;
  out.residual_requests = 0;
  out.unique_residual_requests = 0;
  out.residual_batches = 0;
  out.cuda_residual_batch_groups = 0;
  out.cuda_residual_true_batched_groups = 0;
  out.cuda_residual_true_batched_fits = 0;
  out.cuda_residual_single_fit_calls = 0;
  out.cuda_residual_cpu_fallback_fits = 0;
  out.cuda_residual_unique_designs = 0;
  out.cuda_residual_duplicate_design_fits = 0;
  out.cuda_residual_max_fits_per_design = 0;
  out.max_level_tasks = 0;
  out.max_level_unique_residuals = 0;
  out.dcov_batch_size_requested = dcov_batch_size_requested;
  out.dcov_batch_size_used = 0;
  out.residual_batch_size_requested = residual_batch_size_requested;
  out.residual_batch_size_used = 0;
  out.plan_elapsed_sec = 0.0;
  out.residual_prefetch_elapsed_sec = 0.0;
  out.residual_request_collect_sec = 0.0;
  out.residual_prefetch_missing_scan_sec = 0.0;
  out.residual_prefetch_batch_input_sec = 0.0;
  out.residual_batch_call_wall_sec = 0.0;
  out.residual_diagnostic_merge_sec = 0.0;
  out.residual_prefetch_unaccounted_sec = 0.0;
  out.residual_grouping_sec = 0.0;
  out.residual_grouping_condition_key_sec = 0.0;
  out.residual_grouping_group_key_sec = 0.0;
  out.residual_grouping_design_build_sec = 0.0;
  out.residual_grouping_map_insert_sec = 0.0;
  out.residual_grouping_design_cache_lookup_sec = 0.0;
  out.residual_grouping_design_cache_insert_sec = 0.0;
  out.residual_grouping_group_lookup_sec = 0.0;
  out.residual_grouping_group_insert_sec = 0.0;
  out.residual_grouping_group_design_lookup_sec = 0.0;
  out.residual_grouping_group_design_copy_sec = 0.0;
  out.residual_grouping_group_design_index_insert_sec = 0.0;
  out.residual_grouping_request_insert_sec = 0.0;
  out.residual_grouping_unaccounted_sec = 0.0;
  out.residual_grouping_group_count = 0;
  out.residual_grouping_design_count = 0;
  out.residual_grouping_condition_key_sort_count = 0;
  out.residual_grouping_string_key_count = 0;
  out.residual_structural_group_key_count = 0;
  out.residual_structural_condition_key_count = 0;
  out.residual_string_group_key_count = 0;
  out.residual_string_condition_key_count = 0;
  out.residual_grouping_group_design_copy_count = 0;
  out.residual_grouping_group_design_x_values = 0;
  out.residual_grouping_group_design_p_values = 0;
  out.residual_grouping_request_insert_count = 0;
  out.residual_design_cache_hit_count = 0;
  out.residual_design_cache_miss_count = 0;
  out.residual_design_cache_insert_count = 0;
  out.residual_design_cache_entries = 0;
  out.residual_design_build_total_sec = 0.0;
  out.residual_design_build_basis_sec = 0.0;
  out.residual_design_build_penalty_sec = 0.0;
  out.residual_design_build_x_pack_sec = 0.0;
  out.residual_design_build_p_pack_sec = 0.0;
  out.residual_design_build_alloc_sec = 0.0;
  out.residual_design_build_column_extract_sec = 0.0;
  out.residual_design_build_finite_check_sec = 0.0;
  out.residual_design_build_unaccounted_sec = 0.0;
  out.residual_design_build_count = 0;
  out.residual_design_build_x_values = 0;
  out.residual_design_build_p_values = 0;
  out.residual_design_build_basis_values = 0;
  out.residual_design_build_penalty_values = 0;
  out.residual_design_build_condition_cols = 0;
  out.residual_design_build_finite_check_values = 0;
  out.residual_basis_cache_hit_count = 0;
  out.residual_basis_cache_miss_count = 0;
  out.residual_basis_cache_insert_count = 0;
  out.residual_basis_cache_entries = 0;
  out.residual_basis_cache_hit_sec = 0.0;
  out.residual_basis_cache_miss_build_sec = 0.0;
  out.residual_basis_build_total_sec = 0.0;
  out.residual_basis_build_alloc_sec = 0.0;
  out.residual_basis_build_near_constant_sec = 0.0;
  out.residual_basis_build_knots_sec = 0.0;
  out.residual_basis_build_knots_copy_sec = 0.0;
  out.residual_basis_build_knots_sort_sec = 0.0;
  out.residual_basis_build_knots_center_sec = 0.0;
  out.residual_basis_build_min_gap_sec = 0.0;
  out.residual_basis_build_width_sec = 0.0;
  out.residual_basis_build_eval_sec = 0.0;
  out.residual_basis_build_eval_fill_sec = 0.0;
  out.residual_basis_build_normalize_sec = 0.0;
  out.residual_basis_build_normalize_scale_sec = 0.0;
  out.residual_basis_build_fallback_sec = 0.0;
  out.residual_basis_build_return_sec = 0.0;
  out.residual_basis_build_unaccounted_sec = 0.0;
  out.residual_basis_build_count = 0;
  out.residual_basis_build_rows = 0;
  out.residual_basis_build_cols = 0;
  out.residual_basis_build_values = 0;
  out.residual_basis_build_near_constant_count = 0;
  out.residual_basis_build_fallback_row_count = 0;
  out.residual_host_pack_sec = 0.0;
  out.residual_alloc_sec = 0.0;
  out.residual_h2d_sec = 0.0;
  out.residual_h2d_design_sec = 0.0;
  out.residual_h2d_penalty_sec = 0.0;
  out.residual_h2d_y_sec = 0.0;
  out.residual_h2d_index_sec = 0.0;
  out.residual_h2d_lambda_sec = 0.0;
  out.residual_h2d_active_sec = 0.0;
  out.residual_h2d_copy_count = 0;
  out.residual_h2d_bytes = 0.0;
  out.residual_h2d_design_bytes = 0.0;
  out.residual_h2d_y_bytes = 0.0;
  out.residual_h2d_metadata_bytes = 0.0;
  out.residual_h2d_metadata_coalesced_count = 0;
  out.residual_h2d_metadata_coalesced_bytes = 0.0;
  out.residual_h2d_selected_metadata_copy_count = 0;
  out.residual_xtx_xty_sec = 0.0;
  out.residual_pointer_setup_sec = 0.0;
  out.residual_active_copy_sec = 0.0;
  out.residual_build_system_sec = 0.0;
  out.residual_factor_solve_sec = 0.0;
  out.residual_factor_cholesky_sec = 0.0;
  out.residual_factor_rhs_solve_sec = 0.0;
  out.residual_factor_inverse_solve_sec = 0.0;
  out.residual_summary_sec = 0.0;
  out.residual_d2h_sec = 0.0;
  out.residual_d2h_residuals_sec = 0.0;
  out.residual_d2h_metadata_sec = 0.0;
  out.residual_d2h_info_sec = 0.0;
  out.residual_d2h_copy_count = 0;
  out.residual_d2h_bytes = 0.0;
  out.residual_d2h_residual_bytes = 0.0;
  out.residual_d2h_metadata_bytes = 0.0;
  out.residual_d2h_metadata_coalesced_count = 0;
  out.residual_d2h_metadata_coalesced_bytes = 0.0;
  out.residual_host_select_sec = 0.0;
  out.residual_free_sec = 0.0;
  out.residual_true_batch_total_sec = 0.0;
  out.residual_factorization_count = 0;
  out.residual_rhs_solve_count = 0;
  out.residual_inverse_solve_count = 0;
  out.residual_rhs_solve_api_calls = 0;
  out.residual_rhs_target_solves = 0;
  out.residual_rhs_custom_solve_count = 0;
  out.residual_rhs_cublas_solve_count = 0;
  out.residual_rhs_solve_fallback_count = 0;
  out.residual_rhs_custom_solve_sec = 0.0;
  out.residual_rhs_cublas_solve_sec = 0.0;
  out.residual_candidate_rhs_fused_solve_count = 0;
  out.residual_candidate_rhs_materialized_solve_count = 0;
  out.residual_selected_rhs_materialized_solve_count = 0;
  out.residual_candidate_beta_values_avoided = 0;
  out.residual_summary_candidate_launch_count = 0;
  out.residual_summary_group_batched_launch_count = 0;
  out.residual_summary_group_batched_candidate_count = 0;
  out.residual_winning_factor_reuse_count = 0;
  out.residual_factor_cache_hits = 0;
  out.residual_factor_cache_misses = 0;
  out.residual_factor_cache_entries = 0;
  out.residual_factor_cache_bytes = 0.0;
  out.residual_lambda_candidates = 0;
  out.residual_workspace_reuse_count = 0;
  out.residual_workspace_grow_count = 0;
  out.residual_workspace_slab_grow_count = 0;
  out.residual_workspace_slab_reuse_count = 0;
  out.residual_workspace_slab_bytes = 0.0;
  out.residual_workspace_legacy_alloc_count = 0;
  out.residual_solver_handle_create_count = 0;
  out.residual_per_request_design_x_values = 0;
  out.residual_duplicate_design_x_values_avoided = 0;
  out.residual_cache_insert_sec = 0.0;
  out.residual_cache_move_insert_count = 0;
  out.residual_cache_copy_insert_count = 0;
  out.residual_algebraic_rss_count = 0;
  out.residual_candidate_residual_materialize_count = 0;
  out.residual_winning_residual_materialize_count = 0;
  out.residual_algebraic_rss_clamp_count = 0;
  out.residual_only_batch_count = 0;
  out.residual_full_fit_batch_count = 0;
  out.residual_only_fit_count = 0;
  out.residual_full_fit_materialize_count = 0;
  out.residual_fitted_values_avoided = 0;
  out.residual_result_materialize_sec = 0.0;
  out.residual_fitted_materialize_sec = 0.0;
  out.residual_batch_top_level_wall_sec = 0.0;
  out.residual_batch_top_level_unaccounted_sec = 0.0;
  out.ci_eval_elapsed_sec = 0.0;
  out.ci_host_pack_sec = 0.0;
  out.ci_dcov_call_wall_sec = 0.0;
  out.ci_pvalue_copy_sec = 0.0;
  out.ci_diagnostic_append_sec = 0.0;
  out.ci_eval_unaccounted_sec = 0.0;
  out.replay_elapsed_sec = 0.0;
  out.total_elapsed_sec = 0.0;
  out.dcov_alloc_sec = 0.0;
  out.dcov_h2d_sec = 0.0;
  out.dcov_memset_sec = 0.0;
  out.dcov_rowsum_sec = 0.0;
  out.dcov_totals_d2h_sec = 0.0;
  out.dcov_reduce_sec = 0.0;
  out.dcov_scalars_d2h_sec = 0.0;
  out.dcov_host_scalar_sec = 0.0;
  out.dcov_result_materialize_sec = 0.0;
  out.dcov_free_sec = 0.0;
  out.dcov_total_sec = 0.0;
  out.dcov_top_level_wall_sec = 0.0;
  out.dcov_grid_limit_query_sec = 0.0;
  out.dcov_chunk_dispatch_sec = 0.0;
  out.dcov_top_level_unaccounted_sec = 0.0;
  out.dcov_chunks = 0;
  out.dcov_max_chunk_batch = 0;
  out.dcov_workspace_reuse_count = 0;
  out.dcov_workspace_grow_count = 0;
  out.dcov_raw_aggregate_fused_count = 0;
  out.dcov_row_product_reduce_count = 0;
  out.dcov_pvalue_only_count = 0;
  out.dcov_full_result_materialize_count = 0;
  out.dcov_grid_limit_query_count = 0;
  out.dcov_grid_limit_cache_hit_count = 0;
  out.dcov_grid_limit_process_cache_hit_count = 0;
  return out;
}

int resolve_dcov_batch_size(int requested_batch_size,
                            int,
                            int planned_tasks) {
  if (requested_batch_size > 0) return requested_batch_size;
  const int default_auto_batch_size = 512;
  if (planned_tasks <= 0) return 1;
  return std::min(planned_tasks, default_auto_batch_size);
}

int resolve_residual_batch_size(int requested_residual_batch_size,
                                int unique_residual_requests) {
  if (requested_residual_batch_size > 0) return requested_residual_batch_size;
  const int default_auto_residual_batch_size = 256;
  if (unique_residual_requests <= 0) return 1;
  return std::min(unique_residual_requests, default_auto_residual_batch_size);
}
