#include "dcov_batch_types.hpp"
#include "fastspline_basis.hpp"
#include "hsic_cpu.hpp"
#include "orientation_types.hpp"
#include "regrvonps_device.hpp"
#include "residual_backend_registry.hpp"
#include "skeleton_engine_cuda.hpp"
#include "wanpdag_engine.hpp"
#include "cuda/cuda_status.hpp"
#include "cuda/dcov_batch_cuda.hpp"
#include "cuda/fastspline_residual_cuda.hpp"
#include "cuda/hsic_batch_cuda.hpp"
#include "cuda/mgcv_extract_fixed_sp_cuda.hpp"

#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {

bool all_finite(Rcpp::NumericMatrix values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

bool all_finite_vector(Rcpp::NumericVector values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

Rcpp::LogicalMatrix adjacency_to_matrix(const std::vector<int>& adjacency, int p) {
  Rcpp::LogicalMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = adjacency[static_cast<std::size_t>(i) * p + j] != 0;
    }
  }
  return out;
}

Rcpp::NumericMatrix pmax_to_matrix(const std::vector<double>& pmax, int p) {
  Rcpp::NumericMatrix out(p, p);
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      out(i, j) = pmax[static_cast<std::size_t>(i) * p + j];
    }
  }
  return out;
}

Rcpp::List sepsets_to_list(const std::vector<std::vector<std::vector<int> > >& sepsets) {
  const int p = static_cast<int>(sepsets.size());
  Rcpp::List out(p);
  for (int i = 0; i < p; ++i) {
    Rcpp::List row(p);
    for (int j = 0; j < p; ++j) {
      Rcpp::IntegerVector value(sepsets[i][j].size());
      for (int k = 0; k < value.size(); ++k) value[k] = sepsets[i][j][k] + 1;
      row[j] = value;
    }
    out[i] = row;
  }
  return out;
}

Rcpp::List level_log_to_list(const std::vector<std::vector<LevelDeletion> >& logs) {
  Rcpp::List out(logs.size());
  for (int level = 0; level < static_cast<int>(logs.size()); ++level) {
    Rcpp::List entries(logs[level].size());
    for (int i = 0; i < static_cast<int>(logs[level].size()); ++i) {
      const LevelDeletion& entry = logs[level][i];
      Rcpp::IntegerVector cond(entry.conditioning_set.size());
      for (int k = 0; k < cond.size(); ++k) cond[k] = entry.conditioning_set[k] + 1;
      entries[i] = Rcpp::List::create(
        Rcpp::Named("x") = entry.x + 1,
        Rcpp::Named("y") = entry.y + 1,
        Rcpp::Named("S") = cond,
        Rcpp::Named("p.value") = entry.p_value
      );
    }
    out[level] = entries;
  }
  return out;
}

Rcpp::List residual_cache_stats_to_list(const SkeletonResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("enabled") = result.residual_cache_enabled,
    Rcpp::Named("requests") = result.residual_cache_requests,
    Rcpp::Named("hits") = result.residual_cache_hits,
    Rcpp::Named("misses") = result.residual_cache_misses,
    Rcpp::Named("computations") = result.residual_cache_computations,
    Rcpp::Named("stored_vectors") = result.residual_cache_stored_vectors,
    Rcpp::Named("stored_values") = result.residual_cache_stored_values,
    Rcpp::Named("backend_name") = result.residual_backend,
    Rcpp::Named("residual_device") = result.residual_device
  );
}

Rcpp::List hsic_batch_result_to_list(const HsicBatchResult& result,
                                     int pair,
                                     double sig) {
  Rcpp::NumericVector replicates;
  if (!result.permutation_replicates.empty()) {
    const int reps = result.diagnostics.permutation_replicates;
    replicates = Rcpp::NumericVector(reps);
    const std::size_t base = static_cast<std::size_t>(pair) * reps;
    for (int i = 0; i < reps; ++i) replicates[i] = result.permutation_replicates[base + i];
  }

  return Rcpp::List::create(
    Rcpp::Named("method") =
      result.diagnostics.permutation_replicates > 0 ? "hsic.perm" : "hsic.gamma",
    Rcpp::Named("backend") = result.diagnostics.backend,
    Rcpp::Named("statistic") = result.statistics[pair],
    Rcpp::Named("estimate") = result.statistics[pair],
    Rcpp::Named("estimates") = Rcpp::NumericVector::create(
      Rcpp::Named("HSIC") = result.statistics[pair],
      Rcpp::Named("HSIC mean") = result.means[pair],
      Rcpp::Named("HSIC variance") = result.variances[pair]
    ),
    Rcpp::Named("p.value") = result.p_values[pair],
    Rcpp::Named("replicates") = replicates,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = result.diagnostics.n,
      Rcpp::Named("pairs") = result.diagnostics.pairs,
      Rcpp::Named("backend") = result.diagnostics.backend,
      Rcpp::Named("kernel") = "rbf",
      Rcpp::Named("sig") = sig,
      Rcpp::Named("bytes_allocated") =
        static_cast<double>(result.diagnostics.bytes_allocated),
      Rcpp::Named("cuda_blocks") = result.diagnostics.cuda_blocks,
      Rcpp::Named("cuda_threads") = result.diagnostics.cuda_threads,
      Rcpp::Named("replicates") = result.diagnostics.permutation_replicates,
      Rcpp::Named("used_seed") = result.diagnostics.used_seed,
      Rcpp::Named("seed") = result.diagnostics.used_seed ?
        static_cast<int>(result.diagnostics.seed) : NA_INTEGER,
      Rcpp::Named("reason") = result.diagnostics.reason,
      Rcpp::Named("shape") = result.shapes[pair],
      Rcpp::Named("scale") = result.scales[pair]
    )
  );
}

Rcpp::DataFrame scheduler_levels_to_data_frame(
  const std::vector<LayerDiagnosticsLevel>& levels) {
  const int n = static_cast<int>(levels.size());
  Rcpp::IntegerVector level(n), tasks_planned(n), tasks_evaluated(n);
  Rcpp::IntegerVector tests_replayed(n), tasks_ignored_after_delete(n);
  Rcpp::IntegerVector deletions(n), unconditional_tasks(n), conditional_tasks(n);
  Rcpp::IntegerVector unique_residual_requests(n), dcov_batches(n);
  Rcpp::IntegerVector residual_batches(n);
  Rcpp::NumericVector plan_elapsed_sec(n), residual_prefetch_elapsed_sec(n);
  Rcpp::NumericVector ci_eval_elapsed_sec(n), replay_elapsed_sec(n);
  Rcpp::NumericVector total_elapsed_sec(n);
  for (int i = 0; i < n; ++i) {
    level[i] = levels[i].level;
    tasks_planned[i] = levels[i].tasks_planned;
    tasks_evaluated[i] = levels[i].tasks_evaluated;
    tests_replayed[i] = levels[i].tests_replayed;
    tasks_ignored_after_delete[i] = levels[i].tasks_ignored_after_delete;
    deletions[i] = levels[i].deletions;
    unconditional_tasks[i] = levels[i].unconditional_tasks;
    conditional_tasks[i] = levels[i].conditional_tasks;
    unique_residual_requests[i] = levels[i].unique_residual_requests;
    dcov_batches[i] = levels[i].dcov_batches;
    residual_batches[i] = levels[i].residual_batches;
    plan_elapsed_sec[i] = levels[i].plan_elapsed_sec;
    residual_prefetch_elapsed_sec[i] = levels[i].residual_prefetch_elapsed_sec;
    ci_eval_elapsed_sec[i] = levels[i].ci_eval_elapsed_sec;
    replay_elapsed_sec[i] = levels[i].replay_elapsed_sec;
    total_elapsed_sec[i] = levels[i].total_elapsed_sec;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("tasks_planned") = tasks_planned,
    Rcpp::Named("tasks_evaluated") = tasks_evaluated,
    Rcpp::Named("tests_replayed") = tests_replayed,
    Rcpp::Named("tasks_ignored_after_delete") = tasks_ignored_after_delete,
    Rcpp::Named("deletions") = deletions,
    Rcpp::Named("unconditional_tasks") = unconditional_tasks,
    Rcpp::Named("conditional_tasks") = conditional_tasks,
    Rcpp::Named("unique_residual_requests") = unique_residual_requests,
    Rcpp::Named("dcov_batches") = dcov_batches,
    Rcpp::Named("residual_batches") = residual_batches,
    Rcpp::Named("plan_elapsed_sec") = plan_elapsed_sec,
    Rcpp::Named("residual_prefetch_elapsed_sec") =
      residual_prefetch_elapsed_sec,
    Rcpp::Named("ci_eval_elapsed_sec") = ci_eval_elapsed_sec,
    Rcpp::Named("replay_elapsed_sec") = replay_elapsed_sec,
    Rcpp::Named("total_elapsed_sec") = total_elapsed_sec,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::DataFrame scheduler_batches_to_data_frame(
  const std::vector<SchedulerBatchDiagnostic>& batches) {
  const int n = static_cast<int>(batches.size());
  Rcpp::IntegerVector level(n), batch_id(n), start_task_id(n), task_count(n), rows(n);
  Rcpp::IntegerVector groups(n), true_batched_groups(n), true_batched_fits(n);
  Rcpp::IntegerVector single_fit_calls(n), cpu_fallback_fits(n);
  Rcpp::IntegerVector unique_designs(n), duplicate_design_fits(n);
  Rcpp::IntegerVector max_fits_per_design(n);
  Rcpp::IntegerVector max_group_size(n), min_group_size(n);
  Rcpp::IntegerVector max_design_cols(n), min_design_cols(n);
  Rcpp::CharacterVector kind(n), status(n);
  for (int i = 0; i < n; ++i) {
    level[i] = batches[i].level;
    batch_id[i] = batches[i].batch_id;
    kind[i] = batches[i].kind;
    start_task_id[i] = batches[i].start_task_id;
    task_count[i] = batches[i].task_count;
    rows[i] = batches[i].n;
    status[i] = batches[i].status;
    groups[i] = batches[i].groups;
    true_batched_groups[i] = batches[i].true_batched_groups;
    true_batched_fits[i] = batches[i].true_batched_fits;
    single_fit_calls[i] = batches[i].single_fit_calls;
    cpu_fallback_fits[i] = batches[i].cpu_fallback_fits;
    unique_designs[i] = batches[i].unique_designs;
    duplicate_design_fits[i] = batches[i].duplicate_design_fits;
    max_fits_per_design[i] = batches[i].max_fits_per_design;
    max_group_size[i] = batches[i].max_group_size;
    min_group_size[i] = batches[i].min_group_size;
    max_design_cols[i] = batches[i].max_design_cols;
    min_design_cols[i] = batches[i].min_design_cols;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("batch_id") = batch_id,
    Rcpp::Named("kind") = kind,
    Rcpp::Named("start_task_id") = start_task_id,
    Rcpp::Named("task_count") = task_count,
    Rcpp::Named("n") = rows,
    Rcpp::Named("status") = status,
    Rcpp::Named("groups") = groups,
    Rcpp::Named("true_batched_groups") = true_batched_groups,
    Rcpp::Named("true_batched_fits") = true_batched_fits,
    Rcpp::Named("single_fit_calls") = single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = cpu_fallback_fits,
    Rcpp::Named("unique_designs") = unique_designs,
    Rcpp::Named("duplicate_design_fits") = duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = max_fits_per_design,
    Rcpp::Named("max_group_size") = max_group_size,
    Rcpp::Named("min_group_size") = min_group_size,
    Rcpp::Named("max_design_cols") = max_design_cols,
    Rcpp::Named("min_design_cols") = min_design_cols,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::DataFrame scheduler_residuals_to_data_frame(
  const std::vector<SchedulerResidualDiagnostic>& residuals) {
  const int n = static_cast<int>(residuals.size());
  Rcpp::IntegerVector level(n), request_id(n), target(n), conditioning_size(n);
  Rcpp::CharacterVector residual_backend(n), residual_device(n), reason(n);
  Rcpp::LogicalVector materialized(n), fallback_used(n);
  for (int i = 0; i < n; ++i) {
    level[i] = residuals[i].level;
    request_id[i] = residuals[i].request_id;
    target[i] = residuals[i].target + 1;
    conditioning_size[i] = residuals[i].conditioning_size;
    residual_backend[i] = residuals[i].residual_backend;
    residual_device[i] = residuals[i].residual_device;
    materialized[i] = residuals[i].materialized;
    fallback_used[i] = residuals[i].fallback_used;
    reason[i] = residuals[i].reason;
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("level") = level,
    Rcpp::Named("request_id") = request_id,
    Rcpp::Named("target") = target,
    Rcpp::Named("conditioning_size") = conditioning_size,
    Rcpp::Named("residual_backend") = residual_backend,
    Rcpp::Named("residual_device") = residual_device,
    Rcpp::Named("materialized") = materialized,
    Rcpp::Named("fallback_used") = fallback_used,
    Rcpp::Named("reason") = reason,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List scheduler_diagnostics_to_list(const SchedulerDiagnostics& diagnostics) {
  return Rcpp::List::create(
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("scheduler") = diagnostics.scheduler,
      Rcpp::Named("scheduler_requested") = diagnostics.scheduler_requested,
      Rcpp::Named("levels") = diagnostics.levels,
      Rcpp::Named("tasks_planned") = diagnostics.tasks_planned,
      Rcpp::Named("tasks_evaluated") = diagnostics.tasks_evaluated,
      Rcpp::Named("tests_replayed") = diagnostics.tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") =
        diagnostics.tasks_ignored_after_delete,
      Rcpp::Named("dcov_batches") = diagnostics.dcov_batches,
      Rcpp::Named("residual_requests") = diagnostics.residual_requests,
      Rcpp::Named("unique_residual_requests") =
        diagnostics.unique_residual_requests,
      Rcpp::Named("residual_batches") = diagnostics.residual_batches,
      Rcpp::Named("cuda_residual_batch_groups") =
        diagnostics.cuda_residual_batch_groups,
      Rcpp::Named("cuda_residual_true_batched_groups") =
        diagnostics.cuda_residual_true_batched_groups,
      Rcpp::Named("cuda_residual_true_batched_fits") =
        diagnostics.cuda_residual_true_batched_fits,
      Rcpp::Named("cuda_residual_single_fit_calls") =
        diagnostics.cuda_residual_single_fit_calls,
      Rcpp::Named("cuda_residual_cpu_fallback_fits") =
        diagnostics.cuda_residual_cpu_fallback_fits,
      Rcpp::Named("cuda_residual_unique_designs") =
        diagnostics.cuda_residual_unique_designs,
      Rcpp::Named("cuda_residual_duplicate_design_fits") =
        diagnostics.cuda_residual_duplicate_design_fits,
      Rcpp::Named("cuda_residual_max_fits_per_design") =
        diagnostics.cuda_residual_max_fits_per_design,
      Rcpp::Named("max_level_tasks") = diagnostics.max_level_tasks,
      Rcpp::Named("max_level_unique_residuals") =
        diagnostics.max_level_unique_residuals,
      Rcpp::Named("dcov_batch_size_requested") =
        diagnostics.dcov_batch_size_requested,
      Rcpp::Named("dcov_batch_size_used") = diagnostics.dcov_batch_size_used,
      Rcpp::Named("residual_batch_size_requested") =
        diagnostics.residual_batch_size_requested,
      Rcpp::Named("residual_batch_size_used") =
        diagnostics.residual_batch_size_used,
      Rcpp::Named("plan_elapsed_sec") = diagnostics.plan_elapsed_sec,
      Rcpp::Named("residual_prefetch_elapsed_sec") =
        diagnostics.residual_prefetch_elapsed_sec,
      Rcpp::Named("residual_request_collect_sec") =
        diagnostics.residual_request_collect_sec,
      Rcpp::Named("residual_prefetch_missing_scan_sec") =
        diagnostics.residual_prefetch_missing_scan_sec,
      Rcpp::Named("residual_prefetch_batch_input_sec") =
        diagnostics.residual_prefetch_batch_input_sec,
      Rcpp::Named("residual_batch_call_wall_sec") =
        diagnostics.residual_batch_call_wall_sec,
      Rcpp::Named("residual_diagnostic_merge_sec") =
        diagnostics.residual_diagnostic_merge_sec,
      Rcpp::Named("residual_prefetch_unaccounted_sec") =
        diagnostics.residual_prefetch_unaccounted_sec,
      Rcpp::Named("residual_grouping_sec") =
        diagnostics.residual_grouping_sec,
      Rcpp::Named("residual_grouping_condition_key_sec") =
        diagnostics.residual_grouping_condition_key_sec,
      Rcpp::Named("residual_grouping_group_key_sec") =
        diagnostics.residual_grouping_group_key_sec,
      Rcpp::Named("residual_grouping_design_build_sec") =
        diagnostics.residual_grouping_design_build_sec,
      Rcpp::Named("residual_grouping_map_insert_sec") =
        diagnostics.residual_grouping_map_insert_sec,
      Rcpp::Named("residual_grouping_unaccounted_sec") =
        diagnostics.residual_grouping_unaccounted_sec,
      Rcpp::Named("residual_grouping_group_count") =
        diagnostics.residual_grouping_group_count,
      Rcpp::Named("residual_grouping_design_count") =
        diagnostics.residual_grouping_design_count,
      Rcpp::Named("residual_grouping_condition_key_sort_count") =
        diagnostics.residual_grouping_condition_key_sort_count,
      Rcpp::Named("residual_grouping_string_key_count") =
        diagnostics.residual_grouping_string_key_count,
      Rcpp::Named("residual_host_pack_sec") =
        diagnostics.residual_host_pack_sec,
      Rcpp::Named("residual_alloc_sec") = diagnostics.residual_alloc_sec,
      Rcpp::Named("residual_h2d_sec") = diagnostics.residual_h2d_sec,
      Rcpp::Named("residual_h2d_design_sec") =
        diagnostics.residual_h2d_design_sec,
      Rcpp::Named("residual_h2d_penalty_sec") =
        diagnostics.residual_h2d_penalty_sec,
      Rcpp::Named("residual_h2d_y_sec") =
        diagnostics.residual_h2d_y_sec,
      Rcpp::Named("residual_h2d_index_sec") =
        diagnostics.residual_h2d_index_sec,
      Rcpp::Named("residual_h2d_lambda_sec") =
        diagnostics.residual_h2d_lambda_sec,
      Rcpp::Named("residual_h2d_active_sec") =
        diagnostics.residual_h2d_active_sec,
      Rcpp::Named("residual_h2d_copy_count") =
        diagnostics.residual_h2d_copy_count,
      Rcpp::Named("residual_h2d_bytes") = diagnostics.residual_h2d_bytes,
      Rcpp::Named("residual_h2d_design_bytes") =
        diagnostics.residual_h2d_design_bytes,
      Rcpp::Named("residual_h2d_y_bytes") =
        diagnostics.residual_h2d_y_bytes,
      Rcpp::Named("residual_h2d_metadata_bytes") =
        diagnostics.residual_h2d_metadata_bytes,
      Rcpp::Named("residual_h2d_metadata_coalesced_count") =
        diagnostics.residual_h2d_metadata_coalesced_count,
      Rcpp::Named("residual_h2d_metadata_coalesced_bytes") =
        diagnostics.residual_h2d_metadata_coalesced_bytes,
      Rcpp::Named("residual_h2d_selected_metadata_copy_count") =
        diagnostics.residual_h2d_selected_metadata_copy_count,
      Rcpp::Named("residual_xtx_xty_sec") =
        diagnostics.residual_xtx_xty_sec,
      Rcpp::Named("residual_pointer_setup_sec") =
        diagnostics.residual_pointer_setup_sec,
      Rcpp::Named("residual_active_copy_sec") =
        diagnostics.residual_active_copy_sec,
      Rcpp::Named("residual_build_system_sec") =
        diagnostics.residual_build_system_sec,
      Rcpp::Named("residual_factor_solve_sec") =
        diagnostics.residual_factor_solve_sec,
      Rcpp::Named("residual_factor_cholesky_sec") =
        diagnostics.residual_factor_cholesky_sec,
      Rcpp::Named("residual_factor_rhs_solve_sec") =
        diagnostics.residual_factor_rhs_solve_sec,
      Rcpp::Named("residual_factor_inverse_solve_sec") =
        diagnostics.residual_factor_inverse_solve_sec,
      Rcpp::Named("residual_summary_sec") =
        diagnostics.residual_summary_sec,
      Rcpp::Named("residual_d2h_sec") = diagnostics.residual_d2h_sec,
      Rcpp::Named("residual_d2h_residuals_sec") =
        diagnostics.residual_d2h_residuals_sec,
      Rcpp::Named("residual_d2h_metadata_sec") =
        diagnostics.residual_d2h_metadata_sec,
      Rcpp::Named("residual_d2h_info_sec") =
        diagnostics.residual_d2h_info_sec,
      Rcpp::Named("residual_d2h_copy_count") =
        diagnostics.residual_d2h_copy_count,
      Rcpp::Named("residual_d2h_bytes") = diagnostics.residual_d2h_bytes,
      Rcpp::Named("residual_d2h_residual_bytes") =
        diagnostics.residual_d2h_residual_bytes,
      Rcpp::Named("residual_d2h_metadata_bytes") =
        diagnostics.residual_d2h_metadata_bytes,
      Rcpp::Named("residual_d2h_metadata_coalesced_count") =
        diagnostics.residual_d2h_metadata_coalesced_count,
      Rcpp::Named("residual_d2h_metadata_coalesced_bytes") =
        diagnostics.residual_d2h_metadata_coalesced_bytes,
      Rcpp::Named("residual_host_select_sec") =
        diagnostics.residual_host_select_sec,
      Rcpp::Named("residual_free_sec") = diagnostics.residual_free_sec,
      Rcpp::Named("residual_true_batch_total_sec") =
        diagnostics.residual_true_batch_total_sec,
      Rcpp::Named("residual_factorization_count") =
        diagnostics.residual_factorization_count,
      Rcpp::Named("residual_rhs_solve_count") =
        diagnostics.residual_rhs_solve_count,
      Rcpp::Named("residual_inverse_solve_count") =
        diagnostics.residual_inverse_solve_count,
      Rcpp::Named("residual_rhs_solve_api_calls") =
        diagnostics.residual_rhs_solve_api_calls,
      Rcpp::Named("residual_rhs_target_solves") =
        diagnostics.residual_rhs_target_solves,
      Rcpp::Named("residual_rhs_custom_solve_count") =
        diagnostics.residual_rhs_custom_solve_count,
      Rcpp::Named("residual_rhs_cublas_solve_count") =
        diagnostics.residual_rhs_cublas_solve_count,
      Rcpp::Named("residual_rhs_solve_fallback_count") =
        diagnostics.residual_rhs_solve_fallback_count,
      Rcpp::Named("residual_rhs_custom_solve_sec") =
        diagnostics.residual_rhs_custom_solve_sec,
      Rcpp::Named("residual_rhs_cublas_solve_sec") =
        diagnostics.residual_rhs_cublas_solve_sec,
      Rcpp::Named("residual_candidate_rhs_fused_solve_count") =
        diagnostics.residual_candidate_rhs_fused_solve_count,
      Rcpp::Named("residual_candidate_rhs_materialized_solve_count") =
        diagnostics.residual_candidate_rhs_materialized_solve_count,
      Rcpp::Named("residual_selected_rhs_materialized_solve_count") =
        diagnostics.residual_selected_rhs_materialized_solve_count,
      Rcpp::Named("residual_candidate_beta_values_avoided") =
        diagnostics.residual_candidate_beta_values_avoided,
      Rcpp::Named("residual_summary_candidate_launch_count") =
        diagnostics.residual_summary_candidate_launch_count,
      Rcpp::Named("residual_summary_group_batched_launch_count") =
        diagnostics.residual_summary_group_batched_launch_count,
      Rcpp::Named("residual_summary_group_batched_candidate_count") =
        diagnostics.residual_summary_group_batched_candidate_count,
      Rcpp::Named("residual_winning_factor_reuse_count") =
        diagnostics.residual_winning_factor_reuse_count,
      Rcpp::Named("residual_factor_cache_hits") =
        diagnostics.residual_factor_cache_hits,
      Rcpp::Named("residual_factor_cache_misses") =
        diagnostics.residual_factor_cache_misses,
      Rcpp::Named("residual_factor_cache_entries") =
        diagnostics.residual_factor_cache_entries,
      Rcpp::Named("residual_factor_cache_bytes") =
        diagnostics.residual_factor_cache_bytes,
      Rcpp::Named("residual_lambda_candidates") =
        diagnostics.residual_lambda_candidates,
      Rcpp::Named("residual_workspace_reuse_count") =
        diagnostics.residual_workspace_reuse_count,
      Rcpp::Named("residual_workspace_grow_count") =
        diagnostics.residual_workspace_grow_count,
      Rcpp::Named("residual_workspace_slab_grow_count") =
        diagnostics.residual_workspace_slab_grow_count,
      Rcpp::Named("residual_workspace_slab_reuse_count") =
        diagnostics.residual_workspace_slab_reuse_count,
      Rcpp::Named("residual_workspace_slab_bytes") =
        diagnostics.residual_workspace_slab_bytes,
      Rcpp::Named("residual_workspace_legacy_alloc_count") =
        diagnostics.residual_workspace_legacy_alloc_count,
      Rcpp::Named("residual_solver_handle_create_count") =
        diagnostics.residual_solver_handle_create_count,
      Rcpp::Named("residual_per_request_design_x_values") =
        diagnostics.residual_per_request_design_x_values,
      Rcpp::Named("residual_duplicate_design_x_values_avoided") =
        diagnostics.residual_duplicate_design_x_values_avoided,
      Rcpp::Named("residual_cache_insert_sec") =
        diagnostics.residual_cache_insert_sec,
      Rcpp::Named("residual_cache_move_insert_count") =
        diagnostics.residual_cache_move_insert_count,
      Rcpp::Named("residual_cache_copy_insert_count") =
        diagnostics.residual_cache_copy_insert_count,
      Rcpp::Named("residual_algebraic_rss_count") =
        diagnostics.residual_algebraic_rss_count,
      Rcpp::Named("residual_candidate_residual_materialize_count") =
        diagnostics.residual_candidate_residual_materialize_count,
      Rcpp::Named("residual_winning_residual_materialize_count") =
        diagnostics.residual_winning_residual_materialize_count,
      Rcpp::Named("residual_algebraic_rss_clamp_count") =
        diagnostics.residual_algebraic_rss_clamp_count,
      Rcpp::Named("residual_only_batch_count") =
        diagnostics.residual_only_batch_count,
      Rcpp::Named("residual_full_fit_batch_count") =
        diagnostics.residual_full_fit_batch_count,
      Rcpp::Named("residual_only_fit_count") =
        diagnostics.residual_only_fit_count,
      Rcpp::Named("residual_full_fit_materialize_count") =
        diagnostics.residual_full_fit_materialize_count,
      Rcpp::Named("residual_fitted_values_avoided") =
        diagnostics.residual_fitted_values_avoided,
      Rcpp::Named("residual_result_materialize_sec") =
        diagnostics.residual_result_materialize_sec,
      Rcpp::Named("residual_fitted_materialize_sec") =
        diagnostics.residual_fitted_materialize_sec,
      Rcpp::Named("residual_batch_top_level_wall_sec") =
        diagnostics.residual_batch_top_level_wall_sec,
      Rcpp::Named("residual_batch_top_level_unaccounted_sec") =
        diagnostics.residual_batch_top_level_unaccounted_sec,
      Rcpp::Named("ci_eval_elapsed_sec") = diagnostics.ci_eval_elapsed_sec,
      Rcpp::Named("ci_host_pack_sec") = diagnostics.ci_host_pack_sec,
      Rcpp::Named("ci_dcov_call_wall_sec") =
        diagnostics.ci_dcov_call_wall_sec,
      Rcpp::Named("ci_pvalue_copy_sec") = diagnostics.ci_pvalue_copy_sec,
      Rcpp::Named("ci_diagnostic_append_sec") =
        diagnostics.ci_diagnostic_append_sec,
      Rcpp::Named("ci_eval_unaccounted_sec") =
        diagnostics.ci_eval_unaccounted_sec,
      Rcpp::Named("replay_elapsed_sec") = diagnostics.replay_elapsed_sec,
      Rcpp::Named("total_elapsed_sec") = diagnostics.total_elapsed_sec,
      Rcpp::Named("dcov_alloc_sec") = diagnostics.dcov_alloc_sec,
      Rcpp::Named("dcov_h2d_sec") = diagnostics.dcov_h2d_sec,
      Rcpp::Named("dcov_memset_sec") = diagnostics.dcov_memset_sec,
      Rcpp::Named("dcov_rowsum_sec") = diagnostics.dcov_rowsum_sec,
      Rcpp::Named("dcov_totals_d2h_sec") = diagnostics.dcov_totals_d2h_sec,
      Rcpp::Named("dcov_reduce_sec") = diagnostics.dcov_reduce_sec,
      Rcpp::Named("dcov_scalars_d2h_sec") = diagnostics.dcov_scalars_d2h_sec,
      Rcpp::Named("dcov_host_scalar_sec") = diagnostics.dcov_host_scalar_sec,
      Rcpp::Named("dcov_result_materialize_sec") =
        diagnostics.dcov_result_materialize_sec,
      Rcpp::Named("dcov_free_sec") = diagnostics.dcov_free_sec,
      Rcpp::Named("dcov_total_sec") = diagnostics.dcov_total_sec,
      Rcpp::Named("dcov_top_level_wall_sec") =
        diagnostics.dcov_top_level_wall_sec,
      Rcpp::Named("dcov_grid_limit_query_sec") =
        diagnostics.dcov_grid_limit_query_sec,
      Rcpp::Named("dcov_chunk_dispatch_sec") =
        diagnostics.dcov_chunk_dispatch_sec,
      Rcpp::Named("dcov_top_level_unaccounted_sec") =
        diagnostics.dcov_top_level_unaccounted_sec,
      Rcpp::Named("dcov_chunks") = diagnostics.dcov_chunks,
      Rcpp::Named("dcov_max_chunk_batch") = diagnostics.dcov_max_chunk_batch,
      Rcpp::Named("dcov_workspace_reuse_count") =
        diagnostics.dcov_workspace_reuse_count,
      Rcpp::Named("dcov_workspace_grow_count") =
        diagnostics.dcov_workspace_grow_count,
      Rcpp::Named("dcov_raw_aggregate_fused_count") =
        diagnostics.dcov_raw_aggregate_fused_count,
      Rcpp::Named("dcov_row_product_reduce_count") =
        diagnostics.dcov_row_product_reduce_count,
      Rcpp::Named("dcov_pvalue_only_count") =
        diagnostics.dcov_pvalue_only_count,
      Rcpp::Named("dcov_full_result_materialize_count") =
        diagnostics.dcov_full_result_materialize_count,
      Rcpp::Named("dcov_grid_limit_query_count") =
        diagnostics.dcov_grid_limit_query_count,
      Rcpp::Named("dcov_grid_limit_cache_hit_count") =
        diagnostics.dcov_grid_limit_cache_hit_count,
      Rcpp::Named("dcov_grid_limit_process_cache_hit_count") =
        diagnostics.dcov_grid_limit_process_cache_hit_count
    ),
    Rcpp::Named("levels") =
      scheduler_levels_to_data_frame(diagnostics.per_level),
    Rcpp::Named("batches") =
      scheduler_batches_to_data_frame(diagnostics.batches),
    Rcpp::Named("residuals") =
      scheduler_residuals_to_data_frame(diagnostics.residuals)
  );
}

Rcpp::List skeleton_result_to_list(const SkeletonResult& result, int p) {
  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(result.adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(result.sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(result.pmax, p),
    Rcpp::Named("n.edgetests") = Rcpp::IntegerVector(result.n_edge_tests.begin(),
                                                     result.n_edge_tests.end()),
    Rcpp::Named("per.level.log") = level_log_to_list(result.per_level_log),
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_device") = result.residual_device,
    Rcpp::Named("residual_device_requested") = result.residual_device_requested,
    Rcpp::Named("residual_device_reason") = result.residual_device_reason,
    Rcpp::Named("scheduler") = result.scheduler,
    Rcpp::Named("scheduler_requested") = result.scheduler_requested,
    Rcpp::Named("scheduler_diagnostics") =
      scheduler_diagnostics_to_list(result.scheduler_diagnostics),
    Rcpp::Named("residual_cache") = residual_cache_stats_to_list(result),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") = result.ci_backend.empty() ? "cuda" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("ci_dcc_gamma_tests") = result.ci_dcc_gamma_tests,
      Rcpp::Named("ci_hsic_gamma_tests") = result.ci_hsic_gamma_tests,
      Rcpp::Named("ci_hsic_perm_tests") = result.ci_hsic_perm_tests,
      Rcpp::Named("ci_hsic_permutation_replicates") =
        result.ci_hsic_permutation_replicates,
      Rcpp::Named("ci_hsic_gamma_cuda_tests") =
        result.ci_hsic_gamma_cuda_tests,
      Rcpp::Named("ci_hsic_perm_cuda_tests") =
        result.ci_hsic_perm_cuda_tests,
      Rcpp::Named("ci_hsic_cuda_batches") = result.ci_hsic_cuda_batches,
      Rcpp::Named("ci_hsic_cuda_pairs") = result.ci_hsic_cuda_pairs,
      Rcpp::Named("ci_hsic_cuda_fallback_tests") =
        result.ci_hsic_cuda_fallback_tests,
      Rcpp::Named("ci_hsic_cuda_memory_bytes") =
        static_cast<double>(result.ci_hsic_cuda_memory_bytes),
      Rcpp::Named("ci_hsic_cuda_max_n") = result.ci_hsic_cuda_max_n,
      Rcpp::Named("ci_hsic_cuda_max_batch_pairs") =
        result.ci_hsic_cuda_max_batch_pairs
    )
  );
}

std::string replay_s_key(const std::vector<int>& conditioning_set) {
  std::ostringstream out;
  for (std::size_t i = 0; i < conditioning_set.size(); ++i) {
    if (i != 0) out << "|";
    out << conditioning_set[i] + 1;
  }
  return out.str();
}

Rcpp::IntegerMatrix pdag_to_matrix(const std::vector<int>& pdag, int p) {
  Rcpp::IntegerMatrix out(p, p);
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      out(row, col) = pdag[static_cast<std::size_t>(row) * p + col];
    }
  }
  return out;
}

Rcpp::List orientation_events_to_list(const std::vector<OrientationEvent>& events) {
  Rcpp::List out(events.size());
  for (int i = 0; i < static_cast<int>(events.size()); ++i) {
    const OrientationEvent& event = events[i];
    Rcpp::IntegerVector S(event.S.size());
    for (int j = 0; j < S.size(); ++j) S[j] = event.S[j] + 1;
    out[i] = Rcpp::List::create(
      Rcpp::Named("phase") = event.phase,
      Rcpp::Named("rule") = event.rule,
      Rcpp::Named("x") = event.x < 0 ? NA_INTEGER : event.x + 1,
      Rcpp::Named("y") = event.y < 0 ? NA_INTEGER : event.y + 1,
      Rcpp::Named("z") = event.z < 0 ? NA_INTEGER : event.z + 1,
      Rcpp::Named("S") = S,
      Rcpp::Named("p.value") = event.p_value,
      Rcpp::Named("accepted") = event.accepted,
      Rcpp::Named("message") = event.message
    );
  }
  return out;
}

Rcpp::List orientation_result_to_list(const OrientationResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("pdag") = pdag_to_matrix(result.pdag, result.p),
    Rcpp::Named("events") = orientation_events_to_list(result.events),
    Rcpp::Named("counts") = Rcpp::List::create(
      Rcpp::Named("collider") = result.collider_orientations,
      Rcpp::Named("rule1") = result.rule1_orientations,
      Rcpp::Named("rule2") = result.rule2_orientations,
      Rcpp::Named("rule3") = result.rule3_orientations,
      Rcpp::Named("generalized") = result.generalized_orientations,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls
    ),
    Rcpp::Named("residual_backend") = result.residual_backend,
    Rcpp::Named("residual_backend_params") = result.residual_backend_params,
    Rcpp::Named("residual_device") = result.residual_device,
    Rcpp::Named("residual_device_requested") = result.residual_device_requested,
    Rcpp::Named("residual_device_reason") = result.residual_device_reason,
    Rcpp::Named("orientation_batch_size_requested") =
      result.orientation_batch_size_requested,
    Rcpp::Named("orientation_batch_size_used") =
      result.orientation_batch_size_used,
    Rcpp::Named("residual_cache") = Rcpp::List::create(
      Rcpp::Named("requests") = result.residual_cache_requests,
      Rcpp::Named("hits") = result.residual_cache_hits,
      Rcpp::Named("computations") = result.residual_cache_computations
    ),
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("orientation_residual_device") = result.residual_device,
      Rcpp::Named("orientation_residual_device_requested") =
        result.residual_device_requested,
      Rcpp::Named("orientation_residual_device_reason") =
        result.residual_device_reason,
      Rcpp::Named("orientation_batch_size_requested") =
        result.orientation_batch_size_requested,
      Rcpp::Named("orientation_batch_size_used") =
        result.orientation_batch_size_used,
      Rcpp::Named("regrvonps_calls") = result.regrvonps_calls,
      Rcpp::Named("regrvonps_cuda_calls") = result.regrvonps_cuda_calls,
      Rcpp::Named("regrvonps_cpu_calls") = result.regrvonps_cpu_calls,
      Rcpp::Named("orientation_dcov_batches") =
        result.orientation_dcov_batches,
      Rcpp::Named("orientation_dcov_pairs") = result.orientation_dcov_pairs,
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests,
      Rcpp::Named("orientation_residual_fits") =
        result.orientation_residual_fits,
      Rcpp::Named("orientation_cuda_residual_fits") =
        result.orientation_cuda_residual_fits,
      Rcpp::Named("orientation_cpu_fallback_fits") =
        result.orientation_cpu_fallback_fits,
      Rcpp::Named("orientation_cache_requests") =
        result.residual_cache_requests,
      Rcpp::Named("orientation_cache_hits") = result.residual_cache_hits,
      Rcpp::Named("orientation_cache_computations") =
        result.residual_cache_computations
    ),
    Rcpp::Named("ci_method") = result.ci_method.empty() ? "dcc.gamma" : result.ci_method,
    Rcpp::Named("ci_backend") =
      result.ci_backend.empty() ? "native-cpu" : result.ci_backend,
    Rcpp::Named("ci_backend_reason") = result.ci_backend_reason,
    Rcpp::Named("ci_diagnostics") = Rcpp::List::create(
      Rcpp::Named("regrvonps_dcc_gamma_tests") =
        result.regrvonps_dcc_gamma_tests,
      Rcpp::Named("regrvonps_hsic_gamma_tests") =
        result.regrvonps_hsic_gamma_tests,
      Rcpp::Named("regrvonps_hsic_perm_tests") =
        result.regrvonps_hsic_perm_tests,
      Rcpp::Named("regrvonps_hsic_permutation_replicates") =
        result.regrvonps_hsic_permutation_replicates,
      Rcpp::Named("regrvonps_hsic_gamma_cuda_tests") =
        result.regrvonps_hsic_gamma_cuda_tests,
      Rcpp::Named("regrvonps_hsic_perm_cuda_tests") =
        result.regrvonps_hsic_perm_cuda_tests,
      Rcpp::Named("regrvonps_hsic_cuda_batches") =
        result.regrvonps_hsic_cuda_batches,
      Rcpp::Named("regrvonps_hsic_cuda_pairs") =
        result.regrvonps_hsic_cuda_pairs,
      Rcpp::Named("regrvonps_hsic_cuda_fallback_tests") =
        result.regrvonps_hsic_cuda_fallback_tests
    )
  );
}

Rcpp::List fastspline_cuda_fit_to_list(const FastSplineCudaFit& result) {
  const FastSplineFit& fit = result.fit;
  return Rcpp::List::create(
    Rcpp::Named("residuals") =
      Rcpp::NumericVector(fit.residuals.begin(), fit.residuals.end()),
    Rcpp::Named("fitted") =
      Rcpp::NumericVector(fit.fitted.begin(), fit.fitted.end()),
    Rcpp::Named("selected_lambda") = fit.selected_lambda,
    Rcpp::Named("gcv") = fit.gcv,
    Rcpp::Named("rss") = fit.rss,
    Rcpp::Named("edf") = fit.edf,
    Rcpp::Named("design_cols") = fit.design_cols,
    Rcpp::Named("ridge_attempts") = fit.ridge_attempts,
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = "fastSpline",
    Rcpp::Named("residual_device") =
      result.diagnostics.fallback_used ? "cuda-fallback-cpu" : "cuda",
    Rcpp::Named("fallback_used") = result.diagnostics.fallback_used,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("cuda_used") = result.diagnostics.cuda_used,
      Rcpp::Named("fallback_used") = result.diagnostics.fallback_used,
      Rcpp::Named("reason") = result.diagnostics.reason,
      Rcpp::Named("batch_group_id") = result.diagnostics.batch_group_id,
      Rcpp::Named("batch_position") = result.diagnostics.batch_position,
      Rcpp::Named("true_batched") = result.diagnostics.true_batched,
      Rcpp::Named("cholesky_backend") = result.diagnostics.cholesky_backend
    )
  );
}

Rcpp::DataFrame fastspline_batch_group_table_to_df(
  const FastSplineCudaBatchDiagnostics& diagnostics) {
  const int n = static_cast<int>(diagnostics.group_id.size());
  Rcpp::IntegerVector group_id(n), rows(n), design_cols(n), fit_count(n);
  Rcpp::IntegerVector single_fit_calls(n), cpu_fallback_fits(n);
  Rcpp::IntegerVector unique_designs(n), duplicate_design_fits(n);
  Rcpp::IntegerVector max_fits_per_design(n);
  Rcpp::LogicalVector true_batched(n);
  Rcpp::CharacterVector cholesky_backend(n), status(n), reason(n);
  for (int i = 0; i < n; ++i) {
    group_id[i] = diagnostics.group_id[i];
    rows[i] = diagnostics.group_n[i];
    design_cols[i] = diagnostics.group_design_cols[i];
    fit_count[i] = diagnostics.group_fit_count[i];
    true_batched[i] = diagnostics.group_true_batched[i] != 0;
    single_fit_calls[i] = diagnostics.group_single_fit_calls[i];
    cpu_fallback_fits[i] = diagnostics.group_cpu_fallback_fits[i];
    unique_designs[i] = diagnostics.group_unique_designs[i];
    duplicate_design_fits[i] = diagnostics.group_duplicate_design_fits[i];
    max_fits_per_design[i] = diagnostics.group_max_fits_per_design[i];
    cholesky_backend[i] = diagnostics.group_cholesky_backend[i];
    status[i] = diagnostics.group_status[i];
    reason[i] = diagnostics.group_reason[i];
  }
  return Rcpp::DataFrame::create(
    Rcpp::Named("group_id") = group_id,
    Rcpp::Named("n") = rows,
    Rcpp::Named("design_cols") = design_cols,
    Rcpp::Named("fit_count") = fit_count,
    Rcpp::Named("true_batched") = true_batched,
    Rcpp::Named("single_fit_calls") = single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = cpu_fallback_fits,
    Rcpp::Named("unique_designs") = unique_designs,
    Rcpp::Named("duplicate_design_fits") = duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = max_fits_per_design,
    Rcpp::Named("cholesky_backend") = cholesky_backend,
    Rcpp::Named("status") = status,
    Rcpp::Named("reason") = reason,
    Rcpp::Named("stringsAsFactors") = false
  );
}

Rcpp::List fastspline_batch_diagnostics_to_list(
  const FastSplineCudaBatchDiagnostics& diagnostics) {
  return Rcpp::List::create(
    Rcpp::Named("requested_fits") = diagnostics.requested_fits,
    Rcpp::Named("groups") = diagnostics.groups,
    Rcpp::Named("true_batched_groups") = diagnostics.true_batched_groups,
    Rcpp::Named("true_batched_fits") = diagnostics.true_batched_fits,
    Rcpp::Named("single_fit_calls") = diagnostics.single_fit_calls,
    Rcpp::Named("cpu_fallback_fits") = diagnostics.cpu_fallback_fits,
    Rcpp::Named("unique_designs") = diagnostics.unique_designs,
    Rcpp::Named("duplicate_design_fits") = diagnostics.duplicate_design_fits,
    Rcpp::Named("max_fits_per_design") = diagnostics.max_fits_per_design,
    Rcpp::Named("max_group_size") = diagnostics.max_group_size,
    Rcpp::Named("min_group_size") = diagnostics.min_group_size,
    Rcpp::Named("cholesky_backend") = diagnostics.cholesky_backend,
    Rcpp::Named("batch_mode") = diagnostics.batch_mode,
    Rcpp::Named("grouping_sec") = diagnostics.grouping_sec,
    Rcpp::Named("grouping_condition_key_sec") =
      diagnostics.grouping_condition_key_sec,
    Rcpp::Named("grouping_group_key_sec") =
      diagnostics.grouping_group_key_sec,
    Rcpp::Named("grouping_design_build_sec") =
      diagnostics.grouping_design_build_sec,
    Rcpp::Named("grouping_map_insert_sec") =
      diagnostics.grouping_map_insert_sec,
    Rcpp::Named("grouping_unaccounted_sec") =
      diagnostics.grouping_unaccounted_sec,
    Rcpp::Named("grouping_group_count") =
      diagnostics.grouping_group_count,
    Rcpp::Named("grouping_design_count") =
      diagnostics.grouping_design_count,
    Rcpp::Named("grouping_condition_key_sort_count") =
      diagnostics.grouping_condition_key_sort_count,
    Rcpp::Named("grouping_string_key_count") =
      diagnostics.grouping_string_key_count,
    Rcpp::Named("host_pack_sec") = diagnostics.host_pack_sec,
    Rcpp::Named("alloc_sec") = diagnostics.alloc_sec,
    Rcpp::Named("h2d_sec") = diagnostics.h2d_sec,
    Rcpp::Named("h2d_design_sec") = diagnostics.h2d_design_sec,
    Rcpp::Named("h2d_penalty_sec") = diagnostics.h2d_penalty_sec,
    Rcpp::Named("h2d_y_sec") = diagnostics.h2d_y_sec,
    Rcpp::Named("h2d_index_sec") = diagnostics.h2d_index_sec,
    Rcpp::Named("h2d_lambda_sec") = diagnostics.h2d_lambda_sec,
    Rcpp::Named("h2d_active_sec") = diagnostics.h2d_active_sec,
    Rcpp::Named("h2d_copy_count") = diagnostics.h2d_copy_count,
    Rcpp::Named("h2d_bytes") = diagnostics.h2d_bytes,
    Rcpp::Named("h2d_design_bytes") = diagnostics.h2d_design_bytes,
    Rcpp::Named("h2d_y_bytes") = diagnostics.h2d_y_bytes,
    Rcpp::Named("h2d_metadata_bytes") = diagnostics.h2d_metadata_bytes,
    Rcpp::Named("h2d_metadata_coalesced_count") =
      diagnostics.h2d_metadata_coalesced_count,
    Rcpp::Named("h2d_metadata_coalesced_bytes") =
      diagnostics.h2d_metadata_coalesced_bytes,
    Rcpp::Named("h2d_selected_metadata_copy_count") =
      diagnostics.h2d_selected_metadata_copy_count,
    Rcpp::Named("xtx_xty_sec") = diagnostics.xtx_xty_sec,
    Rcpp::Named("pointer_setup_sec") = diagnostics.pointer_setup_sec,
    Rcpp::Named("active_copy_sec") = diagnostics.active_copy_sec,
    Rcpp::Named("build_system_sec") = diagnostics.build_system_sec,
    Rcpp::Named("factor_solve_sec") = diagnostics.factor_solve_sec,
    Rcpp::Named("factor_cholesky_sec") = diagnostics.factor_cholesky_sec,
    Rcpp::Named("factor_rhs_solve_sec") = diagnostics.factor_rhs_solve_sec,
    Rcpp::Named("factor_inverse_solve_sec") =
      diagnostics.factor_inverse_solve_sec,
    Rcpp::Named("residual_summary_sec") = diagnostics.residual_summary_sec,
    Rcpp::Named("d2h_sec") = diagnostics.d2h_sec,
    Rcpp::Named("d2h_residuals_sec") = diagnostics.d2h_residuals_sec,
    Rcpp::Named("d2h_metadata_sec") = diagnostics.d2h_metadata_sec,
    Rcpp::Named("d2h_info_sec") = diagnostics.d2h_info_sec,
    Rcpp::Named("d2h_copy_count") = diagnostics.d2h_copy_count,
    Rcpp::Named("d2h_bytes") = diagnostics.d2h_bytes,
    Rcpp::Named("d2h_residual_bytes") = diagnostics.d2h_residual_bytes,
    Rcpp::Named("d2h_metadata_bytes") = diagnostics.d2h_metadata_bytes,
    Rcpp::Named("d2h_metadata_coalesced_count") =
      diagnostics.d2h_metadata_coalesced_count,
    Rcpp::Named("d2h_metadata_coalesced_bytes") =
      diagnostics.d2h_metadata_coalesced_bytes,
    Rcpp::Named("host_select_sec") = diagnostics.host_select_sec,
    Rcpp::Named("free_sec") = diagnostics.free_sec,
    Rcpp::Named("true_batch_total_sec") = diagnostics.true_batch_total_sec,
    Rcpp::Named("factorization_count") = diagnostics.factorization_count,
    Rcpp::Named("rhs_solve_count") = diagnostics.rhs_solve_count,
    Rcpp::Named("inverse_solve_count") = diagnostics.inverse_solve_count,
    Rcpp::Named("rhs_solve_api_calls") = diagnostics.rhs_solve_api_calls,
    Rcpp::Named("rhs_target_solves") = diagnostics.rhs_target_solves,
    Rcpp::Named("rhs_custom_solve_count") =
      diagnostics.rhs_custom_solve_count,
    Rcpp::Named("rhs_cublas_solve_count") =
      diagnostics.rhs_cublas_solve_count,
    Rcpp::Named("rhs_solve_fallback_count") =
      diagnostics.rhs_solve_fallback_count,
    Rcpp::Named("rhs_custom_solve_sec") =
      diagnostics.rhs_custom_solve_sec,
    Rcpp::Named("rhs_cublas_solve_sec") =
      diagnostics.rhs_cublas_solve_sec,
    Rcpp::Named("candidate_rhs_fused_solve_count") =
      diagnostics.candidate_rhs_fused_solve_count,
    Rcpp::Named("candidate_rhs_materialized_solve_count") =
      diagnostics.candidate_rhs_materialized_solve_count,
    Rcpp::Named("selected_rhs_materialized_solve_count") =
      diagnostics.selected_rhs_materialized_solve_count,
    Rcpp::Named("candidate_beta_values_avoided") =
      diagnostics.candidate_beta_values_avoided,
    Rcpp::Named("summary_candidate_launch_count") =
      diagnostics.summary_candidate_launch_count,
    Rcpp::Named("summary_group_batched_launch_count") =
      diagnostics.summary_group_batched_launch_count,
    Rcpp::Named("summary_group_batched_candidate_count") =
      diagnostics.summary_group_batched_candidate_count,
    Rcpp::Named("winning_factor_reuse_count") =
      diagnostics.winning_factor_reuse_count,
    Rcpp::Named("factor_cache_hits") = diagnostics.factor_cache_hits,
    Rcpp::Named("factor_cache_misses") = diagnostics.factor_cache_misses,
    Rcpp::Named("factor_cache_entries") = diagnostics.factor_cache_entries,
    Rcpp::Named("factor_cache_bytes") = diagnostics.factor_cache_bytes,
    Rcpp::Named("lambda_candidates") = diagnostics.lambda_candidates,
    Rcpp::Named("workspace_reuse_count") = diagnostics.workspace_reuse_count,
    Rcpp::Named("workspace_grow_count") = diagnostics.workspace_grow_count,
    Rcpp::Named("workspace_slab_grow_count") =
      diagnostics.workspace_slab_grow_count,
    Rcpp::Named("workspace_slab_reuse_count") =
      diagnostics.workspace_slab_reuse_count,
    Rcpp::Named("workspace_slab_bytes") = diagnostics.workspace_slab_bytes,
    Rcpp::Named("workspace_legacy_alloc_count") =
      diagnostics.workspace_legacy_alloc_count,
    Rcpp::Named("solver_handle_create_count") =
      diagnostics.solver_handle_create_count,
    Rcpp::Named("per_request_design_x_values") =
      diagnostics.per_request_design_x_values,
    Rcpp::Named("duplicate_design_x_values_avoided") =
      diagnostics.duplicate_design_x_values_avoided,
    Rcpp::Named("algebraic_rss_count") = diagnostics.algebraic_rss_count,
    Rcpp::Named("candidate_residual_materialize_count") =
      diagnostics.candidate_residual_materialize_count,
    Rcpp::Named("winning_residual_materialize_count") =
      diagnostics.winning_residual_materialize_count,
    Rcpp::Named("algebraic_rss_clamp_count") =
      diagnostics.algebraic_rss_clamp_count,
    Rcpp::Named("residual_only_batch_count") =
      diagnostics.residual_only_batch_count,
    Rcpp::Named("residual_full_fit_batch_count") =
      diagnostics.residual_full_fit_batch_count,
    Rcpp::Named("residual_only_fit_count") =
      diagnostics.residual_only_fit_count,
    Rcpp::Named("residual_full_fit_materialize_count") =
      diagnostics.residual_full_fit_materialize_count,
    Rcpp::Named("residual_fitted_values_avoided") =
      diagnostics.residual_fitted_values_avoided,
    Rcpp::Named("residual_result_materialize_sec") =
      diagnostics.residual_result_materialize_sec,
    Rcpp::Named("residual_fitted_materialize_sec") =
      diagnostics.residual_fitted_materialize_sec,
    Rcpp::Named("residual_batch_top_level_wall_sec") =
      diagnostics.residual_batch_top_level_wall_sec,
    Rcpp::Named("residual_batch_top_level_unaccounted_sec") =
      diagnostics.residual_batch_top_level_unaccounted_sec,
    Rcpp::Named("group_table") =
      fastspline_batch_group_table_to_df(diagnostics)
  );
}

double get_named_double(Rcpp::List values, const char* name, double fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<double>(values[name]);
}

int get_named_int(Rcpp::List values, const char* name, int fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<int>(values[name]);
}

std::string get_named_string(Rcpp::List values, const char* name,
                             const std::string& fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<std::string>(values[name]);
}

bool get_named_bool(Rcpp::List values, const char* name, bool fallback) {
  if (!values.containsElementNamed(name)) return fallback;
  return Rcpp::as<bool>(values[name]);
}

FastSplineParams parse_fastspline_params(Rcpp::List values) {
  FastSplineParams params = default_fastspline_params();
  params.degree = get_named_int(values, "degree", params.degree);
  params.knots = get_named_int(values, "knots", params.knots);
  params.lambda_min = get_named_double(values, "lambda_min", params.lambda_min);
  params.lambda_max = get_named_double(values, "lambda_max", params.lambda_max);
  params.lambda_count = get_named_int(values, "lambda_count", params.lambda_count);
  params.ridge = get_named_double(values, "ridge", params.ridge);
  params.mode = get_named_string(values, "mode", params.mode);
  return params;
}

HsicOptions parse_hsic_options(Rcpp::List hsic_params,
                               Rcpp::List permutation_params) {
  HsicOptions options = default_hsic_options();
  options.sig = get_named_double(hsic_params, "sig", options.sig);
  options.cuda_max_n = get_named_int(hsic_params, "cuda_max_n",
                                     options.cuda_max_n);
  options.cuda_max_batch_pairs =
    get_named_int(hsic_params, "cuda_max_batch_pairs",
                  options.cuda_max_batch_pairs);
  options.cuda_memory_fallback =
    get_named_bool(hsic_params, "cuda_memory_fallback",
                   options.cuda_memory_fallback);
  options.replicates = get_named_int(permutation_params, "replicates",
                                     options.replicates);
  options.include_observed = get_named_bool(permutation_params,
                                            "include_observed",
                                            options.include_observed);
  if (permutation_params.containsElementNamed("seed")) {
    SEXP seed = permutation_params["seed"];
    if (!Rf_isNull(seed)) {
      options.has_seed = true;
      options.seed = static_cast<unsigned int>(Rcpp::as<int>(seed));
    }
  }
  return options;
}

void apply_ci_options(SkeletonOptions* options,
                      const std::string& ci_method,
                      Rcpp::List hsic_params,
                      Rcpp::List permutation_params,
                      bool ci_diagnostics) {
  options->ci_method = ci_method.empty() ? "dcc.gamma" : ci_method;
  options->hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options->ci_diagnostics_enabled = ci_diagnostics;
}

OrientationOptions make_orientation_options(double alpha,
                                            double index,
                                            bool legacy_index,
                                            bool residual_cache,
                                            const std::string& residual_backend,
                                            const std::string& orientation_device,
                                            int orientation_batch_size,
                                            bool orientation_diagnostics,
                                            bool cuda_residual_fallback,
                                            const FastSplineParams& fastspline_params,
                                            bool orient_collider,
                                            bool solve_confl,
                                            Rcpp::LogicalVector rules,
                                            const std::string& ci_method = "dcc.gamma",
                                            Rcpp::List hsic_params = Rcpp::List::create(),
                                            Rcpp::List permutation_params = Rcpp::List::create(),
                                            bool ci_diagnostics = true) {
  if (rules.size() != 3) {
    Rcpp::stop("rules must have length 3");
  }
  if (rules[0] == NA_LOGICAL || rules[1] == NA_LOGICAL ||
      rules[2] == NA_LOGICAL) {
    Rcpp::stop("rules must not contain NA");
  }
  make_residual_backend_config(residual_backend, fastspline_params);
  if (orientation_device != "auto" && orientation_device != "cpu" &&
      orientation_device != "cuda") {
    Rcpp::stop("Unknown orientation residual device: " + orientation_device);
  }
  OrientationOptions options = default_orientation_options();
  options.alpha = alpha;
  options.index = index;
  options.legacy_index = legacy_index;
  options.residual_cache_enabled = residual_cache;
  options.residual_backend_name = residual_backend;
  options.orientation_residual_device_requested = orientation_device;
  options.orientation_residual_device_reason = "";
  if (residual_backend == "linear") {
    options.orientation_residual_device = "cpu";
    if (orientation_device == "cuda") {
      options.orientation_residual_device_reason =
        "linear orientation residual CUDA device is not implemented";
    } else if (orientation_device == "auto") {
      options.orientation_residual_device_reason =
        "linear orientation residuals use CPU";
    }
  } else if (orientation_device == "cpu") {
    options.orientation_residual_device = "cpu";
  } else {
    options.orientation_residual_device = "cuda";
  }
  options.orientation_batch_size = orientation_batch_size;
  options.orientation_diagnostics_enabled = orientation_diagnostics;
  options.cuda_residual_fallback = cuda_residual_fallback;
  options.fastspline_params = fastspline_params;
  options.orient_collider = orient_collider;
  options.solve_confl = solve_confl;
  options.rule1 = rules[0] == TRUE;
  options.rule2 = rules[1] == TRUE;
  options.rule3 = rules[2] == TRUE;
  options.ci_method = ci_method;
  options.hsic_options = parse_hsic_options(hsic_params, permutation_params);
  options.ci_diagnostics_enabled = ci_diagnostics;
  return options;
}

}  // namespace

extern "C" SEXP C_fastkpc_cuda_available() {
  BEGIN_RCPP
  std::string error;
  return Rcpp::wrap(fastkpc_cuda_available(&error));
  END_RCPP
}

extern "C" SEXP C_fastkpc_cuda_device_info() {
  BEGIN_RCPP
  const CudaDeviceInfo info = fastkpc_cuda_device_info();
  return Rcpp::List::create(
    Rcpp::Named("device_id") = info.device_id,
    Rcpp::Named("name") = info.name,
    Rcpp::Named("compute_capability") =
      std::to_string(info.major) + "." + std::to_string(info.minor),
    Rcpp::Named("total_global_mem") = info.total_global_mem
  );
  END_RCPP
}

extern "C" SEXP C_fast_dcov_batch_cuda(SEXP xs, SEXP ys, SEXP indexs,
                                        SEXP legacy_indexs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys) || !Rf_isMatrix(xs) || !Rf_isMatrix(ys)) {
    Rcpp::stop("x and y must be numeric matrices");
  }
  Rcpp::NumericMatrix x(xs);
  Rcpp::NumericMatrix y(ys);
  if (x.nrow() != y.nrow() || x.ncol() != y.ncol()) {
    Rcpp::stop("x and y must have identical dimensions");
  }
  const int n = x.nrow();
  const int batch = x.ncol();
  if (n <= 5) Rcpp::stop("gamma approximation requires n > 5");
  if (!all_finite(x) || !all_finite(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  double index = Rf_asReal(indexs);
  if (index < 0.0 || index > 2.0) {
    Rf_warning("index must be in [0,2), using default index=1");
    index = 1.0;
  }
  DcovBatchOptions options;
  options.index = index;
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  DcovBatchResult result = dcov_batch_cuda(REAL(xs), REAL(ys), n, batch, options);

  Rcpp::NumericMatrix raw(batch, 5);
  for (int k = 0; k < batch; ++k) {
    for (int j = 0; j < 5; ++j) {
      raw(k, j) = result.raw_scalars[static_cast<std::size_t>(k) * 5 + j];
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("p.value") = result.p_values,
    Rcpp::Named("nV2") = result.nV2,
    Rcpp::Named("mean") = result.means,
    Rcpp::Named("variance") = result.variances,
    Rcpp::Named("raw") = raw,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("alloc_sec") = result.alloc_sec,
      Rcpp::Named("h2d_sec") = result.h2d_sec,
      Rcpp::Named("memset_sec") = result.memset_sec,
      Rcpp::Named("rowsum_sec") = result.rowsum_sec,
      Rcpp::Named("totals_d2h_sec") = result.totals_d2h_sec,
      Rcpp::Named("reduce_sec") = result.reduce_sec,
      Rcpp::Named("scalars_d2h_sec") = result.scalars_d2h_sec,
      Rcpp::Named("host_scalar_sec") = result.host_scalar_sec,
      Rcpp::Named("free_sec") = result.free_sec,
      Rcpp::Named("total_sec") = result.total_sec,
      Rcpp::Named("chunks") = result.chunks,
      Rcpp::Named("max_chunk_batch") = result.max_chunk_batch,
      Rcpp::Named("workspace_reuse_count") = result.workspace_reuse_count,
      Rcpp::Named("workspace_grow_count") = result.workspace_grow_count,
      Rcpp::Named("raw_aggregate_fused_count") =
        result.raw_aggregate_fused_count,
      Rcpp::Named("row_product_reduce_count") =
        result.row_product_reduce_count,
      Rcpp::Named("pvalue_only_count") = result.pvalue_only_count,
      Rcpp::Named("full_result_materialize_count") =
        result.full_result_materialize_count,
      Rcpp::Named("result_materialize_sec") =
        result.result_materialize_sec,
      Rcpp::Named("top_level_wall_sec") = result.top_level_wall_sec,
      Rcpp::Named("grid_limit_query_sec") = result.grid_limit_query_sec,
      Rcpp::Named("grid_limit_query_count") =
        result.grid_limit_query_count,
      Rcpp::Named("grid_limit_cache_hit_count") =
        result.grid_limit_cache_hit_count,
      Rcpp::Named("grid_limit_process_cache_hit_count") =
        result.grid_limit_process_cache_hit_count,
      Rcpp::Named("chunk_dispatch_sec") = result.chunk_dispatch_sec,
      Rcpp::Named("top_level_unaccounted_sec") =
        result.top_level_unaccounted_sec
    )
  );
  END_RCPP
}

extern "C" SEXP C_fast_hsic_gamma_cuda(SEXP xs, SEXP ys, SEXP sigs) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  if (x.size() < 4) Rcpp::stop("HSIC requires at least 4 observations");
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  HsicBatchOptions options = default_hsic_batch_options();
  options.sig = Rf_asReal(sigs);
  const HsicBatchResult result =
    hsic_gamma_batch_cuda(REAL(xs), REAL(ys), x.size(), 1, options);
  return hsic_batch_result_to_list(result, 0, options.sig);
  END_RCPP
}

extern "C" SEXP C_fast_hsic_perm_cuda(SEXP xs, SEXP ys, SEXP sigs,
                                       SEXP replicatess, SEXP seeds,
                                       SEXP include_observeds) {
  BEGIN_RCPP
  if (!Rf_isReal(xs) || !Rf_isReal(ys)) {
    Rcpp::stop("x and y must be numeric");
  }
  Rcpp::NumericVector x(xs);
  Rcpp::NumericVector y(ys);
  if (x.size() != y.size()) Rcpp::stop("Sample sizes must agree");
  if (x.size() < 4) Rcpp::stop("HSIC requires at least 4 observations");
  if (!all_finite_vector(x) || !all_finite_vector(y)) {
    Rcpp::stop("Data contains missing or infinite values");
  }
  HsicBatchOptions options = default_hsic_batch_options();
  options.sig = Rf_asReal(sigs);
  options.permutation_replicates = Rf_asInteger(replicatess);
  options.has_seed = true;
  options.seed = static_cast<unsigned int>(Rf_asInteger(seeds));
  options.include_observed = Rcpp::as<bool>(include_observeds);
  options.return_replicates = true;
  const HsicBatchResult result =
    hsic_permutation_batch_cuda(REAL(xs), REAL(ys), x.size(), 1, options);
  return hsic_batch_result_to_list(result, 0, options.sig);
  END_RCPP
}

extern "C" SEXP C_fastspline_residual_cuda(SEXP ys,
                                            SEXP Ss,
                                            SEXP fastspline_paramss,
                                            SEXP fallbacks) {
  BEGIN_RCPP
  if (!Rf_isReal(ys)) Rcpp::stop("y must be numeric");
  if (!Rf_isReal(Ss) || !Rf_isMatrix(Ss)) Rcpp::stop("S must be a numeric matrix");
  Rcpp::NumericVector y(ys);
  Rcpp::NumericMatrix S(Ss);
  if (y.size() != S.nrow()) {
    Rcpp::stop("y and S must have the same number of rows");
  }
  for (int i = 0; i < y.size(); ++i) {
    if (!std::isfinite(y[i])) Rcpp::stop("y contains missing or infinite values");
  }
  if (!all_finite(S)) Rcpp::stop("S contains missing or infinite values");

  Rcpp::NumericMatrix data(y.size(), S.ncol() + 1);
  for (int row = 0; row < y.size(); ++row) {
    data(row, 0) = y[row];
    for (int col = 0; col < S.ncol(); ++col) data(row, col + 1) = S(row, col);
  }
  std::vector<int> cond;
  for (int col = 0; col < S.ncol(); ++col) cond.push_back(col + 1);

  const FastSplineParams params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  const FastSplineCudaFit result =
    fit_fastspline_residuals_cuda(data, 0, cond, params,
                                  Rcpp::as<bool>(fallbacks));
  return fastspline_cuda_fit_to_list(result);
  END_RCPP
}

extern "C" SEXP C_fastspline_residual_batch_cuda(SEXP datas,
                                                  SEXP targetss,
                                                  SEXP conditioning_setss,
                                                  SEXP fastspline_paramss,
                                                  SEXP fallbacks) {
  BEGIN_RCPP
  if (!Rf_isReal(datas) || !Rf_isMatrix(datas)) {
    Rcpp::stop("data must be a numeric matrix");
  }
  Rcpp::NumericMatrix data(datas);
  if (!all_finite(data)) Rcpp::stop("data contains missing or infinite values");
  Rcpp::IntegerVector targets(targetss);
  Rcpp::List conditioning_sets(conditioning_setss);
  if (targets.size() != conditioning_sets.size()) {
    Rcpp::stop("targets and conditioning_sets length mismatch");
  }

  std::vector<int> cpp_targets;
  std::vector<std::vector<int> > cpp_conditioning_sets;
  cpp_targets.reserve(targets.size());
  cpp_conditioning_sets.reserve(targets.size());
  for (int i = 0; i < targets.size(); ++i) {
    if (Rcpp::IntegerVector::is_na(targets[i])) Rcpp::stop("targets contain NA");
    const int target = targets[i] - 1;
    if (target < 0 || target >= data.ncol()) Rcpp::stop("target index out of range");
    cpp_targets.push_back(target);

    Rcpp::IntegerVector cond = conditioning_sets[i];
    std::vector<int> cpp_cond;
    cpp_cond.reserve(cond.size());
    for (int j = 0; j < cond.size(); ++j) {
      if (Rcpp::IntegerVector::is_na(cond[j])) Rcpp::stop("conditioning set contains NA");
      const int value = cond[j] - 1;
      if (value < 0 || value >= data.ncol()) {
        Rcpp::stop("conditioning set index out of range");
      }
      cpp_cond.push_back(value);
    }
    cpp_conditioning_sets.push_back(cpp_cond);
  }

  const FastSplineParams params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  const FastSplineCudaBatchResult batch_result =
    fit_fastspline_residuals_cuda_batch_result(data, cpp_targets,
                                               cpp_conditioning_sets, params,
                                               Rcpp::as<bool>(fallbacks));
  const std::vector<FastSplineCudaFit>& fits = batch_result.fits;

  const int n = data.nrow();
  const int batch = static_cast<int>(fits.size());
  Rcpp::NumericMatrix residuals(n, batch);
  Rcpp::NumericMatrix fitted(n, batch);
  Rcpp::NumericVector selected_lambda(batch);
  Rcpp::NumericVector gcv(batch);
  Rcpp::NumericVector rss(batch);
  Rcpp::NumericVector edf(batch);
  Rcpp::IntegerVector design_cols(batch);
  Rcpp::IntegerVector ridge_attempts(batch);
  Rcpp::CharacterVector residual_device(batch);
  Rcpp::LogicalVector fallback_used(batch);
  Rcpp::List diagnostics(batch);

  for (int k = 0; k < batch; ++k) {
    const FastSplineFit& fit = fits[k].fit;
    if (static_cast<int>(fit.residuals.size()) != n ||
        static_cast<int>(fit.fitted.size()) != n) {
      Rcpp::stop("CUDA residual batch result dimension mismatch");
    }
    for (int row = 0; row < n; ++row) {
      residuals(row, k) = fit.residuals[row];
      fitted(row, k) = fit.fitted[row];
    }
    selected_lambda[k] = fit.selected_lambda;
    gcv[k] = fit.gcv;
    rss[k] = fit.rss;
    edf[k] = fit.edf;
    design_cols[k] = fit.design_cols;
    ridge_attempts[k] = fit.ridge_attempts;
    residual_device[k] = fits[k].diagnostics.fallback_used ?
      "cuda-fallback-cpu" : "cuda";
    fallback_used[k] = fits[k].diagnostics.fallback_used;
    diagnostics[k] = Rcpp::List::create(
      Rcpp::Named("cuda_used") = fits[k].diagnostics.cuda_used,
      Rcpp::Named("fallback_used") = fits[k].diagnostics.fallback_used,
      Rcpp::Named("reason") = fits[k].diagnostics.reason,
      Rcpp::Named("batch_group_id") = fits[k].diagnostics.batch_group_id,
      Rcpp::Named("batch_position") = fits[k].diagnostics.batch_position,
      Rcpp::Named("true_batched") = fits[k].diagnostics.true_batched,
      Rcpp::Named("cholesky_backend") = fits[k].diagnostics.cholesky_backend
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("residuals") = residuals,
    Rcpp::Named("fitted") = fitted,
    Rcpp::Named("selected_lambda") = selected_lambda,
    Rcpp::Named("gcv") = gcv,
    Rcpp::Named("rss") = rss,
    Rcpp::Named("edf") = edf,
    Rcpp::Named("design_cols") = design_cols,
    Rcpp::Named("ridge_attempts") = ridge_attempts,
    Rcpp::Named("backend") = "cuda",
    Rcpp::Named("residual_backend") = "fastSpline",
    Rcpp::Named("residual_device") = residual_device,
    Rcpp::Named("fallback_used") = fallback_used,
    Rcpp::Named("diagnostics") = diagnostics,
    Rcpp::Named("batch_diagnostics") =
      fastspline_batch_diagnostics_to_list(batch_result.diagnostics)
  );
  END_RCPP
}

extern "C" SEXP C_mgcv_extract_gpu_solve_handle_fixed_sp(
    SEXP Xs,
    SEXP ys,
    SEXP Zs,
    SEXP XtX_nulls,
    SEXP penalty_nulls,
    SEXP Xty_nulls) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs)) Rcpp::stop("X must be a numeric matrix");
  if (!Rf_isReal(ys)) Rcpp::stop("y must be numeric");
  if (!Rf_isReal(Zs) || !Rf_isMatrix(Zs)) Rcpp::stop("Z must be a numeric matrix");
  if (!Rf_isReal(XtX_nulls) || !Rf_isMatrix(XtX_nulls)) {
    Rcpp::stop("XtX_null must be a numeric matrix");
  }
  if (!Rf_isReal(penalty_nulls) || !Rf_isMatrix(penalty_nulls)) {
    Rcpp::stop("penalty_null must be a numeric matrix");
  }
  if (!Rf_isReal(Xty_nulls)) Rcpp::stop("Xty_null must be numeric");

  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericVector y(ys);
  Rcpp::NumericMatrix Z(Zs);
  Rcpp::NumericMatrix XtX_null(XtX_nulls);
  Rcpp::NumericMatrix penalty_null(penalty_nulls);
  Rcpp::NumericVector Xty_null(Xty_nulls);

  const int n = X.nrow();
  const int p = X.ncol();
  const int q = Z.ncol();
  if (y.size() != n) Rcpp::stop("length(y) must equal nrow(X)");
  if (Z.nrow() != p) Rcpp::stop("nrow(Z) must equal ncol(X)");
  if (XtX_null.nrow() != q || XtX_null.ncol() != q) {
    Rcpp::stop("XtX_null dimensions must match ncol(Z)");
  }
  if (penalty_null.nrow() != q || penalty_null.ncol() != q) {
    Rcpp::stop("penalty_null dimensions must match ncol(Z)");
  }
  if (Xty_null.size() != q) Rcpp::stop("length(Xty_null) must equal ncol(Z)");
  if (!all_finite(X)) Rcpp::stop("X contains missing or infinite values");
  if (!all_finite_vector(y)) Rcpp::stop("y contains missing or infinite values");
  if (!all_finite(Z)) Rcpp::stop("Z contains missing or infinite values");
  if (!all_finite(XtX_null)) {
    Rcpp::stop("XtX_null contains missing or infinite values");
  }
  if (!all_finite(penalty_null)) {
    Rcpp::stop("penalty_null contains missing or infinite values");
  }
  if (!all_finite_vector(Xty_null)) {
    Rcpp::stop("Xty_null contains missing or infinite values");
  }

  const MgcvExtractGpuFixedSpResult result =
    mgcv_extract_fixed_sp_solve_cuda(
      REAL(Xs), n, p, REAL(ys), REAL(Zs), REAL(XtX_nulls),
      REAL(penalty_nulls), REAL(Xty_nulls), q);

  return Rcpp::List::create(
    Rcpp::Named("theta") =
      Rcpp::NumericVector(result.theta.begin(), result.theta.end()),
    Rcpp::Named("coefficients") =
      Rcpp::NumericVector(result.coefficients.begin(),
                          result.coefficients.end()),
    Rcpp::Named("fitted") =
      Rcpp::NumericVector(result.fitted.begin(), result.fitted.end()),
    Rcpp::Named("residuals") =
      Rcpp::NumericVector(result.residuals.begin(), result.residuals.end()),
    Rcpp::Named("rss") = result.rss,
    Rcpp::Named("diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = result.n,
      Rcpp::Named("coefficient_dim") = result.coefficient_dim,
      Rcpp::Named("null_dim") = result.null_dim,
      Rcpp::Named("solve_stage") = "native-gpu-handle-linear-solve",
      Rcpp::Named("cholesky_backend") = result.cholesky_backend
    )
  );
  END_RCPP
}

extern "C" SEXP C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp(
    SEXP Xs,
    SEXP Ys,
    SEXP Zs,
    SEXP XtX_nulls,
    SEXP penalty_null_list_s,
    SEXP Xty_nulls) {
  BEGIN_RCPP
  if (!Rf_isReal(Xs) || !Rf_isMatrix(Xs)) Rcpp::stop("X must be a numeric matrix");
  if (!Rf_isReal(Ys) || !Rf_isMatrix(Ys)) Rcpp::stop("Y must be a numeric matrix");
  if (!Rf_isReal(Zs) || !Rf_isMatrix(Zs)) Rcpp::stop("Z must be a numeric matrix");
  if (!Rf_isReal(XtX_nulls) || !Rf_isMatrix(XtX_nulls)) {
    Rcpp::stop("XtX_null must be a numeric matrix");
  }
  if (!Rf_isNewList(penalty_null_list_s)) {
    Rcpp::stop("penalty_null_list must be a list of numeric matrices");
  }
  if (!Rf_isReal(Xty_nulls) || !Rf_isMatrix(Xty_nulls)) {
    Rcpp::stop("Xty_null must be a numeric matrix");
  }

  Rcpp::NumericMatrix X(Xs);
  Rcpp::NumericMatrix Y(Ys);
  Rcpp::NumericMatrix Z(Zs);
  Rcpp::NumericMatrix XtX_null(XtX_nulls);
  Rcpp::List penalty_list(penalty_null_list_s);
  Rcpp::NumericMatrix Xty_null(Xty_nulls);

  const int n = X.nrow();
  const int p = X.ncol();
  const int targets = Y.ncol();
  const int q = Z.ncol();
  if (Y.nrow() != n) Rcpp::stop("nrow(Y) must equal nrow(X)");
  if (Z.nrow() != p) Rcpp::stop("nrow(Z) must equal ncol(X)");
  if (XtX_null.nrow() != q || XtX_null.ncol() != q) {
    Rcpp::stop("XtX_null dimensions must match ncol(Z)");
  }
  if (penalty_list.size() != targets) {
    Rcpp::stop("penalty_null_list length must equal ncol(Y)");
  }
  if (Xty_null.nrow() != q || Xty_null.ncol() != targets) {
    Rcpp::stop("Xty_null dimensions must be ncol(Z) by ncol(Y)");
  }
  if (!all_finite(X)) Rcpp::stop("X contains missing or infinite values");
  if (!all_finite(Y)) Rcpp::stop("Y contains missing or infinite values");
  if (!all_finite(Z)) Rcpp::stop("Z contains missing or infinite values");
  if (!all_finite(XtX_null)) {
    Rcpp::stop("XtX_null contains missing or infinite values");
  }
  if (!all_finite(Xty_null)) {
    Rcpp::stop("Xty_null contains missing or infinite values");
  }

  Rcpp::NumericMatrix residuals(n, targets);
  Rcpp::NumericMatrix fitted(n, targets);
  Rcpp::NumericMatrix theta(q, targets);
  Rcpp::NumericMatrix coefficients(p, targets);
  Rcpp::NumericVector rss(targets);
  Rcpp::List diagnostics(targets);

  for (int target = 0; target < targets; ++target) {
    Rcpp::NumericMatrix penalty(penalty_list[target]);
    if (penalty.nrow() != q || penalty.ncol() != q) {
      Rcpp::stop("each penalty_null matrix dimensions must match ncol(Z)");
    }
    if (!all_finite(penalty)) {
      Rcpp::stop("penalty_null contains missing or infinite values");
    }
    const MgcvExtractGpuFixedSpResult result =
      mgcv_extract_fixed_sp_solve_cuda(
        REAL(Xs), n, p, &REAL(Ys)[static_cast<std::size_t>(target) * n],
        REAL(Zs), REAL(XtX_nulls), REAL(penalty),
        &REAL(Xty_nulls)[static_cast<std::size_t>(target) * q], q);
    for (int row = 0; row < n; ++row) {
      fitted(row, target) = result.fitted[row];
      residuals(row, target) = result.residuals[row];
    }
    for (int j = 0; j < q; ++j) theta(j, target) = result.theta[j];
    for (int j = 0; j < p; ++j) coefficients(j, target) = result.coefficients[j];
    rss[target] = result.rss;
    diagnostics[target] = Rcpp::List::create(
      Rcpp::Named("n") = result.n,
      Rcpp::Named("coefficient_dim") = result.coefficient_dim,
      Rcpp::Named("null_dim") = result.null_dim,
      Rcpp::Named("target_index") = target + 1,
      Rcpp::Named("solve_stage") = "native-gpu-same-setup-batch-linear-solve",
      Rcpp::Named("cholesky_backend") = result.cholesky_backend
    );
  }

  return Rcpp::List::create(
    Rcpp::Named("theta") = theta,
    Rcpp::Named("coefficients") = coefficients,
    Rcpp::Named("fitted") = fitted,
    Rcpp::Named("residuals") = residuals,
    Rcpp::Named("rss") = rss,
    Rcpp::Named("diagnostics") = diagnostics,
    Rcpp::Named("batch_diagnostics") = Rcpp::List::create(
      Rcpp::Named("n") = n,
      Rcpp::Named("targets") = targets,
      Rcpp::Named("coefficient_dim") = p,
      Rcpp::Named("null_dim") = q,
      Rcpp::Named("native_batch_call") = true,
      Rcpp::Named("setup_reused") = true,
      Rcpp::Named("true_batched_kernel") = false,
      Rcpp::Named("batch_stage") = "native-same-setup-repeated-cuda-solve"
    )
  );
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda(SEXP data, SEXP alphas, SEXP max_ords,
                                      SEXP indexs, SEXP legacy_indexs,
                                      SEXP batch_sizes) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = false;
  options.residual_backend_name = "linear";
  options.residual_device_requested = "cpu";
  options.cuda_residual_fallback = true;
  options.scheduler_requested = "legacy";
  options.residual_batch_size = 0;
  options.scheduler_diagnostics_enabled = true;
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda_cached(SEXP data, SEXP alphas,
                                             SEXP max_ords, SEXP indexs,
                                             SEXP legacy_indexs,
                                             SEXP batch_sizes,
                                             SEXP residual_caches) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = Rcpp::as<bool>(residual_caches);
  options.residual_backend_name = "linear";
  options.residual_device_requested = "cpu";
  options.cuda_residual_fallback = true;
  options.scheduler_requested = "legacy";
  options.residual_batch_size = 0;
  options.scheduler_diagnostics_enabled = true;
  options.fastspline_params = default_fastspline_params();
  apply_ci_options(&options, "dcc.gamma", Rcpp::List::create(),
                   Rcpp::List::create(), true);
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_skeleton_cuda_backend(SEXP data, SEXP alphas,
                                             SEXP max_ords, SEXP indexs,
                                             SEXP legacy_indexs,
                                             SEXP batch_sizes,
                                             SEXP residual_caches,
                                             SEXP residual_backends,
                                             SEXP residual_devices,
                                             SEXP residual_batch_sizes,
                                             SEXP schedulers,
                                             SEXP scheduler_diagnosticss,
                                             SEXP fastspline_paramss,
                                             SEXP cuda_residual_fallbacks,
                                             SEXP ci_methods,
                                             SEXP hsic_paramss,
                                             SEXP permutation_paramss,
                                             SEXP ci_diagnosticss) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  const std::string residual_backend = Rcpp::as<std::string>(residual_backends);
  const FastSplineParams fastspline_params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  make_residual_backend_config(residual_backend, fastspline_params);

  SkeletonOptions options;
  options.alpha = Rf_asReal(alphas);
  options.max_conditioning_size = Rf_asInteger(max_ords);
  options.na_delete = true;
  options.stable = true;
  options.index = Rf_asReal(indexs);
  options.legacy_index = Rcpp::as<bool>(legacy_indexs);
  options.residual_cache_enabled = Rcpp::as<bool>(residual_caches);
  options.residual_backend_name = residual_backend;
  options.residual_device_requested = Rcpp::as<std::string>(residual_devices);
  options.cuda_residual_fallback = Rcpp::as<bool>(cuda_residual_fallbacks);
  options.scheduler_requested = Rcpp::as<std::string>(schedulers);
  options.residual_batch_size = Rf_asInteger(residual_batch_sizes);
  options.scheduler_diagnostics_enabled = Rcpp::as<bool>(scheduler_diagnosticss);
  options.fastspline_params = fastspline_params;
  apply_ci_options(&options, Rcpp::as<std::string>(ci_methods),
                   Rcpp::as<Rcpp::List>(hsic_paramss),
                   Rcpp::as<Rcpp::List>(permutation_paramss),
                   Rcpp::as<bool>(ci_diagnosticss));
  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult result = run_skeleton_cuda_batch(matrix, options, batch_size);
  return skeleton_result_to_list(result, matrix.ncol());
  END_RCPP
}

extern "C" SEXP C_fast_kpc_wanpdag_cuda(SEXP data, SEXP alphas,
                                         SEXP max_ords, SEXP indexs,
                                         SEXP legacy_indexs,
                                         SEXP batch_sizes,
                                         SEXP residual_caches,
                                         SEXP residual_backends,
                                         SEXP residual_devices,
                                         SEXP orientation_devices,
                                         SEXP residual_batch_sizes,
                                         SEXP orientation_batch_sizes,
                                         SEXP schedulers,
                                         SEXP scheduler_diagnosticss,
                                         SEXP orientation_diagnosticss,
                                         SEXP fastspline_paramss,
                                         SEXP cuda_residual_fallbacks,
                                         SEXP orient_colliders,
                                         SEXP solve_confls,
                                         SEXP ruless,
                                         SEXP ci_methods,
                                         SEXP hsic_paramss,
                                         SEXP permutation_paramss,
                                         SEXP ci_diagnosticss) {
  BEGIN_RCPP
  Rcpp::NumericMatrix matrix(data);
  const double alpha = Rf_asReal(alphas);
  const double index = Rf_asReal(indexs);
  const bool legacy_index = Rcpp::as<bool>(legacy_indexs);
  const bool residual_cache = Rcpp::as<bool>(residual_caches);
  const std::string residual_backend = Rcpp::as<std::string>(residual_backends);
  const std::string residual_device = Rcpp::as<std::string>(residual_devices);
  const std::string orientation_device =
    Rcpp::as<std::string>(orientation_devices);
  const FastSplineParams fastspline_params =
    parse_fastspline_params(Rcpp::as<Rcpp::List>(fastspline_paramss));
  make_residual_backend_config(residual_backend, fastspline_params);

  SkeletonOptions skeleton_options;
  skeleton_options.alpha = alpha;
  skeleton_options.max_conditioning_size = Rf_asInteger(max_ords);
  skeleton_options.na_delete = true;
  skeleton_options.stable = true;
  skeleton_options.index = index;
  skeleton_options.legacy_index = legacy_index;
  skeleton_options.residual_cache_enabled = residual_cache;
  skeleton_options.residual_backend_name = residual_backend;
  skeleton_options.residual_device_requested = residual_device;
  skeleton_options.cuda_residual_fallback = Rcpp::as<bool>(cuda_residual_fallbacks);
  skeleton_options.scheduler_requested = Rcpp::as<std::string>(schedulers);
  skeleton_options.residual_batch_size = Rf_asInteger(residual_batch_sizes);
  skeleton_options.scheduler_diagnostics_enabled =
    Rcpp::as<bool>(scheduler_diagnosticss);
  skeleton_options.fastspline_params = fastspline_params;
  apply_ci_options(&skeleton_options, Rcpp::as<std::string>(ci_methods),
                   Rcpp::as<Rcpp::List>(hsic_paramss),
                   Rcpp::as<Rcpp::List>(permutation_paramss),
                   Rcpp::as<bool>(ci_diagnosticss));

  const int batch_size = Rf_asInteger(batch_sizes);
  const SkeletonResult skeleton =
    run_skeleton_cuda_batch(matrix, skeleton_options, batch_size);
  const OrientationOptions orientation_options = make_orientation_options(
    alpha, index, legacy_index, residual_cache, residual_backend,
    orientation_device, Rf_asInteger(orientation_batch_sizes),
    Rcpp::as<bool>(orientation_diagnosticss),
    Rcpp::as<bool>(cuda_residual_fallbacks),
    fastspline_params, Rcpp::as<bool>(orient_colliders),
    Rcpp::as<bool>(solve_confls), Rcpp::as<Rcpp::LogicalVector>(ruless),
    Rcpp::as<std::string>(ci_methods), Rcpp::as<Rcpp::List>(hsic_paramss),
    Rcpp::as<Rcpp::List>(permutation_paramss),
    Rcpp::as<bool>(ci_diagnosticss));
  const OrientationResult orientation =
    orient_wanpdag_native(matrix, skeleton, orientation_options,
                          regrvonps_device);
  Rcpp::List orientation_list = orientation_result_to_list(orientation);
  return Rcpp::List::create(
    Rcpp::Named("skeleton") = skeleton_result_to_list(skeleton, matrix.ncol()),
    Rcpp::Named("orientation") = orientation_list
  );
  END_RCPP
}

extern "C" SEXP C_precision_replay_layer_native(SEXP adjacencys,
                                                SEXP pmaxs,
                                                SEXP edge_xs,
                                                SEXP edge_ys,
                                                SEXP xs,
                                                SEXP ys,
                                                SEXP conditioning_setss,
                                                SEXP p_values,
                                                SEXP alphas,
                                                SEXP trace_levels) {
  BEGIN_RCPP
  Rcpp::IntegerMatrix adjacency_in(adjacencys);
  Rcpp::NumericMatrix pmax_in(pmaxs);
  Rcpp::IntegerVector edge_x(edge_xs);
  Rcpp::IntegerVector edge_y(edge_ys);
  Rcpp::IntegerVector x(xs);
  Rcpp::IntegerVector y(ys);
  Rcpp::List conditioning_sets(conditioning_setss);
  Rcpp::NumericVector pvals(p_values);
  const double alpha = Rf_asReal(alphas);
  const std::string trace_level = Rcpp::as<std::string>(trace_levels);
  const bool full_trace = trace_level == "full";

  const int p = adjacency_in.nrow();
  if (adjacency_in.ncol() != p || pmax_in.nrow() != p || pmax_in.ncol() != p) {
    Rcpp::stop("adjacency and pmax must be square matrices with matching dimensions");
  }
  const int n_tasks = pvals.size();
  if (edge_x.size() != n_tasks || edge_y.size() != n_tasks ||
      x.size() != n_tasks || y.size() != n_tasks ||
      conditioning_sets.size() != n_tasks) {
    Rcpp::stop("task arrays must have the same length");
  }

  std::vector<int> adjacency(static_cast<std::size_t>(p) * p, 0);
  std::vector<double> pmax(static_cast<std::size_t>(p) * p, 0.0);
  std::vector<std::vector<std::vector<int> > > sepsets(
    p, std::vector<std::vector<int> >(p));
  for (int row = 0; row < p; ++row) {
    for (int col = 0; col < p; ++col) {
      adjacency[static_cast<std::size_t>(row) * p + col] =
        adjacency_in(row, col) != 0 ? 1 : 0;
      pmax[static_cast<std::size_t>(row) * p + col] = pmax_in(row, col);
    }
  }

  Rcpp::IntegerVector trace_id, trace_edge_x, trace_edge_y, trace_x, trace_y;
  Rcpp::CharacterVector trace_s_key;
  Rcpp::NumericVector trace_p;
  Rcpp::LogicalVector trace_deleted, trace_ignored;
  Rcpp::IntegerVector replayed_task_index, ignored_task_index, deleted_task_index;

  int tests_replayed = 0;
  int ignored_after_delete = 0;
  int deletions = 0;
  std::map<int, bool> edge_done;
  std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
  std::vector<LevelDeletion> level_log;

  for (int i = 0; i < n_tasks; ++i) {
    const int ex = edge_x[i] - 1;
    const int ey = edge_y[i] - 1;
    const int ox = x[i] - 1;
    const int oy = y[i] - 1;
    if (ex < 0 || ex >= p || ey < 0 || ey >= p ||
        ox < 0 || ox >= p || oy < 0 || oy >= p) {
      Rcpp::stop("task vertex index out of range");
    }
    const int edge_key = ex < ey ? ex * p + ey : ey * p + ex;
    Rcpp::IntegerVector cond_in = conditioning_sets[i];
    std::vector<int> cond;
    cond.reserve(cond_in.size());
    for (int j = 0; j < cond_in.size(); ++j) {
      if (Rcpp::IntegerVector::is_na(cond_in[j])) {
        Rcpp::stop("conditioning set contains NA");
      }
      const int value = cond_in[j] - 1;
      if (value < 0 || value >= p) Rcpp::stop("conditioning set index out of range");
      cond.push_back(value);
    }

    const bool ignored = edge_done[edge_key] ||
      adjacency[static_cast<std::size_t>(ex) * p + ey] == 0;
    if (ignored) {
      ++ignored_after_delete;
      ignored_task_index.push_back(i + 1);
      if (full_trace) {
        trace_id.push_back(i + 1);
        trace_edge_x.push_back(ex + 1);
        trace_edge_y.push_back(ey + 1);
        trace_x.push_back(ox + 1);
        trace_y.push_back(oy + 1);
        trace_s_key.push_back(replay_s_key(cond));
        trace_p.push_back(NA_REAL);
        trace_deleted.push_back(false);
        trace_ignored.push_back(true);
      }
      continue;
    }

    ++tests_replayed;
    replayed_task_index.push_back(i + 1);
    double pval = pvals[i];
    if (!std::isfinite(pval)) pval = 1.0;
    const std::size_t pmax_idx = static_cast<std::size_t>(ex) * p + ey;
    const std::size_t pmax_rev = static_cast<std::size_t>(ey) * p + ex;
    if (pval > pmax[pmax_idx]) {
      pmax[pmax_idx] = pval;
      pmax[pmax_rev] = pval;
    }

    const bool deleted = pval >= alpha;
    if (deleted) {
      ++deletions;
      deleted_task_index.push_back(i + 1);
      delete_edges[static_cast<std::size_t>(ex) * p + ey] = 1;
      delete_edges[static_cast<std::size_t>(ey) * p + ex] = 1;
      sepsets[ex][ey] = cond;
      sepsets[ey][ex] = cond;
      level_log.push_back(LevelDeletion{ex, ey, cond, pval});
      edge_done[edge_key] = true;
    }

    if (full_trace) {
      trace_id.push_back(i + 1);
      trace_edge_x.push_back(ex + 1);
      trace_edge_y.push_back(ey + 1);
      trace_x.push_back(ox + 1);
      trace_y.push_back(oy + 1);
      trace_s_key.push_back(replay_s_key(cond));
      trace_p.push_back(pval);
      trace_deleted.push_back(deleted);
      trace_ignored.push_back(false);
    }
  }

  for (int i = 0; i < p * p; ++i) {
    if (delete_edges[i] != 0) adjacency[i] = 0;
  }

  Rcpp::DataFrame replay_rows = Rcpp::DataFrame::create(
    Rcpp::Named("task_index") = trace_id,
    Rcpp::Named("edge_x") = trace_edge_x,
    Rcpp::Named("edge_y") = trace_edge_y,
    Rcpp::Named("x") = trace_x,
    Rcpp::Named("y") = trace_y,
    Rcpp::Named("S_key") = trace_s_key,
    Rcpp::Named("p_used") = trace_p,
    Rcpp::Named("edge_deleted") = trace_deleted,
    Rcpp::Named("edge_already_deleted") = trace_ignored,
    Rcpp::Named("stringsAsFactors") = false
  );

  return Rcpp::List::create(
    Rcpp::Named("adjacency") = adjacency_to_matrix(adjacency, p),
    Rcpp::Named("sepsets") = sepsets_to_list(sepsets),
    Rcpp::Named("pMax") = pmax_to_matrix(pmax, p),
    Rcpp::Named("n.edgetests") = tests_replayed,
    Rcpp::Named("per.level.log") =
      level_log_to_list(std::vector<std::vector<LevelDeletion> >(1, level_log))[0],
    Rcpp::Named("summary") = Rcpp::List::create(
      Rcpp::Named("tasks_planned") = n_tasks,
      Rcpp::Named("tests_replayed") = tests_replayed,
      Rcpp::Named("tasks_ignored_after_delete") = ignored_after_delete,
      Rcpp::Named("deletions") = deletions
    ),
    Rcpp::Named("replayed_task_index") = replayed_task_index,
    Rcpp::Named("ignored_task_index") = ignored_task_index,
    Rcpp::Named("deleted_task_index") = deleted_task_index,
    Rcpp::Named("replay_rows") = replay_rows
  );
  END_RCPP
}

static const R_CallMethodDef call_methods[] = {
  {"C_fastkpc_cuda_available", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_available), 0},
  {"C_fastkpc_cuda_device_info", reinterpret_cast<DL_FUNC>(&C_fastkpc_cuda_device_info), 0},
  {"C_fast_dcov_batch_cuda", reinterpret_cast<DL_FUNC>(&C_fast_dcov_batch_cuda), 4},
  {"C_fast_hsic_gamma_cuda", reinterpret_cast<DL_FUNC>(&C_fast_hsic_gamma_cuda), 3},
  {"C_fast_hsic_perm_cuda", reinterpret_cast<DL_FUNC>(&C_fast_hsic_perm_cuda), 6},
  {"C_fastspline_residual_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_cuda), 4},
  {"C_fastspline_residual_batch_cuda", reinterpret_cast<DL_FUNC>(&C_fastspline_residual_batch_cuda), 5},
  {"C_mgcv_extract_gpu_solve_handle_fixed_sp", reinterpret_cast<DL_FUNC>(&C_mgcv_extract_gpu_solve_handle_fixed_sp), 6},
  {"C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp", reinterpret_cast<DL_FUNC>(&C_mgcv_extract_gpu_solve_same_setup_batch_fixed_sp), 6},
  {"C_fast_skeleton_cuda", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda), 6},
  {"C_fast_skeleton_cuda_cached", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda_cached), 7},
  {"C_fast_skeleton_cuda_backend", reinterpret_cast<DL_FUNC>(&C_fast_skeleton_cuda_backend), 18},
  {"C_fast_kpc_wanpdag_cuda", reinterpret_cast<DL_FUNC>(&C_fast_kpc_wanpdag_cuda), 24},
  {"C_precision_replay_layer_native", reinterpret_cast<DL_FUNC>(&C_precision_replay_layer_native), 10},
  {nullptr, nullptr, 0}
};

extern "C" void R_init_fastkpc_cuda(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_methods, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}
