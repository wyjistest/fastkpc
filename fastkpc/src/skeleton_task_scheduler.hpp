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
  double residual_grouping_sec;
  double residual_host_pack_sec;
  double residual_alloc_sec;
  double residual_h2d_sec;
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
  double residual_host_select_sec;
  double residual_free_sec;
  double residual_true_batch_total_sec;
  int residual_factorization_count;
  int residual_rhs_solve_count;
  int residual_inverse_solve_count;
  double ci_eval_elapsed_sec;
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
  double dcov_free_sec;
  double dcov_total_sec;
  int dcov_chunks;
  int dcov_max_chunk_batch;
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
