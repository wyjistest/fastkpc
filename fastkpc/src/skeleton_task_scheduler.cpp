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
  out.max_level_tasks = 0;
  out.max_level_unique_residuals = 0;
  out.dcov_batch_size_requested = dcov_batch_size_requested;
  out.dcov_batch_size_used = 0;
  out.residual_batch_size_requested = residual_batch_size_requested;
  out.residual_batch_size_used = 0;
  out.plan_elapsed_sec = 0.0;
  out.residual_prefetch_elapsed_sec = 0.0;
  out.ci_eval_elapsed_sec = 0.0;
  out.replay_elapsed_sec = 0.0;
  out.total_elapsed_sec = 0.0;
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
