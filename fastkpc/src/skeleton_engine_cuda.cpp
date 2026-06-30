#include "skeleton_engine_cuda.hpp"

#include "ci_method.hpp"
#include "cuda/dcov_batch_cuda.hpp"
#include "cuda/fastspline_residual_cuda.hpp"
#include "cuda/hsic_batch_cuda.hpp"
#include "dcov_exact_cpu.hpp"
#include "hsic_batch_types.hpp"
#include "residual_cache.hpp"
#include "skeleton_engine.hpp"
#include "skeleton_task_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <stdexcept>
#include <vector>

namespace {

int idx(int row, int col, int p) {
  return row * p + col;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

double nonnegative_gap(double total, double accounted) {
  return std::max(0.0, total - accounted);
}

struct HsicCudaEvalCounters {
  int gamma_tests = 0;
  int perm_tests = 0;
  int batches = 0;
  int pairs = 0;
  int permutation_replicates = 0;
  std::size_t memory_bytes = 0;
  int max_n = 0;
  int max_batch_pairs = 0;
};

struct DcovWorkspaceHolder {
  DcovCudaWorkspace* value = nullptr;
  explicit DcovWorkspaceHolder(bool enabled)
      : value(enabled ? create_dcov_cuda_workspace() : nullptr) {}
  ~DcovWorkspaceHolder() { destroy_dcov_cuda_workspace(value); }
};

struct MissingResidual {
  int position;
  ResidualCacheKey key;
};

std::vector<double> column_as_vector(const Rcpp::NumericMatrix& data, int col) {
  std::vector<double> out(data.nrow());
  for (int i = 0; i < data.nrow(); ++i) out[i] = data(i, col);
  return out;
}

std::pair<int, int> minmax_design_cols(
    const FastSplineCudaBatchDiagnostics& diagnostics) {
  if (diagnostics.group_design_cols.empty()) return std::make_pair(0, 0);
  int min_value = diagnostics.group_design_cols[0];
  int max_value = diagnostics.group_design_cols[0];
  for (int value : diagnostics.group_design_cols) {
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
  return std::make_pair(min_value, max_value);
}

class CudaSkeletonResidualCache {
 public:
  CudaSkeletonResidualCache(const std::string& backend_name,
                            const FastSplineParams& fastspline_params,
                            bool enabled,
                            const std::string& requested_device,
                            bool fallback)
      : backend_(make_residual_backend_config(backend_name, fastspline_params)),
        enabled_(enabled),
        fallback_(fallback) {
    requested_device_ = requested_device.empty() ? "cpu" : requested_device;
    if (requested_device_ != "cpu" && requested_device_ != "cuda" &&
        requested_device_ != "auto") {
      throw std::runtime_error("Unknown residual device: " + requested_device_);
    }
    used_device_ = resolve_device();
    residual_workspace_ =
      backend_.kind == ResidualBackendKind::FastSpline &&
      used_device_ == "cuda" ?
        create_fastspline_cuda_workspace() : nullptr;
    stats_.enabled = enabled_;
    stats_.requests = 0;
    stats_.hits = 0;
    stats_.misses = 0;
    stats_.computations = 0;
    stats_.stored_vectors = 0;
    stats_.stored_values = 0;
    stats_.backend_name = backend_.name;
  }

  ~CudaSkeletonResidualCache() {
    destroy_fastspline_cuda_workspace(residual_workspace_);
    residual_workspace_ = nullptr;
  }

  const std::vector<double>& get(const Rcpp::NumericMatrix& data,
                                 int target,
                                 const std::vector<int>& conditioning_set) {
    ++stats_.requests;
    const ResidualCacheKey key = make_residual_cache_key(
      target, conditioning_set, data.nrow(), data.ncol(), backend_.name,
      backend_.params);

    if (!enabled_) {
      ++stats_.computations;
      scratch_ = compute(data, target, key.conditioning_set);
      return scratch_;
    }

    std::map<ResidualCacheKey, std::vector<double> >::iterator it = values_.find(key);
    if (it != values_.end()) {
      ++stats_.hits;
      return it->second;
    }

    std::vector<double> residuals = compute_and_count(data, target, key.conditioning_set);
    std::pair<std::map<ResidualCacheKey, std::vector<double> >::iterator, bool> inserted =
      values_.insert(std::make_pair(key, residuals));
    update_storage(data.nrow());
    return inserted.first->second;
  }

  int prefetch_level(const Rcpp::NumericMatrix& data,
                     const std::vector<LayerResidualRequest>& requests,
                     int level,
                     int residual_batch_size,
                     SchedulerDiagnostics* diagnostics) {
    if (!enabled_ || requests.empty()) return 0;
    const int actual_batch_size = resolve_residual_batch_size(
      residual_batch_size, static_cast<int>(requests.size()));
    diagnostics->residual_batch_size_used =
      std::max(diagnostics->residual_batch_size_used, actual_batch_size);
    const int expected_batches =
      (static_cast<int>(requests.size()) + actual_batch_size - 1) /
      actual_batch_size;
    diagnostics->batches.reserve(diagnostics->batches.size() +
                                 static_cast<std::size_t>(expected_batches));
    diagnostics->residuals.reserve(diagnostics->residuals.size() +
                                   requests.size());

    int batch_count = 0;
    for (int start = 0; start < static_cast<int>(requests.size());
         start += actual_batch_size) {
      const int count = std::min(actual_batch_size,
                                 static_cast<int>(requests.size()) - start);
      std::vector<MissingResidual> missing;
      missing.reserve(count);

      std::chrono::steady_clock::time_point stage =
        std::chrono::steady_clock::now();
      for (int k = 0; k < count; ++k) {
        const LayerResidualRequest& request = requests[start + k];
        ResidualCacheKey key = make_key(data, request.target,
                                        request.conditioning_set);
        if (values_.find(key) == values_.end()) {
          MissingResidual miss;
          miss.position = start + k;
          miss.key = std::move(key);
          missing.push_back(std::move(miss));
        } else {
          diagnostics->residuals.push_back(SchedulerResidualDiagnostic{
            level, request.request_id, request.target,
            static_cast<int>(request.conditioning_set.size()), backend_.name,
            used_device_, true, false, "cache-hit"});
        }
      }
      diagnostics->residual_prefetch_missing_scan_sec += seconds_since(stage);

      if (missing.empty()) continue;
      ++batch_count;
      SchedulerBatchDiagnostic batch_diag{
        level, static_cast<int>(diagnostics->batches.size()), "residual",
        requests[missing.front().position].request_id,
        static_cast<int>(missing.size()), data.nrow(), "ok",
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      if (backend_.kind == ResidualBackendKind::FastSpline &&
          used_device_ == "cuda") {
        stage = std::chrono::steady_clock::now();
        std::vector<int> targets;
        std::vector<std::vector<int> > conditioning_sets;
        targets.reserve(missing.size());
        conditioning_sets.reserve(missing.size());
        for (const MissingResidual& miss : missing) {
          const LayerResidualRequest& request = requests[miss.position];
          targets.push_back(request.target);
          conditioning_sets.push_back(miss.key.conditioning_set);
        }
        diagnostics->residual_prefetch_batch_input_sec += seconds_since(stage);
        stage = std::chrono::steady_clock::now();
        FastSplineCudaBatchResult batch_result =
          fit_fastspline_residuals_cuda_batch_result(
            data, targets, conditioning_sets, backend_.fastspline, fallback_,
            residual_workspace_);
        diagnostics->residual_batch_call_wall_sec += seconds_since(stage);
        std::vector<FastSplineCudaFit>& fits = batch_result.fits;
        stage = std::chrono::steady_clock::now();
        const std::pair<int, int> design_cols =
          minmax_design_cols(batch_result.diagnostics);
        batch_diag.groups = batch_result.diagnostics.groups;
        batch_diag.true_batched_groups =
          batch_result.diagnostics.true_batched_groups;
        batch_diag.true_batched_fits =
          batch_result.diagnostics.true_batched_fits;
        batch_diag.single_fit_calls =
          batch_result.diagnostics.single_fit_calls;
        batch_diag.cpu_fallback_fits =
          batch_result.diagnostics.cpu_fallback_fits;
        batch_diag.unique_designs = batch_result.diagnostics.unique_designs;
        batch_diag.duplicate_design_fits =
          batch_result.diagnostics.duplicate_design_fits;
        batch_diag.max_fits_per_design =
          batch_result.diagnostics.max_fits_per_design;
        batch_diag.max_group_size = batch_result.diagnostics.max_group_size;
        batch_diag.min_group_size = batch_result.diagnostics.min_group_size;
        batch_diag.min_design_cols = design_cols.first;
        batch_diag.max_design_cols = design_cols.second;
        diagnostics->cuda_residual_batch_groups += batch_result.diagnostics.groups;
        diagnostics->cuda_residual_true_batched_groups +=
          batch_result.diagnostics.true_batched_groups;
        diagnostics->cuda_residual_true_batched_fits +=
          batch_result.diagnostics.true_batched_fits;
        diagnostics->cuda_residual_single_fit_calls +=
          batch_result.diagnostics.single_fit_calls;
        diagnostics->cuda_residual_cpu_fallback_fits +=
          batch_result.diagnostics.cpu_fallback_fits;
        diagnostics->cuda_residual_unique_designs +=
          batch_result.diagnostics.unique_designs;
        diagnostics->cuda_residual_duplicate_design_fits +=
          batch_result.diagnostics.duplicate_design_fits;
        diagnostics->cuda_residual_max_fits_per_design =
          std::max(diagnostics->cuda_residual_max_fits_per_design,
                   batch_result.diagnostics.max_fits_per_design);
        diagnostics->residual_grouping_sec +=
          batch_result.diagnostics.grouping_sec;
        diagnostics->residual_host_pack_sec +=
          batch_result.diagnostics.host_pack_sec;
        diagnostics->residual_alloc_sec +=
          batch_result.diagnostics.alloc_sec;
        diagnostics->residual_h2d_sec += batch_result.diagnostics.h2d_sec;
        diagnostics->residual_xtx_xty_sec +=
          batch_result.diagnostics.xtx_xty_sec;
        diagnostics->residual_pointer_setup_sec +=
          batch_result.diagnostics.pointer_setup_sec;
        diagnostics->residual_active_copy_sec +=
          batch_result.diagnostics.active_copy_sec;
        diagnostics->residual_build_system_sec +=
          batch_result.diagnostics.build_system_sec;
        diagnostics->residual_factor_solve_sec +=
          batch_result.diagnostics.factor_solve_sec;
        diagnostics->residual_factor_cholesky_sec +=
          batch_result.diagnostics.factor_cholesky_sec;
        diagnostics->residual_factor_rhs_solve_sec +=
          batch_result.diagnostics.factor_rhs_solve_sec;
        diagnostics->residual_factor_inverse_solve_sec +=
          batch_result.diagnostics.factor_inverse_solve_sec;
        diagnostics->residual_summary_sec +=
          batch_result.diagnostics.residual_summary_sec;
        diagnostics->residual_d2h_sec += batch_result.diagnostics.d2h_sec;
        diagnostics->residual_host_select_sec +=
          batch_result.diagnostics.host_select_sec;
        diagnostics->residual_free_sec += batch_result.diagnostics.free_sec;
        diagnostics->residual_true_batch_total_sec +=
          batch_result.diagnostics.true_batch_total_sec;
        diagnostics->residual_factorization_count +=
          batch_result.diagnostics.factorization_count;
        diagnostics->residual_rhs_solve_count +=
          batch_result.diagnostics.rhs_solve_count;
        diagnostics->residual_inverse_solve_count +=
          batch_result.diagnostics.inverse_solve_count;
        diagnostics->residual_rhs_solve_api_calls +=
          batch_result.diagnostics.rhs_solve_api_calls;
        diagnostics->residual_rhs_target_solves +=
          batch_result.diagnostics.rhs_target_solves;
        diagnostics->residual_winning_factor_reuse_count +=
          batch_result.diagnostics.winning_factor_reuse_count;
        diagnostics->residual_factor_cache_hits +=
          batch_result.diagnostics.factor_cache_hits;
        diagnostics->residual_factor_cache_misses +=
          batch_result.diagnostics.factor_cache_misses;
        diagnostics->residual_factor_cache_entries +=
          batch_result.diagnostics.factor_cache_entries;
        diagnostics->residual_factor_cache_bytes +=
          batch_result.diagnostics.factor_cache_bytes;
        diagnostics->residual_lambda_candidates = std::max(
          diagnostics->residual_lambda_candidates,
          batch_result.diagnostics.lambda_candidates);
        diagnostics->residual_workspace_reuse_count +=
          batch_result.diagnostics.workspace_reuse_count;
        diagnostics->residual_workspace_grow_count +=
          batch_result.diagnostics.workspace_grow_count;
        diagnostics->residual_solver_handle_create_count +=
          batch_result.diagnostics.solver_handle_create_count;
        diagnostics->residual_per_request_design_x_values +=
          batch_result.diagnostics.per_request_design_x_values;
        diagnostics->residual_duplicate_design_x_values_avoided +=
          batch_result.diagnostics.duplicate_design_x_values_avoided;
        diagnostics->residual_algebraic_rss_count +=
          batch_result.diagnostics.algebraic_rss_count;
        diagnostics->residual_candidate_residual_materialize_count +=
          batch_result.diagnostics.candidate_residual_materialize_count;
        diagnostics->residual_winning_residual_materialize_count +=
          batch_result.diagnostics.winning_residual_materialize_count;
        diagnostics->residual_algebraic_rss_clamp_count +=
          batch_result.diagnostics.algebraic_rss_clamp_count;
        diagnostics->residual_diagnostic_merge_sec += seconds_since(stage);
        for (int i = 0; i < static_cast<int>(missing.size()); ++i) {
          const LayerResidualRequest& request = requests[missing[i].position];
          insert_prefetched_key(data.nrow(), missing[i].key,
                                std::move(fits[i].fit.residuals),
                                diagnostics);
          stage = std::chrono::steady_clock::now();
          if (fits[i].diagnostics.fallback_used) {
            used_device_ = "cuda-fallback-cpu";
            if (reason_.empty()) reason_ = fits[i].diagnostics.reason;
          }
          diagnostics->residuals.push_back(SchedulerResidualDiagnostic{
            level, request.request_id, request.target,
            static_cast<int>(request.conditioning_set.size()), backend_.name,
            fits[i].diagnostics.fallback_used ? "cuda-fallback-cpu" : "cuda",
            true, fits[i].diagnostics.fallback_used,
            fits[i].diagnostics.reason});
          diagnostics->residual_diagnostic_merge_sec += seconds_since(stage);
        }
      } else {
        batch_diag.groups = 1;
        batch_diag.unique_designs = static_cast<int>(missing.size());
        batch_diag.max_group_size = static_cast<int>(missing.size());
        batch_diag.min_group_size = static_cast<int>(missing.size());
        for (MissingResidual& miss : missing) {
          const LayerResidualRequest& request = requests[miss.position];
          std::vector<double> residuals =
            compute_residuals_with_backend(data, request.target,
                                           miss.key.conditioning_set, backend_);
          insert_prefetched_key(data.nrow(), miss.key, std::move(residuals),
                                diagnostics);
          stage = std::chrono::steady_clock::now();
          diagnostics->residuals.push_back(SchedulerResidualDiagnostic{
            level, request.request_id, request.target,
            static_cast<int>(request.conditioning_set.size()), backend_.name,
            used_device_, true, false, ""});
          diagnostics->residual_diagnostic_merge_sec += seconds_since(stage);
        }
      }
      stage = std::chrono::steady_clock::now();
      diagnostics->batches.push_back(batch_diag);
      diagnostics->residual_diagnostic_merge_sec += seconds_since(stage);
    }
    return batch_count;
  }

  ResidualCacheStats stats() const {
    ResidualCacheStats out = stats_;
    out.stored_vectors = static_cast<int>(values_.size());
    out.stored_values = out.stored_vectors * (values_.empty() ? 0 :
      static_cast<int>(values_.begin()->second.size()));
    return out;
  }

  const ResidualBackendConfig& backend() const { return backend_; }
  const std::string& used_device() const { return used_device_; }
  const std::string& requested_device() const { return requested_device_; }
  const std::string& reason() const { return reason_; }

 private:
  std::string resolve_device() {
    if (backend_.kind == ResidualBackendKind::Linear) {
      if (requested_device_ == "cuda") {
        reason_ = "linear residual CUDA device is not implemented";
      }
      return "cpu";
    }
    if (requested_device_ == "cpu") return "cpu";
    return "cuda";
  }

  ResidualCacheKey make_key(const Rcpp::NumericMatrix& data,
                            int target,
                            const std::vector<int>& conditioning_set) const {
    return make_residual_cache_key(target, conditioning_set, data.nrow(),
                                   data.ncol(), backend_.name, backend_.params);
  }

  std::vector<double> compute(const Rcpp::NumericMatrix& data,
                              int target,
                              const std::vector<int>& conditioning_set) {
    if (backend_.kind == ResidualBackendKind::FastSpline &&
        used_device_ == "cuda") {
      const FastSplineCudaFit fit = fit_fastspline_residuals_cuda(
        data, target, conditioning_set, backend_.fastspline, fallback_);
      if (fit.diagnostics.fallback_used) {
        used_device_ = "cuda-fallback-cpu";
        if (reason_.empty()) reason_ = fit.diagnostics.reason;
      }
      return fit.fit.residuals;
    }
    return compute_residuals_with_backend(data, target, conditioning_set, backend_);
  }

  std::vector<double> compute_and_count(const Rcpp::NumericMatrix& data,
                                        int target,
                                        const std::vector<int>& conditioning_set) {
    ++stats_.misses;
    ++stats_.computations;
    return compute(data, target, conditioning_set);
  }

  void insert_prefetched(const Rcpp::NumericMatrix& data,
                         int target,
                         const std::vector<int>& conditioning_set,
                         const std::vector<double>& residuals) {
    const ResidualCacheKey key = make_key(data, target, conditioning_set);
    if (values_.find(key) != values_.end()) return;
    ++stats_.misses;
    ++stats_.computations;
    values_.insert(std::make_pair(key, residuals));
    update_storage(data.nrow());
  }

  void insert_prefetched_key(int n,
                             const ResidualCacheKey& key,
                             std::vector<double>&& residuals,
                             SchedulerDiagnostics* diagnostics) {
    const std::chrono::steady_clock::time_point stage =
      std::chrono::steady_clock::now();
    if (values_.find(key) == values_.end()) {
      ++stats_.misses;
      ++stats_.computations;
      values_.insert(std::make_pair(key, std::move(residuals)));
      update_storage(n);
      if (diagnostics != nullptr) ++diagnostics->residual_cache_move_insert_count;
    }
    if (diagnostics != nullptr) {
      diagnostics->residual_cache_insert_sec += seconds_since(stage);
    }
  }

  void update_storage(int n) {
    stats_.stored_vectors = static_cast<int>(values_.size());
    stats_.stored_values = stats_.stored_vectors * n;
  }

  ResidualBackendConfig backend_;
  bool enabled_;
  bool fallback_;
  std::string requested_device_;
  std::string used_device_;
  std::string reason_;
  ResidualCacheStats stats_;
  std::map<ResidualCacheKey, std::vector<double> > values_;
  std::vector<double> scratch_;
  FastSplineCudaWorkspace* residual_workspace_ = nullptr;
};

void copy_column_to_buffer(const Rcpp::NumericMatrix& data,
                           int col,
                           double* out) {
  for (int row = 0; row < data.nrow(); ++row) out[row] = data(row, col);
}

void copy_vector_to_buffer(const std::vector<double>& values,
                           int n,
                           double* out) {
  if (static_cast<int>(values.size()) != n) {
    throw std::runtime_error("CI host pack residual length mismatch");
  }
  std::copy(values.begin(), values.end(), out);
}

void pack_task_vectors_directly(const Rcpp::NumericMatrix& data,
                                const LayerCiTask& task,
                                CudaSkeletonResidualCache* residual_cache,
                                double* x,
                                double* y) {
  const int n = data.nrow();
  if (task.conditioning_set.empty()) {
    copy_column_to_buffer(data, task.orientation_x, x);
    copy_column_to_buffer(data, task.orientation_y, y);
    return;
  }

  const std::vector<double>& x_residual =
    residual_cache->get(data, task.orientation_x, task.conditioning_set);
  copy_vector_to_buffer(x_residual, n, x);
  const std::vector<double>& y_residual =
    residual_cache->get(data, task.orientation_y, task.conditioning_set);
  copy_vector_to_buffer(y_residual, n, y);
}

double pack_ci_task_batch(const Rcpp::NumericMatrix& data,
                          const std::vector<LayerCiTask>& tasks,
                          int start,
                          int count,
                          int n,
                          CudaSkeletonResidualCache* residual_cache,
                          double* xmat,
                          double* ymat) {
  const std::chrono::steady_clock::time_point pack_start =
    std::chrono::steady_clock::now();
  for (int k = 0; k < count; ++k) {
    pack_task_vectors_directly(
      data, tasks[start + k], residual_cache,
      xmat + static_cast<std::size_t>(k) * n,
      ymat + static_cast<std::size_t>(k) * n);
  }
  return seconds_since(pack_start);
}

std::vector<double> evaluate_tasks_cuda(const Rcpp::NumericMatrix& data,
                                        const std::vector<LayerCiTask>& tasks,
                                        int batch_size,
                                        double index,
                                        bool legacy_index,
                                        int level,
                                        CudaSkeletonResidualCache* residual_cache,
                                        DcovCudaWorkspace* dcov_workspace,
                                        SchedulerDiagnostics* diagnostics,
                                        int* dcov_batches) {
  const int n = data.nrow();
  std::vector<double> pvalues(tasks.size(), std::numeric_limits<double>::quiet_NaN());
  if (tasks.empty()) return pvalues;
  const int actual_batch_size = resolve_dcov_batch_size(
    batch_size, n, static_cast<int>(tasks.size()));
  diagnostics->dcov_batch_size_used =
    std::max(diagnostics->dcov_batch_size_used, actual_batch_size);
  const int expected_batches =
    (static_cast<int>(tasks.size()) + actual_batch_size - 1) /
    actual_batch_size;
  diagnostics->batches.reserve(diagnostics->batches.size() +
                               static_cast<std::size_t>(expected_batches));

  DcovBatchOptions options;
  options.index = index;
  options.legacy_index = legacy_index;

  std::vector<double> xmat(static_cast<std::size_t>(n) * actual_batch_size);
  std::vector<double> ymat(static_cast<std::size_t>(n) * actual_batch_size);

  for (int start = 0; start < static_cast<int>(tasks.size()); start += actual_batch_size) {
    const int count = std::min(actual_batch_size, static_cast<int>(tasks.size()) - start);
    diagnostics->ci_host_pack_sec += pack_ci_task_batch(
      data, tasks, start, count, n, residual_cache, xmat.data(), ymat.data());

    std::chrono::steady_clock::time_point stage =
      std::chrono::steady_clock::now();
    DcovBatchResult batch = dcov_batch_cuda_pvalues_into(
      xmat.data(), ymat.data(), n, count, options, dcov_workspace,
      pvalues.data() + start);
    diagnostics->ci_dcov_call_wall_sec += seconds_since(stage);
    stage = std::chrono::steady_clock::now();
    diagnostics->ci_pvalue_copy_sec += seconds_since(stage);
    stage = std::chrono::steady_clock::now();
    diagnostics->dcov_alloc_sec += batch.alloc_sec;
    diagnostics->dcov_h2d_sec += batch.h2d_sec;
    diagnostics->dcov_memset_sec += batch.memset_sec;
    diagnostics->dcov_rowsum_sec += batch.rowsum_sec;
    diagnostics->dcov_totals_d2h_sec += batch.totals_d2h_sec;
    diagnostics->dcov_reduce_sec += batch.reduce_sec;
    diagnostics->dcov_scalars_d2h_sec += batch.scalars_d2h_sec;
    diagnostics->dcov_host_scalar_sec += batch.host_scalar_sec;
    diagnostics->dcov_result_materialize_sec += batch.result_materialize_sec;
    diagnostics->dcov_free_sec += batch.free_sec;
    diagnostics->dcov_total_sec += batch.total_sec;
    diagnostics->dcov_top_level_wall_sec += batch.top_level_wall_sec;
    diagnostics->dcov_grid_limit_query_sec += batch.grid_limit_query_sec;
    diagnostics->dcov_chunk_dispatch_sec += batch.chunk_dispatch_sec;
    diagnostics->dcov_top_level_unaccounted_sec +=
      batch.top_level_unaccounted_sec;
    diagnostics->dcov_chunks += batch.chunks;
    diagnostics->dcov_max_chunk_batch =
      std::max(diagnostics->dcov_max_chunk_batch, batch.max_chunk_batch);
    diagnostics->dcov_workspace_reuse_count += batch.workspace_reuse_count;
    diagnostics->dcov_workspace_grow_count += batch.workspace_grow_count;
    diagnostics->dcov_raw_aggregate_fused_count +=
      batch.raw_aggregate_fused_count;
    diagnostics->dcov_row_product_reduce_count +=
      batch.row_product_reduce_count;
    diagnostics->dcov_pvalue_only_count += batch.pvalue_only_count;
    diagnostics->dcov_full_result_materialize_count +=
      batch.full_result_materialize_count;
    diagnostics->dcov_grid_limit_query_count +=
      batch.grid_limit_query_count;
    diagnostics->dcov_grid_limit_cache_hit_count +=
      batch.grid_limit_cache_hit_count;
    ++(*dcov_batches);
    diagnostics->batches.push_back(SchedulerBatchDiagnostic{
      level, *dcov_batches - 1, "dcov", tasks[start].task_id, count, n, "ok",
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    diagnostics->ci_diagnostic_append_sec += seconds_since(stage);
  }
  return pvalues;
}

HsicBatchOptions make_hsic_batch_options(const HsicOptions& hsic_options,
                                         CiMethodKind kind) {
  HsicBatchOptions options = default_hsic_batch_options();
  if (std::isfinite(hsic_options.sig) && hsic_options.sig > 0.0) {
    options.sig = hsic_options.sig;
  }
  options.permutation_replicates = hsic_options.replicates;
  options.include_observed = hsic_options.include_observed;
  options.has_seed = hsic_options.has_seed;
  options.seed = hsic_options.seed;
  options.return_replicates = false;
  options.max_n = hsic_options.cuda_max_n;
  options.max_batch_pairs = hsic_options.cuda_max_batch_pairs;
  if (kind == CiMethodKind::HsicGamma) {
    options.permutation_replicates = 0;
  }
  return options;
}

std::vector<double> evaluate_tasks_hsic_cuda(
    const Rcpp::NumericMatrix& data,
    const std::vector<LayerCiTask>& tasks,
    int batch_size,
    CiMethodKind kind,
    const HsicOptions& hsic_options,
    int level,
    CudaSkeletonResidualCache* residual_cache,
    SchedulerDiagnostics* diagnostics,
    HsicCudaEvalCounters* counters) {
  const int n = data.nrow();
  std::vector<double> pvalues(tasks.size(), std::numeric_limits<double>::quiet_NaN());
  if (tasks.empty()) return pvalues;

  HsicBatchOptions options = make_hsic_batch_options(hsic_options, kind);
  counters->max_n = options.max_n;
  counters->max_batch_pairs = options.max_batch_pairs;

  int actual_batch_size = resolve_dcov_batch_size(
    batch_size, n, static_cast<int>(tasks.size()));
  if (options.max_batch_pairs > 0) {
    actual_batch_size = std::min(actual_batch_size, options.max_batch_pairs);
  }
  actual_batch_size = std::max(1, actual_batch_size);
  diagnostics->dcov_batch_size_used =
    std::max(diagnostics->dcov_batch_size_used, actual_batch_size);

  std::vector<double> xmat(static_cast<std::size_t>(n) * actual_batch_size);
  std::vector<double> ymat(static_cast<std::size_t>(n) * actual_batch_size);

  for (int start = 0; start < static_cast<int>(tasks.size()); start += actual_batch_size) {
    const int count = std::min(actual_batch_size, static_cast<int>(tasks.size()) - start);
    diagnostics->ci_host_pack_sec += pack_ci_task_batch(
      data, tasks, start, count, n, residual_cache, xmat.data(), ymat.data());

    const HsicBatchResult batch =
      kind == CiMethodKind::HsicGamma ?
        hsic_gamma_batch_cuda(xmat.data(), ymat.data(), n, count, options) :
        hsic_permutation_batch_cuda(xmat.data(), ymat.data(), n, count, options);
    if (batch.diagnostics.backend != "cuda-hsic") {
      throw std::runtime_error("CUDA HSIC batch did not report cuda-hsic backend");
    }
    for (int k = 0; k < count; ++k) {
      pvalues[start + k] = batch.p_values[k];
    }

    ++counters->batches;
    counters->pairs += count;
    counters->memory_bytes =
      std::max(counters->memory_bytes, batch.diagnostics.bytes_allocated);
    if (kind == CiMethodKind::HsicGamma) {
      counters->gamma_tests += count;
    } else {
      counters->perm_tests += count;
      counters->permutation_replicates +=
        count * batch.diagnostics.permutation_replicates;
    }
    diagnostics->batches.push_back(SchedulerBatchDiagnostic{
      level, counters->batches - 1, "hsic", tasks[start].task_id, count, n, "ok",
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  }
  return pvalues;
}

void replay_layer_pvalues(const LayerPlan& plan,
                          const std::vector<double>& pvalues,
                          const SkeletonOptions& options,
                          int p,
                          std::vector<int>* delete_edges,
                          SkeletonResult* result,
                          int* tests_replayed,
                          std::vector<LevelDeletion>* level_log) {
  std::map<int, bool> edge_done;
  *tests_replayed = 0;

  for (int i = 0; i < static_cast<int>(plan.tasks.size()); ++i) {
    const LayerCiTask& task = plan.tasks[i];
    if (edge_done[task.edge_key]) continue;

    ++(*tests_replayed);
    double pval = pvalues[i];
    if (!std::isfinite(pval)) pval = options.na_delete ? 1.0 : 0.0;

    const double current = result->pmax[idx(task.edge_x, task.edge_y, p)];
    if (pval > current) {
      result->pmax[idx(task.edge_x, task.edge_y, p)] = pval;
      result->pmax[idx(task.edge_y, task.edge_x, p)] = pval;
    }

    if (pval >= options.alpha) {
      (*delete_edges)[idx(task.edge_x, task.edge_y, p)] = 1;
      (*delete_edges)[idx(task.edge_y, task.edge_x, p)] = 1;
      result->sepsets[task.edge_x][task.edge_y] = task.conditioning_set;
      result->sepsets[task.edge_y][task.edge_x] = task.conditioning_set;
      level_log->push_back(LevelDeletion{task.edge_x, task.edge_y,
                                         task.conditioning_set, pval});
      edge_done[task.edge_key] = true;
    }
  }
}

void finalize_scheduler_diagnostics(SchedulerDiagnostics* diagnostics,
                                    const ResidualCacheStats& stats,
                                    const SkeletonResult& result) {
  diagnostics->residual_requests = stats.requests;
  diagnostics->tasks_ignored_after_delete =
    diagnostics->tasks_evaluated - diagnostics->tests_replayed;
  if (diagnostics->tasks_ignored_after_delete < 0) {
    throw std::runtime_error("scheduler diagnostic identity failed: ignored tasks");
  }
  int replayed = 0;
  for (int value : result.n_edge_tests) replayed += value;
  if (replayed != diagnostics->tests_replayed) {
    throw std::runtime_error("scheduler diagnostic identity failed: replayed tests");
  }
}

SkeletonResult run_skeleton_cuda_impl(const Rcpp::NumericMatrix& data,
                                      const SkeletonOptions& options,
                                      int batch_size,
                                      const std::string& scheduler) {
  const int p = data.ncol();
  const std::string residual_backend_name =
    options.residual_backend_name.empty() ? "linear" : options.residual_backend_name;
  const std::string residual_device_requested =
    options.residual_device_requested.empty() ? "cpu" : options.residual_device_requested;
  const std::string scheduler_requested =
    options.scheduler_requested.empty() ? scheduler : options.scheduler_requested;
  const CiMethodKind ci_method = parse_ci_method_kind(options.ci_method);

  CudaSkeletonResidualCache residual_cache(
    residual_backend_name, options.fastspline_params,
    options.residual_cache_enabled, residual_device_requested,
    options.cuda_residual_fallback);

  SkeletonResult result;
  result.adjacency.assign(static_cast<std::size_t>(p) * p, 1);
  result.pmax.assign(static_cast<std::size_t>(p) * p,
                     -std::numeric_limits<double>::infinity());
  result.sepsets.resize(p, std::vector<std::vector<int> >(p));
  result.scheduler = scheduler;
  result.scheduler_requested = scheduler_requested;
  result.ci_method = ci_method_name(ci_method);
  result.ci_backend =
    ci_method == CiMethodKind::DccGamma ? "cuda-dcov" : "cuda-hsic";
  result.ci_backend_reason = "";
  result.ci_dcc_gamma_tests = 0;
  result.ci_hsic_gamma_tests = 0;
  result.ci_hsic_perm_tests = 0;
  result.ci_hsic_permutation_replicates = 0;
  result.ci_hsic_gamma_cuda_tests = 0;
  result.ci_hsic_perm_cuda_tests = 0;
  result.ci_hsic_cuda_batches = 0;
  result.ci_hsic_cuda_pairs = 0;
  result.ci_hsic_cuda_fallback_tests = 0;
  result.ci_hsic_cuda_memory_bytes = 0;
  result.ci_hsic_cuda_max_n = 0;
  result.ci_hsic_cuda_max_batch_pairs = 0;

  for (int i = 0; i < p; ++i) {
    result.adjacency[idx(i, i, p)] = 0;
    result.pmax[idx(i, i, p)] = 1.0;
  }

  SchedulerDiagnostics diagnostics = make_scheduler_diagnostics(
    scheduler, scheduler_requested, batch_size, options.residual_batch_size);
  HsicCudaEvalCounters hsic_counters;
  DcovWorkspaceHolder dcov_workspace(ci_method == CiMethodKind::DccGamma);

  const int max_order = std::max(0, options.max_conditioning_size);
  for (int ord = 0; ord <= max_order; ++ord) {
    const std::chrono::steady_clock::time_point level_start =
      std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point plan_start =
      std::chrono::steady_clock::now();
    const std::vector<int> snapshot = result.adjacency;
    LayerPlan plan = make_layer_plan(snapshot, p, ord);
    const double plan_elapsed_sec = seconds_since(plan_start);

    std::vector<LayerResidualRequest> residual_requests;
    int residual_batches = 0;

    const ResidualCacheStats before_stats = residual_cache.stats();
    const double residual_request_collect_before =
      diagnostics.residual_request_collect_sec;
    const double residual_missing_scan_before =
      diagnostics.residual_prefetch_missing_scan_sec;
    const double residual_batch_input_before =
      diagnostics.residual_prefetch_batch_input_sec;
    const double residual_batch_call_before =
      diagnostics.residual_batch_call_wall_sec;
    const double residual_diagnostic_merge_before =
      diagnostics.residual_diagnostic_merge_sec;
    const double residual_cache_insert_before =
      diagnostics.residual_cache_insert_sec;
    const std::chrono::steady_clock::time_point residual_start =
      std::chrono::steady_clock::now();
    if (scheduler == "layer") {
      const std::chrono::steady_clock::time_point collect_start =
        std::chrono::steady_clock::now();
      residual_requests = collect_unique_residual_requests(
        plan, data.nrow(), data.ncol(), residual_cache.backend().name,
        residual_cache.backend().params, residual_cache.used_device());
      diagnostics.residual_request_collect_sec += seconds_since(collect_start);
      plan.unique_residual_requests = static_cast<int>(residual_requests.size());
      residual_batches = residual_cache.prefetch_level(
        data, residual_requests, ord, options.residual_batch_size, &diagnostics);
    }
    const double residual_prefetch_elapsed_sec = seconds_since(residual_start);
    const double residual_prefetch_accounted_sec =
      diagnostics.residual_request_collect_sec - residual_request_collect_before +
      diagnostics.residual_prefetch_missing_scan_sec - residual_missing_scan_before +
      diagnostics.residual_prefetch_batch_input_sec - residual_batch_input_before +
      diagnostics.residual_batch_call_wall_sec - residual_batch_call_before +
      diagnostics.residual_diagnostic_merge_sec - residual_diagnostic_merge_before +
      diagnostics.residual_cache_insert_sec - residual_cache_insert_before;
    diagnostics.residual_prefetch_unaccounted_sec +=
      nonnegative_gap(residual_prefetch_elapsed_sec,
                      residual_prefetch_accounted_sec);

    int dcov_batches = 0;
    std::vector<double> pvalues;
    const double ci_host_pack_before = diagnostics.ci_host_pack_sec;
    const double ci_dcov_call_before = diagnostics.ci_dcov_call_wall_sec;
    const double ci_pvalue_copy_before = diagnostics.ci_pvalue_copy_sec;
    const double ci_diagnostic_append_before =
      diagnostics.ci_diagnostic_append_sec;
    const std::chrono::steady_clock::time_point ci_eval_start =
      std::chrono::steady_clock::now();
    if (ci_method == CiMethodKind::DccGamma) {
      pvalues = evaluate_tasks_cuda(
        data, plan.tasks, batch_size, options.index, options.legacy_index,
        ord, &residual_cache, dcov_workspace.value, &diagnostics,
        &dcov_batches);
    } else {
      pvalues = evaluate_tasks_hsic_cuda(
        data, plan.tasks, batch_size, ci_method, options.hsic_options,
        ord, &residual_cache, &diagnostics, &hsic_counters);
    }
    const double ci_eval_elapsed_sec = seconds_since(ci_eval_start);
    const double ci_eval_accounted_sec =
      diagnostics.ci_host_pack_sec - ci_host_pack_before +
      diagnostics.ci_dcov_call_wall_sec - ci_dcov_call_before +
      diagnostics.ci_pvalue_copy_sec - ci_pvalue_copy_before +
      diagnostics.ci_diagnostic_append_sec - ci_diagnostic_append_before;
    diagnostics.ci_eval_unaccounted_sec +=
      nonnegative_gap(ci_eval_elapsed_sec, ci_eval_accounted_sec);

    const ResidualCacheStats after_stats = residual_cache.stats();
    if (scheduler != "layer") {
      plan.unique_residual_requests = after_stats.computations - before_stats.computations;
      residual_batches = plan.unique_residual_requests;
    }

    std::vector<int> delete_edges(static_cast<std::size_t>(p) * p, 0);
    int level_tests = 0;
    std::vector<LevelDeletion> level_log;
    const std::chrono::steady_clock::time_point replay_start =
      std::chrono::steady_clock::now();
    replay_layer_pvalues(plan, pvalues, options, p, &delete_edges, &result,
                         &level_tests, &level_log);
    const double replay_elapsed_sec = seconds_since(replay_start);

    for (int i = 0; i < p * p; ++i) {
      if (delete_edges[i] != 0) result.adjacency[i] = 0;
    }
    result.n_edge_tests.push_back(level_tests);
    result.per_level_log.push_back(level_log);

    const int task_count = static_cast<int>(plan.tasks.size());
    const int ignored = task_count - level_tests;
    const double total_elapsed_sec = seconds_since(level_start);
    diagnostics.per_level.push_back(LayerDiagnosticsLevel{
      ord, task_count, task_count, level_tests, ignored,
      static_cast<int>(level_log.size()), plan.unconditional_tasks,
      plan.conditional_tasks, plan.unique_residual_requests,
      dcov_batches, residual_batches, plan_elapsed_sec,
      residual_prefetch_elapsed_sec, ci_eval_elapsed_sec, replay_elapsed_sec,
      total_elapsed_sec});
    diagnostics.levels = static_cast<int>(diagnostics.per_level.size());
    diagnostics.tasks_planned += task_count;
    diagnostics.tasks_evaluated += task_count;
    diagnostics.tests_replayed += level_tests;
    diagnostics.dcov_batches += dcov_batches;
    diagnostics.unique_residual_requests += plan.unique_residual_requests;
    diagnostics.residual_batches += residual_batches;
    diagnostics.max_level_tasks = std::max(diagnostics.max_level_tasks, task_count);
    diagnostics.max_level_unique_residuals =
      std::max(diagnostics.max_level_unique_residuals,
               plan.unique_residual_requests);
    diagnostics.plan_elapsed_sec += plan_elapsed_sec;
    diagnostics.residual_prefetch_elapsed_sec += residual_prefetch_elapsed_sec;
    diagnostics.ci_eval_elapsed_sec += ci_eval_elapsed_sec;
    diagnostics.replay_elapsed_sec += replay_elapsed_sec;
    diagnostics.total_elapsed_sec += total_elapsed_sec;
  }

  const ResidualCacheStats stats = residual_cache.stats();
  finalize_scheduler_diagnostics(&diagnostics, stats, result);
  result.scheduler_diagnostics = diagnostics;
  if (ci_method == CiMethodKind::DccGamma) {
    result.ci_dcc_gamma_tests = diagnostics.tests_replayed;
  } else if (ci_method == CiMethodKind::HsicGamma) {
    result.ci_hsic_gamma_tests = diagnostics.tests_replayed;
    result.ci_hsic_gamma_cuda_tests = hsic_counters.gamma_tests;
  } else {
    result.ci_hsic_perm_tests = diagnostics.tests_replayed;
    result.ci_hsic_perm_cuda_tests = hsic_counters.perm_tests;
    result.ci_hsic_permutation_replicates =
      hsic_counters.permutation_replicates;
  }
  result.ci_hsic_cuda_batches = hsic_counters.batches;
  result.ci_hsic_cuda_pairs = hsic_counters.pairs;
  result.ci_hsic_cuda_memory_bytes = hsic_counters.memory_bytes;
  result.ci_hsic_cuda_max_n = hsic_counters.max_n;
  result.ci_hsic_cuda_max_batch_pairs = hsic_counters.max_batch_pairs;

  result.residual_cache_enabled = stats.enabled;
  result.residual_cache_requests = stats.requests;
  result.residual_cache_hits = stats.hits;
  result.residual_cache_misses = stats.misses;
  result.residual_cache_computations = stats.computations;
  result.residual_cache_stored_vectors = stats.stored_vectors;
  result.residual_cache_stored_values = stats.stored_values;
  result.residual_backend = stats.backend_name;
  result.residual_backend_params =
    make_residual_backend_config(residual_backend_name, options.fastspline_params).params;
  result.residual_device = residual_cache.used_device();
  result.residual_device_requested = residual_cache.requested_device();
  result.residual_device_reason = residual_cache.reason();
  return result;
}

std::string resolve_scheduler(const SkeletonOptions& options) {
  const std::string requested =
    options.scheduler_requested.empty() ? "legacy" : options.scheduler_requested;
  if (requested == "auto") return "layer";
  if (requested == "layer" || requested == "legacy") return requested;
  throw std::runtime_error("Unknown scheduler: " + requested);
}

SkeletonResult run_skeleton_cuda_cpu_fallback(const Rcpp::NumericMatrix& data,
                                              const SkeletonOptions& options,
                                              int batch_size,
                                              CiMethodKind kind,
                                              const std::string& reason) {
  SkeletonOptions cpu_options = options;
  cpu_options.residual_device_requested = "cpu";
  cpu_options.scheduler_requested =
    options.scheduler_requested.empty() ? "legacy" : options.scheduler_requested;
  SkeletonResult result = run_skeleton_exact(data, cpu_options);
  result.residual_device = "cpu";
  result.residual_device_requested =
    options.residual_device_requested.empty() ? "cpu" : options.residual_device_requested;
  result.scheduler = resolve_scheduler(options);
  result.scheduler_requested = cpu_options.scheduler_requested;
  result.scheduler_diagnostics = make_scheduler_diagnostics(
    result.scheduler, result.scheduler_requested, batch_size,
    options.residual_batch_size);
  result.ci_backend = "native-cpu";
  result.ci_backend_reason = reason;
  result.ci_hsic_cuda_max_n = options.hsic_options.cuda_max_n;
  result.ci_hsic_cuda_max_batch_pairs =
    options.hsic_options.cuda_max_batch_pairs;
  if (kind == CiMethodKind::HsicGamma) {
    result.ci_hsic_cuda_fallback_tests = result.ci_hsic_gamma_tests;
  } else if (kind == CiMethodKind::HsicPermutation) {
    result.ci_hsic_cuda_fallback_tests = result.ci_hsic_perm_tests;
  }
  return result;
}

}  // namespace

SkeletonResult run_skeleton_cuda_batch(const Rcpp::NumericMatrix& data,
                                       const SkeletonOptions& options,
                                       int batch_size) {
  const CiMethodKind kind = parse_ci_method_kind(options.ci_method);
  if (kind == CiMethodKind::HsicPermutation &&
      !options.hsic_options.has_seed) {
    return run_skeleton_cuda_cpu_fallback(
      data, options, batch_size, kind,
      "CUDA HSIC permutation requires explicit seed in this stage");
  }
  if (kind != CiMethodKind::DccGamma) {
    std::string reason;
    if (!hsic_cuda_available(&reason)) {
      if (!options.hsic_options.cuda_memory_fallback) {
        throw std::runtime_error(
          reason.empty() ? "CUDA HSIC backend is unavailable" : reason);
      }
      return run_skeleton_cuda_cpu_fallback(
        data, options, batch_size, kind,
        reason.empty() ? "CUDA HSIC backend is unavailable" : reason);
    }
    try {
      return run_skeleton_cuda_impl(data, options, batch_size,
                                    resolve_scheduler(options));
    } catch (const std::exception& ex) {
      if (!options.hsic_options.cuda_memory_fallback) throw;
      return run_skeleton_cuda_cpu_fallback(
        data, options, batch_size, kind, ex.what());
    }
  }
  return run_skeleton_cuda_impl(data, options, batch_size,
                                resolve_scheduler(options));
}
