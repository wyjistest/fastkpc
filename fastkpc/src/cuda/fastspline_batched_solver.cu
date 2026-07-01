#include "fastspline_batched_solver.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBlock = 256;
constexpr int kMaxTrueBatchedDesignCols = 128;
constexpr int kSmallPRhsMaxDesignCols = 64;

struct FastSplineScoreMetadata {
  int info;
  int pad0;
  double rss;
  double edf;
  double pad1;
};

struct FastSplineSelectedFactorDescriptor {
  double lambda;
  double ridge;
  int design_index;
  int pad;
};

struct FastSplineSelectedFitDescriptor {
  int factor_index;
  int active;
};

template <typename T>
struct DeviceArena {
  T* base = nullptr;
  std::size_t capacity = 0;
  std::size_t used = 0;
};

struct TrueBatchArenaCounts {
  std::size_t double_count = 0;
  std::size_t int_count = 0;
  std::size_t ptr_count = 0;
  std::size_t score_metadata_count = 0;
};

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

double nonnegative_gap(double total, double accounted) {
  return std::max(0.0, total - accounted);
}

void add_d2h_timing(FastSplineCudaBatchDiagnostics* timing,
                    double elapsed_sec,
                    std::size_t bytes,
                    bool residuals,
                    bool metadata) {
  timing->d2h_sec += elapsed_sec;
  timing->d2h_copy_count += 1;
  timing->d2h_bytes += static_cast<double>(bytes);
  if (residuals) {
    timing->d2h_residuals_sec += elapsed_sec;
    timing->d2h_residual_bytes += static_cast<double>(bytes);
  } else if (metadata) {
    timing->d2h_metadata_sec += elapsed_sec;
    timing->d2h_metadata_bytes += static_cast<double>(bytes);
  } else {
    timing->d2h_info_sec += elapsed_sec;
  }
}

enum class H2dTransferKind {
  Design,
  Penalty,
  Y,
  Index,
  Lambda,
  Active
};

void add_h2d_timing(FastSplineCudaBatchDiagnostics* timing,
                    double elapsed_sec,
                    std::size_t bytes,
                    H2dTransferKind kind) {
  timing->h2d_sec += elapsed_sec;
  timing->h2d_copy_count += 1;
  timing->h2d_bytes += static_cast<double>(bytes);
  switch (kind) {
    case H2dTransferKind::Design:
      timing->h2d_design_sec += elapsed_sec;
      timing->h2d_design_bytes += static_cast<double>(bytes);
      break;
    case H2dTransferKind::Penalty:
      timing->h2d_penalty_sec += elapsed_sec;
      timing->h2d_design_bytes += static_cast<double>(bytes);
      break;
    case H2dTransferKind::Y:
      timing->h2d_y_sec += elapsed_sec;
      timing->h2d_y_bytes += static_cast<double>(bytes);
      break;
    case H2dTransferKind::Index:
      timing->h2d_index_sec += elapsed_sec;
      timing->h2d_metadata_bytes += static_cast<double>(bytes);
      break;
    case H2dTransferKind::Lambda:
      timing->h2d_lambda_sec += elapsed_sec;
      timing->h2d_metadata_bytes += static_cast<double>(bytes);
      break;
    case H2dTransferKind::Active:
      timing->h2d_active_sec += elapsed_sec;
      timing->h2d_metadata_bytes += static_cast<double>(bytes);
      break;
  }
}

void add_batch_timing(FastSplineCudaBatchDiagnostics* out,
                      const FastSplineCudaBatchDiagnostics& value) {
  out->grouping_sec += value.grouping_sec;
  out->grouping_condition_key_sec += value.grouping_condition_key_sec;
  out->grouping_group_key_sec += value.grouping_group_key_sec;
  out->grouping_design_build_sec += value.grouping_design_build_sec;
  out->grouping_map_insert_sec += value.grouping_map_insert_sec;
  out->grouping_unaccounted_sec += value.grouping_unaccounted_sec;
  out->grouping_group_count += value.grouping_group_count;
  out->grouping_design_count += value.grouping_design_count;
  out->grouping_condition_key_sort_count +=
    value.grouping_condition_key_sort_count;
  out->grouping_string_key_count += value.grouping_string_key_count;
  out->design_cache_hit_count += value.design_cache_hit_count;
  out->design_cache_miss_count += value.design_cache_miss_count;
  out->design_cache_insert_count += value.design_cache_insert_count;
  out->design_cache_entries =
    std::max(out->design_cache_entries, value.design_cache_entries);
  out->design_build_total_sec += value.design_build_total_sec;
  out->design_build_basis_sec += value.design_build_basis_sec;
  out->design_build_penalty_sec += value.design_build_penalty_sec;
  out->design_build_x_pack_sec += value.design_build_x_pack_sec;
  out->design_build_p_pack_sec += value.design_build_p_pack_sec;
  out->design_build_alloc_sec += value.design_build_alloc_sec;
  out->design_build_column_extract_sec +=
    value.design_build_column_extract_sec;
  out->design_build_unaccounted_sec += value.design_build_unaccounted_sec;
  out->design_build_count += value.design_build_count;
  out->design_build_x_values += value.design_build_x_values;
  out->design_build_p_values += value.design_build_p_values;
  out->design_build_basis_values += value.design_build_basis_values;
  out->design_build_penalty_values += value.design_build_penalty_values;
  out->design_build_condition_cols += value.design_build_condition_cols;
  out->host_pack_sec += value.host_pack_sec;
  out->alloc_sec += value.alloc_sec;
  out->h2d_sec += value.h2d_sec;
  out->h2d_design_sec += value.h2d_design_sec;
  out->h2d_penalty_sec += value.h2d_penalty_sec;
  out->h2d_y_sec += value.h2d_y_sec;
  out->h2d_index_sec += value.h2d_index_sec;
  out->h2d_lambda_sec += value.h2d_lambda_sec;
  out->h2d_active_sec += value.h2d_active_sec;
  out->h2d_copy_count += value.h2d_copy_count;
  out->h2d_bytes += value.h2d_bytes;
  out->h2d_design_bytes += value.h2d_design_bytes;
  out->h2d_y_bytes += value.h2d_y_bytes;
  out->h2d_metadata_bytes += value.h2d_metadata_bytes;
  out->h2d_metadata_coalesced_count += value.h2d_metadata_coalesced_count;
  out->h2d_metadata_coalesced_bytes += value.h2d_metadata_coalesced_bytes;
  out->h2d_selected_metadata_copy_count +=
    value.h2d_selected_metadata_copy_count;
  out->xtx_xty_sec += value.xtx_xty_sec;
  out->pointer_setup_sec += value.pointer_setup_sec;
  out->active_copy_sec += value.active_copy_sec;
  out->build_system_sec += value.build_system_sec;
  out->factor_solve_sec += value.factor_solve_sec;
  out->factor_cholesky_sec += value.factor_cholesky_sec;
  out->factor_rhs_solve_sec += value.factor_rhs_solve_sec;
  out->factor_inverse_solve_sec += value.factor_inverse_solve_sec;
  out->residual_summary_sec += value.residual_summary_sec;
  out->d2h_sec += value.d2h_sec;
  out->d2h_residuals_sec += value.d2h_residuals_sec;
  out->d2h_metadata_sec += value.d2h_metadata_sec;
  out->d2h_info_sec += value.d2h_info_sec;
  out->d2h_copy_count += value.d2h_copy_count;
  out->d2h_bytes += value.d2h_bytes;
  out->d2h_residual_bytes += value.d2h_residual_bytes;
  out->d2h_metadata_bytes += value.d2h_metadata_bytes;
  out->d2h_metadata_coalesced_count += value.d2h_metadata_coalesced_count;
  out->d2h_metadata_coalesced_bytes += value.d2h_metadata_coalesced_bytes;
  out->host_select_sec += value.host_select_sec;
  out->free_sec += value.free_sec;
  out->true_batch_total_sec += value.true_batch_total_sec;
  out->factorization_count += value.factorization_count;
  out->rhs_solve_count += value.rhs_solve_count;
  out->inverse_solve_count += value.inverse_solve_count;
  out->rhs_solve_api_calls += value.rhs_solve_api_calls;
  out->rhs_target_solves += value.rhs_target_solves;
  out->rhs_custom_solve_count += value.rhs_custom_solve_count;
  out->rhs_cublas_solve_count += value.rhs_cublas_solve_count;
  out->rhs_solve_fallback_count += value.rhs_solve_fallback_count;
  out->rhs_custom_solve_sec += value.rhs_custom_solve_sec;
  out->rhs_cublas_solve_sec += value.rhs_cublas_solve_sec;
  out->candidate_rhs_fused_solve_count +=
    value.candidate_rhs_fused_solve_count;
  out->candidate_rhs_materialized_solve_count +=
    value.candidate_rhs_materialized_solve_count;
  out->selected_rhs_materialized_solve_count +=
    value.selected_rhs_materialized_solve_count;
  out->candidate_beta_values_avoided += value.candidate_beta_values_avoided;
  out->summary_candidate_launch_count += value.summary_candidate_launch_count;
  out->summary_group_batched_launch_count +=
    value.summary_group_batched_launch_count;
  out->summary_group_batched_candidate_count +=
    value.summary_group_batched_candidate_count;
  out->winning_factor_reuse_count += value.winning_factor_reuse_count;
  out->factor_cache_hits += value.factor_cache_hits;
  out->factor_cache_misses += value.factor_cache_misses;
  out->factor_cache_entries += value.factor_cache_entries;
  out->factor_cache_bytes += value.factor_cache_bytes;
  out->lambda_candidates = std::max(out->lambda_candidates,
                                    value.lambda_candidates);
  out->workspace_reuse_count += value.workspace_reuse_count;
  out->workspace_grow_count += value.workspace_grow_count;
  out->workspace_slab_grow_count += value.workspace_slab_grow_count;
  out->workspace_slab_reuse_count += value.workspace_slab_reuse_count;
  out->workspace_slab_bytes = std::max(out->workspace_slab_bytes,
                                       value.workspace_slab_bytes);
  out->workspace_legacy_alloc_count += value.workspace_legacy_alloc_count;
  out->solver_handle_create_count += value.solver_handle_create_count;
  out->per_request_design_x_values += value.per_request_design_x_values;
  out->duplicate_design_x_values_avoided +=
    value.duplicate_design_x_values_avoided;
  out->algebraic_rss_count += value.algebraic_rss_count;
  out->candidate_residual_materialize_count +=
    value.candidate_residual_materialize_count;
  out->winning_residual_materialize_count +=
    value.winning_residual_materialize_count;
  out->algebraic_rss_clamp_count += value.algebraic_rss_clamp_count;
  out->residual_only_fit_count += value.residual_only_fit_count;
  out->residual_full_fit_materialize_count +=
    value.residual_full_fit_materialize_count;
  out->residual_fitted_values_avoided +=
    value.residual_fitted_values_avoided;
  out->residual_result_materialize_sec +=
    value.residual_result_materialize_sec;
  out->residual_fitted_materialize_sec +=
    value.residual_fitted_materialize_sec;
}

__device__ __host__ inline std::size_t matrix_offset(int fit,
                                                     int row,
                                                     int col,
                                                     int rows,
                                                     int cols) {
  return (static_cast<std::size_t>(fit) * rows * cols) +
         (static_cast<std::size_t>(row) * cols) + col;
}

__device__ __host__ inline std::size_t colmajor_square_offset(int fit,
                                                              int row,
                                                              int col,
                                                              int p) {
  return (static_cast<std::size_t>(fit) * p * p) +
         (static_cast<std::size_t>(col) * p) + row;
}

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
  }
}

void check_cublas(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(stage) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void check_cusolver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(stage) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

std::vector<double> response_vector(const Rcpp::NumericMatrix& data, int target) {
  if (target < 0 || target >= data.ncol()) {
    throw std::runtime_error("target column out of range");
  }
  std::vector<double> y(data.nrow());
  for (int row = 0; row < data.nrow(); ++row) y[row] = data(row, target);
  return y;
}

std::string group_key(const FastSplineDesign& design,
                      const FastSplineParams& params) {
  std::ostringstream out;
  out << std::setprecision(17)
      << design.n << "|"
      << design.p << "|"
      << params.degree << "|"
      << params.knots << "|"
      << params.lambda_min << "|"
      << params.lambda_max << "|"
      << params.lambda_count << "|"
      << params.ridge << "|"
      << params.mode;
  return out.str();
}

std::string conditioning_set_key(std::vector<int> conditioning_set) {
  std::sort(conditioning_set.begin(), conditioning_set.end());
  std::ostringstream out;
  for (std::size_t i = 0; i < conditioning_set.size(); ++i) {
    if (i != 0) out << ",";
    out << conditioning_set[i];
  }
  return out.str();
}

FastSplineCudaDiagnostics make_diagnostics(bool cuda_used,
                                           bool fallback_used,
                                           const std::string& reason,
                                           int group_id,
                                           int batch_position,
                                           bool true_batched,
                                           const std::string& backend) {
  FastSplineCudaDiagnostics out;
  out.cuda_used = cuda_used;
  out.fallback_used = fallback_used;
  out.reason = reason;
  out.batch_group_id = group_id;
  out.batch_position = batch_position;
  out.true_batched = true_batched;
  out.cholesky_backend = backend;
  return out;
}

FastSplineCudaFit cpu_fallback_fit(const Rcpp::NumericMatrix& data,
                                   const FastSplineBatchRequest& request,
                                   const FastSplineParams& params,
                                   const std::string& reason,
                                   int group_id,
                                   int batch_position) {
  FastSplineCudaFit out;
  out.fit = fit_fastspline_residuals(data, request.target,
                                     request.conditioning_set, params);
  out.diagnostics = make_diagnostics(false, true, reason, group_id,
                                     batch_position, false, "cpu-fallback");
  return out;
}

std::vector<double> pack_group_design_x(const FastSplineBatchGroup& group) {
  const int design_count = static_cast<int>(group.designs.size());
  const int n = group.n;
  const int p = group.design_cols;
  std::vector<double> out(static_cast<std::size_t>(design_count) * n * p);
  for (int design_index = 0; design_index < design_count; ++design_index) {
    const FastSplineDesign& design = group.designs[design_index];
    if (design.n != n || design.p != p) {
      throw std::runtime_error("fastSpline design batch X shape mismatch");
    }
    std::copy(design.X.begin(), design.X.end(),
              out.begin() + static_cast<std::size_t>(design_index) * n * p);
  }
  return out;
}

std::vector<double> pack_group_design_p(const FastSplineBatchGroup& group) {
  const int design_count = static_cast<int>(group.designs.size());
  const int p = group.design_cols;
  std::vector<double> out(static_cast<std::size_t>(design_count) * p * p);
  for (int design_index = 0; design_index < design_count; ++design_index) {
    const FastSplineDesign& design = group.designs[design_index];
    if (design.p != p || static_cast<int>(design.P.size()) != p * p) {
      throw std::runtime_error("fastSpline design batch penalty shape mismatch");
    }
    std::copy(design.P.begin(), design.P.end(),
              out.begin() + static_cast<std::size_t>(design_index) * p * p);
  }
  return out;
}

std::vector<double> pack_group_y(const Rcpp::NumericMatrix& data,
                                 const FastSplineBatchGroup& group) {
  const int group_size = static_cast<int>(group.requests.size());
  const int n = group.n;
  std::vector<double> out(static_cast<std::size_t>(group_size) * n);
  for (int fit = 0; fit < group_size; ++fit) {
    const std::vector<double> y = response_vector(data, group.requests[fit].target);
    std::copy(y.begin(), y.end(), out.begin() + static_cast<std::size_t>(fit) * n);
  }
  return out;
}

std::vector<int> pack_group_request_design_index(
  const FastSplineBatchGroup& group) {
  const int group_size = static_cast<int>(group.requests.size());
  const int design_count = static_cast<int>(group.designs.size());
  std::vector<int> out(group_size);
  for (int fit = 0; fit < group_size; ++fit) {
    const int design_index = group.requests[fit].design_index;
    if (design_index < 0 || design_index >= design_count) {
      throw std::runtime_error("fastSpline request design index out of range");
    }
    out[fit] = design_index;
  }
  return out;
}

bool finite_vec(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

int max_fits_per_design(const FastSplineBatchGroup& group) {
  if (group.designs.empty()) return 0;
  std::vector<int> counts(group.designs.size(), 0);
  int max_count = 0;
  for (const FastSplineBatchRequest& request : group.requests) {
    if (request.design_index < 0 ||
        request.design_index >= static_cast<int>(counts.size())) {
      throw std::runtime_error("fastSpline batch design index out of range");
    }
    ++counts[request.design_index];
    max_count = std::max(max_count, counts[request.design_index]);
  }
  return max_count;
}

__global__ void batched_xtx_kernel(const double* X,
                                   int group_size,
                                   int n,
                                   int p,
                                   double* XtX) {
  __shared__ double scratch[kBlock];
  const int a = blockIdx.x;
  const int b = blockIdx.y;
  const int fit = blockIdx.z;
  if (fit >= group_size || a >= p || b >= p) return;

  double acc = 0.0;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    acc += X[matrix_offset(fit, row, a, n, p)] *
           X[matrix_offset(fit, row, b, n, p)];
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    XtX[colmajor_square_offset(fit, a, b, p)] = scratch[0];
  }
}

__global__ void batched_xty_by_design_kernel(const double* design_X,
                                             const double* y,
                                             const int* design_index,
                                             int group_size,
                                             int n,
                                             int p,
                                             double* Xty) {
  __shared__ double scratch[kBlock];
  const int col = blockIdx.x;
  const int fit = blockIdx.y;
  if (fit >= group_size || col >= p) return;

  const int design = design_index[fit];
  double acc = 0.0;
  const std::size_t y_base = static_cast<std::size_t>(fit) * n;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    acc += design_X[matrix_offset(design, row, col, n, p)] *
           y[y_base + row];
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    Xty[static_cast<std::size_t>(fit) * p + col] = scratch[0];
  }
}

__global__ void batched_build_lambda_design_system_kernel(
  const double* XtX,
  const double* P,
  const double* lambdas,
  int design_count,
  int lambda_count,
  int p,
  double ridge,
  double* A) {
  const int total = design_count * lambda_count * p * p;
  for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += gridDim.x * blockDim.x) {
    const int pp = p * p;
    const int factor = linear / pp;
    const int within = linear - factor * pp;
    const int design = factor % design_count;
    const int lambda_index = factor / design_count;
    const int row = within % p;
    const int col = within / p;
    const std::size_t out_idx = colmajor_square_offset(factor, row, col, p);
    double value = XtX[colmajor_square_offset(design, row, col, p)] +
      lambdas[lambda_index] *
        P[static_cast<std::size_t>(design) * pp +
          static_cast<std::size_t>(row) * p + col];
    if (row == col && row > 0) value += ridge;
    A[out_idx] = value;
  }
}

__global__ void batched_build_selected_factor_system_kernel(
  const double* XtX,
  const double* P,
  const FastSplineSelectedFactorDescriptor* factors,
  int factor_count,
  int p,
  double* A) {
  const int total = factor_count * p * p;
  for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += gridDim.x * blockDim.x) {
    const int pp = p * p;
    const int factor = linear / pp;
    const int within = linear - factor * pp;
    const int row = within % p;
    const int col = within / p;
    const FastSplineSelectedFactorDescriptor descriptor = factors[factor];
    const int design = descriptor.design_index;
    const std::size_t out_idx = colmajor_square_offset(factor, row, col, p);
    double value = XtX[colmajor_square_offset(design, row, col, p)] +
      descriptor.lambda * P[static_cast<std::size_t>(design) * pp +
                            static_cast<std::size_t>(row) * p + col];
    if (row == col && row > 0) value += descriptor.ridge;
    A[out_idx] = value;
  }
}

__global__ void make_matrix_pointer_array(double* base,
                                          int group_size,
                                          int p,
                                          double** ptrs) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= group_size) return;
  ptrs[fit] = base + static_cast<std::size_t>(fit) * p * p;
}

__global__ void make_vector_pointer_array(double* base,
                                          int group_size,
                                          int p,
                                          double** ptrs) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= group_size) return;
  ptrs[fit] = base + static_cast<std::size_t>(fit) * p;
}

__global__ void batched_copy_xty_to_beta_kernel(const double* Xty,
                                                int group_size,
                                                int p,
                                                double* beta) {
  const int total = group_size * p;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    beta[idx] = Xty[idx];
  }
}

__global__ void small_p_cholesky_rhs_solve_kernel(
  const double* factors,
  const double* Xty,
  const int* factor_index,
  const int* active,
  int group_size,
  int p,
  double* beta,
  int* info) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= group_size) return;
  if (active != nullptr && active[fit] == 0) {
    if (info != nullptr) info[fit] = 0;
    for (int row = 0; row < p; ++row) {
      beta[static_cast<std::size_t>(fit) * p + row] = 0.0;
    }
    return;
  }

  double work[kSmallPRhsMaxDesignCols];
  const int factor = factor_index == nullptr ? fit : factor_index[fit];
  const std::size_t beta_base = static_cast<std::size_t>(fit) * p;
  const std::size_t factor_base = static_cast<std::size_t>(factor) * p * p;
  for (int row = 0; row < p; ++row) {
    work[row] = Xty[beta_base + row];
  }

  int solve_info = 0;
  for (int row = 0; row < p; ++row) {
    double value = work[row];
    for (int col = 0; col < row; ++col) {
      value -= factors[factor_base + static_cast<std::size_t>(row) * p + col] *
        work[col];
    }
    const double diag =
      factors[factor_base + static_cast<std::size_t>(row) * p + row];
    if (!isfinite(diag) || fabs(diag) <= 1e-14) {
      solve_info = row + 1;
      break;
    }
    work[row] = value / diag;
  }

  if (solve_info == 0) {
    for (int row = p - 1; row >= 0; --row) {
      double value = work[row];
      for (int col = row + 1; col < p; ++col) {
        value -= factors[factor_base + static_cast<std::size_t>(col) * p + row] *
          work[col];
      }
      const double diag =
        factors[factor_base + static_cast<std::size_t>(row) * p + row];
      if (!isfinite(diag) || fabs(diag) <= 1e-14) {
        solve_info = row + 1;
        break;
      }
      work[row] = value / diag;
    }
  }

  if (solve_info != 0) {
    if (info != nullptr) info[fit] = solve_info;
    return;
  }

  for (int row = 0; row < p; ++row) {
    beta[beta_base + row] = work[row];
  }
  if (info != nullptr) info[fit] = 0;
}

__global__ void small_p_cholesky_rhs_solve_selected_kernel(
  const double* factors,
  const double* Xty,
  const FastSplineSelectedFitDescriptor* selected_fits,
  int group_size,
  int p,
  double* beta,
  int* info) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= group_size) return;
  if (selected_fits != nullptr && selected_fits[fit].active == 0) {
    if (info != nullptr) info[fit] = 0;
    for (int row = 0; row < p; ++row) {
      beta[static_cast<std::size_t>(fit) * p + row] = 0.0;
    }
    return;
  }

  double work[kSmallPRhsMaxDesignCols];
  const int factor = selected_fits == nullptr ? fit :
    selected_fits[fit].factor_index;
  const std::size_t beta_base = static_cast<std::size_t>(fit) * p;
  const std::size_t factor_base = static_cast<std::size_t>(factor) * p * p;
  for (int row = 0; row < p; ++row) {
    work[row] = Xty[beta_base + row];
  }

  int solve_info = 0;
  for (int row = 0; row < p; ++row) {
    double value = work[row];
    for (int col = 0; col < row; ++col) {
      value -= factors[factor_base + static_cast<std::size_t>(row) * p + col] *
        work[col];
    }
    const double diag =
      factors[factor_base + static_cast<std::size_t>(row) * p + row];
    if (!isfinite(diag) || fabs(diag) <= 1e-14) {
      solve_info = row + 1;
      break;
    }
    work[row] = value / diag;
  }

  if (solve_info == 0) {
    for (int row = p - 1; row >= 0; --row) {
      double value = work[row];
      for (int col = row + 1; col < p; ++col) {
        value -= factors[factor_base + static_cast<std::size_t>(col) * p + row] *
          work[col];
      }
      const double diag =
        factors[factor_base + static_cast<std::size_t>(row) * p + row];
      if (!isfinite(diag) || fabs(diag) <= 1e-14) {
        solve_info = row + 1;
        break;
      }
      work[row] = value / diag;
    }
  }

  if (solve_info != 0) {
    if (info != nullptr) info[fit] = solve_info;
    return;
  }

  for (int row = 0; row < p; ++row) {
    beta[beta_base + row] = work[row];
  }
  if (info != nullptr) info[fit] = 0;
}

__device__ int solve_small_p_cholesky_rhs_local(const double* factors,
                                                const double* Xty,
                                                int factor,
                                                int fit,
                                                int p,
                                                double* beta) {
  const std::size_t beta_base = static_cast<std::size_t>(fit) * p;
  const std::size_t factor_base = static_cast<std::size_t>(factor) * p * p;
  for (int row = 0; row < p; ++row) {
    beta[row] = Xty[beta_base + row];
  }

  for (int row = 0; row < p; ++row) {
    double value = beta[row];
    for (int col = 0; col < row; ++col) {
      value -= factors[factor_base + static_cast<std::size_t>(row) * p + col] *
        beta[col];
    }
    const double diag =
      factors[factor_base + static_cast<std::size_t>(row) * p + row];
    if (!isfinite(diag) || fabs(diag) <= 1e-14) {
      return row + 1;
    }
    beta[row] = value / diag;
  }

  for (int row = p - 1; row >= 0; --row) {
    double value = beta[row];
    for (int col = row + 1; col < p; ++col) {
      value -= factors[factor_base + static_cast<std::size_t>(col) * p + row] *
        beta[col];
    }
    const double diag =
      factors[factor_base + static_cast<std::size_t>(row) * p + row];
    if (!isfinite(diag) || fabs(diag) <= 1e-14) {
      return row + 1;
    }
    beta[row] = value / diag;
  }

  return 0;
}

__global__ void make_request_lambda_matrix_pointer_array(
  double* base,
  const int* design_index,
  int group_size,
  int design_count,
  int lambda_index,
  int p,
  double** ptrs) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= group_size) return;
  const int design = design_index[fit];
  const int factor = lambda_index * design_count + design;
  ptrs[fit] = base + static_cast<std::size_t>(factor) * p * p;
}

__global__ void make_selected_matrix_pointer_array(
  double* base,
  const FastSplineSelectedFitDescriptor* fits,
  int count,
  int p,
  double** ptrs) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= count) return;
  const int matrix = fits[fit].factor_index;
  ptrs[fit] = base + static_cast<std::size_t>(matrix) * p * p;
}

__global__ void batched_identity_kernel(double* matrix,
                                        int group_size,
                                        int p) {
  const int total = group_size * p * p;
  for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total;
       linear += gridDim.x * blockDim.x) {
    const int pp = p * p;
    const int within = linear % pp;
    const int row = within % p;
    const int col = within / p;
    matrix[linear] = row == col ? 1.0 : 0.0;
  }
}

__global__ void batched_fitted_residual_by_design_kernel(
  const double* design_X,
  const double* y,
  const double* beta,
  const int* design_index,
  const int* active,
  int group_size,
  int n,
  int p,
  double* fitted,
  double* residuals) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int fit = blockIdx.y;
  if (fit >= group_size || row >= n) return;
  const std::size_t out_idx = static_cast<std::size_t>(fit) * n + row;
  if (active != nullptr && active[fit] == 0) {
    fitted[out_idx] = 0.0;
    residuals[out_idx] = 0.0;
    return;
  }
  const int design = design_index[fit];
  double value = 0.0;
  for (int col = 0; col < p; ++col) {
    value += design_X[matrix_offset(design, row, col, n, p)] *
             beta[static_cast<std::size_t>(fit) * p + col];
  }
  fitted[out_idx] = value;
  residuals[out_idx] = y[out_idx] - value;
}

__global__ void batched_residual_by_design_kernel(
  const double* design_X,
  const double* y,
  const double* beta,
  const int* design_index,
  const int* active,
  int group_size,
  int n,
  int p,
  double* residuals) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int fit = blockIdx.y;
  if (fit >= group_size || row >= n) return;
  const std::size_t out_idx = static_cast<std::size_t>(fit) * n + row;
  if (active != nullptr && active[fit] == 0) {
    residuals[out_idx] = 0.0;
    return;
  }
  const int design = design_index[fit];
  double value = 0.0;
  for (int col = 0; col < p; ++col) {
    value += design_X[matrix_offset(design, row, col, n, p)] *
             beta[static_cast<std::size_t>(fit) * p + col];
  }
  residuals[out_idx] = y[out_idx] - value;
}

__global__ void batched_y_norm2_kernel(const double* y,
                                       int group_size,
                                       int n,
                                       double* y_norm2) {
  __shared__ double scratch[kBlock];
  const int fit = blockIdx.x;
  if (fit >= group_size) return;

  double acc = 0.0;
  const std::size_t y_base = static_cast<std::size_t>(fit) * n;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    const double value = y[y_base + row];
    acc += value * value;
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) y_norm2[fit] = scratch[0];
}

__global__ void batched_algebraic_rss_edf_kernel(const double* y_norm2,
                                                 const double* Xty,
                                                 const double* XtX,
                                                 const double* beta,
                                                 const double* Ainv,
                                                 const int* info,
                                                 const int* active,
                                                 const int* design_index,
                                                 int design_count,
                                                 int lambda_index,
                                                 int group_size,
                                                 int p,
                                                 double* rss,
                                                 double* edf,
                                                 FastSplineScoreMetadata* metadata,
                                                 int* clamp_count) {
  __shared__ double scratch_rss[kBlock];
  __shared__ double scratch_edf[kBlock];
  const int fit = blockIdx.x;
  if (fit >= group_size) return;
  const std::size_t metadata_index =
    static_cast<std::size_t>(lambda_index) * group_size + fit;
  if (active[fit] == 0) {
    if (threadIdx.x == 0) {
      rss[fit] = nan("");
      edf[fit] = nan("");
      metadata[metadata_index].info = info[fit];
      metadata[metadata_index].pad0 = 0;
      metadata[metadata_index].rss = nan("");
      metadata[metadata_index].edf = nan("");
      metadata[metadata_index].pad1 = 0.0;
    }
    return;
  }

  double rss_acc = 0.0;
  double edf_acc = 0.0;
  const int pp = p * p;
  const int design = design_index[fit];
  const int factor = lambda_index * design_count + design;
  const std::size_t fit_vec_base = static_cast<std::size_t>(fit) * p;
  for (int col = threadIdx.x; col < p; col += blockDim.x) {
    const double beta_col = beta[fit_vec_base + col];
    rss_acc += -2.0 * beta_col * Xty[fit_vec_base + col];
    for (int row = 0; row < p; ++row) {
      rss_acc += beta_col *
        XtX[colmajor_square_offset(design, row, col, p)] *
        beta[fit_vec_base + row];
    }
  }

  for (int linear = threadIdx.x; linear < pp; linear += blockDim.x) {
    const int row = linear % p;
    const int col = linear / p;
    edf_acc += XtX[colmajor_square_offset(design, row, col, p)] *
               Ainv[colmajor_square_offset(factor, col, row, p)];
  }

  scratch_rss[threadIdx.x] = rss_acc;
  scratch_edf[threadIdx.x] = edf_acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch_rss[threadIdx.x] += scratch_rss[threadIdx.x + stride];
      scratch_edf[threadIdx.x] += scratch_edf[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    double value = y_norm2[fit] + scratch_rss[0];
    if (value < 0.0 && value > -1e-8) {
      value = 0.0;
      atomicAdd(clamp_count, 1);
    }
    edf[fit] = scratch_edf[0];
    rss[fit] = value;
    metadata[metadata_index].info = info[fit];
    metadata[metadata_index].pad0 = 0;
    metadata[metadata_index].rss = value;
    metadata[metadata_index].edf = scratch_edf[0];
    metadata[metadata_index].pad1 = 0.0;
  }
}

__global__ void batched_algebraic_rss_edf_all_candidates_kernel(
  const double* y_norm2,
  const double* Xty,
  const double* XtX,
  const double* factors,
  const double* Ainv,
  const int* active,
  const int* design_index,
  int design_count,
  int lambda_count,
  int group_size,
  int p,
  double* rss,
  double* edf,
  FastSplineScoreMetadata* metadata,
  int* clamp_count) {
  __shared__ double scratch_rss[kBlock];
  __shared__ double scratch_edf[kBlock];
  __shared__ double beta_s[kSmallPRhsMaxDesignCols];
  __shared__ int solve_info;
  const int fit = blockIdx.x;
  const int lambda_index = blockIdx.y;
  if (fit >= group_size || lambda_index >= lambda_count) return;
  const std::size_t metadata_index =
    static_cast<std::size_t>(lambda_index) * group_size + fit;
  const int design = design_index[fit];
  const int factor = lambda_index * design_count + design;

  if (threadIdx.x == 0) {
    solve_info = 0;
    if (active[fit] == 0) {
      for (int row = 0; row < p; ++row) beta_s[row] = 0.0;
    } else {
      solve_info = solve_small_p_cholesky_rhs_local(
        factors, Xty, factor, fit, p, beta_s);
    }
  }
  __syncthreads();

  if (active[fit] == 0 || solve_info != 0) {
    if (threadIdx.x == 0) {
      rss[fit] = nan("");
      edf[fit] = nan("");
      metadata[metadata_index].info = solve_info;
      metadata[metadata_index].pad0 = 0;
      metadata[metadata_index].rss = nan("");
      metadata[metadata_index].edf = nan("");
      metadata[metadata_index].pad1 = 0.0;
    }
    return;
  }

  double rss_acc = 0.0;
  double edf_acc = 0.0;
  const int pp = p * p;
  const std::size_t fit_vec_base = static_cast<std::size_t>(fit) * p;
  for (int col = threadIdx.x; col < p; col += blockDim.x) {
    const double beta_col = beta_s[col];
    rss_acc += -2.0 * beta_col * Xty[fit_vec_base + col];
    for (int row = 0; row < p; ++row) {
      rss_acc += beta_col *
        XtX[colmajor_square_offset(design, row, col, p)] *
        beta_s[row];
    }
  }

  for (int linear = threadIdx.x; linear < pp; linear += blockDim.x) {
    const int row = linear % p;
    const int col = linear / p;
    edf_acc += XtX[colmajor_square_offset(design, row, col, p)] *
               Ainv[colmajor_square_offset(factor, col, row, p)];
  }

  scratch_rss[threadIdx.x] = rss_acc;
  scratch_edf[threadIdx.x] = edf_acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch_rss[threadIdx.x] += scratch_rss[threadIdx.x + stride];
      scratch_edf[threadIdx.x] += scratch_edf[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    double value = y_norm2[fit] + scratch_rss[0];
    if (value < 0.0 && value > -1e-8) {
      value = 0.0;
      atomicAdd(clamp_count, 1);
    }
    edf[fit] = scratch_edf[0];
    rss[fit] = value;
    metadata[metadata_index].info = 0;
    metadata[metadata_index].pad0 = 0;
    metadata[metadata_index].rss = value;
    metadata[metadata_index].edf = scratch_edf[0];
    metadata[metadata_index].pad1 = 0.0;
  }
}

struct DeviceGroupBuffers {
  double* d_X = nullptr;
  std::size_t d_X_capacity = 0;
  double* d_P = nullptr;
  std::size_t d_P_capacity = 0;
  double* d_design_X = nullptr;
  std::size_t d_design_X_capacity = 0;
  double* d_design_P = nullptr;
  std::size_t d_design_P_capacity = 0;
  double* d_y = nullptr;
  std::size_t d_y_capacity = 0;
  int* d_request_design_index = nullptr;
  std::size_t d_request_design_index_capacity = 0;
  double* d_XtX = nullptr;
  std::size_t d_XtX_capacity = 0;
  double* d_design_XtX = nullptr;
  std::size_t d_design_XtX_capacity = 0;
  double* d_Xty = nullptr;
  std::size_t d_Xty_capacity = 0;
  double* d_y_norm2 = nullptr;
  std::size_t d_y_norm2_capacity = 0;
  double* d_A = nullptr;
  std::size_t d_A_capacity = 0;
  double* d_design_A = nullptr;
  std::size_t d_design_A_capacity = 0;
  double* d_beta = nullptr;
  std::size_t d_beta_capacity = 0;
  double* d_Ainv = nullptr;
  std::size_t d_Ainv_capacity = 0;
  double* d_design_Ainv = nullptr;
  std::size_t d_design_Ainv_capacity = 0;
  double* d_fitted = nullptr;
  std::size_t d_fitted_capacity = 0;
  double* d_residuals = nullptr;
  std::size_t d_residuals_capacity = 0;
  double* d_rss = nullptr;
  std::size_t d_rss_capacity = 0;
  double* d_edf = nullptr;
  std::size_t d_edf_capacity = 0;
  FastSplineScoreMetadata* d_score_metadata = nullptr;
  std::size_t d_score_metadata_capacity = 0;
  FastSplineSelectedFactorDescriptor* d_selected_factor_descriptors = nullptr;
  std::size_t d_selected_factor_descriptors_capacity = 0;
  FastSplineSelectedFitDescriptor* d_selected_fit_descriptors = nullptr;
  std::size_t d_selected_fit_descriptors_capacity = 0;
  double* d_lambda_grid = nullptr;
  std::size_t d_lambda_grid_capacity = 0;
  int* d_info = nullptr;
  std::size_t d_info_capacity = 0;
  int* d_active = nullptr;
  std::size_t d_active_capacity = 0;
  int* d_algebraic_rss_clamp_count = nullptr;
  std::size_t d_algebraic_rss_clamp_count_capacity = 0;
  double** d_A_ptrs = nullptr;
  std::size_t d_A_ptrs_capacity = 0;
  double** d_design_A_ptrs = nullptr;
  std::size_t d_design_A_ptrs_capacity = 0;
  double** d_beta_ptrs = nullptr;
  std::size_t d_beta_ptrs_capacity = 0;
  double** d_design_Ainv_ptrs = nullptr;
  std::size_t d_design_Ainv_ptrs_capacity = 0;
  DeviceArena<double> double_arena;
  DeviceArena<int> int_arena;
  DeviceArena<double*> ptr_arena;
  DeviceArena<FastSplineScoreMetadata> score_metadata_arena;
  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas = nullptr;
};

template <typename T>
std::size_t arena_capacity_bytes(const DeviceArena<T>& arena) {
  return arena.capacity * sizeof(T);
}

template <typename T>
void free_arena(DeviceArena<T>* arena) {
  cudaFree(arena->base);
  arena->base = nullptr;
  arena->capacity = 0;
  arena->used = 0;
}

std::size_t grow_arena_capacity(std::size_t current,
                                std::size_t required) {
  const std::size_t doubled = current > 0 ? current * 2 : 0;
  return std::max(required, std::max(doubled,
                                     static_cast<std::size_t>(256)));
}

template <typename T>
void ensure_arena_capacity(DeviceArena<T>* arena,
                           std::size_t required,
                           const char* stage,
                           FastSplineCudaBatchDiagnostics* timing) {
  if (required <= arena->capacity) {
    arena->used = 0;
    if (required > 0 && timing != nullptr) {
      ++timing->workspace_slab_reuse_count;
    }
    return;
  }
  const std::size_t old_capacity = arena->capacity;
  if (arena->base != nullptr) check_cuda(cudaFree(arena->base), stage);
  arena->base = nullptr;
  arena->capacity = 0;
  arena->used = 0;
  if (required > 0) {
    const std::size_t new_capacity =
      grow_arena_capacity(old_capacity, required);
    void* raw = nullptr;
    check_cuda(cudaMalloc(&raw, sizeof(T) * new_capacity), stage);
    arena->base = static_cast<T*>(raw);
    arena->capacity = new_capacity;
  }
  if (timing != nullptr) {
    ++timing->workspace_slab_grow_count;
    ++timing->workspace_grow_count;
  }
}

template <typename T>
T* carve_arena(DeviceArena<T>* arena,
               std::size_t count,
               const char* stage) {
  if (count == 0) return nullptr;
  if (arena->base == nullptr || arena->used + count > arena->capacity) {
    throw std::runtime_error(std::string("insufficient arena capacity for ") +
                             stage);
  }
  T* out = arena->base + arena->used;
  arena->used += count;
  return out;
}

template <typename StorageT, typename ValueT>
std::size_t arena_storage_count(std::size_t value_count) {
  const std::size_t bytes = sizeof(ValueT) * value_count;
  return (bytes + sizeof(StorageT) - 1) / sizeof(StorageT);
}

TrueBatchArenaCounts compute_arena_counts(std::size_t x_size,
                                          std::size_t design_x_size,
                                          std::size_t design_pp_size,
                                          std::size_t y_size,
                                          int group_size,
                                          std::size_t vec_size,
                                          std::size_t factor_pp_size,
                                          std::size_t candidate_pp_size,
                                          int lambda_count,
                                          int selected_capacity,
                                          int info_count,
                                          int candidate_factor_count,
                                          bool need_fitted) {
  TrueBatchArenaCounts counts;
  const std::size_t group_count = static_cast<std::size_t>(group_size);
  const std::size_t selected_count =
    static_cast<std::size_t>(selected_capacity);
  const std::size_t lambda_count_size =
    static_cast<std::size_t>(lambda_count);
  const std::size_t info_count_size = static_cast<std::size_t>(info_count);
  const std::size_t candidate_factor_count_size =
    static_cast<std::size_t>(candidate_factor_count);

  counts.double_count =
    x_size +
    design_x_size +
    design_pp_size +
    y_size +
    design_pp_size +
    vec_size +
    group_count +
    factor_pp_size +
    candidate_pp_size +
    vec_size +
    candidate_pp_size +
    (need_fitted ? y_size : static_cast<std::size_t>(0)) +
    y_size +
    group_count +
    group_count +
    lambda_count_size +
    arena_storage_count<double, FastSplineSelectedFactorDescriptor>(
      selected_count);
  counts.int_count =
    group_count +
    info_count_size +
    group_count +
    static_cast<std::size_t>(1) +
    arena_storage_count<int, FastSplineSelectedFitDescriptor>(group_count);
  counts.ptr_count =
    group_count +
    static_cast<std::size_t>(std::max(group_size, candidate_factor_count)) +
    group_count +
    candidate_factor_count_size;
  counts.score_metadata_count = group_count * lambda_count_size;
  return counts;
}

double arena_total_bytes(const DeviceGroupBuffers& buffers) {
  return static_cast<double>(
    arena_capacity_bytes(buffers.double_arena) +
    arena_capacity_bytes(buffers.int_arena) +
    arena_capacity_bytes(buffers.ptr_arena) +
    arena_capacity_bytes(buffers.score_metadata_arena));
}

void assign_group_buffer_slices(DeviceGroupBuffers* buffers,
                                std::size_t x_size,
                                std::size_t design_x_size,
                                std::size_t design_pp_size,
                                std::size_t y_size,
                                int group_size,
                                std::size_t vec_size,
                                std::size_t factor_pp_size,
                                std::size_t candidate_pp_size,
                                int lambda_count,
                                int selected_capacity,
                                int info_count,
                                int candidate_factor_count,
                                bool need_fitted) {
  const std::size_t group_count = static_cast<std::size_t>(group_size);
  const std::size_t selected_count =
    static_cast<std::size_t>(selected_capacity);
  const std::size_t lambda_count_size =
    static_cast<std::size_t>(lambda_count);
  const std::size_t info_count_size = static_cast<std::size_t>(info_count);
  const std::size_t candidate_factor_count_size =
    static_cast<std::size_t>(candidate_factor_count);
  buffers->double_arena.used = 0;
  buffers->int_arena.used = 0;
  buffers->ptr_arena.used = 0;
  buffers->score_metadata_arena.used = 0;

  buffers->d_X = carve_arena(&buffers->double_arena, x_size, "batched X");
  buffers->d_X_capacity = x_size;
  buffers->d_P = nullptr;
  buffers->d_P_capacity = 0;
  buffers->d_design_X = carve_arena(&buffers->double_arena, design_x_size,
                                    "batched design X");
  buffers->d_design_X_capacity = design_x_size;
  buffers->d_design_P = carve_arena(&buffers->double_arena, design_pp_size,
                                    "batched design P");
  buffers->d_design_P_capacity = design_pp_size;
  buffers->d_y = carve_arena(&buffers->double_arena, y_size, "batched y");
  buffers->d_y_capacity = y_size;
  buffers->d_XtX = nullptr;
  buffers->d_XtX_capacity = 0;
  buffers->d_design_XtX = carve_arena(&buffers->double_arena,
                                      design_pp_size,
                                      "batched design XtX");
  buffers->d_design_XtX_capacity = design_pp_size;
  buffers->d_Xty = carve_arena(&buffers->double_arena, vec_size,
                               "batched Xty");
  buffers->d_Xty_capacity = vec_size;
  buffers->d_y_norm2 = carve_arena(&buffers->double_arena, group_count,
                                   "batched y norm2");
  buffers->d_y_norm2_capacity = group_count;
  buffers->d_A = carve_arena(&buffers->double_arena, factor_pp_size,
                             "batched selected factor A");
  buffers->d_A_capacity = factor_pp_size;
  buffers->d_design_A = carve_arena(&buffers->double_arena,
                                    candidate_pp_size,
                                    "batched design A");
  buffers->d_design_A_capacity = candidate_pp_size;
  buffers->d_beta = carve_arena(&buffers->double_arena, vec_size,
                                "batched beta");
  buffers->d_beta_capacity = vec_size;
  buffers->d_Ainv = nullptr;
  buffers->d_Ainv_capacity = 0;
  buffers->d_design_Ainv = carve_arena(&buffers->double_arena,
                                       candidate_pp_size,
                                       "batched design A inverse");
  buffers->d_design_Ainv_capacity = candidate_pp_size;
  buffers->d_fitted = carve_arena(&buffers->double_arena,
                                  need_fitted ? y_size :
                                    static_cast<std::size_t>(0),
                                  "batched fitted");
  buffers->d_fitted_capacity = need_fitted ? y_size : 0;
  buffers->d_residuals = carve_arena(&buffers->double_arena, y_size,
                                     "batched residuals");
  buffers->d_residuals_capacity = y_size;
  buffers->d_rss = carve_arena(&buffers->double_arena, group_count,
                               "batched rss");
  buffers->d_rss_capacity = group_count;
  buffers->d_edf = carve_arena(&buffers->double_arena, group_count,
                               "batched edf");
  buffers->d_edf_capacity = group_count;
  buffers->d_lambda_grid = carve_arena(&buffers->double_arena,
                                       lambda_count_size,
                                       "lambda grid");
  buffers->d_lambda_grid_capacity = lambda_count_size;

  buffers->d_request_design_index = carve_arena(&buffers->int_arena,
                                                group_count,
                                                "request design index");
  buffers->d_request_design_index_capacity = group_count;
  buffers->d_info = carve_arena(&buffers->int_arena, info_count_size,
                                "batched info");
  buffers->d_info_capacity = info_count_size;
  buffers->d_active = carve_arena(&buffers->int_arena, group_count,
                                  "batched active");
  buffers->d_active_capacity = group_count;
  buffers->d_algebraic_rss_clamp_count =
    carve_arena(&buffers->int_arena, static_cast<std::size_t>(1),
                "algebraic rss clamp count");
  buffers->d_algebraic_rss_clamp_count_capacity = 1;

  buffers->d_A_ptrs = carve_arena(&buffers->ptr_arena, group_count,
                                  "batched A ptrs");
  buffers->d_A_ptrs_capacity = group_count;
  buffers->d_design_A_ptrs =
    carve_arena(&buffers->ptr_arena,
                static_cast<std::size_t>(
                  std::max(group_size, candidate_factor_count)),
                "batched design A ptrs");
  buffers->d_design_A_ptrs_capacity =
    static_cast<std::size_t>(std::max(group_size, candidate_factor_count));
  buffers->d_beta_ptrs = carve_arena(&buffers->ptr_arena, group_count,
                                     "batched beta ptrs");
  buffers->d_beta_ptrs_capacity = group_count;
  buffers->d_design_Ainv_ptrs =
    carve_arena(&buffers->ptr_arena, candidate_factor_count_size,
                "batched design inverse ptrs");
  buffers->d_design_Ainv_ptrs_capacity = candidate_factor_count_size;

  buffers->d_score_metadata =
    carve_arena(&buffers->score_metadata_arena,
                group_count * lambda_count_size,
                "batched score metadata");
  buffers->d_score_metadata_capacity = group_count * lambda_count_size;
  double* selected_factor_descriptor_storage =
    carve_arena(
      &buffers->double_arena,
      arena_storage_count<double, FastSplineSelectedFactorDescriptor>(
        selected_count),
      "selected factor descriptors");
  buffers->d_selected_factor_descriptors =
    reinterpret_cast<FastSplineSelectedFactorDescriptor*>(
      selected_factor_descriptor_storage);
  buffers->d_selected_factor_descriptors_capacity = selected_count;
  int* selected_fit_descriptor_storage =
    carve_arena(&buffers->int_arena,
                arena_storage_count<int, FastSplineSelectedFitDescriptor>(
                  group_count),
                "selected fit descriptors");
  buffers->d_selected_fit_descriptors =
    reinterpret_cast<FastSplineSelectedFitDescriptor*>(
      selected_fit_descriptor_storage);
  buffers->d_selected_fit_descriptors_capacity = group_count;
}

template <typename T>
void ensure_device_capacity(T** ptr,
                            std::size_t* capacity,
                            std::size_t required,
                            const char* stage,
                            FastSplineCudaBatchDiagnostics* timing) {
  if (required <= *capacity) return;
  if (*ptr != nullptr) check_cuda(cudaFree(*ptr), stage);
  *ptr = nullptr;
  *capacity = 0;
  if (required > 0) {
    check_cuda(cudaMalloc(ptr, sizeof(T) * required), stage);
    *capacity = required;
  }
  if (timing != nullptr) ++timing->workspace_grow_count;
  if (timing != nullptr) ++timing->workspace_legacy_alloc_count;
}

void ensure_handles(DeviceGroupBuffers* buffers,
                    FastSplineCudaBatchDiagnostics* timing) {
  bool created = false;
  if (buffers->solver == nullptr) {
    check_cusolver(cusolverDnCreate(&buffers->solver),
                   "create batched cuSOLVER handle");
    created = true;
  }
  if (buffers->blas == nullptr) {
    check_cublas(cublasCreate(&buffers->blas),
                 "create batched cuBLAS handle");
    created = true;
  }
  if (created && timing != nullptr) ++timing->solver_handle_create_count;
}

void ensure_group_buffers(DeviceGroupBuffers* buffers,
                          std::size_t x_size,
                          std::size_t design_x_size,
                          std::size_t design_pp_size,
                          std::size_t y_size,
                          int group_size,
                          std::size_t vec_size,
                          std::size_t factor_pp_size,
                          std::size_t candidate_pp_size,
                          int lambda_count,
                          int selected_capacity,
                          int info_count,
                          int candidate_factor_count,
                          bool need_fitted,
                          FastSplineCudaBatchDiagnostics* timing) {
  const TrueBatchArenaCounts counts =
    compute_arena_counts(x_size, design_x_size, design_pp_size, y_size,
                         group_size, vec_size, factor_pp_size,
                         candidate_pp_size, lambda_count, selected_capacity,
                         info_count, candidate_factor_count, need_fitted);
  ensure_arena_capacity(&buffers->double_arena, counts.double_count,
                        "alloc batched double workspace slab", timing);
  ensure_arena_capacity(&buffers->int_arena, counts.int_count,
                        "alloc batched int workspace slab", timing);
  ensure_arena_capacity(&buffers->ptr_arena, counts.ptr_count,
                        "alloc batched pointer workspace slab", timing);
  ensure_arena_capacity(&buffers->score_metadata_arena,
                        counts.score_metadata_count,
                        "alloc batched score metadata workspace slab",
                        timing);
  assign_group_buffer_slices(buffers, x_size, design_x_size, design_pp_size,
                             y_size, group_size, vec_size, factor_pp_size,
                             candidate_pp_size, lambda_count,
                             selected_capacity, info_count,
                             candidate_factor_count, need_fitted);
  if (timing != nullptr) {
    timing->workspace_slab_bytes =
      std::max(timing->workspace_slab_bytes, arena_total_bytes(*buffers));
  }
  ensure_handles(buffers, timing);
}

void solve_rhs_batched(DeviceGroupBuffers* buffers,
                       int group_size,
                       int p,
                       std::size_t vec_size,
                       double* factor_base,
                       const int* factor_index,
                       const int* active,
                       const char* copy_stage,
                       const char* launch_stage,
                       const char* cusolver_stage,
                       FastSplineCudaBatchDiagnostics* timing) {
  const int beta_blocks = static_cast<int>(
    (vec_size + kBlock - 1) / kBlock);
  batched_copy_xty_to_beta_kernel<<<std::max(1, beta_blocks), kBlock>>>(
    buffers->d_Xty, group_size, p, buffers->d_beta);
  check_cuda(cudaGetLastError(), copy_stage);

  if (p <= kSmallPRhsMaxDesignCols) {
    check_cuda(cudaMemset(buffers->d_info, 0, sizeof(int) * group_size),
               "zero small-p rhs info");
    const int solve_blocks = (group_size + kBlock - 1) / kBlock;
    small_p_cholesky_rhs_solve_kernel<<<std::max(1, solve_blocks), kBlock>>>(
      factor_base, buffers->d_Xty, factor_index, active, group_size, p,
      buffers->d_beta, buffers->d_info);
    check_cuda(cudaGetLastError(), launch_stage);
    check_cuda(cudaDeviceSynchronize(), "synchronize small-p RHS solve");
    timing->rhs_custom_solve_count += group_size;
    return;
  }

  timing->rhs_solve_fallback_count += group_size;
  timing->rhs_cublas_solve_count += group_size;
  check_cusolver(cusolverDnDpotrsBatched(
    buffers->solver, CUBLAS_FILL_MODE_UPPER, p, 1, buffers->d_A_ptrs, p,
    buffers->d_beta_ptrs, p, buffers->d_info, group_size),
    cusolver_stage);
  check_cuda(cudaDeviceSynchronize(), "synchronize batched RHS solve");
}

void solve_rhs_batched_selected(DeviceGroupBuffers* buffers,
                                int group_size,
                                int p,
                                std::size_t vec_size,
                                double* factor_base,
                                const char* copy_stage,
                                const char* launch_stage,
                                const char* cusolver_stage,
                                FastSplineCudaBatchDiagnostics* timing) {
  const int beta_blocks = static_cast<int>(
    (vec_size + kBlock - 1) / kBlock);
  batched_copy_xty_to_beta_kernel<<<std::max(1, beta_blocks), kBlock>>>(
    buffers->d_Xty, group_size, p, buffers->d_beta);
  check_cuda(cudaGetLastError(), copy_stage);

  if (p <= kSmallPRhsMaxDesignCols) {
    check_cuda(cudaMemset(buffers->d_info, 0, sizeof(int) * group_size),
               "zero small-p selected rhs info");
    const int solve_blocks = (group_size + kBlock - 1) / kBlock;
    small_p_cholesky_rhs_solve_selected_kernel<<<
      std::max(1, solve_blocks), kBlock>>>(
        factor_base, buffers->d_Xty, buffers->d_selected_fit_descriptors,
        group_size, p, buffers->d_beta, buffers->d_info);
    check_cuda(cudaGetLastError(), launch_stage);
    check_cuda(cudaDeviceSynchronize(),
               "synchronize small-p selected RHS solve");
    timing->rhs_custom_solve_count += group_size;
    return;
  }

  timing->rhs_solve_fallback_count += group_size;
  timing->rhs_cublas_solve_count += group_size;
  check_cusolver(cusolverDnDpotrsBatched(
    buffers->solver, CUBLAS_FILL_MODE_UPPER, p, 1, buffers->d_A_ptrs, p,
    buffers->d_beta_ptrs, p, buffers->d_info, group_size),
    cusolver_stage);
  check_cuda(cudaDeviceSynchronize(), "synchronize selected batched RHS solve");
}

void clear_group_buffer_views(DeviceGroupBuffers* buffers) {
  buffers->d_X = nullptr;
  buffers->d_X_capacity = 0;
  buffers->d_P = nullptr;
  buffers->d_P_capacity = 0;
  buffers->d_design_X = nullptr;
  buffers->d_design_X_capacity = 0;
  buffers->d_design_P = nullptr;
  buffers->d_design_P_capacity = 0;
  buffers->d_y = nullptr;
  buffers->d_y_capacity = 0;
  buffers->d_request_design_index = nullptr;
  buffers->d_request_design_index_capacity = 0;
  buffers->d_XtX = nullptr;
  buffers->d_XtX_capacity = 0;
  buffers->d_design_XtX = nullptr;
  buffers->d_design_XtX_capacity = 0;
  buffers->d_Xty = nullptr;
  buffers->d_Xty_capacity = 0;
  buffers->d_y_norm2 = nullptr;
  buffers->d_y_norm2_capacity = 0;
  buffers->d_A = nullptr;
  buffers->d_A_capacity = 0;
  buffers->d_design_A = nullptr;
  buffers->d_design_A_capacity = 0;
  buffers->d_beta = nullptr;
  buffers->d_beta_capacity = 0;
  buffers->d_Ainv = nullptr;
  buffers->d_Ainv_capacity = 0;
  buffers->d_design_Ainv = nullptr;
  buffers->d_design_Ainv_capacity = 0;
  buffers->d_fitted = nullptr;
  buffers->d_fitted_capacity = 0;
  buffers->d_residuals = nullptr;
  buffers->d_residuals_capacity = 0;
  buffers->d_rss = nullptr;
  buffers->d_rss_capacity = 0;
  buffers->d_edf = nullptr;
  buffers->d_edf_capacity = 0;
  buffers->d_score_metadata = nullptr;
  buffers->d_score_metadata_capacity = 0;
  buffers->d_selected_factor_descriptors = nullptr;
  buffers->d_selected_factor_descriptors_capacity = 0;
  buffers->d_selected_fit_descriptors = nullptr;
  buffers->d_selected_fit_descriptors_capacity = 0;
  buffers->d_lambda_grid = nullptr;
  buffers->d_lambda_grid_capacity = 0;
  buffers->d_info = nullptr;
  buffers->d_info_capacity = 0;
  buffers->d_active = nullptr;
  buffers->d_active_capacity = 0;
  buffers->d_algebraic_rss_clamp_count = nullptr;
  buffers->d_algebraic_rss_clamp_count_capacity = 0;
  buffers->d_A_ptrs = nullptr;
  buffers->d_A_ptrs_capacity = 0;
  buffers->d_design_A_ptrs = nullptr;
  buffers->d_design_A_ptrs_capacity = 0;
  buffers->d_beta_ptrs = nullptr;
  buffers->d_beta_ptrs_capacity = 0;
  buffers->d_design_Ainv_ptrs = nullptr;
  buffers->d_design_Ainv_ptrs_capacity = 0;
}

void free_buffers(DeviceGroupBuffers* buffers) {
  free_arena(&buffers->double_arena);
  free_arena(&buffers->int_arena);
  free_arena(&buffers->ptr_arena);
  free_arena(&buffers->score_metadata_arena);
  clear_group_buffer_views(buffers);
  if (buffers->solver != nullptr) {
    cusolverDnDestroy(buffers->solver);
    buffers->solver = nullptr;
  }
  if (buffers->blas != nullptr) {
    cublasDestroy(buffers->blas);
    buffers->blas = nullptr;
  }
}

}  // namespace

struct FastSplineCudaWorkspace {
  DeviceGroupBuffers group_buffers;
  std::map<std::string, FastSplineDesign> design_cache;
  const double* design_cache_data_ptr = nullptr;
  int design_cache_nrow = 0;
  int design_cache_ncol = 0;
  FastSplineParams design_cache_params;
  bool design_cache_params_set = false;
};

namespace {

struct BestFitState {
  bool found = false;
  double gcv = std::numeric_limits<double>::infinity();
  double lambda = std::numeric_limits<double>::quiet_NaN();
  double rss = std::numeric_limits<double>::quiet_NaN();
  double edf = std::numeric_limits<double>::quiet_NaN();
  double ridge = std::numeric_limits<double>::quiet_NaN();
  int ridge_attempt = 0;
  std::vector<double> fitted;
  std::vector<double> residuals;
};

struct TrueBatchGroupResult {
  std::vector<FastSplineCudaFit> full_fits;
  std::vector<FastSplineCudaResidualOnlyFit> residual_only_fits;
};

TrueBatchGroupResult run_true_batched_group(
  const Rcpp::NumericMatrix& data,
  const FastSplineBatchGroup& group,
  const FastSplineParams& params,
  const std::string& backend,
  FastSplineCudaBatchDiagnostics* timing,
  FastSplineCudaWorkspace* workspace,
  bool residual_only) {
  const int group_size = static_cast<int>(group.requests.size());
  const int n = group.n;
  const int p = group.design_cols;
  if (group_size <= 1) {
    throw std::runtime_error("true batched group requires at least two fits");
  }
  if (p > kMaxTrueBatchedDesignCols) {
    throw std::runtime_error("CUDA fastSpline batch unsupported design_cols=" +
                             std::to_string(p) + " for true batched solve");
  }

  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point stage =
    std::chrono::steady_clock::now();
  const std::vector<double> host_design_X = pack_group_design_x(group);
  const std::vector<double> host_design_P = pack_group_design_p(group);
  const std::vector<double> host_y = pack_group_y(data, group);
  const std::vector<int> host_design_index =
    pack_group_request_design_index(group);
  timing->host_pack_sec += elapsed_since(stage);
  const std::vector<double> lambdas = lambda_grid(params);
  const int design_count = static_cast<int>(group.designs.size());
  const int lambda_count = static_cast<int>(lambdas.size());
  const std::size_t per_request_x_size =
    static_cast<std::size_t>(group_size) * n * p;
  const std::size_t x_size = 0;
  const std::size_t design_x_size =
    static_cast<std::size_t>(design_count) * n * p;
  timing->duplicate_design_x_values_avoided +=
    static_cast<int>(per_request_x_size - design_x_size);
  const std::size_t y_size = static_cast<std::size_t>(group_size) * n;
  const std::size_t design_pp_size =
    static_cast<std::size_t>(design_count) * p * p;
  const int candidate_factor_count = design_count * lambda_count;
  const std::size_t candidate_pp_size =
    static_cast<std::size_t>(candidate_factor_count) * p * p;
  const std::size_t factor_pp_size =
    static_cast<std::size_t>(group_size) * p * p;
  const std::size_t vec_size = static_cast<std::size_t>(group_size) * p;
  const int info_count = std::max(group_size, candidate_factor_count);
  timing->factor_cache_bytes += static_cast<double>(candidate_pp_size) *
    sizeof(double) * 2.0 + static_cast<double>(factor_pp_size) *
    sizeof(double) + static_cast<double>(design_pp_size) * sizeof(double);
  timing->lambda_candidates = std::max(timing->lambda_candidates,
                                       lambda_count);

  DeviceGroupBuffers local_buffers;
  DeviceGroupBuffers* buffers = workspace == nullptr ?
    &local_buffers : &workspace->group_buffers;
  if (workspace != nullptr) ++timing->workspace_reuse_count;
  try {
    stage = std::chrono::steady_clock::now();
    ensure_group_buffers(buffers, x_size, design_x_size, design_pp_size,
                         y_size, group_size, vec_size, factor_pp_size,
                         candidate_pp_size, lambda_count, group_size,
                         info_count, candidate_factor_count,
                         !residual_only, timing);
    timing->alloc_sec += elapsed_since(stage);

    std::chrono::steady_clock::time_point copy_stage =
      std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_design_X, host_design_X.data(),
                          sizeof(double) * design_x_size,
                          cudaMemcpyHostToDevice), "copy batched design X");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   sizeof(double) * design_x_size, H2dTransferKind::Design);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_design_P, host_design_P.data(),
                          sizeof(double) * design_pp_size,
                          cudaMemcpyHostToDevice), "copy batched design P");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   sizeof(double) * design_pp_size, H2dTransferKind::Penalty);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_y, host_y.data(), sizeof(double) * y_size,
                          cudaMemcpyHostToDevice), "copy batched y");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   sizeof(double) * y_size, H2dTransferKind::Y);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_request_design_index,
                          host_design_index.data(), sizeof(int) * group_size,
                          cudaMemcpyHostToDevice),
               "copy batched request design index");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   sizeof(int) * static_cast<std::size_t>(group_size),
                   H2dTransferKind::Index);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_lambda_grid, lambdas.data(),
                          sizeof(double) * lambda_count,
                          cudaMemcpyHostToDevice), "copy lambda grid");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   sizeof(double) * static_cast<std::size_t>(lambda_count),
                   H2dTransferKind::Lambda);

    stage = std::chrono::steady_clock::now();
    const dim3 xtx_grid(p, p, design_count);
    batched_xtx_kernel<<<xtx_grid, kBlock>>>(buffers->d_design_X,
                                             design_count, n, p,
                                             buffers->d_design_XtX);
    const dim3 xty_grid(p, group_size);
    batched_xty_by_design_kernel<<<xty_grid, kBlock>>>(
      buffers->d_design_X, buffers->d_y, buffers->d_request_design_index,
      group_size, n, p, buffers->d_Xty);
    check_cuda(cudaGetLastError(), "launch batched XtX/Xty kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched XtX/Xty kernels");
    timing->xtx_xty_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    batched_y_norm2_kernel<<<group_size, kBlock>>>(
      buffers->d_y, group_size, n, buffers->d_y_norm2);
    check_cuda(cudaGetLastError(), "launch batched y norm2 kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched y norm2 kernel");
    timing->residual_summary_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    const int ptr_blocks = (group_size + kBlock - 1) / kBlock;
    const int candidate_ptr_blocks =
      (candidate_factor_count + kBlock - 1) / kBlock;
    make_vector_pointer_array<<<std::max(1, ptr_blocks), kBlock>>>(
      buffers->d_beta, group_size, p, buffers->d_beta_ptrs);
    make_matrix_pointer_array<<<std::max(1, candidate_ptr_blocks), kBlock>>>(
      buffers->d_design_A, candidate_factor_count, p,
      buffers->d_design_A_ptrs);
    make_matrix_pointer_array<<<std::max(1, candidate_ptr_blocks), kBlock>>>(
      buffers->d_design_Ainv, candidate_factor_count, p,
      buffers->d_design_Ainv_ptrs);
    check_cuda(cudaGetLastError(), "launch batched pointer setup kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched pointer setup kernels");
    timing->pointer_setup_sec += elapsed_since(stage);

    std::vector<int> active(group_size, 1);
    std::vector<BestFitState> best(group_size);
    double ridge = params.ridge;
    int ridge_attempt = 0;
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemset(buffers->d_algebraic_rss_clamp_count, 0,
                          sizeof(int)), "zero algebraic rss clamp count");
    timing->active_copy_sec += elapsed_since(stage);
    while (ridge <= 1e-4 * (1.0 + 1e-12)) {
      std::vector<BestFitState> ridge_best(group_size);
      stage = std::chrono::steady_clock::now();
      check_cuda(cudaMemcpy(buffers->d_active, active.data(),
                            sizeof(int) * group_size, cudaMemcpyHostToDevice),
                 "copy active flags");
      timing->active_copy_sec += elapsed_since(stage);
      const int system_blocks = static_cast<int>(
        (candidate_pp_size + kBlock - 1) / kBlock);
      stage = std::chrono::steady_clock::now();
      batched_build_lambda_design_system_kernel<<<std::max(1, system_blocks),
                                                  kBlock>>>(
        buffers->d_design_XtX, buffers->d_design_P, buffers->d_lambda_grid,
        design_count, lambda_count, p, ridge, buffers->d_design_A);
      check_cuda(cudaGetLastError(), "launch batched build system kernel");
      check_cuda(cudaDeviceSynchronize(), "synchronize batched build system kernel");
      timing->build_system_sec += elapsed_since(stage);

      stage = std::chrono::steady_clock::now();
      check_cusolver(cusolverDnDpotrfBatched(
        buffers->solver, CUBLAS_FILL_MODE_UPPER, p, buffers->d_design_A_ptrs,
        p, buffers->d_info, candidate_factor_count), "batched potrf");
      check_cuda(cudaDeviceSynchronize(), "synchronize batched potrf");
      const double cholesky_sec = elapsed_since(stage);
      timing->factor_cholesky_sec += cholesky_sec;
      timing->factor_solve_sec += cholesky_sec;
      timing->factorization_count += candidate_factor_count;
      timing->factor_cache_misses += candidate_factor_count;
      timing->factor_cache_hits += std::max(
        0, group_size * lambda_count - candidate_factor_count);
      timing->factor_cache_entries += candidate_factor_count;

      stage = std::chrono::steady_clock::now();
      batched_identity_kernel<<<std::max(1, system_blocks), kBlock>>>(
        buffers->d_design_Ainv, candidate_factor_count, p);
      check_cuda(cudaGetLastError(), "launch batched identity kernel");
      const double one = 1.0;
      check_cublas(cublasDtrsmBatched(
        buffers->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, p, p, &one,
        const_cast<const double**>(buffers->d_design_A_ptrs), p,
        buffers->d_design_Ainv_ptrs, p, candidate_factor_count),
        "batched inverse triangular solve 1");
      check_cublas(cublasDtrsmBatched(
        buffers->blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, p, p, &one,
        const_cast<const double**>(buffers->d_design_A_ptrs), p,
        buffers->d_design_Ainv_ptrs, p, candidate_factor_count),
        "batched inverse triangular solve 2");
      check_cuda(cudaDeviceSynchronize(), "synchronize batched inverse solve");
      const double inverse_sec = elapsed_since(stage);
      timing->factor_inverse_solve_sec += inverse_sec;
      timing->factor_solve_sec += inverse_sec;
      timing->inverse_solve_count += candidate_factor_count;

      if (p <= kSmallPRhsMaxDesignCols) {
        stage = std::chrono::steady_clock::now();
        const dim3 fused_grid(group_size, lambda_count);
        batched_algebraic_rss_edf_all_candidates_kernel<<<fused_grid, kBlock>>>(
          buffers->d_y_norm2, buffers->d_Xty, buffers->d_design_XtX,
          buffers->d_design_A, buffers->d_design_Ainv, buffers->d_active,
          buffers->d_request_design_index, design_count, lambda_count,
          group_size, p, buffers->d_rss, buffers->d_edf,
          buffers->d_score_metadata,
          buffers->d_algebraic_rss_clamp_count);
        check_cuda(cudaGetLastError(),
                   "launch batched fused all-candidate RSS kernels");
        check_cuda(cudaDeviceSynchronize(),
                   "synchronize batched fused all-candidate solve");
        const int fused_candidates = group_size * lambda_count;
        timing->candidate_rhs_fused_solve_count += fused_candidates;
        timing->candidate_beta_values_avoided += fused_candidates * p;
        timing->algebraic_rss_count += fused_candidates;
        timing->summary_group_batched_launch_count += 1;
        timing->summary_group_batched_candidate_count += fused_candidates;
        timing->residual_summary_sec += elapsed_since(stage);
      } else {
        for (int lambda_index = 0; lambda_index < lambda_count; ++lambda_index) {
          stage = std::chrono::steady_clock::now();
          make_request_lambda_matrix_pointer_array<<<std::max(1, ptr_blocks),
                                                     kBlock>>>(
            buffers->d_design_A, buffers->d_request_design_index, group_size,
            design_count, lambda_index, p, buffers->d_A_ptrs);
          check_cuda(cudaGetLastError(), "launch candidate pointer setup kernel");
          check_cuda(cudaDeviceSynchronize(),
                     "synchronize candidate pointer setup kernel");
          timing->pointer_setup_sec += elapsed_since(stage);

          stage = std::chrono::steady_clock::now();
          solve_rhs_batched(buffers, group_size, p, vec_size,
                            buffers->d_design_A +
                              static_cast<std::size_t>(lambda_index) *
                                design_count * p * p,
                            buffers->d_request_design_index, buffers->d_active,
                            "launch batched beta copy kernel",
                            "launch small-p candidate RHS solve kernel",
                            "batched potrs beta", timing);
          const double rhs_sec = elapsed_since(stage);
          timing->factor_rhs_solve_sec += rhs_sec;
          timing->factor_solve_sec += rhs_sec;
          timing->rhs_cublas_solve_sec += rhs_sec;
          timing->rhs_solve_count += group_size;
          timing->rhs_solve_api_calls += 1;
          timing->rhs_target_solves += group_size;
          timing->candidate_rhs_materialized_solve_count += group_size;

          stage = std::chrono::steady_clock::now();
          batched_algebraic_rss_edf_kernel<<<group_size, kBlock>>>(
            buffers->d_y_norm2, buffers->d_Xty, buffers->d_design_XtX,
            buffers->d_beta, buffers->d_design_Ainv, buffers->d_info,
            buffers->d_active, buffers->d_request_design_index, design_count,
            lambda_index,
            group_size, p, buffers->d_rss, buffers->d_edf,
            buffers->d_score_metadata,
            buffers->d_algebraic_rss_clamp_count);
          check_cuda(cudaGetLastError(), "launch batched algebraic RSS kernels");
          check_cuda(cudaDeviceSynchronize(),
                     "synchronize batched solve candidate");
          timing->algebraic_rss_count += group_size;
          timing->summary_candidate_launch_count += 1;
          timing->residual_summary_sec += elapsed_since(stage);
        }
      }

      std::vector<FastSplineScoreMetadata> score_metadata(
        static_cast<std::size_t>(lambda_count) * group_size);
      stage = std::chrono::steady_clock::now();
      const std::size_t score_metadata_bytes =
        sizeof(FastSplineScoreMetadata) *
        static_cast<std::size_t>(lambda_count) *
        static_cast<std::size_t>(group_size);
      check_cuda(cudaMemcpy(score_metadata.data(),
                            buffers->d_score_metadata,
                            score_metadata_bytes,
                            cudaMemcpyDeviceToHost),
                 "copy batched score metadata");
      add_d2h_timing(timing, elapsed_since(stage),
                     score_metadata_bytes, false, true);
      timing->d2h_metadata_coalesced_count += 1;
      timing->d2h_metadata_coalesced_bytes +=
        static_cast<double>(score_metadata_bytes);

      stage = std::chrono::steady_clock::now();
      for (int lambda_index = 0; lambda_index < lambda_count; ++lambda_index) {
        const double lambda = lambdas[lambda_index];
        const std::size_t lambda_offset =
          static_cast<std::size_t>(lambda_index) * group_size;
        for (int fit = 0; fit < group_size; ++fit) {
          const FastSplineScoreMetadata& meta =
            score_metadata[lambda_offset + fit];
          if (active[fit] == 0 || meta.info != 0) continue;
          const double denom = static_cast<double>(n) - meta.edf;
          if (!std::isfinite(meta.rss) || !std::isfinite(meta.edf) ||
              denom <= 1e-8) {
            continue;
          }
          const double gcv = static_cast<double>(n) * meta.rss /
            (denom * denom);
          if (!std::isfinite(gcv)) continue;

          if (!ridge_best[fit].found || gcv < ridge_best[fit].gcv ||
              (std::abs(gcv - ridge_best[fit].gcv) <= 1e-14 &&
               lambda < ridge_best[fit].lambda)) {
            ridge_best[fit].found = true;
            ridge_best[fit].gcv = gcv;
            ridge_best[fit].lambda = lambda;
            ridge_best[fit].rss = meta.rss;
            ridge_best[fit].edf = meta.edf;
            ridge_best[fit].ridge = ridge;
            ridge_best[fit].ridge_attempt = ridge_attempt;
          }
        }
      }
      timing->host_select_sec += elapsed_since(stage);

      bool any_active = false;
      for (int fit = 0; fit < group_size; ++fit) {
        if (active[fit] == 0) continue;
        if (ridge_best[fit].found) {
          best[fit] = ridge_best[fit];
          active[fit] = 0;
        } else {
          any_active = true;
        }
      }
      if (!any_active) break;
      ridge *= 100.0;
      if (ridge <= 0.0) ridge = 1e-8;
      ++ridge_attempt;
    }
    int clamp_count = 0;
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(&clamp_count, buffers->d_algebraic_rss_clamp_count,
                          sizeof(int), cudaMemcpyDeviceToHost),
               "copy algebraic rss clamp count");
    timing->algebraic_rss_clamp_count += clamp_count;
    add_d2h_timing(timing, elapsed_since(stage), sizeof(int), false, false);

    std::vector<FastSplineSelectedFitDescriptor> selected_fit_descriptors(
      group_size);
    std::vector<FastSplineSelectedFactorDescriptor>
      selected_factor_descriptors;
    std::map<std::string, int> selected_factor_by_key;
    for (int fit = 0; fit < group_size; ++fit) {
      if (!best[fit].found) {
        throw std::runtime_error("no finite CUDA fastSpline batch solve");
      }
      std::ostringstream selected_key;
      selected_key << host_design_index[fit] << "|"
                   << std::setprecision(17) << best[fit].lambda << "|"
                   << best[fit].ridge;
      std::map<std::string, int>::iterator selected_it =
        selected_factor_by_key.find(selected_key.str());
      if (selected_it == selected_factor_by_key.end()) {
        const int factor_index =
          static_cast<int>(selected_factor_descriptors.size());
        selected_factor_by_key[selected_key.str()] = factor_index;
        FastSplineSelectedFactorDescriptor descriptor;
        descriptor.lambda = best[fit].lambda;
        descriptor.ridge = best[fit].ridge;
        descriptor.design_index = host_design_index[fit];
        descriptor.pad = 0;
        selected_factor_descriptors.push_back(descriptor);
        selected_fit_descriptors[fit].factor_index = factor_index;
      } else {
        selected_fit_descriptors[fit].factor_index = selected_it->second;
      }
      selected_fit_descriptors[fit].active = 1;
    }
    const int selected_factor_count =
      static_cast<int>(selected_factor_descriptors.size());
    timing->winning_factor_reuse_count +=
      std::max(0, group_size - selected_factor_count);
    timing->factor_cache_entries += selected_factor_count;
    timing->factor_cache_misses += selected_factor_count;
    timing->factor_cache_hits += std::max(0, group_size - selected_factor_count);

    const std::size_t selected_factor_descriptor_bytes =
      sizeof(FastSplineSelectedFactorDescriptor) *
      static_cast<std::size_t>(selected_factor_count);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_selected_factor_descriptors,
                          selected_factor_descriptors.data(),
                          selected_factor_descriptor_bytes,
                          cudaMemcpyHostToDevice),
               "copy selected factor descriptors");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   selected_factor_descriptor_bytes,
                   H2dTransferKind::Lambda);
    const std::size_t selected_fit_descriptor_bytes =
      sizeof(FastSplineSelectedFitDescriptor) *
      static_cast<std::size_t>(group_size);
    copy_stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_selected_fit_descriptors,
                          selected_fit_descriptors.data(),
                          selected_fit_descriptor_bytes,
                          cudaMemcpyHostToDevice),
               "copy selected fit descriptors");
    add_h2d_timing(timing, elapsed_since(copy_stage),
                   selected_fit_descriptor_bytes,
                   H2dTransferKind::Index);
    timing->h2d_metadata_coalesced_count += 2;
    timing->h2d_metadata_coalesced_bytes +=
      static_cast<double>(selected_factor_descriptor_bytes +
                          selected_fit_descriptor_bytes);
    timing->h2d_selected_metadata_copy_count += 2;

    const int selected_system_blocks = static_cast<int>(
      (static_cast<std::size_t>(selected_factor_count) * p * p + kBlock - 1) /
      kBlock);
    stage = std::chrono::steady_clock::now();
    batched_build_selected_factor_system_kernel<<<
      std::max(1, selected_system_blocks), kBlock>>>(
      buffers->d_design_XtX, buffers->d_design_P,
      buffers->d_selected_factor_descriptors, selected_factor_count, p,
      buffers->d_A);
    check_cuda(cudaGetLastError(), "launch selected build system kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected build system kernel");
    timing->build_system_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    make_selected_matrix_pointer_array<<<std::max(1, ptr_blocks), kBlock>>>(
      buffers->d_A, buffers->d_selected_fit_descriptors, group_size, p,
      buffers->d_A_ptrs);
    make_matrix_pointer_array<<<std::max(1, selected_system_blocks), kBlock>>>(
      buffers->d_A, selected_factor_count, p, buffers->d_design_A_ptrs);
    check_cuda(cudaGetLastError(), "launch selected pointer setup kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected pointer setup kernels");
    timing->pointer_setup_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cusolver(cusolverDnDpotrfBatched(
      buffers->solver, CUBLAS_FILL_MODE_UPPER, p, buffers->d_design_A_ptrs, p,
      buffers->d_info, selected_factor_count), "selected batched potrf");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected potrf");
    double cholesky_sec = elapsed_since(stage);
    timing->factor_cholesky_sec += cholesky_sec;
    timing->factor_solve_sec += cholesky_sec;
    timing->factorization_count += selected_factor_count;

    stage = std::chrono::steady_clock::now();
    solve_rhs_batched_selected(buffers, group_size, p, vec_size, buffers->d_A,
                               "launch selected beta copy kernel",
                               "launch small-p selected RHS solve kernel",
                               "selected batched potrs beta", timing);
    double rhs_sec = elapsed_since(stage);
    timing->factor_rhs_solve_sec += rhs_sec;
    timing->factor_solve_sec += rhs_sec;
    if (p <= kSmallPRhsMaxDesignCols) {
      timing->rhs_custom_solve_sec += rhs_sec;
    } else {
      timing->rhs_cublas_solve_sec += rhs_sec;
    }
    timing->rhs_solve_count += group_size;
    timing->rhs_solve_api_calls += 1;
    timing->rhs_target_solves += group_size;
    timing->selected_rhs_materialized_solve_count += group_size;

    stage = std::chrono::steady_clock::now();
    const int row_blocks = (n + kBlock - 1) / kBlock;
    const dim3 fit_grid(std::max(1, row_blocks), group_size);
    if (residual_only) {
      batched_residual_by_design_kernel<<<fit_grid, kBlock>>>(
        buffers->d_design_X, buffers->d_y, buffers->d_beta,
        buffers->d_request_design_index, nullptr, group_size, n, p,
        buffers->d_residuals);
    } else {
      batched_fitted_residual_by_design_kernel<<<fit_grid, kBlock>>>(
        buffers->d_design_X, buffers->d_y, buffers->d_beta,
        buffers->d_request_design_index, nullptr, group_size, n, p,
        buffers->d_fitted, buffers->d_residuals);
    }
    check_cuda(cudaGetLastError(), "launch selected residual kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected residual kernels");
    timing->winning_residual_materialize_count += group_size;
    if (residual_only) {
      timing->residual_fitted_values_avoided += static_cast<int>(y_size);
    }
    timing->residual_summary_sec += elapsed_since(stage);

    std::vector<int> final_info(group_size);
    std::vector<double> residuals(y_size);
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(final_info.data(), buffers->d_info,
                          sizeof(int) * group_size, cudaMemcpyDeviceToHost),
               "copy selected info");
    add_d2h_timing(timing, elapsed_since(stage),
                   sizeof(int) * static_cast<std::size_t>(group_size),
                   false, false);
    std::vector<double> fitted;
    if (!residual_only) {
      fitted.resize(y_size);
      const std::chrono::steady_clock::time_point fitted_stage =
        std::chrono::steady_clock::now();
      check_cuda(cudaMemcpy(fitted.data(), buffers->d_fitted,
                            sizeof(double) * y_size, cudaMemcpyDeviceToHost),
                 "copy selected fitted");
      const double fitted_sec = elapsed_since(fitted_stage);
      timing->residual_fitted_materialize_sec += fitted_sec;
      add_d2h_timing(timing, fitted_sec,
                     sizeof(double) * static_cast<std::size_t>(y_size),
                     false, true);
    }
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(residuals.data(), buffers->d_residuals,
                          sizeof(double) * y_size, cudaMemcpyDeviceToHost),
               "copy selected residuals");
    add_d2h_timing(timing, elapsed_since(stage),
                   sizeof(double) * static_cast<std::size_t>(y_size),
                   true, false);

    TrueBatchGroupResult out;
    stage = std::chrono::steady_clock::now();
    if (residual_only) {
      out.residual_only_fits.resize(group_size);
      for (int fit = 0; fit < group_size; ++fit) {
        if (final_info[fit] != 0) {
          throw std::runtime_error("selected CUDA fastSpline batch solve failed");
        }
        const std::size_t offset = static_cast<std::size_t>(fit) * n;
        std::vector<double> fit_residuals(n);
        std::copy(residuals.begin() + offset, residuals.begin() + offset + n,
                  fit_residuals.begin());
        if (!finite_vec(fit_residuals)) {
          throw std::runtime_error("selected CUDA fastSpline residual contains non-finite values");
        }
        out.residual_only_fits[fit].residuals = std::move(fit_residuals);
        out.residual_only_fits[fit].diagnostics =
          make_diagnostics(true, false, "", group.group_id, fit, true,
                           backend);
      }
      timing->residual_only_fit_count += group_size;
    } else {
      out.full_fits.resize(group_size);
      for (int fit = 0; fit < group_size; ++fit) {
        if (final_info[fit] != 0) {
          throw std::runtime_error("selected CUDA fastSpline batch solve failed");
        }
        const std::size_t offset = static_cast<std::size_t>(fit) * n;
        std::vector<double> fit_fitted(n);
        std::vector<double> fit_residuals(n);
        std::copy(fitted.begin() + offset, fitted.begin() + offset + n,
                  fit_fitted.begin());
        std::copy(residuals.begin() + offset, residuals.begin() + offset + n,
                  fit_residuals.begin());
        if (!finite_vec(fit_fitted) || !finite_vec(fit_residuals)) {
          throw std::runtime_error("selected CUDA fastSpline residual contains non-finite values");
        }
        FastSplineFit value;
        value.residuals = fit_residuals;
        value.fitted = fit_fitted;
        value.selected_lambda = best[fit].lambda;
        value.gcv = best[fit].gcv;
        value.rss = best[fit].rss;
        value.edf = best[fit].edf;
        value.design_cols = p;
        value.ridge_attempts = best[fit].ridge_attempt;
        out.full_fits[fit].fit = value;
        out.full_fits[fit].diagnostics =
          make_diagnostics(true, false, "", group.group_id, fit, true,
                           backend);
      }
      timing->residual_full_fit_materialize_count += group_size;
    }
    const double materialize_sec = elapsed_since(stage);
    timing->residual_result_materialize_sec += materialize_sec;
    timing->host_select_sec += materialize_sec;
    stage = std::chrono::steady_clock::now();
    if (workspace == nullptr) free_buffers(buffers);
    timing->free_sec += elapsed_since(stage);
    timing->true_batch_total_sec += elapsed_since(total_start);
    return out;
  } catch (...) {
    stage = std::chrono::steady_clock::now();
    if (workspace == nullptr) free_buffers(buffers);
    timing->free_sec += elapsed_since(stage);
    throw;
  }
}

void append_group_diagnostics(FastSplineCudaBatchDiagnostics* diagnostics,
                              const FastSplineBatchGroup& group,
                              bool true_batched,
                              int single_fit_calls,
                              int cpu_fallback_fits,
                              const std::string& backend,
                              const std::string& status,
                              const std::string& reason) {
  const int fit_count = static_cast<int>(group.requests.size());
  diagnostics->group_id.push_back(group.group_id);
  diagnostics->group_n.push_back(group.n);
  diagnostics->group_design_cols.push_back(group.design_cols);
  diagnostics->group_fit_count.push_back(fit_count);
  diagnostics->group_true_batched.push_back(true_batched ? 1 : 0);
  diagnostics->group_single_fit_calls.push_back(single_fit_calls);
  diagnostics->group_cpu_fallback_fits.push_back(cpu_fallback_fits);
  const int unique_designs = static_cast<int>(group.designs.size());
  const int duplicate_design_fits = std::max(0, fit_count - unique_designs);
  const int group_max_fits_per_design = max_fits_per_design(group);
  diagnostics->group_unique_designs.push_back(unique_designs);
  diagnostics->group_duplicate_design_fits.push_back(duplicate_design_fits);
  diagnostics->group_max_fits_per_design.push_back(group_max_fits_per_design);
  diagnostics->group_cholesky_backend.push_back(backend);
  diagnostics->group_status.push_back(status);
  diagnostics->group_reason.push_back(reason);
  diagnostics->groups += 1;
  diagnostics->true_batched_groups += true_batched ? 1 : 0;
  diagnostics->true_batched_fits += true_batched ? fit_count : 0;
  diagnostics->single_fit_calls += single_fit_calls;
  diagnostics->cpu_fallback_fits += cpu_fallback_fits;
  diagnostics->unique_designs += unique_designs;
  diagnostics->duplicate_design_fits += duplicate_design_fits;
  diagnostics->max_fits_per_design =
    std::max(diagnostics->max_fits_per_design, group_max_fits_per_design);
  diagnostics->max_group_size = std::max(diagnostics->max_group_size, fit_count);
  diagnostics->min_group_size = diagnostics->min_group_size == 0 ?
    fit_count : std::min(diagnostics->min_group_size, fit_count);
  if (true_batched && diagnostics->cholesky_backend.empty()) {
    diagnostics->cholesky_backend = backend;
  }
}

FastSplineCudaBatchDiagnostics make_empty_batch_diagnostics(int requested_fits) {
  FastSplineCudaBatchDiagnostics out;
  out.requested_fits = requested_fits;
  out.groups = 0;
  out.true_batched_groups = 0;
  out.true_batched_fits = 0;
  out.single_fit_calls = 0;
  out.cpu_fallback_fits = 0;
  out.unique_designs = 0;
  out.duplicate_design_fits = 0;
  out.max_fits_per_design = 0;
  out.max_group_size = 0;
  out.min_group_size = 0;
  out.cholesky_backend = "";
  out.batch_mode = requested_fits == 0 ? "empty" : "true-batch";
  out.grouping_sec = 0.0;
  out.grouping_condition_key_sec = 0.0;
  out.grouping_group_key_sec = 0.0;
  out.grouping_design_build_sec = 0.0;
  out.grouping_map_insert_sec = 0.0;
  out.grouping_unaccounted_sec = 0.0;
  out.grouping_group_count = 0;
  out.grouping_design_count = 0;
  out.grouping_condition_key_sort_count = 0;
  out.grouping_string_key_count = 0;
  out.design_cache_hit_count = 0;
  out.design_cache_miss_count = 0;
  out.design_cache_insert_count = 0;
  out.design_cache_entries = 0;
  out.design_build_total_sec = 0.0;
  out.design_build_basis_sec = 0.0;
  out.design_build_penalty_sec = 0.0;
  out.design_build_x_pack_sec = 0.0;
  out.design_build_p_pack_sec = 0.0;
  out.design_build_alloc_sec = 0.0;
  out.design_build_column_extract_sec = 0.0;
  out.design_build_unaccounted_sec = 0.0;
  out.design_build_count = 0;
  out.design_build_x_values = 0;
  out.design_build_p_values = 0;
  out.design_build_basis_values = 0;
  out.design_build_penalty_values = 0;
  out.design_build_condition_cols = 0;
  out.host_pack_sec = 0.0;
  out.alloc_sec = 0.0;
  out.h2d_sec = 0.0;
  out.h2d_design_sec = 0.0;
  out.h2d_penalty_sec = 0.0;
  out.h2d_y_sec = 0.0;
  out.h2d_index_sec = 0.0;
  out.h2d_lambda_sec = 0.0;
  out.h2d_active_sec = 0.0;
  out.h2d_copy_count = 0;
  out.h2d_bytes = 0.0;
  out.h2d_design_bytes = 0.0;
  out.h2d_y_bytes = 0.0;
  out.h2d_metadata_bytes = 0.0;
  out.h2d_metadata_coalesced_count = 0;
  out.h2d_metadata_coalesced_bytes = 0.0;
  out.h2d_selected_metadata_copy_count = 0;
  out.xtx_xty_sec = 0.0;
  out.pointer_setup_sec = 0.0;
  out.active_copy_sec = 0.0;
  out.build_system_sec = 0.0;
  out.factor_solve_sec = 0.0;
  out.factor_cholesky_sec = 0.0;
  out.factor_rhs_solve_sec = 0.0;
  out.factor_inverse_solve_sec = 0.0;
  out.residual_summary_sec = 0.0;
  out.d2h_sec = 0.0;
  out.d2h_residuals_sec = 0.0;
  out.d2h_metadata_sec = 0.0;
  out.d2h_info_sec = 0.0;
  out.d2h_copy_count = 0;
  out.d2h_bytes = 0.0;
  out.d2h_residual_bytes = 0.0;
  out.d2h_metadata_bytes = 0.0;
  out.d2h_metadata_coalesced_count = 0;
  out.d2h_metadata_coalesced_bytes = 0.0;
  out.host_select_sec = 0.0;
  out.free_sec = 0.0;
  out.true_batch_total_sec = 0.0;
  out.factorization_count = 0;
  out.rhs_solve_count = 0;
  out.inverse_solve_count = 0;
  out.rhs_solve_api_calls = 0;
  out.rhs_target_solves = 0;
  out.rhs_custom_solve_count = 0;
  out.rhs_cublas_solve_count = 0;
  out.rhs_solve_fallback_count = 0;
  out.rhs_custom_solve_sec = 0.0;
  out.rhs_cublas_solve_sec = 0.0;
  out.candidate_rhs_fused_solve_count = 0;
  out.candidate_rhs_materialized_solve_count = 0;
  out.selected_rhs_materialized_solve_count = 0;
  out.candidate_beta_values_avoided = 0;
  out.summary_candidate_launch_count = 0;
  out.summary_group_batched_launch_count = 0;
  out.summary_group_batched_candidate_count = 0;
  out.winning_factor_reuse_count = 0;
  out.factor_cache_hits = 0;
  out.factor_cache_misses = 0;
  out.factor_cache_entries = 0;
  out.factor_cache_bytes = 0.0;
  out.lambda_candidates = 0;
  out.workspace_reuse_count = 0;
  out.workspace_grow_count = 0;
  out.workspace_slab_grow_count = 0;
  out.workspace_slab_reuse_count = 0;
  out.workspace_slab_bytes = 0.0;
  out.workspace_legacy_alloc_count = 0;
  out.solver_handle_create_count = 0;
  out.per_request_design_x_values = 0;
  out.duplicate_design_x_values_avoided = 0;
  out.algebraic_rss_count = 0;
  out.candidate_residual_materialize_count = 0;
  out.winning_residual_materialize_count = 0;
  out.algebraic_rss_clamp_count = 0;
  out.residual_only_batch_count = 0;
  out.residual_full_fit_batch_count = 0;
  out.residual_only_fit_count = 0;
  out.residual_full_fit_materialize_count = 0;
  out.residual_fitted_values_avoided = 0;
  out.residual_result_materialize_sec = 0.0;
  out.residual_fitted_materialize_sec = 0.0;
  out.residual_batch_top_level_wall_sec = 0.0;
  out.residual_batch_top_level_unaccounted_sec = 0.0;
  return out;
}

}  // namespace

FastSplineCudaWorkspace* create_fastspline_cuda_workspace() {
  return new FastSplineCudaWorkspace();
}

void destroy_fastspline_cuda_workspace(FastSplineCudaWorkspace* workspace) {
  if (workspace == nullptr) return;
  free_buffers(&workspace->group_buffers);
  delete workspace;
}

void prewarm_fastspline_cuda_workspace(FastSplineCudaWorkspace* workspace) {
  if (workspace == nullptr) return;
  ensure_handles(&workspace->group_buffers, nullptr);
}

bool same_fastspline_params(const FastSplineParams& lhs,
                            const FastSplineParams& rhs) {
  return lhs.degree == rhs.degree &&
    lhs.knots == rhs.knots &&
    lhs.lambda_min == rhs.lambda_min &&
    lhs.lambda_max == rhs.lambda_max &&
    lhs.lambda_count == rhs.lambda_count &&
    lhs.ridge == rhs.ridge &&
    lhs.mode == rhs.mode;
}

void bind_fastspline_design_cache(FastSplineCudaWorkspace* workspace,
                                  const Rcpp::NumericMatrix& data,
                                  const FastSplineParams& params) {
  if (workspace == nullptr) return;
  const double* data_ptr = REAL(data);
  if (workspace->design_cache_data_ptr == data_ptr &&
      workspace->design_cache_nrow == data.nrow() &&
      workspace->design_cache_ncol == data.ncol() &&
      workspace->design_cache_params_set &&
      same_fastspline_params(workspace->design_cache_params, params)) {
    return;
  }
  workspace->design_cache.clear();
  workspace->design_cache_data_ptr = data_ptr;
  workspace->design_cache_nrow = data.nrow();
  workspace->design_cache_ncol = data.ncol();
  workspace->design_cache_params = params;
  workspace->design_cache_params_set = true;
}

std::vector<FastSplineBatchGroup> make_fastspline_batch_groups(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  FastSplineCudaBatchDiagnostics* diagnostics,
  std::map<std::string, FastSplineDesign>* run_design_cache) {
  const std::chrono::steady_clock::time_point grouping_start =
    std::chrono::steady_clock::now();
  if (targets.size() != conditioning_sets.size()) {
    throw std::runtime_error("targets and conditioning_sets length mismatch");
  }
  std::vector<FastSplineBatchGroup> groups;
  std::map<std::string, int> group_by_key;
  std::map<std::string, FastSplineDesign> local_design_cache;
  std::map<std::string, FastSplineDesign>* design_cache =
    run_design_cache == nullptr ? &local_design_cache : run_design_cache;
  std::set<std::string> grouping_design_keys;
  std::vector<std::map<std::string, int> > group_design_by_key;
  for (int i = 0; i < static_cast<int>(targets.size()); ++i) {
    if (targets[i] < 0 || targets[i] >= data.ncol()) {
      throw std::runtime_error("target column out of range");
    }
    std::chrono::steady_clock::time_point substage =
      std::chrono::steady_clock::now();
    std::vector<int> normalized_conditioning_set = conditioning_sets[i];
    std::sort(normalized_conditioning_set.begin(),
              normalized_conditioning_set.end());
    const std::string exact_design_key =
      conditioning_set_key(normalized_conditioning_set);
    if (diagnostics != nullptr) {
      diagnostics->grouping_condition_key_sec += elapsed_since(substage);
      diagnostics->grouping_condition_key_sort_count += 2;
      diagnostics->grouping_string_key_count += 1;
      if (grouping_design_keys.insert(exact_design_key).second) {
        diagnostics->grouping_design_count += 1;
      }
    }

    substage = std::chrono::steady_clock::now();
    std::map<std::string, FastSplineDesign>::iterator design_it =
      design_cache->find(exact_design_key);
    bool inserted_design = false;
    if (diagnostics != nullptr) {
      if (design_it == design_cache->end()) {
        diagnostics->design_cache_miss_count += 1;
      } else {
        diagnostics->design_cache_hit_count += 1;
      }
    }
    if (design_it == design_cache->end()) {
      if (diagnostics != nullptr) {
        diagnostics->grouping_map_insert_sec += elapsed_since(substage);
      }
      substage = std::chrono::steady_clock::now();
      FastSplineDesignBuildDiagnostics build_diagnostics =
        make_empty_fastspline_design_build_diagnostics();
      FastSplineDesign design =
        make_fastspline_design(data, normalized_conditioning_set, params,
                               diagnostics == nullptr ?
                                 nullptr : &build_diagnostics);
      if (diagnostics != nullptr) {
        diagnostics->grouping_design_build_sec += elapsed_since(substage);
        diagnostics->design_build_total_sec +=
          build_diagnostics.total_sec;
        diagnostics->design_build_basis_sec +=
          build_diagnostics.basis_sec;
        diagnostics->design_build_penalty_sec +=
          build_diagnostics.penalty_sec;
        diagnostics->design_build_x_pack_sec +=
          build_diagnostics.x_pack_sec;
        diagnostics->design_build_p_pack_sec +=
          build_diagnostics.p_pack_sec;
        diagnostics->design_build_alloc_sec +=
          build_diagnostics.alloc_sec;
        diagnostics->design_build_column_extract_sec +=
          build_diagnostics.column_extract_sec;
        diagnostics->design_build_unaccounted_sec +=
          build_diagnostics.unaccounted_sec;
        diagnostics->design_build_count +=
          build_diagnostics.build_count;
        diagnostics->design_build_x_values +=
          build_diagnostics.x_values;
        diagnostics->design_build_p_values +=
          build_diagnostics.p_values;
        diagnostics->design_build_basis_values +=
          build_diagnostics.basis_values;
        diagnostics->design_build_penalty_values +=
          build_diagnostics.penalty_values;
        diagnostics->design_build_condition_cols +=
          build_diagnostics.condition_cols;
      }
      substage = std::chrono::steady_clock::now();
      design_it = design_cache->insert(
        std::make_pair(exact_design_key, design)).first;
      inserted_design = true;
    }
    if (diagnostics != nullptr) {
      diagnostics->grouping_map_insert_sec += elapsed_since(substage);
      if (inserted_design) {
        diagnostics->design_cache_insert_count += 1;
      }
      diagnostics->design_cache_entries =
        static_cast<int>(design_cache->size());
    }

    FastSplineBatchRequest request;
    request.original_index = i;
    request.target = targets[i];
    request.conditioning_set = normalized_conditioning_set;
    request.design_index = -1;

    substage = std::chrono::steady_clock::now();
    const std::string key = group_key(design_it->second, params);
    if (diagnostics != nullptr) {
      diagnostics->grouping_group_key_sec += elapsed_since(substage);
      diagnostics->grouping_string_key_count += 1;
    }
    substage = std::chrono::steady_clock::now();
    std::map<std::string, int>::iterator it = group_by_key.find(key);
    if (it == group_by_key.end()) {
      FastSplineBatchGroup group;
      group.group_id = static_cast<int>(groups.size());
      group.n = design_it->second.n;
      group.design_cols = design_it->second.p;
      groups.push_back(group);
      group_design_by_key.push_back(std::map<std::string, int>());
      group_by_key[key] = group.group_id;
      it = group_by_key.find(key);
      if (diagnostics != nullptr) diagnostics->grouping_group_count += 1;
    }
    FastSplineBatchGroup& group = groups[it->second];
    std::map<std::string, int>& designs_for_group =
      group_design_by_key[it->second];
    std::map<std::string, int>::iterator group_design_it =
      designs_for_group.find(exact_design_key);
    if (group_design_it == designs_for_group.end()) {
      const int design_index = static_cast<int>(group.designs.size());
      group.designs.push_back(design_it->second);
      designs_for_group[exact_design_key] = design_index;
      request.design_index = design_index;
    } else {
      request.design_index = group_design_it->second;
    }
    groups[it->second].requests.push_back(request);
    if (diagnostics != nullptr) {
      diagnostics->grouping_map_insert_sec += elapsed_since(substage);
    }
  }
  if (diagnostics != nullptr) {
    const double total = elapsed_since(grouping_start);
    const double accounted =
      diagnostics->grouping_condition_key_sec +
      diagnostics->grouping_group_key_sec +
      diagnostics->grouping_design_build_sec +
      diagnostics->grouping_map_insert_sec;
    diagnostics->grouping_unaccounted_sec += nonnegative_gap(total, accounted);
  }
  return groups;
}

std::vector<FastSplineBatchGroup> make_fastspline_batch_groups(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params) {
  return make_fastspline_batch_groups(data, targets, conditioning_sets, params,
                                      nullptr, nullptr);
}

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_true_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace) {
  const int requested_fits = static_cast<int>(targets.size());
  FastSplineCudaBatchResult result;
  result.fits.resize(requested_fits);
  result.diagnostics = make_empty_batch_diagnostics(requested_fits);
  if (requested_fits > 0) result.diagnostics.residual_full_fit_batch_count = 1;
  if (requested_fits == 0) return result;

  std::chrono::steady_clock::time_point stage =
    std::chrono::steady_clock::now();
  const std::vector<FastSplineBatchGroup> groups =
    make_fastspline_batch_groups(data, targets, conditioning_sets, params,
                                 &result.diagnostics, nullptr);
  result.diagnostics.grouping_sec += elapsed_since(stage);
  const std::string true_backend = "cusolver-batched";

  for (const FastSplineBatchGroup& group : groups) {
    const int fit_count = static_cast<int>(group.requests.size());
    if (fit_count == 1) {
      const FastSplineBatchRequest& request = group.requests[0];
      FastSplineCudaFit fit = fit_fastspline_residuals_cuda(
        data, request.target, request.conditioning_set, params, fallback);
      fit.diagnostics.batch_group_id = group.group_id;
      fit.diagnostics.batch_position = 0;
      fit.diagnostics.true_batched = false;
      if (fit.diagnostics.cholesky_backend.empty()) {
        fit.diagnostics.cholesky_backend = fit.diagnostics.fallback_used ?
          "cpu-fallback" : "single-fit-cusolver";
      }
      result.diagnostics.residual_full_fit_materialize_count += 1;
      result.fits[request.original_index] = fit;
      append_group_diagnostics(&result.diagnostics, group, false, 1,
                               fit.diagnostics.fallback_used ? 1 : 0,
                               fit.diagnostics.cholesky_backend,
                               fit.diagnostics.fallback_used ? "fallback" : "ok",
                               fit.diagnostics.reason);
      continue;
    }

    if (group.design_cols > kMaxTrueBatchedDesignCols) {
      const std::string reason = "CUDA fastSpline batch unsupported design_cols=" +
        std::to_string(group.design_cols) + " for true batched solve";
      if (!fallback) throw std::runtime_error(reason);
      for (int i = 0; i < fit_count; ++i) {
        const FastSplineBatchRequest& request = group.requests[i];
        result.fits[request.original_index] =
          cpu_fallback_fit(data, request, params, reason, group.group_id, i);
      }
      result.diagnostics.residual_full_fit_materialize_count += fit_count;
      append_group_diagnostics(&result.diagnostics, group, false, 0, fit_count,
                               "cpu-fallback", "fallback", reason);
      continue;
    }

    try {
      FastSplineCudaBatchDiagnostics group_timing =
        make_empty_batch_diagnostics(fit_count);
      const TrueBatchGroupResult group_result =
        run_true_batched_group(data, group, params, true_backend,
                               &group_timing, workspace, false);
      add_batch_timing(&result.diagnostics, group_timing);
      for (int i = 0; i < fit_count; ++i) {
        result.fits[group.requests[i].original_index] =
          group_result.full_fits[i];
      }
      append_group_diagnostics(&result.diagnostics, group, true, 0, 0,
                               true_backend, "ok", "");
    } catch (const std::exception& e) {
      if (!fallback) throw;
      const std::string reason = e.what();
      for (int i = 0; i < fit_count; ++i) {
        const FastSplineBatchRequest& request = group.requests[i];
        result.fits[request.original_index] =
          cpu_fallback_fit(data, request, params, reason, group.group_id, i);
      }
      result.diagnostics.residual_full_fit_materialize_count += fit_count;
      append_group_diagnostics(&result.diagnostics, group, false, 0, fit_count,
                               "cpu-fallback", "fallback", reason);
    }
  }
  if (result.diagnostics.cholesky_backend.empty()) {
    result.diagnostics.cholesky_backend =
      result.diagnostics.single_fit_calls > 0 ? "single-fit-cusolver" : "";
  }
  return result;
}

FastSplineCudaResidualOnlyBatchResult
fit_fastspline_residuals_cuda_true_batch_residuals_only(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace) {
  const int requested_fits = static_cast<int>(targets.size());
  FastSplineCudaResidualOnlyBatchResult result;
  result.fits.resize(requested_fits);
  result.diagnostics = make_empty_batch_diagnostics(requested_fits);
  if (requested_fits > 0) result.diagnostics.residual_only_batch_count = 1;
  if (requested_fits == 0) return result;

  bind_fastspline_design_cache(workspace, data, params);
  std::chrono::steady_clock::time_point stage =
    std::chrono::steady_clock::now();
  const std::vector<FastSplineBatchGroup> groups =
    make_fastspline_batch_groups(data, targets, conditioning_sets, params,
                                 &result.diagnostics,
                                 workspace == nullptr ?
                                   nullptr : &workspace->design_cache);
  result.diagnostics.grouping_sec += elapsed_since(stage);
  const std::string true_backend = "cusolver-batched";

  for (const FastSplineBatchGroup& group : groups) {
    const int fit_count = static_cast<int>(group.requests.size());
    if (fit_count == 1) {
      const FastSplineBatchRequest& request = group.requests[0];
      FastSplineCudaFit fit = fit_fastspline_residuals_cuda(
        data, request.target, request.conditioning_set, params, fallback);
      fit.diagnostics.batch_group_id = group.group_id;
      fit.diagnostics.batch_position = 0;
      fit.diagnostics.true_batched = false;
      if (fit.diagnostics.cholesky_backend.empty()) {
        fit.diagnostics.cholesky_backend = fit.diagnostics.fallback_used ?
          "cpu-fallback" : "single-fit-cusolver";
      }
      result.fits[request.original_index].residuals =
        std::move(fit.fit.residuals);
      result.fits[request.original_index].diagnostics = fit.diagnostics;
      result.diagnostics.residual_only_fit_count += 1;
      append_group_diagnostics(&result.diagnostics, group, false, 1,
                               fit.diagnostics.fallback_used ? 1 : 0,
                               fit.diagnostics.cholesky_backend,
                               fit.diagnostics.fallback_used ? "fallback" : "ok",
                               fit.diagnostics.reason);
      continue;
    }

    if (group.design_cols > kMaxTrueBatchedDesignCols) {
      const std::string reason = "CUDA fastSpline batch unsupported design_cols=" +
        std::to_string(group.design_cols) + " for true batched solve";
      if (!fallback) throw std::runtime_error(reason);
      for (int i = 0; i < fit_count; ++i) {
        const FastSplineBatchRequest& request = group.requests[i];
        FastSplineCudaFit fit =
          cpu_fallback_fit(data, request, params, reason, group.group_id, i);
        result.fits[request.original_index].residuals =
          std::move(fit.fit.residuals);
        result.fits[request.original_index].diagnostics = fit.diagnostics;
      }
      result.diagnostics.residual_only_fit_count += fit_count;
      append_group_diagnostics(&result.diagnostics, group, false, 0, fit_count,
                               "cpu-fallback", "fallback", reason);
      continue;
    }

    try {
      FastSplineCudaBatchDiagnostics group_timing =
        make_empty_batch_diagnostics(fit_count);
      const TrueBatchGroupResult group_result =
        run_true_batched_group(data, group, params, true_backend,
                               &group_timing, workspace, true);
      add_batch_timing(&result.diagnostics, group_timing);
      for (int i = 0; i < fit_count; ++i) {
        result.fits[group.requests[i].original_index] =
          group_result.residual_only_fits[i];
      }
      append_group_diagnostics(&result.diagnostics, group, true, 0, 0,
                               true_backend, "ok", "");
    } catch (const std::exception& e) {
      if (!fallback) throw;
      const std::string reason = e.what();
      for (int i = 0; i < fit_count; ++i) {
        const FastSplineBatchRequest& request = group.requests[i];
        FastSplineCudaFit fit =
          cpu_fallback_fit(data, request, params, reason, group.group_id, i);
        result.fits[request.original_index].residuals =
          std::move(fit.fit.residuals);
        result.fits[request.original_index].diagnostics = fit.diagnostics;
      }
      result.diagnostics.residual_only_fit_count += fit_count;
      append_group_diagnostics(&result.diagnostics, group, false, 0, fit_count,
                               "cpu-fallback", "fallback", reason);
    }
  }
  if (result.diagnostics.cholesky_backend.empty()) {
    result.diagnostics.cholesky_backend =
      result.diagnostics.single_fit_calls > 0 ? "single-fit-cusolver" : "";
  }
  return result;
}

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_true_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback) {
  return fit_fastspline_residuals_cuda_true_batch(
    data, targets, conditioning_sets, params, fallback, nullptr);
}
