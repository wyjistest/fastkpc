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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;
constexpr int kMaxTrueBatchedDesignCols = 128;

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

void add_batch_timing(FastSplineCudaBatchDiagnostics* out,
                      const FastSplineCudaBatchDiagnostics& value) {
  out->grouping_sec += value.grouping_sec;
  out->host_pack_sec += value.host_pack_sec;
  out->alloc_sec += value.alloc_sec;
  out->h2d_sec += value.h2d_sec;
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
  out->host_select_sec += value.host_select_sec;
  out->free_sec += value.free_sec;
  out->true_batch_total_sec += value.true_batch_total_sec;
  out->factorization_count += value.factorization_count;
  out->rhs_solve_count += value.rhs_solve_count;
  out->inverse_solve_count += value.inverse_solve_count;
  out->rhs_solve_api_calls += value.rhs_solve_api_calls;
  out->rhs_target_solves += value.rhs_target_solves;
  out->winning_factor_reuse_count += value.winning_factor_reuse_count;
  out->factor_cache_hits += value.factor_cache_hits;
  out->factor_cache_misses += value.factor_cache_misses;
  out->factor_cache_entries += value.factor_cache_entries;
  out->factor_cache_bytes += value.factor_cache_bytes;
  out->lambda_candidates = std::max(out->lambda_candidates,
                                    value.lambda_candidates);
  out->workspace_reuse_count += value.workspace_reuse_count;
  out->workspace_grow_count += value.workspace_grow_count;
  out->solver_handle_create_count += value.solver_handle_create_count;
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

const FastSplineDesign& request_design(const FastSplineBatchGroup& group,
                                       const FastSplineBatchRequest& request) {
  if (request.design_index < 0 ||
      request.design_index >= static_cast<int>(group.designs.size())) {
    throw std::runtime_error("fastSpline batch design index out of range");
  }
  return group.designs[request.design_index];
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

std::vector<double> pack_group_x(const FastSplineBatchGroup& group) {
  const int group_size = static_cast<int>(group.requests.size());
  const int n = group.n;
  const int p = group.design_cols;
  std::vector<double> out(static_cast<std::size_t>(group_size) * n * p);
  for (int fit = 0; fit < group_size; ++fit) {
    const FastSplineDesign& design = request_design(group, group.requests[fit]);
    if (design.n != n || design.p != p) {
      throw std::runtime_error("fastSpline batch X shape mismatch");
    }
    std::copy(design.X.begin(), design.X.end(),
              out.begin() + static_cast<std::size_t>(fit) * n * p);
  }
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

__global__ void batched_xty_kernel(const double* X,
                                   const double* y,
                                   int group_size,
                                   int n,
                                   int p,
                                   double* Xty) {
  __shared__ double scratch[kBlock];
  const int col = blockIdx.x;
  const int fit = blockIdx.y;
  if (fit >= group_size || col >= p) return;

  double acc = 0.0;
  const std::size_t y_base = static_cast<std::size_t>(fit) * n;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    acc += X[matrix_offset(fit, row, col, n, p)] * y[y_base + row];
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
  const int* factor_design_index,
  const double* lambdas,
  const double* ridges,
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
    const int design = factor_design_index[factor];
    const std::size_t out_idx = colmajor_square_offset(factor, row, col, p);
    double value = XtX[colmajor_square_offset(design, row, col, p)] +
      lambdas[factor] * P[static_cast<std::size_t>(design) * pp +
                          static_cast<std::size_t>(row) * p + col];
    if (row == col && row > 0) value += ridges[factor];
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

__global__ void make_indexed_matrix_pointer_array(double* base,
                                                  const int* matrix_index,
                                                  int count,
                                                  int p,
                                                  double** ptrs) {
  const int fit = blockIdx.x * blockDim.x + threadIdx.x;
  if (fit >= count) return;
  const int matrix = matrix_index[fit];
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

__global__ void batched_fitted_residual_kernel(const double* X,
                                               const double* y,
                                               const double* beta,
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
  if (active[fit] == 0) {
    fitted[out_idx] = 0.0;
    residuals[out_idx] = 0.0;
    return;
  }
  double value = 0.0;
  for (int col = 0; col < p; ++col) {
    value += X[matrix_offset(fit, row, col, n, p)] *
             beta[static_cast<std::size_t>(fit) * p + col];
  }
  fitted[out_idx] = value;
  residuals[out_idx] = y[out_idx] - value;
}

__global__ void batched_rss_edf_kernel(const double* residuals,
                                       const double* XtX,
                                       const double* Ainv,
                                       const int* active,
                                       const int* design_index,
                                       int design_count,
                                       int lambda_index,
                                       int group_size,
                                       int n,
                                       int p,
                                       double* rss,
                                       double* edf) {
  __shared__ double scratch_rss[kBlock];
  __shared__ double scratch_edf[kBlock];
  const int fit = blockIdx.x;
  if (fit >= group_size) return;
  if (active[fit] == 0) {
    if (threadIdx.x == 0) {
      rss[fit] = nan("");
      edf[fit] = nan("");
    }
    return;
  }

  double rss_acc = 0.0;
  const std::size_t row_base = static_cast<std::size_t>(fit) * n;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    const double value = residuals[row_base + row];
    rss_acc += value * value;
  }

  double edf_acc = 0.0;
  const int pp = p * p;
  const int design = design_index[fit];
  const int factor = lambda_index * design_count + design;
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
    rss[fit] = scratch_rss[0];
    edf[fit] = scratch_edf[0];
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
  double* d_selected_lambda = nullptr;
  std::size_t d_selected_lambda_capacity = 0;
  double* d_selected_ridge = nullptr;
  std::size_t d_selected_ridge_capacity = 0;
  double* d_lambda_grid = nullptr;
  std::size_t d_lambda_grid_capacity = 0;
  int* d_selected_factor_design_index = nullptr;
  std::size_t d_selected_factor_design_index_capacity = 0;
  int* d_selected_factor_index = nullptr;
  std::size_t d_selected_factor_index_capacity = 0;
  int* d_info = nullptr;
  std::size_t d_info_capacity = 0;
  int* d_active = nullptr;
  std::size_t d_active_capacity = 0;
  double** d_A_ptrs = nullptr;
  std::size_t d_A_ptrs_capacity = 0;
  double** d_design_A_ptrs = nullptr;
  std::size_t d_design_A_ptrs_capacity = 0;
  double** d_beta_ptrs = nullptr;
  std::size_t d_beta_ptrs_capacity = 0;
  double** d_design_Ainv_ptrs = nullptr;
  std::size_t d_design_Ainv_ptrs_capacity = 0;
  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas = nullptr;
};

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
                          FastSplineCudaBatchDiagnostics* timing) {
  ensure_device_capacity(&buffers->d_X, &buffers->d_X_capacity, x_size,
                         "alloc batched X", timing);
  ensure_device_capacity(&buffers->d_design_X,
                         &buffers->d_design_X_capacity, design_x_size,
                         "alloc batched design X", timing);
  ensure_device_capacity(&buffers->d_design_P,
                         &buffers->d_design_P_capacity, design_pp_size,
                         "alloc batched design P", timing);
  ensure_device_capacity(&buffers->d_y, &buffers->d_y_capacity, y_size,
                         "alloc batched y", timing);
  ensure_device_capacity(&buffers->d_request_design_index,
                         &buffers->d_request_design_index_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched request design index", timing);
  ensure_device_capacity(&buffers->d_design_XtX,
                         &buffers->d_design_XtX_capacity, design_pp_size,
                         "alloc batched design XtX", timing);
  ensure_device_capacity(&buffers->d_Xty, &buffers->d_Xty_capacity, vec_size,
                         "alloc batched Xty", timing);
  ensure_device_capacity(&buffers->d_A, &buffers->d_A_capacity,
                         factor_pp_size,
                         "alloc batched selected factor A", timing);
  ensure_device_capacity(&buffers->d_design_A,
                         &buffers->d_design_A_capacity, candidate_pp_size,
                         "alloc batched design A", timing);
  ensure_device_capacity(&buffers->d_beta, &buffers->d_beta_capacity,
                         vec_size, "alloc batched beta", timing);
  ensure_device_capacity(&buffers->d_design_Ainv,
                         &buffers->d_design_Ainv_capacity, candidate_pp_size,
                         "alloc batched design A inverse", timing);
  ensure_device_capacity(&buffers->d_fitted, &buffers->d_fitted_capacity,
                         y_size, "alloc batched fitted", timing);
  ensure_device_capacity(&buffers->d_residuals,
                         &buffers->d_residuals_capacity, y_size,
                         "alloc batched residuals", timing);
  ensure_device_capacity(&buffers->d_rss, &buffers->d_rss_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched rss", timing);
  ensure_device_capacity(&buffers->d_edf, &buffers->d_edf_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched edf", timing);
  ensure_device_capacity(&buffers->d_selected_lambda,
                         &buffers->d_selected_lambda_capacity,
                         static_cast<std::size_t>(selected_capacity),
                         "alloc selected lambda", timing);
  ensure_device_capacity(&buffers->d_selected_ridge,
                         &buffers->d_selected_ridge_capacity,
                         static_cast<std::size_t>(selected_capacity),
                         "alloc selected ridge", timing);
  ensure_device_capacity(&buffers->d_lambda_grid,
                         &buffers->d_lambda_grid_capacity,
                         static_cast<std::size_t>(lambda_count),
                         "alloc lambda grid", timing);
  ensure_device_capacity(&buffers->d_selected_factor_design_index,
                         &buffers->d_selected_factor_design_index_capacity,
                         static_cast<std::size_t>(selected_capacity),
                         "alloc selected factor design index", timing);
  ensure_device_capacity(&buffers->d_selected_factor_index,
                         &buffers->d_selected_factor_index_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc selected factor index", timing);
  ensure_device_capacity(&buffers->d_info, &buffers->d_info_capacity,
                         static_cast<std::size_t>(info_count),
                         "alloc batched info", timing);
  ensure_device_capacity(&buffers->d_active, &buffers->d_active_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched active", timing);
  ensure_device_capacity(&buffers->d_A_ptrs, &buffers->d_A_ptrs_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched A ptrs", timing);
  ensure_device_capacity(&buffers->d_design_A_ptrs,
                         &buffers->d_design_A_ptrs_capacity,
                         static_cast<std::size_t>(
                           std::max(group_size, candidate_factor_count)),
                         "alloc batched design A ptrs", timing);
  ensure_device_capacity(&buffers->d_beta_ptrs,
                         &buffers->d_beta_ptrs_capacity,
                         static_cast<std::size_t>(group_size),
                         "alloc batched beta ptrs", timing);
  ensure_device_capacity(&buffers->d_design_Ainv_ptrs,
                         &buffers->d_design_Ainv_ptrs_capacity,
                         static_cast<std::size_t>(candidate_factor_count),
                         "alloc batched design inverse ptrs", timing);
  ensure_handles(buffers, timing);
}

void free_buffers(DeviceGroupBuffers* buffers) {
  cudaFree(buffers->d_X);
  buffers->d_X = nullptr;
  buffers->d_X_capacity = 0;
  cudaFree(buffers->d_P);
  buffers->d_P = nullptr;
  buffers->d_P_capacity = 0;
  cudaFree(buffers->d_design_X);
  buffers->d_design_X = nullptr;
  buffers->d_design_X_capacity = 0;
  cudaFree(buffers->d_design_P);
  buffers->d_design_P = nullptr;
  buffers->d_design_P_capacity = 0;
  cudaFree(buffers->d_y);
  buffers->d_y = nullptr;
  buffers->d_y_capacity = 0;
  cudaFree(buffers->d_request_design_index);
  buffers->d_request_design_index = nullptr;
  buffers->d_request_design_index_capacity = 0;
  cudaFree(buffers->d_XtX);
  buffers->d_XtX = nullptr;
  buffers->d_XtX_capacity = 0;
  cudaFree(buffers->d_design_XtX);
  buffers->d_design_XtX = nullptr;
  buffers->d_design_XtX_capacity = 0;
  cudaFree(buffers->d_Xty);
  buffers->d_Xty = nullptr;
  buffers->d_Xty_capacity = 0;
  cudaFree(buffers->d_A);
  buffers->d_A = nullptr;
  buffers->d_A_capacity = 0;
  cudaFree(buffers->d_design_A);
  buffers->d_design_A = nullptr;
  buffers->d_design_A_capacity = 0;
  cudaFree(buffers->d_beta);
  buffers->d_beta = nullptr;
  buffers->d_beta_capacity = 0;
  cudaFree(buffers->d_Ainv);
  buffers->d_Ainv = nullptr;
  buffers->d_Ainv_capacity = 0;
  cudaFree(buffers->d_design_Ainv);
  buffers->d_design_Ainv = nullptr;
  buffers->d_design_Ainv_capacity = 0;
  cudaFree(buffers->d_fitted);
  buffers->d_fitted = nullptr;
  buffers->d_fitted_capacity = 0;
  cudaFree(buffers->d_residuals);
  buffers->d_residuals = nullptr;
  buffers->d_residuals_capacity = 0;
  cudaFree(buffers->d_rss);
  buffers->d_rss = nullptr;
  buffers->d_rss_capacity = 0;
  cudaFree(buffers->d_edf);
  buffers->d_edf = nullptr;
  buffers->d_edf_capacity = 0;
  cudaFree(buffers->d_selected_lambda);
  buffers->d_selected_lambda = nullptr;
  buffers->d_selected_lambda_capacity = 0;
  cudaFree(buffers->d_selected_ridge);
  buffers->d_selected_ridge = nullptr;
  buffers->d_selected_ridge_capacity = 0;
  cudaFree(buffers->d_lambda_grid);
  buffers->d_lambda_grid = nullptr;
  buffers->d_lambda_grid_capacity = 0;
  cudaFree(buffers->d_selected_factor_design_index);
  buffers->d_selected_factor_design_index = nullptr;
  buffers->d_selected_factor_design_index_capacity = 0;
  cudaFree(buffers->d_selected_factor_index);
  buffers->d_selected_factor_index = nullptr;
  buffers->d_selected_factor_index_capacity = 0;
  cudaFree(buffers->d_info);
  buffers->d_info = nullptr;
  buffers->d_info_capacity = 0;
  cudaFree(buffers->d_active);
  buffers->d_active = nullptr;
  buffers->d_active_capacity = 0;
  cudaFree(buffers->d_A_ptrs);
  buffers->d_A_ptrs = nullptr;
  buffers->d_A_ptrs_capacity = 0;
  cudaFree(buffers->d_design_A_ptrs);
  buffers->d_design_A_ptrs = nullptr;
  buffers->d_design_A_ptrs_capacity = 0;
  cudaFree(buffers->d_beta_ptrs);
  buffers->d_beta_ptrs = nullptr;
  buffers->d_beta_ptrs_capacity = 0;
  cudaFree(buffers->d_design_Ainv_ptrs);
  buffers->d_design_Ainv_ptrs = nullptr;
  buffers->d_design_Ainv_ptrs_capacity = 0;
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

std::vector<FastSplineCudaFit> run_true_batched_group(
  const Rcpp::NumericMatrix& data,
  const FastSplineBatchGroup& group,
  const FastSplineParams& params,
  const std::string& backend,
  FastSplineCudaBatchDiagnostics* timing,
  FastSplineCudaWorkspace* workspace) {
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
  const std::vector<double> host_X = pack_group_x(group);
  const std::vector<double> host_design_X = pack_group_design_x(group);
  const std::vector<double> host_design_P = pack_group_design_p(group);
  const std::vector<double> host_y = pack_group_y(data, group);
  const std::vector<int> host_design_index =
    pack_group_request_design_index(group);
  timing->host_pack_sec += elapsed_since(stage);
  const std::vector<double> lambdas = lambda_grid(params);
  const int design_count = static_cast<int>(group.designs.size());
  const int lambda_count = static_cast<int>(lambdas.size());
  const std::size_t x_size = static_cast<std::size_t>(group_size) * n * p;
  const std::size_t design_x_size =
    static_cast<std::size_t>(design_count) * n * p;
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
                         info_count, candidate_factor_count, timing);
    timing->alloc_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_X, host_X.data(), sizeof(double) * x_size,
                          cudaMemcpyHostToDevice), "copy batched X");
    check_cuda(cudaMemcpy(buffers->d_design_X, host_design_X.data(),
                          sizeof(double) * design_x_size,
                          cudaMemcpyHostToDevice), "copy batched design X");
    check_cuda(cudaMemcpy(buffers->d_design_P, host_design_P.data(),
                          sizeof(double) * design_pp_size,
                          cudaMemcpyHostToDevice), "copy batched design P");
    check_cuda(cudaMemcpy(buffers->d_y, host_y.data(), sizeof(double) * y_size,
                          cudaMemcpyHostToDevice), "copy batched y");
    check_cuda(cudaMemcpy(buffers->d_request_design_index,
                          host_design_index.data(), sizeof(int) * group_size,
                          cudaMemcpyHostToDevice),
               "copy batched request design index");
    check_cuda(cudaMemcpy(buffers->d_lambda_grid, lambdas.data(),
                          sizeof(double) * lambda_count,
                          cudaMemcpyHostToDevice), "copy lambda grid");
    timing->h2d_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    const dim3 xtx_grid(p, p, design_count);
    batched_xtx_kernel<<<xtx_grid, kBlock>>>(buffers->d_design_X,
                                             design_count, n, p,
                                             buffers->d_design_XtX);
    const dim3 xty_grid(p, group_size);
    batched_xty_kernel<<<xty_grid, kBlock>>>(buffers->d_X, buffers->d_y,
                                             group_size, n, p, buffers->d_Xty);
    check_cuda(cudaGetLastError(), "launch batched XtX/Xty kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched XtX/Xty kernels");
    timing->xtx_xty_sec += elapsed_since(stage);

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

      for (int lambda_index = 0; lambda_index < lambda_count; ++lambda_index) {
        const double lambda = lambdas[lambda_index];
        stage = std::chrono::steady_clock::now();
        make_request_lambda_matrix_pointer_array<<<std::max(1, ptr_blocks),
                                                   kBlock>>>(
          buffers->d_design_A, buffers->d_request_design_index, group_size,
          design_count, lambda_index, p, buffers->d_A_ptrs);
        check_cuda(cudaGetLastError(), "launch candidate pointer setup kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronize candidate pointer setup kernel");
        timing->pointer_setup_sec += elapsed_since(stage);

        stage = std::chrono::steady_clock::now();
        const int beta_blocks = static_cast<int>(
          (vec_size + kBlock - 1) / kBlock);
        batched_copy_xty_to_beta_kernel<<<std::max(1, beta_blocks), kBlock>>>(
          buffers->d_Xty, group_size, p, buffers->d_beta);
        check_cuda(cudaGetLastError(), "launch batched beta copy kernel");
        check_cusolver(cusolverDnDpotrsBatched(
          buffers->solver, CUBLAS_FILL_MODE_UPPER, p, 1, buffers->d_A_ptrs, p,
          buffers->d_beta_ptrs, p, buffers->d_info, group_size),
          "batched potrs beta");
        check_cuda(cudaDeviceSynchronize(), "synchronize batched RHS solve");
        const double rhs_sec = elapsed_since(stage);
        timing->factor_rhs_solve_sec += rhs_sec;
        timing->factor_solve_sec += rhs_sec;
        timing->rhs_solve_count += group_size;
        timing->rhs_solve_api_calls += 1;
        timing->rhs_target_solves += group_size;

        stage = std::chrono::steady_clock::now();
        const int row_blocks = (n + kBlock - 1) / kBlock;
        const dim3 fit_grid(std::max(1, row_blocks), group_size);
        batched_fitted_residual_kernel<<<fit_grid, kBlock>>>(
          buffers->d_X, buffers->d_y, buffers->d_beta, buffers->d_active,
          group_size, n, p, buffers->d_fitted, buffers->d_residuals);
        batched_rss_edf_kernel<<<group_size, kBlock>>>(
          buffers->d_residuals, buffers->d_design_XtX, buffers->d_design_Ainv,
          buffers->d_active, buffers->d_request_design_index, design_count,
          lambda_index, group_size, n, p, buffers->d_rss, buffers->d_edf);
        check_cuda(cudaGetLastError(), "launch batched residual summary kernels");
        check_cuda(cudaDeviceSynchronize(), "synchronize batched solve candidate");
        timing->residual_summary_sec += elapsed_since(stage);

        std::vector<int> info(group_size);
        std::vector<double> rss(group_size);
        std::vector<double> edf(group_size);
        stage = std::chrono::steady_clock::now();
        check_cuda(cudaMemcpy(info.data(), buffers->d_info,
                              sizeof(int) * group_size, cudaMemcpyDeviceToHost),
                   "copy batched info");
        check_cuda(cudaMemcpy(rss.data(), buffers->d_rss,
                              sizeof(double) * group_size, cudaMemcpyDeviceToHost),
                   "copy batched rss");
        check_cuda(cudaMemcpy(edf.data(), buffers->d_edf,
                              sizeof(double) * group_size, cudaMemcpyDeviceToHost),
                   "copy batched edf");
        timing->d2h_sec += elapsed_since(stage);

        stage = std::chrono::steady_clock::now();
        for (int fit = 0; fit < group_size; ++fit) {
          if (active[fit] == 0 || info[fit] != 0) continue;
          const double denom = static_cast<double>(n) - edf[fit];
          if (!std::isfinite(rss[fit]) || !std::isfinite(edf[fit]) ||
              denom <= 1e-8) {
            continue;
          }
          const double gcv = static_cast<double>(n) * rss[fit] / (denom * denom);
          if (!std::isfinite(gcv)) continue;

          if (!ridge_best[fit].found || gcv < ridge_best[fit].gcv ||
              (std::abs(gcv - ridge_best[fit].gcv) <= 1e-14 &&
               lambda < ridge_best[fit].lambda)) {
            ridge_best[fit].found = true;
            ridge_best[fit].gcv = gcv;
            ridge_best[fit].lambda = lambda;
            ridge_best[fit].rss = rss[fit];
            ridge_best[fit].edf = edf[fit];
            ridge_best[fit].ridge = ridge;
            ridge_best[fit].ridge_attempt = ridge_attempt;
          }
        }
        timing->host_select_sec += elapsed_since(stage);
      }

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

    std::vector<double> selected_lambdas(group_size);
    std::vector<double> selected_ridges(group_size);
    std::vector<int> selected_factor_index(group_size);
    std::vector<int> selected_factor_design_index;
    std::vector<double> selected_factor_lambdas;
    std::vector<double> selected_factor_ridges;
    std::map<std::string, int> selected_factor_by_key;
    for (int fit = 0; fit < group_size; ++fit) {
      if (!best[fit].found) {
        throw std::runtime_error("no finite CUDA fastSpline batch solve");
      }
      selected_lambdas[fit] = best[fit].lambda;
      selected_ridges[fit] = best[fit].ridge;
      std::ostringstream selected_key;
      selected_key << host_design_index[fit] << "|"
                   << std::setprecision(17) << best[fit].lambda << "|"
                   << best[fit].ridge;
      std::map<std::string, int>::iterator selected_it =
        selected_factor_by_key.find(selected_key.str());
      if (selected_it == selected_factor_by_key.end()) {
        const int factor_index =
          static_cast<int>(selected_factor_design_index.size());
        selected_factor_by_key[selected_key.str()] = factor_index;
        selected_factor_design_index.push_back(host_design_index[fit]);
        selected_factor_lambdas.push_back(best[fit].lambda);
        selected_factor_ridges.push_back(best[fit].ridge);
        selected_factor_index[fit] = factor_index;
      } else {
        selected_factor_index[fit] = selected_it->second;
      }
    }
    const int selected_factor_count =
      static_cast<int>(selected_factor_design_index.size());
    timing->winning_factor_reuse_count +=
      std::max(0, group_size - selected_factor_count);
    timing->factor_cache_entries += selected_factor_count;
    timing->factor_cache_misses += selected_factor_count;
    timing->factor_cache_hits += std::max(0, group_size - selected_factor_count);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_selected_lambda,
                          selected_factor_lambdas.data(),
                          sizeof(double) * selected_factor_count,
                          cudaMemcpyHostToDevice),
               "copy selected lambdas");
    check_cuda(cudaMemcpy(buffers->d_selected_ridge,
                          selected_factor_ridges.data(),
                          sizeof(double) * selected_factor_count,
                          cudaMemcpyHostToDevice),
               "copy selected ridges");
    check_cuda(cudaMemcpy(buffers->d_selected_factor_design_index,
                          selected_factor_design_index.data(),
                          sizeof(int) * selected_factor_count,
                          cudaMemcpyHostToDevice),
               "copy selected factor design index");
    check_cuda(cudaMemcpy(buffers->d_selected_factor_index,
                          selected_factor_index.data(),
                          sizeof(int) * group_size, cudaMemcpyHostToDevice),
               "copy selected factor index");
    std::vector<int> final_active(group_size, 1);
    check_cuda(cudaMemcpy(buffers->d_active, final_active.data(),
                          sizeof(int) * group_size, cudaMemcpyHostToDevice),
               "copy selected active flags");
    timing->h2d_sec += elapsed_since(stage);

    const int selected_system_blocks = static_cast<int>(
      (static_cast<std::size_t>(selected_factor_count) * p * p + kBlock - 1) /
      kBlock);
    stage = std::chrono::steady_clock::now();
    batched_build_selected_factor_system_kernel<<<
      std::max(1, selected_system_blocks), kBlock>>>(
      buffers->d_design_XtX, buffers->d_design_P,
      buffers->d_selected_factor_design_index, buffers->d_selected_lambda,
      buffers->d_selected_ridge, selected_factor_count, p, buffers->d_A);
    check_cuda(cudaGetLastError(), "launch selected build system kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected build system kernel");
    timing->build_system_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    make_indexed_matrix_pointer_array<<<std::max(1, ptr_blocks), kBlock>>>(
      buffers->d_A, buffers->d_selected_factor_index, group_size, p,
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
    const int beta_blocks = static_cast<int>((vec_size + kBlock - 1) / kBlock);
    batched_copy_xty_to_beta_kernel<<<std::max(1, beta_blocks), kBlock>>>(
      buffers->d_Xty, group_size, p, buffers->d_beta);
    check_cuda(cudaGetLastError(), "launch selected beta copy kernel");
    check_cusolver(cusolverDnDpotrsBatched(
      buffers->solver, CUBLAS_FILL_MODE_UPPER, p, 1, buffers->d_A_ptrs, p,
      buffers->d_beta_ptrs, p, buffers->d_info, group_size),
      "selected batched potrs beta");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected RHS solve");
    double rhs_sec = elapsed_since(stage);
    timing->factor_rhs_solve_sec += rhs_sec;
    timing->factor_solve_sec += rhs_sec;
    timing->rhs_solve_count += group_size;
    timing->rhs_solve_api_calls += 1;
    timing->rhs_target_solves += group_size;

    stage = std::chrono::steady_clock::now();
    const int row_blocks = (n + kBlock - 1) / kBlock;
    const dim3 fit_grid(std::max(1, row_blocks), group_size);
    batched_fitted_residual_kernel<<<fit_grid, kBlock>>>(
      buffers->d_X, buffers->d_y, buffers->d_beta, buffers->d_active,
      group_size, n, p, buffers->d_fitted, buffers->d_residuals);
    check_cuda(cudaGetLastError(), "launch selected residual kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize selected residual kernels");
    timing->residual_summary_sec += elapsed_since(stage);

    std::vector<int> final_info(group_size);
    std::vector<double> fitted(y_size);
    std::vector<double> residuals(y_size);
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(final_info.data(), buffers->d_info,
                          sizeof(int) * group_size, cudaMemcpyDeviceToHost),
               "copy selected info");
    check_cuda(cudaMemcpy(fitted.data(), buffers->d_fitted,
                          sizeof(double) * y_size, cudaMemcpyDeviceToHost),
               "copy selected fitted");
    check_cuda(cudaMemcpy(residuals.data(), buffers->d_residuals,
                          sizeof(double) * y_size, cudaMemcpyDeviceToHost),
               "copy selected residuals");
    timing->d2h_sec += elapsed_since(stage);

    std::vector<FastSplineCudaFit> out(group_size);
    stage = std::chrono::steady_clock::now();
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
      out[fit].fit = value;
      out[fit].diagnostics = make_diagnostics(true, false, "", group.group_id,
                                              fit, true, backend);
    }
    timing->host_select_sec += elapsed_since(stage);
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
  out.host_pack_sec = 0.0;
  out.alloc_sec = 0.0;
  out.h2d_sec = 0.0;
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
  out.host_select_sec = 0.0;
  out.free_sec = 0.0;
  out.true_batch_total_sec = 0.0;
  out.factorization_count = 0;
  out.rhs_solve_count = 0;
  out.inverse_solve_count = 0;
  out.rhs_solve_api_calls = 0;
  out.rhs_target_solves = 0;
  out.winning_factor_reuse_count = 0;
  out.factor_cache_hits = 0;
  out.factor_cache_misses = 0;
  out.factor_cache_entries = 0;
  out.factor_cache_bytes = 0.0;
  out.lambda_candidates = 0;
  out.workspace_reuse_count = 0;
  out.workspace_grow_count = 0;
  out.solver_handle_create_count = 0;
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

std::vector<FastSplineBatchGroup> make_fastspline_batch_groups(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params) {
  if (targets.size() != conditioning_sets.size()) {
    throw std::runtime_error("targets and conditioning_sets length mismatch");
  }
  std::vector<FastSplineBatchGroup> groups;
  std::map<std::string, int> group_by_key;
  std::map<std::string, FastSplineDesign> design_cache;
  std::vector<std::map<std::string, int> > group_design_by_key;
  for (int i = 0; i < static_cast<int>(targets.size()); ++i) {
    if (targets[i] < 0 || targets[i] >= data.ncol()) {
      throw std::runtime_error("target column out of range");
    }
    std::vector<int> normalized_conditioning_set = conditioning_sets[i];
    std::sort(normalized_conditioning_set.begin(),
              normalized_conditioning_set.end());
    const std::string exact_design_key =
      conditioning_set_key(normalized_conditioning_set);

    std::map<std::string, FastSplineDesign>::iterator design_it =
      design_cache.find(exact_design_key);
    if (design_it == design_cache.end()) {
      FastSplineDesign design =
        make_fastspline_design(data, normalized_conditioning_set, params);
      design_it = design_cache.insert(
        std::make_pair(exact_design_key, design)).first;
    }

    FastSplineBatchRequest request;
    request.original_index = i;
    request.target = targets[i];
    request.conditioning_set = normalized_conditioning_set;
    request.design_index = -1;

    const std::string key = group_key(design_it->second, params);
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
  }
  return groups;
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
  if (requested_fits == 0) return result;

  std::chrono::steady_clock::time_point stage =
    std::chrono::steady_clock::now();
  const std::vector<FastSplineBatchGroup> groups =
    make_fastspline_batch_groups(data, targets, conditioning_sets, params);
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
      append_group_diagnostics(&result.diagnostics, group, false, 0, fit_count,
                               "cpu-fallback", "fallback", reason);
      continue;
    }

    try {
      FastSplineCudaBatchDiagnostics group_timing =
        make_empty_batch_diagnostics(fit_count);
      const std::vector<FastSplineCudaFit> group_fits =
        run_true_batched_group(data, group, params, true_backend,
                               &group_timing, workspace);
      add_batch_timing(&result.diagnostics, group_timing);
      for (int i = 0; i < fit_count; ++i) {
        result.fits[group.requests[i].original_index] = group_fits[i];
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
