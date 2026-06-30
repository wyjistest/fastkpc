#include "fastspline_residual_cuda.hpp"
#include "fastspline_batched_solver.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBlock = 256;

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

double nonnegative_gap(double total, double accounted) {
  return std::max(0.0, total - accounted);
}

__global__ void xtx_kernel(const double* X,
                           int n,
                           int p,
                           double* XtX) {
  __shared__ double scratch[kBlock];
  const int a = blockIdx.x;
  const int b = blockIdx.y;
  if (a >= p || b >= p) return;

  double acc = 0.0;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    acc += X[static_cast<std::size_t>(row) * p + a] *
      X[static_cast<std::size_t>(row) * p + b];
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) XtX[a + b * p] = scratch[0];
}

__global__ void xty_kernel(const double* X,
                           const double* y,
                           int n,
                           int p,
                           double* Xty) {
  __shared__ double scratch[kBlock];
  const int col = blockIdx.x;
  if (col >= p) return;

  double acc = 0.0;
  for (int row = threadIdx.x; row < n; row += blockDim.x) {
    acc += X[static_cast<std::size_t>(row) * p + col] * y[row];
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) Xty[col] = scratch[0];
}

__global__ void build_system_kernel(const double* XtX,
                                    const double* P,
                                    int p,
                                    double lambda,
                                    double ridge,
                                    double* A) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = p * p;
  for (int idx = linear; idx < total; idx += gridDim.x * blockDim.x) {
    const int row = idx % p;
    const int col = idx / p;
    double value = XtX[idx] + lambda * P[static_cast<std::size_t>(row) * p + col];
    if (row == col && row > 0) value += ridge;
    A[idx] = value;
  }
}

__global__ void identity_kernel(double* matrix, int p) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = p * p;
  for (int idx = linear; idx < total; idx += gridDim.x * blockDim.x) {
    const int row = idx % p;
    const int col = idx / p;
    matrix[idx] = row == col ? 1.0 : 0.0;
  }
}

__global__ void fitted_residual_kernel(const double* X,
                                       const double* y,
                                       const double* beta,
                                       int n,
                                       int p,
                                       double* fitted,
                                       double* residuals) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  double value = 0.0;
  for (int col = 0; col < p; ++col) {
    value += X[static_cast<std::size_t>(row) * p + col] * beta[col];
  }
  fitted[row] = value;
  residuals[row] = y[row] - value;
}

__global__ void rss_kernel(const double* residuals,
                           int n,
                           double* rss) {
  __shared__ double scratch[kBlock];
  double acc = 0.0;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < n;
       row += gridDim.x * blockDim.x) {
    const double value = residuals[row];
    acc += value * value;
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(rss, scratch[0]);
}

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
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

double edf_from_host_matrices(const std::vector<double>& XtX,
                              const std::vector<double>& A_inv,
                              int p) {
  double edf = 0.0;
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      edf += XtX[i + j * p] * A_inv[j + i * p];
    }
  }
  return edf;
}

bool finite_vec(const std::vector<double>& values) {
  for (double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

FastSplineFit fit_fastspline_residuals_cuda_impl(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const FastSplineParams& params) {
  const FastSplineDesign design = make_fastspline_design(data, conditioning_set, params);
  const std::vector<double> y = response_vector(data, target);
  const std::vector<double> lambdas = lambda_grid(params);
  const int n = design.n;
  const int p = design.p;
  if (n <= 0 || p <= 0) throw std::runtime_error("invalid fastSpline design dimensions");

  double* d_X = nullptr;
  double* d_P = nullptr;
  double* d_y = nullptr;
  double* d_XtX = nullptr;
  double* d_Xty = nullptr;
  double* d_A = nullptr;
  double* d_beta = nullptr;
  double* d_Ainv = nullptr;
  double* d_fitted = nullptr;
  double* d_residuals = nullptr;
  double* d_rss = nullptr;
  int* d_info = nullptr;
  double* d_work = nullptr;
  cusolverDnHandle_t solver = nullptr;

  const std::size_t x_size = static_cast<std::size_t>(n) * p;
  const std::size_t pp_size = static_cast<std::size_t>(p) * p;

  try {
    check_cusolver(cusolverDnCreate(&solver), "create cuSOLVER handle");
    check_cuda(cudaMalloc(&d_X, sizeof(double) * x_size), "alloc X");
    check_cuda(cudaMalloc(&d_P, sizeof(double) * pp_size), "alloc P");
    check_cuda(cudaMalloc(&d_y, sizeof(double) * n), "alloc y");
    check_cuda(cudaMalloc(&d_XtX, sizeof(double) * pp_size), "alloc XtX");
    check_cuda(cudaMalloc(&d_Xty, sizeof(double) * p), "alloc Xty");
    check_cuda(cudaMalloc(&d_A, sizeof(double) * pp_size), "alloc A");
    check_cuda(cudaMalloc(&d_beta, sizeof(double) * p), "alloc beta");
    check_cuda(cudaMalloc(&d_Ainv, sizeof(double) * pp_size), "alloc A inverse");
    check_cuda(cudaMalloc(&d_fitted, sizeof(double) * n), "alloc fitted");
    check_cuda(cudaMalloc(&d_residuals, sizeof(double) * n), "alloc residuals");
    check_cuda(cudaMalloc(&d_rss, sizeof(double)), "alloc rss");
    check_cuda(cudaMalloc(&d_info, sizeof(int)), "alloc cuSOLVER info");

    check_cuda(cudaMemcpy(d_X, design.X.data(), sizeof(double) * x_size,
                          cudaMemcpyHostToDevice), "copy X");
    check_cuda(cudaMemcpy(d_P, design.P.data(), sizeof(double) * pp_size,
                          cudaMemcpyHostToDevice), "copy P");
    check_cuda(cudaMemcpy(d_y, y.data(), sizeof(double) * n,
                          cudaMemcpyHostToDevice), "copy y");

    const dim3 xtx_grid(p, p);
    xtx_kernel<<<xtx_grid, kBlock>>>(d_X, n, p, d_XtX);
    xty_kernel<<<p, kBlock>>>(d_X, d_y, n, p, d_Xty);
    check_cuda(cudaGetLastError(), "launch XtX/Xty kernels");
    check_cuda(cudaDeviceSynchronize(), "synchronize XtX/Xty kernels");

    int lwork = 0;
    check_cusolver(cusolverDnDpotrf_bufferSize(
      solver, CUBLAS_FILL_MODE_UPPER, p, d_A, p, &lwork), "potrf buffer size");
    check_cuda(cudaMalloc(&d_work, sizeof(double) * std::max(1, lwork)),
               "alloc potrf workspace");

    std::vector<double> host_XtX(pp_size);
    check_cuda(cudaMemcpy(host_XtX.data(), d_XtX, sizeof(double) * pp_size,
                          cudaMemcpyDeviceToHost), "copy XtX to host");

    bool found = false;
    double best_gcv = std::numeric_limits<double>::infinity();
    double best_lambda = std::numeric_limits<double>::quiet_NaN();
    double best_rss = std::numeric_limits<double>::quiet_NaN();
    double best_edf = std::numeric_limits<double>::quiet_NaN();
    int best_ridge_attempts = 0;
    std::vector<double> best_residuals;
    std::vector<double> best_fitted;

    double ridge = params.ridge;
    for (int ridge_attempt = 0; ridge <= 1e-4 * (1.0 + 1e-12); ++ridge_attempt) {
      for (double lambda : lambdas) {
        const int system_blocks = static_cast<int>((pp_size + kBlock - 1) / kBlock);
        build_system_kernel<<<std::max(1, system_blocks), kBlock>>>(
          d_XtX, d_P, p, lambda, ridge, d_A);
        check_cuda(cudaGetLastError(), "launch build system kernel");

        check_cusolver(cusolverDnDpotrf(
          solver, CUBLAS_FILL_MODE_UPPER, p, d_A, p, d_work, lwork, d_info),
          "potrf");
        int info = 0;
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "copy potrf info");
        if (info != 0) continue;

        check_cuda(cudaMemcpy(d_beta, d_Xty, sizeof(double) * p,
                              cudaMemcpyDeviceToDevice), "copy Xty to beta");
        check_cusolver(cusolverDnDpotrs(
          solver, CUBLAS_FILL_MODE_UPPER, p, 1, d_A, p, d_beta, p, d_info),
          "potrs beta");
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "copy potrs beta info");
        if (info != 0) continue;

        identity_kernel<<<std::max(1, system_blocks), kBlock>>>(d_Ainv, p);
        check_cuda(cudaGetLastError(), "launch identity kernel");
        check_cusolver(cusolverDnDpotrs(
          solver, CUBLAS_FILL_MODE_UPPER, p, p, d_A, p, d_Ainv, p, d_info),
          "potrs inverse");
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "copy potrs inverse info");
        if (info != 0) continue;

        const int row_blocks = (n + kBlock - 1) / kBlock;
        fitted_residual_kernel<<<std::max(1, row_blocks), kBlock>>>(
          d_X, d_y, d_beta, n, p, d_fitted, d_residuals);
        check_cuda(cudaGetLastError(), "launch fitted residual kernel");
        check_cuda(cudaMemset(d_rss, 0, sizeof(double)), "zero rss");
        rss_kernel<<<std::max(1, row_blocks), kBlock>>>(d_residuals, n, d_rss);
        check_cuda(cudaGetLastError(), "launch rss kernel");
        check_cuda(cudaDeviceSynchronize(), "synchronize residual kernels");

        std::vector<double> host_Ainv(pp_size);
        std::vector<double> fitted(n);
        std::vector<double> residuals(n);
        double rss = std::numeric_limits<double>::quiet_NaN();
        check_cuda(cudaMemcpy(host_Ainv.data(), d_Ainv, sizeof(double) * pp_size,
                              cudaMemcpyDeviceToHost), "copy inverse to host");
        check_cuda(cudaMemcpy(fitted.data(), d_fitted, sizeof(double) * n,
                              cudaMemcpyDeviceToHost), "copy fitted to host");
        check_cuda(cudaMemcpy(residuals.data(), d_residuals, sizeof(double) * n,
                              cudaMemcpyDeviceToHost), "copy residuals to host");
        check_cuda(cudaMemcpy(&rss, d_rss, sizeof(double), cudaMemcpyDeviceToHost),
                   "copy rss to host");

        const double edf = edf_from_host_matrices(host_XtX, host_Ainv, p);
        const double denom = static_cast<double>(n) - edf;
        if (!std::isfinite(rss) || !std::isfinite(edf) || denom <= 1e-8 ||
            !finite_vec(fitted) || !finite_vec(residuals)) {
          continue;
        }
        const double gcv = static_cast<double>(n) * rss / (denom * denom);
        if (!std::isfinite(gcv)) continue;

        if (!found || gcv < best_gcv ||
            (std::abs(gcv - best_gcv) <= 1e-14 && lambda < best_lambda)) {
          found = true;
          best_gcv = gcv;
          best_lambda = lambda;
          best_rss = rss;
          best_edf = edf;
          best_ridge_attempts = ridge_attempt;
          best_fitted = fitted;
          best_residuals = residuals;
        }
      }
      if (found) break;
      ridge *= 100.0;
      if (ridge <= 0.0) ridge = 1e-8;
    }

    if (!found) throw std::runtime_error("no finite CUDA fastSpline solve");

    FastSplineFit fit;
    fit.residuals = best_residuals;
    fit.fitted = best_fitted;
    fit.selected_lambda = best_lambda;
    fit.gcv = best_gcv;
    fit.rss = best_rss;
    fit.edf = best_edf;
    fit.design_cols = p;
    fit.ridge_attempts = best_ridge_attempts;

    cudaFree(d_X);
    cudaFree(d_P);
    cudaFree(d_y);
    cudaFree(d_XtX);
    cudaFree(d_Xty);
    cudaFree(d_A);
    cudaFree(d_beta);
    cudaFree(d_Ainv);
    cudaFree(d_fitted);
    cudaFree(d_residuals);
    cudaFree(d_rss);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(solver);
    return fit;
  } catch (...) {
    cudaFree(d_X);
    cudaFree(d_P);
    cudaFree(d_y);
    cudaFree(d_XtX);
    cudaFree(d_Xty);
    cudaFree(d_A);
    cudaFree(d_beta);
    cudaFree(d_Ainv);
    cudaFree(d_fitted);
    cudaFree(d_residuals);
    cudaFree(d_rss);
    cudaFree(d_info);
    cudaFree(d_work);
    if (solver != nullptr) cusolverDnDestroy(solver);
    throw;
  }
}

FastSplineCudaFit fallback_fit(const Rcpp::NumericMatrix& data,
                               int target,
                               const std::vector<int>& conditioning_set,
                               const FastSplineParams& params,
                               const std::string& reason) {
  FastSplineCudaFit out;
  out.fit = fit_fastspline_residuals(data, target, conditioning_set, params);
  out.diagnostics.cuda_used = false;
  out.diagnostics.fallback_used = true;
  out.diagnostics.reason = reason;
  out.diagnostics.batch_group_id = -1;
  out.diagnostics.batch_position = 0;
  out.diagnostics.true_batched = false;
  out.diagnostics.cholesky_backend = "cpu-fallback";
  return out;
}

}  // namespace

FastSplineCudaFit fit_fastspline_residuals_cuda(
  const Rcpp::NumericMatrix& data,
  int target,
  const std::vector<int>& conditioning_set,
  const FastSplineParams& params,
  bool fallback) {
  try {
    FastSplineCudaFit out;
    out.fit = fit_fastspline_residuals_cuda_impl(data, target, conditioning_set, params);
    out.diagnostics.cuda_used = true;
    out.diagnostics.fallback_used = false;
    out.diagnostics.reason = "";
    out.diagnostics.batch_group_id = -1;
    out.diagnostics.batch_position = 0;
    out.diagnostics.true_batched = false;
    out.diagnostics.cholesky_backend = "single-fit-cusolver";
    return out;
  } catch (const std::exception& e) {
    if (fallback) return fallback_fit(data, target, conditioning_set, params, e.what());
    throw std::runtime_error(std::string("CUDA fastSpline residual fit failed: ") +
                             e.what());
  }
}

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_batch_result(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace) {
  const std::chrono::steady_clock::time_point call_start =
    std::chrono::steady_clock::now();
  try {
    FastSplineCudaBatchResult result = fit_fastspline_residuals_cuda_true_batch(
      data, targets, conditioning_sets, params, fallback, workspace);
    const double wall = elapsed_since(call_start);
    result.diagnostics.residual_batch_top_level_wall_sec += wall;
    const double accounted =
      result.diagnostics.grouping_sec +
      result.diagnostics.true_batch_total_sec;
    result.diagnostics.residual_batch_top_level_unaccounted_sec +=
      nonnegative_gap(wall, accounted);
    return result;
  } catch (const std::exception& e) {
    if (!fallback) {
      throw std::runtime_error(std::string("CUDA fastSpline residual batch failed: ") +
                               e.what());
    }

    FastSplineCudaBatchResult result;
    result.fits.reserve(targets.size());
    result.diagnostics.requested_fits = static_cast<int>(targets.size());
    result.diagnostics.groups = targets.empty() ? 0 : 1;
    result.diagnostics.true_batched_groups = 0;
    result.diagnostics.true_batched_fits = 0;
    result.diagnostics.single_fit_calls = 0;
    result.diagnostics.cpu_fallback_fits = static_cast<int>(targets.size());
    result.diagnostics.unique_designs = 0;
    result.diagnostics.duplicate_design_fits = 0;
    result.diagnostics.max_fits_per_design = 0;
    result.diagnostics.max_group_size = static_cast<int>(targets.size());
    result.diagnostics.min_group_size = static_cast<int>(targets.size());
    result.diagnostics.cholesky_backend = "cpu-fallback";
    result.diagnostics.batch_mode = targets.empty() ? "empty" : "fallback";
    result.diagnostics.grouping_sec = 0.0;
    result.diagnostics.host_pack_sec = 0.0;
    result.diagnostics.alloc_sec = 0.0;
    result.diagnostics.h2d_sec = 0.0;
    result.diagnostics.xtx_xty_sec = 0.0;
    result.diagnostics.pointer_setup_sec = 0.0;
    result.diagnostics.active_copy_sec = 0.0;
    result.diagnostics.build_system_sec = 0.0;
    result.diagnostics.factor_solve_sec = 0.0;
    result.diagnostics.factor_cholesky_sec = 0.0;
    result.diagnostics.factor_rhs_solve_sec = 0.0;
    result.diagnostics.factor_inverse_solve_sec = 0.0;
    result.diagnostics.residual_summary_sec = 0.0;
    result.diagnostics.d2h_sec = 0.0;
    result.diagnostics.d2h_residuals_sec = 0.0;
    result.diagnostics.d2h_metadata_sec = 0.0;
    result.diagnostics.d2h_info_sec = 0.0;
    result.diagnostics.d2h_copy_count = 0;
    result.diagnostics.d2h_bytes = 0.0;
    result.diagnostics.d2h_residual_bytes = 0.0;
    result.diagnostics.d2h_metadata_bytes = 0.0;
    result.diagnostics.d2h_metadata_coalesced_count = 0;
    result.diagnostics.d2h_metadata_coalesced_bytes = 0.0;
    result.diagnostics.host_select_sec = 0.0;
    result.diagnostics.free_sec = 0.0;
    result.diagnostics.true_batch_total_sec = 0.0;
    result.diagnostics.factorization_count = 0;
    result.diagnostics.rhs_solve_count = 0;
    result.diagnostics.inverse_solve_count = 0;
    result.diagnostics.rhs_solve_api_calls = 0;
    result.diagnostics.rhs_target_solves = 0;
    result.diagnostics.rhs_custom_solve_count = 0;
    result.diagnostics.rhs_cublas_solve_count = 0;
    result.diagnostics.rhs_solve_fallback_count = 0;
    result.diagnostics.rhs_custom_solve_sec = 0.0;
    result.diagnostics.rhs_cublas_solve_sec = 0.0;
    result.diagnostics.candidate_rhs_fused_solve_count = 0;
    result.diagnostics.candidate_rhs_materialized_solve_count = 0;
    result.diagnostics.selected_rhs_materialized_solve_count = 0;
    result.diagnostics.candidate_beta_values_avoided = 0;
    result.diagnostics.summary_candidate_launch_count = 0;
    result.diagnostics.summary_group_batched_launch_count = 0;
    result.diagnostics.summary_group_batched_candidate_count = 0;
    result.diagnostics.winning_factor_reuse_count = 0;
    result.diagnostics.factor_cache_hits = 0;
    result.diagnostics.factor_cache_misses = 0;
    result.diagnostics.factor_cache_entries = 0;
    result.diagnostics.factor_cache_bytes = 0.0;
    result.diagnostics.lambda_candidates = 0;
    result.diagnostics.workspace_reuse_count = 0;
    result.diagnostics.workspace_grow_count = 0;
    result.diagnostics.workspace_slab_grow_count = 0;
    result.diagnostics.workspace_slab_reuse_count = 0;
    result.diagnostics.workspace_slab_bytes = 0.0;
    result.diagnostics.workspace_legacy_alloc_count = 0;
    result.diagnostics.solver_handle_create_count = 0;
    result.diagnostics.per_request_design_x_values = 0;
    result.diagnostics.duplicate_design_x_values_avoided = 0;
    result.diagnostics.algebraic_rss_count = 0;
    result.diagnostics.candidate_residual_materialize_count = 0;
    result.diagnostics.winning_residual_materialize_count = 0;
    result.diagnostics.algebraic_rss_clamp_count = 0;
    result.diagnostics.residual_only_batch_count = 0;
    result.diagnostics.residual_full_fit_batch_count = targets.empty() ? 0 : 1;
    result.diagnostics.residual_only_fit_count = 0;
    result.diagnostics.residual_full_fit_materialize_count =
      static_cast<int>(targets.size());
    result.diagnostics.residual_fitted_values_avoided = 0;
    result.diagnostics.residual_result_materialize_sec = 0.0;
    result.diagnostics.residual_fitted_materialize_sec = 0.0;
    result.diagnostics.residual_batch_top_level_wall_sec =
      elapsed_since(call_start);
    result.diagnostics.residual_batch_top_level_unaccounted_sec =
      result.diagnostics.residual_batch_top_level_wall_sec;
    if (!targets.empty()) {
      result.diagnostics.group_id.push_back(0);
      result.diagnostics.group_n.push_back(data.nrow());
      result.diagnostics.group_design_cols.push_back(-1);
      result.diagnostics.group_fit_count.push_back(static_cast<int>(targets.size()));
      result.diagnostics.group_true_batched.push_back(0);
      result.diagnostics.group_single_fit_calls.push_back(0);
      result.diagnostics.group_cpu_fallback_fits.push_back(static_cast<int>(targets.size()));
      result.diagnostics.group_unique_designs.push_back(0);
      result.diagnostics.group_duplicate_design_fits.push_back(0);
      result.diagnostics.group_max_fits_per_design.push_back(0);
      result.diagnostics.group_cholesky_backend.push_back("cpu-fallback");
      result.diagnostics.group_status.push_back("fallback");
      result.diagnostics.group_reason.push_back(e.what());
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
      FastSplineCudaFit fit;
      fit.fit = fit_fastspline_residuals(data, targets[i], conditioning_sets[i], params);
      fit.diagnostics.cuda_used = false;
      fit.diagnostics.fallback_used = true;
      fit.diagnostics.reason = e.what();
      fit.diagnostics.batch_group_id = 0;
      fit.diagnostics.batch_position = static_cast<int>(i);
      fit.diagnostics.true_batched = false;
      fit.diagnostics.cholesky_backend = "cpu-fallback";
      result.fits.push_back(fit);
    }
    return result;
  }
}

FastSplineCudaResidualOnlyBatchResult
fit_fastspline_residuals_cuda_batch_residuals_only(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback,
  FastSplineCudaWorkspace* workspace) {
  const std::chrono::steady_clock::time_point call_start =
    std::chrono::steady_clock::now();
  try {
    FastSplineCudaResidualOnlyBatchResult result =
      fit_fastspline_residuals_cuda_true_batch_residuals_only(
        data, targets, conditioning_sets, params, fallback, workspace);
    const double wall = elapsed_since(call_start);
    result.diagnostics.residual_batch_top_level_wall_sec += wall;
    const double accounted =
      result.diagnostics.grouping_sec +
      result.diagnostics.true_batch_total_sec;
    result.diagnostics.residual_batch_top_level_unaccounted_sec +=
      nonnegative_gap(wall, accounted);
    return result;
  } catch (const std::exception& e) {
    if (!fallback) {
      throw std::runtime_error(std::string("CUDA fastSpline residual batch failed: ") +
                               e.what());
    }

    FastSplineCudaResidualOnlyBatchResult result;
    result.fits.reserve(targets.size());
    result.diagnostics = FastSplineCudaBatchDiagnostics();
    result.diagnostics.requested_fits = static_cast<int>(targets.size());
    result.diagnostics.groups = targets.empty() ? 0 : 1;
    result.diagnostics.true_batched_groups = 0;
    result.diagnostics.true_batched_fits = 0;
    result.diagnostics.single_fit_calls = 0;
    result.diagnostics.cpu_fallback_fits = static_cast<int>(targets.size());
    result.diagnostics.unique_designs = 0;
    result.diagnostics.duplicate_design_fits = 0;
    result.diagnostics.max_fits_per_design = 0;
    result.diagnostics.max_group_size = static_cast<int>(targets.size());
    result.diagnostics.min_group_size = static_cast<int>(targets.size());
    result.diagnostics.cholesky_backend = "cpu-fallback";
    result.diagnostics.batch_mode = targets.empty() ? "empty" : "fallback";
    result.diagnostics.residual_only_batch_count = targets.empty() ? 0 : 1;
    result.diagnostics.residual_full_fit_batch_count = 0;
    result.diagnostics.residual_only_fit_count = static_cast<int>(targets.size());
    result.diagnostics.residual_full_fit_materialize_count = 0;
    result.diagnostics.residual_fitted_values_avoided = 0;
    result.diagnostics.residual_batch_top_level_wall_sec =
      elapsed_since(call_start);
    result.diagnostics.residual_batch_top_level_unaccounted_sec =
      result.diagnostics.residual_batch_top_level_wall_sec;
    if (!targets.empty()) {
      result.diagnostics.group_id.push_back(0);
      result.diagnostics.group_n.push_back(data.nrow());
      result.diagnostics.group_design_cols.push_back(-1);
      result.diagnostics.group_fit_count.push_back(
        static_cast<int>(targets.size()));
      result.diagnostics.group_true_batched.push_back(0);
      result.diagnostics.group_single_fit_calls.push_back(0);
      result.diagnostics.group_cpu_fallback_fits.push_back(
        static_cast<int>(targets.size()));
      result.diagnostics.group_unique_designs.push_back(0);
      result.diagnostics.group_duplicate_design_fits.push_back(0);
      result.diagnostics.group_max_fits_per_design.push_back(0);
      result.diagnostics.group_cholesky_backend.push_back("cpu-fallback");
      result.diagnostics.group_status.push_back("fallback");
      result.diagnostics.group_reason.push_back(e.what());
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
      FastSplineFit fit =
        fit_fastspline_residuals(data, targets[i], conditioning_sets[i],
                                 params);
      FastSplineCudaResidualOnlyFit residual_fit;
      residual_fit.residuals = std::move(fit.residuals);
      residual_fit.diagnostics.cuda_used = false;
      residual_fit.diagnostics.fallback_used = true;
      residual_fit.diagnostics.reason = e.what();
      residual_fit.diagnostics.batch_group_id = 0;
      residual_fit.diagnostics.batch_position = static_cast<int>(i);
      residual_fit.diagnostics.true_batched = false;
      residual_fit.diagnostics.cholesky_backend = "cpu-fallback";
      result.fits.push_back(std::move(residual_fit));
    }
    return result;
  }
}

FastSplineCudaBatchResult fit_fastspline_residuals_cuda_batch_result(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback) {
  return fit_fastspline_residuals_cuda_batch_result(
    data, targets, conditioning_sets, params, fallback, nullptr);
}

std::vector<FastSplineCudaFit> fit_fastspline_residuals_cuda_batch(
  const Rcpp::NumericMatrix& data,
  const std::vector<int>& targets,
  const std::vector<std::vector<int> >& conditioning_sets,
  const FastSplineParams& params,
  bool fallback) {
  if (targets.size() != conditioning_sets.size()) {
    throw std::runtime_error("targets and conditioning_sets length mismatch");
  }
  return fit_fastspline_residuals_cuda_batch_result(
    data, targets, conditioning_sets, params, fallback).fits;
}
