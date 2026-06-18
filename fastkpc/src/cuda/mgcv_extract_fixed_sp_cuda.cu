#include "mgcv_extract_fixed_sp_cuda.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void build_system_kernel(const double* XtX,
                                    const double* P,
                                    int q,
                                    double* A) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = q * q;
  for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
    A[i] = XtX[i] + P[i];
  }
}

__global__ void beta_from_nullspace_kernel(const double* Z,
                                           const double* theta,
                                           int p,
                                           int q,
                                           double* beta) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= p) return;
  double value = 0.0;
  for (int col = 0; col < q; ++col) {
    value += Z[row + col * p] * theta[col];
  }
  beta[row] = value;
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
    value += X[row + col * n] * beta[col];
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

void require_finite(const double* values, int n, const char* name) {
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error(std::string(name) + " contains missing or infinite values");
    }
  }
}

}  // namespace

MgcvExtractGpuFixedSpResult mgcv_extract_fixed_sp_solve_cuda(
  const double* X,
  int n,
  int coefficient_dim,
  const double* y,
  const double* Z,
  const double* XtX_null,
  const double* penalty_null,
  const double* Xty_null,
  int null_dim) {
  if (n <= 0) throw std::runtime_error("n must be positive");
  if (coefficient_dim <= 0) throw std::runtime_error("coefficient_dim must be positive");
  if (null_dim <= 0) throw std::runtime_error("null_dim must be positive");

  require_finite(X, n * coefficient_dim, "X");
  require_finite(y, n, "y");
  require_finite(Z, coefficient_dim * null_dim, "Z");
  require_finite(XtX_null, null_dim * null_dim, "XtX_null");
  require_finite(penalty_null, null_dim * null_dim, "penalty_null");
  require_finite(Xty_null, null_dim, "Xty_null");

  double* d_X = nullptr;
  double* d_y = nullptr;
  double* d_Z = nullptr;
  double* d_XtX = nullptr;
  double* d_P = nullptr;
  double* d_A = nullptr;
  double* d_theta = nullptr;
  double* d_beta = nullptr;
  double* d_fitted = nullptr;
  double* d_residuals = nullptr;
  double* d_rss = nullptr;
  int* d_info = nullptr;
  double* d_work = nullptr;
  cusolverDnHandle_t solver = nullptr;

  const std::size_t x_size = static_cast<std::size_t>(n) * coefficient_dim;
  const std::size_t z_size = static_cast<std::size_t>(coefficient_dim) * null_dim;
  const std::size_t qq_size = static_cast<std::size_t>(null_dim) * null_dim;

  try {
    check_cusolver(cusolverDnCreate(&solver), "create cuSOLVER handle");
    check_cuda(cudaMalloc(&d_X, sizeof(double) * x_size), "alloc X");
    check_cuda(cudaMalloc(&d_y, sizeof(double) * n), "alloc y");
    check_cuda(cudaMalloc(&d_Z, sizeof(double) * z_size), "alloc Z");
    check_cuda(cudaMalloc(&d_XtX, sizeof(double) * qq_size), "alloc XtX");
    check_cuda(cudaMalloc(&d_P, sizeof(double) * qq_size), "alloc penalty");
    check_cuda(cudaMalloc(&d_A, sizeof(double) * qq_size), "alloc system");
    check_cuda(cudaMalloc(&d_theta, sizeof(double) * null_dim), "alloc theta");
    check_cuda(cudaMalloc(&d_beta, sizeof(double) * coefficient_dim), "alloc beta");
    check_cuda(cudaMalloc(&d_fitted, sizeof(double) * n), "alloc fitted");
    check_cuda(cudaMalloc(&d_residuals, sizeof(double) * n), "alloc residuals");
    check_cuda(cudaMalloc(&d_rss, sizeof(double)), "alloc rss");
    check_cuda(cudaMalloc(&d_info, sizeof(int)), "alloc cuSOLVER info");

    check_cuda(cudaMemcpy(d_X, X, sizeof(double) * x_size,
                          cudaMemcpyHostToDevice), "copy X");
    check_cuda(cudaMemcpy(d_y, y, sizeof(double) * n,
                          cudaMemcpyHostToDevice), "copy y");
    check_cuda(cudaMemcpy(d_Z, Z, sizeof(double) * z_size,
                          cudaMemcpyHostToDevice), "copy Z");
    check_cuda(cudaMemcpy(d_XtX, XtX_null, sizeof(double) * qq_size,
                          cudaMemcpyHostToDevice), "copy XtX");
    check_cuda(cudaMemcpy(d_P, penalty_null, sizeof(double) * qq_size,
                          cudaMemcpyHostToDevice), "copy penalty");
    check_cuda(cudaMemcpy(d_theta, Xty_null, sizeof(double) * null_dim,
                          cudaMemcpyHostToDevice), "copy rhs");

    const int matrix_blocks =
      std::max(1, static_cast<int>((qq_size + kBlock - 1) / kBlock));
    build_system_kernel<<<matrix_blocks, kBlock>>>(d_XtX, d_P, null_dim, d_A);
    check_cuda(cudaGetLastError(), "launch build system");
    check_cuda(cudaDeviceSynchronize(), "sync build system");

    int work_size = 0;
    check_cusolver(cusolverDnDpotrf_bufferSize(
                     solver, CUBLAS_FILL_MODE_UPPER, null_dim, d_A, null_dim,
                     &work_size),
                   "cuSOLVER potrf buffer");
    check_cuda(cudaMalloc(&d_work, sizeof(double) * work_size), "alloc cuSOLVER work");
    check_cusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, null_dim,
                                    d_A, null_dim, d_work, work_size, d_info),
                   "cuSOLVER potrf");
    int info = 0;
    check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "copy potrf info");
    if (info != 0) {
      throw std::runtime_error("cuSOLVER potrf failed with info " +
                               std::to_string(info));
    }

    check_cusolver(cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_UPPER, null_dim, 1,
                                    d_A, null_dim, d_theta, null_dim, d_info),
                   "cuSOLVER potrs");
    check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
               "copy potrs info");
    if (info != 0) {
      throw std::runtime_error("cuSOLVER potrs failed with info " +
                               std::to_string(info));
    }

    const int beta_blocks =
      std::max(1, (coefficient_dim + kBlock - 1) / kBlock);
    beta_from_nullspace_kernel<<<beta_blocks, kBlock>>>(
      d_Z, d_theta, coefficient_dim, null_dim, d_beta);
    check_cuda(cudaGetLastError(), "launch beta from nullspace");

    const int row_blocks = std::max(1, (n + kBlock - 1) / kBlock);
    fitted_residual_kernel<<<row_blocks, kBlock>>>(
      d_X, d_y, d_beta, n, coefficient_dim, d_fitted, d_residuals);
    check_cuda(cudaGetLastError(), "launch fitted residual");

    check_cuda(cudaMemset(d_rss, 0, sizeof(double)), "zero rss");
    rss_kernel<<<row_blocks, kBlock>>>(d_residuals, n, d_rss);
    check_cuda(cudaGetLastError(), "launch rss");
    check_cuda(cudaDeviceSynchronize(), "sync mgcvExtractGPU fixed-sp solve");

    MgcvExtractGpuFixedSpResult out;
    out.theta.resize(null_dim);
    out.coefficients.resize(coefficient_dim);
    out.fitted.resize(n);
    out.residuals.resize(n);
    out.rss = 0.0;
    out.n = n;
    out.coefficient_dim = coefficient_dim;
    out.null_dim = null_dim;
    out.cholesky_backend = "cusolver-potrf-potrs";

    check_cuda(cudaMemcpy(out.theta.data(), d_theta, sizeof(double) * null_dim,
                          cudaMemcpyDeviceToHost), "copy theta");
    check_cuda(cudaMemcpy(out.coefficients.data(), d_beta,
                          sizeof(double) * coefficient_dim,
                          cudaMemcpyDeviceToHost), "copy beta");
    check_cuda(cudaMemcpy(out.fitted.data(), d_fitted, sizeof(double) * n,
                          cudaMemcpyDeviceToHost), "copy fitted");
    check_cuda(cudaMemcpy(out.residuals.data(), d_residuals, sizeof(double) * n,
                          cudaMemcpyDeviceToHost), "copy residuals");
    check_cuda(cudaMemcpy(&out.rss, d_rss, sizeof(double),
                          cudaMemcpyDeviceToHost), "copy rss");

    if (solver != nullptr) cusolverDnDestroy(solver);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_Z);
    cudaFree(d_XtX);
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_theta);
    cudaFree(d_beta);
    cudaFree(d_fitted);
    cudaFree(d_residuals);
    cudaFree(d_rss);
    cudaFree(d_info);
    cudaFree(d_work);
    return out;
  } catch (...) {
    if (solver != nullptr) cusolverDnDestroy(solver);
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_Z);
    cudaFree(d_XtX);
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_theta);
    cudaFree(d_beta);
    cudaFree(d_fitted);
    cudaFree(d_residuals);
    cudaFree(d_rss);
    cudaFree(d_info);
    cudaFree(d_work);
    throw;
  }
}
