#include "legacy_dcov_spectra_matvec_cuda.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace fastkpc {
namespace {

constexpr int kBlock = 256;

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + stage + "): " +
                             cudaGetErrorString(err));
  }
}

double elapsed_ms_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

__global__ void dense_sym_matvec_kernel(const double* matrix,
                                        const double* rhs,
                                        int n,
                                        double* output) {
  __shared__ double partial[kBlock];
  const int row = blockIdx.x;
  if (row >= n) return;

  double sum = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    sum += matrix[row + static_cast<std::size_t>(j) * n] * rhs[j];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) output[row] = partial[0];
}

}  // namespace

LegacyDcovSpectraMatvecCudaResult legacy_dcov_spectra_matvec_cuda(
    const double* matrix,
    const double* rhs,
    int n,
    int rhs_count) {
  if (matrix == nullptr) throw std::runtime_error("matrix pointer is null");
  if (rhs == nullptr) throw std::runtime_error("rhs pointer is null");
  if (n <= 0) throw std::runtime_error("matrix dimension must be positive");
  if (rhs_count <= 0) throw std::runtime_error("rhs count must be positive");

  const auto total_start = std::chrono::steady_clock::now();
  const std::size_t matrix_size = static_cast<std::size_t>(n) * n;
  const std::size_t rhs_size = static_cast<std::size_t>(n) * rhs_count;

  double* d_matrix = nullptr;
  double* d_rhs = nullptr;
  double* d_output = nullptr;
  LegacyDcovSpectraMatvecCudaResult result;
  result.n = n;
  result.rhs_count = rhs_count;
  result.values.assign(rhs_size, 0.0);

  try {
    auto stage = std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(&d_matrix, sizeof(double) * matrix_size),
               "alloc dense matvec matrix");
    check_cuda(cudaMalloc(&d_rhs, sizeof(double) * rhs_size),
               "alloc dense matvec rhs");
    check_cuda(cudaMalloc(&d_output, sizeof(double) * rhs_size),
               "alloc dense matvec output");
    result.alloc_ms += elapsed_ms_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(d_matrix, matrix, sizeof(double) * matrix_size,
                          cudaMemcpyHostToDevice),
               "copy dense matvec matrix");
    check_cuda(cudaMemcpy(d_rhs, rhs, sizeof(double) * rhs_size,
                          cudaMemcpyHostToDevice),
               "copy dense matvec rhs");
    result.h2d_ms += elapsed_ms_since(stage);

    stage = std::chrono::steady_clock::now();
    for (int k = 0; k < rhs_count; ++k) {
      const double* rhs_col = d_rhs + static_cast<std::size_t>(k) * n;
      double* output_col = d_output + static_cast<std::size_t>(k) * n;
      dense_sym_matvec_kernel<<<n, kBlock>>>(d_matrix, rhs_col, n, output_col);
      check_cuda(cudaGetLastError(), "launch dense symmetric matvec");
      ++result.kernel_launch_count;
    }
    check_cuda(cudaDeviceSynchronize(), "dense symmetric matvec synchronize");
    result.kernel_ms += elapsed_ms_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(result.values.data(), d_output,
                          sizeof(double) * rhs_size,
                          cudaMemcpyDeviceToHost),
               "copy dense matvec output");
    result.d2h_ms += elapsed_ms_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaFree(d_matrix), "free dense matvec matrix");
    check_cuda(cudaFree(d_rhs), "free dense matvec rhs");
    check_cuda(cudaFree(d_output), "free dense matvec output");
    d_matrix = nullptr;
    d_rhs = nullptr;
    d_output = nullptr;
    result.free_ms += elapsed_ms_since(stage);
  } catch (...) {
    if (d_matrix != nullptr) cudaFree(d_matrix);
    if (d_rhs != nullptr) cudaFree(d_rhs);
    if (d_output != nullptr) cudaFree(d_output);
    throw;
  }

  result.total_ms = elapsed_ms_since(total_start);
  return result;
}

}  // namespace fastkpc
