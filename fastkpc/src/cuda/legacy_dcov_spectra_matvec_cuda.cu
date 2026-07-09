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

LegacyDcovSpectraMatvecCudaResult apply_dense_sym_matvec_device_matrix(
    const double* d_matrix,
    const double* rhs,
    int n,
    int rhs_count,
    double** d_rhs_workspace = nullptr,
    double** d_output_workspace = nullptr,
    std::size_t* workspace_capacity = nullptr) {
  if (d_matrix == nullptr) {
    throw std::runtime_error("CUDA matvec handle has been freed");
  }
  if (rhs == nullptr) throw std::runtime_error("rhs pointer is null");
  if (n <= 0) throw std::runtime_error("matrix dimension must be positive");
  if (rhs_count <= 0) throw std::runtime_error("rhs count must be positive");

  const auto total_start = std::chrono::steady_clock::now();
  const std::size_t rhs_size = static_cast<std::size_t>(n) * rhs_count;

  const bool use_workspace = d_rhs_workspace != nullptr &&
                             d_output_workspace != nullptr &&
                             workspace_capacity != nullptr;
  double* local_d_rhs = nullptr;
  double* local_d_output = nullptr;
  double*& d_rhs = use_workspace ? *d_rhs_workspace : local_d_rhs;
  double*& d_output = use_workspace ? *d_output_workspace : local_d_output;
  LegacyDcovSpectraMatvecCudaResult result;
  result.n = n;
  result.rhs_count = rhs_count;
  result.values.assign(rhs_size, 0.0);

  try {
    auto stage = std::chrono::steady_clock::now();
    if (!use_workspace) {
      stage = std::chrono::steady_clock::now();
      check_cuda(cudaMalloc(&d_rhs, sizeof(double) * rhs_size),
                 "alloc dense matvec rhs");
      check_cuda(cudaMalloc(&d_output, sizeof(double) * rhs_size),
                 "alloc dense matvec output");
      result.alloc_ms += elapsed_ms_since(stage);
    } else if (*workspace_capacity < rhs_size) {
      stage = std::chrono::steady_clock::now();
      double* new_d_rhs = nullptr;
      double* new_d_output = nullptr;
      check_cuda(cudaMalloc(&new_d_rhs, sizeof(double) * rhs_size),
                 "alloc dense matvec rhs workspace");
      const cudaError_t output_alloc =
        cudaMalloc(&new_d_output, sizeof(double) * rhs_size);
      if (output_alloc != cudaSuccess) {
        cudaFree(new_d_rhs);
        check_cuda(output_alloc, "alloc dense matvec output workspace");
      }
      if (d_rhs != nullptr) {
        check_cuda(cudaFree(d_rhs), "free dense matvec rhs workspace");
      }
      if (d_output != nullptr) {
        check_cuda(cudaFree(d_output), "free dense matvec output workspace");
      }
      d_rhs = new_d_rhs;
      d_output = new_d_output;
      *workspace_capacity = rhs_size;
      result.workspace_realloc_count = 1;
      result.workspace_alloc_ms = elapsed_ms_since(stage);
      result.alloc_ms += result.workspace_alloc_ms;
    } else {
      result.device_workspace_reuse_count = 1;
    }
    if (use_workspace) {
      result.workspace_bytes = sizeof(double) * (*workspace_capacity) * 2;
    }

    stage = std::chrono::steady_clock::now();
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

    if (!use_workspace) {
      stage = std::chrono::steady_clock::now();
      check_cuda(cudaFree(d_rhs), "free dense matvec rhs");
      check_cuda(cudaFree(d_output), "free dense matvec output");
      d_rhs = nullptr;
      d_output = nullptr;
      result.free_ms += elapsed_ms_since(stage);
    }
  } catch (...) {
    if (!use_workspace) {
      if (d_rhs != nullptr) cudaFree(d_rhs);
      if (d_output != nullptr) cudaFree(d_output);
    }
    throw;
  }

  result.total_ms = elapsed_ms_since(total_start);
  return result;
}

}  // namespace

struct LegacyDcovSpectraMatvecCudaHandle {
  double* d_matrix = nullptr;
  double* d_rhs = nullptr;
  double* d_output = nullptr;
  int n = 0;
  std::size_t matrix_size = 0;
  std::size_t workspace_capacity = 0;
  double matrix_h2d_ms = 0.0;
};

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

  double* d_matrix = nullptr;
  LegacyDcovSpectraMatvecCudaResult result;
  result.n = n;
  result.rhs_count = rhs_count;
  result.matrix_bytes = sizeof(double) * matrix_size;

  try {
    auto stage = std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(&d_matrix, sizeof(double) * matrix_size),
               "alloc dense matvec matrix");
    result.alloc_ms += elapsed_ms_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(d_matrix, matrix, sizeof(double) * matrix_size,
                          cudaMemcpyHostToDevice),
               "copy dense matvec matrix");
    result.matrix_h2d_ms = elapsed_ms_since(stage);
    result.h2d_ms += result.matrix_h2d_ms;

    LegacyDcovSpectraMatvecCudaResult applied =
      apply_dense_sym_matvec_device_matrix(d_matrix, rhs, n, rhs_count);
    result.values.swap(applied.values);
    result.kernel_launch_count = applied.kernel_launch_count;
    result.alloc_ms += applied.alloc_ms;
    result.h2d_ms += applied.h2d_ms;
    result.kernel_ms = applied.kernel_ms;
    result.d2h_ms = applied.d2h_ms;
    result.free_ms = applied.free_ms;

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaFree(d_matrix), "free dense matvec matrix");
    d_matrix = nullptr;
    result.free_ms += elapsed_ms_since(stage);
  } catch (...) {
    if (d_matrix != nullptr) cudaFree(d_matrix);
    throw;
  }

  result.total_ms = elapsed_ms_since(total_start);
  return result;
}

LegacyDcovSpectraMatvecCudaHandle*
legacy_dcov_spectra_matvec_cuda_handle_create(
    const double* matrix,
    int n) {
  if (matrix == nullptr) throw std::runtime_error("matrix pointer is null");
  if (n <= 0) throw std::runtime_error("matrix dimension must be positive");

  LegacyDcovSpectraMatvecCudaHandle* handle =
    new LegacyDcovSpectraMatvecCudaHandle();
  handle->n = n;
  handle->matrix_size = static_cast<std::size_t>(n) * n;

  try {
    check_cuda(cudaMalloc(&handle->d_matrix,
                          sizeof(double) * handle->matrix_size),
               "alloc dense matvec handle matrix");
    const auto stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(handle->d_matrix, matrix,
                          sizeof(double) * handle->matrix_size,
                          cudaMemcpyHostToDevice),
               "copy dense matvec handle matrix");
    handle->matrix_h2d_ms = elapsed_ms_since(stage);
  } catch (...) {
    if (handle->d_matrix != nullptr) cudaFree(handle->d_matrix);
    delete handle;
    throw;
  }

  return handle;
}

void legacy_dcov_spectra_matvec_cuda_handle_destroy(
    LegacyDcovSpectraMatvecCudaHandle* handle) {
  if (handle == nullptr) return;
  if (handle->d_matrix != nullptr) {
    cudaFree(handle->d_matrix);
    handle->d_matrix = nullptr;
  }
  if (handle->d_rhs != nullptr) {
    cudaFree(handle->d_rhs);
    handle->d_rhs = nullptr;
  }
  if (handle->d_output != nullptr) {
    cudaFree(handle->d_output);
    handle->d_output = nullptr;
  }
  delete handle;
}

int legacy_dcov_spectra_matvec_cuda_handle_n(
    const LegacyDcovSpectraMatvecCudaHandle* handle) {
  if (handle == nullptr || handle->d_matrix == nullptr) {
    throw std::runtime_error("CUDA matvec handle has been freed");
  }
  return handle->n;
}

std::size_t legacy_dcov_spectra_matvec_cuda_handle_matrix_bytes(
    const LegacyDcovSpectraMatvecCudaHandle* handle) {
  if (handle == nullptr || handle->d_matrix == nullptr) {
    throw std::runtime_error("CUDA matvec handle has been freed");
  }
  return sizeof(double) * handle->matrix_size;
}

double legacy_dcov_spectra_matvec_cuda_handle_matrix_h2d_ms(
    const LegacyDcovSpectraMatvecCudaHandle* handle) {
  if (handle == nullptr || handle->d_matrix == nullptr) {
    throw std::runtime_error("CUDA matvec handle has been freed");
  }
  return handle->matrix_h2d_ms;
}

LegacyDcovSpectraMatvecCudaResult
legacy_dcov_spectra_matvec_cuda_handle_apply(
    LegacyDcovSpectraMatvecCudaHandle* handle,
    const double* rhs,
    int rhs_count) {
  if (handle == nullptr || handle->d_matrix == nullptr) {
    throw std::runtime_error("CUDA matvec handle has been freed");
  }
  LegacyDcovSpectraMatvecCudaResult result =
    apply_dense_sym_matvec_device_matrix(handle->d_matrix, rhs, handle->n,
                                         rhs_count, &handle->d_rhs,
                                         &handle->d_output,
                                         &handle->workspace_capacity);
  result.device_matrix_reuse_count = 1;
  result.matrix_bytes = sizeof(double) * handle->matrix_size;
  result.matrix_h2d_ms = 0.0;
  result.workspace_bytes = sizeof(double) * handle->workspace_capacity * 2;
  return result;
}

}  // namespace fastkpc
