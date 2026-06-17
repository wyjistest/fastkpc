#include "dcov_batch_cuda.hpp"

#include <Rmath.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ double pair_dist_1d(const double* values,
                                               int n,
                                               int task,
                                               int i,
                                               int j,
                                               double index,
                                               bool legacy_index) {
  const double* column = values + static_cast<std::size_t>(task) * n;
  double dist = fabs(column[i] - column[j]);
  if (!legacy_index && index != 1.0) dist = pow(dist, index);
  return dist;
}

__global__ void rowsum_kernel(const double* values,
                              int n,
                              int batch,
                              double index,
                              bool legacy_index,
                              double* rowsums,
                              double* totals) {
  __shared__ double scratch[kBlock];
  const int row = blockIdx.x;
  const int task = blockIdx.y;
  if (row >= n || task >= batch) return;

  double acc = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    acc += pair_dist_1d(values, n, task, row, j, index, legacy_index);
  }
  scratch[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    rowsums[static_cast<std::size_t>(task) * n + row] = scratch[0];
    atomicAdd(&totals[task], scratch[0]);
  }
}

__global__ void fused_center_reduce_kernel(const double* x,
                                           const double* y,
                                           int n,
                                           int batch,
                                           double index,
                                           bool legacy_index,
                                           const double* row_k,
                                           const double* row_l,
                                           const double* total_k,
                                           const double* total_l,
                                           double* scalars) {
  __shared__ double sab_s[kBlock];
  __shared__ double saa_s[kBlock];
  __shared__ double sbb_s[kBlock];

  const int task = blockIdx.z;
  if (task >= batch) return;

  const double inv_n = 1.0 / static_cast<double>(n);
  const double grand_k = total_k[task] * inv_n * inv_n;
  const double grand_l = total_l[task] * inv_n * inv_n;
  const double* rk = row_k + static_cast<std::size_t>(task) * n;
  const double* rl = row_l + static_cast<std::size_t>(task) * n;

  double sab = 0.0;
  double saa = 0.0;
  double sbb = 0.0;

  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    const double left_k = grand_k - rk[i] * inv_n;
    const double left_l = grand_l - rl[i] * inv_n;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < n;
         j += gridDim.x * blockDim.x) {
      const double a = pair_dist_1d(x, n, task, i, j, index, legacy_index) -
        rk[j] * inv_n + left_k;
      const double b = pair_dist_1d(y, n, task, i, j, index, legacy_index) -
        rl[j] * inv_n + left_l;
      sab += a * b;
      saa += a * a;
      sbb += b * b;
    }
  }

  sab_s[threadIdx.x] = sab;
  saa_s[threadIdx.x] = saa;
  sbb_s[threadIdx.x] = sbb;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sab_s[threadIdx.x] += sab_s[threadIdx.x + stride];
      saa_s[threadIdx.x] += saa_s[threadIdx.x + stride];
      sbb_s[threadIdx.x] += sbb_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&scalars[static_cast<std::size_t>(task) * 5 + 0], sab_s[0]);
    atomicAdd(&scalars[static_cast<std::size_t>(task) * 5 + 1], saa_s[0]);
    atomicAdd(&scalars[static_cast<std::size_t>(task) * 5 + 2], sbb_s[0]);
  }
}

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + stage + "): " +
                             cudaGetErrorString(err));
  }
}

int dcov_max_grid_batch_dimension() {
  int device = 0;
  cudaDeviceProp prop;
  check_cuda(cudaGetDevice(&device), "get device");
  check_cuda(cudaGetDeviceProperties(&prop, device), "get device properties");
  const int y_limit = prop.maxGridSize[1];
  const int z_limit = prop.maxGridSize[2];
  const int limit = std::min(y_limit, z_limit);
  return std::max(1, limit);
}

DcovBatchResult dcov_batch_cuda_chunk(const double* x,
                                      const double* y,
                                      int n,
                                      int batch,
                                      const DcovBatchOptions& options) {
  if (n <= 5) throw std::runtime_error("gamma approximation requires n > 5");
  if (batch < 1) throw std::runtime_error("batch must be positive");

  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_row_k = nullptr;
  double* d_row_l = nullptr;
  double* d_total_k = nullptr;
  double* d_total_l = nullptr;
  double* d_scalars = nullptr;

  const std::size_t matrix_size = static_cast<std::size_t>(n) * batch;
  const std::size_t row_size = matrix_size;
  const std::size_t scalar_size = static_cast<std::size_t>(batch) * 5;

  try {
    check_cuda(cudaMalloc(&d_x, sizeof(double) * matrix_size), "alloc x");
    check_cuda(cudaMalloc(&d_y, sizeof(double) * matrix_size), "alloc y");
    check_cuda(cudaMalloc(&d_row_k, sizeof(double) * row_size), "alloc row K");
    check_cuda(cudaMalloc(&d_row_l, sizeof(double) * row_size), "alloc row L");
    check_cuda(cudaMalloc(&d_total_k, sizeof(double) * batch), "alloc total K");
    check_cuda(cudaMalloc(&d_total_l, sizeof(double) * batch), "alloc total L");
    check_cuda(cudaMalloc(&d_scalars, sizeof(double) * scalar_size), "alloc scalars");

    check_cuda(cudaMemcpy(d_x, x, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy x");
    check_cuda(cudaMemcpy(d_y, y, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy y");
    check_cuda(cudaMemset(d_total_k, 0, sizeof(double) * batch), "zero total K");
    check_cuda(cudaMemset(d_total_l, 0, sizeof(double) * batch), "zero total L");
    check_cuda(cudaMemset(d_scalars, 0, sizeof(double) * scalar_size), "zero scalars");

    const dim3 rowsum_grid(n, batch);
    rowsum_kernel<<<rowsum_grid, kBlock>>>(d_x, n, batch, options.index,
                                           options.legacy_index, d_row_k, d_total_k);
    rowsum_kernel<<<rowsum_grid, kBlock>>>(d_y, n, batch, options.index,
                                           options.legacy_index, d_row_l, d_total_l);
    check_cuda(cudaGetLastError(), "launch rowsum");
    check_cuda(cudaDeviceSynchronize(), "rowsum synchronize");

    std::vector<double> total_k(batch);
    std::vector<double> total_l(batch);
    check_cuda(cudaMemcpy(total_k.data(), d_total_k, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total K");
    check_cuda(cudaMemcpy(total_l.data(), d_total_l, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total L");

    const int gy = n < 1024 ? n : 1024;
    int gx = 2048 / gy;
    if (gx < 1) gx = 1;
    const dim3 reduce_grid(gx, gy, batch);
    fused_center_reduce_kernel<<<reduce_grid, kBlock>>>(
      d_x, d_y, n, batch, options.index, options.legacy_index, d_row_k, d_row_l,
      d_total_k, d_total_l, d_scalars);
    check_cuda(cudaGetLastError(), "launch fused reduce");
    check_cuda(cudaDeviceSynchronize(), "fused reduce synchronize");

    DcovBatchResult result;
    result.p_values.assign(batch, 0.0);
    result.nV2.assign(batch, 0.0);
    result.means.assign(batch, 0.0);
    result.variances.assign(batch, 0.0);
    result.raw_scalars.assign(scalar_size, 0.0);
    check_cuda(cudaMemcpy(result.raw_scalars.data(), d_scalars,
                          sizeof(double) * scalar_size, cudaMemcpyDeviceToHost),
               "copy scalars");

    for (int task = 0; task < batch; ++task) {
      const std::size_t base = static_cast<std::size_t>(task) * 5;
      const double sab = result.raw_scalars[base + 0];
      const double saa = result.raw_scalars[base + 1];
      const double sbb = result.raw_scalars[base + 2];
      result.raw_scalars[base + 3] = total_k[task];
      result.raw_scalars[base + 4] = total_l[task];

      const double nd = static_cast<double>(n);
      const double nV2 = sab / nd;
      const double mean = (total_k[task] / (nd * nd)) *
        (total_l[task] / (nd * nd));
      const double variance = 2.0 * (nd - 4.0) * (nd - 5.0) /
        nd / (nd - 1.0) / (nd - 2.0) / (nd - 3.0) * saa * sbb / (nd * nd);
      const double alpha = mean * mean / variance;
      const double scale = variance / mean;
      const double p = Rf_pgamma(nV2, alpha, scale, false, false);

      result.nV2[task] = nV2;
      result.means[task] = mean;
      result.variances[task] = variance;
      result.p_values[task] = p;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_k);
    cudaFree(d_row_l);
    cudaFree(d_total_k);
    cudaFree(d_total_l);
    cudaFree(d_scalars);
    return result;
  } catch (...) {
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_k);
    cudaFree(d_row_l);
    cudaFree(d_total_k);
    cudaFree(d_total_l);
    cudaFree(d_scalars);
    throw;
  }
}

}  // namespace

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options) {
  if (n <= 5) throw std::runtime_error("gamma approximation requires n > 5");
  if (batch < 1) throw std::runtime_error("batch must be positive");

  const int chunk_limit = dcov_max_grid_batch_dimension();
  if (batch <= chunk_limit) {
    return dcov_batch_cuda_chunk(x, y, n, batch, options);
  }

  DcovBatchResult result;
  result.p_values.assign(batch, 0.0);
  result.nV2.assign(batch, 0.0);
  result.means.assign(batch, 0.0);
  result.variances.assign(batch, 0.0);
  result.raw_scalars.assign(static_cast<std::size_t>(batch) * 5, 0.0);

  for (int start = 0; start < batch; start += chunk_limit) {
    const int count = std::min(chunk_limit, batch - start);
    const std::size_t offset = static_cast<std::size_t>(start) * n;
    const DcovBatchResult chunk = dcov_batch_cuda_chunk(
      x + offset, y + offset, n, count, options);

    for (int k = 0; k < count; ++k) {
      const int dest = start + k;
      result.p_values[dest] = chunk.p_values[k];
      result.nV2[dest] = chunk.nV2[k];
      result.means[dest] = chunk.means[k];
      result.variances[dest] = chunk.variances[k];
      for (int s = 0; s < 5; ++s) {
        result.raw_scalars[static_cast<std::size_t>(dest) * 5 + s] =
          chunk.raw_scalars[static_cast<std::size_t>(k) * 5 + s];
      }
    }
  }

  return result;
}
