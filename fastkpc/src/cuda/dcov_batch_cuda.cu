#include "dcov_batch_cuda.hpp"

#include <Rmath.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
                                           double* scalars) {
  __shared__ double sab_s[kBlock];
  __shared__ double saa_s[kBlock];
  __shared__ double sbb_s[kBlock];
  __shared__ double row_ab_s[kBlock];
  __shared__ double row_aa_s[kBlock];
  __shared__ double row_bb_s[kBlock];

  const int task = blockIdx.z;
  if (task >= batch) return;

  const double* rk = row_k + static_cast<std::size_t>(task) * n;
  const double* rl = row_l + static_cast<std::size_t>(task) * n;

  double sab = 0.0;
  double saa = 0.0;
  double sbb = 0.0;
  double row_ab = 0.0;
  double row_aa = 0.0;
  double row_bb = 0.0;

  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      row_ab += rk[i] * rl[i];
      row_aa += rk[i] * rk[i];
      row_bb += rl[i] * rl[i];
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < n;
         j += gridDim.x * blockDim.x) {
      const double a = pair_dist_1d(x, n, task, i, j, index, legacy_index);
      const double b = pair_dist_1d(y, n, task, i, j, index, legacy_index);
      sab += a * b;
      saa += a * a;
      sbb += b * b;
    }
  }

  sab_s[threadIdx.x] = sab;
  saa_s[threadIdx.x] = saa;
  sbb_s[threadIdx.x] = sbb;
  row_ab_s[threadIdx.x] = row_ab;
  row_aa_s[threadIdx.x] = row_aa;
  row_bb_s[threadIdx.x] = row_bb;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sab_s[threadIdx.x] += sab_s[threadIdx.x + stride];
      saa_s[threadIdx.x] += saa_s[threadIdx.x + stride];
      sbb_s[threadIdx.x] += sbb_s[threadIdx.x + stride];
      row_ab_s[threadIdx.x] += row_ab_s[threadIdx.x + stride];
      row_aa_s[threadIdx.x] += row_aa_s[threadIdx.x + stride];
      row_bb_s[threadIdx.x] += row_bb_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const std::size_t base = static_cast<std::size_t>(task) * 6;
    atomicAdd(&scalars[base + 0], sab_s[0]);
    atomicAdd(&scalars[base + 1], saa_s[0]);
    atomicAdd(&scalars[base + 2], sbb_s[0]);
    atomicAdd(&scalars[base + 3], row_ab_s[0]);
    atomicAdd(&scalars[base + 4], row_aa_s[0]);
    atomicAdd(&scalars[base + 5], row_bb_s[0]);
  }
}

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + stage + "): " +
                             cudaGetErrorString(err));
  }
}

double elapsed_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
}

void add_timing(DcovBatchResult* out, const DcovBatchResult& value) {
  out->alloc_sec += value.alloc_sec;
  out->h2d_sec += value.h2d_sec;
  out->memset_sec += value.memset_sec;
  out->rowsum_sec += value.rowsum_sec;
  out->totals_d2h_sec += value.totals_d2h_sec;
  out->reduce_sec += value.reduce_sec;
  out->scalars_d2h_sec += value.scalars_d2h_sec;
  out->host_scalar_sec += value.host_scalar_sec;
  out->free_sec += value.free_sec;
  out->total_sec += value.total_sec;
  out->chunks += value.chunks;
  out->max_chunk_batch = std::max(out->max_chunk_batch, value.max_chunk_batch);
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

  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();

  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_row_k = nullptr;
  double* d_row_l = nullptr;
  double* d_total_k = nullptr;
  double* d_total_l = nullptr;
  double* d_scalars = nullptr;

  const std::size_t matrix_size = static_cast<std::size_t>(n) * batch;
  const std::size_t row_size = matrix_size;
  const std::size_t raw_scalar_size = static_cast<std::size_t>(batch) * 5;
  const std::size_t device_scalar_size = static_cast<std::size_t>(batch) * 6;
  DcovBatchResult result;
  result.chunks = 1;
  result.max_chunk_batch = batch;

  try {
    std::chrono::steady_clock::time_point stage =
      std::chrono::steady_clock::now();
    check_cuda(cudaMalloc(&d_x, sizeof(double) * matrix_size), "alloc x");
    check_cuda(cudaMalloc(&d_y, sizeof(double) * matrix_size), "alloc y");
    check_cuda(cudaMalloc(&d_row_k, sizeof(double) * row_size), "alloc row K");
    check_cuda(cudaMalloc(&d_row_l, sizeof(double) * row_size), "alloc row L");
    check_cuda(cudaMalloc(&d_total_k, sizeof(double) * batch), "alloc total K");
    check_cuda(cudaMalloc(&d_total_l, sizeof(double) * batch), "alloc total L");
    check_cuda(cudaMalloc(&d_scalars, sizeof(double) * device_scalar_size),
               "alloc scalars");
    result.alloc_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(d_x, x, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy x");
    check_cuda(cudaMemcpy(d_y, y, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy y");
    result.h2d_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemset(d_total_k, 0, sizeof(double) * batch), "zero total K");
    check_cuda(cudaMemset(d_total_l, 0, sizeof(double) * batch), "zero total L");
    check_cuda(cudaMemset(d_scalars, 0, sizeof(double) * device_scalar_size),
               "zero scalars");
    result.memset_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    const dim3 rowsum_grid(n, batch);
    rowsum_kernel<<<rowsum_grid, kBlock>>>(d_x, n, batch, options.index,
                                           options.legacy_index, d_row_k, d_total_k);
    rowsum_kernel<<<rowsum_grid, kBlock>>>(d_y, n, batch, options.index,
                                           options.legacy_index, d_row_l, d_total_l);
    check_cuda(cudaGetLastError(), "launch rowsum");
    check_cuda(cudaDeviceSynchronize(), "rowsum synchronize");
    result.rowsum_sec += elapsed_since(stage);

    std::vector<double> total_k(batch);
    std::vector<double> total_l(batch);
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(total_k.data(), d_total_k, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total K");
    check_cuda(cudaMemcpy(total_l.data(), d_total_l, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total L");
    result.totals_d2h_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    const int gy = n < 1024 ? n : 1024;
    int gx = 2048 / gy;
    if (gx < 1) gx = 1;
    const dim3 reduce_grid(gx, gy, batch);
    fused_center_reduce_kernel<<<reduce_grid, kBlock>>>(
      d_x, d_y, n, batch, options.index, options.legacy_index, d_row_k, d_row_l,
      d_scalars);
    check_cuda(cudaGetLastError(), "launch fused reduce");
    check_cuda(cudaDeviceSynchronize(), "fused reduce synchronize");
    result.reduce_sec += elapsed_since(stage);

    result.p_values.assign(batch, 0.0);
    result.nV2.assign(batch, 0.0);
    result.means.assign(batch, 0.0);
    result.variances.assign(batch, 0.0);
    result.raw_scalars.assign(raw_scalar_size, 0.0);
    std::vector<double> device_scalars(device_scalar_size, 0.0);
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(device_scalars.data(), d_scalars,
                          sizeof(double) * device_scalar_size,
                          cudaMemcpyDeviceToHost),
               "copy scalars");
    result.scalars_d2h_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    for (int task = 0; task < batch; ++task) {
      const double nd = static_cast<double>(n);
      const std::size_t device_base = static_cast<std::size_t>(task) * 6;
      const double raw_ab = device_scalars[device_base + 0];
      const double raw_aa = device_scalars[device_base + 1];
      const double raw_bb = device_scalars[device_base + 2];
      const double row_ab = device_scalars[device_base + 3];
      const double row_aa = device_scalars[device_base + 4];
      const double row_bb = device_scalars[device_base + 5];
      const double total_ab = total_k[task] * total_l[task];
      const double total_aa = total_k[task] * total_k[task];
      const double total_bb = total_l[task] * total_l[task];
      const double sab = raw_ab - 2.0 * row_ab / nd + total_ab / (nd * nd);
      const double saa = raw_aa - 2.0 * row_aa / nd + total_aa / (nd * nd);
      const double sbb = raw_bb - 2.0 * row_bb / nd + total_bb / (nd * nd);
      const std::size_t base = static_cast<std::size_t>(task) * 5;
      result.raw_scalars[base + 0] = sab;
      result.raw_scalars[base + 1] = saa;
      result.raw_scalars[base + 2] = sbb;
      result.raw_scalars[base + 3] = total_k[task];
      result.raw_scalars[base + 4] = total_l[task];

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
    result.host_scalar_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_k);
    cudaFree(d_row_l);
    cudaFree(d_total_k);
    cudaFree(d_total_l);
    cudaFree(d_scalars);
    result.free_sec += elapsed_since(stage);
    result.total_sec += elapsed_since(total_start);
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
  result.chunks = 0;
  result.max_chunk_batch = 0;

  for (int start = 0; start < batch; start += chunk_limit) {
    const int count = std::min(chunk_limit, batch - start);
    const std::size_t offset = static_cast<std::size_t>(start) * n;
    const DcovBatchResult chunk = dcov_batch_cuda_chunk(
      x + offset, y + offset, n, count, options);
    add_timing(&result, chunk);

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
