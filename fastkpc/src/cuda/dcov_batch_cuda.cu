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

__global__ void rowsum_raw_kernel(const double* x,
                                  const double* y,
                                  int n,
                                  int batch,
                                  double index,
                                  bool legacy_index,
                                  double* row_k,
                                  double* row_l,
                                  double* total_k,
                                  double* total_l,
                                  double* scalars) {
  __shared__ double sx_s[kBlock];
  __shared__ double sy_s[kBlock];
  __shared__ double raw_ab_s[kBlock];
  __shared__ double raw_aa_s[kBlock];
  __shared__ double raw_bb_s[kBlock];

  const int row = blockIdx.x;
  const int task = blockIdx.y;
  if (row >= n || task >= batch) return;

  double sx = 0.0;
  double sy = 0.0;
  double raw_ab = 0.0;
  double raw_aa = 0.0;
  double raw_bb = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const double a = pair_dist_1d(x, n, task, row, j, index, legacy_index);
    const double b = pair_dist_1d(y, n, task, row, j, index, legacy_index);
    sx += a;
    sy += b;
    raw_ab += a * b;
    raw_aa += a * a;
    raw_bb += b * b;
  }

  sx_s[threadIdx.x] = sx;
  sy_s[threadIdx.x] = sy;
  raw_ab_s[threadIdx.x] = raw_ab;
  raw_aa_s[threadIdx.x] = raw_aa;
  raw_bb_s[threadIdx.x] = raw_bb;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sx_s[threadIdx.x] += sx_s[threadIdx.x + stride];
      sy_s[threadIdx.x] += sy_s[threadIdx.x + stride];
      raw_ab_s[threadIdx.x] += raw_ab_s[threadIdx.x + stride];
      raw_aa_s[threadIdx.x] += raw_aa_s[threadIdx.x + stride];
      raw_bb_s[threadIdx.x] += raw_bb_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const std::size_t row_base = static_cast<std::size_t>(task) * n + row;
    const std::size_t scalar_base = static_cast<std::size_t>(task) * 6;
    row_k[row_base] = sx_s[0];
    row_l[row_base] = sy_s[0];
    atomicAdd(&total_k[task], sx_s[0]);
    atomicAdd(&total_l[task], sy_s[0]);
    atomicAdd(&scalars[scalar_base + 0], raw_ab_s[0]);
    atomicAdd(&scalars[scalar_base + 1], raw_aa_s[0]);
    atomicAdd(&scalars[scalar_base + 2], raw_bb_s[0]);
  }
}

__global__ void row_product_reduce_kernel(const double* row_k,
                                          const double* row_l,
                                          int n,
                                          int batch,
                                          double* scalars) {
  __shared__ double row_ab_s[kBlock];
  __shared__ double row_aa_s[kBlock];
  __shared__ double row_bb_s[kBlock];

  const int task = blockIdx.x;
  if (task >= batch) return;

  const double* rk = row_k + static_cast<std::size_t>(task) * n;
  const double* rl = row_l + static_cast<std::size_t>(task) * n;

  double row_ab = 0.0;
  double row_aa = 0.0;
  double row_bb = 0.0;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    row_ab += rk[i] * rl[i];
    row_aa += rk[i] * rk[i];
    row_bb += rl[i] * rl[i];
  }

  row_ab_s[threadIdx.x] = row_ab;
  row_aa_s[threadIdx.x] = row_aa;
  row_bb_s[threadIdx.x] = row_bb;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      row_ab_s[threadIdx.x] += row_ab_s[threadIdx.x + stride];
      row_aa_s[threadIdx.x] += row_aa_s[threadIdx.x + stride];
      row_bb_s[threadIdx.x] += row_bb_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const std::size_t base = static_cast<std::size_t>(task) * 6;
    scalars[base + 3] = row_ab_s[0];
    scalars[base + 4] = row_aa_s[0];
    scalars[base + 5] = row_bb_s[0];
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
  out->workspace_reuse_count += value.workspace_reuse_count;
  out->workspace_grow_count += value.workspace_grow_count;
  out->raw_aggregate_fused_count += value.raw_aggregate_fused_count;
  out->row_product_reduce_count += value.row_product_reduce_count;
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

struct DcovChunkBuffers {
  double* d_x = nullptr;
  std::size_t d_x_capacity = 0;
  double* d_y = nullptr;
  std::size_t d_y_capacity = 0;
  double* d_row_k = nullptr;
  std::size_t d_row_k_capacity = 0;
  double* d_row_l = nullptr;
  std::size_t d_row_l_capacity = 0;
  double* d_total_k = nullptr;
  std::size_t d_total_k_capacity = 0;
  double* d_total_l = nullptr;
  std::size_t d_total_l_capacity = 0;
  double* d_scalars = nullptr;
  std::size_t d_scalars_capacity = 0;
  std::vector<double> h_total_k;
  std::vector<double> h_total_l;
  std::vector<double> h_device_scalars;
};

template <typename T>
void ensure_device_capacity(T** ptr,
                            std::size_t* capacity,
                            std::size_t required,
                            const char* stage,
                            DcovBatchResult* result,
                            bool track_workspace_grow) {
  if (required <= *capacity) return;
  if (*ptr != nullptr) check_cuda(cudaFree(*ptr), stage);
  *ptr = nullptr;
  *capacity = 0;
  if (required > 0) {
    check_cuda(cudaMalloc(ptr, sizeof(T) * required), stage);
    *capacity = required;
  }
  if (track_workspace_grow && result != nullptr) {
    ++result->workspace_grow_count;
  }
}

void ensure_host_capacity(std::vector<double>* values, std::size_t required) {
  if (values->size() < required) values->resize(required);
}

void ensure_dcov_buffers(DcovChunkBuffers* buffers,
                         std::size_t matrix_size,
                         std::size_t row_size,
                         int batch,
                         std::size_t device_scalar_size,
                         DcovBatchResult* result,
                         bool track_workspace_grow) {
  ensure_device_capacity(&buffers->d_x, &buffers->d_x_capacity, matrix_size,
                         "alloc x", result, track_workspace_grow);
  ensure_device_capacity(&buffers->d_y, &buffers->d_y_capacity, matrix_size,
                         "alloc y", result, track_workspace_grow);
  ensure_device_capacity(&buffers->d_row_k, &buffers->d_row_k_capacity,
                         row_size, "alloc row K", result,
                         track_workspace_grow);
  ensure_device_capacity(&buffers->d_row_l, &buffers->d_row_l_capacity,
                         row_size, "alloc row L", result,
                         track_workspace_grow);
  ensure_device_capacity(&buffers->d_total_k, &buffers->d_total_k_capacity,
                         static_cast<std::size_t>(batch), "alloc total K",
                         result, track_workspace_grow);
  ensure_device_capacity(&buffers->d_total_l, &buffers->d_total_l_capacity,
                         static_cast<std::size_t>(batch), "alloc total L",
                         result, track_workspace_grow);
  ensure_device_capacity(&buffers->d_scalars, &buffers->d_scalars_capacity,
                         device_scalar_size, "alloc scalars", result,
                         track_workspace_grow);
  ensure_host_capacity(&buffers->h_total_k, static_cast<std::size_t>(batch));
  ensure_host_capacity(&buffers->h_total_l, static_cast<std::size_t>(batch));
  ensure_host_capacity(&buffers->h_device_scalars, device_scalar_size);
}

void free_dcov_buffers(DcovChunkBuffers* buffers) {
  cudaFree(buffers->d_x);
  buffers->d_x = nullptr;
  buffers->d_x_capacity = 0;
  cudaFree(buffers->d_y);
  buffers->d_y = nullptr;
  buffers->d_y_capacity = 0;
  cudaFree(buffers->d_row_k);
  buffers->d_row_k = nullptr;
  buffers->d_row_k_capacity = 0;
  cudaFree(buffers->d_row_l);
  buffers->d_row_l = nullptr;
  buffers->d_row_l_capacity = 0;
  cudaFree(buffers->d_total_k);
  buffers->d_total_k = nullptr;
  buffers->d_total_k_capacity = 0;
  cudaFree(buffers->d_total_l);
  buffers->d_total_l = nullptr;
  buffers->d_total_l_capacity = 0;
  cudaFree(buffers->d_scalars);
  buffers->d_scalars = nullptr;
  buffers->d_scalars_capacity = 0;
}

}  // namespace

struct DcovCudaWorkspace {
  DcovChunkBuffers buffers;
};

namespace {

DcovBatchResult dcov_batch_cuda_chunk(const double* x,
                                      const double* y,
                                      int n,
                                      int batch,
                                      const DcovBatchOptions& options,
                                      DcovCudaWorkspace* workspace) {
  if (n <= 5) throw std::runtime_error("gamma approximation requires n > 5");
  if (batch < 1) throw std::runtime_error("batch must be positive");

  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();

  const std::size_t matrix_size = static_cast<std::size_t>(n) * batch;
  const std::size_t row_size = matrix_size;
  const std::size_t raw_scalar_size = static_cast<std::size_t>(batch) * 5;
  const std::size_t device_scalar_size = static_cast<std::size_t>(batch) * 6;
  DcovBatchResult result;
  result.chunks = 1;
  result.max_chunk_batch = batch;
  DcovChunkBuffers local_buffers;
  DcovChunkBuffers* buffers = workspace == nullptr ?
    &local_buffers : &workspace->buffers;
  const bool has_workspace = workspace != nullptr;
  if (has_workspace) ++result.workspace_reuse_count;

  try {
    std::chrono::steady_clock::time_point stage =
      std::chrono::steady_clock::now();
    ensure_dcov_buffers(buffers, matrix_size, row_size, batch,
                        device_scalar_size, &result, has_workspace);
    result.alloc_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(buffers->d_x, x, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy x");
    check_cuda(cudaMemcpy(buffers->d_y, y, sizeof(double) * matrix_size, cudaMemcpyHostToDevice),
               "copy y");
    result.h2d_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemset(buffers->d_total_k, 0, sizeof(double) * batch),
               "zero total K");
    check_cuda(cudaMemset(buffers->d_total_l, 0, sizeof(double) * batch),
               "zero total L");
    check_cuda(cudaMemset(buffers->d_scalars, 0,
                          sizeof(double) * device_scalar_size), "zero scalars");
    result.memset_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    const dim3 rowsum_grid(n, batch);
    rowsum_raw_kernel<<<rowsum_grid, kBlock>>>(
      buffers->d_x, buffers->d_y, n, batch, options.index,
      options.legacy_index, buffers->d_row_k, buffers->d_row_l,
      buffers->d_total_k, buffers->d_total_l, buffers->d_scalars);
    check_cuda(cudaGetLastError(), "launch rowsum raw aggregate");
    check_cuda(cudaDeviceSynchronize(), "rowsum raw aggregate synchronize");
    result.raw_aggregate_fused_count += batch;
    result.rowsum_sec += elapsed_since(stage);

    double* total_k = buffers->h_total_k.data();
    double* total_l = buffers->h_total_l.data();
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(total_k, buffers->d_total_k, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total K");
    check_cuda(cudaMemcpy(total_l, buffers->d_total_l, sizeof(double) * batch,
                          cudaMemcpyDeviceToHost), "copy total L");
    result.totals_d2h_sec += elapsed_since(stage);

    stage = std::chrono::steady_clock::now();
    row_product_reduce_kernel<<<batch, kBlock>>>(
      buffers->d_row_k, buffers->d_row_l, n, batch, buffers->d_scalars);
    check_cuda(cudaGetLastError(), "launch row product reduce");
    check_cuda(cudaDeviceSynchronize(), "row product reduce synchronize");
    result.row_product_reduce_count += batch;
    result.reduce_sec += elapsed_since(stage);

    result.p_values.assign(batch, 0.0);
    result.nV2.assign(batch, 0.0);
    result.means.assign(batch, 0.0);
    result.variances.assign(batch, 0.0);
    result.raw_scalars.assign(raw_scalar_size, 0.0);
    double* device_scalars = buffers->h_device_scalars.data();
    stage = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(device_scalars, buffers->d_scalars,
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
    if (!has_workspace) free_dcov_buffers(buffers);
    result.free_sec += elapsed_since(stage);
    result.total_sec += elapsed_since(total_start);
    return result;
  } catch (...) {
    if (!has_workspace) free_dcov_buffers(buffers);
    throw;
  }
}

}  // namespace

DcovCudaWorkspace* create_dcov_cuda_workspace() {
  return new DcovCudaWorkspace();
}

void destroy_dcov_cuda_workspace(DcovCudaWorkspace* workspace) {
  if (workspace == nullptr) return;
  free_dcov_buffers(&workspace->buffers);
  delete workspace;
}

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options,
                                DcovCudaWorkspace* workspace) {
  if (n <= 5) throw std::runtime_error("gamma approximation requires n > 5");
  if (batch < 1) throw std::runtime_error("batch must be positive");

  const int chunk_limit = dcov_max_grid_batch_dimension();
  if (batch <= chunk_limit) {
    return dcov_batch_cuda_chunk(x, y, n, batch, options, workspace);
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
      x + offset, y + offset, n, count, options, workspace);
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

DcovBatchResult dcov_batch_cuda(const double* x,
                                const double* y,
                                int n,
                                int batch,
                                const DcovBatchOptions& options) {
  return dcov_batch_cuda(x, y, n, batch, options, nullptr);
}
