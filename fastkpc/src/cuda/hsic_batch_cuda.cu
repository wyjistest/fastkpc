#include "hsic_batch_cuda.hpp"

#include <Rmath.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlock = 256;

__global__ void hsic_rbf_kernel(const double* values,
                                int n,
                                int pairs,
                                double sigma,
                                double* gram) {
  const int total = n * n * pairs;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int pair_offset = index / (n * n);
    const int cell = index - pair_offset * n * n;
    const int row = cell / n;
    const int col = cell - row * n;
    const double* column = values + static_cast<std::size_t>(pair_offset) * n;
    const double diff = column[row] - column[col];
    gram[index] = exp(-sigma * diff * diff);
  }
}

__global__ void hsic_rowsum_kernel(const double* gram,
                                   int n,
                                   int pairs,
                                   double* rowsums,
                                   double* totals) {
  __shared__ double scratch[kBlock];
  const int row = blockIdx.x;
  const int pair = blockIdx.y;
  if (row >= n || pair >= pairs) return;

  const double* matrix =
    gram + static_cast<std::size_t>(pair) * n * n;
  double sum = 0.0;
  for (int col = threadIdx.x; col < n; col += blockDim.x) {
    sum += matrix[row * n + col];
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    rowsums[static_cast<std::size_t>(pair) * n + row] = scratch[0];
    atomicAdd(&totals[pair], scratch[0]);
  }
}

__global__ void hsic_center_kernel(const double* gram,
                                   const double* rowsums,
                                   const double* totals,
                                   int n,
                                   int pairs,
                                   double* centered) {
  const int total = n * n * pairs;
  const double inv_n = 1.0 / static_cast<double>(n);
  const double inv_n2 = inv_n * inv_n;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += blockDim.x * gridDim.x) {
    const int pair = index / (n * n);
    const int cell = index - pair * n * n;
    const int row = cell / n;
    const int col = cell - row * n;
    const double* row_values = rowsums + static_cast<std::size_t>(pair) * n;
    centered[index] =
      gram[index] - row_values[row] * inv_n - row_values[col] * inv_n +
      totals[pair] * inv_n2;
  }
}

__global__ void hsic_reduce_gamma_scalars_kernel(const double* K,
                                                 const double* L,
                                                 const double* Kc,
                                                 const double* Lc,
                                                 int n,
                                                 int pairs,
                                                 double* scalars) {
  __shared__ double hsic_s[kBlock];
  __shared__ double ksq_s[kBlock];
  __shared__ double lsq_s[kBlock];
  __shared__ double off_k_s[kBlock];
  __shared__ double off_l_s[kBlock];

  const int pair = blockIdx.x;
  if (pair >= pairs) return;
  const std::size_t base = static_cast<std::size_t>(pair) * n * n;
  double hsic = 0.0;
  double ksq = 0.0;
  double lsq = 0.0;
  double off_k = 0.0;
  double off_l = 0.0;
  for (int cell = threadIdx.x; cell < n * n; cell += blockDim.x) {
    const double kc = Kc[base + cell];
    const double lc = Lc[base + cell];
    hsic += kc * lc;
    ksq += kc * kc;
    lsq += lc * lc;
    const int row = cell / n;
    const int col = cell - row * n;
    if (row != col) {
      off_k += K[base + cell];
      off_l += L[base + cell];
    }
  }

  hsic_s[threadIdx.x] = hsic;
  ksq_s[threadIdx.x] = ksq;
  lsq_s[threadIdx.x] = lsq;
  off_k_s[threadIdx.x] = off_k;
  off_l_s[threadIdx.x] = off_l;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      hsic_s[threadIdx.x] += hsic_s[threadIdx.x + stride];
      ksq_s[threadIdx.x] += ksq_s[threadIdx.x + stride];
      lsq_s[threadIdx.x] += lsq_s[threadIdx.x + stride];
      off_k_s[threadIdx.x] += off_k_s[threadIdx.x + stride];
      off_l_s[threadIdx.x] += off_l_s[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const std::size_t out = static_cast<std::size_t>(pair) * 5;
    scalars[out + 0] = hsic_s[0];
    scalars[out + 1] = ksq_s[0];
    scalars[out + 2] = lsq_s[0];
    scalars[out + 3] = off_k_s[0];
    scalars[out + 4] = off_l_s[0];
  }
}

__global__ void hsic_permutation_reduce_kernel(const double* Kc,
                                               const double* Lc,
                                               const int* permutations,
                                               int n,
                                               int pairs,
                                               int replicates,
                                               double* out) {
  __shared__ double scratch[kBlock];
  const int replicate = blockIdx.x;
  const int pair = blockIdx.y;
  if (replicate >= replicates || pair >= pairs) return;

  const std::size_t pair_base = static_cast<std::size_t>(pair) * n * n;
  const int* perm =
    permutations + (static_cast<std::size_t>(pair) * replicates + replicate) * n;

  double sum = 0.0;
  for (int cell = threadIdx.x; cell < n * n; cell += blockDim.x) {
    const int row = cell / n;
    const int col = cell - row * n;
    sum += Kc[pair_base + cell] * Lc[pair_base + perm[row] * n + perm[col]];
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[static_cast<std::size_t>(pair) * replicates + replicate] =
      scratch[0] / (static_cast<double>(n) * n);
  }
}

void check_cuda(cudaError_t err, const char* stage) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error (") + stage + "): " +
                             cudaGetErrorString(err));
  }
}

void check_options(int n, int pairs, const HsicBatchOptions& options) {
  if (n < 4) throw std::runtime_error("HSIC requires at least 4 observations");
  if (pairs < 1) throw std::runtime_error("pairs must be positive");
  if (!std::isfinite(options.sig) || options.sig <= 0.0) {
    throw std::runtime_error("HSIC sig must be positive and finite");
  }
  if (options.max_n > 0 && n > options.max_n) {
    throw std::runtime_error("CUDA HSIC n exceeds configured max_n");
  }
  if (options.max_batch_pairs > 0 && pairs > options.max_batch_pairs) {
    throw std::runtime_error("CUDA HSIC pairs exceed configured max_batch_pairs");
  }
}

std::size_t gamma_bytes(int n, int pairs) {
  const std::size_t matrix = static_cast<std::size_t>(n) * n * pairs;
  const std::size_t rows = static_cast<std::size_t>(n) * pairs;
  return sizeof(double) * (matrix * 4 + rows * 2 + pairs * 2 + pairs * 5);
}

HsicBatchDiagnostics make_diagnostics(int n,
                                      int pairs,
                                      int replicates,
                                      const HsicBatchOptions& options,
                                      std::size_t bytes,
                                      int blocks) {
  HsicBatchDiagnostics diagnostics;
  diagnostics.backend = "cuda-hsic";
  diagnostics.reason = "";
  diagnostics.n = n;
  diagnostics.pairs = pairs;
  diagnostics.batches = 1;
  diagnostics.permutation_replicates = replicates;
  diagnostics.used_seed = options.has_seed;
  diagnostics.seed = options.has_seed ? options.seed : 0U;
  diagnostics.bytes_allocated = bytes;
  diagnostics.cuda_blocks = blocks;
  diagnostics.cuda_threads = kBlock;
  return diagnostics;
}

std::vector<int> make_permutation_table(int n,
                                        int pairs,
                                        int replicates,
                                        unsigned int seed) {
  std::vector<int> table(static_cast<std::size_t>(n) * pairs * replicates);
  for (int pair = 0; pair < pairs; ++pair) {
    for (int replicate = 0; replicate < replicates; ++replicate) {
      std::vector<int> perm(n);
      std::iota(perm.begin(), perm.end(), 0);
      std::mt19937 rng(seed + static_cast<unsigned int>(pair * 1000003 + replicate));
      std::shuffle(perm.begin(), perm.end(), rng);
      const std::size_t base =
        (static_cast<std::size_t>(pair) * replicates + replicate) * n;
      for (int row = 0; row < n; ++row) table[base + row] = perm[row];
    }
  }
  return table;
}

struct DeviceHsicStorage {
  double* d_x = nullptr;
  double* d_y = nullptr;
  double* d_K = nullptr;
  double* d_L = nullptr;
  double* d_Kc = nullptr;
  double* d_Lc = nullptr;
  double* d_row_k = nullptr;
  double* d_row_l = nullptr;
  double* d_total_k = nullptr;
  double* d_total_l = nullptr;
  double* d_scalars = nullptr;
};

void free_storage(DeviceHsicStorage* storage) {
  cudaFree(storage->d_x);
  cudaFree(storage->d_y);
  cudaFree(storage->d_K);
  cudaFree(storage->d_L);
  cudaFree(storage->d_Kc);
  cudaFree(storage->d_Lc);
  cudaFree(storage->d_row_k);
  cudaFree(storage->d_row_l);
  cudaFree(storage->d_total_k);
  cudaFree(storage->d_total_l);
  cudaFree(storage->d_scalars);
}

void compute_centered_grams(const double* x,
                            const double* y,
                            int n,
                            int pairs,
                            const HsicBatchOptions& options,
                            DeviceHsicStorage* storage,
                            int* blocks) {
  const std::size_t matrix = static_cast<std::size_t>(n) * n * pairs;
  const std::size_t columns = static_cast<std::size_t>(n) * pairs;
  const std::size_t rows = static_cast<std::size_t>(n) * pairs;
  const std::size_t pair_count = static_cast<std::size_t>(pairs);
  const std::size_t scalar_count = pair_count * 5;

  check_cuda(cudaMalloc(&storage->d_x, sizeof(double) * columns), "alloc hsic x");
  check_cuda(cudaMalloc(&storage->d_y, sizeof(double) * columns), "alloc hsic y");
  check_cuda(cudaMalloc(&storage->d_K, sizeof(double) * matrix), "alloc hsic K");
  check_cuda(cudaMalloc(&storage->d_L, sizeof(double) * matrix), "alloc hsic L");
  check_cuda(cudaMalloc(&storage->d_Kc, sizeof(double) * matrix), "alloc hsic Kc");
  check_cuda(cudaMalloc(&storage->d_Lc, sizeof(double) * matrix), "alloc hsic Lc");
  check_cuda(cudaMalloc(&storage->d_row_k, sizeof(double) * rows), "alloc hsic row K");
  check_cuda(cudaMalloc(&storage->d_row_l, sizeof(double) * rows), "alloc hsic row L");
  check_cuda(cudaMalloc(&storage->d_total_k, sizeof(double) * pair_count),
             "alloc hsic total K");
  check_cuda(cudaMalloc(&storage->d_total_l, sizeof(double) * pair_count),
             "alloc hsic total L");
  check_cuda(cudaMalloc(&storage->d_scalars, sizeof(double) * scalar_count),
             "alloc hsic scalars");

  check_cuda(cudaMemcpy(storage->d_x, x, sizeof(double) * columns,
                        cudaMemcpyHostToDevice), "copy hsic x");
  check_cuda(cudaMemcpy(storage->d_y, y, sizeof(double) * columns,
                        cudaMemcpyHostToDevice), "copy hsic y");
  check_cuda(cudaMemset(storage->d_total_k, 0, sizeof(double) * pair_count),
             "zero hsic total K");
  check_cuda(cudaMemset(storage->d_total_l, 0, sizeof(double) * pair_count),
             "zero hsic total L");
  check_cuda(cudaMemset(storage->d_scalars, 0, sizeof(double) * scalar_count),
             "zero hsic scalars");

  const int linear_blocks = std::max(1, std::min(4096, static_cast<int>((matrix + kBlock - 1) / kBlock)));
  *blocks = linear_blocks;
  const double sigma = 1.0 / options.sig;
  hsic_rbf_kernel<<<linear_blocks, kBlock>>>(storage->d_x, n, pairs, sigma,
                                             storage->d_K);
  hsic_rbf_kernel<<<linear_blocks, kBlock>>>(storage->d_y, n, pairs, sigma,
                                             storage->d_L);
  check_cuda(cudaGetLastError(), "launch hsic rbf");
  check_cuda(cudaDeviceSynchronize(), "hsic rbf synchronize");

  const dim3 rowsum_grid(n, pairs);
  hsic_rowsum_kernel<<<rowsum_grid, kBlock>>>(storage->d_K, n, pairs,
                                              storage->d_row_k,
                                              storage->d_total_k);
  hsic_rowsum_kernel<<<rowsum_grid, kBlock>>>(storage->d_L, n, pairs,
                                              storage->d_row_l,
                                              storage->d_total_l);
  check_cuda(cudaGetLastError(), "launch hsic rowsum");
  check_cuda(cudaDeviceSynchronize(), "hsic rowsum synchronize");

  hsic_center_kernel<<<linear_blocks, kBlock>>>(storage->d_K, storage->d_row_k,
                                                storage->d_total_k, n, pairs,
                                                storage->d_Kc);
  hsic_center_kernel<<<linear_blocks, kBlock>>>(storage->d_L, storage->d_row_l,
                                                storage->d_total_l, n, pairs,
                                                storage->d_Lc);
  check_cuda(cudaGetLastError(), "launch hsic center");
  check_cuda(cudaDeviceSynchronize(), "hsic center synchronize");

  hsic_reduce_gamma_scalars_kernel<<<pairs, kBlock>>>(
    storage->d_K, storage->d_L, storage->d_Kc, storage->d_Lc, n, pairs,
    storage->d_scalars);
  check_cuda(cudaGetLastError(), "launch hsic gamma reduce");
  check_cuda(cudaDeviceSynchronize(), "hsic gamma reduce synchronize");
}

HsicBatchResult finish_gamma_result(const std::vector<double>& scalars,
                                    int n,
                                    int pairs,
                                    const HsicBatchOptions& options,
                                    std::size_t bytes,
                                    int blocks) {
  HsicBatchResult result;
  result.statistics.assign(pairs, 0.0);
  result.p_values.assign(pairs, 1.0);
  result.means.assign(pairs, 0.0);
  result.variances.assign(pairs, 0.0);
  result.shapes.assign(pairs, std::numeric_limits<double>::quiet_NaN());
  result.scales.assign(pairs, std::numeric_limits<double>::quiet_NaN());
  result.diagnostics = make_diagnostics(n, pairs, 0, options, bytes, blocks);

  const double nd = static_cast<double>(n);
  for (int pair = 0; pair < pairs; ++pair) {
    const std::size_t base = static_cast<std::size_t>(pair) * 5;
    const double hsic_sum = scalars[base + 0];
    const double ksq = scalars[base + 1];
    const double lsq = scalars[base + 2];
    const double off_k = scalars[base + 3];
    const double off_l = scalars[base + 4];

    const double statistic = hsic_sum / (nd * nd);
    const double mux = off_k / (nd * (nd - 1.0));
    const double muy = off_l / (nd * (nd - 1.0));
    const double mean = (1.0 + mux * muy - mux - muy) / nd;
    const double variance = (2.0 * (nd - 4.0) * (nd - 5.0) /
      (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0))) *
      ksq * lsq / (nd * nd * nd * nd);

    result.statistics[pair] = statistic;
    result.means[pair] = mean;
    result.variances[pair] = variance;
    if (std::isfinite(mean) && std::isfinite(variance) &&
        mean > 0.0 && variance > 0.0) {
      const double shape = mean * mean / variance;
      const double scale = variance / mean;
      result.shapes[pair] = shape;
      result.scales[pair] = scale;
      result.p_values[pair] = Rf_pgamma(statistic, shape, scale, false, false);
    } else {
      result.p_values[pair] = 1.0;
    }
    if (!std::isfinite(result.p_values[pair])) result.p_values[pair] = 1.0;
    if (result.p_values[pair] < 0.0) result.p_values[pair] = 0.0;
    if (result.p_values[pair] > 1.0) result.p_values[pair] = 1.0;
  }
  return result;
}

}  // namespace

HsicBatchOptions default_hsic_batch_options() {
  HsicBatchOptions options;
  options.sig = 1.0;
  options.permutation_replicates = 100;
  options.include_observed = true;
  options.has_seed = false;
  options.seed = 0U;
  options.return_replicates = true;
  options.max_n = 2048;
  options.max_batch_pairs = 64;
  return options;
}

bool hsic_cuda_available(std::string* reason) {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    if (reason != nullptr) *reason = cudaGetErrorString(err);
    return false;
  }
  if (count <= 0) {
    if (reason != nullptr) *reason = "no CUDA devices available";
    return false;
  }
  if (reason != nullptr) reason->clear();
  return true;
}

HsicBatchResult hsic_gamma_batch_cuda(const double* x,
                                      const double* y,
                                      int n,
                                      int pairs,
                                      const HsicBatchOptions& options) {
  check_options(n, pairs, options);
  DeviceHsicStorage storage;
  int blocks = 0;
  const std::size_t bytes = gamma_bytes(n, pairs);
  try {
    compute_centered_grams(x, y, n, pairs, options, &storage, &blocks);
    std::vector<double> scalars(static_cast<std::size_t>(pairs) * 5, 0.0);
    check_cuda(cudaMemcpy(scalars.data(), storage.d_scalars,
                          sizeof(double) * scalars.size(),
                          cudaMemcpyDeviceToHost), "copy hsic scalars");
    HsicBatchResult result = finish_gamma_result(scalars, n, pairs, options,
                                                 bytes, blocks);
    free_storage(&storage);
    return result;
  } catch (...) {
    free_storage(&storage);
    throw;
  }
}

HsicBatchResult hsic_permutation_batch_cuda(const double* x,
                                            const double* y,
                                            int n,
                                            int pairs,
                                            const HsicBatchOptions& options) {
  check_options(n, pairs, options);
  if (options.permutation_replicates < 0) {
    throw std::runtime_error("HSIC permutation replicates must be non-negative");
  }
  if (!options.has_seed) {
    throw std::runtime_error("CUDA HSIC permutation requires explicit seed in this stage");
  }

  DeviceHsicStorage storage;
  double* d_replicates = nullptr;
  int* d_permutations = nullptr;
  int blocks = 0;
  const int replicates = options.permutation_replicates;
  const std::size_t bytes =
    gamma_bytes(n, pairs) +
    sizeof(double) * static_cast<std::size_t>(pairs) * std::max(1, replicates) +
    sizeof(int) * static_cast<std::size_t>(pairs) * std::max(1, replicates) * n;

  try {
    compute_centered_grams(x, y, n, pairs, options, &storage, &blocks);
    std::vector<double> scalars(static_cast<std::size_t>(pairs) * 5, 0.0);
    check_cuda(cudaMemcpy(scalars.data(), storage.d_scalars,
                          sizeof(double) * scalars.size(),
                          cudaMemcpyDeviceToHost), "copy hsic permutation scalars");
    HsicBatchResult result = finish_gamma_result(scalars, n, pairs, options,
                                                 bytes, blocks);
    result.diagnostics.permutation_replicates = replicates;

    if (replicates > 0) {
      const std::vector<int> permutations =
        make_permutation_table(n, pairs, replicates, options.seed);
      const std::size_t replicate_count = static_cast<std::size_t>(pairs) * replicates;
      check_cuda(cudaMalloc(&d_permutations,
                            sizeof(int) * permutations.size()),
                 "alloc hsic permutations");
      check_cuda(cudaMalloc(&d_replicates,
                            sizeof(double) * replicate_count),
                 "alloc hsic replicate stats");
      check_cuda(cudaMemcpy(d_permutations, permutations.data(),
                            sizeof(int) * permutations.size(),
                            cudaMemcpyHostToDevice),
                 "copy hsic permutations");

      const dim3 perm_grid(replicates, pairs);
      hsic_permutation_reduce_kernel<<<perm_grid, kBlock>>>(
        storage.d_Kc, storage.d_Lc, d_permutations, n, pairs, replicates,
        d_replicates);
      check_cuda(cudaGetLastError(), "launch hsic permutation reduce");
      check_cuda(cudaDeviceSynchronize(), "hsic permutation reduce synchronize");

      result.permutation_replicates.assign(replicate_count, 0.0);
      check_cuda(cudaMemcpy(result.permutation_replicates.data(), d_replicates,
                            sizeof(double) * replicate_count,
                            cudaMemcpyDeviceToHost),
                 "copy hsic replicate stats");

      for (int pair = 0; pair < pairs; ++pair) {
        int exceedances = options.include_observed ? 1 : 0;
        const std::size_t base = static_cast<std::size_t>(pair) * replicates;
        for (int r = 0; r < replicates; ++r) {
          if (result.permutation_replicates[base + r] >= result.statistics[pair]) {
            ++exceedances;
          }
        }
        const int denominator = replicates + (options.include_observed ? 1 : 0);
        result.p_values[pair] = denominator > 0 ?
          static_cast<double>(exceedances) / static_cast<double>(denominator) : 1.0;
      }
    } else {
      result.permutation_replicates.clear();
      for (int pair = 0; pair < pairs; ++pair) result.p_values[pair] = 1.0;
    }

    cudaFree(d_replicates);
    cudaFree(d_permutations);
    free_storage(&storage);
    return result;
  } catch (...) {
    cudaFree(d_replicates);
    cudaFree(d_permutations);
    free_storage(&storage);
    throw;
  }
}
