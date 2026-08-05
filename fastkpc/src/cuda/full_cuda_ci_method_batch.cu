#include "full_cuda_ci_method_batch.hpp"

#include "../full_cuda_ci_contract.hpp"
#include "third_party/glibc_2_35_exp_fma.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr int kBlockSize = 256;
constexpr int kStatusOk = 0;
constexpr int kStatusInvalid = 1;
constexpr int kStatusGammaFailure = 2;
constexpr double kIncholTolerance = 0.001;

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

bool is_lower_sha256(const std::string& value) {
  return value.size() == 64U &&
    std::all_of(value.begin(), value.end(), [](unsigned char character) {
      return (character >= '0' && character <= '9') ||
        (character >= 'a' && character <= 'f');
    });
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             const char* label) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(label) + " size overflow");
  }
  return left * right;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }
  ~DeviceBuffer() { if (pointer_ != nullptr) cudaFree(pointer_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(std::size_t bytes) {
    if (pointer_ != nullptr || bytes == 0U) {
      throw std::runtime_error("invalid strict CI device allocation");
    }
    check_cuda(cudaMalloc(&pointer_, bytes), "allocate strict CI buffer");
    bytes_ = bytes;
  }

  void close() {
    if (pointer_ == nullptr) return;
    check_cuda(cudaFree(pointer_), "free strict CI buffer");
    pointer_ = nullptr;
    bytes_ = 0U;
  }

  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

class CudaStream {
 public:
  CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking),
               "create strict CI stream");
  }
  ~CudaStream() { if (value_ != nullptr) cudaStreamDestroy(value_); }
  cudaStream_t get() const { return value_; }
  void close() {
    if (value_ == nullptr) return;
    check_cuda(cudaStreamDestroy(value_), "destroy strict CI stream");
    value_ = nullptr;
  }
 private:
  cudaStream_t value_ = nullptr;
};

class CudaEvent {
 public:
  explicit CudaEvent(bool timing) {
    check_cuda(cudaEventCreateWithFlags(
                 &value_, timing ? cudaEventDefault : cudaEventDisableTiming),
               "create strict CI event");
  }
  ~CudaEvent() { if (value_ != nullptr) cudaEventDestroy(value_); }
  cudaEvent_t get() const { return value_; }
  void close() {
    if (value_ == nullptr) return;
    check_cuda(cudaEventDestroy(value_), "destroy strict CI event");
    value_ = nullptr;
  }
 private:
  cudaEvent_t value_ = nullptr;
};

struct DeviceMethodRecord {
  unsigned long long logical_sequence_id = 0ULL;
  double p_value = 1.0;
  double statistic = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  int status = kStatusInvalid;
};

__device__ double regularized_gamma_q(double shape,
                                      double x,
                                      int* converged) {
  constexpr int max_iterations = 10000;
  constexpr double epsilon = 2.0e-15;
  constexpr double tiny = 1.0e-300;
  *converged = 0;
  if (!(shape > 0.0) || !(x >= 0.0) ||
      !isfinite(shape) || !isfinite(x)) {
    return nan("");
  }
  if (x == 0.0) {
    *converged = 1;
    return 1.0;
  }
  const double log_prefactor = -x + shape * log(x) - lgamma(shape);
  if (x < shape + 1.0) {
    double ap = shape;
    double sum = 1.0 / shape;
    double term = sum;
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
      ap += 1.0;
      term *= x / ap;
      sum += term;
      if (fabs(term) <= fabs(sum) * epsilon) {
        *converged = 1;
        const double lower = fmin(1.0, fmax(0.0,
          sum * exp(log_prefactor)));
        return fmin(1.0, fmax(0.0, 1.0 - lower));
      }
    }
    return nan("");
  }

  double b = x + 1.0 - shape;
  double c = 1.0 / tiny;
  double d = 1.0 / fmax(fabs(b), tiny);
  if (b < 0.0) d = -d;
  double fraction = d;
  for (int iteration = 1; iteration <= max_iterations; ++iteration) {
    const double iter = static_cast<double>(iteration);
    const double an = -iter * (iter - shape);
    b += 2.0;
    d = an * d + b;
    if (fabs(d) < tiny) d = copysign(tiny, d == 0.0 ? 1.0 : d);
    c = b + an / c;
    if (fabs(c) < tiny) c = copysign(tiny, c == 0.0 ? 1.0 : c);
    d = 1.0 / d;
    const double delta = d * c;
    fraction *= delta;
    if (fabs(delta - 1.0) <= epsilon) {
      *converged = 1;
      return fmin(1.0, fmax(0.0, exp(log_prefactor) * fraction));
    }
  }
  return nan("");
}

__global__ void build_distance_components_kernel(
    const double* residuals,
    const int* component_targets,
    int n,
    int component_count,
    double* centered,
    double* row_sums) {
  __shared__ double reduction[kBlockSize];
  const int row = static_cast<int>(blockIdx.x);
  const int component = static_cast<int>(blockIdx.y);
  if (row >= n || component >= component_count) return;
  const int target = component_targets[component];
  const double* values = residuals + static_cast<std::size_t>(target) * n;
  double* matrix = centered +
    static_cast<std::size_t>(component) * n * n;
  const double row_value = values[row];
  double sum = 0.0;
  for (int column = static_cast<int>(threadIdx.x);
       column < n; column += static_cast<int>(blockDim.x)) {
    const double value = fabs(row_value - values[column]);
    matrix[static_cast<std::size_t>(row) * n + column] = value;
    sum += value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_sums[static_cast<std::size_t>(component) * n + row] = reduction[0];
  }
}

__global__ void center_distance_components_kernel(
    double* centered,
    const double* row_sums,
    int n,
    int component_count) {
  const int component = static_cast<int>(blockIdx.y);
  const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
    threadIdx.x;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  if (component >= component_count || cell >= cells) return;
  const double* rows = row_sums + static_cast<std::size_t>(component) * n;
  double total = 0.0;
  for (int row = 0; row < n; ++row) total += rows[row];
  const int row = static_cast<int>(cell / static_cast<std::size_t>(n));
  const int column = static_cast<int>(cell % static_cast<std::size_t>(n));
  const double nd = static_cast<double>(n);
  double* matrix = centered + static_cast<std::size_t>(component) * cells;
  matrix[cell] = matrix[cell] - rows[row] / nd - rows[column] / nd +
    total / (nd * nd);
}

__global__ void initialize_dcc_permutation_pairs_kernel(
    const double* centered,
    const int* left_components,
    const int* right_components,
    int n,
    int pair_count,
    int include_observed,
    double* observed_sums,
    int* exceedances) {
  __shared__ double reduction[kBlockSize];
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  const double* left = centered +
    static_cast<std::size_t>(left_components[pair]) * cells;
  const double* right = centered +
    static_cast<std::size_t>(right_components[pair]) * cells;

  double sum = 0.0;
  for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x) {
    sum += left[cell] * right[cell];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    observed_sums[pair] = reduction[0];
    exceedances[pair] = include_observed != 0 ? 1 : 0;
  }
}

__global__ void evaluate_dcc_permutation_replicates_kernel(
    const double* centered,
    const int* left_components,
    const int* right_components,
    const int* permutations,
    const double* observed_sums,
    int n,
    int pair_count,
    int replicates,
    int* exceedances) {
  __shared__ double reduction[kBlockSize];
  const int pair = static_cast<int>(blockIdx.x);
  const int replicate = static_cast<int>(blockIdx.y);
  if (pair >= pair_count || replicate >= replicates) return;
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  const double* left = centered +
    static_cast<std::size_t>(left_components[pair]) * cells;
  const double* right = centered +
    static_cast<std::size_t>(right_components[pair]) * cells;
  const int* permutation = permutations +
    (static_cast<std::size_t>(pair) * replicates + replicate) * n;
  double replicate_sum = 0.0;
  for (std::size_t cell = threadIdx.x; cell < cells;
       cell += blockDim.x) {
    const int row = static_cast<int>(cell / static_cast<std::size_t>(n));
    const int column = static_cast<int>(cell % static_cast<std::size_t>(n));
    replicate_sum += left[cell] * right[
      static_cast<std::size_t>(permutation[row]) * n +
        permutation[column]];
  }
  reduction[threadIdx.x] = replicate_sum;
  __syncthreads();
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && reduction[0] >= observed_sums[pair]) {
    atomicAdd(exceedances + pair, 1);
  }
}

__global__ void finalize_dcc_permutation_pairs_kernel(
    const unsigned long long* logical_sequence_ids,
    const double* observed_sums,
    const int* exceedances,
    int n,
    int pair_count,
    int replicates,
    int include_observed,
    DeviceMethodRecord* records) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  DeviceMethodRecord output;
  output.logical_sequence_id = logical_sequence_ids[pair];
  output.statistic = observed_sums[pair] / static_cast<double>(n);
  const int denominator = replicates + (include_observed != 0 ? 1 : 0);
  if (!(observed_sums[pair] > 0.0) || denominator <= 0) {
    output.p_value = 1.0;
  } else {
    output.p_value = static_cast<double>(exceedances[pair]) /
      static_cast<double>(denominator);
  }
  output.status = kStatusOk;
  records[pair] = output;
}

__global__ void build_inchol_components_kernel(
    const double* residuals,
    const int* component_targets,
    int n,
    int component_count,
    int max_rank,
    double sigma,
    double* factors,
    double* triangular,
    double* diagonal_residuals,
    double* kernel_column,
    double* solve_vector,
    int* pivots,
    int* ranks,
    double* off_diagonal_sums) {
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count || threadIdx.x != 0) return;
  const int target = component_targets[component];
  const double* values = residuals + static_cast<std::size_t>(target) * n;
  double* factor = factors +
    static_cast<std::size_t>(component) * n * max_rank;
  double* upper = triangular +
    static_cast<std::size_t>(component) * max_rank * max_rank;
  double* diagonal = diagonal_residuals +
    static_cast<std::size_t>(component) * n;
  double* column = kernel_column + static_cast<std::size_t>(component) * n;
  double* solve = solve_vector +
    static_cast<std::size_t>(component) * max_rank;
  int* component_pivots = pivots +
    static_cast<std::size_t>(component) * max_rank;

  for (int row = 0; row < n; ++row) diagonal[row] = 1.0;
  for (int entry = 0; entry < max_rank * max_rank; ++entry) {
    upper[entry] = 0.0;
  }
  double residue = 1.0;
  int pivot = 0;
  int rank = 0;
  while (residue > kIncholTolerance && rank < max_rank) {
    const double pivot_value = values[pivot];
    for (int row = 0; row < n; ++row) {
      const double difference = values[row] - pivot_value;
      column[row] = glibc235::exp_fma_rn(
        -sigma * difference * difference);
    }

    double tau_square = column[pivot];
    if (rank == 0) {
      upper[0] = sqrt(tau_square);
    } else {
      for (int row = 0; row < rank; ++row) {
        double value = column[component_pivots[row]];
        for (int previous = 0; previous < row; ++previous) {
          value -= upper[previous + static_cast<std::size_t>(max_rank) * row] *
            solve[previous];
        }
        solve[row] = value /
          upper[row + static_cast<std::size_t>(max_rank) * row];
        tau_square -= solve[row] * solve[row];
        upper[row + static_cast<std::size_t>(max_rank) * rank] = solve[row];
      }
    }
    if (!(tau_square > 0.0) || !isfinite(tau_square)) break;
    const double tau = sqrt(tau_square);
    upper[rank + static_cast<std::size_t>(max_rank) * rank] = tau;
    for (int row = 0; row < n; ++row) {
      double projection = 0.0;
      for (int previous = 0; previous < rank; ++previous) {
        projection += factor[row + static_cast<std::size_t>(n) * previous] *
          solve[previous];
      }
      const double update = (column[row] - projection) / tau;
      factor[row + static_cast<std::size_t>(n) * rank] = update;
      diagonal[row] -= update * update;
    }
    component_pivots[rank] = pivot;
    ++rank;
    residue = diagonal[0];
    pivot = 0;
    for (int row = 1; row < n; ++row) {
      if (diagonal[row] > residue) {
        residue = diagonal[row];
        pivot = row;
      }
    }
  }
  ranks[component] = rank;

  double total_sum = 0.0;
  double diagonal_sum = 0.0;
  for (int column_index = 0; column_index < rank; ++column_index) {
    double column_sum = 0.0;
    for (int row = 0; row < n; ++row) {
      const double value = factor[
        row + static_cast<std::size_t>(n) * column_index];
      column_sum += value;
      diagonal_sum += value * value;
    }
    total_sum += column_sum * column_sum;
  }
  off_diagonal_sums[component] = total_sum - diagonal_sum;
}

__global__ void center_inchol_components_kernel(double* factors,
                                                const int* ranks,
                                                int n,
                                                int component_count,
                                                int max_rank) {
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count || threadIdx.x != 0) return;
  double* factor = factors +
    static_cast<std::size_t>(component) * n * max_rank;
  const int rank = ranks[component];
  for (int column = 0; column < rank; ++column) {
    double mean = 0.0;
    for (int row = 0; row < n; ++row) {
      mean += factor[row + static_cast<std::size_t>(n) * column];
    }
    mean /= static_cast<double>(n);
    for (int row = 0; row < n; ++row) {
      factor[row + static_cast<std::size_t>(n) * column] -= mean;
    }
  }
}

__global__ void reduce_inchol_self_moments_kernel(
    const double* factors,
    const int* ranks,
    int n,
    int component_count,
    int max_rank,
    double* self_moments) {
  extern __shared__ double products[];
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count) return;
  const int rank = ranks[component];
  const double* factor = factors +
    static_cast<std::size_t>(component) * n * max_rank;
  const int square = max_rank * max_rank;
  for (int entry = static_cast<int>(threadIdx.x); entry < square;
       entry += static_cast<int>(blockDim.x)) {
    const int left = entry / max_rank;
    const int right = entry % max_rank;
    double dot = 0.0;
    if (left < rank && right < rank) {
      for (int row = 0; row < n; ++row) {
        dot += factor[row + static_cast<std::size_t>(n) * left] *
          factor[row + static_cast<std::size_t>(n) * right];
      }
    }
    products[entry] = dot * dot;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int entry = 0; entry < square; ++entry) sum += products[entry];
    self_moments[component] = sum;
  }
}

__global__ void evaluate_hsic_kernel(
    const double* factors,
    const int* ranks,
    const double* off_diagonal_sums,
    const double* self_moments,
    const int* left_components,
    const int* right_components,
    const unsigned long long* logical_sequence_ids,
    const int* permutations,
    int n,
    int pair_count,
    int max_rank,
    int replicates,
    int include_observed,
    double alpha,
    DeviceMethodRecord* records) {
  extern __shared__ double products[];
  __shared__ double observed;
  __shared__ int exceedances;
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const int left_component = left_components[pair];
  const int right_component = right_components[pair];
  const int left_rank = ranks[left_component];
  const int right_rank = ranks[right_component];
  const double* left = factors +
    static_cast<std::size_t>(left_component) * n * max_rank;
  const double* right = factors +
    static_cast<std::size_t>(right_component) * n * max_rank;
  const int square = max_rank * max_rank;

  for (int entry = static_cast<int>(threadIdx.x); entry < square;
       entry += static_cast<int>(blockDim.x)) {
    const int left_column = entry / max_rank;
    const int right_column = entry % max_rank;
    double dot = 0.0;
    if (left_column < left_rank && right_column < right_rank) {
      for (int row = 0; row < n; ++row) {
        dot += left[row + static_cast<std::size_t>(n) * left_column] *
          right[row + static_cast<std::size_t>(n) * right_column];
      }
    }
    products[entry] = dot * dot;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int entry = 0; entry < square; ++entry) sum += products[entry];
    observed = sum / (static_cast<double>(n) * n);
    exceedances = include_observed != 0 ? 1 : 0;
  }
  __syncthreads();

  for (int replicate = 0; replicate < replicates; ++replicate) {
    const int* permutation = permutations +
      (static_cast<std::size_t>(pair) * replicates + replicate) * n;
    for (int entry = static_cast<int>(threadIdx.x); entry < square;
         entry += static_cast<int>(blockDim.x)) {
      const int left_column = entry / max_rank;
      const int right_column = entry % max_rank;
      double dot = 0.0;
      if (left_column < left_rank && right_column < right_rank) {
        for (int row = 0; row < n; ++row) {
          dot += left[row + static_cast<std::size_t>(n) * left_column] *
            right[permutation[row] +
              static_cast<std::size_t>(n) * right_column];
        }
      }
      products[entry] = dot * dot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      double sum = 0.0;
      for (int entry = 0; entry < square; ++entry) sum += products[entry];
      const double statistic = sum / (static_cast<double>(n) * n);
      if (statistic >= observed) ++exceedances;
    }
    __syncthreads();
  }

  if (threadIdx.x != 0) return;
  DeviceMethodRecord output;
  output.logical_sequence_id = logical_sequence_ids[pair];
  output.statistic = observed;
  if (replicates > 0) {
    const int denominator = replicates + (include_observed != 0 ? 1 : 0);
    output.p_value = denominator > 0 ?
      static_cast<double>(exceedances) / static_cast<double>(denominator) :
      1.0;
    output.status = kStatusOk;
    records[pair] = output;
    return;
  }

  const double nd = static_cast<double>(n);
  const double mux = off_diagonal_sums[left_component] /
    (nd * (nd - 1.0));
  const double muy = off_diagonal_sums[right_component] /
    (nd * (nd - 1.0));
  output.mean = (1.0 + mux * muy - mux - muy) / nd;
  const double variance_factor =
    2.0 * (nd - 4.0) * (nd - 5.0) /
    (nd * (nd - 1.0) * (nd - 2.0) * (nd - 3.0));
  output.variance = variance_factor * self_moments[left_component] *
    self_moments[right_component] / (nd * nd * nd * nd);
  if (!isfinite(output.statistic) || !isfinite(output.mean) ||
      !isfinite(output.variance) || !(output.mean > 0.0) ||
      !(output.variance > 0.0)) {
    output.p_value = 1.0;
    output.status = kStatusInvalid;
    records[pair] = output;
    return;
  }
  const double shape = output.mean * output.mean / output.variance;
  const double scale = output.variance / output.mean;
  int converged = 0;
  const double upper_tail = regularized_gamma_q(
    shape, fmax(0.0, output.statistic / scale), &converged);
  if (converged == 0 || !isfinite(upper_tail)) {
    output.p_value = 1.0;
    output.status = kStatusGammaFailure;
  } else {
    // kpcalg::hsic.gamma returns 1 - pgamma(lower_tail), including the
    // binary64 cancellation behavior for very small upper tails.
    output.p_value = 1.0 - (1.0 - fmin(1.0, fmax(0.0, upper_tail)));
    output.status = kStatusOk;
  }
  (void)alpha;
  records[pair] = output;
}

__global__ void initialize_hsic_permutation_pairs_kernel(
    const double* factors,
    const int* ranks,
    const int* left_components,
    const int* right_components,
    int n,
    int pair_count,
    int max_rank,
    int include_observed,
    double* observed_statistics,
    int* exceedances) {
  extern __shared__ double products[];
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const int left_component = left_components[pair];
  const int right_component = right_components[pair];
  const int left_rank = ranks[left_component];
  const int right_rank = ranks[right_component];
  const double* left = factors +
    static_cast<std::size_t>(left_component) * n * max_rank;
  const double* right = factors +
    static_cast<std::size_t>(right_component) * n * max_rank;
  const int square = max_rank * max_rank;
  for (int entry = static_cast<int>(threadIdx.x); entry < square;
       entry += static_cast<int>(blockDim.x)) {
    const int left_column = entry / max_rank;
    const int right_column = entry % max_rank;
    double dot = 0.0;
    if (left_column < left_rank && right_column < right_rank) {
      for (int row = 0; row < n; ++row) {
        dot += left[row + static_cast<std::size_t>(n) * left_column] *
          right[row + static_cast<std::size_t>(n) * right_column];
      }
    }
    products[entry] = dot * dot;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int entry = 0; entry < square; ++entry) sum += products[entry];
    observed_statistics[pair] = sum / (static_cast<double>(n) * n);
    exceedances[pair] = include_observed != 0 ? 1 : 0;
  }
}

__global__ void evaluate_hsic_permutation_replicates_kernel(
    const double* factors,
    const int* ranks,
    const int* left_components,
    const int* right_components,
    const int* permutations,
    const double* observed_statistics,
    int n,
    int pair_count,
    int max_rank,
    int replicates,
    int* exceedances) {
  extern __shared__ double products[];
  const int pair = static_cast<int>(blockIdx.x);
  const int replicate = static_cast<int>(blockIdx.y);
  if (pair >= pair_count || replicate >= replicates) return;
  const int left_component = left_components[pair];
  const int right_component = right_components[pair];
  const int left_rank = ranks[left_component];
  const int right_rank = ranks[right_component];
  const double* left = factors +
    static_cast<std::size_t>(left_component) * n * max_rank;
  const double* right = factors +
    static_cast<std::size_t>(right_component) * n * max_rank;
  const int* permutation = permutations +
    (static_cast<std::size_t>(pair) * replicates + replicate) * n;
  const int square = max_rank * max_rank;
  for (int entry = static_cast<int>(threadIdx.x); entry < square;
       entry += static_cast<int>(blockDim.x)) {
    const int left_column = entry / max_rank;
    const int right_column = entry % max_rank;
    double dot = 0.0;
    if (left_column < left_rank && right_column < right_rank) {
      for (int row = 0; row < n; ++row) {
        dot += left[row + static_cast<std::size_t>(n) * left_column] *
          right[permutation[row] +
            static_cast<std::size_t>(n) * right_column];
      }
    }
    products[entry] = dot * dot;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0.0;
    for (int entry = 0; entry < square; ++entry) sum += products[entry];
    const double statistic = sum / (static_cast<double>(n) * n);
    if (statistic >= observed_statistics[pair]) {
      atomicAdd(exceedances + pair, 1);
    }
  }
}

__global__ void finalize_hsic_permutation_pairs_kernel(
    const unsigned long long* logical_sequence_ids,
    const double* observed_statistics,
    const int* exceedances,
    int pair_count,
    int replicates,
    int include_observed,
    DeviceMethodRecord* records) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  DeviceMethodRecord output;
  output.logical_sequence_id = logical_sequence_ids[pair];
  output.statistic = observed_statistics[pair];
  const int denominator = replicates + (include_observed != 0 ? 1 : 0);
  output.p_value = denominator > 0 ?
    static_cast<double>(exceedances[pair]) /
      static_cast<double>(denominator) :
    1.0;
  output.status = kStatusOk;
  records[pair] = output;
}

float elapsed_event_ms(cudaEvent_t start, cudaEvent_t stop) {
  float elapsed = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
             "measure strict CI CUDA stage");
  return elapsed;
}

void validate_request(const FullCudaCiMethodBatchRequest& request,
                      const FixedSpBatchHostView& batch) {
  if (request.ci_method != "dcc.perm" &&
      request.ci_method != "hsic.gamma" &&
      request.ci_method != "hsic.perm") {
    throw std::runtime_error("strict CI method batch method is unsupported");
  }
  if (!is_lower_sha256(request.expected_prepared_s_key_sha256) ||
      request.pairs.empty() || batch.target_count < 2 || batch.n < 6) {
    throw std::runtime_error("strict CI method batch request is malformed");
  }
  if (request.alpha != 0.1 || request.index != 1.0 ||
      request.num_col < 1 || request.num_col > 64 ||
      !std::isfinite(request.hsic_sig) || request.hsic_sig <= 0.0) {
    throw std::runtime_error("strict CI method batch numeric contract changed");
  }
  std::uint64_t previous = 0;
  for (const FullCudaCiMethodPairRequest& pair : request.pairs) {
    if (pair.logical_sequence_id == 0 ||
        pair.logical_sequence_id <= previous ||
        pair.left_target_index < 0 ||
        pair.left_target_index >= batch.target_count ||
        pair.right_target_index < 0 ||
        pair.right_target_index >= batch.target_count ||
        pair.left_target_index == pair.right_target_index) {
      throw std::runtime_error("strict CI method pair request is invalid");
    }
    previous = pair.logical_sequence_id;
  }
  const bool permutation = request.ci_method == "dcc.perm" ||
    request.ci_method == "hsic.perm";
  if (permutation) {
    if (request.permutation_replicates < 1) {
      throw std::runtime_error("strict permutation CI requires replicates");
    }
    const std::size_t expected = checked_multiply(
      checked_multiply(request.pairs.size(),
                       static_cast<std::size_t>(request.permutation_replicates),
                       "strict CI permutation pairs"),
      static_cast<std::size_t>(batch.n), "strict CI permutation rows");
    if (request.permutations.size() != expected) {
      throw std::runtime_error("strict CI permutation table size mismatch");
    }
    for (int index : request.permutations) {
      if (index < 0 || index >= batch.n) {
        throw std::runtime_error("strict CI permutation index is invalid");
      }
    }
  } else if (request.permutation_replicates != 0 ||
             !request.permutations.empty()) {
    throw std::runtime_error("HSIC gamma received permutation metadata");
  }
}

}  // namespace

std::vector<int> full_cuda_ci_method_seeded_permutation_table(
    const std::string& ci_method,
    int n,
    int replicates,
    const std::vector<unsigned int>& seeds) {
  if ((ci_method != "dcc.perm" && ci_method != "hsic.perm") ||
      n < 2 || replicates < 1 || seeds.empty()) {
    throw std::runtime_error("strict CI seeded permutation request is invalid");
  }
  const std::size_t values = checked_multiply(
    checked_multiply(seeds.size(), static_cast<std::size_t>(replicates),
                     "strict CI seeded permutation pairs"),
    static_cast<std::size_t>(n), "strict CI seeded permutation rows");
  std::vector<int> table(values);
  for (std::size_t pair = 0; pair < seeds.size(); ++pair) {
    if (ci_method == "dcc.perm") {
      for (int replicate = 0; replicate < replicates; ++replicate) {
        std::vector<int> permutation(static_cast<std::size_t>(n));
        std::iota(permutation.begin(), permutation.end(), 0);
        std::mt19937 rng(
          seeds[pair] + static_cast<unsigned int>(replicate));
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::copy(permutation.begin(), permutation.end(), table.begin() +
          (pair * static_cast<std::size_t>(replicates) + replicate) * n);
      }
    } else {
      std::mt19937 rng(seeds[pair]);
      for (int replicate = 0; replicate < replicates; ++replicate) {
        std::vector<int> permutation(static_cast<std::size_t>(n));
        std::iota(permutation.begin(), permutation.end(), 0);
        std::shuffle(permutation.begin(), permutation.end(), rng);
        std::copy(permutation.begin(), permutation.end(), table.begin() +
          (pair * static_cast<std::size_t>(replicates) + replicate) * n);
      }
    }
  }
  return table;
}

std::string full_cuda_ci_method_batch_request_identity(
    const FullCudaCiMethodBatchRequest& request,
    const std::vector<std::string>& target_keys,
    int n) {
  std::string permutation_bytes;
  if (!request.permutations.empty()) {
    permutation_bytes.assign(
      reinterpret_cast<const char*>(request.permutations.data()),
      request.permutations.size() * sizeof(int));
  }
  const std::string permutation_sha = full_cuda_ci_sha256_utf8(
    permutation_bytes);
  std::ostringstream payload;
  payload << "schema=" << kFullCudaCiMethodBatchRequestSchemaVersion << "\n"
          << "prepared=" << request.expected_prepared_s_key_sha256 << "\n"
          << "method=" << request.ci_method << "\n"
          << "n=" << n << "\n"
          << "alpha_bits=" << std::hexfloat << request.alpha << "\n"
          << "index_bits=" << request.index << "\n"
          << "numCol=" << std::defaultfloat << request.num_col << "\n"
          << "hsic_sig_bits=" << std::hexfloat << request.hsic_sig << "\n"
          << "replicates=" << std::defaultfloat
          << request.permutation_replicates << "\n"
          << "include_observed="
          << (request.permutation_include_observed ? 1 : 0) << "\n"
          << "permutation_sha256=" << permutation_sha << "\n";
  for (std::size_t target = 0; target < target_keys.size(); ++target) {
    payload << "target[" << target << "]=" << target_keys[target] << "\n";
  }
  for (std::size_t pair = 0; pair < request.pairs.size(); ++pair) {
    const FullCudaCiMethodPairRequest& value = request.pairs[pair];
    payload << "pair[" << pair << "]=" << value.logical_sequence_id << "|"
            << value.left_target_index << "|" << value.right_target_index
            << "\n";
  }
  return full_cuda_ci_sha256_utf8(payload.str());
}

FullCudaCiMethodBatchResult run_full_cuda_ci_method_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiMethodBatchRequest& request) {
  const auto call_started = std::chrono::steady_clock::now();
  validate_request(request, batch);
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  if (prepared_info.prepared_s_key_sha256 !=
      request.expected_prepared_s_key_sha256 ||
      batch.target_keys.size() != static_cast<std::size_t>(batch.target_count) ||
      batch.output_mask != FixedSpOutputResiduals) {
    throw std::runtime_error("strict CI method prepared/batch identity mismatch");
  }
  const std::string actual_identity = full_cuda_ci_method_batch_request_identity(
    request, batch.target_keys, batch.n);
  if (!is_lower_sha256(request.request_identity_sha256) ||
      request.request_identity_sha256 != actual_identity) {
    throw std::runtime_error("strict CI method request identity mismatch");
  }

  FullCudaCiMethodBatchResult result;
  result.schema_version = kFullCudaCiMethodBatchResultSchemaVersion;
  result.request_identity_sha256 = request.request_identity_sha256;
  result.prepared_s_key_sha256 = prepared_info.prepared_s_key_sha256;
  result.target_keys = batch.target_keys;
  FullCudaCiMethodBatchDiagnostics& diagnostics = result.diagnostics;
  diagnostics.ci_method = request.ci_method;
  diagnostics.component_semantic_version = request.ci_method == "dcc.perm" ?
    "energy-dcov-centered-distance-v1" :
    "kernlab-inchol-rbf-tol0.001-centered-factor-v1";
  diagnostics.n = batch.n;
  diagnostics.target_count = batch.target_count;
  diagnostics.pair_count = static_cast<int>(request.pairs.size());
  diagnostics.permutation_replicates = request.permutation_replicates;
  diagnostics.request_identity_authenticated = true;
  diagnostics.prepared_identity_authenticated = true;

  std::vector<int> component_targets;
  std::vector<int> target_to_component(
    static_cast<std::size_t>(batch.target_count), -1);
  for (const FullCudaCiMethodPairRequest& pair : request.pairs) {
    for (int target : {pair.left_target_index, pair.right_target_index}) {
      if (target_to_component[static_cast<std::size_t>(target)] < 0) {
        target_to_component[static_cast<std::size_t>(target)] =
          static_cast<int>(component_targets.size());
        component_targets.push_back(target);
      }
    }
  }
  diagnostics.referenced_component_count =
    static_cast<int>(component_targets.size());
  diagnostics.component_build_count = diagnostics.referenced_component_count;
  diagnostics.pair_evaluation_count = diagnostics.pair_count;

  std::vector<int> left_components(request.pairs.size());
  std::vector<int> right_components(request.pairs.size());
  std::vector<unsigned long long> logical_ids(request.pairs.size());
  for (std::size_t pair = 0; pair < request.pairs.size(); ++pair) {
    left_components[pair] = target_to_component[
      static_cast<std::size_t>(request.pairs[pair].left_target_index)];
    right_components[pair] = target_to_component[
      static_cast<std::size_t>(request.pairs[pair].right_target_index)];
    logical_ids[pair] = request.pairs[pair].logical_sequence_id;
  }

  int caller_device = -1;
  check_cuda(cudaGetDevice(&caller_device), "capture strict CI caller device");
  std::shared_ptr<DeviceResidualBatch> residual_token;
  cudaEvent_t registered_completion = nullptr;
  try {
    const auto solve_started = std::chrono::steady_clock::now();
    residual_token = solve_fixed_sp_batch(prepared_s, batch);
    diagnostics.residual_solve_host_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - solve_started).count();
    const DeviceResidualConsumerView residual_view =
      acquire_device_residual_consumer_view(residual_token);
    if (residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        residual_view.target_keys != batch.target_keys ||
        residual_view.residuals == nullptr ||
        residual_view.producer_completion_event == nullptr) {
      throw std::runtime_error("strict CI residual view identity mismatch");
    }
    diagnostics.target_identity_authenticated = true;
    diagnostics.residuals_device_resident = true;
    check_cuda(cudaSetDevice(residual_view.device_id),
               "select strict CI residual device");

    CudaStream stream;
    CudaEvent stage_start(true);
    CudaEvent stage_stop(true);
    CudaEvent consumer_completion(false);
    check_cuda(cudaStreamWaitEvent(
                 stream.get(), residual_view.producer_completion_event, 0),
               "wait for strict CI residual producer");

    const std::size_t component_count = component_targets.size();
    const std::size_t pair_count = request.pairs.size();
    const std::size_t cells = checked_multiply(
      static_cast<std::size_t>(batch.n), static_cast<std::size_t>(batch.n),
      "strict CI component cells");
    DeviceBuffer d_component_targets(sizeof(int) * component_count);
    DeviceBuffer d_left_components(sizeof(int) * pair_count);
    DeviceBuffer d_right_components(sizeof(int) * pair_count);
    DeviceBuffer d_logical_ids(sizeof(unsigned long long) * pair_count);
    DeviceBuffer d_records(sizeof(DeviceMethodRecord) * pair_count);
    diagnostics.metadata_h2d_count = 4;
    diagnostics.metadata_h2d_bytes = d_component_targets.bytes() +
      d_left_components.bytes() + d_right_components.bytes() +
      d_logical_ids.bytes();
    check_cuda(cudaMemcpyAsync(
      d_component_targets.get(), component_targets.data(),
      d_component_targets.bytes(), cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI component targets");
    check_cuda(cudaMemcpyAsync(
      d_left_components.get(), left_components.data(),
      d_left_components.bytes(), cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI left components");
    check_cuda(cudaMemcpyAsync(
      d_right_components.get(), right_components.data(),
      d_right_components.bytes(), cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI right components");
    check_cuda(cudaMemcpyAsync(
      d_logical_ids.get(), logical_ids.data(), d_logical_ids.bytes(),
      cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI logical IDs");

    DeviceBuffer d_permutations;
    if (!request.permutations.empty()) {
      d_permutations.allocate(sizeof(int) * request.permutations.size());
      ++diagnostics.metadata_h2d_count;
      diagnostics.metadata_h2d_bytes += d_permutations.bytes();
      check_cuda(cudaMemcpyAsync(
        d_permutations.get(), request.permutations.data(),
        d_permutations.bytes(), cudaMemcpyHostToDevice, stream.get()),
        "copy strict CI permutations");
    }

    check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
               "record strict CI component start");
    std::vector<std::unique_ptr<DeviceBuffer>> method_buffers;
    if (request.ci_method == "dcc.perm") {
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count * cells));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count * batch.n));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * pair_count));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(int) * pair_count));
      double* centered = static_cast<double*>(method_buffers[0]->get());
      double* row_sums = static_cast<double*>(method_buffers[1]->get());
      double* observed_sums =
        static_cast<double*>(method_buffers[2]->get());
      int* exceedances = static_cast<int*>(method_buffers[3]->get());
      const dim3 distance_grid(batch.n,
                               static_cast<unsigned int>(component_count));
      build_distance_components_kernel<<<distance_grid, kBlockSize, 0,
          stream.get()>>>(
        residual_view.residuals,
        static_cast<const int*>(d_component_targets.get()), batch.n,
        static_cast<int>(component_count), centered, row_sums);
      const int cell_blocks = static_cast<int>(
        (cells + kBlockSize - 1U) / kBlockSize);
      const dim3 center_grid(cell_blocks,
                             static_cast<unsigned int>(component_count));
      center_distance_components_kernel<<<center_grid, kBlockSize, 0,
          stream.get()>>>(centered, row_sums, batch.n,
                          static_cast<int>(component_count));
      check_cuda(cudaGetLastError(), "launch strict dcc.perm components");
      check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
                 "record strict dcc.perm component stop");
      check_cuda(cudaEventSynchronize(stage_stop.get()),
                 "wait strict dcc.perm components");
      diagnostics.component_build_cuda_ms =
        elapsed_event_ms(stage_start.get(), stage_stop.get());

      check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
                 "record strict dcc.perm pair start");
      initialize_dcc_permutation_pairs_kernel<<<
        static_cast<unsigned int>(pair_count), kBlockSize, 0, stream.get()>>>(
        centered,
        static_cast<const int*>(d_left_components.get()),
        static_cast<const int*>(d_right_components.get()),
        batch.n, static_cast<int>(pair_count),
        request.permutation_include_observed ? 1 : 0,
        observed_sums, exceedances);
      const dim3 permutation_grid(
        static_cast<unsigned int>(pair_count),
        static_cast<unsigned int>(request.permutation_replicates));
      evaluate_dcc_permutation_replicates_kernel<<<
        permutation_grid, kBlockSize, 0, stream.get()>>>(
        centered,
        static_cast<const int*>(d_left_components.get()),
        static_cast<const int*>(d_right_components.get()),
        static_cast<const int*>(d_permutations.get()), observed_sums,
        batch.n, static_cast<int>(pair_count),
        request.permutation_replicates, exceedances);
      const int finalize_blocks = static_cast<int>(
        (pair_count + kBlockSize - 1U) / kBlockSize);
      finalize_dcc_permutation_pairs_kernel<<<
        finalize_blocks, kBlockSize, 0, stream.get()>>>(
        static_cast<const unsigned long long*>(d_logical_ids.get()),
        observed_sums, exceedances, batch.n,
        static_cast<int>(pair_count), request.permutation_replicates,
        request.permutation_include_observed ? 1 : 0,
        static_cast<DeviceMethodRecord*>(d_records.get()));
      check_cuda(cudaGetLastError(), "launch strict dcc.perm pairs");
    } else {
      const int max_rank = std::min(request.num_col, batch.n);
      const std::size_t factor_values = checked_multiply(
        checked_multiply(component_count, static_cast<std::size_t>(batch.n),
                         "strict HSIC factor components"),
        static_cast<std::size_t>(max_rank), "strict HSIC factor rank");
      const std::size_t triangular_values = checked_multiply(
        component_count,
        checked_multiply(static_cast<std::size_t>(max_rank),
                         static_cast<std::size_t>(max_rank),
                         "strict HSIC triangular square"),
        "strict HSIC triangular components");
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * factor_values));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * triangular_values));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count * batch.n));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count * batch.n));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count * max_rank));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(int) * component_count * max_rank));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(int) * component_count));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count));
      method_buffers.emplace_back(new DeviceBuffer(
        sizeof(double) * component_count));
      if (request.permutation_replicates > 0) {
        method_buffers.emplace_back(new DeviceBuffer(
          sizeof(double) * pair_count));
        method_buffers.emplace_back(new DeviceBuffer(
          sizeof(int) * pair_count));
      }
      double* factors = static_cast<double*>(method_buffers[0]->get());
      int* ranks = static_cast<int*>(method_buffers[6]->get());
      double* off_diagonal = static_cast<double*>(method_buffers[7]->get());
      double* self_moments = static_cast<double*>(method_buffers[8]->get());
      check_cuda(cudaMemsetAsync(
        factors, 0, method_buffers[0]->bytes(), stream.get()),
        "zero strict HSIC factors");
      build_inchol_components_kernel<<<
        static_cast<unsigned int>(component_count), 1, 0, stream.get()>>>(
        residual_view.residuals,
        static_cast<const int*>(d_component_targets.get()), batch.n,
        static_cast<int>(component_count), max_rank, 1.0 / request.hsic_sig,
        factors,
        static_cast<double*>(method_buffers[1]->get()),
        static_cast<double*>(method_buffers[2]->get()),
        static_cast<double*>(method_buffers[3]->get()),
        static_cast<double*>(method_buffers[4]->get()),
        static_cast<int*>(method_buffers[5]->get()), ranks, off_diagonal);
      center_inchol_components_kernel<<<
        static_cast<unsigned int>(component_count), 1, 0, stream.get()>>>(
        factors, ranks, batch.n, static_cast<int>(component_count), max_rank);
      const std::size_t shared_bytes = sizeof(double) *
        static_cast<std::size_t>(max_rank) * max_rank;
      reduce_inchol_self_moments_kernel<<<
        static_cast<unsigned int>(component_count), kBlockSize, shared_bytes,
        stream.get()>>>(factors, ranks, batch.n,
                        static_cast<int>(component_count), max_rank,
                        self_moments);
      check_cuda(cudaGetLastError(), "launch strict HSIC components");
      check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
                 "record strict HSIC component stop");
      check_cuda(cudaEventSynchronize(stage_stop.get()),
                 "wait strict HSIC components");
      diagnostics.component_build_cuda_ms =
        elapsed_event_ms(stage_start.get(), stage_stop.get());

      check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
                 "record strict HSIC pair start");
      if (request.permutation_replicates > 0) {
        double* observed_statistics =
          static_cast<double*>(method_buffers[9]->get());
        int* exceedances = static_cast<int*>(method_buffers[10]->get());
        initialize_hsic_permutation_pairs_kernel<<<
          static_cast<unsigned int>(pair_count), kBlockSize, shared_bytes,
          stream.get()>>>(
          factors, ranks,
          static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()),
          batch.n, static_cast<int>(pair_count), max_rank,
          request.permutation_include_observed ? 1 : 0,
          observed_statistics, exceedances);
        const dim3 permutation_grid(
          static_cast<unsigned int>(pair_count),
          static_cast<unsigned int>(request.permutation_replicates));
        evaluate_hsic_permutation_replicates_kernel<<<
          permutation_grid, kBlockSize, shared_bytes, stream.get()>>>(
          factors, ranks,
          static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()),
          static_cast<const int*>(d_permutations.get()),
          observed_statistics, batch.n, static_cast<int>(pair_count),
          max_rank, request.permutation_replicates, exceedances);
        const int finalize_blocks = static_cast<int>(
          (pair_count + kBlockSize - 1U) / kBlockSize);
        finalize_hsic_permutation_pairs_kernel<<<
          finalize_blocks, kBlockSize, 0, stream.get()>>>(
          static_cast<const unsigned long long*>(d_logical_ids.get()),
          observed_statistics, exceedances, static_cast<int>(pair_count),
          request.permutation_replicates,
          request.permutation_include_observed ? 1 : 0,
          static_cast<DeviceMethodRecord*>(d_records.get()));
      } else {
        evaluate_hsic_kernel<<<
          static_cast<unsigned int>(pair_count), kBlockSize, shared_bytes,
          stream.get()>>>(
          factors, ranks, off_diagonal, self_moments,
          static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()),
          static_cast<const unsigned long long*>(d_logical_ids.get()),
          nullptr, batch.n, static_cast<int>(pair_count), max_rank, 0,
          request.permutation_include_observed ? 1 : 0, request.alpha,
          static_cast<DeviceMethodRecord*>(d_records.get()));
      }
      check_cuda(cudaGetLastError(), "launch strict HSIC pairs");
    }
    check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
               "record strict CI pair stop");
    check_cuda(cudaEventSynchronize(stage_stop.get()),
               "wait strict CI pairs");
    diagnostics.pair_evaluation_cuda_ms =
      elapsed_event_ms(stage_start.get(), stage_stop.get());

    check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
               "record strict CI compact start");
    std::vector<DeviceMethodRecord> host_records(pair_count);
    check_cuda(cudaMemcpyAsync(
      host_records.data(), d_records.get(), d_records.bytes(),
      cudaMemcpyDeviceToHost, stream.get()),
      "copy strict CI compact records");
    check_cuda(cudaEventRecord(consumer_completion.get(), stream.get()),
               "record strict CI consumer completion");
    registered_completion = consumer_completion.get();
    register_device_residual_consumer_event(
      residual_token, registered_completion);
    check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
               "record strict CI compact stop");
    check_cuda(cudaEventSynchronize(stage_stop.get()),
               "wait strict CI compact results");
    diagnostics.compact_d2h_cuda_ms =
      elapsed_event_ms(stage_start.get(), stage_stop.get());
    diagnostics.compact_result_d2h_count = 1;
    diagnostics.compact_result_d2h_bytes = d_records.bytes();
    diagnostics.compact_result_only_d2h = true;

    result.records.reserve(pair_count);
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      const DeviceMethodRecord& source = host_records[pair];
      if (source.logical_sequence_id != request.pairs[pair].logical_sequence_id ||
          !std::isfinite(source.p_value) || source.p_value < 0.0 ||
          source.p_value > 1.0) {
        throw std::runtime_error("strict CI compact result is invalid");
      }
      FullCudaCiMethodCompactRecord record;
      record.logical_sequence_id = source.logical_sequence_id;
      record.p_value = source.p_value;
      record.statistic = source.statistic;
      record.mean = source.mean;
      record.variance = source.variance;
      record.status = source.status;
      const int left_target = request.pairs[pair].left_target_index;
      const int right_target = request.pairs[pair].right_target_index;
      record.solver_route = std::string(fixed_sp_route_name(
        residual_view.executed_routes[static_cast<std::size_t>(left_target)])) +
        "+" + fixed_sp_route_name(
          residual_view.executed_routes[static_cast<std::size_t>(right_target)]);
      result.records.push_back(std::move(record));
    }

    std::size_t allocated = d_component_targets.bytes() +
      d_left_components.bytes() + d_right_components.bytes() +
      d_logical_ids.bytes() + d_records.bytes() + d_permutations.bytes();
    for (const std::unique_ptr<DeviceBuffer>& buffer : method_buffers) {
      allocated += buffer->bytes();
    }
    diagnostics.device_allocation_bytes = allocated;
    diagnostics.peak_device_allocation_bytes = allocated;

    check_cuda(cudaEventSynchronize(consumer_completion.get()),
               "wait strict CI residual consumer completion");
    release_device_residual(residual_token);
    free_device_residual(&residual_token);
    registered_completion = nullptr;
    for (std::unique_ptr<DeviceBuffer>& buffer : method_buffers) buffer->close();
    d_permutations.close();
    d_records.close();
    d_logical_ids.close();
    d_right_components.close();
    d_left_components.close();
    d_component_targets.close();
    consumer_completion.close();
    stage_stop.close();
    stage_start.close();
    stream.close();
    check_cuda(cudaSetDevice(caller_device), "restore strict CI caller device");
    diagnostics.caller_device_restored = true;
  } catch (...) {
    if (residual_token) {
      try {
        release_device_residual(residual_token);
      } catch (...) {
      }
      try {
        free_device_residual(&residual_token);
      } catch (...) {
      }
    }
    (void)registered_completion;
    cudaSetDevice(caller_device);
    throw;
  }
  diagnostics.total_host_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - call_started).count();
  return result;
}

}  // namespace fastkpc
