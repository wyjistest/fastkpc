#include "full_cuda_ci_method_batch.hpp"

#include "../full_cuda_ci_contract.hpp"
#include "third_party/glibc_2_35_exp_fma.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <list>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

struct StrictMethodFailureInjectionState {
  std::mutex mutex;
  std::atomic<bool> armed{false};
  std::atomic<bool> identity_tamper_armed{false};
  std::string armed_stage;
  std::string observed_stage;
  std::string identity_tamper_layer;
  int checkpoint_count = 0;
  bool triggered = false;
};

StrictMethodFailureInjectionState& strict_method_failure_injection_state() {
  static StrictMethodFailureInjectionState state;
  return state;
}

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

  bool ensure_capacity(std::size_t bytes) {
    if (bytes == 0U) {
      throw std::runtime_error("invalid strict CI device capacity request");
    }
    if (bytes <= bytes_) return false;
    close();
    allocate(bytes);
    return true;
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
  if (component >= component_count) return;
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

  for (int row = static_cast<int>(threadIdx.x); row < n;
       row += static_cast<int>(blockDim.x)) {
    diagonal[row] = 1.0;
  }
  for (int entry = static_cast<int>(threadIdx.x);
       entry < max_rank * max_rank;
       entry += static_cast<int>(blockDim.x)) {
    upper[entry] = 0.0;
  }
  __shared__ double shared_pivot_value;
  __shared__ double shared_residue;
  __shared__ double shared_tau;
  __shared__ int shared_pivot;
  __shared__ int shared_rank;
  __shared__ int shared_active;
  if (threadIdx.x == 0) {
    shared_residue = 1.0;
    shared_pivot = 0;
    shared_rank = 0;
    shared_active = 1;
  }
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0) {
      shared_active = shared_residue > kIncholTolerance &&
        shared_rank < max_rank;
      if (shared_active != 0) {
        shared_pivot_value = values[shared_pivot];
      }
    }
    __syncthreads();
    if (shared_active == 0) break;

    for (int row = static_cast<int>(threadIdx.x); row < n;
         row += static_cast<int>(blockDim.x)) {
      const double pivot_value = shared_pivot_value;
      const double difference = values[row] - pivot_value;
      column[row] = glibc235::exp_fma_rn(
        -sigma * difference * difference);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      double tau_square = column[shared_pivot];
      if (shared_rank == 0) {
        upper[0] = sqrt(tau_square);
      } else {
        for (int row = 0; row < shared_rank; ++row) {
          double value = column[component_pivots[row]];
          for (int previous = 0; previous < row; ++previous) {
            value -= upper[
              previous + static_cast<std::size_t>(max_rank) * row] *
              solve[previous];
          }
          solve[row] = value /
            upper[row + static_cast<std::size_t>(max_rank) * row];
          tau_square -= solve[row] * solve[row];
          upper[row + static_cast<std::size_t>(max_rank) * shared_rank] =
            solve[row];
        }
      }
      if (!(tau_square > 0.0) || !isfinite(tau_square)) {
        shared_active = 0;
      } else {
        shared_tau = sqrt(tau_square);
        upper[shared_rank +
              static_cast<std::size_t>(max_rank) * shared_rank] = shared_tau;
      }
    }
    __syncthreads();
    if (shared_active == 0) break;

    const int rank = shared_rank;
    const double tau = shared_tau;
    for (int row = static_cast<int>(threadIdx.x); row < n;
         row += static_cast<int>(blockDim.x)) {
      double projection = 0.0;
      for (int previous = 0; previous < rank; ++previous) {
        projection += factor[row + static_cast<std::size_t>(n) * previous] *
          solve[previous];
      }
      const double update = (column[row] - projection) / tau;
      factor[row + static_cast<std::size_t>(n) * rank] = update;
      diagonal[row] -= update * update;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      component_pivots[shared_rank] = shared_pivot;
      ++shared_rank;
      shared_residue = diagonal[0];
      shared_pivot = 0;
      for (int row = 1; row < n; ++row) {
        if (diagonal[row] > shared_residue) {
          shared_residue = diagonal[row];
          shared_pivot = row;
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    ranks[component] = shared_rank;
    double total_sum = 0.0;
    double diagonal_sum = 0.0;
    for (int column_index = 0; column_index < shared_rank; ++column_index) {
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
}

__global__ void center_inchol_components_kernel(double* factors,
                                                const int* ranks,
                                                int n,
                                                int component_count,
                                                int max_rank) {
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count) return;
  double* factor = factors +
    static_cast<std::size_t>(component) * n * max_rank;
  const int rank = ranks[component];
  __shared__ double shared_mean;
  for (int column = 0; column < rank; ++column) {
    if (threadIdx.x == 0) {
      double mean = 0.0;
      for (int row = 0; row < n; ++row) {
        mean += factor[row + static_cast<std::size_t>(n) * column];
      }
      shared_mean = mean / static_cast<double>(n);
    }
    __syncthreads();
    for (int row = static_cast<int>(threadIdx.x); row < n;
         row += static_cast<int>(blockDim.x)) {
      factor[row + static_cast<std::size_t>(n) * column] -= shared_mean;
    }
    __syncthreads();
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

void validate_static_request(const FullCudaCiMethodStaticRequest& request,
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
  } else if (request.permutation_replicates != 0) {
    throw std::runtime_error("HSIC gamma received permutation metadata");
  }
}

void validate_permutation_request(
    const FullCudaCiMethodStaticRequest& request,
    const SealedPermutationTableHandle& permutation_table,
    const FixedSpBatchHostView& batch) {
  const bool permutation = request.ci_method == "dcc.perm" ||
    request.ci_method == "hsic.perm";
  if (permutation) {
    if (!permutation_table.sealed()) {
      throw std::runtime_error("strict permutation CI requires replicates");
    }
    const std::size_t expected = checked_multiply(
      checked_multiply(request.pairs.size(),
                       static_cast<std::size_t>(request.permutation_replicates),
                       "strict CI permutation pairs"),
      static_cast<std::size_t>(batch.n), "strict CI permutation rows");
    if (permutation_table.size() != expected) {
      throw std::runtime_error("strict CI permutation table size mismatch");
    }
  } else if (permutation_table.sealed() ||
             permutation_table.size() != 0U) {
    throw std::runtime_error("HSIC gamma received permutation metadata");
  }
}

void validate_request(const FullCudaCiMethodBatchRequest& request,
                      const FixedSpBatchHostView& batch) {
  const FullCudaCiMethodStaticRequest static_request =
    full_cuda_ci_method_static_request(request);
  validate_static_request(static_request, batch);
  validate_permutation_request(static_request, request.permutation_table, batch);
}

}  // namespace

void test_arm_strict_method_failure_injection(const std::string& stage) {
  if (stage.empty() || stage.size() > 96U) {
    throw std::runtime_error("strict method failure stage is invalid");
  }
  StrictMethodFailureInjectionState& state =
    strict_method_failure_injection_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.armed.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
      "strict method failure injection is already armed");
  }
  state.armed_stage = stage;
  state.observed_stage.clear();
  state.checkpoint_count = 0;
  state.triggered = false;
  state.armed.store(true, std::memory_order_release);
}

StrictMethodFailureInjectionSnapshot
test_strict_method_failure_injection_snapshot() {
  StrictMethodFailureInjectionState& state =
    strict_method_failure_injection_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  StrictMethodFailureInjectionSnapshot snapshot;
  snapshot.armed_stage = state.armed_stage;
  snapshot.observed_stage = state.observed_stage;
  snapshot.checkpoint_count = state.checkpoint_count;
  snapshot.triggered = state.triggered;
  return snapshot;
}

void strict_method_failure_checkpoint(const char* stage) {
  StrictMethodFailureInjectionState& state =
    strict_method_failure_injection_state();
  if (!state.armed.load(std::memory_order_acquire)) return;
  const std::string observed = stage == nullptr ? std::string() : stage;
  bool trigger = false;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.armed.load(std::memory_order_relaxed)) return;
    state.checkpoint_count += 1;
    if (observed == state.armed_stage) {
      state.observed_stage = observed;
      state.triggered = true;
      state.armed.store(false, std::memory_order_release);
      trigger = true;
    }
  }
  if (trigger) {
    throw std::runtime_error(
      "injected strict method failure: " + observed);
  }
}

void test_arm_strict_method_identity_tamper(const std::string& layer) {
  if (layer != "static" && layer != "permutation" && layer != "combined") {
    throw std::runtime_error("strict method identity tamper layer is invalid");
  }
  StrictMethodFailureInjectionState& state =
    strict_method_failure_injection_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.identity_tamper_armed.load(std::memory_order_relaxed)) {
    throw std::runtime_error(
      "strict method identity tamper injection is already armed");
  }
  state.identity_tamper_layer = layer;
  state.identity_tamper_armed.store(true, std::memory_order_release);
}

std::string test_take_strict_method_identity_tamper() {
  StrictMethodFailureInjectionState& state =
    strict_method_failure_injection_state();
  if (!state.identity_tamper_armed.load(std::memory_order_acquire)) {
    return std::string();
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.identity_tamper_armed.load(std::memory_order_relaxed)) {
    return std::string();
  }
  std::string layer = state.identity_tamper_layer;
  state.identity_tamper_layer.clear();
  state.identity_tamper_armed.store(false, std::memory_order_release);
  return layer;
}

SealedPermutationTableHandle::SealedPermutationTableHandle(
    SealedPermutationTableHandle&& other) noexcept
    : values_(std::move(other.values_)),
      sha256_(other.sha256_),
      sealed_(other.sealed_) {
  other.sha256_.fill(0U);
  other.sealed_ = false;
}

SealedPermutationTableHandle& SealedPermutationTableHandle::operator=(
    SealedPermutationTableHandle&& other) noexcept {
  if (this == &other) return *this;
  values_ = std::move(other.values_);
  sha256_ = other.sha256_;
  sealed_ = other.sealed_;
  other.sha256_.fill(0U);
  other.sealed_ = false;
  return *this;
}

const int* SealedPermutationTableHandle::data() const noexcept {
  return values_.empty() ? nullptr : values_.data();
}

std::size_t SealedPermutationTableHandle::size() const noexcept {
  return values_.size();
}

std::size_t SealedPermutationTableHandle::byte_size() const noexcept {
  return values_.size() * sizeof(int);
}

const std::array<unsigned char, 32>&
SealedPermutationTableHandle::sha256() const noexcept {
  return sha256_;
}

bool SealedPermutationTableHandle::sealed() const noexcept {
  return sealed_;
}

class FullCudaCiMethodResidualCache {
 public:
  static constexpr std::size_t kGatherCapacity = 64U;

  struct Entry {
    std::size_t slot = 0U;
    FixedSpRoute route = FixedSpRoute::Unset;
    FixedSpStatus status = static_cast<FixedSpStatus>(-1);
  };

  FullCudaCiMethodResidualCache(int n,
                                int device_id,
                                std::size_t capacity_entries)
      : n_(n), device_id_(device_id), capacity_entries_(capacity_entries),
        slab_(checked_multiply(
          checked_multiply(capacity_entries_, static_cast<std::size_t>(n_),
                           "strict CI residual cache entries"),
          sizeof(double), "strict CI residual cache bytes")),
        gather_(checked_multiply(
          checked_multiply(kGatherCapacity, static_cast<std::size_t>(n_),
                           "strict CI residual gather entries"),
          sizeof(double), "strict CI residual gather bytes")),
        slot_keys_(capacity_entries_) {
    if (n_ <= 0 || device_id_ < 0 || capacity_entries_ < 2U) {
      throw std::runtime_error("strict CI residual cache shape is invalid");
    }
  }

  FullCudaCiMethodResidualCache(const FullCudaCiMethodResidualCache&) = delete;
  FullCudaCiMethodResidualCache& operator=(
    const FullCudaCiMethodResidualCache&) = delete;

  int device_id() const { return device_id_; }
  std::size_t gather_capacity() const { return kGatherCapacity; }

  bool lookup_all(const std::vector<std::string>& keys,
                  const std::vector<FixedSpRoute>& planned_routes,
                  std::vector<Entry>* found,
                  FullCudaCiMethodBatchDiagnostics* diagnostics) const {
    if (keys.empty() || keys.size() > kGatherCapacity ||
        planned_routes.size() != keys.size() ||
        found == nullptr || diagnostics == nullptr) {
      throw std::runtime_error("strict CI residual cache lookup is malformed");
    }
    found->clear();
    found->reserve(keys.size());
    bool all_hit = true;
    diagnostics->residual_cache_lookup_count +=
      static_cast<int>(keys.size());
    diagnostics->residual_cache_capacity_entries = capacity_entries_;
    diagnostics->residual_cache_device_bytes = slab_.bytes() + gather_.bytes();
    for (std::size_t target = 0; target < keys.size(); ++target) {
      const auto entry = entries_.find(cache_identity(
        keys[target], planned_routes[target]));
      if (entry == entries_.end()) {
        all_hit = false;
        found->push_back(Entry{});
      } else {
        diagnostics->residual_cache_hit_count += 1;
        found->push_back(entry->second);
      }
    }
    if (all_hit) {
      diagnostics->residual_cache_all_hit_batch_count += 1;
      diagnostics->residual_cache_bypassed_target_count +=
        static_cast<int>(keys.size());
    }
    return all_hit;
  }

  void store_missing(const DeviceResidualConsumerView& residual,
                     const std::vector<FixedSpRoute>& planned_routes,
                     cudaStream_t stream,
                     FullCudaCiMethodBatchDiagnostics* diagnostics) {
    if (diagnostics == nullptr || residual.n != n_ ||
        residual.device_id != device_id_ || residual.residuals == nullptr ||
        residual.target_keys.size() !=
          static_cast<std::size_t>(residual.target_count) ||
        planned_routes.size() != residual.target_keys.size() ||
        residual.executed_routes.size() != residual.target_keys.size() ||
        residual.solver_statuses.size() != residual.target_keys.size()) {
      throw std::runtime_error("strict CI residual cache insert is malformed");
    }
    for (int target = 0; target < residual.target_count; ++target) {
      const std::string key = cache_identity(
        residual.target_keys[static_cast<std::size_t>(target)],
        planned_routes[static_cast<std::size_t>(target)]);
      if (entries_.find(key) != entries_.end()) continue;

      std::size_t slot = 0U;
      if (slots_used_ < capacity_entries_) {
        slot = slots_used_++;
      } else {
        slot = next_evict_slot_;
        const std::string& evicted = slot_keys_[slot];
        const std::size_t erased = entries_.erase(evicted);
        if (evicted.empty() || erased != 1U) {
          throw std::runtime_error(
            "strict CI residual cache eviction identity changed");
        }
        next_evict_slot_ = (next_evict_slot_ + 1U) % capacity_entries_;
        diagnostics->residual_cache_eviction_count += 1;
      }

      check_cuda(cudaMemcpyAsync(
        static_cast<double*>(slab_.get()) + slot * n_,
        residual.residuals + static_cast<std::size_t>(target) * n_,
        sizeof(double) * static_cast<std::size_t>(n_),
        cudaMemcpyDeviceToDevice, stream),
        "store strict CI residual cache entry");
      Entry entry;
      entry.slot = slot;
      entry.route = residual.executed_routes[static_cast<std::size_t>(target)];
      entry.status = residual.solver_statuses[static_cast<std::size_t>(target)];
      entries_.emplace(key, entry);
      slot_keys_[slot] = key;
      diagnostics->residual_cache_insert_count += 1;
    }
  }

  const double* gather(const std::vector<Entry>& entries,
                       cudaStream_t stream,
                       std::vector<FixedSpRoute>* routes,
                       std::vector<FixedSpStatus>* statuses,
                       FullCudaCiMethodBatchDiagnostics* diagnostics) {
    if (entries.empty() || entries.size() > kGatherCapacity ||
        routes == nullptr || statuses == nullptr ||
        diagnostics == nullptr) {
      throw std::runtime_error("strict CI residual cache gather is malformed");
    }
    routes->clear();
    statuses->clear();
    routes->reserve(entries.size());
    statuses->reserve(entries.size());
    for (std::size_t target = 0; target < entries.size(); ++target) {
      if (entries[target].route == FixedSpRoute::Unset ||
          static_cast<int>(entries[target].status) < 0 ||
          entries[target].slot >= slots_used_) {
        throw std::runtime_error(
          "strict CI residual cache gathered an invalid entry");
      }
      check_cuda(cudaMemcpyAsync(
        static_cast<double*>(gather_.get()) + target * n_,
        static_cast<const double*>(slab_.get()) + entries[target].slot * n_,
        sizeof(double) * static_cast<std::size_t>(n_),
        cudaMemcpyDeviceToDevice, stream),
        "gather strict CI residual cache entry");
      routes->push_back(entries[target].route);
      statuses->push_back(entries[target].status);
    }
    diagnostics->residual_cache_gather_d2d_bytes +=
      sizeof(double) * static_cast<std::size_t>(n_) * entries.size();
    return static_cast<const double*>(gather_.get());
  }

 private:
  static std::string cache_identity(const std::string& residual_key,
                                    FixedSpRoute planned_route) {
    return residual_key + "|planned-route=" +
      fixed_sp_route_name(planned_route);
  }

  int n_ = 0;
  int device_id_ = -1;
  std::size_t capacity_entries_ = 0U;
  DeviceBuffer slab_;
  DeviceBuffer gather_;
  std::unordered_map<std::string, Entry> entries_;
  std::vector<std::string> slot_keys_;
  std::size_t slots_used_ = 0U;
  std::size_t next_evict_slot_ = 0U;
};

class FullCudaCiMethodExecutionContext {
 public:
  struct HsicComponentEntry {
    std::size_t slot = 0U;
    std::list<std::string>::iterator order;
  };

  FullCudaCiMethodExecutionContext(int n,
                                   std::string ci_method,
                                   int num_col,
                                   int permutation_replicates,
                                   int device_id)
      : n_(n), ci_method_(std::move(ci_method)), num_col_(num_col),
        permutation_replicates_(permutation_replicates),
        device_id_(device_id), stage_start_(true), stage_stop_(true),
        consumer_completion_(false) {
    if (n_ < 6 || device_id_ < 0 || num_col_ < 1 || num_col_ > 64 ||
        (ci_method_ != "dcc.perm" && ci_method_ != "hsic.gamma" &&
         ci_method_ != "hsic.perm") ||
        ((ci_method_ == "dcc.perm" || ci_method_ == "hsic.perm") &&
         permutation_replicates_ < 1) ||
        (ci_method_ == "hsic.gamma" && permutation_replicates_ != 0)) {
      throw std::runtime_error(
        "strict CI execution context shape is invalid");
    }
    initialize_hsic_component_cache();
  }

  void validate(const FullCudaCiMethodStaticRequest& request, int n) const {
    if (n != n_ || request.ci_method != ci_method_ ||
        request.num_col != num_col_ ||
        request.permutation_replicates != permutation_replicates_) {
      throw std::runtime_error(
        "strict CI execution context identity changed");
    }
  }

  bool begin_call() {
    const bool reused = call_count_ > 0;
    ++call_count_;
    return reused;
  }

  void ensure(DeviceBuffer* buffer, std::size_t bytes) {
    if (buffer == nullptr) {
      throw std::runtime_error("strict CI execution buffer is missing");
    }
    if (buffer->ensure_capacity(bytes)) ++buffer_growth_count_;
  }

  std::size_t device_bytes() const {
    std::size_t bytes = component_targets_.bytes() +
      left_components_.bytes() + right_components_.bytes() +
      logical_ids_.bytes() + records_.bytes() + permutations_.bytes();
    for (const DeviceBuffer& buffer : method_buffers_) {
      bytes += buffer.bytes();
    }
    bytes += hsic_cache_factors_.bytes() + hsic_cache_ranks_.bytes();
    return bytes;
  }

  bool hsic_component_cache_enabled() const {
    return hsic_cache_capacity_ >= 2U;
  }

  std::size_t hsic_component_cache_capacity() const {
    return hsic_cache_capacity_;
  }

  std::size_t hsic_component_cache_bytes() const {
    return hsic_cache_factors_.bytes() + hsic_cache_ranks_.bytes();
  }

  std::string hsic_component_key(const std::string& residual_key,
                                 FixedSpRoute route,
                                 FixedSpStatus status,
                                 double hsic_sig) const {
    std::ostringstream output;
    output << "semantic=kernlab-inchol-rbf-tol0.001-centered-factor-v1\n"
           << "residual=" << residual_key << "\n"
           << "route=" << fixed_sp_route_name(route) << "\n"
           << "status=" << static_cast<int>(status) << "\n"
           << "n=" << n_ << "\n"
           << "numCol=" << num_col_ << "\n"
           << "sig=" << std::hexfloat << hsic_sig << "\n";
    return output.str();
  }

  bool lookup_hsic_component(const std::string& key,
                             std::size_t* slot) {
    const auto found = hsic_cache_entries_.find(key);
    if (found == hsic_cache_entries_.end()) return false;
    hsic_cache_order_.splice(
      hsic_cache_order_.begin(), hsic_cache_order_, found->second.order);
    found->second.order = hsic_cache_order_.begin();
    *slot = found->second.slot;
    return true;
  }

  std::size_t reserve_hsic_component(const std::string& key,
                                     bool* inserted,
                                     bool* evicted) {
    const auto found = hsic_cache_entries_.find(key);
    if (found != hsic_cache_entries_.end()) {
      hsic_cache_order_.splice(
        hsic_cache_order_.begin(), hsic_cache_order_, found->second.order);
      found->second.order = hsic_cache_order_.begin();
      *inserted = false;
      *evicted = false;
      return found->second.slot;
    }
    std::size_t slot = 0U;
    *evicted = false;
    if (hsic_cache_slots_used_ < hsic_cache_capacity_) {
      slot = hsic_cache_slots_used_++;
    } else {
      if (hsic_cache_order_.empty()) {
        throw std::runtime_error("strict HSIC component cache is malformed");
      }
      const std::string victim = hsic_cache_order_.back();
      const auto victim_entry = hsic_cache_entries_.find(victim);
      if (victim_entry == hsic_cache_entries_.end()) {
        throw std::runtime_error(
          "strict HSIC component cache victim is missing");
      }
      slot = victim_entry->second.slot;
      hsic_cache_entries_.erase(victim_entry);
      hsic_cache_order_.pop_back();
      *evicted = true;
    }
    hsic_cache_order_.push_front(key);
    hsic_cache_entries_.emplace(
      key, HsicComponentEntry{slot, hsic_cache_order_.begin()});
    *inserted = true;
    return slot;
  }

  std::size_t hsic_factor_stride() const {
    return static_cast<std::size_t>(n_) * std::min(num_col_, n_);
  }

 private:
  void initialize_hsic_component_cache() {
    if (ci_method_ != "hsic.perm") return;
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "query strict HSIC component cache memory");
    (void)total_bytes;
    constexpr std::size_t budget = 384U * 1024U * 1024U;
    const std::size_t bounded_budget = std::min(budget, free_bytes / 8U);
    const std::size_t slot_bytes = hsic_factor_stride() * sizeof(double) +
      sizeof(int);
    hsic_cache_capacity_ = std::min<std::size_t>(
      131072U, slot_bytes == 0U ? 0U : bounded_budget / slot_bytes);
    if (hsic_cache_capacity_ < 2U) {
      hsic_cache_capacity_ = 0U;
      return;
    }
    hsic_cache_factors_.allocate(
      hsic_cache_capacity_ * hsic_factor_stride() * sizeof(double));
    hsic_cache_ranks_.allocate(hsic_cache_capacity_ * sizeof(int));
  }

 public:

  int device_id() const { return device_id_; }
  int call_count() const { return call_count_; }
  int buffer_growth_count() const { return buffer_growth_count_; }

  int n_ = 0;
  std::string ci_method_;
  int num_col_ = 0;
  int permutation_replicates_ = 0;
  int device_id_ = -1;
  int call_count_ = 0;
  int buffer_growth_count_ = 0;
  CudaStream stream_;
  CudaEvent stage_start_;
  CudaEvent stage_stop_;
  CudaEvent consumer_completion_;
  DeviceBuffer component_targets_;
  DeviceBuffer left_components_;
  DeviceBuffer right_components_;
  DeviceBuffer logical_ids_;
  DeviceBuffer records_;
  DeviceBuffer permutations_;
  std::array<DeviceBuffer, 11> method_buffers_;
  std::vector<DeviceMethodRecord> host_records_;
  std::size_t hsic_cache_capacity_ = 0U;
  std::size_t hsic_cache_slots_used_ = 0U;
  DeviceBuffer hsic_cache_factors_;
  DeviceBuffer hsic_cache_ranks_;
  std::list<std::string> hsic_cache_order_;
  std::unordered_map<std::string, HsicComponentEntry> hsic_cache_entries_;
};

std::shared_ptr<FullCudaCiMethodResidualCache>
create_full_cuda_ci_method_residual_cache(int n, std::size_t byte_budget) {
  if (n <= 0) {
    return std::shared_ptr<FullCudaCiMethodResidualCache>();
  }
  int device_id = -1;
  check_cuda(cudaGetDevice(&device_id),
             "capture strict CI residual cache device");
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
             "query strict CI residual cache memory");
  (void)total_bytes;
  const std::size_t bounded_budget = std::min(byte_budget, free_bytes / 8U);
  const std::size_t bytes_per_entry =
    sizeof(double) * static_cast<std::size_t>(n);
  const std::size_t gather_bytes =
    FullCudaCiMethodResidualCache::kGatherCapacity * bytes_per_entry;
  if (bounded_budget < gather_bytes + 2U * bytes_per_entry) {
    return std::shared_ptr<FullCudaCiMethodResidualCache>();
  }
  const std::size_t capacity = std::min<std::size_t>(
    131072U, (bounded_budget - gather_bytes) / bytes_per_entry);
  if (capacity < 2U) {
    return std::shared_ptr<FullCudaCiMethodResidualCache>();
  }
  return std::make_shared<FullCudaCiMethodResidualCache>(
    n, device_id, capacity);
}

std::shared_ptr<FullCudaCiMethodExecutionContext>
create_full_cuda_ci_method_execution_context(
    int n,
    const std::string& ci_method,
    int num_col,
    int permutation_replicates) {
  int device_id = -1;
  check_cuda(cudaGetDevice(&device_id),
             "capture strict CI execution context device");
  return std::make_shared<FullCudaCiMethodExecutionContext>(
    n, ci_method, num_col, permutation_replicates, device_id);
}

struct MethodPreparationTicket::Impl {
  std::chrono::steady_clock::time_point call_started;
  FullCudaCiMethodStaticRequest request;
  FixedSpBatchHostView batch;
  std::shared_ptr<PreparedSGpuHandle> prepared_s;
  std::shared_ptr<FullCudaCiMethodResidualCache> residual_cache;
  std::shared_ptr<FullCudaCiMethodExecutionContext> active_context;
  FullCudaCiMethodBatchResult result;
  std::shared_ptr<DeviceResidualBatch> residual_token;
  cudaEvent_t registered_completion = nullptr;
  cudaEvent_t preparation_completion_event = nullptr;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<FixedSpStatus> solver_statuses;
  std::vector<std::size_t> method_buffer_bytes;
  std::size_t component_count = 0U;
  std::size_t pair_count = 0U;
  std::size_t cells = 0U;
  std::size_t component_target_bytes = 0U;
  std::size_t component_index_bytes = 0U;
  std::size_t logical_id_bytes = 0U;
  std::size_t record_bytes = 0U;
  int context_growth_before = 0;
  int caller_device = -1;
  bool hsic_component_cache = false;
  bool finalized = false;

  void cleanup_noexcept() noexcept {
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
    registered_completion = nullptr;
    if (caller_device >= 0) cudaSetDevice(caller_device);
    finalized = true;
  }
};

MethodPreparationTicket::~MethodPreparationTicket() {
  if (impl_ && !impl_->finalized) impl_->cleanup_noexcept();
}

MethodPreparationTicket::MethodPreparationTicket(
    MethodPreparationTicket&&) noexcept = default;
MethodPreparationTicket& MethodPreparationTicket::operator=(
    MethodPreparationTicket&& other) noexcept {
  if (this == &other) return *this;
  if (impl_ && !impl_->finalized) impl_->cleanup_noexcept();
  impl_ = std::move(other.impl_);
  return *this;
}

bool MethodPreparationTicket::valid() const noexcept {
  return impl_ != nullptr && !impl_->finalized;
}

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

FullCudaCiMethodStaticRequest full_cuda_ci_method_static_request(
    const FullCudaCiMethodBatchRequest& request) {
  FullCudaCiMethodStaticRequest output;
  output.expected_prepared_s_key_sha256 =
    request.expected_prepared_s_key_sha256;
  output.identity = request.static_identity;
  output.ci_method = request.ci_method;
  output.pairs = request.pairs;
  output.alpha = request.alpha;
  output.index = request.index;
  output.num_col = request.num_col;
  output.hsic_sig = request.hsic_sig;
  output.permutation_replicates = request.permutation_replicates;
  output.permutation_include_observed =
    request.permutation_include_observed;
  return output;
}

StaticRequestIdentity full_cuda_ci_method_static_request_identity(
    const FullCudaCiMethodStaticRequest& request,
    const std::vector<std::string>& target_keys,
    const std::vector<FixedSpRoute>& planned_routes,
    int n) {
  if (target_keys.size() != planned_routes.size()) {
    throw std::runtime_error(
      "strict CI static identity target route count changed");
  }
  std::ostringstream payload;
  payload << "schema="
          << kFullCudaCiMethodStaticRequestIdentitySchemaVersion << "\n"
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
          << (request.permutation_include_observed ? 1 : 0) << "\n";
  for (std::size_t target = 0; target < target_keys.size(); ++target) {
    payload << "target[" << target << "]=" << target_keys[target] << "\n";
    payload << "planned_route[" << target << "]="
            << fixed_sp_route_name(planned_routes[target]) << "\n";
  }
  for (std::size_t pair = 0; pair < request.pairs.size(); ++pair) {
    const FullCudaCiMethodPairRequest& value = request.pairs[pair];
    payload << "pair[" << pair << "]=" << value.logical_sequence_id << "|"
            << value.left_target_index << "|" << value.right_target_index
            << "\n";
  }
  StaticRequestIdentity identity;
  identity.schema_version =
    kFullCudaCiMethodStaticRequestIdentitySchemaVersion;
  identity.sha256 = full_cuda_ci_sha256_utf8(payload.str());
  return identity;
}

StaticRequestIdentity full_cuda_ci_method_static_request_identity(
    const FullCudaCiMethodBatchRequest& request,
    const std::vector<std::string>& target_keys,
    const std::vector<FixedSpRoute>& planned_routes,
    int n) {
  return full_cuda_ci_method_static_request_identity(
    full_cuda_ci_method_static_request(request), target_keys, planned_routes,
    n);
}

namespace {

PermutationAttestation permutation_attestation_from_table(
    const SealedPermutationTableHandle& permutation_table,
    int permutation_replicates) {
  static_assert(sizeof(int) == 4U,
                "strict permutation payload requires 32-bit int");
  PermutationAttestation attestation;
  attestation.schema_version =
    kFullCudaCiMethodPermutationAttestationSchemaVersion;
  if (permutation_table.sealed()) {
    attestation.payload_sha256 = permutation_table.sha256();
    attestation.value_count = permutation_table.size();
    attestation.byte_count = permutation_table.byte_size();
  } else {
    FullCudaCiSha256Builder empty_digest;
    empty_digest.reset();
    attestation.payload_sha256 = empty_digest.finish();
  }
  attestation.replicates = permutation_replicates;
  std::ostringstream payload;
  payload << "schema="
          << kFullCudaCiMethodPermutationAttestationSchemaVersion << "\n"
          << "payload_layout=pair-major-replicate-major-row-major-int32\n"
          << "value_count=" << attestation.value_count << "\n"
          << "byte_count=" << attestation.byte_count << "\n"
          << "replicates=" << attestation.replicates << "\n"
          << "payload_sha256="
          << full_cuda_ci_sha256_hex(attestation.payload_sha256) << "\n";
  attestation.sha256 = full_cuda_ci_sha256_utf8(payload.str());
  return attestation;
}

}  // namespace

PermutationAttestation full_cuda_ci_method_permutation_attestation(
    const FullCudaCiMethodBatchRequest& request) {
  return permutation_attestation_from_table(
    request.permutation_table, request.permutation_replicates);
}

CombinedRequestIdentity full_cuda_ci_method_combined_request_identity(
    const StaticRequestIdentity& static_identity,
    const PermutationAttestation& permutation_attestation) {
  std::ostringstream payload;
  payload << "schema="
          << kFullCudaCiMethodCombinedRequestIdentitySchemaVersion << "\n"
          << "static_schema=" << static_identity.schema_version << "\n"
          << "static_sha256=" << static_identity.sha256 << "\n"
          << "permutation_schema="
          << permutation_attestation.schema_version << "\n"
          << "permutation_attestation_sha256="
          << permutation_attestation.sha256 << "\n";
  CombinedRequestIdentity identity;
  identity.schema_version =
    kFullCudaCiMethodCombinedRequestIdentitySchemaVersion;
  identity.sha256 = full_cuda_ci_sha256_utf8(payload.str());
  return identity;
}

MethodPreparationTicket submit_method_preparation(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiMethodStaticRequest& request,
    const std::shared_ptr<FullCudaCiMethodResidualCache>& residual_cache,
    const std::shared_ptr<FullCudaCiMethodExecutionContext>&
      execution_context) {
  const auto call_started = std::chrono::steady_clock::now();
  strict_method_failure_checkpoint("before_validate_request");
  validate_static_request(request, batch);
  strict_method_failure_checkpoint("after_validate_request");
  std::shared_ptr<FullCudaCiMethodExecutionContext> active_context =
    execution_context;
  if (!active_context) {
    active_context = create_full_cuda_ci_method_execution_context(
      batch.n, request.ci_method, request.num_col,
      request.permutation_replicates);
  }
  active_context->validate(request, batch.n);
  strict_method_failure_checkpoint("after_execution_context_validate");
  const int context_growth_before = active_context->buffer_growth_count();
  const bool context_reused = active_context->begin_call();
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  if (prepared_info.prepared_s_key_sha256 !=
      request.expected_prepared_s_key_sha256 ||
      batch.target_keys.size() != static_cast<std::size_t>(batch.target_count) ||
      batch.output_mask != FixedSpOutputResiduals) {
    throw std::runtime_error("strict CI method prepared/batch identity mismatch");
  }
  strict_method_failure_checkpoint("after_prepared_identity_validate");
  const auto identity_started = std::chrono::steady_clock::now();
  const StaticRequestIdentity actual_static_identity =
    full_cuda_ci_method_static_request_identity(
      request, batch.target_keys, batch.planned_routes, batch.n);
  const double static_identity_validation_host_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - identity_started).count();
  if (request.identity.schema_version !=
        kFullCudaCiMethodStaticRequestIdentitySchemaVersion ||
      !is_lower_sha256(request.identity.sha256) ||
      request.identity.sha256 != actual_static_identity.sha256) {
    throw std::runtime_error("strict CI method static identity mismatch");
  }
  strict_method_failure_checkpoint("after_static_identity_validate");

  FullCudaCiMethodBatchResult result;
  result.schema_version = kFullCudaCiMethodBatchResultSchemaVersion;
  result.static_request_identity_sha256 = request.identity.sha256;
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
  diagnostics.execution_context_call_count = 1;
  diagnostics.execution_context_reuse_count = context_reused ? 1 : 0;
  diagnostics.preparation_submit_count = 1;
  diagnostics.static_identity_validation_host_ms =
    static_identity_validation_host_ms;
  diagnostics.request_identity_validation_host_ms =
    static_identity_validation_host_ms;
  diagnostics.static_identity_authenticated = true;
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
  std::vector<FullCudaCiMethodResidualCache::Entry> cached_entries;
  const bool residual_cache_eligible = residual_cache &&
    batch.target_count >= 1 &&
    static_cast<std::size_t>(batch.target_count) <=
      residual_cache->gather_capacity();
  const bool residual_cache_all_hit = residual_cache_eligible &&
    residual_cache->lookup_all(
      batch.target_keys, batch.planned_routes, &cached_entries, &diagnostics);
  try {
    DeviceResidualConsumerView residual_view;
    if (!residual_cache_all_hit) {
      const auto solve_started = std::chrono::steady_clock::now();
      strict_method_failure_checkpoint("before_residual_solve");
      residual_token = solve_fixed_sp_batch(prepared_s, batch);
      diagnostics.residual_solve_host_ms =
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - solve_started).count();
      residual_view = acquire_device_residual_consumer_view(residual_token);
      strict_method_failure_checkpoint("after_residual_acquire");
      if (residual_view.n != batch.n ||
          residual_view.target_count != batch.target_count ||
          residual_view.target_keys != batch.target_keys ||
          residual_view.residuals == nullptr ||
          residual_view.producer_completion_event == nullptr) {
        throw std::runtime_error("strict CI residual view identity mismatch");
      }
      check_cuda(cudaSetDevice(residual_view.device_id),
                 "select strict CI residual device");
      if (residual_view.device_id != active_context->device_id()) {
        throw std::runtime_error(
          "strict CI execution context device changed");
      }
    } else {
      check_cuda(cudaSetDevice(residual_cache->device_id()),
                 "select strict CI cached residual device");
      if (residual_cache->device_id() != active_context->device_id()) {
        throw std::runtime_error(
          "strict CI cached execution context device changed");
      }
    }
    diagnostics.target_identity_authenticated = true;
    diagnostics.residuals_device_resident = true;

    CudaStream& stream = active_context->stream_;
    CudaEvent& stage_start = active_context->stage_start_;
    CudaEvent& stage_stop = active_context->stage_stop_;
    CudaEvent& consumer_completion = active_context->consumer_completion_;
    const double* residual_values = nullptr;
    std::vector<FixedSpRoute> executed_routes;
    std::vector<FixedSpStatus> solver_statuses;
    if (residual_cache_all_hit) {
      residual_values = residual_cache->gather(
        cached_entries, stream.get(), &executed_routes, &solver_statuses,
        &diagnostics);
    } else {
      check_cuda(cudaStreamWaitEvent(
                   stream.get(), residual_view.producer_completion_event, 0),
                 "wait for strict CI residual producer");
      if (residual_cache_eligible) {
        residual_cache->store_missing(
          residual_view, batch.planned_routes, stream.get(), &diagnostics);
      }
      residual_values = residual_view.residuals;
      executed_routes = residual_view.executed_routes;
      solver_statuses = residual_view.solver_statuses;
    }
    if (residual_values == nullptr ||
        executed_routes.size() != static_cast<std::size_t>(batch.target_count) ||
        solver_statuses.size() != static_cast<std::size_t>(batch.target_count)) {
      throw std::runtime_error("strict CI residual source identity changed");
    }

    std::vector<int> component_misses;
    std::vector<unsigned char> component_store_after_build(
      component_targets.size(), 0U);
    std::vector<std::size_t> component_cache_slots(
      component_targets.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::string> component_cache_keys(component_targets.size());
    const bool hsic_component_cache =
      active_context->hsic_component_cache_enabled();
    if (hsic_component_cache) {
      diagnostics.component_cache_persistent_capacity_entries =
        active_context->hsic_component_cache_capacity();
      diagnostics.component_cache_persistent_device_bytes =
        active_context->hsic_component_cache_bytes();
      diagnostics.component_cache_persistent_request_count =
        static_cast<int>(component_targets.size());
    }
    for (std::size_t component = 0;
         component < component_targets.size(); ++component) {
      const int target = component_targets[component];
      const bool first_residual = residual_cache_eligible &&
        !residual_cache_all_hit &&
        cached_entries[static_cast<std::size_t>(target)].route ==
          FixedSpRoute::Unset;
      const bool cache_payload_authoritative =
        residual_cache_all_hit || first_residual;
      bool hit = false;
      if (hsic_component_cache && residual_cache_all_hit) {
        component_cache_keys[component] =
          active_context->hsic_component_key(
            batch.target_keys[static_cast<std::size_t>(target)],
            executed_routes[static_cast<std::size_t>(target)],
            solver_statuses[static_cast<std::size_t>(target)],
            request.hsic_sig);
        diagnostics.component_cache_persistent_lookup_count += 1;
        hit = active_context->lookup_hsic_component(
          component_cache_keys[component],
          &component_cache_slots[component]);
        if (hit) diagnostics.component_cache_persistent_hit_count += 1;
      }
      if (!hit) {
        component_misses.push_back(static_cast<int>(component));
        if (hsic_component_cache && cache_payload_authoritative) {
          if (component_cache_keys[component].empty()) {
            component_cache_keys[component] =
              active_context->hsic_component_key(
                batch.target_keys[static_cast<std::size_t>(target)],
                executed_routes[static_cast<std::size_t>(target)],
                solver_statuses[static_cast<std::size_t>(target)],
                request.hsic_sig);
          }
          component_store_after_build[component] = 1U;
        }
      }
    }
    diagnostics.component_build_count =
      static_cast<int>(component_misses.size());
    diagnostics.component_cache_persistent_miss_count =
      hsic_component_cache ? diagnostics.component_build_count : 0;

    const std::size_t component_count = component_targets.size();
    const std::size_t pair_count = request.pairs.size();
    const std::size_t cells = checked_multiply(
      static_cast<std::size_t>(batch.n), static_cast<std::size_t>(batch.n),
      "strict CI component cells");
    const std::size_t component_target_bytes =
      sizeof(int) * component_count;
    const std::size_t component_index_bytes = sizeof(int) * pair_count;
    const std::size_t logical_id_bytes =
      sizeof(unsigned long long) * pair_count;
    const std::size_t record_bytes =
      sizeof(DeviceMethodRecord) * pair_count;
    DeviceBuffer& d_component_targets = active_context->component_targets_;
    DeviceBuffer& d_left_components = active_context->left_components_;
    DeviceBuffer& d_right_components = active_context->right_components_;
    DeviceBuffer& d_logical_ids = active_context->logical_ids_;
    DeviceBuffer& d_records = active_context->records_;
    active_context->ensure(&d_component_targets, component_target_bytes);
    active_context->ensure(&d_left_components, component_index_bytes);
    active_context->ensure(&d_right_components, component_index_bytes);
    active_context->ensure(&d_logical_ids, logical_id_bytes);
    active_context->ensure(&d_records, record_bytes);
    diagnostics.metadata_h2d_count = 4;
    diagnostics.metadata_h2d_bytes = component_target_bytes +
      2U * component_index_bytes + logical_id_bytes;
    check_cuda(cudaMemcpyAsync(
      d_component_targets.get(), component_targets.data(),
      component_target_bytes, cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI component targets");
    check_cuda(cudaMemcpyAsync(
      d_left_components.get(), left_components.data(),
      component_index_bytes, cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI left components");
    check_cuda(cudaMemcpyAsync(
      d_right_components.get(), right_components.data(),
      component_index_bytes, cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI right components");
    check_cuda(cudaMemcpyAsync(
      d_logical_ids.get(), logical_ids.data(), logical_id_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy strict CI logical IDs");

    check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
               "record strict CI component start");
    std::vector<DeviceBuffer*> method_buffers;
    std::vector<std::size_t> method_buffer_bytes;
    const auto add_method_buffer = [&](std::size_t bytes) {
      const std::size_t index = method_buffers.size();
      if (index >= active_context->method_buffers_.size()) {
        throw std::runtime_error(
          "strict CI method workspace capacity changed");
      }
      DeviceBuffer* buffer = &active_context->method_buffers_[index];
      active_context->ensure(buffer, bytes);
      method_buffers.push_back(buffer);
      method_buffer_bytes.push_back(bytes);
    };
    if (request.ci_method == "dcc.perm") {
      add_method_buffer(sizeof(double) * component_count * cells);
      add_method_buffer(sizeof(double) * component_count * batch.n);
      add_method_buffer(sizeof(double) * pair_count);
      add_method_buffer(sizeof(int) * pair_count);
      double* centered = static_cast<double*>(method_buffers[0]->get());
      double* row_sums = static_cast<double*>(method_buffers[1]->get());
      const dim3 distance_grid(batch.n,
                               static_cast<unsigned int>(component_count));
      build_distance_components_kernel<<<distance_grid, kBlockSize, 0,
          stream.get()>>>(
        residual_values,
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
      diagnostics.component_host_wait_count += 1;
      strict_method_failure_checkpoint("after_component_wait");
      diagnostics.component_build_cuda_ms =
        elapsed_event_ms(stage_start.get(), stage_stop.get());
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
      add_method_buffer(sizeof(double) * factor_values);
      add_method_buffer(sizeof(double) * triangular_values);
      add_method_buffer(sizeof(double) * component_count * batch.n);
      add_method_buffer(sizeof(double) * component_count * batch.n);
      add_method_buffer(sizeof(double) * component_count * max_rank);
      add_method_buffer(sizeof(int) * component_count * max_rank);
      add_method_buffer(sizeof(int) * component_count);
      add_method_buffer(sizeof(double) * component_count);
      add_method_buffer(sizeof(double) * component_count);
      if (request.permutation_replicates > 0) {
        add_method_buffer(sizeof(double) * pair_count);
        add_method_buffer(sizeof(int) * pair_count);
      }
      double* factors = static_cast<double*>(method_buffers[0]->get());
      int* ranks = static_cast<int*>(method_buffers[6]->get());
      double* off_diagonal = static_cast<double*>(method_buffers[7]->get());
      double* self_moments = static_cast<double*>(method_buffers[8]->get());
      const std::size_t factor_stride =
        static_cast<std::size_t>(batch.n) * max_rank;
      const std::size_t triangular_stride =
        static_cast<std::size_t>(max_rank) * max_rank;
      for (std::size_t component = 0; component < component_count;
           ++component) {
        const std::size_t slot = component_cache_slots[component];
        if (slot == std::numeric_limits<std::size_t>::max()) continue;
        check_cuda(cudaMemcpyAsync(
          factors + component * factor_stride,
          static_cast<const double*>(
            active_context->hsic_cache_factors_.get()) +
              slot * factor_stride,
          sizeof(double) * factor_stride, cudaMemcpyDeviceToDevice,
          stream.get()),
          "gather strict HSIC component factors");
        check_cuda(cudaMemcpyAsync(
          ranks + component,
          static_cast<const int*>(active_context->hsic_cache_ranks_.get()) +
            slot,
          sizeof(int), cudaMemcpyDeviceToDevice, stream.get()),
          "gather strict HSIC component rank");
        diagnostics.component_cache_persistent_gather_d2d_bytes +=
          sizeof(double) * factor_stride + sizeof(int);
      }
      const std::size_t shared_bytes = sizeof(double) *
        static_cast<std::size_t>(max_rank) * max_rank;
      if (component_misses.size() == component_count) {
        check_cuda(cudaMemsetAsync(
          factors, 0, sizeof(double) * factor_values, stream.get()),
          "zero strict HSIC factors");
        build_inchol_components_kernel<<<
          static_cast<unsigned int>(component_count), kBlockSize, 0,
          stream.get()>>>(
          residual_values,
          static_cast<const int*>(d_component_targets.get()), batch.n,
          static_cast<int>(component_count), max_rank, 1.0 / request.hsic_sig,
          factors,
          static_cast<double*>(method_buffers[1]->get()),
          static_cast<double*>(method_buffers[2]->get()),
          static_cast<double*>(method_buffers[3]->get()),
          static_cast<double*>(method_buffers[4]->get()),
          static_cast<int*>(method_buffers[5]->get()), ranks, off_diagonal);
        center_inchol_components_kernel<<<
          static_cast<unsigned int>(component_count), kBlockSize, 0,
          stream.get()>>>(
          factors, ranks, batch.n, static_cast<int>(component_count),
          max_rank);
        reduce_inchol_self_moments_kernel<<<
          static_cast<unsigned int>(component_count), kBlockSize, shared_bytes,
          stream.get()>>>(factors, ranks, batch.n,
                          static_cast<int>(component_count), max_rank,
                          self_moments);
      } else {
        for (int component : component_misses) {
          const std::size_t output = static_cast<std::size_t>(component);
          check_cuda(cudaMemsetAsync(
            factors + output * factor_stride, 0,
            sizeof(double) * factor_stride, stream.get()),
            "zero strict HSIC component factors");
          build_inchol_components_kernel<<<1U, kBlockSize, 0, stream.get()>>>(
            residual_values,
            static_cast<const int*>(d_component_targets.get()) + output,
            batch.n, 1, max_rank, 1.0 / request.hsic_sig,
            factors + output * factor_stride,
            static_cast<double*>(method_buffers[1]->get()) +
              output * triangular_stride,
            static_cast<double*>(method_buffers[2]->get()) + output * batch.n,
            static_cast<double*>(method_buffers[3]->get()) + output * batch.n,
            static_cast<double*>(method_buffers[4]->get()) + output * max_rank,
            static_cast<int*>(method_buffers[5]->get()) + output * max_rank,
            ranks + output, off_diagonal + output);
          center_inchol_components_kernel<<<1U, kBlockSize, 0, stream.get()>>>(
            factors + output * factor_stride, ranks + output, batch.n, 1,
            max_rank);
          reduce_inchol_self_moments_kernel<<<
            1U, kBlockSize, shared_bytes, stream.get()>>>(
            factors + output * factor_stride, ranks + output, batch.n, 1,
            max_rank, self_moments + output);
        }
      }
      for (int component : component_misses) {
        const std::size_t output = static_cast<std::size_t>(component);
        if (component_store_after_build[output] == 0U) continue;
        bool inserted = false;
        bool evicted = false;
        const std::size_t slot = active_context->reserve_hsic_component(
          component_cache_keys[output], &inserted, &evicted);
        if (!inserted) {
          throw std::runtime_error(
            "strict HSIC component cache duplicate store identity");
        }
        diagnostics.component_cache_persistent_insert_count += 1;
        if (evicted) {
          diagnostics.component_cache_persistent_eviction_count += 1;
        }
        check_cuda(cudaMemcpyAsync(
          static_cast<double*>(active_context->hsic_cache_factors_.get()) +
            slot * factor_stride,
          factors + output * factor_stride,
          sizeof(double) * factor_stride, cudaMemcpyDeviceToDevice,
          stream.get()),
          "store strict HSIC component factors");
        check_cuda(cudaMemcpyAsync(
          static_cast<int*>(active_context->hsic_cache_ranks_.get()) + slot,
          ranks + output, sizeof(int), cudaMemcpyDeviceToDevice, stream.get()),
          "store strict HSIC component rank");
        diagnostics.component_cache_persistent_store_d2d_bytes +=
          sizeof(double) * factor_stride + sizeof(int);
      }
      check_cuda(cudaGetLastError(), "launch strict HSIC components");
      check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
                 "record strict HSIC component stop");
      check_cuda(cudaEventSynchronize(stage_stop.get()),
                 "wait strict HSIC components");
      diagnostics.component_host_wait_count += 1;
      strict_method_failure_checkpoint("after_component_wait");
      diagnostics.component_build_cuda_ms =
        elapsed_event_ms(stage_start.get(), stage_stop.get());
    }
    MethodPreparationTicket ticket;
    ticket.impl_.reset(new MethodPreparationTicket::Impl());
    MethodPreparationTicket::Impl& state = *ticket.impl_;
    state.call_started = call_started;
    state.request = request;
    state.batch = batch;
    state.prepared_s = prepared_s;
    state.residual_cache = residual_cache;
    state.active_context = active_context;
    state.result = std::move(result);
    state.residual_token = std::move(residual_token);
    state.registered_completion = registered_completion;
    state.preparation_completion_event = stage_stop.get();
    state.executed_routes = std::move(executed_routes);
    state.solver_statuses = std::move(solver_statuses);
    state.method_buffer_bytes = std::move(method_buffer_bytes);
    state.component_count = component_count;
    state.pair_count = pair_count;
    state.cells = cells;
    state.component_target_bytes = component_target_bytes;
    state.component_index_bytes = component_index_bytes;
    state.logical_id_bytes = logical_id_bytes;
    state.record_bytes = record_bytes;
    state.context_growth_before = context_growth_before;
    state.caller_device = caller_device;
    state.hsic_component_cache = hsic_component_cache;
    return ticket;
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
}

FullCudaCiMethodBatchResult finalize_method_from_permutation(
    MethodPreparationTicket&& preparation,
    const PermutationAttestation& permutation_attestation,
    const CombinedRequestIdentity& combined_identity,
    const SealedPermutationTableHandle& permutation_table) {
  if (!preparation.valid()) {
    throw std::runtime_error("strict CI method preparation ticket is invalid");
  }
  std::unique_ptr<MethodPreparationTicket::Impl> state =
    std::move(preparation.impl_);
  FullCudaCiMethodBatchDiagnostics& diagnostics = state->result.diagnostics;
  try {
    validate_permutation_request(
      state->request, permutation_table, state->batch);

    const auto attestation_started = std::chrono::steady_clock::now();
    const PermutationAttestation actual_attestation =
      permutation_attestation_from_table(
        permutation_table, state->request.permutation_replicates);
    diagnostics.permutation_attestation_validation_host_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - attestation_started).count();
    if (permutation_attestation.schema_version !=
          kFullCudaCiMethodPermutationAttestationSchemaVersion ||
        !is_lower_sha256(permutation_attestation.sha256) ||
        permutation_attestation.payload_sha256 !=
          actual_attestation.payload_sha256 ||
        permutation_attestation.value_count != actual_attestation.value_count ||
        permutation_attestation.byte_count != actual_attestation.byte_count ||
        permutation_attestation.replicates != actual_attestation.replicates ||
        permutation_attestation.sha256 != actual_attestation.sha256) {
      throw std::runtime_error(
        "strict CI method permutation attestation mismatch");
    }
    strict_method_failure_checkpoint(
      "after_permutation_attestation_validate");

    const auto combined_started = std::chrono::steady_clock::now();
    const CombinedRequestIdentity actual_combined =
      full_cuda_ci_method_combined_request_identity(
        state->request.identity, actual_attestation);
    diagnostics.combined_identity_validation_host_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - combined_started).count();
    if (combined_identity.schema_version !=
          kFullCudaCiMethodCombinedRequestIdentitySchemaVersion ||
        !is_lower_sha256(combined_identity.sha256) ||
        combined_identity.sha256 != actual_combined.sha256) {
      throw std::runtime_error("strict CI method combined identity mismatch");
    }
    strict_method_failure_checkpoint("after_combined_identity_validate");

    diagnostics.request_identity_validation_host_ms +=
      diagnostics.permutation_attestation_validation_host_ms +
      diagnostics.combined_identity_validation_host_ms;
    diagnostics.finalization_count = 1;
    diagnostics.preparation_ticket_consumed = true;
    diagnostics.permutation_attestation_authenticated = true;
    diagnostics.combined_identity_authenticated = true;
    diagnostics.request_identity_authenticated = true;
    state->result.permutation_attestation_sha256 =
      permutation_attestation.sha256;
    state->result.combined_request_identity_sha256 = combined_identity.sha256;
    state->result.request_identity_sha256 = combined_identity.sha256;

    check_cuda(cudaSetDevice(state->active_context->device_id()),
               "select strict CI finalization device");
    CudaStream& stream = state->active_context->stream_;
    CudaEvent& stage_start = state->active_context->stage_start_;
    CudaEvent& stage_stop = state->active_context->stage_stop_;
    CudaEvent& consumer_completion =
      state->active_context->consumer_completion_;
    DeviceBuffer& d_left_components =
      state->active_context->left_components_;
    DeviceBuffer& d_right_components =
      state->active_context->right_components_;
    DeviceBuffer& d_logical_ids = state->active_context->logical_ids_;
    DeviceBuffer& d_records = state->active_context->records_;
    DeviceBuffer& d_permutations = state->active_context->permutations_;
    const std::size_t permutation_bytes = permutation_table.byte_size();
    if (permutation_table.size() != 0U) {
      state->active_context->ensure(&d_permutations, permutation_bytes);
      diagnostics.metadata_h2d_count += 1;
      diagnostics.metadata_h2d_bytes += permutation_bytes;
      const auto permutation_h2d_started = std::chrono::steady_clock::now();
      check_cuda(cudaMemcpyAsync(
        d_permutations.get(), permutation_table.data(), permutation_bytes,
        cudaMemcpyHostToDevice, stream.get()),
        "copy strict CI permutations");
      diagnostics.permutation_h2d_submit_host_ms =
        std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - permutation_h2d_started).count();
    }

    check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
               "record strict CI pair start");
    if (state->request.ci_method == "dcc.perm") {
      double* centered = static_cast<double*>(
        state->active_context->method_buffers_[0].get());
      double* observed_sums = static_cast<double*>(
        state->active_context->method_buffers_[2].get());
      int* exceedances = static_cast<int*>(
        state->active_context->method_buffers_[3].get());
      initialize_dcc_permutation_pairs_kernel<<<
        static_cast<unsigned int>(state->pair_count), kBlockSize, 0,
        stream.get()>>>(
        centered, static_cast<const int*>(d_left_components.get()),
        static_cast<const int*>(d_right_components.get()), state->batch.n,
        static_cast<int>(state->pair_count),
        state->request.permutation_include_observed ? 1 : 0,
        observed_sums, exceedances);
      const dim3 permutation_grid(
        static_cast<unsigned int>(state->pair_count),
        static_cast<unsigned int>(state->request.permutation_replicates));
      evaluate_dcc_permutation_replicates_kernel<<<
        permutation_grid, kBlockSize, 0, stream.get()>>>(
        centered, static_cast<const int*>(d_left_components.get()),
        static_cast<const int*>(d_right_components.get()),
        static_cast<const int*>(d_permutations.get()), observed_sums,
        state->batch.n, static_cast<int>(state->pair_count),
        state->request.permutation_replicates, exceedances);
      const int finalize_blocks = static_cast<int>(
        (state->pair_count + kBlockSize - 1U) / kBlockSize);
      finalize_dcc_permutation_pairs_kernel<<<
        finalize_blocks, kBlockSize, 0, stream.get()>>>(
        static_cast<const unsigned long long*>(d_logical_ids.get()),
        observed_sums, exceedances, state->batch.n,
        static_cast<int>(state->pair_count),
        state->request.permutation_replicates,
        state->request.permutation_include_observed ? 1 : 0,
        static_cast<DeviceMethodRecord*>(d_records.get()));
      check_cuda(cudaGetLastError(), "launch strict dcc.perm pairs");
    } else {
      const int max_rank = std::min(state->request.num_col, state->batch.n);
      const std::size_t shared_bytes = sizeof(double) *
        static_cast<std::size_t>(max_rank) * max_rank;
      double* factors = static_cast<double*>(
        state->active_context->method_buffers_[0].get());
      int* ranks = static_cast<int*>(
        state->active_context->method_buffers_[6].get());
      double* off_diagonal = static_cast<double*>(
        state->active_context->method_buffers_[7].get());
      double* self_moments = static_cast<double*>(
        state->active_context->method_buffers_[8].get());
      if (state->request.permutation_replicates > 0) {
        double* observed_statistics = static_cast<double*>(
          state->active_context->method_buffers_[9].get());
        int* exceedances = static_cast<int*>(
          state->active_context->method_buffers_[10].get());
        initialize_hsic_permutation_pairs_kernel<<<
          static_cast<unsigned int>(state->pair_count), kBlockSize,
          shared_bytes, stream.get()>>>(
          factors, ranks, static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()), state->batch.n,
          static_cast<int>(state->pair_count), max_rank,
          state->request.permutation_include_observed ? 1 : 0,
          observed_statistics, exceedances);
        const dim3 permutation_grid(
          static_cast<unsigned int>(state->pair_count),
          static_cast<unsigned int>(state->request.permutation_replicates));
        evaluate_hsic_permutation_replicates_kernel<<<
          permutation_grid, kBlockSize, shared_bytes, stream.get()>>>(
          factors, ranks, static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()),
          static_cast<const int*>(d_permutations.get()), observed_statistics,
          state->batch.n, static_cast<int>(state->pair_count), max_rank,
          state->request.permutation_replicates, exceedances);
        const int finalize_blocks = static_cast<int>(
          (state->pair_count + kBlockSize - 1U) / kBlockSize);
        finalize_hsic_permutation_pairs_kernel<<<
          finalize_blocks, kBlockSize, 0, stream.get()>>>(
          static_cast<const unsigned long long*>(d_logical_ids.get()),
          observed_statistics, exceedances,
          static_cast<int>(state->pair_count),
          state->request.permutation_replicates,
          state->request.permutation_include_observed ? 1 : 0,
          static_cast<DeviceMethodRecord*>(d_records.get()));
      } else {
        evaluate_hsic_kernel<<<
          static_cast<unsigned int>(state->pair_count), kBlockSize,
          shared_bytes, stream.get()>>>(
          factors, ranks, off_diagonal, self_moments,
          static_cast<const int*>(d_left_components.get()),
          static_cast<const int*>(d_right_components.get()),
          static_cast<const unsigned long long*>(d_logical_ids.get()),
          nullptr, state->batch.n, static_cast<int>(state->pair_count),
          max_rank, 0,
          state->request.permutation_include_observed ? 1 : 0,
          state->request.alpha,
          static_cast<DeviceMethodRecord*>(d_records.get()));
      }
      check_cuda(cudaGetLastError(), "launch strict HSIC pairs");
    }
    check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
               "record strict CI pair stop");
    check_cuda(cudaEventSynchronize(stage_stop.get()),
               "wait strict CI pairs");
    diagnostics.pair_host_wait_count += 1;
    strict_method_failure_checkpoint("after_pair_wait");
    diagnostics.pair_evaluation_cuda_ms =
      elapsed_event_ms(stage_start.get(), stage_stop.get());

    check_cuda(cudaEventRecord(stage_start.get(), stream.get()),
               "record strict CI compact start");
    std::vector<DeviceMethodRecord>& host_records =
      state->active_context->host_records_;
    host_records.resize(state->pair_count);
    check_cuda(cudaMemcpyAsync(
      host_records.data(), d_records.get(), state->record_bytes,
      cudaMemcpyDeviceToHost, stream.get()),
      "copy strict CI compact records");
    check_cuda(cudaEventRecord(consumer_completion.get(), stream.get()),
               "record strict CI consumer completion");
    if (state->residual_token) {
      state->registered_completion = consumer_completion.get();
      register_device_residual_consumer_event(
        state->residual_token, state->registered_completion);
    }
    check_cuda(cudaEventRecord(stage_stop.get(), stream.get()),
               "record strict CI compact stop");
    check_cuda(cudaEventSynchronize(stage_stop.get()),
               "wait strict CI compact results");
    diagnostics.compact_host_wait_count += 1;
    strict_method_failure_checkpoint("after_compact_wait");
    diagnostics.compact_d2h_cuda_ms =
      elapsed_event_ms(stage_start.get(), stage_stop.get());
    diagnostics.compact_result_d2h_count = 1;
    diagnostics.compact_result_d2h_bytes = state->record_bytes;
    diagnostics.compact_result_only_d2h = true;

    state->result.records.reserve(state->pair_count);
    for (std::size_t pair = 0; pair < state->pair_count; ++pair) {
      const DeviceMethodRecord& source = host_records[pair];
      if (source.logical_sequence_id !=
            state->request.pairs[pair].logical_sequence_id ||
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
      const int left_target =
        state->request.pairs[pair].left_target_index;
      const int right_target =
        state->request.pairs[pair].right_target_index;
      record.solver_route = std::string(fixed_sp_route_name(
        state->executed_routes[static_cast<std::size_t>(left_target)])) +
        "+" + fixed_sp_route_name(
          state->executed_routes[static_cast<std::size_t>(right_target)]);
      state->result.records.push_back(std::move(record));
    }

    std::size_t allocated = state->component_target_bytes +
      2U * state->component_index_bytes + state->logical_id_bytes +
      state->record_bytes + permutation_bytes;
    for (std::size_t bytes : state->method_buffer_bytes) allocated += bytes;
    diagnostics.device_allocation_bytes = allocated;
    diagnostics.peak_device_allocation_bytes = allocated;
    diagnostics.execution_context_buffer_growth_count =
      state->active_context->buffer_growth_count() -
        state->context_growth_before;
    diagnostics.execution_context_device_bytes =
      state->active_context->device_bytes();
    if (state->hsic_component_cache) {
      const std::size_t cached_component_bytes = sizeof(double) *
        state->active_context->hsic_factor_stride() + sizeof(int);
      if (diagnostics.component_cache_persistent_request_count !=
            diagnostics.component_cache_persistent_hit_count +
              diagnostics.component_cache_persistent_miss_count ||
          diagnostics.component_cache_persistent_miss_count !=
            diagnostics.component_build_count ||
          diagnostics.component_cache_persistent_gather_d2d_bytes !=
            static_cast<std::size_t>(
              diagnostics.component_cache_persistent_hit_count) *
              cached_component_bytes ||
          diagnostics.component_cache_persistent_store_d2d_bytes !=
            static_cast<std::size_t>(
              diagnostics.component_cache_persistent_insert_count) *
              cached_component_bytes) {
        throw std::runtime_error(
          "strict HSIC component cache accounting changed");
      }
    }

    check_cuda(cudaEventSynchronize(consumer_completion.get()),
               "wait strict CI residual consumer completion");
    diagnostics.consumer_host_wait_count += 1;
    strict_method_failure_checkpoint("after_consumer_wait");
    if (state->residual_token) {
      release_device_residual(state->residual_token);
      free_device_residual(&state->residual_token);
    }
    state->registered_completion = nullptr;
    check_cuda(cudaSetDevice(state->caller_device),
               "restore strict CI caller device");
    diagnostics.caller_device_restored = true;
    diagnostics.total_host_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - state->call_started).count();
    state->finalized = true;
    return std::move(state->result);
  } catch (...) {
    state->cleanup_noexcept();
    throw;
  }
}

FullCudaCiMethodBatchResult run_full_cuda_ci_method_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiMethodBatchRequest& request,
    const std::shared_ptr<FullCudaCiMethodResidualCache>& residual_cache,
    const std::shared_ptr<FullCudaCiMethodExecutionContext>&
      execution_context) {
  const auto call_started = std::chrono::steady_clock::now();
  validate_request(request, batch);
  const StaticRequestIdentity actual_static =
    full_cuda_ci_method_static_request_identity(
      request, batch.target_keys, batch.planned_routes, batch.n);
  if (request.static_identity.schema_version !=
        kFullCudaCiMethodStaticRequestIdentitySchemaVersion ||
      !is_lower_sha256(request.static_identity.sha256) ||
      request.static_identity.sha256 != actual_static.sha256) {
    throw std::runtime_error("strict CI method static identity mismatch");
  }
  strict_method_failure_checkpoint("after_static_identity_validate");
  const PermutationAttestation actual_attestation =
    full_cuda_ci_method_permutation_attestation(request);
  if (request.permutation_attestation.schema_version !=
        kFullCudaCiMethodPermutationAttestationSchemaVersion ||
      !is_lower_sha256(request.permutation_attestation.sha256) ||
      request.permutation_attestation.payload_sha256 !=
        actual_attestation.payload_sha256 ||
      request.permutation_attestation.value_count !=
        actual_attestation.value_count ||
      request.permutation_attestation.byte_count !=
        actual_attestation.byte_count ||
      request.permutation_attestation.replicates !=
        actual_attestation.replicates ||
      request.permutation_attestation.sha256 != actual_attestation.sha256) {
    throw std::runtime_error(
      "strict CI method permutation attestation mismatch");
  }
  strict_method_failure_checkpoint("after_permutation_attestation_validate");
  const CombinedRequestIdentity actual_combined =
    full_cuda_ci_method_combined_request_identity(
      actual_static, actual_attestation);
  if (request.combined_identity.schema_version !=
        kFullCudaCiMethodCombinedRequestIdentitySchemaVersion ||
      !is_lower_sha256(request.combined_identity.sha256) ||
      request.combined_identity.sha256 != actual_combined.sha256) {
    throw std::runtime_error("strict CI method combined identity mismatch");
  }
  strict_method_failure_checkpoint("after_combined_identity_validate");

  MethodPreparationTicket preparation = submit_method_preparation(
    prepared_s, batch, full_cuda_ci_method_static_request(request),
    residual_cache, execution_context);
  strict_method_failure_checkpoint("after_method_preparation_submit");
  strict_method_failure_checkpoint("before_method_finalization");
  FullCudaCiMethodBatchResult result = finalize_method_from_permutation(
    std::move(preparation), request.permutation_attestation,
    request.combined_identity, request.permutation_table);
  strict_method_failure_checkpoint("after_method_finalization");
  result.diagnostics.total_host_ms =
    std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - call_started).count();
  return result;
}

}  // namespace fastkpc
