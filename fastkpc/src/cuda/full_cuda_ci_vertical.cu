#include "full_cuda_ci_vertical.hpp"

#include "../full_cuda_ci_contract.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fastkpc {
namespace {

constexpr int kVerticalBlockSize = 256;
constexpr double kCanonicalAlpha = 0.1;
constexpr int kDcovStatusOk = 0;
constexpr int kDcovStatusInvalidMoment = 1;
constexpr int kDcovStatusNonfinite = 2;
constexpr int kDcovStatusGammaNoConvergence = 3;
constexpr int kDiagnosticExactCenteredDistance = 1 << 0;
constexpr int kDiagnosticEvictionReplay = 1 << 1;
constexpr int kDiagnosticLegacyFullEig = 1 << 2;

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(stage) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuBLAS status " +
      std::to_string(static_cast<int>(status)));
  }
}

void check_cusolver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
      std::string(stage) + ": cuSOLVER status " +
      std::to_string(static_cast<int>(status)));
  }
}

class VerticalCublasHandle {
 public:
  explicit VerticalCublasHandle(cudaStream_t stream) {
    check_cublas(cublasCreate(&handle_), "create legacy eig cuBLAS handle");
    try {
      check_cublas(cublasSetStream(handle_, stream),
                   "set legacy eig cuBLAS stream");
      check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                   "set legacy eig pedantic math");
      check_cublas(cublasSetAtomicsMode(
        handle_, CUBLAS_ATOMICS_NOT_ALLOWED),
        "disable legacy eig cuBLAS atomics");
    } catch (...) {
      cublasDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }
  ~VerticalCublasHandle() {
    if (handle_ != nullptr) cublasDestroy(handle_);
  }
  VerticalCublasHandle(const VerticalCublasHandle&) = delete;
  VerticalCublasHandle& operator=(const VerticalCublasHandle&) = delete;
  cublasHandle_t get() const { return handle_; }
  void close() {
    if (handle_ == nullptr) return;
    check_cublas(cublasDestroy(handle_),
                 "destroy legacy eig cuBLAS handle");
    handle_ = nullptr;
  }
 private:
  cublasHandle_t handle_ = nullptr;
};

class VerticalCusolverHandle {
 public:
  explicit VerticalCusolverHandle(cudaStream_t stream) {
    check_cusolver(cusolverDnCreate(&handle_),
                   "create legacy eig cuSOLVER handle");
    try {
      check_cusolver(cusolverDnSetStream(handle_, stream),
                     "set legacy eig cuSOLVER stream");
      check_cusolver(cusolverDnSetDeterministicMode(
        handle_, CUSOLVER_DETERMINISTIC_RESULTS),
        "set legacy eig deterministic mode");
    } catch (...) {
      cusolverDnDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }
  ~VerticalCusolverHandle() {
    if (handle_ != nullptr) cusolverDnDestroy(handle_);
  }
  VerticalCusolverHandle(const VerticalCusolverHandle&) = delete;
  VerticalCusolverHandle& operator=(const VerticalCusolverHandle&) = delete;
  cusolverDnHandle_t get() const { return handle_; }
  void close() {
    if (handle_ == nullptr) return;
    check_cusolver(cusolverDnDestroy(handle_),
                   "destroy legacy eig cuSOLVER handle");
    handle_ = nullptr;
  }
 private:
  cusolverDnHandle_t handle_ = nullptr;
};

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

struct VerticalResourceLedger {
  std::atomic<std::int64_t> live_device_allocations{0};
  std::atomic<std::int64_t> live_device_bytes{0};
  std::atomic<std::int64_t> live_streams{0};
  std::atomic<std::int64_t> live_events{0};
  std::atomic<std::int64_t> total_device_allocations{0};
  std::atomic<std::int64_t> total_device_frees{0};
  std::atomic<std::int64_t> total_stream_creates{0};
  std::atomic<std::int64_t> total_stream_destroys{0};
  std::atomic<std::int64_t> total_event_creates{0};
  std::atomic<std::int64_t> total_event_destroys{0};
};

VerticalResourceLedger& vertical_resource_ledger() {
  static VerticalResourceLedger ledger;
  return ledger;
}

struct VerticalCallAccounting {
  int device_allocation_count = 0;
  int device_free_count = 0;
  std::size_t device_allocation_bytes = 0;
  std::size_t live_device_bytes = 0;
  std::size_t peak_live_device_bytes = 0;
};

class TrackedDeviceBuffer {
 public:
  TrackedDeviceBuffer() = default;
  TrackedDeviceBuffer(std::size_t bytes, VerticalCallAccounting* accounting) {
    allocate(bytes, accounting);
  }
  ~TrackedDeviceBuffer() { close_noexcept(); }

  TrackedDeviceBuffer(const TrackedDeviceBuffer&) = delete;
  TrackedDeviceBuffer& operator=(const TrackedDeviceBuffer&) = delete;

  void allocate(std::size_t bytes, VerticalCallAccounting* accounting) {
    if (pointer_ != nullptr || bytes == 0U || accounting == nullptr) {
      throw std::runtime_error("invalid vertical device allocation request");
    }
    check_cuda(cudaMalloc(&pointer_, bytes), "allocate vertical device buffer");
    bytes_ = bytes;
    accounting_ = accounting;
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_device_allocations.fetch_add(1, std::memory_order_relaxed);
    ledger.live_device_bytes.fetch_add(
      static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    ledger.total_device_allocations.fetch_add(1, std::memory_order_relaxed);
    accounting_->device_allocation_count += 1;
    accounting_->device_allocation_bytes += bytes;
    accounting_->live_device_bytes += bytes;
    accounting_->peak_live_device_bytes = std::max(
      accounting_->peak_live_device_bytes, accounting_->live_device_bytes);
  }

  void close() {
    if (pointer_ == nullptr) return;
    check_cuda(cudaFree(pointer_), "free vertical device buffer");
    account_free();
  }

  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void account_free() noexcept {
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_device_allocations.fetch_sub(1, std::memory_order_relaxed);
    ledger.live_device_bytes.fetch_sub(
      static_cast<std::int64_t>(bytes_), std::memory_order_relaxed);
    ledger.total_device_frees.fetch_add(1, std::memory_order_relaxed);
    if (accounting_ != nullptr) {
      accounting_->device_free_count += 1;
      if (accounting_->live_device_bytes >= bytes_) {
        accounting_->live_device_bytes -= bytes_;
      } else {
        accounting_->live_device_bytes = 0U;
      }
    }
    pointer_ = nullptr;
    bytes_ = 0U;
    accounting_ = nullptr;
  }

  void close_noexcept() noexcept {
    if (pointer_ == nullptr) return;
    if (cudaFree(pointer_) == cudaSuccess) account_free();
  }

  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
  VerticalCallAccounting* accounting_ = nullptr;
};

class TrackedCudaStream {
 public:
  TrackedCudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create vertical consumer stream");
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_streams.fetch_add(1, std::memory_order_relaxed);
    ledger.total_stream_creates.fetch_add(1, std::memory_order_relaxed);
  }
  ~TrackedCudaStream() { close_noexcept(); }
  TrackedCudaStream(const TrackedCudaStream&) = delete;
  TrackedCudaStream& operator=(const TrackedCudaStream&) = delete;

  cudaStream_t get() const { return stream_; }

  void close() {
    if (stream_ == nullptr) return;
    check_cuda(cudaStreamDestroy(stream_), "destroy vertical consumer stream");
    stream_ = nullptr;
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_streams.fetch_sub(1, std::memory_order_relaxed);
    ledger.total_stream_destroys.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  void close_noexcept() noexcept {
    if (stream_ == nullptr) return;
    if (cudaStreamDestroy(stream_) == cudaSuccess) {
      stream_ = nullptr;
      VerticalResourceLedger& ledger = vertical_resource_ledger();
      ledger.live_streams.fetch_sub(1, std::memory_order_relaxed);
      ledger.total_stream_destroys.fetch_add(1, std::memory_order_relaxed);
    }
  }

  cudaStream_t stream_ = nullptr;
};

class TrackedCudaEvent {
 public:
  explicit TrackedCudaEvent(bool timing_enabled = false) {
    check_cuda(cudaEventCreateWithFlags(
                 &event_, timing_enabled ? cudaEventDefault :
                   cudaEventDisableTiming),
               "create vertical consumer event");
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_events.fetch_add(1, std::memory_order_relaxed);
    ledger.total_event_creates.fetch_add(1, std::memory_order_relaxed);
  }
  ~TrackedCudaEvent() { close_noexcept(); }
  TrackedCudaEvent(const TrackedCudaEvent&) = delete;
  TrackedCudaEvent& operator=(const TrackedCudaEvent&) = delete;

  cudaEvent_t get() const { return event_; }

  void close() {
    if (event_ == nullptr) return;
    check_cuda(cudaEventDestroy(event_), "destroy vertical consumer event");
    event_ = nullptr;
    VerticalResourceLedger& ledger = vertical_resource_ledger();
    ledger.live_events.fetch_sub(1, std::memory_order_relaxed);
    ledger.total_event_destroys.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  void close_noexcept() noexcept {
    if (event_ == nullptr) return;
    if (cudaEventDestroy(event_) == cudaSuccess) {
      event_ = nullptr;
      VerticalResourceLedger& ledger = vertical_resource_ledger();
      ledger.live_events.fetch_sub(1, std::memory_order_relaxed);
      ledger.total_event_destroys.fetch_add(1, std::memory_order_relaxed);
    }
  }

  cudaEvent_t event_ = nullptr;
};

struct DeviceCompactCiResult {
  unsigned long long logical_sequence_id = 0ULL;
  double p_value = 0.0;
  double statistic = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  double gamma_shape = 0.0;
  double gamma_scale = 0.0;
  int status = kDcovStatusNonfinite;
  int independent = 0;
  int gamma_iterations = 0;
  int diagnostic_flags = 0;
};

__device__ double regularized_gamma_q(double shape,
                                      double x,
                                      int* iterations,
                                      int* converged) {
  constexpr int max_iterations = 10000;
  constexpr double epsilon = 2.0e-15;
  constexpr double tiny = 1.0e-300;
  *iterations = 0;
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
        *iterations = iteration;
        *converged = 1;
        double lower = sum * exp(log_prefactor);
        lower = fmin(1.0, fmax(0.0, lower));
        return fmin(1.0, fmax(0.0, 1.0 - lower));
      }
    }
    *iterations = max_iterations;
    return nan("");
  }

  double b = x + 1.0 - shape;
  double c = 1.0 / tiny;
  double d = 1.0 / fmax(fabs(b), tiny);
  if (b < 0.0) d = -d;
  double fraction = d;
  for (int iteration = 1; iteration <= max_iterations; ++iteration) {
    const double i = static_cast<double>(iteration);
    const double an = -i * (i - shape);
    b += 2.0;
    d = an * d + b;
    if (fabs(d) < tiny) d = copysign(tiny, d == 0.0 ? 1.0 : d);
    c = b + an / c;
    if (fabs(c) < tiny) c = copysign(tiny, c == 0.0 ? 1.0 : c);
    d = 1.0 / d;
    const double delta = d * c;
    fraction *= delta;
    if (fabs(delta - 1.0) <= epsilon) {
      *iterations = iteration;
      *converged = 1;
      const double upper = exp(log_prefactor) * fraction;
      return fmin(1.0, fmax(0.0, upper));
    }
  }
  *iterations = max_iterations;
  return nan("");
}

__global__ void build_distance_rows_kernel(const double* residuals,
                                           int n,
                                           int target_index,
                                           double* distance,
                                           double* row_sums) {
  __shared__ double reduction[kVerticalBlockSize];
  const int row = static_cast<int>(blockIdx.x);
  if (row >= n) return;
  const double* values = residuals +
    static_cast<std::size_t>(target_index) * static_cast<std::size_t>(n);
  const double row_value = values[row];
  double sum = 0.0;
  for (int column = static_cast<int>(threadIdx.x);
       column < n; column += static_cast<int>(blockDim.x)) {
    const double value = fabs(row_value - values[column]);
    distance[static_cast<std::size_t>(row) * n + column] = value;
    sum += value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) row_sums[row] = reduction[0];
}

__global__ void reduce_rows_kernel(const double* row_sums,
                                   int n,
                                   double* total) {
  __shared__ double reduction[kVerticalBlockSize];
  double sum = 0.0;
  for (int row = static_cast<int>(threadIdx.x);
       row < n; row += static_cast<int>(blockDim.x)) {
    sum += row_sums[row];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *total = reduction[0];
}

__global__ void center_distance_kernel(double* distance,
                                       const double* row_sums,
                                       const double* total,
                                       int n,
                                       std::size_t cell_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (index >= cell_count) return;
  const int row = static_cast<int>(index / static_cast<std::size_t>(n));
  const int column = static_cast<int>(index % static_cast<std::size_t>(n));
  const double nd = static_cast<double>(n);
  distance[index] = distance[index] - row_sums[row] / nd -
    row_sums[column] / nd + *total / (nd * nd);
}

__global__ void self_moment_kernel(const double* centered,
                                   std::size_t cell_count,
                                   double* self_moment) {
  __shared__ double reduction[kVerticalBlockSize];
  double sum = 0.0;
  for (std::size_t index = threadIdx.x;
       index < cell_count; index += blockDim.x) {
    const double value = centered[index];
    sum += value * value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *self_moment = reduction[0];
}

__global__ void build_distance_rows_batch_kernel(
    const double* residuals,
    const int* target_indices,
    int n,
    int component_count,
    std::size_t cell_count,
    double* distance,
    double* row_sums) {
  __shared__ double reduction[kVerticalBlockSize];
  const int row = static_cast<int>(blockIdx.x);
  const int component = static_cast<int>(blockIdx.y);
  if (row >= n || component >= component_count) return;
  const int target_index = target_indices[component];
  const double* values = residuals +
    static_cast<std::size_t>(target_index) * static_cast<std::size_t>(n);
  const double row_value = values[row];
  double* component_distance = distance +
    static_cast<std::size_t>(component) * cell_count;
  double sum = 0.0;
  for (int column = static_cast<int>(threadIdx.x);
       column < n; column += static_cast<int>(blockDim.x)) {
    const double value = fabs(row_value - values[column]);
    component_distance[static_cast<std::size_t>(row) * n + column] = value;
    sum += value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    row_sums[static_cast<std::size_t>(component) * n + row] = reduction[0];
  }
}

__global__ void reduce_rows_batch_kernel(const double* row_sums,
                                         int n,
                                         int component_count,
                                         double* totals) {
  __shared__ double reduction[kVerticalBlockSize];
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count) return;
  const double* component_rows = row_sums +
    static_cast<std::size_t>(component) * static_cast<std::size_t>(n);
  double sum = 0.0;
  for (int row = static_cast<int>(threadIdx.x);
       row < n; row += static_cast<int>(blockDim.x)) {
    sum += component_rows[row];
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) totals[component] = reduction[0];
}

__global__ void center_distance_batch_kernel(
    double* distance,
    const double* row_sums,
    const double* totals,
    int n,
    std::size_t cell_count,
    std::size_t total_cell_count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
    blockDim.x + threadIdx.x;
  if (index >= total_cell_count) return;
  const std::size_t component = index / cell_count;
  const std::size_t local_index = index - component * cell_count;
  const int row = static_cast<int>(local_index /
    static_cast<std::size_t>(n));
  const int column = static_cast<int>(local_index %
    static_cast<std::size_t>(n));
  const double* component_rows = row_sums + component *
    static_cast<std::size_t>(n);
  const double nd = static_cast<double>(n);
  distance[index] = distance[index] - component_rows[row] / nd -
    component_rows[column] / nd + totals[component] / (nd * nd);
}

__global__ void self_moment_batch_kernel(const double* centered,
                                         int component_count,
                                         std::size_t cell_count,
                                         double* self_moments) {
  __shared__ double reduction[kVerticalBlockSize];
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count) return;
  const double* values = centered +
    static_cast<std::size_t>(component) * cell_count;
  double sum = 0.0;
  for (std::size_t index = threadIdx.x;
       index < cell_count; index += blockDim.x) {
    const double value = values[index];
    sum += value * value;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) self_moments[component] = reduction[0];
}

__global__ void evaluate_pair_kernel(
    const double* left_centered,
    const double* right_centered,
    const double* left_total,
    const double* right_total,
    const double* left_self_moment,
    const double* right_self_moment,
    std::size_t cell_count,
    int n,
    unsigned long long logical_sequence_id,
    double alpha,
    int diagnostic_flags,
    DeviceCompactCiResult* result) {
  __shared__ double reduction[kVerticalBlockSize];
  double cross_sum = 0.0;
  for (std::size_t index = threadIdx.x;
       index < cell_count; index += blockDim.x) {
    cross_sum += left_centered[index] * right_centered[index];
  }
  reduction[threadIdx.x] = cross_sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) return;

  DeviceCompactCiResult output;
  output.logical_sequence_id = logical_sequence_id;
  output.diagnostic_flags = diagnostic_flags;
  const double nd = static_cast<double>(n);
  output.statistic = reduction[0] / nd;
  output.mean = (*left_total / (nd * nd)) *
    (*right_total / (nd * nd));
  const double variance_factor =
    2.0 * (nd - 4.0) * (nd - 5.0) /
    nd / (nd - 1.0) / (nd - 2.0) / (nd - 3.0);
  output.variance = variance_factor * (*left_self_moment) *
    (*right_self_moment) / (nd * nd);

  if (!isfinite(output.statistic) || !isfinite(output.mean) ||
      !isfinite(output.variance)) {
    output.status = kDcovStatusNonfinite;
    *result = output;
    return;
  }
  if (!(output.mean > 0.0) || !(output.variance > 0.0)) {
    output.status = kDcovStatusInvalidMoment;
    *result = output;
    return;
  }
  output.gamma_shape = output.mean * output.mean / output.variance;
  output.gamma_scale = output.variance / output.mean;
  if (!(output.gamma_shape > 0.0) || !(output.gamma_scale > 0.0) ||
      !isfinite(output.gamma_shape) || !isfinite(output.gamma_scale)) {
    output.status = kDcovStatusInvalidMoment;
    *result = output;
    return;
  }

  int converged = 0;
  const double scaled_statistic = output.statistic / output.gamma_scale;
  output.p_value = regularized_gamma_q(
    output.gamma_shape, fmax(0.0, scaled_statistic),
    &output.gamma_iterations, &converged);
  if (converged == 0) {
    output.status = kDcovStatusGammaNoConvergence;
    *result = output;
    return;
  }
  if (!isfinite(output.p_value) || output.p_value < 0.0 ||
      output.p_value > 1.0) {
    output.status = kDcovStatusNonfinite;
    *result = output;
    return;
  }
  output.independent = output.p_value >= alpha ? 1 : 0;
  output.status = kDcovStatusOk;
  *result = output;
}

__global__ void evaluate_pair_batch_kernel(
    const double* centered,
    const double* totals,
    const double* self_moments,
    const int* left_components,
    const int* right_components,
    const unsigned long long* logical_sequence_ids,
    int pair_count,
    std::size_t cell_count,
    int n,
    double alpha,
    int diagnostic_flags,
    DeviceCompactCiResult* results) {
  __shared__ double reduction[kVerticalBlockSize];
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const int left_component = left_components[pair];
  const int right_component = right_components[pair];
  const double* left = centered +
    static_cast<std::size_t>(left_component) * cell_count;
  const double* right = centered +
    static_cast<std::size_t>(right_component) * cell_count;
  double cross_sum = 0.0;
  for (std::size_t index = threadIdx.x;
       index < cell_count; index += blockDim.x) {
    cross_sum += left[index] * right[index];
  }
  reduction[threadIdx.x] = cross_sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) return;

  DeviceCompactCiResult output;
  output.logical_sequence_id = logical_sequence_ids[pair];
  output.diagnostic_flags = diagnostic_flags;
  const double nd = static_cast<double>(n);
  output.statistic = reduction[0] / nd;
  output.mean = (totals[left_component] / (nd * nd)) *
    (totals[right_component] / (nd * nd));
  const double variance_factor =
    2.0 * (nd - 4.0) * (nd - 5.0) /
    nd / (nd - 1.0) / (nd - 2.0) / (nd - 3.0);
  output.variance = variance_factor * self_moments[left_component] *
    self_moments[right_component] / (nd * nd);

  if (!isfinite(output.statistic) || !isfinite(output.mean) ||
      !isfinite(output.variance)) {
    output.status = kDcovStatusNonfinite;
    results[pair] = output;
    return;
  }
  if (!(output.mean > 0.0) || !(output.variance > 0.0)) {
    output.status = kDcovStatusInvalidMoment;
    results[pair] = output;
    return;
  }
  output.gamma_shape = output.mean * output.mean / output.variance;
  output.gamma_scale = output.variance / output.mean;
  if (!(output.gamma_shape > 0.0) || !(output.gamma_scale > 0.0) ||
      !isfinite(output.gamma_shape) || !isfinite(output.gamma_scale)) {
    output.status = kDcovStatusInvalidMoment;
    results[pair] = output;
    return;
  }

  int converged = 0;
  const double scaled_statistic = output.statistic / output.gamma_scale;
  output.p_value = regularized_gamma_q(
    output.gamma_shape, fmax(0.0, scaled_statistic),
    &output.gamma_iterations, &converged);
  if (converged == 0) {
    output.status = kDcovStatusGammaNoConvergence;
    results[pair] = output;
    return;
  }
  if (!isfinite(output.p_value) || output.p_value < 0.0 ||
      output.p_value > 1.0) {
    output.status = kDcovStatusNonfinite;
    results[pair] = output;
    return;
  }
  output.independent = output.p_value >= alpha ? 1 : 0;
  output.status = kDcovStatusOk;
  results[pair] = output;
}

__global__ void select_center_legacy_eigencomponent_kernel(
    const double* eigenvalues,
    const double* eigenvectors,
    int n,
    int num_col,
    int component_count,
    double* selected_values,
    double* centered_vectors) {
  extern __shared__ unsigned char selected_mask[];
  __shared__ int selected_indices[64];
  __shared__ double reduction[kVerticalBlockSize];
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count || num_col > 64) return;
  eigenvalues += static_cast<std::size_t>(component) * n;
  eigenvectors += static_cast<std::size_t>(component) *
    static_cast<std::size_t>(n) * n;
  for (int index = static_cast<int>(threadIdx.x);
       index < n; index += static_cast<int>(blockDim.x)) {
    selected_mask[index] = 0U;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int selected = 0; selected < num_col; ++selected) {
      int best_index = -1;
      double best_magnitude = -1.0;
      for (int candidate = 0; candidate < n; ++candidate) {
        if (selected_mask[candidate] != 0U) continue;
        const double magnitude = fabs(eigenvalues[candidate]);
        if (magnitude > best_magnitude ||
            (magnitude == best_magnitude && candidate > best_index)) {
          best_magnitude = magnitude;
          best_index = candidate;
        }
      }
      selected_indices[selected] = best_index;
      selected_values[
        static_cast<std::size_t>(component) * num_col + selected] =
          eigenvalues[best_index];
      selected_mask[best_index] = 1U;
    }
  }
  __syncthreads();

  double* output = centered_vectors +
    static_cast<std::size_t>(component) *
      static_cast<std::size_t>(n) * num_col;
  for (int selected = 0; selected < num_col; ++selected) {
    const int source_column = selected_indices[selected];
    const double* source = eigenvectors +
      static_cast<std::size_t>(source_column) * n;
    double sum = 0.0;
    for (int row = static_cast<int>(threadIdx.x);
         row < n; row += static_cast<int>(blockDim.x)) {
      sum += source[row];
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
      if (static_cast<int>(threadIdx.x) < stride) {
        reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      }
      __syncthreads();
    }
    const double mean = reduction[0] / static_cast<double>(n);
    for (int row = static_cast<int>(threadIdx.x);
         row < n; row += static_cast<int>(blockDim.x)) {
      output[static_cast<std::size_t>(selected) * n + row] =
        source[row] - mean;
    }
    __syncthreads();
  }
}

__global__ void reduce_legacy_self_moment_kernel(
    const double* component_grams,
    const double* selected_values,
    int component_count,
    int num_col,
    double* self_moments) {
  __shared__ double reduction[kVerticalBlockSize];
  const int component = static_cast<int>(blockIdx.x);
  if (component >= component_count) return;
  const std::size_t square =
    static_cast<std::size_t>(num_col) * num_col;
  const double* gram = component_grams +
    static_cast<std::size_t>(component) * square;
  const double* values = selected_values +
    static_cast<std::size_t>(component) * num_col;
  double sum = 0.0;
  for (std::size_t index = threadIdx.x;
       index < square; index += blockDim.x) {
    const int row = static_cast<int>(index % num_col);
    const int column = static_cast<int>(index / num_col);
    const double cross = gram[index];
    sum += values[row] * values[column] * cross * cross;
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) self_moments[component] = reduction[0];
}

__global__ void evaluate_legacy_eig_pair_batch_kernel(
    const double* pair_cross,
    const double* selected_values,
    const double* totals,
    const double* self_moments,
    const int* left_components,
    const int* right_components,
    const unsigned long long* logical_sequence_ids,
    int pair_count,
    int num_col,
    int n,
    double alpha,
    DeviceCompactCiResult* results) {
  __shared__ double reduction[kVerticalBlockSize];
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const int left_component = left_components[pair];
  const int right_component = right_components[pair];
  const double* left_values = selected_values +
    static_cast<std::size_t>(left_component) * num_col;
  const double* right_values = selected_values +
    static_cast<std::size_t>(right_component) * num_col;
  const std::size_t square =
    static_cast<std::size_t>(num_col) * num_col;
  const double* cross = pair_cross +
    static_cast<std::size_t>(pair) * square;
  double weighted_cross = 0.0;
  for (std::size_t index = threadIdx.x;
       index < square; index += blockDim.x) {
    const int row = static_cast<int>(index % num_col);
    const int column = static_cast<int>(index / num_col);
    const double value = cross[index];
    weighted_cross += left_values[row] * right_values[column] *
      value * value;
  }
  reduction[threadIdx.x] = weighted_cross;
  __syncthreads();
  for (int stride = kVerticalBlockSize / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) return;

  DeviceCompactCiResult output;
  output.logical_sequence_id = logical_sequence_ids[pair];
  output.diagnostic_flags = kDiagnosticLegacyFullEig;
  const double nd = static_cast<double>(n);
  output.statistic = reduction[0] / nd;
  output.mean = (totals[left_component] / (nd * nd)) *
    (totals[right_component] / (nd * nd));
  const double variance_factor =
    2.0 * (nd - 4.0) * (nd - 5.0) /
    nd / (nd - 1.0) / (nd - 2.0) / (nd - 3.0);
  // Preserve the qualified legacy low-rank representation scaling.
  output.variance = variance_factor * self_moments[left_component] *
    self_moments[right_component] / (nd * nd);
  if (!isfinite(output.statistic) || !isfinite(output.mean) ||
      !isfinite(output.variance)) {
    output.status = kDcovStatusNonfinite;
    results[pair] = output;
    return;
  }
  if (!(output.mean > 0.0) || !(output.variance > 0.0)) {
    output.status = kDcovStatusInvalidMoment;
    results[pair] = output;
    return;
  }
  output.gamma_shape = output.mean * output.mean / output.variance;
  output.gamma_scale = output.variance / output.mean;
  int converged = 0;
  output.p_value = regularized_gamma_q(
    output.gamma_shape, fmax(0.0, output.statistic / output.gamma_scale),
    &output.gamma_iterations, &converged);
  if (converged == 0) {
    output.status = kDcovStatusGammaNoConvergence;
    results[pair] = output;
    return;
  }
  if (!isfinite(output.p_value) || output.p_value < 0.0 ||
      output.p_value > 1.0) {
    output.status = kDcovStatusNonfinite;
    results[pair] = output;
    return;
  }
  output.independent = output.p_value >= alpha ? 1 : 0;
  output.status = kDcovStatusOk;
  results[pair] = output;
}

const char* dcov_status_name(int status) {
  switch (status) {
    case kDcovStatusOk: return "OK_EXACT_CUDA_GAMMA";
    case kDcovStatusInvalidMoment: return "ERR_INVALID_GAMMA_MOMENT";
    case kDcovStatusNonfinite: return "ERR_NONFINITE_DCOV_OUTPUT";
    case kDcovStatusGammaNoConvergence:
      return "ERR_DEVICE_GAMMA_NO_CONVERGENCE";
  }
  return "ERR_UNKNOWN_DCOV_STATUS";
}

bool equal_double_bits(double left, double right) {
  return std::memcmp(&left, &right, sizeof(double)) == 0;
}

double host_elapsed_ms(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

double cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t end) {
  float elapsed = 0.0F;
  check_cuda(cudaEventElapsedTime(&elapsed, start, end),
             "measure vertical CUDA stage");
  return static_cast<double>(elapsed);
}

bool compact_result_bit_identical(const DeviceCompactCiResult& left,
                                  const DeviceCompactCiResult& right) {
  return left.logical_sequence_id == right.logical_sequence_id &&
    equal_double_bits(left.p_value, right.p_value) &&
    equal_double_bits(left.statistic, right.statistic) &&
    equal_double_bits(left.mean, right.mean) &&
    equal_double_bits(left.variance, right.variance) &&
    equal_double_bits(left.gamma_shape, right.gamma_shape) &&
    equal_double_bits(left.gamma_scale, right.gamma_scale) &&
    left.status == right.status &&
    left.independent == right.independent &&
    left.gamma_iterations == right.gamma_iterations &&
    left.diagnostic_flags == right.diagnostic_flags;
}

FullCudaCiCompactHostRecord host_record(
    const DeviceCompactCiResult& device_result,
    const DeviceResidualConsumerView& residual_view,
    const FullCudaCiVerticalRequest& request) {
  FullCudaCiCompactHostRecord result;
  result.logical_sequence_id = device_result.logical_sequence_id;
  result.p_value = device_result.p_value;
  result.status = device_result.status == kDcovStatusOk ? "OK" : "ERROR";
  result.solver_route = std::string(fixed_sp_route_name(
    residual_view.executed_routes[
      static_cast<std::size_t>(request.left_target_index)])) + "|" +
    fixed_sp_route_name(residual_view.executed_routes[
      static_cast<std::size_t>(request.right_target_index)]);
  result.optimizer_status = "ORACLE_SP_ACCEPTED";
  result.dcov_status = dcov_status_name(device_result.status);
  result.diagnostic_flags = device_result.diagnostic_flags;
  return result;
}

FullCudaCiCompactHostRecord host_record(
    const DeviceCompactCiResult& device_result,
    const DeviceResidualConsumerView& residual_view,
    const FullCudaCiExactBatchPairRequest& request) {
  FullCudaCiCompactHostRecord result;
  result.logical_sequence_id = device_result.logical_sequence_id;
  result.p_value = device_result.p_value;
  result.status = device_result.status == kDcovStatusOk ? "OK" : "ERROR";
  result.solver_route = std::string(fixed_sp_route_name(
    residual_view.executed_routes[
      static_cast<std::size_t>(request.left_target_index)])) + "|" +
    fixed_sp_route_name(residual_view.executed_routes[
      static_cast<std::size_t>(request.right_target_index)]);
  result.optimizer_status = "ORACLE_SP_ACCEPTED";
  result.dcov_status = dcov_status_name(device_result.status);
  result.diagnostic_flags = device_result.diagnostic_flags;
  return result;
}

FullCudaCiNumericalDiagnostics numerical_diagnostics(
    const DeviceCompactCiResult& device_result) {
  FullCudaCiNumericalDiagnostics result;
  result.statistic = device_result.statistic;
  result.mean = device_result.mean;
  result.variance = device_result.variance;
  result.gamma_shape = device_result.gamma_shape;
  result.gamma_scale = device_result.gamma_scale;
  result.gamma_iterations = device_result.gamma_iterations;
  return result;
}

void require_ok(const DeviceCompactCiResult& result) {
  if (result.status != kDcovStatusOk) {
    throw std::runtime_error(
      std::string("full CUDA CI pair evaluation failed closed: ") +
      dcov_status_name(result.status));
  }
}

}  // namespace

class DcovComponentHandle {
 public:
  DcovComponentHandle(int n,
                      int target_index,
                      std::string target_key,
                      VerticalCallAccounting* accounting)
      : n_(n),
        target_index_(target_index),
        target_key_(std::move(target_key)),
        cell_count_(checked_multiply(
          static_cast<std::size_t>(n), static_cast<std::size_t>(n),
          "dCov component matrix")),
        centered_(checked_multiply(cell_count_, sizeof(double),
                                   "dCov centered component"), accounting),
        row_sums_(checked_multiply(static_cast<std::size_t>(n), sizeof(double),
                                   "dCov row sums"), accounting),
        total_(sizeof(double), accounting),
        self_moment_(sizeof(double), accounting) {}

  void build(const DeviceResidualConsumerView& residual_view,
             cudaStream_t stream) {
    build_distance_rows_kernel<<<
      static_cast<unsigned int>(n_), kVerticalBlockSize, 0, stream
    >>>(residual_view.residuals, n_, target_index_, centered(), row_sums());
    check_cuda(cudaGetLastError(), "launch vertical distance rows");
    reduce_rows_kernel<<<1, kVerticalBlockSize, 0, stream>>>(
      row_sums(), n_, total());
    check_cuda(cudaGetLastError(), "launch vertical distance total");
    const std::size_t blocks =
      (cell_count_ + kVerticalBlockSize - 1U) / kVerticalBlockSize;
    if (blocks == 0U || blocks > std::numeric_limits<unsigned int>::max()) {
      throw std::runtime_error("vertical distance center launch overflow");
    }
    center_distance_kernel<<<
      static_cast<unsigned int>(blocks), kVerticalBlockSize, 0, stream
    >>>(centered(), row_sums(), total(), n_, cell_count_);
    check_cuda(cudaGetLastError(), "launch vertical distance center");
    self_moment_kernel<<<1, kVerticalBlockSize, 0, stream>>>(
      centered(), cell_count_, self_moment());
    check_cuda(cudaGetLastError(), "launch vertical component self moment");
  }

  void close() {
    self_moment_.close();
    total_.close();
    row_sums_.close();
    centered_.close();
  }

  double* centered() const {
    return static_cast<double*>(centered_.get());
  }
  double* row_sums() const {
    return static_cast<double*>(row_sums_.get());
  }
  double* total() const { return static_cast<double*>(total_.get()); }
  double* self_moment() const {
    return static_cast<double*>(self_moment_.get());
  }
  std::size_t cell_count() const { return cell_count_; }
  std::size_t device_bytes() const {
    return centered_.bytes() + row_sums_.bytes() + total_.bytes() +
      self_moment_.bytes();
  }

 private:
  int n_ = 0;
  int target_index_ = -1;
  std::string target_key_;
  std::size_t cell_count_ = 0U;
  TrackedDeviceBuffer centered_;
  TrackedDeviceBuffer row_sums_;
  TrackedDeviceBuffer total_;
  TrackedDeviceBuffer self_moment_;
};

class LogicalCiBatchHandle {
 public:
  FullCudaCiVerticalRequest request;
  std::shared_ptr<DcovComponentHandle> left;
  std::shared_ptr<DcovComponentHandle> right;
};

class CompactCiResult {
 public:
  DeviceCompactCiResult device_record;
};

namespace {

std::shared_ptr<DcovComponentHandle> build_component(
    const DeviceResidualConsumerView& residual_view,
    int target_index,
    cudaStream_t stream,
    VerticalCallAccounting* accounting) {
  auto component = std::make_shared<DcovComponentHandle>(
    residual_view.n, target_index,
    residual_view.target_keys[static_cast<std::size_t>(target_index)],
    accounting);
  component->build(residual_view, stream);
  return component;
}

void evaluate_pair(
    const LogicalCiBatchHandle& batch,
    cudaStream_t stream,
    cudaEvent_t completion_event,
    cudaEvent_t pair_start_event,
    cudaEvent_t pair_end_event,
    TrackedDeviceBuffer* device_result,
    FullCudaCiVerticalDiagnostics* diagnostics,
    DeviceCompactCiResult* host_result) {
  if (!batch.left || !batch.right || device_result == nullptr ||
      diagnostics == nullptr || host_result == nullptr) {
    throw std::runtime_error("vertical logical CI batch is incomplete");
  }
  const int flags = kDiagnosticExactCenteredDistance |
    (batch.request.exercise_eviction ? kDiagnosticEvictionReplay : 0);
  check_cuda(cudaEventRecord(pair_start_event, stream),
             "record vertical pair start");
  evaluate_pair_kernel<<<1, kVerticalBlockSize, 0, stream>>>(
    batch.left->centered(), batch.right->centered(),
    batch.left->total(), batch.right->total(),
    batch.left->self_moment(), batch.right->self_moment(),
    batch.left->cell_count(), diagnostics->n,
    static_cast<unsigned long long>(batch.request.logical_sequence_id),
    batch.request.alpha, flags,
    static_cast<DeviceCompactCiResult*>(device_result->get()));
  check_cuda(cudaGetLastError(), "launch vertical pair evaluation");
  check_cuda(cudaEventRecord(pair_end_event, stream),
             "record vertical pair end");

  check_cuda(cudaMemcpyAsync(
    host_result, device_result->get(), sizeof(DeviceCompactCiResult),
    cudaMemcpyDeviceToHost, stream), "copy compact vertical CI result");
  diagnostics->compact_result_d2h_count += 1;
  diagnostics->compact_result_d2h_bytes += sizeof(DeviceCompactCiResult);
  diagnostics->pair_evaluation_count += 1;
  check_cuda(cudaEventRecord(completion_event, stream),
             "record vertical CI completion");
}

void close_component(std::shared_ptr<DcovComponentHandle>* component) {
  if (component == nullptr || !*component) return;
  (*component)->close();
  component->reset();
}

}  // namespace

std::string full_cuda_ci_vertical_request_identity(
    const FullCudaCiVerticalRequest& request,
    const std::vector<std::string>& target_keys) {
  if (!is_lower_sha256(request.expected_prepared_s_key_sha256)) {
    throw std::runtime_error(
      "vertical expected PreparedSKey must be a lowercase SHA-256");
  }
  if (request.logical_sequence_id == 0U ||
      request.logical_sequence_id > 9007199254740991ULL) {
    throw std::runtime_error(
      "vertical logical_sequence_id must be a positive safe-53-bit integer");
  }
  if (request.left_target_index < 0 || request.right_target_index < 0 ||
      request.left_target_index == request.right_target_index ||
      request.left_target_index >= static_cast<int>(target_keys.size()) ||
      request.right_target_index >= static_cast<int>(target_keys.size())) {
    throw std::runtime_error("vertical target ordinals are invalid");
  }
  if (request.alpha != kCanonicalAlpha) {
    throw std::runtime_error("vertical prototype requires canonical alpha 0.1");
  }
  if (!request.exercise_eviction) {
    throw std::runtime_error(
      "vertical prototype requires deterministic eviction replay");
  }
  for (const std::string& key : target_keys) {
    if (!is_lower_sha256(key)) {
      throw std::runtime_error(
        "vertical target keys must be lowercase SHA-256 strings");
    }
  }

  std::ostringstream payload;
  payload << "schema_version=" << kFullCudaCiVerticalRequestSchemaVersion
          << "\nexpected_prepared_s_key_sha256="
          << request.expected_prepared_s_key_sha256
          << "\nlogical_sequence_id=" << request.logical_sequence_id
          << "\nleft_target_key="
          << target_keys[static_cast<std::size_t>(request.left_target_index)]
          << "\nright_target_key="
          << target_keys[static_cast<std::size_t>(request.right_target_index)]
          << "\nalpha=0.1\nexercise_eviction=true";
  return full_cuda_ci_sha256_utf8(payload.str());
}

std::string full_cuda_ci_exact_batch_request_identity(
    const FullCudaCiExactBatchRequest& request,
    const std::vector<std::string>& target_keys) {
  if (!is_lower_sha256(request.expected_prepared_s_key_sha256)) {
    throw std::runtime_error(
      "exact batch expected PreparedSKey must be a lowercase SHA-256");
  }
  if (target_keys.size() < 2U || request.pairs.empty()) {
    throw std::runtime_error(
      "exact batch requires at least two targets and one pair");
  }
  if (request.component_capacity < 2 ||
      request.component_capacity > static_cast<int>(target_keys.size())) {
    throw std::runtime_error("exact batch component capacity is invalid");
  }
  for (std::size_t index = 0; index < target_keys.size(); ++index) {
    if (!is_lower_sha256(target_keys[index]) ||
        std::find(target_keys.begin(), target_keys.begin() + index,
                  target_keys[index]) != target_keys.begin() + index) {
      throw std::runtime_error(
        "exact batch target keys must be distinct lowercase SHA-256 strings");
    }
  }

  std::vector<bool> referenced(target_keys.size(), false);
  std::uint64_t previous_sequence = 0U;
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    if (pair.logical_sequence_id == 0U ||
        pair.logical_sequence_id > 9007199254740991ULL ||
        pair.logical_sequence_id <= previous_sequence) {
      throw std::runtime_error(
        "exact batch logical sequence IDs must be strictly increasing safe-53-bit integers");
    }
    if (pair.left_target_index < 0 || pair.right_target_index < 0 ||
        pair.left_target_index == pair.right_target_index ||
        pair.left_target_index >= static_cast<int>(target_keys.size()) ||
        pair.right_target_index >= static_cast<int>(target_keys.size())) {
      throw std::runtime_error("exact batch target ordinals are invalid");
    }
    if (pair.alpha != kCanonicalAlpha) {
      throw std::runtime_error("exact batch requires canonical alpha 0.1");
    }
    referenced[static_cast<std::size_t>(pair.left_target_index)] = true;
    referenced[static_cast<std::size_t>(pair.right_target_index)] = true;
    previous_sequence = pair.logical_sequence_id;
  }
  const int referenced_count = static_cast<int>(std::count(
    referenced.begin(), referenced.end(), true));
  if (referenced_count > request.component_capacity) {
    throw std::runtime_error(
      "exact batch referenced components exceed component capacity");
  }

  std::ostringstream payload;
  payload << "schema_version=" << kFullCudaCiExactBatchRequestSchemaVersion
          << "\nexpected_prepared_s_key_sha256="
          << request.expected_prepared_s_key_sha256
          << "\ncomponent_capacity=" << request.component_capacity
          << "\ntarget_count=" << target_keys.size();
  for (const std::string& key : target_keys) {
    payload << "\ntarget_key=" << key;
  }
  payload << "\npair_count=" << request.pairs.size();
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    payload << "\npair=" << pair.logical_sequence_id << "|"
            << target_keys[static_cast<std::size_t>(pair.left_target_index)]
            << "|"
            << target_keys[static_cast<std::size_t>(pair.right_target_index)]
            << "|0.1";
  }
  return full_cuda_ci_sha256_utf8(payload.str());
}

std::string full_cuda_ci_legacy_eig_batch_request_identity(
    const FullCudaCiLegacyEigBatchRequest& request,
    const std::vector<std::string>& target_keys) {
  if (!is_lower_sha256(request.expected_prepared_s_key_sha256)) {
    throw std::runtime_error(
      "legacy eig expected PreparedSKey must be a lowercase SHA-256");
  }
  if (target_keys.size() < 2U || request.pairs.empty()) {
    throw std::runtime_error(
      "legacy eig batch requires at least two targets and one pair");
  }
  if (request.component_capacity < 2 ||
      request.component_capacity > static_cast<int>(target_keys.size())) {
    throw std::runtime_error("legacy eig component capacity is invalid");
  }
  if (request.num_col != 35) {
    throw std::runtime_error("legacy eig batch requires canonical numCol 35");
  }
  for (std::size_t index = 0; index < target_keys.size(); ++index) {
    if (!is_lower_sha256(target_keys[index]) ||
        std::find(target_keys.begin(), target_keys.begin() + index,
                  target_keys[index]) != target_keys.begin() + index) {
      throw std::runtime_error(
        "legacy eig target keys must be distinct lowercase SHA-256 strings");
    }
  }
  std::vector<bool> referenced(target_keys.size(), false);
  std::uint64_t previous_sequence = 0U;
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    if (pair.logical_sequence_id == 0U ||
        pair.logical_sequence_id > 9007199254740991ULL ||
        pair.logical_sequence_id <= previous_sequence) {
      throw std::runtime_error(
        "legacy eig logical sequence IDs must be strictly increasing safe-53-bit integers");
    }
    if (pair.left_target_index < 0 || pair.right_target_index < 0 ||
        pair.left_target_index == pair.right_target_index ||
        pair.left_target_index >= static_cast<int>(target_keys.size()) ||
        pair.right_target_index >= static_cast<int>(target_keys.size())) {
      throw std::runtime_error("legacy eig target ordinals are invalid");
    }
    if (pair.alpha != kCanonicalAlpha) {
      throw std::runtime_error("legacy eig batch requires canonical alpha 0.1");
    }
    referenced[static_cast<std::size_t>(pair.left_target_index)] = true;
    referenced[static_cast<std::size_t>(pair.right_target_index)] = true;
    previous_sequence = pair.logical_sequence_id;
  }
  const int referenced_count = static_cast<int>(std::count(
    referenced.begin(), referenced.end(), true));
  if (referenced_count > request.component_capacity) {
    throw std::runtime_error(
      "legacy eig referenced components exceed component capacity");
  }

  std::ostringstream payload;
  payload << "schema_version="
          << kFullCudaCiLegacyEigBatchRequestSchemaVersion
          << "\nexpected_prepared_s_key_sha256="
          << request.expected_prepared_s_key_sha256
          << "\ncomponent_capacity=" << request.component_capacity
          << "\nnum_col=" << request.num_col
          << "\ntarget_count=" << target_keys.size();
  for (const std::string& key : target_keys) {
    payload << "\ntarget_key=" << key;
  }
  payload << "\npair_count=" << request.pairs.size();
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    payload << "\npair=" << pair.logical_sequence_id << "|"
            << target_keys[static_cast<std::size_t>(pair.left_target_index)]
            << "|"
            << target_keys[static_cast<std::size_t>(pair.right_target_index)]
            << "|0.1";
  }
  return full_cuda_ci_sha256_utf8(payload.str());
}

FullCudaCiVerticalResourceSnapshot
full_cuda_ci_vertical_resource_snapshot() {
  const VerticalResourceLedger& ledger = vertical_resource_ledger();
  FullCudaCiVerticalResourceSnapshot result;
  result.live_device_allocations =
    ledger.live_device_allocations.load(std::memory_order_relaxed);
  result.live_device_bytes =
    ledger.live_device_bytes.load(std::memory_order_relaxed);
  result.live_streams = ledger.live_streams.load(std::memory_order_relaxed);
  result.live_events = ledger.live_events.load(std::memory_order_relaxed);
  result.total_device_allocations =
    ledger.total_device_allocations.load(std::memory_order_relaxed);
  result.total_device_frees =
    ledger.total_device_frees.load(std::memory_order_relaxed);
  result.total_stream_creates =
    ledger.total_stream_creates.load(std::memory_order_relaxed);
  result.total_stream_destroys =
    ledger.total_stream_destroys.load(std::memory_order_relaxed);
  result.total_event_creates =
    ledger.total_event_creates.load(std::memory_order_relaxed);
  result.total_event_destroys =
    ledger.total_event_destroys.load(std::memory_order_relaxed);
  return result;
}

FullCudaCiVerticalResult run_full_cuda_ci_phase35_vertical(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiVerticalRequest& request) {
  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  if (prepared_info.prepared_s_key_sha256 !=
      request.expected_prepared_s_key_sha256) {
    throw std::runtime_error(
      "vertical PreparedSKey does not match the authenticated request");
  }
  if (batch.output_mask != FixedSpOutputResiduals) {
    throw std::runtime_error(
      "vertical prototype accepts residual output only");
  }
  if (batch.target_keys.size() !=
      static_cast<std::size_t>(batch.target_count)) {
    throw std::runtime_error("vertical target key count mismatch");
  }
  const std::string actual_request_identity =
    full_cuda_ci_vertical_request_identity(request, batch.target_keys);
  if (!is_lower_sha256(request.request_identity_sha256) ||
      request.request_identity_sha256 != actual_request_identity) {
    throw std::runtime_error(
      "vertical request identity SHA-256 mismatch");
  }

  const FullCudaCiVerticalResourceSnapshot resources_before =
    full_cuda_ci_vertical_resource_snapshot();
  VerticalCallAccounting accounting;
  FullCudaCiVerticalResult result;
  result.schema_version = kFullCudaCiVerticalResultSchemaVersion;
  result.request_identity_sha256 = request.request_identity_sha256;
  result.prepared_s_key_sha256 = prepared_info.prepared_s_key_sha256;
  result.target_keys = batch.target_keys;
  FullCudaCiVerticalDiagnostics& diagnostics = result.diagnostics;
  diagnostics.component_semantic_version =
    kFullCudaCiExactComponentSemanticVersion;
  diagnostics.n = batch.n;
  diagnostics.target_count = batch.target_count;
  diagnostics.live_device_allocations_before =
    resources_before.live_device_allocations;
  diagnostics.live_device_bytes_before = resources_before.live_device_bytes;
  diagnostics.request_identity_authenticated = true;
  diagnostics.prepared_identity_authenticated = true;
  diagnostics.target_identity_authenticated = true;

  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller CUDA device for vertical call");
  std::shared_ptr<DeviceResidualBatch> residual_token;
  bool consumer_registered = false;
  cudaStream_t producer_stream = nullptr;
  try {
    const std::chrono::steady_clock::time_point residual_solve_start =
      std::chrono::steady_clock::now();
    residual_token = solve_fixed_sp_batch(prepared_s, batch);
    const DeviceResidualConsumerView residual_view =
      acquire_device_residual_consumer_view(residual_token);
    producer_stream = residual_view.producer_stream;
    if (residual_view.target_keys != batch.target_keys ||
        residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        residual_view.residuals == nullptr ||
        residual_view.producer_completion_event == nullptr) {
      throw std::runtime_error(
        "vertical residual view does not match the authenticated request");
    }
    const DeviceResidualInfo residual_info =
      device_residual_info(residual_token);
    if (residual_info.implicit_residual_d2h_count != 0 ||
        residual_info.shadow_d2h_bytes != 0U ||
        residual_info.shadow_materialize_call_count != 0) {
      throw std::runtime_error(
        "vertical residual path observed a forbidden numeric D2H transfer");
    }
    diagnostics.residuals_device_resident = true;
    diagnostics.residual_solve_host_ms =
      host_elapsed_ms(residual_solve_start);

    check_cuda(cudaSetDevice(residual_view.device_id),
               "select vertical residual device");
    TrackedCudaStream consumer_stream;
    TrackedCudaEvent consumer_completion(true);
    TrackedCudaEvent component_start(true);
    TrackedCudaEvent component_end(true);
    TrackedCudaEvent pair_start(true);
    TrackedCudaEvent pair_end(true);
    check_cuda(cudaStreamWaitEvent(
      consumer_stream.get(), residual_view.producer_completion_event, 0),
      "wait for vertical residual producer");

    TrackedDeviceBuffer device_result(
      sizeof(DeviceCompactCiResult), &accounting);
    LogicalCiBatchHandle first_batch;
    first_batch.request = request;
    check_cuda(cudaEventRecord(component_start.get(), consumer_stream.get()),
               "record first component start");
    first_batch.left = build_component(
      residual_view, request.left_target_index, consumer_stream.get(),
      &accounting);
    first_batch.right = build_component(
      residual_view, request.right_target_index, consumer_stream.get(),
      &accounting);
    check_cuda(cudaEventRecord(component_end.get(), consumer_stream.get()),
               "record first component end");
    diagnostics.component_build_count += 2;
    diagnostics.component_bytes_per_target = first_batch.left->device_bytes();
    if (first_batch.right->device_bytes() !=
        diagnostics.component_bytes_per_target) {
      throw std::runtime_error("vertical component byte accounting mismatch");
    }
    diagnostics.peak_component_bytes =
      2U * diagnostics.component_bytes_per_target;
    diagnostics.components_device_resident = true;

    CompactCiResult first_compact;
    evaluate_pair(
      first_batch, consumer_stream.get(), consumer_completion.get(),
      pair_start.get(), pair_end.get(), &device_result, &diagnostics,
      &first_compact.device_record);
    check_cuda(cudaEventSynchronize(consumer_completion.get()),
               "wait for first vertical compact result");
    diagnostics.explicit_host_wait_count += 1;
    diagnostics.first_component_build_cuda_ms = cuda_elapsed_ms(
      component_start.get(), component_end.get());
    diagnostics.first_pair_evaluation_cuda_ms = cuda_elapsed_ms(
      pair_start.get(), pair_end.get());
    diagnostics.first_compact_d2h_cuda_ms = cuda_elapsed_ms(
      pair_end.get(), consumer_completion.get());
    require_ok(first_compact.device_record);

    close_component(&first_batch.left);
    close_component(&first_batch.right);
    diagnostics.component_cache_eviction_count += 2;

    LogicalCiBatchHandle replay_batch;
    replay_batch.request = request;
    check_cuda(cudaEventRecord(component_start.get(), consumer_stream.get()),
               "record replay component start");
    replay_batch.right = build_component(
      residual_view, request.right_target_index, consumer_stream.get(),
      &accounting);
    replay_batch.left = build_component(
      residual_view, request.left_target_index, consumer_stream.get(),
      &accounting);
    check_cuda(cudaEventRecord(component_end.get(), consumer_stream.get()),
               "record replay component end");
    diagnostics.component_build_count += 2;

    CompactCiResult replay_compact;
    evaluate_pair(
      replay_batch, consumer_stream.get(), consumer_completion.get(),
      pair_start.get(), pair_end.get(), &device_result, &diagnostics,
      &replay_compact.device_record);
    register_device_residual_consumer_event(
      residual_token, consumer_completion.get());
    consumer_registered = true;
    diagnostics.consumer_event_registration_count += 1;
    check_cuda(cudaEventSynchronize(consumer_completion.get()),
               "wait for replay vertical compact result");
    diagnostics.explicit_host_wait_count += 1;
    diagnostics.replay_component_build_cuda_ms = cuda_elapsed_ms(
      component_start.get(), component_end.get());
    diagnostics.replay_pair_evaluation_cuda_ms = cuda_elapsed_ms(
      pair_start.get(), pair_end.get());
    diagnostics.replay_compact_d2h_cuda_ms = cuda_elapsed_ms(
      pair_end.get(), consumer_completion.get());
    check_cuda(cudaStreamSynchronize(residual_view.producer_stream),
               "wait for vertical residual consumer proxy");
    diagnostics.explicit_host_wait_count += 1;
    require_ok(replay_compact.device_record);

    diagnostics.eviction_result_bit_identical = compact_result_bit_identical(
      first_compact.device_record, replay_compact.device_record);
    if (!diagnostics.eviction_result_bit_identical) {
      throw std::runtime_error(
        "vertical eviction reconstruction changed the compact result");
    }
    diagnostics.deterministic_replay_count = 1;
    diagnostics.deterministic_logical_replay =
      first_compact.device_record.logical_sequence_id ==
        request.logical_sequence_id &&
      replay_compact.device_record.logical_sequence_id ==
        request.logical_sequence_id;
    if (!diagnostics.deterministic_logical_replay) {
      throw std::runtime_error(
        "vertical logical replay sequence identity changed");
    }

    result.first_result = host_record(
      first_compact.device_record, residual_view, request);
    result.replay_result = host_record(
      replay_compact.device_record, residual_view, request);
    result.first_numerical = numerical_diagnostics(first_compact.device_record);
    result.replay_numerical =
      numerical_diagnostics(replay_compact.device_record);

    const std::chrono::steady_clock::time_point teardown_start =
      std::chrono::steady_clock::now();
    close_component(&replay_batch.left);
    close_component(&replay_batch.right);
    device_result.close();
    pair_end.close();
    pair_start.close();
    component_end.close();
    component_start.close();
    consumer_completion.close();
    consumer_stream.close();

    release_device_residual(residual_token);
    free_device_residual(&residual_token);
    check_cuda(cudaSetDevice(caller_device_id),
               "restore caller CUDA device after vertical call");
    diagnostics.caller_device_restored = true;
    diagnostics.teardown_host_ms = host_elapsed_ms(teardown_start);
  } catch (...) {
    if (consumer_registered && producer_stream != nullptr) {
      (void)cudaStreamSynchronize(producer_stream);
    }
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
    if (caller_device_id >= 0) {
      (void)cudaSetDevice(caller_device_id);
    }
    throw;
  }

  const FullCudaCiVerticalResourceSnapshot resources_after =
    full_cuda_ci_vertical_resource_snapshot();
  diagnostics.device_allocation_count = accounting.device_allocation_count;
  diagnostics.device_free_count = accounting.device_free_count;
  diagnostics.device_allocation_bytes = accounting.device_allocation_bytes;
  diagnostics.peak_live_device_bytes = accounting.peak_live_device_bytes;
  diagnostics.live_device_allocations_after =
    resources_after.live_device_allocations;
  diagnostics.live_device_bytes_after = resources_after.live_device_bytes;
  diagnostics.compact_result_only_d2h =
    diagnostics.residual_d2h_count == 0 &&
    diagnostics.residual_d2h_bytes == 0U &&
    diagnostics.component_d2h_count == 0 &&
    diagnostics.component_d2h_bytes == 0U &&
    diagnostics.cpu_dcov_component_count == 0 &&
    diagnostics.cpu_dcov_pair_statistic_count == 0 &&
    diagnostics.cpu_gamma_p_value_count == 0 &&
    diagnostics.compact_result_d2h_count == 2;
  diagnostics.bounded_allocation =
    accounting.peak_live_device_bytes <=
      diagnostics.peak_component_bytes + sizeof(DeviceCompactCiResult);
  diagnostics.leak_free_teardown =
    resources_after.live_device_allocations ==
      resources_before.live_device_allocations &&
    resources_after.live_device_bytes == resources_before.live_device_bytes &&
    resources_after.live_streams == resources_before.live_streams &&
    resources_after.live_events == resources_before.live_events &&
    accounting.device_allocation_count == accounting.device_free_count &&
    accounting.live_device_bytes == 0U;
  if (!diagnostics.compact_result_only_d2h ||
      !diagnostics.bounded_allocation ||
      !diagnostics.leak_free_teardown ||
      !diagnostics.caller_device_restored) {
    throw std::runtime_error(
      "vertical structural accounting gate failed closed");
  }
  diagnostics.total_host_ms = host_elapsed_ms(total_start);
  return result;
}

FullCudaCiExactBatchResult run_full_cuda_ci_phase35_exact_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiExactBatchRequest& request) {
  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  if (prepared_info.prepared_s_key_sha256 !=
      request.expected_prepared_s_key_sha256) {
    throw std::runtime_error(
      "exact batch PreparedSKey does not match the authenticated request");
  }
  if (batch.output_mask != FixedSpOutputResiduals) {
    throw std::runtime_error("exact batch accepts residual output only");
  }
  if (batch.target_keys.size() !=
      static_cast<std::size_t>(batch.target_count)) {
    throw std::runtime_error("exact batch target key count mismatch");
  }
  const std::string actual_request_identity =
    full_cuda_ci_exact_batch_request_identity(request, batch.target_keys);
  if (!is_lower_sha256(request.request_identity_sha256) ||
      request.request_identity_sha256 != actual_request_identity) {
    throw std::runtime_error("exact batch request identity SHA-256 mismatch");
  }

  std::vector<int> target_to_component(
    static_cast<std::size_t>(batch.target_count), -1);
  std::vector<int> component_targets;
  component_targets.reserve(static_cast<std::size_t>(batch.target_count));
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    target_to_component[static_cast<std::size_t>(pair.left_target_index)] = 0;
    target_to_component[static_cast<std::size_t>(pair.right_target_index)] = 0;
  }
  for (int target = 0; target < batch.target_count; ++target) {
    if (target_to_component[static_cast<std::size_t>(target)] >= 0) {
      target_to_component[static_cast<std::size_t>(target)] =
        static_cast<int>(component_targets.size());
      component_targets.push_back(target);
    }
  }
  const int component_count = static_cast<int>(component_targets.size());
  const int pair_count = static_cast<int>(request.pairs.size());
  std::vector<int> left_components(static_cast<std::size_t>(pair_count));
  std::vector<int> right_components(static_cast<std::size_t>(pair_count));
  std::vector<unsigned long long> logical_sequence_ids(
    static_cast<std::size_t>(pair_count));
  for (int pair_index = 0; pair_index < pair_count; ++pair_index) {
    const FullCudaCiExactBatchPairRequest& pair =
      request.pairs[static_cast<std::size_t>(pair_index)];
    left_components[static_cast<std::size_t>(pair_index)] =
      target_to_component[static_cast<std::size_t>(pair.left_target_index)];
    right_components[static_cast<std::size_t>(pair_index)] =
      target_to_component[static_cast<std::size_t>(pair.right_target_index)];
    logical_sequence_ids[static_cast<std::size_t>(pair_index)] =
      static_cast<unsigned long long>(pair.logical_sequence_id);
  }

  const std::size_t n_size = static_cast<std::size_t>(batch.n);
  const std::size_t component_size = static_cast<std::size_t>(component_count);
  const std::size_t pair_size = static_cast<std::size_t>(pair_count);
  const std::size_t cell_count = checked_multiply(
    n_size, n_size, "exact batch component cells");
  const std::size_t component_cell_count = checked_multiply(
    component_size, cell_count, "exact batch component pool cells");
  const std::size_t component_row_count = checked_multiply(
    component_size, n_size, "exact batch component row sums");
  const std::size_t component_matrix_bytes = checked_multiply(
    component_cell_count, sizeof(double),
    "exact batch component matrix bytes");
  const std::size_t component_row_bytes = checked_multiply(
    component_row_count, sizeof(double),
    "exact batch component row bytes");
  const std::size_t component_scalar_bytes = checked_multiply(
    component_size, sizeof(double),
    "exact batch component scalar bytes");
  const std::size_t component_target_bytes = checked_multiply(
    component_size, sizeof(int), "exact batch component target bytes");
  const std::size_t pair_index_bytes = checked_multiply(
    pair_size, sizeof(int), "exact batch pair index bytes");
  const std::size_t pair_sequence_bytes = checked_multiply(
    pair_size, sizeof(unsigned long long),
    "exact batch pair sequence bytes");
  const std::size_t result_bytes = checked_multiply(
    pair_size, sizeof(DeviceCompactCiResult),
    "exact batch compact result bytes");

  const FullCudaCiVerticalResourceSnapshot resources_before =
    full_cuda_ci_vertical_resource_snapshot();
  VerticalCallAccounting accounting;
  FullCudaCiExactBatchResult result;
  result.schema_version = kFullCudaCiExactBatchResultSchemaVersion;
  result.request_identity_sha256 = request.request_identity_sha256;
  result.prepared_s_key_sha256 = prepared_info.prepared_s_key_sha256;
  result.target_keys = batch.target_keys;
  FullCudaCiExactBatchDiagnostics& diagnostics = result.diagnostics;
  diagnostics.component_semantic_version =
    kFullCudaCiExactComponentSemanticVersion;
  diagnostics.n = batch.n;
  diagnostics.target_count = batch.target_count;
  diagnostics.pair_count = pair_count;
  diagnostics.referenced_component_count = component_count;
  diagnostics.component_capacity = request.component_capacity;
  diagnostics.component_cache_lookup_count = 2 * pair_count;
  diagnostics.component_cache_miss_count = component_count;
  diagnostics.component_cache_hit_count =
    diagnostics.component_cache_lookup_count - component_count;
  diagnostics.component_build_count = component_count;
  diagnostics.pair_evaluation_count = pair_count;
  diagnostics.live_device_allocations_before =
    resources_before.live_device_allocations;
  diagnostics.live_device_bytes_before = resources_before.live_device_bytes;
  diagnostics.request_identity_authenticated = true;
  diagnostics.prepared_identity_authenticated = true;
  diagnostics.target_identity_authenticated = true;
  diagnostics.component_bytes_per_target = checked_multiply(
    cell_count + n_size + 2U, sizeof(double),
    "exact batch component bytes per target");
  diagnostics.peak_component_bytes = checked_multiply(
    component_size, diagnostics.component_bytes_per_target,
    "exact batch peak component bytes");

  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller CUDA device for exact batch");
  std::shared_ptr<DeviceResidualBatch> residual_token;
  bool consumer_registered = false;
  cudaStream_t producer_stream = nullptr;
  try {
    const std::chrono::steady_clock::time_point residual_solve_start =
      std::chrono::steady_clock::now();
    residual_token = solve_fixed_sp_batch(prepared_s, batch);
    const DeviceResidualConsumerView residual_view =
      acquire_device_residual_consumer_view(residual_token);
    producer_stream = residual_view.producer_stream;
    if (residual_view.target_keys != batch.target_keys ||
        residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        residual_view.residuals == nullptr ||
        residual_view.producer_completion_event == nullptr) {
      throw std::runtime_error(
        "exact batch residual view does not match the authenticated request");
    }
    const DeviceResidualInfo residual_info =
      device_residual_info(residual_token);
    if (residual_info.implicit_residual_d2h_count != 0 ||
        residual_info.shadow_d2h_bytes != 0U ||
        residual_info.shadow_materialize_call_count != 0) {
      throw std::runtime_error(
        "exact batch residual path observed a forbidden numeric D2H transfer");
    }
    diagnostics.residuals_device_resident = true;
    diagnostics.residual_solve_host_ms =
      host_elapsed_ms(residual_solve_start);

    check_cuda(cudaSetDevice(residual_view.device_id),
               "select exact batch residual device");
    const std::chrono::steady_clock::time_point dcov_start =
      std::chrono::steady_clock::now();
    TrackedCudaStream consumer_stream;
    TrackedCudaEvent metadata_start(true);
    TrackedCudaEvent component_start(true);
    TrackedCudaEvent pair_start(true);
    TrackedCudaEvent pair_end(true);
    TrackedCudaEvent completion(true);
    check_cuda(cudaStreamWaitEvent(
      consumer_stream.get(), residual_view.producer_completion_event, 0),
      "wait for exact batch residual producer");

    TrackedDeviceBuffer centered(
      component_matrix_bytes, &accounting);
    TrackedDeviceBuffer row_sums(component_row_bytes, &accounting);
    TrackedDeviceBuffer totals(component_scalar_bytes, &accounting);
    TrackedDeviceBuffer self_moments(component_scalar_bytes, &accounting);
    TrackedDeviceBuffer device_component_targets(
      component_target_bytes, &accounting);
    TrackedDeviceBuffer device_left_components(pair_index_bytes, &accounting);
    TrackedDeviceBuffer device_right_components(pair_index_bytes, &accounting);
    TrackedDeviceBuffer device_logical_sequence_ids(
      pair_sequence_bytes, &accounting);
    TrackedDeviceBuffer device_results(result_bytes, &accounting);

    check_cuda(cudaEventRecord(metadata_start.get(), consumer_stream.get()),
               "record exact batch metadata start");
    check_cuda(cudaMemcpyAsync(
      device_component_targets.get(), component_targets.data(),
      component_target_bytes, cudaMemcpyHostToDevice, consumer_stream.get()),
      "copy exact batch component targets");
    check_cuda(cudaMemcpyAsync(
      device_left_components.get(), left_components.data(), pair_index_bytes,
      cudaMemcpyHostToDevice, consumer_stream.get()),
      "copy exact batch left components");
    check_cuda(cudaMemcpyAsync(
      device_right_components.get(), right_components.data(), pair_index_bytes,
      cudaMemcpyHostToDevice, consumer_stream.get()),
      "copy exact batch right components");
    check_cuda(cudaMemcpyAsync(
      device_logical_sequence_ids.get(), logical_sequence_ids.data(),
      pair_sequence_bytes, cudaMemcpyHostToDevice, consumer_stream.get()),
      "copy exact batch logical sequence IDs");
    diagnostics.metadata_h2d_count = 4;
    diagnostics.metadata_h2d_bytes = component_target_bytes +
      2U * pair_index_bytes + pair_sequence_bytes;
    check_cuda(cudaEventRecord(component_start.get(), consumer_stream.get()),
               "record exact batch component start");

    const dim3 distance_grid(
      static_cast<unsigned int>(batch.n),
      static_cast<unsigned int>(component_count), 1U);
    build_distance_rows_batch_kernel<<<
      distance_grid, kVerticalBlockSize, 0, consumer_stream.get()
    >>>(
      residual_view.residuals,
      static_cast<const int*>(device_component_targets.get()), batch.n,
      component_count, cell_count, static_cast<double*>(centered.get()),
      static_cast<double*>(row_sums.get()));
    check_cuda(cudaGetLastError(),
               "launch exact batch distance rows");
    reduce_rows_batch_kernel<<<
      static_cast<unsigned int>(component_count), kVerticalBlockSize, 0,
      consumer_stream.get()
    >>>(
      static_cast<const double*>(row_sums.get()), batch.n, component_count,
      static_cast<double*>(totals.get()));
    check_cuda(cudaGetLastError(),
               "launch exact batch distance totals");
    const std::size_t center_blocks =
      (component_cell_count + kVerticalBlockSize - 1U) /
      kVerticalBlockSize;
    if (center_blocks == 0U ||
        center_blocks > std::numeric_limits<unsigned int>::max()) {
      throw std::runtime_error("exact batch distance center launch overflow");
    }
    center_distance_batch_kernel<<<
      static_cast<unsigned int>(center_blocks), kVerticalBlockSize, 0,
      consumer_stream.get()
    >>>(
      static_cast<double*>(centered.get()),
      static_cast<const double*>(row_sums.get()),
      static_cast<const double*>(totals.get()), batch.n, cell_count,
      component_cell_count);
    check_cuda(cudaGetLastError(),
               "launch exact batch distance center");
    self_moment_batch_kernel<<<
      static_cast<unsigned int>(component_count), kVerticalBlockSize, 0,
      consumer_stream.get()
    >>>(
      static_cast<const double*>(centered.get()), component_count,
      cell_count, static_cast<double*>(self_moments.get()));
    check_cuda(cudaGetLastError(),
               "launch exact batch self moments");
    diagnostics.components_device_resident = true;
    check_cuda(cudaEventRecord(pair_start.get(), consumer_stream.get()),
               "record exact batch pair start");

    evaluate_pair_batch_kernel<<<
      static_cast<unsigned int>(pair_count), kVerticalBlockSize, 0,
      consumer_stream.get()
    >>>(
      static_cast<const double*>(centered.get()),
      static_cast<const double*>(totals.get()),
      static_cast<const double*>(self_moments.get()),
      static_cast<const int*>(device_left_components.get()),
      static_cast<const int*>(device_right_components.get()),
      static_cast<const unsigned long long*>(
        device_logical_sequence_ids.get()),
      pair_count, cell_count, batch.n, kCanonicalAlpha,
      kDiagnosticExactCenteredDistance,
      static_cast<DeviceCompactCiResult*>(device_results.get()));
    check_cuda(cudaGetLastError(),
               "launch exact batch pair evaluation");
    check_cuda(cudaEventRecord(pair_end.get(), consumer_stream.get()),
               "record exact batch pair end");

    std::vector<DeviceCompactCiResult> host_results(pair_size);
    check_cuda(cudaMemcpyAsync(
      host_results.data(), device_results.get(), result_bytes,
      cudaMemcpyDeviceToHost, consumer_stream.get()),
      "copy compact exact batch results");
    diagnostics.compact_result_d2h_count = 1;
    diagnostics.compact_result_d2h_bytes = result_bytes;
    check_cuda(cudaEventRecord(completion.get(), consumer_stream.get()),
               "record exact batch completion");
    register_device_residual_consumer_event(
      residual_token, completion.get());
    consumer_registered = true;
    diagnostics.consumer_event_registration_count = 1;
    check_cuda(cudaEventSynchronize(completion.get()),
               "wait for compact exact batch results");
    diagnostics.explicit_host_wait_count += 1;
    diagnostics.metadata_h2d_cuda_ms = cuda_elapsed_ms(
      metadata_start.get(), component_start.get());
    diagnostics.component_build_cuda_ms = cuda_elapsed_ms(
      component_start.get(), pair_start.get());
    diagnostics.pair_evaluation_cuda_ms = cuda_elapsed_ms(
      pair_start.get(), pair_end.get());
    diagnostics.compact_d2h_cuda_ms = cuda_elapsed_ms(
      pair_end.get(), completion.get());
    check_cuda(cudaStreamSynchronize(residual_view.producer_stream),
               "wait for exact batch residual consumer proxy");
    diagnostics.explicit_host_wait_count += 1;

    result.records.reserve(pair_size);
    result.numerical.reserve(pair_size);
    diagnostics.deterministic_logical_order = true;
    for (int pair_index = 0; pair_index < pair_count; ++pair_index) {
      const DeviceCompactCiResult& device_result =
        host_results[static_cast<std::size_t>(pair_index)];
      const FullCudaCiExactBatchPairRequest& pair =
        request.pairs[static_cast<std::size_t>(pair_index)];
      require_ok(device_result);
      diagnostics.deterministic_logical_order =
        diagnostics.deterministic_logical_order &&
        device_result.logical_sequence_id == pair.logical_sequence_id;
      result.records.push_back(host_record(
        device_result, residual_view, pair));
      result.numerical.push_back(numerical_diagnostics(device_result));
    }
    if (!diagnostics.deterministic_logical_order) {
      throw std::runtime_error(
        "exact batch logical result order changed");
    }

    const std::chrono::steady_clock::time_point teardown_start =
      std::chrono::steady_clock::now();
    device_results.close();
    device_logical_sequence_ids.close();
    device_right_components.close();
    device_left_components.close();
    device_component_targets.close();
    self_moments.close();
    totals.close();
    row_sums.close();
    centered.close();
    completion.close();
    pair_end.close();
    pair_start.close();
    component_start.close();
    metadata_start.close();
    consumer_stream.close();
    diagnostics.dcov_host_boundary_ms = host_elapsed_ms(dcov_start);

    release_device_residual(residual_token);
    free_device_residual(&residual_token);
    check_cuda(cudaSetDevice(caller_device_id),
               "restore caller CUDA device after exact batch");
    diagnostics.caller_device_restored = true;
    diagnostics.teardown_host_ms = host_elapsed_ms(teardown_start);
  } catch (...) {
    if (consumer_registered && producer_stream != nullptr) {
      (void)cudaStreamSynchronize(producer_stream);
    }
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
    if (caller_device_id >= 0) {
      (void)cudaSetDevice(caller_device_id);
    }
    throw;
  }

  const FullCudaCiVerticalResourceSnapshot resources_after =
    full_cuda_ci_vertical_resource_snapshot();
  diagnostics.device_allocation_count = accounting.device_allocation_count;
  diagnostics.device_free_count = accounting.device_free_count;
  diagnostics.device_allocation_bytes = accounting.device_allocation_bytes;
  diagnostics.peak_live_device_bytes = accounting.peak_live_device_bytes;
  diagnostics.live_device_allocations_after =
    resources_after.live_device_allocations;
  diagnostics.live_device_bytes_after = resources_after.live_device_bytes;
  diagnostics.compact_result_only_d2h =
    diagnostics.residual_d2h_count == 0 &&
    diagnostics.residual_d2h_bytes == 0U &&
    diagnostics.component_d2h_count == 0 &&
    diagnostics.component_d2h_bytes == 0U &&
    diagnostics.cpu_dcov_component_count == 0 &&
    diagnostics.cpu_dcov_pair_statistic_count == 0 &&
    diagnostics.cpu_gamma_p_value_count == 0 &&
    diagnostics.compact_result_d2h_count == 1 &&
    diagnostics.compact_result_d2h_bytes == result_bytes;
  diagnostics.component_capacity_respected =
    component_count <= request.component_capacity &&
    diagnostics.peak_component_bytes <= checked_multiply(
      static_cast<std::size_t>(request.component_capacity),
      diagnostics.component_bytes_per_target,
      "exact batch component capacity bytes");
  const std::size_t expected_peak_bytes =
    diagnostics.peak_component_bytes + diagnostics.metadata_h2d_bytes +
    result_bytes;
  diagnostics.bounded_allocation =
    accounting.peak_live_device_bytes == expected_peak_bytes;
  diagnostics.leak_free_teardown =
    resources_after.live_device_allocations ==
      resources_before.live_device_allocations &&
    resources_after.live_device_bytes == resources_before.live_device_bytes &&
    resources_after.live_streams == resources_before.live_streams &&
    resources_after.live_events == resources_before.live_events &&
    accounting.device_allocation_count == accounting.device_free_count &&
    accounting.live_device_bytes == 0U;
  if (!diagnostics.compact_result_only_d2h ||
      !diagnostics.deterministic_logical_order ||
      !diagnostics.component_capacity_respected ||
      !diagnostics.bounded_allocation ||
      !diagnostics.leak_free_teardown ||
      !diagnostics.caller_device_restored) {
    throw std::runtime_error(
      "exact batch structural accounting gate failed closed");
  }
  diagnostics.total_host_ms = host_elapsed_ms(total_start);
  return result;
}

FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request) {
  const std::chrono::steady_clock::time_point total_start =
    std::chrono::steady_clock::now();
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  if (prepared_info.prepared_s_key_sha256 !=
      request.expected_prepared_s_key_sha256) {
    throw std::runtime_error(
      "legacy eig PreparedSKey does not match the authenticated request");
  }
  if (batch.output_mask != FixedSpOutputResiduals) {
    throw std::runtime_error("legacy eig batch accepts residual output only");
  }
  if (batch.target_keys.size() !=
      static_cast<std::size_t>(batch.target_count)) {
    throw std::runtime_error("legacy eig batch target key count mismatch");
  }
  const std::string actual_request_identity =
    full_cuda_ci_legacy_eig_batch_request_identity(
      request, batch.target_keys);
  if (!is_lower_sha256(request.request_identity_sha256) ||
      request.request_identity_sha256 != actual_request_identity) {
    throw std::runtime_error(
      "legacy eig batch request identity SHA-256 mismatch");
  }

  std::vector<int> target_to_component(
    static_cast<std::size_t>(batch.target_count), -1);
  std::vector<int> component_targets;
  component_targets.reserve(static_cast<std::size_t>(batch.target_count));
  for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
    target_to_component[static_cast<std::size_t>(pair.left_target_index)] = 0;
    target_to_component[static_cast<std::size_t>(pair.right_target_index)] = 0;
  }
  for (int target = 0; target < batch.target_count; ++target) {
    if (target_to_component[static_cast<std::size_t>(target)] >= 0) {
      target_to_component[static_cast<std::size_t>(target)] =
        static_cast<int>(component_targets.size());
      component_targets.push_back(target);
    }
  }
  const int component_count = static_cast<int>(component_targets.size());
  const int pair_count = static_cast<int>(request.pairs.size());
  const int num_col = request.num_col;
  std::vector<int> left_components(static_cast<std::size_t>(pair_count));
  std::vector<int> right_components(static_cast<std::size_t>(pair_count));
  std::vector<unsigned long long> logical_sequence_ids(
    static_cast<std::size_t>(pair_count));
  for (int index = 0; index < pair_count; ++index) {
    const FullCudaCiExactBatchPairRequest& pair =
      request.pairs[static_cast<std::size_t>(index)];
    left_components[static_cast<std::size_t>(index)] =
      target_to_component[static_cast<std::size_t>(pair.left_target_index)];
    right_components[static_cast<std::size_t>(index)] =
      target_to_component[static_cast<std::size_t>(pair.right_target_index)];
    logical_sequence_ids[static_cast<std::size_t>(index)] =
      static_cast<unsigned long long>(pair.logical_sequence_id);
  }

  const std::size_t n_size = static_cast<std::size_t>(batch.n);
  const std::size_t component_size = static_cast<std::size_t>(component_count);
  const std::size_t pair_size = static_cast<std::size_t>(pair_count);
  const std::size_t k_size = static_cast<std::size_t>(num_col);
  const std::size_t cell_count = checked_multiply(
    n_size, n_size, "legacy eig component cells");
  const std::size_t component_cells = checked_multiply(
    component_size, cell_count, "legacy eig component pool cells");
  const std::size_t component_rows = checked_multiply(
    component_size, n_size, "legacy eig component rows");
  const std::size_t vector_cells = checked_multiply(
    checked_multiply(component_size, n_size,
                     "legacy eig component vector rows"),
    k_size, "legacy eig component vectors");
  const std::size_t component_value_count = checked_multiply(
    component_size, k_size, "legacy eig component values");
  const std::size_t component_gram_cells = checked_multiply(
    checked_multiply(component_size, k_size,
                     "legacy eig component Gram rows"),
    k_size, "legacy eig component Gram cells");
  const std::size_t pair_cross_cells = checked_multiply(
    checked_multiply(pair_size, k_size, "legacy eig pair cross rows"),
    k_size, "legacy eig pair cross cells");
  const std::size_t result_bytes = checked_multiply(
    pair_size, sizeof(DeviceCompactCiResult),
    "legacy eig compact result bytes");
  const std::size_t component_target_bytes = checked_multiply(
    component_size, sizeof(int), "legacy eig component target bytes");
  const std::size_t pair_index_bytes = checked_multiply(
    pair_size, sizeof(int), "legacy eig pair index bytes");
  const std::size_t pair_sequence_bytes = checked_multiply(
    pair_size, sizeof(unsigned long long),
    "legacy eig pair sequence bytes");
  const std::size_t pair_pointer_bytes = checked_multiply(
    pair_size, sizeof(double*), "legacy eig pair pointer bytes");

  const FullCudaCiVerticalResourceSnapshot resources_before =
    full_cuda_ci_vertical_resource_snapshot();
  VerticalCallAccounting accounting;
  FullCudaCiLegacyEigBatchResult result;
  result.schema_version = kFullCudaCiLegacyEigBatchResultSchemaVersion;
  result.request_identity_sha256 = request.request_identity_sha256;
  result.prepared_s_key_sha256 = prepared_info.prepared_s_key_sha256;
  result.target_keys = batch.target_keys;
  FullCudaCiLegacyEigBatchDiagnostics& diagnostics = result.diagnostics;
  diagnostics.component_semantic_version =
    kFullCudaCiLegacyEigComponentSemanticVersion;
  diagnostics.n = batch.n;
  diagnostics.target_count = batch.target_count;
  diagnostics.pair_count = pair_count;
  diagnostics.referenced_component_count = component_count;
  diagnostics.component_capacity = request.component_capacity;
  diagnostics.num_col = num_col;
  diagnostics.request_identity_authenticated = true;
  diagnostics.prepared_identity_authenticated = true;
  diagnostics.target_identity_authenticated = true;
  diagnostics.deterministic_logical_order = true;
  diagnostics.component_capacity_respected =
    component_count <= request.component_capacity;

  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller CUDA device for legacy eig batch");
  std::shared_ptr<DeviceResidualBatch> residual_token;
  bool consumer_registered = false;
  cudaStream_t producer_stream = nullptr;
  try {
    const std::chrono::steady_clock::time_point residual_start =
      std::chrono::steady_clock::now();
    residual_token = solve_fixed_sp_batch(prepared_s, batch);
    const DeviceResidualConsumerView residual_view =
      acquire_device_residual_consumer_view(residual_token);
    producer_stream = residual_view.producer_stream;
    if (residual_view.target_keys != batch.target_keys ||
        residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        residual_view.residuals == nullptr ||
        residual_view.producer_completion_event == nullptr) {
      throw std::runtime_error(
        "legacy eig residual view does not match the authenticated request");
    }
    const DeviceResidualInfo residual_info =
      device_residual_info(residual_token);
    if (residual_info.implicit_residual_d2h_count != 0 ||
        residual_info.shadow_d2h_bytes != 0U ||
        residual_info.shadow_materialize_call_count != 0) {
      throw std::runtime_error(
        "legacy eig residual path observed forbidden numeric D2H");
    }
    diagnostics.residuals_device_resident = true;
    diagnostics.residual_solve_host_ms = host_elapsed_ms(residual_start);
    check_cuda(cudaSetDevice(residual_view.device_id),
               "select legacy eig residual device");

    const std::chrono::steady_clock::time_point dcov_start =
      std::chrono::steady_clock::now();
    TrackedCudaStream stream;
    TrackedCudaEvent completion(true);
    TrackedCudaEvent metadata_start(true);
    TrackedCudaEvent metadata_end(true);
    TrackedCudaEvent distance_start(true);
    TrackedCudaEvent distance_end(true);
    TrackedCudaEvent eig_start(true);
    TrackedCudaEvent eig_end(true);
    TrackedCudaEvent finalize_end(true);
    TrackedCudaEvent pair_start(true);
    TrackedCudaEvent pair_end(true);
    VerticalCublasHandle blas(stream.get());
    VerticalCusolverHandle solver(stream.get());
    check_cuda(cudaStreamWaitEvent(
      stream.get(), residual_view.producer_completion_event, 0),
      "wait for legacy eig residual producer");

    TrackedDeviceBuffer component_target_buffer(
      component_target_bytes, &accounting);
    TrackedDeviceBuffer component_matrix_buffer(
      checked_multiply(component_cells, sizeof(double),
                       "legacy eig component matrix bytes"), &accounting);
    TrackedDeviceBuffer row_sum_buffer(
      checked_multiply(component_rows, sizeof(double),
                       "legacy eig row sum bytes"), &accounting);
    TrackedDeviceBuffer total_buffer(
      checked_multiply(component_size, sizeof(double),
                       "legacy eig total bytes"), &accounting);
    TrackedDeviceBuffer eigenvalue_buffer(
      checked_multiply(component_rows, sizeof(double),
                       "legacy eig eigenvalue bytes"), &accounting);
    TrackedDeviceBuffer selected_value_buffer(
      checked_multiply(component_value_count, sizeof(double),
                       "legacy eig selected value bytes"), &accounting);
    TrackedDeviceBuffer vector_buffer(
      checked_multiply(vector_cells, sizeof(double),
                       "legacy eig vector bytes"), &accounting);
    TrackedDeviceBuffer component_gram_buffer(
      checked_multiply(component_gram_cells, sizeof(double),
                       "legacy eig component Gram bytes"), &accounting);
    TrackedDeviceBuffer self_moment_buffer(
      checked_multiply(component_size, sizeof(double),
                       "legacy eig moment bytes"), &accounting);
    TrackedDeviceBuffer left_component_buffer(pair_index_bytes, &accounting);
    TrackedDeviceBuffer right_component_buffer(pair_index_bytes, &accounting);
    TrackedDeviceBuffer sequence_buffer(pair_sequence_bytes, &accounting);
    TrackedDeviceBuffer pair_cross_buffer(
      checked_multiply(pair_cross_cells, sizeof(double),
                       "legacy eig pair cross bytes"), &accounting);
    TrackedDeviceBuffer device_result(result_bytes, &accounting);
    TrackedDeviceBuffer solver_info_buffer(
      checked_multiply(component_size, sizeof(int),
                       "legacy eig solver info bytes"), &accounting);
    TrackedDeviceBuffer left_pointer_buffer(pair_pointer_bytes, &accounting);
    TrackedDeviceBuffer right_pointer_buffer(pair_pointer_bytes, &accounting);
    TrackedDeviceBuffer cross_pointer_buffer(pair_pointer_bytes, &accounting);

    std::vector<double*> left_pointers(pair_size);
    std::vector<double*> right_pointers(pair_size);
    std::vector<double*> cross_pointers(pair_size);
    double* vectors = static_cast<double*>(vector_buffer.get());
    double* pair_cross = static_cast<double*>(pair_cross_buffer.get());
    const std::size_t vector_stride = checked_multiply(
      n_size, k_size, "legacy eig vector stride");
    const std::size_t cross_stride = checked_multiply(
      k_size, k_size, "legacy eig cross stride");
    for (int index = 0; index < pair_count; ++index) {
      left_pointers[static_cast<std::size_t>(index)] = vectors +
        static_cast<std::size_t>(
          left_components[static_cast<std::size_t>(index)]) * vector_stride;
      right_pointers[static_cast<std::size_t>(index)] = vectors +
        static_cast<std::size_t>(
          right_components[static_cast<std::size_t>(index)]) * vector_stride;
      cross_pointers[static_cast<std::size_t>(index)] = pair_cross +
        static_cast<std::size_t>(index) * cross_stride;
    }

    check_cuda(cudaEventRecord(metadata_start.get(), stream.get()),
               "record legacy eig metadata start");
    check_cuda(cudaMemcpyAsync(
      component_target_buffer.get(), component_targets.data(),
      component_target_bytes, cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig component targets");
    check_cuda(cudaMemcpyAsync(
      left_component_buffer.get(), left_components.data(), pair_index_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig left components");
    check_cuda(cudaMemcpyAsync(
      right_component_buffer.get(), right_components.data(), pair_index_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig right components");
    check_cuda(cudaMemcpyAsync(
      sequence_buffer.get(), logical_sequence_ids.data(), pair_sequence_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig logical sequence IDs");
    check_cuda(cudaMemcpyAsync(
      left_pointer_buffer.get(), left_pointers.data(), pair_pointer_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig left pointers");
    check_cuda(cudaMemcpyAsync(
      right_pointer_buffer.get(), right_pointers.data(), pair_pointer_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig right pointers");
    check_cuda(cudaMemcpyAsync(
      cross_pointer_buffer.get(), cross_pointers.data(), pair_pointer_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy legacy eig output pointers");
    check_cuda(cudaEventRecord(metadata_end.get(), stream.get()),
               "record legacy eig metadata end");
    diagnostics.metadata_h2d_count = 7;
    diagnostics.metadata_h2d_bytes = component_target_bytes +
      2U * pair_index_bytes + pair_sequence_bytes +
      3U * pair_pointer_bytes;

    check_cuda(cudaEventRecord(distance_start.get(), stream.get()),
               "record legacy eig distance start");
    const dim3 distance_grid(
      static_cast<unsigned int>(batch.n),
      static_cast<unsigned int>(component_count), 1U);
    build_distance_rows_batch_kernel<<<
      distance_grid, kVerticalBlockSize, 0, stream.get()
    >>>(
      residual_view.residuals,
      static_cast<const int*>(component_target_buffer.get()), batch.n,
      component_count, cell_count,
      static_cast<double*>(component_matrix_buffer.get()),
      static_cast<double*>(row_sum_buffer.get()));
    check_cuda(cudaGetLastError(), "launch legacy eig distance rows");
    reduce_rows_batch_kernel<<<
      static_cast<unsigned int>(component_count), kVerticalBlockSize,
      0, stream.get()
    >>>(
      static_cast<const double*>(row_sum_buffer.get()), batch.n,
      component_count, static_cast<double*>(total_buffer.get()));
    check_cuda(cudaGetLastError(), "launch legacy eig distance totals");
    check_cuda(cudaEventRecord(distance_end.get(), stream.get()),
               "record legacy eig distance end");

    int eigen_lwork = 0;
    check_cusolver(cusolverDnDsyevd_bufferSize(
      solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      batch.n, static_cast<double*>(component_matrix_buffer.get()), batch.n,
      static_cast<double*>(eigenvalue_buffer.get()), &eigen_lwork),
      "query legacy eig workspace");
    if (eigen_lwork <= 0) {
      throw std::runtime_error("legacy eig workspace query failed");
    }
    TrackedDeviceBuffer eigen_work_buffer(
      checked_multiply(static_cast<std::size_t>(eigen_lwork), sizeof(double),
                       "legacy eig solver workspace bytes"), &accounting);
    diagnostics.eig_workspace_bytes = eigen_work_buffer.bytes();
    check_cuda(cudaEventRecord(eig_start.get(), stream.get()),
               "record legacy eig solve start");
    double* component_matrices =
      static_cast<double*>(component_matrix_buffer.get());
    double* all_eigenvalues =
      static_cast<double*>(eigenvalue_buffer.get());
    int* solver_info = static_cast<int*>(solver_info_buffer.get());
    for (int component = 0; component < component_count; ++component) {
      check_cusolver(cusolverDnDsyevd(
        solver.get(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        batch.n,
        component_matrices + static_cast<std::size_t>(component) * cell_count,
        batch.n,
        all_eigenvalues + static_cast<std::size_t>(component) * n_size,
        static_cast<double*>(eigen_work_buffer.get()), eigen_lwork,
        solver_info + component), "run legacy full eig component");
    }
    check_cuda(cudaEventRecord(eig_end.get(), stream.get()),
               "record legacy eig solve end");
    select_center_legacy_eigencomponent_kernel<<<
      static_cast<unsigned int>(component_count), kVerticalBlockSize,
      n_size * sizeof(unsigned char), stream.get()
    >>>(
      all_eigenvalues, component_matrices, batch.n, num_col, component_count,
      static_cast<double*>(selected_value_buffer.get()), vectors);
    check_cuda(cudaGetLastError(),
               "launch legacy eig component selection");
    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(cublasDgemmStridedBatched(
      blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
      num_col, num_col, batch.n, &one,
      vectors, batch.n, static_cast<long long>(vector_stride),
      vectors, batch.n, static_cast<long long>(vector_stride),
      &zero, static_cast<double*>(component_gram_buffer.get()), num_col,
      static_cast<long long>(cross_stride), component_count),
      "form legacy eig component Gram matrices");
    reduce_legacy_self_moment_kernel<<<
      static_cast<unsigned int>(component_count), kVerticalBlockSize,
      0, stream.get()
    >>>(
      static_cast<const double*>(component_gram_buffer.get()),
      static_cast<const double*>(selected_value_buffer.get()),
      component_count, num_col,
      static_cast<double*>(self_moment_buffer.get()));
    check_cuda(cudaGetLastError(), "launch legacy eig self moments");
    check_cuda(cudaEventRecord(finalize_end.get(), stream.get()),
               "record legacy eig finalize end");

    diagnostics.component_build_count = component_count;
    diagnostics.cuda_full_eig_count = component_count;
    diagnostics.components_device_resident = true;
    diagnostics.persistent_component_bytes =
      selected_value_buffer.bytes() + vector_buffer.bytes() +
      total_buffer.bytes() + self_moment_buffer.bytes();

    check_cuda(cudaEventRecord(pair_start.get(), stream.get()),
               "record legacy eig pair start");
    check_cublas(cublasDgemmBatched(
      blas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
      num_col, num_col, batch.n, &one,
      reinterpret_cast<const double**>(left_pointer_buffer.get()), batch.n,
      reinterpret_cast<const double**>(right_pointer_buffer.get()), batch.n,
      &zero, reinterpret_cast<double**>(cross_pointer_buffer.get()), num_col,
      pair_count), "form legacy eig pair cross matrices");
    evaluate_legacy_eig_pair_batch_kernel<<<
      static_cast<unsigned int>(pair_count), kVerticalBlockSize,
      0, stream.get()
    >>>(
      static_cast<const double*>(pair_cross_buffer.get()),
      static_cast<const double*>(selected_value_buffer.get()),
      static_cast<const double*>(total_buffer.get()),
      static_cast<const double*>(self_moment_buffer.get()),
      static_cast<const int*>(left_component_buffer.get()),
      static_cast<const int*>(right_component_buffer.get()),
      static_cast<const unsigned long long*>(sequence_buffer.get()),
      pair_count, num_col, batch.n, kCanonicalAlpha,
      static_cast<DeviceCompactCiResult*>(device_result.get()));
    check_cuda(cudaGetLastError(), "launch legacy eig pair evaluation");
    check_cuda(cudaEventRecord(pair_end.get(), stream.get()),
               "record legacy eig pair end");
    diagnostics.pair_evaluation_count = pair_count;
    diagnostics.cuda_pair_count = pair_count;
    diagnostics.cuda_gamma_count = pair_count;
    diagnostics.pair_workspace_bytes = pair_cross_buffer.bytes();

    std::vector<DeviceCompactCiResult> host_results(pair_size);
    std::vector<int> host_solver_info(component_size, -1);
    check_cuda(cudaMemcpyAsync(
      host_results.data(), device_result.get(), result_bytes,
      cudaMemcpyDeviceToHost, stream.get()),
      "copy compact legacy eig results");
    check_cuda(cudaMemcpyAsync(
      host_solver_info.data(), solver_info_buffer.get(),
      checked_multiply(component_size, sizeof(int),
                       "legacy eig solver status copy bytes"),
      cudaMemcpyDeviceToHost, stream.get()),
      "copy compact legacy eig solver status");
    diagnostics.compact_result_d2h_count = 1;
    diagnostics.compact_result_d2h_bytes = result_bytes;
    diagnostics.compact_status_d2h_count = 1;
    diagnostics.compact_status_d2h_bytes =
      checked_multiply(component_size, sizeof(int),
                       "legacy eig solver status bytes");
    check_cuda(cudaEventRecord(completion.get(), stream.get()),
               "record legacy eig completion");
    register_device_residual_consumer_event(
      residual_token, completion.get());
    consumer_registered = true;
    diagnostics.consumer_event_registration_count = 1;
    check_cuda(cudaEventSynchronize(completion.get()),
               "wait for compact legacy eig results");
    diagnostics.explicit_host_wait_count += 1;
    diagnostics.metadata_h2d_cuda_ms = cuda_elapsed_ms(
      metadata_start.get(), metadata_end.get());
    diagnostics.distance_build_cuda_ms = cuda_elapsed_ms(
      distance_start.get(), distance_end.get());
    diagnostics.full_eig_cuda_ms = cuda_elapsed_ms(
      eig_start.get(), eig_end.get());
    diagnostics.component_finalize_cuda_ms = cuda_elapsed_ms(
      eig_end.get(), finalize_end.get());
    diagnostics.component_build_cuda_ms =
      diagnostics.distance_build_cuda_ms + diagnostics.full_eig_cuda_ms +
      diagnostics.component_finalize_cuda_ms;
    diagnostics.pair_evaluation_cuda_ms = cuda_elapsed_ms(
      pair_start.get(), pair_end.get());
    diagnostics.compact_d2h_cuda_ms = cuda_elapsed_ms(
      pair_end.get(), completion.get());
    diagnostics.dcov_host_boundary_ms = host_elapsed_ms(dcov_start);
    diagnostics.solver_failure_count = static_cast<int>(std::count_if(
      host_solver_info.begin(), host_solver_info.end(),
      [](int value) { return value != 0; }));
    if (diagnostics.solver_failure_count != 0) {
      throw std::runtime_error("legacy full eig component solver failed");
    }
    for (const DeviceCompactCiResult& host_result : host_results) {
      require_ok(host_result);
    }
    check_cuda(cudaStreamSynchronize(residual_view.producer_stream),
               "wait for legacy eig residual consumer proxy");
    diagnostics.explicit_host_wait_count += 1;

    result.records.reserve(pair_size);
    result.numerical.reserve(pair_size);
    for (int index = 0; index < pair_count; ++index) {
      result.records.push_back(host_record(
        host_results[static_cast<std::size_t>(index)], residual_view,
        request.pairs[static_cast<std::size_t>(index)]));
      result.numerical.push_back(numerical_diagnostics(
        host_results[static_cast<std::size_t>(index)]));
    }

    const std::chrono::steady_clock::time_point teardown_start =
      std::chrono::steady_clock::now();
    cross_pointer_buffer.close();
    right_pointer_buffer.close();
    left_pointer_buffer.close();
    solver_info_buffer.close();
    device_result.close();
    pair_cross_buffer.close();
    sequence_buffer.close();
    right_component_buffer.close();
    left_component_buffer.close();
    self_moment_buffer.close();
    component_gram_buffer.close();
    vector_buffer.close();
    selected_value_buffer.close();
    eigen_work_buffer.close();
    eigenvalue_buffer.close();
    total_buffer.close();
    row_sum_buffer.close();
    component_matrix_buffer.close();
    component_target_buffer.close();
    solver.close();
    blas.close();
    pair_end.close();
    pair_start.close();
    finalize_end.close();
    eig_end.close();
    eig_start.close();
    distance_end.close();
    distance_start.close();
    metadata_end.close();
    metadata_start.close();
    completion.close();
    stream.close();
    release_device_residual(residual_token);
    free_device_residual(&residual_token);
    check_cuda(cudaSetDevice(caller_device_id),
               "restore caller CUDA device after legacy eig batch");
    diagnostics.caller_device_restored = true;
    diagnostics.teardown_host_ms = host_elapsed_ms(teardown_start);
  } catch (...) {
    if (consumer_registered && producer_stream != nullptr) {
      (void)cudaStreamSynchronize(producer_stream);
    }
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
    if (caller_device_id >= 0) (void)cudaSetDevice(caller_device_id);
    throw;
  }

  const FullCudaCiVerticalResourceSnapshot resources_after =
    full_cuda_ci_vertical_resource_snapshot();
  diagnostics.device_allocation_count = accounting.device_allocation_count;
  diagnostics.device_free_count = accounting.device_free_count;
  diagnostics.device_allocation_bytes = accounting.device_allocation_bytes;
  diagnostics.peak_live_device_bytes = accounting.peak_live_device_bytes;
  diagnostics.compact_result_only_d2h =
    diagnostics.residual_d2h_count == 0 &&
    diagnostics.residual_d2h_bytes == 0U &&
    diagnostics.component_d2h_count == 0 &&
    diagnostics.component_d2h_bytes == 0U &&
    diagnostics.cpu_dcov_component_count == 0 &&
    diagnostics.cpu_dcov_eigen_count == 0 &&
    diagnostics.cpu_dcov_pair_statistic_count == 0 &&
    diagnostics.cpu_gamma_p_value_count == 0 &&
    diagnostics.compact_result_d2h_count == 1 &&
    diagnostics.compact_status_d2h_count == 1;
  diagnostics.bounded_allocation =
    accounting.peak_live_device_bytes <= accounting.device_allocation_bytes;
  diagnostics.leak_free_teardown =
    resources_after.live_device_allocations ==
      resources_before.live_device_allocations &&
    resources_after.live_device_bytes == resources_before.live_device_bytes &&
    resources_after.live_streams == resources_before.live_streams &&
    resources_after.live_events == resources_before.live_events &&
    accounting.device_allocation_count == accounting.device_free_count &&
    accounting.live_device_bytes == 0U;
  if (!diagnostics.compact_result_only_d2h ||
      !diagnostics.deterministic_logical_order ||
      !diagnostics.component_capacity_respected ||
      !diagnostics.bounded_allocation ||
      !diagnostics.leak_free_teardown ||
      !diagnostics.caller_device_restored ||
      diagnostics.solver_failure_count != 0) {
    throw std::runtime_error(
      "legacy eig batch structural accounting gate failed closed");
  }
  diagnostics.total_host_ms = host_elapsed_ms(total_start);
  return result;
}

}  // namespace fastkpc
