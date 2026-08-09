#include "full_cuda_ci_vertical.hpp"
#include "mgcv_multi_penalty_gcv.hpp"

#include "../full_cuda_ci_contract.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace fastkpc {
namespace {

constexpr char kFixedSpResidualProducerSemanticIdentity[] =
  "full-cuda-ci-fixed-sp-runtime-residual-v1";
constexpr char kOptimizerResidualProducerSemanticIdentity[] =
  "full-cuda-ci-multi-penalty-optimizer-accepted-residual-v1";
constexpr char kLegacyComponentCacheResidualBypassSemanticIdentity[] =
  "full-cuda-ci-legacy-component-cache-residual-bypass-v1";

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

bool is_successful_fixed_sp_status(FixedSpStatus status) {
  return status == FixedSpStatus::OkCholeskyBatched ||
    status == FixedSpStatus::OkCholeskySingle ||
    status == FixedSpStatus::OkAugmentedQr ||
    status == FixedSpStatus::OkAugmentedSvd;
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
    const double* const* left_selected_values,
    const double* const* right_selected_values,
    const double* const* left_totals,
    const double* const* right_totals,
    const double* const* left_self_moments,
    const double* const* right_self_moments,
    const unsigned long long* logical_sequence_ids,
    int pair_count,
    int num_col,
    int n,
    double alpha,
    DeviceCompactCiResult* results) {
  __shared__ double reduction[kVerticalBlockSize];
  const int pair = static_cast<int>(blockIdx.x);
  if (pair >= pair_count) return;
  const double* left_values = left_selected_values[pair];
  const double* right_values = right_selected_values[pair];
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
  output.mean = (*left_totals[pair] / (nd * nd)) *
    (*right_totals[pair] / (nd * nd));
  const double variance_factor =
    2.0 * (nd - 4.0) * (nd - 5.0) /
    nd / (nd - 1.0) / (nd - 2.0) / (nd - 3.0);
  // Preserve the qualified legacy low-rank representation scaling.
  output.variance = variance_factor * *left_self_moments[pair] *
    *right_self_moments[pair] / (nd * nd);
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

struct DeviceResidualParityAccumulator {
  unsigned long long mismatch_count = 0ULL;
  unsigned long long max_abs_difference_bits = 0ULL;
  double squared_difference_sum = 0.0;
  double squared_reference_sum = 0.0;
};

struct DeviceFixedResidualIdentityAccumulator {
  unsigned long long residual_mismatch_count = 0ULL;
  unsigned long long centered_mismatch_count = 0ULL;
  unsigned long long row_sum_mismatch_count = 0ULL;
  unsigned long long total_mismatch_count = 0ULL;
  unsigned long long self_moment_mismatch_count = 0ULL;
  unsigned long long residual_max_abs_difference_bits = 0ULL;
  unsigned long long component_max_abs_difference_bits = 0ULL;
  double residual_squared_difference_sum = 0.0;
  double residual_squared_reference_sum = 0.0;
  double component_squared_difference_sum = 0.0;
  double component_squared_reference_sum = 0.0;
};

__device__ void accumulate_fixed_identity_value(
    double left,
    double right,
    unsigned long long* mismatch_count,
    unsigned long long* max_abs_difference_bits,
    double* squared_difference_sum,
    double* squared_reference_sum) {
  if (__double_as_longlong(left) != __double_as_longlong(right)) {
    atomicAdd(mismatch_count, 1ULL);
  }
  const double difference = fabs(left - right);
  atomicMax(
    max_abs_difference_bits,
    static_cast<unsigned long long>(__double_as_longlong(difference)));
  atomicAdd(
    squared_difference_sum, __dmul_rn(difference, difference));
  atomicAdd(squared_reference_sum, __dmul_rn(left, left));
}

__global__ void compare_fixed_residuals_by_key_kernel(
    const double* left_residuals,
    const double* right_residuals,
    const int* left_target_indices,
    const int* right_target_indices,
    int n,
    int matched_target_count,
    DeviceFixedResidualIdentityAccumulator* accumulators) {
  const int match = static_cast<int>(blockIdx.x);
  if (match >= matched_target_count) return;
  DeviceFixedResidualIdentityAccumulator* accumulator = accumulators + match;
  const double* left = left_residuals +
    static_cast<std::size_t>(left_target_indices[match]) * n;
  const double* right = right_residuals +
    static_cast<std::size_t>(right_target_indices[match]) * n;
  for (int row = static_cast<int>(threadIdx.x); row < n;
       row += static_cast<int>(blockDim.x)) {
    accumulate_fixed_identity_value(
      left[row], right[row], &accumulator->residual_mismatch_count,
      &accumulator->residual_max_abs_difference_bits,
      &accumulator->residual_squared_difference_sum,
      &accumulator->residual_squared_reference_sum);
  }
}

__global__ void compare_exact_components_by_key_kernel(
    const double* left_centered,
    const double* right_centered,
    const double* left_row_sums,
    const double* right_row_sums,
    const double* left_totals,
    const double* right_totals,
    const double* left_self_moments,
    const double* right_self_moments,
    int n,
    int matched_target_count,
    std::size_t cell_count,
    DeviceFixedResidualIdentityAccumulator* accumulators) {
  const int match = static_cast<int>(blockIdx.x);
  if (match >= matched_target_count) return;
  DeviceFixedResidualIdentityAccumulator* accumulator = accumulators + match;
  const std::size_t cell_offset = static_cast<std::size_t>(match) * cell_count;
  const std::size_t row_offset = static_cast<std::size_t>(match) * n;
  for (std::size_t cell = threadIdx.x; cell < cell_count;
       cell += blockDim.x) {
    accumulate_fixed_identity_value(
      left_centered[cell_offset + cell], right_centered[cell_offset + cell],
      &accumulator->centered_mismatch_count,
      &accumulator->component_max_abs_difference_bits,
      &accumulator->component_squared_difference_sum,
      &accumulator->component_squared_reference_sum);
  }
  for (int row = static_cast<int>(threadIdx.x); row < n;
       row += static_cast<int>(blockDim.x)) {
    accumulate_fixed_identity_value(
      left_row_sums[row_offset + row], right_row_sums[row_offset + row],
      &accumulator->row_sum_mismatch_count,
      &accumulator->component_max_abs_difference_bits,
      &accumulator->component_squared_difference_sum,
      &accumulator->component_squared_reference_sum);
  }
  if (threadIdx.x == 0) {
    accumulate_fixed_identity_value(
      left_totals[match], right_totals[match],
      &accumulator->total_mismatch_count,
      &accumulator->component_max_abs_difference_bits,
      &accumulator->component_squared_difference_sum,
      &accumulator->component_squared_reference_sum);
    accumulate_fixed_identity_value(
      left_self_moments[match], right_self_moments[match],
      &accumulator->self_moment_mismatch_count,
      &accumulator->component_max_abs_difference_bits,
      &accumulator->component_squared_difference_sum,
      &accumulator->component_squared_reference_sum);
  }
}

__global__ void compare_residuals_by_target_kernel(
    const double* optimizer_residuals,
    const double* fixed_residuals,
    int n,
    int target_count,
    DeviceResidualParityAccumulator* accumulators) {
  const int target = static_cast<int>(blockIdx.x);
  if (target >= target_count) return;
  DeviceResidualParityAccumulator* accumulator = accumulators + target;
  for (int row = static_cast<int>(threadIdx.x); row < n;
       row += static_cast<int>(blockDim.x)) {
    const std::size_t index = static_cast<std::size_t>(target) * n + row;
    const double optimizer = optimizer_residuals[index];
    const double fixed = fixed_residuals[index];
    if (__double_as_longlong(optimizer) != __double_as_longlong(fixed)) {
      atomicAdd(&accumulator->mismatch_count, 1ULL);
    }
    const double difference = fabs(optimizer - fixed);
    atomicMax(
      &accumulator->max_abs_difference_bits,
      static_cast<unsigned long long>(__double_as_longlong(difference)));
    atomicAdd(
      &accumulator->squared_difference_sum,
      __dmul_rn(difference, difference));
    atomicAdd(
      &accumulator->squared_reference_sum,
      __dmul_rn(fixed, fixed));
  }
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

class FullCudaCiLegacyEigExecutionContext {
 public:
  FullCudaCiLegacyEigExecutionContext(
      int device_id, int n, int num_col, int maximum_component_count,
      int maximum_pair_count, int eig_lane_count,
      VerticalCallAccounting* accounting)
      : device_id_(device_id),
        n_(n),
        num_col_(num_col),
        maximum_component_count_(maximum_component_count),
        maximum_pair_count_(maximum_pair_count),
        eig_lane_count_(eig_lane_count),
        accounting_(accounting) {
    if (device_id_ < 0 || n_ < 2 || num_col_ < 1 || num_col_ > 64 ||
        maximum_component_count_ < 0 || maximum_pair_count_ < 1 ||
        eig_lane_count_ < 0 ||
        (maximum_component_count_ == 0 && eig_lane_count_ != 0) ||
        (maximum_component_count_ > 0 && eig_lane_count_ < 1) ||
        eig_lane_count_ > maximum_component_count_ || accounting_ == nullptr) {
      throw std::runtime_error(
        "legacy eig execution context shape is invalid");
    }
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device),
               "query legacy eig execution context device");
    if (current_device != device_id_) {
      throw std::runtime_error(
        "legacy eig execution context device does not match caller");
    }

    stream_ = std::make_unique<TrackedCudaStream>();
    completion_ = std::make_unique<TrackedCudaEvent>(true);
    metadata_start_ = std::make_unique<TrackedCudaEvent>(true);
    metadata_end_ = std::make_unique<TrackedCudaEvent>(true);
    pair_start_ = std::make_unique<TrackedCudaEvent>(true);
    pair_end_ = std::make_unique<TrackedCudaEvent>(true);
    blas_ = std::make_unique<VerticalCublasHandle>(stream_->get());
    if (maximum_component_count_ > 0) {
      distance_start_ = std::make_unique<TrackedCudaEvent>(true);
      distance_end_ = std::make_unique<TrackedCudaEvent>(true);
      eig_start_ = std::make_unique<TrackedCudaEvent>(true);
      eig_end_ = std::make_unique<TrackedCudaEvent>(true);
      finalize_end_ = std::make_unique<TrackedCudaEvent>(true);
    }

    const std::size_t n_size = static_cast<std::size_t>(n_);
    const std::size_t component_size =
      static_cast<std::size_t>(maximum_component_count_);
    const std::size_t pair_size = static_cast<std::size_t>(maximum_pair_count_);
    const std::size_t k_size = static_cast<std::size_t>(num_col_);
    const std::size_t cells = checked_multiply(
      n_size, n_size, "legacy eig context component cells");
    const std::size_t vector_stride = checked_multiply(
      n_size, k_size, "legacy eig context vector stride");
    const std::size_t cross_stride = checked_multiply(
      k_size, k_size, "legacy eig context cross stride");
    if (maximum_component_count_ > 0) {
      component_target_buffer_ = allocate_buffer(checked_multiply(
        component_size, sizeof(int),
        "legacy eig context component target bytes"));
      component_matrix_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, cells,
                         "legacy eig context component matrix values"),
        sizeof(double), "legacy eig context component matrix bytes"));
      row_sum_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, n_size,
                         "legacy eig context row sum values"),
        sizeof(double), "legacy eig context row sum bytes"));
      total_buffer_ = allocate_buffer(checked_multiply(
        component_size, sizeof(double), "legacy eig context total bytes"));
      eigenvalue_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, n_size,
                         "legacy eig context eigenvalue values"),
        sizeof(double), "legacy eig context eigenvalue bytes"));
      selected_value_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, k_size,
                         "legacy eig context selected values"),
        sizeof(double), "legacy eig context selected value bytes"));
      vector_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, vector_stride,
                         "legacy eig context vector values"),
        sizeof(double), "legacy eig context vector bytes"));
      component_gram_buffer_ = allocate_buffer(checked_multiply(
        checked_multiply(component_size, cross_stride,
                         "legacy eig context component Gram values"),
        sizeof(double), "legacy eig context component Gram bytes"));
      self_moment_buffer_ = allocate_buffer(checked_multiply(
        component_size, sizeof(double),
        "legacy eig context self moment bytes"));
      solver_info_buffer_ = allocate_buffer(checked_multiply(
        component_size, sizeof(int), "legacy eig context solver info bytes"));
    }
    const std::size_t metadata_words = checked_multiply(
      10U, pair_size, "legacy eig context pair metadata words");
    pair_metadata_buffer_ = allocate_buffer(checked_multiply(
      metadata_words, sizeof(std::uintptr_t),
      "legacy eig context pair metadata bytes"));
    pair_cross_buffer_ = allocate_buffer(checked_multiply(
      checked_multiply(pair_size, cross_stride,
                       "legacy eig context pair cross values"),
      sizeof(double), "legacy eig context pair cross bytes"));
    device_result_buffer_ = allocate_buffer(checked_multiply(
      pair_size, sizeof(DeviceCompactCiResult),
      "legacy eig context result bytes"));

    if (maximum_component_count_ > 0) {
      auxiliary_streams_.reserve(
        static_cast<std::size_t>(eig_lane_count_ - 1));
      auxiliary_completion_events_.reserve(
        static_cast<std::size_t>(eig_lane_count_ - 1));
      for (int lane = 1; lane < eig_lane_count_; ++lane) {
        auxiliary_streams_.push_back(
          std::make_unique<TrackedCudaStream>());
        auxiliary_completion_events_.push_back(
          std::make_unique<TrackedCudaEvent>(false));
      }
      solvers_.reserve(static_cast<std::size_t>(eig_lane_count_));
      for (int lane = 0; lane < eig_lane_count_; ++lane) {
        solvers_.push_back(std::make_unique<VerticalCusolverHandle>(
          lane_stream(lane)));
      }
      check_cusolver(cusolverDnDsyevd_bufferSize(
        solvers_.front()->get(), CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, n_, component_matrices(), n_,
        eigenvalues(), &eigen_lwork_),
        "query persistent legacy eig workspace");
      if (eigen_lwork_ <= 0) {
        throw std::runtime_error(
          "persistent legacy eig workspace query failed");
      }
      eigen_work_buffers_.reserve(static_cast<std::size_t>(eig_lane_count_));
      for (int lane = 0; lane < eig_lane_count_; ++lane) {
        eigen_work_buffers_.push_back(allocate_buffer(checked_multiply(
          static_cast<std::size_t>(eigen_lwork_), sizeof(double),
          "persistent legacy eig workspace bytes")));
      }
    }
  }

  FullCudaCiLegacyEigExecutionContext(
      const FullCudaCiLegacyEigExecutionContext&) = delete;
  FullCudaCiLegacyEigExecutionContext& operator=(
      const FullCudaCiLegacyEigExecutionContext&) = delete;

  cudaStream_t stream() const { return stream_->get(); }
  cudaEvent_t completion() const { return completion_->get(); }
  cudaEvent_t metadata_start() const { return metadata_start_->get(); }
  cudaEvent_t metadata_end() const { return metadata_end_->get(); }
  cudaEvent_t pair_start() const { return pair_start_->get(); }
  cudaEvent_t pair_end() const { return pair_end_->get(); }
  cudaEvent_t distance_start() const { return distance_start_->get(); }
  cudaEvent_t distance_end() const { return distance_end_->get(); }
  cudaEvent_t eig_start() const { return eig_start_->get(); }
  cudaEvent_t eig_end() const { return eig_end_->get(); }
  cudaEvent_t finalize_end() const { return finalize_end_->get(); }
  cublasHandle_t blas() const { return blas_->get(); }
  void* component_targets() const { return component_target_buffer_->get(); }
  double* component_matrices() const {
    return static_cast<double*>(component_matrix_buffer_->get());
  }
  double* row_sums() const {
    return static_cast<double*>(row_sum_buffer_->get());
  }
  double* totals() const {
    return static_cast<double*>(total_buffer_->get());
  }
  double* eigenvalues() const {
    return static_cast<double*>(eigenvalue_buffer_->get());
  }
  double* selected_values() const {
    return static_cast<double*>(selected_value_buffer_->get());
  }
  double* vectors() const {
    return static_cast<double*>(vector_buffer_->get());
  }
  double* component_grams() const {
    return static_cast<double*>(component_gram_buffer_->get());
  }
  double* self_moments() const {
    return static_cast<double*>(self_moment_buffer_->get());
  }
  int* solver_info() const {
    return static_cast<int*>(solver_info_buffer_->get());
  }
  void* pair_metadata() const { return pair_metadata_buffer_->get(); }
  double* pair_cross() const {
    return static_cast<double*>(pair_cross_buffer_->get());
  }
  void* device_results() const { return device_result_buffer_->get(); }
  std::size_t device_bytes() const { return device_bytes_; }
  std::size_t eig_workspace_bytes() const {
    return checked_multiply(
      checked_multiply(static_cast<std::size_t>(eig_lane_count_),
                       static_cast<std::size_t>(eigen_lwork_),
                       "legacy eig context total workspace values"),
      sizeof(double), "legacy eig context total workspace bytes");
  }
  int eig_lane_count() const { return eig_lane_count_; }

  void require_capacity(int component_count, int pair_count) const {
    if (component_count < 0 || component_count > maximum_component_count_ ||
        pair_count < 1 || pair_count > maximum_pair_count_) {
      throw std::runtime_error(
        "legacy eig execution context capacity is insufficient");
    }
  }

  int submit_eigensolves(int component_count) {
    if (component_count < 1 || component_count > maximum_component_count_ ||
        eig_lane_count_ < 1) {
      throw std::runtime_error(
        "legacy eig context submission count is invalid");
    }
    const int active_lane_count = std::min(component_count, eig_lane_count_);
    check_cuda(cudaEventRecord(eig_start(), stream()),
               "record persistent legacy eig solve start");
    for (int lane = 1; lane < active_lane_count; ++lane) {
      check_cuda(cudaStreamWaitEvent(lane_stream(lane), eig_start(), 0),
                 "gate persistent legacy eig lane");
    }
    const std::size_t cells = checked_multiply(
      static_cast<std::size_t>(n_), static_cast<std::size_t>(n_),
      "legacy eig context submission cells");
    for (int component = 0; component < component_count; ++component) {
      const int lane = component % active_lane_count;
      check_cusolver(cusolverDnDsyevd(
        solvers_[static_cast<std::size_t>(lane)]->get(),
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        n_, component_matrices() + static_cast<std::size_t>(component) * cells,
        n_, eigenvalues() + static_cast<std::size_t>(component) * n_,
        static_cast<double*>(
          eigen_work_buffers_[static_cast<std::size_t>(lane)]->get()),
        eigen_lwork_, solver_info() + component),
        "run persistent legacy full eig component");
    }
    for (int lane = 1; lane < active_lane_count; ++lane) {
      cudaEvent_t lane_completion =
        auxiliary_completion_events_[static_cast<std::size_t>(lane - 1)]->get();
      check_cuda(cudaEventRecord(lane_completion, lane_stream(lane)),
                 "record persistent legacy eig lane completion");
      check_cuda(cudaStreamWaitEvent(stream(), lane_completion, 0),
                 "join persistent legacy eig lane");
    }
    check_cuda(cudaEventRecord(eig_end(), stream()),
               "record persistent legacy eig solve end");
    return active_lane_count;
  }

 private:
  std::unique_ptr<TrackedDeviceBuffer> allocate_buffer(std::size_t bytes) {
    auto buffer = std::make_unique<TrackedDeviceBuffer>(bytes, accounting_);
    device_bytes_ += bytes;
    return buffer;
  }

  cudaStream_t lane_stream(int lane) const {
    if (lane == 0) return stream_->get();
    if (lane < 0 || lane >= eig_lane_count_) {
      throw std::runtime_error("legacy eig lane index is invalid");
    }
    return auxiliary_streams_[static_cast<std::size_t>(lane - 1)]->get();
  }

  int device_id_ = -1;
  int n_ = 0;
  int num_col_ = 0;
  int maximum_component_count_ = 0;
  int maximum_pair_count_ = 0;
  int eig_lane_count_ = 0;
  int eigen_lwork_ = 0;
  VerticalCallAccounting* accounting_ = nullptr;
  std::size_t device_bytes_ = 0U;
  std::unique_ptr<TrackedCudaStream> stream_;
  std::unique_ptr<TrackedCudaEvent> completion_;
  std::unique_ptr<TrackedCudaEvent> metadata_start_;
  std::unique_ptr<TrackedCudaEvent> metadata_end_;
  std::unique_ptr<TrackedCudaEvent> pair_start_;
  std::unique_ptr<TrackedCudaEvent> pair_end_;
  std::unique_ptr<TrackedCudaEvent> distance_start_;
  std::unique_ptr<TrackedCudaEvent> distance_end_;
  std::unique_ptr<TrackedCudaEvent> eig_start_;
  std::unique_ptr<TrackedCudaEvent> eig_end_;
  std::unique_ptr<TrackedCudaEvent> finalize_end_;
  std::unique_ptr<VerticalCublasHandle> blas_;
  std::vector<std::unique_ptr<TrackedCudaStream>> auxiliary_streams_;
  std::vector<std::unique_ptr<TrackedCudaEvent>>
    auxiliary_completion_events_;
  std::vector<std::unique_ptr<VerticalCusolverHandle>> solvers_;
  std::vector<std::unique_ptr<TrackedDeviceBuffer>> eigen_work_buffers_;
  std::unique_ptr<TrackedDeviceBuffer> component_target_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> component_matrix_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> row_sum_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> total_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> eigenvalue_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> selected_value_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> vector_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> component_gram_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> self_moment_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> solver_info_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> pair_metadata_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> pair_cross_buffer_;
  std::unique_ptr<TrackedDeviceBuffer> device_result_buffer_;
};

class FullCudaCiLegacyEigComponentCache {
 public:
  FullCudaCiLegacyEigComponentCache(
      int device_id, int n, int num_col, std::size_t capacity_entries,
      bool residual_bypass_enabled, int maximum_component_count,
      int maximum_pair_count, int eig_lane_count)
      : device_id_(device_id),
        n_(n),
        num_col_(num_col),
        vector_value_count_(checked_multiply(
          static_cast<std::size_t>(n), static_cast<std::size_t>(num_col),
          "legacy eig cache vector values")),
        entry_value_count_(vector_value_count_ +
          static_cast<std::size_t>(num_col) + 2U),
        capacity_entries_(capacity_entries),
        residual_bypass_enabled_(residual_bypass_enabled),
        storage_(checked_multiply(
          checked_multiply(capacity_entries_, entry_value_count_,
                           "legacy eig cache values"),
          sizeof(double), "legacy eig cache bytes"), &accounting_) {
    if (device_id_ < 0 || n_ < 2 || num_col_ < 1 ||
        capacity_entries_ < 2U || maximum_component_count < 0 ||
        maximum_pair_count < 0 || eig_lane_count < 0 ||
        (eig_lane_count > 0 &&
          (maximum_component_count < eig_lane_count ||
           maximum_pair_count < 1))) {
      throw std::runtime_error("legacy eig component cache shape is invalid");
    }
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device),
               "query legacy eig component cache device");
    if (current_device != device_id_) {
      throw std::runtime_error(
        "legacy eig component cache device does not match caller");
    }
    keys_.reserve(capacity_entries_);
    if (eig_lane_count > 0) {
      execution_context_ =
        std::make_unique<FullCudaCiLegacyEigExecutionContext>(
          device_id_, n_, num_col_, maximum_component_count,
          maximum_pair_count, eig_lane_count, &context_accounting_);
    }
  }

  FullCudaCiLegacyEigComponentCache(
      const FullCudaCiLegacyEigComponentCache&) = delete;
  FullCudaCiLegacyEigComponentCache& operator=(
      const FullCudaCiLegacyEigComponentCache&) = delete;

  bool lookup(const std::string& key, std::size_t* slot,
              FixedSpRoute* route, FixedSpStatus* status) {
    if (slot == nullptr || route == nullptr || status == nullptr ||
        !is_lower_sha256(key)) {
      throw std::runtime_error("legacy eig component cache lookup is invalid");
    }
    lookup_count_ += 1U;
    const auto found = keys_.find(key);
    if (found == keys_.end()) {
      miss_count_ += 1U;
      return false;
    }
    hit_count_ += 1U;
    *slot = found->second.slot;
    *route = found->second.route;
    *status = found->second.status;
    return true;
  }

  bool reserve_slot(std::size_t* slot) {
    if (slot == nullptr) {
      throw std::runtime_error(
        "legacy eig component cache reservation is invalid");
    }
    if (next_slot_ >= capacity_entries_) {
      bypass_count_ += 1U;
      return false;
    }
    *slot = next_slot_++;
    return true;
  }

  void commit(const std::string& key, std::size_t slot,
              FixedSpRoute route, FixedSpStatus status) {
    if (!is_lower_sha256(key) || slot >= next_slot_ ||
        route == FixedSpRoute::Unset ||
        !is_successful_fixed_sp_status(status)) {
      throw std::runtime_error("legacy eig component cache commit is invalid");
    }
    const auto inserted = keys_.emplace(
      key, EntryMetadata{slot, route, status});
    if (!inserted.second || inserted.first->second.slot != slot) {
      throw std::runtime_error(
        "legacy eig component cache key was committed twice");
    }
    insert_count_ += 1U;
  }

  double* vectors(std::size_t slot) const {
    return entry(slot);
  }
  double* selected_values(std::size_t slot) const {
    return entry(slot) + vector_value_count_;
  }
  double* total(std::size_t slot) const {
    return selected_values(slot) + num_col_;
  }
  double* self_moment(std::size_t slot) const {
    return total(slot) + 1;
  }
  bool residual_bypass_enabled() const noexcept {
    return residual_bypass_enabled_;
  }
  bool contains(const std::string& key) const {
    if (!is_lower_sha256(key)) {
      throw std::runtime_error(
        "legacy eig component cache contains key is invalid");
    }
    return keys_.find(key) != keys_.end();
  }
  FullCudaCiLegacyEigExecutionContext* execution_context() const noexcept {
    return execution_context_.get();
  }

  FullCudaCiLegacyEigComponentCacheInfo info() const {
    FullCudaCiLegacyEigComponentCacheInfo value;
    value.device_id = device_id_;
    value.n = n_;
    value.num_col = num_col_;
    value.capacity_entries = capacity_entries_;
    value.entries = keys_.size();
    value.bytes_per_entry = entry_value_count_ * sizeof(double);
    value.device_bytes = storage_.bytes();
    value.lookup_count = lookup_count_;
    value.hit_count = hit_count_;
    value.miss_count = miss_count_;
    value.insert_count = insert_count_;
    value.bypass_count = bypass_count_;
    value.eig_lane_count = execution_context_ ?
      execution_context_->eig_lane_count() : 0;
    value.execution_context_device_bytes = execution_context_ ?
      execution_context_->device_bytes() : 0U;
    value.residual_bypass_enabled = residual_bypass_enabled_;
    return value;
  }

 private:
  struct EntryMetadata {
    std::size_t slot = 0U;
    FixedSpRoute route = FixedSpRoute::Unset;
    FixedSpStatus status = FixedSpStatus::ErrInternalCuda;
  };

  double* entry(std::size_t slot) const {
    if (slot >= capacity_entries_) {
      throw std::runtime_error("legacy eig component cache slot is invalid");
    }
    return static_cast<double*>(storage_.get()) + slot * entry_value_count_;
  }

  int device_id_ = -1;
  int n_ = 0;
  int num_col_ = 0;
  std::size_t vector_value_count_ = 0U;
  std::size_t entry_value_count_ = 0U;
  std::size_t capacity_entries_ = 0U;
  bool residual_bypass_enabled_ = false;
  std::size_t next_slot_ = 0U;
  VerticalCallAccounting accounting_;
  VerticalCallAccounting context_accounting_;
  TrackedDeviceBuffer storage_;
  std::unique_ptr<FullCudaCiLegacyEigExecutionContext> execution_context_;
  std::unordered_map<std::string, EntryMetadata> keys_;
  std::uint64_t lookup_count_ = 0U;
  std::uint64_t hit_count_ = 0U;
  std::uint64_t miss_count_ = 0U;
  std::uint64_t insert_count_ = 0U;
  std::uint64_t bypass_count_ = 0U;
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

std::shared_ptr<FullCudaCiLegacyEigComponentCache>
create_full_cuda_ci_legacy_eig_component_cache(
    int n, int num_col, std::size_t budget_bytes,
    bool residual_bypass_enabled, int maximum_component_count,
    int maximum_pair_count, int eig_lane_count) {
  if (n < 2 || num_col < 1 || num_col > 64 || budget_bytes == 0U ||
      maximum_component_count < 0 || maximum_pair_count < 0 ||
      eig_lane_count < 0 ||
      (eig_lane_count > 0 &&
        (maximum_component_count < eig_lane_count ||
         maximum_pair_count < 1))) {
    throw std::runtime_error(
      "legacy eig component cache creation parameters are invalid");
  }
  const std::size_t vector_values = checked_multiply(
    static_cast<std::size_t>(n), static_cast<std::size_t>(num_col),
    "legacy eig cache vector values");
  const std::size_t entry_bytes = checked_multiply(
    vector_values + static_cast<std::size_t>(num_col) + 2U,
    sizeof(double), "legacy eig cache entry bytes");
  const std::size_t capacity = budget_bytes / entry_bytes;
  if (capacity < 2U) {
    throw std::runtime_error(
      "legacy eig component cache budget holds fewer than two entries");
  }
  int device_id = -1;
  check_cuda(cudaGetDevice(&device_id),
             "capture legacy eig component cache device");
  return std::make_shared<FullCudaCiLegacyEigComponentCache>(
    device_id, n, num_col, capacity, residual_bypass_enabled,
    maximum_component_count, maximum_pair_count, eig_lane_count);
}

FullCudaCiLegacyEigComponentCacheInfo
full_cuda_ci_legacy_eig_component_cache_info(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  if (!cache) {
    throw std::runtime_error("legacy eig component cache is null");
  }
  return cache->info();
}

std::vector<unsigned char>
full_cuda_ci_legacy_eig_component_cache_contains(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache,
    const std::vector<std::string>& keys) {
  if (!cache) {
    throw std::runtime_error("legacy eig component cache is null");
  }
  std::unordered_set<std::string> unique;
  unique.reserve(keys.size());
  std::vector<unsigned char> present;
  present.reserve(keys.size());
  for (const std::string& key : keys) {
    if (!unique.insert(key).second) {
      throw std::runtime_error(
        "legacy eig component cache contains keys are not unique");
    }
    present.push_back(cache->contains(key) ? 1U : 0U);
  }
  return present;
}

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

FullCudaCiExactBatchResult run_full_cuda_ci_phase35_exact_batch_impl(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiExactBatchRequest& request,
    const MultiPenaltyGcvCudaResidualConsumerView*
      optimizer_residual_view) {
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
  diagnostics.residual_producer_semantic_identity =
    optimizer_residual_view == nullptr ?
      kFixedSpResidualProducerSemanticIdentity :
      kOptimizerResidualProducerSemanticIdentity;
  diagnostics.residual_solve_bypassed = optimizer_residual_view != nullptr;
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
  DeviceResidualConsumerView residual_view;
  bool consumer_registered = false;
  cudaStream_t producer_stream = nullptr;
  try {
    const std::chrono::steady_clock::time_point residual_solve_start =
      std::chrono::steady_clock::now();
    if (optimizer_residual_view == nullptr) {
      residual_token = solve_fixed_sp_batch(prepared_s, batch);
      residual_view = acquire_device_residual_consumer_view(residual_token);
    } else {
      residual_view.residuals = optimizer_residual_view->residuals;
      residual_view.n = optimizer_residual_view->n;
      residual_view.target_count = optimizer_residual_view->target_count;
      residual_view.device_id = optimizer_residual_view->device_id;
      residual_view.producer_stream =
        optimizer_residual_view->producer_stream;
      residual_view.producer_completion_event =
        optimizer_residual_view->producer_completion_event;
      residual_view.target_keys = optimizer_residual_view->target_keys;
      residual_view.executed_routes.assign(
        static_cast<std::size_t>(residual_view.target_count),
        FixedSpRoute::Unset);
      residual_view.solver_statuses.assign(
        static_cast<std::size_t>(residual_view.target_count),
        FixedSpStatus::OkAugmentedQr);
    }
    producer_stream = residual_view.producer_stream;
    if (residual_view.target_keys != batch.target_keys ||
        residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        residual_view.residuals == nullptr ||
        residual_view.producer_completion_event == nullptr) {
      throw std::runtime_error(
        "exact batch residual view does not match the authenticated request");
    }
    if (residual_token) {
      const DeviceResidualInfo residual_info =
        device_residual_info(residual_token);
      if (residual_info.implicit_residual_d2h_count != 0 ||
          residual_info.shadow_d2h_bytes != 0U ||
          residual_info.shadow_materialize_call_count != 0) {
        throw std::runtime_error(
          "exact batch residual path observed a forbidden numeric D2H transfer");
      }
    }
    diagnostics.residuals_device_resident = true;
    diagnostics.residual_solve_host_ms = residual_token ?
      host_elapsed_ms(residual_solve_start) : 0.0;

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
    if (residual_token) {
      register_device_residual_consumer_event(
        residual_token, completion.get());
      consumer_registered = true;
      diagnostics.consumer_event_registration_count = 1;
    }
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

    if (residual_token) {
      release_device_residual(residual_token);
      free_device_residual(&residual_token);
    }
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

FullCudaCiExactBatchResult run_full_cuda_ci_phase35_exact_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiExactBatchRequest& request) {
  return run_full_cuda_ci_phase35_exact_batch_impl(
    prepared_s, batch, request, nullptr);
}

FullCudaCiExactBatchResult
run_full_cuda_ci_phase35_exact_batch_from_optimizer_residual(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual,
    const FullCudaCiExactBatchRequest& request) {
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  const MultiPenaltyGcvCudaResidualConsumerView residual_view =
    acquire_multi_penalty_gcv_cuda_residual_consumer_view(residual);
  FixedSpBatchHostView batch;
  batch.n = residual_view.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = residual_view.target_count;
  batch.output_mask = FixedSpOutputResiduals;
  batch.target_keys = residual_view.target_keys;
  FullCudaCiExactBatchResult result =
    run_full_cuda_ci_phase35_exact_batch_impl(
      prepared_s, batch, request, &residual_view);
  for (FullCudaCiCompactHostRecord& record : result.records) {
    record.solver_route = "OPTIMIZER_ACCEPTED|OPTIMIZER_ACCEPTED";
    record.optimizer_status = "LIVE_CUDA_GCV_ACCEPTED";
  }
  return result;
}

FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request,
    const MultiPenaltyGcvCudaResidualConsumerView*
      optimizer_residual_view,
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>&
      component_cache) {
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
  FullCudaCiLegacyEigComponentCacheInfo cache_info_before;
  if (component_cache) {
    cache_info_before = component_cache->info();
    if (cache_info_before.n != batch.n ||
        cache_info_before.num_col != num_col) {
      throw std::runtime_error(
        "legacy eig component cache shape does not match request");
    }
  }
  std::vector<std::size_t> component_cache_slots(
    static_cast<std::size_t>(component_count),
    std::numeric_limits<std::size_t>::max());
  std::vector<unsigned char> component_cache_hits(
    static_cast<std::size_t>(component_count), 0U);
  std::vector<unsigned char> component_cache_reservations(
    static_cast<std::size_t>(component_count), 0U);
  std::vector<FixedSpRoute> component_cache_routes(
    static_cast<std::size_t>(component_count), FixedSpRoute::Unset);
  std::vector<FixedSpStatus> component_cache_statuses(
    static_cast<std::size_t>(component_count),
    FixedSpStatus::ErrInternalCuda);
  std::vector<int> component_to_miss(
    static_cast<std::size_t>(component_count), -1);
  std::vector<int> miss_targets;
  std::vector<std::string> miss_keys;
  miss_targets.reserve(static_cast<std::size_t>(component_count));
  miss_keys.reserve(static_cast<std::size_t>(component_count));
  for (int component = 0; component < component_count; ++component) {
    const int target = component_targets[static_cast<std::size_t>(component)];
    const std::string& key =
      batch.target_keys[static_cast<std::size_t>(target)];
    std::size_t slot = std::numeric_limits<std::size_t>::max();
    const bool hit = component_cache && component_cache->lookup(
      key, &slot,
      &component_cache_routes[static_cast<std::size_t>(component)],
      &component_cache_statuses[static_cast<std::size_t>(component)]);
    component_cache_slots[static_cast<std::size_t>(component)] = slot;
    component_cache_hits[static_cast<std::size_t>(component)] = hit ? 1U : 0U;
    if (!hit) {
      component_to_miss[static_cast<std::size_t>(component)] =
        static_cast<int>(miss_targets.size());
      miss_targets.push_back(target);
      miss_keys.push_back(key);
      if (component_cache && component_cache->reserve_slot(&slot)) {
        component_cache_slots[static_cast<std::size_t>(component)] = slot;
        component_cache_reservations[static_cast<std::size_t>(component)] = 1U;
      }
    }
  }
  const int miss_count = static_cast<int>(miss_targets.size());
  const bool residual_cache_bypass = component_cache && miss_count == 0 &&
    component_cache->residual_bypass_enabled();
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
  const std::size_t miss_size = static_cast<std::size_t>(miss_count);
  const std::size_t pair_size = static_cast<std::size_t>(pair_count);
  const std::size_t k_size = static_cast<std::size_t>(num_col);
  const std::size_t cell_count = checked_multiply(
    n_size, n_size, "legacy eig component cells");
  const std::size_t component_cells = checked_multiply(
    miss_size, cell_count, "legacy eig component pool cells");
  const std::size_t component_rows = checked_multiply(
    miss_size, n_size, "legacy eig component rows");
  const std::size_t vector_cells = checked_multiply(
    checked_multiply(miss_size, n_size,
                     "legacy eig component vector rows"),
    k_size, "legacy eig component vectors");
  const std::size_t component_value_count = checked_multiply(
    miss_size, k_size, "legacy eig component values");
  const std::size_t component_gram_cells = checked_multiply(
    checked_multiply(miss_size, k_size,
                     "legacy eig component Gram rows"),
    k_size, "legacy eig component Gram cells");
  const std::size_t pair_cross_cells = checked_multiply(
    checked_multiply(pair_size, k_size, "legacy eig pair cross rows"),
    k_size, "legacy eig pair cross cells");
  const std::size_t result_bytes = checked_multiply(
    pair_size, sizeof(DeviceCompactCiResult),
    "legacy eig compact result bytes");
  const std::size_t component_target_copy_bytes = checked_multiply(
    miss_size, sizeof(int), "legacy eig component target copy bytes");
  static_assert(sizeof(std::uintptr_t) == sizeof(unsigned long long),
                "legacy eig metadata requires 64-bit device pointers");
  const std::size_t pair_metadata_word_count = checked_multiply(
    10U, pair_size, "legacy eig pair metadata words");
  const std::size_t pair_metadata_bytes = checked_multiply(
    pair_metadata_word_count, sizeof(std::uintptr_t),
    "legacy eig pair metadata bytes");

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
  diagnostics.component_cache_persistent_request_count =
    component_cache ? component_count : 0;
  diagnostics.component_cache_persistent_hit_count =
    component_count - miss_count;
  diagnostics.component_cache_persistent_miss_count =
    component_cache ? miss_count : 0;
  diagnostics.component_cache_persistent_capacity_entries =
    cache_info_before.capacity_entries;
  diagnostics.component_cache_persistent_entries_before =
    cache_info_before.entries;
  diagnostics.component_cache_persistent_device_bytes =
    cache_info_before.device_bytes;
  diagnostics.request_identity_authenticated = true;
  diagnostics.prepared_identity_authenticated = true;
  diagnostics.target_identity_authenticated = true;
  diagnostics.deterministic_logical_order = true;
  diagnostics.component_capacity_respected =
    component_count <= request.component_capacity;
  diagnostics.residual_producer_semantic_identity = residual_cache_bypass ?
    kLegacyComponentCacheResidualBypassSemanticIdentity :
    (optimizer_residual_view == nullptr ?
      kFixedSpResidualProducerSemanticIdentity :
      kOptimizerResidualProducerSemanticIdentity);
  diagnostics.residual_solve_bypassed =
    optimizer_residual_view != nullptr || residual_cache_bypass;
  diagnostics.residual_solve_bypassed_by_component_cache =
    residual_cache_bypass;
  diagnostics.component_cache_persistent_residual_bypass_count =
    residual_cache_bypass ? 1 : 0;

  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller CUDA device for legacy eig batch");
  if (component_cache && cache_info_before.device_id != caller_device_id) {
    throw std::runtime_error(
      "legacy eig component cache device does not match batch device");
  }
  std::shared_ptr<DeviceResidualBatch> residual_token;
  DeviceResidualConsumerView residual_view;
  bool consumer_registered = false;
  cudaStream_t producer_stream = nullptr;
  try {
    const std::chrono::steady_clock::time_point residual_start =
      std::chrono::steady_clock::now();
    if (residual_cache_bypass) {
      residual_view.n = batch.n;
      residual_view.target_count = batch.target_count;
      residual_view.device_id = cache_info_before.device_id;
      residual_view.target_keys = batch.target_keys;
      residual_view.executed_routes.assign(
        static_cast<std::size_t>(batch.target_count), FixedSpRoute::Unset);
      residual_view.solver_statuses.assign(
        static_cast<std::size_t>(batch.target_count),
        FixedSpStatus::ErrInternalCuda);
      for (int component = 0; component < component_count; ++component) {
        const int target =
          component_targets[static_cast<std::size_t>(component)];
        residual_view.executed_routes[static_cast<std::size_t>(target)] =
          component_cache_routes[static_cast<std::size_t>(component)];
        residual_view.solver_statuses[static_cast<std::size_t>(target)] =
          component_cache_statuses[static_cast<std::size_t>(component)];
      }
      for (const FullCudaCiExactBatchPairRequest& pair : request.pairs) {
        const std::size_t left =
          static_cast<std::size_t>(pair.left_target_index);
        const std::size_t right =
          static_cast<std::size_t>(pair.right_target_index);
        if (residual_view.executed_routes[left] == FixedSpRoute::Unset ||
            residual_view.executed_routes[right] == FixedSpRoute::Unset ||
            !is_successful_fixed_sp_status(residual_view.solver_statuses[left]) ||
            !is_successful_fixed_sp_status(residual_view.solver_statuses[right])) {
          throw std::runtime_error(
            "legacy eig component cache residual metadata is incomplete");
        }
      }
    } else if (optimizer_residual_view == nullptr) {
      residual_token = solve_fixed_sp_batch(prepared_s, batch);
      residual_view = acquire_device_residual_consumer_view(residual_token);
    } else {
      residual_view.residuals = optimizer_residual_view->residuals;
      residual_view.n = optimizer_residual_view->n;
      residual_view.target_count = optimizer_residual_view->target_count;
      residual_view.device_id = optimizer_residual_view->device_id;
      residual_view.producer_stream =
        optimizer_residual_view->producer_stream;
      residual_view.producer_completion_event =
        optimizer_residual_view->producer_completion_event;
      residual_view.target_keys = optimizer_residual_view->target_keys;
      residual_view.executed_routes.assign(
        static_cast<std::size_t>(residual_view.target_count),
        FixedSpRoute::Unset);
      residual_view.solver_statuses.assign(
        static_cast<std::size_t>(residual_view.target_count),
        FixedSpStatus::OkAugmentedQr);
    }
    producer_stream = residual_view.producer_stream;
    if (residual_view.target_keys != batch.target_keys ||
        residual_view.n != batch.n ||
        residual_view.target_count != batch.target_count ||
        (!residual_cache_bypass &&
          (residual_view.residuals == nullptr ||
           residual_view.producer_completion_event == nullptr))) {
      throw std::runtime_error(
        "legacy eig residual view does not match the authenticated request");
    }
    if (residual_token) {
      const DeviceResidualInfo residual_info =
        device_residual_info(residual_token);
      if (residual_info.implicit_residual_d2h_count != 0 ||
          residual_info.shadow_d2h_bytes != 0U ||
          residual_info.shadow_materialize_call_count != 0) {
        throw std::runtime_error(
          "legacy eig residual path observed forbidden numeric D2H");
      }
    }
    diagnostics.residuals_device_resident = !residual_cache_bypass;
    for (int component = 0;
         !residual_cache_bypass && component < component_count; ++component) {
      if (component_cache_hits[static_cast<std::size_t>(component)] == 0U) {
        continue;
      }
      const int target =
        component_targets[static_cast<std::size_t>(component)];
      if (component_cache_routes[static_cast<std::size_t>(component)] !=
          residual_view.executed_routes[static_cast<std::size_t>(target)]) {
        diagnostics.component_cache_persistent_route_mismatch_count += 1;
      }
      if (component_cache_statuses[static_cast<std::size_t>(component)] !=
          residual_view.solver_statuses[static_cast<std::size_t>(target)]) {
        diagnostics.component_cache_persistent_status_mismatch_count += 1;
      }
    }
    if (diagnostics.component_cache_persistent_route_mismatch_count != 0 ||
        diagnostics.component_cache_persistent_status_mismatch_count != 0) {
      throw std::runtime_error(
        "legacy eig component cache residual route identity mismatch");
    }
    diagnostics.residual_solve_host_ms = residual_token ?
      host_elapsed_ms(residual_start) : 0.0;
    check_cuda(cudaSetDevice(residual_view.device_id),
               "select legacy eig residual device");

    const std::chrono::steady_clock::time_point dcov_start =
      std::chrono::steady_clock::now();
    FullCudaCiLegacyEigExecutionContext* execution_context =
      component_cache ? component_cache->execution_context() : nullptr;
    const bool persistent_execution_context = execution_context != nullptr;
    std::unique_ptr<FullCudaCiLegacyEigExecutionContext>
      local_execution_context;
    if (execution_context == nullptr) {
      local_execution_context =
        std::make_unique<FullCudaCiLegacyEigExecutionContext>(
          residual_view.device_id, batch.n, num_col, miss_count, pair_count,
          miss_count > 0 ? 1 : 0, &accounting);
      execution_context = local_execution_context.get();
    }
    execution_context->require_capacity(miss_count, pair_count);
    diagnostics.component_eig_configured_lane_count =
      execution_context->eig_lane_count();
    diagnostics.component_execution_context_reuse_count =
      persistent_execution_context ? 1 : 0;
    diagnostics.component_execution_context_device_bytes =
      persistent_execution_context ? execution_context->device_bytes() : 0U;
    cudaStream_t stream = execution_context->stream();
    if (!residual_cache_bypass) {
      check_cuda(cudaStreamWaitEvent(
        stream, residual_view.producer_completion_event, 0),
        "wait for legacy eig residual producer");
    }

    std::vector<std::uintptr_t> pair_metadata(pair_metadata_word_count, 0U);
    double* vectors = miss_count > 0 ? execution_context->vectors() : nullptr;
    double* selected_values = miss_count > 0 ?
      execution_context->selected_values() : nullptr;
    double* totals = miss_count > 0 ? execution_context->totals() : nullptr;
    double* self_moments = miss_count > 0 ?
      execution_context->self_moments() : nullptr;
    double* pair_cross = execution_context->pair_cross();
    const std::size_t vector_stride = checked_multiply(
      n_size, k_size, "legacy eig vector stride");
    const std::size_t cross_stride = checked_multiply(
      k_size, k_size, "legacy eig cross stride");
    struct ComponentPointers {
      double* vectors = nullptr;
      double* selected_values = nullptr;
      double* total = nullptr;
      double* self_moment = nullptr;
    };
    std::vector<ComponentPointers> component_pointers(component_size);
    for (int component = 0; component < component_count; ++component) {
      ComponentPointers& pointers =
        component_pointers[static_cast<std::size_t>(component)];
      if (component_cache_hits[static_cast<std::size_t>(component)] != 0U) {
        const std::size_t slot =
          component_cache_slots[static_cast<std::size_t>(component)];
        pointers.vectors = component_cache->vectors(slot);
        pointers.selected_values = component_cache->selected_values(slot);
        pointers.total = component_cache->total(slot);
        pointers.self_moment = component_cache->self_moment(slot);
      } else {
        const std::size_t miss = static_cast<std::size_t>(
          component_to_miss[static_cast<std::size_t>(component)]);
        pointers.vectors = vectors + miss * vector_stride;
        pointers.selected_values = selected_values + miss * k_size;
        pointers.total = totals + miss;
        pointers.self_moment = self_moments + miss;
      }
    }
    for (int index = 0; index < pair_count; ++index) {
      const std::size_t pair = static_cast<std::size_t>(index);
      const ComponentPointers& left = component_pointers[
        static_cast<std::size_t>(left_components[pair])];
      const ComponentPointers& right = component_pointers[
        static_cast<std::size_t>(right_components[pair])];
      pair_metadata[pair] = static_cast<std::uintptr_t>(
        logical_sequence_ids[pair]);
      pair_metadata[pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(left.vectors);
      pair_metadata[2U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(right.vectors);
      pair_metadata[3U * pair_size + pair] = reinterpret_cast<std::uintptr_t>(
        pair_cross + pair * cross_stride);
      pair_metadata[4U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(left.selected_values);
      pair_metadata[5U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(right.selected_values);
      pair_metadata[6U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(left.total);
      pair_metadata[7U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(right.total);
      pair_metadata[8U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(left.self_moment);
      pair_metadata[9U * pair_size + pair] =
        reinterpret_cast<std::uintptr_t>(right.self_moment);
    }

    check_cuda(cudaEventRecord(execution_context->metadata_start(), stream),
               "record legacy eig metadata start");
    if (miss_count > 0) {
      check_cuda(cudaMemcpyAsync(
        execution_context->component_targets(), miss_targets.data(),
        component_target_copy_bytes, cudaMemcpyHostToDevice, stream),
        "copy legacy eig component targets");
    }
    check_cuda(cudaMemcpyAsync(
      execution_context->pair_metadata(), pair_metadata.data(),
      pair_metadata_bytes, cudaMemcpyHostToDevice, stream),
      "copy legacy eig pair metadata");
    check_cuda(cudaEventRecord(execution_context->metadata_end(), stream),
               "record legacy eig metadata end");
    diagnostics.metadata_h2d_count = 1 + (miss_count > 0 ? 1 : 0);
    diagnostics.metadata_h2d_bytes =
      component_target_copy_bytes + pair_metadata_bytes;

    const double one = 1.0;
    const double zero = 0.0;
    if (miss_count > 0) {
      check_cuda(cudaEventRecord(execution_context->distance_start(), stream),
                 "record legacy eig distance start");
      const dim3 distance_grid(
        static_cast<unsigned int>(batch.n),
        static_cast<unsigned int>(miss_count), 1U);
      build_distance_rows_batch_kernel<<<
        distance_grid, kVerticalBlockSize, 0, stream
      >>>(
        residual_view.residuals,
        static_cast<const int*>(execution_context->component_targets()), batch.n,
        miss_count, cell_count,
        execution_context->component_matrices(),
        execution_context->row_sums());
      check_cuda(cudaGetLastError(), "launch legacy eig distance rows");
      reduce_rows_batch_kernel<<<
        static_cast<unsigned int>(miss_count), kVerticalBlockSize,
        0, stream
      >>>(
        execution_context->row_sums(), batch.n,
        miss_count, execution_context->totals());
      check_cuda(cudaGetLastError(), "launch legacy eig distance totals");
      check_cuda(cudaEventRecord(execution_context->distance_end(), stream),
                 "record legacy eig distance end");
      diagnostics.eig_workspace_bytes =
        execution_context->eig_workspace_bytes();
      diagnostics.component_eig_active_lane_count =
        execution_context->submit_eigensolves(miss_count);
      diagnostics.component_eig_multilane_batch_count =
        diagnostics.component_eig_active_lane_count > 1 ? 1 : 0;
      double* component_matrices = execution_context->component_matrices();
      double* all_eigenvalues = execution_context->eigenvalues();
      select_center_legacy_eigencomponent_kernel<<<
        static_cast<unsigned int>(miss_count), kVerticalBlockSize,
        n_size * sizeof(unsigned char), stream
      >>>(
        all_eigenvalues, component_matrices, batch.n, num_col, miss_count,
        execution_context->selected_values(), vectors);
      check_cuda(cudaGetLastError(),
                 "launch legacy eig component selection");
      check_cublas(cublasDgemmStridedBatched(
        execution_context->blas(), CUBLAS_OP_T, CUBLAS_OP_N,
        num_col, num_col, batch.n, &one,
        vectors, batch.n, static_cast<long long>(vector_stride),
        vectors, batch.n, static_cast<long long>(vector_stride),
        &zero, execution_context->component_grams(), num_col,
        static_cast<long long>(cross_stride), miss_count),
        "form legacy eig component Gram matrices");
      reduce_legacy_self_moment_kernel<<<
        static_cast<unsigned int>(miss_count), kVerticalBlockSize,
        0, stream
      >>>(
        execution_context->component_grams(),
        execution_context->selected_values(),
        miss_count, num_col,
        execution_context->self_moments());
      check_cuda(cudaGetLastError(), "launch legacy eig self moments");
      const std::size_t selected_value_bytes = k_size * sizeof(double);
      const std::size_t vector_bytes = vector_stride * sizeof(double);
      for (int component = 0; component < component_count; ++component) {
        if (component_cache_reservations[
              static_cast<std::size_t>(component)] == 0U) {
          continue;
        }
        const std::size_t miss = static_cast<std::size_t>(
          component_to_miss[static_cast<std::size_t>(component)]);
        const std::size_t slot =
          component_cache_slots[static_cast<std::size_t>(component)];
        check_cuda(cudaMemcpyAsync(
          component_cache->vectors(slot), vectors + miss * vector_stride,
          vector_bytes, cudaMemcpyDeviceToDevice, stream),
          "store legacy eig cached vectors");
        check_cuda(cudaMemcpyAsync(
          component_cache->selected_values(slot),
          selected_values + miss * k_size, selected_value_bytes,
          cudaMemcpyDeviceToDevice, stream),
          "store legacy eig cached eigenvalues");
        check_cuda(cudaMemcpyAsync(
          component_cache->total(slot), totals + miss, sizeof(double),
          cudaMemcpyDeviceToDevice, stream),
          "store legacy eig cached total");
        check_cuda(cudaMemcpyAsync(
          component_cache->self_moment(slot), self_moments + miss,
          sizeof(double), cudaMemcpyDeviceToDevice, stream),
          "store legacy eig cached self moment");
        diagnostics.component_cache_persistent_store_d2d_bytes +=
          vector_bytes + selected_value_bytes + 2U * sizeof(double);
        diagnostics.component_cache_persistent_insert_count += 1;
      }
      check_cuda(cudaEventRecord(execution_context->finalize_end(), stream),
                 "record legacy eig finalize end");
    }

    diagnostics.component_build_count = miss_count;
    diagnostics.cuda_full_eig_count = miss_count;
    diagnostics.components_device_resident = true;
    diagnostics.persistent_component_bytes = miss_size *
      (vector_stride * sizeof(double) + k_size * sizeof(double) +
       2U * sizeof(double));

    check_cuda(cudaEventRecord(execution_context->pair_start(), stream),
               "record legacy eig pair start");
    std::uintptr_t* device_pair_metadata =
      static_cast<std::uintptr_t*>(execution_context->pair_metadata());
    check_cublas(cublasDgemmBatched(
      execution_context->blas(), CUBLAS_OP_T, CUBLAS_OP_N,
      num_col, num_col, batch.n, &one,
      reinterpret_cast<const double**>(device_pair_metadata + pair_size),
      batch.n,
      reinterpret_cast<const double**>(device_pair_metadata +
                                       2U * pair_size), batch.n,
      &zero,
      reinterpret_cast<double**>(device_pair_metadata + 3U * pair_size),
      num_col,
      pair_count), "form legacy eig pair cross matrices");
    evaluate_legacy_eig_pair_batch_kernel<<<
      static_cast<unsigned int>(pair_count), kVerticalBlockSize,
      0, stream
    >>>(
      execution_context->pair_cross(),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 4U * pair_size),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 5U * pair_size),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 6U * pair_size),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 7U * pair_size),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 8U * pair_size),
      reinterpret_cast<const double* const*>(
        device_pair_metadata + 9U * pair_size),
      reinterpret_cast<const unsigned long long*>(device_pair_metadata),
      pair_count, num_col, batch.n, kCanonicalAlpha,
      static_cast<DeviceCompactCiResult*>(execution_context->device_results()));
    check_cuda(cudaGetLastError(), "launch legacy eig pair evaluation");
    check_cuda(cudaEventRecord(execution_context->pair_end(), stream),
               "record legacy eig pair end");
    diagnostics.pair_evaluation_count = pair_count;
    diagnostics.cuda_pair_count = pair_count;
    diagnostics.cuda_gamma_count = pair_count;
    diagnostics.pair_workspace_bytes =
      checked_multiply(pair_cross_cells, sizeof(double),
                       "legacy eig pair cross diagnostic bytes");

    std::vector<DeviceCompactCiResult> host_results(pair_size);
    std::vector<int> host_solver_info(miss_size, -1);
    check_cuda(cudaMemcpyAsync(
      host_results.data(), execution_context->device_results(), result_bytes,
      cudaMemcpyDeviceToHost, stream),
      "copy compact legacy eig results");
    if (miss_count > 0) {
      check_cuda(cudaMemcpyAsync(
        host_solver_info.data(), execution_context->solver_info(),
        checked_multiply(miss_size, sizeof(int),
                         "legacy eig solver status copy bytes"),
        cudaMemcpyDeviceToHost, stream),
        "copy compact legacy eig solver status");
    }
    diagnostics.compact_result_d2h_count = 1;
    diagnostics.compact_result_d2h_bytes = result_bytes;
    diagnostics.compact_status_d2h_count = miss_count > 0 ? 1 : 0;
    diagnostics.compact_status_d2h_bytes =
      checked_multiply(miss_size, sizeof(int),
                       "legacy eig solver status bytes");
    check_cuda(cudaEventRecord(execution_context->completion(), stream),
               "record legacy eig completion");
    if (residual_token) {
      register_device_residual_consumer_event(
        residual_token, execution_context->completion());
      consumer_registered = true;
      diagnostics.consumer_event_registration_count = 1;
    }
    check_cuda(cudaEventSynchronize(execution_context->completion()),
               "wait for compact legacy eig results");
    diagnostics.explicit_host_wait_count += 1;
    diagnostics.metadata_h2d_cuda_ms = cuda_elapsed_ms(
      execution_context->metadata_start(), execution_context->metadata_end());
    if (miss_count > 0) {
      diagnostics.distance_build_cuda_ms = cuda_elapsed_ms(
        execution_context->distance_start(), execution_context->distance_end());
      diagnostics.full_eig_cuda_ms = cuda_elapsed_ms(
        execution_context->eig_start(), execution_context->eig_end());
      diagnostics.component_finalize_cuda_ms = cuda_elapsed_ms(
        execution_context->eig_end(), execution_context->finalize_end());
    }
    diagnostics.component_build_cuda_ms =
      diagnostics.distance_build_cuda_ms + diagnostics.full_eig_cuda_ms +
      diagnostics.component_finalize_cuda_ms;
    diagnostics.pair_evaluation_cuda_ms = cuda_elapsed_ms(
      execution_context->pair_start(), execution_context->pair_end());
    diagnostics.compact_d2h_cuda_ms = cuda_elapsed_ms(
      execution_context->pair_end(), execution_context->completion());
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
    if (component_cache) {
      for (int component = 0; component < component_count; ++component) {
        if (component_cache_reservations[
              static_cast<std::size_t>(component)] == 0U) {
          continue;
        }
        const int target =
          component_targets[static_cast<std::size_t>(component)];
        component_cache->commit(
          batch.target_keys[static_cast<std::size_t>(target)],
          component_cache_slots[static_cast<std::size_t>(component)],
          residual_view.executed_routes[static_cast<std::size_t>(target)],
          residual_view.solver_statuses[static_cast<std::size_t>(target)]);
      }
      const FullCudaCiLegacyEigComponentCacheInfo cache_info_after =
        component_cache->info();
      diagnostics.component_cache_persistent_entries_after =
        cache_info_after.entries;
      diagnostics.component_cache_persistent_bypass_count = static_cast<int>(
        cache_info_after.bypass_count - cache_info_before.bypass_count);
    }
    if (!residual_cache_bypass) {
      check_cuda(cudaStreamSynchronize(residual_view.producer_stream),
                 "wait for legacy eig residual consumer proxy");
      diagnostics.explicit_host_wait_count += 1;
    }

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
    local_execution_context.reset();
    if (residual_token) {
      release_device_residual(residual_token);
      free_device_residual(&residual_token);
    }
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
    diagnostics.compact_status_d2h_count == (miss_count > 0 ? 1 : 0);
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
  const bool component_cache_accounted = component_cache ?
    diagnostics.component_cache_persistent_request_count == component_count &&
      diagnostics.component_cache_persistent_request_count ==
        diagnostics.component_cache_persistent_hit_count +
          diagnostics.component_cache_persistent_miss_count &&
      diagnostics.component_cache_persistent_miss_count ==
        diagnostics.component_cache_persistent_insert_count +
          diagnostics.component_cache_persistent_bypass_count &&
      diagnostics.component_cache_persistent_route_mismatch_count == 0 &&
      diagnostics.component_cache_persistent_status_mismatch_count == 0 &&
      diagnostics.component_cache_persistent_residual_bypass_count ==
        (residual_cache_bypass ? 1 : 0) &&
      diagnostics.component_cache_persistent_entries_after ==
        diagnostics.component_cache_persistent_entries_before +
          static_cast<std::size_t>(
            diagnostics.component_cache_persistent_insert_count) &&
      diagnostics.component_cache_persistent_capacity_entries >=
        diagnostics.component_cache_persistent_entries_after &&
      diagnostics.component_cache_persistent_device_bytes ==
        cache_info_before.device_bytes &&
      diagnostics.component_cache_persistent_store_d2d_bytes ==
        static_cast<std::size_t>(
          diagnostics.component_cache_persistent_insert_count) *
            cache_info_before.bytes_per_entry :
    diagnostics.component_cache_persistent_request_count == 0 &&
      diagnostics.component_cache_persistent_hit_count == 0 &&
      diagnostics.component_cache_persistent_miss_count == 0 &&
      diagnostics.component_cache_persistent_insert_count == 0 &&
      diagnostics.component_cache_persistent_bypass_count == 0 &&
      diagnostics.component_cache_persistent_route_mismatch_count == 0 &&
      diagnostics.component_cache_persistent_status_mismatch_count == 0 &&
      diagnostics.component_cache_persistent_residual_bypass_count == 0 &&
      diagnostics.component_cache_persistent_capacity_entries == 0U &&
      diagnostics.component_cache_persistent_entries_before == 0U &&
      diagnostics.component_cache_persistent_entries_after == 0U &&
      diagnostics.component_cache_persistent_device_bytes == 0U &&
      diagnostics.component_cache_persistent_store_d2d_bytes == 0U;
  if (!diagnostics.compact_result_only_d2h ||
      !diagnostics.deterministic_logical_order ||
      !diagnostics.component_capacity_respected ||
      (!diagnostics.residuals_device_resident &&
       !diagnostics.residual_solve_bypassed_by_component_cache) ||
      !diagnostics.bounded_allocation ||
      !diagnostics.leak_free_teardown ||
      !component_cache_accounted ||
      !diagnostics.caller_device_restored ||
      diagnostics.solver_failure_count != 0) {
    throw std::runtime_error(
      "legacy eig batch structural accounting gate failed closed");
  }
  diagnostics.total_host_ms = host_elapsed_ms(total_start);
  return result;
}

FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request) {
  return run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    prepared_s, batch, request, nullptr, nullptr);
}

FullCudaCiLegacyEigBatchResult
run_full_cuda_ci_phase35_legacy_eig_batch_with_component_cache(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request,
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  if (!cache) {
    throw std::runtime_error("legacy eig component cache is null");
  }
  return run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    prepared_s, batch, request, nullptr, cache);
}

FullCudaCiLegacyEigBatchResult
run_full_cuda_ci_phase35_legacy_eig_batch_from_optimizer_residual(
    const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>& residual,
    const FullCudaCiLegacyEigBatchRequest& request) {
  const PreparedSInfo prepared_info = prepared_s_gpu_info(prepared_s);
  const MultiPenaltyGcvCudaResidualConsumerView residual_view =
    acquire_multi_penalty_gcv_cuda_residual_consumer_view(residual);
  FixedSpBatchHostView batch;
  batch.n = residual_view.n;
  batch.null_dim = prepared_info.null_dim;
  batch.penalty_count = prepared_info.penalty_count;
  batch.target_count = residual_view.target_count;
  batch.output_mask = FixedSpOutputResiduals;
  batch.target_keys = residual_view.target_keys;
  FullCudaCiLegacyEigBatchResult result =
    run_full_cuda_ci_phase35_legacy_eig_batch_impl(
      prepared_s, batch, request, &residual_view, nullptr);
  for (FullCudaCiCompactHostRecord& record : result.records) {
    record.solver_route = "OPTIMIZER_ACCEPTED|OPTIMIZER_ACCEPTED";
    record.optimizer_status = "LIVE_CUDA_GCV_ACCEPTED";
  }
  return result;
}

FullCudaCiOptimizerResidualParityDiagnostics
compare_full_cuda_ci_optimizer_and_fixed_residuals(
    const std::shared_ptr<MultiPenaltyGcvCudaResidualBatch>&
      optimizer_residual,
    const std::shared_ptr<DeviceResidualBatch>& fixed_residual) {
  const auto started = std::chrono::steady_clock::now();
  FullCudaCiOptimizerResidualParityDiagnostics diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-phase10-optimizer-residual-parity-v1";
  const MultiPenaltyGcvCudaResidualInfo optimizer_info =
    multi_penalty_gcv_cuda_residual_info(optimizer_residual);
  const DeviceResidualInfo fixed_info = device_residual_info(fixed_residual);
  diagnostics.n = optimizer_info.n;
  diagnostics.target_count = optimizer_info.target_count;
  diagnostics.optimizer_status_failure_count = static_cast<int>(
    std::count_if(
      optimizer_info.optimizer_status.begin(),
      optimizer_info.optimizer_status.end(),
      [](int status) { return status != 0; }));
  diagnostics.fixed_status_failure_count = static_cast<int>(std::count_if(
    fixed_info.solver_statuses.begin(), fixed_info.solver_statuses.end(),
    [](FixedSpStatus status) {
      return static_cast<int>(status) >= 10;
    }));
  for (std::size_t index = 0; index < fixed_info.executed_routes.size();
       ++index) {
    const FixedSpRoute route = fixed_info.executed_routes[index];
    const FixedSpStatus status = fixed_info.solver_statuses[index];
    const bool consistent =
      (route == FixedSpRoute::CholeskyBatched &&
       (status == FixedSpStatus::OkCholeskyBatched ||
        status == FixedSpStatus::OkCholeskySingle)) ||
      (route == FixedSpRoute::AugmentedQr &&
       status == FixedSpStatus::OkAugmentedQr) ||
      (route == FixedSpRoute::AugmentedSvd &&
       status == FixedSpStatus::OkAugmentedSvd);
    if (!consistent) diagnostics.fixed_route_status_mismatch_count += 1;
  }
  if (optimizer_info.released || !optimizer_info.device_resident ||
      diagnostics.optimizer_status_failure_count != 0 ||
      diagnostics.fixed_status_failure_count != 0 ||
      diagnostics.fixed_route_status_mismatch_count != 0) {
    throw std::runtime_error(
      "optimizer residual parity rejects an unsuccessful producer");
  }
  const MultiPenaltyGcvCudaResidualConsumerView optimizer_view =
    acquire_multi_penalty_gcv_cuda_residual_consumer_view(
      optimizer_residual);
  const DeviceResidualConsumerView fixed_view =
    acquire_device_residual_consumer_view(fixed_residual);
  if (optimizer_view.n != fixed_view.n ||
      optimizer_view.target_count != fixed_view.target_count ||
      optimizer_view.device_id != fixed_view.device_id ||
      optimizer_view.target_keys != fixed_view.target_keys ||
      optimizer_view.residuals == nullptr || fixed_view.residuals == nullptr ||
      optimizer_view.producer_completion_event == nullptr ||
      fixed_view.producer_completion_event == nullptr) {
    throw std::runtime_error(
      "optimizer residual parity identity mismatch");
  }
  diagnostics.target_identity_authenticated = true;
  diagnostics.device_identity_authenticated = true;
  diagnostics.residual_payload_device_resident = true;
  diagnostics.value_count = checked_multiply(
    static_cast<std::size_t>(optimizer_view.n),
    static_cast<std::size_t>(optimizer_view.target_count),
    "optimizer residual parity values");

  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller device for optimizer residual parity");
  try {
    check_cuda(cudaSetDevice(optimizer_view.device_id),
               "select optimizer residual parity device");
    VerticalCallAccounting accounting;
    const std::size_t compact_bytes = checked_multiply(
      static_cast<std::size_t>(optimizer_view.target_count),
      sizeof(DeviceResidualParityAccumulator),
      "optimizer residual parity compact bytes");
    TrackedCudaStream stream;
    TrackedCudaEvent comparison_start(true);
    TrackedCudaEvent comparison_end(true);
    TrackedCudaEvent completion(true);
    TrackedDeviceBuffer device_accumulators(compact_bytes, &accounting);
    check_cuda(cudaStreamWaitEvent(
      stream.get(), optimizer_view.producer_completion_event, 0),
      "wait for optimizer residual producer during parity");
    check_cuda(cudaStreamWaitEvent(
      stream.get(), fixed_view.producer_completion_event, 0),
      "wait for fixed residual producer during parity");
    diagnostics.producer_event_wait_count = 2;
    check_cuda(cudaMemsetAsync(
      device_accumulators.get(), 0, compact_bytes, stream.get()),
      "initialize optimizer residual parity accumulators");
    check_cuda(cudaEventRecord(comparison_start.get(), stream.get()),
               "record optimizer residual parity start");
    compare_residuals_by_target_kernel<<<
      static_cast<unsigned int>(optimizer_view.target_count),
      kVerticalBlockSize, 0, stream.get()
    >>>(
      optimizer_view.residuals, fixed_view.residuals, optimizer_view.n,
      optimizer_view.target_count,
      static_cast<DeviceResidualParityAccumulator*>(
        device_accumulators.get()));
    check_cuda(cudaGetLastError(),
               "launch optimizer residual parity kernel");
    check_cuda(cudaEventRecord(comparison_end.get(), stream.get()),
               "record optimizer residual parity end");
    std::vector<DeviceResidualParityAccumulator> host_accumulators(
      static_cast<std::size_t>(optimizer_view.target_count));
    check_cuda(cudaMemcpyAsync(
      host_accumulators.data(), device_accumulators.get(), compact_bytes,
      cudaMemcpyDeviceToHost, stream.get()),
      "copy optimizer residual parity compact diagnostics");
    diagnostics.compact_d2h_count = 1;
    diagnostics.compact_d2h_bytes = compact_bytes;
    check_cuda(cudaEventRecord(completion.get(), stream.get()),
               "record optimizer residual parity completion");
    register_device_residual_consumer_event(
      fixed_residual, completion.get());
    diagnostics.consumer_event_registration_count = 1;
    check_cuda(cudaEventSynchronize(completion.get()),
               "wait for optimizer residual parity diagnostics");
    check_cuda(cudaStreamSynchronize(fixed_view.producer_stream),
               "wait for fixed residual parity consumer proxy");
    diagnostics.comparison_cuda_ms = cuda_elapsed_ms(
      comparison_start.get(), comparison_end.get());

    double squared_difference_sum = 0.0;
    double squared_reference_sum = 0.0;
    for (const DeviceResidualParityAccumulator& value : host_accumulators) {
      diagnostics.mismatch_value_count += value.mismatch_count;
      diagnostics.mismatch_target_count += value.mismatch_count > 0 ? 1 : 0;
      double maximum = 0.0;
      const std::uint64_t maximum_bits = value.max_abs_difference_bits;
      std::memcpy(&maximum, &maximum_bits, sizeof(maximum));
      diagnostics.max_abs_difference = std::max(
        diagnostics.max_abs_difference, maximum);
      squared_difference_sum += value.squared_difference_sum;
      squared_reference_sum += value.squared_reference_sum;
    }
    diagnostics.bitwise_equal_target_count =
      diagnostics.target_count - diagnostics.mismatch_target_count;
    diagnostics.bitwise_equal_value_count =
      diagnostics.value_count - diagnostics.mismatch_value_count;
    diagnostics.relative_l2_difference = std::sqrt(squared_difference_sum) /
      std::max(std::sqrt(squared_reference_sum), 1e-300);
    diagnostics.compact_diagnostics_only_d2h =
      diagnostics.residual_d2h_count == 0 &&
      diagnostics.residual_d2h_bytes == 0U &&
      diagnostics.compact_d2h_count == 1;

    device_accumulators.close();
    completion.close();
    comparison_end.close();
    comparison_start.close();
    stream.close();
    check_cuda(cudaSetDevice(caller_device_id),
               "restore caller device after optimizer residual parity");
    diagnostics.caller_device_restored = true;
  } catch (...) {
    if (caller_device_id >= 0) (void)cudaSetDevice(caller_device_id);
    throw;
  }
  diagnostics.total_host_ms = host_elapsed_ms(started);
  return diagnostics;
}

FullCudaCiFixedResidualIdentityDiagnostics
compare_full_cuda_ci_fixed_residuals_and_exact_components(
    const std::shared_ptr<DeviceResidualBatch>& left_residual,
    const std::shared_ptr<DeviceResidualBatch>& right_residual) {
  const auto started = std::chrono::steady_clock::now();
  FullCudaCiFixedResidualIdentityDiagnostics diagnostics;
  diagnostics.schema_version =
    "full-cuda-ci-phase10-fixed-residual-identity-v1";
  const DeviceResidualInfo left_info = device_residual_info(left_residual);
  const DeviceResidualInfo right_info = device_residual_info(right_residual);
  diagnostics.n = left_info.n;
  diagnostics.left_target_count = left_info.target_count;
  diagnostics.right_target_count = right_info.target_count;
  diagnostics.left_status_failure_count = static_cast<int>(std::count_if(
    left_info.solver_statuses.begin(), left_info.solver_statuses.end(),
    [](FixedSpStatus status) { return static_cast<int>(status) >= 10; }));
  diagnostics.right_status_failure_count = static_cast<int>(std::count_if(
    right_info.solver_statuses.begin(), right_info.solver_statuses.end(),
    [](FixedSpStatus status) { return static_cast<int>(status) >= 10; }));
  if (diagnostics.left_status_failure_count != 0 ||
      diagnostics.right_status_failure_count != 0) {
    throw std::runtime_error(
      "fixed residual identity rejects an unsuccessful producer");
  }

  const DeviceResidualConsumerView left_view =
    acquire_device_residual_consumer_view(left_residual);
  const DeviceResidualConsumerView right_view =
    acquire_device_residual_consumer_view(right_residual);
  if (left_view.n != right_view.n || left_view.n <= 0 ||
      left_view.device_id != right_view.device_id ||
      left_view.residuals == nullptr || right_view.residuals == nullptr ||
      left_view.producer_completion_event == nullptr ||
      right_view.producer_completion_event == nullptr) {
    throw std::runtime_error("fixed residual identity producer mismatch");
  }
  diagnostics.device_identity_authenticated = true;
  diagnostics.residual_payload_device_resident = true;

  std::unordered_map<std::string, int> right_index;
  right_index.reserve(right_info.target_keys.size());
  for (int index = 0; index < right_info.target_count; ++index) {
    const bool inserted = right_index.emplace(
      right_info.target_keys[static_cast<std::size_t>(index)], index).second;
    if (!inserted) {
      throw std::runtime_error(
        "fixed residual identity found duplicate right TargetKeys");
    }
  }
  std::vector<int> left_target_indices;
  std::vector<int> right_target_indices;
  left_target_indices.reserve(left_info.target_keys.size());
  right_target_indices.reserve(left_info.target_keys.size());
  diagnostics.matched_target_keys.reserve(left_info.target_keys.size());
  for (int left = 0; left < left_info.target_count; ++left) {
    const std::string& key =
      left_info.target_keys[static_cast<std::size_t>(left)];
    const auto found = right_index.find(key);
    if (found == right_index.end()) {
      diagnostics.left_only_target_count += 1;
      continue;
    }
    const int right = found->second;
    left_target_indices.push_back(left);
    right_target_indices.push_back(right);
    diagnostics.matched_target_keys.push_back(key);
    if (left_info.planned_routes[static_cast<std::size_t>(left)] !=
        right_info.planned_routes[static_cast<std::size_t>(right)]) {
      diagnostics.planned_route_mismatch_count += 1;
    }
    if (left_info.executed_routes[static_cast<std::size_t>(left)] !=
        right_info.executed_routes[static_cast<std::size_t>(right)]) {
      diagnostics.executed_route_mismatch_count += 1;
    }
    if (left_info.solver_statuses[static_cast<std::size_t>(left)] !=
        right_info.solver_statuses[static_cast<std::size_t>(right)]) {
      diagnostics.solver_status_mismatch_count += 1;
    }
  }
  diagnostics.matched_target_count =
    static_cast<int>(left_target_indices.size());
  diagnostics.right_only_target_count = right_info.target_count -
    diagnostics.matched_target_count;
  if (diagnostics.matched_target_count <= 0) {
    throw std::runtime_error(
      "fixed residual identity requires at least one common TargetKey");
  }
  diagnostics.target_identity_authenticated = true;

  const std::size_t matched =
    static_cast<std::size_t>(diagnostics.matched_target_count);
  const std::size_t n = static_cast<std::size_t>(diagnostics.n);
  const std::size_t cell_count = checked_multiply(
    n, n, "fixed residual identity component cells");
  const std::size_t component_cells = checked_multiply(
    matched, cell_count, "fixed residual identity component pool cells");
  const std::size_t component_rows = checked_multiply(
    matched, n, "fixed residual identity component rows");
  diagnostics.residual_value_count = checked_multiply(
    matched, n, "fixed residual identity residual values");
  diagnostics.centered_component_value_count = component_cells;
  diagnostics.row_sum_value_count = component_rows;

  const std::size_t index_bytes = checked_multiply(
    2U * matched, sizeof(int), "fixed residual identity target indices");
  const std::size_t accumulator_bytes = checked_multiply(
    matched, sizeof(DeviceFixedResidualIdentityAccumulator),
    "fixed residual identity compact accumulators");
  const std::size_t component_bytes = checked_multiply(
    component_cells, sizeof(double),
    "fixed residual identity component bytes");
  const std::size_t row_bytes = checked_multiply(
    component_rows, sizeof(double), "fixed residual identity row bytes");
  const std::size_t scalar_bytes = checked_multiply(
    matched, sizeof(double), "fixed residual identity scalar bytes");

  const FullCudaCiVerticalResourceSnapshot resources_before =
    full_cuda_ci_vertical_resource_snapshot();
  VerticalCallAccounting accounting;
  int caller_device_id = -1;
  check_cuda(cudaGetDevice(&caller_device_id),
             "capture caller device for fixed residual identity");
  try {
    check_cuda(cudaSetDevice(left_view.device_id),
               "select fixed residual identity device");
    TrackedCudaStream stream;
    TrackedCudaEvent residual_start(true);
    TrackedCudaEvent residual_end(true);
    TrackedCudaEvent component_end(true);
    TrackedCudaEvent comparison_end(true);
    TrackedCudaEvent completion(true);
    TrackedDeviceBuffer target_indices(index_bytes, &accounting);
    TrackedDeviceBuffer accumulators(accumulator_bytes, &accounting);
    TrackedDeviceBuffer left_centered(component_bytes, &accounting);
    TrackedDeviceBuffer right_centered(component_bytes, &accounting);
    TrackedDeviceBuffer left_rows(row_bytes, &accounting);
    TrackedDeviceBuffer right_rows(row_bytes, &accounting);
    TrackedDeviceBuffer left_totals(scalar_bytes, &accounting);
    TrackedDeviceBuffer right_totals(scalar_bytes, &accounting);
    TrackedDeviceBuffer left_self_moments(scalar_bytes, &accounting);
    TrackedDeviceBuffer right_self_moments(scalar_bytes, &accounting);

    check_cuda(cudaStreamWaitEvent(
      stream.get(), left_view.producer_completion_event, 0),
      "wait for left fixed residual producer");
    check_cuda(cudaStreamWaitEvent(
      stream.get(), right_view.producer_completion_event, 0),
      "wait for right fixed residual producer");
    diagnostics.producer_event_wait_count = 2;
    std::vector<int> host_indices;
    host_indices.reserve(2U * matched);
    host_indices.insert(host_indices.end(), left_target_indices.begin(),
                        left_target_indices.end());
    host_indices.insert(host_indices.end(), right_target_indices.begin(),
                        right_target_indices.end());
    check_cuda(cudaMemcpyAsync(
      target_indices.get(), host_indices.data(), index_bytes,
      cudaMemcpyHostToDevice, stream.get()),
      "copy fixed residual identity target indices");
    diagnostics.metadata_h2d_count = 1;
    diagnostics.metadata_h2d_bytes = index_bytes;
    check_cuda(cudaMemsetAsync(
      accumulators.get(), 0, accumulator_bytes, stream.get()),
      "initialize fixed residual identity accumulators");

    const int* device_left_indices =
      static_cast<const int*>(target_indices.get());
    const int* device_right_indices = device_left_indices + matched;
    auto* device_accumulators =
      static_cast<DeviceFixedResidualIdentityAccumulator*>(
        accumulators.get());
    check_cuda(cudaEventRecord(residual_start.get(), stream.get()),
               "record fixed residual identity residual start");
    compare_fixed_residuals_by_key_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      left_view.residuals, right_view.residuals, device_left_indices,
      device_right_indices, diagnostics.n, diagnostics.matched_target_count,
      device_accumulators);
    check_cuda(cudaGetLastError(),
               "launch fixed residual identity residual comparison");
    check_cuda(cudaEventRecord(residual_end.get(), stream.get()),
               "record fixed residual identity residual end");

    const dim3 distance_grid(
      static_cast<unsigned int>(diagnostics.n),
      static_cast<unsigned int>(matched), 1U);
    build_distance_rows_batch_kernel<<<
      distance_grid, kVerticalBlockSize, 0, stream.get()
    >>>(
      left_view.residuals, device_left_indices, diagnostics.n,
      diagnostics.matched_target_count, cell_count,
      static_cast<double*>(left_centered.get()),
      static_cast<double*>(left_rows.get()));
    check_cuda(cudaGetLastError(),
               "build left fixed residual identity components");
    build_distance_rows_batch_kernel<<<
      distance_grid, kVerticalBlockSize, 0, stream.get()
    >>>(
      right_view.residuals, device_right_indices, diagnostics.n,
      diagnostics.matched_target_count, cell_count,
      static_cast<double*>(right_centered.get()),
      static_cast<double*>(right_rows.get()));
    check_cuda(cudaGetLastError(),
               "build right fixed residual identity components");
    reduce_rows_batch_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      static_cast<const double*>(left_rows.get()), diagnostics.n,
      diagnostics.matched_target_count,
      static_cast<double*>(left_totals.get()));
    check_cuda(cudaGetLastError(),
               "reduce left fixed residual identity component rows");
    reduce_rows_batch_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      static_cast<const double*>(right_rows.get()), diagnostics.n,
      diagnostics.matched_target_count,
      static_cast<double*>(right_totals.get()));
    check_cuda(cudaGetLastError(),
               "reduce right fixed residual identity component rows");
    const std::size_t center_blocks =
      (component_cells + kVerticalBlockSize - 1U) / kVerticalBlockSize;
    if (center_blocks == 0U ||
        center_blocks > std::numeric_limits<unsigned int>::max()) {
      throw std::runtime_error(
        "fixed residual identity component launch overflow");
    }
    center_distance_batch_kernel<<<
      static_cast<unsigned int>(center_blocks), kVerticalBlockSize, 0,
      stream.get()
    >>>(
      static_cast<double*>(left_centered.get()),
      static_cast<const double*>(left_rows.get()),
      static_cast<const double*>(left_totals.get()), diagnostics.n,
      cell_count, component_cells);
    check_cuda(cudaGetLastError(),
               "center left fixed residual identity components");
    center_distance_batch_kernel<<<
      static_cast<unsigned int>(center_blocks), kVerticalBlockSize, 0,
      stream.get()
    >>>(
      static_cast<double*>(right_centered.get()),
      static_cast<const double*>(right_rows.get()),
      static_cast<const double*>(right_totals.get()), diagnostics.n,
      cell_count, component_cells);
    check_cuda(cudaGetLastError(),
               "center right fixed residual identity components");
    self_moment_batch_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      static_cast<const double*>(left_centered.get()),
      diagnostics.matched_target_count, cell_count,
      static_cast<double*>(left_self_moments.get()));
    check_cuda(cudaGetLastError(),
               "build left fixed residual identity self moments");
    self_moment_batch_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      static_cast<const double*>(right_centered.get()),
      diagnostics.matched_target_count, cell_count,
      static_cast<double*>(right_self_moments.get()));
    check_cuda(cudaGetLastError(),
               "build right fixed residual identity self moments");
    check_cuda(cudaEventRecord(component_end.get(), stream.get()),
               "record fixed residual identity component end");

    compare_exact_components_by_key_kernel<<<
      static_cast<unsigned int>(matched), kVerticalBlockSize, 0, stream.get()
    >>>(
      static_cast<const double*>(left_centered.get()),
      static_cast<const double*>(right_centered.get()),
      static_cast<const double*>(left_rows.get()),
      static_cast<const double*>(right_rows.get()),
      static_cast<const double*>(left_totals.get()),
      static_cast<const double*>(right_totals.get()),
      static_cast<const double*>(left_self_moments.get()),
      static_cast<const double*>(right_self_moments.get()), diagnostics.n,
      diagnostics.matched_target_count, cell_count, device_accumulators);
    check_cuda(cudaGetLastError(),
               "compare fixed residual identity exact components");
    check_cuda(cudaEventRecord(comparison_end.get(), stream.get()),
               "record fixed residual identity component comparison end");

    std::vector<DeviceFixedResidualIdentityAccumulator> host_accumulators(
      matched);
    check_cuda(cudaMemcpyAsync(
      host_accumulators.data(), accumulators.get(), accumulator_bytes,
      cudaMemcpyDeviceToHost, stream.get()),
      "copy fixed residual identity compact diagnostics");
    diagnostics.compact_d2h_count = 1;
    diagnostics.compact_d2h_bytes = accumulator_bytes;
    check_cuda(cudaEventRecord(completion.get(), stream.get()),
               "record fixed residual identity completion");
    register_device_residual_consumer_event(left_residual, completion.get());
    register_device_residual_consumer_event(right_residual, completion.get());
    diagnostics.consumer_event_registration_count = 2;
    check_cuda(cudaEventSynchronize(completion.get()),
               "wait for fixed residual identity diagnostics");
    check_cuda(cudaStreamSynchronize(left_view.producer_stream),
               "wait for left fixed residual consumer proxy");
    if (right_view.producer_stream != left_view.producer_stream) {
      check_cuda(cudaStreamSynchronize(right_view.producer_stream),
                 "wait for right fixed residual consumer proxy");
    }
    diagnostics.residual_compare_cuda_ms = cuda_elapsed_ms(
      residual_start.get(), residual_end.get());
    diagnostics.component_build_cuda_ms = cuda_elapsed_ms(
      residual_end.get(), component_end.get());
    diagnostics.component_compare_cuda_ms = cuda_elapsed_ms(
      component_end.get(), comparison_end.get());

    double residual_squared_difference_sum = 0.0;
    double residual_squared_reference_sum = 0.0;
    double component_squared_difference_sum = 0.0;
    double component_squared_reference_sum = 0.0;
    for (const DeviceFixedResidualIdentityAccumulator& value :
         host_accumulators) {
      diagnostics.residual_mismatch_value_count +=
        value.residual_mismatch_count;
      diagnostics.residual_mismatch_target_count +=
        value.residual_mismatch_count > 0 ? 1 : 0;
      diagnostics.centered_component_mismatch_value_count +=
        value.centered_mismatch_count;
      diagnostics.row_sum_mismatch_value_count +=
        value.row_sum_mismatch_count;
      diagnostics.total_mismatch_value_count += value.total_mismatch_count;
      diagnostics.self_moment_mismatch_value_count +=
        value.self_moment_mismatch_count;
      const bool component_mismatch = value.centered_mismatch_count != 0 ||
        value.row_sum_mismatch_count != 0 ||
        value.total_mismatch_count != 0 ||
        value.self_moment_mismatch_count != 0;
      diagnostics.component_mismatch_target_count +=
        component_mismatch ? 1 : 0;
      double residual_maximum = 0.0;
      const std::uint64_t residual_maximum_bits =
        value.residual_max_abs_difference_bits;
      std::memcpy(
        &residual_maximum, &residual_maximum_bits,
        sizeof(residual_maximum));
      diagnostics.residual_max_abs_difference = std::max(
        diagnostics.residual_max_abs_difference, residual_maximum);
      double component_maximum = 0.0;
      const std::uint64_t component_maximum_bits =
        value.component_max_abs_difference_bits;
      std::memcpy(
        &component_maximum, &component_maximum_bits,
        sizeof(component_maximum));
      diagnostics.component_max_abs_difference = std::max(
        diagnostics.component_max_abs_difference, component_maximum);
      residual_squared_difference_sum +=
        value.residual_squared_difference_sum;
      residual_squared_reference_sum +=
        value.residual_squared_reference_sum;
      component_squared_difference_sum +=
        value.component_squared_difference_sum;
      component_squared_reference_sum +=
        value.component_squared_reference_sum;
    }
    diagnostics.residual_relative_l2_difference =
      std::sqrt(residual_squared_difference_sum) /
      std::max(std::sqrt(residual_squared_reference_sum), 1e-300);
    diagnostics.component_relative_l2_difference =
      std::sqrt(component_squared_difference_sum) /
      std::max(std::sqrt(component_squared_reference_sum), 1e-300);
    diagnostics.component_payload_device_resident = true;
    diagnostics.compact_diagnostics_only_d2h =
      diagnostics.residual_d2h_count == 0 &&
      diagnostics.residual_d2h_bytes == 0U &&
      diagnostics.compact_d2h_count == 1;

    right_self_moments.close();
    left_self_moments.close();
    right_totals.close();
    left_totals.close();
    right_rows.close();
    left_rows.close();
    right_centered.close();
    left_centered.close();
    accumulators.close();
    target_indices.close();
    completion.close();
    comparison_end.close();
    component_end.close();
    residual_end.close();
    residual_start.close();
    stream.close();
    check_cuda(cudaSetDevice(caller_device_id),
               "restore caller device after fixed residual identity");
    diagnostics.caller_device_restored = true;
  } catch (...) {
    if (caller_device_id >= 0) (void)cudaSetDevice(caller_device_id);
    throw;
  }

  const FullCudaCiVerticalResourceSnapshot resources_after =
    full_cuda_ci_vertical_resource_snapshot();
  const std::size_t expected_allocation_bytes = index_bytes +
    accumulator_bytes + 2U * (component_bytes + row_bytes +
                               2U * scalar_bytes);
  diagnostics.bounded_allocation =
    accounting.device_allocation_count == 10 &&
    accounting.device_allocation_bytes == expected_allocation_bytes;
  diagnostics.leak_free_teardown =
    resources_after.live_device_allocations ==
      resources_before.live_device_allocations &&
    resources_after.live_device_bytes == resources_before.live_device_bytes &&
    resources_after.live_streams == resources_before.live_streams &&
    resources_after.live_events == resources_before.live_events &&
    accounting.device_allocation_count == accounting.device_free_count &&
    accounting.live_device_bytes == 0U;
  if (!diagnostics.compact_diagnostics_only_d2h ||
      !diagnostics.bounded_allocation || !diagnostics.leak_free_teardown ||
      !diagnostics.caller_device_restored) {
    throw std::runtime_error(
      "fixed residual identity structural accounting gate failed closed");
  }
  diagnostics.total_host_ms = host_elapsed_ms(started);
  return diagnostics;
}

}  // namespace fastkpc
