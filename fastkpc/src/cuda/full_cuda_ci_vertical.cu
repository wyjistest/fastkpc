#include "full_cuda_ci_vertical.hpp"

#include "../full_cuda_ci_contract.hpp"

#include <cuda_runtime.h>

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

}  // namespace fastkpc
