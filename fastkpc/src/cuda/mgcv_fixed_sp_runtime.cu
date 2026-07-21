#include "mgcv_fixed_sp_runtime.hpp"
#include "mgcv_fixed_sp_stable.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr std::size_t kStableBaseIntArraysPerTarget = 6U;
constexpr std::size_t kStableAggregateIntArraysPerTarget = 3U;
constexpr std::size_t kStableDoubleArraysPerTarget = 3U;
static_assert(sizeof(double) % sizeof(int) == 0U,
              "double diagnostics must fit an integral number of ints");
constexpr std::size_t kIntsPerDouble = sizeof(double) / sizeof(int);

void check_cuda(cudaError_t status, const char* stage) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(stage) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* stage) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(stage) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

void check_cusolver(cusolverStatus_t status, const char* stage) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(stage) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

struct ResourceLifecycleCounters {
  int acquire_attempt_count = 0;
  int acquire_success_count = 0;
  int acquire_failure_count = 0;
  int teardown_attempt_count = 0;
  int teardown_success_count = 0;
  int teardown_failure_count = 0;
  int active_count = 0;
  int ownership_indeterminate_count = 0;
};

struct FixedSpResourceCounters {
  ResourceLifecycleCounters cuda_device;
  ResourceLifecycleCounters cuda_host;
  ResourceLifecycleCounters stream;
  ResourceLifecycleCounters event;
  ResourceLifecycleCounters cublas_handle;
  ResourceLifecycleCounters cusolver_handle;
  ResourceLifecycleCounters gesvdj_info;
  int cleanup_error_count = 0;

  int allocation_count() const {
    return cuda_device.acquire_success_count +
      cuda_host.acquire_success_count;
  }

  int handle_create_count() const {
    return stream.acquire_success_count + event.acquire_success_count +
      cublas_handle.acquire_success_count +
      cusolver_handle.acquire_success_count +
      gesvdj_info.acquire_success_count;
  }
};

struct FixedSpResourceLedger {
  mutable std::mutex mutex;
  FixedSpResourceCounters counters;
};

struct AtomicResourceLifecycleCounters {
  std::atomic<std::int64_t> acquire_attempt_count{0};
  std::atomic<std::int64_t> acquire_success_count{0};
  std::atomic<std::int64_t> acquire_failure_count{0};
  std::atomic<std::int64_t> teardown_attempt_count{0};
  std::atomic<std::int64_t> teardown_success_count{0};
  std::atomic<std::int64_t> teardown_failure_count{0};
  std::atomic<std::int64_t> active_count{0};
  std::atomic<std::int64_t> ownership_indeterminate_count{0};
};

struct FixedSpGlobalResourceLedger {
  AtomicResourceLifecycleCounters cuda_device;
  AtomicResourceLifecycleCounters cuda_host;
  AtomicResourceLifecycleCounters stream;
  AtomicResourceLifecycleCounters event;
  AtomicResourceLifecycleCounters cublas_handle;
  AtomicResourceLifecycleCounters cusolver_handle;
  AtomicResourceLifecycleCounters gesvdj_info;
  std::atomic<std::int64_t> cleanup_error_count{0};
  std::atomic<int> inject_next_acquire_failure_kind{-1};
  std::atomic<int> inject_next_teardown_failure_kind{-1};
  std::atomic<int> inject_next_post_call_teardown_failure_kind{-1};
  std::atomic<bool> inject_next_blocked_consumer_launch_failure{false};
  std::atomic<bool> inject_next_prepared_static_shadow_body_failure{false};
};

struct FixedSpBatchedInfoTestState {
  std::mutex mutex;
  bool potrf_armed = false;
  std::vector<int> potrf_info;
  bool potrs_armed = false;
  int potrs_info = 0;
};

enum class FixedSpResourceKind {
  CudaDevice,
  CudaHost,
  Stream,
  Event,
  CublasHandle,
  CusolverHandle,
  GesvdjInfo
};

FixedSpGlobalResourceLedger& global_resource_ledger() {
  static FixedSpGlobalResourceLedger ledger;
  return ledger;
}

FixedSpBatchedInfoTestState& batched_info_test_state() {
  static FixedSpBatchedInfoTestState state;
  return state;
}

std::vector<int> consume_forced_potrf_info() {
  FixedSpBatchedInfoTestState& state = batched_info_test_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.potrf_armed) return std::vector<int>();
  state.potrf_armed = false;
  std::vector<int> info = std::move(state.potrf_info);
  state.potrf_info.clear();
  return info;
}

bool consume_forced_potrs_info(int* info) {
  FixedSpBatchedInfoTestState& state = batched_info_test_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.potrs_armed) return false;
  state.potrs_armed = false;
  *info = state.potrs_info;
  state.potrs_info = 0;
  return true;
}

ResourceLifecycleCounters& local_resource_counters(
    FixedSpResourceLedger* ledger,
    FixedSpResourceKind kind) {
  switch (kind) {
    case FixedSpResourceKind::CudaDevice: return ledger->counters.cuda_device;
    case FixedSpResourceKind::CudaHost: return ledger->counters.cuda_host;
    case FixedSpResourceKind::Stream: return ledger->counters.stream;
    case FixedSpResourceKind::Event: return ledger->counters.event;
    case FixedSpResourceKind::CublasHandle:
      return ledger->counters.cublas_handle;
    case FixedSpResourceKind::CusolverHandle:
      return ledger->counters.cusolver_handle;
    case FixedSpResourceKind::GesvdjInfo:
      return ledger->counters.gesvdj_info;
  }
  return ledger->counters.cuda_device;
}

AtomicResourceLifecycleCounters& global_resource_counters(
    FixedSpResourceKind kind) {
  FixedSpGlobalResourceLedger& ledger = global_resource_ledger();
  switch (kind) {
    case FixedSpResourceKind::CudaDevice: return ledger.cuda_device;
    case FixedSpResourceKind::CudaHost: return ledger.cuda_host;
    case FixedSpResourceKind::Stream: return ledger.stream;
    case FixedSpResourceKind::Event: return ledger.event;
    case FixedSpResourceKind::CublasHandle: return ledger.cublas_handle;
    case FixedSpResourceKind::CusolverHandle: return ledger.cusolver_handle;
    case FixedSpResourceKind::GesvdjInfo: return ledger.gesvdj_info;
  }
  return ledger.cuda_device;
}

void record_resource_acquire_success(FixedSpResourceLedger* ledger,
                                     FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    ResourceLifecycleCounters& local = local_resource_counters(ledger, kind);
    local.acquire_success_count += 1;
    local.active_count += 1;
  }
  AtomicResourceLifecycleCounters& global = global_resource_counters(kind);
  global.acquire_success_count.fetch_add(1, std::memory_order_relaxed);
  global.active_count.fetch_add(1, std::memory_order_relaxed);
}

void record_resource_acquire_attempt(FixedSpResourceLedger* ledger,
                                     FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    local_resource_counters(ledger, kind).acquire_attempt_count += 1;
  }
  global_resource_counters(kind).acquire_attempt_count.fetch_add(
    1, std::memory_order_relaxed);
}

void record_resource_acquire_failure(FixedSpResourceLedger* ledger,
                                     FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    local_resource_counters(ledger, kind).acquire_failure_count += 1;
  }
  global_resource_counters(kind).acquire_failure_count.fetch_add(
    1, std::memory_order_relaxed);
}

bool consume_injected_resource_acquire_failure(
    FixedSpResourceKind kind) noexcept {
  int expected = static_cast<int>(kind);
  return global_resource_ledger().inject_next_acquire_failure_kind
    .compare_exchange_strong(expected, -1, std::memory_order_acq_rel);
}

const char* fixed_sp_resource_kind_name(FixedSpResourceKind kind) noexcept {
  switch (kind) {
    case FixedSpResourceKind::CudaDevice: return "cuda_device";
    case FixedSpResourceKind::CudaHost: return "cuda_host";
    case FixedSpResourceKind::Stream: return "stream";
    case FixedSpResourceKind::Event: return "event";
    case FixedSpResourceKind::CublasHandle: return "cublas_handle";
    case FixedSpResourceKind::CusolverHandle: return "cusolver_handle";
    case FixedSpResourceKind::GesvdjInfo: return "gesvdj_info";
  }
  return "unknown";
}

FixedSpResourceKind fixed_sp_resource_kind_from_name(
    const std::string& resource) {
  if (resource == "cuda_device") return FixedSpResourceKind::CudaDevice;
  if (resource == "cuda_host") return FixedSpResourceKind::CudaHost;
  if (resource == "stream") return FixedSpResourceKind::Stream;
  if (resource == "event") return FixedSpResourceKind::Event;
  if (resource == "cublas_handle") return FixedSpResourceKind::CublasHandle;
  if (resource == "cusolver_handle") {
    return FixedSpResourceKind::CusolverHandle;
  }
  if (resource == "gesvdj_info") return FixedSpResourceKind::GesvdjInfo;
  throw std::runtime_error("unknown fixed-sp resource failure injection target");
}

void arm_injected_resource_teardown_failure(FixedSpResourceKind kind) {
  int expected = -1;
  if (!global_resource_ledger().inject_next_teardown_failure_kind
         .compare_exchange_strong(expected, static_cast<int>(kind),
                                  std::memory_order_acq_rel)) {
    throw std::runtime_error(
      "tracked fixed-sp resource teardown failure injection is already pending");
  }
}

bool consume_injected_resource_teardown_failure(
    FixedSpResourceKind kind) noexcept {
  int expected = static_cast<int>(kind);
  return global_resource_ledger().inject_next_teardown_failure_kind
    .compare_exchange_strong(expected, -1, std::memory_order_acq_rel);
}

void arm_injected_resource_post_call_teardown_failure(
    FixedSpResourceKind kind) {
  int expected = -1;
  if (!global_resource_ledger()
         .inject_next_post_call_teardown_failure_kind.compare_exchange_strong(
           expected, static_cast<int>(kind), std::memory_order_acq_rel)) {
    throw std::runtime_error(
      "tracked fixed-sp resource post-call teardown failure injection is "
      "already pending");
  }
}

bool consume_injected_resource_post_call_teardown_failure(
    FixedSpResourceKind kind) noexcept {
  int expected = static_cast<int>(kind);
  return global_resource_ledger()
    .inject_next_post_call_teardown_failure_kind.compare_exchange_strong(
      expected, -1, std::memory_order_acq_rel);
}

bool has_injected_resource_post_call_teardown_failure(
    FixedSpResourceKind kind) noexcept {
  return global_resource_ledger()
    .inject_next_post_call_teardown_failure_kind.load(
      std::memory_order_acquire) == static_cast<int>(kind);
}

bool consume_injected_prepared_static_shadow_body_failure() noexcept {
  bool expected = true;
  return global_resource_ledger()
    .inject_next_prepared_static_shadow_body_failure.compare_exchange_strong(
      expected, false, std::memory_order_acq_rel);
}

[[noreturn]] void throw_injected_resource_acquire_failure(
    FixedSpResourceKind kind) {
  throw std::runtime_error(
    std::string("injected tracked fixed-sp resource acquire failure: ") +
    fixed_sp_resource_kind_name(kind));
}

void record_resource_teardown_attempt(FixedSpResourceLedger* ledger,
                                      FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    local_resource_counters(ledger, kind).teardown_attempt_count += 1;
  }
  global_resource_counters(kind).teardown_attempt_count.fetch_add(
    1, std::memory_order_relaxed);
}

void record_resource_teardown_success(FixedSpResourceLedger* ledger,
                                      FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    ResourceLifecycleCounters& local = local_resource_counters(ledger, kind);
    local.teardown_success_count += 1;
    local.active_count -= 1;
  }
  AtomicResourceLifecycleCounters& global = global_resource_counters(kind);
  global.teardown_success_count.fetch_add(1, std::memory_order_relaxed);
  global.active_count.fetch_sub(1, std::memory_order_relaxed);
}

void record_cleanup_error(FixedSpResourceLedger* ledger) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    ledger->counters.cleanup_error_count += 1;
  }
  global_resource_ledger().cleanup_error_count.fetch_add(
    1, std::memory_order_relaxed);
}

void record_resource_teardown_failure(FixedSpResourceLedger* ledger,
                                       FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    local_resource_counters(ledger, kind).teardown_failure_count += 1;
  }
  global_resource_counters(kind).teardown_failure_count.fetch_add(
    1, std::memory_order_relaxed);
  record_cleanup_error(ledger);
}

void record_resource_teardown_indeterminate(
    FixedSpResourceLedger* ledger,
    FixedSpResourceKind kind) noexcept {
  {
    std::lock_guard<std::mutex> lock(ledger->mutex);
    ResourceLifecycleCounters& local = local_resource_counters(ledger, kind);
    local.teardown_failure_count += 1;
    local.active_count -= 1;
    local.ownership_indeterminate_count += 1;
  }
  AtomicResourceLifecycleCounters& global = global_resource_counters(kind);
  global.teardown_failure_count.fetch_add(1, std::memory_order_relaxed);
  global.active_count.fetch_sub(1, std::memory_order_relaxed);
  global.ownership_indeterminate_count.fetch_add(
    1, std::memory_order_relaxed);
  record_cleanup_error(ledger);
}

struct CleanupCudaDeviceStatus {
  cudaError_t get_device = cudaSuccess;
  cudaError_t select_device = cudaSuccess;
  cudaError_t restore_device = cudaSuccess;
  bool ready = false;
  bool restore_required = false;
  bool restore_attempted = false;
};

class ScopedCleanupCudaDevice {
 public:
  ScopedCleanupCudaDevice(std::int64_t creator_pid,
                          int device_id,
                          FixedSpResourceLedger* ledger) noexcept
      : creator_pid_(creator_pid), ledger_(ledger) {
    if (creator_pid_ != static_cast<std::int64_t>(getpid()) ||
        device_id < 0) {
      return;
    }
    status_.get_device = cudaGetDevice(&previous_device_);
    if (status_.get_device != cudaSuccess) {
      if (ledger_ != nullptr) record_cleanup_error(ledger_);
      return;
    }
    if (previous_device_ != device_id) {
      status_.select_device = cudaSetDevice(device_id);
      if (status_.select_device != cudaSuccess) {
        if (ledger_ != nullptr) record_cleanup_error(ledger_);
        return;
      }
      status_.restore_required = true;
    }
    status_.ready = true;
  }

  ~ScopedCleanupCudaDevice() { restore_noexcept(); }

  cudaError_t restore_noexcept() noexcept {
    if (status_.restore_attempted) return status_.restore_device;
    status_.restore_attempted = true;
    if (!status_.restore_required ||
        creator_pid_ != static_cast<std::int64_t>(getpid())) {
      return status_.restore_device;
    }
    status_.restore_device = cudaSetDevice(previous_device_);
    if (status_.restore_device != cudaSuccess && ledger_ != nullptr) {
      record_cleanup_error(ledger_);
    }
    return status_.restore_device;
  }

  ScopedCleanupCudaDevice(const ScopedCleanupCudaDevice&) = delete;
  ScopedCleanupCudaDevice& operator=(const ScopedCleanupCudaDevice&) = delete;

  bool ready() const noexcept { return status_.ready; }
  const CleanupCudaDeviceStatus& status() const noexcept { return status_; }

 private:
  std::int64_t creator_pid_ = -1;
  FixedSpResourceLedger* ledger_ = nullptr;
  int previous_device_ = -1;
  CleanupCudaDeviceStatus status_;
};

template <typename T>
void tracked_cuda_malloc(FixedSpResourceLedger* ledger,
                         T** pointer,
                         std::size_t bytes,
                         const char* stage) {
  record_resource_acquire_attempt(
    ledger, FixedSpResourceKind::CudaDevice);
  T* acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::CudaDevice);
  const cudaError_t status = injected ? cudaErrorMemoryAllocation :
    cudaMalloc(reinterpret_cast<void**>(&acquired), bytes);
  if (status != cudaSuccess) {
    record_resource_acquire_failure(
      ledger, FixedSpResourceKind::CudaDevice);
    if (injected) {
      throw_injected_resource_acquire_failure(
        FixedSpResourceKind::CudaDevice);
    }
    check_cuda(status, stage);
  }
  *pointer = acquired;
  record_resource_acquire_success(
    ledger, FixedSpResourceKind::CudaDevice);
}

template <typename T>
void tracked_cuda_malloc_host(FixedSpResourceLedger* ledger,
                              T** pointer,
                              std::size_t bytes,
                              const char* stage) {
  record_resource_acquire_attempt(ledger, FixedSpResourceKind::CudaHost);
  T* acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::CudaHost);
  const cudaError_t status = injected ? cudaErrorMemoryAllocation :
    cudaMallocHost(reinterpret_cast<void**>(&acquired), bytes);
  if (status != cudaSuccess) {
    record_resource_acquire_failure(ledger, FixedSpResourceKind::CudaHost);
    if (injected) {
      throw_injected_resource_acquire_failure(FixedSpResourceKind::CudaHost);
    }
    check_cuda(status, stage);
  }
  *pointer = acquired;
  record_resource_acquire_success(ledger, FixedSpResourceKind::CudaHost);
}

void tracked_cuda_stream_create(FixedSpResourceLedger* ledger,
                                cudaStream_t* stream,
                                unsigned int flags,
                                const char* stage) {
  record_resource_acquire_attempt(ledger, FixedSpResourceKind::Stream);
  cudaStream_t acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::Stream);
  const cudaError_t status = injected ? cudaErrorMemoryAllocation :
    cudaStreamCreateWithFlags(&acquired, flags);
  if (status != cudaSuccess) {
    record_resource_acquire_failure(ledger, FixedSpResourceKind::Stream);
    if (injected) {
      throw_injected_resource_acquire_failure(FixedSpResourceKind::Stream);
    }
    check_cuda(status, stage);
  }
  *stream = acquired;
  record_resource_acquire_success(ledger, FixedSpResourceKind::Stream);
}

void tracked_cuda_event_create(FixedSpResourceLedger* ledger,
                               cudaEvent_t* event,
                               unsigned int flags,
                               const char* stage) {
  record_resource_acquire_attempt(ledger, FixedSpResourceKind::Event);
  cudaEvent_t acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::Event);
  const cudaError_t status = injected ? cudaErrorMemoryAllocation :
    cudaEventCreateWithFlags(&acquired, flags);
  if (status != cudaSuccess) {
    record_resource_acquire_failure(ledger, FixedSpResourceKind::Event);
    if (injected) {
      throw_injected_resource_acquire_failure(FixedSpResourceKind::Event);
    }
    check_cuda(status, stage);
  }
  *event = acquired;
  record_resource_acquire_success(ledger, FixedSpResourceKind::Event);
}

void tracked_cublas_create(FixedSpResourceLedger* ledger,
                           cublasHandle_t* handle,
                           const char* stage) {
  record_resource_acquire_attempt(
    ledger, FixedSpResourceKind::CublasHandle);
  cublasHandle_t acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::CublasHandle);
  const cublasStatus_t status = injected ? CUBLAS_STATUS_ALLOC_FAILED :
    cublasCreate(&acquired);
  if (status != CUBLAS_STATUS_SUCCESS) {
    record_resource_acquire_failure(
      ledger, FixedSpResourceKind::CublasHandle);
    if (injected) {
      throw_injected_resource_acquire_failure(
        FixedSpResourceKind::CublasHandle);
    }
    check_cublas(status, stage);
  }
  *handle = acquired;
  record_resource_acquire_success(
    ledger, FixedSpResourceKind::CublasHandle);
}

void tracked_cusolver_create(FixedSpResourceLedger* ledger,
                             cusolverDnHandle_t* handle,
                             const char* stage) {
  record_resource_acquire_attempt(
    ledger, FixedSpResourceKind::CusolverHandle);
  cusolverDnHandle_t acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::CusolverHandle);
  const cusolverStatus_t status = injected ? CUSOLVER_STATUS_ALLOC_FAILED :
    cusolverDnCreate(&acquired);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    record_resource_acquire_failure(
      ledger, FixedSpResourceKind::CusolverHandle);
    if (injected) {
      throw_injected_resource_acquire_failure(
        FixedSpResourceKind::CusolverHandle);
    }
    check_cusolver(status, stage);
  }
  *handle = acquired;
  record_resource_acquire_success(
    ledger, FixedSpResourceKind::CusolverHandle);
}

void tracked_gesvdj_info_create(FixedSpResourceLedger* ledger,
                                gesvdjInfo_t* info,
                                const char* stage) {
  record_resource_acquire_attempt(ledger, FixedSpResourceKind::GesvdjInfo);
  gesvdjInfo_t acquired = nullptr;
  const bool injected = consume_injected_resource_acquire_failure(
    FixedSpResourceKind::GesvdjInfo);
  const cusolverStatus_t status = injected ? CUSOLVER_STATUS_ALLOC_FAILED :
    cusolverDnCreateGesvdjInfo(&acquired);
  if (status != CUSOLVER_STATUS_SUCCESS) {
    record_resource_acquire_failure(
      ledger, FixedSpResourceKind::GesvdjInfo);
    if (injected) {
      throw_injected_resource_acquire_failure(
        FixedSpResourceKind::GesvdjInfo);
    }
    check_cusolver(status, stage);
  }
  *info = acquired;
  record_resource_acquire_success(ledger, FixedSpResourceKind::GesvdjInfo);
}

enum class TrackedTeardownDisposition {
  Noop,
  NotAttemptedRetryable,
  CalledSuccess,
  CalledOwnershipIndeterminate
};

template <typename Status>
struct TrackedTeardownResult {
  Status status;
  TrackedTeardownDisposition disposition;
};

struct OwnerTeardownStatus {
  bool not_attempted_retryable = false;
  bool ownership_indeterminate = false;

  template <typename Status>
  void observe(const TrackedTeardownResult<Status>& result) noexcept {
    if (result.disposition ==
        TrackedTeardownDisposition::NotAttemptedRetryable) {
      not_attempted_retryable = true;
    } else if (result.disposition ==
               TrackedTeardownDisposition::CalledOwnershipIndeterminate) {
      ownership_indeterminate = true;
    }
  }

  void merge(const OwnerTeardownStatus& other) noexcept {
    not_attempted_retryable =
      not_attempted_retryable || other.not_attempted_retryable;
    ownership_indeterminate =
      ownership_indeterminate || other.ownership_indeterminate;
  }
};

template <typename T>
TrackedTeardownResult<cudaError_t> tracked_cuda_free_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) {
    return {cudaSuccess, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CudaDevice);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::CudaDevice)) {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CudaDevice);
    return {cudaErrorUnknown,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  T* owned = std::exchange(*pointer, nullptr);
  cudaError_t status = cudaFree(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::CudaDevice) && status == cudaSuccess) {
    status = cudaErrorUnknown;
  }
  if (status == cudaSuccess) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CudaDevice);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::CudaDevice);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

template <typename T>
void tracked_cuda_free(FixedSpResourceLedger* ledger,
                       T** pointer,
                       const char* stage) {
  const TrackedTeardownResult<cudaError_t> result =
    tracked_cuda_free_noexcept(ledger, pointer);
  if (result.disposition ==
      TrackedTeardownDisposition::NotAttemptedRetryable) {
    throw std::runtime_error(
      "injected tracked CUDA device free failure");
  }
  check_cuda(result.status, stage);
}

template <typename T>
TrackedTeardownResult<cudaError_t> tracked_cuda_reserve_probe_free_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) {
    return {cudaSuccess, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CudaDevice);
  T* owned = std::exchange(*pointer, nullptr);
  const cudaError_t status = cudaFree(owned);
  if (status == cudaSuccess) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CudaDevice);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::CudaDevice);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

template <typename T>
void tracked_cuda_reserve_probe_free(FixedSpResourceLedger* ledger,
                                     T** pointer,
                                     const char* stage) {
  check_cuda(
    tracked_cuda_reserve_probe_free_noexcept(ledger, pointer).status, stage);
}

template <typename T>
TrackedTeardownResult<cudaError_t> tracked_cuda_free_host_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) {
    return {cudaSuccess, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::CudaHost);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::CudaHost)) {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::CudaHost);
    return {cudaErrorUnknown,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  T* owned = std::exchange(*pointer, nullptr);
  cudaError_t status = cudaFreeHost(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::CudaHost) && status == cudaSuccess) {
    status = cudaErrorUnknown;
  }
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::CudaHost);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::CudaHost);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

template <typename T>
void tracked_cuda_free_host(FixedSpResourceLedger* ledger,
                            T** pointer,
                            const char* stage) {
  check_cuda(tracked_cuda_free_host_noexcept(ledger, pointer).status, stage);
}

TrackedTeardownResult<cudaError_t> tracked_cuda_stream_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cudaStream_t* stream) noexcept {
  if (stream == nullptr || *stream == nullptr) {
    return {cudaSuccess, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::Stream);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::Stream)) {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::Stream);
    return {cudaErrorUnknown,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  cudaStream_t owned = std::exchange(*stream, nullptr);
  cudaError_t status = cudaStreamDestroy(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::Stream) && status == cudaSuccess) {
    status = cudaErrorUnknown;
  }
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::Stream);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(ledger, FixedSpResourceKind::Stream);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

TrackedTeardownResult<cudaError_t> tracked_cuda_event_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cudaEvent_t* event) noexcept {
  if (event == nullptr || *event == nullptr) {
    return {cudaSuccess, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::Event);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::Event)) {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::Event);
    return {cudaErrorUnknown,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  cudaEvent_t owned = std::exchange(*event, nullptr);
  cudaError_t status = cudaEventDestroy(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::Event) && status == cudaSuccess) {
    status = cudaErrorUnknown;
  }
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::Event);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(ledger, FixedSpResourceKind::Event);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

void tracked_cuda_event_destroy(FixedSpResourceLedger* ledger,
                                cudaEvent_t* event,
                                const char* stage) {
  check_cuda(tracked_cuda_event_destroy_noexcept(ledger, event).status, stage);
}

void cleanup_local_cuda_event_noexcept(FixedSpResourceLedger* ledger,
                                       cudaEvent_t* event) noexcept {
  const TrackedTeardownResult<cudaError_t> first =
    tracked_cuda_event_destroy_noexcept(ledger, event);
  if (first.disposition ==
      TrackedTeardownDisposition::NotAttemptedRetryable) {
    tracked_cuda_event_destroy_noexcept(ledger, event);
  }
}

TrackedTeardownResult<cublasStatus_t> tracked_cublas_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cublasHandle_t* handle) noexcept {
  if (handle == nullptr || *handle == nullptr) {
    return {CUBLAS_STATUS_SUCCESS, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CublasHandle);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::CublasHandle)) {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CublasHandle);
    return {CUBLAS_STATUS_INTERNAL_ERROR,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  cublasHandle_t owned = std::exchange(*handle, nullptr);
  cublasStatus_t status = cublasDestroy(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::CublasHandle) &&
      status == CUBLAS_STATUS_SUCCESS) {
    status = CUBLAS_STATUS_INTERNAL_ERROR;
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CublasHandle);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::CublasHandle);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

TrackedTeardownResult<cusolverStatus_t> tracked_cusolver_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cusolverDnHandle_t* handle) noexcept {
  if (handle == nullptr || *handle == nullptr) {
    return {CUSOLVER_STATUS_SUCCESS, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CusolverHandle);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::CusolverHandle)) {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CusolverHandle);
    return {CUSOLVER_STATUS_INTERNAL_ERROR,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  cusolverDnHandle_t owned = std::exchange(*handle, nullptr);
  cusolverStatus_t status = cusolverDnDestroy(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::CusolverHandle) &&
      status == CUSOLVER_STATUS_SUCCESS) {
    status = CUSOLVER_STATUS_INTERNAL_ERROR;
  }
  if (status == CUSOLVER_STATUS_SUCCESS) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CusolverHandle);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::CusolverHandle);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

TrackedTeardownResult<cusolverStatus_t> tracked_gesvdj_info_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    gesvdjInfo_t* info) noexcept {
  if (info == nullptr || *info == nullptr) {
    return {CUSOLVER_STATUS_SUCCESS, TrackedTeardownDisposition::Noop};
  }
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::GesvdjInfo);
  if (consume_injected_resource_teardown_failure(
        FixedSpResourceKind::GesvdjInfo)) {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::GesvdjInfo);
    return {CUSOLVER_STATUS_INTERNAL_ERROR,
            TrackedTeardownDisposition::NotAttemptedRetryable};
  }
  gesvdjInfo_t owned = std::exchange(*info, nullptr);
  cusolverStatus_t status = cusolverDnDestroyGesvdjInfo(owned);
  if (consume_injected_resource_post_call_teardown_failure(
        FixedSpResourceKind::GesvdjInfo) &&
      status == CUSOLVER_STATUS_SUCCESS) {
    status = CUSOLVER_STATUS_INTERNAL_ERROR;
  }
  if (status == CUSOLVER_STATUS_SUCCESS) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::GesvdjInfo);
    return {status, TrackedTeardownDisposition::CalledSuccess};
  }
  record_resource_teardown_indeterminate(
    ledger, FixedSpResourceKind::GesvdjInfo);
  return {status,
          TrackedTeardownDisposition::CalledOwnershipIndeterminate};
}

template <typename T>
void cleanup_local_cuda_allocation_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  const TrackedTeardownResult<cudaError_t> first =
    tracked_cuda_free_noexcept(ledger, pointer);
  if (first.disposition ==
      TrackedTeardownDisposition::NotAttemptedRetryable) {
    tracked_cuda_free_noexcept(ledger, pointer);
  }
}

template <typename T>
void cleanup_local_cuda_host_allocation_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  const TrackedTeardownResult<cudaError_t> first =
    tracked_cuda_free_host_noexcept(ledger, pointer);
  if (first.disposition ==
      TrackedTeardownDisposition::NotAttemptedRetryable) {
    tracked_cuda_free_host_noexcept(ledger, pointer);
  }
}

void copy_resource_counters(const FixedSpResourceCounters& counters,
                            FixedSpRuntimeInfo* info) {
  info->cuda_device_allocation_count =
    counters.cuda_device.acquire_success_count;
  info->cuda_host_allocation_count =
    counters.cuda_host.acquire_success_count;
  info->stream_create_count = counters.stream.acquire_success_count;
  info->event_create_count = counters.event.acquire_success_count;
  info->cublas_handle_create_count =
    counters.cublas_handle.acquire_success_count;
  info->cusolver_handle_create_count =
    counters.cusolver_handle.acquire_success_count;
  info->gesvdj_info_create_count =
    counters.gesvdj_info.acquire_success_count;
  info->gesvdj_info_destroy_count =
    counters.gesvdj_info.teardown_success_count;
}

FixedSpResourceCounters resource_counters_snapshot(
    const std::shared_ptr<FixedSpResourceLedger>& ledger) {
  std::lock_guard<std::mutex> lock(ledger->mutex);
  return ledger->counters;
}

void copy_solve_resource_deltas(
    const FixedSpResourceCounters& before,
    const FixedSpResourceCounters& after,
    DeviceResidualInfo* info) {
  info->resource_allocation_count_before_solve = before.allocation_count();
  info->resource_allocation_count_after_solve = after.allocation_count();
  info->resource_handle_create_count_before_solve =
    before.handle_create_count();
  info->resource_handle_create_count_after_solve =
    after.handle_create_count();
  info->cuda_device_allocation_count_during_solve =
    after.cuda_device.acquire_success_count -
    before.cuda_device.acquire_success_count;
  info->cuda_host_allocation_count_during_solve =
    after.cuda_host.acquire_success_count -
    before.cuda_host.acquire_success_count;
  info->stream_create_count_during_solve =
    after.stream.acquire_success_count - before.stream.acquire_success_count;
  info->event_create_count_during_solve =
    after.event.acquire_success_count - before.event.acquire_success_count;
  info->cublas_handle_create_count_during_solve =
    after.cublas_handle.acquire_success_count -
    before.cublas_handle.acquire_success_count;
  info->cusolver_handle_create_count_during_solve =
    after.cusolver_handle.acquire_success_count -
    before.cusolver_handle.acquire_success_count;
  info->per_target_allocation_count_after_warmup =
    info->resource_allocation_count_after_solve -
    info->resource_allocation_count_before_solve;
  info->per_target_handle_create_count =
    info->resource_handle_create_count_after_solve -
    info->resource_handle_create_count_before_solve;
  info->resource_instrumentation_version = 1;
  info->resource_snapshot_captured = true;
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        const char* name) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::runtime_error(std::string(name) + " size overflow");
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             const char* name) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(name) + " size overflow");
  }
  return left * right;
}

std::size_t allocation_bytes(std::size_t count,
                             std::size_t element_size,
                             const char* name) {
  return checked_multiply(count, element_size, name);
}

std::size_t pointer_alignment(const void* pointer) {
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  if (address == 0) return 0;
  return static_cast<std::size_t>(address & (~address + 1U));
}

void validate_capacities(const FixedSpCapacities& capacities) {
  if (capacities.n <= 0 || capacities.null_dim <= 0 ||
      capacities.target_count <= 0 || capacities.penalty_count <= 0 ||
      capacities.augmented_rows <= 0) {
    throw std::runtime_error("fixed-sp CUDA runtime capacities must be positive");
  }
}

std::size_t positive_size(int value, const char* name) {
  if (value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

std::size_t matrix_count(int rows, int columns, const char* name) {
  return checked_multiply(
    positive_size(rows, name), positive_size(columns, name), name);
}

void require_scale_aware_symmetric(const double* matrix,
                                   int dimension,
                                   const char* error_message) {
  const std::size_t count = matrix_count(
    dimension, dimension, "prepared symmetric matrix");
  double scale = 1.0;
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(matrix[index])) return;
    scale = std::max(scale, std::abs(matrix[index]));
  }

  // Dimension-scaled roundoff with an absolute floor near zero.
  const double tolerance = static_cast<double>(dimension) *
    std::numeric_limits<double>::epsilon() * scale;
  const std::size_t leading_dimension =
    static_cast<std::size_t>(dimension);
  for (int column = 0; column < dimension; ++column) {
    for (int row = column + 1; row < dimension; ++row) {
      const double lower = matrix[static_cast<std::size_t>(row) +
        leading_dimension * static_cast<std::size_t>(column)];
      const double upper = matrix[static_cast<std::size_t>(column) +
        leading_dimension * static_cast<std::size_t>(row)];
      if (std::abs(lower - upper) > tolerance) {
        throw std::runtime_error(error_message);
      }
    }
  }
}

void mirror_lower_triangle(double* matrix, int dimension) {
  const std::size_t leading_dimension =
    static_cast<std::size_t>(dimension);
  for (int column = 0; column < dimension; ++column) {
    for (int row = column + 1; row < dimension; ++row) {
      matrix[static_cast<std::size_t>(column) +
             leading_dimension * static_cast<std::size_t>(row)] =
        matrix[static_cast<std::size_t>(row) +
               leading_dimension * static_cast<std::size_t>(column)];
    }
  }
}

void validate_prepared_host_view(const PreparedSHostView& setup) {
  const std::size_t penalties =
    positive_size(setup.penalty_count, "prepared penalty count");
  positive_size(setup.n, "prepared row count");
  positive_size(setup.coefficient_dim, "prepared coefficient dimension");
  positive_size(setup.null_dim, "prepared null dimension");
  if (setup.X == nullptr || setup.gram == nullptr) {
    throw std::runtime_error("prepared setup matrix pointers are null");
  }
  if (setup.Z == nullptr && setup.null_dim != setup.coefficient_dim) {
    throw std::runtime_error(
      "identity prepared setup must preserve coefficient dimension");
  }
  if (setup.Z != nullptr && setup.null_dim > setup.coefficient_dim) {
    throw std::runtime_error(
      "prepared null dimension exceeds coefficient dimension");
  }
  if (setup.penalty_blocks.size() != penalties ||
      setup.penalty_dimensions.size() != penalties ||
      setup.penalty_offsets_zero_based.size() != penalties ||
      setup.penalty_ranks.size() != penalties ||
      setup.penalty_sp_indices_zero_based.size() != penalties) {
    throw std::runtime_error("prepared penalty metadata size mismatch");
  }
  for (std::size_t index = 0; index < penalties; ++index) {
    const int dimension = setup.penalty_dimensions[index];
    const int offset = setup.penalty_offsets_zero_based[index];
    if (setup.penalty_blocks[index] == nullptr || dimension <= 0 ||
        offset < 0 || dimension > setup.coefficient_dim ||
        offset > setup.coefficient_dim - dimension) {
      throw std::runtime_error("prepared penalty offset is out of range");
    }
    if (setup.penalty_ranks[index] < 0 ||
        setup.penalty_ranks[index] > dimension) {
      throw std::runtime_error("prepared penalty rank is out of range");
    }
    if (setup.penalty_sp_indices_zero_based[index] !=
        static_cast<int>(index)) {
      throw std::runtime_error(
        "prepared penalty-to-SP mapping must be identity");
    }
    require_scale_aware_symmetric(
      setup.penalty_blocks[index], dimension,
      "prepared smooth penalty block must be symmetric");
  }
  if (setup.H != nullptr) {
    require_scale_aware_symmetric(
      setup.H, setup.coefficient_dim, "prepared H must be symmetric");
  }
}

std::vector<double> build_nullspace_design(
    const PreparedSHostView& setup) {
  if (setup.Z == nullptr) return {};
  const std::size_t count =
    matrix_count(setup.n, setup.null_dim, "nullspace design");
  std::vector<double> result(count, 0.0);
  for (int column = 0; column < setup.null_dim; ++column) {
    for (int inner = 0; inner < setup.coefficient_dim; ++inner) {
      const double z = setup.Z[
        static_cast<std::size_t>(inner) +
        static_cast<std::size_t>(setup.coefficient_dim) * column];
      for (int row = 0; row < setup.n; ++row) {
        result[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(setup.n) * column] +=
          setup.X[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(setup.n) * inner] * z;
      }
    }
  }
  return result;
}

std::vector<double> build_projected_penalties(
    const PreparedSHostView& setup) {
  const std::size_t p_squared = matrix_count(
    setup.coefficient_dim, setup.coefficient_dim,
    "full coefficient penalty");
  const std::size_t q_squared =
    matrix_count(setup.null_dim, setup.null_dim, "projected penalty");
  const std::size_t projected_count = checked_multiply(
    positive_size(setup.penalty_count, "prepared penalty count"),
    q_squared, "projected penalties");
  std::vector<double> projected(projected_count, 0.0);
  std::vector<double> full(p_squared, 0.0);
  std::vector<double> penalty_times_z;
  if (setup.Z != nullptr) {
    penalty_times_z.assign(matrix_count(
      setup.coefficient_dim, setup.null_dim, "penalty times nullspace"),
      0.0);
  }

  for (int penalty = 0; penalty < setup.penalty_count; ++penalty) {
    std::fill(full.begin(), full.end(), 0.0);
    const int dimension = setup.penalty_dimensions[penalty];
    const int offset = setup.penalty_offsets_zero_based[penalty];
    const double* block = setup.penalty_blocks[penalty];
    for (int column = 0; column < dimension; ++column) {
      for (int row = 0; row < dimension; ++row) {
        full[static_cast<std::size_t>(offset + row) +
             static_cast<std::size_t>(setup.coefficient_dim) *
               (offset + column)] =
          block[static_cast<std::size_t>(row) +
                static_cast<std::size_t>(dimension) * column];
      }
    }

    double* destination = projected.data() +
      static_cast<std::size_t>(penalty) * q_squared;
    if (setup.Z == nullptr) {
      std::copy(full.begin(), full.end(), destination);
      mirror_lower_triangle(destination, setup.null_dim);
      continue;
    }

    std::fill(penalty_times_z.begin(), penalty_times_z.end(), 0.0);
    for (int column = 0; column < setup.null_dim; ++column) {
      for (int inner = 0; inner < setup.coefficient_dim; ++inner) {
        const double z = setup.Z[
          static_cast<std::size_t>(inner) +
          static_cast<std::size_t>(setup.coefficient_dim) * column];
        for (int row = 0; row < setup.coefficient_dim; ++row) {
          penalty_times_z[static_cast<std::size_t>(row) +
            static_cast<std::size_t>(setup.coefficient_dim) * column] +=
            full[static_cast<std::size_t>(row) +
                 static_cast<std::size_t>(setup.coefficient_dim) * inner] * z;
        }
      }
    }
    for (int column = 0; column < setup.null_dim; ++column) {
      for (int row = 0; row < setup.null_dim; ++row) {
        double value = 0.0;
        for (int inner = 0; inner < setup.coefficient_dim; ++inner) {
          value += setup.Z[static_cast<std::size_t>(inner) +
                           static_cast<std::size_t>(setup.coefficient_dim) *
                             row] *
            penalty_times_z[static_cast<std::size_t>(inner) +
              static_cast<std::size_t>(setup.coefficient_dim) * column];
        }
        destination[static_cast<std::size_t>(row) +
                    static_cast<std::size_t>(setup.null_dim) * column] = value;
      }
    }
    mirror_lower_triangle(destination, setup.null_dim);
  }
  return projected;
}

std::vector<double> build_projected_H(const PreparedSHostView& setup) {
  if (setup.H == nullptr) return {};
  const std::size_t p_squared = matrix_count(
    setup.coefficient_dim, setup.coefficient_dim,
    "full coefficient H");
  const std::size_t q_squared =
    matrix_count(setup.null_dim, setup.null_dim, "projected H");
  if (setup.Z == nullptr) {
    std::vector<double> projected(setup.H, setup.H + p_squared);
    mirror_lower_triangle(projected.data(), setup.null_dim);
    return projected;
  }

  const std::size_t p_q = matrix_count(
    setup.coefficient_dim, setup.null_dim, "H times nullspace");
  std::vector<double> H_times_z(p_q, 0.0);
  std::vector<double> projected(q_squared, 0.0);
  for (int column = 0; column < setup.null_dim; ++column) {
    for (int inner = 0; inner < setup.coefficient_dim; ++inner) {
      const double z = setup.Z[
        static_cast<std::size_t>(inner) +
        static_cast<std::size_t>(setup.coefficient_dim) * column];
      for (int row = 0; row < setup.coefficient_dim; ++row) {
        H_times_z[static_cast<std::size_t>(row) +
          static_cast<std::size_t>(setup.coefficient_dim) * column] +=
          setup.H[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(setup.coefficient_dim) * inner] * z;
      }
    }
  }
  for (int column = 0; column < setup.null_dim; ++column) {
    for (int row = 0; row < setup.null_dim; ++row) {
      double value = 0.0;
      for (int inner = 0; inner < setup.coefficient_dim; ++inner) {
        value += setup.Z[static_cast<std::size_t>(inner) +
                         static_cast<std::size_t>(setup.coefficient_dim) *
                           row] *
          H_times_z[static_cast<std::size_t>(inner) +
            static_cast<std::size_t>(setup.coefficient_dim) * column];
      }
      projected[static_cast<std::size_t>(row) +
                static_cast<std::size_t>(setup.null_dim) * column] = value;
    }
  }
  mirror_lower_triangle(projected.data(), setup.null_dim);
  return projected;
}

void require_finite_derived(const std::vector<double>& values,
                            const char* name) {
  if (!std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value);
      })) {
    throw std::runtime_error(
      std::string("derived prepared ") + name + " must be finite");
  }
}

__global__ void initialize_fixed_sp_outputs_invalid(
    double* coefficients,
    std::size_t coefficient_count,
    double* fitted,
    std::size_t fitted_count,
    double* residuals,
    std::size_t residual_count,
    double* rss,
    std::size_t rss_count,
    double* rhs,
    std::size_t rhs_count) {
  const std::size_t index =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const double invalid = __longlong_as_double(0x7ff8000000000000ULL);
  if (index < coefficient_count) coefficients[index] = invalid;
  if (index < fitted_count) fitted[index] = invalid;
  if (index < residual_count) residuals[index] = invalid;
  if (index < rss_count) rss[index] = invalid;
  if (index < rhs_count) rhs[index] = invalid;
}

__global__ void build_fixed_sp_systems_kernel(
    const double* gram,
    const double* projected_H,
    const double* projected_penalties,
    const double* sp,
    const int* safe_target_indices,
    int safe_count,
    int penalty_count,
    int q,
    double* systems) {
  const int safe_ordinal = static_cast<int>(blockIdx.z);
  const int row = static_cast<int>(blockIdx.x) *
      static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int column = static_cast<int>(blockIdx.y) *
      static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
  if (safe_ordinal >= safe_count || row >= q || column >= q) return;

  const int target = safe_target_indices[safe_ordinal];
  const std::size_t q_squared =
    static_cast<std::size_t>(q) * static_cast<std::size_t>(q);
  const std::size_t element = static_cast<std::size_t>(row) +
    static_cast<std::size_t>(column) * static_cast<std::size_t>(q);
  double value = gram[element] +
    (projected_H == nullptr ? 0.0 : projected_H[element]);
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    value += sp[static_cast<std::size_t>(penalty) +
                static_cast<std::size_t>(penalty_count) *
                  static_cast<std::size_t>(target)] *
      projected_penalties[element + q_squared *
        static_cast<std::size_t>(penalty)];
  }
  systems[element + q_squared * static_cast<std::size_t>(safe_ordinal)] =
    value;
}

__global__ void gather_fixed_sp_safe_rhs_kernel(
    const double* canonical_rhs,
    const int* safe_target_indices,
    int q,
    int safe_count,
    double* safe_theta) {
  const std::size_t index =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(q) *
    static_cast<std::size_t>(safe_count);
  if (index >= count) return;

  const std::size_t safe_ordinal = index / static_cast<std::size_t>(q);
  const std::size_t row = index - safe_ordinal * static_cast<std::size_t>(q);
  const int target = safe_target_indices[safe_ordinal];
  safe_theta[index] = canonical_rhs[
    row + static_cast<std::size_t>(q) * static_cast<std::size_t>(target)];
}

__global__ void make_fixed_sp_pointer_arrays(
    double* systems,
    double* theta,
    int q,
    int count,
    double** system_ptrs,
    double** theta_ptrs) {
  const int index = static_cast<int>(blockIdx.x) *
      static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  if (index >= count) return;
  const std::size_t ordinal = static_cast<std::size_t>(index);
  const std::size_t q_size = static_cast<std::size_t>(q);
  system_ptrs[index] = systems + ordinal * q_size * q_size;
  theta_ptrs[index] = theta + ordinal * q_size;
}

__global__ void compact_fixed_sp_success_pointer_arrays(
    const int* potrf_info,
    int count,
    double** system_ptrs,
    double** theta_ptrs) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) return;
  int output = 0;
  for (int index = 0; index < count; ++index) {
    if (potrf_info[index] == 0) {
      system_ptrs[output] = system_ptrs[index];
      theta_ptrs[output] = theta_ptrs[index];
      output += 1;
    }
  }
}

__global__ void finalize_fixed_sp_coefficients_batch(
    const double* theta,
    const double* Z,
    const int* safe_target_indices,
    const int* potrf_info,
    int coefficient_dim,
    int null_dim,
    int safe_count,
    double* coefficients) {
  const int coefficient = static_cast<int>(blockIdx.x) *
      static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int safe_ordinal = static_cast<int>(blockIdx.y) *
      static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
  if (coefficient >= coefficient_dim || safe_ordinal >= safe_count ||
      potrf_info[safe_ordinal] != 0) {
    return;
  }

  double value = 0.0;
  if (Z == nullptr) {
    value = theta[static_cast<std::size_t>(coefficient) +
      static_cast<std::size_t>(null_dim) *
        static_cast<std::size_t>(safe_ordinal)];
  } else {
    for (int inner = 0; inner < null_dim; ++inner) {
      value += Z[static_cast<std::size_t>(coefficient) +
        static_cast<std::size_t>(coefficient_dim) *
          static_cast<std::size_t>(inner)] *
        theta[static_cast<std::size_t>(inner) +
          static_cast<std::size_t>(null_dim) *
            static_cast<std::size_t>(safe_ordinal)];
    }
  }
  const int target = safe_target_indices[safe_ordinal];
  coefficients[static_cast<std::size_t>(coefficient) +
    static_cast<std::size_t>(coefficient_dim) *
      static_cast<std::size_t>(target)] = value;
}

__global__ void finalize_fixed_sp_fitted_batch(
    const double* X_null,
    const double* theta,
    const int* safe_target_indices,
    const int* potrf_info,
    int n,
    int null_dim,
    int safe_count,
    double* fitted) {
  const int row = static_cast<int>(blockIdx.x) *
      static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int safe_ordinal = static_cast<int>(blockIdx.y) *
      static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
  if (row >= n || safe_ordinal >= safe_count ||
      potrf_info[safe_ordinal] != 0) {
    return;
  }

  double value = 0.0;
  for (int inner = 0; inner < null_dim; ++inner) {
    value += X_null[static_cast<std::size_t>(row) +
      static_cast<std::size_t>(n) * static_cast<std::size_t>(inner)] *
      theta[static_cast<std::size_t>(inner) +
        static_cast<std::size_t>(null_dim) *
          static_cast<std::size_t>(safe_ordinal)];
  }
  const int target = safe_target_indices[safe_ordinal];
  fitted[static_cast<std::size_t>(row) +
    static_cast<std::size_t>(n) * static_cast<std::size_t>(target)] = value;
}

__global__ void finalize_fixed_sp_residual_rss_batch(
    const double* y,
    const double* fitted,
    const int* safe_target_indices,
    const int* potrf_info,
    int n,
    int safe_count,
    double* residuals,
    double* rss,
    bool write_residuals,
    bool write_rss) {
  const int safe_ordinal = static_cast<int>(blockIdx.x);
  if (safe_ordinal >= safe_count || potrf_info[safe_ordinal] != 0) return;

  const int target = safe_target_indices[safe_ordinal];
  const std::size_t target_offset = static_cast<std::size_t>(target);
  const double* target_y = y + static_cast<std::size_t>(n) * target_offset;
  const double* target_fitted =
    fitted + static_cast<std::size_t>(n) * target_offset;
  double* target_residuals =
    residuals + static_cast<std::size_t>(n) * target_offset;
  extern __shared__ double sums[];
  double local_sum = 0.0;
  for (int row = static_cast<int>(threadIdx.x);
       row < n;
       row += static_cast<int>(blockDim.x)) {
    const double residual = target_y[row] - target_fitted[row];
    if (write_residuals) target_residuals[row] = residual;
    local_sum += residual * residual;
  }
  sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U && write_rss) rss[target_offset] = sums[0];
}

__global__ void finalize_fixed_sp_qr_coefficients(
    const double* theta,
    const double* Z,
    int target,
    int coefficient_dim,
    int null_dim,
    double* coefficients) {
  const std::size_t coefficient =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coefficient_count =
    static_cast<std::size_t>(coefficient_dim);
  if (coefficient >= coefficient_count) return;

  double value = 0.0;
  if (Z == nullptr) {
    value = theta[coefficient];
  } else {
    for (std::size_t inner = 0U;
         inner < static_cast<std::size_t>(null_dim); ++inner) {
      value += Z[coefficient + coefficient_count * inner] * theta[inner];
    }
  }
  coefficients[coefficient + coefficient_count *
    static_cast<std::size_t>(target)] = value;
}

__global__ void finalize_fixed_sp_qr_fitted(
    const double* X_null,
    const double* theta,
    int target,
    int n,
    int null_dim,
    double* fitted) {
  const std::size_t row =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row_count = static_cast<std::size_t>(n);
  if (row >= row_count) return;

  double value = 0.0;
  for (std::size_t inner = 0U;
       inner < static_cast<std::size_t>(null_dim); ++inner) {
    value += X_null[row + row_count * inner] * theta[inner];
  }
  fitted[row + row_count * static_cast<std::size_t>(target)] = value;
}

__global__ void finalize_fixed_sp_qr_residual_rss(
    const double* y,
    const double* fitted,
    int target,
    int n,
    double* residuals,
    double* rss,
    bool write_residuals,
    bool write_rss) {
  const std::size_t target_offset = static_cast<std::size_t>(target);
  const std::size_t row_count = static_cast<std::size_t>(n);
  const double* target_y = y + row_count * target_offset;
  const double* target_fitted = fitted + row_count * target_offset;
  double* target_residuals = residuals + row_count * target_offset;
  extern __shared__ double sums[];
  double local_sum = 0.0;
  for (std::size_t row = static_cast<std::size_t>(threadIdx.x);
       row < row_count; row += static_cast<std::size_t>(blockDim.x)) {
    const double residual = target_y[row] - target_fitted[row];
    if (write_residuals) target_residuals[row] = residual;
    local_sum += residual * residual;
  }
  sums[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) {
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0U && write_rss) rss[target_offset] = sums[0];
}

__global__ void merge_fixed_sp_qr_finite_status(
    const int* qr_finite_status,
    int target_count,
    int* finite_status) {
  const std::size_t target =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (target >= static_cast<std::size_t>(target_count)) return;
  if (qr_finite_status[target] >= 0) {
    finite_status[target] = qr_finite_status[target];
  }
}

__global__ void check_fixed_sp_outputs_finite(
    const double* rhs,
    const double* theta,
    const int* safe_target_indices,
    const double* coefficients,
    const double* fitted,
    const double* residuals,
    const double* rss,
    int n,
    int p,
    int q,
    int safe_count,
    std::uint32_t output_mask,
    int* finite_status) {
  const int safe_ordinal =
    static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
    static_cast<int>(threadIdx.x);
  if (safe_ordinal >= safe_count) return;

  const int target = safe_target_indices[safe_ordinal];
  bool finite = true;
  const std::size_t target_offset = static_cast<std::size_t>(target);
  const double* target_rhs = rhs + static_cast<std::size_t>(q) * target_offset;
  const double* target_theta =
    theta + static_cast<std::size_t>(q) *
      static_cast<std::size_t>(safe_ordinal);
  for (int row = 0; row < q; ++row) {
    finite = finite && isfinite(target_rhs[row]) && isfinite(target_theta[row]);
  }
  if ((output_mask & FixedSpOutputCoefficients) != 0U) {
    const double* target_coefficients =
      coefficients + static_cast<std::size_t>(p) * target_offset;
    for (int row = 0; row < p; ++row) {
      finite = finite && isfinite(target_coefficients[row]);
    }
  }
  const bool fitted_computed =
    (output_mask & (FixedSpOutputFitted |
                    FixedSpOutputResiduals |
                    FixedSpOutputRss)) != 0U;
  if (fitted_computed) {
    const double* target_fitted =
      fitted + static_cast<std::size_t>(n) * target_offset;
    for (int row = 0; row < n; ++row) {
      finite = finite && isfinite(target_fitted[row]);
    }
  }
  if ((output_mask & FixedSpOutputResiduals) != 0U) {
    const double* target_residuals =
      residuals + static_cast<std::size_t>(n) * target_offset;
    for (int row = 0; row < n; ++row) {
      finite = finite && isfinite(target_residuals[row]);
    }
  }
  if ((output_mask & FixedSpOutputRss) != 0U) {
    finite = finite && isfinite(rss[target_offset]);
  }
  finite_status[target_offset] = finite ? 0 : 1;
}

}  // namespace

struct TestBlockedConsumerState {
  std::mutex mutex;
  std::condition_variable condition;
  bool completed = false;
};

namespace {

struct TestBlockedConsumerCallbackPayload {
  std::shared_ptr<TestBlockedConsumerState> state;
};

void CUDART_CB wait_for_test_blocked_consumer(void* data) {
  std::unique_ptr<TestBlockedConsumerCallbackPayload> payload(
    static_cast<TestBlockedConsumerCallbackPayload*>(data));
  if (!payload || !payload->state) return;
  const std::shared_ptr<TestBlockedConsumerState> state = payload->state;
  try {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->condition.wait(lock, [&state]() { return state->completed; });
  } catch (...) {
  }
}

void signal_test_blocked_consumer(
    const std::shared_ptr<TestBlockedConsumerState>& state) noexcept {
  if (!state) return;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->completed = true;
  }
  state->condition.notify_all();
}

struct TestBlockedConsumerTeardownStatus {
  CleanupCudaDeviceStatus device;
  cudaError_t stream_synchronize = cudaSuccess;
  TrackedTeardownResult<cudaError_t> event_destroy = {
    cudaSuccess, TrackedTeardownDisposition::Noop
  };
  TrackedTeardownResult<cudaError_t> stream_destroy = {
    cudaSuccess, TrackedTeardownDisposition::Noop
  };
};

class TestBlockedConsumerResources {
 public:
  TestBlockedConsumerResources(
      std::shared_ptr<FixedSpResourceLedger> resource_ledger,
      std::int64_t creator_pid,
      int device_id,
      std::shared_ptr<TestBlockedConsumerState> state)
      : resource_ledger_(std::move(resource_ledger)),
        creator_pid_(creator_pid),
        device_id_(device_id),
        state_(std::move(state)) {}

  ~TestBlockedConsumerResources() { cleanup_noexcept(); }

  TestBlockedConsumerResources(const TestBlockedConsumerResources&) = delete;
  TestBlockedConsumerResources& operator=(
    const TestBlockedConsumerResources&) = delete;

  cudaStream_t* stream_address() noexcept { return &stream_; }
  cudaEvent_t* event_address() noexcept { return &event_; }
  cudaStream_t stream() const noexcept { return stream_; }
  cudaEvent_t event() const noexcept { return event_; }
  bool has_resources() const noexcept {
    return stream_ != nullptr || event_ != nullptr;
  }

  void signal_noexcept() noexcept {
    signal_test_blocked_consumer(state_);
  }

  TestBlockedConsumerTeardownStatus teardown_once_noexcept() noexcept {
    TestBlockedConsumerTeardownStatus status;
    if (!has_resources()) {
      status.device.ready = true;
      status.device.restore_attempted = true;
      return status;
    }
    if (creator_pid_ != static_cast<std::int64_t>(getpid())) return status;

    signal_noexcept();
    FixedSpResourceLedger* ledger = resource_ledger_.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid_, device_id_, ledger);
    if (cleanup_device.ready()) {
      if (stream_ != nullptr) {
        status.stream_synchronize = cudaStreamSynchronize(stream_);
        if (status.stream_synchronize != cudaSuccess && ledger != nullptr) {
          record_cleanup_error(ledger);
        }
      }
      status.event_destroy = tracked_cuda_event_destroy_noexcept(
        ledger, &event_);
      status.stream_destroy = tracked_cuda_stream_destroy_noexcept(
        ledger, &stream_);
    }
    cleanup_device.restore_noexcept();
    status.device = cleanup_device.status();
    return status;
  }

  void cleanup_noexcept() noexcept {
    if (creator_pid_ != static_cast<std::int64_t>(getpid())) {
      stream_ = nullptr;
      event_ = nullptr;
      state_.reset();
      return;
    }
    if (!has_resources()) {
      state_.reset();
      return;
    }
    teardown_once_noexcept();
    if (has_resources()) teardown_once_noexcept();
    if (!has_resources()) state_.reset();
  }

 private:
  std::shared_ptr<FixedSpResourceLedger> resource_ledger_;
  std::int64_t creator_pid_ = -1;
  int device_id_ = -1;
  std::shared_ptr<TestBlockedConsumerState> state_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
};

void check_test_blocked_consumer_teardown(
    const TestBlockedConsumerTeardownStatus& status) {
  check_cuda(status.device.get_device,
             "query CUDA device for blocked consumer test teardown");
  check_cuda(status.device.select_device,
             "select CUDA device for blocked consumer test teardown");
  if (!status.device.ready) {
    throw std::runtime_error(
      "select CUDA device for blocked consumer test teardown");
  }
  check_cuda(status.stream_synchronize,
             "synchronize blocked consumer test stream");
  check_cuda(status.event_destroy.status, "destroy blocked consumer test event");
  check_cuda(status.stream_destroy.status,
             "destroy blocked consumer test stream");
  check_cuda(status.device.restore_device,
             "restore CUDA device after blocked consumer test teardown");
}

}  // namespace

enum class FixedSpOwnerLifecycle {
  Usable,
  TeardownOnly,
  Freed
};

class CudaRuntimeContext {
 public:
  explicit CudaRuntimeContext(int requested_device);
  ~CudaRuntimeContext();
  void reserve(const FixedSpCapacities& requested_capacities);
  FixedSpRuntimeInfo info() const;
  void require_usable() const;
  void require_explicit_close_identity() const;
  OwnerTeardownStatus close_once_noexcept() noexcept;
  void cleanup_noexcept() noexcept;

  int device_id = -1;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 1;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
  gesvdjInfo_t svd_params = nullptr;
  cudaEvent_t cholesky_factor_checkpoint_event = nullptr;
  cudaEvent_t cholesky_solve_checkpoint_event = nullptr;
  double* double_arena = nullptr;
  int* int_arena = nullptr;
  int* host_status_arena = nullptr;
  void** pointer_arena = nullptr;
  void* cublas_workspace = nullptr;
  std::size_t double_capacity = 0;
  std::size_t int_capacity = 0;
  std::size_t host_status_capacity = 0;
  std::size_t pointer_capacity = 0;
  std::size_t cublas_workspace_bytes = kFixedSpCublasWorkspaceBytes;
  int potrf_lwork = 0;
  FixedSpStableWorkspace stable_workspace;
  FixedSpCapacities capacities;
  std::shared_ptr<FixedSpResourceLedger> resource_ledger =
    std::make_shared<FixedSpResourceLedger>();
  FixedSpRuntimeInfo diagnostics;
  std::atomic<int> active_prepared_handle_count{0};
  FixedSpOwnerLifecycle lifecycle = FixedSpOwnerLifecycle::Usable;
  mutable std::mutex mutex;
};

enum class TransientResidualSlotState {
  Free,
  Leased,
  ConsumerRegistrationPending,
  Poisoned
};

struct TransientResidualSlot {
  ~TransientResidualSlot() { cleanup_noexcept(); }
  OwnerTeardownStatus close_once_noexcept() noexcept {
    OwnerTeardownStatus status;
    if (lifecycle == FixedSpOwnerLifecycle::Freed) return status;
    if (creator_pid != static_cast<std::int64_t>(getpid())) {
      coefficients = nullptr;
      fitted = nullptr;
      residuals = nullptr;
      rss = nullptr;
      rhs = nullptr;
      host_finite_status = nullptr;
      solve_completion_event = nullptr;
      consumer_completion_event = nullptr;
      lifecycle = FixedSpOwnerLifecycle::Freed;
      return status;
    }
    lifecycle = FixedSpOwnerLifecycle::TeardownOnly;
    FixedSpResourceLedger* ledger = resource_ledger.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid, device_id, ledger);
    if (cleanup_device.ready()) {
      status.observe(tracked_cuda_event_destroy_noexcept(
        ledger, &consumer_completion_event));
      status.observe(tracked_cuda_event_destroy_noexcept(
        ledger, &solve_completion_event));
      status.observe(
        tracked_cuda_free_host_noexcept(ledger, &host_finite_status));
      status.observe(tracked_cuda_free_noexcept(ledger, &rss));
      status.observe(tracked_cuda_free_noexcept(ledger, &rhs));
      status.observe(tracked_cuda_free_noexcept(ledger, &residuals));
      status.observe(tracked_cuda_free_noexcept(ledger, &fitted));
      status.observe(tracked_cuda_free_noexcept(ledger, &coefficients));
    } else {
      status.not_attempted_retryable = true;
    }
    cleanup_device.restore_noexcept();
    if (!status.not_attempted_retryable) {
      lifecycle = FixedSpOwnerLifecycle::Freed;
    }
    return status;
  }

  void cleanup_noexcept() noexcept {
    if (lifecycle == FixedSpOwnerLifecycle::Freed) return;
    close_once_noexcept();
    if (lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
      close_once_noexcept();
    }
  }

  std::shared_ptr<FixedSpResourceLedger> resource_ledger;
  std::int64_t creator_pid = -1;
  int device_id = -1;
  int target_capacity = 0;
  std::size_t finite_status_capacity = 0;
  std::size_t coefficient_capacity = 0;
  std::size_t rhs_capacity = 0;
  double* coefficients = nullptr;
  double* fitted = nullptr;
  double* residuals = nullptr;
  double* rss = nullptr;
  double* rhs = nullptr;
  int* host_finite_status = nullptr;
  cudaEvent_t solve_completion_event = nullptr;
  cudaEvent_t consumer_completion_event = nullptr;
  std::uint64_t generation = 0;
  TransientResidualSlotState state = TransientResidualSlotState::Free;
  std::weak_ptr<DeviceResidualBatch> lease_owner;
  bool consumer_event_registered = false;
  std::string poison_reason;
  FixedSpOwnerLifecycle lifecycle = FixedSpOwnerLifecycle::Usable;
};

class PreparedSGpuHandle {
 public:
  ~PreparedSGpuHandle() { cleanup_noexcept(); }

  OwnerTeardownStatus close_once_noexcept() noexcept {
    OwnerTeardownStatus status;
    if (lifecycle == FixedSpOwnerLifecycle::Freed) return status;
    if (creator_pid != static_cast<std::int64_t>(getpid())) {
      d_X = nullptr;
      d_Z = nullptr;
      d_X_null = nullptr;
      d_gram = nullptr;
      d_projected_penalties = nullptr;
      d_projected_H = nullptr;
      d_penalty_roots = nullptr;
      d_H_root = nullptr;
      setup_completion_event = nullptr;
      residual_slot.reset();
      context.reset();
      resource_ledger.reset();
      registered_with_context = false;
      lifecycle = FixedSpOwnerLifecycle::Freed;
      return status;
    }
    lifecycle = FixedSpOwnerLifecycle::TeardownOnly;
    if (registered_with_context && context) {
      context->active_prepared_handle_count.fetch_sub(
        1, std::memory_order_acq_rel);
      registered_with_context = false;
    }
    FixedSpResourceLedger* ledger = resource_ledger.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid, device_id, ledger);
    if (cleanup_device.ready()) {
      status.observe(tracked_cuda_event_destroy_noexcept(
        ledger, &setup_completion_event));
      if (residual_slot) {
        status.merge(residual_slot->close_once_noexcept());
        if (residual_slot->lifecycle == FixedSpOwnerLifecycle::Freed) {
          residual_slot.reset();
        }
      }
      status.observe(
        tracked_cuda_free_noexcept(ledger, &d_projected_penalties));
      status.observe(tracked_cuda_free_noexcept(ledger, &d_projected_H));
      status.observe(tracked_cuda_free_noexcept(ledger, &d_H_root));
      status.observe(tracked_cuda_free_noexcept(ledger, &d_penalty_roots));
      status.observe(tracked_cuda_free_noexcept(ledger, &d_gram));
      if (d_X_null == d_X) {
        d_X_null = nullptr;
      } else {
        status.observe(tracked_cuda_free_noexcept(ledger, &d_X_null));
      }
      status.observe(tracked_cuda_free_noexcept(ledger, &d_Z));
      status.observe(tracked_cuda_free_noexcept(ledger, &d_X));
    } else {
      status.not_attempted_retryable = true;
    }
    cleanup_device.restore_noexcept();
    if (!status.not_attempted_retryable) {
      residual_slot.reset();
      context.reset();
      generation += 1;
      lifecycle = FixedSpOwnerLifecycle::Freed;
    }
    return status;
  }

  void cleanup_noexcept() noexcept {
    if (lifecycle == FixedSpOwnerLifecycle::Freed) return;
    close_once_noexcept();
    if (lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
      close_once_noexcept();
    }
  }

  std::shared_ptr<CudaRuntimeContext> context;
  std::shared_ptr<FixedSpResourceLedger> resource_ledger;
  std::int64_t creator_pid = -1;
  int device_id = -1;
  std::uint64_t context_generation = 0;
  std::uint64_t generation = 1;
  std::string prepared_s_key_sha256;
  int n = 0;
  int p = 0;
  int q = 0;
  int penalty_count = 0;
  double* d_X = nullptr;
  double* d_Z = nullptr;
  double* d_X_null = nullptr;
  double* d_gram = nullptr;
  double* d_projected_penalties = nullptr;
  double* d_projected_H = nullptr;
  double* d_penalty_roots = nullptr;
  std::vector<int> penalty_root_offsets;
  std::vector<int> penalty_root_ranks;
  int total_penalty_root_rows = 0;
  bool has_H = false;
  double* d_H_root = nullptr;
  int H_root_rank = 0;
  int penalty_root_build_count = 0;
  int penalty_root_rank_mismatch_count = 0;
  std::size_t penalty_root_bytes = 0;
  double penalty_root_build_ms = 0.0;
  int setup_shadow_d2h_count = 0;
  std::size_t setup_shadow_d2h_bytes = 0;
  int augmented_test_shadow_d2h_count = 0;
  std::size_t augmented_test_shadow_d2h_bytes = 0;
  int projected_H_test_shadow_d2h_count = 0;
  std::size_t projected_H_test_shadow_d2h_bytes = 0;
  std::shared_ptr<TransientResidualSlot> residual_slot;
  cudaEvent_t setup_completion_event = nullptr;
  int setup_h2d_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  bool transient_fixed_sp_compatibility = false;
  bool registered_with_context = false;
  FixedSpOwnerLifecycle lifecycle = FixedSpOwnerLifecycle::Usable;
};

class DeviceResidualBatch {
 public:
  ~DeviceResidualBatch();
  void cleanup_noexcept() noexcept;

  std::shared_ptr<PreparedSGpuHandle> owner;
  std::int64_t creator_pid = -1;
  int device_id = -1;
  std::uint64_t owner_generation = 0;
  std::uint64_t slot_generation = 0;
  int n = 0;
  int target_count = 0;
  std::uint32_t output_mask = 0;
  std::vector<std::string> target_keys;
  std::shared_ptr<TransientResidualSlot> slot;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<std::string> reroute_reasons;
  std::vector<FixedSpStatus> solver_statuses;
  std::vector<bool> target_true_batched;
  DeviceResidualInfo diagnostics;
  std::shared_ptr<TestBlockedConsumerResources> test_blocked_consumer_resources;
  bool output_status_resolved = false;
  bool lease_released = false;
  bool freed = false;
};

namespace {

const char* transient_residual_slot_state_name(
    TransientResidualSlotState state) {
  switch (state) {
    case TransientResidualSlotState::Free: return "free";
    case TransientResidualSlotState::Leased: return "leased";
    case TransientResidualSlotState::ConsumerRegistrationPending:
      return "consumer_registration_pending";
    case TransientResidualSlotState::Poisoned: return "poisoned";
  }
  return "unknown";
}

void poison_transient_residual_slot(
    TransientResidualSlot* slot,
    std::uint64_t expected_generation,
    const char* reason,
    bool clear_lease_owner) noexcept {
  if (slot == nullptr || slot->generation != expected_generation) return;
  if (slot->state != TransientResidualSlotState::Poisoned) {
    slot->state = TransientResidualSlotState::Poisoned;
    try {
      slot->poison_reason =
        reason != nullptr && reason[0] != '\0' ? reason :
          "unknown output-slot failure";
    } catch (...) {
    }
  }
  if (clear_lease_owner) slot->lease_owner.reset();
}

[[noreturn]] void throw_output_slot_poisoned(
    const TransientResidualSlot& slot) {
  std::string message = "ERR_OUTPUT_SLOT_POISONED";
  if (!slot.poison_reason.empty()) {
    message += ": ";
    message += slot.poison_reason;
  }
  throw std::runtime_error(message);
}

std::shared_ptr<CudaRuntimeContext> require_prepared_host_identity(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  if (!handle || handle->lifecycle == FixedSpOwnerLifecycle::Freed) {
    throw std::runtime_error("fixed-sp CUDA prepared handle has been freed");
  }
  if (handle->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle is TeardownOnly");
  }
  if (handle->creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle has a different creator PID");
  }
  const std::shared_ptr<CudaRuntimeContext> context = handle->context;
  if (!context || context->lifecycle == FixedSpOwnerLifecycle::Freed) {
    throw std::runtime_error("fixed-sp CUDA prepared runtime has been freed");
  }
  if (context->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error("fixed-sp CUDA runtime is TeardownOnly");
  }
  if (context->creator_pid != handle->creator_pid ||
      handle->device_id != context->device_id ||
      handle->context_generation != context->generation) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle ownership mismatch");
  }
  if (!handle->residual_slot ||
      handle->residual_slot->lifecycle != FixedSpOwnerLifecycle::Usable ||
      handle->residual_slot->creator_pid != handle->creator_pid ||
      handle->residual_slot->device_id != handle->device_id) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared output-slot ownership mismatch");
  }
  return context;
}

std::shared_ptr<CudaRuntimeContext> require_prepared_close_identity(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  if (!handle || handle->lifecycle == FixedSpOwnerLifecycle::Freed) {
    return std::shared_ptr<CudaRuntimeContext>();
  }
  if (handle->creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle has a different creator PID");
  }
  const std::shared_ptr<CudaRuntimeContext> context = handle->context;
  if (!context || context->lifecycle == FixedSpOwnerLifecycle::Freed) {
    throw std::runtime_error("fixed-sp CUDA prepared runtime has been freed");
  }
  if (context->creator_pid != handle->creator_pid ||
      handle->device_id != context->device_id ||
      handle->context_generation != context->generation) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle ownership mismatch");
  }
  context->require_explicit_close_identity();
  return context;
}

std::shared_ptr<CudaRuntimeContext> require_residual_host_identity(
    const std::shared_ptr<DeviceResidualBatch>& token,
    bool require_active_lease) {
  if (!token || token->freed) {
    throw std::runtime_error("fixed-sp CUDA residual token has been freed");
  }
  if (token->creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA residual token has a different creator PID");
  }
  if (!token->owner ||
      token->owner->lifecycle == FixedSpOwnerLifecycle::Freed ||
      token->owner->creator_pid != token->creator_pid ||
      token->owner_generation != token->owner->generation) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error(
      "STALE_TOKEN: residual owner generation mismatch");
  }
  if (token->owner->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle is TeardownOnly");
  }
  const std::shared_ptr<CudaRuntimeContext> context = token->owner->context;
  if (!context || context->lifecycle == FixedSpOwnerLifecycle::Freed) {
    throw std::runtime_error("fixed-sp CUDA residual runtime has been freed");
  }
  if (context->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error("fixed-sp CUDA runtime is TeardownOnly");
  }
  if (context->creator_pid != token->creator_pid ||
      token->owner->context_generation != context->generation ||
      token->owner->device_id != context->device_id ||
      token->device_id != context->device_id) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error(
      "STALE_TOKEN: residual owner generation mismatch");
  }
  if (!require_active_lease) return context;
  if (token->lease_released || !token->slot ||
      token->slot->lifecycle == FixedSpOwnerLifecycle::Freed) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error("STALE_TOKEN: residual lease has been released");
  }
  if (token->slot->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error(
      "fixed-sp CUDA residual slot is TeardownOnly");
  }
  if (token->slot->creator_pid != token->creator_pid ||
      token->slot->device_id != context->device_id ||
      token->slot_generation != token->slot->generation) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error(
      "STALE_TOKEN: residual slot generation mismatch");
  }
  const std::shared_ptr<DeviceResidualBatch> lease_owner =
    token->slot->lease_owner.lock();
  if (!lease_owner || lease_owner.get() != token.get() ||
      (token->slot->state != TransientResidualSlotState::Leased &&
       token->slot->state !=
         TransientResidualSlotState::ConsumerRegistrationPending &&
       token->slot->state != TransientResidualSlotState::Poisoned)) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error(
      "STALE_TOKEN: residual slot generation mismatch");
  }
  return context;
}

void validate_fixed_sp_batch(const PreparedSGpuHandle& handle,
                             const CudaRuntimeContext& context,
                             const FixedSpBatchHostView& batch) {
  if (batch.Y == nullptr || batch.SP == nullptr) {
    throw std::runtime_error("fixed-sp batch matrix pointers are null");
  }
  if (batch.n != handle.n || batch.null_dim != handle.q ||
      batch.penalty_count != handle.penalty_count) {
    throw std::runtime_error(
      "fixed-sp batch dimensions do not match prepared handle");
  }
  if (batch.target_count <= 0 ||
      batch.target_count > context.capacities.target_count) {
    throw std::runtime_error(
      "fixed-sp batch target count exceeds reserved capacity");
  }
  if (batch.output_mask == 0 ||
      (batch.output_mask & ~kFixedSpPublicOutputMask) != 0) {
    throw std::runtime_error("fixed-sp output mask is invalid");
  }
  const std::size_t targets = static_cast<std::size_t>(batch.target_count);
  if (batch.planned_routes.size() != targets ||
      batch.target_keys.size() != targets) {
    throw std::runtime_error("fixed-sp batch metadata size mismatch");
  }
  if (handle.transient_fixed_sp_compatibility && batch.target_count != 1) {
    throw std::runtime_error(
      "transient fixed-sp compatibility requires exactly one target");
  }

  std::unordered_set<std::string> unique_keys;
  unique_keys.reserve(targets);
  for (std::size_t index = 0; index < targets; ++index) {
    const FixedSpRoute route = batch.planned_routes[index];
    if (handle.transient_fixed_sp_compatibility &&
        route != FixedSpRoute::AugmentedSvd) {
      throw std::runtime_error(
        "transient fixed-sp compatibility requires AUGMENTED_SVD");
    }
    if (route != FixedSpRoute::CholeskyBatched &&
        route != FixedSpRoute::AugmentedQr &&
        route != FixedSpRoute::AugmentedSvd) {
      throw std::runtime_error("fixed-sp planned route metadata is invalid");
    }
    const std::string& target_key = batch.target_keys[index];
    const bool valid_target_key = handle.transient_fixed_sp_compatibility ?
      target_key == "transient-fixed-sp-compatibility-target-v1" :
      target_key.size() == 64U &&
        std::all_of(target_key.begin(), target_key.end(),
                    [](unsigned char character) {
          return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
        });
    if (!valid_target_key || !unique_keys.insert(target_key).second) {
      throw std::runtime_error("fixed-sp target key metadata is invalid");
    }
  }

  const std::size_t y_count = checked_multiply(
    positive_size(batch.n, "fixed-sp batch row count"), targets,
    "fixed-sp batch Y");
  const std::size_t sp_count = checked_multiply(
    positive_size(batch.penalty_count, "fixed-sp batch penalty count"),
    targets, "fixed-sp batch SP");
  for (std::size_t index = 0; index < y_count; ++index) {
    if (!std::isfinite(batch.Y[index])) {
      throw std::runtime_error("Y must be finite");
    }
  }
  for (std::size_t index = 0; index < sp_count; ++index) {
    if (!std::isfinite(batch.SP[index]) || batch.SP[index] < 0.0) {
      throw std::runtime_error("SP must be finite and non-negative");
    }
  }
}

void increment_active_slot_busy(TransientResidualSlot* slot) {
  if (slot == nullptr) return;
  if (std::shared_ptr<DeviceResidualBatch> owner = slot->lease_owner.lock()) {
    owner->diagnostics.output_slot_busy_count += 1;
  }
}

bool fixed_sp_status_is_successful(FixedSpStatus status) {
  return status == FixedSpStatus::OkCholeskyBatched ||
    status == FixedSpStatus::OkCholeskySingle ||
    status == FixedSpStatus::OkAugmentedQr ||
    status == FixedSpStatus::OkAugmentedSvd;
}

void resolve_fixed_sp_output_status_locked(
    DeviceResidualBatch* token,
    CudaRuntimeContext* context) {
  if (token == nullptr || token->output_status_resolved) return;
  if (context == nullptr || !token->slot ||
      token->slot_generation != token->slot->generation ||
      token->slot->solve_completion_event == nullptr ||
      token->slot->host_finite_status == nullptr ||
      static_cast<std::size_t>(token->target_count) >
        token->slot->finite_status_capacity) {
    throw std::runtime_error(
      "STALE_TOKEN: fixed-sp output status storage mismatch");
  }

  check_cuda(cudaEventSynchronize(token->slot->solve_completion_event),
             "wait for fixed-sp output status");
  for (int target = 0; target < token->target_count; ++target) {
    const std::size_t target_offset = static_cast<std::size_t>(target);
    if (fixed_sp_status_is_successful(
          token->solver_statuses[target_offset]) &&
        token->slot->host_finite_status[target_offset] != 0) {
      token->solver_statuses[target_offset] =
        FixedSpStatus::ErrNonfiniteOutput;
      token->diagnostics.nonfinite_output_count += 1;
    }
    if (token->solver_statuses[target_offset] !=
        FixedSpStatus::OkCholeskyBatched) {
      token->target_true_batched[target_offset] = false;
    }
  }
  token->diagnostics.true_batched_target_count =
    static_cast<int>(std::count(
      token->solver_statuses.begin(), token->solver_statuses.end(),
      FixedSpStatus::OkCholeskyBatched));
  token->diagnostics.executed_svd_target_count =
    static_cast<int>(std::count(
      token->executed_routes.begin(), token->executed_routes.end(),
      FixedSpRoute::AugmentedSvd));
  token->diagnostics.true_batched_kernel =
    token->target_count >= 2 &&
    token->diagnostics.true_batched_target_count == token->target_count;
  token->diagnostics.batch_output_finalized_target_count =
    static_cast<int>(std::count_if(
      token->solver_statuses.begin(), token->solver_statuses.end(),
      [](FixedSpStatus status) {
        return fixed_sp_status_is_successful(status);
      }));
  token->diagnostics.solver_statuses = token->solver_statuses;
  token->diagnostics.target_true_batched = token->target_true_batched;
  token->output_status_resolved = true;
}

}  // namespace

DeviceResidualBatch::~DeviceResidualBatch() {
  cleanup_noexcept();
}

void DeviceResidualBatch::cleanup_noexcept() noexcept {
  if (creator_pid != static_cast<std::int64_t>(getpid())) {
    test_blocked_consumer_resources.reset();
    return;
  }
  if (test_blocked_consumer_resources) {
    test_blocked_consumer_resources->signal_noexcept();
  }
  if (freed || lease_released || !owner || !slot) {
    test_blocked_consumer_resources.reset();
    return;
  }
  const std::shared_ptr<CudaRuntimeContext> context = owner->context;
  if (!context) {
    test_blocked_consumer_resources.reset();
    return;
  }
  try {
    std::lock_guard<std::mutex> lock(context->mutex);
    if (owner_generation != owner->generation ||
        slot_generation != slot->generation) {
      test_blocked_consumer_resources.reset();
      return;
    }
    if (slot->state == TransientResidualSlotState::Poisoned) {
      test_blocked_consumer_resources.reset();
      poison_transient_residual_slot(
        slot.get(), slot_generation, nullptr, true);
      lease_released = true;
      freed = true;
      return;
    }
    if (slot->state != TransientResidualSlotState::Leased) {
      test_blocked_consumer_resources.reset();
      return;
    }

    try {
      ScopedCleanupCudaDevice cleanup_device(
        creator_pid, context->device_id, context->resource_ledger.get());
      if (!cleanup_device.ready()) {
        throw std::runtime_error(
          "select CUDA device for residual token cleanup");
      }
      if (slot->consumer_event_registered) {
        check_cuda(cudaEventSynchronize(slot->consumer_completion_event),
                   "wait for residual consumer during token cleanup");
        slot->consumer_event_registered = false;
      }
      test_blocked_consumer_resources.reset();
    } catch (const std::exception& error) {
      test_blocked_consumer_resources.reset();
      poison_transient_residual_slot(
        slot.get(), slot_generation, error.what(), true);
      lease_released = true;
      freed = true;
      return;
    } catch (...) {
      test_blocked_consumer_resources.reset();
      poison_transient_residual_slot(
        slot.get(), slot_generation,
        "unknown residual token cleanup failure", true);
      lease_released = true;
      freed = true;
      return;
    }
    slot->state = TransientResidualSlotState::Free;
    slot->lease_owner.reset();
    lease_released = true;
    diagnostics.output_slot_release_count += 1;
    freed = true;
  } catch (...) {
  }
}

CudaRuntimeContext::CudaRuntimeContext(int requested_device) {
  if (requested_device < 0) {
    throw std::runtime_error("device_id must be non-negative");
  }

  device_id = requested_device;
  creator_pid = static_cast<std::int64_t>(getpid());
  diagnostics.device_id = device_id;
  diagnostics.creator_pid = creator_pid;
  diagnostics.generation = generation;
  diagnostics.runtime_context_create_count = 1;

  try {
    check_cuda(cudaSetDevice(device_id), "set CUDA device");

    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, device_id),
               "query CUDA device properties");
    diagnostics.gpu_name = properties.name;
    diagnostics.compute_capability_major = properties.major;
    diagnostics.compute_capability_minor = properties.minor;
    diagnostics.sm_count = properties.multiProcessorCount;
    check_cuda(cudaRuntimeGetVersion(&diagnostics.cuda_toolkit_version),
               "query CUDA runtime version");
    check_cuda(cudaDriverGetVersion(&diagnostics.cuda_driver_version),
               "query CUDA driver version");

    tracked_cuda_stream_create(
      resource_ledger.get(), &stream, cudaStreamNonBlocking,
      "create CUDA stream");

    tracked_cublas_create(
      resource_ledger.get(), &blas, "create cuBLAS handle");
    check_cublas(cublasSetStream(blas, stream), "bind cuBLAS stream");
    check_cublas(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH),
                 "set cuBLAS pedantic math");
    check_cublas(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED),
                 "disable cuBLAS atomics");

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    check_cublas(cublasGetMathMode(blas, &math_mode),
                 "query cuBLAS math mode");
    if (math_mode != CUBLAS_PEDANTIC_MATH) {
      throw std::runtime_error("cuBLAS pedantic math configuration mismatch");
    }
    diagnostics.cublas_pedantic_math_enabled = true;

    cublasAtomicsMode_t atomics_mode = CUBLAS_ATOMICS_ALLOWED;
    check_cublas(cublasGetAtomicsMode(blas, &atomics_mode),
                 "query cuBLAS atomics mode");
    if (atomics_mode != CUBLAS_ATOMICS_NOT_ALLOWED) {
      throw std::runtime_error("cuBLAS atomics configuration mismatch");
    }
    diagnostics.cublas_atomics_not_allowed = true;

    tracked_cusolver_create(
      resource_ledger.get(), &solver, "create cuSOLVER handle");
    check_cusolver(cusolverDnSetStream(solver, stream),
                   "bind cuSOLVER stream");
    check_cusolver(cusolverDnSetDeterministicMode(
      solver, CUSOLVER_DETERMINISTIC_RESULTS
    ), "enable deterministic cuSOLVER");

    cusolverDeterministicMode_t deterministic_mode =
      CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS;
    check_cusolver(cusolverDnGetDeterministicMode(
      solver, &deterministic_mode
    ), "query deterministic cuSOLVER");
    if (deterministic_mode != CUSOLVER_DETERMINISTIC_RESULTS) {
      throw std::runtime_error(
        "cuSOLVER deterministic configuration mismatch");
    }
    diagnostics.cusolver_deterministic_mode_enabled = true;

    tracked_gesvdj_info_create(
      resource_ledger.get(), &svd_params, "create Phase 3C gesvdjInfo");
    check_cusolver(cusolverDnXgesvdjSetTolerance(svd_params, 1e-12),
                   "set Phase 3C SVD convergence tolerance");
    check_cusolver(cusolverDnXgesvdjSetMaxSweeps(svd_params, 100),
                   "set Phase 3C SVD max sweeps");
    check_cusolver(cusolverDnXgesvdjSetSortEig(svd_params, 1),
                   "sort Phase 3C singular values");

    tracked_cuda_event_create(
      resource_ledger.get(), &cholesky_factor_checkpoint_event,
      cudaEventDisableTiming, "create Cholesky factor checkpoint event");
    tracked_cuda_event_create(
      resource_ledger.get(), &cholesky_solve_checkpoint_event,
      cudaEventDisableTiming, "create Cholesky solve checkpoint event");
  } catch (...) {
    cleanup_noexcept();
    throw;
  }
}

CudaRuntimeContext::~CudaRuntimeContext() {
  cleanup_noexcept();
}

void CudaRuntimeContext::cleanup_noexcept() noexcept {
  if (lifecycle == FixedSpOwnerLifecycle::Freed) return;
  close_once_noexcept();
  if (lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    close_once_noexcept();
  }
}

OwnerTeardownStatus CudaRuntimeContext::close_once_noexcept() noexcept {
  OwnerTeardownStatus status;
  if (lifecycle == FixedSpOwnerLifecycle::Freed) return status;
  if (creator_pid != static_cast<std::int64_t>(getpid())) {
    stream = nullptr;
    blas = nullptr;
    solver = nullptr;
    svd_params = nullptr;
    cholesky_factor_checkpoint_event = nullptr;
    cholesky_solve_checkpoint_event = nullptr;
    double_arena = nullptr;
    int_arena = nullptr;
    host_status_arena = nullptr;
    pointer_arena = nullptr;
    cublas_workspace = nullptr;
    stable_workspace = FixedSpStableWorkspace{};
    diagnostics.freed = true;
    lifecycle = FixedSpOwnerLifecycle::Freed;
    return status;
  }
  lifecycle = FixedSpOwnerLifecycle::TeardownOnly;
  FixedSpResourceLedger* ledger = resource_ledger.get();
  ScopedCleanupCudaDevice cleanup_device(creator_pid, device_id, ledger);
  if (cleanup_device.ready()) {
    status.observe(tracked_cuda_free_noexcept(ledger, &double_arena));
    if (double_arena == nullptr) stable_workspace = FixedSpStableWorkspace{};
    status.observe(tracked_cuda_free_noexcept(ledger, &int_arena));
    status.observe(
      tracked_cuda_free_host_noexcept(ledger, &host_status_arena));
    status.observe(tracked_cuda_free_noexcept(ledger, &pointer_arena));
    status.observe(tracked_cuda_free_noexcept(ledger, &cublas_workspace));

    status.observe(tracked_cuda_event_destroy_noexcept(
      ledger, &cholesky_solve_checkpoint_event));
    status.observe(tracked_cuda_event_destroy_noexcept(
      ledger, &cholesky_factor_checkpoint_event));
    status.observe(tracked_gesvdj_info_destroy_noexcept(ledger, &svd_params));
    status.observe(tracked_cusolver_destroy_noexcept(ledger, &solver));
    status.observe(tracked_cublas_destroy_noexcept(ledger, &blas));
    status.observe(tracked_cuda_stream_destroy_noexcept(ledger, &stream));
  } else {
    status.not_attempted_retryable = true;
  }
  cleanup_device.restore_noexcept();
  if (!status.not_attempted_retryable) {
    generation += 1;
    lifecycle = FixedSpOwnerLifecycle::Freed;
    diagnostics.freed = true;
    diagnostics.generation = generation;
  }
  return status;
}

void CudaRuntimeContext::require_usable() const {
  if (lifecycle == FixedSpOwnerLifecycle::Freed) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
  if (lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
    throw std::runtime_error("fixed-sp CUDA runtime is TeardownOnly");
  }
  require_explicit_close_identity();
}

void CudaRuntimeContext::require_explicit_close_identity() const {
  if (lifecycle == FixedSpOwnerLifecycle::Freed) return;
  if (creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime has a different creator PID");
  }
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device),
             "query current device for fixed-sp CUDA runtime");
  if (current_device != device_id) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime is on the wrong device");
  }
}

void CudaRuntimeContext::reserve(
    const FixedSpCapacities& requested_capacities) {
  std::lock_guard<std::mutex> lock(mutex);
  require_usable();
  validate_capacities(requested_capacities);
  check_cuda(cudaSetDevice(device_id), "set CUDA device for reserve");

  FixedSpCapacities merged_capacities;
  merged_capacities.n = std::max(capacities.n, requested_capacities.n);
  merged_capacities.null_dim =
    std::max(capacities.null_dim, requested_capacities.null_dim);
  merged_capacities.target_count =
    std::max(capacities.target_count, requested_capacities.target_count);
  merged_capacities.penalty_count =
    std::max(capacities.penalty_count, requested_capacities.penalty_count);
  merged_capacities.augmented_rows =
    std::max(capacities.augmented_rows, requested_capacities.augmented_rows);

  const std::size_t merged_stable_rows = checked_add(
    static_cast<std::size_t>(merged_capacities.n),
    static_cast<std::size_t>(merged_capacities.null_dim),
    "stable n plus null dimension");
  if (merged_stable_rows >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("stable row capacity size overflow");
  }
  const int effective_stable_rows = std::max(
    merged_capacities.augmented_rows,
    static_cast<int>(merged_stable_rows));

  const bool within_existing_capacity =
    requested_capacities.n <= capacities.n &&
    requested_capacities.null_dim <= capacities.null_dim &&
    requested_capacities.target_count <= capacities.target_count &&
    requested_capacities.penalty_count <= capacities.penalty_count &&
    requested_capacities.augmented_rows <= capacities.augmented_rows &&
    effective_stable_rows <= stable_workspace.max_rows;
  if (within_existing_capacity) {
    diagnostics.workspace_reserve_count += 1;
    return;
  }
  if (active_prepared_handle_count.load(std::memory_order_acquire) > 0) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime cannot grow with active prepared handles");
  }

  int requested_potrf_lwork = potrf_lwork;
  if (merged_capacities.null_dim > capacities.null_dim) {
    const std::size_t q =
      static_cast<std::size_t>(merged_capacities.null_dim);
    const std::size_t probe_count = checked_multiply(q, q, "potrf probe");
    const std::size_t probe_bytes =
      allocation_bytes(probe_count, sizeof(double), "potrf probe");
    double* probe = nullptr;
    try {
      tracked_cuda_malloc(
        resource_ledger.get(), &probe, probe_bytes, "allocate potrf probe");
      check_cusolver(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, merged_capacities.null_dim,
        probe, merged_capacities.null_dim, &requested_potrf_lwork
      ), "query cuSOLVER potrf workspace");
      tracked_cuda_free(
        resource_ledger.get(), &probe, "free potrf probe");
    } catch (...) {
      cleanup_local_cuda_allocation_noexcept(
        resource_ledger.get(), &probe);
      throw;
    }
    if (requested_potrf_lwork < 0) {
      throw std::runtime_error("cuSOLVER potrf workspace size is invalid");
    }
  }

  FixedSpStableWorkspace requested_stable_workspace = stable_workspace;
  const bool stable_dimensions_grow =
    effective_stable_rows > stable_workspace.max_rows ||
    merged_capacities.null_dim > stable_workspace.max_q;
  if (stable_dimensions_grow) {
    double* stable_probe = nullptr;
    try {
      const std::size_t probe_count =
        fixed_sp_stable_probe_double_count(
          effective_stable_rows,
          merged_capacities.null_dim);
      tracked_cuda_malloc(
        resource_ledger.get(), &stable_probe,
        allocation_bytes(
          probe_count, sizeof(double), "stable reserve probe"),
        "allocate stable reserve probe");
      requested_stable_workspace = fixed_sp_stable_probe_view(
        stable_probe, effective_stable_rows,
        merged_capacities.null_dim);
      query_fixed_sp_stable_workspace(
        solver, svd_params, &requested_stable_workspace);
      // Scoped probes must not intercept teardown injection armed for a
      // persistent arena owner, but their real allocation/free is ledgered.
      tracked_cuda_reserve_probe_free(
        resource_ledger.get(), &stable_probe, "free stable reserve probe");
    } catch (...) {
      tracked_cuda_reserve_probe_free_noexcept(
        resource_ledger.get(), &stable_probe);
      throw;
    }
  }

  const std::size_t targets =
    static_cast<std::size_t>(merged_capacities.target_count);
  const std::size_t n = static_cast<std::size_t>(merged_capacities.n);
  const std::size_t q =
    static_cast<std::size_t>(merged_capacities.null_dim);
  const std::size_t penalties =
    static_cast<std::size_t>(merged_capacities.penalty_count);
  const std::size_t y_count = checked_multiply(n, targets, "Y arena");
  const std::size_t sp_count =
    checked_multiply(penalties, targets, "SP arena");
  const std::size_t rhs_count =
    checked_multiply(q, targets, "RHS arena");
  const std::size_t q_squared =
    checked_multiply(q, q, "system matrix arena");
  const std::size_t system_count =
    checked_multiply(q_squared, targets, "system matrix arena");
  const std::size_t theta_count =
    checked_multiply(q, targets, "theta arena");
  std::size_t double_required = checked_add(y_count, sp_count, "double arena");
  double_required = checked_add(double_required, rhs_count, "double arena");
  double_required = checked_add(double_required, system_count, "double arena");
  double_required = checked_add(double_required, theta_count, "double arena");
  double_required = checked_add(
    double_required, static_cast<std::size_t>(requested_potrf_lwork),
    "double arena");
  const std::size_t stable_double_count = checked_multiply(
    targets, kStableDoubleArraysPerTarget, "stable double diagnostics");
  double_required = checked_add(
    double_required, stable_double_count, "double arena");
  const std::size_t stable_workspace_offset = double_required;
  const std::size_t stable_workspace_count =
    fixed_sp_stable_workspace_double_count(
      requested_stable_workspace.max_rows,
      requested_stable_workspace.max_q,
      requested_stable_workspace.eigen_lwork,
      requested_stable_workspace.qr_lwork,
      requested_stable_workspace.ormqr_lwork,
      requested_stable_workspace.svd_lwork);
  double_required = checked_add(
    double_required, stable_workspace_count, "double arena");

  const std::size_t legacy_int_count = checked_add(
    checked_multiply(targets, 3U, "int arena"), 1U, "int arena");
  const std::size_t stable_compact_width = checked_add(
    q,
    checked_add(
      kStableBaseIntArraysPerTarget,
      kStableAggregateIntArraysPerTarget,
      "stable compact int width"),
    "stable compact int width");
  const std::size_t stable_compact_int_count = checked_multiply(
    targets, stable_compact_width, "stable compact int arena");
  const std::size_t stable_info_offset = checked_add(
    legacy_int_count, stable_compact_int_count, "stable info offset");
  const std::size_t int_required = checked_add(
    stable_info_offset, 1U, "int arena");
  const std::size_t host_sigma_offset = checked_add(
    int_required, int_required % kIntsPerDouble == 0U ? 0U :
      kIntsPerDouble - int_required % kIntsPerDouble,
    "host stable sigma alignment");
  const std::size_t host_status_required = checked_add(
    host_sigma_offset,
    checked_multiply(stable_double_count, kIntsPerDouble,
                     "host stable double diagnostics"),
    "host status arena");
  const std::size_t pointer_required =
    checked_multiply(targets, 2U, "pointer arena");

  double* new_double_arena = nullptr;
  int* new_int_arena = nullptr;
  int* new_host_status_arena = nullptr;
  void** new_pointer_arena = nullptr;
  void* new_cublas_workspace = nullptr;
  std::size_t new_cublas_alignment = diagnostics.cublas_workspace_alignment;
  auto cleanup_new_allocations = [&]() noexcept {
    cleanup_local_cuda_allocation_noexcept(
      resource_ledger.get(), &new_double_arena);
    cleanup_local_cuda_allocation_noexcept(
      resource_ledger.get(), &new_int_arena);
    cleanup_local_cuda_host_allocation_noexcept(
      resource_ledger.get(), &new_host_status_arena);
    cleanup_local_cuda_allocation_noexcept(
      resource_ledger.get(), &new_pointer_arena);
    cleanup_local_cuda_allocation_noexcept(
      resource_ledger.get(), &new_cublas_workspace);
  };

  try {
    if (double_required > double_capacity) {
      tracked_cuda_malloc(
        resource_ledger.get(),
        &new_double_arena,
        allocation_bytes(double_required, sizeof(double), "double arena"),
        "allocate fixed-sp double arena");
    }
    if (int_required > int_capacity) {
      tracked_cuda_malloc(
        resource_ledger.get(),
        &new_int_arena,
        allocation_bytes(int_required, sizeof(int), "int arena"),
        "allocate fixed-sp int arena");
    }
    if (host_status_required > host_status_capacity) {
      tracked_cuda_malloc_host(
        resource_ledger.get(),
        &new_host_status_arena,
        allocation_bytes(host_status_required, sizeof(int),
                         "host status arena"),
        "allocate fixed-sp host status arena");
    }
    if (pointer_required > pointer_capacity) {
      tracked_cuda_malloc(
        resource_ledger.get(),
        &new_pointer_arena,
        allocation_bytes(pointer_required, sizeof(void*), "pointer arena"),
        "allocate fixed-sp pointer arena");
    }
    if (cublas_workspace == nullptr) {
      tracked_cuda_malloc(
        resource_ledger.get(), &new_cublas_workspace,
        cublas_workspace_bytes, "allocate cuBLAS user workspace");
      new_cublas_alignment = pointer_alignment(new_cublas_workspace);
      if (new_cublas_alignment < kFixedSpCublasWorkspaceAlignment) {
        throw std::runtime_error(
          "cuBLAS user workspace alignment is below 256 bytes");
      }
    }
  } catch (...) {
    cleanup_new_allocations();
    throw;
  }

  const bool replace_double = new_double_arena != nullptr;
  const bool replace_int = new_int_arena != nullptr;
  const bool replace_host_status = new_host_status_arena != nullptr;
  const bool replace_pointer = new_pointer_arena != nullptr;
  const bool install_cublas_workspace = new_cublas_workspace != nullptr;
  bool old_owned_resource_consumed = false;
  auto free_old_cuda_owner = [&](auto** pointer, const char* stage) {
    const bool had_owner = pointer != nullptr && *pointer != nullptr;
    const TrackedTeardownResult<cudaError_t> result =
      tracked_cuda_free_noexcept(resource_ledger.get(), pointer);
    if (had_owner &&
        (result.disposition == TrackedTeardownDisposition::CalledSuccess ||
         result.disposition ==
           TrackedTeardownDisposition::CalledOwnershipIndeterminate)) {
      old_owned_resource_consumed = true;
    }
    if (result.disposition ==
        TrackedTeardownDisposition::NotAttemptedRetryable) {
      throw std::runtime_error(
        "injected tracked CUDA device free failure");
    }
    check_cuda(result.status, stage);
  };
  auto free_old_cuda_host_owner = [&](auto** pointer, const char* stage) {
    const bool had_owner = pointer != nullptr && *pointer != nullptr;
    const TrackedTeardownResult<cudaError_t> result =
      tracked_cuda_free_host_noexcept(resource_ledger.get(), pointer);
    if (had_owner &&
        (result.disposition == TrackedTeardownDisposition::CalledSuccess ||
         result.disposition ==
           TrackedTeardownDisposition::CalledOwnershipIndeterminate)) {
      old_owned_resource_consumed = true;
    }
    check_cuda(result.status, stage);
  };
  try {
    if (replace_double) {
      free_old_cuda_owner(
        &double_arena, "free old fixed-sp double arena");
      double_arena = new_double_arena;
      new_double_arena = nullptr;
    }
    if (replace_int) {
      free_old_cuda_owner(&int_arena, "free old fixed-sp int arena");
      int_arena = new_int_arena;
      new_int_arena = nullptr;
    }
    if (replace_host_status) {
      free_old_cuda_host_owner(
        &host_status_arena,
        "free old fixed-sp host status arena");
      host_status_arena = new_host_status_arena;
      new_host_status_arena = nullptr;
    }
    if (replace_pointer) {
      free_old_cuda_owner(
        &pointer_arena,
        "free old fixed-sp pointer arena");
      pointer_arena = new_pointer_arena;
      new_pointer_arena = nullptr;
    }
    if (install_cublas_workspace) {
      check_cublas(cublasSetStream(blas, stream), "rebind cuBLAS stream");
      check_cublas(cublasSetWorkspace(
        blas, new_cublas_workspace, cublas_workspace_bytes
      ), "install cuBLAS workspace");
      cublas_workspace = new_cublas_workspace;
      new_cublas_workspace = nullptr;
      diagnostics.cublas_workspace_alignment = new_cublas_alignment;
      diagnostics.cublas_user_workspace_installed = true;
    }
  } catch (...) {
    if (old_owned_resource_consumed) {
      lifecycle = FixedSpOwnerLifecycle::TeardownOnly;
    }
    cleanup_new_allocations();
    throw;
  }

  if (replace_double) double_capacity = double_required;
  if (replace_int) int_capacity = int_required;
  if (replace_host_status) host_status_capacity = host_status_required;
  if (replace_pointer) pointer_capacity = pointer_required;
  const bool grew = replace_double || replace_int || replace_host_status ||
    replace_pointer || install_cublas_workspace;

  stable_workspace = fixed_sp_stable_workspace_view(
    double_arena + stable_workspace_offset,
    int_arena + stable_info_offset,
    requested_stable_workspace.max_rows,
    requested_stable_workspace.max_q,
    requested_stable_workspace.eigen_lwork,
    requested_stable_workspace.qr_lwork,
    requested_stable_workspace.ormqr_lwork,
    requested_stable_workspace.svd_lwork);

  capacities = merged_capacities;
  potrf_lwork = std::max(potrf_lwork, requested_potrf_lwork);

  std::size_t workspace_bytes =
    allocation_bytes(double_capacity, sizeof(double), "workspace diagnostics");
  workspace_bytes = checked_add(
    workspace_bytes,
    allocation_bytes(int_capacity, sizeof(int), "workspace diagnostics"),
    "workspace diagnostics");
  workspace_bytes = checked_add(
    workspace_bytes,
    allocation_bytes(host_status_capacity, sizeof(int),
                     "workspace diagnostics"),
    "workspace diagnostics");
  workspace_bytes = checked_add(
    workspace_bytes,
    allocation_bytes(pointer_capacity, sizeof(void*),
                     "workspace diagnostics"),
    "workspace diagnostics");
  diagnostics.workspace_bytes = workspace_bytes;
  diagnostics.cublas_workspace_bytes =
    diagnostics.cublas_user_workspace_installed ? cublas_workspace_bytes : 0;
  diagnostics.eigen_workspace_bytes = allocation_bytes(
    static_cast<std::size_t>(stable_workspace.eigen_lwork), sizeof(double),
    "eigensolver workspace diagnostics");
  diagnostics.qr_workspace_bytes = allocation_bytes(
    static_cast<std::size_t>(std::max(
      stable_workspace.qr_lwork, stable_workspace.ormqr_lwork)),
    sizeof(double), "QR workspace diagnostics");
  diagnostics.svd_workspace_bytes = allocation_bytes(
    static_cast<std::size_t>(stable_workspace.svd_lwork), sizeof(double),
    "SVD workspace diagnostics");
  diagnostics.augmented_workspace_bytes = allocation_bytes(
    checked_multiply(
      static_cast<std::size_t>(stable_workspace.max_rows),
      static_cast<std::size_t>(stable_workspace.max_q),
      "augmented workspace diagnostics"),
    sizeof(double), "augmented workspace diagnostics");
  diagnostics.aggregate_factor_workspace_bytes = allocation_bytes(
    checked_add(
      checked_multiply(
        static_cast<std::size_t>(stable_workspace.max_q),
        static_cast<std::size_t>(stable_workspace.max_q),
        "aggregate factor workspace diagnostics"),
      checked_multiply(
        static_cast<std::size_t>(stable_workspace.max_q), 2U,
        "aggregate factor work diagnostics"),
      "aggregate factor workspace diagnostics"),
    sizeof(double), "aggregate factor workspace diagnostics");
  diagnostics.workspace_reserve_count += 1;
  if (grew) diagnostics.workspace_grow_count += 1;
}

FixedSpRuntimeInfo CudaRuntimeContext::info() const {
  std::lock_guard<std::mutex> lock(mutex);
  require_usable();
  FixedSpRuntimeInfo result = diagnostics;
  copy_resource_counters(resource_counters_snapshot(resource_ledger), &result);
  return result;
}

std::shared_ptr<CudaRuntimeContext> create_fixed_sp_runtime(int device_id) {
  return std::make_shared<CudaRuntimeContext>(device_id);
}

void reserve_fixed_sp_runtime(
    const std::shared_ptr<CudaRuntimeContext>& context,
    const FixedSpCapacities& capacities) {
  if (!context) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
  context->reserve(capacities);
}

FixedSpRuntimeInfo fixed_sp_runtime_info(
    const std::shared_ptr<CudaRuntimeContext>& context) {
  if (!context) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
  return context->info();
}

void free_fixed_sp_runtime(std::shared_ptr<CudaRuntimeContext>* context) {
  if (context == nullptr || !*context) return;
  const std::shared_ptr<CudaRuntimeContext> owned_context = *context;
  OwnerTeardownStatus status;
  {
    std::lock_guard<std::mutex> lock(owned_context->mutex);
    owned_context->require_explicit_close_identity();
    if (owned_context.use_count() > 2) {
      context->reset();
      return;
    }
    status = owned_context->close_once_noexcept();
    if (status.ownership_indeterminate &&
        owned_context->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
      status.merge(owned_context->close_once_noexcept());
    }
  }
  if (status.ownership_indeterminate) {
    context->reset();
    throw std::runtime_error(
      "fixed-sp CUDA runtime teardown ownership is indeterminate");
  }
  if (status.not_attempted_retryable) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime has retryable teardown work");
  }
  context->reset();
}

FixedSpResourceSnapshot test_fixed_sp_cuda_resource_snapshot() {
  auto snapshot_one = [](const AtomicResourceLifecycleCounters& counters) {
    FixedSpResourceLifecycleSnapshot snapshot;
    snapshot.acquire_attempt_count = counters.acquire_attempt_count.load(
      std::memory_order_relaxed);
    snapshot.acquire_success_count = counters.acquire_success_count.load(
      std::memory_order_relaxed);
    snapshot.acquire_failure_count = counters.acquire_failure_count.load(
      std::memory_order_relaxed);
    snapshot.teardown_attempt_count = counters.teardown_attempt_count.load(
      std::memory_order_relaxed);
    snapshot.teardown_success_count = counters.teardown_success_count.load(
      std::memory_order_relaxed);
    snapshot.teardown_failure_count = counters.teardown_failure_count.load(
      std::memory_order_relaxed);
    snapshot.active_count = counters.active_count.load(
      std::memory_order_relaxed);
    snapshot.ownership_indeterminate_count =
      counters.ownership_indeterminate_count.load(std::memory_order_relaxed);
    return snapshot;
  };

  const FixedSpGlobalResourceLedger& ledger = global_resource_ledger();
  FixedSpResourceSnapshot snapshot;
  snapshot.cuda_device = snapshot_one(ledger.cuda_device);
  snapshot.cuda_host = snapshot_one(ledger.cuda_host);
  snapshot.stream = snapshot_one(ledger.stream);
  snapshot.event = snapshot_one(ledger.event);
  snapshot.cublas_handle = snapshot_one(ledger.cublas_handle);
  snapshot.cusolver_handle = snapshot_one(ledger.cusolver_handle);
  snapshot.gesvdj_info = snapshot_one(ledger.gesvdj_info);
  snapshot.cleanup_error_count = ledger.cleanup_error_count.load(
    std::memory_order_relaxed);
  return snapshot;
}

void test_inject_next_fixed_sp_cuda_resource_acquire_failure(
    const std::string& resource) {
  const FixedSpResourceKind kind = fixed_sp_resource_kind_from_name(resource);
  int expected = -1;
  if (!global_resource_ledger().inject_next_acquire_failure_kind
         .compare_exchange_strong(expected, static_cast<int>(kind),
                                  std::memory_order_acq_rel)) {
    throw std::runtime_error(
      "tracked fixed-sp resource acquire failure injection is already pending");
  }
}

void test_inject_next_fixed_sp_cuda_resource_teardown_failure(
    const std::string& resource) {
  arm_injected_resource_teardown_failure(
    fixed_sp_resource_kind_from_name(resource));
}

void test_inject_next_fixed_sp_cuda_resource_post_call_teardown_failure(
    const std::string& resource) {
  arm_injected_resource_post_call_teardown_failure(
    fixed_sp_resource_kind_from_name(resource));
}

void test_inject_next_prepared_static_shadow_body_failure() {
  bool expected = false;
  if (!global_resource_ledger()
         .inject_next_prepared_static_shadow_body_failure
         .compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    throw std::runtime_error(
      "prepared static-shadow body failure injection is already pending");
  }
}

void test_exercise_fixed_sp_cuda_resource_teardown_failure(
    const std::string& resource) {
  const FixedSpResourceKind kind = fixed_sp_resource_kind_from_name(resource);
  FixedSpResourceLedger ledger;
  check_cuda(cudaSetDevice(0), "set CUDA device for teardown failure test");
  const bool post_call_mode =
    has_injected_resource_post_call_teardown_failure(kind);
  if (!post_call_mode) arm_injected_resource_teardown_failure(kind);

  auto require_first = [&](TrackedTeardownDisposition disposition,
                           bool has_raw_owner) {
    const bool valid = post_call_mode ?
      disposition ==
        TrackedTeardownDisposition::CalledOwnershipIndeterminate &&
        !has_raw_owner :
      disposition == TrackedTeardownDisposition::NotAttemptedRetryable &&
        has_raw_owner;
    if (!valid) {
      throw std::runtime_error(
        "tracked fixed-sp teardown first-attempt ownership invariant failed");
    }
  };
  auto require_retry = [&](TrackedTeardownDisposition disposition,
                           bool has_raw_owner) {
    if (disposition != TrackedTeardownDisposition::CalledSuccess ||
        has_raw_owner) {
      throw std::runtime_error(
        "tracked fixed-sp teardown retry ownership invariant failed");
    }
  };
  switch (kind) {
    case FixedSpResourceKind::CudaDevice: {
      void* pointer = nullptr;
      tracked_cuda_malloc(
        &ledger, &pointer, 1U, "allocate teardown test device byte");
      const TrackedTeardownResult<cudaError_t> first =
        tracked_cuda_free_noexcept(&ledger, &pointer);
      require_first(first.disposition, pointer != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cudaError_t> second =
          tracked_cuda_free_noexcept(&ledger, &pointer);
        require_retry(second.disposition, pointer != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::CudaHost: {
      void* pointer = nullptr;
      tracked_cuda_malloc_host(
        &ledger, &pointer, 1U, "allocate teardown test host byte");
      const TrackedTeardownResult<cudaError_t> first =
        tracked_cuda_free_host_noexcept(&ledger, &pointer);
      require_first(first.disposition, pointer != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cudaError_t> second =
          tracked_cuda_free_host_noexcept(&ledger, &pointer);
        require_retry(second.disposition, pointer != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::Stream: {
      cudaStream_t stream = nullptr;
      tracked_cuda_stream_create(
        &ledger, &stream, cudaStreamNonBlocking,
        "create teardown test stream");
      const TrackedTeardownResult<cudaError_t> first =
        tracked_cuda_stream_destroy_noexcept(&ledger, &stream);
      require_first(first.disposition, stream != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cudaError_t> second =
          tracked_cuda_stream_destroy_noexcept(&ledger, &stream);
        require_retry(second.disposition, stream != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::Event: {
      cudaEvent_t event = nullptr;
      tracked_cuda_event_create(
        &ledger, &event, cudaEventDisableTiming,
        "create teardown test event");
      const TrackedTeardownResult<cudaError_t> first =
        tracked_cuda_event_destroy_noexcept(&ledger, &event);
      require_first(first.disposition, event != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cudaError_t> second =
          tracked_cuda_event_destroy_noexcept(&ledger, &event);
        require_retry(second.disposition, event != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::CublasHandle: {
      cublasHandle_t handle = nullptr;
      tracked_cublas_create(
        &ledger, &handle, "create teardown test cuBLAS handle");
      const TrackedTeardownResult<cublasStatus_t> first =
        tracked_cublas_destroy_noexcept(&ledger, &handle);
      require_first(first.disposition, handle != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cublasStatus_t> second =
          tracked_cublas_destroy_noexcept(&ledger, &handle);
        require_retry(second.disposition, handle != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::CusolverHandle: {
      cusolverDnHandle_t handle = nullptr;
      tracked_cusolver_create(
        &ledger, &handle, "create teardown test cuSOLVER handle");
      const TrackedTeardownResult<cusolverStatus_t> first =
        tracked_cusolver_destroy_noexcept(&ledger, &handle);
      require_first(first.disposition, handle != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cusolverStatus_t> second =
          tracked_cusolver_destroy_noexcept(&ledger, &handle);
        require_retry(second.disposition, handle != nullptr);
      }
      break;
    }
    case FixedSpResourceKind::GesvdjInfo: {
      gesvdjInfo_t info = nullptr;
      tracked_gesvdj_info_create(
        &ledger, &info, "create teardown test gesvdj info");
      const TrackedTeardownResult<cusolverStatus_t> first =
        tracked_gesvdj_info_destroy_noexcept(&ledger, &info);
      require_first(first.disposition, info != nullptr);
      if (!post_call_mode) {
        const TrackedTeardownResult<cusolverStatus_t> second =
          tracked_gesvdj_info_destroy_noexcept(&ledger, &info);
        require_retry(second.disposition, info != nullptr);
      }
      break;
    }
  }
}

void test_inject_next_fixed_sp_cuda_device_free_failure() {
  arm_injected_resource_teardown_failure(FixedSpResourceKind::CudaDevice);
}

int test_fixed_sp_cuda_device_count() {
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count),
             "query fixed-sp CUDA test device count");
  return device_count;
}

void test_fixed_sp_cuda_set_device(int device_id) {
  if (device_id < 0) {
    throw std::runtime_error(
      "fixed-sp CUDA test device id must be non-negative");
  }
  check_cuda(cudaSetDevice(device_id), "set fixed-sp CUDA test device");
}

int test_fixed_sp_cuda_get_device() {
  int device_id = -1;
  check_cuda(cudaGetDevice(&device_id), "get fixed-sp CUDA test device");
  return device_id;
}

void test_inject_next_blocked_consumer_launch_failure() {
  bool expected = false;
  if (!global_resource_ledger().inject_next_blocked_consumer_launch_failure
         .compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    throw std::runtime_error(
      "blocked-consumer launch failure injection is already pending");
  }
}

void test_force_next_fixed_sp_cuda_potrf_info(
    const std::vector<int>& info) {
  FixedSpBatchedInfoTestState& state = batched_info_test_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (info.empty()) {
    state.potrf_armed = false;
    state.potrf_info.clear();
    return;
  }
  if (state.potrf_armed) {
    throw std::runtime_error(
      "Phase 3B forced potrf info is already pending");
  }
  state.potrf_info = info;
  state.potrf_armed = true;
}

void test_force_next_fixed_sp_cuda_potrs_info(int info) {
  FixedSpBatchedInfoTestState& state = batched_info_test_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (info == 0) {
    state.potrs_armed = false;
    state.potrs_info = 0;
    return;
  }
  if (state.potrs_armed) {
    throw std::runtime_error(
      "Phase 3B forced potrs info is already pending");
  }
  state.potrs_info = info;
  state.potrs_armed = true;
}

static std::shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu_impl(
    const std::shared_ptr<CudaRuntimeContext>& context,
    const PreparedSHostView& setup,
    bool transient_fixed_sp_compatibility) {
  if (!context) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
  std::lock_guard<std::mutex> lock(context->mutex);
  context->require_usable();
  validate_prepared_host_view(setup);
  if (context->diagnostics.workspace_reserve_count == 0) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime must reserve workspace before prepared setup");
  }
  if (setup.n > context->capacities.n ||
      setup.null_dim > context->capacities.null_dim ||
      setup.penalty_count > context->capacities.penalty_count) {
    throw std::runtime_error(
      "prepared setup dimensions exceed reserved CUDA runtime capacity");
  }

  const std::vector<double> nullspace_design =
    build_nullspace_design(setup);
  const std::vector<double> projected_penalties =
    build_projected_penalties(setup);
  const std::vector<double> projected_H = build_projected_H(setup);
  require_finite_derived(nullspace_design, "X_null");
  require_finite_derived(projected_penalties, "projected penalties");
  require_finite_derived(projected_H, "projected H");
  const std::size_t x_count =
    matrix_count(setup.n, setup.coefficient_dim, "prepared X");
  const std::size_t z_count = setup.Z == nullptr ? 0U :
    matrix_count(setup.coefficient_dim, setup.null_dim, "prepared Z");
  const std::size_t x_null_count = setup.Z == nullptr ? 0U :
    matrix_count(setup.n, setup.null_dim, "prepared nullspace design");
  const std::size_t gram_count =
    matrix_count(setup.null_dim, setup.null_dim, "prepared Gram");
  const std::size_t penalty_count = projected_penalties.size();
  const std::vector<int> penalty_root_ranks =
    transient_fixed_sp_compatibility ?
      std::vector<int>(static_cast<std::size_t>(setup.penalty_count), 0) :
      setup.penalty_ranks;
  std::size_t total_penalty_root_rows = 0U;
  std::vector<int> penalty_root_offsets;
  penalty_root_offsets.reserve(penalty_root_ranks.size());
  for (int rank : penalty_root_ranks) {
    if (total_penalty_root_rows >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("prepared penalty root row count overflow");
    }
    penalty_root_offsets.push_back(
      static_cast<int>(total_penalty_root_rows));
    total_penalty_root_rows = checked_add(
      total_penalty_root_rows, static_cast<std::size_t>(rank),
      "prepared penalty root rows");
  }
  if (total_penalty_root_rows >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("prepared penalty root row count overflow");
  }
  const std::size_t penalty_root_count = checked_multiply(
    total_penalty_root_rows,
    positive_size(setup.null_dim, "prepared root null dimension"),
    "prepared penalty roots");
  const bool has_H = setup.H != nullptr;
  const std::size_t validation_count = transient_fixed_sp_compatibility ?
    1U : checked_add(
      positive_size(setup.penalty_count, "prepared penalty count"),
      has_H ? 1U : 0U, "prepared root validation records");
  const std::size_t target_capacity = positive_size(
    context->capacities.target_count, "reserved target capacity");
  const std::size_t coefficient_output_count = checked_multiply(
    positive_size(setup.coefficient_dim, "prepared coefficient dimension"),
    target_capacity, "transient coefficient slot");
  const std::size_t observation_output_count = checked_multiply(
    positive_size(setup.n, "prepared row count"), target_capacity,
    "transient observation slot");
  const std::size_t rhs_output_count = checked_multiply(
    positive_size(setup.null_dim, "prepared null dimension"), target_capacity,
    "transient RHS slot");

  auto handle = std::make_shared<PreparedSGpuHandle>();
  handle->context = context;
  handle->creator_pid = context->creator_pid;
  handle->device_id = context->device_id;
  handle->context_generation = context->generation;
  handle->prepared_s_key_sha256 = setup.prepared_s_key_sha256;
  handle->n = setup.n;
  handle->p = setup.coefficient_dim;
  handle->q = setup.null_dim;
  handle->penalty_count = setup.penalty_count;
  handle->penalty_root_offsets = std::move(penalty_root_offsets);
  handle->penalty_root_ranks = penalty_root_ranks;
  handle->total_penalty_root_rows =
    static_cast<int>(total_penalty_root_rows);
  handle->has_H = has_H;
  handle->transient_fixed_sp_compatibility =
    transient_fixed_sp_compatibility;
  handle->penalty_root_bytes = allocation_bytes(
    penalty_root_count, sizeof(double), "prepared penalty root diagnostics");
  handle->resource_ledger = context->resource_ledger;
  handle->residual_slot = std::make_shared<TransientResidualSlot>();
  handle->residual_slot->resource_ledger = handle->resource_ledger;
  handle->residual_slot->creator_pid = context->creator_pid;
  handle->residual_slot->device_id = context->device_id;
  handle->residual_slot->target_capacity = context->capacities.target_count;
  handle->residual_slot->finite_status_capacity = target_capacity;
  handle->residual_slot->coefficient_capacity = coefficient_output_count;
  handle->residual_slot->rhs_capacity = rhs_output_count;

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for prepared setup");
  FixedSpRootValidationRecord* d_root_validation = nullptr;
  FixedSpRootValidationRecord* host_root_validation = nullptr;
  try {
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->d_X,
      allocation_bytes(x_count, sizeof(double), "prepared X"),
      "allocate prepared X");
    if (setup.Z == nullptr) {
      handle->d_X_null = handle->d_X;
    } else {
      tracked_cuda_malloc(
        context->resource_ledger.get(),
        &handle->d_Z,
        allocation_bytes(z_count, sizeof(double), "prepared Z"),
        "allocate prepared Z");
      tracked_cuda_malloc(
        context->resource_ledger.get(),
        &handle->d_X_null,
        allocation_bytes(
          x_null_count, sizeof(double), "prepared nullspace design"),
        "allocate prepared nullspace design");
    }
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->d_gram,
      allocation_bytes(gram_count, sizeof(double), "prepared Gram"),
      "allocate prepared Gram");
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->d_projected_penalties,
      allocation_bytes(
        penalty_count, sizeof(double), "prepared projected penalties"),
      "allocate prepared projected penalties");
    if (penalty_root_count > 0U) {
      tracked_cuda_malloc(
        context->resource_ledger.get(), &handle->d_penalty_roots,
        handle->penalty_root_bytes, "allocate prepared penalty roots");
    }
    if (has_H) {
      tracked_cuda_malloc(
        context->resource_ledger.get(), &handle->d_projected_H,
        allocation_bytes(gram_count, sizeof(double), "prepared projected H"),
        "allocate prepared projected H");
      tracked_cuda_malloc(
        context->resource_ledger.get(), &handle->d_H_root,
        allocation_bytes(gram_count, sizeof(double), "prepared H root"),
        "allocate prepared H root");
    }
    const std::size_t validation_bytes = allocation_bytes(
      validation_count, sizeof(FixedSpRootValidationRecord),
      "prepared root validation records");
    tracked_cuda_malloc(
      context->resource_ledger.get(), &d_root_validation,
      validation_bytes, "allocate prepared root validation records");
    tracked_cuda_malloc_host(
      context->resource_ledger.get(), &host_root_validation,
      validation_bytes, "allocate prepared root host validation records");

    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->residual_slot->coefficients,
      allocation_bytes(
        coefficient_output_count, sizeof(double),
        "transient coefficient slot"),
      "allocate transient coefficient slot");
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->residual_slot->fitted,
      allocation_bytes(
        observation_output_count, sizeof(double), "transient fitted slot"),
      "allocate transient fitted slot");
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->residual_slot->residuals,
      allocation_bytes(
        observation_output_count, sizeof(double), "transient residual slot"),
      "allocate transient residual slot");
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->residual_slot->rss,
      allocation_bytes(target_capacity, sizeof(double), "transient RSS slot"),
      "allocate transient RSS slot");
    tracked_cuda_malloc(
      context->resource_ledger.get(),
      &handle->residual_slot->rhs,
      allocation_bytes(
        rhs_output_count, sizeof(double), "transient RHS slot"),
      "allocate transient RHS slot");
    tracked_cuda_malloc_host(
      context->resource_ledger.get(),
      &handle->residual_slot->host_finite_status,
      allocation_bytes(
        target_capacity, sizeof(int), "transient finite status slot"),
      "allocate transient finite status slot");
    tracked_cuda_event_create(
      context->resource_ledger.get(),
      &handle->residual_slot->solve_completion_event,
      cudaEventDisableTiming, "create transient solve completion event");
    tracked_cuda_event_create(
      context->resource_ledger.get(),
      &handle->residual_slot->consumer_completion_event,
      cudaEventDisableTiming, "create transient consumer completion event");
    tracked_cuda_event_create(
      context->resource_ledger.get(), &handle->setup_completion_event,
      cudaEventDisableTiming, "create prepared setup completion event");

    auto upload = [&](double* destination, const double* source,
                      std::size_t count, const char* name) {
      const std::size_t bytes = allocation_bytes(count, sizeof(double), name);
      check_cuda(cudaMemcpyAsync(
        destination, source, bytes, cudaMemcpyHostToDevice, context->stream
      ), name);
      handle->setup_h2d_bytes = checked_add(
        handle->setup_h2d_bytes, bytes, "prepared setup H2D diagnostics");
    };
    upload(handle->d_X, setup.X, x_count, "upload prepared X");
    if (setup.Z != nullptr) {
      upload(handle->d_Z, setup.Z, z_count, "upload prepared Z");
      upload(handle->d_X_null, nullspace_design.data(), x_null_count,
             "upload prepared nullspace design");
    }
    upload(handle->d_gram, setup.gram, gram_count,
           "upload prepared Gram");
    upload(handle->d_projected_penalties, projected_penalties.data(),
           penalty_count, "upload prepared projected penalties");
    if (has_H) {
      upload(handle->d_projected_H, projected_H.data(), gram_count,
             "upload prepared projected H");
    }

    if (!transient_fixed_sp_compatibility) {
    FixedSpStableWorkspace& stable = context->stable_workspace;
    if (stable.scaled_projection == nullptr || stable.diagonal == nullptr ||
        stable.eigen_work == nullptr || stable.info == nullptr ||
        stable.eigen_lwork <= 0 || stable.max_q < setup.null_dim) {
      throw std::runtime_error(
        "prepared penalty root eigensolver workspace is unavailable");
    }
    const std::size_t square_bytes = allocation_bytes(
      gram_count, sizeof(double), "prepared root eigensolver matrix");
    const auto root_build_start = std::chrono::steady_clock::now();
    auto decompose_and_build = [&](int validation_index,
                                   double* destination,
                                   int leading_dimension,
                                   int row_offset,
                                   int row_capacity,
                                   int expected_rank,
                                   const char* stage) {
      check_cusolver(cusolverDnDsyevd(
        context->solver, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, setup.null_dim,
        stable.scaled_projection, setup.null_dim, stable.diagonal,
        stable.eigen_work, stable.eigen_lwork, stable.info
      ), stage);
      launch_fixed_sp_root_build(
        stable.scaled_projection, stable.diagonal, stable.info,
        setup.null_dim, destination, leading_dimension, row_offset,
        row_capacity, expected_rank, std::numeric_limits<double>::epsilon(),
        d_root_validation + validation_index, context->stream);
    };

    for (int penalty = 0; penalty < setup.penalty_count; ++penalty) {
      check_cuda(cudaMemcpyAsync(
        stable.scaled_projection,
        handle->d_projected_penalties +
          static_cast<std::size_t>(penalty) * gram_count,
        square_bytes, cudaMemcpyDeviceToDevice, context->stream
      ), "copy projected smooth penalty to eigensolver scratch");
      decompose_and_build(
        penalty, handle->d_penalty_roots,
        handle->total_penalty_root_rows,
        handle->penalty_root_offsets[static_cast<std::size_t>(penalty)],
        handle->penalty_root_ranks[static_cast<std::size_t>(penalty)],
        handle->penalty_root_ranks[static_cast<std::size_t>(penalty)],
        "decompose projected smooth penalty");
    }
    if (has_H) {
      check_cuda(cudaMemcpyAsync(
        stable.scaled_projection, handle->d_projected_H, square_bytes,
        cudaMemcpyDeviceToDevice, context->stream
      ), "copy projected H to eigensolver scratch");
      decompose_and_build(
        setup.penalty_count, handle->d_H_root, setup.null_dim, 0,
        setup.null_dim, -1, "decompose projected H");
    }

    check_cuda(cudaMemcpyAsync(
      host_root_validation, d_root_validation, validation_bytes,
      cudaMemcpyDeviceToHost, context->stream
    ), "download prepared root validation records");
    check_cuda(cudaEventRecord(
      handle->setup_completion_event, context->stream
    ), "record prepared setup completion");
    check_cuda(cudaEventSynchronize(handle->setup_completion_event),
               "wait for prepared setup completion");

    auto validate_root_record = [&](int index, const char* name,
                                    int expected_rank) {
      const FixedSpRootValidationRecord& record =
        host_root_validation[index];
      if (record.solver_info != 0) {
        throw std::runtime_error(
          std::string(name) + " eigendecomposition failed with info " +
          std::to_string(record.solver_info));
      }
      if (record.finite == 0) {
        throw std::runtime_error(
          std::string(name) + " eigendecomposition is nonfinite");
      }
      if (record.ascending == 0) {
        throw std::runtime_error(
          std::string(name) + " eigenvalues are not ascending");
      }
      if (record.psd == 0) {
        throw std::runtime_error(std::string("non-PSD ") + name);
      }
      if (expected_rank >= 0 && record.rank != expected_rank) {
        handle->penalty_root_rank_mismatch_count += 1;
        throw std::runtime_error(
          std::string(name) + " root rank mismatch: expected " +
          std::to_string(expected_rank) + ", derived " +
          std::to_string(record.rank));
      }
      return record.rank;
    };
    for (int penalty = 0; penalty < setup.penalty_count; ++penalty) {
      validate_root_record(
        penalty, "smooth penalty",
        handle->penalty_root_ranks[static_cast<std::size_t>(penalty)]);
    }
    if (has_H) {
      handle->H_root_rank = validate_root_record(
        setup.penalty_count, "H", -1);
    }
    const auto root_build_end = std::chrono::steady_clock::now();
    handle->penalty_root_build_ms =
      std::chrono::duration<double, std::milli>(
        root_build_end - root_build_start).count();
    handle->penalty_root_build_count = 1;
    } else {
      check_cuda(cudaEventRecord(
        handle->setup_completion_event, context->stream
      ), "record transient compatibility setup completion");
      check_cuda(cudaEventSynchronize(handle->setup_completion_event),
                 "wait for transient compatibility setup completion");
    }
    handle->setup_h2d_upload_count = 1;
    tracked_cuda_free(
      context->resource_ledger.get(), &d_root_validation,
      "free prepared root validation records");
    tracked_cuda_free_host(
      context->resource_ledger.get(), &host_root_validation,
      "free prepared root host validation records");
  } catch (...) {
    cleanup_local_cuda_allocation_noexcept(
      context->resource_ledger.get(), &d_root_validation);
    cleanup_local_cuda_host_allocation_noexcept(
      context->resource_ledger.get(), &host_root_validation);
    handle->cleanup_noexcept();
    throw;
  }
  context->active_prepared_handle_count.fetch_add(
    1, std::memory_order_release);
  handle->registered_with_context = true;
  return handle;
}

std::shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu(
    const std::shared_ptr<CudaRuntimeContext>& context,
    const PreparedSHostView& setup) {
  return create_prepared_s_gpu_impl(context, setup, false);
}

std::shared_ptr<PreparedSGpuHandle>
create_transient_fixed_sp_compatibility_prepared_s_gpu(
    const std::shared_ptr<CudaRuntimeContext>& context,
    const TransientFixedSpCompatibilityHostView& view) {
  if (view.n <= 0 || view.null_dim <= 0 || view.X_null == nullptr ||
      view.gram == nullptr || view.projected_penalty == nullptr) {
    throw std::runtime_error(
      "transient fixed-sp compatibility view is malformed");
  }
  const std::size_t x_count = matrix_count(
    view.n, view.null_dim, "transient compatibility X_null");
  const std::size_t square_count = matrix_count(
    view.null_dim, view.null_dim, "transient compatibility square matrix");
  auto require_finite_view = [](const double* values, std::size_t count,
                                const char* name) {
    for (std::size_t index = 0; index < count; ++index) {
      if (!std::isfinite(values[index])) {
        throw std::runtime_error(std::string(name) + " must be finite");
      }
    }
  };
  require_finite_view(view.X_null, x_count,
                      "transient compatibility X_null");
  require_finite_view(view.gram, square_count,
                      "transient compatibility Gram");
  require_finite_view(view.projected_penalty, square_count,
                      "transient compatibility projected penalty");
  require_scale_aware_symmetric(
    view.gram, view.null_dim,
    "transient compatibility Gram must be symmetric");
  require_scale_aware_symmetric(
    view.projected_penalty, view.null_dim,
    "transient compatibility projected penalty must be symmetric");

  PreparedSHostView setup;
  setup.n = view.n;
  setup.coefficient_dim = view.null_dim;
  setup.null_dim = view.null_dim;
  setup.penalty_count = 1;
  setup.X = view.X_null;
  setup.Z = nullptr;
  setup.gram = view.gram;
  setup.H = nullptr;
  setup.penalty_blocks = {view.projected_penalty};
  setup.penalty_dimensions = {view.null_dim};
  setup.penalty_offsets_zero_based = {0};
  setup.penalty_ranks = {0};
  setup.penalty_sp_indices_zero_based = {0};
  return create_prepared_s_gpu_impl(context, setup, true);
}

PreparedSInfo prepared_s_gpu_info(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  context->require_usable();
  PreparedSInfo info;
  info.prepared_s_key_sha256 = handle->prepared_s_key_sha256;
  info.n = handle->n;
  info.coefficient_dim = handle->p;
  info.null_dim = handle->q;
  info.penalty_count = handle->penalty_count;
  info.setup_h2d_upload_count = handle->setup_h2d_upload_count;
  info.setup_h2d_bytes = handle->setup_h2d_bytes;
  info.penalty_root_build_count = handle->penalty_root_build_count;
  info.penalty_root_rank_mismatch_count =
    handle->penalty_root_rank_mismatch_count;
  info.penalty_root_bytes = handle->penalty_root_bytes;
  info.penalty_root_build_ms = handle->penalty_root_build_ms;
  info.penalty_root_matrix_count = handle->penalty_count;
  info.penalty_root_row_count = handle->total_penalty_root_rows;
  info.H_root_matrix_count = handle->has_H ? 1 : 0;
  info.H_root_rank = handle->H_root_rank;
  info.setup_shadow_d2h_count = handle->setup_shadow_d2h_count;
  info.setup_shadow_d2h_bytes = handle->setup_shadow_d2h_bytes;
  info.augmented_test_shadow_d2h_count =
    handle->augmented_test_shadow_d2h_count;
  info.augmented_test_shadow_d2h_bytes =
    handle->augmented_test_shadow_d2h_bytes;
  info.projected_H_test_shadow_d2h_count =
    handle->projected_H_test_shadow_d2h_count;
  info.projected_H_test_shadow_d2h_bytes =
    handle->projected_H_test_shadow_d2h_bytes;
  info.coefficient_output_capacity =
    handle->residual_slot == nullptr ? 0U :
      handle->residual_slot->coefficient_capacity;
  info.generation = handle->generation;
  info.output_slot_leased =
    handle->residual_slot != nullptr &&
    handle->residual_slot->state == TransientResidualSlotState::Leased;
  if (handle->residual_slot != nullptr) {
    info.output_slot_state = transient_residual_slot_state_name(
      handle->residual_slot->state);
    info.output_slot_poison_reason = handle->residual_slot->poison_reason;
  }
  return info;
}

void free_prepared_s_gpu(std::shared_ptr<PreparedSGpuHandle>* handle) {
  if (handle == nullptr || !*handle) return;
  const std::shared_ptr<PreparedSGpuHandle> owned_handle = *handle;
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_close_identity(owned_handle);
  if (!context) {
    handle->reset();
    return;
  }
  OwnerTeardownStatus status;
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    require_prepared_close_identity(owned_handle);
    if (owned_handle.use_count() > 2) {
      handle->reset();
      return;
    }
    status = owned_handle->close_once_noexcept();
    if (status.ownership_indeterminate &&
        owned_handle->lifecycle == FixedSpOwnerLifecycle::TeardownOnly) {
      status.merge(owned_handle->close_once_noexcept());
    }
  }
  if (status.ownership_indeterminate) {
    handle->reset();
    throw std::runtime_error(
      "fixed-sp CUDA prepared teardown ownership is indeterminate");
  }
  if (status.not_attempted_retryable) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle has retryable teardown work");
  }
  handle->reset();
}

std::shared_ptr<DeviceResidualBatch> solve_fixed_sp_batch(
    const std::shared_ptr<PreparedSGpuHandle>& handle,
    const FixedSpBatchHostView& batch) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);

  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  const std::shared_ptr<TransientResidualSlot> slot = handle->residual_slot;
  if (slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*slot);
  }
  if (slot->state != TransientResidualSlotState::Free ||
      slot->consumer_event_registered) {
    increment_active_slot_busy(slot.get());
    throw std::runtime_error("ERR_OUTPUT_SLOT_BUSY");
  }
  context->require_usable();
  validate_fixed_sp_batch(*handle, *context, batch);
  const FixedSpResourceCounters resources_before_solve =
    resource_counters_snapshot(context->resource_ledger);

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for fixed-sp solve");
  slot->generation += 1;
  const std::uint64_t acquired_generation = slot->generation;
  slot->state = TransientResidualSlotState::Leased;
  slot->consumer_event_registered = false;

  std::shared_ptr<DeviceResidualBatch> token;
  auto restore_slot = [&]() noexcept {
    if (slot->generation == acquired_generation &&
        slot->state == TransientResidualSlotState::Leased) {
      slot->state = TransientResidualSlotState::Free;
      slot->consumer_event_registered = false;
      slot->lease_owner.reset();
    }
    if (token) token->lease_released = true;
  };

  try {
    token = std::make_shared<DeviceResidualBatch>();
    token->owner = handle;
    token->creator_pid = handle->creator_pid;
    token->device_id = handle->device_id;
    token->owner_generation = handle->generation;
    token->slot_generation = acquired_generation;
    token->n = batch.n;
    token->target_count = batch.target_count;
    token->output_mask = batch.output_mask;
    token->target_keys = batch.target_keys;
    token->slot = slot;
    token->planned_routes = batch.planned_routes;
    token->executed_routes.assign(
      static_cast<std::size_t>(batch.target_count), FixedSpRoute::Unset);
    token->reroute_reasons.assign(
      static_cast<std::size_t>(batch.target_count), std::string());
    token->solver_statuses.assign(
      static_cast<std::size_t>(batch.target_count),
      FixedSpStatus::ErrStablePathNotImplemented);
    token->target_true_batched.assign(
      static_cast<std::size_t>(batch.target_count), false);

    token->diagnostics.n = batch.n;
    token->diagnostics.coefficient_dim = handle->p;
    token->diagnostics.null_dim = handle->q;
    token->diagnostics.target_count = batch.target_count;
    token->diagnostics.native_batch_call = true;
    token->diagnostics.batch_call_count = 1;
    token->diagnostics.output_slot_acquire_count = 1;
    token->diagnostics.owner_generation = handle->generation;
    token->diagnostics.slot_generation = acquired_generation;
    token->diagnostics.qr_rank.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.geqrf_info.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.ormqr_info.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.effective_rank.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.sigma_max.assign(
      static_cast<std::size_t>(batch.target_count),
      std::numeric_limits<double>::quiet_NaN());
    token->diagnostics.smallest_retained_sigma.assign(
      static_cast<std::size_t>(batch.target_count),
      std::numeric_limits<double>::quiet_NaN());
    token->diagnostics.svd_info.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.aggregate_penalty_root_rank.assign(
      static_cast<std::size_t>(batch.target_count), -1);
    token->diagnostics.aggregate_factor_call_count.assign(
      static_cast<std::size_t>(batch.target_count), 0);
    token->diagnostics.aggregate_b_build_count.assign(
      static_cast<std::size_t>(batch.target_count), 0);
    token->diagnostics.aggregate_penalty_root_pivot.assign(
      checked_multiply(
        static_cast<std::size_t>(batch.target_count),
        static_cast<std::size_t>(handle->q),
        "fixed-sp aggregate pivot diagnostics"),
      -1);
    token->diagnostics.aggregate_dstop.assign(
      static_cast<std::size_t>(batch.target_count),
      std::numeric_limits<double>::quiet_NaN());
    token->diagnostics.target_true_batched.assign(
      static_cast<std::size_t>(batch.target_count), false);
    slot->lease_owner = token;

    const std::size_t targets =
      static_cast<std::size_t>(batch.target_count);
    const std::size_t coefficient_count =
      (batch.output_mask & FixedSpOutputCoefficients) == 0 ? 0U :
        checked_multiply(static_cast<std::size_t>(handle->p), targets,
                         "fixed-sp coefficient outputs");
    const std::size_t fitted_count =
      (batch.output_mask & FixedSpOutputFitted) == 0 ? 0U :
        checked_multiply(static_cast<std::size_t>(batch.n), targets,
                         "fixed-sp fitted outputs");
    const std::size_t residual_count =
      (batch.output_mask & FixedSpOutputResiduals) == 0 ? 0U :
        checked_multiply(static_cast<std::size_t>(batch.n), targets,
                         "fixed-sp residual outputs");
    const std::size_t rss_count =
      (batch.output_mask & FixedSpOutputRss) == 0 ? 0U : targets;
    const std::size_t rhs_count =
      (batch.output_mask & FixedSpOutputRhs) == 0 ? 0U :
        checked_multiply(static_cast<std::size_t>(handle->q), targets,
                         "fixed-sp RHS outputs");
    const std::size_t initialize_count = std::max(
      std::max(std::max(coefficient_count, fitted_count),
               std::max(residual_count, rss_count)), rhs_count);
    constexpr unsigned int threads = 256U;
    const std::size_t block_count =
      (initialize_count + threads - 1U) / threads;
    if (block_count == 0 ||
        block_count > std::numeric_limits<unsigned int>::max()) {
      throw std::runtime_error("fixed-sp output initialization size overflow");
    }
    initialize_fixed_sp_outputs_invalid<<<
      static_cast<unsigned int>(block_count), threads, 0, context->stream
    >>>(
      slot->coefficients, coefficient_count,
      slot->fitted, fitted_count,
      slot->residuals, residual_count,
      slot->rss, rss_count,
      slot->rhs, rhs_count);
    check_cuda(cudaGetLastError(),
               "launch fixed-sp invalid output initialization");
    token->diagnostics.invalid_output_init_count = 1;

    const std::size_t reserved_n = positive_size(
      context->capacities.n, "reserved fixed-sp row count");
    const std::size_t reserved_q = positive_size(
      context->capacities.null_dim, "reserved fixed-sp null dimension");
    const std::size_t reserved_targets = positive_size(
      context->capacities.target_count, "reserved fixed-sp target count");
    const std::size_t reserved_penalties = positive_size(
      context->capacities.penalty_count, "reserved fixed-sp penalty count");
    const std::size_t y_capacity = checked_multiply(
      reserved_n, reserved_targets, "reserved Y arena");
    const std::size_t sp_capacity = checked_multiply(
      reserved_penalties, reserved_targets, "reserved SP arena");
    const std::size_t rhs_capacity = checked_multiply(
      reserved_q, reserved_targets, "reserved RHS arena");
    const std::size_t reserved_q_squared = checked_multiply(
      reserved_q, reserved_q, "reserved system arena");
    const std::size_t system_capacity = checked_multiply(
      reserved_q_squared, reserved_targets, "reserved system arena");
    const std::size_t theta_capacity = checked_multiply(
      reserved_q, reserved_targets, "reserved theta arena");
    if (context->double_arena == nullptr || context->int_arena == nullptr ||
        context->host_status_arena == nullptr) {
      throw std::runtime_error("fixed-sp CUDA arena is unavailable");
    }
    double* d_y = context->double_arena;
    double* d_sp = d_y + y_capacity;
    double* d_rhs = d_sp + sp_capacity;
    double* d_systems = d_rhs + rhs_capacity;
    double* d_theta = d_systems + system_capacity;
    double* d_potrf_work = d_theta + theta_capacity;
    double* d_sigma_max = d_potrf_work +
      static_cast<std::size_t>(context->potrf_lwork);
    double* d_smallest_retained_sigma =
      d_sigma_max + reserved_targets;
    double* d_aggregate_dstop =
      d_smallest_retained_sigma + reserved_targets;
    if (d_aggregate_dstop + reserved_targets !=
        context->stable_workspace.B) {
      throw std::runtime_error(
        "fixed-sp stable double arena binding mismatch");
    }

    const std::size_t stable_index_offset = reserved_targets;
    const std::size_t potrs_info_offset = checked_multiply(
      reserved_targets, 3U, "fixed-sp potrs info offset");
    const std::size_t stable_compact_offset = checked_add(
      potrs_info_offset, 1U, "fixed-sp stable compact offset");
    const std::size_t stable_compact_width = checked_add(
      reserved_q,
      checked_add(
        kStableBaseIntArraysPerTarget,
        kStableAggregateIntArraysPerTarget,
        "fixed-sp stable compact width"),
      "fixed-sp stable compact width");
    const std::size_t stable_compact_count = checked_multiply(
      reserved_targets, stable_compact_width,
      "fixed-sp stable compact layout");
    const std::size_t stable_shared_info_offset = checked_add(
      stable_compact_offset, stable_compact_count,
      "fixed-sp stable shared info offset");
    const std::size_t int_layout_required = checked_add(
      stable_shared_info_offset, 1U, "fixed-sp integer arena layout");
    const std::size_t host_sigma_offset = checked_add(
      int_layout_required,
      int_layout_required % kIntsPerDouble == 0U ? 0U :
        kIntsPerDouble - int_layout_required % kIntsPerDouble,
      "fixed-sp host stable sigma alignment");
    const std::size_t host_layout_required = checked_add(
      host_sigma_offset,
      checked_multiply(
        checked_multiply(reserved_targets, kStableDoubleArraysPerTarget,
                         "fixed-sp host stable double layout"),
        kIntsPerDouble, "fixed-sp host stable double layout"),
      "fixed-sp host status layout");
    if (context->int_capacity < int_layout_required ||
        context->host_status_capacity < host_layout_required) {
      throw std::runtime_error(
        "fixed-sp integer arena does not satisfy stable compact layout");
    }
    int* d_safe_target_indices = context->int_arena;
    int* d_stable_target_indices =
      context->int_arena + stable_index_offset;
    int* d_potrf_info = d_stable_target_indices + reserved_targets;
    int* d_potrs_info = d_potrf_info + reserved_targets;
    int* d_geqrf_info = context->int_arena + stable_compact_offset;
    int* d_ormqr_info = d_geqrf_info + reserved_targets;
    int* d_qr_rank = d_ormqr_info + reserved_targets;
    int* d_qr_reroute = d_qr_rank + reserved_targets;
    int* d_qr_finite_status = d_qr_reroute + reserved_targets;
    int* d_svd_info = d_qr_finite_status + reserved_targets;
    int* d_aggregate_root_rank = d_svd_info + reserved_targets;
    int* d_aggregate_factor_call_count =
      d_aggregate_root_rank + reserved_targets;
    int* d_aggregate_b_build_count =
      d_aggregate_factor_call_count + reserved_targets;
    int* d_aggregate_pivots =
      d_aggregate_b_build_count + reserved_targets;
    int* h_safe_target_indices = context->host_status_arena;
    int* h_stable_target_indices =
      context->host_status_arena + stable_index_offset;
    int* h_potrf_info = h_stable_target_indices + reserved_targets;
    int* h_potrs_info = h_potrf_info + reserved_targets;
    int* h_geqrf_info = context->host_status_arena + stable_compact_offset;
    int* h_ormqr_info = h_geqrf_info + reserved_targets;
    int* h_qr_rank = h_ormqr_info + reserved_targets;
    int* h_qr_reroute = h_qr_rank + reserved_targets;
    int* h_qr_finite_status = h_qr_reroute + reserved_targets;
    int* h_svd_info = h_qr_finite_status + reserved_targets;
    int* h_aggregate_root_rank = h_svd_info + reserved_targets;
    int* h_aggregate_factor_call_count =
      h_aggregate_root_rank + reserved_targets;
    int* h_aggregate_b_build_count =
      h_aggregate_factor_call_count + reserved_targets;
    int* h_aggregate_pivots =
      h_aggregate_b_build_count + reserved_targets;
    double* h_sigma_max = reinterpret_cast<double*>(
      context->host_status_arena + host_sigma_offset);
    double* h_smallest_retained_sigma = h_sigma_max + reserved_targets;
    double* h_aggregate_dstop =
      h_smallest_retained_sigma + reserved_targets;
    if (d_aggregate_pivots +
          checked_multiply(reserved_targets, reserved_q,
                           "fixed-sp aggregate pivot layout") !=
          context->int_arena + stable_shared_info_offset ||
        h_aggregate_pivots +
          checked_multiply(reserved_targets, reserved_q,
                           "fixed-sp host aggregate pivot layout") !=
          context->host_status_arena + stable_shared_info_offset ||
        reinterpret_cast<int*>(h_aggregate_dstop + reserved_targets) !=
          context->host_status_arena + host_layout_required ||
        context->stable_workspace.info !=
          context->int_arena + stable_shared_info_offset) {
      throw std::runtime_error(
        "fixed-sp stable compact arena binding mismatch");
    }
    const std::size_t pointer_layout_required = checked_multiply(
      reserved_targets, 2U, "fixed-sp pointer arena layout");
    if (context->pointer_arena == nullptr ||
        context->pointer_capacity < pointer_layout_required) {
      throw std::runtime_error(
        "fixed-sp pointer arena does not satisfy 2*targets layout");
    }
    double** d_system_ptrs =
      reinterpret_cast<double**>(context->pointer_arena);
    double** d_theta_ptrs = d_system_ptrs + reserved_targets;

    std::fill_n(
      h_safe_target_indices,
      checked_multiply(reserved_targets, 2U,
                       "fixed-sp route partition staging"),
      0);
    int safe_count = 0;
    int stable_count = 0;
    for (int target = 0; target < batch.target_count; ++target) {
      const FixedSpRoute route =
        batch.planned_routes[static_cast<std::size_t>(target)];
      switch (route) {
        case FixedSpRoute::CholeskyBatched:
          h_safe_target_indices[safe_count++] = target;
          token->diagnostics.planned_cholesky_target_count += 1;
          break;
        case FixedSpRoute::AugmentedQr:
          h_stable_target_indices[stable_count++] = target;
          token->diagnostics.planned_qr_target_count += 1;
          break;
        case FixedSpRoute::AugmentedSvd:
          h_stable_target_indices[stable_count++] = target;
          token->diagnostics.planned_svd_target_count += 1;
          break;
        case FixedSpRoute::Unset:
          throw std::runtime_error(
            "fixed-sp planned route metadata is invalid");
      }
    }
    bool canonical_partition_exact =
      safe_count + stable_count == batch.target_count;
    for (int index = 1; index < safe_count; ++index) {
      canonical_partition_exact = canonical_partition_exact &&
        h_safe_target_indices[index - 1] < h_safe_target_indices[index];
    }
    for (int index = 1; index < stable_count; ++index) {
      canonical_partition_exact = canonical_partition_exact &&
        h_stable_target_indices[index - 1] <
          h_stable_target_indices[index];
    }
    if (!canonical_partition_exact) {
      throw std::runtime_error(
        "fixed-sp compact route partition is not canonical");
    }
    check_cuda(cudaMemcpyAsync(
      d_safe_target_indices, h_safe_target_indices,
      allocation_bytes(
        checked_multiply(reserved_targets, 2U,
                         "fixed-sp route partition upload"),
        sizeof(int), "fixed-sp route partition upload"),
      cudaMemcpyHostToDevice, context->stream
    ), "upload fixed-sp compact route partition");
    check_cuda(cudaMemsetAsync(
      d_geqrf_info, 0xff,
      allocation_bytes(
        checked_multiply(
          reserved_targets, kStableBaseIntArraysPerTarget,
          "fixed-sp base stable diagnostics"),
        sizeof(int), "fixed-sp base stable diagnostics"),
      context->stream
    ), "initialize fixed-sp stable compact diagnostics");
    check_cuda(cudaMemsetAsync(
      d_sigma_max, 0,
      allocation_bytes(
        checked_multiply(reserved_targets, 2U,
                         "fixed-sp stable sigma diagnostics"),
        sizeof(double), "fixed-sp stable sigma diagnostics"),
      context->stream
    ), "initialize fixed-sp stable sigma diagnostics");
    launch_fixed_sp_aggregate_diagnostics_init(
      context->capacities.target_count, context->capacities.null_dim,
      d_aggregate_root_rank, d_aggregate_factor_call_count,
      d_aggregate_b_build_count, d_aggregate_pivots,
      d_aggregate_dstop, context->stream);

    const std::size_t y_count = checked_multiply(
      static_cast<std::size_t>(batch.n), targets, "fixed-sp solve Y");
    const std::size_t sp_count = checked_multiply(
      static_cast<std::size_t>(batch.penalty_count), targets,
      "fixed-sp solve SP");
    const std::size_t y_bytes = allocation_bytes(
      y_count, sizeof(double), "fixed-sp solve Y");
    const std::size_t sp_bytes = allocation_bytes(
      sp_count, sizeof(double), "fixed-sp solve SP");
    check_cuda(cudaMemcpyAsync(
      d_y, batch.Y, y_bytes,
      cudaMemcpyHostToDevice, context->stream
    ), "upload fixed-sp solve Y");
    check_cuda(cudaMemcpyAsync(
      d_sp, batch.SP, sp_bytes,
      cudaMemcpyHostToDevice, context->stream
    ), "upload fixed-sp solve SP");
    // Target payload diagnostics cover the numeric Y/SP data plane. The
    // compact route metadata has its own single upload above.
    token->diagnostics.target_batch_h2d_call_count = 1;
    token->diagnostics.target_h2d_copy_count = 2;
    token->diagnostics.target_h2d_bytes = checked_add(
      y_bytes, sp_bytes, "fixed-sp target H2D diagnostics");

    const double one = 1.0;
    const double zero = 0.0;
    const double minus_one = -1.0;
    check_cublas(cublasDgemm(
      context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      handle->q, batch.target_count, batch.n,
      &one, handle->d_X_null, batch.n,
      d_y, batch.n, &zero, d_rhs, handle->q
    ), "build fixed-sp RHS with X_null transpose Y");
    token->diagnostics.rhs_device_build_count = 1;
    token->diagnostics.rhs_authority = "cuda-x0-transpose-y";
    token->diagnostics.full_cuda_data_plane = true;
    if (rhs_count != 0U) {
      if (rhs_count > slot->rhs_capacity) {
        throw std::runtime_error(
          "fixed-sp RHS output exceeds persistent slot capacity");
      }
      check_cuda(cudaMemcpyAsync(
        slot->rhs, d_rhs,
        allocation_bytes(rhs_count, sizeof(double), "persist fixed-sp RHS"),
        cudaMemcpyDeviceToDevice, context->stream
      ), "persist fixed-sp RHS in prepared output slot");
    }

    if (safe_count > 0) {
      constexpr unsigned int system_tile = 16U;
      const dim3 system_block(system_tile, system_tile, 1U);
      const dim3 system_grid(
        (static_cast<unsigned int>(handle->q) + system_tile - 1U) /
          system_tile,
        (static_cast<unsigned int>(handle->q) + system_tile - 1U) /
          system_tile,
        static_cast<unsigned int>(safe_count));
      build_fixed_sp_systems_kernel<<<
        system_grid, system_block, 0, context->stream
      >>>(
        handle->d_gram, handle->d_projected_H,
        handle->d_projected_penalties, d_sp,
        d_safe_target_indices, safe_count, handle->penalty_count,
        handle->q, d_systems);
      check_cuda(cudaGetLastError(), "launch fixed-sp fused system build");

      const std::size_t safe_theta_count = checked_multiply(
        static_cast<std::size_t>(handle->q),
        static_cast<std::size_t>(safe_count),
        "fixed-sp compact safe RHS");
      const std::size_t gather_blocks =
        (safe_theta_count + threads - 1U) / threads;
      if (gather_blocks == 0U ||
          gather_blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("fixed-sp safe RHS gather size overflow");
      }
      gather_fixed_sp_safe_rhs_kernel<<<
        static_cast<unsigned int>(gather_blocks), threads,
        0, context->stream
      >>>(
        d_rhs, d_safe_target_indices, handle->q, safe_count, d_theta);
      check_cuda(cudaGetLastError(), "launch fixed-sp safe RHS gather");
    }
    int successful_factor_count = 0;
    if (safe_count == 1) {
      token->diagnostics.cholesky_single_target_count = 1;
      check_cusolver(cusolverDnDpotrf(
        context->solver, CUBLAS_FILL_MODE_UPPER, handle->q,
        d_systems, handle->q, d_potrf_work, context->potrf_lwork,
        d_potrf_info
      ), "factor fixed-sp Cholesky system");
      check_cuda(cudaMemcpyAsync(
        h_potrf_info, d_potrf_info, sizeof(int),
        cudaMemcpyDeviceToHost, context->stream
      ), "copy fixed-sp Cholesky factor status");
      check_cuda(cudaEventRecord(
        context->cholesky_factor_checkpoint_event, context->stream
      ), "record fixed-sp Cholesky factor checkpoint");
      context->diagnostics.cholesky_factor_checkpoint_record_count += 1;
      check_cuda(cudaEventSynchronize(
        context->cholesky_factor_checkpoint_event
      ), "wait for fixed-sp Cholesky factor checkpoint");
      context->diagnostics.cholesky_factor_checkpoint_wait_count += 1;
      if (*h_potrf_info < 0) {
        throw std::runtime_error(
          "fixed-sp Cholesky potrf reported an invalid argument");
      }
      if (*h_potrf_info > 0) {
        const std::size_t target_offset = static_cast<std::size_t>(
          h_safe_target_indices[0]);
        token->reroute_reasons[target_offset] =
          "CHOLESKY_NON_POSITIVE_PIVOT";
        token->diagnostics.stable_reroute_count += 1;
        token->diagnostics.cholesky_to_svd_count += 1;
      } else {
        check_cusolver(cusolverDnDpotrs(
          context->solver, CUBLAS_FILL_MODE_UPPER, handle->q, 1,
          d_systems, handle->q, d_theta, handle->q, d_potrs_info
        ), "solve fixed-sp Cholesky system");
        check_cuda(cudaMemcpyAsync(
          h_potrs_info, d_potrs_info, sizeof(int),
          cudaMemcpyDeviceToHost, context->stream
        ), "copy fixed-sp Cholesky solve status");
        check_cuda(cudaEventRecord(
          context->cholesky_solve_checkpoint_event, context->stream
        ), "record fixed-sp Cholesky solve checkpoint");
        context->diagnostics.cholesky_solve_checkpoint_record_count += 1;
        check_cuda(cudaEventSynchronize(
          context->cholesky_solve_checkpoint_event
        ), "wait for fixed-sp Cholesky solve checkpoint");
        context->diagnostics.cholesky_solve_checkpoint_wait_count += 1;
        if (*h_potrs_info != 0) {
          throw std::runtime_error(
            "fixed-sp Cholesky potrs reported a batch failure");
        }
        successful_factor_count = 1;
      }
    } else if (safe_count >= 2) {
      token->diagnostics.true_batched_subgroup_count = 1;
      token->diagnostics.true_batched_attempted_target_count = safe_count;

      const std::size_t pointer_blocks =
        (static_cast<std::size_t>(safe_count) + threads - 1U) / threads;
      if (pointer_blocks == 0U ||
          pointer_blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("fixed-sp pointer-array launch overflow");
      }
      make_fixed_sp_pointer_arrays<<<
        static_cast<unsigned int>(pointer_blocks), threads,
        0, context->stream
      >>>(
        d_systems, d_theta, handle->q, safe_count,
        d_system_ptrs, d_theta_ptrs);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp batched pointer arrays");
      check_cuda(cudaMemsetAsync(
        d_potrf_info, 0,
        allocation_bytes(static_cast<std::size_t>(safe_count), sizeof(int),
                         "Phase 3B batched potrf info"),
        context->stream
      ), "zero Phase 3B batched potrf info");

      std::vector<int> forced_potrf_info = consume_forced_potrf_info();
      if (!forced_potrf_info.empty() &&
          forced_potrf_info.size() != static_cast<std::size_t>(safe_count)) {
        throw std::runtime_error(
          "Phase 3B forced potrf info length mismatch");
      }
      check_cusolver(cusolverDnDpotrfBatched(
        context->solver, CUBLAS_FILL_MODE_UPPER, handle->q,
        d_system_ptrs, handle->q, d_potrf_info, safe_count
      ), "Phase 3B batched potrf");
      token->diagnostics.potrf_batched_call_count = 1;
      if (!forced_potrf_info.empty()) {
        std::copy(
          forced_potrf_info.begin(), forced_potrf_info.end(), h_potrf_info);
        check_cuda(cudaMemcpyAsync(
          d_potrf_info, h_potrf_info,
          allocation_bytes(
            static_cast<std::size_t>(safe_count), sizeof(int),
            "forced Phase 3B batched potrf info"),
          cudaMemcpyHostToDevice, context->stream
        ), "force Phase 3B batched potrf info");
      }
      check_cuda(cudaMemcpyAsync(
        h_potrf_info, d_potrf_info,
        allocation_bytes(static_cast<std::size_t>(safe_count), sizeof(int),
                         "Phase 3B batched potrf info"),
        cudaMemcpyDeviceToHost, context->stream
      ), "copy Phase 3B batched potrf info");
      check_cuda(cudaEventRecord(
        context->cholesky_factor_checkpoint_event, context->stream
      ), "record Phase 3B batched factor checkpoint");
      context->diagnostics.cholesky_factor_checkpoint_record_count += 1;
      check_cuda(cudaEventSynchronize(
        context->cholesky_factor_checkpoint_event
      ), "wait for Phase 3B batched factor checkpoint");
      context->diagnostics.cholesky_factor_checkpoint_wait_count += 1;

      for (int safe_ordinal = 0; safe_ordinal < safe_count; ++safe_ordinal) {
        if (h_potrf_info[safe_ordinal] < 0) {
          throw std::runtime_error(
            "Phase 3B batched potrf info contains a negative value");
        }
      }
      for (int safe_ordinal = 0; safe_ordinal < safe_count; ++safe_ordinal) {
        if (h_potrf_info[safe_ordinal] == 0) {
          successful_factor_count += 1;
          continue;
        }
        const std::size_t target_offset = static_cast<std::size_t>(
          h_safe_target_indices[safe_ordinal]);
        token->reroute_reasons[target_offset] =
          "CHOLESKY_NON_POSITIVE_PIVOT";
        token->diagnostics.stable_reroute_count += 1;
        token->diagnostics.cholesky_to_svd_count += 1;
      }

      if (successful_factor_count > 0) {
        compact_fixed_sp_success_pointer_arrays<<<
          1U, 1U, 0, context->stream
        >>>(
          d_potrf_info, safe_count, d_system_ptrs, d_theta_ptrs);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp successful pointer compaction");
        check_cuda(cudaMemsetAsync(
          d_potrs_info, 0, sizeof(int), context->stream
        ), "zero Phase 3B potrs scalar info");

        int forced_potrs_info = 0;
        const bool force_potrs_info =
          consume_forced_potrs_info(&forced_potrs_info);
        check_cusolver(cusolverDnDpotrsBatched(
          context->solver, CUBLAS_FILL_MODE_UPPER, handle->q, 1,
          d_system_ptrs, handle->q, d_theta_ptrs, handle->q,
          d_potrs_info, successful_factor_count
        ), "Phase 3B batched potrs");
        token->diagnostics.potrs_batched_call_count = 1;
        if (force_potrs_info) {
          *h_potrs_info = forced_potrs_info;
          check_cuda(cudaMemcpyAsync(
            d_potrs_info, h_potrs_info, sizeof(int),
            cudaMemcpyHostToDevice, context->stream
          ), "force Phase 3B batched potrs info");
        }
        check_cuda(cudaMemcpyAsync(
          h_potrs_info, d_potrs_info, sizeof(int),
          cudaMemcpyDeviceToHost, context->stream
        ), "copy Phase 3B batched potrs info");
        check_cuda(cudaEventRecord(
          context->cholesky_solve_checkpoint_event, context->stream
        ), "record Phase 3B batched solve checkpoint");
        context->diagnostics.cholesky_solve_checkpoint_record_count += 1;
        check_cuda(cudaEventSynchronize(
          context->cholesky_solve_checkpoint_event
        ), "wait for Phase 3B batched solve checkpoint");
        context->diagnostics.cholesky_solve_checkpoint_wait_count += 1;
        if (*h_potrs_info != 0) {
          throw std::runtime_error(
            "Phase 3B batched potrs info is nonzero");
        }
        token->diagnostics.true_batched_target_count =
          successful_factor_count;
      }
    }

    const bool write_coefficients =
      (batch.output_mask & FixedSpOutputCoefficients) != 0U;
    const bool needs_fitted =
      (batch.output_mask & (FixedSpOutputFitted |
                            FixedSpOutputResiduals |
                            FixedSpOutputRss)) != 0U;
    const bool write_residuals =
      (batch.output_mask & FixedSpOutputResiduals) != 0U;
    const bool write_rss =
      (batch.output_mask & FixedSpOutputRss) != 0U;
    constexpr unsigned int output_columns_per_block = 32U;
    constexpr unsigned int output_targets_per_block = 8U;
    const dim3 output_block(
      output_columns_per_block, output_targets_per_block, 1U);

    if (successful_factor_count > 0 && write_coefficients) {
      const dim3 coefficient_grid(
        (static_cast<unsigned int>(handle->p) +
          output_columns_per_block - 1U) / output_columns_per_block,
        (static_cast<unsigned int>(safe_count) +
          output_targets_per_block - 1U) / output_targets_per_block,
        1U);
      finalize_fixed_sp_coefficients_batch<<<
        coefficient_grid, output_block, 0, context->stream
      >>>(
        d_theta, handle->d_Z, d_safe_target_indices, d_potrf_info,
        handle->p, handle->q, safe_count, slot->coefficients);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp batched coefficient finalization");
      token->diagnostics.coefficient_batch_finalize_call_count = 1;
    }

    if (successful_factor_count > 0 && needs_fitted) {
      const dim3 fitted_grid(
        (static_cast<unsigned int>(batch.n) +
          output_columns_per_block - 1U) / output_columns_per_block,
        (static_cast<unsigned int>(safe_count) +
          output_targets_per_block - 1U) / output_targets_per_block,
        1U);
      finalize_fixed_sp_fitted_batch<<<
        fitted_grid, output_block, 0, context->stream
      >>>(
        handle->d_X_null, d_theta, d_safe_target_indices, d_potrf_info,
        batch.n, handle->q, safe_count, slot->fitted);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp batched fitted finalization");
      token->diagnostics.fitted_batch_finalize_call_count = 1;
    }

    if (successful_factor_count > 0 && (write_residuals || write_rss)) {
      finalize_fixed_sp_residual_rss_batch<<<
        static_cast<unsigned int>(safe_count), threads,
        threads * sizeof(double), context->stream
      >>>(
        d_y, slot->fitted, d_safe_target_indices, d_potrf_info,
        batch.n, safe_count, slot->residuals, slot->rss,
        write_residuals, write_rss);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp batched residual and RSS finalization");
      token->diagnostics.residual_rss_batch_finalize_call_count = 1;
    }

    int stable_ordinal = 0;
    for (int target = 0; target < batch.target_count; ++target) {
      const FixedSpRoute route =
        batch.planned_routes[static_cast<std::size_t>(target)];
      if (route == FixedSpRoute::CholeskyBatched) continue;
      const int target_stable_ordinal = stable_ordinal++;
      if (route != FixedSpRoute::AugmentedQr) continue;

      const std::size_t target_offset = static_cast<std::size_t>(target);
      const double* target_y = d_y +
        static_cast<std::size_t>(batch.n) * target_offset;
      const double* target_sp = d_sp +
        static_cast<std::size_t>(batch.penalty_count) * target_offset;
      const double* host_target_sp = batch.SP +
        static_cast<std::size_t>(batch.penalty_count) * target_offset;
      AugmentedSystemView augmented = build_fixed_sp_augmented_system(
        handle->d_X_null, handle->d_penalty_roots,
        handle->penalty_root_offsets.data(),
        handle->penalty_root_ranks.data(),
        handle->total_penalty_root_rows, handle->penalty_count,
        handle->d_H_root, handle->H_root_rank,
        target_y, target_sp, host_target_sp,
        batch.n, handle->q, target,
        &context->stable_workspace, context->stream);

      check_cusolver(cusolverDnDgeqrf(
        context->solver, augmented.rows, handle->q,
        augmented.B, augmented.leading_dimension,
        context->stable_workspace.tau,
        context->stable_workspace.qr_work,
        context->stable_workspace.qr_lwork,
        context->stable_workspace.info
      ), "factor fixed-sp augmented QR system");
      check_cuda(cudaMemcpyAsync(
        d_geqrf_info + target_offset, context->stable_workspace.info,
        sizeof(int), cudaMemcpyDeviceToDevice, context->stream
      ), "snapshot fixed-sp geqrf info");

      check_cusolver(cusolverDnDormqr(
        context->solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        augmented.rows, 1, handle->q,
        augmented.B, augmented.leading_dimension,
        context->stable_workspace.tau,
        augmented.c, augmented.leading_dimension,
        context->stable_workspace.qr_work,
        context->stable_workspace.ormqr_lwork,
        context->stable_workspace.info
      ), "apply fixed-sp augmented QR Q transpose");
      check_cuda(cudaMemcpyAsync(
        d_ormqr_info + target_offset, context->stable_workspace.info,
        sizeof(int), cudaMemcpyDeviceToDevice, context->stream
      ), "snapshot fixed-sp ormqr info");

      check_cublas(cublasDtrsv(
        context->blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, handle->q,
        augmented.B, augmented.leading_dimension, augmented.c, 1
      ), "solve fixed-sp augmented QR triangular system");
      launch_fixed_sp_qr_rank_status(
        augmented.B, augmented.c, augmented.rows, handle->q, target,
        std::numeric_limits<double>::epsilon(),
        d_qr_rank, d_qr_reroute, d_qr_finite_status, context->stream);

      if (write_coefficients) {
        const std::size_t coefficient_blocks =
          (static_cast<std::size_t>(handle->p) + threads - 1U) / threads;
        finalize_fixed_sp_qr_coefficients<<<
          static_cast<unsigned int>(coefficient_blocks), threads,
          0, context->stream
        >>>(
          augmented.c, handle->d_Z, target, handle->p, handle->q,
          slot->coefficients);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp QR coefficient finalization");
      }
      if (needs_fitted) {
        const std::size_t fitted_blocks =
          (static_cast<std::size_t>(batch.n) + threads - 1U) / threads;
        finalize_fixed_sp_qr_fitted<<<
          static_cast<unsigned int>(fitted_blocks), threads,
          0, context->stream
        >>>(
          handle->d_X_null, augmented.c, target, batch.n, handle->q,
          slot->fitted);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp QR fitted finalization");
      }
      if (write_residuals || write_rss) {
        finalize_fixed_sp_qr_residual_rss<<<
          1U, threads, threads * sizeof(double), context->stream
        >>>(
          d_y, slot->fitted, target, batch.n,
          slot->residuals, slot->rss, write_residuals, write_rss);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp QR residual and RSS finalization");
      }
      if (write_coefficients || needs_fitted) {
        token->diagnostics.per_target_output_finalize_call_count += 1;
      }

      check_fixed_sp_outputs_finite<<<1U, threads, 0, context->stream>>>(
        d_rhs, augmented.c,
        d_stable_target_indices +
          static_cast<std::size_t>(target_stable_ordinal),
        slot->coefficients, slot->fitted, slot->residuals, slot->rss,
        batch.n, handle->p, handle->q, 1,
        batch.output_mask, d_qr_finite_status);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp QR output finite check");
    }
    if (stable_ordinal != stable_count) {
      throw std::runtime_error(
        "fixed-sp stable target ordinal accounting mismatch");
    }

    if (token->diagnostics.planned_qr_target_count > 0) {
      check_cuda(cudaMemcpyAsync(
        h_geqrf_info, d_geqrf_info,
        allocation_bytes(
          checked_multiply(
            reserved_targets, kStableBaseIntArraysPerTarget,
            "fixed-sp QR compact diagnostics"),
          sizeof(int), "fixed-sp QR compact diagnostics"),
        cudaMemcpyDeviceToHost, context->stream
      ), "copy fixed-sp QR compact diagnostics");
      check_cuda(cudaEventRecord(
        context->cholesky_factor_checkpoint_event, context->stream
      ), "record fixed-sp QR checkpoint");
      context->diagnostics.qr_checkpoint_record_count += 1;
      check_cuda(cudaEventSynchronize(
        context->cholesky_factor_checkpoint_event
      ), "wait for fixed-sp QR checkpoint");
      context->diagnostics.qr_checkpoint_wait_count += 1;

      for (int target = 0; target < batch.target_count; ++target) {
        const std::size_t target_offset = static_cast<std::size_t>(target);
        if (batch.planned_routes[target_offset] !=
            FixedSpRoute::AugmentedQr) {
          continue;
        }
        token->diagnostics.geqrf_info[target_offset] =
          h_geqrf_info[target_offset];
        token->diagnostics.ormqr_info[target_offset] =
          h_ormqr_info[target_offset];
        token->diagnostics.qr_rank[target_offset] = h_qr_rank[target_offset];

        if (h_geqrf_info[target_offset] != 0 ||
            h_ormqr_info[target_offset] != 0) {
          token->executed_routes[target_offset] = FixedSpRoute::AugmentedQr;
          token->solver_statuses[target_offset] = FixedSpStatus::ErrQrFailed;
          token->diagnostics.executed_qr_target_count += 1;
          continue;
        }
        if (h_qr_rank[target_offset] < 0 ||
            h_qr_rank[target_offset] > handle->q ||
            (h_qr_reroute[target_offset] != 0 &&
             h_qr_reroute[target_offset] != 1) ||
            (h_qr_finite_status[target_offset] != 0 &&
             h_qr_finite_status[target_offset] != 1)) {
          throw std::runtime_error(
            "fixed-sp QR compact diagnostics are invalid");
        }
        if (h_qr_reroute[target_offset] != 0 ||
            h_qr_rank[target_offset] < handle->q) {
          token->executed_routes[target_offset] = FixedSpRoute::AugmentedSvd;
          token->reroute_reasons[target_offset] =
            "QR_RANK_GUARD_REJECTED";
          token->diagnostics.stable_reroute_count += 1;
          token->diagnostics.qr_to_svd_count += 1;
          continue;
        }
        token->executed_routes[target_offset] = FixedSpRoute::AugmentedQr;
        token->solver_statuses[target_offset] = FixedSpStatus::OkAugmentedQr;
        token->diagnostics.executed_qr_target_count += 1;
      }
    }

    auto target_uses_svd = [&](std::size_t target_offset) {
      const FixedSpRoute route = batch.planned_routes[target_offset];
      const std::string& reason = token->reroute_reasons[target_offset];
      return route == FixedSpRoute::AugmentedSvd ||
        (route == FixedSpRoute::CholeskyBatched &&
         reason == "CHOLESKY_NON_POSITIVE_PIVOT") ||
        (route == FixedSpRoute::AugmentedQr &&
         reason == "QR_RANK_GUARD_REJECTED");
    };
    auto device_target_index = [&](int target) -> const int* {
      const FixedSpRoute route =
        batch.planned_routes[static_cast<std::size_t>(target)];
      if (route == FixedSpRoute::CholeskyBatched) {
        for (int ordinal = 0; ordinal < safe_count; ++ordinal) {
          if (h_safe_target_indices[ordinal] == target) {
            return d_safe_target_indices + ordinal;
          }
        }
      } else {
        for (int ordinal = 0; ordinal < stable_count; ++ordinal) {
          if (h_stable_target_indices[ordinal] == target) {
            return d_stable_target_indices + ordinal;
          }
        }
      }
      throw std::runtime_error(
        "fixed-sp SVD target is absent from the route partition");
    };

    int svd_target_count = 0;
    for (int target = 0; target < batch.target_count; ++target) {
      const std::size_t target_offset = static_cast<std::size_t>(target);
      if (!target_uses_svd(target_offset)) continue;
      svd_target_count += 1;

      const double* target_y = d_y +
        static_cast<std::size_t>(batch.n) * target_offset;
      const double* target_sp = d_sp +
        static_cast<std::size_t>(batch.penalty_count) * target_offset;
      int* target_aggregate_pivots = d_aggregate_pivots +
        target_offset * reserved_q;
      launch_fixed_sp_aggregate_factor(
        handle->d_projected_penalties, handle->d_projected_H,
        target_sp, handle->penalty_count, handle->q,
        context->stable_workspace.aggregate_penalty_factor,
        context->stable_workspace.aggregate_factor_work,
        d_aggregate_root_rank + target_offset,
        d_aggregate_factor_call_count + target_offset,
        target_aggregate_pivots, d_aggregate_dstop + target_offset,
        context->stream);
      AugmentedSystemView augmented =
        build_fixed_sp_aggregate_augmented_system(
        handle->d_X_null,
        context->stable_workspace.aggregate_penalty_factor,
        d_aggregate_root_rank + target_offset,
        target_aggregate_pivots, target_y,
        batch.n, handle->q, target,
        d_aggregate_b_build_count + target_offset,
        &context->stable_workspace, context->stream);

      check_cusolver(cusolverDnDgesvdj(
        context->solver, CUSOLVER_EIG_MODE_VECTOR, 1,
        augmented.rows, handle->q,
        augmented.B, augmented.leading_dimension,
        context->stable_workspace.singular_values,
        context->stable_workspace.U, augmented.leading_dimension,
        context->stable_workspace.V, handle->q,
        context->stable_workspace.svd_work,
        context->stable_workspace.svd_lwork,
        context->stable_workspace.info, context->svd_params
      ), "factor fixed-sp augmented SVD system");
      check_cuda(cudaMemcpyAsync(
        d_svd_info + target_offset, context->stable_workspace.info,
        sizeof(int), cudaMemcpyDeviceToDevice, context->stream
      ), "snapshot fixed-sp SVD info");

      check_cublas(cublasDgemv(
        context->blas, CUBLAS_OP_T, augmented.rows, handle->q,
        &one, context->stable_workspace.U, augmented.leading_dimension,
        augmented.c, 1, &zero,
        context->stable_workspace.scaled_projection, 1
      ), "project fixed-sp augmented response onto SVD U");
      launch_fixed_sp_svd_rank_scale(
        context->stable_workspace.singular_values,
        context->stable_workspace.scaled_projection,
        d_svd_info + target_offset,
        augmented.rows, handle->q, target,
        std::numeric_limits<double>::epsilon(),
        d_qr_rank, d_sigma_max, d_smallest_retained_sigma,
        context->stream);
      check_cublas(cublasDgemv(
        context->blas, CUBLAS_OP_N, handle->q, handle->q,
        &one, context->stable_workspace.V, handle->q,
        context->stable_workspace.scaled_projection, 1,
        &zero, context->stable_workspace.diagonal, 1
      ), "form fixed-sp augmented SVD coefficients from V");

      AugmentedSystemView correction =
        build_fixed_sp_aggregate_augmented_system(
        handle->d_X_null,
        context->stable_workspace.aggregate_penalty_factor,
        d_aggregate_root_rank + target_offset,
        target_aggregate_pivots, target_y,
        batch.n, handle->q, target,
        d_aggregate_b_build_count + target_offset,
        &context->stable_workspace, context->stream);
      check_cublas(cublasDgemv(
        context->blas, CUBLAS_OP_N, correction.rows, handle->q,
        &minus_one, correction.B, correction.leading_dimension,
        context->stable_workspace.diagonal, 1,
        &one, correction.c, 1
      ), "form fixed-sp augmented SVD correction residual");
      check_cublas(cublasDgemv(
        context->blas, CUBLAS_OP_T, correction.rows, handle->q,
        &one, context->stable_workspace.U, correction.leading_dimension,
        correction.c, 1, &zero,
        context->stable_workspace.scaled_projection, 1
      ), "project fixed-sp augmented SVD correction onto U");
      launch_fixed_sp_svd_rank_scale(
        context->stable_workspace.singular_values,
        context->stable_workspace.scaled_projection,
        d_svd_info + target_offset,
        correction.rows, handle->q, target,
        std::numeric_limits<double>::epsilon(),
        d_qr_rank, d_sigma_max, d_smallest_retained_sigma,
        context->stream);
      check_cublas(cublasDgemv(
        context->blas, CUBLAS_OP_N, handle->q, handle->q,
        &one, context->stable_workspace.V, handle->q,
        context->stable_workspace.scaled_projection, 1,
        &zero, context->stable_workspace.tau, 1
      ), "form fixed-sp augmented SVD correction from V");
      check_cublas(cublasDaxpy(
        context->blas, handle->q, &one,
        context->stable_workspace.tau, 1,
        context->stable_workspace.diagonal, 1
      ), "apply fixed-sp augmented SVD correction");

      if (write_coefficients) {
        const std::size_t coefficient_blocks =
          (static_cast<std::size_t>(handle->p) + threads - 1U) / threads;
        finalize_fixed_sp_qr_coefficients<<<
          static_cast<unsigned int>(coefficient_blocks), threads,
          0, context->stream
        >>>(
          context->stable_workspace.diagonal, handle->d_Z, target,
          handle->p, handle->q, slot->coefficients);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp SVD coefficient finalization");
      }
      if (needs_fitted) {
        const std::size_t fitted_blocks =
          (static_cast<std::size_t>(batch.n) + threads - 1U) / threads;
        finalize_fixed_sp_qr_fitted<<<
          static_cast<unsigned int>(fitted_blocks), threads,
          0, context->stream
        >>>(
          handle->d_X_null, context->stable_workspace.diagonal, target,
          batch.n, handle->q, slot->fitted);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp SVD fitted finalization");
      }
      if (write_residuals || write_rss) {
        finalize_fixed_sp_qr_residual_rss<<<
          1U, threads, threads * sizeof(double), context->stream
        >>>(
          d_y, slot->fitted, target, batch.n,
          slot->residuals, slot->rss, write_residuals, write_rss);
        check_cuda(cudaGetLastError(),
                   "launch fixed-sp SVD residual and RSS finalization");
      }
      if (write_coefficients || needs_fitted) {
        token->diagnostics.per_target_output_finalize_call_count += 1;
      }

      check_fixed_sp_outputs_finite<<<1U, threads, 0, context->stream>>>(
        d_rhs, context->stable_workspace.diagonal,
        device_target_index(target),
        slot->coefficients, slot->fitted, slot->residuals, slot->rss,
        batch.n, handle->p, handle->q, 1,
        batch.output_mask, d_qr_finite_status);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp SVD output finite check");
    }

    if (svd_target_count > 0) {
      check_cuda(cudaMemcpyAsync(
        h_geqrf_info, d_geqrf_info,
        allocation_bytes(
          stable_compact_count, sizeof(int),
          "fixed-sp SVD compact integer diagnostics"),
        cudaMemcpyDeviceToHost, context->stream
      ), "copy fixed-sp SVD compact integer diagnostics");
      check_cuda(cudaMemcpyAsync(
        h_sigma_max, d_sigma_max,
        allocation_bytes(
          checked_multiply(
            reserved_targets, kStableDoubleArraysPerTarget,
            "fixed-sp SVD compact double diagnostics"),
          sizeof(double), "fixed-sp SVD compact double diagnostics"),
        cudaMemcpyDeviceToHost, context->stream
      ), "copy fixed-sp SVD compact double diagnostics");
      check_cuda(cudaEventRecord(
        context->cholesky_factor_checkpoint_event, context->stream
      ), "record fixed-sp SVD checkpoint");
      context->diagnostics.svd_checkpoint_record_count += 1;
      check_cuda(cudaEventSynchronize(
        context->cholesky_factor_checkpoint_event
      ), "wait for fixed-sp SVD checkpoint");
      context->diagnostics.svd_checkpoint_wait_count += 1;

      int observed_svd_count = 0;
      for (int target = 0; target < batch.target_count; ++target) {
        const std::size_t target_offset = static_cast<std::size_t>(target);
        if (!target_uses_svd(target_offset)) continue;
        observed_svd_count += 1;

        const int info = h_svd_info[target_offset];
        const int rank = h_qr_rank[target_offset];
        const double maximum = h_sigma_max[target_offset];
        const double smallest = h_smallest_retained_sigma[target_offset];
        const int aggregate_rank = h_aggregate_root_rank[target_offset];
        const int factor_count =
          h_aggregate_factor_call_count[target_offset];
        const int b_build_count = h_aggregate_b_build_count[target_offset];
        const double aggregate_dstop = h_aggregate_dstop[target_offset];
        token->diagnostics.svd_info[target_offset] = info;
        token->diagnostics.effective_rank[target_offset] = rank;
        token->diagnostics.sigma_max[target_offset] = maximum;
        token->diagnostics.smallest_retained_sigma[target_offset] = smallest;
        token->diagnostics.aggregate_penalty_root_rank[target_offset] =
          aggregate_rank;
        token->diagnostics.aggregate_factor_call_count[target_offset] =
          factor_count;
        token->diagnostics.aggregate_b_build_count[target_offset] =
          b_build_count;
        token->diagnostics.aggregate_dstop[target_offset] = aggregate_dstop;
        std::copy_n(
          h_aggregate_pivots + target_offset * reserved_q,
          static_cast<std::size_t>(handle->q),
          token->diagnostics.aggregate_penalty_root_pivot.begin() +
            static_cast<std::ptrdiff_t>(target_offset *
              static_cast<std::size_t>(handle->q)));
        token->executed_routes[target_offset] = FixedSpRoute::AugmentedSvd;
        token->target_true_batched[target_offset] = false;

        const bool rank_diagnostics_valid =
          rank >= 0 && rank <= handle->q &&
          std::isfinite(maximum) && maximum >= 0.0 &&
          std::isfinite(smallest) && smallest >= 0.0 &&
          smallest <= maximum &&
          ((rank == 0 && smallest == 0.0) ||
           (rank > 0 && smallest > 0.0)) &&
          aggregate_rank >= 0 && aggregate_rank <= handle->q &&
          factor_count == 1 && b_build_count == 2 &&
          std::isfinite(aggregate_dstop) && aggregate_dstop >= 0.0;
        if (info != 0 || !rank_diagnostics_valid) {
          token->solver_statuses[target_offset] =
            FixedSpStatus::ErrSvdFailed;
          continue;
        }
        token->solver_statuses[target_offset] = FixedSpStatus::OkAugmentedSvd;
        token->diagnostics.executed_svd_target_count += 1;
      }
      if (observed_svd_count != svd_target_count) {
        throw std::runtime_error(
          "fixed-sp SVD target accounting mismatch");
      }
      token->diagnostics.aggregate_penalty_factor_count =
        std::accumulate(
          token->diagnostics.aggregate_factor_call_count.begin(),
          token->diagnostics.aggregate_factor_call_count.end(), 0);
      token->diagnostics.aggregate_svd_b_build_count =
        std::accumulate(
          token->diagnostics.aggregate_b_build_count.begin(),
          token->diagnostics.aggregate_b_build_count.end(), 0);
    }

    for (int safe_ordinal = 0; safe_ordinal < safe_count; ++safe_ordinal) {
      if (h_potrf_info[safe_ordinal] != 0) continue;
      const std::size_t target_offset = static_cast<std::size_t>(
        h_safe_target_indices[safe_ordinal]);
      token->executed_routes[target_offset] = FixedSpRoute::CholeskyBatched;
      token->solver_statuses[target_offset] = safe_count >= 2 ?
        FixedSpStatus::OkCholeskyBatched : FixedSpStatus::OkCholeskySingle;
      token->target_true_batched[target_offset] = safe_count >= 2;
      token->diagnostics.executed_cholesky_target_count += 1;
    }
    token->diagnostics.batch_output_finalized_target_count =
      token->diagnostics.executed_cholesky_target_count;
    if (targets > slot->finite_status_capacity ||
        slot->host_finite_status == nullptr) {
      throw std::runtime_error("fixed-sp finite status capacity mismatch");
    }
    check_cuda(cudaMemsetAsync(
      d_potrf_info, 0,
      allocation_bytes(targets, sizeof(int), "fixed-sp finite status"),
      context->stream
    ), "initialize fixed-sp finite status");
    if (safe_count > 0) {
      const std::size_t finite_status_blocks =
        (static_cast<std::size_t>(safe_count) + threads - 1U) / threads;
      if (finite_status_blocks == 0U ||
          finite_status_blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("fixed-sp finite status launch overflow");
      }
      check_fixed_sp_outputs_finite<<<
        static_cast<unsigned int>(finite_status_blocks), threads,
        0, context->stream
      >>>(
        d_rhs, d_theta, d_safe_target_indices,
        slot->coefficients, slot->fitted, slot->residuals, slot->rss,
        batch.n, handle->p, handle->q, safe_count,
        batch.output_mask, d_potrf_info);
      check_cuda(cudaGetLastError(), "launch fixed-sp output finite check");
    }
    if (token->diagnostics.planned_qr_target_count > 0 ||
        token->diagnostics.executed_svd_target_count > 0) {
      const std::size_t stable_finite_blocks =
        (targets + threads - 1U) / threads;
      if (stable_finite_blocks == 0U ||
          stable_finite_blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error(
          "fixed-sp stable finite status merge size overflow");
      }
      merge_fixed_sp_qr_finite_status<<<
        static_cast<unsigned int>(stable_finite_blocks), threads,
        0, context->stream
      >>>(d_qr_finite_status, batch.target_count, d_potrf_info);
      check_cuda(cudaGetLastError(),
                 "launch fixed-sp stable finite status merge");
    }

    // Every route writes finite status by canonical target ordinal. Failed
    // solver attempts remain explicit errors, so provisional output is hidden.
    check_cuda(cudaMemcpyAsync(
      slot->host_finite_status, d_potrf_info,
      allocation_bytes(targets, sizeof(int), "fixed-sp finite status"),
      cudaMemcpyDeviceToHost, context->stream
    ), "copy fixed-sp finite status");
    check_cuda(cudaEventRecord(slot->solve_completion_event, context->stream),
               "record fixed-sp solve completion");
    token->diagnostics.canonical_output_order_exact =
      canonical_partition_exact;

    token->diagnostics.target_keys = token->target_keys;
    token->diagnostics.planned_routes = token->planned_routes;
    token->diagnostics.executed_routes = token->executed_routes;
    token->diagnostics.reroute_reasons = token->reroute_reasons;
    token->diagnostics.solver_statuses = token->solver_statuses;
    token->diagnostics.target_true_batched = token->target_true_batched;
    copy_solve_resource_deltas(
      resources_before_solve,
      resource_counters_snapshot(context->resource_ledger),
      &token->diagnostics);
    return token;
  } catch (...) {
    restore_slot();
    throw;
  }
}

DeviceResidualInfo device_residual_info(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  const bool require_active_lease = token && !token->lease_released;
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, require_active_lease);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, require_active_lease);
  if (require_active_lease &&
      token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  context->require_usable();
  if (require_active_lease) {
    resolve_fixed_sp_output_status_locked(token.get(), context.get());
  } else if (!token->output_status_resolved) {
    throw std::runtime_error(
      "STALE_TOKEN: released residual status was not resolved");
  }
  return token->diagnostics;
}

FixedSpShadowResult materialize_fixed_sp_shadow(
    const std::shared_ptr<DeviceResidualBatch>& token,
    std::uint32_t output_mask) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, true);
  if (output_mask == 0U ||
      (output_mask & ~kFixedSpPublicOutputMask) != 0U) {
    throw std::runtime_error("fixed-sp shadow output mask is invalid");
  }
  if ((output_mask & ~token->output_mask) != 0U) {
    throw std::runtime_error(
      "fixed-sp shadow requested an output not computed by the solve");
  }

  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, true);
  if (token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  if (token->slot->state ==
        TransientResidualSlotState::ConsumerRegistrationPending ||
      token->slot->consumer_event_registered) {
    token->diagnostics.output_slot_busy_count += 1;
    throw std::runtime_error("ERR_OUTPUT_SLOT_BUSY");
  }
  context->require_usable();
  if (token->solver_statuses.size() !=
      static_cast<std::size_t>(token->target_count)) {
    throw std::runtime_error("fixed-sp shadow status size mismatch");
  }
  if (token->target_count > token->slot->target_capacity ||
      token->slot->solve_completion_event == nullptr ||
      ((output_mask & FixedSpOutputCoefficients) != 0U &&
       (token->slot->coefficients == nullptr ||
        checked_multiply(
          static_cast<std::size_t>(token->owner->p),
          static_cast<std::size_t>(token->target_count),
          "fixed-sp coefficient shadow capacity") >
          token->slot->coefficient_capacity)) ||
      ((output_mask & FixedSpOutputFitted) != 0U &&
       token->slot->fitted == nullptr) ||
      ((output_mask & FixedSpOutputResiduals) != 0U &&
       token->slot->residuals == nullptr) ||
      ((output_mask & FixedSpOutputRss) != 0U &&
       token->slot->rss == nullptr) ||
      ((output_mask & FixedSpOutputRhs) != 0U &&
       (token->slot->rhs == nullptr ||
        checked_multiply(
          static_cast<std::size_t>(token->owner->q),
          static_cast<std::size_t>(token->target_count),
          "fixed-sp RHS shadow capacity") > token->slot->rhs_capacity))) {
    throw std::runtime_error(
      "fixed-sp shadow exceeds the prepared output slot capacity");
  }
  resolve_fixed_sp_output_status_locked(token.get(), context.get());

  FixedSpShadowResult result;
  result.n = token->n;
  result.coefficient_dim = token->owner->p;
  result.null_dim = token->owner->q;
  result.target_count = token->target_count;
  result.output_mask = output_mask;
  result.successful_targets.assign(
    static_cast<std::size_t>(result.target_count), 0U);
  const std::size_t targets = static_cast<std::size_t>(result.target_count);
  if ((output_mask & FixedSpOutputCoefficients) != 0U) {
    result.coefficients.resize(checked_multiply(
      static_cast<std::size_t>(result.coefficient_dim), targets,
      "fixed-sp coefficient shadow"));
  }
  if ((output_mask & FixedSpOutputFitted) != 0U) {
    result.fitted.resize(checked_multiply(
      static_cast<std::size_t>(result.n), targets,
      "fixed-sp fitted shadow"));
  }
  if ((output_mask & FixedSpOutputResiduals) != 0U) {
    result.residuals.resize(checked_multiply(
      static_cast<std::size_t>(result.n), targets,
      "fixed-sp residual shadow"));
  }
  if ((output_mask & FixedSpOutputRss) != 0U) {
    result.rss.resize(targets);
  }
  if ((output_mask & FixedSpOutputRhs) != 0U) {
    result.cuda_nullspace_rhs.resize(checked_multiply(
      static_cast<std::size_t>(result.null_dim), targets,
      "fixed-sp RHS shadow"));
  }

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for fixed-sp shadow");

  std::size_t copied_bytes = 0U;
  auto download_column = [&](double* destination,
                             const double* source,
                             std::size_t rows,
                             std::size_t target,
                             const char* name) {
    const std::size_t bytes = allocation_bytes(rows, sizeof(double), name);
    check_cuda(cudaMemcpy(
      destination + rows * target, source + rows * target,
      bytes, cudaMemcpyDeviceToHost
    ), name);
    copied_bytes = checked_add(copied_bytes, bytes,
                               "fixed-sp shadow D2H diagnostics");
  };
  for (std::size_t target = 0; target < targets; ++target) {
    if (!fixed_sp_status_is_successful(token->solver_statuses[target])) continue;
    result.successful_targets[target] = 1U;
    if ((output_mask & FixedSpOutputCoefficients) != 0U) {
      download_column(
        result.coefficients.data(), token->slot->coefficients,
        static_cast<std::size_t>(result.coefficient_dim), target,
        "download fixed-sp coefficient shadow");
    }
    if ((output_mask & FixedSpOutputFitted) != 0U) {
      download_column(
        result.fitted.data(), token->slot->fitted,
        static_cast<std::size_t>(result.n), target,
        "download fixed-sp fitted shadow");
    }
    if ((output_mask & FixedSpOutputResiduals) != 0U) {
      download_column(
        result.residuals.data(), token->slot->residuals,
        static_cast<std::size_t>(result.n), target,
        "download fixed-sp residual shadow");
    }
    if ((output_mask & FixedSpOutputRss) != 0U) {
      download_column(
        result.rss.data(), token->slot->rss, 1U, target,
        "download fixed-sp RSS shadow");
    }
    if ((output_mask & FixedSpOutputRhs) != 0U) {
      download_column(
        result.cuda_nullspace_rhs.data(), token->slot->rhs,
        static_cast<std::size_t>(result.null_dim), target,
        "download fixed-sp RHS shadow");
    }
  }
  token->diagnostics.shadow_materialize_call_count += 1;
  token->diagnostics.shadow_materialize_target_count += result.target_count;
  token->diagnostics.shadow_d2h_bytes = checked_add(
    token->diagnostics.shadow_d2h_bytes, copied_bytes,
    "fixed-sp cumulative shadow D2H diagnostics");
  return result;
}

void release_device_residual(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  const bool require_active_lease = token && !token->lease_released;
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, require_active_lease);

  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, require_active_lease);
  if (!require_active_lease) {
    context->require_usable();
    return;
  }
  if (token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  context->require_usable();
  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for residual release");
  resolve_fixed_sp_output_status_locked(token.get(), context.get());
  if (token->slot->consumer_event_registered) {
    const cudaError_t query =
      cudaEventQuery(token->slot->consumer_completion_event);
    if (query == cudaErrorNotReady) {
      token->diagnostics.output_slot_busy_count += 1;
      throw std::runtime_error("ERR_OUTPUT_SLOT_BUSY");
    }
    check_cuda(query, "query fixed-sp consumer completion event");
    token->slot->consumer_event_registered = false;
  }
  token->slot->state = TransientResidualSlotState::Free;
  token->slot->lease_owner.reset();
  token->lease_released = true;
  token->diagnostics.output_slot_release_count += 1;
}

namespace {

void register_device_residual_consumer_event_impl(
    const std::shared_ptr<DeviceResidualBatch>& token,
    cudaEvent_t consumer_completion_event,
    bool inject_failure) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, true);

  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, true);
  if (token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  if (token->slot->consumer_event_registered) {
    throw std::runtime_error(
      "fixed-sp consumer event is already registered");
  }
  context->require_usable();
  resolve_fixed_sp_output_status_locked(token.get(), context.get());

  token->slot->state =
    TransientResidualSlotState::ConsumerRegistrationPending;
  try {
    if (inject_failure) {
      throw std::runtime_error(
        "INJECTED_CONSUMER_REGISTRATION_FAILURE");
    }
    if (consumer_completion_event == nullptr) {
      throw std::runtime_error("consumer completion event must not be null");
    }
    check_cuda(cudaSetDevice(context->device_id),
               "set CUDA device for consumer event registration");
    check_cuda(cudaStreamWaitEvent(
      context->stream, consumer_completion_event, 0
    ), "wait for registered fixed-sp consumer event");
    check_cuda(cudaEventRecord(
      token->slot->consumer_completion_event, context->stream
    ), "record fixed-sp consumer completion proxy");
    token->slot->consumer_event_registered = true;
    token->slot->state = TransientResidualSlotState::Leased;
  } catch (const std::exception& error) {
    poison_transient_residual_slot(
      token->slot.get(), token->slot_generation, error.what(), false);
    throw;
  } catch (...) {
    poison_transient_residual_slot(
      token->slot.get(), token->slot_generation,
      "unknown consumer registration failure", false);
    throw;
  }
}

}  // namespace

void register_device_residual_consumer_event(
    const std::shared_ptr<DeviceResidualBatch>& token,
    cudaEvent_t consumer_completion_event) {
  register_device_residual_consumer_event_impl(
    token, consumer_completion_event, false);
}

void test_register_blocked_device_residual_consumer(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, true);
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    require_residual_host_identity(token, true);
    if (token->slot->state == TransientResidualSlotState::Poisoned) {
      throw_output_slot_poisoned(*token->slot);
    }
    if (token->slot->consumer_event_registered ||
        token->test_blocked_consumer_resources) {
      throw std::runtime_error(
        "fixed-sp blocked consumer test is already registered");
    }
    context->require_usable();
  }

  const std::shared_ptr<TestBlockedConsumerState> state =
    std::make_shared<TestBlockedConsumerState>();
  const std::shared_ptr<TestBlockedConsumerResources> resources =
    std::make_shared<TestBlockedConsumerResources>(
      context->resource_ledger, token->creator_pid, token->device_id, state);
  try {
    tracked_cuda_stream_create(
      context->resource_ledger.get(), resources->stream_address(),
      cudaStreamNonBlocking, "create blocked consumer test stream");
    tracked_cuda_event_create(
      context->resource_ledger.get(), resources->event_address(),
      cudaEventDisableTiming, "create blocked consumer test event");
    std::unique_ptr<TestBlockedConsumerCallbackPayload> payload(
      new TestBlockedConsumerCallbackPayload{state});
    check_cuda(cudaLaunchHostFunc(
      resources->stream(), wait_for_test_blocked_consumer, payload.get()
    ), "launch blocked consumer test host gate");
    payload.release();
    if (global_resource_ledger()
          .inject_next_blocked_consumer_launch_failure.exchange(
            false, std::memory_order_acq_rel)) {
      throw std::runtime_error(
        "INJECTED_BLOCKED_CONSUMER_LAUNCH_FAILURE");
    }
    check_cuda(cudaEventRecord(resources->event(), resources->stream()),
               "record blocked consumer test event");
    register_device_residual_consumer_event(token, resources->event());
    {
      std::lock_guard<std::mutex> lock(context->mutex);
      require_residual_host_identity(token, true);
      token->test_blocked_consumer_resources = resources;
    }
  } catch (...) {
    resources->cleanup_noexcept();
    if (resources->has_resources()) {
      std::lock_guard<std::mutex> lock(context->mutex);
      token->test_blocked_consumer_resources = resources;
    }
    throw;
  }
}

void test_complete_blocked_device_residual_consumer(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, true);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, true);
  if (token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  if (!token->slot->consumer_event_registered ||
      !token->test_blocked_consumer_resources ||
      !token->test_blocked_consumer_resources->has_resources()) {
    throw std::runtime_error(
      "fixed-sp blocked consumer test is not registered");
  }
  context->require_usable();

  const std::shared_ptr<TestBlockedConsumerResources> resources =
    token->test_blocked_consumer_resources;
  resources->signal_noexcept();
  if (resources->event() != nullptr) {
    check_cuda(cudaEventSynchronize(resources->event()),
               "complete blocked consumer test event");
  }
  check_cuda(cudaEventSynchronize(token->slot->consumer_completion_event),
             "complete blocked consumer test proxy event");
  const TestBlockedConsumerTeardownStatus teardown =
    resources->teardown_once_noexcept();
  if (!resources->has_resources()) {
    token->test_blocked_consumer_resources.reset();
  }
  check_test_blocked_consumer_teardown(teardown);
}

void free_device_residual(std::shared_ptr<DeviceResidualBatch>* token) {
  if (token == nullptr || !*token) return;
  const bool current_slot_poisoned =
    (*token)->slot &&
    (*token)->slot_generation == (*token)->slot->generation &&
    (*token)->slot->state == TransientResidualSlotState::Poisoned;
  if (current_slot_poisoned) {
    const std::shared_ptr<CudaRuntimeContext> context =
      require_residual_host_identity(*token, true);
    std::lock_guard<std::mutex> lock(context->mutex);
    require_residual_host_identity(*token, true);
    context->require_usable();
    (*token)->lease_released = true;
  } else {
    release_device_residual(*token);
  }
  (*token)->freed = true;
  (*token)->owner.reset();
  (*token)->slot.reset();
  token->reset();
}

DeviceCoefficientShadow test_device_residual_coefficient_shadow(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_residual_host_identity(token, true);

  std::lock_guard<std::mutex> lock(context->mutex);
  require_residual_host_identity(token, true);
  if (token->slot->state == TransientResidualSlotState::Poisoned) {
    throw_output_slot_poisoned(*token->slot);
  }
  if (token->slot->state ==
        TransientResidualSlotState::ConsumerRegistrationPending ||
      token->slot->consumer_event_registered) {
    token->diagnostics.output_slot_busy_count += 1;
    throw std::runtime_error("ERR_OUTPUT_SLOT_BUSY");
  }
  context->require_usable();
  if ((token->output_mask & FixedSpOutputCoefficients) == 0) {
    throw std::runtime_error(
      "test coefficient shadow requires coefficient output");
  }
  resolve_fixed_sp_output_status_locked(token.get(), context.get());

  DeviceCoefficientShadow shadow;
  shadow.coefficient_dim = token->owner->p;
  shadow.target_count = token->target_count;
  const std::size_t count = matrix_count(
    shadow.coefficient_dim, shadow.target_count,
    "test coefficient shadow");
  if (token->slot->coefficients == nullptr ||
      token->slot->solve_completion_event == nullptr ||
      count > token->slot->coefficient_capacity) {
    throw std::runtime_error(
      "test coefficient shadow exceeds allocated coefficient capacity");
  }
  shadow.coefficients.resize(count);

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for coefficient test shadow");
  check_cuda(cudaMemcpy(
    shadow.coefficients.data(), token->slot->coefficients,
    allocation_bytes(count, sizeof(double), "test coefficient shadow"),
    cudaMemcpyDeviceToHost
  ), "download coefficient test shadow");
  return shadow;
}

void test_inject_device_residual_consumer_registration_failure(
    const std::shared_ptr<DeviceResidualBatch>& token) {
  register_device_residual_consumer_event_impl(token, nullptr, true);
}

FixedSpAugmentedSystemShadow test_build_fixed_sp_augmented_shadow(
    const std::shared_ptr<PreparedSGpuHandle>& handle,
    const double* Y,
    std::size_t Y_count,
    const double* SP,
    std::size_t SP_count,
    int target_index) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  context->require_usable();

  const std::size_t n = positive_size(
    handle->n, "augmented test-shadow row count");
  const std::size_t q = positive_size(
    handle->q, "augmented test-shadow null dimension");
  const std::size_t penalty_count = positive_size(
    handle->penalty_count, "augmented test-shadow penalty count");
  if (Y == nullptr || Y_count != n) {
    throw std::runtime_error("augmented Y shape mismatch");
  }
  if (SP == nullptr || SP_count != penalty_count) {
    throw std::runtime_error("augmented SP shape mismatch");
  }
  if (target_index < 0 ||
      target_index >= context->capacities.target_count) {
    throw std::runtime_error(
      "augmented target_index is out of reserved range");
  }
  for (std::size_t index = 0U; index < n; ++index) {
    if (!std::isfinite(Y[index])) {
      throw std::runtime_error(
        "augmented Y must contain only finite values");
    }
  }
  for (std::size_t index = 0U; index < penalty_count; ++index) {
    if (!std::isfinite(SP[index]) || SP[index] < 0.0) {
      throw std::runtime_error(
        "augmented SP must contain only finite non-negative values");
    }
  }
  if (handle->setup_completion_event == nullptr ||
      handle->penalty_root_build_count != 1 ||
      handle->d_X_null == nullptr ||
      handle->penalty_root_offsets.size() != penalty_count ||
      handle->penalty_root_ranks.size() != penalty_count ||
      (handle->total_penalty_root_rows > 0 &&
       handle->d_penalty_roots == nullptr) ||
      (handle->has_H && handle->d_H_root == nullptr)) {
    throw std::runtime_error(
      "prepared augmented-system state is unavailable");
  }

  if (handle->total_penalty_root_rows < 0 || handle->H_root_rank < 0) {
    throw std::runtime_error(
      "augmented test-shadow root row count is invalid");
  }
  const std::size_t smooth_rows =
    static_cast<std::size_t>(handle->total_penalty_root_rows);
  const std::size_t H_rows =
    static_cast<std::size_t>(handle->H_root_rank);
  const std::size_t root_rows = checked_add(
    smooth_rows, H_rows, "augmented test-shadow root rows");
  const std::size_t rows_size = checked_add(
    n, root_rows, "augmented test-shadow rows");
  if (rows_size >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("augmented test-shadow row count overflow");
  }
  const int rows = static_cast<int>(rows_size);
  FixedSpStableWorkspace& stable = context->stable_workspace;
  if (rows > context->capacities.augmented_rows ||
      rows > stable.max_rows || handle->q > stable.max_q ||
      stable.B == nullptr || stable.c == nullptr) {
    throw std::runtime_error(
      "augmented system exceeds reserved stable workspace capacity");
  }

  const std::size_t reserved_n = positive_size(
    context->capacities.n, "reserved augmented Y rows");
  const std::size_t reserved_targets = positive_size(
    context->capacities.target_count, "reserved augmented target count");
  const std::size_t reserved_penalties = positive_size(
    context->capacities.penalty_count,
    "reserved augmented penalty count");
  if (n > reserved_n || penalty_count > reserved_penalties ||
      context->double_arena == nullptr) {
    throw std::runtime_error(
      "augmented inputs exceed reserved CUDA arena capacity");
  }
  const std::size_t Y_capacity = checked_multiply(
    reserved_n, reserved_targets, "reserved augmented Y arena");
  const std::size_t Y_offset = checked_multiply(
    static_cast<std::size_t>(target_index), reserved_n,
    "augmented target Y offset");
  const std::size_t SP_offset = checked_multiply(
    static_cast<std::size_t>(target_index), reserved_penalties,
    "augmented target SP offset");
  double* device_Y = context->double_arena + Y_offset;
  double* device_SP = context->double_arena + Y_capacity + SP_offset;

  FixedSpAugmentedSystemShadow shadow;
  shadow.leading_dimension = rows;
  shadow.rows = rows;
  shadow.cols = handle->q;
  shadow.target_index = target_index;
  shadow.B.resize(checked_multiply(
    rows_size, q, "augmented test-shadow B"));
  shadow.c.resize(rows_size);

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for augmented test shadow");
  bool stream_work_enqueued = false;
  try {
    check_cuda(cudaStreamWaitEvent(
      context->stream, handle->setup_completion_event, 0
    ), "wait for prepared setup before augmented build");
    check_cuda(cudaMemcpyAsync(
      device_Y, Y,
      allocation_bytes(n, sizeof(double), "augmented test Y upload"),
      cudaMemcpyHostToDevice, context->stream
    ), "upload augmented test Y");
    stream_work_enqueued = true;
    check_cuda(cudaMemcpyAsync(
      device_SP, SP,
      allocation_bytes(
        penalty_count, sizeof(double), "augmented test SP upload"),
      cudaMemcpyHostToDevice, context->stream
    ), "upload augmented test SP");

    const AugmentedSystemView view = build_fixed_sp_augmented_system(
      handle->d_X_null, handle->d_penalty_roots,
      handle->penalty_root_offsets.data(),
      handle->penalty_root_ranks.data(),
      handle->total_penalty_root_rows, handle->penalty_count,
      handle->d_H_root, handle->H_root_rank, device_Y, device_SP, SP,
      handle->n, handle->q, target_index, &stable, context->stream);
    if (view.B != stable.B || view.c != stable.c ||
        view.leading_dimension != rows || view.rows != rows ||
        view.cols != handle->q || view.target_index != target_index) {
      throw std::runtime_error(
        "augmented system view metadata is inconsistent");
    }
    check_cuda(cudaMemcpyAsync(
      shadow.B.data(), view.B,
      allocation_bytes(
        shadow.B.size(), sizeof(double), "augmented test-shadow B"),
      cudaMemcpyDeviceToHost, context->stream
    ), "download augmented test-shadow B");
    check_cuda(cudaMemcpyAsync(
      shadow.c.data(), view.c,
      allocation_bytes(
        shadow.c.size(), sizeof(double), "augmented test-shadow c"),
      cudaMemcpyDeviceToHost, context->stream
    ), "download augmented test-shadow c");
    check_cuda(cudaStreamSynchronize(context->stream),
               "wait for augmented test shadow");
    stream_work_enqueued = false;
  } catch (...) {
    if (stream_work_enqueued) cudaStreamSynchronize(context->stream);
    throw;
  }

  const std::size_t copied_values = checked_add(
    shadow.B.size(), shadow.c.size(),
    "augmented test-shadow copied values");
  const std::size_t copied_bytes = allocation_bytes(
    copied_values, sizeof(double),
    "augmented test-shadow D2H diagnostics");
  handle->augmented_test_shadow_d2h_count += 1;
  handle->augmented_test_shadow_d2h_bytes = checked_add(
    handle->augmented_test_shadow_d2h_bytes, copied_bytes,
    "augmented test-shadow D2H diagnostics");
  return shadow;
}

PreparedSRootsShadow test_prepared_s_roots_shadow(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  context->require_usable();
  if (handle->setup_completion_event == nullptr ||
      handle->penalty_root_build_count != 1 ||
      (handle->total_penalty_root_rows > 0 &&
       handle->d_penalty_roots == nullptr) ||
      (handle->has_H && handle->d_H_root == nullptr)) {
    throw std::runtime_error("prepared penalty root state is unavailable");
  }

  PreparedSRootsShadow shadow;
  shadow.null_dim = handle->q;
  shadow.total_penalty_root_rows = handle->total_penalty_root_rows;
  shadow.penalty_root_offsets = handle->penalty_root_offsets;
  shadow.penalty_root_ranks = handle->penalty_root_ranks;
  shadow.has_H = handle->has_H;
  shadow.H_root_rank = handle->H_root_rank;

  const std::size_t q = positive_size(
    handle->q, "prepared root shadow null dimension");
  const std::size_t penalty_root_count = checked_multiply(
    static_cast<std::size_t>(handle->total_penalty_root_rows), q,
    "prepared root shadow smooth roots");
  const std::size_t H_root_count = checked_multiply(
    static_cast<std::size_t>(handle->H_root_rank), q,
    "prepared root shadow H root");
  shadow.penalty_roots.resize(penalty_root_count);
  if (handle->has_H) shadow.H_root.resize(H_root_count);

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for prepared root shadow");
  check_cuda(cudaEventSynchronize(handle->setup_completion_event),
             "wait for prepared root setup completion");
  if (penalty_root_count > 0U) {
    check_cuda(cudaMemcpy(
      shadow.penalty_roots.data(), handle->d_penalty_roots,
      allocation_bytes(
        penalty_root_count, sizeof(double),
        "prepared root shadow smooth roots"),
      cudaMemcpyDeviceToHost
    ), "download prepared smooth penalty roots");
  }
  if (H_root_count > 0U) {
    check_cuda(cudaMemcpy2D(
      shadow.H_root.data(),
      allocation_bytes(
        static_cast<std::size_t>(handle->H_root_rank), sizeof(double),
        "prepared root shadow H destination pitch"),
      handle->d_H_root,
      allocation_bytes(q, sizeof(double),
                       "prepared root shadow H source pitch"),
      allocation_bytes(
        static_cast<std::size_t>(handle->H_root_rank), sizeof(double),
        "prepared root shadow H width"),
      q, cudaMemcpyDeviceToHost
    ), "download prepared H root");
  }

  const std::size_t copied_values = checked_add(
    penalty_root_count, H_root_count,
    "prepared root shadow copied values");
  const std::size_t copied_bytes = allocation_bytes(
    copied_values, sizeof(double), "prepared root shadow D2H diagnostics");
  handle->setup_shadow_d2h_count += 1;
  handle->setup_shadow_d2h_bytes = checked_add(
    handle->setup_shadow_d2h_bytes, copied_bytes,
    "prepared root shadow D2H diagnostics");
  return shadow;
}

PreparedSStaticShadow test_prepared_s_static_shadow(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  context->require_usable();
  if (handle->d_X_null == nullptr || handle->d_gram == nullptr ||
      handle->d_projected_penalties == nullptr ||
      (handle->has_H && handle->d_projected_H == nullptr)) {
    throw std::runtime_error("prepared static state is unavailable");
  }

  PreparedSStaticShadow shadow;
  shadow.n = handle->n;
  shadow.null_dim = handle->q;
  shadow.penalty_count = handle->penalty_count;
  shadow.has_H = handle->has_H;
  const std::size_t x_null_count =
    matrix_count(handle->n, handle->q, "test shadow X_null");
  const std::size_t gram_count =
    matrix_count(handle->q, handle->q, "test shadow Gram");
  const std::size_t projected_count = checked_multiply(
    positive_size(handle->penalty_count, "test shadow penalty count"),
    gram_count, "test shadow projected penalties");
  shadow.X_null.resize(x_null_count);
  shadow.gram.resize(gram_count);
  shadow.projected_penalties.resize(projected_count);
  if (handle->has_H) shadow.projected_H.resize(gram_count);

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for prepared test shadow");
  cudaEvent_t completion_event = nullptr;
  try {
    tracked_cuda_event_create(
      context->resource_ledger.get(), &completion_event,
      cudaEventDisableTiming, "create prepared test shadow completion event");
    if (consume_injected_prepared_static_shadow_body_failure()) {
      throw std::runtime_error(
        "INJECTED_PREPARED_STATIC_SHADOW_BODY_FAILURE");
    }
    auto download = [&](double* destination, const double* source,
                        std::size_t count, const char* name) {
      check_cuda(cudaMemcpyAsync(
        destination, source,
        allocation_bytes(count, sizeof(double), name),
        cudaMemcpyDeviceToHost, context->stream
      ), name);
    };
    download(shadow.X_null.data(), handle->d_X_null, x_null_count,
             "download prepared test shadow X_null");
    download(shadow.gram.data(), handle->d_gram, gram_count,
             "download prepared test shadow Gram");
    download(shadow.projected_penalties.data(),
             handle->d_projected_penalties, projected_count,
             "download prepared test shadow projected penalties");
    if (handle->has_H) {
      download(shadow.projected_H.data(), handle->d_projected_H, gram_count,
               "download prepared test shadow projected H");
    }
    check_cuda(cudaEventRecord(completion_event, context->stream),
               "record prepared test shadow completion");
    check_cuda(cudaEventSynchronize(completion_event),
               "wait for prepared test shadow completion");
    tracked_cuda_event_destroy(
      context->resource_ledger.get(), &completion_event,
      "destroy prepared test shadow completion event");
    if (handle->has_H) {
      handle->projected_H_test_shadow_d2h_count += 1;
      handle->projected_H_test_shadow_d2h_bytes = checked_add(
        handle->projected_H_test_shadow_d2h_bytes,
        allocation_bytes(gram_count, sizeof(double),
                         "projected H test shadow diagnostics"),
        "projected H test shadow diagnostics");
    }
  } catch (...) {
    cleanup_local_cuda_event_noexcept(
      context->resource_ledger.get(), &completion_event);
    throw;
  }
  return shadow;
}

const char* fixed_sp_status_name(FixedSpStatus status) {
  switch (status) {
    case FixedSpStatus::OkCholeskyBatched: return "OK_CHOLESKY_BATCHED";
    case FixedSpStatus::OkCholeskySingle: return "OK_CHOLESKY_SINGLE";
    case FixedSpStatus::OkAugmentedQr: return "OK_AUGMENTED_QR";
    case FixedSpStatus::OkAugmentedSvd: return "OK_AUGMENTED_SVD";
    case FixedSpStatus::ErrNonfiniteInput: return "ERR_NONFINITE_INPUT";
    case FixedSpStatus::ErrSpShapeOrOrder: return "ERR_SP_SHAPE_OR_ORDER";
    case FixedSpStatus::ErrRouteMetadata: return "ERR_ROUTE_METADATA";
    case FixedSpStatus::ErrStablePathNotImplemented:
      return "ERR_STABLE_PATH_NOT_IMPLEMENTED";
    case FixedSpStatus::ErrQrFailed: return "ERR_QR_FAILED";
    case FixedSpStatus::ErrSvdFailed: return "ERR_SVD_FAILED";
    case FixedSpStatus::ErrNonfiniteOutput: return "ERR_NONFINITE_OUTPUT";
    case FixedSpStatus::ErrInternalCuda: return "ERR_INTERNAL_CUDA";
  }
  return "UNKNOWN";
}

const char* fixed_sp_route_name(FixedSpRoute route) {
  switch (route) {
    case FixedSpRoute::Unset: return "";
    case FixedSpRoute::CholeskyBatched: return "CHOLESKY_BATCHED";
    case FixedSpRoute::AugmentedQr: return "AUGMENTED_QR";
    case FixedSpRoute::AugmentedSvd: return "AUGMENTED_SVD";
  }
  return "UNKNOWN";
}

}  // namespace fastkpc
