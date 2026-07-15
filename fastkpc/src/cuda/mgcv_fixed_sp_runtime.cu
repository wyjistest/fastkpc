#include "mgcv_fixed_sp_runtime.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fastkpc {
namespace {

constexpr std::size_t kCublasWorkspaceBytes = 16U * 1024U * 1024U;

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
};

struct FixedSpResourceCounters {
  ResourceLifecycleCounters cuda_device;
  ResourceLifecycleCounters cuda_host;
  ResourceLifecycleCounters stream;
  ResourceLifecycleCounters event;
  ResourceLifecycleCounters cublas_handle;
  ResourceLifecycleCounters cusolver_handle;
  int cleanup_error_count = 0;

  int allocation_count() const {
    return cuda_device.acquire_success_count +
      cuda_host.acquire_success_count;
  }

  int handle_create_count() const {
    return stream.acquire_success_count + event.acquire_success_count +
      cublas_handle.acquire_success_count +
      cusolver_handle.acquire_success_count;
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
};

struct FixedSpGlobalResourceLedger {
  AtomicResourceLifecycleCounters cuda_device;
  AtomicResourceLifecycleCounters cuda_host;
  AtomicResourceLifecycleCounters stream;
  AtomicResourceLifecycleCounters event;
  AtomicResourceLifecycleCounters cublas_handle;
  AtomicResourceLifecycleCounters cusolver_handle;
  std::atomic<std::int64_t> cleanup_error_count{0};
  std::atomic<int> inject_next_acquire_failure_kind{-1};
  std::atomic<int> inject_next_teardown_failure_kind{-1};
  std::atomic<bool> inject_next_blocked_consumer_launch_failure{false};
};

enum class FixedSpResourceKind {
  CudaDevice,
  CudaHost,
  Stream,
  Event,
  CublasHandle,
  CusolverHandle
};

FixedSpGlobalResourceLedger& global_resource_ledger() {
  static FixedSpGlobalResourceLedger ledger;
  return ledger;
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
    if (cudaGetDevice(&previous_device_) != cudaSuccess) {
      if (ledger_ != nullptr) record_cleanup_error(ledger_);
      return;
    }
    if (previous_device_ != device_id) {
      if (cudaSetDevice(device_id) != cudaSuccess) {
        if (ledger_ != nullptr) record_cleanup_error(ledger_);
        return;
      }
      restore_device_ = true;
    }
    ready_ = true;
  }

  ~ScopedCleanupCudaDevice() {
    if (!restore_device_ ||
        creator_pid_ != static_cast<std::int64_t>(getpid())) {
      return;
    }
    if (cudaSetDevice(previous_device_) != cudaSuccess && ledger_ != nullptr) {
      record_cleanup_error(ledger_);
    }
  }

  ScopedCleanupCudaDevice(const ScopedCleanupCudaDevice&) = delete;
  ScopedCleanupCudaDevice& operator=(const ScopedCleanupCudaDevice&) = delete;

  bool ready() const noexcept { return ready_; }

 private:
  std::int64_t creator_pid_ = -1;
  FixedSpResourceLedger* ledger_ = nullptr;
  int previous_device_ = -1;
  bool restore_device_ = false;
  bool ready_ = false;
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

template <typename T>
cudaError_t tracked_cuda_free_noexcept(FixedSpResourceLedger* ledger,
                                       T** pointer,
                                       bool* injected_failure = nullptr) noexcept {
  if (injected_failure != nullptr) *injected_failure = false;
  if (pointer == nullptr || *pointer == nullptr) return cudaSuccess;
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CudaDevice);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::CudaDevice);
  const cudaError_t status = inject ? cudaErrorUnknown : cudaFree(*pointer);
  if (inject && injected_failure != nullptr) *injected_failure = true;
  if (status == cudaSuccess) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CudaDevice);
    *pointer = nullptr;
  } else {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CudaDevice);
  }
  return status;
}

template <typename T>
void tracked_cuda_free(FixedSpResourceLedger* ledger,
                       T** pointer,
                       const char* stage) {
  bool injected_failure = false;
  const cudaError_t status = tracked_cuda_free_noexcept(
    ledger, pointer, &injected_failure);
  if (injected_failure) {
    throw std::runtime_error(
      "injected tracked CUDA device free failure");
  }
  check_cuda(status, stage);
}

template <typename T>
cudaError_t tracked_cuda_free_host_noexcept(
    FixedSpResourceLedger* ledger,
    T** pointer) noexcept {
  if (pointer == nullptr || *pointer == nullptr) return cudaSuccess;
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::CudaHost);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::CudaHost);
  const cudaError_t status = inject ? cudaErrorUnknown : cudaFreeHost(*pointer);
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::CudaHost);
    *pointer = nullptr;
  } else {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::CudaHost);
  }
  return status;
}

template <typename T>
void tracked_cuda_free_host(FixedSpResourceLedger* ledger,
                            T** pointer,
                            const char* stage) {
  check_cuda(tracked_cuda_free_host_noexcept(ledger, pointer), stage);
}

cudaError_t tracked_cuda_stream_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cudaStream_t* stream) noexcept {
  if (stream == nullptr || *stream == nullptr) return cudaSuccess;
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::Stream);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::Stream);
  const cudaError_t status = inject ? cudaErrorUnknown :
    cudaStreamDestroy(*stream);
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::Stream);
    *stream = nullptr;
  } else {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::Stream);
  }
  return status;
}

cudaError_t tracked_cuda_event_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cudaEvent_t* event) noexcept {
  if (event == nullptr || *event == nullptr) return cudaSuccess;
  record_resource_teardown_attempt(ledger, FixedSpResourceKind::Event);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::Event);
  const cudaError_t status = inject ? cudaErrorUnknown :
    cudaEventDestroy(*event);
  if (status == cudaSuccess) {
    record_resource_teardown_success(ledger, FixedSpResourceKind::Event);
    *event = nullptr;
  } else {
    record_resource_teardown_failure(ledger, FixedSpResourceKind::Event);
  }
  return status;
}

void tracked_cuda_event_destroy(FixedSpResourceLedger* ledger,
                                cudaEvent_t* event,
                                const char* stage) {
  check_cuda(tracked_cuda_event_destroy_noexcept(ledger, event), stage);
}

cublasStatus_t tracked_cublas_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cublasHandle_t* handle) noexcept {
  if (handle == nullptr || *handle == nullptr) return CUBLAS_STATUS_SUCCESS;
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CublasHandle);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::CublasHandle);
  const cublasStatus_t status = inject ? CUBLAS_STATUS_INTERNAL_ERROR :
    cublasDestroy(*handle);
  if (status == CUBLAS_STATUS_SUCCESS) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CublasHandle);
    *handle = nullptr;
  } else {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CublasHandle);
  }
  return status;
}

cusolverStatus_t tracked_cusolver_destroy_noexcept(
    FixedSpResourceLedger* ledger,
    cusolverDnHandle_t* handle) noexcept {
  if (handle == nullptr || *handle == nullptr) {
    return CUSOLVER_STATUS_SUCCESS;
  }
  record_resource_teardown_attempt(
    ledger, FixedSpResourceKind::CusolverHandle);
  const bool inject = consume_injected_resource_teardown_failure(
    FixedSpResourceKind::CusolverHandle);
  const cusolverStatus_t status = inject ? CUSOLVER_STATUS_INTERNAL_ERROR :
    cusolverDnDestroy(*handle);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    record_resource_teardown_success(
      ledger, FixedSpResourceKind::CusolverHandle);
    *handle = nullptr;
  } else {
    record_resource_teardown_failure(
      ledger, FixedSpResourceKind::CusolverHandle);
  }
  return status;
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
  }
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

__global__ void build_fixed_sp_systems(
    const double* gram,
    const double* projected_penalties,
    const double* sp,
    int q,
    int penalty_count,
    int target_count,
    double* systems) {
  const std::size_t index =
    static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t q_squared =
    static_cast<std::size_t>(q) * static_cast<std::size_t>(q);
  const std::size_t system_count =
    q_squared * static_cast<std::size_t>(target_count);
  if (index >= system_count) return;

  const std::size_t target = index / q_squared;
  const std::size_t element = index - target * q_squared;
  double value = gram[element];
  for (int penalty = 0; penalty < penalty_count; ++penalty) {
    value += sp[static_cast<std::size_t>(penalty) +
                static_cast<std::size_t>(penalty_count) * target] *
      projected_penalties[element + q_squared *
        static_cast<std::size_t>(penalty)];
  }
  systems[index] = value;
}

__global__ void build_fixed_sp_residual_and_rss(
    const double* y,
    const double* fitted,
    int n,
    double* residuals,
    double* rss,
    bool write_residuals,
    bool write_rss) {
  extern __shared__ double sums[];
  double local_sum = 0.0;
  for (int row = static_cast<int>(threadIdx.x);
       row < n;
       row += static_cast<int>(blockDim.x)) {
    const double residual = y[row] - fitted[row];
    if (write_residuals) residuals[row] = residual;
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
  if (threadIdx.x == 0U && write_rss) *rss = sums[0];
}

__global__ void check_fixed_sp_outputs_finite(
    const double* rhs,
    const double* theta,
    const double* coefficients,
    const double* fitted,
    const double* residuals,
    const double* rss,
    int n,
    int p,
    int q,
    int target_count,
    std::uint32_t output_mask,
    int* finite_status) {
  const int target =
    static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
    static_cast<int>(threadIdx.x);
  if (target >= target_count) return;

  bool finite = true;
  const std::size_t target_offset = static_cast<std::size_t>(target);
  const double* target_rhs = rhs + static_cast<std::size_t>(q) * target_offset;
  const double* target_theta =
    theta + static_cast<std::size_t>(q) * target_offset;
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
  bool device_ready = false;
  cudaError_t stream_synchronize = cudaSuccess;
  cudaError_t event_destroy = cudaSuccess;
  cudaError_t stream_destroy = cudaSuccess;
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
      status.device_ready = true;
      return status;
    }
    if (creator_pid_ != static_cast<std::int64_t>(getpid())) return status;

    signal_noexcept();
    FixedSpResourceLedger* ledger = resource_ledger_.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid_, device_id_, ledger);
    if (!cleanup_device.ready()) return status;
    status.device_ready = true;

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
  if (!status.device_ready) {
    throw std::runtime_error(
      "select CUDA device for blocked consumer test teardown");
  }
  check_cuda(status.stream_synchronize,
             "synchronize blocked consumer test stream");
  check_cuda(status.event_destroy, "destroy blocked consumer test event");
  check_cuda(status.stream_destroy, "destroy blocked consumer test stream");
}

}  // namespace

class CudaRuntimeContext {
 public:
  explicit CudaRuntimeContext(int requested_device);
  ~CudaRuntimeContext();
  void reserve(const FixedSpCapacities& requested_capacities);
  FixedSpRuntimeInfo info() const;
  void require_usable() const;

  int device_id = -1;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 1;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cusolverDnHandle_t solver = nullptr;
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
  std::size_t cublas_workspace_bytes = kCublasWorkspaceBytes;
  int potrf_lwork = 0;
  FixedSpCapacities capacities;
  std::shared_ptr<FixedSpResourceLedger> resource_ledger =
    std::make_shared<FixedSpResourceLedger>();
  FixedSpRuntimeInfo diagnostics;
  std::atomic<int> active_prepared_handle_count{0};
  bool freed = false;
  mutable std::mutex mutex;

 private:
  void cleanup_noexcept() noexcept;
};

enum class TransientResidualSlotState {
  Free,
  Leased,
  ConsumerRegistrationPending,
  Poisoned
};

struct TransientResidualSlot {
  ~TransientResidualSlot() { cleanup_noexcept(); }
  void cleanup_noexcept() noexcept {
    if (freed) return;
    if (creator_pid != static_cast<std::int64_t>(getpid())) {
      freed = true;
      return;
    }
    FixedSpResourceLedger* ledger = resource_ledger.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid, device_id, ledger);
    if (cleanup_device.ready()) {
      tracked_cuda_event_destroy_noexcept(
        ledger, &consumer_completion_event);
      tracked_cuda_event_destroy_noexcept(
        ledger, &solve_completion_event);
      tracked_cuda_free_host_noexcept(ledger, &host_finite_status);
      tracked_cuda_free_noexcept(ledger, &rss);
      tracked_cuda_free_noexcept(ledger, &rhs);
      tracked_cuda_free_noexcept(ledger, &residuals);
      tracked_cuda_free_noexcept(ledger, &fitted);
      tracked_cuda_free_noexcept(ledger, &coefficients);
    }
    freed = true;
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
  bool freed = false;
};

class PreparedSGpuHandle {
 public:
  ~PreparedSGpuHandle() { cleanup_noexcept(); }

  void cleanup_noexcept() noexcept {
    if (freed) return;
    if (creator_pid != static_cast<std::int64_t>(getpid())) {
      residual_slot.reset();
      context.reset();
      resource_ledger.reset();
      registered_with_context = false;
      freed = true;
      return;
    }
    FixedSpResourceLedger* ledger = resource_ledger.get();
    ScopedCleanupCudaDevice cleanup_device(
      creator_pid, device_id, ledger);
    if (cleanup_device.ready()) {
      tracked_cuda_event_destroy_noexcept(ledger, &setup_completion_event);
      residual_slot.reset();
      tracked_cuda_free_noexcept(ledger, &d_projected_penalties);
      tracked_cuda_free_noexcept(ledger, &d_gram);
      if (d_X_null != d_X) tracked_cuda_free_noexcept(ledger, &d_X_null);
      d_X_null = nullptr;
      tracked_cuda_free_noexcept(ledger, &d_Z);
      tracked_cuda_free_noexcept(ledger, &d_X);
    } else {
      residual_slot.reset();
    }
    if (registered_with_context && context) {
      context->active_prepared_handle_count.fetch_sub(
        1, std::memory_order_acq_rel);
      registered_with_context = false;
    }
    context.reset();
    generation += 1;
    freed = true;
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
  std::shared_ptr<TransientResidualSlot> residual_slot;
  cudaEvent_t setup_completion_event = nullptr;
  int setup_h2d_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  bool registered_with_context = false;
  bool freed = false;
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
  if (!handle || handle->freed) {
    throw std::runtime_error("fixed-sp CUDA prepared handle has been freed");
  }
  if (handle->creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle has a different creator PID");
  }
  const std::shared_ptr<CudaRuntimeContext> context = handle->context;
  if (!context || context->freed) {
    throw std::runtime_error("fixed-sp CUDA prepared runtime has been freed");
  }
  if (context->creator_pid != handle->creator_pid ||
      handle->device_id != context->device_id ||
      handle->context_generation != context->generation) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared handle ownership mismatch");
  }
  if (!handle->residual_slot || handle->residual_slot->freed ||
      handle->residual_slot->creator_pid != handle->creator_pid ||
      handle->residual_slot->device_id != handle->device_id) {
    throw std::runtime_error(
      "fixed-sp CUDA prepared output-slot ownership mismatch");
  }
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
  if (!token->owner || token->owner->freed ||
      token->owner->creator_pid != token->creator_pid ||
      token->owner_generation != token->owner->generation) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error(
      "STALE_TOKEN: residual owner generation mismatch");
  }
  const std::shared_ptr<CudaRuntimeContext> context = token->owner->context;
  if (!context || context->freed) {
    throw std::runtime_error("fixed-sp CUDA residual runtime has been freed");
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
  if (token->lease_released || !token->slot || token->slot->freed) {
    token->diagnostics.stale_token_reject_count += 1;
    throw std::runtime_error("STALE_TOKEN: residual lease has been released");
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

  std::unordered_set<std::string> unique_keys;
  unique_keys.reserve(targets);
  for (std::size_t index = 0; index < targets; ++index) {
    const FixedSpRoute route = batch.planned_routes[index];
    if (route != FixedSpRoute::CholeskyBatched &&
        route != FixedSpRoute::AugmentedQr &&
        route != FixedSpRoute::AugmentedSvd) {
      throw std::runtime_error("fixed-sp planned route metadata is invalid");
    }
    const std::string& target_key = batch.target_keys[index];
    const bool valid_target_key = target_key.size() == 64U &&
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
  }
  token->diagnostics.solver_statuses = token->solver_statuses;
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
  if (freed) return;
  if (creator_pid != static_cast<std::int64_t>(getpid())) {
    diagnostics.freed = true;
    freed = true;
    return;
  }
  FixedSpResourceLedger* ledger = resource_ledger.get();
  ScopedCleanupCudaDevice cleanup_device(creator_pid, device_id, ledger);
  if (cleanup_device.ready()) {
    tracked_cuda_free_noexcept(ledger, &double_arena);
    tracked_cuda_free_noexcept(ledger, &int_arena);
    tracked_cuda_free_host_noexcept(ledger, &host_status_arena);
    tracked_cuda_free_noexcept(ledger, &pointer_arena);
    tracked_cuda_free_noexcept(ledger, &cublas_workspace);

    tracked_cuda_event_destroy_noexcept(
      ledger, &cholesky_solve_checkpoint_event);
    tracked_cuda_event_destroy_noexcept(
      ledger, &cholesky_factor_checkpoint_event);
    tracked_cusolver_destroy_noexcept(ledger, &solver);
    tracked_cublas_destroy_noexcept(ledger, &blas);
    tracked_cuda_stream_destroy_noexcept(ledger, &stream);
  }

  generation += 1;
  freed = true;
  diagnostics.freed = true;
  diagnostics.generation = generation;
}

void CudaRuntimeContext::require_usable() const {
  if (freed) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
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

  const bool within_existing_capacity =
    requested_capacities.n <= capacities.n &&
    requested_capacities.null_dim <= capacities.null_dim &&
    requested_capacities.target_count <= capacities.target_count &&
    requested_capacities.penalty_count <= capacities.penalty_count &&
    requested_capacities.augmented_rows <= capacities.augmented_rows;
  if (within_existing_capacity) {
    diagnostics.workspace_reserve_count += 1;
    return;
  }
  if (active_prepared_handle_count.load(std::memory_order_acquire) > 0) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime cannot grow with active prepared handles");
  }

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
      tracked_cuda_free_noexcept(resource_ledger.get(), &probe);
      throw;
    }
    if (requested_potrf_lwork < 0) {
      throw std::runtime_error("cuSOLVER potrf workspace size is invalid");
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
  const std::size_t int_required = checked_add(targets, 1U, "int arena");
  const std::size_t pointer_required =
    checked_multiply(targets, 2U, "pointer arena");

  double* new_double_arena = nullptr;
  int* new_int_arena = nullptr;
  int* new_host_status_arena = nullptr;
  void** new_pointer_arena = nullptr;
  void* new_cublas_workspace = nullptr;
  std::size_t new_cublas_alignment = diagnostics.cublas_workspace_alignment;
  auto cleanup_new_allocations = [&]() noexcept {
    tracked_cuda_free_noexcept(resource_ledger.get(), &new_double_arena);
    tracked_cuda_free_noexcept(resource_ledger.get(), &new_int_arena);
    tracked_cuda_free_host_noexcept(
      resource_ledger.get(), &new_host_status_arena);
    tracked_cuda_free_noexcept(resource_ledger.get(), &new_pointer_arena);
    tracked_cuda_free_noexcept(resource_ledger.get(), &new_cublas_workspace);
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
    if (int_required > host_status_capacity) {
      tracked_cuda_malloc_host(
        resource_ledger.get(),
        &new_host_status_arena,
        allocation_bytes(int_required, sizeof(int), "host status arena"),
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
      if (new_cublas_alignment < 256U) {
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
  try {
    if (replace_double) {
      tracked_cuda_free(
        resource_ledger.get(), &double_arena, "free old fixed-sp double arena");
      double_arena = new_double_arena;
      new_double_arena = nullptr;
    }
    if (replace_int) {
      tracked_cuda_free(
        resource_ledger.get(), &int_arena, "free old fixed-sp int arena");
      int_arena = new_int_arena;
      new_int_arena = nullptr;
    }
    if (replace_host_status) {
      tracked_cuda_free_host(
        resource_ledger.get(), &host_status_arena,
        "free old fixed-sp host status arena");
      host_status_arena = new_host_status_arena;
      new_host_status_arena = nullptr;
    }
    if (replace_pointer) {
      tracked_cuda_free(
        resource_ledger.get(), &pointer_arena,
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
    cleanup_new_allocations();
    throw;
  }

  if (replace_double) double_capacity = double_required;
  if (replace_int) int_capacity = int_required;
  if (replace_host_status) host_status_capacity = int_required;
  if (replace_pointer) pointer_capacity = pointer_required;
  const bool grew = replace_double || replace_int || replace_host_status ||
    replace_pointer || install_cublas_workspace;

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
  {
    std::lock_guard<std::mutex> lock(owned_context->mutex);
    owned_context->require_usable();
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

void test_exercise_fixed_sp_cuda_resource_teardown_failure(
    const std::string& resource) {
  const FixedSpResourceKind kind = fixed_sp_resource_kind_from_name(resource);
  FixedSpResourceLedger ledger;
  check_cuda(cudaSetDevice(0), "set CUDA device for teardown failure test");

  auto require_cuda_retry = [&](cudaError_t first, cudaError_t second,
                                const void* retained) {
    if (first == cudaSuccess || second != cudaSuccess || retained != nullptr) {
      throw std::runtime_error(
        "tracked fixed-sp CUDA teardown failure retry invariant failed");
    }
  };
  switch (kind) {
    case FixedSpResourceKind::CudaDevice: {
      void* pointer = nullptr;
      tracked_cuda_malloc(
        &ledger, &pointer, 1U, "allocate teardown test device byte");
      arm_injected_resource_teardown_failure(kind);
      const cudaError_t first = tracked_cuda_free_noexcept(
        &ledger, &pointer);
      if (first == cudaSuccess || pointer == nullptr) {
        tracked_cuda_free_noexcept(&ledger, &pointer);
        throw std::runtime_error(
          "tracked fixed-sp device teardown did not retain ownership");
      }
      const cudaError_t second = tracked_cuda_free_noexcept(
        &ledger, &pointer);
      require_cuda_retry(first, second, pointer);
      break;
    }
    case FixedSpResourceKind::CudaHost: {
      void* pointer = nullptr;
      tracked_cuda_malloc_host(
        &ledger, &pointer, 1U, "allocate teardown test host byte");
      arm_injected_resource_teardown_failure(kind);
      const cudaError_t first = tracked_cuda_free_host_noexcept(
        &ledger, &pointer);
      if (first == cudaSuccess || pointer == nullptr) {
        tracked_cuda_free_host_noexcept(&ledger, &pointer);
        throw std::runtime_error(
          "tracked fixed-sp host teardown did not retain ownership");
      }
      const cudaError_t second = tracked_cuda_free_host_noexcept(
        &ledger, &pointer);
      require_cuda_retry(first, second, pointer);
      break;
    }
    case FixedSpResourceKind::Stream: {
      cudaStream_t stream = nullptr;
      tracked_cuda_stream_create(
        &ledger, &stream, cudaStreamNonBlocking,
        "create teardown test stream");
      arm_injected_resource_teardown_failure(kind);
      const cudaError_t first = tracked_cuda_stream_destroy_noexcept(
        &ledger, &stream);
      if (first == cudaSuccess || stream == nullptr) {
        tracked_cuda_stream_destroy_noexcept(&ledger, &stream);
        throw std::runtime_error(
          "tracked fixed-sp stream teardown did not retain ownership");
      }
      const cudaError_t second = tracked_cuda_stream_destroy_noexcept(
        &ledger, &stream);
      require_cuda_retry(first, second, stream);
      break;
    }
    case FixedSpResourceKind::Event: {
      cudaEvent_t event = nullptr;
      tracked_cuda_event_create(
        &ledger, &event, cudaEventDisableTiming,
        "create teardown test event");
      arm_injected_resource_teardown_failure(kind);
      const cudaError_t first = tracked_cuda_event_destroy_noexcept(
        &ledger, &event);
      if (first == cudaSuccess || event == nullptr) {
        tracked_cuda_event_destroy_noexcept(&ledger, &event);
        throw std::runtime_error(
          "tracked fixed-sp event teardown did not retain ownership");
      }
      const cudaError_t second = tracked_cuda_event_destroy_noexcept(
        &ledger, &event);
      require_cuda_retry(first, second, event);
      break;
    }
    case FixedSpResourceKind::CublasHandle: {
      cublasHandle_t handle = nullptr;
      tracked_cublas_create(
        &ledger, &handle, "create teardown test cuBLAS handle");
      arm_injected_resource_teardown_failure(kind);
      const cublasStatus_t first = tracked_cublas_destroy_noexcept(
        &ledger, &handle);
      if (first == CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        tracked_cublas_destroy_noexcept(&ledger, &handle);
        throw std::runtime_error(
          "tracked fixed-sp cuBLAS teardown did not retain ownership");
      }
      const cublasStatus_t second = tracked_cublas_destroy_noexcept(
        &ledger, &handle);
      if (second != CUBLAS_STATUS_SUCCESS || handle != nullptr) {
        tracked_cublas_destroy_noexcept(&ledger, &handle);
        throw std::runtime_error(
          "tracked fixed-sp cuBLAS teardown failure retry invariant failed");
      }
      break;
    }
    case FixedSpResourceKind::CusolverHandle: {
      cusolverDnHandle_t handle = nullptr;
      tracked_cusolver_create(
        &ledger, &handle, "create teardown test cuSOLVER handle");
      arm_injected_resource_teardown_failure(kind);
      const cusolverStatus_t first = tracked_cusolver_destroy_noexcept(
        &ledger, &handle);
      if (first == CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
        tracked_cusolver_destroy_noexcept(&ledger, &handle);
        throw std::runtime_error(
          "tracked fixed-sp cuSOLVER teardown did not retain ownership");
      }
      const cusolverStatus_t second = tracked_cusolver_destroy_noexcept(
        &ledger, &handle);
      if (second != CUSOLVER_STATUS_SUCCESS || handle != nullptr) {
        tracked_cusolver_destroy_noexcept(&ledger, &handle);
        throw std::runtime_error(
          "tracked fixed-sp cuSOLVER teardown failure retry invariant failed");
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

std::shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu(
    const std::shared_ptr<CudaRuntimeContext>& context,
    const PreparedSHostView& setup) {
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
  require_finite_derived(nullspace_design, "X_null");
  require_finite_derived(projected_penalties, "projected penalties");
  const std::size_t x_count =
    matrix_count(setup.n, setup.coefficient_dim, "prepared X");
  const std::size_t z_count = setup.Z == nullptr ? 0U :
    matrix_count(setup.coefficient_dim, setup.null_dim, "prepared Z");
  const std::size_t x_null_count = setup.Z == nullptr ? 0U :
    matrix_count(setup.n, setup.null_dim, "prepared nullspace design");
  const std::size_t gram_count =
    matrix_count(setup.null_dim, setup.null_dim, "prepared Gram");
  const std::size_t penalty_count = projected_penalties.size();
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
    check_cuda(cudaEventRecord(
      handle->setup_completion_event, context->stream
    ), "record prepared setup completion");
    check_cuda(cudaEventSynchronize(handle->setup_completion_event),
               "wait for prepared setup completion");
    tracked_cuda_event_destroy(
      context->resource_ledger.get(), &handle->setup_completion_event,
      "destroy prepared setup completion event");
    handle->setup_h2d_upload_count = 1;
  } catch (...) {
    handle->cleanup_noexcept();
    throw;
  }
  context->active_prepared_handle_count.fetch_add(
    1, std::memory_order_release);
  handle->registered_with_context = true;
  return handle;
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
    require_prepared_host_identity(owned_handle);
  {
    std::lock_guard<std::mutex> lock(context->mutex);
    require_prepared_host_identity(owned_handle);
    context->require_usable();
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

    token->diagnostics.n = batch.n;
    token->diagnostics.coefficient_dim = handle->p;
    token->diagnostics.target_count = batch.target_count;
    token->diagnostics.native_batch_call = true;
    token->diagnostics.output_slot_acquire_count = 1;
    token->diagnostics.owner_generation = handle->generation;
    token->diagnostics.slot_generation = acquired_generation;
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

    const std::size_t y_count = checked_multiply(
      static_cast<std::size_t>(batch.n), targets, "fixed-sp solve Y");
    const std::size_t sp_count = checked_multiply(
      static_cast<std::size_t>(batch.penalty_count), targets,
      "fixed-sp solve SP");
    check_cuda(cudaMemcpyAsync(
      d_y, batch.Y,
      allocation_bytes(y_count, sizeof(double), "fixed-sp solve Y"),
      cudaMemcpyHostToDevice, context->stream
    ), "upload fixed-sp solve Y");
    check_cuda(cudaMemcpyAsync(
      d_sp, batch.SP,
      allocation_bytes(sp_count, sizeof(double), "fixed-sp solve SP"),
      cudaMemcpyHostToDevice, context->stream
    ), "upload fixed-sp solve SP");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(cublasDgemm(
      context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      handle->q, batch.target_count, batch.n,
      &one, handle->d_X_null, batch.n,
      d_y, batch.n, &zero, d_rhs, handle->q
    ), "build fixed-sp RHS with X_null transpose Y");
    token->diagnostics.rhs_device_build_count = 1;
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

    const std::size_t q_squared = matrix_count(
      handle->q, handle->q, "fixed-sp solve system");
    const std::size_t system_count = checked_multiply(
      q_squared, targets, "fixed-sp solve systems");
    const std::size_t system_blocks =
      (system_count + threads - 1U) / threads;
    if (system_blocks == 0U ||
        system_blocks > std::numeric_limits<unsigned int>::max()) {
      throw std::runtime_error("fixed-sp system build size overflow");
    }
    build_fixed_sp_systems<<<
      static_cast<unsigned int>(system_blocks), threads, 0, context->stream
    >>>(
      handle->d_gram, handle->d_projected_penalties, d_sp,
      handle->q, handle->penalty_count, batch.target_count, d_systems);
    check_cuda(cudaGetLastError(), "launch fixed-sp system build");

    for (FixedSpRoute route : batch.planned_routes) {
      switch (route) {
        case FixedSpRoute::CholeskyBatched:
          token->diagnostics.planned_cholesky_target_count += 1;
          break;
        case FixedSpRoute::AugmentedQr:
          token->diagnostics.planned_qr_target_count += 1;
          break;
        case FixedSpRoute::AugmentedSvd:
          token->diagnostics.planned_svd_target_count += 1;
          break;
        case FixedSpRoute::Unset:
          throw std::runtime_error(
            "fixed-sp planned route metadata is invalid");
      }
    }

    for (int target = 0; target < batch.target_count; ++target) {
      const std::size_t target_offset = static_cast<std::size_t>(target);
      if (batch.planned_routes[target_offset] !=
          FixedSpRoute::CholeskyBatched) {
        continue;
      }

      double* system = d_systems + q_squared * target_offset;
      double* theta = d_theta +
        static_cast<std::size_t>(handle->q) * target_offset;
      const double* rhs = d_rhs +
        static_cast<std::size_t>(handle->q) * target_offset;
      check_cuda(cudaMemcpyAsync(
        theta, rhs,
        allocation_bytes(static_cast<std::size_t>(handle->q), sizeof(double),
                         "fixed-sp working RHS"),
        cudaMemcpyDeviceToDevice, context->stream
      ), "copy fixed-sp working RHS");

      int* d_factor_info = context->int_arena + target_offset;
      int* h_factor_info = context->host_status_arena + target_offset;
      check_cusolver(cusolverDnDpotrf(
        context->solver, CUBLAS_FILL_MODE_UPPER, handle->q,
        system, handle->q, d_potrf_work, context->potrf_lwork,
        d_factor_info
      ), "factor fixed-sp Cholesky system");
      check_cuda(cudaMemcpyAsync(
        h_factor_info, d_factor_info, sizeof(int),
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
      if (*h_factor_info < 0) {
        throw std::runtime_error(
          "fixed-sp Cholesky potrf reported an invalid argument");
      }
      if (*h_factor_info > 0) {
        token->reroute_reasons[target_offset] =
          "CHOLESKY_NON_POSITIVE_PIVOT";
        token->diagnostics.stable_reroute_count += 1;
        token->diagnostics.cholesky_to_svd_count += 1;
        continue;
      }

      int* d_solve_info = context->int_arena + reserved_targets;
      int* h_solve_info = context->host_status_arena + reserved_targets;
      check_cusolver(cusolverDnDpotrs(
        context->solver, CUBLAS_FILL_MODE_UPPER, handle->q, 1,
        system, handle->q, theta, handle->q, d_solve_info
      ), "solve fixed-sp Cholesky system");
      check_cuda(cudaMemcpyAsync(
        h_solve_info, d_solve_info, sizeof(int),
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
      if (*h_solve_info != 0) {
        throw std::runtime_error(
          "fixed-sp Cholesky potrs reported a batch failure");
      }

      if ((batch.output_mask & FixedSpOutputCoefficients) != 0U) {
        double* coefficients = slot->coefficients +
          static_cast<std::size_t>(handle->p) * target_offset;
        if (handle->d_Z == nullptr) {
          check_cuda(cudaMemcpyAsync(
            coefficients, theta,
            allocation_bytes(static_cast<std::size_t>(handle->p),
                             sizeof(double), "fixed-sp coefficients"),
            cudaMemcpyDeviceToDevice, context->stream
          ), "copy identity-nullspace fixed-sp coefficients");
        } else {
          check_cublas(cublasDgemv(
            context->blas, CUBLAS_OP_N, handle->p, handle->q,
            &one, handle->d_Z, handle->p, theta, 1,
            &zero, coefficients, 1
          ), "expand fixed-sp coefficients through Z");
        }
      }

      const bool needs_fitted =
        (batch.output_mask & (FixedSpOutputFitted |
                              FixedSpOutputResiduals |
                              FixedSpOutputRss)) != 0U;
      if (needs_fitted) {
        double* fitted = slot->fitted +
          static_cast<std::size_t>(batch.n) * target_offset;
        check_cublas(cublasDgemv(
          context->blas, CUBLAS_OP_N, batch.n, handle->q,
          &one, handle->d_X_null, batch.n, theta, 1,
          &zero, fitted, 1
        ), "build fixed-sp fitted values");

        const bool write_residuals =
          (batch.output_mask & FixedSpOutputResiduals) != 0U;
        const bool write_rss =
          (batch.output_mask & FixedSpOutputRss) != 0U;
        if (write_residuals || write_rss) {
          const double* y = d_y +
            static_cast<std::size_t>(batch.n) * target_offset;
          double* residuals = slot->residuals +
            static_cast<std::size_t>(batch.n) * target_offset;
          build_fixed_sp_residual_and_rss<<<
            1U, threads, threads * sizeof(double), context->stream
          >>>(
            y, fitted, batch.n, residuals, slot->rss + target_offset,
            write_residuals, write_rss);
          check_cuda(cudaGetLastError(),
                     "launch fixed-sp residual and RSS build");
        }
      }

      token->executed_routes[target_offset] = FixedSpRoute::CholeskyBatched;
      token->solver_statuses[target_offset] = FixedSpStatus::OkCholeskySingle;
      token->diagnostics.executed_cholesky_target_count += 1;
    }

    const std::size_t finite_status_blocks =
      (targets + threads - 1U) / threads;
    if (finite_status_blocks == 0U ||
        finite_status_blocks > std::numeric_limits<unsigned int>::max() ||
        targets > slot->finite_status_capacity ||
        slot->host_finite_status == nullptr) {
      throw std::runtime_error("fixed-sp finite status capacity mismatch");
    }
    check_fixed_sp_outputs_finite<<<
      static_cast<unsigned int>(finite_status_blocks), threads,
      0, context->stream
    >>>(
      d_rhs, d_theta, slot->coefficients, slot->fitted,
      slot->residuals, slot->rss, batch.n, handle->p, handle->q,
      batch.target_count, batch.output_mask, context->int_arena);
    check_cuda(cudaGetLastError(), "launch fixed-sp output finite check");

    // Device flags use reserved context scratch. The pinned destination is
    // handle-owned so another prepared handle cannot overwrite an unresolved
    // token's compact status before observation.
    check_cuda(cudaMemcpyAsync(
      slot->host_finite_status, context->int_arena,
      allocation_bytes(targets, sizeof(int), "fixed-sp finite status"),
      cudaMemcpyDeviceToHost, context->stream
    ), "copy fixed-sp finite status");
    check_cuda(cudaEventRecord(slot->solve_completion_event, context->stream),
               "record fixed-sp solve completion");

    token->diagnostics.target_keys = token->target_keys;
    token->diagnostics.planned_routes = token->planned_routes;
    token->diagnostics.executed_routes = token->executed_routes;
    token->diagnostics.reroute_reasons = token->reroute_reasons;
    token->diagnostics.solver_statuses = token->solver_statuses;
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

PreparedSStaticShadow test_prepared_s_static_shadow(
    const std::shared_ptr<PreparedSGpuHandle>& handle) {
  const std::shared_ptr<CudaRuntimeContext> context =
    require_prepared_host_identity(handle);
  std::lock_guard<std::mutex> lock(context->mutex);
  require_prepared_host_identity(handle);
  context->require_usable();
  if (handle->d_X_null == nullptr || handle->d_gram == nullptr ||
      handle->d_projected_penalties == nullptr) {
    throw std::runtime_error("prepared static state is unavailable");
  }

  PreparedSStaticShadow shadow;
  shadow.n = handle->n;
  shadow.null_dim = handle->q;
  shadow.penalty_count = handle->penalty_count;
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

  check_cuda(cudaSetDevice(context->device_id),
             "set CUDA device for prepared test shadow");
  cudaEvent_t completion_event = nullptr;
  try {
    tracked_cuda_event_create(
      context->resource_ledger.get(), &completion_event,
      cudaEventDisableTiming, "create prepared test shadow completion event");
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
    check_cuda(cudaEventRecord(completion_event, context->stream),
               "record prepared test shadow completion");
    check_cuda(cudaEventSynchronize(completion_event),
               "wait for prepared test shadow completion");
    tracked_cuda_event_destroy(
      context->resource_ledger.get(), &completion_event,
      "destroy prepared test shadow completion event");
  } catch (...) {
    tracked_cuda_event_destroy_noexcept(
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
