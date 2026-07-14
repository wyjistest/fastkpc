#include "mgcv_fixed_sp_runtime.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

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
  FixedSpRuntimeInfo diagnostics;
  bool freed = false;
  mutable std::mutex mutex;

 private:
  void cleanup_noexcept() noexcept;
};

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
    diagnostics.compute_capability_major = properties.major;
    diagnostics.compute_capability_minor = properties.minor;
    diagnostics.sm_count = properties.multiProcessorCount;
    check_cuda(cudaRuntimeGetVersion(&diagnostics.cuda_toolkit_version),
               "query CUDA runtime version");
    check_cuda(cudaDriverGetVersion(&diagnostics.cuda_driver_version),
               "query CUDA driver version");

    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create CUDA stream");
    diagnostics.stream_create_count = 1;

    check_cublas(cublasCreate(&blas), "create cuBLAS handle");
    diagnostics.cublas_handle_create_count = 1;
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

    check_cusolver(cusolverDnCreate(&solver), "create cuSOLVER handle");
    diagnostics.cusolver_handle_create_count = 1;
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

    check_cuda(cudaEventCreateWithFlags(
      &cholesky_factor_checkpoint_event, cudaEventDisableTiming
    ), "create Cholesky factor checkpoint event");
    check_cuda(cudaEventCreateWithFlags(
      &cholesky_solve_checkpoint_event, cudaEventDisableTiming
    ), "create Cholesky solve checkpoint event");
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
  if (device_id >= 0) cudaSetDevice(device_id);

  cudaFree(double_arena);
  double_arena = nullptr;
  cudaFree(int_arena);
  int_arena = nullptr;
  cudaFreeHost(host_status_arena);
  host_status_arena = nullptr;
  cudaFree(pointer_arena);
  pointer_arena = nullptr;
  cudaFree(cublas_workspace);
  cublas_workspace = nullptr;

  if (cholesky_solve_checkpoint_event != nullptr) {
    cudaEventDestroy(cholesky_solve_checkpoint_event);
    cholesky_solve_checkpoint_event = nullptr;
  }
  if (cholesky_factor_checkpoint_event != nullptr) {
    cudaEventDestroy(cholesky_factor_checkpoint_event);
    cholesky_factor_checkpoint_event = nullptr;
  }
  if (solver != nullptr) {
    cusolverDnDestroy(solver);
    solver = nullptr;
  }
  if (blas != nullptr) {
    cublasDestroy(blas);
    blas = nullptr;
  }
  if (stream != nullptr) {
    cudaStreamDestroy(stream);
    stream = nullptr;
  }

  freed = true;
  diagnostics.freed = true;
}

void CudaRuntimeContext::require_usable() const {
  if (freed) {
    throw std::runtime_error("fixed-sp CUDA runtime has been freed");
  }
  if (creator_pid != static_cast<std::int64_t>(getpid())) {
    throw std::runtime_error(
      "fixed-sp CUDA runtime cannot be used from a different process");
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
      check_cuda(cudaMalloc(&probe, probe_bytes), "allocate potrf probe");
      check_cusolver(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, merged_capacities.null_dim,
        probe, merged_capacities.null_dim, &requested_potrf_lwork
      ), "query cuSOLVER potrf workspace");
      check_cuda(cudaFree(probe), "free potrf probe");
      probe = nullptr;
    } catch (...) {
      cudaFree(probe);
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

  try {
    if (double_required > double_capacity) {
      check_cuda(cudaMalloc(
        &new_double_arena,
        allocation_bytes(double_required, sizeof(double), "double arena")
      ), "allocate fixed-sp double arena");
    }
    if (int_required > int_capacity) {
      check_cuda(cudaMalloc(
        &new_int_arena,
        allocation_bytes(int_required, sizeof(int), "int arena")
      ), "allocate fixed-sp int arena");
    }
    if (int_required > host_status_capacity) {
      check_cuda(cudaMallocHost(
        &new_host_status_arena,
        allocation_bytes(int_required, sizeof(int), "host status arena")
      ), "allocate fixed-sp host status arena");
    }
    if (pointer_required > pointer_capacity) {
      check_cuda(cudaMalloc(
        &new_pointer_arena,
        allocation_bytes(pointer_required, sizeof(void*), "pointer arena")
      ), "allocate fixed-sp pointer arena");
    }
    if (cublas_workspace == nullptr) {
      check_cuda(cudaMalloc(&new_cublas_workspace, cublas_workspace_bytes),
                 "allocate cuBLAS user workspace");
      new_cublas_alignment = pointer_alignment(new_cublas_workspace);
      if (new_cublas_alignment < 256U) {
        throw std::runtime_error(
          "cuBLAS user workspace alignment is below 256 bytes");
      }
      check_cublas(cublasSetStream(blas, stream), "rebind cuBLAS stream");
      check_cublas(cublasSetWorkspace(
        blas, new_cublas_workspace, cublas_workspace_bytes
      ), "install cuBLAS workspace");
    }
  } catch (...) {
    cudaFree(new_double_arena);
    cudaFree(new_int_arena);
    cudaFreeHost(new_host_status_arena);
    cudaFree(new_pointer_arena);
    cudaFree(new_cublas_workspace);
    throw;
  }

  bool grew = false;
  if (new_double_arena != nullptr) {
    cudaFree(double_arena);
    double_arena = new_double_arena;
    double_capacity = double_required;
    grew = true;
  }
  if (new_int_arena != nullptr) {
    cudaFree(int_arena);
    int_arena = new_int_arena;
    int_capacity = int_required;
    grew = true;
  }
  if (new_host_status_arena != nullptr) {
    cudaFreeHost(host_status_arena);
    host_status_arena = new_host_status_arena;
    host_status_capacity = int_required;
    grew = true;
  }
  if (new_pointer_arena != nullptr) {
    cudaFree(pointer_arena);
    pointer_arena = new_pointer_arena;
    pointer_capacity = pointer_required;
    grew = true;
  }
  if (new_cublas_workspace != nullptr) {
    cublas_workspace = new_cublas_workspace;
    diagnostics.cublas_workspace_alignment = new_cublas_alignment;
    diagnostics.cublas_user_workspace_installed = true;
    grew = true;
  }

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
  return diagnostics;
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
  if (context != nullptr) context->reset();
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
    case FixedSpRoute::CholeskyBatched: return "CHOLESKY_BATCHED";
    case FixedSpRoute::AugmentedQr: return "AUGMENTED_QR";
    case FixedSpRoute::AugmentedSvd: return "AUGMENTED_SVD";
  }
  return "UNKNOWN";
}

}  // namespace fastkpc
