#include "cuda_status.hpp"
#include "cuda_device_properties.hpp"
#include "mgcv_fixed_sp_runtime_types.hpp"

#include <cuda_runtime.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace {
bool device_properties_cache_enabled() {
  const char* value = std::getenv("FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
    return false;
  if (std::strcmp(value, "1") == 0) return true;
  throw std::runtime_error(
    "FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE must be 0 or 1");
}

struct DevicePropertiesCache {
  const bool enabled = device_properties_cache_enabled();
  std::mutex mutex;
  std::map<int, cudaDeviceProp> properties;
  std::atomic<std::uint64_t> requests{0}, queries{0}, failures{0}, hits{0}, entries{0};

  cudaError_t query(cudaDeviceProp* output, int device) {
    queries.fetch_add(1, std::memory_order_relaxed);
    const cudaError_t err = cudaGetDeviceProperties(output, device);
    if (err != cudaSuccess) failures.fetch_add(1, std::memory_order_relaxed);
    return err;
  }
};

DevicePropertiesCache& device_properties_cache() {
  static DevicePropertiesCache cache;
  return cache;
}
}  // namespace

cudaError_t fastkpc_cuda_get_device_properties(cudaDeviceProp* properties,
                                             int device) {
  auto& cache = device_properties_cache();
  cache.requests.fetch_add(1, std::memory_order_relaxed);
  if (!cache.enabled || properties == nullptr || device < 0)
    return cache.query(properties, device);

  // Serialize first access as well as publication: concurrent capacity backends
  // issue at most one successful query per logical device. Failures are retried.
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found = cache.properties.find(device);
  if (found != cache.properties.end()) {
    *properties = found->second;
    cache.hits.fetch_add(1, std::memory_order_relaxed);
    return cudaSuccess;
  }
  const cudaError_t err = cache.query(properties, device);
  if (err == cudaSuccess) {
    cache.properties.emplace(device, *properties);
    cache.entries.fetch_add(1, std::memory_order_relaxed);
  }
  return err;
}

CudaDevicePropertiesCacheStats fastkpc_cuda_device_properties_cache_stats() {
  auto& cache = device_properties_cache();
  return {cache.enabled, cache.requests.load(std::memory_order_relaxed),
          cache.queries.load(std::memory_order_relaxed),
          cache.failures.load(std::memory_order_relaxed),
          cache.hits.load(std::memory_order_relaxed),
          cache.entries.load(std::memory_order_relaxed)};
}

bool fastkpc_cuda_available(std::string* error_message) {
  cudaError_t err = cudaFree(0);
  if (err != cudaSuccess) {
    if (error_message != nullptr) *error_message = cudaGetErrorString(err);
    return false;
  }
  int count = 0;
  err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    if (error_message != nullptr) *error_message = cudaGetErrorString(err);
    return false;
  }
  if (count <= 0) {
    if (error_message != nullptr) *error_message = "no CUDA devices found";
    return false;
  }
  return true;
}

CudaDeviceInfo fastkpc_cuda_device_info() {
  std::string error;
  if (!fastkpc_cuda_available(&error)) {
    throw std::runtime_error("CUDA unavailable: " + error);
  }
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDevice failed: ") +
                             cudaGetErrorString(err));
  }
  cudaDeviceProp prop;
  err = fastkpc_cuda_get_device_properties(&prop, device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") +
                             cudaGetErrorString(err));
  }
  CudaDeviceInfo info;
  info.device_id = device;
  info.name = prop.name;
  info.major = prop.major;
  info.minor = prop.minor;
  info.total_global_mem = static_cast<double>(prop.totalGlobalMem);
  return info;
}

CudaPhase3EnvironmentIdentity fastkpc_cuda_phase3_environment_identity(
    int device_id) {
  if (device_id < 0) {
    throw std::runtime_error("device_id must be non-negative");
  }
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA device count query failed: ") +
                             cudaGetErrorString(err));
  }
  if (device_id >= device_count) {
    throw std::runtime_error("requested CUDA device_id is unavailable");
  }

  cudaDeviceProp properties;
  err = fastkpc_cuda_get_device_properties(&properties, device_id);
  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA static device properties query failed: ") +
      cudaGetErrorString(err));
  }
  int toolkit_version = 0;
  err = cudaRuntimeGetVersion(&toolkit_version);
  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA toolkit version query failed: ") +
      cudaGetErrorString(err));
  }
  int driver_version = 0;
  err = cudaDriverGetVersion(&driver_version);
  if (err != cudaSuccess) {
    throw std::runtime_error(
      std::string("CUDA driver version query failed: ") +
      cudaGetErrorString(err));
  }

  std::ostringstream uuid;
  uuid << "GPU-" << std::hex << std::setfill('0');
  for (int index = 0; index < 16; ++index) {
    uuid << std::setw(2) << static_cast<unsigned int>(
      static_cast<unsigned char>(properties.uuid.bytes[index]));
  }

  CudaPhase3EnvironmentIdentity identity;
  identity.schema_version = "full-cuda-ci-phase3-environment-policy-v1";
  identity.runtime_abi_schema_version =
    fastkpc::kFixedSpRuntimeAbiSchemaVersion;
  identity.configuration_schema_version =
    fastkpc::kFixedSpEnvironmentConfigSchemaVersion;
  identity.device_id = device_id;
  identity.cuda_toolkit_version = toolkit_version;
  identity.cuda_driver_version = driver_version;
  identity.gpu_name = properties.name;
  identity.gpu_uuid = uuid.str();
  identity.compute_capability_major = properties.major;
  identity.compute_capability_minor = properties.minor;
  identity.sm_count = properties.multiProcessorCount;
  identity.cusolver_deterministic_mode_required =
    fastkpc::kFixedSpCusolverDeterministicMode;
  identity.cublas_math_mode_required = fastkpc::kFixedSpCublasMathMode;
  identity.cublas_atomics_mode_required = fastkpc::kFixedSpCublasAtomicsMode;
  identity.cublas_user_workspace_required =
    fastkpc::kFixedSpCublasUserWorkspaceRequired;
  identity.cublas_workspace_bytes_required =
    fastkpc::kFixedSpCublasWorkspaceBytes;
  identity.cublas_workspace_min_alignment_required =
    fastkpc::kFixedSpCublasWorkspaceMinAlignment;
  return identity;
}
