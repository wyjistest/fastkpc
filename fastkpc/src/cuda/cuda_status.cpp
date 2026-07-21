#include "cuda_status.hpp"
#include "mgcv_fixed_sp_runtime_types.hpp"

#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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
  err = cudaGetDeviceProperties(&prop, device);
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
  err = cudaGetDeviceProperties(&properties, device_id);
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
