#ifndef FASTKPC_CUDA_STATUS_HPP
#define FASTKPC_CUDA_STATUS_HPP

#include <cstddef>
#include <string>

struct CudaDeviceInfo {
  int device_id;
  std::string name;
  int major;
  int minor;
  double total_global_mem;
};

bool fastkpc_cuda_available(std::string* error_message);
CudaDeviceInfo fastkpc_cuda_device_info();

struct CudaPhase3EnvironmentIdentity {
  std::string schema_version;
  std::string runtime_abi_schema_version;
  std::string configuration_schema_version;
  int device_id;
  int cuda_toolkit_version;
  int cuda_driver_version;
  std::string gpu_name;
  std::string gpu_uuid;
  int compute_capability_major;
  int compute_capability_minor;
  int sm_count;
  std::string cusolver_deterministic_mode;
  std::string cublas_math_mode;
  std::string cublas_atomics_mode;
  bool cublas_user_workspace_installed;
  std::size_t cublas_workspace_bytes;
  std::size_t cublas_workspace_alignment;
};

CudaPhase3EnvironmentIdentity fastkpc_cuda_phase3_environment_identity(
  int device_id);

#endif
