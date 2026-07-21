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
  std::string cusolver_deterministic_mode_required;
  std::string cublas_math_mode_required;
  std::string cublas_atomics_mode_required;
  bool cublas_user_workspace_required;
  std::size_t cublas_workspace_bytes_required;
  std::size_t cublas_workspace_min_alignment_required;
};

CudaPhase3EnvironmentIdentity fastkpc_cuda_phase3_environment_identity(
  int device_id);

#endif
