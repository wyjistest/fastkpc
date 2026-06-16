#include "cuda_status.hpp"

#include <cuda_runtime.h>
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
