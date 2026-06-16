#ifndef FASTKPC_CUDA_STATUS_HPP
#define FASTKPC_CUDA_STATUS_HPP

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

#endif
