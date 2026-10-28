#ifndef FASTKPC_CUDA_DEVICE_PROPERTIES_HPP
#define FASTKPC_CUDA_DEVICE_PROPERTIES_HPP

#include <cuda_runtime_api.h>
#include <cstdint>

// Process-local static metadata only; callers retain their live device/context
// checks. Set FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE=1 before first use to opt in.
// The shared definition lives in cuda_status.cpp, including for capacity TUs.
cudaError_t fastkpc_cuda_get_device_properties(cudaDeviceProp* properties,
                                             int device);

struct CudaDevicePropertiesCacheStats {
  bool enabled;
  std::uint64_t requests;
  std::uint64_t queries;
  std::uint64_t failures;
  std::uint64_t hits;
  std::uint64_t entries;
};

// Host-only diagnostic snapshot; counters are exact after callers have joined.
CudaDevicePropertiesCacheStats fastkpc_cuda_device_properties_cache_stats();

#endif
