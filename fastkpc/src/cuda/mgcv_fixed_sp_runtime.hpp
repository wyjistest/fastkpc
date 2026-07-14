#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP

#include "mgcv_fixed_sp_runtime_types.hpp"

#include <memory>

namespace fastkpc {

class CudaRuntimeContext;
class PreparedSGpuHandle;
class DeviceResidualBatch;

std::shared_ptr<CudaRuntimeContext> create_fixed_sp_runtime(int device_id);
void reserve_fixed_sp_runtime(
  const std::shared_ptr<CudaRuntimeContext>& context,
  const FixedSpCapacities& capacities);
FixedSpRuntimeInfo fixed_sp_runtime_info(
  const std::shared_ptr<CudaRuntimeContext>& context);
void free_fixed_sp_runtime(std::shared_ptr<CudaRuntimeContext>* context);

std::shared_ptr<PreparedSGpuHandle> create_prepared_s_gpu(
  const std::shared_ptr<CudaRuntimeContext>& context,
  const PreparedSHostView& setup);
PreparedSInfo prepared_s_gpu_info(
  const std::shared_ptr<PreparedSGpuHandle>& handle);
void free_prepared_s_gpu(std::shared_ptr<PreparedSGpuHandle>* handle);

}  // namespace fastkpc

#endif
