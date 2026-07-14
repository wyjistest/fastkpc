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

}  // namespace fastkpc

#endif
