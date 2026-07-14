#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP

#include "mgcv_fixed_sp_runtime_types.hpp"

#include <cuda_runtime_api.h>

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
PreparedSStaticShadow test_prepared_s_static_shadow(
  const std::shared_ptr<PreparedSGpuHandle>& handle);

std::shared_ptr<DeviceResidualBatch> solve_fixed_sp_batch(
  const std::shared_ptr<PreparedSGpuHandle>& handle,
  const FixedSpBatchHostView& batch);
DeviceResidualInfo device_residual_info(
  const std::shared_ptr<DeviceResidualBatch>& token);
void release_device_residual(
  const std::shared_ptr<DeviceResidualBatch>& token);
void register_device_residual_consumer_event(
  const std::shared_ptr<DeviceResidualBatch>& token,
  cudaEvent_t consumer_completion_event);
void free_device_residual(std::shared_ptr<DeviceResidualBatch>* token);

}  // namespace fastkpc

#endif
