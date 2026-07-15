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
FixedSpShadowResult materialize_fixed_sp_shadow(
  const std::shared_ptr<DeviceResidualBatch>& token,
  std::uint32_t output_mask);
void release_device_residual(
  const std::shared_ptr<DeviceResidualBatch>& token);
void register_device_residual_consumer_event(
  const std::shared_ptr<DeviceResidualBatch>& token,
  cudaEvent_t consumer_completion_event);
void free_device_residual(std::shared_ptr<DeviceResidualBatch>* token);
DeviceCoefficientShadow test_device_residual_coefficient_shadow(
  const std::shared_ptr<DeviceResidualBatch>& token);
void test_inject_device_residual_consumer_registration_failure(
  const std::shared_ptr<DeviceResidualBatch>& token);
FixedSpResourceSnapshot test_fixed_sp_cuda_resource_snapshot();
void test_inject_next_fixed_sp_cuda_resource_acquire_failure(
  const std::string& resource);
void test_inject_next_fixed_sp_cuda_resource_teardown_failure(
  const std::string& resource);
void test_inject_next_fixed_sp_cuda_resource_post_call_teardown_failure(
  const std::string& resource);
void test_inject_next_prepared_static_shadow_body_failure();
void test_exercise_fixed_sp_cuda_resource_teardown_failure(
  const std::string& resource);
void test_inject_next_fixed_sp_cuda_device_free_failure();
int test_fixed_sp_cuda_device_count();
void test_fixed_sp_cuda_set_device(int device_id);
int test_fixed_sp_cuda_get_device();
void test_inject_next_blocked_consumer_launch_failure();
void test_force_next_fixed_sp_cuda_potrf_info(
  const std::vector<int>& info);
void test_force_next_fixed_sp_cuda_potrs_info(int info);
void test_register_blocked_device_residual_consumer(
  const std::shared_ptr<DeviceResidualBatch>& token);
void test_complete_blocked_device_residual_consumer(
  const std::shared_ptr<DeviceResidualBatch>& token);

}  // namespace fastkpc

#endif
