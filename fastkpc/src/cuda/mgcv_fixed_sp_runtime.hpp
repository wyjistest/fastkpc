#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_HPP

#include "mgcv_fixed_sp_runtime_types.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

class CudaRuntimeContext;
class PreparedSGpuHandle;
class DeviceResidualBatch;

struct TransientFixedSpCompatibilityHostView {
  int n = 0;
  int null_dim = 0;
  const double* X_null = nullptr;
  const double* gram = nullptr;
  const double* projected_penalty = nullptr;
};

// Internal CUDA consumers use this ephemeral view while retaining the token.
// It is deliberately absent from the public semantic ABI and is never
// serialized or returned to R.
struct DeviceResidualConsumerView {
  const double* residuals = nullptr;
  int n = 0;
  int target_count = 0;
  int device_id = -1;
  cudaStream_t producer_stream = nullptr;
  cudaEvent_t producer_completion_event = nullptr;
  std::vector<std::string> target_keys;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<FixedSpStatus> solver_statuses;
  bool metadata_provisional = false;
};

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
std::shared_ptr<PreparedSGpuHandle>
create_transient_fixed_sp_compatibility_prepared_s_gpu(
  const std::shared_ptr<CudaRuntimeContext>& context,
  const TransientFixedSpCompatibilityHostView& view);
PreparedSInfo prepared_s_gpu_info(
  const std::shared_ptr<PreparedSGpuHandle>& handle);
void free_prepared_s_gpu(std::shared_ptr<PreparedSGpuHandle>* handle);
PreparedSStaticShadow test_prepared_s_static_shadow(
  const std::shared_ptr<PreparedSGpuHandle>& handle);
FixedSpAugmentedSystemShadow test_build_fixed_sp_augmented_shadow(
  const std::shared_ptr<PreparedSGpuHandle>& handle,
  const double* Y,
  std::size_t Y_count,
  const double* SP,
  std::size_t SP_count,
  int target_index);

std::shared_ptr<DeviceResidualBatch> solve_fixed_sp_batch(
  const std::shared_ptr<PreparedSGpuHandle>& handle,
  const FixedSpBatchHostView& batch);
std::shared_ptr<DeviceResidualBatch> submit_fixed_sp_batch_deferred_svd(
  const std::shared_ptr<PreparedSGpuHandle>& handle,
  const FixedSpBatchHostView& batch);
DeviceResidualInfo device_residual_info(
  const std::shared_ptr<DeviceResidualBatch>& token);
DeviceResidualConsumerView acquire_device_residual_consumer_view(
  const std::shared_ptr<DeviceResidualBatch>& token);
DeviceResidualConsumerView acquire_device_residual_submission_view(
  const std::shared_ptr<DeviceResidualBatch>& token);
void complete_device_residual_after_stream_wait(
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
