#ifndef FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP
#define FASTKPC_MGCV_FIXED_SP_RUNTIME_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

constexpr char kFixedSpRuntimeAbiSchemaVersion[] =
  "full-cuda-ci-fixed-sp-runtime-v1";
constexpr char kFixedSpEnvironmentConfigSchemaVersion[] =
  "full-cuda-ci-fixed-sp-environment-config-v1";
constexpr char kFixedSpCusolverDeterministicMode[] = "enabled";
constexpr char kFixedSpCublasMathMode[] = "pedantic";
constexpr char kFixedSpCublasAtomicsMode[] = "not_allowed";
constexpr std::size_t kFixedSpCublasWorkspaceBytes = 16U * 1024U * 1024U;
constexpr std::size_t kFixedSpCublasWorkspaceAlignment = 256U;

enum class FixedSpRoute : int {
  Unset = -1,
  CholeskyBatched = 0,
  AugmentedQr = 1,
  AugmentedSvd = 2
};

enum FixedSpOutputMask : std::uint32_t {
  FixedSpOutputCoefficients = 1U << 0,
  FixedSpOutputFitted = 1U << 1,
  FixedSpOutputResiduals = 1U << 2,
  FixedSpOutputRss = 1U << 3,
  FixedSpOutputRhs = 1U << 4
};

constexpr std::uint32_t kFixedSpPublicOutputMask =
  FixedSpOutputCoefficients | FixedSpOutputFitted |
  FixedSpOutputResiduals | FixedSpOutputRss | FixedSpOutputRhs;

enum class FixedSpStatus : int {
  OkCholeskyBatched = 0,
  OkCholeskySingle = 1,
  OkAugmentedQr = 2,
  OkAugmentedSvd = 3,
  ErrNonfiniteInput = 10,
  ErrSpShapeOrOrder = 11,
  ErrRouteMetadata = 12,
  ErrStablePathNotImplemented = 13,
  ErrQrFailed = 14,
  ErrSvdFailed = 15,
  ErrNonfiniteOutput = 16,
  ErrInternalCuda = 17
};

struct FixedSpCapacities {
  int n = 0;
  int null_dim = 0;
  int target_count = 0;
  int penalty_count = 0;
  int augmented_rows = 0;
};

struct FixedSpRuntimeInfo {
  int device_id = -1;
  std::string gpu_name;
  std::int64_t creator_pid = -1;
  std::uint64_t generation = 0;
  int runtime_context_create_count = 0;
  int cuda_device_allocation_count = 0;
  int cuda_host_allocation_count = 0;
  int stream_create_count = 0;
  int event_create_count = 0;
  int cublas_handle_create_count = 0;
  int cusolver_handle_create_count = 0;
  int gesvdj_info_create_count = 0;
  int gesvdj_info_destroy_count = 0;
  int workspace_reserve_count = 0;
  int workspace_grow_count = 0;
  int stable_workspace_grow_count = 0;
  int cuda_device_synchronize_count = 0;
  int cholesky_factor_checkpoint_record_count = 0;
  int cholesky_factor_checkpoint_wait_count = 0;
  int cholesky_solve_checkpoint_record_count = 0;
  int cholesky_solve_checkpoint_wait_count = 0;
  int qr_checkpoint_record_count = 0;
  int qr_checkpoint_wait_count = 0;
  int svd_checkpoint_record_count = 0;
  int svd_checkpoint_wait_count = 0;
  std::size_t workspace_bytes = 0;
  std::size_t cublas_workspace_bytes = 0;
  std::size_t cublas_workspace_alignment = 0;
  std::size_t eigen_workspace_bytes = 0;
  std::size_t qr_workspace_bytes = 0;
  std::size_t svd_workspace_bytes = 0;
  std::size_t augmented_workspace_bytes = 0;
  std::size_t aggregate_factor_workspace_bytes = 0;
  int cuda_toolkit_version = 0;
  int cuda_driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int sm_count = 0;
  bool cusolver_deterministic_mode_enabled = false;
  bool cublas_pedantic_math_enabled = false;
  bool cublas_atomics_not_allowed = false;
  bool cublas_user_workspace_installed = false;
  bool freed = false;
};

struct FixedSpResourceLifecycleSnapshot {
  std::int64_t acquire_attempt_count = 0;
  std::int64_t acquire_success_count = 0;
  std::int64_t acquire_failure_count = 0;
  std::int64_t teardown_attempt_count = 0;
  std::int64_t teardown_success_count = 0;
  std::int64_t teardown_failure_count = 0;
  std::int64_t active_count = 0;
  std::int64_t ownership_indeterminate_count = 0;
};

struct FixedSpResourceSnapshot {
  FixedSpResourceLifecycleSnapshot cuda_device;
  FixedSpResourceLifecycleSnapshot cuda_host;
  FixedSpResourceLifecycleSnapshot stream;
  FixedSpResourceLifecycleSnapshot event;
  FixedSpResourceLifecycleSnapshot cublas_handle;
  FixedSpResourceLifecycleSnapshot cusolver_handle;
  FixedSpResourceLifecycleSnapshot gesvdj_info;
  std::int64_t cleanup_error_count = 0;
};

struct PreparedSHostView {
  std::string dataset_sha256;
  std::string prepared_s_key_sha256;
  std::string same_s_group_id;
  std::string semantic_fingerprint;
  std::string representation_fingerprint;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  const double* X = nullptr;
  const double* Z = nullptr;
  const double* gram = nullptr;
  const double* H = nullptr;
  std::vector<const double*> penalty_blocks;
  std::vector<int> penalty_dimensions;
  std::vector<int> penalty_offsets_zero_based;
  std::vector<int> penalty_ranks;
  std::vector<int> penalty_sp_indices_zero_based;
};

struct PreparedSInfo {
  std::string prepared_s_key_sha256;
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int penalty_count = 0;
  int setup_h2d_upload_count = 0;
  std::size_t setup_h2d_bytes = 0;
  int penalty_root_build_count = 0;
  int penalty_root_rank_mismatch_count = 0;
  std::size_t penalty_root_bytes = 0;
  double penalty_root_build_ms = 0.0;
  int penalty_root_matrix_count = 0;
  int penalty_root_row_count = 0;
  int H_root_matrix_count = 0;
  int H_root_rank = 0;
  int setup_shadow_d2h_count = 0;
  std::size_t setup_shadow_d2h_bytes = 0;
  int augmented_test_shadow_d2h_count = 0;
  std::size_t augmented_test_shadow_d2h_bytes = 0;
  int projected_H_test_shadow_d2h_count = 0;
  std::size_t projected_H_test_shadow_d2h_bytes = 0;
  std::size_t coefficient_output_capacity = 0;
  std::uint64_t generation = 0;
  bool output_slot_leased = false;
  std::string output_slot_state = "free";
  std::string output_slot_poison_reason;
};

struct FixedSpBatchHostView {
  const double* Y = nullptr;
  const double* SP = nullptr;
  int n = 0;
  int null_dim = 0;
  int penalty_count = 0;
  int target_count = 0;
  std::uint32_t output_mask = 0;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<std::string> target_keys;
};

struct DeviceResidualInfo {
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int target_count = 0;
  std::vector<std::string> target_keys;
  std::vector<FixedSpRoute> planned_routes;
  std::vector<FixedSpRoute> executed_routes;
  std::vector<std::string> reroute_reasons;
  std::vector<FixedSpStatus> solver_statuses;
  std::vector<int> qr_rank;
  std::vector<int> geqrf_info;
  std::vector<int> ormqr_info;
  std::vector<int> effective_rank;
  std::vector<double> sigma_max;
  std::vector<double> smallest_retained_sigma;
  std::vector<int> svd_info;
  std::vector<int> aggregate_penalty_root_rank;
  std::vector<int> aggregate_factor_call_count;
  std::vector<int> aggregate_b_build_count;
  std::vector<int> aggregate_penalty_root_pivot;
  std::vector<double> aggregate_dstop;
  std::vector<bool> target_true_batched;
  bool native_batch_call = false;
  int batch_call_count = 0;
  bool true_batched_kernel = false;
  int true_batched_subgroup_count = 0;
  int true_batched_attempted_target_count = 0;
  int true_batched_target_count = 0;
  int cholesky_single_target_count = 0;
  int potrf_batched_call_count = 0;
  int potrs_batched_call_count = 0;
  int target_batch_h2d_call_count = 0;
  int target_h2d_copy_count = 0;
  std::size_t target_h2d_bytes = 0;
  int coefficient_batch_finalize_call_count = 0;
  int fitted_batch_finalize_call_count = 0;
  int residual_rss_batch_finalize_call_count = 0;
  int per_target_output_finalize_call_count = 0;
  int batch_output_finalized_target_count = 0;
  bool canonical_output_order_exact = false;
  int stable_reroute_count = 0;
  int planned_cholesky_target_count = 0;
  int planned_qr_target_count = 0;
  int planned_svd_target_count = 0;
  int executed_cholesky_target_count = 0;
  int executed_qr_target_count = 0;
  int executed_svd_target_count = 0;
  int cholesky_to_svd_count = 0;
  int qr_to_svd_count = 0;
  int aggregate_penalty_factor_count = 0;
  int aggregate_svd_b_build_count = 0;
  int aggregate_penalty_root_d2h_count = 0;
  std::size_t aggregate_penalty_root_d2h_bytes = 0;
  int output_slot_acquire_count = 0;
  int output_slot_release_count = 0;
  int output_slot_busy_count = 0;
  int stale_token_reject_count = 0;
  int invalid_output_init_count = 0;
  int nonfinite_output_count = 0;
  int cpu_fallback_count = 0;
  int unknown_fallback_count = 0;
  bool resource_snapshot_captured = false;
  int resource_instrumentation_version = 0;
  int resource_allocation_count_before_solve = -1;
  int resource_allocation_count_after_solve = -1;
  int resource_handle_create_count_before_solve = -1;
  int resource_handle_create_count_after_solve = -1;
  int cuda_device_allocation_count_during_solve = -1;
  int cuda_host_allocation_count_during_solve = -1;
  int stream_create_count_during_solve = -1;
  int event_create_count_during_solve = -1;
  int cublas_handle_create_count_during_solve = -1;
  int cusolver_handle_create_count_during_solve = -1;
  int per_target_allocation_count_after_warmup = -1;
  int per_target_handle_create_count = -1;
  int implicit_residual_d2h_count = 0;
  int rhs_device_build_count = 0;
  std::string rhs_authority = "cuda-x0-transpose-y";
  bool full_cuda_data_plane = true;
  int shadow_materialize_call_count = 0;
  int shadow_materialize_target_count = 0;
  std::size_t shadow_d2h_bytes = 0;
  std::uint64_t owner_generation = 0;
  std::uint64_t slot_generation = 0;
};

struct PreparedSStaticShadow {
  int n = 0;
  int null_dim = 0;
  int penalty_count = 0;
  std::vector<double> X_null;
  std::vector<double> gram;
  std::vector<double> projected_penalties;
  bool has_H = false;
  std::vector<double> projected_H;
};

struct PreparedSRootsShadow {
  int null_dim = 0;
  int total_penalty_root_rows = 0;
  std::vector<int> penalty_root_offsets;
  std::vector<int> penalty_root_ranks;
  std::vector<double> penalty_roots;
  bool has_H = false;
  int H_root_rank = 0;
  std::vector<double> H_root;
};

struct FixedSpAugmentedSystemShadow {
  int leading_dimension = 0;
  int rows = 0;
  int cols = 0;
  int target_index = -1;
  std::vector<double> B;
  std::vector<double> c;
};

struct DeviceCoefficientShadow {
  int coefficient_dim = 0;
  int target_count = 0;
  std::vector<double> coefficients;
};

struct FixedSpShadowResult {
  int n = 0;
  int coefficient_dim = 0;
  int null_dim = 0;
  int target_count = 0;
  std::uint32_t output_mask = 0;
  std::vector<std::uint8_t> successful_targets;
  std::vector<double> coefficients;
  std::vector<double> fitted;
  std::vector<double> residuals;
  std::vector<double> rss;
  std::vector<double> cuda_nullspace_rhs;
};

const char* fixed_sp_status_name(FixedSpStatus status);
const char* fixed_sp_route_name(FixedSpRoute route);

class PreparedSGpuHandle;
PreparedSRootsShadow test_prepared_s_roots_shadow(
  const std::shared_ptr<PreparedSGpuHandle>& handle);

}  // namespace fastkpc

#endif
