#ifndef FASTKPC_FULL_CUDA_CI_VERTICAL_HPP
#define FASTKPC_FULL_CUDA_CI_VERTICAL_HPP

#include "mgcv_fixed_sp_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

constexpr char kFullCudaCiVerticalRequestSchemaVersion[] =
  "full-cuda-ci-phase35-vertical-request-v1";
constexpr char kFullCudaCiVerticalResultSchemaVersion[] =
  "full-cuda-ci-phase35-vertical-result-v1";
constexpr char kFullCudaCiExactComponentSemanticVersion[] =
  "full-cuda-ci-exact-centered-distance-component-v1";
constexpr char kFullCudaCiExactBatchRequestSchemaVersion[] =
  "full-cuda-ci-phase35-exact-batch-request-v1";
constexpr char kFullCudaCiExactBatchResultSchemaVersion[] =
  "full-cuda-ci-phase35-exact-batch-result-v1";
constexpr char kFullCudaCiLegacyEigBatchRequestSchemaVersion[] =
  "full-cuda-ci-phase35-legacy-eig-batch-request-v1";
constexpr char kFullCudaCiLegacyEigBatchResultSchemaVersion[] =
  "full-cuda-ci-phase35-legacy-eig-batch-result-v1";
constexpr char kFullCudaCiLegacyEigComponentSemanticVersion[] =
  "full-cuda-ci-legacy-raw-distance-full-eig-numcol35-v1";

struct FullCudaCiVerticalRequest {
  std::string expected_prepared_s_key_sha256;
  std::string request_identity_sha256;
  std::uint64_t logical_sequence_id = 0;
  int left_target_index = -1;
  int right_target_index = -1;
  double alpha = 0.1;
  bool exercise_eviction = true;
};

struct FullCudaCiCompactHostRecord {
  std::uint64_t logical_sequence_id = 0;
  double p_value = 0.0;
  std::string status;
  std::string solver_route;
  std::string optimizer_status;
  std::string dcov_status;
  int diagnostic_flags = 0;
};

struct FullCudaCiNumericalDiagnostics {
  double statistic = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  double gamma_shape = 0.0;
  double gamma_scale = 0.0;
  int gamma_iterations = 0;
};

struct FullCudaCiVerticalDiagnostics {
  std::string component_semantic_version;
  int n = 0;
  int target_count = 0;
  int component_build_count = 0;
  int component_cache_eviction_count = 0;
  int pair_evaluation_count = 0;
  int deterministic_replay_count = 0;
  int residual_d2h_count = 0;
  std::size_t residual_d2h_bytes = 0;
  int component_d2h_count = 0;
  std::size_t component_d2h_bytes = 0;
  int compact_result_d2h_count = 0;
  std::size_t compact_result_d2h_bytes = 0;
  int cpu_dcov_component_count = 0;
  int cpu_dcov_pair_statistic_count = 0;
  int cpu_gamma_p_value_count = 0;
  int consumer_event_registration_count = 0;
  int explicit_host_wait_count = 0;
  int device_allocation_count = 0;
  int device_free_count = 0;
  std::size_t device_allocation_bytes = 0;
  std::size_t peak_live_device_bytes = 0;
  std::size_t component_bytes_per_target = 0;
  std::size_t peak_component_bytes = 0;
  std::int64_t live_device_allocations_before = 0;
  std::int64_t live_device_allocations_after = 0;
  std::int64_t live_device_bytes_before = 0;
  std::int64_t live_device_bytes_after = 0;
  double residual_solve_host_ms = 0.0;
  double first_component_build_cuda_ms = 0.0;
  double first_pair_evaluation_cuda_ms = 0.0;
  double first_compact_d2h_cuda_ms = 0.0;
  double replay_component_build_cuda_ms = 0.0;
  double replay_pair_evaluation_cuda_ms = 0.0;
  double replay_compact_d2h_cuda_ms = 0.0;
  double teardown_host_ms = 0.0;
  double total_host_ms = 0.0;
  bool request_identity_authenticated = false;
  bool prepared_identity_authenticated = false;
  bool target_identity_authenticated = false;
  bool residuals_device_resident = false;
  bool components_device_resident = false;
  bool compact_result_only_d2h = false;
  bool eviction_result_bit_identical = false;
  bool deterministic_logical_replay = false;
  bool bounded_allocation = false;
  bool leak_free_teardown = false;
  bool caller_device_restored = false;
};

struct FullCudaCiVerticalResult {
  std::string schema_version;
  std::string request_identity_sha256;
  std::string prepared_s_key_sha256;
  std::vector<std::string> target_keys;
  FullCudaCiCompactHostRecord first_result;
  FullCudaCiCompactHostRecord replay_result;
  FullCudaCiNumericalDiagnostics first_numerical;
  FullCudaCiNumericalDiagnostics replay_numerical;
  FullCudaCiVerticalDiagnostics diagnostics;
};

struct FullCudaCiVerticalResourceSnapshot {
  std::int64_t live_device_allocations = 0;
  std::int64_t live_device_bytes = 0;
  std::int64_t live_streams = 0;
  std::int64_t live_events = 0;
  std::int64_t total_device_allocations = 0;
  std::int64_t total_device_frees = 0;
  std::int64_t total_stream_creates = 0;
  std::int64_t total_stream_destroys = 0;
  std::int64_t total_event_creates = 0;
  std::int64_t total_event_destroys = 0;
};

struct FullCudaCiExactBatchPairRequest {
  std::uint64_t logical_sequence_id = 0;
  int left_target_index = -1;
  int right_target_index = -1;
  double alpha = 0.1;
};

struct FullCudaCiExactBatchRequest {
  std::string expected_prepared_s_key_sha256;
  std::string request_identity_sha256;
  std::vector<FullCudaCiExactBatchPairRequest> pairs;
  int component_capacity = 0;
};

struct FullCudaCiExactBatchDiagnostics {
  std::string component_semantic_version;
  int n = 0;
  int target_count = 0;
  int pair_count = 0;
  int referenced_component_count = 0;
  int component_capacity = 0;
  int component_cache_lookup_count = 0;
  int component_cache_hit_count = 0;
  int component_cache_miss_count = 0;
  int component_build_count = 0;
  int component_cache_eviction_count = 0;
  int pair_evaluation_count = 0;
  int residual_d2h_count = 0;
  std::size_t residual_d2h_bytes = 0;
  int component_d2h_count = 0;
  std::size_t component_d2h_bytes = 0;
  int compact_result_d2h_count = 0;
  std::size_t compact_result_d2h_bytes = 0;
  int metadata_h2d_count = 0;
  std::size_t metadata_h2d_bytes = 0;
  int cpu_dcov_component_count = 0;
  int cpu_dcov_pair_statistic_count = 0;
  int cpu_gamma_p_value_count = 0;
  int consumer_event_registration_count = 0;
  int explicit_host_wait_count = 0;
  int device_allocation_count = 0;
  int device_free_count = 0;
  std::size_t device_allocation_bytes = 0;
  std::size_t peak_live_device_bytes = 0;
  std::size_t component_bytes_per_target = 0;
  std::size_t peak_component_bytes = 0;
  std::int64_t live_device_allocations_before = 0;
  std::int64_t live_device_allocations_after = 0;
  std::int64_t live_device_bytes_before = 0;
  std::int64_t live_device_bytes_after = 0;
  double residual_solve_host_ms = 0.0;
  double metadata_h2d_cuda_ms = 0.0;
  double component_build_cuda_ms = 0.0;
  double pair_evaluation_cuda_ms = 0.0;
  double compact_d2h_cuda_ms = 0.0;
  double dcov_host_boundary_ms = 0.0;
  double teardown_host_ms = 0.0;
  double total_host_ms = 0.0;
  bool request_identity_authenticated = false;
  bool prepared_identity_authenticated = false;
  bool target_identity_authenticated = false;
  bool residuals_device_resident = false;
  bool components_device_resident = false;
  bool compact_result_only_d2h = false;
  bool deterministic_logical_order = false;
  bool component_capacity_respected = false;
  bool bounded_allocation = false;
  bool leak_free_teardown = false;
  bool caller_device_restored = false;
};

struct FullCudaCiExactBatchResult {
  std::string schema_version;
  std::string request_identity_sha256;
  std::string prepared_s_key_sha256;
  std::vector<std::string> target_keys;
  std::vector<FullCudaCiCompactHostRecord> records;
  std::vector<FullCudaCiNumericalDiagnostics> numerical;
  FullCudaCiExactBatchDiagnostics diagnostics;
};

struct FullCudaCiLegacyEigBatchRequest {
  std::string expected_prepared_s_key_sha256;
  std::string request_identity_sha256;
  std::vector<FullCudaCiExactBatchPairRequest> pairs;
  int component_capacity = 0;
  int num_col = 35;
};

struct FullCudaCiLegacyEigBatchDiagnostics {
  std::string component_semantic_version;
  int n = 0;
  int target_count = 0;
  int pair_count = 0;
  int referenced_component_count = 0;
  int component_capacity = 0;
  int num_col = 0;
  int component_build_count = 0;
  int pair_evaluation_count = 0;
  int solver_failure_count = 0;
  int residual_d2h_count = 0;
  std::size_t residual_d2h_bytes = 0;
  int component_d2h_count = 0;
  std::size_t component_d2h_bytes = 0;
  int compact_result_d2h_count = 0;
  std::size_t compact_result_d2h_bytes = 0;
  int compact_status_d2h_count = 0;
  std::size_t compact_status_d2h_bytes = 0;
  int metadata_h2d_count = 0;
  std::size_t metadata_h2d_bytes = 0;
  int cpu_dcov_component_count = 0;
  int cpu_dcov_eigen_count = 0;
  int cpu_dcov_pair_statistic_count = 0;
  int cpu_gamma_p_value_count = 0;
  int cuda_full_eig_count = 0;
  int cuda_pair_count = 0;
  int cuda_gamma_count = 0;
  int consumer_event_registration_count = 0;
  int explicit_host_wait_count = 0;
  int device_allocation_count = 0;
  int device_free_count = 0;
  std::size_t device_allocation_bytes = 0;
  std::size_t peak_live_device_bytes = 0;
  std::size_t persistent_component_bytes = 0;
  std::size_t eig_workspace_bytes = 0;
  std::size_t pair_workspace_bytes = 0;
  double residual_solve_host_ms = 0.0;
  double metadata_h2d_cuda_ms = 0.0;
  double distance_build_cuda_ms = 0.0;
  double full_eig_cuda_ms = 0.0;
  double component_finalize_cuda_ms = 0.0;
  double component_build_cuda_ms = 0.0;
  double pair_evaluation_cuda_ms = 0.0;
  double compact_d2h_cuda_ms = 0.0;
  double dcov_host_boundary_ms = 0.0;
  double teardown_host_ms = 0.0;
  double total_host_ms = 0.0;
  bool request_identity_authenticated = false;
  bool prepared_identity_authenticated = false;
  bool target_identity_authenticated = false;
  bool residuals_device_resident = false;
  bool components_device_resident = false;
  bool compact_result_only_d2h = false;
  bool deterministic_logical_order = false;
  bool component_capacity_respected = false;
  bool bounded_allocation = false;
  bool leak_free_teardown = false;
  bool caller_device_restored = false;
};

struct FullCudaCiLegacyEigBatchResult {
  std::string schema_version;
  std::string request_identity_sha256;
  std::string prepared_s_key_sha256;
  std::vector<std::string> target_keys;
  std::vector<FullCudaCiCompactHostRecord> records;
  std::vector<FullCudaCiNumericalDiagnostics> numerical;
  FullCudaCiLegacyEigBatchDiagnostics diagnostics;
};

std::string full_cuda_ci_vertical_request_identity(
  const FullCudaCiVerticalRequest& request,
  const std::vector<std::string>& target_keys);

FullCudaCiVerticalResult run_full_cuda_ci_phase35_vertical(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiVerticalRequest& request);

FullCudaCiVerticalResourceSnapshot
full_cuda_ci_vertical_resource_snapshot();

std::string full_cuda_ci_exact_batch_request_identity(
  const FullCudaCiExactBatchRequest& request,
  const std::vector<std::string>& target_keys);

FullCudaCiExactBatchResult run_full_cuda_ci_phase35_exact_batch(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiExactBatchRequest& request);

std::string full_cuda_ci_legacy_eig_batch_request_identity(
  const FullCudaCiLegacyEigBatchRequest& request,
  const std::vector<std::string>& target_keys);

FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiLegacyEigBatchRequest& request);

}  // namespace fastkpc

#endif
