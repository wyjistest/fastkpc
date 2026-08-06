#ifndef FASTKPC_FULL_CUDA_CI_METHOD_BATCH_HPP
#define FASTKPC_FULL_CUDA_CI_METHOD_BATCH_HPP

#include "mgcv_fixed_sp_runtime.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

class FullCudaCiMethodResidualCache;
class FullCudaCiMethodExecutionContext;
struct FullCudaCiMethodBatchResult;

constexpr char kFullCudaCiMethodStaticRequestIdentitySchemaVersion[] =
  "full-cuda-ci-method-static-request-identity-v1";
constexpr char kFullCudaCiMethodPermutationAttestationSchemaVersion[] =
  "full-cuda-ci-method-permutation-attestation-v2";
constexpr char kFullCudaCiMethodCombinedRequestIdentitySchemaVersion[] =
  "full-cuda-ci-method-combined-request-identity-v2";
constexpr char kFullCudaCiMethodBatchResultSchemaVersion[] =
  "full-cuda-ci-method-batch-result-v3";

struct FullCudaCiMethodPairRequest {
  std::uint64_t logical_sequence_id = 0;
  int left_target_index = -1;
  int right_target_index = -1;
};

struct StaticRequestIdentity {
  std::string schema_version;
  std::string sha256;
};

struct RngReceipt {
  std::string initial_state_sha256;
  std::string final_state_sha256;
  std::uint64_t uniform_draw_count = 0U;
  std::uint64_t rejection_count = 0U;
  bool count_exact = true;
};

struct PermutationAttestation {
  std::string schema_version;
  std::array<unsigned char, 32> payload_sha256{};
  std::size_t value_count = 0U;
  std::size_t byte_count = 0U;
  int replicates = 0;
  RngReceipt rng;
  std::string sha256;
};

struct CombinedRequestIdentity {
  std::string schema_version;
  std::string sha256;
};

struct FullCudaCiMethodStaticRequest {
  std::string expected_prepared_s_key_sha256;
  StaticRequestIdentity identity;
  std::string ci_method;
  std::vector<FullCudaCiMethodPairRequest> pairs;
  double alpha = 0.1;
  double index = 1.0;
  int num_col = 35;
  double hsic_sig = 1.0;
  int permutation_replicates = 0;
  bool permutation_include_observed = true;
};

class PermutationTableBuilder;
class SealedPermutationArtifact;

class SealedPermutationTableHandle {
 public:
  SealedPermutationTableHandle() = default;
  ~SealedPermutationTableHandle() = default;
  SealedPermutationTableHandle(SealedPermutationTableHandle&&) noexcept;
  SealedPermutationTableHandle& operator=(
    SealedPermutationTableHandle&&) noexcept;
  SealedPermutationTableHandle(const SealedPermutationTableHandle&) = delete;
  SealedPermutationTableHandle& operator=(
    const SealedPermutationTableHandle&) = delete;

  const int* data() const noexcept;
  std::size_t size() const noexcept;
  std::size_t byte_size() const noexcept;
  const std::array<unsigned char, 32>& sha256() const noexcept;
  bool sealed() const noexcept;

 private:
  friend class PermutationTableBuilder;

  std::vector<int> values_;
  std::array<unsigned char, 32> sha256_{};
  bool sealed_ = false;
};

class SealedPermutationArtifact {
 public:
  SealedPermutationArtifact() = default;
  ~SealedPermutationArtifact() = default;
  SealedPermutationArtifact(SealedPermutationArtifact&&) noexcept;
  SealedPermutationArtifact& operator=(
    SealedPermutationArtifact&&) noexcept;
  SealedPermutationArtifact(const SealedPermutationArtifact&) = delete;
  SealedPermutationArtifact& operator=(
    const SealedPermutationArtifact&) = delete;

  const SealedPermutationTableHandle& table() const noexcept;
  const PermutationAttestation& attestation() const noexcept;
  bool valid() const noexcept;

 private:
  friend class PermutationTableBuilder;
  friend SealedPermutationArtifact
  full_cuda_ci_method_empty_permutation_artifact();

  SealedPermutationTableHandle table_;
  PermutationAttestation attestation_;
  bool valid_ = false;
};

class PermutationTableBuilder {
 public:
  PermutationTableBuilder();
  ~PermutationTableBuilder();
  PermutationTableBuilder(PermutationTableBuilder&&) noexcept;
  PermutationTableBuilder& operator=(PermutationTableBuilder&&) noexcept;
  PermutationTableBuilder(const PermutationTableBuilder&) = delete;
  PermutationTableBuilder& operator=(const PermutationTableBuilder&) = delete;

  void reset(std::size_t size);
  std::size_t capacity() const noexcept;
  std::size_t size() const noexcept;
  void append_row(const int* values, std::size_t size);
  int* begin_row(std::size_t size);
  void commit_row(const int* values, std::size_t size);
  SealedPermutationArtifact seal(
    const RngReceipt& rng_receipt,
    int replicates);
  void reclaim(SealedPermutationArtifact&& artifact);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class MethodPreparationTicket {
 public:
  MethodPreparationTicket() = default;
  ~MethodPreparationTicket();
  MethodPreparationTicket(MethodPreparationTicket&&) noexcept;
  MethodPreparationTicket& operator=(MethodPreparationTicket&&) noexcept;
  MethodPreparationTicket(const MethodPreparationTicket&) = delete;
  MethodPreparationTicket& operator=(const MethodPreparationTicket&) = delete;

  bool valid() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  friend MethodPreparationTicket submit_method_preparation(
    const std::shared_ptr<PreparedSGpuHandle>&,
    const FixedSpBatchHostView&,
    const FullCudaCiMethodStaticRequest&,
    const std::shared_ptr<FullCudaCiMethodResidualCache>&,
    const std::shared_ptr<FullCudaCiMethodExecutionContext>&);
  friend FullCudaCiMethodBatchResult finalize_method_from_permutation(
    MethodPreparationTicket&&,
    const PermutationAttestation&,
    const CombinedRequestIdentity&,
    const SealedPermutationArtifact&);
};

struct FullCudaCiMethodBatchRequest {
  std::string expected_prepared_s_key_sha256;
  StaticRequestIdentity static_identity;
  PermutationAttestation permutation_attestation;
  CombinedRequestIdentity combined_identity;
  std::string ci_method;
  std::vector<FullCudaCiMethodPairRequest> pairs;
  double alpha = 0.1;
  double index = 1.0;
  int num_col = 35;
  double hsic_sig = 1.0;
  int permutation_replicates = 0;
  bool permutation_include_observed = true;
  // Pair-major, then replicate-major, then row-major zero-based indices.
  SealedPermutationArtifact permutation_artifact;
};

struct FullCudaCiMethodCompactRecord {
  std::uint64_t logical_sequence_id = 0;
  double p_value = 0.0;
  double statistic = 0.0;
  double mean = 0.0;
  double variance = 0.0;
  int status = 0;
  std::string solver_route;
};

struct FullCudaCiMethodBatchDiagnostics {
  std::string ci_method;
  std::string component_semantic_version;
  int n = 0;
  int target_count = 0;
  int referenced_component_count = 0;
  int pair_count = 0;
  int permutation_replicates = 0;
  int component_build_count = 0;
  int pair_evaluation_count = 0;
  int residual_d2h_count = 0;
  std::size_t residual_d2h_bytes = 0;
  int component_d2h_count = 0;
  std::size_t component_d2h_bytes = 0;
  int compact_result_d2h_count = 0;
  std::size_t compact_result_d2h_bytes = 0;
  int metadata_h2d_count = 0;
  std::size_t metadata_h2d_bytes = 0;
  std::size_t device_allocation_bytes = 0;
  std::size_t peak_device_allocation_bytes = 0;
  int residual_cache_lookup_count = 0;
  int residual_cache_hit_count = 0;
  int residual_cache_insert_count = 0;
  int residual_cache_eviction_count = 0;
  int residual_cache_all_hit_batch_count = 0;
  int residual_cache_bypassed_target_count = 0;
  std::size_t residual_cache_capacity_entries = 0;
  std::size_t residual_cache_device_bytes = 0;
  std::size_t residual_cache_gather_d2d_bytes = 0;
  int execution_context_call_count = 0;
  int execution_context_reuse_count = 0;
  int execution_context_buffer_growth_count = 0;
  std::size_t execution_context_device_bytes = 0;
  int preparation_submit_count = 0;
  int finalization_count = 0;
  bool preparation_ticket_consumed = false;
  bool deferred_svd_submission = false;
  bool preparation_submit_nonblocking = false;
  int submit_hidden_stream_sync_count = 0;
  int submit_hidden_device_sync_count = 0;
  int submit_completion_event_wait_count = 0;
  int in_flight_peak = 0;
  int intermediate_host_event_wait_count = 0;
  int final_result_host_event_wait_count = 0;
  int component_cache_persistent_request_count = 0;
  int component_cache_persistent_lookup_count = 0;
  int component_cache_persistent_hit_count = 0;
  int component_cache_persistent_miss_count = 0;
  int component_cache_persistent_insert_count = 0;
  int component_cache_persistent_eviction_count = 0;
  std::size_t component_cache_persistent_capacity_entries = 0;
  std::size_t component_cache_persistent_device_bytes = 0;
  std::size_t component_cache_persistent_gather_d2d_bytes = 0;
  std::size_t component_cache_persistent_store_d2d_bytes = 0;
  double residual_solve_host_ms = 0.0;
  double component_build_cuda_ms = 0.0;
  double pair_evaluation_cuda_ms = 0.0;
  double compact_d2h_cuda_ms = 0.0;
  double permutation_h2d_submit_host_ms = 0.0;
  double static_identity_validation_host_ms = 0.0;
  double permutation_attestation_validation_host_ms = 0.0;
  double combined_identity_validation_host_ms = 0.0;
  double request_identity_validation_host_ms = 0.0;
  double total_host_ms = 0.0;
  int component_host_wait_count = 0;
  int pair_host_wait_count = 0;
  int compact_host_wait_count = 0;
  int consumer_host_wait_count = 0;
  int permutation_payload_validation_scan_count = 0;
  std::size_t permutation_payload_validation_scan_bytes = 0;
  bool static_identity_authenticated = false;
  bool request_identity_authenticated = false;
  bool permutation_attestation_authenticated = false;
  bool combined_identity_authenticated = false;
  bool prepared_identity_authenticated = false;
  bool target_identity_authenticated = false;
  bool residuals_device_resident = false;
  bool compact_result_only_d2h = false;
  bool caller_device_restored = false;
};

struct FullCudaCiMethodBatchResult {
  std::string schema_version;
  std::string static_request_identity_sha256;
  std::string permutation_attestation_sha256;
  std::string combined_request_identity_sha256;
  // Compatibility alias for the combined request identity.
  std::string request_identity_sha256;
  std::string prepared_s_key_sha256;
  std::vector<std::string> target_keys;
  std::vector<FullCudaCiMethodCompactRecord> records;
  FullCudaCiMethodBatchDiagnostics diagnostics;
};

struct StrictMethodFailureInjectionSnapshot {
  std::string armed_stage;
  std::string observed_stage;
  int checkpoint_count = 0;
  bool triggered = false;
};

void test_arm_strict_method_failure_injection(const std::string& stage);
StrictMethodFailureInjectionSnapshot
test_strict_method_failure_injection_snapshot();
void strict_method_failure_checkpoint(const char* stage);
void test_arm_strict_method_identity_tamper(const std::string& layer);
std::string test_take_strict_method_identity_tamper();

StaticRequestIdentity full_cuda_ci_method_static_request_identity(
  const FullCudaCiMethodBatchRequest& request,
  const std::vector<std::string>& target_keys,
  const std::vector<FixedSpRoute>& planned_routes,
  int n);
StaticRequestIdentity full_cuda_ci_method_static_request_identity(
  const FullCudaCiMethodStaticRequest& request,
  const std::vector<std::string>& target_keys,
  const std::vector<FixedSpRoute>& planned_routes,
  int n);
PermutationAttestation full_cuda_ci_method_permutation_attestation(
  const FullCudaCiMethodBatchRequest& request);
PermutationAttestation full_cuda_ci_method_permutation_attestation(
  const SealedPermutationTableHandle& permutation_table,
  int replicates,
  const RngReceipt& rng_receipt);
SealedPermutationArtifact full_cuda_ci_method_empty_permutation_artifact();
CombinedRequestIdentity full_cuda_ci_method_combined_request_identity(
  const StaticRequestIdentity& static_identity,
  const PermutationAttestation& permutation_attestation);
FullCudaCiMethodStaticRequest full_cuda_ci_method_static_request(
  const FullCudaCiMethodBatchRequest& request);

std::vector<int> full_cuda_ci_method_seeded_permutation_table(
  const std::string& ci_method,
  int n,
  int replicates,
  const std::vector<unsigned int>& seeds);

std::shared_ptr<FullCudaCiMethodResidualCache>
create_full_cuda_ci_method_residual_cache(
  int n,
  std::size_t byte_budget);

std::shared_ptr<FullCudaCiMethodExecutionContext>
create_full_cuda_ci_method_execution_context(
  int n,
  const std::string& ci_method,
  int num_col,
  int permutation_replicates);

MethodPreparationTicket submit_method_preparation(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiMethodStaticRequest& request,
  const std::shared_ptr<FullCudaCiMethodResidualCache>& residual_cache =
    std::shared_ptr<FullCudaCiMethodResidualCache>(),
  const std::shared_ptr<FullCudaCiMethodExecutionContext>& execution_context =
    std::shared_ptr<FullCudaCiMethodExecutionContext>());
FullCudaCiMethodBatchResult finalize_method_from_permutation(
  MethodPreparationTicket&& preparation,
  const PermutationAttestation& permutation_attestation,
  const CombinedRequestIdentity& combined_identity,
  const SealedPermutationArtifact& permutation_artifact);

FullCudaCiMethodBatchResult run_full_cuda_ci_method_batch(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiMethodBatchRequest& request,
  const std::shared_ptr<FullCudaCiMethodResidualCache>& residual_cache =
    std::shared_ptr<FullCudaCiMethodResidualCache>(),
  const std::shared_ptr<FullCudaCiMethodExecutionContext>& execution_context =
    std::shared_ptr<FullCudaCiMethodExecutionContext>());

}  // namespace fastkpc

#endif
