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

constexpr char kFullCudaCiMethodBatchRequestSchemaVersion[] =
  "full-cuda-ci-method-batch-request-v1";
constexpr char kFullCudaCiMethodBatchResultSchemaVersion[] =
  "full-cuda-ci-method-batch-result-v1";

struct FullCudaCiMethodPairRequest {
  std::uint64_t logical_sequence_id = 0;
  int left_target_index = -1;
  int right_target_index = -1;
};

class PermutationTableBuilder;

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
  SealedPermutationTableHandle seal();
  void reclaim(SealedPermutationTableHandle&& handle);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct FullCudaCiMethodBatchRequest {
  std::string expected_prepared_s_key_sha256;
  std::string request_identity_sha256;
  std::string ci_method;
  std::vector<FullCudaCiMethodPairRequest> pairs;
  double alpha = 0.1;
  double index = 1.0;
  int num_col = 35;
  double hsic_sig = 1.0;
  int permutation_replicates = 0;
  bool permutation_include_observed = true;
  // Pair-major, then replicate-major, then row-major zero-based indices.
  SealedPermutationTableHandle permutation_table;
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
  double request_identity_validation_host_ms = 0.0;
  double total_host_ms = 0.0;
  int permutation_payload_validation_scan_count = 0;
  std::size_t permutation_payload_validation_scan_bytes = 0;
  bool request_identity_authenticated = false;
  bool permutation_attestation_authenticated = false;
  bool prepared_identity_authenticated = false;
  bool target_identity_authenticated = false;
  bool residuals_device_resident = false;
  bool compact_result_only_d2h = false;
  bool caller_device_restored = false;
};

struct FullCudaCiMethodBatchResult {
  std::string schema_version;
  std::string request_identity_sha256;
  std::string prepared_s_key_sha256;
  std::vector<std::string> target_keys;
  std::vector<FullCudaCiMethodCompactRecord> records;
  FullCudaCiMethodBatchDiagnostics diagnostics;
};

std::string full_cuda_ci_method_batch_request_identity(
  const FullCudaCiMethodBatchRequest& request,
  const std::vector<std::string>& target_keys,
  int n);

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
