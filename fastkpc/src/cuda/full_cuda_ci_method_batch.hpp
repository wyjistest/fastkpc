#ifndef FASTKPC_FULL_CUDA_CI_METHOD_BATCH_HPP
#define FASTKPC_FULL_CUDA_CI_METHOD_BATCH_HPP

#include "mgcv_fixed_sp_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastkpc {

constexpr char kFullCudaCiMethodBatchRequestSchemaVersion[] =
  "full-cuda-ci-method-batch-request-v1";
constexpr char kFullCudaCiMethodBatchResultSchemaVersion[] =
  "full-cuda-ci-method-batch-result-v1";

struct FullCudaCiMethodPairRequest {
  std::uint64_t logical_sequence_id = 0;
  int left_target_index = -1;
  int right_target_index = -1;
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
  std::vector<int> permutations;
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
  double residual_solve_host_ms = 0.0;
  double component_build_cuda_ms = 0.0;
  double pair_evaluation_cuda_ms = 0.0;
  double compact_d2h_cuda_ms = 0.0;
  double total_host_ms = 0.0;
  bool request_identity_authenticated = false;
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

FullCudaCiMethodBatchResult run_full_cuda_ci_method_batch(
  const std::shared_ptr<PreparedSGpuHandle>& prepared_s,
  const FixedSpBatchHostView& batch,
  const FullCudaCiMethodBatchRequest& request);

}  // namespace fastkpc

#endif
