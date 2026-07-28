#ifndef FASTKPC_FULL_CUDA_CI_SEMANTIC_ABI_HPP
#define FASTKPC_FULL_CUDA_CI_SEMANTIC_ABI_HPP

#include <string>
#include <utility>
#include <vector>

namespace fastkpc {

class PreparedSGpuHandle;
class TargetOptimizerStateHandle;
class DeviceResidualHandle;
class DcovComponentHandle;
class LogicalCiBatchHandle;
class CompactCiResult;

struct FullCudaCiSemanticAbiInfo {
  std::string schema_version;
  int abi_major = 0;
  int abi_minor = 0;
  std::vector<std::string> capabilities;
  std::vector<std::pair<std::string, std::string>> capability_status;
  std::string backend_semantic_version;
  std::string producer_contract_hash;
  std::vector<std::pair<std::string, bool>> device_residency_flags;
  std::vector<std::string> semantic_objects;
  std::vector<std::string> compact_result_fields;
};

FullCudaCiSemanticAbiInfo full_cuda_ci_semantic_abi_info();

}  // namespace fastkpc

#endif
