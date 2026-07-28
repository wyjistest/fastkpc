#ifndef FASTKPC_FULL_CUDA_CI_CONTRACT_HPP
#define FASTKPC_FULL_CUDA_CI_CONTRACT_HPP

#include <string>

namespace fastkpc {

struct FullCudaCiContractIdentity {
  std::string contract_name;
  std::string contract_schema_version;
  int semantic_major = 0;
  int semantic_minor = 0;
  int semantic_patch = 0;
  std::string canonical_json;
  std::string sha256;
};

std::string full_cuda_ci_sha256_utf8(const std::string& value);

FullCudaCiContractIdentity full_cuda_ci_contract_identity(
  const std::string& json,
  const std::string& expected_contract_name);

}  // namespace fastkpc

#endif
