#ifndef FASTKPC_FULL_CUDA_CI_CONTRACT_HPP
#define FASTKPC_FULL_CUDA_CI_CONTRACT_HPP

#include <array>
#include <cstddef>
#include <memory>
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

class FullCudaCiSha256Builder {
 public:
  FullCudaCiSha256Builder();
  ~FullCudaCiSha256Builder();
  FullCudaCiSha256Builder(FullCudaCiSha256Builder&&) noexcept;
  FullCudaCiSha256Builder& operator=(FullCudaCiSha256Builder&&) noexcept;
  FullCudaCiSha256Builder(const FullCudaCiSha256Builder&) = delete;
  FullCudaCiSha256Builder& operator=(const FullCudaCiSha256Builder&) = delete;

  void reset();
  void update(const void* value, std::size_t size);
  std::array<unsigned char, 32> finish();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string full_cuda_ci_sha256_utf8(const std::string& value);
std::string full_cuda_ci_sha256_bytes(const void* value, std::size_t size);
std::string full_cuda_ci_sha256_hex(
  const std::array<unsigned char, 32>& digest);
const char* full_cuda_ci_sha256_backend();

FullCudaCiContractIdentity full_cuda_ci_contract_identity(
  const std::string& json,
  const std::string& expected_contract_name);

}  // namespace fastkpc

#endif
