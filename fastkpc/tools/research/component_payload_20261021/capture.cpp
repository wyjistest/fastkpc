// Capture actual accepted-native cache commits, including singleton cache
// misses. Explicit diagnostic D2H only; the production numerical calls remain.
#include "full_cuda_ci_vertical.hpp"
#include "mgcv_multi_penalty_gcv.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void cuda_ok(cudaError_t code) {
  require(code == cudaSuccess, cudaGetErrorString(code));
}
template<class F> F native(const char* symbol) {
  const char* path = std::getenv("FASTKPC_PAYLOAD_CAPTURE_NATIVE");
  require(path != nullptr, "missing payload native identity");
  void* handle = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  require(handle != nullptr, "payload native not loaded");
  void* result = dlsym(handle, symbol);
  dlclose(handle);
  require(result != nullptr, "payload native symbol missing");
  return reinterpret_cast<F>(result);
}
const std::unordered_set<std::string>& selected_keys() {
  static const auto keys = [] {
    const char* path = std::getenv("FASTKPC_PAYLOAD_CAPTURE_KEYS");
    require(path != nullptr, "missing payload capture keys");
    std::ifstream in(path);
    std::unordered_set<std::string> result;
    std::string key;
    while (in >> key) {
      require(key.size() == 64, "invalid payload capture key");
      result.insert(key);
    }
    require(in.eof() && !result.empty(), "cannot read payload capture keys");
    return result;
  }();
  return keys;
}
struct Copy { void* destination; std::size_t bytes; };
thread_local std::vector<Copy>* active = nullptr;
std::unordered_set<std::string> captured;
void write_binary(const std::string& path, const std::vector<double>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(double));
  require(out.good(), "payload diagnostic write failed");
}
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst, const void* src,
    std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
  using Function = decltype(&cudaMemcpyAsync);
  static Function original = [] {
    void* handle = dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);
    require(handle != nullptr,"cudart not loaded");
    auto fn = reinterpret_cast<Function>(dlsym(handle,"cudaMemcpyAsync"));
    dlclose(handle);
    require(fn != nullptr,"cudaMemcpyAsync missing");
    return fn;
  }();
  const auto result = original(dst,src,bytes,kind,stream);
  if (active && kind == cudaMemcpyDeviceToDevice && result == cudaSuccess) {
    Dl_info caller{};
    if (dladdr(__builtin_return_address(0),&caller) && caller.dli_fname &&
        std::string(caller.dli_fname) == std::getenv("FASTKPC_PAYLOAD_CAPTURE_NATIVE"))
      active->push_back(Copy{dst,bytes});
  }
  return result;
}

namespace fastkpc {
FullCudaCiLegacyEigBatchResult
run_full_cuda_ci_phase35_legacy_eig_batch_from_exact_primary_reuse_residual(
    const std::shared_ptr<PreparedSGpuHandle>& prepared,
    const MultiPenaltyGcvCudaResidualConsumerView& residual,
    const FullCudaCiLegacyEigBatchRequest& request,
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  using Function = decltype(&run_full_cuda_ci_phase35_legacy_eig_batch_from_exact_primary_reuse_residual);
  static Function original = native<Function>(
    "_ZN7fastkpc75run_full_cuda_ci_phase35_legacy_eig_batch_from_exact_primary_reuse_residualERKSt10shared_ptrINS_18PreparedSGpuHandleEERKNS_39MultiPenaltyGcvCudaResidualConsumerViewERKNS_31FullCudaCiLegacyEigBatchRequestERKS0_INS_33FullCudaCiLegacyEigComponentCacheEE");
  using Contains = decltype(&full_cuda_ci_legacy_eig_component_cache_contains);
  static Contains contains = native<Contains>(
    "_ZN7fastkpc48full_cuda_ci_legacy_eig_component_cache_containsERKSt10shared_ptrINS_33FullCudaCiLegacyEigComponentCacheEERKSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISB_EE");
  require(cache != nullptr && active == nullptr,"invalid payload capture scope");
  const auto hits = contains(cache,residual.target_keys);
  std::vector<bool> referenced(residual.target_count,false);
  for (const auto& pair : request.pairs) {
    referenced.at(pair.left_target_index) = true;
    referenced.at(pair.right_target_index) = true;
  }
  std::vector<int> misses;
  bool wanted = false;
  for (int i=0;i<residual.target_count;++i) if (referenced[i] && !hits.at(i)) {
    misses.push_back(i);
    wanted = wanted || selected_keys().count(residual.target_keys[i]);
  }
  if (!wanted) return original(prepared,residual,request,cache);
  std::vector<Copy> copies;
  active = &copies;
  FullCudaCiLegacyEigBatchResult result;
  try { result = original(prepared,residual,request,cache); }
  catch (...) { active = nullptr; throw; }
  active = nullptr;
  require(copies.size() == 4*misses.size(),"unexpected cache D2D sequence");
  require(result.diagnostics.component_cache_persistent_insert_count == int(misses.size()) &&
          result.diagnostics.component_eig_configured_lane_count == 8,
          "accepted cache capture coverage changed");
  const char* directory = std::getenv("FASTKPC_PAYLOAD_CAPTURE_DIR");
  require(directory != nullptr,"missing payload capture directory");
  const std::size_t lengths[] = {std::size_t(residual.n)*request.num_col,
                                 std::size_t(request.num_col),1U,1U};
  for (std::size_t j=0;j<misses.size();++j) {
    const int i = misses[j];
    const auto& key = residual.target_keys[i];
    if (!selected_keys().count(key)) continue;
    require(captured.insert(key).second,"payload captured twice");
    std::vector<double> payload;
    for (int f=0;f<4;++f) {
      const Copy& copy = copies[4*j+f];
      require(copy.bytes == lengths[f]*sizeof(double),"cache payload layout changed");
      const auto offset = payload.size();
      payload.resize(offset+lengths[f]);
      cuda_ok(cudaMemcpy(payload.data()+offset,copy.destination,copy.bytes,cudaMemcpyDeviceToHost));
    }
    std::vector<double> shadow(residual.n);
    cuda_ok(cudaMemcpy(shadow.data(),residual.residuals+std::size_t(i)*residual.n,
      shadow.size()*sizeof(double),cudaMemcpyDeviceToHost));
    const std::string prefix = std::string(directory)+"/"+key;
    write_binary(prefix+".f64",payload);
    write_binary(prefix+".residual.f64",shadow);
    std::ofstream out(prefix+".json");
    out << "{\"key\":\"" << key << "\",\"request\":\""
        << request.request_identity_sha256 << "\",\"n\":" << residual.n
        << ",\"q\":" << residual.coefficient_dim << ",\"num_col\":" << request.num_col
        << ",\"residual_targets\":" << residual.target_count
        << ",\"component_misses\":" << misses.size()
        << ",\"component_position\":" << j
        << ",\"configured_lanes\":" << result.diagnostics.component_eig_configured_lane_count
        << ",\"active_lanes\":" << result.diagnostics.component_eig_active_lane_count << "}\n";
    require(out.good(),"payload metadata write failed");
  }
  return result;
}
}
