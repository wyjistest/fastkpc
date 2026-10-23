// Research interposition: observe accepted-native consumers, never replace a result.
#include "full_cuda_ci_vertical.hpp"
#include "mgcv_multi_penalty_gcv.hpp"
#include "../full_cuda_ci_contract.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void cuda_ok(cudaError_t s) { require(s == cudaSuccess, cudaGetErrorString(s)); }
template<class F> F native(const char* symbol) {
  const char* path = std::getenv("FASTKPC_SCALAR_CAPTURE_NATIVE");
  require(path, "missing native path");
  void* h = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  require(h, "native not loaded");
  void* p = dlsym(h, symbol); dlclose(h);
  require(p, "native symbol missing");
  return reinterpret_cast<F>(p);
}
std::string field(const std::string& s, const std::string& key) {
  const auto start = s.find("\n" + key + "=");
  require(start != std::string::npos, "identity field missing");
  const auto a = start + key.size() + 2;
  return s.substr(a, s.find('\n', a) - a);
}
std::map<std::string,std::string> prepared_sets, residual_targets;
std::map<std::string,std::pair<std::string,int>> target_ids;
std::set<std::string> captured_keys;
const std::set<std::string>& changed_keys() {
  static const auto keys = [] {
    std::ifstream in(std::getenv("FASTKPC_SCALAR_CAPTURE_KEYS"));
    std::set<std::string> result; std::string key;
    while (in >> key) result.insert(key);
    require(in.eof() && result.size() == 1002, "changed-key count differs");
    return result;
  }();
  return keys;
}
std::string directory() {
  const char* path = std::getenv("FASTKPC_SCALAR_CAPTURE_DIR");
  require(path, "capture directory missing"); return path;
}
template<class T> std::vector<T> download(const void* p, std::size_t count) {
  std::vector<T> values(count);
  cuda_ok(cudaMemcpy(values.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));
  return values;
}
template<class T> void binary(const std::string& name,const std::vector<T>& values) {
  std::ofstream out(directory()+"/"+name,std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
  require(out.good(), "capture write failed");
}
struct Active { int pairs; const std::uintptr_t* metadata=nullptr; const void* results=nullptr; int gemms=0; };
thread_local Active* active=nullptr;
unsigned batch_counter=0;
}

namespace fastkpc {
std::string full_cuda_ci_sha256_utf8(const std::string& input) {
  static auto original=native<decltype(&full_cuda_ci_sha256_utf8)>(
    "_ZN7fastkpc24full_cuda_ci_sha256_utf8ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
  auto result=original(input);
  if (input.rfind("schema=full-cuda-ci-prepared-s-key-v1\n",0)==0)
    prepared_sets[result]=field(input,"S");
  else if (input.rfind("schema=full-cuda-ci-target-key-v1\n",0)==0)
    target_ids[result]={field(input,"prepared"),std::stoi(field(input,"target"))};
  else if (input.rfind("schema=full-cuda-ci-residual-key-v2\n",0)==0 ||
           input.rfind("schema=full-cuda-ci-residual-key-v1\n",0)==0)
    residual_targets[result]=field(input,"target");
  return result;
}
}

extern "C" cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,
    const double* alpha,const double* const A[],int lda,
    const double* const B[],int ldb,const double* beta,double* const C[],int ldc,int count) {
  using F=decltype(&cublasDgemmBatched);
  static F original=[] { void* h=dlopen("libcublas.so.12",RTLD_NOW|RTLD_NOLOAD);
    require(h,"cublas library not loaded");
    auto p=reinterpret_cast<F>(dlsym(h,"cublasDgemmBatched")); dlclose(h);
    require(p,"cublas batched symbol missing"); return p; }();
  auto status=original(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,count);
  if (active && status==CUBLAS_STATUS_SUCCESS) {
    require(count==active->pairs && m==35 && n==35 && k==351,"unexpected pair GEMM");
    active->metadata=reinterpret_cast<const std::uintptr_t*>(A)-count;
    ++active->gemms;
  }
  return status;
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst,const void* src,std::size_t size,
    cudaMemcpyKind kind,cudaStream_t stream) {
  using F=decltype(&cudaMemcpyAsync);
  static F original=[] { void* h=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);
    require(h,"cudart library not loaded");
    auto p=reinterpret_cast<F>(dlsym(h,"cudaMemcpyAsync")); dlclose(h);
    require(p,"cudart copy symbol missing"); return p; }();
  auto status=original(dst,src,size,kind,stream);
  if (active && kind==cudaMemcpyDeviceToHost && size==std::size_t(active->pairs)*72 &&
      status==cudaSuccess) {
    require(active->results==nullptr,"duplicate compact copy"); active->results=src;
  }
  return status;
}

namespace fastkpc {
FullCudaCiLegacyEigBatchResult run_full_cuda_ci_phase35_legacy_eig_batch_impl(
    const std::shared_ptr<PreparedSGpuHandle>& prepared,const FixedSpBatchHostView& batch,
    const FullCudaCiLegacyEigBatchRequest& request,
    const MultiPenaltyGcvCudaResidualConsumerView* residual,const char* semantic,
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  using F=decltype(&run_full_cuda_ci_phase35_legacy_eig_batch_impl);
  static F original=native<F>("_ZN7fastkpc46run_full_cuda_ci_phase35_legacy_eig_batch_implERKSt10shared_ptrINS_18PreparedSGpuHandleEERKNS_20FixedSpBatchHostViewERKNS_31FullCudaCiLegacyEigBatchRequestEPKNS_39MultiPenaltyGcvCudaResidualConsumerViewEPKcRKS0_INS_33FullCudaCiLegacyEigComponentCacheEE");
  const unsigned batch_id=++batch_counter;
  bool wanted=false;
  for (const auto& p:request.pairs)
    wanted=wanted || changed_keys().count(batch.target_keys.at(p.left_target_index)) ||
                     changed_keys().count(batch.target_keys.at(p.right_target_index));
  Active scope{int(request.pairs.size())};
  require(active==nullptr,"nested pair capture");
  if (wanted) active=&scope;
  FullCudaCiLegacyEigBatchResult result;
  try { result=original(prepared,batch,request,residual,semantic,cache); }
  catch (...) { active=nullptr; throw; }
  active=nullptr;
  std::ofstream log(directory()+"/requests.jsonl",std::ios::app);
  log << "{\"batch\":" << batch_id << ",\"request\":\"" << request.request_identity_sha256
      << "\",\"prepared\":\"" << request.expected_prepared_s_key_sha256
      << "\",\"captured\":" << (wanted?"true":"false")
      << ",\"misses\":" << result.diagnostics.component_cache_persistent_miss_count
      << ",\"pairs\":[";
  std::vector<std::uintptr_t> metadata;
  if (wanted) {
    require(scope.gemms==1 && scope.metadata && scope.results,"pair capture hooks missing");
    metadata=download<std::uintptr_t>(scope.metadata,10*scope.pairs);
    binary(std::to_string(batch_id)+".compact",download<unsigned char>(scope.results,72*scope.pairs));
    binary(std::to_string(batch_id)+".cross",download<double>(
      reinterpret_cast<const void*>(metadata[3*scope.pairs]),std::size_t(scope.pairs)*35*35));
  }
  for (int j=0;j<scope.pairs;++j) {
    const auto& pair=request.pairs[j];
    const auto& left=batch.target_keys.at(pair.left_target_index);
    const auto& right=batch.target_keys.at(pair.right_target_index);
    if (j) log << ',';
    log << "{\"id\":" << pair.logical_sequence_id << ",\"left\":\"" << left
        << "\",\"right\":\"" << right << '"';
    if (wanted) {
      require(metadata[j]==pair.logical_sequence_id,"logical id differs");
      const auto l=target_ids.at(residual_targets.at(left));
      const auto r=target_ids.at(residual_targets.at(right));
      require(l.first==r.first && l.first==request.expected_prepared_s_key_sha256,"target identity differs");
      log << ",\"x\":" << l.second << ",\"y\":" << r.second
          << ",\"S\":\"" << prepared_sets.at(l.first) << '"';
      const std::string keys[]={left,right};
      const std::size_t offsets[]={1U,2U};
      for (int side=0;side<2;++side) if (captured_keys.insert(keys[side]).second) {
        std::vector<double> values;
        const std::size_t parts[]={offsets[side],std::size_t(4+side),std::size_t(6+side),std::size_t(8+side)};
        const std::size_t lengths[]={351U*35U,35U,1U,1U};
        for (int f=0;f<4;++f) {
          auto v=download<double>(reinterpret_cast<const void*>(metadata[parts[f]*scope.pairs+j]),lengths[f]);
          values.insert(values.end(),v.begin(),v.end());
        }
        binary(keys[side]+".payload",values);
      }
    }
    log << '}';
  }
  log << "]}\n";
  require(log.good(),"request metadata write failed");
  return result;
}
}
