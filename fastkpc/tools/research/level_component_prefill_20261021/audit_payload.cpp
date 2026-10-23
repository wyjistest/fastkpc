// External qualification observer. Never changes a numerical result. Diagnostic
// D2H occurs after the real pair completion and is excluded from timing evidence.
#include "full_cuda_ci_vertical.hpp"
#include "mgcv_multi_penalty_gcv.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void cuda_ok(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
const char* env(const char* name) { const char* p=std::getenv(name); require(p && *p,"audit environment missing"); return p; }
template<class F> F symbol(const char* library,const char* name) {
  void* h=dlopen(library, RTLD_NOW|RTLD_NOLOAD); require(h,"audit library not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,name)); dlclose(h); require(f,"audit symbol missing"); return f;
}
const std::set<std::string>& references() {
  static const auto result=[] {
    std::set<std::string> keys;
    for (const auto& item:std::filesystem::directory_iterator(env("FASTKPC_LSCP_REFERENCE"))) {
      const auto p=item.path();
      if(p.extension()==".f64" && p.stem().string().size()==64) keys.insert(p.stem().string());
    }
    require(keys.size()==6632,"payload reference coverage changed"); return keys;
  }(); return result;
}
struct Active { int pairs; const std::uintptr_t* metadata=nullptr; int gemms=0; };
thread_local Active* active=nullptr;
std::set<std::string> checked;
template<class T> std::vector<T> download(const void* p,std::size_t n) {
  std::vector<T> result(n); cuda_ok(cudaMemcpy(result.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return result;
}
}

extern "C" cublasStatus_t cublasDgemmBatched(cublasHandle_t h,
    cublasOperation_t ta,cublasOperation_t tb,int m,int n,int k,
    const double* alpha,const double* const a[],int lda,const double* const b[],int ldb,
    const double* beta,double* const c[],int ldc,int count) {
  static auto original=symbol<decltype(&cublasDgemmBatched)>("libcublas.so.12","cublasDgemmBatched");
  const auto status=original(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc,count);
  if(active && status==CUBLAS_STATUS_SUCCESS) {
    require(count==active->pairs && m==35 && n==35 && k==351,"audit pair shape changed");
    active->metadata=reinterpret_cast<const std::uintptr_t*>(a)-count; ++active->gemms;
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
  static F original=symbol<F>(env("FASTKPC_LSCP_NATIVE"),
    "_ZN7fastkpc46run_full_cuda_ci_phase35_legacy_eig_batch_implERKSt10shared_ptrINS_18PreparedSGpuHandleEERKNS_20FixedSpBatchHostViewERKNS_31FullCudaCiLegacyEigBatchRequestEPKNS_39MultiPenaltyGcvCudaResidualConsumerViewEPKcRKS0_INS_33FullCudaCiLegacyEigComponentCacheEE");
  bool wanted=false;
  for(const auto& key:batch.target_keys) wanted=wanted || (references().count(key) && !checked.count(key));
  if(!wanted) return original(prepared,batch,request,residual,semantic,cache);
  Active scope{static_cast<int>(request.pairs.size())};
  require(!active,"nested payload audit"); active=&scope;
  FullCudaCiLegacyEigBatchResult result;
  try { result=original(prepared,batch,request,residual,semantic,cache); }
  catch(...) { active=nullptr; throw; }
  active=nullptr;
  require(scope.gemms==1 && scope.metadata,"audit missed consumer GEMM");
  const auto metadata=download<std::uintptr_t>(scope.metadata,10*scope.pairs);
  for(int j=0;j<scope.pairs;++j) {
    const auto& pair=request.pairs[j];
    const int indices[]={pair.left_target_index,pair.right_target_index};
    for(int side=0;side<2;++side) {
      const auto& key=batch.target_keys.at(indices[side]);
      if(!references().count(key) || !checked.insert(key).second) continue;
      std::vector<double> actual;
      const std::size_t parts[]={std::size_t(1+side),std::size_t(4+side),std::size_t(6+side),std::size_t(8+side)};
      const std::size_t lengths[]={351U*35U,35U,1U,1U};
      for(int f=0;f<4;++f) {
        auto field=download<double>(reinterpret_cast<const void*>(metadata[parts[f]*scope.pairs+j]),lengths[f]);
        actual.insert(actual.end(),field.begin(),field.end());
      }
      std::vector<double> expected(actual.size());
      std::ifstream input(std::string(env("FASTKPC_LSCP_REFERENCE"))+"/"+key+".f64",std::ios::binary);
      input.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(double));
      require(input.gcount()==static_cast<std::streamsize>(expected.size()*sizeof(double)),"payload reference truncated");
      const bool exact=std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(double))==0;
      std::ofstream log(env("FASTKPC_LSCP_PAYLOAD_LOG"),std::ios::app);
      log << key << ' ' << (exact?"PASS":"FAIL") << '\n'; log.close();
      require(exact,"consumed four-field payload differs from accepted native");
    }
  }
  return result;
}
}
