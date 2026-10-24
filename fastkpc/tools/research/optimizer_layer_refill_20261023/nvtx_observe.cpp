// Profile-only setup scopes. No diagnostic CUDA event or memcpy interception.
#include "layer_fixture.hpp"
#include "observe.hpp"
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <unordered_map>
#include "symbols.inc"
namespace window_diag {
bool profile_enabled=false;
std::unordered_map<std::string,size_t> profile_indices;
void observe_begin(const std::vector<std::string>& keys,bool enabled){
  need(!profile_enabled,"nested profile");profile_indices.clear();
  for(size_t i=0;i<keys.size();++i)need(profile_indices.emplace(keys[i],i).second,"duplicate setup");
  profile_enabled=enabled;
}
void observe_end(const std::string&){profile_enabled=false;profile_indices.clear();}
template<class F>F original(const char* symbol){
  auto path=std::getenv("FASTKPC_WINDOW_NATIVE");need(path,"native path missing");
  void* h=dlopen(path,RTLD_NOW|RTLD_NOLOAD);need(h,"native not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,symbol));dlclose(h);need(f,symbol);return f;
}
template<class F,class P>auto call(F f,const P& p,const double* Y,int n,int count,
    const std::vector<std::string>& keys,const fastkpc::MultiPenaltyGcvCudaOptimizerControl& control){
  if(!profile_enabled)return f(p,Y,n,count,keys,control);
  const auto label="setup:"+std::to_string(profile_indices.at(keys.at(0)));nvtxRangePushA(label.c_str());
  try{auto result=f(p,Y,n,count,keys,control);nvtxRangePop();return result;}
  catch(...){nvtxRangePop();throw;}
}
}
namespace fastkpc {
MultiPenaltyGcvCudaBatchResult multi_penalty_gcv_cuda_optimize_batch(
 const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& p,const double* Y,int n,int count,
 const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control){
  static auto f=window_diag::original<decltype(&multi_penalty_gcv_cuda_optimize_batch)>(SYM_batch);
  return window_diag::call(f,p,Y,n,count,keys,control);
}
MultiPenaltyGcvCudaBatchResultsmall64 multi_penalty_gcv_cuda_optimize_batch_small64(
 const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const double* Y,int n,int count,
 const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control){
  static auto f=window_diag::original<decltype(&multi_penalty_gcv_cuda_optimize_batch_small64)>(SYM_small);
  return window_diag::call(f,p,Y,n,count,keys,control);
}
}
