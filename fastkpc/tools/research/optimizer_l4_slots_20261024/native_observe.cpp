// Full-output retention around the real production batch entrypoint.
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include "native_observe.hpp"
#include "symbols.inc"
#include <chrono>
#include <dlfcn.h>
#include <unordered_map>
namespace l4_observe {
using window_diag::need;
static bool active=false;
static std::unordered_map<std::string, Destination*> destinations;
void begin(std::vector<Destination>& rows){
  need(!active,"nested native retention");destinations.clear();
  for(auto& r:rows){need(r.request&&r.event,"retention destination");r.visits=0;
    need(destinations.emplace(r.request->keys.at(0),&r).second,"duplicate retention key");}
  active=true;
}
void end(bool complete){
  active=false;
  if(complete)for(const auto& entry:destinations)need(entry.second->visits==1,"native observer missing/duplicate call");
  destinations.clear();
}
template<class F>F original(const char* symbol){
  const auto path=std::getenv("FASTKPC_WINDOW_NATIVE");need(path,"native path missing");
  void* h=dlopen(path,RTLD_NOW|RTLD_NOLOAD);need(h,"native not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,symbol));dlclose(h);need(f,"native symbol missing");return f;
}
}
namespace fastkpc {
MultiPenaltyGcvCudaBatchResultsmall64 multi_penalty_gcv_cuda_optimize_batch_small64(
 const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const double* Y,int n,int count,
 const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control){
  using namespace l4_observe;using Clock=std::chrono::steady_clock;
  static auto f=original<decltype(&multi_penalty_gcv_cuda_optimize_batch_small64)>(SYM_small);
  need(active,"production batch outside retention scope");
  const auto& map=destinations;auto* d=map.at(keys.at(0));
  need(keys==d->request->keys&&count==int(keys.size())&&n==351,"native retention identity");
  auto t=Clock::now();auto result=f(p,Y,n,count,keys,control);
  d->native_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();t=Clock::now();
  l4_slot::preserve_native_output(p,count,d->coefficients,d->fitted,d->residual,d->event);
  d->retain_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();++d->visits;
  return result;
}
}
