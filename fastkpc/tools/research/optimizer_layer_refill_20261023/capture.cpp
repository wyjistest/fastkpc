// Capture production L5 inputs/outputs without changing numerical execution.
#include "layer_fixture.hpp"
#include "skeleton_task_scheduler.hpp"
#include <dlfcn.h>
#include <map>
#include <iostream>
#include "symbols.inc"
using namespace layer_diag;
namespace {
int level=-1;
std::map<int,int> windows;
std::map<const void*,Geometry> geometries;
const char* env(const char* key){auto p=std::getenv(key);need(p&&*p,key);return p;}
template<class F> F original(const char* name){
  void* h=dlopen(env("FASTKPC_WINDOW_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"native not loaded");auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);
  need(f,name);return f;
}
}
LayerPlan make_layer_plan(const std::vector<int>& a,int p,int l){
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);
  level=l;geometries.clear();return f(a,p,l);
}
namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_gcv_capacity_prepared(
    const double* X,const double* qr,const double* tau,const double* R,const int* pivot,
    const std::vector<std::vector<double>>& roots,const std::vector<std::vector<double>>& matrices,
    const std::vector<int>& ranks,const double* sp,int n,int q,int penalties,int capacity,int device){
  static auto f=original<decltype(&create_multi_penalty_gcv_capacity_prepared)>(SYM_create);
  auto p=f(X,qr,tau,R,pivot,roots,matrices,ranks,sp,n,q,penalties,capacity,device);
  if(level==5){
    Geometry g;g.n=n;g.q=q;g.penalties=penalties;g.capacity=capacity;
    g.X={X,X+size_t(n)*q};g.qr={qr,qr+size_t(n)*q};g.tau={tau,tau+q};
    g.R={R,R+size_t(q)*q};g.pivot={pivot,pivot+q};g.initial={sp,sp+penalties};
    g.roots=roots;g.matrices=matrices;g.ranks=ranks;geometries[p.get()]=std::move(g);
  }
  return p;
}
MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,int concurrency,bool retain){
  static auto f=original<decltype(&multi_penalty_gcv_capacity_optimize_multi)>(SYM_multi);
  const int w=++windows[level];Window fixture;fixture.window=w;
  const size_t count=requests.size();
  if(level==5){
    need(w<=15 && count==size_t(w==15?59:64) &&
      concurrency==int(count) && retain,"capture L5 scheduler changed");
    for(const auto& input:requests){
      Request r;r.geometry=geometries.at(input.prepared.get());r.Y=input.Y;
      r.keys=input.target_keys;r.control=input.control;fixture.requests.push_back(std::move(r));
    }
  }
  auto result=f(std::move(requests),concurrency,retain);
  if(level==5){
    need(result.setups.size()==count && result.residuals.size()==count,"capture result coverage");
    for(size_t i=0;i<count;++i)fixture.requests[i].expected=snapshot(result.setups[i],result.residuals[i]);
    const auto path=std::string(env("FASTKPC_WINDOW_CAPTURE"))+"/L5-W"+std::to_string(w)+".bin";
    fixture.save(path);std::cerr<<"captured "<<path<<" setups="<<count<<std::endl;
  }
  geometries.clear();return result;
}
}
