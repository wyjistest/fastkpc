// External observation only. Native computations and scheduling are unchanged.
#include "fixture.hpp"
#include "skeleton_task_scheduler.hpp"
#include <dlfcn.h>
#include <map>
#include <iostream>
#include "symbols.inc"
using namespace window_diag;
namespace {
int level=-1;
std::map<int,int> windows;
std::map<const void*,Geometry> geometries;
const char* env(const char* key) {auto v=std::getenv(key);need(v&&*v,key);return v;}
template<class F> F original(const char* name) {
  void* h=dlopen(env("FASTKPC_WINDOW_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"native not loaded");auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);
  need(f,name);return f;
}
bool chosen(int l,int w) {return (l==3&&(w==31||w==32))||(l==5&&(w==7||w==8));}
}
LayerPlan make_layer_plan(const std::vector<int>& a,int p,int l) {
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);
  level=l;geometries.clear();return f(a,p,l);
}
namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_gcv_capacity_prepared(
    const double* X,const double* qr,const double* tau,const double* R,const int* pivot,
    const std::vector<std::vector<double>>& roots,const std::vector<std::vector<double>>& matrices,
    const std::vector<int>& ranks,const double* sp,int n,int q,int penalties,int capacity,int device) {
  static auto f=original<decltype(&create_multi_penalty_gcv_capacity_prepared)>(SYM_create);
  auto p=f(X,qr,tau,R,pivot,roots,matrices,ranks,sp,n,q,penalties,capacity,device);
  if (chosen(level,windows[level]+1)) {
    Geometry g;g.n=n;g.q=q;g.penalties=penalties;g.capacity=capacity;
    g.X={X,X+size_t(n)*q};g.qr={qr,qr+size_t(n)*q};g.tau={tau,tau+q};
    g.R={R,R+size_t(q)*q};g.pivot={pivot,pivot+q};g.initial={sp,sp+penalties};
    g.roots=roots;g.matrices=matrices;g.ranks=ranks;geometries[p.get()]=std::move(g);
  }
  return p;
}
MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,int concurrency,bool retain) {
  static auto f=original<decltype(&multi_penalty_gcv_capacity_optimize_multi)>(SYM_multi);
  const int w=++windows[level];const bool selected=chosen(level,w);
  Fixture fixture;fixture.level=level;fixture.window=w;
  if (selected) {
    need(concurrency==64 && requests.size()==64 && retain,"capture window shape changed");
    for (const auto& input:requests) {
      Request r;r.geometry=geometries.at(input.prepared.get());r.Y=input.Y;
      r.keys=input.target_keys;r.control=input.control;
      need(r.control.exact_primary_only && r.control.force_final_selected_evaluation,
        "capture optimizer control changed");fixture.requests.push_back(std::move(r));
    }
  }
  auto result=f(std::move(requests),concurrency,retain);
  if (selected) {
    need(result.setups.size()==64 && result.residuals.size()==64,"capture coverage");
    for (size_t i=0;i<64;++i) fixture.requests[i].expected=snapshot(result.setups[i],result.residuals[i]);
    const std::string path=std::string(env("FASTKPC_WINDOW_CAPTURE"))+"/L"+
      std::to_string(level)+"-W"+std::to_string(w)+".bin";
    fixture.save(path);std::cerr<<"captured "<<path<<std::endl;
  }
  geometries.clear();return result;
}
}
