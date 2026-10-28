// Research-only host interposition. Every numerical call goes to the exact native.
#include "fixture.hpp"
#include "skeleton_task_scheduler.hpp"
#include "symbols.inc"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <mutex>
#define R_NO_REMAP
#include <Rinternals.h>
using namespace late_diag;
namespace {
using Clock=std::chrono::steady_clock;
struct State {
  std::mutex mutex;
  std::map<std::string,const void*> functions;
  int level=-1,l8_creates=0,preload_batches=0,attribute_calls=0,errors=0,captures=0;
  double preload_ms=0;
  size_t free_before=0,free_after=0,capture_bytes=0;
  std::map<int,int> windows;
  std::map<const void*,Geometry> geometries;
};
State& state(){static auto* s=new State;return *s;}
const char* required(const char* key){auto p=std::getenv(key);need(p&&*p,key);return p;}
bool enabled(){static const bool value=[](){const auto* p=required("FASTKPC_LATE_L8_PRELOAD");
  need(std::string(p)=="0"||std::string(p)=="1","invalid L8 preload mode");return std::string(p)=="1";}();return value;}
bool capturing(){const auto* p=std::getenv("FASTKPC_LATE_CAPTURE");return p&&*p;}
template<class F>F original(const char* symbol){
  void* h=dlopen(required("FASTKPC_WINDOW_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"native not loaded");auto f=reinterpret_cast<F>(dlsym(h,symbol));dlclose(h);
  need(f,symbol);return f;
}
bool wanted(const char* name){
  if(!name||std::string(name).find("_mgcv_multi_penalty_gcv_cu_")==std::string::npos)return false;
  for(const auto* suffix:{"target_squared_norm_kernel","magic_qt_y_kernel",
      "optimize_multi_penalty_targets_kernel","gather_multi_penalty_coefficients_kernel",
      "form_multi_penalty_residuals_kernel"})if(std::string(name).find(suffix)!=std::string::npos)return true;
  return false;
}
void preload(){auto& s=state();need(s.preload_batches==0&&s.functions.size()==5,"L8 function registry");
  ++s.preload_batches;size_t total=0;auto check=[&](cudaError_t e){if(e!=cudaSuccess){++s.errors;throw std::runtime_error(cudaGetErrorString(e));}};
  check(cudaMemGetInfo(&s.free_before,&total));auto t=Clock::now();
  for(const auto& entry:s.functions){cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,entry.second));++s.attribute_calls;}
  s.preload_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
  check(cudaMemGetInfo(&s.free_after,&total));
  need(std::min(s.free_before,s.free_after)>=size_t(2)*1024*1024*1024,"L8 preload headroom");
  need(s.free_before<=s.free_after+size_t(256)*1024*1024,"L8 preload memory increment");
}
}
extern "C" void __cudaRegisterFunction(void** fat,const char* host,char* device,
    const char* name,int limit,uint3* tid,uint3* bid,dim3* block,dim3* grid,int* warp){
  using Fn=void(*)(void**,const char*,char*,const char*,int,uint3*,uint3*,dim3*,dim3*,int*);
  static auto f=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"__cudaRegisterFunction"));
  if(!f)std::abort();f(fat,host,device,name,limit,tid,bid,block,grid,warp);
  if(wanted(name)){auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    if(!s.functions.emplace(name,host).second)std::abort();}
}
LayerPlan make_layer_plan(const std::vector<int>& a,int p,int level){
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);
  auto& s=state();s.level=level;s.geometries.clear();return f(a,p,level);
}
namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_gcv_capacity_prepared(
    const double* X,const double* qr,const double* tau,const double* R,const int* pivot,
    const std::vector<std::vector<double>>& roots,const std::vector<std::vector<double>>& matrices,
    const std::vector<int>& ranks,const double* sp,int n,int q,int penalties,int capacity,int device){
  static auto f=original<decltype(&create_multi_penalty_gcv_capacity_prepared)>(SYM_create);
  auto p=f(X,qr,tau,R,pivot,roots,matrices,ranks,sp,n,q,penalties,capacity,device);auto& s=state();
  if(s.level==8){need(n==351&&q==73&&penalties==8,"L8 preload shape");
    if(++s.l8_creates==1&&enabled())preload();}
  if(capturing()&&(s.level==6||s.level==8)){
    Geometry g;g.n=n;g.q=q;g.penalties=penalties;g.capacity=capacity;
    g.X={X,X+size_t(n)*q};g.qr={qr,qr+size_t(n)*q};g.tau={tau,tau+q};
    g.R={R,R+size_t(q)*q};g.pivot={pivot,pivot+q};g.initial={sp,sp+penalties};
    g.roots=roots;g.matrices=matrices;g.ranks=ranks;s.geometries[p.get()]=std::move(g);
  }return p;
}
MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,int concurrency,bool retain){
  static auto f=original<decltype(&multi_penalty_gcv_capacity_optimize_multi)>(SYM_multi);
  auto& s=state();const bool capture=capturing()&&(s.level==6||s.level==8);
  Window fixture;const size_t count=requests.size();
  if(capture){fixture.level=s.level;fixture.window=++s.windows[s.level];
    need(concurrency==int(count)&&retain,"late original concurrency/retention");
    for(const auto& in:requests){Request r;r.geometry=s.geometries.at(in.prepared.get());
      r.Y=in.Y;r.keys=in.target_keys;r.control=in.control;fixture.requests.push_back(std::move(r));}}
  auto result=f(std::move(requests),concurrency,retain);
  if(capture){need(result.setups.size()==count&&result.residuals.size()==count,"late result coverage");
    for(size_t i=0;i<count;++i){fixture.requests[i].expected=snapshot(result.setups[i],result.residuals[i]);
      const auto& r=fixture.requests[i];s.capture_bytes+=(r.geometry.q+2*r.geometry.n)*r.keys.size()*sizeof(double);}
    const auto path=std::string(required("FASTKPC_LATE_CAPTURE"))+"/L"+std::to_string(s.level)+"-W"+std::to_string(fixture.window)+".bin";
    fixture.save(path);++s.captures;std::cerr<<"captured "<<path<<" setups="<<count<<std::endl;
  }s.geometries.clear();return result;
}
}
extern "C" SEXP late_optimizer_stats(){
  auto& s=state();const std::array<const char*,12> names={"enabled","registered_functions","l8_prepared",
    "preload_batches","attribute_calls","errors","preload_ms","free_before","free_after",
    "capture_windows","capture_diagnostic_d2h_bytes","function_names"};
  SEXP result=PROTECT(Rf_allocVector(VECSXP,names.size()));SEXP labels=PROTECT(Rf_allocVector(STRSXP,names.size()));
  for(size_t i=0;i<names.size();++i)SET_STRING_ELT(labels,i,Rf_mkChar(names[i]));
  const std::array<double,11> values={double(enabled()),double(s.functions.size()),double(s.l8_creates),
    double(s.preload_batches),double(s.attribute_calls),double(s.errors),s.preload_ms,double(s.free_before),
    double(s.free_after),double(s.captures),double(s.capture_bytes)};
  for(size_t i=0;i<values.size();++i)SET_VECTOR_ELT(result,i,Rf_ScalarReal(values[i]));
  SEXP funcs=PROTECT(Rf_allocVector(STRSXP,s.functions.size()));size_t i=0;
  for(const auto& entry:s.functions)SET_STRING_ELT(funcs,i++,Rf_mkChar(entry.first.c_str()));
  SET_VECTOR_ELT(result,11,funcs);Rf_setAttrib(result,R_NamesSymbol,labels);UNPROTECT(3);return result;
}
