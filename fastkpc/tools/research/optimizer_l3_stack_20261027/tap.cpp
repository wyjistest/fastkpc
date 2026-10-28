// Observational host wrappers; native numerical calls are forwarded unchanged.
#include "../optimizer_l3_coexistence_20261027/probe.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include "symbols.h"
#include <Rcpp.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace {
using Clock = std::chrono::steady_clock;
uint64_t ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
template<class F> F next(const char* name) {
  auto f = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
  if (!f) { std::fprintf(stderr, "stack tap missing %s\n", name); std::abort(); }
  return f;
}
std::string quote(const std::string& value) {
  std::string out="\"";
  for (unsigned char c:value) {
    if(c=='\\'||c=='\"') { out+='\\';out+=c; }
    else if(c<32) { char b[7];std::snprintf(b,sizeof(b),"\\u%04x",c);out+=b; }
    else out+=c;
  }
  return out+'\"';
}
struct State {
  std::mutex names_mutex, output_mutex;
  std::unordered_map<const void*, std::string> names;
  std::ofstream output;
  std::atomic<bool> active{false}, single_seen{false};
  std::atomic<uint64_t> launches{0}, returned{0}, requests{0}, checked{0}, targets{0}, errors{0};
};
State& state() { static State* s=new State;return *s; }
thread_local std::string request_key;
thread_local int request_targets=0;
thread_local uint64_t request_number=0, request_launch=0;
thread_local cudaStream_t request_stream=nullptr;
void observe(const char* phase, uint64_t launch=0, cudaStream_t stream=nullptr,
             const std::string& kernel="", int api_status=0) {
  auto& s=state();
  const auto started=ns();const auto entered=s.launches.load(),returned=s.returned.load();
  const auto before=layer_audit::counts();
  size_t stack=0,free=0,total=0;
  const auto stack_status=cudaDeviceGetLimit(&stack,cudaLimitStackSize);
  const auto memory_status=cudaMemGetInfo(&free,&total);
  const auto after=layer_audit::counts();const auto ended=ns();
  if(stack_status!=cudaSuccess||memory_status!=cudaSuccess)++s.errors;
  std::ostringstream row;
  row<<"{\"phase\":"<<quote(phase)<<",\"begin_ns\":"<<started<<",\"end_ns\":"<<ended
    <<",\"thread\":"<<syscall(SYS_gettid)<<",\"launch\":"<<launch
    <<",\"request_number\":"<<request_number<<",\"request_key\":"<<quote(request_key)
    <<",\"target_count\":"<<request_targets<<",\"stream\":"<<reinterpret_cast<uintptr_t>(stream)
    <<",\"kernel\":"<<quote(kernel)<<",\"api_status\":"<<api_status
    <<",\"stack_status\":"<<stack_status<<",\"memory_status\":"<<memory_status
    <<",\"stack_bytes\":"<<stack<<",\"free_bytes\":"<<free<<",\"total_bytes\":"<<total
    <<",\"app_bytes_before\":"<<before.bytes<<",\"app_bytes\":"<<after.bytes
    <<",\"allocation_calls_before\":"<<before.allocation_calls<<",\"allocation_calls\":"<<after.allocation_calls
    <<",\"allocations\":"<<after.allocations<<",\"streams\":"<<after.streams
    <<",\"events\":"<<after.events<<",\"blas\":"<<after.blas
    <<",\"worker_mutations\":"<<after.forbidden_calls
    <<",\"launches_entered_before\":"<<entered<<",\"launches_entered_after\":"<<s.launches.load()
    <<",\"launches_returned_before\":"<<returned<<",\"launches_returned_after\":"<<s.returned.load()
    <<",\"checked_setups\":"<<s.checked.load()<<",\"checked_targets\":"<<s.targets.load()<<"}\n";
  std::lock_guard<std::mutex> lock(s.output_mutex);
  if(!s.output.is_open()) {
    const char* path=std::getenv("FASTKPC_L3_STACK_LOG");if(!path)std::abort();
    s.output.open(path,std::ios::out|std::ios::app);
  }
  s.output<<row.str();s.output.flush();if(!s.output)std::abort();
}
}

extern "C" void __cudaRegisterFunction(void** fat,const char* host,char* device,
    const char* name,int limit,uint3* tid,uint3* bid,dim3* block,dim3* grid,int* warp) {
  using F=void(*)(void**,const char*,char*,const char*,int,uint3*,uint3*,dim3*,dim3*,int*);
  static auto f=next<F>("__cudaRegisterFunction");
  f(fat,host,device,name,limit,tid,bid,block,grid,warp);
  if(name && (std::strstr(name,"optimize_multi_penalty_targets_kernel") ||
              std::strstr(name,"replay_fused_exact_optimizer_kernel"))) {
    auto& s=state();std::lock_guard<std::mutex> lock(s.names_mutex);s.names[host]=name;
  }
}
extern "C" cudaError_t cudaLaunchKernel(const void* function,dim3 grid,dim3 block,
    void** args,size_t shared,cudaStream_t stream) {
  using Launch=cudaError_t(*)(const void*,dim3,dim3,void**,size_t,cudaStream_t);
  static auto f=next<Launch>("cudaLaunchKernel");
  auto& s=state();std::string name;
  { std::lock_guard<std::mutex> lock(s.names_mutex);auto i=s.names.find(function);if(i!=s.names.end())name=i->second; }
  const bool multi=s.active.load() && name.find("optimize_multi_penalty_targets_kernel")!=std::string::npos;
  const bool single=name.find("replay_fused_exact_optimizer_kernel")!=std::string::npos && !s.single_seen.exchange(true);
  const uint64_t sequence=multi?++s.launches:0;
  if(multi) { request_launch=sequence;request_stream=stream; }
  const bool measured=single||(multi&&sequence<=64);
  if(measured)observe(single?"single_launch_enter":"l3_launch_enter",sequence,stream,name);
  const auto result=f(function,grid,block,args,shared,stream);
  if(multi)++s.returned;
  if(measured)observe(single?"single_launch_return":"l3_launch_return",sequence,stream,name,result);
  return result;
}

namespace fastkpc {
MultiPenaltyGcvCudaBatchResultsmall64 multi_penalty_gcv_cuda_optimize_batch_small64(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const double* y,
    int n,int targets,const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control) {
  static auto f=next<decltype(&multi_penalty_gcv_cuda_optimize_batch_small64)>(STACK_batch);
  const bool measured=state().active.load();
  if(measured) { request_key=keys.empty()?"":keys.front();request_targets=targets;request_number=++state().requests;request_launch=0;request_stream=nullptr; }
  try {
    auto result=f(p,y,n,targets,keys,control);
    if(measured&&request_launch&&request_launch<=64)observe("request_native_return",request_launch,request_stream);
    if(measured) { request_key.clear();request_targets=0;request_number=request_launch=0;request_stream=nullptr; }
    return result;
  } catch(...) {
    if(measured) { observe("request_native_exception",request_launch,request_stream);request_key.clear();request_targets=0;request_number=request_launch=0;request_stream=nullptr; }
    throw;
  }
}
}
namespace l3_probe {
void begin(bool slots) {
  static auto f=next<decltype(&begin)>("_ZN8l3_probe5beginEb");
  state().active=true;observe("before_l3_begin");f(slots);
}
void checkpoint(const char* name) {
  static auto f=next<decltype(&checkpoint)>("_ZN8l3_probe10checkpointEPKc");
  f(name);
  if(std::string(name)!="stage_a_cohort")observe(name);
}
void workers(bool enabled) {
  static auto f=next<decltype(&workers)>("_ZN8l3_probe7workersEb");f(enabled);
  observe(enabled?"workers_start":"optimizer_workers_joined");
}
void optimizer(const fastkpc::MultiPenaltyGcvCudaOptimization& result,
    const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>& residual) {
  static auto f=next<decltype(&optimizer)>(STACK_optimizer);f(result,residual);
  ++state().checked;state().targets+=result.target_count;
}
void finish() {
  static auto f=next<decltype(&finish)>("_ZN8l3_probe6finishEv");f();
  observe("after_l3_finish");state().active=false;
}
}
extern "C" SEXP fastkpc_l3_stack_finish() { BEGIN_RCPP
  observe("after_native_cleanup");state().active=false;
  const auto c=layer_audit::counts();
  return Rcpp::List::create(Rcpp::Named("resources_closed")=!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,
    Rcpp::Named("observation_errors")=double(state().errors.load()),
    Rcpp::Named("l3_launches")=double(state().launches.load()),
    Rcpp::Named("l3_requests")=double(state().requests.load()),
    Rcpp::Named("optimizer_setups_bitwise_exact")=double(state().checked.load()),
    Rcpp::Named("optimizer_targets_bitwise_exact")=double(state().targets.load()));
END_RCPP }
