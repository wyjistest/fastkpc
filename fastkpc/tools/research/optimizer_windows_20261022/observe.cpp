// Only diagnostic timing/NVTX; production numerical calls stay in the native.
#include "fixture.hpp"
#include "observe.hpp"
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include "symbols.inc"
namespace window_diag {
using Clock=std::chrono::steady_clock;
struct Row {
  cudaEvent_t start=nullptr,end=nullptr;
  cudaStream_t stream=nullptr;
  double host_start_ms=0,host_return_ms=0;
  long tid=0;
  bool entered=false,returned=false;
};
static bool enabled=false;
static std::vector<Row> rows;
static std::unordered_map<std::string,size_t> indices;
static cudaEvent_t reference=nullptr;
static Clock::time_point epoch;
thread_local Row* active_row=nullptr;
inline void check(cudaError_t x) {need(x==cudaSuccess,cudaGetErrorString(x));}
template<class F> F original(const char* name) {
  const char* p=std::getenv("FASTKPC_WINDOW_NATIVE");need(p,"native path missing");
  void* h=dlopen(p,RTLD_NOW|RTLD_NOLOAD);need(h,"native not loaded");
  auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);need(f,name);return f;
}
void observe_begin(const std::vector<std::string>& keys,bool value) {
  need(!enabled && rows.empty(),"overlapping observation");
  if (!value) return;
  indices.clear();rows.resize(keys.size());
  for (size_t i=0;i<keys.size();++i) {
    need(indices.emplace(keys[i],i).second,"duplicate setup identity");
    check(cudaEventCreate(&rows[i].start));check(cudaEventCreate(&rows[i].end));
  }
  check(cudaEventCreate(&reference));check(cudaEventRecord(reference));
  check(cudaEventSynchronize(reference));epoch=Clock::now();enabled=true;
}
void observe_end(const std::string& path) {
  if (!enabled) return;
  enabled=false;std::ofstream out(path);
  out<<"setup,thread_id,host_start_ms,host_return_ms,device_start_ms,device_end_ms,stream_span_ms\n"<<std::setprecision(17);
  for (size_t i=0;i<rows.size();++i) {
    auto& r=rows[i];need(r.entered && r.returned && r.stream,"missing setup observation");
    check(cudaEventSynchronize(r.end));float start=0,end=0,span=0;
    check(cudaEventElapsedTime(&start,reference,r.start));
    check(cudaEventElapsedTime(&end,reference,r.end));
    check(cudaEventElapsedTime(&span,r.start,r.end));
    out<<i<<','<<r.tid<<','<<r.host_start_ms<<','<<r.host_return_ms<<','<<start<<','<<end<<','<<span<<'\n';
    check(cudaEventDestroy(r.start));check(cudaEventDestroy(r.end));
  }
  check(cudaEventDestroy(reference));reference=nullptr;rows.clear();indices.clear();need(bool(out),"observation write");
}
template<class F,class P> auto call(F f,const P& p,const double* Y,int n,int targets,
    const std::vector<std::string>& keys,const fastkpc::MultiPenaltyGcvCudaOptimizerControl& control) {
  if (!enabled) return f(p,Y,n,targets,keys,control);
  const size_t index=indices.at(keys.at(0));Row& r=rows.at(index);
  need(!r.entered && !active_row,"duplicate/nested setup observation");
  r.entered=true;r.tid=syscall(SYS_gettid);
  r.host_start_ms=std::chrono::duration<double,std::milli>(Clock::now()-epoch).count();
  const std::string label="setup:"+std::to_string(index);nvtxRangePushA(label.c_str());
  active_row=&r;
  try {
    auto result=f(p,Y,n,targets,keys,control);
    r.host_return_ms=std::chrono::duration<double,std::milli>(Clock::now()-epoch).count();
    need(r.stream,"native batch made no stream upload");check(cudaEventRecord(r.end,r.stream));
    r.returned=true;active_row=nullptr;nvtxRangePop();return result;
  } catch (...) {active_row=nullptr;nvtxRangePop();throw;}
}
}
extern "C" cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t count,
    cudaMemcpyKind kind,cudaStream_t stream) {
  using namespace window_diag;
  static auto f=[] {auto p=reinterpret_cast<decltype(&cudaMemcpyAsync)>(dlsym(RTLD_NEXT,"cudaMemcpyAsync"));need(p,"cudaMemcpyAsync symbol");return p;}();
  if (active_row && !active_row->stream) {
    need(stream!=nullptr,"unexpected default producer stream");
    active_row->stream=stream;check(cudaEventRecord(active_row->start,stream));
  }
  return f(dst,src,count,kind,stream);
}
namespace fastkpc {
MultiPenaltyGcvCudaBatchResult multi_penalty_gcv_cuda_optimize_batch(
    const std::shared_ptr<MultiPenaltyGcvCudaPrepared>& p,const double* Y,int n,int targets,
    const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control) {
  static auto f=window_diag::original<decltype(&multi_penalty_gcv_cuda_optimize_batch)>(SYM_batch);
  return window_diag::call(f,p,Y,n,targets,keys,control);
}
MultiPenaltyGcvCudaBatchResultsmall64 multi_penalty_gcv_cuda_optimize_batch_small64(
    const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const double* Y,int n,int targets,
    const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control) {
  static auto f=window_diag::original<decltype(&multi_penalty_gcv_cuda_optimize_batch_small64)>(SYM_small);
  return window_diag::call(f,p,Y,n,targets,keys,control);
}
}
