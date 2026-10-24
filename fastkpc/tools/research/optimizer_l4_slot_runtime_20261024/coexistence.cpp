// Diagnostic only: unchanged native producer plus an extra, conservative pool.
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "skeleton_task_scheduler.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include "symbols.inc"
using namespace fastkpc;
using window_diag::need;
namespace {
std::atomic<bool> sampling{false};
std::atomic<std::size_t> minimum{SIZE_MAX}, samples{0};
int level=-1, windows=0, setups=0, targets=0, stored=0, warmed=0;
std::size_t before=0, after_create=0, after_warm=0;
std::thread observer;
std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> pool;
std::vector<cudaEvent_t> events;
double* extra_arena=nullptr;
const char* env(const char* key){const char* p=std::getenv(key);need(p&&*p,key);return p;}
void check(cudaError_t s){need(s==cudaSuccess,cudaGetErrorString(s));}
template<class F> F original(const char* name){
  void* h=dlopen(env("FASTKPC_WINDOW_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"production native not loaded");auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);
  need(f,name);return f;
}
std::size_t observe(){
  std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  if(sampling.load()){
    auto old=minimum.load();while(old>free&&!minimum.compare_exchange_weak(old,free)){}
    ++samples;
  }
  return free;
}
void cleanup(){
  sampling=false;if(observer.joinable())observer.join();
  pool.clear();for(auto e:events)if(e)cudaEventDestroy(e);events.clear();
  if(extra_arena){cudaFree(extra_arena);extra_arena=nullptr;}
}
void start(){
  // Capacity 18 first occurs after W1. Search the original windows without
  // padding a smaller cohort or retaining the entire capture in host memory.
  window_diag::Request warm;
  for(int w=1;w<=34&&warm.keys.empty();++w){
    const auto fixture=l4_diag::Window::read(std::string(env("FASTKPC_L4_PROBE_FIXTURES"))+"/L4-W"+std::to_string(w)+".bin");
    const auto found=std::find_if(fixture.requests.begin(),fixture.requests.end(),
      [](const auto& x){return x.keys.size()==18;});
    if(found!=fixture.requests.end())warm=*found;
  }
  need(warm.keys.size()==18,"18-target warm request absent");const auto* r=&warm;
  const auto& g=r->geometry;
  before=observe();sampling=true;
  try{
    observer=std::thread([]{
      cudaSetDevice(0);
      while(sampling.load()){observe();std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    });
    for(int i=0;i<32;++i){
      pool.push_back(create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),
        g.R.data(),g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),351,37,4,18,0));
      events.push_back(nullptr);
      check(cudaEventCreateWithFlags(&events.back(),cudaEventDisableTiming|cudaEventBlockingSync));
    }
    check(cudaMalloc(reinterpret_cast<void**>(&extra_arena),37ULL*17767*8));
    after_create=observe();
    for(auto& p:pool){
      auto result=multi_penalty_gcv_capacity_optimize_batch_with_residual(
        p,r->Y.data(),351,18,r->keys,r->control);
      const bool same=window_diag::snapshot(result.optimization,result.residual).same(r->expected);
      release_multi_penalty_gcv_capacity_residual(result.residual);
      need(same,"extra native slot warm output mismatch");++warmed;
    }
    after_warm=observe();
    std::cerr<<"L4 extra pool ready free="<<after_warm<<" warm_increment="
      <<(before-std::min(after_create,after_warm))<<std::endl;
  }catch(...){cleanup();throw;}
}
void finish(){
  observe();cleanup();check(cudaDeviceSynchronize());
  const std::size_t increment=before-std::min(before,std::min(after_create,after_warm));
  const bool coverage=windows==34&&setups==2154&&targets==17767&&stored==17767&&warmed==32;
  const bool pass=coverage&&minimum.load()>=2ULL*1024*1024*1024&&increment<=1024ULL*1024*1024;
  std::ofstream out(std::string(env("FASTKPC_L4_PROBE_OUT"))+"/coexistence-summary.json");
  out<<"{\"status\":\""<<(pass?"PASS":"CONSERVATIVE_PROBE_FAILED")
    <<"\",\"windows\":"<<windows<<",\"setups\":"<<setups<<",\"targets\":"<<targets
    <<",\"stage_a_targets\":"<<stored<<",\"extra_slots\":32,\"warm_exact_targets\":"<<warmed*18
    <<",\"extra_coefficient_bytes\":5259032,\"free_before\":"<<before
    <<",\"free_after_create\":"<<after_create<<",\"free_after_warm\":"<<after_warm
    <<",\"warm_increment_bytes\":"<<increment<<",\"minimum_free_bytes\":"<<minimum.load()
    <<",\"observations\":"<<samples.load()<<",\"extra_resources_closed\":true"
    <<",\"baseline_window_handles_also_resident\":true,\"coverage_pass\":"<<(coverage?"true":"false")<<"}\n";
  need(bool(out),"coexistence receipt failed");out.close();
  std::cerr<<"L4 coexistence "<<(pass?"PASS":"FAIL")<<" minimum_free="<<minimum.load()<<std::endl;
  // A conservative failure is recorded; allow baseline closure and full parity audit.
}
}
extern "C" cudaError_t cudaMalloc(void** p,std::size_t bytes){
  static const auto f=reinterpret_cast<cudaError_t(*)(void**,std::size_t)>(dlsym(RTLD_NEXT,"cudaMalloc"));
  if(!f)return cudaErrorUnknown;
  const auto s=f(p,bytes);if(s==cudaSuccess&&sampling.load())observe();return s;
}
LayerPlan make_layer_plan(const std::vector<int>& a,int p,int l){
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);
  level=l;return f(a,p,l);
}
namespace fastkpc {
MultiPenaltyGcvCapacityMultiResult multi_penalty_gcv_capacity_optimize_multi(
    std::vector<MultiPenaltyGcvCapacityRequest> requests,int concurrency,bool retain){
  static auto f=original<decltype(&multi_penalty_gcv_capacity_optimize_multi)>(SYM_multi);
  if(level==4){
    need(retain&&concurrency==32,"production L4 controls changed");
    if(windows==0)start();++windows;setups+=requests.size();
    for(const auto& r:requests)targets+=r.target_count;observe();
  }
  try{auto result=f(std::move(requests),concurrency,retain);if(level==4)observe();return result;}
  catch(...){cleanup();throw;}
}
void store_full_cuda_ci_level_component_residual(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache,
    const MultiPenaltyGcvCudaResidualConsumerView& residual){
  static auto f=original<decltype(&store_full_cuda_ci_level_component_residual)>(SYM_store);
  if(level==4)observe();f(cache,residual);
  if(level==4){stored+=residual.target_count;observe();}
}
void build_full_cuda_ci_level_components(const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache){
  static auto f=original<decltype(&build_full_cuda_ci_level_components)>(SYM_build);
  if(level==4)finish();f(cache);
}
}
