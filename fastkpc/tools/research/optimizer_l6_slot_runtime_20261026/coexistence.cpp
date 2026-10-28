// Real cache/arena residency, with original optimizer window handles released.
#include "../optimizer_l6_slots_20261025/slot_api.hpp"
#include "../optimizer_l6_slots_20261025/fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include "skeleton_task_scheduler.hpp"
#include "cuda/full_cuda_ci_vertical.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include "symbols.inc"
using namespace fastkpc;
using namespace window_diag;
namespace {
std::atomic<bool> sampling{false};
std::atomic<std::size_t> minimum{SIZE_MAX},samples{0};
int level=-1;
bool visited=false;
std::vector<std::weak_ptr<MultiPenaltyGcvCapacityPrepared>> window_handles;
const char* env(const char* key){const char* p=std::getenv(key);need(p&&*p,key);return p;}
void check(cudaError_t s){need(s==cudaSuccess,cudaGetErrorString(s));}
template<class F> F original(const char* name){
  void* h=dlopen(env("FASTKPC_WINDOW_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"production native not loaded");auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);need(f,name);return f;
}
std::size_t observe(){
  std::size_t free=0,total=0;
  if(cudaMemGetInfo(&free,&total)!=cudaSuccess)free=0;
  if(sampling.load()){auto old=minimum.load();while(old>free&&!minimum.compare_exchange_weak(old,free)){};++samples;}
  return free;
}
Snapshot saved_snapshot(const l6_slot::Result& r,const std::vector<double>& saved,size_t offset){
  const auto& a=r.optimization;const auto& d=a.diagnostics;const auto& info=r.info;
  need(d.exact_primary_only&&d.cuda_exact_primary_target_count==a.target_count&&!d.cpu_objective_count&&
    !d.cpu_optimizer_count&&!d.cpu_multi_penalty_solve_count&&!d.fallback_count&&!d.cuda_error_count&&
    !d.cuda_stability_replay_target_count&&!d.cuda_terminal_boundary_confirmation_count,"GPU authority");
  for(int i=0;i<a.target_count;++i)need(info.solver_info.at(i)==0&&info.optimizer_status.at(i)==0&&
    info.decomposition_route.at(i)==kMultiPenaltyGcvDecompositionStableSvd,"stable-SVD status");
  auto take=[&](size_t n){need(offset+n<=saved.size(),"arena bound");std::vector<double> v(saved.begin()+offset,saved.begin()+offset+n);offset+=n;return v;};
  const size_t q_count=size_t(55)*a.target_count,n_count=size_t(351)*a.target_count;
  auto coefficients=take(q_count),fitted=take(n_count),residual=take(n_count);Snapshot s;
  s.floating={a.initial_log_sp,a.selected_log_sp,a.rss,a.edf,a.score,a.condition,a.gradient,a.hessian,a.coefficients,
    a.rms_gradient,a.hessian_eigenvalues,std::move(coefficients),std::move(fitted),std::move(residual)};
  s.integral={a.aggregate_penalty_rank,a.numerical_rank,a.solver_info,a.optimizer_iterations,a.score_calls,
    a.objective_calls,a.step_halving_count,a.newton_trial_count,a.steepest_descent_trial_count,
    a.boundary_probe_count,a.boundary_accepted_count,a.boundary_status,a.fully_converged,
    a.hessian_positive_definite,a.step_failed,a.optimizer_status,info.solver_info,info.decomposition_route,info.numerical_rank};
  return s;
}

bool same_resources(const layer_audit::Counts& a,const layer_audit::Counts& b){
  return a.allocations==b.allocations&&a.bytes==b.bytes&&a.streams==b.streams&&
    a.events==b.events&&a.blas==b.blas&&a.forbidden_calls==b.forbidden_calls;
}
void replay_with_live_cache(){
  need(!visited&&window_handles.size()==245,"original L6 window coverage");
  visited=true;
  for(const auto& p:window_handles)need(p.expired(),"original L6 prepared handle still resident");
  const auto resources_before=layer_audit::counts();
  auto windows=l6_diag::read_layer(env("FASTKPC_L6_PROBE_FIXTURES"));
  std::vector<const Request*> all;for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
  std::vector<l6_slot::Packed> packets;std::vector<std::size_t> offsets;std::size_t values=0;
  for(auto r:all){packets.push_back(l6_slot::pack(*r));offsets.push_back(values);values+=(55+702)*r->keys.size();}
  need(values*8==6952288,"full diagnostic arena shape changed");
  const std::size_t before=observe();need(before>0,"memory query failed");minimum=before;sampling=true;
  std::thread monitor([]{cudaSetDevice(0);while(sampling.load()){
    observe();std::this_thread::sleep_for(std::chrono::milliseconds(10));}});
  std::vector<l6_slot::Handle> pool;std::vector<l6_slot::Result> results(all.size());
  double* arena=nullptr;
  auto cleanup=[&]{
    std::exception_ptr error;
    try{layer_audit::native_scope(false);results.clear();pool.clear();
      if(arena){check(cudaFree(arena));arena=nullptr;}
      check(cudaDeviceSynchronize());observe();
    }catch(...){error=std::current_exception();}
    sampling=false;if(monitor.joinable())monitor.join();
    if(error)std::rethrow_exception(error);
  };
  try{
    for(int i=0;i<32;++i)pool.push_back(l6_slot::create(packets.front(),5));
    check(cudaMalloc(reinterpret_cast<void**>(&arena),values*8));const std::size_t after_create=observe();
    std::atomic<std::size_t> next{0};std::atomic<bool> cancel{false};std::mutex error_mutex;std::exception_ptr error;
    std::vector<std::thread> workers;
    auto worker=[&](int slot){
      l6_slot::Result pending;
      try{check(cudaSetDevice(0));while(!cancel.load()){
        const auto i=next.fetch_add(1);if(i>=all.size())break;
        l6_slot::bind(pool[slot],packets[i],i+1);pending=l6_slot::execute(pool[slot],*all[i]);
        need(pending.info.target_keys==all[i]->keys&&l6_slot::token_live(pending),"slot identity");
        const auto targets=all[i]->keys.size(),offset=offsets[i];
        l6_slot::detach(pool[slot],pending,arena+offset,arena+offset+55*targets,arena+offset+(55+351)*targets);
        need(l6_slot::token_rejected(pending),"released token remained live");results[i]=std::move(pending);
      }}catch(...){cancel=true;const auto caught=std::current_exception();
        try{l6_slot::abandon(pool[slot],pending);}catch(...){}
        std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=caught;}
    };
    layer_audit::native_scope(true);
    try{for(int i=0;i<32;++i)workers.emplace_back(worker,i);}
    catch(...){cancel=true;for(auto& t:workers)t.join();throw;}
    for(auto& t:workers)t.join();for(auto& p:pool)l6_slot::drain(p);
    layer_audit::native_scope(false);
    if(error)std::rethrow_exception(error);const std::size_t after_work=observe();
    std::vector<double> saved(values);check(cudaMemcpy(saved.data(),arena,values*8,cudaMemcpyDeviceToHost));
    std::size_t targets=0,iterations=0;
    for(std::size_t i=0;i<all.size();++i){
      need(saved_snapshot(results[i],saved,offsets[i]).same(all[i]->expected),"captured output mismatch");
      targets+=all[i]->keys.size();iterations+=std::accumulate(results[i].optimization.optimizer_iterations.begin(),
        results[i].optimization.optimizer_iterations.end(),std::size_t(0));
    }
    observe();
    for(auto& p:pool)need(!l6_slot::info(p).residual_slot_leased,"live lease at close");
    cleanup();const std::size_t after_close=observe();
    const std::size_t min=minimum.load(),increment=before-std::min(before,min);
    need(same_resources(resources_before,layer_audit::counts()),"probe resources or worker mutations changed");
    const bool pass=min>=2ULL*1024*1024*1024&&increment<=1024ULL*1024*1024&&targets==1148&&iterations==17062;
    std::ofstream out(std::string(env("FASTKPC_L6_PROBE_OUT"))+"/coexistence-summary.json");
    out<<"{\"status\":\""<<(pass?"PASS":"STOP_REPLACEMENT_CAPACITY")<<"\",\"setups\":"<<all.size()
       <<",\"targets\":"<<targets<<",\"iterations\":"<<iterations<<",\"pool_size\":32,\"target_capacity\":5"
       <<",\"diagnostic_arena_bytes\":"<<values*8<<",\"minimum_free_bytes\":"<<min
       <<",\"warm_increment_bytes\":"<<increment<<",\"free_before\":"<<before
       <<",\"free_after_create\":"<<after_create<<",\"free_after_work\":"<<after_work
       <<",\"free_after_close\":"<<after_close<<",\"observations\":"<<samples.load()
       <<",\"outputs_bitwise_exact\":true,\"all_leases_released\":true,\"original_window_handles_present\":false"
       <<",\"expired_window_handles\":"<<window_handles.size()
       <<",\"resources_restored\":true,\"worker_resource_mutations\":0}\n";
    need(bool(out),"replacement receipt failed");
    std::cerr<<"L6 replacement coexistence "<<(pass?"PASS":"STOP")<<" minimum_free="<<min
             <<" pool_increment="<<increment<<std::endl;
  }catch(...){const auto error=std::current_exception();try{cleanup();}catch(...){}std::rethrow_exception(error);}
}
}
extern "C" cudaError_t cudaMalloc(void** p,std::size_t bytes){
  static const auto f=reinterpret_cast<cudaError_t(*)(void**,std::size_t)>(dlsym(RTLD_NEXT,"cudaMalloc"));
  if(!f)return cudaErrorUnknown;const auto s=f(p,bytes);if(s==cudaSuccess&&sampling.load())observe();return s;
}
extern "C" cudaError_t cudaFree(void* p){
  static const auto f=reinterpret_cast<decltype(&cudaFree)>(dlsym(RTLD_NEXT,"cudaFree"));
  if(!f)return cudaErrorUnknown;if(sampling.load())observe();
  const auto s=f(p);if(s==cudaSuccess&&sampling.load())observe();return s;
}
LayerPlan make_layer_plan(const std::vector<int>& a,int p,int l){
  static auto f=original<decltype(&make_layer_plan)>(SYM_plan);level=l;return f(a,p,l);
}
namespace fastkpc {
std::shared_ptr<MultiPenaltyGcvCapacityPrepared> create_multi_penalty_gcv_capacity_prepared(
    const double* X,const double* qr,const double* tau,const double* R,const int* pivot,
    const std::vector<std::vector<double>>& roots,const std::vector<std::vector<double>>& matrices,
    const std::vector<int>& ranks,const double* sp,int n,int q,int penalties,int capacity,int device){
  static auto f=original<decltype(&create_multi_penalty_gcv_capacity_prepared)>(SYM_create);
  auto p=f(X,qr,tau,R,pivot,roots,matrices,ranks,sp,n,q,penalties,capacity,device);
  if(level==6&&!visited){need(n==351&&q==55&&penalties==6,"original L6 geometry");window_handles.push_back(p);}
  return p;
}
void build_full_cuda_ci_level_components(const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache){
  static auto f=original<decltype(&build_full_cuda_ci_level_components)>(SYM_build);
  if(level==6)replay_with_live_cache();f(cache);
}
}

