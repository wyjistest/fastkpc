// Interpose host entry/status copy only. All CUDA work stays in the current native.
#include "../optimizer_windows_20261022/fixture.hpp"
#include "cuda/mgcv_multi_penalty_gcv_extended.hpp"
#include "cuda/mgcv_multi_penalty_gcv_l4.hpp"
#include "cuda/mgcv_multi_penalty_gcv_l5.hpp"
#include "state_layout.hpp"
#include "symbols.inc"
#include <cupti.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <iomanip>
#include <map>
#include <mutex>

namespace attribution {
using window_diag::need;
using namespace fastkpc;
using state_layout::DeviceOptimizerState;
struct Row {
  int level=0,index=0;
  std::vector<std::string> keys;
  window_diag::Snapshot expected;
  std::vector<DeviceOptimizerState> states;
  std::atomic<int> visits{0};
  uint64_t enter=0,returned=0,observed=0;
  long thread=0;
  uintptr_t prepared=0,stream=0;
  uintptr_t geometry=0;
  uint64_t bind_enter=0,bind_return=0;
  void* status=nullptr;
  size_t status_bytes=0;
  int downloads=0;
  bool pass=false;
  uint64_t external()const{return uint64_t(level)*1000000+index;}
};
struct Catalog {
  std::vector<std::unique_ptr<Row>> rows;
  std::map<std::string,Row*> lookup;
  std::string out;
};
Catalog* catalog=nullptr;
std::once_flag initialized;
thread_local Row* active=nullptr;
struct Binding{uintptr_t prepared=0,geometry=0;uint64_t enter=0,returned=0;};
thread_local Binding binding;
void check(CUptiResult x) {need(x==CUPTI_SUCCESS,"CUPTI observer call failed");}
uint64_t now(){uint64_t t=0;check(cuptiGetTimestamp(&t));return t;}
const char* env(const char* n){const char* p=std::getenv(n);need(p&&*p,std::string("missing ")+n);return p;}
template<class F> F original(const char* name) {
  void* h=dlopen(env("FASTKPC_L45_NATIVE"),RTLD_NOW|RTLD_NOLOAD);
  need(h,"native not loaded");auto f=reinterpret_cast<F>(dlsym(h,name));dlclose(h);
  need(f,"native entry missing");return f;
}
void finish() {
  if(!catalog)return;
  std::ofstream setups(catalog->out+"/setups.tsv"),targets(catalog->out+"/targets.tsv");
  setups<<"level\trequest_index\texternal_id\ttargets\tthread_id\tprepared\tstream\tgeometry_address\tbind_enter_ns\tbind_return_ns\tenter_ns\treturn_ns\tobserver_end_ns\tstatus_bytes\tpass\n";
  targets<<"level\trequest_index\ttarget_index\tkey\titerations\tscore_calls\tobjective_calls\tstep_halving\tnewton_trials\tsteepest_trials\tboundary_probes\tboundary_accepted\trank\troute\tinfo\toptimizer_status\tcomplete_evaluations\tscore_only_evaluations\tstable_svd_evaluations\tguarded_qr_evaluations\tselected_reuse\tselected_refresh\tpenalty_cycles\tsvd_cycles\tscore_cycles\tderivative_cycles";
  for(int k=0;k<8;++k)targets<<"\tdecomposition_"<<k;
  targets<<'\n';
  int valid=0,total=0;uint64_t bytes=0;bool request_order=true;
  for(int level:{4,5}){
    uintptr_t base=0;
    for(const auto& p:catalog->rows)if(p->level==level&&p->index==0)base=p->geometry;
    size_t stride=level==4?sizeof(MultiPenaltyGcvL4Geometry):sizeof(MultiPenaltyGcvL5Geometry);
    for(const auto& p:catalog->rows)if(p->level==level)
      request_order=request_order&&base&&p->geometry==base+stride*p->index;
  }
  for(const auto& p:catalog->rows){const auto& r=*p;
    setups<<r.level<<'\t'<<r.index<<'\t'<<r.external()<<'\t'<<r.keys.size()<<'\t'<<r.thread<<'\t'<<r.prepared<<'\t'<<r.stream
      <<'\t'<<r.geometry<<'\t'<<r.bind_enter<<'\t'<<r.bind_return<<'\t'<<r.enter<<'\t'<<r.returned<<'\t'<<r.observed<<'\t'<<r.status_bytes<<'\t'<<r.pass<<'\n';
    if(r.pass)++valid;bytes+=r.status_bytes;
    for(size_t t=0;t<r.states.size();++t){const auto& s=r.states[t];const auto& v=s.current;const auto& q=s.phase_timing;
      targets<<r.level<<'\t'<<r.index<<'\t'<<t<<'\t'<<r.keys[t]<<'\t'<<s.optimizer_iterations<<'\t'<<s.score_calls
        <<'\t'<<s.objective_calls<<'\t'<<s.step_halving_count<<'\t'<<s.newton_trial_count<<'\t'<<s.steepest_descent_trial_count
        <<'\t'<<s.boundary_probe_count<<'\t'<<s.boundary_accepted_count<<'\t'<<v.numerical_rank<<'\t'<<v.decomposition_route
        <<'\t'<<v.solver_info<<'\t'<<s.optimizer_status<<'\t'<<q.complete_evaluation_count<<'\t'<<q.score_only_evaluation_count
        <<'\t'<<q.stable_svd_evaluation_count<<'\t'<<q.guarded_qr_evaluation_count<<'\t'<<s.selected_evaluation_reuse_count
        <<'\t'<<s.selected_evaluation_refresh_count<<'\t'<<q.penalty_factor_augmentation_cycles<<'\t'<<q.qr_svd_cycles
        <<'\t'<<q.score_construction_cycles<<'\t'<<q.derivative_hessian_cycles;
      for(auto cycles:q.decomposition_stage_cycles)targets<<'\t'<<cycles;
      targets<<'\n';++total;
    }
  }
  std::ofstream summary(catalog->out+"/observer.json");
  summary<<"{\"complete\":"<<(valid==3109&&total==24583?"true":"false")
    <<",\"passed_setups\":"<<valid<<",\"targets\":"<<total<<",\"status_struct_bytes\":"<<sizeof(DeviceOptimizerState)
    <<",\"canonical_request_order_exact\":"<<(request_order?"true":"false")
    <<",\"geometry_stride_bytes\":"<<sizeof(MultiPenaltyGcvL4Geometry)
    <<",\"existing_status_bytes_observed\":"<<bytes<<",\"extra_device_copies\":0,\"extra_device_allocations\":0,\"extra_device_waits\":0}\n";
  if(!setups||!targets||!summary)std::abort();
}
Catalog& get(){std::call_once(initialized,[]{
  auto* c=new Catalog;c->out=env("FASTKPC_L45_OUTPUT");
  std::ifstream in(env("FASTKPC_L45_REFERENCE"),std::ios::binary);window_diag::IO io{&in,nullptr};
  std::string schema;io.value(schema);need(schema=="l45-optimizer-host-reference-v1","reference schema");
  int count=0;io.value(count);need(count==3109,"reference coverage");
  for(int i=0;i<count;++i){auto p=std::make_unique<Row>();io.value(p->level);io.value(p->index);io.value(p->keys);p->expected.io(io);
    need(c->lookup.emplace(p->keys.at(0),p.get()).second,"duplicate reference key");c->rows.push_back(std::move(p));}
  need(in.peek()==EOF,"reference trailing bytes");catalog=c;std::atexit(finish);
});return *catalog;}

void verify(Row& r,const MultiPenaltyGcvCudaOptimization& a){
  const int n=a.target_count,p=a.penalty_count,q=a.coefficient_dim;
  need(n==int(r.keys.size())&&p==r.level&&q==1+9*r.level,"observed shape mismatch");
  need(r.downloads==1&&r.status&&r.status_bytes==sizeof(DeviceOptimizerState)*n,"status layout/transfer mismatch");
  // Copy only from the host buffer after the native's own completion wait.
  r.states.resize(n);std::memcpy(r.states.data(),r.status,r.status_bytes);
  window_diag::Snapshot s;
  s.floating={a.initial_log_sp,a.selected_log_sp,a.rss,a.edf,a.score,a.condition,a.gradient,a.hessian,a.coefficients,a.rms_gradient,a.hessian_eigenvalues};
  s.integral={a.aggregate_penalty_rank,a.numerical_rank,a.solver_info,a.optimizer_iterations,a.score_calls,a.objective_calls,
    a.step_halving_count,a.newton_trial_count,a.steepest_descent_trial_count,a.boundary_probe_count,a.boundary_accepted_count,
    a.boundary_status,a.fully_converged,a.hessian_positive_definite,a.step_failed,a.optimizer_status};
  need(s.same(r.expected),"captured optimizer output mismatch L"+std::to_string(r.level)+" index "+std::to_string(r.index));
  const auto& d=a.diagnostics;
  need(d.exact_primary_only&&d.cuda_exact_primary_target_count==n&&!d.cpu_objective_count&&!d.cpu_optimizer_count&&
    !d.fallback_count&&!d.cuda_error_count&&!d.cuda_stability_replay_target_count&&!d.cuda_terminal_boundary_confirmation_count,
    "exact-primary/authority failed");
  uint64_t penalty=0,svd=0,score=0,deriv=0,stages[8]{};int complete=0,only=0,stable=0,qr=0,reuse=0,refresh=0;
  auto same=[](const double* x,const double* y,size_t count){return !std::memcmp(x,y,count*sizeof(double));};
  for(int t=0;t<n;++t){const auto& v=r.states[t];const auto& f=v.current;const auto& c=v.phase_timing;
    need(v.penalty_count==p&&v.optimizer_iterations==a.optimizer_iterations[t]&&v.score_calls==a.score_calls[t]&&
      v.objective_calls==a.objective_calls[t]&&v.step_halving_count==a.step_halving_count[t]&&
      v.newton_trial_count==a.newton_trial_count[t]&&v.steepest_descent_trial_count==a.steepest_descent_trial_count[t]&&
      v.boundary_probe_count==a.boundary_probe_count[t]&&v.boundary_accepted_count==a.boundary_accepted_count[t]&&
      f.numerical_rank==a.numerical_rank[t]&&f.aggregate_penalty_rank==a.aggregate_penalty_rank[t]&&
      f.solver_info==0&&v.optimizer_status==0&&f.decomposition_route==2&&!v.stability_replay_attempted&&
      !v.terminal_boundary_confirmation_count,"per-target status ABI check");
    need(same(&f.rss,&a.rss[t],1)&&same(&f.edf,&a.edf[t],1)&&same(&f.score,&a.score[t],1)&&same(&f.condition,&a.condition[t],1)&&
      same(v.log_sp,a.selected_log_sp.data()+t*p,p)&&same(f.gradient,a.gradient.data()+t*p,p)&&
      same(f.hessian,a.hessian.data()+t*p*p,p*p)&&same(f.coefficients,a.coefficients.data()+t*q,q),"per-target floating ABI check");
    penalty+=c.penalty_factor_augmentation_cycles;svd+=c.qr_svd_cycles;score+=c.score_construction_cycles;deriv+=c.derivative_hessian_cycles;
    for(int k=0;k<8;++k)stages[k]+=c.decomposition_stage_cycles[k];
    complete+=c.complete_evaluation_count;only+=c.score_only_evaluation_count;stable+=c.stable_svd_evaluation_count;qr+=c.guarded_qr_evaluation_count;
    reuse+=v.selected_evaluation_reuse_count;refresh+=v.selected_evaluation_refresh_count;
  }
  need(penalty==d.cuda_penalty_factor_augmentation_cycles&&svd==d.cuda_qr_svd_cycles&&score==d.cuda_score_construction_cycles&&
    deriv==d.cuda_derivative_hessian_cycles&&complete==d.cuda_complete_evaluation_count&&only==d.cuda_score_only_evaluation_count&&
    stable==d.cuda_stable_svd_evaluation_count&&qr==d.cuda_guarded_qr_evaluation_count&&reuse==d.cuda_selected_evaluation_reuse_count&&
    refresh==d.cuda_selected_evaluation_refresh_count,"aggregate phase/counter ABI check");
  const uint64_t expected[]={d.cuda_qr_bidiagonal_reduction_cycles,d.cuda_bidiagonal_svd_cycles,d.cuda_svd_vector_postback_cycles,
    d.cuda_left_vector_product_cycles,d.cuda_qr_factorization_cycles,d.cuda_q_generation_cycles,d.cuda_qr_guard_cycles,d.cuda_stable_bidiagonal_reduction_cycles};
  for(int k=0;k<8;++k)need(stages[k]==expected[k],"decomposition phase ABI check");
}
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream){
  // R loads the native and its CUDA dependencies with local visibility. RTLD_NEXT
  // alone sees the standalone smoke executable's dependencies but not R's scope.
  static auto f=[](){
    auto p=reinterpret_cast<decltype(&cudaMemcpyAsync)>(dlsym(RTLD_NEXT,"cudaMemcpyAsync"));
    return p?p:attribution::original<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
  }();
  auto* r=attribution::active;
  if(r&&kind==cudaMemcpyDeviceToHost){r->status=dst;r->status_bytes=bytes;r->stream=reinterpret_cast<uintptr_t>(stream);++r->downloads;}
  return f(dst,src,bytes,kind,stream);
}

namespace fastkpc {
void rebind_multi_penalty_gcv_cuda_prepared_l4(
 const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const MultiPenaltyGcvL4Geometry& g){
  using namespace attribution;
  static auto f=original<decltype(&rebind_multi_penalty_gcv_cuda_prepared_l4)>(NATIVE_BIND4_SYMBOL);
  binding={reinterpret_cast<uintptr_t>(p.get()),reinterpret_cast<uintptr_t>(&g),now(),0};f(p,g);binding.returned=now();
}
void rebind_multi_penalty_gcv_cuda_prepared_l5(
 const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& p,const MultiPenaltyGcvL5Geometry& g){
  using namespace attribution;
  static auto f=original<decltype(&rebind_multi_penalty_gcv_cuda_prepared_l5)>(NATIVE_BIND5_SYMBOL);
  binding={reinterpret_cast<uintptr_t>(p.get()),reinterpret_cast<uintptr_t>(&g),now(),0};f(p,g);binding.returned=now();
}
MultiPenaltyGcvCudaBatchResultsmall64 multi_penalty_gcv_cuda_optimize_batch_small64(
 const std::shared_ptr<MultiPenaltyGcvCudaPreparedsmall64>& prepared,const double* Y,int n,int count,
 const std::vector<std::string>& keys,const MultiPenaltyGcvCudaOptimizerControl& control){
  using namespace attribution;
  static auto f=original<decltype(&multi_penalty_gcv_cuda_optimize_batch_small64)>(NATIVE_BATCH_SYMBOL);
  auto& c=get();const auto found=c.lookup.find(keys.at(0));
  if(found==c.lookup.end())return f(prepared,Y,n,count,keys,control);
  Row& r=*found->second;
  need(r.visits.fetch_add(1)==0&&!active&&r.keys==keys&&n==351&&count==int(keys.size()),"duplicate/invalid observer identity");
  need(control.exact_primary_only&&control.force_final_selected_evaluation&&!control.decomposition_trace_capacity_per_target,"observer control mismatch");
  r.thread=syscall(SYS_gettid);r.prepared=reinterpret_cast<uintptr_t>(prepared.get());r.enter=now();active=&r;
  if(binding.prepared==r.prepared){r.geometry=binding.geometry;r.bind_enter=binding.enter;r.bind_return=binding.returned;}
  binding={};
  check(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,r.external()));
  bool pushed=true;
  try{
    auto result=f(prepared,Y,n,count,keys,control);r.returned=now();active=nullptr;
    uint64_t id=0;check(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&id));pushed=false;need(id==r.external(),"correlation stack");
    verify(r,result.optimization);r.pass=true;r.observed=now();return result;
  }catch(...){active=nullptr;if(pushed){uint64_t id=0;cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&id);}throw;}
}
}
