#include "cuda/mgcv_multi_penalty_gcv_postback.hpp"
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/layer_fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
using namespace fastkpc;
using namespace window_diag;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point s){return std::chrono::duration<double>(Clock::now()-s).count();}
void check(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
void closed(){const auto c=layer_audit::counts();need(!c.allocations&&!c.bytes&&!c.events&&!c.streams&&!c.blas,"resource closure");}
template<int Level> using Packet=std::conditional_t<Level==4,MultiPenaltyGcvL4Geometry,MultiPenaltyGcvL5Geometry>;
template<int Level> Packet<Level> pack(const Request& r){
  const auto& a=r.geometry;Packet<Level> p;p.X=a.X;p.qr=a.qr;p.tau=a.tau;p.r=a.R;
  p.pivot=a.pivot;p.ranks=a.ranks;p.initial_log_sp=a.initial;
  int offset=0;for(int i=0;i<Level;++i){p.offsets.push_back(offset);offset+=a.ranks[i];
    p.roots.insert(p.roots.end(),a.roots[i].begin(),a.roots[i].end());
    p.matrices.insert(p.matrices.end(),a.matrices[i].begin(),a.matrices[i].end());}
  if constexpr(Level==4)validate_multi_penalty_gcv_l4_geometry(p);else validate_multi_penalty_gcv_l5_geometry(p);
  return p;
}
template<int Level> auto create(const Packet<Level>& p){
  if constexpr(Level==4)return create_multi_penalty_gcv_capacity_l4_slot(p,0);
  else return create_multi_penalty_gcv_capacity_l5_slot(p,0);
}
template<int Level> void bind(const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& s,const Packet<Level>& p){
  if constexpr(Level==4)rebind_multi_penalty_gcv_capacity_l4_slot(s,p);else rebind_multi_penalty_gcv_capacity_l5_slot(s,p);
}
template<int Level> void drain(const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& s){
  if constexpr(Level==4)drain_multi_penalty_gcv_capacity_l4_slot(s);else drain_multi_penalty_gcv_capacity_l5_slot(s);
}
Snapshot state_snapshot(const MultiPenaltyGcvCudaOptimization& a,const MultiPenaltyGcvCudaResidualInfo& info){
  const auto& d=a.diagnostics;
  need(d.exact_primary_only&&d.cuda_exact_primary_target_count==a.target_count&&!d.cpu_objective_count&&
    !d.cpu_optimizer_count&&!d.cpu_multi_penalty_solve_count&&!d.fallback_count&&!d.cuda_error_count&&
    !d.cuda_stability_replay_target_count&&!d.cuda_terminal_boundary_confirmation_count,"GPU authority");
  for(int i=0;i<a.target_count;++i)need(info.solver_info.at(i)==0&&info.optimizer_status.at(i)==0&&
    info.decomposition_route.at(i)==kMultiPenaltyGcvDecompositionStableSvd,"stable-SVD info/route");
  Snapshot s;
  s.floating={a.initial_log_sp,a.selected_log_sp,a.rss,a.edf,a.score,a.condition,a.gradient,a.hessian,
    a.coefficients,a.rms_gradient,a.hessian_eigenvalues};
  s.integral={a.aggregate_penalty_rank,a.numerical_rank,a.solver_info,a.optimizer_iterations,a.score_calls,
    a.objective_calls,a.step_halving_count,a.newton_trial_count,a.steepest_descent_trial_count,
    a.boundary_probe_count,a.boundary_accepted_count,a.boundary_status,a.fully_converged,
    a.hessian_positive_definite,a.step_failed,a.optimizer_status,info.solver_info,info.decomposition_route,info.numerical_rank};
  return s;
}
template<int Level,class Windows> void layer(const Windows& windows,const std::string& mode,const std::string& prefix){
  constexpr int q=Level==4?37:46;
  const int backend=multi_penalty_gcv_vector_postback_requested();
  const auto counts_before=multi_penalty_gcv_postback_execution_counts();
  std::vector<const Request*> all;for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
  const bool reduced=mode=="reduced",verify=mode!="timing";
  std::vector<std::size_t> indices(all.size());std::iota(indices.begin(),indices.end(),0);
  if(reduced){std::map<int,std::size_t> first;for(std::size_t i=0;i<all.size();++i)first.emplace(all[i]->keys.size(),i);
    indices.clear();for(auto x:first)indices.push_back(x.second);indices.push_back(all.size()-1);
    std::sort(indices.begin(),indices.end());indices.erase(std::unique(indices.begin(),indices.end()),indices.end());}
  closed();std::size_t free_before,total;check(cudaMemGetInfo(&free_before,&total));
  need(free_before>2ULL*1024*1024*1024,"2 GiB headroom before layer");
  auto start=Clock::now();std::vector<Packet<Level>> packets;
  std::vector<std::size_t> offsets;std::size_t values=0,targets=0;
  for(auto i:indices){packets.push_back(pack<Level>(*all[i]));offsets.push_back(values);
    values+=(q+351)*all[i]->keys.size();targets+=all[i]->keys.size();}
  const double packing=elapsed(start);start=Clock::now();
  std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> slots;
  std::vector<cudaEvent_t> events;
  double* arena=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&arena),values*8));
  for(int k=0;k<32;++k){slots.push_back(create<Level>(packets.front()));cudaEvent_t e;
    check(cudaEventCreateWithFlags(&e,cudaEventBlockingSync|cudaEventDisableTiming));events.push_back(e);}
  const double preparation=elapsed(start);
  std::vector<MultiPenaltyGcvCudaOptimization> outputs(indices.size());
  std::vector<MultiPenaltyGcvCudaResidualInfo> infos(indices.size());
  std::vector<int> assignment(indices.size(),-1);std::atomic<std::size_t> next{0};
  std::atomic<bool> cancel{false};std::mutex mutex;std::exception_ptr error;
  std::size_t passed=0;std::atomic<std::size_t> verified{0};
  const auto mutations_before=layer_audit::counts().forbidden_calls;
  auto worker=[&](int worker){try{
    check(cudaSetDevice(0));while(!cancel){std::size_t j=next.fetch_add(1);if(j>=indices.size())break;
      assignment[j]=worker;const auto& r=*all[indices[j]];bind<Level>(slots[worker],packets[j]);
      auto result=multi_penalty_gcv_capacity_optimize_batch_with_residual(slots[worker],r.Y.data(),351,r.keys.size(),r.keys,r.control);
      try{
        const auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(result.residual);
        need(view.target_keys==r.keys,"target order");infos[j]=multi_penalty_gcv_capacity_residual_info(result.residual);
        check(cudaMemcpyAsync(arena+offsets[j],view.coefficients,q*r.keys.size()*8,cudaMemcpyDeviceToDevice,view.producer_stream));
        check(cudaMemcpyAsync(arena+offsets[j]+q*r.keys.size(),view.residuals,351*r.keys.size()*8,cudaMemcpyDeviceToDevice,view.producer_stream));
        check(cudaEventRecord(events[worker],view.producer_stream));check(cudaEventSynchronize(events[worker]));
        if(verify){need(snapshot(result.optimization,result.residual).same(r.expected),
          "full optimizer/fitted/residual BITWISE_MISMATCH L"+std::to_string(Level)+" request "+std::to_string(indices[j]));++verified;}
        release_multi_penalty_gcv_capacity_residual(result.residual);result.residual.reset();
        outputs[j]=std::move(result.optimization);
      }catch(...){release_multi_penalty_gcv_capacity_residual(result.residual);throw;}
    }
  }catch(...){cancel=true;std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();}};
  layer_audit::native_scope(true);start=Clock::now();std::vector<std::thread> threads;
  for(int k=0;k<32;++k)threads.emplace_back(worker,k);for(auto& t:threads)t.join();
  for(auto& slot:slots)drain<Level>(slot);const double work=elapsed(start);layer_audit::native_scope(false);
  if(error)std::rethrow_exception(error);
  need(layer_audit::counts().forbidden_calls==mutations_before,"resource mutation during work");
  std::size_t free_after;check(cudaMemGetInfo(&free_after,&total));need(free_after>=2ULL*1024*1024*1024,"headroom after work");
  const auto live=layer_audit::counts();start=Clock::now();std::vector<double> shadow(values);
  check(cudaMemcpy(shadow.data(),arena,values*8,cudaMemcpyDeviceToHost));
  std::size_t iterations=0;unsigned long long svd=0,postback=0;
  for(std::size_t j=0;j<indices.size();++j){const auto& r=*all[indices[j]];auto actual=state_snapshot(outputs[j],infos[j]);
    auto expected=r.expected;expected.floating.resize(11);
    auto it=shadow.begin()+offsets[j];actual.floating.emplace_back(it,it+q*r.keys.size());
    actual.floating.emplace_back(it+q*r.keys.size(),it+(q+351)*r.keys.size());
    expected.floating.push_back(r.expected.floating.at(11));expected.floating.push_back(r.expected.floating.at(13));
    need(actual.same(expected),"optimizer/coefficients/residual BITWISE_MISMATCH request "+std::to_string(indices[j]));
    need(assignment[j]>=0,"unprocessed request");passed+=r.keys.size();
    iterations+=std::accumulate(outputs[j].optimizer_iterations.begin(),outputs[j].optimizer_iterations.end(),std::size_t(0));
    svd+=outputs[j].diagnostics.cuda_qr_svd_cycles;postback+=outputs[j].diagnostics.cuda_svd_vector_postback_cycles;
  }
  const double verification=elapsed(start);need(passed==targets&&(!verify||verified==indices.size()),"coverage");
  unsigned long long stats[30]={};using ReadStats=int(*)(unsigned long long*,int);
  auto read_stats=reinterpret_cast<ReadStats>(dlsym(RTLD_DEFAULT,"fastkpc_postback_profile_read"));
  if(read_stats)need(read_stats(stats,30)==0,"profile read");
  start=Clock::now();for(auto e:events)check(cudaEventDestroy(e));slots.clear();check(cudaFree(arena));
  check(cudaDeviceSynchronize());const double destruction=elapsed(start);closed();
  const auto counts_after=multi_penalty_gcv_postback_execution_counts();
  for(int m=0;m<2;++m){
    need(counts_after.created[m]-counts_before.created[m]==(m==backend?32:0),"backend created");
    need(counts_after.destroyed[m]-counts_before.destroyed[m]==(m==backend?32:0),"backend destroyed");
    need(counts_after.optimizer_batches[m]-counts_before.optimizer_batches[m]==(m==backend?indices.size():0),"backend batches");
    need(counts_after.optimizer_targets[m]-counts_before.optimizer_targets[m]==(m==backend?targets:0),"backend targets");
    need(counts_after.q37_targets[m]-counts_before.q37_targets[m]==(m==backend&&Level==4?targets:0),"backend q37");
    need(counts_after.q46_targets[m]-counts_before.q46_targets[m]==(m==backend&&Level==5?targets:0),"backend q46");
    need(counts_after.other_targets[m]==counts_before.other_targets[m],"backend other q");
  }
  std::ofstream report(prefix+"-L"+std::to_string(Level)+".json");need(bool(report),"report open");
  report<<std::setprecision(17)<<"{\"status\":\"PASS\",\"level\":"<<Level<<",\"mode\":\""<<mode
    <<"\",\"vector_postback\":"<<backend<<",\"setups\":"<<indices.size()<<",\"targets\":"<<passed<<",\"full_shadow_verified_setups\":"<<verified
    <<",\"work_sec\":"<<work<<",\"packing_sec\":"<<packing<<",\"preparation_sec\":"<<preparation
    <<",\"destruction_sec\":"<<destruction<<",\"verification_sec\":"<<verification
    <<",\"charged_lifecycle_sec\":"<<packing+preparation+work+destruction<<",\"iterations\":"<<iterations
    <<",\"svd_cycles\":"<<svd<<",\"postback_cycles\":"<<postback
    <<",\"application_bytes\":"<<live.bytes<<",\"physical_increment_bytes\":"<<(free_before>free_after?free_before-free_after:0)
    <<",\"free_after_work\":"<<free_after<<",\"resource_closed\":true,\"profile_cumulative\":[";
  for(int i=0;i<30;++i)report<<(i?",":"")<<stats[i];report<<"]}\n";
  std::cout<<"PASS L"<<Level<<" mode="<<mode<<" setups="<<indices.size()<<" targets="<<passed
    <<" work_sec="<<work<<" lifecycle_sec="<<packing+preparation+work+destruction<<std::endl;
}
int main(int argc,char** argv)try{
  need(argc==5,"cohorts L4_RAW L5_RAW PREFIX reduced|verify|timing");std::string mode=argv[4];
  need(mode=="reduced"||mode=="verify"||mode=="timing","mode");check(cudaSetDevice(0));check(cudaFree(nullptr));closed();
  auto l4=l4_diag::read_layer(argv[1]);auto l5=layer_diag::read_layer(argv[2]);
  layer<4>(l4,mode,argv[3]);layer<5>(l5,mode,argv[3]);return 0;
}catch(const std::exception& e){layer_audit::native_scope(false);std::cerr<<e.what()<<std::endl;return 1;}
