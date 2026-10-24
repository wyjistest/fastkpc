#include "slot_api.hpp"
#include "../optimizer_layer_refill_20261023/layer_fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
using namespace fastkpc;
using namespace layer_diag;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(cudaError_t s){need(s==cudaSuccess,cudaGetErrorString(s));}
void closed(){auto c=layer_audit::counts();need(!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,"resource closure");}
struct NativeScope {NativeScope(){layer_audit::native_scope(true);}~NativeScope(){layer_audit::native_scope(false);}};
struct Arena {
  double* ptr=nullptr;size_t values=0;
  explicit Arena(size_t count):values(count){check(cudaMalloc(reinterpret_cast<void**>(&ptr),count*sizeof(double)));}
  ~Arena(){if(ptr)cudaFree(ptr);}
  Arena(const Arena&)=delete;
  std::vector<double> shadow(){std::vector<double> v(values);check(cudaMemcpy(v.data(),ptr,values*sizeof(double),cudaMemcpyDeviceToHost));return v;}
};
Snapshot saved_snapshot(const slot_diag::Result& r,const std::vector<double>& saved,size_t offset){
  const auto& a=r.optimization;const auto& d=a.diagnostics;const auto& info=r.info;
  need(d.exact_primary_only&&d.cuda_exact_primary_target_count==a.target_count&&!d.cpu_objective_count&&
    !d.cpu_optimizer_count&&!d.cpu_multi_penalty_solve_count&&!d.fallback_count&&!d.cuda_error_count&&
    !d.cuda_stability_replay_target_count&&!d.cuda_terminal_boundary_confirmation_count,"GPU authority");
  for(int i=0;i<a.target_count;++i)need(info.solver_info.at(i)==0&&info.optimizer_status.at(i)==0&&
    info.decomposition_route.at(i)==kMultiPenaltyGcvDecompositionStableSvd,"stable-SVD status");
  auto take=[&](size_t n){need(offset+n<=saved.size(),"arena bound");std::vector<double> v(saved.begin()+offset,saved.begin()+offset+n);offset+=n;return v;};
  const size_t q_count=size_t(46)*a.target_count,n_count=size_t(351)*a.target_count;
  auto coefficients=take(q_count),fitted=take(n_count),residual=take(n_count);Snapshot s;
  s.floating={a.initial_log_sp,a.selected_log_sp,a.rss,a.edf,a.score,a.condition,a.gradient,a.hessian,a.coefficients,
    a.rms_gradient,a.hessian_eigenvalues,std::move(coefficients),std::move(fitted),std::move(residual)};
  s.integral={a.aggregate_penalty_rank,a.numerical_rank,a.solver_info,a.optimizer_iterations,a.score_calls,
    a.objective_calls,a.step_halving_count,a.newton_trial_count,a.steepest_descent_trial_count,
    a.boundary_probe_count,a.boundary_accepted_count,a.boundary_status,a.fully_converged,
    a.hessian_positive_definite,a.step_failed,a.optimizer_status,info.solver_info,info.decomposition_route,info.numerical_rank};
  return s;
}
struct Row {
  char variant='S';int cycle=0;std::string kind;
  double work=0,prepare=0,destroy=0,verify=0,pack=0,bind_sum=0,opt_sum=0,detach_sum=0;
  layer_audit::Counts live;
  size_t application=0,physical=0,free_after=0,targets=0,iterations=0,rebindings=0,stale_checks=0;
};
int main(int argc,char** argv){try{
  need(argc==4,"replay RAW PREFIX checks|memcheck|capacity|campaign");
  const std::string prefix=argv[2],mode=argv[3];
  need(mode=="checks"||mode=="memcheck"||mode=="capacity"||mode=="campaign","mode");
  need(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H")&&
    std::string(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H"))=="1","wait required");
  auto windows=read_layer(argv[1]);std::vector<const Request*> all;
  for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
  const auto pack_started=Clock::now();
  std::vector<slot_diag::Packed> packets;for(auto* r:all)packets.push_back(slot_diag::pack(*r));
  const double packet_pack_ms=ms(pack_started);
  size_t largest=0;for(size_t i=1;i<packets.size();++i)if(packets[i].total_rank>packets[largest].total_rank)largest=i;
  std::vector<size_t> indices(all.size());std::iota(indices.begin(),indices.end(),0);
  const bool reduced=mode=="checks"||mode=="memcheck";
  if(reduced){std::map<int,size_t> first;for(size_t i=0;i<all.size();++i)first.emplace(all[i]->keys.size(),i);
    indices={first.at(8),first.at(2),first.at(4),first.at(3),first.at(8),first.at(2),first.at(4),first.at(3)};}
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();size_t initial_free=0,total=0;
  check(cudaMemGetInfo(&initial_free,&total));need(initial_free>size_t(2)*1024*1024*1024,"initial headroom");
  layer_audit::memory_limit(initial_free-size_t(2)*1024*1024*1024);
  std::vector<size_t> offsets(indices.size());size_t arena_values=0;
  for(size_t j=0;j<indices.size();++j){offsets[j]=arena_values;arena_values+=(46+2*351)*all[indices[j]]->keys.size();}
  std::ofstream runs(prefix+"-runs.csv"),details(prefix+"-setups.csv");
  runs<<"variant,cycle,kind,work_ms,prepare_ms,destroy_ms,packet_pack_ms,lifecycle_ms,verify_ms,bind_sum_ms,optimize_sum_ms,detach_sum_ms,application_bytes,physical_increment_bytes,free_after_create,targets,iterations,rebindings,stale_checks,allocations,streams,events,blas_handles\n"<<std::setprecision(17);
  details<<"variant,cycle,kind,request,slot,generation,bind_ms,optimize_ms,detach_ms\n"<<std::setprecision(17);
  size_t completed=0,compared=0,negative=0;bool cancellation=false;
  auto write=[&](const Row& r){runs<<r.variant<<','<<r.cycle<<','<<r.kind<<','<<r.work<<','<<r.prepare<<','<<r.destroy<<','<<r.pack<<','
    <<r.work+r.prepare+r.destroy+r.pack<<','<<r.verify<<','<<r.bind_sum<<','<<r.opt_sum<<','<<r.detach_sum<<','<<r.application<<','<<r.physical<<','
    <<r.free_after<<','<<r.targets<<','<<r.iterations<<','<<r.rebindings<<','<<r.stale_checks<<','<<r.live.allocations<<','<<r.live.streams<<','<<r.live.events<<','<<r.live.blas<<'\n';runs.flush();details.flush();
    ++completed;compared+=r.targets;std::cout<<r.variant<<' '<<r.kind<<" cycle="<<r.cycle<<" PASS work_sec="<<r.work/1000
      <<" lifecycle_sec="<<(r.work+r.prepare+r.destroy+r.pack)/1000<<" targets="<<r.targets<<std::endl;};
  auto resident=[&](const std::string& kind,int cycle){
    Row row;row.variant='R';row.kind=kind;row.cycle=cycle;closed();size_t before=0;check(cudaMemGetInfo(&before,&total));
    auto t=Clock::now();std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> pool;
    std::vector<MultiPenaltyGcvCapacityRequest> requests;
    for(size_t i:indices){const auto& r=*all[i];const auto& g=r.geometry;
      auto p=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),g.roots,g.matrices,
        g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);pool.push_back(p);
      MultiPenaltyGcvCapacityRequest q;q.prepared=p;q.Y=r.Y;q.n=g.n;q.target_count=r.keys.size();q.target_keys=r.keys;q.control=r.control;
      requests.push_back(std::move(q));check(cudaMemGetInfo(&row.free_after,&total));need(row.free_after>=size_t(2)*1024*1024*1024,"resident capacity");}
    row.prepare=ms(t);row.live=layer_audit::counts();row.application=row.live.bytes;row.physical=before-row.free_after;
    MultiPenaltyGcvCapacityMultiResult result;t=Clock::now();
    {NativeScope scope;result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),std::min(32,int(indices.size())),true);}
    row.work=ms(t);t=Clock::now();
    for(size_t j=0;j<indices.size();++j){const auto& r=*all[indices[j]];
      need(multi_penalty_gcv_capacity_residual_info(result.residuals[j]).target_keys==r.keys,"resident identity");
      need(snapshot(result.setups[j],result.residuals[j]).same(r.expected),"resident bitwise mismatch");
      row.targets+=r.keys.size();row.iterations+=std::accumulate(result.setups[j].optimizer_iterations.begin(),result.setups[j].optimizer_iterations.end(),0);}
    row.verify=ms(t);t=Clock::now();
    for(auto& r:result.residuals){release_multi_penalty_gcv_capacity_residual(r);r.reset();}
    result={};pool.clear();check(cudaDeviceSynchronize());row.destroy=ms(t);closed();write(row);
  };
  auto slots=[&](const std::string& kind,int cycle,int workers,bool inject){
    Row row;row.kind=kind;row.cycle=cycle;closed();size_t before=0;check(cudaMemGetInfo(&before,&total));
    auto t=Clock::now();std::vector<slot_diag::Handle> pool;
    for(int k=0;k<workers;++k)pool.push_back(slot_diag::create(packets[largest],8));
    auto arena=std::make_unique<Arena>(arena_values);row.prepare=ms(t);row.pack=packet_pack_ms;
    row.live=layer_audit::counts();row.application=row.live.bytes;
    check(cudaMemGetInfo(&row.free_after,&total));row.physical=before-row.free_after;
    need(row.free_after>=size_t(2)*1024*1024*1024,"slot capacity headroom");
    std::vector<slot_diag::Result> results(indices.size());std::vector<slot_diag::Times> times(indices.size());
    std::vector<int> assignment(indices.size(),-1);std::vector<std::uint64_t> generations(indices.size());
    std::atomic<size_t> next{0};std::atomic<bool> cancel{false},start{false};std::atomic<int> ready{0};
    std::mutex error_mutex;std::exception_ptr error;
    auto worker=[&](int k){ready.fetch_add(1);while(!start.load())std::this_thread::yield();
      slot_diag::Result pending;
      try{check(cudaSetDevice(0));while(!cancel.load()){
        const size_t j=next.fetch_add(1);if(j>=indices.size())break;const size_t i=indices[j];assignment[j]=k;
        auto at=Clock::now();slot_diag::bind(pool[k],packets[i],j+1);times[j].bind_ms=ms(at);
        need(slot_diag::binding(pool[k])==j+1,"binding identity");
        at=Clock::now();pending=slot_diag::execute(pool[k],*all[i]);times[j].optimize_ms=ms(at);
        need(pending.info.target_keys==all[i]->keys&&slot_diag::token_live(pending),"slot identity/live output");
        generations[j]=pending.info.generation;
        if(inject&&j==2)throw std::runtime_error("INJECTED_AFTER_EXECUTE");
        const size_t targets=all[i]->keys.size(),off=offsets[j];at=Clock::now();
        slot_diag::detach(pool[k],pending,arena->ptr+off,arena->ptr+off+46*targets,arena->ptr+off+(46+351)*targets);
        times[j].detach_ms=ms(at);
        need(!slot_diag::token_live(pending)&&slot_diag::token_rejected(pending)&&!slot_diag::info(pool[k]).residual_slot_leased,"released token/lease");
        results[j]=std::move(pending);
      }}catch(...){auto caught=std::current_exception();cancel=true;
        try{slot_diag::abandon(pool[k],pending);}catch(...){}
        std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=caught;}
    };
    t=Clock::now();{
      NativeScope scope;std::vector<std::thread> threads;
      try{for(int k=0;k<workers;++k)threads.emplace_back(worker,k);}
      catch(...){cancel=true;start=true;for(auto& w:threads)w.join();throw;}
      while(ready.load()!=workers)std::this_thread::yield();start=true;for(auto& w:threads)w.join();
      for(auto& p:pool)slot_diag::drain(p);
    }row.work=ms(t);
    if(inject){need(bool(error),"injection did not cancel");try{std::rethrow_exception(error);}catch(const std::exception& e){need(std::string(e.what())=="INJECTED_AFTER_EXECUTE","unexpected failure");}
      for(const auto& p:pool)need(!slot_diag::info(p).residual_slot_leased,"cancel left lease");
      results.clear();pool.clear();arena.reset();check(cudaDeviceSynchronize());closed();cancellation=true;
      std::cout<<"PASS injected cancellation/drain/lease/resource closure"<<std::endl;return;}
    if(error)std::rethrow_exception(error);
    t=Clock::now();auto saved=arena->shadow();
    for(size_t j=0;j<indices.size();++j){const auto& r=*all[indices[j]];
      need(assignment[j]>=0&&saved_snapshot(results[j],saved,offsets[j]).same(r.expected),"slot BITWISE_MISMATCH request "+std::to_string(indices[j]));
      need(slot_diag::token_rejected(results[j]),"stale token revived after rebind");++row.stale_checks;
      row.targets+=r.keys.size();row.iterations+=std::accumulate(results[j].optimization.optimizer_iterations.begin(),results[j].optimization.optimizer_iterations.end(),0);
      row.bind_sum+=times[j].bind_ms;row.opt_sum+=times[j].optimize_ms;row.detach_sum+=times[j].detach_ms;
      details<<'S'<<','<<cycle<<','<<kind<<','<<indices[j]<<','<<assignment[j]<<','<<generations[j]<<','<<times[j].bind_ms<<','<<times[j].optimize_ms<<','<<times[j].detach_ms<<'\n';}
    row.rebindings=indices.size();row.verify=ms(t);t=Clock::now();results.clear();pool.clear();arena.reset();
    check(cudaDeviceSynchronize());row.destroy=ms(t);closed();write(row);
  };
  auto contracts=[&](){
    closed();auto s=slot_diag::create(packets[largest],8);Arena arena(arena_values);slot_diag::Result stale;
    auto rejects=[&](auto fn,const std::string& message){bool rejected=false;try{fn();}catch(const std::runtime_error& e){rejected=std::string(e.what())==message;}need(rejected,"expected "+message);++negative;};
    for(size_t j=0;j<indices.size();++j){const size_t i=indices[j];
      slot_diag::bind(s,packets[i],j+1);if(stale.token)need(slot_diag::token_rejected(stale),"old token after binding");
      auto result=slot_diag::execute(s,*all[i]);stale.token.reset();need(slot_diag::token_live(result),"old token destructor cleared new lease");
      rejects([&]{slot_diag::bind(s,packets[i],1000);},"SLOT_OUTPUT_BUSY");
      const auto generation=slot_diag::info(s).generation;const auto identity=slot_diag::binding(s);
      need(generation==result.info.generation&&identity==j+1,"busy rebind changed identity");
      size_t targets=all[i]->keys.size(),off=offsets[j];
      slot_diag::detach(s,result,arena.ptr+off,arena.ptr+off+46*targets,arena.ptr+off+(46+351)*targets);
      need(slot_diag::token_rejected(result),"detached token accepted");
      Request bad=*all[i];bad.geometry.q=47;auto packet=packets[i];packet.request=&bad;
      rejects([&]{slot_diag::bind(s,packet,1001);},"SLOT_SHAPE_REJECTED");
      need(slot_diag::info(s).generation==generation&&slot_diag::binding(s)==identity,"invalid bind mutated state");
      stale=std::move(result);
    }
    slot_diag::drain(s);stale.token.reset();s.reset();
  };
  if(reduced){contracts();closed();slots("sequential-rebind",0,1,false);slots("parallel-rebind",1,2,false);
    slots("cancel",2,2,true);closed();}
  else if(mode=="capacity")slots("standalone-capacity",0,32,false);
  else{
    resident("warmup",-1);slots("warmup",-1,32,false);
    for(int i=0;i<6;++i){if(i%2==0){resident("timing",i);slots("timing",i,32,false);}
      else{slots("timing",i,32,false);resident("timing",i);}}
  }
  closed();const auto c=layer_audit::counts();need(c.forbidden_calls==0,"hot resource mutations");
  std::ofstream summary(prefix+"-summary.json");summary<<"{\"status\":\"PASS\",\"runs\":"<<completed
    <<",\"setups_per_run\":"<<indices.size()<<",\"compared_target_occurrences\":"<<compared
    <<",\"negative_checks\":"<<negative<<",\"cancellation_checked\":"<<(cancellation?"true":"false")
    <<",\"forbidden_calls\":"<<c.forbidden_calls<<",\"resources_closed\":true,\"outputs_bitwise_exact\":true}\n";
  need(bool(summary)&&bool(runs)&&bool(details),"receipt output");return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
