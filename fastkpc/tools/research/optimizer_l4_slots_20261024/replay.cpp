#include "slot_api.hpp"
#include "native_observe.hpp"
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
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
using namespace l4_diag;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(cudaError_t s){need(s==cudaSuccess,cudaGetErrorString(s));}
void closed(){auto c=layer_audit::counts();need(!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,"resource closure");}
struct NativeScope {NativeScope(){layer_audit::native_scope(true);}~NativeScope(){layer_audit::native_scope(false);}};
struct Arena {
  double* ptr=nullptr;size_t values=0;
  explicit Arena(size_t n):values(n){check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(double)));}
  ~Arena(){if(ptr)cudaFree(ptr);}
  std::vector<double> shadow(){std::vector<double> v(values);check(cudaMemcpy(v.data(),ptr,values*8,cudaMemcpyDeviceToHost));return v;}
};
struct Events {
  std::vector<cudaEvent_t> values;
  explicit Events(size_t n){try{for(size_t k=0;k<n;++k){cudaEvent_t e=nullptr;
    check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming|cudaEventBlockingSync));values.push_back(e);}}
    catch(...){for(auto e:values)cudaEventDestroy(e);throw;}}
  ~Events(){for(auto e:values)cudaEventDestroy(e);}
};
Snapshot saved_snapshot(const l4_slot::Result& r,const std::vector<double>& saved,size_t offset){
  const auto& a=r.optimization;const auto& d=a.diagnostics;const auto& info=r.info;
  need(d.exact_primary_only&&d.cuda_exact_primary_target_count==a.target_count&&!d.cpu_objective_count&&
    !d.cpu_optimizer_count&&!d.cpu_multi_penalty_solve_count&&!d.fallback_count&&!d.cuda_error_count&&
    !d.cuda_stability_replay_target_count&&!d.cuda_terminal_boundary_confirmation_count,"GPU authority");
  for(int i=0;i<a.target_count;++i)need(info.solver_info.at(i)==0&&info.optimizer_status.at(i)==0&&
    info.decomposition_route.at(i)==kMultiPenaltyGcvDecompositionStableSvd,"stable-SVD status");
  auto take=[&](size_t n){need(offset+n<=saved.size(),"arena bound");std::vector<double> v(saved.begin()+offset,saved.begin()+offset+n);offset+=n;return v;};
  const size_t q_count=size_t(37)*a.target_count,n_count=size_t(351)*a.target_count;
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
  double work=0,prepare=0,destroy=0,verify=0,pack=0,lifecycle=0,bind_sum=0,opt_sum=0,detach_sum=0;
  layer_audit::Counts live;
  size_t application=0,physical=0,free_after=0,targets=0,iterations=0,rebindings=0,stale_checks=0;
};
int main(int argc,char** argv){try{
  need(argc==4,"replay RAW PREFIX checks|memcheck|capacity|campaign");
  const std::string prefix=argv[2],mode=argv[3];
  need(mode=="checks"||mode=="memcheck"||mode=="capacity"||mode=="campaign","mode");
  need(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H")&&
    std::string(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H"))=="1","status wait required");
  auto windows=read_layer(argv[1]);std::vector<const Request*> all;
  for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
  std::map<int,size_t> first;
  for(size_t i=0;i<all.size();++i)first.emplace(all[i]->keys.size(),i);
  need(first.size()==14,"capacity coverage");
  std::vector<size_t> indices(all.size());std::iota(indices.begin(),indices.end(),0);
  const bool reduced=mode=="checks"||mode=="memcheck";
  if(reduced){indices.clear();for(const auto& e:first)indices.push_back(e.second);
    indices.push_back(all.size()-1);std::sort(indices.begin(),indices.end());
    indices.erase(std::unique(indices.begin(),indices.end()),indices.end());need(indices.size()==15,"reduced selection");}
  std::vector<size_t> offsets(indices.size());size_t arena_values=0;
  for(size_t j=0;j<indices.size();++j){offsets[j]=arena_values;arena_values+=(37+2*351)*all[indices[j]]->keys.size();}
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();size_t initial_free=0,total=0;
  check(cudaMemGetInfo(&initial_free,&total));need(initial_free>size_t(2)*1024*1024*1024,"initial headroom");
  layer_audit::memory_limit(initial_free-size_t(2)*1024*1024*1024);
  std::ofstream runs(prefix+"-runs.csv"),details(prefix+"-setups.csv");
  runs<<"variant,cycle,kind,work_ms,prepare_ms,destroy_ms,packet_pack_ms,lifecycle_ms,verify_ms,bind_sum_ms,optimize_sum_ms,detach_sum_ms,application_bytes,physical_increment_bytes,free_after_create,targets,iterations,rebindings,stale_checks,allocations,streams,events,blas_handles\n"<<std::setprecision(17);
  details<<"variant,cycle,kind,request,slot,generation,bind_ms,optimize_ms,detach_ms\n"<<std::setprecision(17);
  size_t completed=0,compared=0,negative=0,contract_compared=0;bool cancellation=false,partial_upload=false,partial_detach=false;
  auto observe_capacity=[&](Row& row,size_t before){
    const auto c=layer_audit::counts();size_t free=0;check(cudaMemGetInfo(&free,&total));
    if(c.bytes>=row.application){row.application=c.bytes;row.live=c;}
    row.free_after=row.free_after?std::min(row.free_after,free):free;
    row.physical=std::max(row.physical,before>=free?before-free:0);
    need(free>=size_t(2)*1024*1024*1024,"device headroom");
  };
  auto write=[&](const Row& r){runs<<r.variant<<','<<r.cycle<<','<<r.kind<<','<<r.work<<','<<r.prepare<<','<<r.destroy<<','<<r.pack<<','
    <<r.lifecycle<<','<<r.verify<<','<<r.bind_sum<<','<<r.opt_sum<<','<<r.detach_sum<<','<<r.application<<','<<r.physical<<','
    <<r.free_after<<','<<r.targets<<','<<r.iterations<<','<<r.rebindings<<','<<r.stale_checks<<','<<r.live.allocations<<','<<r.live.streams<<','<<r.live.events<<','<<r.live.blas<<'\n';runs.flush();details.flush();
    ++completed;compared+=r.targets;std::cout<<r.variant<<' '<<r.kind<<" cycle="<<r.cycle<<" PASS work_sec="<<r.work/1000
      <<" lifecycle_sec="<<r.lifecycle/1000<<" targets="<<r.targets<<" physical_MiB="<<r.physical/1048576.0<<std::endl;};
  auto verify=[&](Row& row,const std::vector<l4_slot::Result>& results,Arena& arena){
    auto t=Clock::now();auto saved=arena.shadow();
    for(size_t j=0;j<indices.size();++j){const auto& r=*all[indices[j]];
      need(results[j].info.target_keys==r.keys,"saved identity");
      need(saved_snapshot(results[j],saved,offsets[j]).same(r.expected),"BITWISE_MISMATCH request "+std::to_string(indices[j]));
      row.targets+=r.keys.size();row.iterations+=std::accumulate(results[j].optimization.optimizer_iterations.begin(),results[j].optimization.optimizer_iterations.end(),size_t(0));}
    row.verify+=ms(t);
  };
  auto windowed=[&](const std::string& kind,int cycle){
    closed();Row row;row.variant='W';row.kind=kind;row.cycle=cycle;size_t before=0;check(cudaMemGetInfo(&before,&total));
    const auto life=Clock::now();auto t=Clock::now();
    auto arena=std::make_unique<Arena>(arena_values);auto events=std::make_unique<Events>(64);
    std::vector<l4_slot::Result> results(indices.size());
    std::vector<l4_observe::Destination> observed(indices.size());
    for(size_t j=0;j<indices.size();++j){const auto count=all[indices[j]]->keys.size(),off=offsets[j];
      observed[j]={all[indices[j]],arena->ptr+off,arena->ptr+off+37*count,arena->ptr+off+(37+351)*count,nullptr};}
    row.prepare+=ms(t);
    for(size_t start=0;start<indices.size();){
      size_t end=start+1;while(end<indices.size()&&indices[end]/64==indices[start]/64)++end;
      t=Clock::now();std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> pool;
      std::vector<MultiPenaltyGcvCapacityRequest> requests;std::vector<l4_observe::Destination> dest;
      for(size_t j=start;j<end;++j){const auto& r=*all[indices[j]];const auto& g=r.geometry;
        auto p=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),g.roots,g.matrices,
          g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);pool.push_back(p);
        MultiPenaltyGcvCapacityRequest request;request.prepared=p;request.Y=r.Y;request.n=g.n;request.target_count=r.keys.size();
        request.target_keys=r.keys;request.control=r.control;requests.push_back(std::move(request));
        auto d=observed[j];d.event=events->values[j-start];dest.push_back(d);}
      l4_observe::begin(dest);row.prepare+=ms(t);observe_capacity(row,before);
      MultiPenaltyGcvCapacityMultiResult result;t=Clock::now();
      try{NativeScope scope;result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),std::min(32,int(end-start)),true);}
      catch(...){l4_observe::end(false);throw;}
      row.work+=ms(t);observe_capacity(row,before);t=Clock::now();l4_observe::end();
      for(size_t j=start;j<end;++j){const size_t k=j-start;
        results[j].optimization=std::move(result.setups[k]);results[j].info=multi_penalty_gcv_capacity_residual_info(result.residuals[k]);
        need(results[j].info.target_keys==all[indices[j]]->keys,"W target identity");
        release_multi_penalty_gcv_capacity_residual(result.residuals[k]);result.residuals[k].reset();
        observed[j]=dest[k];row.opt_sum+=dest[k].native_ms;row.detach_sum+=dest[k].retain_ms;}
      result={};pool.clear();check(cudaDeviceSynchronize());row.destroy+=ms(t);start=end;
    }
    verify(row,results,*arena);t=Clock::now();results.clear();events.reset();arena.reset();check(cudaDeviceSynchronize());row.destroy+=ms(t);
    row.lifecycle=ms(life)-row.verify;closed();
    for(size_t j=0;j<indices.size();++j)details<<'W'<<','<<cycle<<','<<kind<<','<<indices[j]<<",-1,0,0,"<<observed[j].native_ms<<','<<observed[j].retain_ms<<'\n';
    write(row);
  };
  auto slots=[&](const std::string& kind,int cycle,int workers,bool inject){
    closed();Row row;row.kind=kind;row.cycle=cycle;size_t before=0;check(cudaMemGetInfo(&before,&total));
    const auto life=Clock::now();auto t=Clock::now();std::vector<l4_slot::Packed> packets;
    for(auto i:indices)packets.push_back(l4_slot::pack(*all[i]));row.pack=ms(t);t=Clock::now();
    std::vector<l4_slot::Handle> pool;for(int k=0;k<workers;++k)pool.push_back(l4_slot::create(packets.front(),18));
    auto arena=std::make_unique<Arena>(arena_values);
    std::vector<l4_slot::Result> results(indices.size());std::vector<l4_slot::Times> times(indices.size());
    std::vector<int> assignment(indices.size(),-1);std::vector<std::uint64_t> generations(indices.size());
    std::atomic<size_t> next{0};std::atomic<bool> cancel{false},start{false};std::atomic<int> ready{0};
    std::mutex error_mutex;std::exception_ptr error;row.prepare=ms(t);observe_capacity(row,before);
    if(!reduced)need(row.physical<=size_t(1024)*1024*1024,"standalone slot capacity limit");
    auto worker=[&](int k){ready.fetch_add(1);while(!start.load())std::this_thread::yield();l4_slot::Result pending;
      try{check(cudaSetDevice(0));while(!cancel.load()){
        const size_t j=next.fetch_add(1);if(j>=indices.size())break;assignment[j]=k;
        auto at=Clock::now();l4_slot::bind(pool[k],packets[j],j+1);times[j].bind_ms=ms(at);
        need(l4_slot::binding(pool[k])==j+1,"binding identity");at=Clock::now();
        pending=l4_slot::execute(pool[k],*all[indices[j]]);times[j].optimize_ms=ms(at);
        need(pending.info.target_keys==all[indices[j]]->keys&&l4_slot::token_live(pending),"slot identity/live output");
        generations[j]=pending.info.generation;if(inject&&j==0)throw std::runtime_error("INJECTED_AFTER_EXECUTE");
        const size_t targets=all[indices[j]]->keys.size(),off=offsets[j];at=Clock::now();
        l4_slot::detach(pool[k],pending,arena->ptr+off,arena->ptr+off+37*targets,arena->ptr+off+(37+351)*targets);times[j].detach_ms=ms(at);
        need(!l4_slot::token_live(pending)&&l4_slot::token_rejected(pending)&&!l4_slot::info(pool[k]).residual_slot_leased,"released token/lease");
        results[j]=std::move(pending);
      }}catch(...){auto caught=std::current_exception();cancel=true;try{l4_slot::abandon(pool[k],pending);}catch(...){}
        std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=caught;}
    };
    t=Clock::now();{NativeScope scope;std::vector<std::thread> threads;
      try{for(int k=0;k<workers;++k)threads.emplace_back(worker,k);}catch(...){cancel=true;start=true;for(auto& w:threads)w.join();throw;}
      while(ready.load()!=workers)std::this_thread::yield();start=true;for(auto& w:threads)w.join();
      for(auto& p:pool)l4_slot::drain(p);
    }row.work=ms(t);
    if(inject){need(bool(error),"injection did not cancel");try{std::rethrow_exception(error);}catch(const std::exception& e){need(std::string(e.what())=="INJECTED_AFTER_EXECUTE","unexpected cancellation error");}
      for(const auto& p:pool)need(!l4_slot::info(p).residual_slot_leased,"cancel left lease");
      results.clear();pool.clear();arena.reset();check(cudaDeviceSynchronize());closed();cancellation=true;
      std::cout<<"PASS parallel cancellation/drain/resource closure"<<std::endl;return;}
    if(error)std::rethrow_exception(error);
    observe_capacity(row,before);
    if(!reduced)need(row.physical<=size_t(1024)*1024*1024,"post-work standalone slot capacity limit");
    verify(row,results,*arena);t=Clock::now();
    for(size_t j=0;j<indices.size();++j){need(assignment[j]>=0&&l4_slot::token_rejected(results[j]),"stale token revived");++row.stale_checks;
      row.bind_sum+=times[j].bind_ms;row.opt_sum+=times[j].optimize_ms;row.detach_sum+=times[j].detach_ms;}
    row.verify+=ms(t);row.rebindings=indices.size();t=Clock::now();results.clear();pool.clear();arena.reset();packets.clear();
    check(cudaDeviceSynchronize());row.destroy=ms(t);row.lifecycle=ms(life)-row.verify;closed();
    for(size_t j=0;j<indices.size();++j)details<<'S'<<','<<cycle<<','<<kind<<','<<indices[j]<<','<<assignment[j]<<','<<generations[j]<<','<<times[j].bind_ms<<','<<times[j].optimize_ms<<','<<times[j].detach_ms<<'\n';
    write(row);
  };
  auto rejects=[&](auto fn,const std::string& message){bool rejected=false;try{fn();}catch(const std::runtime_error& e){rejected=std::string(e.what())==message;}
    need(rejected,"expected "+message);++negative;};
  auto contracts=[&](){
    closed();const auto initial=l4_slot::pack(*all[first.at(2)]);auto s=l4_slot::create(initial,18);Arena arena((37+702)*18);l4_slot::Result stale;
    size_t serial=0;for(auto i:{first.at(2),first.at(18),first.at(2)}){auto packet=l4_slot::pack(*all[i]);
      l4_slot::bind(s,packet,++serial);if(stale.token)need(l4_slot::token_rejected(stale),"old token after rebind");
      auto result=l4_slot::execute(s,*all[i]);stale.token.reset();need(l4_slot::token_live(result),"old destructor cleared new lease");
      rejects([&]{l4_slot::bind(s,packet,1000);},"SLOT_OUTPUT_BUSY");
      const auto generation=l4_slot::info(s).generation,identity=l4_slot::binding(s);
      need(generation==result.info.generation&&identity==serial,"busy rebind mutated state");
      const size_t count=all[i]->keys.size();l4_slot::detach(s,result,arena.ptr,arena.ptr+37*count,arena.ptr+(37+351)*count);
      need(saved_snapshot(result,arena.shadow(),0).same(all[i]->expected),"contract output mismatch");contract_compared+=count;
      Request bad=*all[i];bad.geometry.q=38;auto invalid=packet;invalid.request=&bad;
      rejects([&]{l4_slot::bind(s,invalid,1001);},"SLOT_SHAPE_REJECTED");
      need(l4_slot::info(s).generation==generation&&l4_slot::binding(s)==identity,"invalid rebind mutated state");stale=std::move(result);
    }l4_slot::drain(s);stale.token.reset();s.reset();
    std::cout<<"PASS busy/invalid bind, stale generation and old-token destruction"<<std::endl;
  };
  auto failure=[&](l4_slot::Failure point){
    closed();auto packet=l4_slot::pack(*all[first.at(2)]);auto s=l4_slot::create(packet,18);Arena arena((37+702)*2);l4_slot::Result result;
    if(point==l4_slot::Failure::AfterPartialUpload){l4_slot::arm_failure(s,point);
      rejects([&]{l4_slot::bind(s,packet,1);},"INJECTED_PARTIAL_UPLOAD");partial_upload=true;
    }else{l4_slot::bind(s,packet,1);result=l4_slot::execute(s,*packet.request);l4_slot::arm_failure(s,point);
      rejects([&]{l4_slot::detach(s,result,arena.ptr,arena.ptr+37*2,arena.ptr+(37+351)*2);},"INJECTED_PARTIAL_DETACH");partial_detach=true;}
    rejects([&]{l4_slot::bind(s,packet,2);},"SLOT_POISONED");l4_slot::abandon(s,result);
    need(!l4_slot::info(s).residual_slot_leased,"failed slot lease survived drain");
    if(result.token)need(l4_slot::token_rejected(result),"failed output remained live");result={};s.reset();
    std::cout<<"PASS partial "<<(point==l4_slot::Failure::AfterPartialUpload?"upload":"detach")<<" failure drain"<<std::endl;
  };
  if(reduced){contracts();closed();failure(l4_slot::Failure::AfterPartialUpload);closed();failure(l4_slot::Failure::AfterPartialDetach);closed();
    if(mode=="checks")windowed("reduced-production",0);
    slots("sequential-rebind",0,1,false);
    if(mode=="checks")slots("parallel-rebind",1,2,false);
    slots("cancel",2,2,true);closed();
  }else if(mode=="capacity")slots("standalone-capacity",0,32,false);
  else{windowed("warmup",-1);slots("warmup",-1,32,false);
    for(int i=0;i<6;++i)if(i%2==0){windowed("timing",i);slots("timing",i,32,false);}else{slots("timing",i,32,false);windowed("timing",i);}}
  closed();const auto c=layer_audit::counts();need(c.forbidden_calls==0,"hot resource mutation");
  std::ofstream summary(prefix+"-summary.json");summary<<"{\"status\":\"PASS\",\"runs\":"<<completed
    <<",\"setups_per_run\":"<<indices.size()<<",\"compared_target_occurrences\":"<<compared
    <<",\"contract_compared_targets\":"<<contract_compared<<",\"negative_checks\":"<<negative
    <<",\"cancellation_checked\":"<<(cancellation?"true":"false")
    <<",\"partial_upload_checked\":"<<(partial_upload?"true":"false")<<",\"partial_detach_checked\":"<<(partial_detach?"true":"false")
    <<",\"forbidden_calls\":"<<c.forbidden_calls<<",\"resources_closed\":true,\"outputs_bitwise_exact\":true}\n";
  need(bool(summary)&&bool(runs)&&bool(details),"receipt output");return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
