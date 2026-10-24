// Whole-layer host scheduling diagnostic; numerical calls use the actual native.
#include "layer_fixture.hpp"
#include "audit.hpp"
#include "observe.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
using namespace layer_diag;
using namespace fastkpc;
using Clock=std::chrono::steady_clock;
using Prepared=std::shared_ptr<MultiPenaltyGcvCapacityPrepared>;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void check(cudaError_t status){need(status==cudaSuccess,cudaGetErrorString(status));}
void closed(){auto c=layer_audit::counts();need(c.allocations==0&&c.bytes==0&&c.streams==0&&c.events==0&&c.blas==0,"resource closure failed");}
struct Scope {Scope(){layer_audit::native_scope(true);}~Scope(){layer_audit::native_scope(false);}};
struct Run {
  int id=0,cycle=0,targets=0,iterations=0,calls=0,peak=0;
  char variant='P';std::string kind;
  double native_ms=0,native_report_ms=0,prepare_ms=0,destroy_ms=0,pack_ms=0;
  double verify_ms=0,token_release_ms=0,raw_ms=0,work_ms=0,pool_prepare_ms=0,pool_destroy_ms=0;
  size_t forbidden=0;
};
int main(int argc,char** argv){
 try {
  need(argc==4,"replay RAW PREFIX capacity|campaign|profile|memcheck");
  const std::string prefix=argv[2],mode=argv[3];
  need(mode=="capacity"||mode=="campaign"||mode=="profile"||mode=="memcheck","unknown mode");
  need(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H")&&
    std::string(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H"))=="1","wait must be on");
  auto layer=read_layer(argv[1]);
  std::vector<const Request*> inputs;std::vector<std::pair<size_t,size_t>> windows;
  std::vector<int> window_ids;
  for(const auto& w:layer){
    const size_t begin=inputs.size();
    if(mode=="memcheck"&&w.window!=1&&w.window!=15)continue;
    const size_t count=mode=="memcheck"?2:w.requests.size();
    for(size_t i=0;i<count;++i){inputs.push_back(&w.requests[i]);window_ids.push_back(w.window);}
    windows.emplace_back(begin,inputs.size());
  }
  const size_t count=inputs.size();std::vector<std::string> keys;
  for(auto* r:inputs)keys.push_back(r->keys.front());
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();
  size_t free_start=0,total=0;check(cudaMemGetInfo(&free_start,&total));
  const size_t headroom=size_t(2)*1024*1024*1024;
  need(free_start>headroom,"insufficient initial free memory");
  layer_audit::memory_limit(free_start-headroom);
  std::ofstream setup_log(prefix+"-setups.csv");
  setup_log<<"allocation,setup,window,n,q,penalties,targets,application_bytes,free_before,free_after,prepare_ms\n"<<std::setprecision(17);
  size_t allocation_id=0,minimum_free=free_start;
  auto create=[&](size_t i){
    size_t before=0,after=0;check(cudaMemGetInfo(&before,&total));need(before>headroom,"headroom exhausted");
    const auto c=layer_audit::counts();const auto started=Clock::now();const auto& r=*inputs[i];const auto& g=r.geometry;
    auto p=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),
      g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);
    const double duration=ms(started);check(cudaMemGetInfo(&after,&total));minimum_free=std::min(minimum_free,after);
    need(after>=headroom,"full layer violates 2 GiB headroom");
    const auto d=layer_audit::counts();need(d.bytes>c.bytes&&d.blas>c.blas&&d.events>=c.events+2,"resource interposition incomplete");
    setup_log<<allocation_id++<<','<<i<<','<<window_ids[i]<<','<<g.n<<','<<g.q<<','<<g.penalties<<','
      <<g.capacity<<','<<d.bytes-c.bytes<<','<<before<<','<<after<<','<<duration<<'\n';
    return p;
  };
  std::vector<Prepared> pool(count);
  auto create_pool=[&](){auto t=Clock::now();for(size_t i=0;i<count;++i)pool[i]=create(i);return ms(t);};
  auto destroy_pool=[&](){auto t=Clock::now();for(auto& p:pool)p.reset();check(cudaDeviceSynchronize());closed();return ms(t);};
  if(mode=="capacity"){
    const double prepare=create_pool();const auto live=layer_audit::counts();
    size_t free_all=0;check(cudaMemGetInfo(&free_all,&total));const double release=destroy_pool();
    size_t free_after=0;check(cudaMemGetInfo(&free_after,&total));
    std::ofstream out(prefix+"-summary.json");out<<std::setprecision(17)
      <<"{\"status\":\"PASS\",\"setups\":"<<count<<",\"targets\":6816,\"application_bytes\":"<<live.bytes
      <<",\"application_allocations\":"<<live.allocations<<",\"streams\":"<<live.streams<<",\"events\":"<<live.events
      <<",\"blas_handles\":"<<live.blas<<",\"free_before\":"<<free_start<<",\"free_all_resident\":"<<free_all
      <<",\"free_after_release\":"<<free_after<<",\"headroom_required\":"<<headroom
      <<",\"prepare_ms\":"<<prepare<<",\"release_ms\":"<<release<<",\"resources_closed\":true}\n";
    need(bool(out)&&bool(setup_log),"capacity write failed");std::cout<<"PASS full L5 capacity bytes="<<live.bytes<<" free="<<free_all<<std::endl;return 0;
  }
  std::ofstream runs(prefix+"-runs.csv"),calls(prefix+"-calls.csv");
  runs<<"run,cycle,kind,variant,workers,setups,targets,iterations,calls,peak_host_calls,native_ms,native_report_ms,prepare_ms,destroy_ms,pack_ms,verify_ms,token_release_ms,raw_ms,segmented_work_ms,pool_prepare_ms,pool_destroy_ms,lifecycle_work_ms,forbidden_calls,parity\n"<<std::setprecision(17);
  calls<<"run,variant,call,first_setup,setups,workers,wall_ms,native_wall_ms,peak_host_calls\n"<<std::setprecision(17);
  int run_id=0,total_runs=0,total_targets=0;
  auto write_run=[&](const Run& r){
    runs<<r.id<<','<<r.cycle<<','<<r.kind<<','<<r.variant<<','<<(r.variant=='R'?32:64)<<','<<count
      <<','<<r.targets<<','<<r.iterations<<','<<r.calls<<','<<r.peak<<','<<r.native_ms<<','<<r.native_report_ms
      <<','<<r.prepare_ms<<','<<r.destroy_ms<<','<<r.pack_ms<<','<<r.verify_ms<<','<<r.token_release_ms
      <<','<<r.raw_ms<<','<<r.work_ms<<','<<r.pool_prepare_ms<<','<<r.pool_destroy_ms
      <<','<<r.work_ms+r.pool_prepare_ms+r.pool_destroy_ms<<','<<r.forbidden<<",1\n";
    runs.flush();calls.flush();++total_runs;total_targets+=r.targets;
  };
  auto evaluate=[&](char variant,const std::string& kind,int cycle,bool observe){
    Run r;r.id=run_id++;r.variant=variant;r.kind=kind;r.cycle=cycle;
    const int requested=variant=='R'?32:64;
    std::vector<std::pair<size_t,size_t>> groups=variant=='R'?
      std::vector<std::pair<size_t,size_t>>{{0,count}}:windows;
    std::vector<MultiPenaltyGcvCapacityMultiResult> retained;
    observe_begin(keys,observe);
    const std::string label="layer:L5:"+std::string(1,variant)+":"+kind;
    nvtxRangePushA(label.c_str());const auto whole_start=Clock::now();
    auto verify=[&](MultiPenaltyGcvCapacityMultiResult& result,size_t begin,size_t end){
      auto t=Clock::now();need(result.setups.size()==end-begin&&result.residuals.size()==end-begin,"result size changed");
      for(size_t i=begin;i<end;++i){
        const auto& input=*inputs[i];auto& residual=result.residuals[i-begin];const auto& opt=result.setups[i-begin];
        auto info=multi_penalty_gcv_capacity_residual_info(residual);
        need(info.target_keys==input.keys&&info.device_resident&&!info.released,"identity/order/residency changed");
        need(snapshot(opt,residual).same(input.expected),"BITWISE_MISMATCH setup "+std::to_string(i));
        r.targets+=opt.target_count;r.iterations+=std::accumulate(opt.optimizer_iterations.begin(),opt.optimizer_iterations.end(),0);
      }
      r.verify_ms+=ms(t);t=Clock::now();
      for(size_t i=begin;i<end;++i){auto& p=result.residuals[i-begin];release_multi_penalty_gcv_capacity_residual(p);p.reset();
        need(!multi_penalty_gcv_capacity_prepared_info(pool[i]).residual_slot_leased,"output lease remains");}
      r.token_release_ms+=ms(t);
    };
    for(const auto& group:groups){
      const auto [begin,end]=group;
      if(variant=='P'){auto t=Clock::now();for(size_t i=begin;i<end;++i)pool[i]=create(i);r.prepare_ms+=ms(t);}
      auto t=Clock::now();std::vector<MultiPenaltyGcvCapacityRequest> requests;
      for(size_t i=begin;i<end;++i){const auto& input=*inputs[i];MultiPenaltyGcvCapacityRequest request;
        request.prepared=pool[i];request.Y=input.Y;request.n=input.geometry.n;request.target_count=input.keys.size();
        request.target_keys=input.keys;request.control=input.control;requests.push_back(std::move(request));}
      r.pack_ms+=ms(t);const int workers=std::min(requested,int(end-begin));
      const auto before=layer_audit::counts();t=Clock::now();MultiPenaltyGcvCapacityMultiResult result;
      {Scope scope;result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),workers,true);}
      const double call_ms=ms(t);const auto after=layer_audit::counts();r.forbidden+=after.forbidden_calls-before.forbidden_calls;
      need(after.forbidden_calls==before.forbidden_calls,"GPU resource mutation inside optimizer call");
      const auto& d=result.diagnostics;need(d.setup_count==int(end-begin)&&d.worker_count==workers&&d.setup_stream_count==int(end-begin),"worker/cohort shape changed");
      r.native_ms+=call_ms;r.native_report_ms+=d.wall_host_ms;r.peak=std::max(r.peak,d.max_host_calls_in_flight);
      calls<<r.id<<','<<variant<<','<<r.calls++<<','<<begin<<','<<end-begin<<','<<workers<<','<<call_ms<<','<<d.wall_host_ms<<','<<d.max_host_calls_in_flight<<'\n';
      if(variant=='P'){
        verify(result,begin,end);result={};t=Clock::now();for(size_t i=begin;i<end;++i)pool[i].reset();
        r.destroy_ms+=ms(t);
      }else retained.push_back(std::move(result));
    }
    const double boundary_ms=ms(whole_start);nvtxRangePop();
    // A/R end their primary timing before diagnostic shadow downloads.
    if(variant!='P')for(size_t j=0;j<groups.size();++j)verify(retained[j],groups[j].first,groups[j].second);
    retained.clear();
    r.raw_ms=ms(whole_start);r.work_ms=variant=='P'?boundary_ms-r.verify_ms:boundary_ms+r.token_release_ms;
    observe_end(prefix+"-run"+std::to_string(r.id)+"-timing.csv");
    need(r.targets==(mode=="memcheck"?std::accumulate(inputs.begin(),inputs.end(),0,[](int n,const Request* v){return n+v->keys.size();}):6816),"target coverage");
    std::cout<<label<<" run="<<r.id<<" native_ms="<<r.native_ms<<" work_ms="<<r.work_ms
      <<" verify_ms="<<r.verify_ms<<" PASS targets="<<r.targets<<std::endl;
    return r;
  };
  auto cycle=[&](const std::string& order,const std::string& kind,int number,bool observed){
    std::vector<Run> rows;double prepare=0,destroy=0;int resident_done=0;
    for(char variant:order){
      if(variant!='P'&&resident_done==0){closed();prepare=create_pool();}
      if(variant=='P')closed();
      rows.push_back(evaluate(variant,kind,number,observed));
      if(variant!='P'&&++resident_done==2)destroy=destroy_pool();
      if(variant=='P')closed();
    }
    need(resident_done==2,"resident pair incomplete");
    for(auto& r:rows){if(r.variant!='P'){r.pool_prepare_ms=prepare;r.pool_destroy_ms=destroy;}write_run(r);}
    closed();
  };
  if(mode=="memcheck")cycle("PAR","memcheck",0,false);
  else {
    cycle("PAR","warmup",-1,false);
    if(mode=="campaign"){
      const std::vector<std::string> orders={"PAR","RAP","ARP","PRA","PAR","RAP"};
      for(size_t i=0;i<orders.size();++i)cycle(orders[i],"timing",i,false);
    }else {
      check(cudaProfilerStart());cycle("PAR","profile",0,true);check(cudaProfilerStop());
    }
  }
  closed();const auto c=layer_audit::counts();size_t free_end=0;check(cudaMemGetInfo(&free_end,&total));
  std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
    <<"{\"status\":\"PASS\",\"setups\":"<<count<<",\"runs\":"<<total_runs<<",\"compared_target_occurrences\":"<<total_targets
    <<",\"peak_application_bytes\":"<<c.peak_bytes<<",\"minimum_free_bytes\":"<<minimum_free
    <<",\"free_after_release\":"<<free_end<<",\"forbidden_calls\":"<<c.forbidden_calls
    <<",\"output_leases_closed\":true,\"resources_closed\":true,\"all_outputs_bitwise_exact\":true}\n";
  need(bool(summary)&&bool(runs)&&bool(calls)&&bool(setup_log),"artifact write failed");return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
