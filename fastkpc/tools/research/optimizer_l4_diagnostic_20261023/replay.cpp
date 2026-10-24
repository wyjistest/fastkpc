// Captured L4 replay. Link the actual production native; no numerical clone.
#include "fixture.hpp"
#include "audit.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
using namespace l4_diag;
using namespace fastkpc;
using Clock=std::chrono::steady_clock;
using Prepared=std::shared_ptr<MultiPenaltyGcvCapacityPrepared>;
using Result=MultiPenaltyGcvCapacityMultiResult;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void check(cudaError_t s){need(s==cudaSuccess,cudaGetErrorString(s));}
void closed(){auto c=layer_audit::counts();need(!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,"resources not closed");}
struct Scope{Scope(){layer_audit::native_scope(true);}~Scope(){layer_audit::native_scope(false);}};
struct Run{
  char variant='A';int cycle=0,setups=0,targets=0,iterations=0,calls=0;
  std::string kind;double native=0,reported=0,pack=0,verify=0,tokens=0,prepare=0,destroy=0;
  size_t forbidden=0;
};
int main(int argc,char** argv){
 try{
  need(argc==5,"replay RAW PREFIX capacity|memcheck|campaign pending|resident|windows");
  const std::string prefix=argv[2],mode=argv[3],policy=argv[4];
  need(mode=="capacity"||mode=="memcheck"||mode=="campaign","unknown mode");
  need(mode=="capacity"||policy=="resident"||policy=="windows","unknown scheduling policy");
  need(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H")&&
    std::string(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H"))=="1","status wait must be on");
  const auto layer=read_layer(argv[1]);
  std::vector<const Request*> inputs;std::vector<std::vector<size_t>> windows;
  std::vector<int> original_window;std::map<int,int> capacities,root_ranks;
  int full_targets=0;
  for(const auto& w:layer){
    std::vector<size_t> group;
    for(const auto& r:w.requests){
      group.push_back(inputs.size());inputs.push_back(&r);original_window.push_back(w.window);
      ++capacities[r.geometry.capacity];full_targets+=r.keys.size();
      ++root_ranks[std::accumulate(r.geometry.ranks.begin(),r.geometry.ranks.end(),0)];
    }
    windows.push_back(std::move(group));
  }
  const size_t count=inputs.size();need(count==2154&&full_targets==17767,"full capture coverage");
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();
  size_t free_start=0,total=0;check(cudaMemGetInfo(&free_start,&total));
  const size_t headroom=size_t(2)*1024*1024*1024;
  need(free_start>headroom,"initial headroom missing");layer_audit::memory_limit(free_start-headroom);
  size_t minimum_free=free_start;
  std::ofstream allocations(prefix+"-allocations.csv");
  allocations<<"phase,setup,window,targets,application_bytes,free_before,free_after,prepare_ms\n"<<std::setprecision(17);
  std::vector<Prepared> pool(count);
  auto create=[&](size_t i,const std::string& phase){
    need(!pool[i],"prepared already exists");size_t before=0,after=0;check(cudaMemGetInfo(&before,&total));
    need(before>=headroom,"create starts below headroom");
    const auto c=layer_audit::counts();const auto start=Clock::now();const auto& g=inputs[i]->geometry;
    pool[i]=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),
      g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);
    const double elapsed=ms(start);check(cudaMemGetInfo(&after,&total));minimum_free=std::min(minimum_free,after);
    const auto d=layer_audit::counts();need(d.bytes>c.bytes&&d.blas==c.blas+1&&d.events>=c.events+2,
      "resource interposition incomplete");
    allocations<<phase<<','<<i<<','<<original_window[i]<<','<<g.capacity<<','<<d.bytes-c.bytes<<','
      <<before<<','<<after<<','<<elapsed<<'\n';allocations.flush();
    return std::make_pair(d.bytes-c.bytes,after);
  };
  auto destroy=[&](const std::vector<size_t>& indices){
    const auto start=Clock::now();for(size_t i:indices){
      if(pool[i])need(!multi_penalty_gcv_capacity_prepared_info(pool[i]).residual_slot_leased,"destroy with live lease");
      pool[i].reset();
    }
    check(cudaDeviceSynchronize());return ms(start);
  };
  std::vector<size_t> all(count);std::iota(all.begin(),all.end(),0);
  if(mode=="capacity"){
    size_t application_sum=0,application_max=0;
    for(size_t i:all){
      const auto measured=create(i,"individual-census");
      application_sum+=measured.first;application_max=std::max(application_max,measured.first);
      need(measured.second>=headroom,"individual cohort exceeds memory gate");destroy({i});closed();
    }
    size_t resident_before=0;check(cudaMemGetInfo(&resident_before,&total));
    bool fits=true;size_t created=0;const auto start=Clock::now();
    for(size_t i:all){
      const auto measured=create(i,"resident-attempt");++created;
      if(measured.second<headroom){fits=false;break;}
    }
    const double prepare=ms(start);const auto live=layer_audit::counts();
    size_t resident_free=0;check(cudaMemGetInfo(&resident_free,&total));
    const double release=destroy(all);closed();size_t free_end=0;check(cudaMemGetInfo(&free_end,&total));
    std::ofstream out(prefix+"-summary.json");out<<std::setprecision(17)
      <<"{\"status\":\""<<(fits?"PASS":"STOP_FULL_LAYER_CAPACITY")<<"\",\"setups\":"<<count
      <<",\"targets\":"<<full_targets<<",\"windows\":34,\"census_setups\":"<<count
      <<",\"application_bytes_sum\":"<<application_sum<<",\"application_bytes_max_cohort\":"<<application_max
      <<",\"resident_created\":"<<created<<",\"resident_application_bytes\":"<<live.bytes
      <<",\"resident_streams\":"<<live.streams<<",\"resident_events\":"<<live.events<<",\"resident_blas\":"<<live.blas
      <<",\"free_start\":"<<free_start<<",\"free_before_resident\":"<<resident_before
      <<",\"free_resident_boundary\":"<<resident_free<<",\"free_after_release\":"<<free_end
      <<",\"headroom_required\":"<<headroom<<",\"prepare_ms\":"<<prepare<<",\"release_ms\":"<<release
      <<",\"resources_closed\":true,\"production_cache_present\":false,\"target_capacities\":{";
    bool first=true;for(auto [capacity,number]:capacities){if(!first)out<<',';first=false;out<<'"'<<capacity<<"\":"<<number;}
    out<<"},\"root_rank_sums\":{";first=true;
    for(auto [rank,number]:root_ranks){if(!first)out<<',';first=false;out<<'"'<<rank<<"\":"<<number;}
    out<<"}}\n";need(bool(out)&&bool(allocations),"capacity output failed");
    std::cout<<(fits?"PASS":"STOP_FULL_LAYER_CAPACITY")<<" resident_created="<<created<<'/'<<count
      <<" free="<<resident_free<<" application_sum="<<application_sum<<" resources_closed=1\n";return 0;
  }

  // Reduced sanitizer covers every actual target capacity and both endpoints.
  if(mode=="memcheck"){
    std::set<size_t> selected{0,count-1};std::set<int> seen;
    for(size_t i:all)if(seen.insert(inputs[i]->geometry.capacity).second)selected.insert(i);
    all.assign(selected.begin(),selected.end());
    std::vector<std::vector<size_t>> reduced;
    for(const auto& w:windows){std::vector<size_t> g;for(size_t i:w)if(selected.count(i))g.push_back(i);
      if(!g.empty())reduced.push_back(std::move(g));}
    windows=std::move(reduced);
  }
  const int expected_targets=std::accumulate(all.begin(),all.end(),0,[&](int n,size_t i){return n+inputs[i]->keys.size();});
  const char candidate=policy=="resident"?'R':'W';
  std::ofstream runs(prefix+"-runs.csv"),calls(prefix+"-calls.csv");
  runs<<"cycle,kind,variant,workers,setups,targets,iterations,calls,native_ms,reported_ms,pack_ms,verify_ms,token_release_ms,pool_prepare_ms,pool_destroy_ms,segmented_work_ms,lifecycle_work_ms,forbidden_calls,parity\n"<<std::setprecision(17);
  calls<<"cycle,kind,variant,first_setup,original_window,setups,targets,workers,native_ms,reported_ms,peak_host_calls\n"<<std::setprecision(17);
  int total_runs=0,total_compared=0;
  auto invoke=[&](Run& r,const std::vector<size_t>& indices){
    auto start=Clock::now();std::vector<MultiPenaltyGcvCapacityRequest> requests;
    for(size_t i:indices){const auto& in=*inputs[i];MultiPenaltyGcvCapacityRequest request;
      request.prepared=pool[i];request.Y=in.Y;request.n=in.geometry.n;request.target_count=in.keys.size();
      request.target_keys=in.keys;request.control=in.control;requests.push_back(std::move(request));}
    r.pack+=ms(start);const int workers=std::min(r.variant=='A'?64:32,int(indices.size()));
    const auto before=layer_audit::counts();start=Clock::now();Result result;
    {Scope scope;result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),workers,true);}
    const double elapsed=ms(start);const auto after=layer_audit::counts();
    r.forbidden+=after.forbidden_calls-before.forbidden_calls;
    need(after.forbidden_calls==before.forbidden_calls,"resource mutation inside optimizer");
    const auto& d=result.diagnostics;
    need(d.setup_count==int(indices.size())&&d.worker_count==workers&&d.setup_stream_count==int(indices.size()),
      "native worker/cohort shape changed");
    r.native+=elapsed;r.reported+=d.wall_host_ms;++r.calls;
    calls<<r.cycle<<','<<r.kind<<','<<r.variant<<','<<indices.front()<<','<<original_window[indices.front()]<<','
      <<indices.size()<<','<<d.target_count<<','<<workers<<','<<elapsed<<','<<d.wall_host_ms<<','<<d.max_host_calls_in_flight<<'\n';
    calls.flush();return result;
  };
  auto verify=[&](Run& r,Result& result,const std::vector<size_t>& indices){
    auto start=Clock::now();need(result.setups.size()==indices.size()&&result.residuals.size()==indices.size(),"result coverage");
    for(size_t j=0;j<indices.size();++j){size_t i=indices[j];const auto& in=*inputs[i];
      const auto info=multi_penalty_gcv_capacity_residual_info(result.residuals[j]);
      need(info.target_keys==in.keys&&info.device_resident&&!info.released,"target identity/order changed");
      need(snapshot(result.setups[j],result.residuals[j]).same(in.expected),"BITWISE_MISMATCH setup "+std::to_string(i));
      ++r.setups;r.targets+=in.keys.size();const auto& iterations=result.setups[j].optimizer_iterations;
      r.iterations+=std::accumulate(iterations.begin(),iterations.end(),0);
    }
    r.verify+=ms(start);start=Clock::now();
    for(size_t j=0;j<indices.size();++j){release_multi_penalty_gcv_capacity_residual(result.residuals[j]);result.residuals[j].reset();
      need(!multi_penalty_gcv_capacity_prepared_info(pool[indices[j]]).residual_slot_leased,"output lease remains");}
    r.tokens+=ms(start);
  };
  auto write=[&](const Run& r){
    need(r.setups==int(all.size())&&r.targets==expected_targets,"complete replay coverage failed");
    const double work=r.native+r.pack+r.tokens;
    runs<<r.cycle<<','<<r.kind<<','<<r.variant<<','<<(r.variant=='A'?64:32)<<','<<r.setups<<','<<r.targets<<','
      <<r.iterations<<','<<r.calls<<','<<r.native<<','<<r.reported<<','<<r.pack<<','<<r.verify<<','<<r.tokens<<','
      <<r.prepare<<','<<r.destroy<<','<<work<<','<<work+r.prepare+r.destroy<<','<<r.forbidden<<",1\n";
    runs.flush();++total_runs;total_compared+=r.targets;
    std::cout<<"PASS "<<r.kind<<" cycle="<<r.cycle<<" variant="<<r.variant<<" native_ms="<<r.native
      <<" targets="<<r.targets<<std::endl;
  };
  auto cycle=[&](int number,const std::string& kind,bool candidate_first){
    Run a,b;a.variant='A';b.variant=candidate;a.cycle=b.cycle=number;a.kind=b.kind=kind;
    std::vector<Run*> order=candidate_first?std::vector<Run*>{&b,&a}:std::vector<Run*>{&a,&b};
    if(policy=="resident"){
      auto start=Clock::now();for(size_t i:all)need(create(i,"resident-cycle").second>=headroom,"resident headroom changed");
      a.prepare=b.prepare=ms(start);
      for(auto* r:order){
        const auto groups=r->variant=='A'?windows:std::vector<std::vector<size_t>>{all};
        std::vector<Result> retained;for(const auto& g:groups)retained.push_back(invoke(*r,g));
        for(size_t j=0;j<groups.size();++j)verify(*r,retained[j],groups[j]);
      }
      a.destroy=b.destroy=destroy(all);closed();
    }else{
      for(const auto& g:windows){
        auto start=Clock::now();for(size_t i:g)need(create(i,"window-cycle").second>=headroom,"window headroom changed");
        const double preparation=ms(start);a.prepare+=preparation;b.prepare+=preparation;
        for(auto* r:order){auto result=invoke(*r,g);verify(*r,result,g);}
        const double release=destroy(g);a.destroy+=release;b.destroy+=release;closed();
      }
    }
    for(auto* r:order)write(*r);closed();
  };
  if(mode=="memcheck")cycle(0,"memcheck",false);
  else{cycle(-1,"warmup",false);for(int i=0;i<6;++i)cycle(i,"timing",i%2==1);}
  closed();const auto c=layer_audit::counts();size_t free_end=0;check(cudaMemGetInfo(&free_end,&total));
  std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
    <<"{\"status\":\"PASS\",\"policy\":\""<<policy<<"\",\"setups_per_run\":"<<all.size()
    <<",\"targets_per_run\":"<<expected_targets<<",\"runs\":"<<total_runs
    <<",\"compared_target_occurrences\":"<<total_compared<<",\"peak_application_bytes\":"<<c.peak_bytes
    <<",\"minimum_free_after_create\":"<<minimum_free<<",\"free_after_release\":"<<free_end
    <<",\"forbidden_calls\":"<<c.forbidden_calls
    <<",\"all_outputs_bitwise_exact\":true,\"output_leases_closed\":true,\"resources_closed\":true,\"selected_setups\":[";
  for(size_t j=0;j<all.size();++j){if(j)summary<<',';summary<<all[j];}
  summary<<"]}\n";need(bool(summary)&&bool(runs)&&bool(calls)&&bool(allocations),"replay output failed");return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
