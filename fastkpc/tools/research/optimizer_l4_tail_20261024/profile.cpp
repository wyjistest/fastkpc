// L4 attribution only. Execute the actual production native in original windows.
#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include "../optimizer_windows_20261022/observe.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
using namespace l4_diag;
using namespace fastkpc;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
void check(cudaError_t status) { need(status==cudaSuccess,cudaGetErrorString(status)); }
void closed() {
  const auto c=layer_audit::counts();
  need(!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,"resources not closed");
}
struct Range { explicit Range(const std::string& text){nvtxRangePushA(text.c_str());} ~Range(){nvtxRangePop();} };
struct NativeScope { NativeScope(){layer_audit::native_scope(true);} ~NativeScope(){layer_audit::native_scope(false);} };
int main(int argc,char** argv) { try {
  need(argc==3,"profile RAW PREFIX");
  const std::string prefix=argv[2];const auto layer=read_layer(argv[1]);
  need(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H")&&
       std::string(std::getenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H"))=="1","status wait required");
  std::vector<std::string> keys;std::map<int,int> capacities;
  std::ofstream inputs(prefix+"-inputs.csv");inputs<<"setup,window,targets,iterations\n";
  size_t index=0;int total_targets=0;
  for(const auto& w:layer)for(const auto& r:w.requests){
    keys.push_back(r.keys.front());++capacities[r.geometry.capacity];total_targets+=r.keys.size();
    const auto& it=r.expected.integral.at(3);
    inputs<<index++<<','<<w.window<<','<<r.keys.size()<<','<<std::accumulate(it.begin(),it.end(),0)<<'\n';
  }
  need(keys.size()==2154&&total_targets==17767&&capacities.size()==14,"capture coverage");
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();
  size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  const size_t headroom=size_t(2)*1024*1024*1024;need(free>headroom,"initial headroom");
  layer_audit::memory_limit(free-headroom);size_t minimum_free=free;
  std::ofstream rows(prefix+"-windows.csv");rows<<std::setprecision(17)
    <<"pass,window,setups,targets,iterations,workers,peak_host_calls,prepare_sec,pack_sec,native_sec,reported_sec,verify_sec,release_sec,destroy_sec\n";
  double native_totals[2]{},lifecycle_totals[2]{};int compared=0;
  for(int run=0;run<2;++run){
    const std::string pass=run?"profile":"warmup";
    observe_begin(keys,run==1);if(run)check(cudaProfilerStart());
    {
      Range layer_range("layer:L4:"+pass);
      for(const auto& w:layer){
        std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> pool;
        auto start=Clock::now();
        {
          Range range("prepare:W"+std::to_string(w.window));
          for(const auto& r:w.requests){const auto& g=r.geometry;
            pool.push_back(create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),
              g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0));
            check(cudaMemGetInfo(&free,&total));minimum_free=std::min(minimum_free,free);
            need(free>=headroom,"window headroom");
          }
        }
        const double prepare=seconds(start);start=Clock::now();
        std::vector<MultiPenaltyGcvCapacityRequest> requests;
        for(size_t j=0;j<w.requests.size();++j){const auto& r=w.requests[j];MultiPenaltyGcvCapacityRequest v;
          v.prepared=pool[j];v.Y=r.Y;v.n=r.geometry.n;v.target_count=r.keys.size();
          v.target_keys=r.keys;v.control=r.control;requests.push_back(std::move(v));}
        const double pack=seconds(start);MultiPenaltyGcvCapacityMultiResult result;
        const auto before=layer_audit::counts();start=Clock::now();
        {Range range("window:W"+std::to_string(w.window));NativeScope scope;
          result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),32,true);}
        const double native=seconds(start);const auto after=layer_audit::counts();
        need(after.forbidden_calls==before.forbidden_calls,"native resource mutation");
        const auto& d=result.diagnostics;
        need(d.setup_count==int(w.requests.size())&&d.worker_count==32&&d.setup_stream_count==int(w.requests.size()),"native shape");
        need(result.setups.size()==w.requests.size()&&result.residuals.size()==w.requests.size(),"output coverage");
        start=Clock::now();int targets=0,iterations=0;
        {
          Range range("verify:W"+std::to_string(w.window));
          for(size_t j=0;j<w.requests.size();++j){const auto& r=w.requests[j];
            const auto info=multi_penalty_gcv_capacity_residual_info(result.residuals[j]);
            need(info.target_keys==r.keys&&info.device_resident&&!info.released,"target identity");
            need(snapshot(result.setups[j],result.residuals[j]).same(r.expected),"BITWISE_MISMATCH W"+std::to_string(w.window)+" setup "+std::to_string(j));
            targets+=r.keys.size();const auto& it=result.setups[j].optimizer_iterations;
            iterations+=std::accumulate(it.begin(),it.end(),0);
          }
        }
        const double verify=seconds(start);start=Clock::now();
        {
          Range range("release:W"+std::to_string(w.window));
          for(size_t j=0;j<pool.size();++j){release_multi_penalty_gcv_capacity_residual(result.residuals[j]);
            result.residuals[j].reset();need(!multi_penalty_gcv_capacity_prepared_info(pool[j]).residual_slot_leased,"lease remains");}
        }
        const double release=seconds(start);start=Clock::now();
        {Range range("destroy:W"+std::to_string(w.window));pool.clear();check(cudaDeviceSynchronize());closed();}
        const double destroy=seconds(start);
        rows<<pass<<','<<w.window<<','<<w.requests.size()<<','<<targets<<','<<iterations<<",32,"<<d.max_host_calls_in_flight<<','
          <<prepare<<','<<pack<<','<<native<<','<<d.wall_host_ms/1000<<','<<verify<<','<<release<<','<<destroy<<'\n';rows.flush();
        native_totals[run]+=native;lifecycle_totals[run]+=prepare+pack+native+release+destroy;compared+=targets;
      }
    }
    if(run)check(cudaProfilerStop());observe_end("");closed();
    std::cout<<std::setprecision(12)<<"PASS "<<pass<<" native_sec="<<native_totals[run]
      <<" segmented_lifecycle_sec="<<lifecycle_totals[run]<<" targets=17767"<<std::endl;
  }
  const auto c=layer_audit::counts();need(!c.forbidden_calls,"forbidden calls");
  std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
    <<"{\"status\":\"PASS\",\"setups_per_pass\":2154,\"targets_per_pass\":17767,\"windows_per_pass\":34,\"passes\":2,\"compared_target_occurrences\":"<<compared
    <<",\"unprofiled_warmup_native_sec\":"<<native_totals[0]<<",\"profiled_native_sec\":"<<native_totals[1]
    <<",\"unprofiled_segmented_lifecycle_sec\":"<<lifecycle_totals[0]<<",\"profiled_segmented_lifecycle_sec\":"<<lifecycle_totals[1]
    <<",\"peak_application_bytes\":"<<c.peak_bytes<<",\"minimum_free_after_create\":"<<minimum_free
    <<",\"forbidden_calls\":0,\"resources_closed\":true,\"output_leases_closed\":true,\"all_outputs_bitwise_exact\":true,\"production_cache_present\":false}\n";
  need(bool(inputs)&&bool(rows)&&bool(summary),"output write failed");return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
