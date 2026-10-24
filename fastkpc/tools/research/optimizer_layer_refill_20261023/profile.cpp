// Separate A/R Nsight processes, with all CUDA resources created before capture.
#include "layer_fixture.hpp"
#include "audit.hpp"
#include "observe.hpp"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <iostream>
#include <iomanip>
using namespace layer_diag;
using namespace fastkpc;
using Clock=std::chrono::steady_clock;
void check(cudaError_t status){need(status==cudaSuccess,cudaGetErrorString(status));}
int main(int argc,char** argv){try{
  need(argc==4,"profile RAW PREFIX A|R");const std::string prefix=argv[2],variant=argv[3];
  need(variant=="A"||variant=="R","profile variant");
  auto windows=read_layer(argv[1]);std::vector<const Request*> inputs;std::vector<std::string> keys;
  for(const auto& w:windows)for(const auto& r:w.requests){inputs.push_back(&r);keys.push_back(r.keys.front());}
  check(cudaSetDevice(0));check(cudaFree(nullptr));size_t free=0,total=0;
  check(cudaMemGetInfo(&free,&total));const size_t headroom=size_t(2)*1024*1024*1024;
  need(free>headroom,"profile initial capacity");layer_audit::memory_limit(free-headroom);
  std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> pool;
  for(const auto* r:inputs){const auto& g=r->geometry;
    pool.push_back(create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),
      g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0));
    check(cudaMemGetInfo(&free,&total));need(free>=headroom,"profile headroom");}
  int compared=0;double wall[2]{};
  for(int run=0;run<2;++run){
    const size_t width=variant=="A"?64:inputs.size();const int workers=variant=="A"?64:32;
    std::vector<std::vector<MultiPenaltyGcvCapacityRequest>> groups;
    for(size_t begin=0;begin<inputs.size();begin+=width){std::vector<MultiPenaltyGcvCapacityRequest> group;
      for(size_t i=begin;i<std::min(begin+width,inputs.size());++i){const auto& r=*inputs[i];MultiPenaltyGcvCapacityRequest v;
        v.prepared=pool[i];v.Y=r.Y;v.n=r.geometry.n;v.target_count=r.keys.size();v.target_keys=r.keys;v.control=r.control;group.push_back(std::move(v));}
      groups.push_back(std::move(group));}
    std::vector<MultiPenaltyGcvCapacityMultiResult> results;
    observe_begin(keys,run==1);if(run==1)check(cudaProfilerStart());
    const auto label="layer:L5:"+variant+":"+(run==1?"profile":"warmup");nvtxRangePushA(label.c_str());
    const auto started=Clock::now();layer_audit::native_scope(true);
    try{for(auto& group:groups){const int n=group.size();results.push_back(multi_penalty_gcv_capacity_optimize_multi(std::move(group),std::min(workers,n),true));}}
    catch(...){layer_audit::native_scope(false);throw;}
    layer_audit::native_scope(false);wall[run]=std::chrono::duration<double>(Clock::now()-started).count();
    nvtxRangePop();if(run==1)check(cudaProfilerStop());observe_end("");
    size_t index=0;for(auto& result:results)for(size_t j=0;j<result.setups.size();++j,++index){
      auto& residual=result.residuals[j];const auto& r=*inputs[index];
      need(multi_penalty_gcv_capacity_residual_info(residual).target_keys==r.keys,"profile target identity");
      need(snapshot(result.setups[j],residual).same(r.expected),"profile bitwise mismatch");
      compared+=r.keys.size();release_multi_penalty_gcv_capacity_residual(residual);residual.reset();
      need(!multi_penalty_gcv_capacity_prepared_info(pool[index]).residual_slot_leased,"profile lease leak");}
    need(index==955,"profile coverage");results.clear();
    std::cout<<label<<" PASS seconds="<<wall[run]<<std::endl;
  }
  pool.clear();check(cudaDeviceSynchronize());const auto c=layer_audit::counts();
  need(c.allocations==0&&c.bytes==0&&c.streams==0&&c.events==0&&c.blas==0&&c.forbidden_calls==0,"profile resource gate");
  std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
    <<"{\"status\":\"PASS\",\"setups\":955,\"runs\":2,\"compared_target_occurrences\":"<<compared
    <<",\"warmup_sec\":"<<wall[0]<<",\"profile_sec\":"<<wall[1]
    <<",\"forbidden_calls\":0,\"resources_closed\":true,\"output_leases_closed\":true,\"all_outputs_bitwise_exact\":true}\n";
  need(bool(summary),"profile summary write");return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
