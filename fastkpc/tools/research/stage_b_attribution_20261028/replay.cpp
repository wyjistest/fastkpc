#include "common.hpp"
#include <iostream>
using namespace fastkpc;
using namespace stage_b_diag;
int main(int argc,char** argv) {
  try {
    need(argc==2,"usage: replay CAPTURE_DIR");check(cudaSetDevice(0));
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    std::ofstream hardware(env("STAGE_B_OUTPUT")+"/hardware.json");
    hardware<<"{\"sm_count\":"<<prop.multiProcessorCount<<",\"max_threads_per_sm\":"<<prop.maxThreadsPerMultiProcessor
      <<",\"shared_per_sm\":"<<prop.sharedMemPerMultiprocessor<<",\"reserved_shared_per_block\":"<<prop.reservedSharedMemPerBlock
      <<",\"registers_per_sm\":"<<prop.regsPerMultiprocessor<<",\"warp_size\":"<<prop.warpSize<<"}\n";hardware.close();
    const auto before=full_cuda_ci_vertical_resource_snapshot();
    {
      auto cache=create_full_cuda_ci_legacy_eig_component_cache(n,k,11811160064ULL,true,48,2304,8);
      configure_full_cuda_ci_structured_prefill(cache,true);
      cudaStream_t stream=nullptr;cudaEvent_t event=nullptr;double* device=nullptr;
      check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
      check(cudaMalloc(&device,16*n*sizeof(double)));
      for(int level:{2,3}) {
        const size_t count=level==2?49919:54878;
        const auto prefix=std::string(argv[1])+"/l"+std::to_string(level);
        std::ifstream raw(prefix+"-residuals.f64",std::ios::binary),meta(prefix+"-targets.tsv");
        need(raw.good()&&meta.good(),"capture absent");
        struct Target {size_t cohort,index;std::string key;int info,route,rank;};
        std::vector<Target> rows;Target row;
        while(meta>>row.cohort>>row.index>>row.key>>row.info>>row.route>>row.rank)rows.push_back(row);
        need(rows.size()==count,"capture target coverage");
        begin_full_cuda_ci_level_component_prefill(cache,level,count);
        for(size_t off=0;off<count;) {
          size_t end=off+1;while(end<count&&rows[end].cohort==rows[off].cohort)++end;
          need(end-off<=16,"cohort changed");std::vector<double> h((end-off)*n);
          raw.read(reinterpret_cast<char*>(h.data()),h.size()*8);need(size_t(raw.gcount())==h.size()*8,"residual truncated");
          check(cudaMemcpyAsync(device,h.data(),h.size()*8,cudaMemcpyHostToDevice,stream));
          check(cudaEventRecord(event,stream));
          MultiPenaltyGcvCudaResidualConsumerView v;
          v.n=n;v.device_id=0;v.target_count=end-off;v.residuals=device;
          v.producer_stream=stream;v.producer_completion_event=event;v.fixed_sp_evaluation=true;
          for(size_t i=off;i<end;++i) {
            need(rows[i].index==i,"target order changed");v.target_keys.push_back(rows[i].key);
            v.solver_info.push_back(rows[i].info);v.decomposition_route.push_back(rows[i].route);
            v.numerical_rank.push_back(rows[i].rank);
          }
          store_full_cuda_ci_level_component_residual(cache,v);off=end;
        }
        need(raw.peek()==std::char_traits<char>::eof(),"residual trailing bytes");
        build_full_cuda_ci_level_components(cache);
        finish_full_cuda_ci_level_component_prefill(cache);
      }
      check(cudaFree(device));check(cudaEventDestroy(event));check(cudaStreamDestroy(stream));
    }
    const auto after=full_cuda_ci_vertical_resource_snapshot();
    need(before.live_device_allocations==0&&before.live_device_bytes==0&&before.live_streams==0&&before.live_events==0,
      "dirty initial resource ledger");
    need(after.live_device_allocations==0&&after.live_device_bytes==0&&after.live_streams==0&&after.live_events==0,
      "tracked resources leaked");
    std::cout<<"PASS native replay resource closure"<<std::endl;
  } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
