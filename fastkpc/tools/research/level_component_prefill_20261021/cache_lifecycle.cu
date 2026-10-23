// Resource and fail-closed tests of the new cache API, using a captured residual
// as an input fixture. This is not a numerical producer qualification.
#include "full_cuda_ci_vertical.cu"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <functional>
#include <stdexcept>

using namespace fastkpc;
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void rejects(const std::function<void()>& f,const char* message) {
  bool rejected=false;
  try { f(); } catch(const std::runtime_error& e) {
    rejected=std::string(e.what()).find(message)!=std::string::npos;
  }
  require(rejected,"expected fail-closed rejection was missing");
}
int main(int argc,char** argv) {
  require(argc==2,"usage: cache_lifecycle captured.residual.f64");
  std::vector<double> input(702);
  std::ifstream file(argv[1],std::ios::binary);
  file.read(reinterpret_cast<char*>(input.data()),351*sizeof(double));
  require(file.gcount()==351*sizeof(double),"captured residual shape changed");
  std::copy(input.begin(),input.begin()+351,input.begin()+351);
  check(cudaSetDevice(0));
  double* device=nullptr; cudaStream_t stream=nullptr; cudaEvent_t ready=nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device),input.size()*sizeof(double)));
  check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
  check(cudaMemcpyAsync(device,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
  check(cudaEventRecord(ready,stream));
  const auto before=full_cuda_ci_vertical_resource_snapshot();
  {
    auto cache=create_full_cuda_ci_legacy_eig_component_cache(351,35,1U<<20,true,22,484,8);
    const auto capacity=full_cuda_ci_legacy_eig_component_cache_info(cache).capacity_entries;
    rejects([&]{begin_full_cuda_ci_level_component_prefill(cache,3,capacity+1);},"capacity/lifecycle");
    rejects([&]{build_full_cuda_ci_level_components(cache);},"coverage");
    MultiPenaltyGcvCudaResidualConsumerView view;
    view.residuals=device; view.n=351; view.coefficient_dim=28; view.target_count=2;
    view.device_id=0; view.producer_stream=stream; view.producer_completion_event=ready;
    view.fixed_sp_evaluation=true;
    view.target_keys={std::string(64,'a'),std::string(64,'b')};
    view.solver_info={0,0}; view.decomposition_route={kMultiPenaltyGcvDecompositionStableSvd,
      kMultiPenaltyGcvDecompositionStableSvd}; view.numerical_rank={28,28};
    for(int level=3;level<=5;++level) {
      begin_full_cuda_ci_level_component_prefill(cache,level,2);
      rejects([&]{begin_full_cuda_ci_level_component_prefill(cache,level,2);},"capacity/lifecycle");
      rejects([&]{build_full_cuda_ci_level_components(cache);},"coverage");
      store_full_cuda_ci_level_component_residual(cache,view);
      build_full_cuda_ci_level_components(cache);
      rejects([&]{store_full_cuda_ci_level_component_residual(cache,view);},"identity");
      rejects([&]{build_full_cuda_ci_level_components(cache);},"coverage");
      if(level==3) {
        // A singleton first use, then a repeated resolution rejection. One
        // entry remains pending and must be discarded at the next reset.
        cache->resolve_gram({view.target_keys[0]});
        rejects([&]{cache->resolve_gram({view.target_keys[0]});},"twice");
      } else if(level==4) {
        cache->resolve_gram(view.target_keys);
      }
      finish_full_cuda_ci_level_component_prefill(cache);
      rejects([&]{finish_full_cuda_ci_level_component_prefill(cache);},"lifecycle");
      const auto info=full_cuda_ci_legacy_eig_component_cache_info(cache);
      const int resolved=level==3?1:(level==4?2:0);
      require(info.level_prefill.back().pending_at_end==2-resolved &&
        info.level_prefill.back().resolved==resolved,
        "unconsumed pending entries were not preserved");
    }
    const auto info=full_cuda_ci_legacy_eig_component_cache_info(cache);
    require(info.reset_count==3 && info.discarded_pending==1 && info.discarded_complete==3 &&
      info.entries==2 && info.insert_count==6,"reset/insert conservation changed");
  }
  // Unwind a partially populated window on invalid rank and duplicate keys.
  for(bool duplicate:{false,true}) {
    auto cache=create_full_cuda_ci_legacy_eig_component_cache(351,35,1U<<20,true,22,484,8);
    begin_full_cuda_ci_level_component_prefill(cache,3,2);
    MultiPenaltyGcvCudaResidualConsumerView view;
    view.residuals=device; view.n=351; view.target_count=2; view.device_id=0;
    view.producer_stream=stream; view.producer_completion_event=ready; view.fixed_sp_evaluation=true;
    view.target_keys={std::string(64,'a'),std::string(64,duplicate?'a':'b')};
    view.solver_info={0,0}; view.decomposition_route={kMultiPenaltyGcvDecompositionStableSvd,
      kMultiPenaltyGcvDecompositionStableSvd}; view.numerical_rank={28,duplicate?28:0};
    rejects([&]{store_full_cuda_ci_level_component_residual(cache,view);},"metadata");
  }
  const auto after=full_cuda_ci_vertical_resource_snapshot();
  require(before.live_device_allocations==after.live_device_allocations &&
    before.live_device_bytes==after.live_device_bytes && before.live_streams==after.live_streams &&
    before.live_events==after.live_events,"cache API leaked resources");
  check(cudaEventDestroy(ready)); check(cudaStreamDestroy(stream)); check(cudaFree(device));
  std::cout << "PASS capacity, lifecycle, pending reset, invalid metadata, error teardown\n";
  return 0;
}
