// Reuse every existing source-shared lifecycle test, then test device restoration
// on success and exception paths with the cache on GPU 0 and caller on GPU 1.
#define main previous_lifecycle_main
#include "../level_component_prefill_20261021/cache_lifecycle.cu"
#undef main
int main(int argc,char** argv) {
  const int previous=previous_lifecycle_main(argc,argv);
  require(previous==0,"prior lifecycle tests failed");
  int count=0; check(cudaGetDeviceCount(&count)); require(count>=2,"device guard test needs two visible devices");
  check(cudaSetDevice(0));
  std::vector<double> input(351);
  std::ifstream file(argv[1],std::ios::binary);
  file.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(double));
  require(file.gcount()==static_cast<std::streamsize>(input.size()*sizeof(double)),"fixture truncated");
  double* device=nullptr; cudaStream_t stream=nullptr; cudaEvent_t ready=nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&device),input.size()*sizeof(double)));
  check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
  check(cudaMemcpyAsync(device,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
  check(cudaEventRecord(ready,stream));
  const auto before=full_cuda_ci_vertical_resource_snapshot();
  {
    auto cache=create_full_cuda_ci_legacy_eig_component_cache(351,35,1U<<20,true,48,2304,8);
    MultiPenaltyGcvCudaResidualConsumerView view;
    view.residuals=device; view.n=351; view.coefficient_dim=28; view.target_count=1;
    view.device_id=0; view.producer_stream=stream; view.producer_completion_event=ready;
    view.fixed_sp_evaluation=true; view.target_keys={std::string(64,'a')};
    view.solver_info={0}; view.decomposition_route={kMultiPenaltyGcvDecompositionStableSvd};
    view.numerical_rank={28};
    auto restored=[] { int device=-1; check(cudaGetDevice(&device)); require(device==1,"prefill leaked current device"); };
    check(cudaSetDevice(1));
    begin_full_cuda_ci_level_component_prefill(cache,3,1); restored();
    auto bad=view; bad.numerical_rank[0]=0;
    rejects([&]{store_full_cuda_ci_level_component_residual(cache,bad);},"metadata"); restored();
    store_full_cuda_ci_level_component_residual(cache,view); restored();
    build_full_cuda_ci_level_components(cache); restored();
    cache->resolve_gram(view.target_keys); restored();
    finish_full_cuda_ci_level_component_prefill(cache); restored();
    cache->finish_gram_timing(); restored();
    rejects([&]{finish_full_cuda_ci_level_component_prefill(cache);},"lifecycle"); restored();
    begin_full_cuda_ci_level_component_prefill(cache,4,1); restored();
    store_full_cuda_ci_level_component_residual(cache,view); restored();
    build_full_cuda_ci_level_components(cache); restored();
    cache->resolve_gram(view.target_keys); restored();
    rejects([&]{cache->resolve_gram(view.target_keys);},"twice"); restored();
    cache.reset(); restored();
  }
  const auto after=full_cuda_ci_vertical_resource_snapshot();
  require(before.live_device_allocations==after.live_device_allocations &&
    before.live_device_bytes==after.live_device_bytes && before.live_streams==after.live_streams &&
    before.live_events==after.live_events,"device guard test leaked native resources");
  check(cudaSetDevice(0));
  check(cudaEventDestroy(ready)); check(cudaStreamDestroy(stream)); check(cudaFree(device));
  std::cout << "PASS caller device restored on success, exception and destruction\n";
}
