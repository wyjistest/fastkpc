// Canonical 48-wide source-shared integration test. Extra D2H is diagnostic only.
#define FASTKPC_STRUCTURED_TESTING 1
#include "full_cuda_ci_vertical.cu"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <functional>

using namespace fastkpc;
void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
void rejects(const std::function<void()>& f,const char* part) {
  bool caught=false;
  try {f();} catch(const std::exception& e) {caught=std::string(e.what()).find(part)!=std::string::npos;}
  require(caught,"expected rejection missing");
}
std::string key(int i) {
  std::ostringstream s; s<<std::hex<<std::setw(64)<<std::setfill('0')<<i+1; return s.str();
}
std::vector<double> load(const char* path,size_t count) {
  std::vector<double> x(count); std::ifstream f(path,std::ios::binary);
  f.read(reinterpret_cast<char*>(x.data()),count*sizeof(double));
  require(f.gcount()==count*sizeof(double),"fixture length mismatch"); return x;
}
struct Input {
  VerticalCallAccounting accounting;
  TrackedDeviceBuffer buffer;
  TrackedCudaStream stream;
  TrackedCudaEvent ready;
  Input(const std::vector<double>& x):buffer(x.size()*sizeof(double),&accounting) {
    check_cuda(cudaMemcpyAsync(buffer.get(),x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice,
      stream.get()),"fixture H2D");
    check_cuda(cudaEventRecord(ready.get(),stream.get()),"fixture ready");
    check_cuda(cudaEventSynchronize(ready.get()),"finish fixture input");
  }
  void populate(const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache,int level,int count) {
    cache->begin_level(level,count);
    for(int off=0;off<count;off+=16) {
      MultiPenaltyGcvCudaResidualConsumerView v;
      v.n=351; v.device_id=0; v.target_count=std::min(16,count-off);
      v.residuals=static_cast<const double*>(buffer.get())+size_t(off)*351;
      v.producer_stream=stream.get();v.producer_completion_event=ready.get();v.fixed_sp_evaluation=true;
      for(int i=0;i<v.target_count;++i) {
        v.target_keys.push_back(key(off+i)); v.solver_info.push_back(0);
        v.decomposition_route.push_back(kMultiPenaltyGcvDecompositionStableSvd);v.numerical_rank.push_back(28);
      }
      cache->store_residual(v);
    }
  }
};
std::vector<double> shadow(const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& c,int count) {
  const int fields=351*35+35+1;
  std::vector<double> out(size_t(count)*fields);
  for(int i=0;i<count;++i) {
    size_t slot=0;FixedSpRoute route;FixedSpStatus status;
    require(c->lookup(key(i),&slot,&route,&status),"missing produced identity");
    require(route==FixedSpRoute::AugmentedSvd && status==FixedSpStatus::OkAugmentedSvd,"residual route changed");
    check_cuda(cudaMemcpy(out.data()+size_t(i)*fields,c->vectors(slot),fields*sizeof(double),
      cudaMemcpyDeviceToHost),"diagnostic three fields");
  }
  return out;
}
int main(int argc,char** argv) {
  require(argc==3 || argc==4,"usage: lifecycle residuals.f64 three-fields.f64 [smoke]");
  const int real_count=argc==4?4:640;
  check_cuda(cudaSetDevice(0),"select test device");
  const auto before=full_cuda_ci_vertical_resource_snapshot();
  auto x=load(argv[1],640*351); auto expected=load(argv[2],640*(351*35+35+1));
  {
    Input input(x);
    auto cache=create_full_cuda_ci_legacy_eig_component_cache(351,35,96U<<20,true,48,2304,8);
    cache->configure_structured(true);
    input.populate(cache,3,real_count);
    rejects([&]{size_t slot;FixedSpRoute r;FixedSpStatus s;cache->lookup(key(0),&slot,&r,&s);},"unfinished");
    cache->build_level();
    auto actual=shadow(cache,real_count);
    require(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(double))==0,
      "production pool differs from qualified adapter bytes");
    auto s=cache->info().level_prefill.back();
    require(s.structured_accepted==real_count && s.dense_fallback==0 && s.structured_peak_inflight==real_count,
      "pool coverage changed");
    cache->resolve_gram({key(0)});
    rejects([&]{cache->resolve_gram({key(0)});},"twice");
    cache->finish_level();
    input.populate(cache,4,1);cache->build_level();cache->finish_level();
    require(cache->info().discarded_pending==real_count-1,"pending reset changed");
  }
  // First 56 of 70 are true small-gap inputs. They exceed dense capacity 48.
  x.resize(70*351);
  for(int c=0;c<56;++c) for(int r=0;r<351;++r) x[size_t(c)*351+r]=r<175?-1.0:1.0;
  {
    Input input(x);
    auto dense=create_full_cuda_ci_legacy_eig_component_cache(351,35,16U<<20,true,48,2304,8);
    auto mixed=create_full_cuda_ci_legacy_eig_component_cache(351,35,16U<<20,true,48,2304,8);
    mixed->configure_structured(true);
    input.populate(dense,3,70);dense->build_level();
    input.populate(mixed,3,70);mixed->build_level();
    auto a=shadow(dense,70),b=shadow(mixed,70);
    require(std::memcmp(a.data(),b.data(),56*(351*35+35+1)*sizeof(double))==0,"fallback payload bits changed");
    auto s=mixed->info().level_prefill.back();
    require(s.fallback_gap==56 && s.dense_fallback_batches==2 && s.structured_accepted==14,
      "bounded mixed fallback partition changed");
    dense->finish_level();mixed->finish_level();
    auto bad=create_full_cuda_ci_legacy_eig_component_cache(351,35,16U<<20,true,48,2304,8);
    bad->configure_structured(true);bad->test_dense_failure=true;
    input.populate(bad,3,70);
    rejects([&]{bad->build_level();},"no wave published");
    require(bad->info().entries==0,"failed wave published entries");
    rejects([&]{size_t slot;FixedSpRoute r;FixedSpStatus s;bad->lookup(key(0),&slot,&r,&s);},"failed closed");
    rejects([&]{bad->build_level();},"coverage");
  }
  {
    // Actual rotation overflow, plus destruction with an abandoned submission.
    Input input(x);
    VerticalCallAccounting accounting;
    StructuredComponentLane lane(351,35,1,&accounting,1);
    lane.submit(static_cast<double*>(input.buffer.get())+56*351,1,input.ready.get());
    lane.read_status(1);
    require(lane.statuses[0].reason==StructuredRotationOverflow,"rotation limit did not trigger fallback");
    lane.submit(static_cast<double*>(input.buffer.get())+56*351,1,input.ready.get());
  }
  const auto after=full_cuda_ci_vertical_resource_snapshot();
  require(before.live_device_allocations==after.live_device_allocations &&
    before.live_device_bytes==after.live_device_bytes && before.live_streams==after.live_streams &&
    before.live_events==after.live_events,"structured integration resource leak");
  std::cout<<"PASS "<<real_count<<" adapter bytes, original Gram, pending reset, 56/70 mixed GPU fallback, "
    "48-capacity chunking, failure no-publication, rotation overflow, abandoned submit, resource closure\n";
}

