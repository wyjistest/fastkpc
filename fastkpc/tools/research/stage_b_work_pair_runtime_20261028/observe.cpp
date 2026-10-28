// Host-only observation. No CUDA kernels, device allocations or numerical edits.
#include "common.hpp"
#include "symbols.inc"
#include <iostream>
#include <cstring>

using namespace stage_b_diag;
namespace {
int level=-1, field=0;
size_t planned=0, inputs=0, stores=0, batches=0, joined=0;
bool building=false;
double* base=nullptr;
std::ofstream residuals, metadata;
std::ofstream waves;
std::chrono::steady_clock::time_point stage_start;
const size_t counts[] = {0,0,49919,54878,17767,6816,1148,124,18};
bool wanted() { return level>=4 && level<=8; }
template<class F> F original(const char* symbol) {
  static void* h=dlopen(env("STAGE_B_NATIVE").c_str(),RTLD_NOW|RTLD_NOLOAD);
  need(h,"native not loaded");auto f=reinterpret_cast<F>(dlsym(h,symbol));need(f,"native symbol absent");return f;
}
std::string file(const char* suffix) { return env("STAGE_B_OUTPUT")+"/l"+std::to_string(level)+suffix; }
}

extern "C" cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t bytes,
    cudaMemcpyKind kind,cudaStream_t stream) {
  using F=cudaError_t(*)(void*,const void*,size_t,cudaMemcpyKind,cudaStream_t);
  static F f=[] {void* h=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);
    need(h,"cudart absent");return reinterpret_cast<F>(dlsym(h,"cudaMemcpyAsync"));}();
  const auto result=f(dst,src,bytes,kind,stream);
  if(result==cudaSuccess&&building&&kind==cudaMemcpyDeviceToDevice) {
    const size_t expected[]={size_t(n*k)*8,size_t(k)*8,8};
    const size_t offset[]={0,size_t(n*k),size_t(n*k+k)};
    need(bytes==expected[field],"unexpected Stage B D2D shape");
    if(stores==0&&field==0)base=static_cast<double*>(dst);
    need(dst==base+stores*stride+offset[field],"cache store order/stride changed");
    if(++field==3){field=0;++stores;}
  }
  return result;
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  using F=cudaError_t(*)(cudaEvent_t);
  static F f=[] {void* h=dlopen("libcudart.so.12",RTLD_NOW|RTLD_NOLOAD);
    need(h,"cudart absent");return reinterpret_cast<F>(dlsym(h,"cudaEventSynchronize"));}();
  const auto result=f(event);
  if(result==cudaSuccess&&building&&stores>joined&&(stores%640==0||stores==planned)) {
    const auto label="wave_end_"+std::to_string(joined/640);
    mark(label.c_str(),level);
    waves<<std::setprecision(17)<<joined/640<<'\t'<<stores-joined<<'\t'<<seconds(stage_start)<<'\n';
    joined=stores;
  }
  return result;
}

namespace fastkpc {
void begin_full_cuda_ci_level_component_prefill(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache,int l,size_t count) {
  static auto f=original<decltype(&begin_full_cuda_ci_level_component_prefill)>(SYM_begin);
  f(cache,l,count);level=l;planned=count;inputs=stores=batches=0;field=0;base=nullptr;
  if(!wanted())return;
  need(count==counts[l],"canonical Stage B coverage changed");
  if(env("STAGE_B_MODE")=="capture") {
    residuals.open(file("-residuals.f64"),std::ios::binary);
    metadata.open(file("-targets.tsv"));need(residuals.good()&&metadata.good(),"capture output failed");
  }
}
void store_full_cuda_ci_level_component_residual(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache,
    const MultiPenaltyGcvCudaResidualConsumerView& v) {
  static auto f=original<decltype(&store_full_cuda_ci_level_component_residual)>(SYM_store);
  f(cache,v);if(!wanted())return;
  need(v.n==n&&v.target_count>=1&&v.target_count<=16,"input cohort shape changed");
  if(env("STAGE_B_MODE")=="capture") {
    std::vector<double> h(size_t(v.target_count)*n);
    check(cudaMemcpy(h.data(),v.residuals,h.size()*8,cudaMemcpyDeviceToHost));
    residuals.write(reinterpret_cast<char*>(h.data()),h.size()*8);
    for(int i=0;i<v.target_count;++i)
      metadata<<batches<<'\t'<<inputs+i<<'\t'<<v.target_keys[i]<<'\t'
        <<v.solver_info[i]<<'\t'<<v.decomposition_route[i]<<'\t'<<v.numerical_rank[i]<<'\n';
  }
  inputs+=v.target_count;++batches;
}
void build_full_cuda_ci_level_components(
    const std::shared_ptr<FullCudaCiLegacyEigComponentCache>& cache) {
  static auto f=original<decltype(&build_full_cuda_ci_level_components)>(SYM_build);
  if(!wanted()){f(cache);return;}
  need(inputs==planned,"incomplete residual capture");
  if(residuals.is_open()){residuals.close();metadata.close();}
  stores=joined=0;field=0;base=nullptr;
  waves.open(file("-waves.tsv"));need(waves.good(),"wave output failed");building=true;
  const auto start=std::chrono::steady_clock::now();stage_start=start;mark("begin",level);
  try {f(cache);}catch(...){building=false;throw;}
  mark("end",level);const double elapsed=seconds(start);building=false;waves.close();
  need(stores==planned&&joined==planned&&field==0&&base,"incomplete component stores/joins");
  // Original build has joined all producers. Diagnostic reads cannot perturb it.
  std::ofstream hashes(file("-payload-sha256.txt"));need(hashes.good(),"hash output failed");
  std::vector<double> h(64*stride);
  for(size_t off=0;off<planned;off+=64) {
    const size_t count=std::min(size_t(64),planned-off);
    check(cudaMemcpy(h.data(),base+off*stride,count*stride*8,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<count;++i) hashes<<hash(h.data()+i*stride,fields*8)<<'\n';
  }
  hashes.close();
  static auto info=original<decltype(&full_cuda_ci_legacy_eig_component_cache_info)>(SYM_info);
  const auto s=info(cache).level_prefill.back();
  need(s.structured_accepted==planned&&s.dense_fallback==0&&s.structured_contexts==std::min(size_t(10),(planned+63)/64)&&
    s.structured_width==64&&s.structured_waves==(planned+639)/640,"structured shape changed");
  std::ofstream receipt(file("-observation.json"));
  receipt<<std::setprecision(17)<<"{\"level\":"<<level<<",\"components\":"<<planned
    <<",\"input_cohorts\":"<<batches<<",\"three_field_hashes\":"<<stores
    <<",\"contexts\":"<<s.structured_contexts<<",\"width\":64,\"waves\":"<<s.structured_waves
    <<",\"stage_b_sec\":"<<elapsed<<",\"native_stage_b_sec\":"<<s.prefill_host_ms/1000
    <<",\"dense_fallback\":0,\"diagnostic_payload_d2h_bytes\":"<<planned*stride*8
    <<",\"diagnostic_residual_d2h_bytes\":"<<(env("STAGE_B_MODE")=="capture"?planned*n*8:0)
    <<",\"sampled_device_used_bytes\":"<<s.sampled_device_used_bytes<<"}\n";
  std::cout<<"PASS Stage B "<<level<<" components "<<planned<<" hashes "<<stores<<std::endl;
}
}
