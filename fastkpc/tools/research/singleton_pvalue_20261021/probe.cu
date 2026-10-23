// GPU-only pair replay. Numerical pair statements are extracted verbatim.
#include "full_cuda_ci_vertical.cu"
#include <cstdio>

namespace fastkpc {
struct ScalarProbe {
  VerticalCallAccounting accounting;
  std::unique_ptr<FullCudaCiLegacyEigExecutionContext> context;
  std::unique_ptr<TrackedDeviceBuffer> payload;
  int n, k, components, pairs;
  ScalarProbe(int rows,int cols,int width,int pair_capacity)
      : n(rows),k(cols),components(width),pairs(pair_capacity) {
    int device=-1; check_cuda(cudaGetDevice(&device),"scalar replay device");
    context=std::make_unique<FullCudaCiLegacyEigExecutionContext>(
      device,n,k,0,pairs,0,&accounting);
    payload=std::make_unique<TrackedDeviceBuffer>(
      std::size_t(components)*(std::size_t(n)*k+k+2)*sizeof(double),&accounting);
    static_assert(sizeof(DeviceCompactCiResult)==72,"compact result layout changed");
  }
  void run(const double* values,int count,const int* left,const int* right,
           const unsigned long long* ids,int pair_count,void* output,double* cross_output) {
    if (count<1 || count>components || pair_count<1 || pair_count>pairs)
      throw std::runtime_error("scalar replay capacity exceeded");
    auto* execution_context=context.get();
    const auto stream=execution_context->stream();
    const int num_col=k;
    const std::size_t pair_size=pair_count,cross_stride=std::size_t(k)*k;
    const std::size_t entry=std::size_t(n)*k+k+2;
    FixedSpBatchHostView batch; batch.n=n;
    check_cuda(cudaMemcpyAsync(payload->get(),values,count*entry*sizeof(double),
      cudaMemcpyHostToDevice,stream),"upload captured component payload");
    double* base=static_cast<double*>(payload->get());
    std::vector<std::uintptr_t> metadata(10*pair_size);
    for (int j=0;j<pair_count;++j) {
      if (left[j]<0 || right[j]<0 || left[j]>=count || right[j]>=count)
        throw std::runtime_error("scalar replay target index invalid");
      double* l=base+left[j]*entry; double* r=base+right[j]*entry;
      metadata[j]=ids[j];
      metadata[pair_size+j]=reinterpret_cast<std::uintptr_t>(l);
      metadata[2*pair_size+j]=reinterpret_cast<std::uintptr_t>(r);
      metadata[3*pair_size+j]=reinterpret_cast<std::uintptr_t>(execution_context->pair_cross()+j*cross_stride);
      metadata[4*pair_size+j]=reinterpret_cast<std::uintptr_t>(l+std::size_t(n)*k);
      metadata[5*pair_size+j]=reinterpret_cast<std::uintptr_t>(r+std::size_t(n)*k);
      metadata[6*pair_size+j]=reinterpret_cast<std::uintptr_t>(l+entry-2);
      metadata[7*pair_size+j]=reinterpret_cast<std::uintptr_t>(r+entry-2);
      metadata[8*pair_size+j]=reinterpret_cast<std::uintptr_t>(l+entry-1);
      metadata[9*pair_size+j]=reinterpret_cast<std::uintptr_t>(r+entry-1);
    }
    check_cuda(cudaMemcpyAsync(execution_context->pair_metadata(),metadata.data(),
      metadata.size()*sizeof(std::uintptr_t),cudaMemcpyHostToDevice,stream),"upload original pair ordering");
    const double one=1.0,zero=0.0;
#include "pair_stage.inc"
    check_cuda(cudaMemcpyAsync(output,execution_context->device_results(),
      pair_size*sizeof(DeviceCompactCiResult),cudaMemcpyDeviceToHost,stream),"read scalar replay compact");
    check_cuda(cudaMemcpyAsync(cross_output,execution_context->pair_cross(),
      pair_size*cross_stride*sizeof(double),cudaMemcpyDeviceToHost,stream),"read diagnostic pair cross");
    check_cuda(cudaEventRecord(execution_context->completion(),stream),"record scalar replay completion");
    check_cuda(cudaEventSynchronize(execution_context->completion()),"wait scalar replay diagnostic");
  }
};
}
extern "C" void* scalar_probe_create(int n,int k,int c,int p) {
  try { return new fastkpc::ScalarProbe(n,k,c,p); }
  catch(const std::exception& e) { std::fprintf(stderr,"scalar create: %s\n",e.what()); return nullptr; }
}
extern "C" int scalar_probe_run(void* handle,const double* values,int count,
    const int* l,const int* r,const unsigned long long* ids,int pairs,void* output,double* cross) {
  try { static_cast<fastkpc::ScalarProbe*>(handle)->run(values,count,l,r,ids,pairs,output,cross); return 0; }
  catch(const std::exception& e) { std::fprintf(stderr,"scalar run: %s\n",e.what()); return 1; }
}
extern "C" void scalar_probe_destroy(void* handle) { delete static_cast<fastkpc::ScalarProbe*>(handle); }
extern "C" int scalar_probe_snapshot(long long* values) {
  auto s=fastkpc::full_cuda_ci_vertical_resource_snapshot();
  values[0]=s.live_device_allocations; values[1]=s.live_device_bytes;
  values[2]=s.live_streams; values[3]=s.live_events; return 0;
}
