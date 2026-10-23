// Separate TU keeps the research solver's --fmad=false build contract; the
// original consumer TU retains its production compiler flags.
#include "../exact_speedup_20261020/structured_eig_gpu.cu"
#include "device_api.hpp"
#include <memory>

namespace {
struct DeviceAdapter {
  std::unique_ptr<Context> context;
  StructuredStatus* status=nullptr;
  int k=0;
};
__global__ void inject_info(Meta* meta,int index) {
  if (threadIdx.x==0) meta[index].info=91;
}
__global__ void check_boundary(const Meta* meta,const double* values,int k,
    int count,int fault_index,int fault,StructuredStatus* out) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  const Meta z=meta[i];
  StructuredStatus s{z.info,z.m,z.rotations,z.deflated,0,0,0,0,0};
  if(z.info) s.reason=z.info>=21 && z.info<=25 ? kRotationOverflow : kStructuredInfo;
  else {
    s.kth=fabs(values[size_t(i)*(k+1)+k-1]);
    s.next=fabs(values[size_t(i)*(k+1)+k]);
    if(i==fault_index && fault==3) s.next=s.kth; // diagnostic guard injection only
    s.gap=(s.kth-s.next)/fmax(fmax(s.kth,s.next),DBL_MIN);
    if(!isfinite(s.gap) || s.gap<kStructuredAdapterGapThreshold)
      s.reason=kSmallBoundaryGap;
  }
  out[i]=s;
}
}
extern "C" void* adapter_structured_create(int n,int k,int capacity,int rotations,char* error) {
  try {
    auto a=std::make_unique<DeviceAdapter>();
    // Recover k+1, not only k: the acceptance guard needs the first excluded root.
    a->context.reset(static_cast<Context*>(structured_eig_create(n,k+1,capacity,rotations,error)));
    if(!a->context) return nullptr;
    a->k=k; a->status=a->context->allocate<StructuredStatus>(capacity);
    return a.release();
  } catch(const std::exception& e) { error_text(error,e.what()); return nullptr; }
}
extern "C" int adapter_structured_submit(void* handle,const double* input,int count,
    cudaEvent_t ready,int fault_index,int fault,StructuredDeviceView* view,char* error) {
  try {
    auto* a=static_cast<DeviceAdapter*>(handle);
    if(!a || !input || !ready || !view || count<1 || count>a->context->count ||
       fault<0 || fault>3 || (fault && (fault_index<0 || fault_index>=count)))
      throw std::runtime_error("invalid device adapter submission");
    auto* c=a->context.get();
    check(cudaStreamWaitEvent(c->stream,ready,0));
    prepare<<<count,128,0,c->stream>>>(input,c->n,c->data,c->ints,c->meta);
    if(fault==1) inject_info<<<1,1,0,c->stream>>>(c->meta,fault_index);
    bidiagonal<<<count,32,7*c->n*sizeof(double),c->stream>>>(c->n,c->data,c->ints,
      c->meta,c->rotations,c->capacity);
    if(fault==2) {
      // Re-run exactly one target with a real one-rotation capacity; its buffer
      // base still uses the allocated stride. Other targets remain valid.
      const int i=fault_index;
      prepare<<<1,128,0,c->stream>>>(input+size_t(i)*c->n,c->n,
        c->data+size_t(i)*c->n*vectors_per_component,c->ints+size_t(i)*4*c->n,c->meta+i);
      bidiagonal<<<1,32,7*c->n*sizeof(double),c->stream>>>(c->n,
        c->data+size_t(i)*c->n*vectors_per_component,c->ints+size_t(i)*4*c->n,
        c->meta+i,c->rotations+size_t(i)*c->capacity,1);
    }
    roots<<<count,128,0,c->stream>>>(c->n,c->k,c->data,c->ints,c->meta,c->candidates);
    recover<<<dim3(count,(c->k+c->tile-1)/c->tile),64,c->tile*c->n*sizeof(double),c->stream>>>(
      c->n,c->k,c->data,c->ints,c->meta,c->rotations,c->capacity,c->candidates,
      c->tile,c->evals,c->vectors,c->means);
    check_boundary<<<(count+127)/128,128,0,c->stream>>>(c->meta,c->evals,a->k,count,
      fault_index,fault,a->status);
    check(cudaGetLastError());
    check(cudaEventRecord(c->events[5],c->stream));
    *view={c->evals,c->vectors,a->status,c->events[5]};
    return 0;
  } catch(const std::exception& e) { error_text(error,e.what()); return 1; }
}
extern "C" void adapter_structured_destroy(void* p) { delete static_cast<DeviceAdapter*>(p); }
extern "C" size_t adapter_structured_bytes(void* p) {
  return static_cast<DeviceAdapter*>(p)->context->bytes;
}
