// External software-failure injection, never linked into the production native.
#include "fault.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
namespace { thread_local int copies=0; thread_local bool event=false; }
namespace l6_fault {
void upload_after(int value){copies=value;}
void next_event_record(){event=true;}
void clear(){copies=0;event=false;}
}
extern "C" cudaError_t cudaMemcpyAsync(void* dst,const void* src,std::size_t bytes,
                                     cudaMemcpyKind kind,cudaStream_t stream){
  static auto f=reinterpret_cast<decltype(&cudaMemcpyAsync)>(dlsym(RTLD_NEXT,"cudaMemcpyAsync"));
  if(!f)std::abort();
  if(kind==cudaMemcpyHostToDevice&&copies>0&&--copies==0)return cudaErrorInvalidValue;
  return f(dst,src,bytes,kind,stream);
}
extern "C" cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t stream){
  static auto f=reinterpret_cast<cudaError_t(*)(cudaEvent_t,cudaStream_t)>(dlsym(RTLD_NEXT,"cudaEventRecord"));
  if(!f)std::abort();
  if(event){event=false;return cudaErrorInvalidValue;}return f(e,stream);
}

