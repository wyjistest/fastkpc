// External host resource accounting, never injected into production sources.
#include "audit.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <limits>
namespace {
std::mutex mutex;
std::unordered_map<void*,size_t> allocations;
std::unordered_set<cudaStream_t> streams;
std::unordered_set<cudaEvent_t> events;
std::unordered_set<cublasHandle_t> handles;
layer_audit::Counts state;
std::atomic<bool> active{false};
size_t limit=std::numeric_limits<size_t>::max();
template<class F>F next(const char* name){auto f=reinterpret_cast<F>(dlsym(RTLD_NEXT,name));if(!f)std::abort();return f;}
void mutation(){if(active.load())++state.forbidden_calls;}
}
namespace layer_audit {
Counts counts(){std::lock_guard<std::mutex> guard(mutex);auto s=state;
  s.allocations=allocations.size();s.streams=streams.size();s.events=events.size();s.blas=handles.size();return s;}
void native_scope(bool value){active=value;}
void memory_limit(size_t value){std::lock_guard<std::mutex> guard(mutex);limit=value;}
}
extern "C" cudaError_t cudaMalloc(void** p,size_t bytes){
  static auto f=next<cudaError_t(*)(void**,size_t)>("cudaMalloc");
  {std::lock_guard<std::mutex> guard(mutex);mutation();if(bytes>limit-state.bytes)return cudaErrorMemoryAllocation;}
  auto status=f(p,bytes);if(status==cudaSuccess&&*p){std::lock_guard<std::mutex> guard(mutex);
    allocations[*p]=bytes;state.bytes+=bytes;++state.allocation_calls;state.peak_bytes=std::max(state.peak_bytes,state.bytes);}return status;
}
extern "C" cudaError_t cudaFree(void* p){
  static auto f=next<decltype(&cudaFree)>("cudaFree");
  if(p){std::lock_guard<std::mutex> guard(mutex);mutation();}
  auto status=f(p);if(status==cudaSuccess&&p){std::lock_guard<std::mutex> guard(mutex);
    auto i=allocations.find(p);if(i!=allocations.end()){state.bytes-=i->second;allocations.erase(i);}}return status;
}
extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p,unsigned flags){
  static auto f=next<decltype(&cudaStreamCreateWithFlags)>("cudaStreamCreateWithFlags");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p,flags);
  if(status==cudaSuccess){std::lock_guard<std::mutex> guard(mutex);streams.insert(*p);}return status;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t p){
  static auto f=next<decltype(&cudaStreamDestroy)>("cudaStreamDestroy");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p);
  if(status==cudaSuccess){std::lock_guard<std::mutex> guard(mutex);streams.erase(p);}return status;
}
extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t* p,unsigned flags){
  static auto f=next<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p,flags);
  if(status==cudaSuccess){std::lock_guard<std::mutex> guard(mutex);events.insert(*p);}return status;
}
extern "C" cudaError_t cudaEventCreate(cudaEvent_t* p){
  static auto f=next<cudaError_t(*)(cudaEvent_t*)>("cudaEventCreate");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p);
  if(status==cudaSuccess){std::lock_guard<std::mutex> guard(mutex);events.insert(*p);}return status;
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t p){
  static auto f=next<decltype(&cudaEventDestroy)>("cudaEventDestroy");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p);
  if(status==cudaSuccess){std::lock_guard<std::mutex> guard(mutex);events.erase(p);}return status;
}
extern "C" cublasStatus_t cublasCreate_v2(cublasHandle_t* p){
  static auto f=next<decltype(&cublasCreate_v2)>("cublasCreate_v2");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p);
  if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(mutex);handles.insert(*p);}return status;
}
extern "C" cublasStatus_t cublasDestroy_v2(cublasHandle_t p){
  static auto f=next<decltype(&cublasDestroy_v2)>("cublasDestroy_v2");
  {std::lock_guard<std::mutex> guard(mutex);mutation();}auto status=f(p);
  if(status==CUBLAS_STATUS_SUCCESS){std::lock_guard<std::mutex> guard(mutex);handles.erase(p);}return status;
}
