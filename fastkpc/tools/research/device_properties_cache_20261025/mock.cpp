// Fault/concurrency tests of the production helper and unchanged status callers.
// CUDA is mocked here; this test is never a numerical authority.
#include "cuda/cuda_device_properties.hpp"
#include "cuda/cuda_status.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

std::atomic<int> queries{0}, counts{0}, currents{0}, versions{0}, drivers{0}, frees{0};
std::atomic<bool> fail_once{false};
cudaError_t count_error=cudaSuccess, current_error=cudaSuccess, init_error=cudaSuccess;
cudaError_t version_error=cudaSuccess, driver_error=cudaSuccess;
extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p,int d) {
  ++queries;
  if(!p) return cudaErrorInvalidValue;
  if(d<0 || d>=4) return cudaErrorInvalidDevice;
  if(fail_once.exchange(false)) return cudaErrorUnknown;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  *p={};std::snprintf(p->name,sizeof(p->name),"fixture-%d",d);
  p->uuid.bytes[0]=d+1;p->major=8;p->minor=9;p->totalGlobalMem=24ULL<<30;
  p->multiProcessorCount=128;p->maxGridSize[1]=65535;p->maxGridSize[2]=65535;
  return cudaSuccess;
}
extern "C" cudaError_t cudaGetDeviceCount(int* n){++counts;*n=4;return count_error;}
extern "C" cudaError_t cudaGetDevice(int* n){++currents;*n=0;return current_error;}
extern "C" cudaError_t cudaFree(void*){++frees;return init_error;}
extern "C" cudaError_t cudaRuntimeGetVersion(int* n){++versions;*n=12080;return version_error;}
extern "C" cudaError_t cudaDriverGetVersion(int* n){++drivers;*n=12080;return driver_error;}
extern "C" const char* cudaGetErrorString(cudaError_t){return "injected";}
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F> void error(F f,const char* message){
  try{f();}catch(const std::runtime_error& e){need(e.what()==std::string(message),e.what());return;}
  throw std::runtime_error("expected error absent");
}
int main(int argc,char** argv)try{
  need(argc==2,"expected enable mode");
  if(std::string(argv[1])=="invalid"){
    error([]{fastkpc_cuda_device_properties_cache_stats();},
      "FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE must be 0 or 1");
    need(queries==0,"invalid configuration queried CUDA");
    std::cout<<"PASS invalid configuration\n";return 0;
  }
  const bool enabled=std::string(argv[1])=="1";
  // Race first initialization and first device lookup across callers.
  std::atomic<int> ready{0},bad{0};std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  for(int k=0;k<32;++k)workers.emplace_back([&]{
    ++ready;while(!go.load())std::this_thread::yield();
    for(int i=0;i<8;++i){cudaDeviceProp p{};
      if(fastkpc_cuda_get_device_properties(&p,0)!=cudaSuccess ||
         std::strcmp(p.name,"fixture-0") || p.uuid.bytes[0]!=1)++bad;}
  });
  while(ready!=32)std::this_thread::yield();go=true;
  for(auto& w:workers)w.join();
  need(bad==0,"concurrent metadata mismatch");
  auto s=fastkpc_cuda_device_properties_cache_stats();
  need(s.enabled==enabled && s.requests==256,"initial counters");
  need(s.queries==(enabled?1:256) && queries==s.queries,"first access duplicated");
  need(s.hits==(enabled?255:0) && s.entries==(enabled?1:0),"cache counters");
  cudaDeviceProp p{};int before=queries;
  fail_once=true;
  need(fastkpc_cuda_get_device_properties(&p,1)==cudaErrorUnknown,"failure status changed");
  need(fastkpc_cuda_get_device_properties(&p,1)==cudaSuccess,"failed query cached");
  need(queries==before+2 && !std::strcmp(p.name,"fixture-1") && p.uuid.bytes[0]==2,
       "retry or per-device isolation");
  need(fastkpc_cuda_get_device_properties(&p,1)==cudaSuccess,"repeat device query");
  need(queries==before+(enabled?2:3),"cache not per-device");
  auto info=fastkpc_cuda_device_info();
  need(info.device_id==0 && info.name=="fixture-0" && info.major==8 && info.minor==9,
       "public device info changed");
  const auto a=fastkpc_cuda_phase3_environment_identity(0);
  before=counts;int v=versions,d=drivers;
  const auto b=fastkpc_cuda_phase3_environment_identity(0);
  need(counts==before+1 && versions==v+1 && drivers==d+1,"live queries bypassed");
  need(a.gpu_uuid==b.gpu_uuid && a.gpu_name==b.gpu_name && a.sm_count==b.sm_count &&
       a.configuration_schema_version==b.configuration_schema_version,"identity changed");
  init_error=cudaErrorUnknown;
  error([]{fastkpc_cuda_device_info();},"CUDA unavailable: injected");init_error=cudaSuccess;
  current_error=cudaErrorUnknown;
  error([]{fastkpc_cuda_device_info();},"cudaGetDevice failed: injected");current_error=cudaSuccess;
  count_error=cudaErrorUnknown;
  error([]{fastkpc_cuda_phase3_environment_identity(0);},"CUDA device count query failed: injected");
  count_error=cudaSuccess;
  version_error=cudaErrorUnknown;
  error([]{fastkpc_cuda_phase3_environment_identity(0);},"CUDA toolkit version query failed: injected");
  version_error=cudaSuccess;driver_error=cudaErrorUnknown;
  error([]{fastkpc_cuda_phase3_environment_identity(0);},"CUDA driver version query failed: injected");
  driver_error=cudaSuccess;
  error([]{fastkpc_cuda_phase3_environment_identity(-1);},"device_id must be non-negative");
  error([]{fastkpc_cuda_phase3_environment_identity(4);},"requested CUDA device_id is unavailable");
  fail_once=true;
  error([]{fastkpc_cuda_phase3_environment_identity(2);},"CUDA static device properties query failed: injected");
  need(fastkpc_cuda_phase3_environment_identity(2).gpu_name=="fixture-2","diagnostic retry");
  before=queries;
  for(int i=0;i<2;++i)need(fastkpc_cuda_get_device_properties(&p,-1)==cudaErrorInvalidDevice,
                         "invalid device status");
  need(fastkpc_cuda_get_device_properties(nullptr,0)==cudaErrorInvalidValue,"null status");
  need(queries==before+3,"invalid arguments not delegated");
  setenv("FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE",enabled?"0":"1",1);
  need(fastkpc_cuda_device_properties_cache_stats().enabled==enabled,"configuration not frozen");
  s=fastkpc_cuda_device_properties_cache_stats();
  need(s.queries==queries && s.requests==s.queries+s.hits && s.failures==5,"final counters");
  need(s.entries==(enabled?3:0),"failed entry retained");
  std::cout<<"PASS mode="<<enabled<<" queries="<<s.queries<<" hits="<<s.hits
           <<" failures="<<s.failures<<" entries="<<s.entries<<"\n";
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
