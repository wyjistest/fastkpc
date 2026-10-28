#include "cuda/cuda_device_properties.hpp"
#include "cuda/cuda_status.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main()try{
  need(cudaSetDevice(0)==cudaSuccess,"set device");
  cudaDeviceProp real{};need(cudaGetDeviceProperties(&real,0)==cudaSuccess,"direct query");
  const auto started=std::chrono::steady_clock::now();
  std::atomic<int> bad{0};std::vector<std::thread> workers;
  for(int t=0;t<32;++t)workers.emplace_back([&]{
    if(cudaSetDevice(0)!=cudaSuccess){++bad;return;}
    for(int i=0;i<16;++i){cudaDeviceProp p{};
      if(fastkpc_cuda_get_device_properties(&p,0)!=cudaSuccess ||
         std::memcmp(&p,&real,sizeof(p)))++bad;}
  });
  for(auto& w:workers)w.join();need(bad==0,"real metadata mismatch");
  const double host_ms=std::chrono::duration<double,std::milli>(
    std::chrono::steady_clock::now()-started).count();
  auto info=fastkpc_cuda_device_info();auto id=fastkpc_cuda_phase3_environment_identity(0);
  need(info.name==real.name && id.gpu_name==real.name && id.sm_count==real.multiProcessorCount,
       "diagnostic identity mismatch");
  auto s=fastkpc_cuda_device_properties_cache_stats();
  need(s.requests==514 && s.queries==(s.enabled?1:514) && s.entries==(s.enabled?1:0) &&
       s.requests==s.hits+s.queries && s.failures==0,"native counters");
  std::cout<<"PASS native mode="<<s.enabled<<" requests="<<s.requests<<" queries="<<s.queries
           <<" name="<<id.gpu_name<<" uuid="<<id.gpu_uuid<<" SM="<<id.sm_count
           <<" concurrent_helper_host_ms="<<host_ms<<"\n";
  need(cudaDeviceReset()==cudaSuccess,"reset");
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
