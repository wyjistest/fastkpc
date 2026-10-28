#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
namespace {
long long wp_creates=0,wp_destroys=0,wp_submits=0,wp_targets=0;
int wp_blocks=0,wp_regs=0,wp_local=0,wp_static=0,wp_dynamic=0;
template<class K> void wp_resource(K kernel,int n) {
  cudaFuncAttributes attr{};
  wp_dynamic=(3+WP_WORK_FACTOR)*n*sizeof(double);
  if(cudaFuncGetAttributes(&attr,kernel)!=cudaSuccess ||
     cudaOccupancyMaxActiveBlocksPerMultiprocessor(&wp_blocks,kernel,32,wp_dynamic)!=cudaSuccess)
    throw std::runtime_error("work-pair resource query failed");
  wp_regs=attr.numRegs;wp_local=attr.localSizeBytes;wp_static=attr.sharedSizeBytes;
}
__attribute__((destructor)) void wp_report() {
  const char* dir=std::getenv("STAGE_B_OUTPUT");if(!dir)return;
  std::ofstream f(std::string(dir)+"/variant.json");
  f<<"{\"work_factor\":"<<WP_WORK_FACTOR<<",\"creates\":"<<wp_creates<<",\"destroys\":"<<wp_destroys
   <<",\"submits\":"<<wp_submits<<",\"targets\":"<<wp_targets<<",\"block_size\":32,\"dynamic_shared\":"<<wp_dynamic
   <<",\"static_shared\":"<<wp_static<<",\"registers\":"<<wp_regs<<",\"local_bytes\":"<<wp_local
   <<",\"occupancy_api_blocks\":"<<wp_blocks<<"}\n";
}
}
