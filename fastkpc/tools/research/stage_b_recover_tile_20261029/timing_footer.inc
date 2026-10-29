// Host-side metadata only; no device instrumentation or arithmetic changes.
#include <fstream>
extern "C" void rt_resources(int* out) {
  structured_check(cudaSetDevice(0));
  structured_check(cudaFuncSetAttribute(recover,cudaFuncAttributeMaxDynamicSharedMemorySize,RT_TILE*351*8));
  cudaFuncAttributes a{};structured_check(cudaFuncGetAttributes(&a,recover));
  int blocks=0;structured_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,recover,64,RT_TILE*351*8));
  out[0]=blocks;out[1]=a.numRegs;out[2]=int(a.localSizeBytes);out[3]=int(a.sharedSizeBytes);out[4]=RT_TILE*351*8;
}
__attribute__((destructor)) void rt_report() {
  const char* path=std::getenv("STAGE_B_OUTPUT");if(!path)return;
  const auto c=fastkpc::structured_component_execution_counts();
  std::ofstream out(std::string(path)+"/tile.json");
  out<<"{\"tile\":"<<RT_TILE<<",\"threads\":64,\"dynamic_shared\":"<<RT_TILE*351*8
     <<",\"created\":"<<c.created[1]<<",\"destroyed\":"<<c.destroyed[1]
     <<",\"submissions\":"<<c.submissions[1]<<",\"targets\":"<<c.targets[1]
     <<",\"other_created\":"<<c.created[0]<<",\"other_destroyed\":"<<c.destroyed[0]
     <<",\"other_submissions\":"<<c.submissions[0]<<",\"other_targets\":"<<c.targets[0]<<"}\n";
}
