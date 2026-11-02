// Symmetric host metadata only; no device diagnostic code.
#include <fstream>
extern "C" void ip_resources(int* out) {
  structured_check(cudaSetDevice(0));
  auto kernel=bidiagonal<StructuredDbdsqrMode::FixedVtWorkPair>;
  structured_check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,IP_FACTOR*351*8));
  cudaFuncAttributes a{};structured_check(cudaFuncGetAttributes(&a,kernel));
  int blocks=0;structured_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,32,IP_FACTOR*351*8));
  out[0]=blocks;out[1]=a.numRegs;out[2]=int(a.localSizeBytes);out[3]=int(a.sharedSizeBytes);out[4]=IP_FACTOR*351*8;
}
__attribute__((destructor)) void ip_report() {
  const char* path=std::getenv("STAGE_B_OUTPUT");if(!path)return;
  const auto c=fastkpc::structured_component_execution_counts();
  std::ofstream out(std::string(path)+"/immediate.json");
  out<<"{\"factor\":"<<IP_FACTOR<<",\"dynamic_shared\":"<<IP_FACTOR*351*8
    <<",\"recover_tile\":36,\"created\":"<<c.created[1]<<",\"destroyed\":"<<c.destroyed[1]
    <<",\"submissions\":"<<c.submissions[1]<<",\"targets\":"<<c.targets[1]
    <<",\"other_created\":"<<c.created[0]<<",\"other_destroyed\":"<<c.destroyed[0]
    <<",\"other_submissions\":"<<c.submissions[0]<<",\"other_targets\":"<<c.targets[0]<<"}\n";
}
