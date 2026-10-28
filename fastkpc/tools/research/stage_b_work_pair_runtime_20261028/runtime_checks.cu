// GPU output equality plus host configuration/lifetime rejection tests.
#include "full_cuda_ci_structured_eig.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastkpc;
void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int rejects=0;
template<class F> void reject(F f){bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}
  need(caught,"invalid configuration not rejected");++rejects;}
long long live_allocations=0,live_bytes=0;
struct Memory {
  std::vector<std::pair<void*,size_t>> items;
  static void* allocate(void* owner,size_t bytes){auto& m=*static_cast<Memory*>(owner);void* p=nullptr;
    check(cudaMalloc(&p,bytes));check(cudaMemset(p,0,bytes));m.items.push_back({p,bytes});
    ++live_allocations;live_bytes+=bytes;return p;}
  ~Memory(){for(auto item:items){cudaFree(item.first);--live_allocations;live_bytes-=item.second;}}
};
__global__ void compare(const unsigned char* a,const unsigned char* b,size_t n,unsigned* mismatch){
  for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
    if(a[i]!=b[i])atomicOr(mismatch,1u);
}
void compare_bytes(const void* a,const void* b,size_t n,unsigned* bad,cudaStream_t stream){
  compare<<<64,256,0,stream>>>(static_cast<const unsigned char*>(a),static_cast<const unsigned char*>(b),n,bad);
  check(cudaGetLastError());
}
int main(int argc,char** argv){try {
  need(argc==3,"runtime_checks DEEP_CAPTURE REPORT_JSON");
  using M=StructuredDbdsqrMode;
  need(parse_structured_dbdsqr_mode(nullptr)==M::Original,"default mode");
  need(parse_structured_dbdsqr_mode("")==M::Original,"empty mode");
  need(parse_structured_dbdsqr_mode("0")==M::Original,"off mode");
  need(parse_structured_dbdsqr_mode("1")==M::FixedVtWorkPair,"on mode");
  for(const char* s:{"2","true","01","-1","1 "})reject([&]{parse_structured_dbdsqr_mode(s);});
  for(int n:{6,351,1024})for(M m:{M::Original,M::FixedVtWorkPair}) {
    const size_t bytes=structured_dbdsqr_shared_bytes(m,n);
    validate_structured_dbdsqr_launch(m,m,n,bytes,'L',1,0,0);
    reject([&]{validate_structured_dbdsqr_launch(m,m,n,bytes-8,'L',1,0,0);});
    reject([&]{validate_structured_dbdsqr_launch(m,m,n,bytes+8,'L',1,0,0);});
    reject([&]{validate_structured_dbdsqr_launch(m,m==M::Original?M::FixedVtWorkPair:M::Original,n,bytes,'L',1,0,0);});
  }
  // This is the exact dangerous combination. Validation only: no CUDA launch.
  reject([&]{validate_structured_dbdsqr_launch(M::Original,M::Original,351,5*351*8,'L',1,0,0);});
  reject([&]{validate_structured_dbdsqr_launch(M::FixedVtWorkPair,M::FixedVtWorkPair,351,5*351*8,'U',1,0,0);});
  for(int field=0;field<3;++field)reject([&]{validate_structured_dbdsqr_launch(M::FixedVtWorkPair,M::FixedVtWorkPair,
    351,5*351*8,'L',field==0?2:1,field==1?1:0,field==2?1:0);});
  reject([&]{structured_dbdsqr_shared_bytes(static_cast<M>(9),351);});
  reject([&]{structured_dbdsqr_shared_bytes(M::Original,0);});
  reject([&]{structured_dbdsqr_shared_bytes(M::FixedVtWorkPair,1025);});
  check(cudaSetDevice(0));
  auto before=structured_component_execution_counts();
  {
    Memory memory;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    setenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR","invalid",1);
    reject([&]{create_structured_component_plan(351,35,64,170662,stream,Memory::allocate,&memory);});
    need(live_allocations==0,"invalid option allocated device memory");
    unsetenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR");
    auto* off=create_structured_component_plan(351,35,64,170662,stream,Memory::allocate,&memory);
    setenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR","1",1);
    auto* on=create_structured_component_plan(351,35,64,170662,stream,Memory::allocate,&memory);
    setenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR","0",1);
    need(structured_component_plan_mode(off)==M::Original && structured_component_plan_mode(on)==M::FixedVtWorkPair,
         "plan mode changed with environment");
    auto* input=static_cast<double*>(Memory::allocate(&memory,64*351*8));
    auto* bad=static_cast<unsigned*>(Memory::allocate(&memory,sizeof(unsigned)));
    for(int level:{4,5,6,7,8}) {
      const int count=level==8?18:64;
      std::vector<double> data(size_t(count)*351);
      std::ifstream f(std::string(argv[1])+"/l"+std::to_string(level)+"-residuals.f64",std::ios::binary);
      f.read(reinterpret_cast<char*>(data.data()),data.size()*8);
      need(size_t(f.gcount())==data.size()*8,"captured input truncated");
      check(cudaMemcpyAsync(input,data.data(),data.size()*8,cudaMemcpyHostToDevice,stream));
      for(int repeat=0;repeat<2;++repeat){
        // Change the option in both directions: neither existing plan can change.
        setenv("FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR",repeat?"1":"0",1);
        auto a=submit_structured_components(off,input,count),b=submit_structured_components(on,input,count);
        compare_bytes(a.values,b.values,size_t(count)*36*8,bad,stream);
        compare_bytes(a.vectors,b.vectors,size_t(count)*36*351*8,bad,stream);
        compare_bytes(a.status,b.status,size_t(count)*sizeof(StructuredComponentStatus),bad,stream);
      }
    }
    auto submitted=structured_component_execution_counts();
    reject([&]{submit_structured_components(on,input,65);});
    reject([&]{submit_structured_components(on,nullptr,1);});
    reject([&]{submit_structured_components(nullptr,input,1);});
    auto unchanged=structured_component_execution_counts();
    for(int m=0;m<2;++m)need(submitted.submissions[m]==unchanged.submissions[m] &&
      submitted.targets[m]==unchanged.targets[m],"invalid submit reached launch");
    unsigned mismatch=0;
    check(cudaMemcpyAsync(&mismatch,bad,sizeof(mismatch),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));need(mismatch==0,"production OFF/ON GPU bitwise mismatch");
    destroy_structured_component_plan(off);destroy_structured_component_plan(on);
    check(cudaStreamDestroy(stream));
  }
  auto after=structured_component_execution_counts();
  for(int m=0;m<2;++m)need(after.created[m]-before.created[m]==1 &&
    after.destroyed[m]-before.destroyed[m]==1 && after.submissions[m]-before.submissions[m]==10 &&
    after.targets[m]-before.targets[m]==548,"production mode dispatch/lifecycle counter mismatch");
  need(live_allocations==0 && live_bytes==0,"test allocation leak");
  std::ofstream f(argv[2]);need(f.good(),"report output");
  f<<"{\"status\":\"PASS_PRODUCTION_MODE_LIFETIME_AND_GPU_PARITY\",\"rejected_configurations\":"<<rejects
   <<",\"targets_per_mode\":548,\"submissions_per_mode\":10,\"plan_mode_immutable\":true,"
     "\"undersized_original_launched\":false,\"live_device_allocations\":0,\"live_device_bytes\":0}\n";
  std::cout<<"PASS production mode/lifetime, "<<rejects<<" prelaunch rejects and 548 GPU target comparisons"<<std::endl;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
