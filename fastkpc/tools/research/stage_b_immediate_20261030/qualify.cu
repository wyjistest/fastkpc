#include "diagnostic.hpp"
#include "full_cuda_ci_structured_eig.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Result {
  unsigned mask;int coverage,rotations,info,overflow,sweeps;
  int checkpoints,checkpoint_bad,first_event,first_stage;
  unsigned long long fields,stages[7];
};
__device__ bool unequal(const void* av,const void* bv,size_t bytes) {
  auto* a=static_cast<const unsigned char*>(av);auto* b=static_cast<const unsigned char*>(bv);
  for(size_t i=threadIdx.x;i<bytes;i+=blockDim.x)if(a[i]!=b[i])return true;
  return false;
}
__global__ void compare(WorkPairView a,WorkPairView b,Result* result,int count,int direct) {
  const int c=blockIdx.x;if(c>=count)return;
  auto& out=result[c];const WorkPairState sa=a.state[c],sb=b.state[c];
  const auto ca=a.checkpoints[c],cb=b.checkpoints[c];
  if(!threadIdx.x){out={};out.coverage=sa.coverage;out.rotations=sa.count;out.info=sa.info;
    out.overflow=sa.overflow;out.sweeps=sa.sweeps;out.checkpoints=ca.count;
    out.checkpoint_bad=ca.bad|cb.bad;out.first_event=cb.first_event;out.first_stage=cb.first_stage;
    out.fields=cb.fields;for(int j=0;j<7;++j)out.stages[j]=cb.stages[j];}
  __syncthreads();unsigned bad=0;
  if(unequal(a.state+c,b.state+c,sizeof(WorkPairState))||sa.canary||sb.canary)bad|=1;
  for(int j=0;j<3;++j)if(unequal(a.deq+(size_t(c)*3+j)*a.n,b.deq+(size_t(c)*3+j)*b.n,size_t(sa.r>0?sa.r:0)*8))bad|=2;
  if(sa.count!=sb.count||sa.count<0||sa.count>a.capacity)bad|=4;
  else if(unequal(static_cast<const char*>(a.rotations)+size_t(c)*a.capacity*a.rotation_size,
      static_cast<const char*>(b.rotations)+size_t(c)*b.capacity*b.rotation_size,size_t(sa.count)*a.rotation_size))bad|=4;
  if(ca.bad||cb.bad||ca.count!=cb.count||ca.fields!=cb.fields)bad|=512;
  for(int j=0;j<7;++j)if(ca.stages[j]!=cb.stages[j])bad|=512;
  if(!direct) {
    if(unequal(a.data+size_t(c)*20*a.n,b.data+size_t(c)*20*b.n,size_t(20)*a.n*8))bad|=8;
    if(unequal(a.ints+size_t(c)*4*a.n,b.ints+size_t(c)*4*b.n,size_t(4)*a.n*4))bad|=16;
    if(unequal(static_cast<const char*>(a.meta)+size_t(c)*a.meta_size,
               static_cast<const char*>(b.meta)+size_t(c)*b.meta_size,a.meta_size))bad|=32;
    if(unequal(static_cast<const char*>(a.candidates)+size_t(c)*(2*a.k+4)*a.candidate_size,
               static_cast<const char*>(b.candidates)+size_t(c)*(2*b.k+4)*b.candidate_size,size_t(2*a.k+4)*a.candidate_size))bad|=64;
    if(unequal(a.values+size_t(c)*a.k,b.values+size_t(c)*b.k,size_t(a.k)*8)||
       unequal(a.vectors+size_t(c)*a.k*a.n,b.vectors+size_t(c)*b.k*b.n,size_t(a.k)*a.n*8)||
       unequal(a.means+c,b.means+c,8))bad|=128;
    if(unequal(static_cast<const char*>(a.status)+size_t(c)*a.status_size,
               static_cast<const char*>(b.status)+size_t(c)*b.status_size,a.status_size))bad|=256;
  }
  if(bad)atomicOr(&out.mask,bad);
}

long long live_allocations=0,live_bytes=0,live_streams=0,live_plans=0;
int cleanup_errors=0;
struct Memory {
  std::vector<std::pair<void*,size_t>> items;
  static void* allocate(void* owner,size_t bytes) {
    auto& m=*static_cast<Memory*>(owner);void* raw=nullptr;check(cudaMalloc(&raw,bytes+512));
    void* p=static_cast<char*>(raw)+256;
    try{m.items.push_back({p,bytes});}catch(...){cudaFree(raw);throw;}
    ++live_allocations;live_bytes+=bytes;
    check(cudaMemset(raw,0xa5,256));check(cudaMemset(p,0,bytes));
    check(cudaMemset(static_cast<char*>(p)+bytes,0xa5,256));return p;
  }
  ~Memory() noexcept {
    for(auto v:items){unsigned char guards[512]{};
      if(cudaMemcpy(guards,static_cast<char*>(v.first)-256,256,cudaMemcpyDeviceToHost)!=cudaSuccess)++cleanup_errors;
      if(cudaMemcpy(guards+256,static_cast<char*>(v.first)+v.second,256,cudaMemcpyDeviceToHost)!=cudaSuccess)++cleanup_errors;
      for(auto x:guards)if(x!=0xa5){++cleanup_errors;break;}
      if(cudaFree(static_cast<char*>(v.first)-256)!=cudaSuccess)++cleanup_errors;
      --live_allocations;live_bytes-=v.second;}
  }
};
struct Library {
  void* handle=nullptr;WpCreate create;WpDestroy destroy;WpSubmit submit;WpDirect direct;WpGet get;
  IpConfigure configure;IpEntry entry;IpRotation rotation;
  template<class T>T symbol(const char* name){auto p=dlsym(handle,name);need(p,"missing diagnostic symbol");return reinterpret_cast<T>(p);}
  Library(const std::string& path){handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
    try{create=symbol<WpCreate>("wp_create");destroy=symbol<WpDestroy>("wp_destroy");submit=symbol<WpSubmit>("wp_submit");
      direct=symbol<WpDirect>("wp_direct");get=symbol<WpGet>("wp_get");configure=symbol<IpConfigure>("ip_configure");
      entry=symbol<IpEntry>("ip_entry");rotation=symbol<IpRotation>("ip_rotation");
    }catch(...){dlclose(handle);handle=nullptr;throw;}}
  ~Library(){if(handle)dlclose(handle);}
};
struct Lane {
  Memory memory;Library &a,&b;void *pa=nullptr,*pb=nullptr;cudaStream_t stream=nullptr;
  double* input=nullptr;Result* result=nullptr;
  Lane(Library& x,Library& y,int k=35,int capacity=170662):a(x),b(y){
    try{
      check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));++live_streams;
      pa=a.create(351,k,64,capacity,stream,Memory::allocate,&memory);++live_plans;
      pb=b.create(351,k,64,capacity,stream,Memory::allocate,&memory);++live_plans;
      input=static_cast<double*>(Memory::allocate(&memory,64*351*8));
      result=static_cast<Result*>(Memory::allocate(&memory,64*sizeof(Result)));
      auto* headers=static_cast<CheckpointHeader*>(Memory::allocate(&memory,size_t(64)*ip_checkpoint_capacity*sizeof(CheckpointHeader)));
      auto* values=static_cast<unsigned long long*>(Memory::allocate(&memory,size_t(64)*ip_checkpoint_capacity*3*351*8));
      a.configure(pa,headers,values,nullptr,ip_checkpoint_capacity,0);
      b.configure(pb,headers,values,a.get(pa).checkpoints,ip_checkpoint_capacity,1);
      // All allocations/guards were initialized on the default stream.
      check(cudaStreamSynchronize(nullptr));
    }catch(...){cleanup();throw;}}
  void cleanup() noexcept {
    if(stream&&cudaStreamSynchronize(stream)!=cudaSuccess)++cleanup_errors;
    if(pa){a.destroy(pa);pa=nullptr;--live_plans;}if(pb){b.destroy(pb);pb=nullptr;--live_plans;}
    if(stream){if(cudaStreamDestroy(stream)!=cudaSuccess)++cleanup_errors;stream=nullptr;--live_streams;}
  }
  ~Lane(){cleanup();}
  void enqueue(const double* host,int count){
    check(cudaMemcpyAsync(input,host,size_t(count)*351*8,cudaMemcpyHostToDevice,stream));
    auto av=a.submit(pa,input,count),bv=b.submit(pb,input,count);
    compare<<<count,256,0,stream>>>(av,bv,result,count,0);check(cudaGetLastError());}
  std::vector<Result> finish(int count){std::vector<Result> r(count);
    check(cudaMemcpyAsync(r.data(),result,count*sizeof(Result),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return r;}
  Result run_entry(const std::vector<double>& h,EntryConfig cfg,int corrupt=-1,int helper=0){
    check(cudaMemcpyAsync(input,h.data(),h.size()*8,cudaMemcpyHostToDevice,stream));
    if(helper)a.rotation(pa,helper,cfg.capacity,0,input);else a.entry(pa,cfg,input);
    if(corrupt>=0)a.symbol<void(*)(void*,int)>("ip_corrupt_trace")(pa,corrupt);
    if(helper)b.rotation(pb,helper,cfg.capacity,0,input);else b.entry(pb,cfg,input);
    compare<<<1,256,0,stream>>>(a.get(pa),b.get(pb),result,1,1);check(cudaGetLastError());return finish(1)[0];}
  Result run_direct(const std::vector<double>& h,int r,int cap=170662,int force=0){
    return run_entry(h,{r,1,0,0,std::max(1,r),1,1,cap,force,'L'});}
};
struct Totals {
  long long targets=0,rotations=0,sweeps=0,failures=0,checkpoints=0;
  unsigned long long fields=0,stages[7]={};int max_checkpoints=0;
  unsigned coverage=0;long long branches[11]={};
  void add(const Result& r){++targets;rotations+=r.rotations;sweeps+=r.sweeps;coverage|=r.coverage;
    checkpoints+=r.checkpoints;max_checkpoints=std::max(max_checkpoints,r.checkpoints);fields+=r.fields;
    for(int j=0;j<7;++j)stages[j]+=r.stages[j];
    for(int j=0;j<11;++j)branches[j]+=bool(r.coverage&(1<<j));
    if(r.mask){++failures;std::cerr<<"mismatch target "<<targets<<" mask "<<r.mask<<" info "<<r.info
      <<" checkpoint_bad "<<r.checkpoint_bad<<" event "<<r.first_event<<" stage "<<r.first_stage<<std::endl;}}
  void json(std::ostream& f){f<<"{\"targets\":"<<targets<<",\"rotations_compared\":"<<rotations
    <<",\"sweeps\":"<<sweeps<<",\"mismatches\":"<<failures<<",\"coverage_mask\":"<<coverage
    <<",\"checkpoints\":"<<checkpoints<<",\"max_checkpoints\":"<<max_checkpoints
    <<",\"checkpoint_fields_compared\":"<<fields<<",\"checkpoint_stages\":[";
    for(int j=0;j<7;++j){if(j)f<<',';f<<stages[j];}f<<"],\"branch_targets\":[";
    for(int j=0;j<11;++j){if(j)f<<',';f<<branches[j];}f<<"]}";}
};
std::vector<double> direct_input(int r,int kind) {
  std::vector<double> h(3*351,0);
  for(int j=0;j<r;++j){h[j]=1+(j*17%29)/8.;h[351+j]=j+1<r?.125*(1+j%5):0;h[702+j]=(j%7-3)*.25;}
  if(kind==1||kind==2)for(int j=0;j<r;++j)h[j]=std::ldexp(1.,kind==1?80-160*j/std::max(1,r-1):-80+160*j/std::max(1,r-1));
  if(kind==3)for(int j=0;j<r;++j){h[j]=(j%2?-1.:1.)*2.;h[351+j]=0.;}
  if(kind==4)for(int j=0;j<r;++j){h[j]=j%3?-0.:0.;h[351+j]=0.;h[702+j]=j%2?-0.:0.;}
  if(kind==5)for(int j=0;j<r;++j){h[j]=(j<r/2?r/2-j:j-r/2+1);if(j==r/2-1)h[351+j]=0.;}
  if(kind==6)for(int j=0;j<r;++j){h[j]=std::ldexp(h[j],-450);h[351+j]=std::ldexp(h[351+j],-450);}
  if(kind==7)for(int j=0;j<r;++j){h[j]=std::ldexp(h[j],450);h[351+j]=std::ldexp(h[351+j],450);}
  return h;
}
#include "../stage_b_recover_tile_20261029/boundary_cases.inc"

int main(int argc,char** argv) {
  try {
    need(argc==5,"usage: qualify BASELINE_SO CANDIDATE_SO CAPTURE_DIR REPORT_JSON");check(cudaSetDevice(0));
    const bool smoke=std::getenv("IP_SMOKE")!=nullptr;
    Totals captured,synthetic;int overflow_checks=0,entry_cases=0,wrapper_cases=0,negative_controls=0,extra_cases=0;
    {
      Library a(argv[1]),b(argv[2]);std::vector<std::unique_ptr<Lane>> lanes;
      for(int j=0;j<(smoke?1:10);++j)lanes.push_back(std::make_unique<Lane>(a,b));
      auto& lane=*lanes[0];
      for(int r=1;r<=350;++r){if(smoke&&r!=1&&r!=2&&r!=16&&r!=350)continue;
        auto v=lane.run_direct(direct_input(r,0),r);synthetic.add(v);need(!v.mask,"dimension parity");}
      for(int r:{1,2,3,4,7,16,35,64,127,350})for(int kind=1;kind<=7;++kind){
        if(smoke&&r!=2&&r!=350)continue;
        auto v=lane.run_direct(direct_input(r,kind),r);synthetic.add(v);need(!v.mask,"synthetic parity");}
      auto input=direct_input(16,0);auto regular=lane.run_direct(input,16);
      need(!regular.mask&&regular.info==0&&regular.rotations>1,"rotation boundary setup");
      for(int capacity:{0,1,regular.rotations-1,regular.rotations}){
        auto v=lane.run_direct(input,16,capacity);synthetic.add(v);need(!v.mask,"rotation boundary parity");
        need(v.overflow==(capacity<regular.rotations?1:0),"expected rotation overflow");++overflow_checks;}
      auto forced=lane.run_direct(direct_input(350,0),350,170662,1);synthetic.add(forced);
      need(!forced.mask&&forced.info>0&&(forced.coverage&1024),"injected nonconvergence gate");
      EntryConfig normal{16,1,0,0,16,1,1,170662,0,'L'};
      for(int which=0;which<10;++which){auto cfg=normal;int expected=0;
        switch(which){case 0:cfg.r=0;break;case 1:cfg.r=-1;expected=-2;break;
          case 2:cfg.uplo='U';expected=-101;break;case 3:cfg.ncvt=0;expected=-101;break;
          case 4:cfg.nru=1;expected=-101;break;case 5:cfg.ncc=1;expected=-101;break;
          case 6:cfg.ldvt=1;expected=-9;break;case 7:cfg.ldu=0;expected=-11;break;
          case 8:cfg.ldc=0;expected=-13;break;case 9:cfg.ldvt=cfg.ldu=cfg.ldc=0;expected=-9;break;}
        auto v=lane.run_entry(input,cfg);synthetic.add(v);need(!v.mask&&v.info==expected,"entry error precedence");++entry_cases;}
      for(int helper:{1,2})for(int sign:{0,1})for(int capacity:{0,1}){
        auto h=direct_input(2,0);h[0]=1.;h[1]=sign?-0.:0.;auto cfg=normal;cfg.r=2;cfg.capacity=capacity;
        auto v=lane.run_entry(h,cfg,-1,helper);synthetic.add(v);
        need(!v.mask&&v.rotations==(helper==2?capacity:0)&&v.overflow==(helper==2&&capacity==0),"identity wrapper behavior");++wrapper_cases;}
      for(int helper:{1,2}){auto h=direct_input(2,0);h[0]=.6;h[1]=-.8;auto cfg=normal;cfg.r=2;cfg.capacity=0;
        auto v=lane.run_entry(h,cfg,-1,helper);synthetic.add(v);need(!v.mask&&v.overflow==1,"overflow q completion");++wrapper_cases;}
      for(int numeric:{0,1}){auto v=lane.run_entry(input,normal,numeric);
        need(v.mask==512&&(v.checkpoint_bad&(numeric?32:8)),"checkpoint negative control missed corruption");++negative_controls;}
      for(int kind=0;kind<8;++kind){std::vector<double> h(351);
        for(int j=0;j<351;++j)h[j]=kind==0?1.:kind==1?(j%2?-0.:0.):double(j%std::max(2,kind));
        if(kind==6)h[0]=std::numeric_limits<double>::infinity();
        if(kind==7)h[0]=std::numeric_limits<double>::quiet_NaN();
        lane.enqueue(h.data(),1);auto v=lane.finish(1)[0];synthetic.add(v);need(!v.mask,"prepare/boundary parity");}
      extra_cases=recover_boundaries(a,b,synthetic);
      std::cout<<"PASS synthetic cases "<<synthetic.targets<<" coverage "<<synthetic.coverage
        <<" checkpoint fields "<<synthetic.fields<<std::endl;
      for(int level:{2,3}){
        const int total=smoke?8:(level==2?49919:54878);
        std::ifstream raw(std::string(argv[3])+"/l"+std::to_string(level)+"-residuals.f64",std::ios::binary);
        need(raw.good(),"missing captured residuals");
        for(int base=0;base<total;base+=640){int count=std::min(640,total-base);std::vector<double> h(size_t(count)*351);
          raw.read(reinterpret_cast<char*>(h.data()),h.size()*8);need(size_t(raw.gcount())==h.size()*8,"capture truncated");
          for(int j=0;j<(count+63)/64;++j)lanes[j]->enqueue(h.data()+size_t(j)*64*351,std::min(64,count-j*64));
          for(int j=0;j<(count+63)/64;++j)for(auto v:lanes[j]->finish(std::min(64,count-j*64))){captured.add(v);need(!v.mask,"captured parity");}
          if(base%6400==0)std::cout<<"PASS level "<<level<<" through "<<base+count<<" / "<<total
            <<" max checkpoints "<<captured.max_checkpoints<<std::endl;
        }
        if(!smoke)need(raw.peek()==std::char_traits<char>::eof(),"capture trailing bytes");
      }
      lanes.clear();
    }
    need(live_allocations==0&&live_bytes==0&&live_streams==0&&live_plans==0&&cleanup_errors==0,"diagnostic resource closure");
    check(cudaDeviceSynchronize());need(captured.targets==(smoke?16:104797),"captured coverage");
    if(!smoke)need(((captured.coverage|synthetic.coverage)&2047)==2047,"missing branch coverage");
    need(captured.stages[1]==static_cast<unsigned long long>(captured.sweeps),"missing sweep checkpoints");
    std::ofstream f(argv[4]);need(f.good(),"report output");
    f<<"{\"status\":\"PASS_GPU_BITWISE_QUALIFICATION\",\"captured\":";captured.json(f);
    f<<",\"synthetic\":";synthetic.json(f);
    f<<",\"rotation_capacity_cases\":"<<overflow_checks<<",\"entry_cases\":"<<entry_cases
      <<",\"wrapper_cases\":"<<wrapper_cases<<",\"negative_controls\":"<<negative_controls
      <<",\"recover_boundary_cases\":"<<extra_cases<<",\"checkpoint_capacity\":"<<ip_checkpoint_capacity
      <<",\"live_device_allocations\":0,\"live_device_bytes\":0,\"live_streams\":0,\"live_plans\":0"
      <<",\"cleanup_errors\":0,\"cpu_numerical_authority\":false,\"performance_evidence\":false}\n";
    std::cout<<"PASS GPU fields/checkpoints/log/payload and resource closure"<<std::endl;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" resources "<<live_allocations<<' '<<live_bytes
      <<' '<<live_streams<<' '<<live_plans<<" cleanup_errors "<<cleanup_errors<<std::endl;return 1;}
}
