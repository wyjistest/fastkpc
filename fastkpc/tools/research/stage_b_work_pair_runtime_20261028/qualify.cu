#include "diagnostic.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Result {unsigned mask;int coverage,rotations,info,overflow,sweeps;};
__device__ bool unequal(const void* av,const void* bv,size_t bytes) {
  auto* a=static_cast<const unsigned char*>(av);auto* b=static_cast<const unsigned char*>(bv);
  for(size_t i=threadIdx.x;i<bytes;i+=blockDim.x)if(a[i]!=b[i])return true;
  return false;
}
__global__ void compare(WorkPairView a,WorkPairView b,Result* result,int count,int direct) {
  const int c=blockIdx.x;if(c>=count)return;
  auto& out=result[c];const WorkPairState sa=a.state[c],sb=b.state[c];
  if(!threadIdx.x)out={0,sa.coverage,sa.count,sa.info,sa.overflow,sa.sweeps};
  __syncthreads();
  unsigned bad=0;
  if(unequal(a.state+c,b.state+c,sizeof(WorkPairState))||sa.canary||sb.canary)bad|=1;
  for(int j=0;j<3;++j)if(unequal(a.deq+(size_t(c)*3+j)*a.n,b.deq+(size_t(c)*3+j)*b.n,size_t(sa.r>0?sa.r:0)*8))bad|=2;
  if(sa.count!=sb.count||sa.count<0||sa.count>a.capacity)bad|=4;
  else if(unequal(static_cast<const char*>(a.rotations)+size_t(c)*a.capacity*a.rotation_size,
                  static_cast<const char*>(b.rotations)+size_t(c)*b.capacity*b.rotation_size,size_t(sa.count)*a.rotation_size))bad|=4;
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

long long live_allocations=0,live_bytes=0;
struct Memory {
  std::vector<std::pair<void*,size_t>> items;
  static void* allocate(void* owner,size_t bytes) {
    auto& m=*static_cast<Memory*>(owner);void* p=nullptr;check(cudaMalloc(&p,bytes));
    check(cudaMemset(p,0,bytes));m.items.push_back({p,bytes});++live_allocations;live_bytes+=bytes;return p;
  }
  ~Memory(){for(auto v:items){cudaFree(v.first);--live_allocations;live_bytes-=v.second;}}
};
struct Library {
  void* handle;WpCreate create;WpDestroy destroy;WpSubmit submit;WpDirect direct;WpGet get;
  template<class T>T symbol(const char* name){auto p=dlsym(handle,name);need(p,"missing diagnostic symbol");return reinterpret_cast<T>(p);}
  Library(const std::string& path){handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!handle)throw std::runtime_error(dlerror());
    create=symbol<WpCreate>("wp_create");destroy=symbol<WpDestroy>("wp_destroy");submit=symbol<WpSubmit>("wp_submit");
    direct=symbol<WpDirect>("wp_direct");get=symbol<WpGet>("wp_get");}
  ~Library(){dlclose(handle);}
};
struct Lane {
  Memory memory;Library &a,&b;void *pa,*pb;cudaStream_t stream;double* input;Result* result;
  Lane(Library& x,Library& y):a(x),b(y){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    pa=a.create(351,35,64,170662,stream,Memory::allocate,&memory);
    pb=b.create(351,35,64,170662,stream,Memory::allocate,&memory);
    input=static_cast<double*>(Memory::allocate(&memory,64*351*8));
    result=static_cast<Result*>(Memory::allocate(&memory,64*sizeof(Result)));}
  ~Lane(){cudaStreamSynchronize(stream);a.destroy(pa);b.destroy(pb);cudaStreamDestroy(stream);}
  void enqueue(const double* host,int count){check(cudaMemcpyAsync(input,host,size_t(count)*351*8,cudaMemcpyHostToDevice,stream));
    auto av=a.submit(pa,input,count),bv=b.submit(pb,input,count);
    compare<<<count,256,0,stream>>>(av,bv,result,count,0);check(cudaGetLastError());}
  std::vector<Result> finish(int count){std::vector<Result> r(count);
    check(cudaMemcpyAsync(r.data(),result,count*sizeof(Result),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return r;}
  Result run_direct(const std::vector<double>& h,int r,int cap=170662,int force=0){
    check(cudaMemcpyAsync(input,h.data(),h.size()*8,cudaMemcpyHostToDevice,stream));
    a.direct(pa,r,cap,force,input);b.direct(pb,r,cap,force,input);
    compare<<<1,256,0,stream>>>(a.get(pa),b.get(pb),result,1,1);check(cudaGetLastError());return finish(1)[0];}
};
struct Totals {
  long long targets=0,rotations=0,sweeps=0,failures=0;
  unsigned coverage=0;long long branches[11]={};
  void add(const Result& r){++targets;rotations+=r.rotations;sweeps+=r.sweeps;coverage|=r.coverage;
    for(int j=0;j<11;++j)branches[j]+=bool(r.coverage&(1<<j));
    if(r.mask){++failures;std::cerr<<"mismatch target "<<targets<<" mask "<<r.mask<<" info "<<r.info<<std::endl;}}
  void json(std::ostream& f){f<<"{\"targets\":"<<targets<<",\"rotations_compared\":"<<rotations
    <<",\"sweeps\":"<<sweeps<<",\"mismatches\":"<<failures<<",\"coverage_mask\":"<<coverage<<",\"branch_targets\":[";
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
int main(int argc,char** argv) {
  try {
    need(argc==5,"usage: qualify BASELINE_SO CANDIDATE_SO CAPTURE_DIR REPORT_JSON");check(cudaSetDevice(0));
    Totals captured,synthetic;int overflow_checks=0,injected_nonconvergence=0,precheck_cases=0;
    {
      Library a(argv[1]),b(argv[2]);std::vector<Lane*> lanes;
      for(int j=0;j<10;++j)lanes.push_back(new Lane(a,b));
      auto& lane=*lanes[0];
      for(int r=1;r<=350;++r){auto v=lane.run_direct(direct_input(r,0),r);synthetic.add(v);need(!v.mask,"dimension parity");}
      for(int r:{1,2,3,4,7,16,35,64,127,350})for(int kind=1;kind<=7;++kind){
        auto v=lane.run_direct(direct_input(r,kind),r);synthetic.add(v);need(!v.mask,"synthetic parity");}
      auto input=direct_input(16,0);auto regular=lane.run_direct(input,16);need(!regular.mask&&regular.info==0&&regular.rotations>1,"rotation boundary setup");
      for(int capacity:{0,1,regular.rotations-1,regular.rotations}){
        auto v=lane.run_direct(input,16,capacity);synthetic.add(v);need(!v.mask,"rotation boundary parity");
        need(v.overflow==(capacity<regular.rotations?1:0),"rotation boundary expected overflow");++overflow_checks;}
      auto forced=lane.run_direct(direct_input(350,0),350,170662,1);synthetic.add(forced);
      need(!forced.mask&&forced.info>0&&(forced.coverage&1024),"injected nonconvergence gate");++injected_nonconvergence;
      for(int kind=0;kind<8;++kind){std::vector<double> h(351);
        for(int j=0;j<351;++j)h[j]=kind==0?1.:kind==1?(j%2?-0.:0.):double(j%std::max(2,kind));
        if(kind==6)h[0]=std::numeric_limits<double>::infinity();
        if(kind==7)h[0]=std::numeric_limits<double>::quiet_NaN();
        lane.enqueue(h.data(),1);auto v=lane.finish(1)[0];synthetic.add(v);need(!v.mask,"prepare/boundary parity");++precheck_cases;}
      std::cout<<"PASS synthetic cases "<<synthetic.targets<<" coverage "<<synthetic.coverage<<std::endl;
      const int counts[] = {0,0,49919,54878,17767,6816,1148,124,18};
      for(int level:{4,5,6,7,8}) {
        const int total=counts[level];
        std::ifstream raw(std::string(argv[3])+"/l"+std::to_string(level)+"-residuals.f64",std::ios::binary);
        need(raw.good(),"missing captured residuals");
        for(int base=0;base<total;base+=640){int count=std::min(640,total-base);std::vector<double> h(size_t(count)*351);
          raw.read(reinterpret_cast<char*>(h.data()),h.size()*8);need(size_t(raw.gcount())==h.size()*8,"capture truncated");
          for(int j=0;j<(count+63)/64;++j)lanes[j]->enqueue(h.data()+size_t(j)*64*351,std::min(64,count-j*64));
          for(int j=0;j<(count+63)/64;++j)for(auto v:lanes[j]->finish(std::min(64,count-j*64))){captured.add(v);need(!v.mask,"captured parity");}
          if(base%6400==0)std::cout<<"PASS level "<<level<<" through "<<base+count<<" / "<<total<<std::endl;
        }
        need(raw.peek()==std::char_traits<char>::eof(),"capture trailing bytes");
      }
      for(auto* p:lanes)delete p;
    }
    need(live_allocations==0&&live_bytes==0,"diagnostic allocation leak");check(cudaDeviceSynchronize());
    need(captured.targets==25873,"captured coverage");
    // Every sweep family must be exercised naturally in captured or constructed inputs.
    need(((captured.coverage|synthetic.coverage)&127)==127,"missing sweep branch coverage");
    std::ofstream f(argv[4]);need(f.good(),"report output");
    f<<"{\"status\":\"PASS_GPU_BITWISE_QUALIFICATION\",\"captured\":";captured.json(f);
    f<<",\"synthetic\":";synthetic.json(f);
    f<<",\"rotation_capacity_cases\":"<<overflow_checks<<",\"injected_nonconvergence_cases\":"<<injected_nonconvergence
     <<",\"prepare_boundary_cases\":"<<precheck_cases<<",\"live_device_allocations\":"<<live_allocations
     <<",\"live_device_bytes\":"<<live_bytes<<",\"cpu_numerical_authority\":false,\"performance_evidence\":false}\n";
    std::cout<<"PASS all GPU state/log/output comparisons and resource closure"<<std::endl;
  }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
