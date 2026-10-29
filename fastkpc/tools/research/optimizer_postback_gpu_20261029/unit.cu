// Both numerical references and candidates run on GPU. Host generates bit-pattern
// inputs and checks counters only; it does not evaluate reflectors or SVD.
#define FASTKPC_LAPACK_SMALL_MAX_COLUMNS 64
#define FASTKPC_LAPACK_SMALL_NAMESPACE postback_unit_lapack
#include "cuda/lapack_312_small_dgesdd.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace lp = fastkpc::postback_unit_lapack;
constexpr int G=64, VC=92*46, TC=65*32, CC=92*46, WC=3*46*46+4*46, AC=46;
constexpr int VO=G, TO=VO+VC+2*G, CO=TO+TC+2*G, WO=CO+CC+2*G, AO=WO+WC+2*G;
constexpr int STRIDE=AO+AC+G;
constexpr std::uint64_t POISON=0x7ff80000badbeef1ULL;
struct Case { int q,left,rows,offset,ib,m,n,ldv,ldc,outer,pattern,seed; };
void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}

template<bool Candidate> __global__ void run(Case* cases,double* data,int* infos,int count){
  const int index=blockIdx.x;
  if(index>=count)return;
  const Case x=cases[index];double* b=data+std::size_t(index)*STRIDE;
  int m=x.m,n=x.n,k=x.ib,ldv=x.ldv,ldc=x.ldc,ldt=65,ldw=x.q,info=0;
  if(threadIdx.x<32){
    if(x.outer){
      if constexpr(Candidate){
        info=lp::postback_apply<true>(x.left,m,x.q,b+VO,ldv,b+AO,b+CO,ldc,b+WO);
      }else if(threadIdx.x==0){
        char vect=x.left?'Q':'P',side=x.left?'L':'R',trans=x.left?'N':'T';
        int q=x.q,lwork=3*q*q+4*q;
        lp::reference_blocked_reflector::dormbr_reference::dormbr_(
          &vect,&side,&trans,&m,&n,&q,b+VO,&ldv,b+AO,b+CO,&ldc,b+WO,&lwork,&info,1,1,1);
      }
    }else{
      char side=x.left?'L':'R',trans=x.left?'N':'T',direct='F',store=x.left?'C':'R';
      if constexpr(Candidate){
        for(int v=threadIdx.x;v<x.q;v+=32){
          int sm=x.left?m:1,sn=x.left?1:n;
          lp::reference_blocked_reflector::dlarfb_(&side,&trans,&direct,&store,
            &sm,&sn,&k,b+VO,&ldv,b+TO,&ldt,b+CO+(x.left?v*ldc:v),&ldc,
            b+WO+v,&ldw,1,1,1,1);
        }
      }else if(threadIdx.x==0){
        lp::reference_blocked_reflector::dlarfb_(&side,&trans,&direct,&store,
          &m,&n,&k,b+VO,&ldv,b+TO,&ldt,b+CO,&ldc,b+WO,&ldw,1,1,1,1);
      }
    }
    __syncwarp();
  }
  __syncthreads();
  if(threadIdx.x==0)infos[index]=info;
}

__global__ void compare(const Case* cases,const unsigned long long* input,
    const unsigned long long* ref,const unsigned long long* candidate,
    const int* ri,const int* ci,unsigned long long* errors,int count){
  for(std::size_t flat=blockIdx.x*blockDim.x+threadIdx.x;
      flat<std::size_t(count)*STRIDE;flat+=gridDim.x*blockDim.x){
    int test=flat/STRIDE,at=flat%STRIDE;Case x=cases[test];
    if(ref[flat]!=candidate[flat]){atomicAdd(errors,1ULL);atomicMin(errors+4,static_cast<unsigned long long>(flat));}
    const int pos=at-CO;
    const bool c=pos>=0&&pos<x.ldc*x.n&&pos%x.ldc<x.m;
    const bool w=at>=WO&&at<WO+(x.outer?3*x.q*x.q+4*x.q:x.q*x.ib);
    if(!c&&!w&&(input[flat]!=ref[flat]||input[flat]!=candidate[flat]))atomicAdd(errors+1,1ULL);
    if(c&&x.pattern<3){
      const auto a=ref[flat]&0x7fffffffffffffffULL,b=candidate[flat]&0x7fffffffffffffffULL;
      if(a>0x7ff0000000000000ULL||b>0x7ff0000000000000ULL)atomicAdd(errors+2,1ULL);
    }
    if(at==0&&(ri[test]!=0||ci[test]!=0))atomicAdd(errors+3,1ULL);
  }
}
std::uint64_t random_bits(std::uint64_t& state){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
std::uint64_t value(std::uint64_t& state,int pattern,bool special){
  auto r=random_bits(state);
  if(pattern>=1&&r%10<3)return (r&1)<<63;
  if(pattern==3&&special&&r%100>=97){
    if(r%3==0)return 0x7ff0000000000000ULL;
    if(r%3==1)return 0xfff0000000000000ULL;
    return 0x7ff8000000000000ULL|(r&0x0007ffffffffffffULL);
  }
  return (r&0x800fffffffffffffULL)|((std::uint64_t(1018)+((r>>52)%6))<<52);
}
void fill(const Case& x,std::uint64_t* p,std::size_t id){
  std::fill(p,p+STRIDE,POISON);std::uint64_t state=0x9e3779b97f4a7c15ULL^(id+1)*1000003;
  if(x.outer){
    for(int j=0;j<x.q;++j)for(int i=0;i<x.ldv;++i)p[VO+i+x.ldv*j]=value(state,x.pattern,false);
    for(int i=0;i<x.q;++i)p[AO+i]=value(state,x.pattern,false);
  }else{
    for(int j=0;j<(x.left?x.ib:x.n);++j)for(int i=0;i<(x.left?x.m:x.ib);++i){
      bool unref=x.left?(i<x.ib&&i<=j):(j<x.ib&&i>=j);
      if(!unref)p[VO+i+x.ldv*j]=value(state,x.pattern,false);
    }
    for(int j=0;j<x.ib;++j)for(int i=0;i<=j;++i)
      p[TO+i+65*j]=(x.pattern==2&&i==j)?((random_bits(state)&1)?0x3ff0000000000000ULL:0):value(state,x.pattern,false);
  }
  for(int j=0;j<x.n;++j)for(int i=0;i<x.m;++i)p[CO+i+x.ldc*j]=value(state,x.pattern,true);
}
int main(int argc,char** argv)try{
  need(argc==2||argc==3,"unit BLOCKS_TSV [--reduced]");
  const bool reduced=argc==3;need(!reduced||std::string(argv[2])=="--reduced","mode");
  std::ifstream input(argv[1]);need(bool(input),"open inventory");std::string line;
  std::vector<Case> shapes,cases;
  while(std::getline(input,line)){
    Case x{};std::string side,path;std::istringstream row(line);
    need(bool(row>>x.q>>side>>x.rows>>path>>x.offset>>x.ib>>x.m>>x.n>>x.ldv>>x.ldc),"block row");
    need((row>>std::ws).eof()&&(x.q==37||x.q==46)&&(side=="Q"||side=="P"),"block identity");
    x.left=side=="Q";shapes.push_back(x);
  }
  need(shapes.size()==1553,"complete block coverage");
  for(int q:{37,46})for(int rows=q+1;rows<=2*q;++rows)for(int left:{0,1}){
    Case x{};x.q=q;x.rows=rows;x.left=left;x.outer=1;
    x.ldv=rows<11*q/6?rows:q;x.m=left?x.ldv:q;x.n=q;x.ldc=x.m;
    shapes.push_back(x);
  }
  need(shapes.size()==1719,"outer shape coverage");
  for(auto x:shapes)for(int pattern=0;pattern<(reduced?1:4);++pattern)for(int seed=0;seed<(reduced?1:3);++seed){
    x.pattern=pattern;x.seed=seed;cases.push_back(x);
  }
  check(cudaSetDevice(0));constexpr int batch=128;const std::size_t bytes=std::size_t(batch)*STRIDE*8;
  std::uint64_t *di=nullptr,*dr=nullptr,*dc=nullptr;Case* ds=nullptr;int *ri=nullptr,*ci=nullptr;unsigned long long* de=nullptr;
  check(cudaMalloc(&di,bytes));check(cudaMalloc(&dr,bytes));check(cudaMalloc(&dc,bytes));check(cudaMalloc(&ds,batch*sizeof(Case)));
  check(cudaMalloc(&ri,batch*sizeof(int)));check(cudaMalloc(&ci,batch*sizeof(int)));check(cudaMalloc(&de,5*8));
  unsigned long long totals[4]={};std::size_t completed=0;
  for(std::size_t start=0;start<cases.size();start+=batch){
    int count=std::min<std::size_t>(batch,cases.size()-start);std::vector<std::uint64_t> h(std::size_t(count)*STRIDE);
    for(int i=0;i<count;++i)fill(cases[start+i],h.data()+std::size_t(i)*STRIDE,start+i);
    check(cudaMemcpy(di,h.data(),h.size()*8,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dr,di,h.size()*8,cudaMemcpyDeviceToDevice));check(cudaMemcpy(dc,di,h.size()*8,cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(ds,cases.data()+start,count*sizeof(Case),cudaMemcpyHostToDevice));
    unsigned long long errors[5]={0,0,0,0,~0ULL};check(cudaMemcpy(de,errors,sizeof(errors),cudaMemcpyHostToDevice));
    run<false><<<count,64>>>(ds,reinterpret_cast<double*>(dr),ri,count);check(cudaGetLastError());
    run<true><<<count,64>>>(ds,reinterpret_cast<double*>(dc),ci,count);check(cudaGetLastError());
    compare<<<256,256>>>(ds,reinterpret_cast<unsigned long long*>(di),reinterpret_cast<unsigned long long*>(dr),
      reinterpret_cast<unsigned long long*>(dc),ri,ci,de,count);check(cudaGetLastError());
    check(cudaMemcpy(errors,de,sizeof(errors),cudaMemcpyDeviceToHost));
    for(int j=0;j<4;++j)totals[j]+=errors[j];completed+=count;
    if(errors[0]||errors[1]||errors[2]||errors[3]){
      std::cerr<<"FAIL batch_start="<<start<<" diff="<<errors[0]<<" readonly="<<errors[1]
        <<" finite_nan="<<errors[2]<<" info="<<errors[3]<<" first_flat="<<errors[4]<<'\n';
      if(errors[4]!=~0ULL){const auto& x=cases[start+errors[4]/STRIDE];
        std::cerr<<"q="<<x.q<<" side="<<(x.left?'Q':'P')<<" rows="<<x.rows<<" offset="<<x.offset
          <<" ib="<<x.ib<<" outer="<<x.outer<<" pattern="<<x.pattern<<" seed="<<x.seed<<'\n';}
      break;
    }
  }
  check(cudaFree(de));check(cudaFree(ci));check(cudaFree(ri));check(cudaFree(ds));
  check(cudaFree(dc));check(cudaFree(dr));check(cudaFree(di));check(cudaDeviceSynchronize());
  std::cout<<"{\"cases\":"<<completed<<",\"expected\":"<<cases.size()<<",\"bit_mismatches\":"<<totals[0]
    <<",\"readonly_or_guard_changes\":"<<totals[1]<<",\"finite_output_nan_count\":"<<totals[2]
    <<",\"info_errors\":"<<totals[3]<<",\"gpu_reference_and_candidate\":true}"<<std::endl;
  need(completed==cases.size()&&!totals[0]&&!totals[1]&&!totals[2]&&!totals[3],"GPU unit parity failed");
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
