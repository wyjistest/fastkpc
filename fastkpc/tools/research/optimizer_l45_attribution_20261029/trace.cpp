// CUPTI activity only; external correlation gives exact setup identity.
#include <cupti.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
namespace {
#pragma pack(push,1)
struct Kernel {uint64_t start,end,lmem_total;uint32_t stream,corr,name,lmem,channel,grid,block;uint16_t regs;};
struct Runtime {uint64_t start,end;uint32_t corr,cbid,thread;};
struct External {uint64_t external;uint32_t corr,kind;};
#pragma pack(pop)
std::mutex mutex;
FILE *kernels=nullptr,*runtime=nullptr,*external=nullptr,*names=nullptr;
std::unordered_map<std::string,uint32_t> ids;
std::atomic<uint64_t> nk{0},nr{0},ne{0},dropped{0},invalid{0};
std::string dir;
void check(CUptiResult x){if(x!=CUPTI_SUCCESS){const char* s=nullptr;cuptiGetResultString(x,&s);std::fprintf(stderr,"trace %s\n",s);std::abort();}}
template<class T>void put(FILE* f,const T& x){if(std::fwrite(&x,sizeof(x),1,f)!=1)std::abort();}
void CUPTIAPI requested(uint8_t** buffer,size_t* size,size_t* max){*size=size_t(64)<<20;*buffer=static_cast<uint8_t*>(std::aligned_alloc(8,*size));*max=0;if(!*buffer)std::abort();}
void CUPTIAPI completed(CUcontext ctx,uint32_t stream,uint8_t* buffer,size_t,size_t valid){
  {std::lock_guard<std::mutex> lock(mutex);CUpti_Activity* record=nullptr;
   CUptiResult result;
   while((result=cuptiActivityGetNextRecord(buffer,valid,&record))==CUPTI_SUCCESS){
    if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){
      const auto* k=reinterpret_cast<const CUpti_ActivityKernel9*>(record);std::string name=k->name?k->name:"?";
      auto it=ids.find(name);if(it==ids.end()){uint32_t id=ids.size();it=ids.emplace(name,id).first;std::fprintf(names,"%u\t%s\n",id,name.c_str());}
      put(kernels,Kernel{k->start,k->end,k->localMemoryTotal_v2,k->streamId,k->correlationId,it->second,k->localMemoryPerThread,k->channelID,uint32_t(k->gridX),uint32_t(k->blockX),k->registersPerThread});
      ++nk;if(!k->start||k->end<k->start)++invalid;
    }else if(record->kind==CUPTI_ACTIVITY_KIND_RUNTIME){auto* r=reinterpret_cast<const CUpti_ActivityAPI*>(record);
      put(runtime,Runtime{r->start,r->end,r->correlationId,r->cbid,r->threadId});++nr;if(!r->start||r->end<r->start)++invalid;
    }else if(record->kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION){auto* r=reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record);
      put(external,External{r->externalId,r->correlationId,uint32_t(r->externalKind)});++ne;}
   }if(result!=CUPTI_ERROR_MAX_LIMIT_REACHED)check(result);
  }
  size_t count=0;check(cuptiActivityGetNumDroppedRecords(ctx,stream,&count));dropped+=count;std::free(buffer);
}
void finish(){check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));std::lock_guard<std::mutex> lock(mutex);
  for(FILE* f:{kernels,runtime,external,names})if(std::fclose(f))std::abort();
  FILE* f=std::fopen((dir+"/trace.json").c_str(),"wx");if(!f)std::abort();
  std::fprintf(f,"{\"kernels\":%llu,\"runtime\":%llu,\"external\":%llu,\"dropped\":%llu,\"invalid\":%llu}\n",
    (unsigned long long)nk.load(),(unsigned long long)nr.load(),(unsigned long long)ne.load(),(unsigned long long)dropped.load(),(unsigned long long)invalid.load());std::fclose(f);
}
}
extern "C" int InitializeInjection(){
  const char* p=std::getenv("KTRACE_DIR");if(!p)std::abort();dir=p;
  kernels=std::fopen((dir+"/kernels.bin").c_str(),"wbx");runtime=std::fopen((dir+"/runtime.bin").c_str(),"wbx");
  external=std::fopen((dir+"/external.bin").c_str(),"wbx");names=std::fopen((dir+"/kernel-names.tsv").c_str(),"wx");
  if(!kernels||!runtime||!external||!names)std::abort();
  check(cuptiActivityRegisterCallbacks(requested,completed));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
  check(cuptiActivityFlushPeriod(2000));std::atexit(finish);return 1;
}
