// Activity tracing only, no hardware counters and no serialized kernel tracing.
#include <cupti.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
#pragma pack(push,1)
struct Event {
  uint64_t start=0,end=0,bytes=0;
  uint32_t kind=0,device=0,context=0,stream=0,correlation=0,name=0,detail=0,thread=0,
    gx=0,gy=0,gz=0,bx=0,by=0,bz=0,regs=0,ss=0,ds=0,lmem=0,channel=0;
};
#pragma pack(pop)
static_assert(sizeof(Event)==100,"event schema changed");
std::mutex mutex;
FILE *events=nullptr,*names=nullptr,*markers=nullptr;
std::unordered_map<std::string,uint32_t> ids;
std::atomic<uint64_t> dropped{0},invalid{0},count{0};
std::string directory;
void check(CUptiResult r) {
  if(r!=CUPTI_SUCCESS){const char* text=nullptr;cuptiGetResultString(r,&text);
    std::fprintf(stderr,"CUPTI activity failure: %s\n",text?text:"unknown");std::abort();}
}
uint32_t name_id(const std::string& name) {
  auto p=ids.find(name);if(p!=ids.end())return p->second;
  uint32_t id=ids.size();ids.emplace(name,id);std::fprintf(names,"%u\t%s\n",id,name.c_str());return id;
}
void CUPTIAPI request(uint8_t** buffer,size_t* bytes,size_t* max) {
  *bytes=64U<<20;*buffer=static_cast<uint8_t*>(std::aligned_alloc(8,*bytes));*max=0;
  if(!*buffer)std::abort();
}
void CUPTIAPI complete(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    CUpti_Activity* p=nullptr;CUptiResult next;
    while((next=cuptiActivityGetNextRecord(buffer,valid,&p))==CUPTI_SUCCESS) {
      Event e;
      if(p->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto* k=reinterpret_cast<CUpti_ActivityKernel9*>(p);e.kind=1;e.start=k->start;e.end=k->end;
        e.device=k->deviceId;e.context=k->contextId;e.stream=k->streamId;e.correlation=k->correlationId;
        e.name=name_id(k->name);e.gx=k->gridX;e.gy=k->gridY;e.gz=k->gridZ;
        e.bx=k->blockX;e.by=k->blockY;e.bz=k->blockZ;e.regs=k->registersPerThread;
        e.ss=k->staticSharedMemory;e.ds=k->dynamicSharedMemory;e.lmem=k->localMemoryPerThread;e.channel=k->channelID;
      } else if(p->kind==CUPTI_ACTIVITY_KIND_MEMCPY) {
        auto* m=reinterpret_cast<CUpti_ActivityMemcpy5*>(p);e.kind=2;e.start=m->start;e.end=m->end;
        e.bytes=m->bytes;e.device=m->deviceId;e.context=m->contextId;e.stream=m->streamId;
        e.correlation=m->correlationId;e.detail=m->copyKind;e.channel=m->channelID;e.name=m->runtimeCorrelationId;
      } else if(p->kind==CUPTI_ACTIVITY_KIND_MEMSET) {
        auto* m=reinterpret_cast<CUpti_ActivityMemset4*>(p);e.kind=3;e.start=m->start;e.end=m->end;
        e.bytes=m->bytes;e.device=m->deviceId;e.context=m->contextId;e.stream=m->streamId;
        e.correlation=m->correlationId;e.detail=m->memoryKind;e.channel=m->channelID;
      } else if(p->kind==CUPTI_ACTIVITY_KIND_RUNTIME||p->kind==CUPTI_ACTIVITY_KIND_DRIVER) {
        auto* a=reinterpret_cast<CUpti_ActivityAPI*>(p);e.kind=p->kind==CUPTI_ACTIVITY_KIND_RUNTIME?4:5;
        e.start=a->start;e.end=a->end;e.correlation=a->correlationId;e.thread=a->threadId;e.detail=a->cbid;
        const char* name=nullptr;check(cuptiGetCallbackName(e.kind==4?CUPTI_CB_DOMAIN_RUNTIME_API:CUPTI_CB_DOMAIN_DRIVER_API,
          a->cbid,&name));e.name=name_id(name?name:"unknown API");
      } else if(p->kind==CUPTI_ACTIVITY_KIND_SYNCHRONIZATION) {
        auto* s=reinterpret_cast<CUpti_ActivitySynchronization*>(p);e.kind=6;e.start=s->start;e.end=s->end;
        e.context=s->contextId;e.stream=s->streamId;e.correlation=s->correlationId;e.detail=s->type;e.name=s->cudaEventId;
      } else continue;
      if(e.start==0||e.end<e.start)++invalid;
      if(std::fwrite(&e,sizeof(e),1,events)!=1)std::abort();++count;
    }
    if(next!=CUPTI_ERROR_MAX_LIMIT_REACHED)check(next);
  }
  size_t lost=0;check(cuptiActivityGetNumDroppedRecords(context,stream,&lost));dropped+=lost;
  std::free(buffer);
}
void finish() {
  check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  std::lock_guard<std::mutex> lock(mutex);
  std::fclose(events);std::fclose(names);std::fclose(markers);
  FILE* f=std::fopen((directory+"/trace-summary.json").c_str(),"wx");if(!f)std::abort();
  std::fprintf(f,"{\"schema\":\"stage-b-activity-v1\",\"event_bytes\":100,\"events\":%llu,\"dropped\":%llu,\"invalid_timestamps\":%llu,\"hardware_counters\":false}\n",
    (unsigned long long)count.load(),(unsigned long long)dropped.load(),(unsigned long long)invalid.load());std::fclose(f);
}
}
extern "C" void stage_b_trace_mark(const char* name,int level) {
  uint64_t now=0;check(cuptiGetTimestamp(&now));std::lock_guard<std::mutex> lock(mutex);
  std::fprintf(markers,"%llu\t%d\t%s\n",(unsigned long long)now,level,name);
}
extern "C" int InitializeInjection() {
  const char* dir=std::getenv("STAGE_B_TRACE");if(!dir)std::abort();directory=dir;
  events=std::fopen((directory+"/events.bin").c_str(),"wbx");
  names=std::fopen((directory+"/names.tsv").c_str(),"wx");
  markers=std::fopen((directory+"/markers.tsv").c_str(),"wx");
  if(!events||!names||!markers)std::abort();
  check(cuptiActivityRegisterCallbacks(request,complete));
  for(auto kind:{CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,CUPTI_ACTIVITY_KIND_RUNTIME,CUPTI_ACTIVITY_KIND_DRIVER,
      CUPTI_ACTIVITY_KIND_SYNCHRONIZATION})check(cuptiActivityEnable(kind));
  check(cuptiActivityFlushPeriod(1000));std::atexit(finish);return 1;
}
