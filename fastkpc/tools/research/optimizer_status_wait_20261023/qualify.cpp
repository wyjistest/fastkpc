// Exercise the actual capacity-dispatched native. CUDA interposition observes
// only test behavior; no failure hook or test counter enters production code.
#include "../optimizer_windows_20261022/fixture.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>

using namespace fastkpc;
using namespace window_diag;
namespace audit {
struct Entry { unsigned flags; int device; cudaStream_t stream=nullptr; };
std::mutex mutex;
std::map<cudaEvent_t, Entry> events;
std::atomic<bool> active{false};
std::atomic<int> creates{0}, records{0}, waits{0}, downloads{0}, failure{0};
std::atomic<int> wrong_device{0};
template<class F> F real(const char* name) {
  auto p=reinterpret_cast<F>(dlsym(RTLD_NEXT,name));
  if(!p) std::abort();
  return p;
}
bool blocking(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(mutex);
  const auto it=events.find(event);
  return it!=events.end() && (it->second.flags & cudaEventBlockingSync);
}
bool producer_topology(cudaEvent_t event,cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(mutex);
  const auto producer=events.find(event);
  if(producer==events.end() || producer->second.flags!=cudaEventDisableTiming) return false;
  int same_stream_wait=0;
  for(const auto& row:events)
    if(row.second.stream==stream && row.second.flags==(cudaEventDisableTiming|cudaEventBlockingSync))
      ++same_stream_wait;
  return same_stream_wait==1;
}
int live() {
  std::lock_guard<std::mutex> lock(mutex);int count=0;
  for(const auto& e:events) count+=(e.second.flags & cudaEventBlockingSync)!=0;
  return count;
}
int all_live() {std::lock_guard<std::mutex> lock(mutex);return events.size();}
void begin(int fail=0) {
  creates=0;records=0;waits=0;downloads=0;failure=fail;active=true;
}
void end() {active=false;}
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event,unsigned flags) {
  static auto next=audit::real<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags");
  if(audit::active && (flags & cudaEventBlockingSync)) {
    int expected=3;
    if(audit::failure.compare_exchange_strong(expected,0)) return cudaErrorMemoryAllocation;
  }
  const auto code=next(event,flags);
  if(code==cudaSuccess) {
    int device=-1;cudaGetDevice(&device);
    std::lock_guard<std::mutex> lock(audit::mutex);
    audit::events[*event]={flags,device,nullptr};
    if(audit::active && (flags & cudaEventBlockingSync)) ++audit::creates;
  }
  return code;
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  static auto next=audit::real<decltype(&cudaEventDestroy)>("cudaEventDestroy");
  const auto code=next(event);
  if(code==cudaSuccess) {std::lock_guard<std::mutex> lock(audit::mutex);audit::events.erase(event);}
  return code;
}
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,cudaStream_t stream) {
  static auto next=audit::real<decltype(&cudaEventRecord)>("cudaEventRecord");
  if(audit::blocking(event)) {
    int device=-1;cudaGetDevice(&device);
    {std::lock_guard<std::mutex> lock(audit::mutex);
      auto& row=audit::events.at(event);row.stream=stream;
      if(row.device!=device) ++audit::wrong_device;}
    if(audit::active) {
      ++audit::records;int expected=1;
      if(audit::failure.compare_exchange_strong(expected,0)) return cudaErrorInvalidResourceHandle;
    }
  }
  return next(event,stream);
}
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
  static auto next=audit::real<decltype(&cudaEventSynchronize)>("cudaEventSynchronize");
  const auto code=next(event);
  if(audit::active && audit::blocking(event)) {
    ++audit::waits;int expected=2;
    if(audit::failure.compare_exchange_strong(expected,0)) return cudaErrorInvalidResourceHandle;
  }
  return code;
}
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* dst,const void* src,size_t bytes,
                                               cudaMemcpyKind kind,cudaStream_t stream) {
  static auto next=audit::real<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
  if(audit::active && kind==cudaMemcpyDeviceToHost) ++audit::downloads;
  return next(dst,src,bytes,kind,stream);
}

constexpr const char* flag="FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H";
bool baseline_only=false;
bool resource_memcheck=false;
using Prepared=std::shared_ptr<MultiPenaltyGcvCapacityPrepared>;
Prepared prepare(const Request& r) {
  const auto& g=r.geometry;
  return create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),
    g.R.data(),g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),
    g.n,g.q,g.penalties,g.capacity,0);
}
MultiPenaltyGcvCapacityBatchResult solve(const Prepared& p,const Request& r) {
  return multi_penalty_gcv_capacity_optimize_batch_with_residual(p,r.Y.data(),r.geometry.n,
    r.keys.size(),r.keys,r.control);
}
Request synthetic(int q) {
  Request r;auto& g=r.geometry;g.q=q;g.n=q+8;g.penalties=1;g.capacity=2;
  g.X.assign(g.n*q,0);g.qr=g.X;g.tau.assign(q,0);g.R.assign(q*q,0);g.pivot.resize(q);
  for(int j=0;j<q;++j) {g.X[j+g.n*j]=g.qr[j+g.n*j]=g.R[j+q*j]=1;g.pivot[j]=j;}
  g.roots={std::vector<double>(q,0)};g.matrices={std::vector<double>(q*q,0)};
  g.roots[0][0]=g.matrices[0][0]=1;g.ranks={1};g.initial={0};
  // Nonzero residual energy and a finite interior smoothing optimum. An
  // all-zero response is a degenerate GCV problem, not a valid capacity test.
  r.Y.assign(g.n*2,0);
  for(int k=0;k<2;++k) {
    r.Y[g.n*k]=3+k;
    for(int j=1;j<q;++j) r.Y[j+g.n*k]=(j%7-3)*0.125;
    for(int j=q;j<g.n;++j) r.Y[j+g.n*k]=1;
  }
  r.keys={"synthetic-"+std::to_string(q)+"-0","synthetic-"+std::to_string(q)+"-1"};
  r.control.exact_primary_only=true;r.control.force_final_selected_evaluation=true;
  return r;
}
void release(MultiPenaltyGcvCapacityBatchResult& value) {
  release_multi_penalty_gcv_capacity_residual(value.residual);value.residual.reset();
}
void configuration_checks(const Request& r,int original) {
  const int all_events=audit::all_live();
  setenv(flag,"1",1);audit::begin(3);bool allocation_rejected=false;
  try{auto p=prepare(r);}catch(const std::exception&){allocation_rejected=true;}
  audit::end();
  need(allocation_rejected && audit::failure==0 && audit::all_live()==all_events,
       "event allocation failure leaked resources or missed injection");
  setenv(flag,"invalid",1);bool rejected=false;
  try{auto p=prepare(r);}catch(const std::exception& e){rejected=std::string(e.what()).find(flag)!=std::string::npos;}
  need(rejected && audit::live()==original && audit::all_live()==all_events,"invalid flag did not unwind resources");
  for(const char* disabled:{"","0"}) {
    setenv(flag,disabled,1);auto p=prepare(r);
    need(audit::live()==original,"explicit disabled value allocated a wait event");
  }
  setenv(flag,"0",1);
}
void resource_case(const Request& r) {
  const int original=audit::live();
  unsetenv(flag);auto off=prepare(r);
  need(audit::live()==original,"default allocated wait event");off.reset();
  setenv(flag,"1",1);auto on=prepare(r);
  need(audit::live()==original+1,"prepared wait event missing");on.reset();
  need(audit::live()==original,"prepared wait event leak");
  configuration_checks(r,original);
  std::cout<<"PASS resource lifecycle q="<<r.geometry.q<<std::endl;
}
void run_case(const Request& r,bool captured,bool extended_checks) {
  const int original=audit::live();
  unsetenv(flag);auto off=prepare(r);need(audit::live()==original,"default allocated wait event");
  setenv(flag,"1",1); // A later environment change must not enable an existing object.
  audit::begin();auto baseline=solve(off,r);audit::end();
  need(audit::creates==0 && audit::records==0 && audit::waits==0,"default path has extra wait");
  const auto expected=snapshot(baseline.optimization,baseline.residual);
  if(captured) need(expected.same(r.expected),"baseline capture mismatch");
  release(baseline);off.reset();
  if(baseline_only) {std::cout<<"PASS baseline q="<<r.geometry.q<<std::endl;return;}

  setenv(flag,"1",1);auto on=prepare(r);
  need(audit::live()==original+1,"prepared must own one blocking event");
  setenv(flag,"0",1); // Prepared policy is immutable even if the environment changes.
  for(int rep=0;rep<2;++rep) {
    audit::begin();auto value=solve(on,r);audit::end();
    need(audit::creates==0 && audit::records==1 && audit::waits==1 && audit::downloads==1,
         "wait/creation/download topology mismatch");
    need(snapshot(value.optimization,value.residual).same(expected),"enabled numerical drift");
    auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(value.residual);
    need(audit::producer_topology(view.producer_completion_event,view.producer_stream),
         "downstream producer event or prepared wait stream changed");
    bool busy=false;try{solve(on,r);}catch(const std::exception& e){busy=std::string(e.what()).find("OUTPUT_SLOT_BUSY")!=std::string::npos;}
    need(busy,"leased output did not reject reuse");
    release(value);
  }
  if(extended_checks) {
    // The opt-in affects optimizer status only, not the independent fixed
    // evaluator, which still follows its original completion/download path.
    setenv(flag,"0",1);auto fixed_off=prepare(r);
    auto fixed=[&](const Prepared& p) {
      return multi_penalty_gcv_capacity_evaluate_fixed_batch(p,r.Y.data(),
        expected.floating.at(1).data(),r.geometry.n,r.keys.size(),r.keys,r.control.rank_tolerance);
    };
    auto fixed_reference=fixed(fixed_off);
    const auto fixed_shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(fixed_reference);
    release_multi_penalty_gcv_capacity_residual(fixed_reference);fixed_reference.reset();fixed_off.reset();
    audit::begin();auto fixed_on=fixed(on);audit::end();
    need(audit::records==0 && audit::waits==0 && audit::creates==0 && audit::downloads==1,
         "fixed evaluator acquired optimizer status wait");
    const auto enabled_shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(fixed_on);
    Snapshot fixed_a,fixed_b;
    fixed_a.floating={fixed_shadow.coefficients,fixed_shadow.fitted,fixed_shadow.residuals};
    fixed_b.floating={enabled_shadow.coefficients,enabled_shadow.fitted,enabled_shadow.residuals};
    need(fixed_a.same(fixed_b),"fixed evaluator numerical drift");
    release_multi_penalty_gcv_capacity_residual(fixed_on);fixed_on.reset();
    auto traced=r;traced.control.decomposition_trace_capacity_per_target=512;
    audit::begin();auto value=solve(on,traced);audit::end();
    need(audit::creates==0 && audit::records==1 && audit::waits==1 && audit::downloads==3,
         "trace downloads added redundant waits");
    need(snapshot(value.optimization,value.residual).same(expected),"trace changed numerical output");
    release(value);
    for(int fail:{1,2}) {
      bool rejected=false;audit::begin(fail);
      try{auto result=solve(on,r);release(result);}catch(const std::exception&){rejected=true;}
      audit::end();need(rejected && audit::failure==0 && audit::downloads==0,
                        "failed wait published/downloaded status or missed injection");
      need(cudaDeviceSynchronize()==cudaSuccess,"failure cleanup synchronization");
      need(!multi_penalty_gcv_capacity_prepared_info(on).residual_slot_leased,"failure leaked output lease");
      auto recovered=solve(on,r);need(snapshot(recovered.optimization,recovered.residual).same(expected),"failed wait poisoned prepared");release(recovered);
    }
  }
  on.reset();need(audit::live()==original,"prepared wait event leak");
  configuration_checks(r,original);
  std::cout<<"PASS q="<<r.geometry.q<<" captured="<<captured<<" extended="<<extended_checks<<std::endl;
}
void cancellation(const Request& r) {
  const int original=audit::live();setenv(flag,"1",1);
  std::vector<Prepared> handles;std::vector<MultiPenaltyGcvCapacityRequest> requests;
  for(int i=0;i<3;++i) {
    auto p=prepare(r);handles.push_back(p);MultiPenaltyGcvCapacityRequest req;
    req.prepared=p;req.Y=r.Y;req.n=r.geometry.n;req.target_count=r.keys.size();req.target_keys=r.keys;req.control=r.control;
    requests.push_back(std::move(req));
  }
  bool rejected=false;audit::begin(1);
  try{multi_penalty_gcv_capacity_optimize_multi(std::move(requests),3,true);}catch(const std::exception&){rejected=true;}
  audit::end();need(rejected && audit::failure==0,"multi-worker failure did not propagate");
  need(cudaDeviceSynchronize()==cudaSuccess,"cancel drain failed");
  for(const auto& p:handles) need(!multi_penalty_gcv_capacity_prepared_info(p).residual_slot_leased,"cancelled output lease leaked");
  handles.clear();need(audit::live()==original,"cancelled wait event leak");
  std::cout<<"PASS multi-worker cancellation and resource closure\n";
}
int main(int argc,char** argv) {
  try {
    need(argc==3 || (argc==4 && (std::string(argv[3])=="--baseline-only" ||
         std::string(argv[3])=="--resource-memcheck")),
         "qualify L3_CAPTURE L5_CAPTURE [--baseline-only|--resource-memcheck]");
    baseline_only=argc==4 && std::string(argv[3])=="--baseline-only";
    resource_memcheck=argc==4 && std::string(argv[3])=="--resource-memcheck";
    need(cudaSetDevice(0)==cudaSuccess,"select GPU");
    const auto l3=Fixture::read(argv[1]),l5=Fixture::read(argv[2]);
    run_case(l3.requests.front(),true,true);run_case(l5.requests.front(),true,false);
    for(int q:{73,82,193,385}) {
      if(resource_memcheck) resource_case(synthetic(q));
      else run_case(synthetic(q),false,false);
    }
    if(!baseline_only) cancellation(l3.requests.front());
    need(audit::live()==0 && audit::wrong_device==0,"event ownership closure");
    std::cout<<(baseline_only ? "PASS baseline geometry and GPU output only\n" : resource_memcheck ?
      "PASS resource memcheck: real complete paths and five-bucket allocation/lifecycle\n" :
      "PASS all five capacity buckets, bitwise state/residual, event reuse, trace, failures, cancellation\n");
    return 0;
  }catch(const std::exception& e){audit::end();std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
