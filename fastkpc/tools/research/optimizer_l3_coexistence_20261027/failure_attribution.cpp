// Logging-only repeat of a rejected capacity gate. No threshold or math changes.
#include "probe.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include <Rcpp.h>
#include <dlfcn.h>
#include <atomic>
#include <fstream>
#include <mutex>
namespace {
std::atomic<std::size_t> completed{0},targets{0};
std::size_t before=0,created=0,after_work=0;
layer_audit::Counts before_counts,created_counts,work_counts;
bool joined=false;
const char* root(){const char* p=std::getenv("FASTKPC_L3_PROBE_OUT");if(!p)throw std::runtime_error("missing attribution output");return p;}
template<class F>F next(const char* name){auto f=reinterpret_cast<F>(dlsym(RTLD_NEXT,name));if(!f)throw std::runtime_error(name);return f;}
std::size_t memory(){std::size_t free=0,total=0;if(cudaMemGetInfo(&free,&total)!=cudaSuccess)throw std::runtime_error("memory attribution query");return free;}
void record(const char* name,std::size_t free,const layer_audit::Counts& c){
  std::ofstream out(std::string(root())+"/failure-points.csv",std::ios::app);
  out<<name<<','<<free<<','<<c.bytes<<','<<c.allocations<<','<<c.streams<<','<<c.events<<','<<c.blas<<','<<c.forbidden_calls<<','<<completed.load()<<','<<targets.load()<<'\n';
}
}
namespace l3_probe {
void checkpoint(const char* name){
  static auto f=next<decltype(&checkpoint)>("_ZN8l3_probe10checkpointEPKc");f(name);
  const std::string label(name);if(label=="pool_before"||label=="pool_created"){
    const auto free=memory();const auto counts=layer_audit::counts();
    if(label=="pool_before"){before=free;before_counts=counts;}else{created=free;created_counts=counts;}
    record(name,free,counts);
  }
}
void optimizer(const fastkpc::MultiPenaltyGcvCudaOptimization& result,
    const std::shared_ptr<fastkpc::MultiPenaltyGcvCapacityResidualBatch>& residual){
  static auto f=next<decltype(&optimizer)>(ATTR_optimizer);f(result,residual);
  ++completed;targets+=result.target_count;
}
void workers(bool active){
  static auto f=next<decltype(&workers)>("_ZN8l3_probe7workersEb");f(active);
  if(!active){joined=true;after_work=memory();work_counts=layer_audit::counts();record("optimizer_workers_joined",after_work,work_counts);}
}
}
extern "C" SEXP fastkpc_l3_failure_report(){BEGIN_RCPP
  const auto c=layer_audit::counts();const auto free=memory();record("after_native_exception",free,c);
  const bool resources=!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas;
  std::ofstream out(std::string(root())+"/failure.json");
  out<<"{\"schema\":\"l3-failed-capacity-attribution-v1\",\"optimizer_workers_joined\":"<<(joined?"true":"false")
     <<",\"optimizer_setups_bitwise_exact\":"<<completed.load()<<",\"optimizer_targets_bitwise_exact\":"<<targets.load()
     <<",\"free_before_pool_bytes\":"<<before<<",\"free_after_create_bytes\":"<<created
     <<",\"free_after_optimizer_bytes\":"<<after_work<<",\"free_after_exception_bytes\":"<<free
     <<",\"create_increment_bytes\":"<<(before-created)<<",\"work_increment_bytes\":"<<(before-after_work)
     <<",\"additional_physical_bytes_after_first_execution\":"<<(created-after_work)
     <<",\"pool_application_bytes\":"<<(created_counts.bytes-before_counts.bytes)
     <<",\"additional_application_bytes_during_optimizer\":"<<(work_counts.bytes-created_counts.bytes)
     <<",\"worker_resource_mutations\":"<<work_counts.forbidden_calls
     <<",\"resources_closed_after_exception\":"<<(resources?"true":"false")
     <<",\"remaining_allocations\":"<<c.allocations<<",\"remaining_bytes\":"<<c.bytes
     <<",\"remaining_streams\":"<<c.streams<<",\"remaining_events\":"<<c.events<<",\"remaining_blas\":"<<c.blas<<"}\n";
  return Rcpp::wrap(resources);
END_RCPP}
