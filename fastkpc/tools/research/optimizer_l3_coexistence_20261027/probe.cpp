#include "probe.hpp"
#include "../optimizer_l3_slots_20261027/fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include <Rcpp.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
namespace l3_probe {
using namespace fastkpc;
using namespace window_diag;
namespace {
bool begun=false,done=false,slot_mode=false;
std::atomic<bool> sampling{false};
std::atomic<std::size_t> minimum{SIZE_MAX},sample_count{0};
std::thread monitor;
std::mutex mutex;
std::vector<l3_diag::Window> fixtures;
std::map<std::string,const Request*> requests;
std::set<std::string> optimized;
std::size_t optimizer_targets=0,stage_targets=0,stage_batches=0,geometry_count=0,shadow_bytes=0;
std::size_t pool_before_free=0,pool_increment=0;
layer_audit::Counts pool_before,pool_created,pool_closing;
bool pool_closed=false;
std::ofstream checkpoints,reference_out;
std::ifstream reference_in;
const char* env(const char* k){const char* p=std::getenv(k);need(p&&*p,k);return p;}
void check(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
std::size_t observe(){std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  auto old=minimum.load();while(old>free&&!minimum.compare_exchange_weak(old,free)){};++sample_count;return free;}
template<class T>bool bits(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(T));}
const Request& request(const std::vector<std::string>& keys){need(!keys.empty(),"empty L3 keys");auto i=requests.find(keys.front());
  need(i!=requests.end()&&i->second->keys==keys,"L3 original cohort identity");return *i->second;}
bool delta_closed(const layer_audit::Counts& before,const layer_audit::Counts& after){
  return pool_created.bytes-pool_before.bytes==before.bytes-after.bytes&&
    pool_created.allocations-pool_before.allocations==before.allocations-after.allocations&&
    pool_created.streams-pool_before.streams==before.streams-after.streams&&
    pool_created.events-pool_before.events==before.events-after.events&&
    pool_created.blas-pool_before.blas==before.blas-after.blas;
}
}
void checkpoint(const char* name){
  if(!begun||done)return;const auto free=observe();const auto c=layer_audit::counts();
  if(std::string(name)=="pool_before"){pool_before=c;pool_before_free=free;}
  if(std::string(name)=="pool_created")pool_created=c;
  if(std::string(name)=="pool_created"||std::string(name)=="optimizer_done"){
    pool_increment=std::max(pool_increment,pool_before_free-std::min(pool_before_free,free));
    need(pool_increment<=1536ULL*1024*1024,"L3 pool exceeded predeclared 1.5 GiB");}
  if(std::string(name)=="pool_closing")pool_closing=c;
  if(std::string(name)=="pool_closed"){
    pool_closed=delta_closed(pool_closing,c);need(pool_closed,"L3 pool resources not restored");}
  checkpoints<<name<<','<<free<<','<<c.bytes<<','<<c.allocations<<','<<c.streams<<','<<c.events<<','<<c.blas<<','<<stage_targets<<'\n';checkpoints.flush();
  need(free>=2ULL*1024*1024*1024,"L3 joint headroom below predeclared 2 GiB");
}
void begin(bool slots){
  need(!begun,"duplicate L3 probe");slot_mode=slots;
  fixtures=l3_diag::read_layer(env("FASTKPC_L3_PROBE_FIXTURES"));
  for(const auto& w:fixtures)for(const auto& r:w.requests)need(requests.emplace(r.keys.front(),&r).second,"duplicate L3 cohort");
  checkpoints.open(std::string(env("FASTKPC_L3_PROBE_OUT"))+"/checkpoints.csv");
  checkpoints<<"checkpoint,free_bytes,application_bytes,allocations,streams,events,blas,stage_a_targets\n";
  if(slots)reference_in.open(env("FASTKPC_L3_STAGE_A_REFERENCE"),std::ios::binary);
  else reference_out.open(env("FASTKPC_L3_STAGE_A_REFERENCE"),std::ios::binary);
  need(slots?bool(reference_in):bool(reference_out),"Stage A reference stream");begun=true;
  checkpoint("level_begin");sampling=true;
  monitor=std::thread([]{if(cudaSetDevice(0)!=cudaSuccess){minimum=0;return;}
    while(sampling.load()){try{observe();}catch(...){minimum=0;return;}
      std::this_thread::sleep_for(std::chrono::milliseconds(10));}});
}
void workers(bool active){layer_audit::native_scope(active);}
void geometry(const Geometry& g,const MultiPenaltyGcvCapacityRequest& input){
  validate(g);const auto& r=request(input.target_keys);const auto& a=r.geometry;
  std::vector<double> roots,matrices;for(const auto& v:a.roots)roots.insert(roots.end(),v.begin(),v.end());
  for(const auto& v:a.matrices)matrices.insert(matrices.end(),v.begin(),v.end());
  need(!input.prepared&&input.n==351&&input.target_count==int(r.keys.size())&&bits(input.Y,r.Y)&&
    bits(g.X,a.X)&&bits(g.qr,a.qr)&&bits(g.tau,a.tau)&&bits(g.r,a.R)&&g.pivot==a.pivot&&g.ranks==a.ranks&&
    bits(g.initial_log_sp,a.initial)&&bits(g.roots,roots)&&bits(g.matrices,matrices),"L3 replacement input mismatch or retained prepared");
  ++geometry_count;
}
void optimizer(const MultiPenaltyGcvCudaOptimization& result,const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual){
  const auto info=multi_penalty_gcv_capacity_residual_info(residual);const auto& r=request(info.target_keys);
  const auto saved=snapshot(result,residual);need(saved.same(r.expected),"L3 optimizer captured output mismatch");
  std::lock_guard<std::mutex> lock(mutex);need(optimized.insert(r.keys.front()).second,"L3 duplicate optimizer result");
  optimizer_targets+=r.keys.size();shadow_bytes+=(28+702)*r.keys.size()*8;
}
void stage_a(const std::vector<std::string>& keys,int offset,int count,
    const MultiPenaltyGcvCudaCoefficientConsumerView& c,const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& residual){
  const auto& r=request(keys);need(offset>=0&&count>0&&offset+count<=int(keys.size()),"Stage A cohort range");
  checkpoint("stage_a_cohort");check(cudaEventSynchronize(c.producer_completion_event));
  std::vector<double> coefficients(28*count);check(cudaMemcpy(coefficients.data(),c.coefficients,coefficients.size()*8,cudaMemcpyDeviceToHost));
  std::vector<double> expected(r.expected.floating[11].begin()+28*offset,r.expected.floating[11].begin()+28*(offset+count));
  need(bits(coefficients,expected),"coefficient arena detach/gather changed bits");
  const auto info=multi_penalty_gcv_capacity_residual_info(residual);
  const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(residual);
  Snapshot saved;saved.floating={shadow.coefficients,shadow.fitted,shadow.residuals};
  saved.integral={info.solver_info,info.decomposition_route,info.numerical_rank};
  need(info.target_keys==c.target_keys&&info.target_count==count&&shadow.n==351&&shadow.coefficient_dim==28&&
    info.solver_info==c.solver_info&&info.decomposition_route==c.decomposition_route&&info.numerical_rank==c.numerical_rank,
    "Stage A metadata mismatch");
  auto stored_keys=c.target_keys;IO io{slot_mode?&reference_in:nullptr,slot_mode?nullptr:&reference_out};
  if(slot_mode){std::vector<std::string> original;io.value(original);Snapshot original_saved;original_saved.io(io);
    need(original==stored_keys&&original_saved.same(saved),"original Stage A order/output mismatch");}
  else {io.value(stored_keys);saved.io(io);}
  stage_targets+=count;++stage_batches;shadow_bytes+=(28+28+702)*count*8;
}
void finish(){
  checkpoint("level_end");sampling=false;if(monitor.joinable())monitor.join();
  const auto c=layer_audit::counts();const bool pass=optimized.size()==4065&&optimizer_targets==54878&&stage_targets==54878&&
    minimum.load()>=2ULL*1024*1024*1024&&(!slot_mode||(geometry_count==4065&&pool_closed&&c.forbidden_calls==0));
  if(slot_mode)need(reference_in.peek()==EOF,"unconsumed original Stage A cohorts");else reference_out.close();
  std::ofstream out(std::string(env("FASTKPC_L3_PROBE_OUT"))+"/probe.json");
  out<<"{\"status\":\""<<(pass?"PASS":"FAIL")<<"\",\"slots\":"<<(slot_mode?"true":"false")
     <<",\"optimizer_setups\":"<<optimized.size()<<",\"optimizer_targets\":"<<optimizer_targets
     <<",\"geometry_checked\":"<<geometry_count<<",\"stage_a_targets\":"<<stage_targets<<",\"stage_a_batches\":"<<stage_batches
     <<",\"minimum_free_bytes\":"<<minimum.load()<<",\"pool_increment_bytes\":"<<pool_increment
     <<",\"observations\":"<<sample_count.load()<<",\"diagnostic_shadow_d2h_bytes\":"<<shadow_bytes
     <<",\"pool_application_bytes\":"<<(slot_mode?pool_created.bytes-pool_before.bytes:0)
     <<",\"pool_resources_closed\":"<<(pool_closed?"true":"false")<<",\"worker_resource_mutations\":"<<c.forbidden_calls<<"}\n";
  need(bool(out),"probe receipt write");done=true;need(pass,"L3 coexistence qualification failed");
  std::cerr<<"L3 coexistence "<<(slot_mode?"S":"W")<<" PASS, minimum_free="<<minimum.load()<<std::endl;
}
extern "C" SEXP fastkpc_l3_probe_finish(){BEGIN_RCPP
  need(done,"L3 probe not completed");const auto c=layer_audit::counts();
  need(!c.allocations&&!c.bytes&&!c.streams&&!c.events&&!c.blas,"diagnostic application resources still live");
  return Rcpp::wrap(true);
END_RCPP}
// Join even if a native error unwinds the R call; retain the failing attempt.
struct Cleanup {~Cleanup(){sampling=false;if(monitor.joinable())monitor.join();}} cleanup;
}
