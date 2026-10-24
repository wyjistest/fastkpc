#include "../optimizer_layer_refill_20261023/layer_fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

using namespace fastkpc;
using namespace window_diag;
void checked(cudaError_t e) { need(e == cudaSuccess, cudaGetErrorString(e)); }
MultiPenaltyGcvL5Geometry packet(const Request& r) {
  const auto& a = r.geometry;
  MultiPenaltyGcvL5Geometry g;
  g.X=a.X;g.qr=a.qr;g.tau=a.tau;g.r=a.R;g.pivot=a.pivot;
  g.ranks=a.ranks;g.initial_log_sp=a.initial;
  int rank=0;
  for(int j=0;j<5;++j){g.offsets.push_back(rank);rank+=a.ranks[j];
    g.roots.insert(g.roots.end(),a.roots[j].begin(),a.roots[j].end());
    g.matrices.insert(g.matrices.end(),a.matrices[j].begin(),a.matrices[j].end());}
  validate_multi_penalty_gcv_l5_geometry(g);return g;
}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"expected rejection");}
void closed(){auto s=layer_audit::counts();need(!s.allocations&&!s.bytes&&!s.streams&&!s.events&&!s.blas,"resource leak");}
struct Store {
  double* p=nullptr;cudaStream_t stream=nullptr;cudaEvent_t event=nullptr;
  Store(){checked(cudaMalloc(reinterpret_cast<void**>(&p),46*8*sizeof(double)));
    checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    checked(cudaEventCreateWithFlags(&event,cudaEventDisableTiming|cudaEventBlockingSync));}
  ~Store(){cudaStreamSynchronize(stream);cudaEventDestroy(event);cudaStreamDestroy(stream);cudaFree(p);}
};
int main(int argc,char**argv)try{
  need(argc==2,"fixture directory required");setenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H","1",1);
  checked(cudaSetDevice(0));auto windows=layer_diag::read_layer(argv[1]);
  std::vector<const Request*> chosen;
  for(int targets:{8,2,4,3}){
    const Request* found=nullptr;
    for(auto& w:windows)for(auto& r:w.requests)if(!found&&int(r.keys.size())==targets)found=&r;
    need(found,"missing cohort width");chosen.push_back(found);
  }
  std::vector<MultiPenaltyGcvL5Geometry> packets;for(auto r:chosen)packets.push_back(packet(*r));
  auto largest=std::max_element(packets.begin(),packets.end(),[](const auto&a,const auto&b){return a.roots.size()<b.roots.size();});
  int compared=0,negative=0;
  layer_audit::memory_limit(0);
  rejects([&]{create_multi_penalty_gcv_capacity_l5_slot(*largest,0);});++negative;
  layer_audit::memory_limit(std::numeric_limits<std::size_t>::max());closed();
  closed();{
    auto slot=create_multi_penalty_gcv_capacity_l5_slot(*largest,0);Store arena;
    std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> old;
    for(int repeat=0;repeat<2;++repeat)for(int i=0;i<4;++i){
      const auto&r=*chosen[i];rebind_multi_penalty_gcv_capacity_l5_slot(slot,packets[i]);
      if(old){rejects([&]{acquire_multi_penalty_gcv_capacity_residual_consumer_view(old);});++negative;}
      auto value=multi_penalty_gcv_capacity_optimize_batch_with_residual(slot,r.Y.data(),351,r.keys.size(),r.keys,r.control);
      old.reset();need(multi_penalty_gcv_capacity_residual_info(value.residual).device_resident,"old destructor released new output");
      need(snapshot(value.optimization,value.residual).same(r.expected),"captured optimizer bits differ");
      compared+=r.keys.size();
      auto before=multi_penalty_gcv_capacity_prepared_info(slot);
      rejects([&]{rebind_multi_penalty_gcv_capacity_l5_slot(slot,packets[i]);});++negative;
      need(before.generation==multi_penalty_gcv_capacity_prepared_info(slot).generation,"busy bind mutated generation");
      const auto info=multi_penalty_gcv_capacity_residual_info(value.residual);
      const auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(value.residual);
      layer_audit::native_scope(true);
      checked(cudaStreamWaitEvent(arena.stream,view.producer_completion_event,0));
      checked(cudaMemcpyAsync(arena.p,view.coefficients,46*r.keys.size()*sizeof(double),cudaMemcpyDeviceToDevice,arena.stream));
      checked(cudaEventRecord(arena.event,arena.stream));checked(cudaEventSynchronize(arena.event));
      release_multi_penalty_gcv_capacity_residual(value.residual);old=value.residual;value.residual.reset();
      rebind_multi_penalty_gcv_capacity_l5_slot(slot,packets[(i+1)%4]);
      rebind_multi_penalty_gcv_capacity_l5_slot(slot,packets[i]);
      MultiPenaltyGcvCudaCoefficientConsumerView c;c.coefficients=arena.p;c.coefficient_dim=46;
      c.target_count=r.keys.size();c.device_id=0;c.producer_stream=arena.stream;c.producer_completion_event=arena.event;
      c.target_keys=r.keys;c.solver_info=info.solver_info;c.decomposition_route=info.decomposition_route;c.numerical_rank=info.numerical_rank;
      auto finalized=multi_penalty_gcv_capacity_finalize_selected_coefficients(slot,r.Y.data(),c,351,r.keys.size(),r.keys);
      layer_audit::native_scope(false);
      const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(finalized);
      Snapshot actual,expected;actual.floating={shadow.coefficients,shadow.fitted,shadow.residuals};
      expected.floating={r.expected.floating.at(11),r.expected.floating.at(12),r.expected.floating.at(13)};
      need(actual.same(expected),"detached coefficient Stage A bits differ");compared+=r.keys.size();
      release_multi_penalty_gcv_capacity_residual(finalized);finalized.reset();
      before=multi_penalty_gcv_capacity_prepared_info(slot);auto bad=packets[i];bad.q=47;
      rejects([&]{rebind_multi_penalty_gcv_capacity_l5_slot(slot,bad);});++negative;
      need(before.generation==multi_penalty_gcv_capacity_prepared_info(slot).generation,"invalid bind mutated generation");
    }
    drain_multi_penalty_gcv_capacity_l5_slot(slot);old.reset();slot.reset();
  }closed();
  // Two real native workers, with a failure injected after one optimizer
  // completed. Every surviving output and stream is explicitly drained.
  {
    std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> slots;
    for(int i=0;i<2;++i)slots.push_back(create_multi_penalty_gcv_capacity_l5_slot(*largest,0));
    std::atomic<bool> cancelled{false};std::atomic<int> failures{0};std::vector<std::thread> threads;
    layer_audit::native_scope(true);
    for(int worker=0;worker<2;++worker)threads.emplace_back([&,worker]{
      std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> pending;
      try{
        checked(cudaSetDevice(0));const auto&r=*chosen[worker];
        rebind_multi_penalty_gcv_capacity_l5_slot(slots[worker],packets[worker]);
        auto result=multi_penalty_gcv_capacity_optimize_batch_with_residual(slots[worker],r.Y.data(),351,r.keys.size(),r.keys,r.control);
        pending=result.residual;
        if(worker==0)throw std::runtime_error("injected completed-output cancellation");
        release_multi_penalty_gcv_capacity_residual(pending);pending.reset();
      }catch(...){cancelled=true;++failures;drain_multi_penalty_gcv_capacity_l5_slot(slots[worker]);
        if(pending)release_multi_penalty_gcv_capacity_residual(pending);}
    });
    for(auto&t:threads)t.join();layer_audit::native_scope(false);
    need(cancelled&&failures==1,"cancellation injection failed");
    for(auto&s:slots){drain_multi_penalty_gcv_capacity_l5_slot(s);need(!multi_penalty_gcv_capacity_prepared_info(s).residual_slot_leased,"cancel lease");}
  }closed();need(layer_audit::counts().forbidden_calls==0,"resource mutation inside native interval");
  std::cout<<"PASS captured optimizer and detached Stage A; compared="<<compared<<" negative="<<negative
           <<" cancellation=PASS resources=PASS forbidden=0\n";
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
