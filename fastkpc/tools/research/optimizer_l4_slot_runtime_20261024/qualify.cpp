#include "../optimizer_l4_diagnostic_20261023/fixture.hpp"
#include "../optimizer_layer_refill_20261023/audit.hpp"
#include "fault.hpp"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
using namespace fastkpc;
using namespace window_diag;
void check(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
void closed(){auto s=layer_audit::counts();need(!s.allocations&&!s.bytes&&!s.streams&&!s.events&&!s.blas,"resource leak");}
template<class F>void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}need(failed,"expected rejection");}
MultiPenaltyGcvL4Geometry packet(const Request& r){
  const auto& a=r.geometry;MultiPenaltyGcvL4Geometry g;
  g.X=a.X;g.qr=a.qr;g.tau=a.tau;g.r=a.R;g.pivot=a.pivot;g.ranks=a.ranks;g.initial_log_sp=a.initial;
  int rank=0;for(int j=0;j<4;++j){g.offsets.push_back(rank);rank+=a.ranks[j];
    g.roots.insert(g.roots.end(),a.roots[j].begin(),a.roots[j].end());
    g.matrices.insert(g.matrices.end(),a.matrices[j].begin(),a.matrices[j].end());}
  validate_multi_penalty_gcv_l4_geometry(g);return g;
}
struct Store {
  double* p=nullptr;cudaStream_t stream=nullptr;cudaEvent_t event=nullptr;
  Store(){check(cudaMalloc(reinterpret_cast<void**>(&p),37*18*sizeof(double)));
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming|cudaEventBlockingSync));}
  ~Store(){cudaStreamSynchronize(stream);cudaEventDestroy(event);cudaStreamDestroy(stream);cudaFree(p);}
};
int main(int argc,char** argv)try{
  need(argc==3,"captured L4 directory and original Stage A reference required");setenv("FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H","1",1);
  check(cudaSetDevice(0));auto windows=l4_diag::read_layer(argv[1]);
  std::map<int,const Request*> first;
  for(const auto& w:windows)for(const auto& r:w.requests)first.emplace(r.keys.size(),&r);
  need(first.size()==14,"all fourteen L4 capacities required");
  std::vector<const Request*> chosen;for(const auto& entry:first)chosen.push_back(entry.second);
  chosen.push_back(&windows.back().requests.back());
  std::ifstream reference(argv[2],std::ios::binary);IO reader{&reference,nullptr};
  std::string schema;reader.value(schema);need(schema=="l4-original-stage-a-v1","reference schema");
  int reference_count=0;reader.value(reference_count);need(reference_count==int(chosen.size()),"reference coverage");
  std::vector<Snapshot> stage_a(reference_count);
  for(int i=0;i<reference_count;++i){std::vector<std::string> keys;reader.value(keys);
    need(keys==chosen[i]->keys,"reference target identity");stage_a[i].io(reader);}
  need(reference.peek()==EOF,"reference trailing bytes");
  std::vector<MultiPenaltyGcvL4Geometry> packets;for(auto r:chosen)packets.push_back(packet(*r));
  int compared=0,negative=0;
  layer_audit::memory_limit(0);rejects([&]{create_multi_penalty_gcv_capacity_l4_slot(packets.front(),0);});++negative;
  layer_audit::memory_limit(std::numeric_limits<std::size_t>::max());closed();
  {
    auto slot=create_multi_penalty_gcv_capacity_l4_slot(packets.front(),0);Store arena;
    std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> old;
    for(int repeat=0;repeat<2;++repeat)for(std::size_t i=0;i<chosen.size();++i){
      const auto& r=*chosen[i];rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets[i]);
      if(old){rejects([&]{acquire_multi_penalty_gcv_capacity_residual_consumer_view(old);});++negative;}
      auto value=multi_penalty_gcv_capacity_optimize_batch_with_residual(slot,r.Y.data(),351,r.keys.size(),r.keys,r.control);
      old.reset();need(multi_penalty_gcv_capacity_residual_info(value.residual).device_resident,"old destructor released new output");
      need(snapshot(value.optimization,value.residual).same(r.expected),"captured optimizer bits differ");compared+=r.keys.size();
      auto before=multi_penalty_gcv_capacity_prepared_info(slot);
      rejects([&]{rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets[i]);});++negative;
      need(before.generation==multi_penalty_gcv_capacity_prepared_info(slot).generation,"busy bind changed generation");
      const auto info=multi_penalty_gcv_capacity_residual_info(value.residual);
      const auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(value.residual);
      layer_audit::native_scope(true);
      check(cudaStreamWaitEvent(arena.stream,view.producer_completion_event,0));
      check(cudaMemcpyAsync(arena.p,view.coefficients,37*r.keys.size()*8,cudaMemcpyDeviceToDevice,arena.stream));
      check(cudaEventRecord(arena.event,arena.stream));check(cudaEventSynchronize(arena.event));
      release_multi_penalty_gcv_capacity_residual(value.residual);old=value.residual;value.residual.reset();
      rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets[(i+1)%packets.size()]);
      rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets[i]);
      const int total=r.keys.size(),cohorts=(total+15)/16;int offset=0;
      for(int cohort=0;cohort<cohorts;++cohort){
        const int count=total/cohorts+(cohort<total%cohorts?1:0);
        MultiPenaltyGcvCudaCoefficientConsumerView c;c.coefficients=arena.p+37*offset;c.coefficient_dim=37;
        c.target_count=count;c.device_id=0;c.producer_stream=arena.stream;c.producer_completion_event=arena.event;
        c.target_keys={r.keys.begin()+offset,r.keys.begin()+offset+count};
        c.solver_info={info.solver_info.begin()+offset,info.solver_info.begin()+offset+count};
        c.decomposition_route={info.decomposition_route.begin()+offset,info.decomposition_route.begin()+offset+count};
        c.numerical_rank={info.numerical_rank.begin()+offset,info.numerical_rank.begin()+offset+count};
        auto finalized=multi_penalty_gcv_capacity_finalize_selected_coefficients(slot,r.Y.data()+351*offset,c,351,count,c.target_keys);
        layer_audit::native_scope(false);
        const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(finalized);
        const auto final_info=multi_penalty_gcv_capacity_residual_info(finalized);
        Snapshot actual,expected;actual.floating={shadow.coefficients,shadow.fitted,shadow.residuals};
        actual.integral={final_info.solver_info,final_info.decomposition_route,final_info.numerical_rank};
        for(int index=0;index<3;++index){const auto& values=stage_a[i].floating.at(index);const int stride=index==0?37:351;
          expected.floating.emplace_back(values.begin()+stride*offset,values.begin()+stride*(offset+count));
          const auto& ints=stage_a[i].integral.at(index);
          expected.integral.emplace_back(ints.begin()+offset,ints.begin()+offset+count);}
        need(actual.same(expected),"Stage A coefficient/fitted/residual bits differ");compared+=count;
        release_multi_penalty_gcv_capacity_residual(finalized);finalized.reset();offset+=count;layer_audit::native_scope(true);
      }
      layer_audit::native_scope(false);
      before=multi_penalty_gcv_capacity_prepared_info(slot);
      auto bad=packets[i];bad.q=38;rejects([&]{rebind_multi_penalty_gcv_capacity_l4_slot(slot,bad);});++negative;
      bad=packets[i];bad.pivot[1]=bad.pivot[0];rejects([&]{rebind_multi_penalty_gcv_capacity_l4_slot(slot,bad);});++negative;
      need(before.generation==multi_penalty_gcv_capacity_prepared_info(slot).generation,"invalid bind changed generation");
    }
    drain_multi_penalty_gcv_capacity_l4_slot(slot);old.reset();slot.reset();
  }closed();
  // Partial uploads require draining and discarding the slot, never resuming it.
  {
    auto slot=create_multi_penalty_gcv_capacity_l4_slot(packets.front(),0);
    l4_fault::upload_after(3);rejects([&]{rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets.front());});++negative;
    l4_fault::clear();drain_multi_penalty_gcv_capacity_l4_slot(slot);slot.reset();
  }closed();
  // Fail after coefficient D2D, before its completion event. Drain consumer
  // before dropping the producer lease, exactly as runtime cancellation does.
  {
    auto slot=create_multi_penalty_gcv_capacity_l4_slot(packets.front(),0);Store arena;const auto& r=*chosen.front();
    rebind_multi_penalty_gcv_capacity_l4_slot(slot,packets.front());
    auto value=multi_penalty_gcv_capacity_optimize_batch_with_residual(slot,r.Y.data(),351,r.keys.size(),r.keys,r.control);
    const auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(value.residual);
    check(cudaStreamWaitEvent(arena.stream,view.producer_completion_event,0));
    check(cudaMemcpyAsync(arena.p,view.coefficients,37*r.keys.size()*8,cudaMemcpyDeviceToDevice,arena.stream));
    l4_fault::next_event_record();rejects([&]{check(cudaEventRecord(arena.event,arena.stream));});++negative;l4_fault::clear();
    check(cudaStreamSynchronize(arena.stream));drain_multi_penalty_gcv_capacity_l4_slot(slot);
    release_multi_penalty_gcv_capacity_residual(value.residual);value.residual.reset();slot.reset();
  }closed();
  {
    std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> slots;
    for(int i=0;i<2;++i)slots.push_back(create_multi_penalty_gcv_capacity_l4_slot(packets.front(),0));
    std::atomic<int> failures{0};std::vector<std::thread> threads;layer_audit::native_scope(true);
    for(int worker=0;worker<2;++worker)threads.emplace_back([&,worker]{
      std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch> pending;
      try{check(cudaSetDevice(0));const auto& r=*chosen[worker];
        rebind_multi_penalty_gcv_capacity_l4_slot(slots[worker],packets[worker]);
        auto result=multi_penalty_gcv_capacity_optimize_batch_with_residual(slots[worker],r.Y.data(),351,r.keys.size(),r.keys,r.control);
        pending=result.residual;if(worker==0)throw std::runtime_error("injected completed-output cancellation");
        release_multi_penalty_gcv_capacity_residual(pending);pending.reset();
      }catch(...){++failures;drain_multi_penalty_gcv_capacity_l4_slot(slots[worker]);
        if(pending)release_multi_penalty_gcv_capacity_residual(pending);}
    });
    for(auto& t:threads)t.join();layer_audit::native_scope(false);need(failures==1,"cancellation injection failed");
    for(auto& s:slots){drain_multi_penalty_gcv_capacity_l4_slot(s);need(!multi_penalty_gcv_capacity_prepared_info(s).residual_slot_leased,"cancel lease");}
  }closed();need(layer_audit::counts().forbidden_calls==0,"worker resource mutation");
  std::cout<<"PASS capacities=14 requests=15 compared="<<compared<<" negative="<<negative
    <<" partial_upload=PASS partial_detach=PASS cancellation=PASS resources=PASS forbidden=0\n";
}catch(const std::exception& e){layer_audit::native_scope(false);l4_fault::clear();std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
