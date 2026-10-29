// Reuse the frozen GPU-only fixture and snapshot helpers, not a CPU solver.
#define main postback_cohort_main
#include "cohorts.cpp"
#undef main
#include <cstdlib>

template<class F> void rejects(F f,const std::string& message){
  bool caught=false;
  try{f();}catch(const std::exception& e){caught=std::string(e.what()).find(message)!=std::string::npos;}
  need(caught,"expected fail-closed: "+message);
}
void mode(int value){setenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK",value?"1":"0",1);}
Snapshot residual_snapshot(const std::shared_ptr<MultiPenaltyGcvCapacityResidualBatch>& r){
  auto i=multi_penalty_gcv_capacity_residual_info(r);
  auto s=materialize_multi_penalty_gcv_capacity_residual_shadow(r);
  for(auto v:i.solver_info)need(v==0,"fixed solver info");
  for(auto v:i.decomposition_route)need(v==kMultiPenaltyGcvDecompositionStableSvd,"fixed stable SVD");
  Snapshot result;result.floating={s.coefficients,s.fitted,s.residuals};
  result.integral={i.solver_info,i.decomposition_route,i.numerical_rank};return result;
}
MultiPenaltyGcvCapacityRequest request(const Request& r,const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& p){
  MultiPenaltyGcvCapacityRequest v;v.prepared=p;v.Y=r.Y;v.n=r.geometry.n;
  v.target_count=r.keys.size();v.target_keys=r.keys;v.control=r.control;return v;
}
template<int Level> void contracts(const Request& r,const Request& second){
  closed();auto packet=pack<Level>(r);
  unsetenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK");need(!multi_penalty_gcv_vector_postback_requested(),"default OFF");
  auto before=multi_penalty_gcv_postback_execution_counts();
  auto off=create<Level>(packet);mode(1);auto on=create<Level>(packet);
  auto counts=multi_penalty_gcv_postback_execution_counts();
  need(counts.created[0]==before.created[0]+1&&counts.created[1]==before.created[1]+1,"creation mode");
  for(const char* bad:{"2","true","-1"," 1"}){
    setenv("FASTKPC_DCC_GAMMA_VECTOR_POSTBACK",bad,1);
    auto calls=layer_audit::counts().allocation_calls;
    rejects([&]{auto invalid=create<Level>(packet);},"must be 0 or 1");
    need(layer_audit::counts().allocation_calls==calls,"invalid flag before GPU allocation");
  }
  // Mixed mode requests must fail before any GPU execution or counter update.
  before=multi_penalty_gcv_postback_execution_counts();
  layer_audit::native_scope(true);auto mutations=layer_audit::counts().forbidden_calls;
  rejects([&]{multi_penalty_gcv_capacity_optimize_multi({request(r,off),request(r,on)},2,true);},"vector-postback modes");
  layer_audit::native_scope(false);
  need(layer_audit::counts().forbidden_calls==mutations,"mixed modes GPU mutation");
  counts=multi_penalty_gcv_postback_execution_counts();
  need(counts.optimizer_targets[0]==before.optimizer_targets[0]&&counts.optimizer_targets[1]==before.optimizer_targets[1],"mixed mode execution");
  Snapshot fixed[2];const int targets=r.keys.size();
  for(int m=0;m<2;++m){
    auto p=m?on:off;mode(!m); // opposite flag after creation
    before=multi_penalty_gcv_postback_execution_counts();
    bind<Level>(p,packet);
    auto optimized=multi_penalty_gcv_capacity_optimize_batch_with_residual(p,r.Y.data(),351,targets,r.keys,r.control);
    need(snapshot(optimized.optimization,optimized.residual).same(r.expected),"immutable mode optimizer parity");
    rejects([&]{bind<Level>(p,packet);},"lease");
    auto v=acquire_multi_penalty_gcv_capacity_residual_consumer_view(optimized.residual);
    MultiPenaltyGcvCudaCoefficientConsumerView detached;
    double* coef=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&coef),v.coefficient_dim*targets*8));
    check(cudaMemcpyAsync(coef,v.coefficients,v.coefficient_dim*targets*8,cudaMemcpyDeviceToDevice,v.producer_stream));
    cudaEvent_t detached_ready=nullptr;check(cudaEventCreateWithFlags(&detached_ready,cudaEventDisableTiming));
    check(cudaEventRecord(detached_ready,v.producer_stream));check(cudaEventSynchronize(detached_ready));
    detached.coefficients=coef;detached.coefficient_dim=v.coefficient_dim;detached.target_count=targets;
    detached.device_id=v.device_id;detached.target_keys=v.target_keys;
    detached.producer_stream=v.producer_stream;detached.producer_completion_event=detached_ready;
    detached.solver_info=v.solver_info;detached.decomposition_route=v.decomposition_route;detached.numerical_rank=v.numerical_rank;
    release_multi_penalty_gcv_capacity_residual(optimized.residual);
    rejects([&]{acquire_multi_penalty_gcv_capacity_residual_consumer_view(optimized.residual);},"released");
    optimized.residual.reset();
    auto finalized=multi_penalty_gcv_capacity_finalize_selected_coefficients(p,r.Y.data(),detached,351,targets,r.keys);
    auto final=residual_snapshot(finalized);Snapshot expected;
    expected.floating={r.expected.floating.at(11),r.expected.floating.at(12),r.expected.floating.at(13)};
    expected.integral={r.expected.integral.at(16),r.expected.integral.at(17),r.expected.integral.at(18)};
    need(final.same(expected),"coefficient finalization parity");
    release_multi_penalty_gcv_capacity_residual(finalized);finalized.reset();check(cudaFree(coef));check(cudaEventDestroy(detached_ready));
    auto evaluated=multi_penalty_gcv_capacity_evaluate_fixed_batch(p,r.Y.data(),optimized.optimization.selected_log_sp.data(),
      351,targets,r.keys,r.control.rank_tolerance);
    fixed[m]=residual_snapshot(evaluated);release_multi_penalty_gcv_capacity_residual(evaluated);evaluated.reset();
    counts=multi_penalty_gcv_postback_execution_counts();
    for(int k=0;k<2;++k){
      const int expected_count=k==m?targets:0;
      need(counts.optimizer_targets[k]-before.optimizer_targets[k]==expected_count,"frozen optimizer counter");
      need(counts.fixed_targets[k]-before.fixed_targets[k]==expected_count,"frozen fixed counter");
      need(counts.coefficient_finalization_targets[k]-before.coefficient_finalization_targets[k]==expected_count,"frozen finalization counter");
    }
    bind<Level>(p,packet);drain<Level>(p);
  }
  need(fixed[0].same(fixed[1]),"OFF/ON exact fixed parity");
  off.reset();on.reset();check(cudaDeviceSynchronize());closed();
  for(int m=0;m<2;++m){
    mode(m);auto a=create<Level>(packet);auto b=create<Level>(pack<Level>(second));mode(!m);
    auto result=multi_penalty_gcv_capacity_optimize_multi({request(r,a),request(second,b)},2,true);
    need(result.setups.size()==2&&result.residuals.size()==2,"multi result size");
    need(snapshot(result.setups[0],result.residuals[0]).same(r.expected),"multi first parity");
    need(snapshot(result.setups[1],result.residuals[1]).same(second.expected),"multi second parity");
    for(auto& residual:result.residuals)release_multi_penalty_gcv_capacity_residual(residual);
    result.residuals.clear();drain<Level>(a);drain<Level>(b);a.reset();b.reset();closed();
  }
  std::cout<<"PASS L"<<Level<<" dispatch/immutability/mixed/fixed/finalization/lease/multi/resource contracts"<<std::endl;
}
int main(int argc,char** argv)try{
  need(argc==4,"contracts L4_RAW L5_RAW L3_FIXTURE");check(cudaSetDevice(0));check(cudaFree(nullptr));closed();
  auto l4=l4_diag::read_layer(argv[1]);auto l5=layer_diag::read_layer(argv[2]);
  contracts<4>(l4[0].requests[0],l4[0].requests[1]);contracts<5>(l5[0].requests[0],l5[0].requests[1]);
  // q=28 keeps the original object even with ON requested.
  auto l3=Fixture::read(argv[3]);const auto& r=l3.requests[0];const auto& g=r.geometry;mode(1);
  need(g.q==28,"non-target fixture q");auto before=multi_penalty_gcv_postback_execution_counts();
  auto p=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),
    g.roots,g.matrices,g.ranks,g.initial.data(),g.n,g.q,g.penalties,g.capacity,0);
  auto v=multi_penalty_gcv_capacity_optimize_batch_with_residual(p,r.Y.data(),g.n,r.keys.size(),r.keys,r.control);
  need(snapshot(v.optimization,v.residual).same(r.expected),"non-target q parity");
  release_multi_penalty_gcv_capacity_residual(v.residual);v.residual.reset();p.reset();check(cudaDeviceSynchronize());closed();
  auto after=multi_penalty_gcv_postback_execution_counts();
  need(after.optimizer_targets[1]==before.optimizer_targets[1]&&after.other_targets[0]-before.other_targets[0]==r.keys.size(),"non-target mode");
  need(after.created[0]==after.destroyed[0]&&after.created[1]==after.destroyed[1],"wrapper counts close");
  std::cout<<"PASS all runtime contracts; GPU numerical authority only"<<std::endl;return 0;
}catch(const std::exception& e){layer_audit::native_scope(false);std::cerr<<e.what()<<std::endl;return 1;}
