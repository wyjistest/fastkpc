// Linked only to the frozen parent native. Reproduce its actual Stage A cohorts,
// preserving each original L6 target cohort (2, 3 or 5).
#include "../optimizer_l6_slots_20261025/fixture.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <map>
using namespace fastkpc;
using namespace window_diag;
void check(cudaError_t e){need(e==cudaSuccess,cudaGetErrorString(e));}
int main(int argc,char** argv)try{
  need(argc==3,"captured L6 directory and output required");
  check(cudaSetDevice(0));auto windows=l6_diag::read_layer(argv[1]);
  std::map<int,const Request*> first;for(const auto& w:windows)for(const auto& r:w.requests)first.emplace(r.keys.size(),&r);
  need(first.size()==3,"capacity coverage");std::vector<const Request*> chosen;
  for(const auto& e:first)chosen.push_back(e.second);chosen.push_back(&windows.back().requests.back());
  double* coefficients=nullptr;cudaStream_t stream=nullptr;cudaEvent_t event=nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&coefficients),55*5*8));
  check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming|cudaEventBlockingSync));
  std::ofstream out(argv[2],std::ios::binary);IO io{nullptr,&out};
  std::string schema="l6-original-stage-a-v1";io.value(schema);int count=chosen.size();io.value(count);
  std::size_t different=0;double max_error=0;
  for(const auto* input:chosen){
    const auto& r=*input;const auto& g=r.geometry;
    auto prepared=create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),g.tau.data(),g.R.data(),g.pivot.data(),
      g.roots,g.matrices,g.ranks,g.initial.data(),351,55,6,r.keys.size(),0);
    auto optimized=multi_penalty_gcv_capacity_optimize_batch_with_residual(prepared,r.Y.data(),351,r.keys.size(),r.keys,r.control);
    need(snapshot(optimized.optimization,optimized.residual).same(r.expected),"parent optimizer capture mismatch");
    const auto info=multi_penalty_gcv_capacity_residual_info(optimized.residual);
    const auto view=acquire_multi_penalty_gcv_capacity_residual_consumer_view(optimized.residual);
    check(cudaStreamWaitEvent(stream,view.producer_completion_event,0));
    check(cudaMemcpyAsync(coefficients,view.coefficients,55*r.keys.size()*8,cudaMemcpyDeviceToDevice,stream));
    check(cudaEventRecord(event,stream));check(cudaEventSynchronize(event));
    release_multi_penalty_gcv_capacity_residual(optimized.residual);optimized.residual.reset();
    const int total=r.keys.size(),cohorts=(total+15)/16;int offset=0;
    Snapshot expected;expected.floating.resize(3);expected.integral.resize(3);
    for(int cohort=0;cohort<cohorts;++cohort){
      const int n=total/cohorts+(cohort<total%cohorts?1:0);
      MultiPenaltyGcvCudaCoefficientConsumerView c;c.coefficients=coefficients+55*offset;c.coefficient_dim=55;
      c.target_count=n;c.device_id=0;c.producer_stream=stream;c.producer_completion_event=event;
      c.target_keys={r.keys.begin()+offset,r.keys.begin()+offset+n};
      c.solver_info={info.solver_info.begin()+offset,info.solver_info.begin()+offset+n};
      c.decomposition_route={info.decomposition_route.begin()+offset,info.decomposition_route.begin()+offset+n};
      c.numerical_rank={info.numerical_rank.begin()+offset,info.numerical_rank.begin()+offset+n};
      auto final=multi_penalty_gcv_capacity_finalize_selected_coefficients(prepared,r.Y.data()+351*offset,c,351,n,c.target_keys);
      const auto shadow=materialize_multi_penalty_gcv_capacity_residual_shadow(final);
      const auto final_info=multi_penalty_gcv_capacity_residual_info(final);
      const std::vector<std::vector<double>> floating={shadow.coefficients,shadow.fitted,shadow.residuals};
      const std::vector<std::vector<int>> integral={final_info.solver_info,final_info.decomposition_route,final_info.numerical_rank};
      for(int j=0;j<3;++j){
        expected.floating[j].insert(expected.floating[j].end(),floating[j].begin(),floating[j].end());
        expected.integral[j].insert(expected.integral[j].end(),integral[j].begin(),integral[j].end());
      }
      release_multi_penalty_gcv_capacity_residual(final);offset+=n;
    }
    std::size_t local=0;double local_max=0;
    for(int j=0;j<3;++j){const auto& a=expected.floating[j];const auto& b=r.expected.floating[11+j];
      need(a.size()==b.size(),"parent Stage A size");for(std::size_t k=0;k<a.size();++k){
        local+=std::memcmp(&a[k],&b[k],8)!=0;local_max=std::max(local_max,std::abs(a[k]-b[k]));}}
    different+=local;max_error=std::max(max_error,local_max);
    std::cout<<"parent cohort_targets="<<total<<" stage_a_vs_optimizer_changed_values="<<local
             <<" max_abs="<<local_max<<std::endl;
    auto keys=r.keys;io.value(keys);expected.io(io);
  }
  check(cudaStreamSynchronize(stream));check(cudaEventDestroy(event));check(cudaStreamDestroy(stream));check(cudaFree(coefficients));
  need(bool(out),"reference write failed");
  std::cout<<"PASS original Stage A references requests="<<chosen.size()<<" changed_vs_optimizer="<<different
           <<" max_abs="<<max_error<<std::endl;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}

