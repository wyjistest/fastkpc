// Fixed input/cohort replay through the production cross-setup optimizer.
#include "fixture.hpp"
#include "observe.hpp"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace window_diag;
using namespace fastkpc;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a) {return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
void check(cudaError_t x) {need(x==cudaSuccess,cudaGetErrorString(x));}
int main(int argc,char** argv) {
  try {
    need(argc==4,"replay FIXTURE PREFIX campaign|profile|smoke");
    auto fixture=Fixture::read(argv[1]);const std::string prefix=argv[2],mode=argv[3];
    need(mode=="campaign"||mode=="profile"||mode=="smoke","unknown replay mode");
    const int count=mode=="smoke"?2:64;
    check(cudaSetDevice(0));check(cudaFree(nullptr));
    size_t free_before=0,total=0;check(cudaMemGetInfo(&free_before,&total));
    std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> prepared;
    std::vector<std::string> setup_keys;
    std::ofstream setup_file(prefix+"-setups.csv");
    setup_file<<"setup,n,q,penalties,targets,prepare_ms\n"<<std::setprecision(17);
    const auto preparing=Clock::now();
    for (int i=0;i<count;++i) {
      const auto& r=fixture.requests.at(i);const auto& g=r.geometry;
      need(g.n==351 && g.q<=64 && g.capacity==int(r.keys.size()),"captured prepared shape changed");
      need(r.control.exact_primary_only&&r.control.force_final_selected_evaluation,"exact control missing");
      auto start=Clock::now();
      prepared.push_back(create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),
        g.tau.data(),g.R.data(),g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),
        g.n,g.q,g.penalties,g.capacity,0));
      setup_keys.push_back(r.keys.front());
      setup_file<<i<<','<<g.n<<','<<g.q<<','<<g.penalties<<','<<r.keys.size()<<','<<elapsed(start)<<'\n';
    }
    const double prepare_ms=elapsed(preparing);
    size_t free_prepared=0;check(cudaMemGetInfo(&free_prepared,&total));
    std::ofstream out(prefix+"-runs.csv");
    out<<"run,kind,workers,wall_ms,native_wall_ms,summed_setup_host_ms,max_setup_host_ms,targets,iterations,parity\n"<<std::setprecision(17);
    int run=0,all_targets=0;
    auto evaluate=[&](int workers,const std::string& kind,bool observe) {
      std::vector<MultiPenaltyGcvCapacityRequest> requests;
      for (int i=0;i<count;++i) {
        const auto& r=fixture.requests[i];MultiPenaltyGcvCapacityRequest input;
        input.prepared=prepared[i];input.Y=r.Y;input.n=r.geometry.n;
        input.target_count=r.keys.size();input.target_keys=r.keys;input.control=r.control;
        requests.push_back(std::move(input));
      }
      observe_begin(setup_keys,observe);
      const std::string label="window:L"+std::to_string(fixture.level)+":W"+
        std::to_string(fixture.window)+":workers"+std::to_string(workers)+":"+kind;
      nvtxRangePushA(label.c_str());auto started=Clock::now();
      auto result=multi_penalty_gcv_capacity_optimize_multi(std::move(requests),workers,true);
      const double wall=elapsed(started);nvtxRangePop();
      observe_end(prefix+"-run"+std::to_string(run)+"-timing.csv");
      need(result.setups.size()==size_t(count)&&result.residuals.size()==size_t(count),"replay result coverage");
      int targets=0,iterations=0;double maximum=0;
      for (int i=0;i<count;++i) {
        auto actual=snapshot(result.setups[i],result.residuals[i]);
        need(actual.same(fixture.requests[i].expected),"BITWISE_MISMATCH setup "+std::to_string(i));
        targets+=result.setups[i].target_count;
        for (auto v:result.setups[i].optimizer_iterations) iterations+=v;
        maximum=std::max(maximum,result.setups[i].diagnostics.total_host_ms);
        release_multi_penalty_gcv_capacity_residual(result.residuals[i]);result.residuals[i].reset();
        need(!multi_penalty_gcv_capacity_prepared_info(prepared[i]).residual_slot_leased,"unreleased output slot");
      }
      all_targets+=targets;
      out<<run<<','<<kind<<','<<workers<<','<<wall<<','<<result.diagnostics.wall_host_ms<<','
        <<result.diagnostics.summed_setup_host_ms<<','<<maximum<<','<<targets<<','<<iterations<<",1\n";out.flush();
      std::cout<<label<<" wall_ms="<<wall<<" PASS targets="<<targets<<std::endl;++run;
    };
    evaluate(mode=="smoke"?2:64,"warmup",false);
    if (mode=="campaign") {
      for (int workers:{64,32,16,32,16,64,16,64,32}) evaluate(workers,"timing",false);
      for (int workers:{64,32,16}) evaluate(workers,"observed",true);
    } else if (mode=="profile") {
      check(cudaProfilerStart());
      for (int workers:{64,32,16}) evaluate(workers,"profile",true);
      check(cudaProfilerStop());
    } else evaluate(2,"smoke",true);
    size_t free_after_runs=0;check(cudaMemGetInfo(&free_after_runs,&total));
    const auto releasing=Clock::now();prepared.clear();
    const double release_ms=elapsed(releasing);
    check(cudaDeviceSynchronize());size_t free_after=0;check(cudaMemGetInfo(&free_after,&total));
    std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
      <<"{\"schema\":\"optimizer-window-replay-v1\",\"status\":\"PASS\",\"level\":"<<fixture.level
      <<",\"window\":"<<fixture.window<<",\"setups\":"<<count<<",\"runs\":"<<run
      <<",\"compared_target_occurrences\":"<<all_targets<<",\"prepare_ms\":"<<prepare_ms
      <<",\"release_ms\":"<<release_ms<<",\"free_before\":"<<free_before
      <<",\"free_prepared\":"<<free_prepared<<",\"free_after_runs\":"<<free_after_runs
      <<",\"free_after_release\":"<<free_after<<",\"observed_events_closed\":true}\n";
    need(bool(out)&&bool(setup_file)&&bool(summary),"replay output write failed");return 0;
  } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
