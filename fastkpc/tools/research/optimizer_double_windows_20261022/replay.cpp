// Bounded adjacent-window experiment; all numerical work stays in the native.
#include "fixture.hpp"
#include "observe.hpp"
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>

using namespace window_diag;
using namespace fastkpc;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now()-t).count();
}
void check(cudaError_t x) { need(x==cudaSuccess,cudaGetErrorString(x)); }

int main(int argc,char** argv) {
  try {
    need(argc==6,"replay FIRST_FIXTURE SECOND_FIXTURE PREFIX campaign|profile|smoke 6");
    const auto first=Fixture::read(argv[1]),second=Fixture::read(argv[2]);
    const std::string prefix=argv[3],mode=argv[4];
    // The final argument explicitly fixes the number of timing cycles.
    const int cycles=std::stoi(argv[5]);
    need(first.level==second.level && second.window==first.window+1,"nonadjacent windows");
    need(mode=="campaign" || mode=="profile" || mode=="smoke","unknown mode");
    need(cycles==6,"campaign order is frozen to six balanced cycles");
    const int half=mode=="smoke"?2:64,count=2*half;
    std::vector<const Request*> inputs;
    for (const auto* f:{&first,&second})
      for(int i=0;i<half;++i) inputs.push_back(&f->requests.at(i));
    std::vector<std::string> keys;
    std::set<std::string> unique;
    for(const auto* r:inputs) {
      need(r->control.exact_primary_only && r->control.force_final_selected_evaluation,
           "exact control missing");
      need(unique.insert(r->keys.at(0)).second,"duplicate setup identity across windows");
      keys.push_back(r->keys.front());
    }
    check(cudaSetDevice(0));check(cudaFree(nullptr));
    size_t free_before=0,free_half=0,free_all=0,total=0;
    check(cudaMemGetInfo(&free_before,&total));
    std::vector<std::shared_ptr<MultiPenaltyGcvCapacityPrepared>> prepared;
    std::ofstream setups(prefix+"-setups.csv");
    setups<<"setup,window,local_setup,n,q,penalties,targets,prepare_ms\n"<<std::setprecision(17);
    const auto prepare_start=Clock::now();
    for(int i=0;i<count;++i) {
      const auto& r=*inputs[i];const auto& g=r.geometry;
      need(g.n==351 && g.q<=64 && g.capacity==int(r.keys.size()),"captured shape changed");
      const auto start=Clock::now();
      prepared.push_back(create_multi_penalty_gcv_capacity_prepared(g.X.data(),g.qr.data(),
        g.tau.data(),g.R.data(),g.pivot.data(),g.roots,g.matrices,g.ranks,g.initial.data(),
        g.n,g.q,g.penalties,g.capacity,0));
      setups<<i<<','<<(i<half?first.window:second.window)<<','<<i%half<<','<<g.n<<','
        <<g.q<<','<<g.penalties<<','<<r.keys.size()<<','<<elapsed(start)<<'\n';
      if(i+1==half) check(cudaMemGetInfo(&free_half,&total));
    }
    const double prepare_ms=elapsed(prepare_start);
    check(cudaMemGetInfo(&free_all,&total));
    std::ofstream runs(prefix+"-runs.csv"),calls(prefix+"-calls.csv");
    runs<<"run,cycle,kind,variant,workers,wall_ms,native_wall_sum_ms,targets,iterations,peak_host_calls,parity\n"<<std::setprecision(17);
    calls<<"run,variant,call,first_setup,setups,workers,wall_ms,native_wall_ms,summed_setup_host_ms,peak_host_calls\n"<<std::setprecision(17);
    int run=0,total_targets=0,total_calls=0;
    auto evaluate=[&](char variant,const std::string& kind,int cycle,bool observe) {
      const int workers=mode=="smoke"?(variant=='C'?2:4):(variant=='C'?32:64);
      const int call_count=variant=='A'?2:1;
      std::vector<std::vector<MultiPenaltyGcvCapacityRequest>> groups(call_count);
      for(int i=0;i<count;++i) {
        const auto& r=*inputs[i];MultiPenaltyGcvCapacityRequest request;
        request.prepared=prepared[i];request.Y=r.Y;request.n=r.geometry.n;
        request.target_count=r.keys.size();request.target_keys=r.keys;request.control=r.control;
        groups[variant=='A'?i/half:0].push_back(std::move(request));
      }
      std::vector<MultiPenaltyGcvCapacityMultiResult> results(call_count);
      std::vector<double> call_ms(call_count);
      observe_begin(keys,observe);
      const std::string label="pair:L"+std::to_string(first.level)+":W"+
        std::to_string(first.window)+"+"+std::to_string(second.window)+":"+variant+":"+kind;
      nvtxRangePushA(label.c_str());const auto started=Clock::now();
      for(int j=0;j<call_count;++j) {
        const auto t=Clock::now();
        results[j]=multi_penalty_gcv_capacity_optimize_multi(std::move(groups[j]),workers,true);
        call_ms[j]=elapsed(t);
      }
      const double wall_ms=elapsed(started);nvtxRangePop();
      observe_end(prefix+"-run"+std::to_string(run)+"-timing.csv");
      int index=0,targets=0,iterations=0,peak=0;double native_ms=0;
      for(int j=0;j<call_count;++j) {
        auto& result=results[j];const auto& d=result.diagnostics;
        const int expected_count=variant=='A'?half:count;
        need(result.setups.size()==size_t(expected_count) &&
          result.residuals.size()==size_t(expected_count),"result coverage");
        need(d.setup_count==expected_count && d.setup_stream_count==expected_count &&
          d.worker_count==std::min(workers,expected_count),"scheduler shape changed");
        native_ms+=d.wall_host_ms;peak=std::max(peak,d.max_host_calls_in_flight);
        calls<<run<<','<<variant<<','<<j<<','<<index<<','<<expected_count<<','<<d.worker_count
          <<','<<call_ms[j]<<','<<d.wall_host_ms<<','<<d.summed_setup_host_ms<<','
          <<d.max_host_calls_in_flight<<'\n';
        for(int k=0;k<expected_count;++k,++index) {
          auto& residual=result.residuals[k];const auto& expected=*inputs[index];
          const auto info=multi_penalty_gcv_capacity_residual_info(residual);
          need(info.target_keys==expected.keys && info.device_resident && !info.released,
               "target identity/order/residency changed");
          need(snapshot(result.setups[k],residual).same(expected.expected),
               "BITWISE_MISMATCH setup "+std::to_string(index));
          targets+=result.setups[k].target_count;
          for(int n:result.setups[k].optimizer_iterations) iterations+=n;
          release_multi_penalty_gcv_capacity_residual(residual);residual.reset();
          need(!multi_penalty_gcv_capacity_prepared_info(prepared[index]).residual_slot_leased,
               "unreleased output lease");
        }
      }
      need(index==count,"missing output setup");total_targets+=targets;total_calls+=call_count;
      runs<<run<<','<<cycle<<','<<kind<<','<<variant<<','<<workers<<','<<wall_ms<<','
        <<native_ms<<','<<targets<<','<<iterations<<','<<peak<<",1\n";
      runs.flush();calls.flush();
      std::cout<<label<<" run="<<run<<" cycle="<<cycle<<" wall_ms="<<wall_ms
        <<" PASS setups="<<count<<" targets="<<targets<<std::endl;
      ++run;
    };
    for(char variant:std::string("ABC")) evaluate(variant,"warmup",-1,false);
    if(mode=="campaign") {
      const std::vector<std::string> orders={"ABC","BCA","CAB","CBA","BAC","ACB"};
      for(int cycle=0;cycle<cycles;++cycle)
        for(char variant:orders[cycle]) evaluate(variant,"timing",cycle,false);
      for(char variant:std::string("ABC")) evaluate(variant,"observed",-1,true);
    } else if(mode=="profile") {
      check(cudaProfilerStart());
      for(char variant:std::string("ABC")) evaluate(variant,"profile",-1,true);
      check(cudaProfilerStop());
    } else {
      for(char variant:std::string("ABC")) evaluate(variant,"smoke",-1,true);
    }
    size_t free_after_runs=0,free_after_release=0;
    check(cudaMemGetInfo(&free_after_runs,&total));
    const auto release_start=Clock::now();prepared.clear();
    const double release_ms=elapsed(release_start);check(cudaDeviceSynchronize());
    check(cudaMemGetInfo(&free_after_release,&total));
    std::ofstream summary(prefix+"-summary.json");summary<<std::setprecision(17)
      <<"{\"schema\":\"optimizer-double-window-replay-v1\",\"status\":\"PASS\",\"level\":"
      <<first.level<<",\"first_window\":"<<first.window<<",\"second_window\":"<<second.window
      <<",\"setups\":"<<count<<",\"runs\":"<<run<<",\"native_calls\":"<<total_calls
      <<",\"compared_target_occurrences\":"<<total_targets<<",\"prepare_ms\":"<<prepare_ms
      <<",\"release_ms\":"<<release_ms<<",\"free_before\":"<<free_before
      <<",\"free_first_window\":"<<free_half<<",\"free_both_windows\":"<<free_all
      <<",\"free_after_runs\":"<<free_after_runs<<",\"free_after_release\":"<<free_after_release
      <<",\"output_leases_closed\":true,\"observation_events_closed\":true}\n";
    need(bool(summary)&&bool(runs)&&bool(calls)&&bool(setups),"output write failed");return 0;
  } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}
}
