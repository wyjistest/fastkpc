// Follow-up to a failed capacity gate only. No performance campaign, relaxation,
// production integration or reuse of this run as a successful capacity receipt.
// Reuse the source-bound verifier and fixture reader from the original harness.
#define main original_replay_main_not_called
#include "replay.cpp"
#undef main

int main(int argc,char** argv){try{
  need(argc==3,"capacity-diagnostic RAW PREFIX");const std::string prefix=argv[2];
  auto windows=read_layer(argv[1]);std::vector<const Request*> all;
  for(const auto& w:windows)for(const auto& r:w.requests)all.push_back(&r);
  std::vector<l4_slot::Packed> packets;std::vector<size_t> offsets;size_t values=0;
  for(auto r:all){packets.push_back(l4_slot::pack(*r));offsets.push_back(values);values+=(37+702)*r->keys.size();}
  check(cudaSetDevice(0));check(cudaFree(nullptr));closed();size_t initial=0,total=0;
  check(cudaMemGetInfo(&initial,&total));layer_audit::memory_limit(initial-size_t(2)*1024*1024*1024);
  std::ofstream phases(prefix+"-phases.csv");phases<<"phase,free_bytes,physical_increment_bytes,application_bytes,allocations,streams,events,blas_handles\n";
  size_t maximum=0,min_free=initial;
  auto record=[&](const std::string& phase){size_t free=0;check(cudaMemGetInfo(&free,&total));const auto c=layer_audit::counts();
    const size_t increment=initial>=free?initial-free:0;maximum=std::max(maximum,increment);min_free=std::min(min_free,free);
    phases<<phase<<','<<free<<','<<increment<<','<<c.bytes<<','<<c.allocations<<','<<c.streams<<','<<c.events<<','<<c.blas<<'\n';phases.flush();
    std::cout<<phase<<" physical_increment_bytes="<<increment<<" application_bytes="<<c.bytes<<std::endl;};
  record("context_initialized");
  std::vector<l4_slot::Handle> pool;for(int k=0;k<32;++k){pool.push_back(l4_slot::create(packets.front(),18));if(k==0)record("first_slot_created");}
  check(cudaDeviceSynchronize());record("all_slots_created");
  auto arena=std::make_unique<Arena>(values);record("arena_created");
  std::vector<l4_slot::Result> results(all.size());std::atomic<size_t> next{0};
  std::atomic<bool> cancel{false},start{false};std::atomic<int> ready{0};std::mutex error_mutex;std::exception_ptr error;
  auto worker=[&](int k){ready.fetch_add(1);while(!start.load())std::this_thread::yield();l4_slot::Result pending;
    try{check(cudaSetDevice(0));while(!cancel.load()){
      const size_t j=next.fetch_add(1);if(j>=all.size())break;
      l4_slot::bind(pool[k],packets[j],j+1);pending=l4_slot::execute(pool[k],*all[j]);
      need(pending.info.target_keys==all[j]->keys&&l4_slot::token_live(pending),"diagnostic target identity");
      const size_t count=all[j]->keys.size(),off=offsets[j];
      l4_slot::detach(pool[k],pending,arena->ptr+off,arena->ptr+off+37*count,arena->ptr+off+(37+351)*count);
      need(!l4_slot::token_live(pending)&&l4_slot::token_rejected(pending)&&!l4_slot::info(pool[k]).residual_slot_leased,"diagnostic release");
      results[j]=std::move(pending);
    }}catch(...){cancel=true;auto caught=std::current_exception();try{l4_slot::abandon(pool[k],pending);}catch(...){}
      std::lock_guard<std::mutex> guard(error_mutex);if(!error)error=caught;}
  };
  {NativeScope scope;std::vector<std::thread> threads;
    try{for(int k=0;k<32;++k)threads.emplace_back(worker,k);}catch(...){cancel=true;start=true;for(auto& t:threads)t.join();throw;}
    while(ready.load()!=32)std::this_thread::yield();start=true;for(auto& t:threads)t.join();for(auto& s:pool)l4_slot::drain(s);}
  if(error)std::rethrow_exception(error);record("work_completed");
  auto saved=arena->shadow();size_t targets=0,iterations=0;
  for(size_t j=0;j<all.size();++j){need(results[j].info.target_keys==all[j]->keys&&
      saved_snapshot(results[j],saved,offsets[j]).same(all[j]->expected),"diagnostic BITWISE_MISMATCH "+std::to_string(j));
    need(l4_slot::token_rejected(results[j]),"diagnostic stale token revived");targets+=all[j]->keys.size();
    iterations+=std::accumulate(results[j].optimization.optimizer_iterations.begin(),results[j].optimization.optimizer_iterations.end(),size_t(0));}
  record("outputs_verified");results.clear();pool.clear();arena.reset();check(cudaDeviceSynchronize());closed();record("resources_closed");
  need(layer_audit::counts().forbidden_calls==0,"diagnostic worker resource mutation");
  std::ofstream summary(prefix+"-summary.json");summary<<"{\"status\":\"DIAGNOSTIC_COMPLETE_CAPACITY_GATE_UNCHANGED\",\"setups\":"<<all.size()
    <<",\"targets\":"<<targets<<",\"iterations\":"<<iterations<<",\"outputs_bitwise_exact\":true,\"resources_closed\":true,\"forbidden_calls\":0,\"max_physical_increment_bytes\":"<<maximum
    <<",\"minimum_free_bytes\":"<<min_free<<",\"original_capacity_gate_bytes\":1073741824,\"capacity_gate_pass\":"<<(maximum<=size_t(1024)*1024*1024?"true":"false")
    <<",\"performance_campaign_authorized\":false}\n";
  need(bool(summary)&&bool(phases),"diagnostic receipt");std::cout<<"PASS diagnostic full outputs/resource closure; capacity gate unchanged"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<"FAIL diagnostic "<<e.what()<<std::endl;return 1;}}
