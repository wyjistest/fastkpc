"""Generate a host-only overlay from frozen v1 inputs; reuse unchanged CUDA objects."""
import difflib
from pathlib import Path
import subprocess
import time

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
OLD_TOOL = TOOL.parent / 'optimizer_l3_coexistence_20261027'


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def prepare(out, old, original, native, sha, write_new):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'protocol.json', dict(
        schema='l3-replacement-coexistence-split-raw-capacity-v2', declared_epoch=time.time(),
        declaration_sha256=sha(TOOL/'README.md'), old_manifest_sha256=sha(old/'manifest.json'),
        native_sha256=sha(native), slots=32, capacity=33, n=351, q=28,
        penalty_count=3, total_root_rank=24, target_count_min=2, target_count_max=33,
        creation_limit_bytes=1536*1024**2, optimizer_growth_limit_bytes=1280*1024**2,
        optimizer_total_limit_bytes=2816*1024**2, minimum_observed_free_bytes=2*1024**3,
        baseline='after actual coefficient arena reserve; before any pool creation',
        creation_boundary='pool complete, before any L3 launch',
        optimizer_boundary='all workers joined, before Stage A',
        stage_a_total_increment_limit_bytes=None, memory_sample_ms=10,
        W_S_free_comparison='descriptive only, not noninferiority',
        original_window_handles_retained=False, extra_device_output_arena_bytes=0,
        stack_model_subtraction=False, added_warmup=False, stack_limit_modified=False,
        existing_L4_L6_increment_limit_bytes=1024**3, performance_evidence=False,
        physical_gpu=1, selected_gpu_exclusive=True, external_gpu0_allowed=True,
        evidence_class='GPU1-exclusive shared-host semantic and resource qualification',
        success='GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT', runtime_integration=False))

    text=(old/'one_call.cpp').read_text()
    text='#include "capacity.hpp"\n'+text
    start=text.index('  void observe_capacity(bool pool_increment) {')
    end=text.index('  void close_noexcept()', start)
    unchanged=text[start:end]
    legacy=replace(unchanged, 'Policy::level == 3 || Policy::level == 4 || Policy::level == 6',
                   'Policy::level == 4 || Policy::level == 6')
    legacy=replace(legacy, '(Policy::level == 3 ? 1536ULL : 1024ULL)', '1024ULL')
    legacy=legacy.replace('  void observe_capacity(bool pool_increment) {',
                          '  void observe_legacy_capacity(bool pool_increment) {')
    text=text[:start]+'''  bool l3_creation_checked_ = false;
  void observe_capacity(bool pool_increment) {
    if constexpr (Policy::level == 3) {
      std::size_t free = 0, total = 0;
      check_one_call_cuda(cudaMemGetInfo(&free, &total), "L3 split capacity");
      diagnostics_->minimum_free = diagnostics_->minimum_free ?
        std::min(diagnostics_->minimum_free, free) : free;
      if (pool_increment) diagnostics_->maximum_pool_increment = std::max(
        diagnostics_->maximum_pool_increment,
        l3_capacity::increment(diagnostics_->free_before, free));
      const auto phase = !l3_creation_checked_ ? l3_capacity::Phase::creation :
        (pool_increment ? l3_capacity::Phase::optimizer_done : l3_capacity::Phase::stage_a);
      require(l3_capacity::accepted(phase, diagnostics_->free_before,
                diagnostics_->free_after, free), "L3 split raw capacity contract exceeded");
      l3_creation_checked_ = true;
    } else {
      observe_legacy_capacity(pool_increment);
    }
  }
'''+legacy+text[end:]
    # The L4/L6 guard body is exactly the production implementation.
    production=(ROOT/'fastkpc/src/full_cuda_ci_one_call.cpp').read_text()
    a=production.index('  void observe_capacity(bool pool_increment) {')
    b=production.index('  void close_noexcept()',a)
    assert legacy.replace('observe_legacy_capacity','observe_capacity') == production[a:b]
    (out/'one_call.cpp').write_text(text)
    (out/'overlay-v2.patch').write_text(''.join(difflib.unified_diff(
        (old/'one_call.cpp').read_text().splitlines(True), text.splitlines(True),
        fromfile='frozen-v1/one_call.cpp', tofile='v2/one_call.cpp')))

    probe=(OLD_TOOL/'probe.cpp').read_text()
    probe='#include "capacity.hpp"\n'+probe
    probe=replace(probe,'std::mutex mutex;', 'std::mutex mutex, sample_mutex;')
    probe=replace(probe,'std::size_t pool_before_free=0,pool_increment=0;', '''std::size_t pool_before_free=0,pool_increment=0,pool_created_free=0;
std::size_t creation_increment=0,optimizer_growth=0,optimizer_total=0;
std::atomic<std::size_t> observed_baseline{0}, observed_created{0}, stage_minimum{SIZE_MAX},stage_total_max{0};
std::atomic<int> sample_phase{0};
std::ofstream memory_samples;''')
    a=probe.index('std::size_t observe()');b=probe.index('template<class T>bool bits',a)
    probe=probe[:a]+'''std::size_t observe(bool sampled=false){
  std::size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
  auto old=minimum.load();while(old>free&&!minimum.compare_exchange_weak(old,free)){}
  ++sample_count;
  const auto baseline=observed_baseline.load(), created=observed_created.load();
  const int phase=sample_phase.load();
  const auto raw=l3_capacity::increment(baseline,free);
  if(phase==2){
    old=stage_minimum.load();while(old>free&&!stage_minimum.compare_exchange_weak(old,free)){}
    old=stage_total_max.load();while(old<raw&&!stage_total_max.compare_exchange_weak(old,raw)){}
  }
  const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  std::lock_guard<std::mutex> lock(sample_mutex);
  memory_samples<<(sampled?"sample":"checkpoint")<<','<<ns<<','<<phase<<','<<free<<','
    <<baseline<<','<<created<<','<<raw<<'\\n';
  return free;
}
'''+probe[b:]
    a=probe.index('void checkpoint(const char* name)');b=probe.index('void begin(bool slots)',a)
    probe=probe[:a]+'''void checkpoint(const char* name){
  if(!begun||done)return;
  const std::string point(name);
  if(point=="stage_a_cohort")sample_phase=2;
  const auto free=observe();const auto c=layer_audit::counts();
  bool capacity_pass=free>=l3_capacity::minimum_free;
  if(point=="pool_before"){
    pool_before=c;pool_before_free=free;observed_baseline=free;sample_phase=1;
  }
  if(point=="pool_created"){
    pool_created=c;pool_created_free=free;observed_created=free;
    creation_increment=l3_capacity::increment(pool_before_free,free);
    pool_increment=creation_increment;
    capacity_pass=l3_capacity::accepted(l3_capacity::Phase::creation,pool_before_free,free,free);
  }
  if(point=="optimizer_done"){
    optimizer_growth=l3_capacity::increment(pool_created_free,free);
    optimizer_total=l3_capacity::increment(pool_before_free,free);
    pool_increment=std::max(pool_increment,optimizer_total);
    capacity_pass=l3_capacity::accepted(l3_capacity::Phase::optimizer_done,pool_before_free,pool_created_free,free);
    sample_phase=2;
  }
  if(point=="pool_closing")pool_closing=c;
  if(point=="pool_closed"){
    pool_closed=delta_closed(pool_closing,c);need(pool_closed,"L3 pool resources not restored");
  }
  checkpoints<<name<<','<<free<<','<<c.bytes<<','<<c.allocations<<','<<c.streams<<','<<c.events<<','<<c.blas<<','<<stage_targets
    <<','<<creation_increment<<','<<optimizer_growth<<','<<l3_capacity::increment(pool_before_free,free)<<','<<capacity_pass<<'\\n';
  checkpoints.flush();
  need(capacity_pass,"L3 split raw capacity contract exceeded at checkpoint");
}
'''+probe[b:]
    probe=replace(probe,'checkpoints<<"checkpoint,free_bytes,application_bytes,allocations,streams,events,blas,stage_a_targets\\n";',
        '''checkpoints<<"checkpoint,free_bytes,application_bytes,allocations,streams,events,blas,stage_a_targets,creation_increment_bytes,optimizer_growth_bytes,raw_total_increment_bytes,capacity_pass\\n";
  memory_samples.open(std::string(env("FASTKPC_L3_PROBE_OUT"))+"/memory-samples.csv");
  memory_samples<<"kind,monotonic_ns,phase,free_bytes,baseline_free_bytes,created_free_bytes,raw_total_increment_bytes\\n";''')
    probe=replace(probe,'try{observe();}', 'try{observe(true);}')
    probe=replace(probe,'sampling=false;if(monitor.joinable())monitor.join();\n  const auto c=',
        'sampling=false;if(monitor.joinable())monitor.join();memory_samples.close();\n  const auto c=')
    probe=replace(probe,'optimizer_targets==54878&&stage_targets==54878&&',
        'optimizer_targets==54878&&stage_targets==54878&&stage_batches==5573&&')
    probe=replace(probe,'<<",\\"observations\\":"<<sample_count.load()',
        '''<<",\\"baseline_free_bytes\\":"<<pool_before_free<<",\\"created_free_bytes\\":"<<pool_created_free
     <<",\\"creation_increment_bytes\\":"<<creation_increment<<",\\"optimizer_growth_bytes\\":"<<optimizer_growth
     <<",\\"optimizer_total_increment_bytes\\":"<<optimizer_total
     <<",\\"stage_a_minimum_free_bytes\\":"<<stage_minimum.load()
     <<",\\"stage_a_maximum_raw_total_increment_bytes\\":"<<stage_total_max.load()
     <<",\\"observations\\":"<<sample_count.load()''')
    probe=replace(probe,'// Join even if a native error', '''extern "C" SEXP fastkpc_l3_probe_cleanup_snapshot(){BEGIN_RCPP
  sampling=false;if(monitor.joinable())monitor.join();memory_samples.flush();
  const auto c=layer_audit::counts();
  return Rcpp::List::create(Rcpp::Named("done")=done,
    Rcpp::Named("allocations")=double(c.allocations),Rcpp::Named("bytes")=double(c.bytes),
    Rcpp::Named("streams")=double(c.streams),Rcpp::Named("events")=double(c.events),
    Rcpp::Named("blas")=double(c.blas),Rcpp::Named("worker_resource_mutations")=double(c.forbidden_calls),
    Rcpp::Named("optimizer_setups")=double(optimized.size()),Rcpp::Named("optimizer_targets")=double(optimizer_targets),
    Rcpp::Named("stage_a_targets")=double(stage_targets),Rcpp::Named("minimum_free_bytes")=double(minimum.load()));
END_RCPP}
// Join even if a native error''')
    (out/'probe.cpp').write_text(probe)
    (out/'probe-v2.patch').write_text(''.join(difflib.unified_diff(
        (OLD_TOOL/'probe.cpp').read_text().splitlines(True),probe.splitlines(True),
        fromfile='frozen-v1/probe.cpp',tofile='v2/probe.cpp')))

    runner=(old/'run.R').read_text()
    runner=replace(runner,'result <- fastkpc_compatible_cuda_skeleton(',
                    'result <- tryCatch(fastkpc_compatible_cuda_skeleton(')
    runner=replace(runner,'ci_method="dcc.gamma"))', '''ci_method="dcc.gamma")), error=function(e) {
  jsonlite::write_json(list(error=conditionMessage(e),
    resources=.Call("fastkpc_l3_probe_cleanup_snapshot")),
    file.path(out,"failure.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
  stop(e)
})''')
    runner=replace(runner,'elapsed <- unname(proc.time()', '''cleanup <- .Call("fastkpc_l3_probe_cleanup_snapshot")
jsonlite::write_json(cleanup,file.path(out,"cleanup.json"),auto_unbox=TRUE,pretty=TRUE,digits=NA)
elapsed <- unname(proc.time()''')
    runner=replace(runner,'l3pool <- s$l3_probe_slots', '''l3pool <- s$l3_probe_slots
l3probe <- jsonlite::read_json(file.path(out,"probe.json"),simplifyVector=TRUE)''')
    runner=replace(runner,'l3pool$minimum_free_bytes>=2*1024^3 && l3pool$maximum_pool_increment_bytes<=1536*1024^2 &&',
        '''l3pool$minimum_free_bytes>=2*1024^3 && l3pool$maximum_pool_increment_bytes<=2816*1024^2 &&
    l3probe$creation_increment_bytes<=1536*1024^2 && l3probe$optimizer_growth_bytes<=1280*1024^2 &&
    l3probe$optimizer_total_increment_bytes<=2816*1024^2 && l3probe$minimum_free_bytes>=2*1024^3 &&
    l3probe$stage_a_minimum_free_bytes>=2*1024^3 && l3probe$status=="PASS" &&''')
    runner=replace(runner,'schema="optimizer-l6-slot-runtime-canonical-v1"',
                   'schema="optimizer-l3-coexistence-split-capacity-v2"')
    runner=replace(runner,'receipt$l3_probe_slots <- l3pool',
                   'receipt$l3_probe_slots <- l3pool; receipt$l3_capacity_probe <- l3probe')
    runner=replace(runner,'sha("fastkpc/tools/research/optimizer_l6_slot_runtime_20261026/run.R")',
                   'sha(file.path(dirname(out),"run.R"))')
    (out/'run.R').write_text(runner)
    (out/'runner-v2.patch').write_text(''.join(difflib.unified_diff(
        (old/'run.R').read_text().splitlines(True),runner.splitlines(True),
        fromfile='frozen-v1/run.R',tofile='v2/run.R')))

    commands=[['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(TOOL),
               str(TOOL/'test_capacity.cpp'),'-o',str(out/'test_capacity')],
              [str(out/'test_capacity')]]
    for name in ['one_call','probe']:
        cmd=next(c for c in original['commands'] if c[-1]==str(old/(name+'.o')))
        cmd=[s.replace(str(old),str(out)) for s in cmd]
        cmd=[str(out/'probe.cpp') if s==str(OLD_TOOL/'probe.cpp') else s for s in cmd]
        cmd.insert(1,'-I'+str(TOOL))
        commands.append(cmd)
    link=[s.replace(str(old),str(out)) for s in original['commands'][-1]]
    for name in ['bridge','rebind']:
        link=[str(old/(name+'.o')) if s==str(out/(name+'.o')) else s for s in link]
    commands.append(link)
    for i,cmd in enumerate(commands):
        print('BUILD',i,Path(cmd[0]).name,flush=True)
        with (out/('build-'+str(i)+'.log')).open('x') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    inputs=[p for p in TOOL.iterdir() if p.is_file()]+[
        old/'one_call.cpp',old/'run.R',old/'bridge.o',old/'rebind.o',old/'device-code.json',
        old/'build.json',old/'manifest.json',OLD_TOOL/'probe.cpp',OLD_TOOL/'probe.hpp',
        OLD_TOOL/'geometry.hpp',TOOL.parent/'optimizer_l3_stack_20261027/campaign.py',
        TOOL.parent/'optimizer_l3_slots_20261027/common.py']
    write_new(out/'build.json',dict(status='PASS',native_sha256=sha(native),
        source_sha256=original['source_sha256'],protocol_sha256=sha(out/'protocol.json'),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},commands=commands,
        diagnostic_overlay_sha256=sha(out/'probe.so'),
        device_code='reuse original bridge/rebind objects; all 14 functions retain frozen instruction identity',
        source_class='diagnostic host capacity overlay; unchanged production math entrypoints'))
    print('PASS build',sha(out/'probe.so'),flush=True)
