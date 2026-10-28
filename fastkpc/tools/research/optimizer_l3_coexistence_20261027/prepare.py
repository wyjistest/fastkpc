"""Isolated diagnostic overlay; never replace production source/native."""
import difflib
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,PARENT,sha,load,write_new,check_identity
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
captured=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027'
identity=check_identity(captured)
assert load(captured/'closure.json')['status']=='GO_CAPTURED_L3_SLOTS'
write_new(out/'protocol.json',dict(schema='l3-production-replacement-coexistence-v1',
    declared_epoch=time.time(),declaration_sha256=sha(TOOL/'README.md'),slots=32,capacity=33,
    maximum_pool_increment_bytes=1536*1024**2,minimum_observed_free_bytes=2*1024**3,
    memory_sample_ms=10,original_window_handles_retained=False,extra_device_output_arena_bytes=0,
    capacity_basis='after coefficient arena reserve, before pool; pre/post optimizer',
    joint_basis='whole L3 optimizer and Stage A lifetime; checkpoints and samples',
    variants=['original windows W','replacement slots S'],performance_evidence=False,
    native_sha256=sha(NATIVE),capture_closure_sha256=sha(captured/'closure.json')))
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old[:120],text.count(old),count)
    return text.replace(old,new)

capacity=(ROOT/'fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.cpp').read_text()
layout=capacity[capacity.index('using CapacityHandle ='):capacity.index('\nnamespace {')]
(out/'capacity-layout.inc').write_text('namespace fastkpc {\n'+layout+'\n}\n')
cuda=(ROOT/'fastkpc/src/cuda/mgcv_multi_penalty_gcv_small64.cu').read_text()
rebind=cuda[cuda.index('void rebind_multi_penalty_gcv_cuda_prepared_l4('):cuda.index('void rebind_multi_penalty_gcv_cuda_prepared_l6(')]
for old,new in [('rebind_multi_penalty_gcv_cuda_prepared_l4','rebind_cuda'),
    ('drain_multi_penalty_gcv_cuda_prepared_l4','drain_cuda'),('validate_multi_penalty_gcv_l4_geometry','validate'),
    ('MultiPenaltyGcvL4Geometry','Geometry'),('p.target_capacity != 18','p.target_capacity != 33'),('L4 slot','L3 probe slot')]:
    rebind=rebind.replace(old,new)
(out/'rebind.cu').write_text('#include "geometry.hpp"\n#include "cuda/mgcv_multi_penalty_gcv_small64.cu"\nnamespace l3_probe {\nusing namespace fastkpc;\n'+rebind+'\n}\n')
original=(ROOT/'fastkpc/src/full_cuda_ci_one_call.cpp').read_text()
text='#include "probe.hpp"\n'+original
text=replace(text,'OptimizerSlotDiagnostics l4_slots, l5_slots, l6_slots;','OptimizerSlotDiagnostics l3_probe, l4_slots, l5_slots, l6_slots;')
policy='''struct L3ProbePolicy {
  using Geometry = l3_probe::Geometry;
  static constexpr int level = 3, q = 28, penalties = 3;
  static void validate(const Geometry& g) { l3_probe::validate(g); }
  static bool cohort(int n) { return n >= 2 && n <= 33; }
  static auto create(const Geometry& g,int device) { return l3_probe::create(g,device); }
  static void bind(const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& p,const Geometry& g) { l3_probe::bind(p,g); }
  static void drain(const std::shared_ptr<MultiPenaltyGcvCapacityPrepared>& p) { l3_probe::drain(p); }
};
'''
text=replace(text,'struct L4OptimizerSlotPolicy {',policy+'struct L4OptimizerSlotPolicy {')
text=replace(text,'if constexpr (Policy::level == 4 || Policy::level == 6)',
    'if constexpr (Policy::level == 3 || Policy::level == 4 || Policy::level == 6)')
text=replace(text,'diagnostics_->maximum_pool_increment <= 1024ULL * 1024 * 1024',
    'diagnostics_->maximum_pool_increment <= (Policy::level == 3 ? 1536ULL : 1024ULL) * 1024 * 1024')
text=replace(text,'static_assert(Policy::level == 4 || Policy::level == 5 || Policy::level == 6,',
    'static_assert(Policy::level == 3 || Policy::level == 4 || Policy::level == 5 || Policy::level == 6,')
text=replace(text,'auto& d = Policy::level == 4 ? diagnostics->l4_slots :',
    'auto& d = Policy::level == 3 ? diagnostics->l3_probe : (Policy::level == 4 ? diagnostics->l4_slots :')
text=replace(text,'(Policy::level == 5 ? diagnostics->l5_slots : diagnostics->l6_slots);',
    '(Policy::level == 5 ? diagnostics->l5_slots : diagnostics->l6_slots));')
text=replace(text,'geometry.push_back(pack_optimizer_slot_geometry<Policy>(work.metadata[i].context));',
    'geometry.push_back(pack_optimizer_slot_geometry<Policy>(work.metadata[i].context));\n    if constexpr (Policy::level == 3) l3_probe::geometry(geometry.back(),request);')
text=replace(text,'OptimizerSlotPool<Policy> pool(geometry, &d);',
    'if constexpr (Policy::level == 3) l3_probe::checkpoint("pool_before");\n  OptimizerSlotPool<Policy> pool(geometry, &d);\n  if constexpr (Policy::level == 3) l3_probe::checkpoint("pool_created");')
text=replace(text,'pending = std::move(result.residual);\n        const auto& values',
    'pending = std::move(result.residual);\n        if constexpr (Policy::level == 3) l3_probe::optimizer(result.optimization,pending);\n        const auto& values')
text=replace(text,'try {\n    for (int slot = 0; slot < pool.size(); ++slot) workers.emplace_back(worker, slot);',
    'if constexpr (Policy::level == 3) l3_probe::workers(true);\n  try {\n    for (int slot = 0; slot < pool.size(); ++slot) workers.emplace_back(worker, slot);')
text=replace(text,'for (auto& thread : workers) thread.join();\n  if (error)',
    'for (auto& thread : workers) thread.join();\n  if constexpr (Policy::level == 3) l3_probe::workers(false);\n  if (error)')
text=replace(text,'pool.after_work();','pool.after_work();\n  if constexpr (Policy::level == 3) l3_probe::checkpoint("optimizer_done");')
text=replace(text,'pool.close();\n  diagnostics->cuda_optimizer_host_ms',
    'if constexpr (Policy::level == 3) l3_probe::checkpoint("pool_closing");\n  pool.close();\n  if constexpr (Policy::level == 3) l3_probe::checkpoint("pool_closed");\n  diagnostics->cuda_optimizer_host_ms')
text=replace(text,'optimization.setups[setup_index];\n      const int target_count',
    'optimization.setups[setup_index];\n      if (work->batch.level == 3 && !work->coefficients_already_detached) l3_probe::optimizer(result,optimization.residuals[setup_index]);\n      const int target_count')
text=replace(text,'// store waits for the detach:',
    'if (work->batch.level == 3) l3_probe::stage_a(request.base_target_keys,offset,count,coefficients,residual);\n              // store waits for the detach:')
text=replace(text,'work->batch.batch_wall_ms = logical_boundary_ms;\n}\n\nvoid prefill_multi_penalty_target_states',
    'work->batch.batch_wall_ms = logical_boundary_ms;\n  if (work->batch.level == 3) l3_probe::checkpoint("stage_a_window_completed");\n}\n\nvoid prefill_multi_penalty_target_states')
start=text.index('void prefill_multi_penalty_level_target_states(');end=text.index('\nGroupResult execute_group(',start)
block=text[start:end]
block=replace(block,'std::map<std::string, std::vector<int>> tasks_by_conditioning;',
    'if (plan.level == 3) l3_probe::begin(diagnostics->l3_probe.enabled);\n  std::map<std::string, std::vector<int>> tasks_by_conditioning;')
block=replace(block,'if ((diagnostics->l4_slots.enabled && plan.level == 4)',
    'if ((diagnostics->l3_probe.enabled && plan.level == 3) ||\n      (diagnostics->l4_slots.enabled && plan.level == 4)')
block=replace(block,'if (plan.level == 4) {',
    'if (plan.level == 3) {\n      prefill_optimizer_slot_target_states<L3ProbePolicy>(data, plan, window, coefficient_arena, diagnostics);\n      l3_probe::finish();\n    } else if (plan.level == 4) {')
assert block.endswith('}\n')
block=block[:-2]+'  if (plan.level == 3) l3_probe::finish();\n}\n';text=text[:start]+block+text[end:]
text=replace(text,'diagnostics.l4_slots.enabled = level_component_flag',
    'diagnostics.l3_probe.enabled = level_component_flag("FASTKPC_DIAG_L3_REPLACEMENT");\n  diagnostics.l4_slots.enabled = level_component_flag')
text=replace(text,'require(diagnostics.l4_slots.created == diagnostics.l4_slots.destroyed &&',
    'require(diagnostics.l3_probe.created == diagnostics.l3_probe.destroyed && diagnostics.l3_probe.optimizer_binds == diagnostics.l3_probe.detached && diagnostics.l3_probe.stage_a_binds == diagnostics.l3_probe.detached, "L3 probe lifecycle");\n  require(diagnostics.l4_slots.created == diagnostics.l4_slots.destroyed &&')
a=text.index('      Rcpp::Named("l4_optimizer_slots")');b=text.index('      Rcpp::Named("l6_optimizer_slots")',a)
text=text[:a]+text[a:b].replace('l4_optimizer_slots','l3_probe_slots').replace('diagnostics.l4_slots','diagnostics.l3_probe')+text[a:]
(out/'one_call.cpp').write_text(text)
(out/'overlay.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='production/full_cuda_ci_one_call.cpp',tofile='diagnostic/one_call.cpp')))

runner=(TOOL.parent/'optimizer_l6_slot_runtime_20261026/run.R').read_text()
runner='dyn.load(Sys.getenv("FASTKPC_L3_PROBE_LIBRARY"))\n'+runner
runner=replace(runner,'GPU1 isolated incremental source-closure development; not full rebuild, clean producer or promotion',
    'GPU1 isolated L3 replacement coexistence diagnostic host overlay; not performance or production integration')
runner=replace(runner,'elapsed <- unname(proc.time()',
    'stopifnot(isTRUE(.Call("fastkpc_l3_probe_finish")))\nelapsed <- unname(proc.time()')
runner=replace(runner,'x$level!=6L','!(x$level %in% c(3L,6L))',2)
runner=replace(runner,'if(l6slot_mode=="1")5292L else 5537L',
    '(5292L - if(Sys.getenv("FASTKPC_DIAG_L3_REPLACEMENT")=="1")4065L else 0L)')
extra='''l3 <- subset(s$prefill_batches,level==3L)
l3pool <- s$l3_probe_slots
l3_on <- Sys.getenv("FASTKPC_DIAG_L3_REPLACEMENT")=="1"
g$l3_coverage <- sum(l3$optimizer_setup_count)==4065L && sum(l3$target_optimization_count)==54878L
g$l3_requested <- identical(l3pool$enabled,l3_on)
g$l3_shape <- if(l3_on) {
  nrow(l3)==1L && l3pool$created==32L && l3pool$destroyed==32L &&
    l3pool$optimizer_binds==4065L && l3pool$stage_a_binds==4065L && l3pool$detached==4065L &&
    l3pool$minimum_free_bytes>=2*1024^3 && l3pool$maximum_pool_increment_bytes<=1536*1024^2 &&
    l3$optimizer_requested_concurrency==32L && l3$optimizer_worker_count==32L
} else {
  nrow(l3)==64L && identical(l3$conditioning_group_count,c(rep(64L,63L),33L)) &&
    l3pool$created==0L && l3pool$destroyed==0L
}
'''
runner=replace(runner,'changed <- t[different,,drop=FALSE]',extra+'changed <- t[different,,drop=FALSE]')
runner=replace(runner,'receipt$gates<-g;',
    'receipt$diagnostic_overlay_sha256 <- sha(Sys.getenv("FASTKPC_L3_PROBE_LIBRARY")); receipt$l3_probe_slots <- l3pool\nreceipt$gates<-g;')
(out/'run.R').write_text(runner)

includes=['-I'+str(TOOL),'-I'+str(out),'-I'+str(ROOT/'fastkpc/src'),'-I'+str(ROOT/'fastkpc/src/cuda'),'-I/usr/local/cuda/include']
cuda_cmd=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-diag-suppress','2464',
    '-Xcompiler','-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden','-std=c++17',*includes,'-c',str(out/'rebind.cu'),'-o',str(out/'rebind.o')]
rcpp_includes=subprocess.check_output(['Rscript','--vanilla','-e','cat(vapply(c("Rcpp","RcppEigen","RcppArmadillo","RSpectra"),function(x)system.file("include",package=x),""),sep="\\n")'],text=True).splitlines()
cpp=['g++','-std=c++17',*shlex.split(subprocess.check_output(['R','CMD','config','CXX17FLAGS'],text=True)),'-fPIC','-pthread',*includes,
    '-I/usr/local/lib/R/include',*['-I'+p for p in rcpp_includes]]
commands=[cuda_cmd,*[cpp+['-c',str(src),'-o',str(out/(name+'.o'))] for name,src in
    [('one_call',out/'one_call.cpp'),('bridge',TOOL/'bridge.cpp'),('probe',TOOL/'probe.cpp')]]]
commands.append(['g++','-shared','-pthread',*[str(out/(n+'.o')) for n in ['one_call','bridge','probe','rebind']],
    str(captured/'audit.so'),str(NATIVE),'-Wl,-rpath,'+str(captured),'-Wl,-rpath,'+str(NATIVE.parent),
    '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-lcusolver','-ldl',
    '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRblas','-lRlapack','-o',str(out/'probe.so')])
for i,cmd in enumerate(commands):
    print('BUILD',i,Path(cmd[0]).name,flush=True)
    with (out/('build-'+str(i)+'.log')).open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
def sass(path):
    data={};name=None
    p=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],stdout=subprocess.PIPE,text=True)
    for line in p.stdout:
        if 'Function :' in line:
            name=re.sub(r'\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+_cu_[0-9a-f]{8}','4ANON',line.split('Function :')[1].strip())
            assert name not in data;data[name]=[]
        elif name:data[name].extend(re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',line))
    assert p.wait()==0 and len(data)==14 and all(data.values())
    return {k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in data.items()}
reference=PARENT/'build-source/fastkpc/build/mgcv_multi_penalty_gcv_small64.o'
original_code,overlay_code=sass(reference),sass(out/'rebind.o');assert original_code==overlay_code,'diagnostic CUDA code changed'
write_new(out/'device-code.json',dict(status='PASS',functions=14,reference=original_code,overlay=overlay_code))
check_identity(captured)
write_new(out/'build.json',dict(status='PASS',native_sha256=sha(NATIVE),source_sha256=identity['source_sha256'],
    diagnostic_overlay_sha256=sha(out/'probe.so'),source_class='diagnostic host overlay, production math entrypoints unchanged',
    protocol_sha256=sha(out/'protocol.json'),commands=commands,
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [captured/'prototype-build.json',captured/'audit.so',
      TOOL.parent/'optimizer_l6_slot_runtime_20261026/run.R',TOOL.parent/'optimizer_layer_refill_20261023/audit.hpp',
      TOOL.parent/'optimizer_l3_slots_20261027/fixture.hpp',TOOL.parent/'optimizer_windows_20261022/fixture.hpp']},
    outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
print('PASS overlay build',sha(out/'probe.so'),flush=True)
