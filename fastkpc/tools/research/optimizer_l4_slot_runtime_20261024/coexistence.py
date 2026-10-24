"""Actual production cache coexistence, with unchanged native and a conservative extra pool."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
os.chdir(ROOT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream: json.dump(value,stream,indent=2);stream.write('\n')
native=ROOT/'fastkpc/build/fastkpc_cuda.so'
attestation=json.loads((ROOT/'fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261024.json').read_text())
assert sha(native)==attestation['native_sha256']
assert all(sha(ROOT/p)==h for p,h in attestation['source_sha256'].items())
prior=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_runtime_20261023'
assert sha(native)==json.loads((prior/'build.json').read_text())['native_sha256']
fixture=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023/raw/L4-W1.bin'
manifest=json.loads((fixture.parent.parent/'manifest.json').read_text())
assert sha(fixture)==manifest['files']['raw/L4-W1.bin']
fixtures={p:h for p,h in manifest['files'].items() if p.startswith('raw/')}
assert len(fixtures)==34 and all(sha(fixture.parent.parent/p)==h for p,h in fixtures.items())
symbols=[s.split()[-1] for s in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
names=dict(plan='make_layer_plan',multi='fastkpc::multi_penalty_gcv_capacity_optimize_multi',
    store='fastkpc::store_full_cuda_ci_level_component_residual',build='fastkpc::build_full_cuda_ci_level_components')
lines=[]
for alias,name in names.items():
    matches=[s for s,d in zip(symbols,demangled) if d.startswith(name+'(')]
    assert len(matches)==1,(alias,matches)
    lines.append('constexpr const char* SYM_'+alias+' = "'+matches[0]+'";')
symbol_text='\n'.join(lines)+'\n'
if (out/'symbols.inc').exists():
    assert (out/'symbols.inc').read_text()==symbol_text and not (out/'build.json').exists()
else:
    with (out/'symbols.inc').open('x') as stream:stream.write(symbol_text)
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
cmd=['g++','-O2','-std=c++17','-fPIC','-shared','-pthread','-I'+str(out),
    '-I'+str(ROOT/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
    str(TOOL/'coexistence.cpp'),str(native),'-Wl,-rpath,'+str(native.parent),
    '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl',
    '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(out/'coexistence.so')]
subprocess.run(cmd,check=True)
runner=ROOT/'fastkpc/tools/research/optimizer_l4_runtime_20261023/run.R'
dependencies=[runner,TOOL/'coexistence.cpp',TOOL/'coexistence.py',TOOL/'README.md',
    TOOL.parent/'optimizer_l4_diagnostic_20261023/fixture.hpp',
    TOOL.parent/'optimizer_windows_20261022/fixture.hpp',
    ROOT/'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh']
build=dict(status='PASS',native_sha256=sha(native),source_sha256=attestation['source_sha256'],
    probe_sha256=sha(out/'coexistence.so'),command=cmd,fixtures=fixtures,
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in dependencies})
write(out/'build.json',build)
write(out/'protocol.json',dict(schema='l4-real-cache-coexistence-v1',declaration_epoch=time.time(),
    minimum_free_bytes=2*1024**3,maximum_warm_increment_bytes=1024**3,
    retained_extra_slots=32,target_capacity=18,extra_coefficient_bytes=37*17767*8,
    original_window_handles_retained=True,interval='first L4 optimizer through last Stage A store',
    baseline='complete canonical S2 + L5 slots + L4 workers32',performance_evidence=False))
os.sched_setaffinity(0,range(20))
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy()
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',
    FASTKPC_WINDOW_NATIVE=str(native),FASTKPC_L4_PROBE_FIXTURES=str(fixture.parent),
    FASTKPC_L4_PROBE_OUT=str(out),LD_PRELOAD=str(out/'coexistence.so'),
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(runner),
    str(out/'canonical'),'1',str(prior/'on1/receipt.rds'),'-']
start=time.time();samples=[];unexpected=set()
print('START actual cache/arena coexistence',flush=True)
with (out/'run.log').open('x') as log, (out/'gpu.csv').open('x') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
        '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
    try:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            active={int(row.split(',')[0]):row.split(',')[1].strip() for row in raw.splitlines() if row.strip()}
            samples.append(dict(epoch=time.time(),processes=active));unexpected.update(set(active)-{proc.pid})
            if unexpected:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            try:proc.wait(timeout=5);break
            except subprocess.TimeoutExpired:pass
    finally:monitor.terminate();monitor.wait(timeout=10)
write(out/'process.json',dict(command=command,exit_code=proc.returncode,start_epoch=start,end_epoch=time.time(),
    physical_gpu=1,affinity=sorted(os.sched_getaffinity(0)),unexpected_gpu_pids=sorted(unexpected)))
write(out/'gpu-processes.json',samples)
assert proc.returncode==0 and not unexpected
assert sha(native)==build['native_sha256']
assert all(sha(ROOT/p)==h for p,h in {**build['source_sha256'],**build['dependencies']}.items())
report=json.loads((out/'canonical/report.json').read_text())
assert report['gates'] and all(report['gates'].values())
capacity=json.loads((out/'coexistence-summary.json').read_text())
assert capacity['status']=='PASS', 'conservative coexistence failed; preserve baseline and diagnose replacement model'
write(out/'closure.json',dict(status='PASS_REAL_CACHE_ARENA_CONSERVATIVE_COEXISTENCE',
    native_sha256=sha(native),capacity=capacity,canonical_hard_gates=report['gates'],
    oracle_errors=report['oracle_errors'],evidence_class='diagnostic; no performance claim',
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
print('PASS',capacity,flush=True)
