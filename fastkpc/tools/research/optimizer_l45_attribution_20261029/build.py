"""Build host-only observer and CUPTI tracer against unchanged production native."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
identity=json.loads((ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_recover_tile_20261029/attempt3/manifest.json').read_text())
native=ROOT/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)==identity['native_sha256']
assert all(sha(ROOT/p)==h for p,h in identity['source_sha256'].items())
captures={}
for name,count in [('optimizer_l4_diagnostic_20261023',34),('optimizer_layer_refill_20261023',15)]:
    base=ROOT/'fastkpc/artifacts/full_cuda_ci'/name
    manifest=json.loads((base/'manifest.json').read_text())
    files={p:h for p,h in manifest['files'].items() if p.startswith('raw/') and p.endswith('.bin')}
    assert len(files)==count
    for p,h in files.items():assert sha(base/p)==h;captures[str(base/p)]=h
    captures[str(base/'manifest.json')]=sha(base/'manifest.json')

source=(ROOT/'fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu').read_text()
parts=[]
for name in ['DeviceTargetEvaluation','DevicePhaseTiming','DeviceOptimizerState']:
    found=re.findall(r'struct '+name+r' \{.*?\n\};',source,re.S)
    assert len(found)==1
    parts.append(found[0])
(OUT/'state_layout.hpp').write_text('#pragma once\nnamespace state_layout {\n'
    'constexpr int kMultiPenaltyGcvMaximumCoefficientDim=64;\n'
    'constexpr int kMultiPenaltyGcvMaximumPenaltyCount=7;\n'+'\n'.join(parts)+'\n}\n')
symbols=[s.split()[-1] for s in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
exports={}
for macro,name in [('NATIVE_BATCH_SYMBOL','multi_penalty_gcv_cuda_optimize_batch_small64'),
                   ('NATIVE_BIND4_SYMBOL','rebind_multi_penalty_gcv_cuda_prepared_l4'),
                   ('NATIVE_BIND5_SYMBOL','rebind_multi_penalty_gcv_cuda_prepared_l5')]:
    chosen=[s for s,d in zip(symbols,demangled) if d.startswith('fastkpc::'+name+'(')]
    assert len(chosen)==1
    exports[macro]=chosen[0]
(OUT/'symbols.inc').write_text(''.join('#define '+m+' "'+s+'"\n' for m,s in exports.items()))
cuda=Path('/usr/local/cuda-12.5/targets/x86_64-linux')
inc=['-I'+str(ROOT/'fastkpc/src'),'-I'+str(cuda/'include'),'-I'+str(OUT)]
libs=['-L'+str(cuda/'lib'),'-Wl,-rpath,'+str(cuda/'lib'),'-lcudart','-lcupti','-ldl']
commands=[
 ['g++','-O2','-std=c++17',*inc,str(TOOL/'fixture_index.cpp'),'-o',str(OUT/'fixture_index')],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,str(TOOL/'observer.cpp'),*libs,'-o',str(OUT/'observer.so')],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,str(TOOL/'trace.cpp'),*libs,'-o',str(OUT/'trace.so')],
 ['g++','-O2','-std=c++17',*inc,str(TOOL/'smoke.cpp'),str(native),*libs,
  '-Wl,-rpath,'+str(native.parent),'-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib',
  '-lR','-lRlapack','-lRblas','-o',str(OUT/'smoke-check')]]
for cmd in commands:subprocess.run(cmd,check=True,cwd=ROOT)
subprocess.run([str(OUT/'fixture_index'),str(ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023/raw'),
    str(ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023/raw'),str(OUT/'reference')],check=True,cwd=ROOT)
# Reuse the exact hard-gate runner, changing only evidence labels and own path.
runner_path=ROOT/'fastkpc/tools/research/stage_b_work_pair_runtime_20261028/run.R'
runner=runner_path.read_text()
begin=runner.index('evidence_class <- ');end=runner.index('\nSys.setenv(',begin)
runner=runner[:begin]+'evidence_class <- "unchanged-native CUPTI/host-metadata attribution; not performance or promotion"'+runner[end:]
runner=runner.replace('schema="structured-work-pair-runtime-canonical-v1"','schema="l45-optimizer-attribution-canonical-v1"')
runner=runner.replace('sha("fastkpc/tools/research/stage_b_work_pair_runtime_20261028/run.R")',
                      'sha('+json.dumps(str(OUT/'run.R'))+')')
runner += '''\nwriteLines("PASS full canonical hard gates; request-array order checked by observer at exit",file.path(out,"identity-order.txt"))
'''
(OUT/'run.R').write_text(runner)
write(OUT/'build.json',dict(status='PASS',native_sha256=identity['native_sha256'],
    source_sha256=identity['source_sha256'],capture_sha256=captures,commands=commands,
    production_changes=False,device_compilation=False,runner_source_sha256=sha(runner_path),
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
print('PASS host-only diagnostic build',flush=True)
