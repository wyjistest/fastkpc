"""Build external companions against the archived production native, without installing."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
expected='8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874'
assert sha(native)==expected
closure=json.loads((root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022/closure.json').read_text())
assert all(sha(root/p)==s for p,s in closure['source_sha256'].items())
symbols=[r.split()[-1] for r in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
names=dict(plan='make_layer_plan',create='fastkpc::create_multi_penalty_gcv_capacity_prepared',
 multi='fastkpc::multi_penalty_gcv_capacity_optimize_multi',
 batch='fastkpc::multi_penalty_gcv_cuda_optimize_batch',
 small='fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64')
lines=[]
for alias,name in names.items():
    matches=[s for s,d in zip(symbols,demangled) if d.startswith(name+'(')]
    assert len(matches)==1,(alias,matches)
    lines.append('constexpr const char* SYM_'+alias+' = "'+matches[0]+'";')
(out/'symbols.inc').write_text('\n'.join(lines)+'\n')
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
common=['g++','-O2','-std=c++17','-fPIC','-pthread','-I'+str(out),'-I'+str(here),
 '-I'+str(root/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib']
mode=sys.argv[2] if len(sys.argv)>2 else 'capture'
sources=[here/'fixture.hpp',here/'build.py'];commands=[]
if mode=='capture':
    sources.append(here/'capture.cpp')
    commands.append(common+['-shared',str(here/'capture.cpp'),'-lR','-lcudart','-ldl',
      str(native),'-Wl,-rpath,'+str(native.parent),'-o',str(out/'capture.so')])
elif mode=='replay':
    sources.extend([here/'observe.cpp',here/'observe.hpp',here/'replay.cpp'])
    commands.append(common+['-shared',str(here/'observe.cpp'),'-lcudart','-lnvToolsExt','-ldl',
      str(native),'-Wl,-rpath,'+str(native.parent),'-o',str(out/'observe.so')])
    commands.append(common+[str(here/'replay.cpp'),str(out/'observe.so'),str(native),
      '-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(native.parent),'-lR','-lcudart','-lnvToolsExt',
      '-ldl','-o',str(out/'replay')])
else: raise ValueError(mode)
for c in commands:
    print(' '.join(c),flush=True);subprocess.run(c,check=True)
assert sha(native)==expected
(out/(mode+'-build.json')).write_text(json.dumps(dict(native_sha256=expected,commands=commands,
 sources={str(p.relative_to(root)):sha(p) for p in sources},source_closure=closure['source_sha256'],
 outputs={p.name:sha(p) for p in out.iterdir() if p.suffix=='.so' or p.name=='replay'}),indent=2)+'\n')
