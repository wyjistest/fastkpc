"""External L5 companions; link the qualified status-wait native unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
mode=sys.argv[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
closure_path=root/'fastkpc/artifacts/full_cuda_ci/optimizer_status_wait_20261023/closure.json'
closure=json.loads(closure_path.read_text())
assert closure['status']=='PASS_SHARED_HOST_ORIGINAL_WINDOW_OPT_IN'
assert sha(native)==closure['native_sha256']
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
old=here.parent/'optimizer_windows_20261022'
common=['g++','-O2','-std=c++17','-fPIC','-pthread','-I'+str(out),'-I'+str(here),'-I'+str(old),
 '-I'+str(root/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib']
sources=[here/'layer_fixture.hpp',old/'fixture.hpp',here/'build.py'];commands=[]
if mode=='capture':
    sources.append(here/'capture.cpp')
    commands.append(common+['-shared',str(here/'capture.cpp'),'-lR','-lcudart','-ldl',
      str(native),'-Wl,-rpath,'+str(native.parent),'-o',str(out/'capture.so')])
elif mode=='replay':
    sources.extend([old/'observe.cpp',old/'observe.hpp',here/'replay.cpp',here/'audit.cpp',here/'audit.hpp'])
    commands.append(common+['-shared',str(old/'observe.cpp'),str(here/'audit.cpp'),
      '-lcudart','-lcublas','-lnvToolsExt','-ldl',str(native),'-Wl,-rpath,'+str(native.parent),
      '-o',str(out/'observe.so')])
    commands.append(common+[str(here/'replay.cpp'),str(out/'observe.so'),str(native),
      '-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(native.parent),'-lR','-lcudart','-lnvToolsExt',
      '-ldl','-o',str(out/'replay')])
else:raise ValueError(mode)
for command in commands:
    print(' '.join(command),flush=True);subprocess.run(command,check=True)
assert sha(native)==closure['native_sha256']
(out/(mode+'-build.json')).write_text(json.dumps(dict(native_sha256=sha(native),
 commands=commands,sources={str(p.relative_to(root)):sha(p) for p in sources},
 source_closure=closure['source_sha256'],closure_sha256=sha(closure_path),
 outputs={p.name:sha(p) for p in out.iterdir() if p.suffix=='.so' or p.name=='replay'}),indent=2)+'\n')
