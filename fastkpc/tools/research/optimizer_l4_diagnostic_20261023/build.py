"""Build external L4 companions against the unchanged qualified runtime."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4];here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True);mode=sys.argv[2]
assert mode in ['capture','replay'] and not (out/(mode+'-build.json')).exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_slot_runtime_20261023'
closure=json.loads((prior/'closure.json').read_text());manifest=json.loads((prior/'manifest.json').read_text())
assert closure['status']=='GO_L5_OPT_IN_RUNTIME_DEVELOPMENT'
assert sha(native)==closure['native_sha256']==manifest['native_sha256']
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
symbols=[r.split()[-1] for r in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
names=dict(plan='make_layer_plan',create='fastkpc::create_multi_penalty_gcv_capacity_prepared',
 multi='fastkpc::multi_penalty_gcv_capacity_optimize_multi')
lines=[]
for alias,name in names.items():
 matches=[s for s,d in zip(symbols,demangled) if d.startswith(name+'(')]
 assert len(matches)==1,(alias,matches)
 lines.append('constexpr const char* SYM_'+alias+' = "'+matches[0]+'";')
(out/'symbols.inc').write_text('\n'.join(lines)+'\n')
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
old=here.parent/'optimizer_layer_refill_20261023'
common=['g++','-O2','-std=c++17','-fPIC','-pthread','-I'+str(out),'-I'+str(here),'-I'+str(old),
 '-I'+str(root/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib']
sources=[here/'fixture.hpp',here.parent/'optimizer_windows_20261022/fixture.hpp',here/'build.py']
commands=[];outputs=[]
if mode=='capture':
 sources.extend([here/'capture.cpp',here/'capture.R']);outputs=['capture.so']
 commands.append(common+['-shared',str(here/'capture.cpp'),'-lR','-lcudart','-ldl',
  str(native),'-Wl,-rpath,'+str(native.parent),'-o',str(out/'capture.so')])
else:
 sources.extend([here/'replay.cpp',old/'audit.cpp',old/'audit.hpp']);outputs=['audit.so','replay']
 commands.append(common+['-shared',str(old/'audit.cpp'),'-lcudart','-lcublas','-ldl','-o',str(out/'audit.so')])
 commands.append(common+[str(here/'replay.cpp'),str(out/'audit.so'),str(native),
  '-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(native.parent),'-lR','-lcudart','-ldl','-o',str(out/'replay')])
for command in commands:subprocess.run(command,check=True)
assert sha(native)==closure['native_sha256']
(out/(mode+'-build.json')).write_text(json.dumps(dict(native_sha256=sha(native),commands=commands,
 sources={str(p.relative_to(root)):sha(p) for p in sources},source_closure=manifest['source_sha256'],
 prior_manifest_sha256=sha(prior/'manifest.json'),outputs={p:sha(out/p) for p in outputs}),indent=2)+'\n')
print('PASS build',mode,flush=True)
