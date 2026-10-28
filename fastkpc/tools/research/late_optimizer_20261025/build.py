"""Build only the research host interposer; preserve the installed native."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
tool=Path(__file__).resolve().parent;root=tool.parents[3]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=root/'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025'
build=json.loads((parent/'build.json').read_text());native=root/'fastkpc/build/fastkpc_cuda.so'
assert not (out/'observe-build.json').exists()
assert sha(native)==build['native_sha256']=='994c43d386c859f23c34509b857db22064fb9a009b8fe871c719dd608f820df9'
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
symbols=[r.split()[-1] for r in subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
aliases=dict(plan='make_layer_plan',create='fastkpc::create_multi_penalty_gcv_capacity_prepared',multi='fastkpc::multi_penalty_gcv_capacity_optimize_multi')
lines=[]
for alias,name in aliases.items():
    matches=[s for s,d in zip(symbols,demangled) if d.startswith(name+'(')]
    assert len(matches)==1,(alias,matches)
    lines.append('constexpr const char* SYM_'+alias+' = "'+matches[0]+'";')
(out/'symbols.inc').write_text('\n'.join(lines)+'\n')
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
command=['g++','-O2','-std=c++17','-fPIC','-pthread','-shared','-I'+str(out),'-I'+str(tool),
 '-I'+str(root/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 str(tool/'observe.cpp'),str(native),'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-Wl,-rpath,'+str(native.parent),
 '-lR','-lcudart','-ldl','-o',str(out/'observe.so')]
subprocess.run(command,check=True)
for name in ['build.json','checks.json','stats.so']:shutil.copyfile(parent/name,out/name)
inputs=[tool/'build.py',tool/'observe.cpp',tool/'fixture.hpp',tool/'README.md',
        tool.parent/'optimizer_windows_20261022/fixture.hpp']
record=dict(status='PASS',native_sha256=sha(native),source_sha256=build['source_sha256'],
 parent_build=str((parent/'build.json').relative_to(root)),parent_build_sha256=sha(parent/'build.json'),
 command=command,inputs={str(p.relative_to(root)):sha(p) for p in inputs},
 outputs={p:sha(out/p) for p in ['observe.so','symbols.inc','build.json','checks.json','stats.so']},
 production_native_unchanged=True,gpu_code_added=False)
assert record['native_sha256']==build['native_sha256']
with (out/'observe-build.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print('PASS research interposer build; native unchanged',flush=True)
