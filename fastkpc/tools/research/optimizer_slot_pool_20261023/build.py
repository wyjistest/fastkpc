"""Build a hidden small64 research DSO; never rebuild/replace production native."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[4];tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023'
manifest=json.loads((prior/'manifest.json').read_text())
native=root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)==manifest['native_sha256']
assert all(sha(root/p)==h for p,h in manifest['source_sha256'].items())
commands=[]
includes=['-I'+str(root/'fastkpc/src'),'-I'+str(tool),'-I/usr/local/cuda/include']
commands.append(['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-diag-suppress','2464',
 '-Xcompiler','-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden','-std=c++17',*includes,
 '-c',str(tool/'slot_backend.cu'),'-o',str(out/'slot_backend.o')])
commands.append(['g++','-shared','-pthread',str(out/'slot_backend.o'),str(native),
 '-Wl,-rpath,'+str(native.parent),'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-lcudart','-lcublas','-o',str(out/'slot_backend.so')])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*includes,
 str(tool.parent/'optimizer_layer_refill_20261023/audit.cpp'),'-lcudart','-lcublas','-ldl',
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-o',str(out/'audit.so')])
commands.append(['g++','-O2','-std=c++17','-pthread',*includes,str(tool/'replay.cpp'),
 str(out/'audit.so'),str(out/'slot_backend.so'),str(native),'-Wl,-rpath,'+str(out),
 '-Wl,-rpath,'+str(native.parent),'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-lcudart','-ldl','-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib',
 '-lR','-lRlapack','-lRblas','-o',str(out/'replay')])
link_only='--link-only' in sys.argv[2:]
cuda_inputs=[tool/'slot_backend.cu',tool/'slot_api.hpp',tool.parent/'optimizer_windows_20261022/fixture.hpp']
cuda_inputs += [root/p for p in manifest['source_sha256']]
if link_only:
    # Developer iteration after a host-only compile error. Numeric instructions
    # are still checked below; no changed CUDA input may postdate the object.
    assert (out/'slot_backend.o').stat().st_mtime_ns>=max(p.stat().st_mtime_ns for p in cuda_inputs)
for cmd in commands[1 if link_only else 0:]:
    print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
exports=subprocess.check_output(['nm','-D','--defined-only',str(out/'slot_backend.so')],text=True)
assert 'create_multi_penalty_gcv_cuda_prepared_small64' not in exports,'production entrypoint leaked'
def sass(path):
    result={};name=None
    proc=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],stdout=subprocess.PIPE,text=True)
    for line in proc.stdout:
        if 'Function :' in line:
            name=line.split('Function :')[1].strip()
            name=re.sub(r'\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+_cu_[0-9a-f]{8}','4ANON',name)
            assert name not in result;result[name]=[]
        elif name:
            result[name].extend(re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',line))
    assert proc.wait()==0 and result and all(result.values())
    return {n:dict(words=len(x),sha256=hashlib.sha256('\n'.join(x).encode()).hexdigest()) for n,x in result.items()}
reference=root/'fastkpc/artifacts/full_cuda_ci/optimizer_status_wait_20261023/build-source/fastkpc/build/mgcv_multi_penalty_gcv_small64.o'
original=sass(reference);prototype=sass(out/'slot_backend.o')
device=dict(status='PASS' if original==prototype else 'FAIL',original=original,prototype=prototype,
 reference_object_sha256=sha(reference),prototype_object_sha256=sha(out/'slot_backend.o'))
(out/'device-code.json').write_text(json.dumps(device,indent=2)+'\n')
assert original==prototype,'GPU instruction mismatch'
sources=[p for p in tool.iterdir() if p.suffix in ['.cu','.cpp','.hpp','.py']]
sources += [tool.parent/'optimizer_windows_20261022/fixture.hpp',
 tool.parent/'optimizer_layer_refill_20261023/layer_fixture.hpp',
 tool.parent/'optimizer_layer_refill_20261023/audit.cpp',tool.parent/'optimizer_layer_refill_20261023/audit.hpp']
(out/'build.json').write_text(json.dumps(dict(native_sha256=sha(native),source_sha256=manifest['source_sha256'],
 commands=commands,link_only=link_only,cuda_input_sha256={str(p.relative_to(root)):sha(p) for p in cuda_inputs},
 sources={str(p.relative_to(root)):sha(p) for p in sources},
 outputs={p:sha(out/p) for p in ['slot_backend.o','slot_backend.so','audit.so','replay']},
 fixture_manifest_sha256=sha(prior/'manifest.json'),device_code='PASS'),indent=2)+'\n')
print('PASS isolated build and device instructions',len(original),'functions',flush=True)
