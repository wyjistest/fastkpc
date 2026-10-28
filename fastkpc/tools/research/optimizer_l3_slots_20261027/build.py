"""Isolated L3 scheduler; preserve production native and compare device instructions."""
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from common import ROOT,TOOL,NATIVE,PARENT,sha,load,write_new,source_identity

out=Path(sys.argv[1]).resolve();identity=source_identity()
includes=['-I'+str(ROOT/'fastkpc/src'),'-I'+str(TOOL),'-I'+str(out),'-I/usr/local/cuda/include']
cuda_inputs=[TOOL/'slot_backend.cu',TOOL/'slot_api.hpp',TOOL.parent/'optimizer_windows_20261022/fixture.hpp']
cuda_inputs += [ROOT/p for p in identity['source_sha256']]
cuda_hashes={str(p.relative_to(ROOT)):sha(p) for p in cuda_inputs}
cuda_cmd=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-diag-suppress','2464',
    '-Xcompiler','-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden','-std=c++17',*includes,
    '-c',str(TOOL/'slot_backend.cu'),'-o',str(out/'slot_backend.o')]
if '--link-only' in sys.argv[2:]:
    b=load(out/'cuda-build.json')
    assert b['inputs']==cuda_hashes and b['object_sha256']==sha(out/'slot_backend.o')
else:
    assert not (out/'cuda-build.json').exists(),'preserve CUDA build'
    print('Compiling isolated L3 CUDA object',flush=True)
    subprocess.run(cuda_cmd,check=True)
    write_new(out/'cuda-build.json',dict(inputs=cuda_hashes,command=cuda_cmd,object_sha256=sha(out/'slot_backend.o')))
if '--cuda-only' in sys.argv[2:]:sys.exit(0)
assert not (out/'prototype-build.json').exists(),'preserve prototype build'
links=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl']
rpath=['-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(NATIVE.parent)]
audit=TOOL.parent/'optimizer_layer_refill_20261023'
ktrace=TOOL.parent/'optimizer_slot_line_review_20261025/ktrace.cpp'
commands=[
    ['g++','-shared','-pthread',str(out/'slot_backend.o'),str(NATIVE),*rpath,*links,'-o',str(out/'slot_backend.so')],
    ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*includes,str(audit/'audit.cpp'),*links,'-o',str(out/'audit.so')],
    ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*includes,str(TOOL/'native_observe.cpp'),
        str(out/'slot_backend.so'),*rpath,*links,'-o',str(out/'native_observe.so')],
    ['g++','-O2','-std=c++17','-pthread',*includes,str(TOOL/'replay.cpp'),str(out/'audit.so'),
        str(out/'native_observe.so'),str(out/'slot_backend.so'),str(NATIVE),*rpath,*links,
        '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRlapack','-lRblas','-o',str(out/'replay')],
    ['g++','-O2','-std=c++17','-shared','-fPIC','-pthread','-I/usr/local/cuda/include',
        str(ktrace),'-L/usr/local/cuda/lib64',
        '-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-o',str(out/'ktrace.so')]]
for cmd in commands:subprocess.run(cmd,check=True)
exports=subprocess.check_output(['nm','-D','--defined-only',str(out/'slot_backend.so')],text=True)
assert 'create_multi_penalty_gcv_cuda_prepared_small64' not in exports
assert 'multi_penalty_gcv_cuda_optimize_batch_small64' not in exports

def sass(path):
    result={};name=None
    proc=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],stdout=subprocess.PIPE,text=True)
    for line in proc.stdout:
        if 'Function :' in line:
            name=line.split('Function :')[1].strip()
            name=re.sub(r'\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+_cu_[0-9a-f]{8}','4ANON',name)
            assert name not in result;result[name]=[]
        elif name:result[name].extend(re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',line))
    assert proc.wait()==0 and len(result)==14 and all(result.values())
    return {n:dict(words=len(x),sha256=hashlib.sha256('\n'.join(x).encode()).hexdigest()) for n,x in result.items()}

reference=PARENT/'build-source/fastkpc/build/mgcv_multi_penalty_gcv_small64.o'
assert sha(reference)==identity['objects']['mgcv_multi_penalty_gcv_small64.o']
original,prototype=sass(reference),sass(out/'slot_backend.o')
write_new(out/'device-code.json',dict(status='PASS' if original==prototype else 'FAIL',original=original,
    prototype=prototype,reference_object_sha256=sha(reference),prototype_object_sha256=sha(out/'slot_backend.o')))
assert original==prototype,'GPU instructions changed'
source_identity()
deps=[PARENT/'build.json',audit/'audit.cpp',audit/'audit.hpp',ktrace,TOOL.parent/'optimizer_windows_20261022/fixture.hpp']
write_new(out/'prototype-build.json',dict(native_sha256=sha(NATIVE),producer_head=identity['producer_head'],
    source_sha256=identity['source_sha256'],commands=[cuda_cmd,*commands],device_code='PASS',
    evidence_class='source-closure captured prototype; no production integration',
    execution_tools={str(p.relative_to(ROOT)):sha(p) for p in TOOL.iterdir() if p.is_file()},
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in deps},
    outputs={p:sha(out/p) for p in ['slot_backend.o','slot_backend.so','native_observe.so','audit.so',
        'replay','ktrace.so','symbols.inc','device-code.json']}))
print('PASS isolated build: all 14 GPU functions match production',flush=True)
