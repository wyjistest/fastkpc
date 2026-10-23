"""Build research interposer/replay without touching the accepted native."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path.cwd(); here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve(); out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)=='fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50'
source=root/'fastkpc/src/cuda/full_cuda_ci_vertical.cu'
text=source.read_text()
start=text.index('    check_cuda(cudaEventRecord(execution_context->pair_start(), stream)')
last='               "record legacy eig pair end");'
end=text.index(last,start)+len(last)
stage=text[start:end]
assert stage.count('cublasDgemmBatched')==1 and stage.count('evaluate_legacy_eig_pair_batch_kernel')==1
(out/'pair_stage.inc').write_text(stage+'\n')
commands=[['g++','-O2','-std=c++17','-fPIC','-shared',str(here/'capture.cpp'),
  '-I'+str(root/'fastkpc/src/cuda'),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64',
  '-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl','-o',str(out/'capture.so')],
 ['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','-std=c++17','-Xcompiler=-fPIC','-shared',
  str(here/'probe.cu'),'-I'+str(root/'fastkpc/src/cuda'),'-I'+str(out),'-Xlinker=-Bsymbolic',
  str(native),'-Xlinker=-rpath='+str(native.parent),'-lcublas','-lcusolver','-o',str(out/'probe.so')]]
for command in commands:
    print(' '.join(command),flush=True)
    subprocess.run(command,check=True)
sources=[here/'capture.cpp',here/'probe.cu',here/'build.py',source,
         root/'fastkpc/src/cuda/full_cuda_ci_vertical.hpp',root/'fastkpc/src/full_cuda_ci_contract.hpp']
report=dict(schema='singleton-pvalue-build-v1',native_sha256=sha(native),commands=commands,
 sources={str(p.relative_to(root)):sha(p) for p in sources},
 artifacts={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.so','.inc')})
(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
