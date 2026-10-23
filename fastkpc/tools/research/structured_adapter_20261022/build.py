"""Build isolated device adapter, sharing original CUDA consumer statements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path.cwd(); here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve(); out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
expected='b1e7eab4297a13b7ff29f0bf40b2fc3ce4d6950542dff4318ee3e2dc76694ee9'
assert sha(native)==expected
source=root/'fastkpc/src/cuda/full_cuda_ci_vertical.cu'; text=source.read_text()
start=text.index('  double* output = centered_vectors +',text.index('__global__ void select_center_legacy'))
end=text.index('\n}\n',start)
(out/'center_stage.inc').write_text(text[start:end]+'\n')
start=text.index('    check_cuda(cudaEventRecord(execution_context->pair_start(), stream)')
last='               "record legacy eig pair end");'
stage=text[start:text.index(last,start)+len(last)]
assert stage.count('cublasDgemmBatched')==1 and stage.count('evaluate_legacy_eig_pair_batch_kernel')==1
(out/'pair_stage.inc').write_text(stage+'\n')
files=list(here.glob('*.*'))+[source,root/'fastkpc/src/cuda/full_cuda_ci_vertical.hpp',
 root/'fastkpc/src/full_cuda_ci_contract.hpp',
 here.parent/'exact_speedup_20261020/structured_eig_gpu.cu',
 here.parent/'exact_speedup_20261020/generate_recorded_dbdsqr.py']
files+=list((root/'fastkpc/src/cuda').glob('lapack_312*'))
files+=list((root/'fastkpc/src/cuda/third_party').glob('lapack_3_12*device.inc'))
before={str(p.relative_to(root)):sha(p) for p in files if p.is_file()}
nvcc='/usr/local/cuda/bin/nvcc'
common=[nvcc,'-O3','-arch=sm_89','-std=c++17','-Xcompiler=-fPIC',
 '-I'+str(out),'-I'+str(root/'fastkpc/src/cuda')]
commands=[
 [sys.executable,'-I',str(here.parent/'exact_speedup_20261020/generate_recorded_dbdsqr.py'),
  str(out/'recorded_dbdsqr.inc')],
 common+['--fmad=false','-diag-suppress=2464','-c',str(here/'structured_device.cu'),'-o',str(out/'structured.o')],
 common+['-shared',str(here/'adapter.cu'),str(out/'structured.o'),'-Xlinker=-Bsymbolic',
  str(native),'-Xlinker=-rpath='+str(native.parent),'-lcublas','-lcusolver',
  '-o',str(out/'adapter.so')]]
for cmd in commands:
    print(' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
assert sha(native)==expected and all(sha(root/p)==h for p,h in before.items())
(out/'build.json').write_text(json.dumps(dict(schema='structured-device-adapter-build-v1',
 native_sha256=expected,sources=before,commands=commands,
 nvcc=subprocess.check_output([nvcc,'--version'],text=True),
 artifacts={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.so','.inc','.o')}),indent=2)+'\n')
