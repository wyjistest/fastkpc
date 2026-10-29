"""Relink a research-only native from 42 authenticated objects and one CUDA TU."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT, TOOL, sha, write

out=Path(sys.argv[1]).resolve();variant=sys.argv[2]
assert variant in ('candidate','reference-profile','candidate-profile')
assert json.loads((out/'unit-closure.json').read_text())['status']=='PASS_GPU_VECTOR_AND_BLOCKED_POSTBACK'
identity=json.loads((out/'source.json').read_text())
parent=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/build.json'
old=json.loads(parent.read_text())
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==old['native_sha256']==identity['native_sha256']
for p,h in identity['production_source_sha256'].items():assert sha(ROOT/p)==h,p
for name,h in old['objects'].items():assert sha(ROOT/'fastkpc/build'/name)==h,name
dest=out/variant;dest.mkdir(exist_ok=False)
profile=variant.endswith('-profile');slices=not variant.startswith('reference')
source=dest/'backend.cu'
source.write_text('#include "'+str(out/'source/cuda/mgcv_multi_penalty_gcv_small64.cu')+'"\n'+'''
extern "C" int fastkpc_postback_profile_read(unsigned long long* values, int count) {
  if (count != 30) return -1;
#if POSTBACK_PROFILE
  return static_cast<int>(cudaMemcpyFromSymbol(values,
    fastkpc::lapack312_multi_small64::postback_stats, 30 * sizeof(unsigned long long)));
#else
  for (int i = 0; i < 30; ++i) values[i] = 0;
  return 0;
#endif
}
''')
cuda=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-diag-suppress','2464',
    '-Xcompiler','-fPIC','-std=c++17','-Xptxas=-v',
    '-DPOSTBACK_SLICES='+str(int(slices)),'-DPOSTBACK_PROFILE='+str(int(profile)),
    '-I'+str(out/'source'),'-I'+str(out/'source/cuda'),'-I'+str(out/'generated'),
    '-c',str(source),'-o',str(dest/'small64.o')]
objects=[dest/'small64.o' if name=='mgcv_multi_penalty_gcv_small64.o' else ROOT/'fastkpc/build'/name
         for name in old['objects']]
libs=[]
for name in ['LAPACK_LIBS','BLAS_LIBS','FLIBS']:
    libs+=shlex.split(subprocess.check_output(['R','CMD','config',name],text=True))
link=['g++','-shared','-pthread','-o',str(dest/'fastkpc_cuda.so'),*map(str,objects),*libs,'-lcrypto',
    '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-lcusolver',
    '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR']
write(dest/'build-protocol.json',dict(variant=variant,commands=[cuda,link],parent_sha256=sha(parent),
    original_native_sha256=old['native_sha256'],original_objects=old['objects'],
    source_overlay_sha256={str(p.relative_to(out/'source')):sha(p)
        for p in (out/'source').rglob('*') if p.is_file()},
    generated_sha256=sha(out/'generated/recorded_dbdsqr.inc'),
    tools={str(p.relative_to(ROOT)):sha(p) for p in [TOOL/'build_backend.py',TOOL/'candidate.inc']},
    production_modified=False))
start=time.time()
with (dest/'build.log').open('x') as log:
    for command in [cuda,link]:
        print('START',variant,'compile' if command==cuda else 'link',flush=True)
        subprocess.run(command,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
for name,h in old['objects'].items():assert sha(ROOT/'fastkpc/build'/name)==h,name
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==old['native_sha256']
write(dest/'build.json',dict(status='PASS',variant=variant,elapsed_sec=time.time()-start,
    native_sha256=sha(dest/'fastkpc_cuda.so'),original_native_sha256=old['native_sha256'],
    object_sha256=sha(dest/'small64.o'),production_source_sha256=identity['production_source_sha256'],
    protocol_sha256=sha(dest/'build-protocol.json'),profile=profile))
print('PASS research backend',variant,flush=True)
