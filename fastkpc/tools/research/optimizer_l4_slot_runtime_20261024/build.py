"""Audited incremental build; L4 host scheduling only, CUDA instructions checked separately."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4]
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
coexist=out/'coexistence-replacement/closure.json'
assert json.loads(coexist.read_text())['status']=='PASS_REAL_CACHE_ARENA_REPLACEMENT_COEXISTENCE'
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_runtime_20261023'
old=json.loads((prior/'build.json').read_text())
native=root/'fastkpc/build/fastkpc_cuda.so'
assert old['status']=='PASS' and sha(native)==old['native_sha256']
snapshot=out/'build-source';assert not snapshot.exists(),'preserve previous attempts'
sources=[p for p in (root/'fastkpc/src').rglob('*') if p.is_file() and p.suffix in
         ('.cpp','.c','.h','.hpp','.cu','.cuh','.inc','.txt')]
sources += [root/'fastkpc/tools/build_cuda_native.sh',root/'fastkpc/tools/generate_recorded_dbdsqr.py']
hashes={str(p.relative_to(root)):sha(p) for p in sorted(sources)}
allowed={'fastkpc/src/full_cuda_ci_one_call.cpp','fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.cpp',
         'fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.hpp','fastkpc/src/cuda/mgcv_multi_penalty_gcv_small64.cu',
         'fastkpc/src/cuda/mgcv_multi_penalty_gcv_l4.hpp'}
changed={p for p,h in hashes.items() if old['source_sha256'].get(p)!=h}
assert changed==allowed,(changed,allowed)
assert set(hashes)==set(old['source_sha256'])|{'fastkpc/src/cuda/mgcv_multi_penalty_gcv_l4.hpp'}
for p in sources:
    dest=snapshot/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
build=snapshot/'fastkpc/build';build.mkdir()
prior_build=prior/'build-source/fastkpc/build'
for name,h in old['objects'].items():
    assert sha(prior_build/name)==h;shutil.copyfile(prior_build/name,build/name)
shutil.copyfile(prior_build/'recorded_dbdsqr.inc',build/'recorded_dbdsqr.inc')
assert sha(build/'recorded_dbdsqr.inc')==old['generator_output_sha256']
with native.open('rb') as src,(out/'previous-native.so.gz').open('xb') as dest:
    with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as zipped:shutil.copyfileobj(src,zipped)
env=os.environ.copy()
for key in list(env):
    if key.startswith('FASTKPC_INCREMENTAL_') or key=='FASTKPC_MULTI_PENALTY_BUCKETS':del env[key]
env.update(FASTKPC_INCREMENTAL_EXACT_FIXED_EVALUATOR_ONLY='1',FASTKPC_MULTI_PENALTY_BUCKETS='small64')
started=time.time()
with (out/'build.log').open('x') as log:
    subprocess.run(['sh','-x',str(snapshot/'fastkpc/tools/build_cuda_native.sh')],cwd=snapshot,
                   env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
assert all(sha(root/p)==h and sha(snapshot/p)==h for p,h in hashes.items())
objects={p.name:sha(p) for p in sorted(build.glob('*.o'))}
rebuilt=['full_cuda_ci_one_call.o','mgcv_multi_penalty_gcv_capacity.o','r_api_multi_penalty_capacity.o',
         'r_api_cuda.o','full_cuda_ci_vertical.o','mgcv_multi_penalty_gcv_small64.o','full_cuda_ci_structured_eig.o']
assert objects.keys()==old['objects'].keys()
assert all(h==old['objects'][name] for name,h in objects.items() if name not in rebuilt)
assert sha(build/'recorded_dbdsqr.inc')==old['generator_output_sha256']
for name in [*rebuilt,'recorded_dbdsqr.inc','fastkpc_cuda.so']:shutil.copyfile(build/name,root/'fastkpc/build'/name)
assert all(sha(root/'fastkpc/build'/name)==h for name,h in objects.items())
receipt=dict(schema='optimizer-l4-slot-incremental-build-v1',status='PASS',
    source_build='verified parent objects; seven rebuilt objects and relink; unchanged numerical source',
    evidence_class='audited incremental source closure; not clean producer or full rebuild',
    parent_build=str((prior/'build.json').relative_to(root)),parent_build_sha256=sha(prior/'build.json'),
    coexistence_closure_sha256=sha(coexist),previous_native_sha256=old['native_sha256'],native_sha256=sha(native),
    source_sha256=hashes,changed_sources=sorted(changed),objects=objects,rebuilt_objects=rebuilt,
    retained_objects_exact=True,generator_output_sha256=sha(build/'recorded_dbdsqr.inc'),
    elapsed_sec=time.time()-started,producer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
with (out/'build.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print('PASS source-closure build',receipt['native_sha256'],flush=True)
