"""Close this source/binary/receipt epoch; never promote the runtime."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

out=Path(sys.argv[1]);root=Path.cwd();tool=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
native=root/'fastkpc/build/fastkpc_cuda.so';native_sha=sha(native)
reference_native_sha=hashlib.sha256(gzip.decompress((out/'reference-native.so.gz').read_bytes())).hexdigest()
assert reference_native_sha=='b1e7eab4297a13b7ff29f0bf40b2fc3ce4d6950542dff4318ee3e2dc76694ee9'
reports={name:read(out/name/'report.json') for name in ('off','dense','structured')}
for name,r in reports.items():
    assert r['gates'] and all(r['gates'].values()),name
    assert r['native_sha256']==native_sha,name
    assert r['runner_sha256']==sha(tool/'run.R')
    assert r['receipt_sha256']==sha(out/name/'receipt.rds')
    assert r['authority_sha256']==sha(out/'authority.rds')
    assert all(sha(p)==h for p,h in r['source_sha256'].items())

old=Path('fastkpc/tools/research/exact_speedup_20261020/structured_eig_gpu.cu').read_text()
new=Path('fastkpc/src/cuda/full_cuda_ci_structured_eig.cu').read_text()
marker='namespace fastkpc { namespace lapack312 {'
original_kernels=old[old.index(marker):old.index('void check(cudaError_t code)')].rstrip()
new_kernels=new[new.index(marker):new.index('__global__ void check_structured_boundary')].rstrip()
assert original_kernels==new_kernels,'arithmetic kernel source changed'
vertical=Path('fastkpc/src/cuda/full_cuda_ci_vertical.cu').read_text()
def center(name):
    start=vertical.index('  double* output = centered_vectors +',vertical.index(name))
    return vertical[start:vertical.index('\n}',start)]
assert center('__global__ void select_center_legacy')==center('__global__ void center_structured_component')
recipe=Path('fastkpc/tools/build_cuda_native.sh').read_text()
assert '--fmad=false -DFASTKPC_STRUCTURED_FMAD_DISABLED=1' in recipe

logs={
 'build.log':'/tmp/fastkpc-structured-prefill-build-final.log',
 'lifecycle-build.log':'/tmp/fastkpc-structured-prefill-lifecycle-final-build.log',
 'lifecycle.log':'/tmp/fastkpc-structured-prefill-lifecycle-final.log',
 'memcheck.log':'/tmp/fastkpc-structured-prefill-memcheck.log',
 'configuration.log':'/tmp/fastkpc-structured-prefill-config.log',
 'existing-configuration.log':'/tmp/fastkpc-structured-prefill-old-config.log',
 'cache-regression.log':'/tmp/fastkpc-structured-prefill-cache-regression.log',
 'authority.log':'/tmp/fastkpc-structured-prefill-authority.log',
 'campaign.log':'/tmp/fastkpc-structured-prefill-campaign.log',
 'completion-timing.log':'/tmp/fastkpc-structured-completion-timing.log'}
assert 'PASS 640 adapter bytes' in Path(logs['lifecycle.log']).read_text()
memcheck=Path(logs['memcheck.log']).read_text()
assert 'PASS 4 adapter bytes' in memcheck and '0 bytes leaked' in memcheck and 'ERROR SUMMARY: 0 errors' in memcheck
for name in ('configuration.log','existing-configuration.log','cache-regression.log'):
    assert 'PASS' in Path(logs[name]).read_text()
(out/'logs').mkdir(exist_ok=True)
for name,path in logs.items():shutil.copyfile(path,out/'logs'/name)
shutil.copyfile('/tmp/fastkpc-structured-prefill-lifecycle-final',out/'lifecycle')

sources=set(reports['off']['source_sha256'])
sources.update(str(p) for p in Path('fastkpc/src').rglob('*')
               if p.is_file() and p.suffix in ('.cpp','.c','.h','.hpp','.cu','.cuh','.inc'))
sources.update(str(p.relative_to(root)) for p in tool.iterdir() if p.is_file())
sources.update(['fastkpc/tests/test_full_cuda_ci_structured_prefill.R',
 'fastkpc/tests/test_full_cuda_ci_level_component_prefill.R',
 'fastkpc/tests/test_full_cuda_ci_legacy_component_cache.R',
 'fastkpc/tools/research/exact_speedup_20261020/structured_eig_gpu.cu',
 'fastkpc/tools/research/exact_speedup_20261020/generate_recorded_dbdsqr.py',
 'fastkpc/tools/research/structured_adapter_20261022/run.py',
 'fastkpc/src/cuda/lapack_312_small_dgesdd.cuh',
 'fastkpc/src/cuda/third_party/lapack_3_12_small_dgesdd_device.inc',
 'fastkpc/docs/PHASE10_STRUCTURED_PREFILL_RESULT_20261022.md',
 'fastkpc/docs/PHASE10_STRUCTURED_DEVICE_ADAPTER_REVIEW_20261022.md',
 'goal-5.6.md'])
for name in sorted(sources):
    dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
with native.open('rb') as source,(out/'fastkpc_cuda.so.gz').open('wb') as f:
    with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as target:shutil.copyfileobj(source,target)
with (out/'runtime.patch').open('w') as f:
    subprocess.run(['git','diff','--','fastkpc/src','fastkpc/tools/build_cuda_native.sh'],stdout=f,check=True)
objects={str(p):sha(p) for p in sorted(Path('fastkpc/build').glob('*.o'))}
(out/'build-objects.json').write_text(json.dumps(dict(
    mode='incremental EIG_ONLY plus dedicated structured TU; retained objects explicitly hashed',
    native_sha256=native_sha,objects=objects,
    recorded_dbdsqr_sha256=sha('fastkpc/build/recorded_dbdsqr.inc')),indent=2)+'\n')
memory={}
for name in reports:
    samples=[line.split(',') for line in (out/(name+'-memory.csv')).read_text().splitlines()]
    memory[name]=max(float(row[1]) for row in samples if len(row)==3)
rows=reports['structured']['level_component_prefill']
closure=dict(schema='structured-prefill-subset22-closure-v1',status='PASS_SUBSET22_OPT_IN',
    native_sha256=native_sha,
    reference_native_sha256=reference_native_sha,
    arithmetic_kernel_source_exact=True,centering_source_exact=True,
    source_closure=reports['off']['source_sha256'],
    elapsed_sec={name:r['elapsed_sec'] for name,r in reports.items()},
    stage_b_sec={name:sum(row['stage_b_host_ms'] for row in r['level_component_prefill'])/1000
                 for name,r in reports.items()},
    all_ci_count=24264,produced=7270,consumed=6632,
    structured_accepted=sum(row['structured']['accepted'] for row in rows),
    dense_fallback=sum(row['structured']['dense_fallback'] for row in rows),
    maximum_p_error=reports['structured']['oracle_errors'],
    device_peak_sampled_mib=memory,
    evidence_class='one source-closure development run per mode; not a repeated median or promotion',
    numerical_production='GPU only; independent offline kpcalg test oracle is comparison-only',
    production_default=False,canonical_qualification='NOT_RUN',final_pdag_campaign='NOT_RUN',
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
inputs=['fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/manifest.json',
 'fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/adapter.so',
 'fastkpc/artifacts/full_cuda_ci/structured_real_20261020/manifest.json',
 'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds',
 'fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/subset-off/receipt.rds.gz',
 'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']
files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
       for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
manifest=dict(schema='structured-prefill-subset22-manifest-v1',status='PASS',
    native_sha256=native_sha,files=files,
    source_sha256={p:sha(p) for p in sorted(sources)},inputs={p:sha(p) for p in inputs})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(closure,indent=2));print('PASS manifest',sha(out/'manifest.json'))
