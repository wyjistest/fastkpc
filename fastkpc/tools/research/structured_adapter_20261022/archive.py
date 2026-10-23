"""Close the isolated campaign without relabeling it runtime/promotion evidence."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

work=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();root=Path.cwd()
assert not out.exists(), 'refuse to overwrite a prior evidence archive'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda name:json.loads((work/name).read_text())
build=read('build.json');report=read('qualification.json');smoke=read('smoke.json')
assert report['status']=='PASS_ADAPTER_GATES' and smoke['status']=='PASS_ADAPTER_GATES'
assert report['resources_closed'] and smoke['resources_closed']
assert len(report['rows'])==6
assert all([v['width'] for v in r['batch_variants']]==[1,2,17,31,64] for r in report['rows'])
assert all(r['original_total_bitwise'] and r['resources_closed'] for r in report['rows'])
assert report['concurrent']['contexts']==10 and report['concurrent']['bitwise']
assert report['fault_tests']['dense_error_no_publication']
assert report['build_sha256']==smoke['build_sha256']==sha(work/'build.json')
assert report['runner_sha256']==smoke['runner_sha256']==sha(Path(__file__).with_name('run.py'))
assert report['native_sha256']==smoke['native_sha256']==build['native_sha256']
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']
for p,h in build['sources'].items():assert sha(root/p)==h,p
for p,h in build['artifacts'].items():assert sha(work/p)==h,p
memcheck=Path('/tmp/fastkpc-structured-adapter-memcheck.log')
text=memcheck.read_text()
assert 'ERROR SUMMARY: 0 errors' in text
assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in text
out.mkdir(parents=True)
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    assert sha(source)==sha(dest)
for p in work.iterdir():
    if not p.is_file():continue
    if p.suffix=='.npz':
        # Lossless packaging after all GPU/diagnostic timing boundaries.
        with p.open('rb') as a,(out/(p.name+'.gz')).open('wb') as b:
            with gzip.GzipFile(filename='',mode='wb',fileobj=b,mtime=0) as z:shutil.copyfileobj(a,z)
    elif p.suffix in ('.json','.inc','.so','.tsv'):
        copy(p,out/p.name)
sources=set(build['sources'])
sources.update(str(p.relative_to(root)) for p in Path(__file__).resolve().parent.iterdir() if p.is_file())
sources.add('fastkpc/docs/PHASE10_STRUCTURED_DEVICE_ADAPTER_20261022.md')
source_hash={}
for name in sorted(sources):
    copy(root/name,out/'sources'/name);source_hash[name]=sha(root/name)
for kind in ('build','qualification','smoke','memcheck'):
    p=Path(f'/tmp/fastkpc-structured-adapter-{kind}.log')
    if p.exists():copy(p,out/'logs'/p.name)
inputs=[
 'fastkpc/artifacts/full_cuda_ci/structured_real_20261020/manifest.json',
 'fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/manifest.json',
 'fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/fastkpc_cuda.so.gz',
 'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds']
for base,names in [
 ('fastkpc/artifacts/full_cuda_ci/structured_real_20261020',
  ['pairs.csv','residuals.f64']+[f'witnesses/{n}.{suffix}' for n in (242,253,351,364) for suffix in ('csv','f64')]),
 ('fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021',
  ['audit-on/components.tsv.gz','audit-on/pairs.tsv.gz','audit-on/wide-raw.tar.gz','component-attribution.csv'])]:
    reference=json.loads((root/base/'manifest.json').read_text())
    for name in names:
        expected=reference['files'][name]
        if isinstance(expected,dict):expected=expected['sha256']
        assert sha(root/base/name)==expected,name
        inputs.append(str(Path(base)/name))
with gzip.open(root/inputs[2],'rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==build['native_sha256']
rows=report['rows']
closure=dict(schema='structured-device-adapter-closure-v1',status='PASS_NARROW_DEVICE_ADAPTER',
 created_utc=datetime.now(timezone.utc).isoformat(),native_sha256=build['native_sha256'],
 components=sum(r['components'] for r in rows),pairs=sum(r['structured_authority']['pairs'] for r in rows),
 max_authority_p_error=max(r['structured_authority']['max_p_error'] for r in rows),
 max_p_difference_vs_dense=max(r['max_p_vs_dense'] for r in rows),
 batch_widths=[1,2,17,31,64],batch_order_bits='PASS',gpu_dense_fallback='PASS',
 concurrency_contexts=report['concurrent']['contexts'],concurrency_bits='PASS',
 sanitizer=dict(errors=0,leaked_bytes=0,scope='four n witnesses and adapter failure paths; not full method'),
 numerical_production='GPU only; archived CPU authority read only',
 component_solver_identity={1:'structured-device-three-field-v1',2:'retained-dense-three-field-v1'},
 gate_gap=1e-8,full_method_integration='NOT_RUN',next_gate='OPT_IN_22_COLUMN_FULL_METHOD',
 evidence_class='isolated source-bound development qualification; no performance promotion',
 production_default=False,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
manifest=dict(schema='structured-device-adapter-evidence-v1',status='PASS',files=files,
 sources=source_hash,external_inputs={p:sha(root/p) for p in inputs})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for name,h in files.items():assert sha(out/name)==h
print('PASS',len(files),'files; manifest',sha(out/'manifest.json'))
