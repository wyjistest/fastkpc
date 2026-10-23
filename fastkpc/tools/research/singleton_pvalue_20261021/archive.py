"""Archive a source/native-bound scalar-only GPU diagnostic, including raw anchors."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

work,old,destination=(Path(p).resolve() for p in sys.argv[1:])
root=Path.cwd(); here=Path(__file__).resolve().parent
destination.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(name):return json.loads((work/name).read_text())
build=read('build/build.json'); capture=read('native/capture.json')
export=read('candidates/report.json'); full=read('full/report.json'); smoke=read('memcheck/report.json')
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']==capture['native_sha256']==full['native_sha256']
assert all(capture['gates'].values()) and capture['physical_pairs']==capture['tasks']==24264
assert capture['receipt_sha256']==sha(work/'native/receipt.rds')
assert capture['interposer_sha256']==build['artifacts']['capture.so']
assert capture['runner_sha256']==sha(here/'capture_native.R')
assert export['runner_sha256']==sha(here/'export_candidates.py')
assert full['runner_sha256']==smoke['runner_sha256']==sha(here/'replay.py')
for path,digest in build['sources'].items(): assert sha(root/path)==digest,path
for name,digest in build['artifacts'].items(): assert sha(work/'build'/name)==digest,name
for receipt in (full,smoke):
    assert receipt['build_sha256']==sha(work/'build/build.json')
    assert receipt['native_capture_sha256']==sha(work/'native/capture.json')
    assert receipt['candidate_export_sha256']==sha(work/'candidates/report.json')
    assert receipt['resources_before']==receipt['resources_after']==[0,0,0,0]
assert export['resources_before']==export['resources_after']==[0,0,0,0]
assert full['affected_components']==1002 and export['changed']==1002
assert full['scope'].startswith('all affected') and smoke['scope']=='smoke'
assert export['scalars_sha256']==sha(work/'candidates/scalars.npz')
assert export['keys_sha256']==sha(work/'candidates/keys.json')
for prefix,receipt in (('full',full),('memcheck',smoke)):
    for name,digest in receipt['outputs'].items(): assert sha(work/prefix/name)==digest
sanitizer=(work/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in sanitizer
assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in sanitizer
dependency=root/'fastkpc/artifacts/full_cuda_ci/component_payload_20261021'
prior_raw=json.loads((dependency/'native/raw-manifest.json').read_text())
for name,item in prior_raw.items():
    assert sha(old/'native/raw'/name)==item['sha256'],name
assert export['original_report_sha256']==sha(dependency/'native-full/report.json')
assert export['build_sha256']==sha(dependency/'build/build.json')
assert export['probe_sha256']==sha(dependency/'build/probe.so')
files=['build/build.json','build/probe.so','build/capture.so','build/pair_stage.inc',
 'native/capture.json','native/receipt.rds','native/tasks.tsv','native/task-pvalues.f64',
 'candidates/changed-keys.txt','candidates/keys.json','candidates/scalars.npz','candidates/report.json',
 'full/report.json','full/pairs.jsonl','full/task-pvalues.npz',
 'memcheck/report.json','memcheck/pairs.jsonl','memcheck/task-pvalues.npz','memcheck.log']
for name in files:
    target=destination/name; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(work/name,target)
for path in here.iterdir():
    if path.is_file():
        target=destination/'sources'/path.name; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path,target)
raw=work/'native/raw'; raw_files=sorted(p for p in raw.iterdir() if p.is_file())
manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in raw_files}
archive_path=destination/'native/raw.tar.gz'
with tarfile.open(archive_path,'w:gz',compresslevel=1) as archive:
    for path in raw_files:archive.add(path,arcname=path.name,recursive=False)
with tarfile.open(archive_path,'r:gz') as archive:
    members=archive.getmembers(); assert len(members)==len(manifest)
    for member in members:
        blob=archive.extractfile(member).read(); expected=manifest[member.name]
        assert len(blob)==expected['bytes'] and hashlib.sha256(blob).hexdigest()==expected['sha256']
(destination/'native/raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
artifacts={str(p.relative_to(destination)):dict(bytes=p.stat().st_size,sha256=sha(p))
 for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='manifest.json'}
report=dict(schema='singleton-gpu-pvalue-campaign-v1',audit='PASS',diagnostic_result=full['status'],
 original_payload_gate='STOP_PAYLOAD_BITS',production_integration=False,
 native_sha256=build['native_sha256'],runtime_producer='dcb1749',
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 dependency_manifest=str((dependency/'manifest.json').relative_to(root)),
 dependency_manifest_sha256=sha(dependency/'manifest.json'),
 gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True).strip().splitlines(),
 artifacts=artifacts,evidence_class='GPU scalar-only diagnostic, no full-method candidate, oracle or performance claim')
(destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('audit PASS',full['status'],'manifest',sha(destination/'manifest.json'),flush=True)
