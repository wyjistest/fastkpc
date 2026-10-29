"""Preserve protocol tools, then bind completed attribution to immutable evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
protocol=read(out/'protocol.json')
if '--snapshot' in sys.argv:
    dest=out/'protocol-tools';dest.mkdir(exist_ok=False)
    for p,h in protocol['tools'].items():
        source=ROOT/p;assert sha(source)==h,p
        shutil.copy2(source,dest/source.name)
    write(dest/'index.json',protocol['tools']);print('PASS preserved protocol tools');sys.exit(0)
assert not (out/'manifest.json').exists()
b=read(out/'build.json');a=read(out/'analysis.json');r=read(out/'canonical/report.json')
assert a['status']=='PASS_L45_IDENTITY_AND_EXISTING_PHASE_ATTRIBUTION'
assert all(r['gates'].values()) and a['observer']['canonical_request_order_exact']
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==b['native_sha256']
for p,h in b['source_sha256'].items():assert sha(ROOT/p)==h,p
for p,h in protocol['dependencies'].items():assert sha(Path(p))==h,p
for p,h in protocol['tools'].items():
    snapshot=out/'protocol-tools'/Path(p).name
    assert sha(snapshot if snapshot.exists() else ROOT/p)==h,p
for name in ['smoke','canonical']:
    process=read(out/(name+'-process.json'));assert process['exit_code']==0 and process['isolated'] and not process['unexpected']
assert sha(out/'canonical/receipt.rds')==r['receipt_sha256']
# These shared inputs were not all predeclared in protocol.json. Bind them at
# archival time, distinguishing current committed source from a pre-run freeze.
shared_paths={
    'fastkpc/tools/research/stage_b_work_pair_runtime_20261028/support.py',
    'fastkpc/tools/research/stage_b_work_pair_runtime_20261028/run.R',
    'fastkpc/tools/research/optimizer_windows_20261022/fixture.hpp',
    'fastkpc/tools/research/optimizer_l4_diagnostic_20261023/fixture.hpp',
    'fastkpc/tools/research/optimizer_layer_refill_20261023/layer_fixture.hpp',
    'fastkpc/tools/with_phase10_exact_development_env.sh',
    'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    'fastkpc/R/fast_kpc.R',
    'fastkpc/R/full_cuda_ci_phase35_vertical.R',
}
pending=list(shared_paths)
while pending:
    path=pending.pop()
    if not path.endswith('.R'):continue
    # Include static source() dependencies, even those behind unused branches.
    for child in re.findall(r'''\bsource\(\s*["'](fastkpc/[^"']+)["']''', (ROOT/path).read_text()):
        if child not in shared_paths:shared_paths.add(child);pending.append(child)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
shared={}
for path in sorted(shared_paths):
    blob=subprocess.check_output(['git','show',head+':'+path],cwd=ROOT)
    digest=sha(ROOT/path)
    assert digest==hashlib.sha256(blob).hexdigest(),path
    shared[path]=dict(sha256=digest,git_commit=head,
        git_blob=subprocess.check_output(['git','rev-parse',head+':'+path],cwd=ROOT,text=True).strip())
assert shared['fastkpc/tools/research/stage_b_work_pair_runtime_20261028/run.R']['sha256']==b['runner_source_sha256']
offline_fields=json.loads(subprocess.check_output(['Rscript','--vanilla','-e',
    'x <- readRDS(commandArgs(TRUE)[[1L]]); '
    'cat(jsonlite::toJSON(list(dataset_sha256=x$dataset_sha256, '
    'native_sha256=x$native_sha256), auto_unbox=TRUE))',str(out/'canonical/receipt.rds')],
    cwd=ROOT,text=True))
assert offline_fields['native_sha256']==b['native_sha256']
data_path=ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds'
assert sha(data_path)==offline_fields['dataset_sha256']
data_dependencies={str(data_path):offline_fields['dataset_sha256']}
for name,digest in r['authority_sha256'].items():
    path=ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/(name+'.rds')
    assert sha(path)==digest,str(path)
    data_dependencies[str(path)]=digest
assert sha(out/'run.R')==r['runner_sha256']
reference=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/on2/receipt.rds'
assert sha(reference)==r['reference_sha256']
write(out/'shared-dependencies.json',dict(
    evidence_class='post-run archival source closure, not additional predeclared protocol inputs',
    static_source_closure_only=True,source=shared,data_sha256=data_dependencies,
    data_binding='dataset from saved receipt; authority files from saved canonical report',
    no_gpu_execution=True))
prior={}
for attempt in sorted(out.parent.glob('attempt*')):
    if attempt==out:continue
    prior[attempt.name]={str(p.relative_to(attempt)):sha(p) for p in sorted(attempt.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
write(out/'prior-attempts.json',prior)
write(out/'closure.json',dict(status='COMPLETE_L45_DIAGNOSTIC_NO_PRODUCTION_CHANGE',
    native_sha256=b['native_sha256'],setups=3109,targets=24583,logical_ci=240498,
    correlated_kernels=a['correlated_kernels'],canonical_request_order_exact=True,
    metadata_and_captured_optimizer_outputs_exact=True,extra_device_copies=0,
    analysis_sha256=sha(out/'analysis.json'),report_sha256=sha(out/'canonical/report.json'),
    shared_dependencies_sha256=sha(out/'shared-dependencies.json'),
    numerical_gates=len(r['gates']),cross_domain_end_discrepancies=len(a['cross_domain_end_violations']),
    receipt_sha256=r['receipt_sha256'],production_changes=False,performance_evidence=False,
    accepted_opt_in_development_median_sec=311.440,phase10='ACTIVE',candidate='NOT_FROZEN',
    holdout='SEALED',recommended=False,default_enabled=False,l8_optimizer='STOP_UNCHANGED'))
deps=dict(protocol['dependencies'])
deps.update({str(ROOT/p):v['sha256'] for p,v in shared.items()})
deps.update(data_dependencies)
for name in ['ktrace-run','ktrace-run-cache-on']:
    base=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025'/name
    for filename in ['kernels.bin','runtime.bin','kernel-names.tsv','profile-result.json','ktrace-summary.txt']:
        p=base/filename;deps[str(p)]=sha(p)
tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()}
files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
manifest=dict(schema='l45-attribution-manifest-v1',native_sha256=b['native_sha256'],
    source_sha256=b['source_sha256'],tools=tools,dependencies=deps,files=files)
write(out/'manifest.json',manifest)
print(json.dumps(dict(closure_sha256=sha(out/'closure.json'),manifest_sha256=sha(out/'manifest.json'),
    artifacts=len(files),sources=len(b['source_sha256']),tools=len(tools),dependencies=len(deps)),indent=2))
