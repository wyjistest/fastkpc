"""Seal completed numerical/capacity work without accepting or installing it.

The isolated OFF/ON campaign is an independent outstanding gate. This audit
cannot produce GO_CAPACITY or replace the accepted installed native.
"""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,PARENT,BASE,TOOL,sha,read,write

parent=read(PARENT/'closure.json');build=read(BASE/'candidate/build.json')
unit=read(BASE/'qualification/closure.json');semantic=read(BASE/'semantic/closure.json')
assert parent['status']=='PASS_BUILD_FIX_PARENT'
assert read(PARENT/'object-payload-comparison.json')['status']=='PASS'
assert unit['status']=='PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES'
assert semantic['status']=='PASS_NUMERICAL_CAPACITY_PENDING_OUTER'
assert all(semantic['capacity_gates'].values())
for p,h in build['source_sha256'].items():
    assert sha(ROOT/p)==h and sha(BASE/'candidate/source'/p)==h,p
for name,h in build['objects'].items():assert sha(BASE/'candidate/objects'/name)==h,name
assert sha(build['native_path'])==build['native_sha256']
for mode in ['parent','full','incremental']:
    folder=PARENT/mode/'qualification'
    assert read(folder/'closure.json')['status']=='PASS'
    report=read(folder/'report.json')
    assert all(all(r['gates'].values()) for r in report['reports'].values())
    for p,h in read(folder/'protocol.json')['dependencies'].items():
        if sha(p)!=h:
            archived=folder/'tools'/Path(p).name
            assert archived.is_file() and sha(archived)==h,p
for folder in [PARENT/'full/regressions2',PARENT/'canonical',BASE/'qualification',BASE/'semantic']:
    protocol=read(folder/('run-protocol.json' if folder==BASE/'qualification' else 'protocol.json'))
    for p,h in protocol['dependencies'].items():assert sha(p)==h,p
    for p in folder.glob('*-process.json'):
        record=read(p)
        ok=record['isolated'] or (record.get('numerical_only') and record.get('gpu1_exclusive') and not record['unexpected'])
        assert ok and record['exit_code']==0,str(p)
for folder in [PARENT/'canonical/qualification',BASE/'semantic/qualification']:
    report=read(folder/'report.json')
    assert all(report['gates'].values())
    assert report['comparisons']['changed_ci_count']==report['comparisons']['changed_pmax_count']==0
    assert sha(folder/'receipt.rds')==report['receipt_sha256']
original=read(PARENT/'parent/identity.json')
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==original['native_sha256']
for name,h in original['objects'].items():assert sha(ROOT/'fastkpc/build'/name)==h,name
assert not (BASE/'canonical').exists(),'timing campaign already started; use its actual status'

gate=dict(median_nonregression_ratio_max=1.02,each_pair_ratio_max=1.05,
    speed_median_ratio_max=.98,speed_paired_wins=3,
    measured_pool_bytes_reduction=630789120,minimum_free_improvement_bytes=600*1024**2)
write(BASE/'pending-outer-protocol.json',dict(epoch=time.time(),status='NOT_STARTED',
    sequence=[['off',1],['on',1],['on',2],['off',2],['off',3],['on',3]],gate=gate,
    both_gpus_free_of_external_compute=True,physical_gpu=1,
    parent_native_sha256=parent['native_sha256'],candidate_native_sha256=build['native_sha256'],
    pending_command='python3 fastkpc/tools/research/structured_rows_20261029/campaign.py',
    declaration_sha256=sha(TOOL/'README.md'),campaign_tool_sha256=sha(TOOL/'campaign.py'),
    accepts_shared_host_timings=False,install_before_nonregression=False))
tools={}
for folder in [TOOL,TOOL.parent/'build_consistency_20261029']:
    for p in folder.iterdir():
        if p.is_file():
            import shutil
            target=BASE/'pending-tools'/folder.name/p.name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);tools[str(p.relative_to(ROOT))]=sha(p)
write(BASE/'pending-closure.json',dict(status='NUMERICAL_AND_CAPACITY_QUALIFIED_PERFORMANCE_PENDING',
    machine_audit='PASS_COMPLETED_SCOPE_ONLY',parent_closure_sha256=sha(PARENT/'closure.json'),
    native_sha256=build['native_sha256'],build_sha256=sha(BASE/'candidate/build.json'),
    gpu_qualification_sha256=sha(BASE/'qualification/closure.json'),
    canonical_qualification_sha256=sha(BASE/'semantic/closure.json'),
    capacity_pool_reduction_bytes=630789120,capacity_checkpoint_improvement_bytes=650117120,
    outer_nonregression='NOT_TESTED',speed='NOT_TESTED',installed=False,
    installed_native_sha256=original['native_sha256'],accepted_development_median_sec=290.491,
    block='GPU0 external compute prevents the predeclared isolated timing campaign',
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
    no_q55_expansion=True,no_lane_slot_or_recover_tile_change=True,
    evidence_class='source-closure GPU1 numerical/capacity qualification; shared host; no timing claim',
    preserved_failures=['raw object metadata assertion','R test cleanup wrapper','standalone R library link path'],
    pending_protocol_sha256=sha(BASE/'pending-outer-protocol.json')))
files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and p.name!='pending-manifest.json'}
parent_files={str(p.relative_to(PARENT)):sha(p) for p in PARENT.rglob('*') if p.is_file()}
write(BASE/'pending-manifest.json',dict(schema='structured-rows-pending-closure-v1',
    production_source_sha256=build['source_sha256'],tools=tools,files=files,parent_files=parent_files,
    native_sha256=build['native_sha256'],performance_evidence=False))
print('PASS completed-scope audit; no installation; isolated timing still required',flush=True)
