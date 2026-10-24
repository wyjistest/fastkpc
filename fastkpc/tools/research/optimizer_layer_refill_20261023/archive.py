"""Audit and close the bounded L5 diagnostic, without runtime promotion."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parents[4]
tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda name:json.loads((out/name).read_text())
build=read('replay-build.json');capture=read('capture.json');capacity=read('capacity-summary.json')
assert all(capture['gates'].values()) and capture['l5_setups']==955 and capture['l5_targets']==6816
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']==capture['native_sha256']
for name in ['capture-build.json','replay-build.json','profile-build.json']:
    b=read(name)
    assert b['native_sha256']==build['native_sha256']
    assert all(sha(root/p)==h for p,h in b['sources'].items())
    assert all(sha(root/p)==h for p,h in b.get('source_closure',{}).items())
    assert all(sha(out/p)==h for p,h in b['outputs'].items())
assert all(sha(out/p)==h for p,h in build['outputs'].items())
# capture.R serializes its named vector as a JSON array in numeric W1..W15 order.
assert len(capture['fixture_sha256'])==15
assert all(sha(out/'raw'/('L5-W'+str(i)+'.bin'))==h
           for i,h in enumerate(capture['fixture_sha256'],1))
assert capacity['status']=='PASS' and capacity['resources_closed']
assert capacity['free_all_resident']>=capacity['headroom_required']
checks={name:read(name+'-summary.json') for name in ['memcheck','campaign','profile-A','profile-R']}
for name,r in checks.items():
    assert r['status']=='PASS' and r['resources_closed'] and r['output_leases_closed']
    assert r['all_outputs_bitwise_exact'] and r['forbidden_calls']==0
assert checks['campaign']['runs']==21 and checks['campaign']['compared_target_occurrences']==21*6816
for name in ['profile-A','profile-R']:
    assert checks[name]['runs']==2 and checks[name]['compared_target_occurrences']==2*6816
assert checks['memcheck']['compared_target_occurrences']==30
mem=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
process={name:read(name+'-process.json') for name in ['capture','capacity','memcheck','campaign','profile-A','profile-R']}
assert all(p['exit_code']==0 and p['physical_gpu']==1 and not p['unexpected_gpu_pids'] for p in process.values())
with (out/'campaign-runs.csv').open() as stream:rows=list(csv.DictReader(stream))
timed=[r for r in rows if r['kind']=='timing']
assert len(timed)==18 and all(r['parity']=='1' and r['forbidden_calls']=='0' for r in rows)
metrics=['native_ms','prepare_ms','destroy_ms','pack_ms','verify_ms','token_release_ms',
         'raw_ms','segmented_work_ms','pool_prepare_ms','pool_destroy_ms','lifecycle_work_ms']
summary={}
for v in 'PAR':
    group=[r for r in timed if r['variant']==v]
    assert len(group)==6 and all(int(r['targets'])==6816 and int(r['setups'])==955 for r in group)
    assert all(int(r['calls'])==(1 if v=='R' else 15) for r in group)
    summary[v]={m:dict(median=statistics.median(float(r[m])/1000 for r in group),
        minimum=min(float(r[m])/1000 for r in group),maximum=max(float(r[m])/1000 for r in group)) for m in metrics}
ratios=[]
for cycle in range(6):
    group={r['variant']:r for r in timed if int(r['cycle'])==cycle}
    assert set(group)==set('PAR')
    assert group['A']['pool_prepare_ms']==group['R']['pool_prepare_ms']
    assert group['A']['pool_destroy_ms']==group['R']['pool_destroy_ms']
    ratios.append(float(group['R']['native_ms'])/float(group['A']['native_ms']))
ratio=summary['R']['native_ms']['median']/summary['A']['native_ms']['median']
lifecycle_ratio=summary['R']['lifecycle_work_ms']['median']/summary['P']['segmented_work_ms']['median']
go=ratio<=0.90 and lifecycle_ratio<1.0
peak={}
for name in process:
    with (out/(name+'-gpu.csv')).open() as stream:gpu=list(csv.reader(stream))
    selected=[r for r in gpu if len(r)==7 and int(r[1])==1]
    other=[r for r in gpu if len(r)==7 and int(r[1])==0]
    peak[name]=dict(gpu1_peak_mib=max(int(r[2]) for r in selected),
        gpu0_peak_utilization=max(int(r[3]) for r in other))
profile=read('profile-analysis.json')
assert profile['status']=='PASS'
for variant in 'AR':
    assert profile['variants'][variant]['setups']==955
    assert profile['variants'][variant]['other_kernels_in_range']==0
assert profile['variants']['A']['correlated_kernels']==profile['variants']['R']['correlated_kernels']
assert profile['variants']['A']['cross_original_window_kernel_overlap_sec']==0
go=go and profile['variants']['R']['cross_original_window_kernel_overlap_sec']>0
closure=dict(schema='optimizer-layer-refill-closure-v1',
 status='GO_L5_RESIDENT_DIAGNOSTIC' if go else 'STOP_OR_CONDITIONAL_L5_RESIDENT_DIAGNOSTIC',
 evidence_class='GPU1 shared-host captured-layer source-closure development; not outer-method or promotion',
 native_sha256=build['native_sha256'],source_sha256=build['source_closure'],
 layer=5,setups=955,targets=6816,original_windows=15,capture_ci_count=capture['ci_count'],
 all_outputs_bitwise_exact=True,resources_closed=True,hard_gates='PASS',
 capacity=capacity,seconds=summary,R_over_A_native_ratio=ratio,
 same_cycle_R_over_A=ratios,same_cycle_R_wins=sum(r<1 for r in ratios),
 R_lifecycle_over_P_segmented_reference=lifecycle_ratio,
 predeclared_R_over_A_limit=0.90,P_reference_limitations='inter-window diagnostic D2H/verification pauses; not end-to-end runtime timing',
 compared_target_occurrences=sum(r['compared_target_occurrences'] for r in checks.values()),
 memcheck_scope='four actual setups from first and last windows, all three configurations',
 memcheck_errors=0,memcheck_leaked_bytes=0,telemetry=peak,processes=process,
 profile=profile,incomplete_profile=(dict(status='NOT_COMPLETED_NOT_PASS',
     process=read('profile-process.json'),fault_location='libcupti.so.12.5; exact trigger unproven')
     if (out/'profile-process.json').exists() else None),
 slot_pool_implemented=False,runtime_rolling='NOT_YET_GO',
 phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
manifest=dict(schema='optimizer-layer-refill-manifest-v1',native_sha256=build['native_sha256'],
 source_sha256=build['source_closure'],
 tools={str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()},
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'},
 references={str(p.relative_to(root)):sha(p) for p in [
   root/'fastkpc/artifacts/full_cuda_ci/optimizer_status_wait_20261023/manifest.json',
   root/'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022/manifest.json']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['source_sha256','processes','profile','seconds']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
