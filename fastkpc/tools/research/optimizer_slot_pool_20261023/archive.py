"""Close the bounded slot prototype as development evidence only."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
root=Path(__file__).resolve().parents[4];tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((out/n).read_text())
build=read('build.json');assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']
for key in ['sources','source_sha256','cuda_input_sha256']:
    assert all(sha(root/p)==h for p,h in build[key].items()),key
assert all(sha(out/p)==h for p,h in build['outputs'].items())
device=read('device-code.json');assert device['status']=='PASS' and device['original']==device['prototype']
assert len(device['original'])==14
reports={n:read(n+'-summary.json') for n in ['checks','memcheck','capacity','campaign']}
for n,r in reports.items():
    assert r['status']=='PASS' and r['resources_closed'] and r['outputs_bitwise_exact'] and r['forbidden_calls']==0,n
for n in ['checks','memcheck']:
    assert reports[n]['negative_checks']==16 and reports[n]['cancellation_checked']
    assert reports[n]['runs']==2 and reports[n]['compared_target_occurrences']==68
assert reports['capacity']['runs']==1 and reports['capacity']['compared_target_occurrences']==6816
assert reports['campaign']['runs']==14 and reports['campaign']['compared_target_occurrences']==14*6816
mem=(out/'memcheck.log').read_text();assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
processes={n:read(n+'-process.json') for n in reports}
assert all(p['exit_code']==0 and p['physical_gpu']==1 and not p['unexpected_gpu_pids'] for p in processes.values())
rows=lambda n:list(csv.DictReader((out/n).open()))
capacity=rows('capacity-runs.csv');assert len(capacity)==1
capacity=capacity[0];assert int(capacity['physical_increment_bytes'])<=1024**3
assert int(capacity['free_after_create'])>=2*1024**3
assert tuple(int(capacity[k]) for k in ['allocations','streams','events','blas_handles'])==(33,32,96,32)
assert int(capacity['application_bytes'])==32*7000312+40786944
campaign=rows('campaign-runs.csv');timed=[r for r in campaign if r['kind']=='timing'];assert len(timed)==12
for r in campaign:
    assert int(r['targets'])==6816 and int(r['iterations'])==90859
    assert int(r['rebindings'])==(955 if r['variant']=='S' else 0)
    assert int(r['stale_checks'])==(955 if r['variant']=='S' else 0)
    assert int(r['streams'])==(32 if r['variant']=='S' else 955)
details=rows('campaign-setups.csv')
for cycle in range(-1,6):
    g=[r for r in details if int(r['cycle'])==cycle]
    assert len(g)==955 and {int(r['request']) for r in g}==set(range(955))
    assert all(0<=int(r['slot'])<32 for r in g)
    generations={}
    for r in g:
        k=int(r['slot']);generation=int(r['generation']);assert generation>generations.get(k,0)
        generations[k]=generation
summary={}
for v in 'RS':
    g=[r for r in timed if r['variant']==v];assert len(g)==6
    summary[v]={k:dict(median=statistics.median(float(r[k])/1000 for r in g),
        minimum=min(float(r[k])/1000 for r in g),maximum=max(float(r[k])/1000 for r in g))
        for k in ['work_ms','prepare_ms','destroy_ms','packet_pack_ms','lifecycle_ms','verify_ms',
                  'bind_sum_ms','optimize_sum_ms','detach_sum_ms']}
ratios=[];lifecycle_wins=0
for cycle in range(6):
    g={r['variant']:r for r in timed if int(r['cycle'])==cycle};assert set(g)==set('RS')
    ratios.append(float(g['S']['work_ms'])/float(g['R']['work_ms']))
    lifecycle_wins+=float(g['S']['lifecycle_ms'])<float(g['R']['lifecycle_ms'])
ratio=summary['S']['work_ms']['median']/summary['R']['work_ms']['median']
life_ratio=summary['S']['lifecycle_ms']['median']/summary['R']['lifecycle_ms']['median']
go=ratio<=1.15 and life_ratio<1
telemetry={}
for n in reports:
    gpu=list(csv.reader((out/(n+'-gpu.csv')).open()))
    selected=[r for r in gpu if len(r)==7 and int(r[1])==1]
    other=[r for r in gpu if len(r)==7 and int(r[1])==0]
    telemetry[n]=dict(gpu1_peak_mib=max(int(r[2]) for r in selected),gpu0_peak_utilization=max(int(r[3]) for r in other))
closure=dict(schema='optimizer-slot-pool-closure-v1',status='GO_L5_BOUNDED_SLOT_PROTOTYPE' if go else 'CONDITIONAL_OR_STOP_L5_SLOT_PERFORMANCE',
 evidence_class='GPU1 shared-host captured-L5 research prototype; not runtime integration or promotion',
 native_sha256=build['native_sha256'],prototype_sha256=build['outputs'],device_functions_exact=14,
 setups=955,targets=6816,slots=32,target_capacity=8,actual_target_counts=[2,3,4,8],
 all_outputs_bitwise_exact=True,hard_gates='PASS',resources_closed=True,forbidden_calls=0,
 completed_full_layer_runs=15,compared_target_occurrences=sum(r['compared_target_occurrences'] for r in reports.values()),
 reduced_memcheck_errors=0,reduced_memcheck_leaked_bytes=0,
 cancellation_and_token_checks='PASS',capacity=capacity,seconds=summary,S_over_R_work_ratio=ratio,
 S_over_R_lifecycle_ratio=life_ratio,paired_work_ratios=ratios,paired_S_lifecycle_wins=lifecycle_wins,
 predeclared_work_ratio_limit=1.15,predeclared_capacity_increment_bytes=1024**3,
 full_output_arena_bytes=40786944,processes=processes,telemetry=telemetry,
 review_corrections=read('review-audit.json'),production_sources_changed=False,
 joint_runtime_capacity_qualified=False,other_layers_qualified=False,
 runtime_rolling='NOT_YET_GO',phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
references=[root/'fastkpc/artifacts/full_cuda_ci'/name/'manifest.json' for name in
 ['optimizer_layer_refill_20261023','optimizer_layer_refill_review_20261023','optimizer_status_wait_20261023']]
manifest=dict(schema='optimizer-slot-pool-manifest-v1',native_sha256=build['native_sha256'],source_sha256=build['source_sha256'],
 corrected_review={str(p.relative_to(root)):sha(p) for p in [root/'fastkpc/docs/PHASE10_OPTIMIZER_LAYER_REFILL_REVIEW_20261023.md']},
 tools={str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()},
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'},
 references={str(p.relative_to(root)):sha(p) for p in references})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['processes','seconds','review_corrections']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
