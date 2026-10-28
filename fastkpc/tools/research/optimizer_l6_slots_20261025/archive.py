"""Audit captured L6 qualification without claiming full-runtime integration."""
import csv
import json
from pathlib import Path
import statistics
import sys
from common import CAPTURE, ROOT, TOOL, check_identity, load, sha, write_new

out=Path(sys.argv[1]).resolve();build=check_identity(out);protocol=load(out/'protocol.json')
assert protocol['build_sha256']==sha(out/'build.json') and protocol['declaration_sha256']==sha(TOOL/'README.md')
assert protocol['capture_sha256']==sha(CAPTURE/'capture-closure.json')
device=load(out/'device-code.json')
assert device['status']=='PASS' and device['original']==device['prototype'] and len(device['original'])==14
modes=['checks','memcheck','capacity','campaign']
reports={n:load(out/(n+'-summary.json')) for n in modes}
processes={n:load(out/(n+'-process.json')) for n in modes}
contexts={n:load(out/(n+'-context.json')) for n in modes}
for n,r in reports.items():
    assert r['status']=='PASS' and r['resources_closed'] and r['outputs_bitwise_exact'] and r['forbidden_calls']==0,n
    p=processes[n]
    assert p['exit_code']==0 and p['physical_gpu']==1 and not p['unexpected_gpu_pids']
    assert p['protocol_sha256']==sha(out/'protocol.json') and p['native_sha256']==build['native_sha256']
    c=contexts[n]
    assert c['native_warm_targets']==5 and c['native_warm_outputs_exact'] and c['native_warm_resources_closed']
    assert c['compute_warm_free_bytes']>=2*1024**3
    assert c['common_retained_bytes']==max(0,c['cold_context_free_bytes']-c['compute_warm_free_bytes'])
for n,runs in [('checks',3),('memcheck',1)]:
    r=reports[n]
    assert r['setups_per_run']==4 and r['runs']==runs and r['negative_checks']==10 and r['contract_compared_targets']==9
    assert all(r[k] for k in ['cancellation_checked','partial_upload_checked','partial_detach_checked'])
for n,runs in [('capacity',1),('campaign',14)]:
    r=reports[n]
    assert r['setups_per_run']==245 and r['runs']==runs and r['compared_target_occurrences']==1148*runs
mem=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
rows=lambda p:list(csv.DictReader((out/p).open()))
capacity=rows('capacity-runs.csv');assert len(capacity)==1;capacity=capacity[0]
assert int(capacity['physical_increment_bytes'])<=1024**3 and int(capacity['free_after_create'])>=2*1024**3
campaign=rows('campaign-runs.csv');timed=[r for r in campaign if r['kind']=='timing']
assert len(campaign)==14 and len(timed)==12
assert [(r['variant'],int(r['cycle'])) for r in campaign]==[('W',-1),('S',-1)]+[
    (v,i) for i in range(6) for v in ('WS' if i%2==0 else 'SW')]
for r in [capacity,*campaign]:
    assert int(r['targets'])==1148 and int(r['iterations'])==17062
    assert int(r['rebindings'])==(245 if r['variant']=='S' else 0)
    assert int(r['stale_checks'])==(245 if r['variant']=='S' else 0)
    assert int(r['free_after_create'])>=2*1024**3
    assert int(r['streams'])==(32 if r['variant']=='S' else 64)
    if r['variant']=='S':assert int(r['physical_increment_bytes'])<=1024**3
details=rows('campaign-setups.csv')
for cycle in range(-1,6):
    for variant in 'WS':
        g=[r for r in details if int(r['cycle'])==cycle and r['variant']==variant]
        assert len(g)==245 and [int(r['request']) for r in g]==list(range(245))
        if variant=='S':
            generations={}
            for r in g:
                k,generation=int(r['slot']),int(r['generation'])
                assert 0<=k<32 and generation>generations.get(k,0)
                generations[k]=generation
            assert len(generations)==32
summary={}
for v in 'WS':
    g=[r for r in timed if r['variant']==v]
    summary[v]={k:dict(median=statistics.median(float(r[k])/1000 for r in g),
                      minimum=min(float(r[k])/1000 for r in g),maximum=max(float(r[k])/1000 for r in g))
                for k in ['work_ms','prepare_ms','destroy_ms','packet_pack_ms','lifecycle_ms','verify_ms',
                          'bind_sum_ms','optimize_sum_ms','detach_sum_ms']}
pairs=[]
for i in range(6):
    p={r['variant']:r for r in timed if int(r['cycle'])==i}
    pairs.append(dict(cycle=i,work_ratio=float(p['S']['work_ms'])/float(p['W']['work_ms']),
      lifecycle_ratio=float(p['S']['lifecycle_ms'])/float(p['W']['lifecycle_ms']),
      lifecycle_saved_sec=(float(p['W']['lifecycle_ms'])-float(p['S']['lifecycle_ms']))/1000))
ratio=summary['S']['lifecycle_ms']['median']/summary['W']['lifecycle_ms']['median']
wins=sum(p['lifecycle_saved_sec']>0 for p in pairs)
status='GO_CAPTURED_L6_SLOTS' if ratio<=.90 and wins==6 else 'CONDITIONAL_CAPTURED_L6_SLOTS' if ratio<1 else 'STOP_CAPTURED_L6_PERFORMANCE'
closure=dict(schema='optimizer-l6-slots-closure-v1',status=status,
  evidence_class='GPU1-exclusive captured layer, host sharing recorded; not canonical runtime or promotion',
  native_sha256=build['native_sha256'],source_sha256=build['source_sha256'],
  build_sha256=sha(out/'build.json'),protocol_sha256=sha(out/'protocol.json'),capture_sha256=sha(CAPTURE/'capture-closure.json'),
  device_functions_exact=14,setups=245,targets=1148,iterations=17062,slots=32,target_capacity=5,
  hard_gates='PASS',outputs_bitwise_exact=True,resources_closed=True,forbidden_calls=0,
  memcheck_errors=0,memcheck_leaked_bytes=0,capacity=capacity,
  compared_target_occurrences=sum(r['compared_target_occurrences']+r['contract_compared_targets'] for r in reports.values())+5*len(modes),
  native_warmup_targets_per_process=5,seconds=summary,S_over_W_lifecycle_ratio=ratio,
  S_over_W_work_ratio=summary['S']['work_ms']['median']/summary['W']['work_ms']['median'],paired_wins=wins,pairs=pairs,
  reports=reports,processes=processes,context_memory=contexts,production_sources_changed=False,joint_runtime_capacity_qualified=False,
  runtime_integration='NOT_YET_GO',new_canonical_seconds=None,accepted_canonical_median_sec=407.531,
  phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
write_new(out/'closure.json',closure)
write_new(out/'manifest.json',dict(schema='optimizer-l6-slots-manifest-v1',native_sha256=build['native_sha256'],
  tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
  files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'},
  references={str(CAPTURE/'capture-closure.json'):sha(CAPTURE/'capture-closure.json')}))
print(json.dumps(dict(status=status,seconds=summary,lifecycle_ratio=ratio,paired_wins=wins,
                     manifest_sha256=sha(out/'manifest.json')),indent=2))
