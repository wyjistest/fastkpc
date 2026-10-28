"""Machine closure of the frozen L3 captured experiment, without runtime promotion."""
import csv
from pathlib import Path
import statistics as stats
import sys
from common import ROOT,TOOL,sha,load,write_new,check_identity

out=Path(sys.argv[1]).resolve();build=check_identity(out);protocol=load(out/'protocol.json')
assert protocol['build_sha256']==sha(out/'prototype-build.json')
assert protocol['declaration_sha256']==sha(TOOL/'README.md')
assert protocol['capture_sha256']==sha(out/'capture-closure.json')
device=load(out/'device-code.json')
assert device['status']=='PASS' and device['original']==device['prototype'] and len(device['original'])==14
modes=['checks','memcheck','capacity','campaign','trace-W','trace-S']
reports={n:load(out/(n+'-summary.json')) for n in modes}
processes={n:load(out/(n+'-process.json')) for n in modes}
contexts={n:load(out/(n+'-context.json')) for n in modes}
for n,r in reports.items():
    assert r['status']=='PASS' and r['resources_closed'] and r['outputs_bitwise_exact'] and r['forbidden_calls']==0,n
    p=processes[n]
    assert p['exit_code']==0 and p['isolated'] and p['physical_gpu']==1 and not p['unexpected_gpu_pids']
    assert p['protocol_sha256']==sha(out/'protocol.json') and p['native_sha256']==build['native_sha256']
    c=contexts[n];traced=n.startswith('trace-')
    assert c['native_warm_targets']==(0 if traced else 33) and c['native_warm_resources_closed']
    assert c['native_warm_outputs_exact']==(not traced) and c['trace_cold_context']==traced
    assert c['compute_warm_free_bytes']>=2*1024**3
rows=lambda name:list(csv.DictReader((out/name).open()))
metadata=rows('campaign-metadata.csv')
assert len(metadata)==4065 and [int(r['request']) for r in metadata]==list(range(4065))
assert sum(int(r['targets']) for r in metadata)==54878
assert sum(int(r['iterations']) for r in metadata)==protocol['iterations']
first={}
for i,r in enumerate(metadata):first.setdefault(int(r['targets']),i)
reduced=sorted(set(first.values())|{4064});reduced_targets=sum(int(metadata[i]['targets']) for i in reduced)
for n,count in [('checks',3),('memcheck',1)]:
    r=reports[n]
    assert r['runs']==count and r['setups_per_run']==len(reduced)
    assert r['compared_target_occurrences']==reduced_targets*count
    assert r['negative_checks']==10 and r['contract_compared_targets']==37
    assert all(r[k] for k in ['cancellation_checked','partial_upload_checked','partial_detach_checked'])
for n,count in [('capacity',1),('campaign',14),('trace-W',1),('trace-S',1)]:
    r=reports[n]
    assert r['runs']==count and r['setups_per_run']==4065 and r['compared_target_occurrences']==54878*count
mem=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
capacity=rows('capacity-runs.csv');assert len(capacity)==1;capacity=capacity[0]
campaign=rows('campaign-runs.csv');timed=[r for r in campaign if r['kind']=='timing']
assert len(campaign)==14 and len(timed)==12
assert [(r['variant'],int(r['cycle'])) for r in campaign]==[('W',-1),('S',-1)]+[
    (v,i) for i in range(6) for v in ('WS' if i%2==0 else 'SW')]
for r in [capacity,*campaign]:
    assert int(r['targets'])==54878 and int(r['iterations'])==protocol['iterations']
    assert int(r['rebindings'])==(4065 if r['variant']=='S' else 0)
    assert int(r['stale_checks'])==(4065 if r['variant']=='S' else 0)
    assert int(r['free_after_create'])>=2*1024**3
    assert int(r['streams'])==(32 if r['variant']=='S' else 64)
    if r['variant']=='S':assert int(r['physical_increment_bytes'])<=2*1024**3
details=rows('campaign-setups.csv')
services={}
for cycle in range(-1,6):
    for variant in 'WS':
        g=[r for r in details if int(r['cycle'])==cycle and r['variant']==variant]
        assert len(g)==4065 and [int(r['request']) for r in g]==list(range(4065))
        if variant=='S':
            generations={}
            for r in g:
                k,generation=int(r['slot']),int(r['generation'])
                assert 0<=k<32 and generation>generations.get(k,0);generations[k]=generation
            assert len(generations)==32
        if cycle>=0:
            for r in g:services.setdefault((variant,int(r['request'])),[]).append(float(r['optimize_ms']))
def distribution(values):
    values=sorted(values);return dict(minimum=values[0],median=stats.median(values),
        p95=values[min(len(values)-1,int(.95*len(values)))],maximum=values[-1])
service_ratios=[stats.median(services['S',i])/stats.median(services['W',i]) for i in range(4065)]
service_by_count={str(t):dict(setups=sum(int(r['targets'])==t for r in metadata),
    host_service_S_over_W=distribution([service_ratios[i] for i,r in enumerate(metadata) if int(r['targets'])==t])) for t in sorted(first)}
summary={}
for v in 'WS':
    g=[r for r in timed if r['variant']==v]
    summary[v]={k:distribution([float(r[k])/1000 for r in g]) for k in
        ['work_ms','prepare_ms','destroy_ms','packet_pack_ms','lifecycle_ms','verify_ms','bind_sum_ms','optimize_sum_ms','detach_sum_ms']}
pairs=[]
for i in range(6):
    p={r['variant']:r for r in timed if int(r['cycle'])==i}
    pairs.append(dict(cycle=i,lifecycle_ratio=float(p['S']['lifecycle_ms'])/float(p['W']['lifecycle_ms']),
        lifecycle_saved_sec=(float(p['W']['lifecycle_ms'])-float(p['S']['lifecycle_ms']))/1000))
ratio=summary['S']['lifecycle_ms']['median']/summary['W']['lifecycle_ms']['median']
wins=sum(p['lifecycle_saved_sec']>0 for p in pairs)
status='GO_CAPTURED_L3_SLOTS' if ratio<=.90 and wins==6 else 'CONDITIONAL_CAPTURED_L3_SLOTS' if ratio<1 else 'STOP_CAPTURED_L3_PERFORMANCE'
trace=load(out/'trace-analysis.json')
for v in 'WS':assert trace['results'][v]['persistent_kernels']==4065 and trace['results'][v]['dropped']==0
for p,h in trace['files'].items():assert sha(out/p)==h,p
closure=dict(schema='optimizer-l3-slots-closure-v1',status=status,evidence_class='isolated captured cohort; not full-method or promotion',
    native_sha256=build['native_sha256'],build_sha256=sha(out/'prototype-build.json'),protocol_sha256=sha(out/'protocol.json'),
    capture_sha256=sha(out/'capture-closure.json'),device_functions_exact=14,setups=4065,targets=54878,
    iterations=protocol['iterations'],slots=32,target_capacity=33,hard_gates='PASS',outputs_bitwise_exact=True,
    resources_closed=True,forbidden_calls=0,memcheck_errors=0,memcheck_leaked_bytes=0,capacity=capacity,
    diagnostic_output_arena_bytes=(28+2*351)*54878*8,
    compared_target_occurrences=sum(r['compared_target_occurrences']+r['contract_compared_targets'] for r in reports.values())+33*4,
    seconds=summary,S_over_W_lifecycle_ratio=ratio,paired_wins=wins,pairs=pairs,
    host_service_ratio=distribution(service_ratios),host_service_by_target_count=service_by_count,
    host_service_note='paired per-request medians include device queuing and host API; not isolated device contention',
    reports=reports,processes=processes,context_memory=contexts,trace_analysis_sha256=sha(out/'trace-analysis.json'),
    production_sources_changed=False,joint_runtime_capacity_qualified=False,runtime_integration='NOT_YET_GO',
    new_canonical_seconds=None,accepted_canonical_median_sec=368.851,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
write_new(out/'closure.json',closure)
write_new(out/'manifest.json',dict(schema='optimizer-l3-slots-manifest-v1',native_sha256=build['native_sha256'],
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}))
print(status,dict(lifecycle_ratio=ratio,paired_wins=wins,seconds=summary,manifest_sha256=sha(out/'manifest.json')))
