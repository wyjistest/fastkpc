"""Audit the six untraced runs and apply the predeclared small-optimization gate."""
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
read=lambda p:json.loads((out/p).read_text())
build,checks,protocol=(read(p) for p in ['build.json','checks.json','protocol.json'])
assert build['status']==checks['status']=='PASS'
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']==checks['native_sha256']
assert sha(out/'stats.so')==checks['stats_sha256']==protocol['stats_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert all(sha(root/p)==h for p,h in protocol['frozen_tools'].items())
assert all(sha(root/p)==h for p,h in protocol['frozen_dependencies'].items())
reports,processes,peaks,references={},{},{},{}
for name in protocol['sequence']:
    r,p=read(name+'/report.json'),read(name+'-process.json')
    assert r['gates'] and all(r['gates'].values()),name
    assert sha(out/name/'receipt.rds')==r['receipt_sha256']
    assert r['runner_sha256']==sha(tool/'run.R')
    ref=root/p['command'][-2]
    assert r['reference_sha256']==sha(ref)
    references[str(ref.relative_to(root))]=sha(ref)
    assert r['source_sha256']==build['source_sha256']
    assert r['native_sha256']==build['native_sha256']==p['native_sha256']
    assert p['exit_code']==0 and p['physical_gpu']==1 and p['isolated']
    assert p['l4_slots']==p['l5_slots']==p['l4_workers32']=='1'
    assert r['l4_slots']==r['l5_slots']=='1'
    mode='1' if name.startswith('on') else '0'
    assert r['properties_mode']==p['properties_cache']==mode
    stats=r['properties_cache']
    assert stats['failures']==0 and stats['requests']==stats['queries']+stats['hits']
    if mode=='1':assert stats['enabled'] and stats['queries']==stats['entries']==1
    else:assert not stats['enabled'] and stats['hits']==stats['entries']==0
    assert r['comparisons']['changed_ci_count']==r['comparisons']['changed_pmax_count']==0
    assert r['comparisons']['max_ci_difference']==r['comparisons']['alpha_flips']==0
    assert len(r['oracle_errors'])==2 and max(r['oracle_errors'])<=1e-10
    for field,filename in [('logical_ci_trace','logical_ci_trace.rds'),('pmax','pmax.rds'),('adjacency','adjacency.rds')]:
        path=root/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/filename
        assert r['authority_sha256'][field]==sha(path)
        references[str(path.relative_to(root))]=sha(path)
    frozen=root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds'
    assert r['comparisons']['frozen_l2']['receipt_sha256']==sha(frozen)
    references[str(frozen.relative_to(root))]=sha(frozen)
    with (out/(name+'-gpu.csv')).open() as stream:
        peaks[name]=max(int(row[2]) for row in csv.reader(stream) if len(row)==7 and int(row[1])==1)
    reports[name],processes[name]=r,p
assert len({r['properties_cache']['requests'] for r in reports.values()})==1
medians,ranges={},{}
for mode in ['off','on']:
    group=[reports[mode+str(i)] for i in range(1,4)]
    medians[mode]={k:statistics.median(r['stages'][k] for r in group) for k in
        ['elapsed_sec','optimizer_sec','stage_a_sec','stage_b_sec','native_setup_sec','prepared_build_sec']}
    medians[mode]['optimizer_by_level_sec']={str(level):statistics.median(
        next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==level)
        for r in group) for level in range(1,9)}
    ranges[mode]=max(r['elapsed_sec'] for r in group)-min(r['elapsed_sec'] for r in group)
saving=medians['off']['elapsed_sec']-medians['on']['elapsed_sec']
required=max(protocol['minimum_saving_seconds'],
    protocol['minimum_saving_fraction']*medians['off']['elapsed_sec'],*ranges.values())
pair_deltas={str(i):reports['off'+str(i)]['elapsed_sec']-reports['on'+str(i)]['elapsed_sec']
             for i in range(1,4)}
wins=sum(x>0 for x in pair_deltas.values())
if saving>=required and wins==protocol['required_pair_wins']:
    status='GO_DEVICE_PROPERTIES_CACHE_DEVELOPMENT'
elif saving<=0:
    status='STOP_DEVICE_PROPERTIES_CACHE_WALL_TIME'
else:
    status='CONDITIONAL_DEVICE_PROPERTIES_CACHE_PERFORMANCE'
closure=dict(schema='device-properties-cache-closure-v1',status=status,hard_gates='PASS',
    evidence_class='GPU1 isolated audited incremental source-closure development; not clean producer or promotion',
    native_sha256=build['native_sha256'],producer_head=build['producer_head'],
    build_lineage=build['source_build'],full_canonical_runs=6,logical_ci_per_run=240498,
    ci_and_pmax_bits_exact=True,seconds_medians=medians,seconds_within_mode_ranges=ranges,
    median_saving_seconds=saving,median_saving_fraction=saving/medians['off']['elapsed_sec'],
    required_saving_seconds=required,paired_saving_seconds=pair_deltas,paired_on_wins=wins,
    elapsed_sec_by_run={n:r['elapsed_sec'] for n,r in reports.items()},
    cache_stats_by_run={n:r['properties_cache'] for n,r in reports.items()},
    gpu1_sampled_total_peak_mib=peaks,
    authority_max_ci_error=max(r['oracle_errors'][0] for r in reports.values()),
    authority_max_pmax_error=max(r['oracle_errors'][1] for r in reports.values()),
    checks=checks,processes=processes,default_enabled=False,
    qualified_scope='canonical S2, L4/L5 slots and status wait enabled; only properties cache differs',
    new_real_data_qualified=False,new_final_pdag_qualified=False,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
assert not (out/'closure.json').exists(),'preserve previous closure'
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
references[build['parent_build']]=sha(root/build['parent_build'])
assert references[build['parent_build']]==build['parent_build_sha256']
manifest=dict(schema='device-properties-cache-manifest-v1',native_sha256=build['native_sha256'],
    source_sha256=build['source_sha256'],tools=protocol['frozen_tools'],
    dependencies=protocol['frozen_dependencies'],references=references,
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['checks','processes']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
