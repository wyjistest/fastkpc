"""Join authenticated setup/target metadata with exact CUPTI external IDs."""
import collections
import csv
import json
from pathlib import Path
import statistics
import sys
import numpy as np
from legacy import KERNEL, RUNTIME, read_trace

EXTERNAL=np.dtype([('external','<u8'),('corr','<u4'),('kind','<u4')])
PHASES=['penalty_cycles','svd_cycles','score_cycles','derivative_cycles']
def read_tsv(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def quantiles(xs):
    return dict(zip(['min','p50','p90','p99','max'],map(float,np.percentile(xs,[0,50,90,99,100]))))
def corr(xs,ys):
    return float(np.corrcoef(xs,ys)[0,1]) if np.std(xs)>0 and np.std(ys)>0 else None
def fraction(n,d):return n/d if d else None
def union(intervals):
    result=0;end=0
    for a,b in sorted(intervals):
        result+=max(0,b-max(a,end));end=max(b,end)
    return result
def independent_union(intervals):
    events=collections.Counter()
    for a,b in intervals:events[a]+=1;events[b]-=1
    active=0;last=0;total=0
    for t,delta in sorted(events.items()):
        if active:total+=t-last
        active+=delta;last=t
        assert active>=0
    assert active==0
    return total

out=Path(sys.argv[1]).resolve();run=out/'canonical'
metadata=json.loads((run/'observer.json').read_text());report=json.loads((run/'report.json').read_text())
trace=json.loads((run/'trace.json').read_text())
assert metadata['complete'] and metadata['canonical_request_order_exact']
assert all(report['gates'].values()) and trace['dropped']==trace['invalid']==0
rows=[{k:int(v) for k,v in r.items()} for r in read_tsv(run/'setups.tsv')]
targets=[{k:(v if k=='key' else int(v)) for k,v in r.items()} for r in read_tsv(run/'targets.tsv')]
assert len(rows)==3109 and len(targets)==24583 and len({r['external_id'] for r in rows})==3109
by_id={r['external_id']:r for r in rows}
by_target=collections.defaultdict(list)
for t in targets:
    t['phase_cycles']=sum(t[k] for k in PHASES)
    t['evaluations']=t['complete_evaluations']+t['score_only_evaluations']
    t['cycles_per_evaluation']=t['phase_cycles']/t['evaluations']
    assert t['route']==2 and t['info']==t['optimizer_status']==0 and t['guarded_qr_evaluations']==0
    by_target[t['level']*1000000+t['request_index']].append(t)
names,k,r=read_trace(run)
e=np.fromfile(run/'external.bin',dtype=EXTERNAL)
assert len(k)==trace['kernels'] and len(r)==trace['runtime'] and len(e)==trace['external']
external={}
for x in e:
    ci,ext=int(x['corr']),int(x['external'])
    assert ext in by_id
    assert ci not in external or external[ci]==ext
    external[ci]=ext
groups=collections.defaultdict(list)
for i,x in enumerate(k):
    ci=int(x['corr'])
    if ci in external:groups[external[ci]].append(i)
assert set(groups)==set(by_id)
api={int(x['corr']):x for x in r if int(x['corr']) in external}
assert len(api)==len({int(x['corr']) for x in r if int(x['corr']) in external})
paired=0
thread_mapping={}
cross_domain_end_violations=[]
for ext,row in by_id.items():
    indexes=groups[ext];opts=[i for i in indexes if 'optimize_multi_penalty_targets_kernel' in names[int(k[i]['name'])]]
    assert len(opts)==1
    opt=k[opts[0]];calls=by_target[ext]
    assert len(calls)==row['targets'] and int(opt['grid'])==row['targets'] and int(opt['block'])==64 and int(opt['regs'])==255
    assert row['enter_ns']<=int(opt['start'])<=int(opt['end'])
    assert row['enter_ns']<=row['return_ns']<=row['observer_end_ns']
    # Preserve, rather than silently clip, CUPTI device/host boundary discrepancies.
    # Identity comes from external IDs, never a timestamp-containment heuristic.
    if int(opt['end'])>row['return_ns']:
        cross_domain_end_violations.append(dict(external_id=ext,
            kernel_end_after_host_return_ns=int(opt['end'])-row['return_ns']))
    assert row['bind_enter_ns']<=row['bind_return_ns']<=row['enter_ns'] and row['bind_enter_ns']>0
    for i in indexes:
        a=api[int(k[i]['corr'])]
        assert row['enter_ns']<=int(a['start'])<=int(a['end'])<=row['return_ns']
        assert int(a['start'])<=int(k[i]['start'])
        # CUPTI threadId and Linux gettid use different ID spaces. Worker pthread
        # IDs may also be reused between levels; verify a bijection within each.
        thread_key=(row['level'],int(a['thread']))
        assert thread_key not in thread_mapping or thread_mapping[thread_key]==row['thread_id']
        thread_mapping[thread_key]=row['thread_id']
        paired+=1
    assert len({int(k[i]['stream']) for i in indexes})==1
    first=min(int(k[i]['start']) for i in indexes);last=max(int(k[i]['end']) for i in indexes)
    busy=sum(int(k[i]['end'])-int(k[i]['start']) for i in indexes)
    heaviest=max(calls,key=lambda t:t['phase_cycles'])
    summed=sum(t['phase_cycles'] for t in calls);maximal=heaviest['phase_cycles']
    row.update(device_first_ns=first,device_last_ns=last,optimizer_start_ns=int(opt['start']),optimizer_end_ns=int(opt['end']),
        optimizer_sec=float(opt['end']-opt['start'])/1e9,kernel_busy_sec=busy/1e9,stream_span_sec=(last-first)/1e9,
        host_batch_sec=(row['return_ns']-row['enter_ns'])/1e9,bind_sec=(row['bind_return_ns']-row['bind_enter_ns'])/1e9,
        observer_tail_sec=(row['observer_end_ns']-row['return_ns'])/1e9,kernels=len(indexes),cupti_stream=int(opt['stream']),
        critical_candidate_target=heaviest['target_index'],critical_candidate_key=heaviest['key'],
        max_target_phase_cycles=maximal,sum_target_phase_cycles=summed,
        max_to_mean_target_phase_ratio=maximal*len(calls)/summed,max_to_median_target_phase_ratio=maximal/statistics.median(t['phase_cycles'] for t in calls),
        max_target_iterations=max(t['iterations'] for t in calls),max_target_evaluations=max(t['evaluations'] for t in calls),
        critical_candidate_iterations=heaviest['iterations'],critical_candidate_evaluations=heaviest['evaluations'],
        critical_candidate_step_halving=heaviest['step_halving'],
        critical_candidate_cycles_per_evaluation=heaviest['cycles_per_evaluation'])

levels={}
for lv,expected_setups,expected_targets in [(4,2154,17767),(5,955,6816)]:
    setup=[x for x in rows if x['level']==lv];target=[x for x in targets if x['level']==lv]
    assert len(setup)==expected_setups and len(target)==expected_targets
    assert len({x['prepared'] for x in setup})==len({x['cupti_stream'] for x in setup})==32
    mapping={k[1]:v for k,v in thread_mapping.items() if k[0]==lv}
    assert len(mapping)==len(set(mapping.values()))==32
    address0=next(x['geometry_address'] for x in setup if x['request_index']==0)
    assert all(x['geometry_address']==address0+x['request_index']*metadata['geometry_stride_bytes'] for x in setup)
    start=min(x['device_first_ns'] for x in setup);end=max(x['device_last_ns'] for x in setup)
    for rank,x in enumerate(sorted(setup,key=lambda x:x['device_first_ns'])):
        x.update(observed_start_rank=rank,start_sec=(x['device_first_ns']-start)/1e9,
            end_sec=(x['device_last_ns']-start)/1e9,
            optimizer_start_sec=(x['optimizer_start_ns']-start)/1e9)
    all_intervals=[(int(k[i]['start']),int(k[i]['end'])) for x in setup for i in groups[x['external_id']]]
    opts=[(x['optimizer_start_ns'],x['optimizer_end_ns']) for x in setup]
    assert union(all_intervals)==independent_union(all_intervals)
    assert union(opts)==independent_union(opts)
    phase={p:sum(t[p] for t in target) for p in PHASES};total=sum(phase.values())
    decomp={str(d):sum(t['decomposition_'+str(d)] for t in target) for d in range(8)}
    candidates=[by_target[x['external_id']][x['critical_candidate_target']] for x in setup]
    candidate_phase={p:sum(t[p] for t in candidates) for p in PHASES};ctotal=sum(candidate_phase.values())
    counts={p:sum(t[p] for t in target) for p in ['iterations','score_calls','objective_calls','step_halving','newton_trials',
        'steepest_trials','boundary_probes','boundary_accepted','complete_evaluations','score_only_evaluations','stable_svd_evaluations','selected_reuse','selected_refresh']}
    longest=sorted(setup,key=lambda x:x['optimizer_sec'],reverse=True)
    tail=sorted(setup,key=lambda x:x['device_last_ns'],reverse=True)
    levels[str(lv)]=dict(setups=len(setup),targets=len(target),kernel_span_sec=(end-start)/1e9,
        host_optimizer_boundary_sec=next(x['host_sec'] for x in report['stages']['optimizer_by_level'] if x['level']==lv),
        kernel_union_sec=union(all_intervals)/1e9,optimizer_union_sec=union(opts)/1e9,
        setup_optimizer_sec=quantiles([x['optimizer_sec'] for x in setup]),
        target_iterations=quantiles([t['iterations'] for t in target]),target_evaluations=quantiles([t['evaluations'] for t in target]),
        target_cycles_per_evaluation=quantiles([t['cycles_per_evaluation'] for t in target]),
        critical_candidate_to_mean_ratio=quantiles([x['max_to_mean_target_phase_ratio'] for x in setup]),
        phase_cycles=phase,phase_fractions={p:v/total for p,v in phase.items()},
        critical_candidate_phase_fractions={p:v/ctotal for p,v in candidate_phase.items()},
        decomposition_cycles=decomp,decomposition_fractions_of_svd={p:v/phase['svd_cycles'] for p,v in decomp.items()},
        counts=counts,longest_setups=longest[:20],latest_finishes=tail[:10],
        phase_clock_ratio_to_kernel_hz=quantiles([x['max_target_phase_cycles']/x['optimizer_sec'] for x in setup]),
        correlations=dict(kernel_vs_max_iterations=corr([x['optimizer_sec'] for x in setup],[x['max_target_iterations'] for x in setup]),
            kernel_vs_max_evaluations=corr([x['optimizer_sec'] for x in setup],[x['max_target_evaluations'] for x in setup]),
            kernel_vs_max_target_cycles=corr([x['optimizer_sec'] for x in setup],[x['max_target_phase_cycles'] for x in setup])),
        reordered_gpu_start_count=sum(x['observed_start_rank']!=x['request_index'] for x in setup),
        maximum_start_rank_displacement=max(abs(x['observed_start_rank']-x['request_index']) for x in setup),
        observer_tail_sum_sec=sum(x['observer_tail_sec'] for x in setup))
with (out/'joined-setups.tsv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
with (out/'joined-targets.tsv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(targets[0]),delimiter='\t');w.writeheader();w.writerows(targets)
value=dict(status='PASS_L45_IDENTITY_AND_EXISTING_PHASE_ATTRIBUTION',native_sha256=report['native_sha256'],
    correlated_kernels=paired,external_correlations=len(e),levels=levels,
    thread_identity_check='32-to-32 CUPTI/Linux ID bijection per level; different numeric ID spaces',
    cross_domain_end_violations=cross_domain_end_violations,
    limitations=['diagnostic run, not performance evidence','phase cycles overlap across targets and include stalls',
      'no per-target start/end timestamps; largest phase sum is a candidate critical target',
      'selected refresh count is known, its cycles are not separately timed','no measured occupancy, stall or bandwidth counters',
      'CUPTI device ends can exceed host return timestamps; no cross-domain wait/gap inference'],
    numerical_gates=report['gates'],oracle_errors=report['oracle_errors'],observer=metadata)
with (out/'analysis.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({lv:{'span':v['kernel_span_sec'],'phases':v['phase_fractions'],'counts':v['counts'],
    'correlations':v['correlations'],'slowest':v['longest_setups'][0],'latest':v['latest_finishes'][0]} for lv,v in levels.items()},indent=2))
