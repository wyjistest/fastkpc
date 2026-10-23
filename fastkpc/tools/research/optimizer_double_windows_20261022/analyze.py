"""Analyze primary timings and separately observed device/host timelines."""
import collections
import csv
import json
from pathlib import Path
import sqlite3
import statistics
import sys

out=Path(sys.argv[1]).resolve()
def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def describe(xs):
    assert xs
    return dict(values=xs,min=min(xs),median=statistics.median(xs),max=max(xs))
def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>=end:
            total+=b-a;end=b
        elif b>end:
            total+=b-end;end=b
    return total
def concurrency(intervals):
    events=collections.defaultdict(int)
    for a,b in intervals:
        events[a]+=1;events[b]-=1
    active=peak=0
    for t,delta in sorted(events.items()):
        active+=delta;peak=max(peak,active)
    assert active==0
    return peak

report=dict(schema='optimizer-double-window-analysis-v1',pairs={},profiles={})
for name in ('L3-W31+W32','L5-W7+W8'):
    runs=rows(out/f'campaign-{name}-runs.csv')
    assert len(runs)==24 and all(r['parity']=='1' for r in runs)
    timings={v:describe([float(r['wall_ms']) for r in runs if r['kind']=='timing' and r['variant']==v]) for v in 'ABC'}
    assert all(len(t['values'])==6 for t in timings.values())
    paired={}
    for variant in 'BC':
        diffs=[];ratios=[]
        for cycle in range(6):
            cycle_rows={r['variant']:float(r['wall_ms']) for r in runs if r['kind']=='timing' and int(r['cycle'])==cycle}
            diffs.append(cycle_rows['A']-cycle_rows[variant])
            ratios.append(cycle_rows['A']/cycle_rows[variant])
        paired[variant]=dict(A_minus_variant_ms=describe(diffs),A_over_variant=describe(ratios),wins=sum(d>0 for d in diffs))
    summary=json.loads((out/f'campaign-{name}-summary.json').read_text())
    report['pairs'][name]=dict(timings_ms=timings,paired_cycles=paired,summary=summary,
        first_window_prepared_delta_bytes=summary['free_before']-summary['free_first_window'],
        second_window_prepared_delta_bytes=summary['free_first_window']-summary['free_both_windows'],
        median_speedup_vs_A={v:timings['A']['median']/timings[v]['median'] for v in 'BC'})
    print(name,'primary medians ms',{v:round(t['median'],3) for v,t in timings.items()},flush=True)

for path in sorted(out.glob('profile-*.sqlite')):
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=[dict(start=s,end=e,label=t or strings.get(ti,''),tid=tid)
            for s,e,t,ti,tid in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS') if e is not None]
    pairs=sorted([r for r in ranges if r['label'].startswith('pair:')],key=lambda r:r['start'])
    assert len(pairs)==3
    api={co:dict(start=s,end=e,tid=tid,name=strings[n]) for s,e,tid,co,n in
         db.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    kernels=[dict(start=s,end=e,stream=st,corr=co,name=strings[n],grid=g)
             for s,e,st,co,n,g in db.execute('select start,end,streamId,correlationId,shortName,gridX*gridY*gridZ from CUPTI_ACTIVITY_KIND_KERNEL')]
    copies=[dict(start=s,end=e,stream=st,corr=co,kind=kind,bytes=size)
            for s,e,st,co,kind,size in db.execute('select start,end,streamId,correlationId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY')]
    pair_report={}
    for pair in pairs:
        start,end=pair['start'],pair['end'];variant=pair['label'].split(':')[-2]
        assert variant in 'ABC'
        setups=[r for r in ranges if r['label'].startswith('setup:') and start<=r['start']<r['end']<=end]
        ks=[k for k in kernels if start<=k['start']<k['end']<=end]
        opts=[k for k in ks if 'optimize_multi_penalty_targets' in k['name']]
        ms=[m for m in copies if start<=m['start']<m['end']<=end]
        assert len(setups)==len(opts)==128
        timeline=[]
        for k in opts:
            launch=api[k['corr']]
            matches=[s for s in setups if s['tid']==launch['tid'] and s['start']<=launch['start']<s['end']]
            assert len(matches)==1
            setup=matches[0];index=int(setup['label'].split(':')[1])
            own_ks=[x for x in ks if x['stream']==k['stream']]
            own_ms=[m for m in ms if m['stream']==k['stream']]
            h2d=[m for m in own_ms if m['kind']==1];d2h=[m for m in own_ms if m['kind']==2]
            assert len(h2d)==len(d2h)==1
            upload,download=h2d[0],d2h[0];hu=api[upload['corr']];hd=api[download['corr']]
            row=dict(setup=index,window_side=int(index>=64),stream=k['stream'],grid_blocks=k['grid'])
            absolute=dict(host_start=setup['start'],host_return=setup['end'],
                h2d_api_enter=hu['start'],h2d_api_return=hu['end'],h2d_start=upload['start'],h2d_end=upload['end'],
                launch_enter=launch['start'],launch_return=launch['end'],optimizer_start=k['start'],optimizer_end=k['end'],
                last_kernel_end=max(x['end'] for x in own_ks),
                d2h_api_enter=hd['start'],d2h_api_return=hd['end'],d2h_start=download['start'],d2h_end=download['end'])
            row.update({field+'_ms':(value-start)/1e6 for field,value in absolute.items()})
            row.update(optimizer_duration_ms=(k['end']-k['start'])/1e6,
                h2d_api_ms=(hu['end']-hu['start'])/1e6,h2d_after_api_return_ms=(upload['start']-hu['end'])/1e6,
                d2h_api_ms=(hd['end']-hd['start'])/1e6,
                last_kernel_to_d2h_start_ms=(download['start']-max(x['end'] for x in own_ks))/1e6,
                d2h_device_duration_ms=(download['end']-download['start'])/1e6,
                d2h_return_after_last_kernel_ms=(hd['end']-max(x['end'] for x in own_ks))/1e6,
                d2h_return_after_copy_ms=(hd['end']-download['end'])/1e6)
            timeline.append(row)
        timeline.sort(key=lambda r:r['setup']);assert [r['setup'] for r in timeline]==list(range(128))
        with (out/f'{path.stem}-{variant}-timeline.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
        left=[(r['optimizer_start_ms'],r['optimizer_end_ms']) for r in timeline if r['window_side']==0]
        right=[(r['optimizer_start_ms'],r['optimizer_end_ms']) for r in timeline if r['window_side']==1]
        all_opt=left+right
        kernel_intervals=[(k['start'],k['end']) for k in ks]
        runtime=collections.defaultdict(lambda:dict(calls=0,summed_host_ms=0))
        for a in api.values():
            if start<=a['start']<a['end']<=end:
                runtime[a['name']]['calls']+=1
                runtime[a['name']]['summed_host_ms']+=(a['end']-a['start'])/1e6
        allocation_calls={n:v for n,v in runtime.items() if n.startswith(('cudaMalloc','cudaFree'))}
        pair_report[variant]=dict(label=pair['label'],wall_ms=(end-start)/1e6,
            optimizer_kernel_span_ms=max(b for a,b in all_opt)-min(a for a,b in all_opt),
            optimizer_duration_sum_ms=sum(b-a for a,b in all_opt),
            optimizer_peak_concurrency=concurrency(all_opt),
            optimizer_time_weighted_full_grid_upper_bound=sum(r['optimizer_duration_ms']*r['grid_blocks'] for r in timeline)/((end-start)/1e6),
            no_kernel_ms=((end-start)-union(kernel_intervals))/1e6,
            cross_window_optimizer_overlap_ms=union(left)+union(right)-union(all_opt),
            first_window_last_optimizer_end_ms=max(b for a,b in left),
            second_window_first_optimizer_start_ms=min(a for a,b in right),
            second_window_setups_launched_before_first_finishes=sum(a<max(b for a,b in left) for a,b in right),
            delayed_h2d_over_5ms=sum(r['h2d_after_api_return_ms']>5 for r in timeline),
            h2d_after_api_return_ms=describe([r['h2d_after_api_return_ms'] for r in timeline]),
            last_kernel_to_d2h_start_ms=describe([r['last_kernel_to_d2h_start_ms'] for r in timeline]),
            result_d2h_device_duration_ms=describe([r['d2h_device_duration_ms'] for r in timeline]),
            d2h_return_after_last_kernel_ms=describe([r['d2h_return_after_last_kernel_ms'] for r in timeline]),
            d2h_return_after_copy_ms=describe([r['d2h_return_after_copy_ms'] for r in timeline]),
            optimizer_start_after_host_entry_ms=describe([r['optimizer_start_ms']-r['host_start_ms'] for r in timeline]),
            summed_result_d2h_api_ms=sum(r['d2h_api_ms'] for r in timeline),
            summed_result_d2h_device_ms=sum(r['d2h_end_ms']-r['d2h_start_ms'] for r in timeline),
            timed_allocation_calls=allocation_calls,runtime=dict(runtime))
    # Match the same setup rather than comparing aggregate kernel sums only.
    matched={}
    base=rows(out/f'{path.stem}-A-timeline.csv')
    for variant in 'BC':
        candidate=rows(out/f'{path.stem}-{variant}-timeline.csv')
        matched[variant]=describe([float(b['optimizer_duration_ms'])/float(a['optimizer_duration_ms'])
                                  for a,b in zip(base,candidate)])
    report['profiles'][path.stem]=dict(variants=pair_report,kernel_duration_ratio_vs_A=matched)
    print(path.stem,{v:dict(wall_ms=round(r['wall_ms'],3),overlap_ms=round(r['cross_window_optimizer_overlap_ms'],3),
                          delayed_h2d=r['delayed_h2d_over_5ms']) for v,r in pair_report.items()},flush=True)
    db.close()
assert len(report['profiles'])==2
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
