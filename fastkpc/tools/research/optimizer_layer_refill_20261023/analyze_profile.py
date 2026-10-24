"""Correlate kernels to setup NVTX ranges in separate static-pool A/R traces."""
import bisect
import collections
import csv
import json
from pathlib import Path
import sqlite3
import sys

out=Path(sys.argv[1]).resolve()
timeline=[]
def analyze(variant):
    db=sqlite3.connect((out/('profile-'+variant+'.sqlite')).as_uri()+'?mode=ro',uri=True)
    names=dict(db.execute('select id,value from StringIds'))
    ranges=[(s,e,t or names.get(i,''),tid) for s,e,t,i,tid in
            db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
    apis=collections.defaultdict(list)
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER']:
        if table in tables:
            for s,e,tid,co in db.execute('select start,end,globalTid,correlationId from '+table):
                apis[tid].append((s,e,co))
    for group in apis.values():group.sort()
    starts={tid:[r[0] for r in rows] for tid,rows in apis.items()}
    kernels=collections.defaultdict(list)
    for s,e,co,stream in db.execute('select start,end,correlationId,streamId from CUPTI_ACTIVITY_KIND_KERNEL'):
        kernels[co].append((s,e,stream))
    db.close()
    layer=[r for r in ranges if r[2]=='layer:L5:'+variant+':profile']
    assert len(layer)==1
    start,end,_,_=layer[0];spans={};points=[];accounted=set()
    for s,e,text,tid in ranges:
        if not text.startswith('setup:') or not (start<=s and e<=end):continue
        index=int(text.split(':')[1]);assert index not in spans
        first=bisect.bisect_left(starts[tid],s);last=bisect.bisect_right(starts[tid],e)
        correlations={co for a,b,co in apis[tid][first:last] if b<=e}
        found=[k for co in correlations for k in kernels.get(co,[])]
        assert found,(variant,index,'no correlated kernels')
        assert not accounted.intersection(found),'kernel assigned to two setups'
        accounted.update(found);window=index//64+1
        ks=min(k[0] for k in found);ke=max(k[1] for k in found);spans[index]=(ks,ke,window)
        timeline.append(dict(variant=variant,setup=index,window=window,
            host_start_sec=(s-start)/1e9,host_return_sec=(e-start)/1e9,
            first_kernel_sec=(ks-start)/1e9,last_kernel_sec=(ke-start)/1e9,
            kernel_count=len(found),kernel_sum_sec=sum(k[1]-k[0] for k in found)/1e9))
        for a,b,stream in found:points.extend([(a,1,window),(b,-1,window)])
    assert set(spans)==set(range(955)),(variant,'setup coverage')
    all_in_range={k for group in kernels.values() for k in group if start<=k[0] and k[1]<=end}
    assert accounted==all_in_range,(variant,'uncorrelated kernels')
    active=collections.Counter();active_kernels=0;previous=None;overlap=0;peak=0;peak_windows=0
    for time,delta,window in sorted(points):
        if previous is not None and len(active)>=2:overlap+=time-previous
        active[window]+=delta;active_kernels+=delta
        if active[window]==0:del active[window]
        assert active_kernels>=0 and all(n>0 for n in active.values())
        peak=max(peak,active_kernels);peak_windows=max(peak_windows,len(active));previous=time
    assert not active and active_kernels==0
    by_window=[]
    for w in range(1,16):
        rows=[r for r in spans.values() if r[2]==w]
        by_window.append(dict(window=w,first_kernel_sec=(min(r[0] for r in rows)-start)/1e9,
            last_kernel_sec=(max(r[1] for r in rows)-start)/1e9))
    early=sum(by_window[i]['first_kernel_sec']<max(r['last_kernel_sec'] for r in by_window[:i])
              for i in range(1,15))
    return dict(setups=len(spans),correlated_kernels=len(accounted),other_kernels_in_range=0,
        profile_range_sec=(end-start)/1e9,cross_original_window_kernel_overlap_sec=overlap/1e9,
        peak_concurrent_kernels=peak,peak_original_windows_with_active_kernels=peak_windows,
        windows_started_before_prior_windows_completed=early,windows=by_window)
result={variant:analyze(variant) for variant in 'AR'}
assert result['A']['cross_original_window_kernel_overlap_sec']==0
with (out/'profile-kernel-timeline.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
report=dict(status='PASS',measurement='all CUPTI kernel intervals correlated via CUDA API and setup NVTX range; static prepared pool; profile excluded from primary timing',
            variants=result)
(out/'profile-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({v:{k:x for k,x in r.items() if k!='windows'} for v,r in result.items()},indent=2))
