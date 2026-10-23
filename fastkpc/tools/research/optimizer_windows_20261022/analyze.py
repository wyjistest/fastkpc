"""Offline attribution. Kernel coverage and launched grids are not SM occupancy."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

out=Path(sys.argv[1]).resolve()
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def union(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
def clip(intervals,a,b):return [(max(a,s),min(b,e)) for s,e in intervals if s<b and e>a]
def describe(v):return dict(median=statistics.median(v),minimum=min(v),maximum=max(v),values=v)
report=dict(schema='optimizer-window-diagnostic-v1',status='PASS_CAPTURED_WINDOWS',windows={},profiles={},
 evidence_class='fixed captured windows; isolated optimizer diagnostics; no outer-method performance claim',
 native_sha256=json.loads((out/'capture.json').read_text())['native_sha256'])
for name in ('L3-W31','L3-W32','L5-W7','L5-W8'):
    summary=json.loads((out/('replay-'+name+'-summary.json')).read_text())
    runrows=rows(out/('replay-'+name+'-runs.csv'))
    assert all(int(r['parity'])==1 for r in runrows)
    assert len({r['iterations'] for r in runrows})==1
    assert len({r['targets'] for r in runrows})==1
    timed=[r for r in runrows if r['kind']=='timing'];observed={}
    timings={str(w):describe([float(r['wall_ms']) for r in timed if int(r['workers'])==w]) for w in (16,32,64)}
    for r in runrows:
        if r['kind']!='observed':continue
        t=rows(out/('replay-'+name+'-run'+r['run']+'-timing.csv'))
        starts=[float(x['host_start_ms']) for x in t];ends=[float(x['host_return_ms']) for x in t]
        spans=[b-a for a,b in zip(starts,ends)]
        observed[r['workers']]=dict(wall_ms=float(r['wall_ms']),host_start_spread_ms=max(starts)-min(starts),
          last_completion_ms=max(ends),median_completion_ms=statistics.median(ends),
          host_slot_fraction=sum(spans)/(int(r['workers'])*(max(ends)-min(starts))),
          setup_host_span_ms=describe(spans),stream_span_ms=describe([float(x['stream_span_ms']) for x in t]))
    report['windows'][name]=dict(summary=summary,targets=int(runrows[0]['targets']),
      iterations=int(runrows[0]['iterations']),timings_ms=timings,observed=observed,
      ratio_to64={str(w):timings[str(w)]['median']/timings['64']['median'] for w in (16,32)})
    print(name,{w:round(t['median'],3) for w,t in timings.items()},flush=True)
for path in sorted(out.glob('profile-*.sqlite')):
    db=sqlite3.connect(path);strings=dict(db.execute('select id,value from StringIds'))
    cols=[r[1] for r in db.execute('pragma table_info(NVTX_EVENTS)')]
    entries=[dict(zip(cols,r)) for r in db.execute('select * from NVTX_EVENTS')]
    for e in entries:e['label']=e.get('text') or strings.get(e.get('textId'),'')
    windows=[e for e in entries if e['label'].startswith('window:') and e['end'] is not None]
    kernels=db.execute('select start,end,shortName,gridX*gridY*gridZ,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    api=db.execute('select start,end,nameId,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME').fetchall()
    api_by_id={x[4]:x for x in api}
    copy_labels=dict(db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER'))
    copies=db.execute('select start,end,copyKind,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY').fetchall()
    sm_count=db.execute('select smCount from TARGET_INFO_GPU where cuDevice=0').fetchone()[0]
    items=[]
    for w in windows:
        a,b=w['start'],w['end'];duration=b-a
        ks=[k for k in kernels if k[0]<b and k[1]>a]
        intervals=clip([(k[0],k[1]) for k in ks],a,b)
        optimizers=[k for k in ks if 'optimize_multi_penalty' in strings[k[2]]]
        names=collections.Counter(strings[k[2]] for k in ks)
        runtime=collections.defaultdict(lambda:dict(calls=0,summed_host_ms=0))
        for s,e,n,tid,cid in api:
            length=max(0,min(e,b)-max(s,a))
            if length:
                rt=runtime[strings[n]];rt['calls']+=1;rt['summed_host_ms']+=length/1e6
        delays=[(k[0]-api_by_id[k[5]][1])/1e6 for k in optimizers if k[5] in api_by_id]
        copy_summary=collections.defaultdict(lambda:dict(calls=0,bytes=0,summed_api_host_ms=0,summed_device_copy_ms=0))
        for cs,ce,kind,size,cid in copies:
            if cid not in api_by_id:continue
            hs,he,_,_,_=api_by_id[cid]
            if not (a<=hs<he<=b):continue
            item=copy_summary[copy_labels[kind]];item['calls']+=1;item['bytes']+=size
            item['summed_api_host_ms']+=(he-hs)/1e6;item['summed_device_copy_ms']+=(ce-cs)/1e6
        tail=a+int(.75*duration)
        boundaries=collections.defaultdict(lambda:[0,0])
        boundaries[a];boundaries[b]
        for k in optimizers:
            s,e=max(a,k[0]),min(b,k[1]);boundaries[s][0]+=1;boundaries[e][0]-=1
            boundaries[s][1]+=k[3];boundaries[e][1]-=k[3]
        active=grids=0;previous=a;segments=[]
        for at,(dk,dg) in sorted(boundaries.items()):
            if at>previous:segments.append((previous,at,active,grids))
            active+=dk;grids+=dg;previous=at
        items.append(dict(label=w['label'],wall_ms=duration/1e6,kernel_count=len(ks),
          kernel_names=dict(names),any_kernel_coverage_fraction=union(intervals)/duration,
          no_kernel_ms=(duration-union(intervals))/1e6,
          optimizer_launches=len(optimizers),optimizer_streams=len({k[4] for k in optimizers}),
          optimizer_average_overlapping_kernels=sum(max(0,min(k[1],b)-max(k[0],a)) for k in optimizers)/duration,
          optimizer_launched_grid_upper_bound=sum(max(0,min(k[1],b)-max(k[0],a))*k[3] for k in optimizers)/duration,
          gpu_sm_count=sm_count,
          optimizer_max_overlapping_kernels=max(s[2] for s in segments),
          optimizer_grid_below_sm_fraction=sum(e-s for s,e,n,g in segments if 0<g<sm_count)/duration,
          last_quarter_optimizer_average_kernels=sum(max(0,e-max(s,tail))*n for s,e,n,g in segments if e>tail)/(b-tail),
          last_quarter_optimizer_launched_grid_bound=sum(max(0,e-max(s,tail))*g for s,e,n,g in segments if e>tail)/(b-tail),
          last_quarter_no_kernel_ms=((b-tail)-union(clip(intervals,tail,b)))/1e6,
          optimizer_enqueue_to_start_ms=describe(delays) if delays else None,
          copies=dict(copy_summary),
          runtime=dict(runtime)))
    assert len(items)==3,(path,len(items))
    report['profiles'][path.stem]=items
    print(path.name,[(i['label'],round(i['no_kernel_ms'],3),round(i['optimizer_average_overlapping_kernels'],2)) for i in items])
    db.close()
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
