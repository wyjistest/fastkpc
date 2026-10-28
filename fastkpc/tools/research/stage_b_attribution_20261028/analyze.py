"""Post-capture interval analysis; never infer occupancy or removable work from sums."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import numpy as np

out=Path(sys.argv[1]).resolve()
assert json.loads((out/'campaign.json').read_text())['status']=='PASS_AUTHENTICATED_ACTIVITY_DIAGNOSTIC'
trace=out/'trace'
summary=json.loads((trace/'trace-summary.json').read_text())
assert summary['dropped']==0 and summary['invalid_timestamps']==0
fields64=['start','end','bytes']
fields32=['kind','device','context','stream','correlation','name','detail','thread',
          'gx','gy','gz','bx','by','bz','regs','ss','ds','lmem','channel']
dtype=np.dtype([(k,'<u8') for k in fields64]+[(k,'<u4') for k in fields32])
assert dtype.itemsize==100 and (trace/'events.bin').stat().st_size%100==0
events=np.fromfile(trace/'events.bin',dtype=dtype)
assert len(events)==summary['events'] and (events['end']>=events['start']).all()
names={int(line.split('\t',1)[0]):line.rstrip('\n').split('\t',1)[1] for line in (trace/'names.tsv').open()}
markers=[(int(ts),int(level),name) for ts,level,name in
         (line.rstrip().split('\t') for line in (trace/'markers.tsv').open())]

def merge(rows,lo,hi):
    if not len(rows):return []
    order=np.argsort(rows['start'],kind='stable');result=[]
    for r in rows[order]:
        a=max(lo,int(r['start']));b=min(hi,int(r['end']))
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result
def length(intervals):return sum(b-a for a,b in intervals)/1e9
def intersection(a,b):
    i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total/1e9
def gaps(intervals,lo,hi):
    end=lo;result=[]
    for a,b in intervals:
        if a>end:result.append([end,a])
        end=max(end,b)
    if end<hi:result.append([end,hi])
    return result
def durations(rows):return (rows['end'].astype(np.int64)-rows['start'].astype(np.int64))/1e6
def quant(values):
    return {name:float(np.quantile(values,q)) for name,q in [('p50',.5),('p95',.95),('max',1)]} if len(values) else {}
def family(name):
    for word in ['build_distance_rows','reduce_rows_batch','bidiagonal','recover','roots','check_structured_boundary',
                 'center_structured_component','prepare']:
        if word in name:return word
    return name
def ledger(rows,lo,hi):
    kernel=rows[rows['kind']==1];copies=rows[(rows['kind']==2)|(rows['kind']==3)]
    device=rows[rows['kind']<=3]
    uk=merge(kernel,lo,hi);uc=merge(copies,lo,hi);ud=merge(device,lo,hi)
    gap=gaps(ud,lo,hi)
    return dict(boundary_sec=(hi-lo)/1e9,kernel_union_sec=length(uk),copy_set_union_sec=length(uc),
      recorded_device_activity_union_sec=length(ud),kernel_copy_overlap_sec=length(uk)+length(uc)-length(ud),
      no_kernel_interval_sec=(hi-lo)/1e9-length(uk),no_recorded_device_activity_sec=length(gap),
      no_recorded_activity_gap_ms=quant([(b-a)/1e6 for a,b in gap]),
      no_recorded_activity_over_1ms_sec=sum(b-a for a,b in gap if b-a>1e6)/1e9,
      kernel_count=len(kernel),memcpy_count=int((rows['kind']==2).sum()),memset_count=int((rows['kind']==3).sum()))

result=dict(schema='stage-b-activity-analysis-v1',hardware_occupancy_measured=False,warp_stalls_measured=False,
    cautions=['Activity intervals do not measure SM occupancy, eligible warps or warp stall causes.',
      'Per-kernel and per-lane duration sums include overlap and are not critical-path time.',
      'No-recorded-device-activity gaps are not guaranteed idle or removable.',
      'Captured Stage B excludes optimizer workspaces and intervening CI consumers; no full-method speedup/capacity claim.',
      'NCU kernel replay serializes kernels; concurrent range metrics require separate qualification.'],levels=[])
hardware=json.loads((trace/'hardware.json').read_text())
for level in [2,3]:
    lo=[t for t,l,n in markers if l==level and n=='begin'];hi=[t for t,l,n in markers if l==level and n=='end']
    assert len(lo)==len(hi)==1;lo,hi=lo[0],hi[0]
    rows=events[(events['end']>lo)&(events['start']<hi)]
    d=ledger(rows,lo,hi);d['level']=level
    d['unprofiled_sec']=[json.loads((out/name/f'l{level}-observation.json').read_text())['stage_b_sec']
                         for name in ['replay1','replay2','replay3']]
    d['unprofiled_median_sec']=statistics.median(d['unprofiled_sec'])
    d['profile_to_unprofiled_ratio']=d['boundary_sec']/d['unprofiled_median_sec']
    kernel=rows[rows['kind']==1]
    families=np.array([family(names[int(i)]) for i in kernel['name']])
    groups=[]
    for f in sorted(set(families)):
        selected=kernel[families==f];other=rows[(rows['kind']<=3)&
            ~((rows['kind']==1)&np.isin(rows['name'],selected['name']))]
        configs=sorted(set(tuple(int(r[x]) for x in ['gx','gy','gz','bx','by','bz','regs','ss','ds','lmem']) for r in selected))
        static=[]
        for config in configs:
            gx,gy,gz,bx,by,bz,regs,ss,ds,lmem=config
            shared_bound=hardware['shared_per_sm']//(ss+ds+hardware['reserved_shared_per_block'])
            thread_bound=hardware['max_threads_per_sm']//(bx*by*bz)
            register_upper=hardware['registers_per_sm']//max(1,regs*bx*by*bz)
            static.append(dict(grid=[gx,gy,gz],block=[bx,by,bz],registers_per_thread=regs,
                shared_static_bytes=ss,shared_dynamic_bytes=ds,local_bytes_per_thread=lmem,
                shared_residency_upper_blocks=shared_bound,thread_residency_upper_blocks=thread_bound,
                unrounded_register_residency_upper_blocks=register_upper))
        groups.append(dict(name=f,count=len(selected),summed_duration_sec=float(durations(selected).sum()/1000),
            union_sec=length(merge(selected,lo,hi)),duration_ms=quant(durations(selected)),
            only_this_family_recorded_device_activity_sec=d['recorded_device_activity_union_sec']-length(merge(other,lo,hi)),
            launch_configs=static))
    d['kernels']=groups
    d['memory']=[]
    for kind in [2,3]:
        for detail in sorted(set(int(x) for x in rows['detail'][rows['kind']==kind])):
            r=rows[(rows['kind']==kind)&(rows['detail']==detail)]
            d['memory'].append(dict(kind='memcpy' if kind==2 else 'memset',cupti_detail=detail,count=len(r),
                bytes=int(r['bytes'].sum()),summed_duration_sec=float(durations(r).sum()/1000),union_sec=length(merge(r,lo,hi))))
    runtime=rows[rows['kind']==4];apis=[]
    for name in sorted(set(runtime['name'])):
        r=runtime[runtime['name']==name]
        apis.append(dict(name=names[int(name)],count=len(r),summed_host_sec=float(durations(r).sum()/1000),
                         duration_ms=quant(durations(r))))
    d['runtime_apis']=sorted(apis,key=lambda r:-r['summed_host_sec'])
    # This CUPTI runtime associates launches/copies directly with runtime IDs.
    # Other versions may use driver IDs. Match either exact ID, never time/order.
    api_rows=rows[(rows['kind']==4)|(rows['kind']==5)]
    api={int(r['correlation']):r for r in api_rows}
    assert len(api)==len(api_rows),'ambiguous API correlation IDs'
    delays={};unmatched=0;matched_domains={'runtime':0,'driver':0}
    for r,f in zip(kernel,families):
        a=api.get(int(r['correlation']))
        if a is None:unmatched+=1;continue
        matched_domains['runtime' if a['kind']==4 else 'driver']+=1
        delays.setdefault(str(f),[]).append((int(r['start'])-int(a['end']))/1e6)
    d['post_matched_api_return_to_kernel_start_ms']={f:dict(quant(v),count=len(v),negative_count=sum(x<0 for x in v)) for f,v in delays.items()}
    d['unmatched_kernel_api_correlations']=unmatched
    d['kernel_api_match_domains']=matched_domains
    assert unmatched==0
    d['memcpy_api_attribution']=[]
    device_union=merge(rows[rows['kind']<=3],lo,hi)
    for detail in sorted(set(int(x) for x in rows['detail'][rows['kind']==2])):
        copies=rows[(rows['kind']==2)&(rows['detail']==detail)]
        calls=[]
        for c in copies:
            a=api.get(int(c['correlation']))
            if a is None and c['name']:a=api.get(int(c['name']))
            assert a is not None,'unmatched memcpy API'
            calls.append(a)
        calls=np.array(calls,dtype=dtype)
        intervals=merge(calls,lo,hi)
        d['memcpy_api_attribution'].append(dict(cupti_copy_kind=detail,count=len(copies),
            api_names=sorted(set(names[int(x)] for x in calls['name'])),
            api_summed_sec=float(durations(calls).sum()/1000),api_interval_union_sec=length(intervals),
            api_overlap_recorded_device_activity_sec=intersection(intervals,device_union),
            actual_copy_summed_sec=float(durations(copies).sum()/1000),
            api_duration_ms=quant(durations(calls))))
    waves=sorted([(int(n[len('wave_end_'):]),t) for t,l,n in markers if l==level and n.startswith('wave_end_')])
    expected=78 if level==2 else 86
    assert [i for i,t in waves]==list(range(expected))
    d['wave_count']=len(waves);d['wave_boundary_sec']=[];previous=lo
    for i,t in waves:
        d['wave_boundary_sec'].append((t-previous)/1e9);previous=t
    d['selected_waves']=[]
    for i in [0,len(waves)//2,len(waves)-1]:
        a=lo if i==0 else waves[i-1][1];b=waves[i][1]
        r=rows[(rows['end']>a)&(rows['start']<b)]
        d['selected_waves'].append(dict(index=i,**ledger(r,a,b)))
    d['post_final_wave_sec']=(hi-waves[-1][1])/1e9
    result['levels'].append(d)
result['input_sha256']={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
    [trace/'events.bin',trace/'names.tsv',trace/'markers.tsv',trace/'trace-summary.json',out/'campaign.json']}
result['analysis_tool_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
original=json.loads((out/'analysis-v1.json').read_text()) if (out/'analysis-v1.json').exists() else None
if original:
    stable=['boundary_sec','kernel_union_sec','copy_set_union_sec','recorded_device_activity_union_sec',
            'no_kernel_interval_sec','no_recorded_device_activity_sec','kernels','selected_waves']
    assert all(a[f]==b[f] for a,b in zip(original['levels'],result['levels']) for f in stable)
    result['previous_analysis_sha256']=hashlib.sha256((out/'analysis-v1.json').read_bytes()).hexdigest()
    result['interval_results_unchanged']=True
with (out/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
for d in result['levels']:
    print('L',d['level'],'unprofiled',d['unprofiled_sec'],'trace',d['boundary_sec'],
          'device union',d['recorded_device_activity_union_sec'],'no activity',d['no_recorded_device_activity_sec'])
    for k in sorted(d['kernels'],key=lambda x:-x['union_sec']):
        print(' ',k['name'],k['count'],'union',k['union_sec'],'exclusive',k['only_this_family_recorded_device_activity_sec'])
