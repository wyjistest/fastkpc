"""Observe waits in both window and slot traces; temporal alignment is not causality."""
import collections
from pathlib import Path
import sys
import numpy as np
from common import load,sha,write_new

out=Path(sys.argv[1]).resolve()
KD=np.dtype([('start','<u8'),('end','<u8'),('lmem_total','<u8'),('stream','<u4'),
    ('corr','<u4'),('name','<u4'),('lmem','<u4'),('channel','<u4'),('grid','<u4'),('block','<u4'),('regs','<u2')])
RD=np.dtype([('start','<u8'),('end','<u8'),('corr','<u4'),('cbid','<u4'),('thread','<u4')])
def dist(v):
    a=np.asarray(v,dtype=float)
    return dict(count=len(a),minimum=float(a.min()),p50=float(np.median(a)),
        p95=float(np.percentile(a,95)),maximum=float(a.max()),sum=float(a.sum())) if len(a) else dict(count=0)

results={}
for variant in 'WS':
    run=out/('trace-'+variant)
    summary=dict(line.split() for line in (run/'ktrace-summary.txt').read_text().splitlines())
    assert summary['dropped']=='0'
    K=np.fromfile(run/'kernels.bin',dtype=KD);R=np.fromfile(run/'runtime.bin',dtype=RD)
    assert len(K)==int(summary['kernels']) and len(R)==int(summary['runtime'])
    names={int(k):v for k,v in (line.split('\t',1) for line in (run/'kernel-names.tsv').read_text().splitlines())}
    launch={int(r['corr']):(int(r['start']),int(r['end'])) for r in R}
    ki=np.array([i for i,k in enumerate(K) if 'optimize_multi_penalty_targets_kernel' in names[int(k['name'])]],dtype=int)
    ki=ki[np.argsort(K['start'][ki],kind='stable')]
    assert len(ki)==4065
    P=K[ki];st=P['start'].astype(np.int64);en=P['end'].astype(np.int64)
    starts=np.array([launch[int(k['corr'])][0] for k in P],dtype=np.int64)
    streams=P['stream'];channels=P['channel'];late=np.nonzero(st-starts>50_000_000)[0]
    unique_streams=np.unique(streams)
    assert len(unique_streams)==(4065 if variant=='W' else 32)
    # Reconstruct original W groups from launch order; no cross-window candidate ends.
    if variant=='W':
        ordered=sorted(unique_streams,key=lambda s:min(st[streams==s]))
        group={int(s):i//64 for i,s in enumerate(ordered)}
    else:group={int(s):0 for s in unique_streams}
    groups=np.array([group[int(s)] for s in streams])
    def near(i,t):
        peers=(streams!=streams[i])&(groups==groups[i])&(np.abs(en-t)<=2_000_000)
        same=bool(np.any(peers&(channels==channels[i])))
        return same,bool(np.any(peers)) and not same
    same=other=overlap=shift_same=shift_other=shift_trials=0
    for i in range(len(P)):
        overlap+=bool(np.any((streams!=streams[i])&(channels==channels[i])&(st<=st[i])&(en>st[i])))
    for i in late:
        a,b=near(i,st[i]);same+=a;other+=b
        peers=groups==groups[i];lo=int(st[peers].min());hi=int(en[peers].max())
        for shift in (-500_000_000,-300_000_000,-200_000_000,200_000_000,300_000_000,500_000_000):
            t=int(st[i])+shift
            if lo<=t<=hi:
                a,b=near(i,t);shift_same+=a;shift_other+=b;shift_trials+=1
    first_use=[]
    all_st=K['start'].astype(np.int64);all_en=K['end'].astype(np.int64)
    seen=set()
    for i in sorted(range(len(K)),key=lambda i:launch.get(int(K['corr'][i]),(int(K['start'][i]),))[0]):
        name=int(K['name'][i]);api=launch.get(int(K['corr'][i]))
        if name in seen or api is None:continue
        seen.add(name);a0,a1=api
        running=np.nonzero((all_st<a0)&(all_en>a0))[0]
        drain=int(all_en[running].max()) if len(running) else a0
        released=np.nonzero((all_st>=drain)&(all_st<=drain+2_000_000)&(K['stream']!=K['stream'][i]))[0]
        released=[j for j in released if launch.get(int(K['corr'][j]),(0,))[0]>a0]
        first_use.append(dict(function=names[name],launch_api_ms=(a1-a0)/1e6,
            launch_to_start_ms=(int(K['start'][i])-a0)/1e6,running_at_launch=len(running),
            latest_running_end_after_launch_ms=(drain-a0)/1e6,released_near_drain=len(released)))
    results[variant]=dict(persistent_kernels=len(P),streams=len(unique_streams),dropped=0,
        persistent_span_sec=(int(en.max())-int(st.min()))/1e9,
        device_service_ms=dist((en-st)/1e6),launch_to_start_ms=dist((st-starts)/1e6),
        device_service_by_target_count={str(int(q)):dist((en[P['grid']==q]-st[P['grid']==q])/1e6) for q in np.unique(P['grid'])},
        persistent_grid_target_sum=int(P['grid'].sum()),
        same_channel_overlapping_starts=overlap,late_gt50ms=len(late),late_near_same_channel_end=same,
        late_near_other_channel_end_only=other,shifted_chance=dict(trials=shift_trials,same=shift_same,other_only=shift_other),
        channels_per_stream=dist([len(np.unique(channels[streams==s])) for s in unique_streams]),
        first_use=first_use,first_use_gemm=[r for r in first_use if 'gemm' in r['function'].lower()],
        evidence_class='separate cold-context captured trace; not performance or full-method evidence')
    assert results[variant]['persistent_grid_target_sum']==54878
write_new(out/'trace-analysis.json',dict(schema='l3-slot-trace-analysis-v1',results=results,
    definitions=dict(late_launch_to_start_ms=50,end_alignment_tolerance_ms=2,
        note='Both tests apply to slot and window modes. Alignment and channel reuse are observations, not driver-policy proof.'),
    files={str(p.relative_to(out)):sha(p) for v in 'WS' for p in (out/('trace-'+v)).iterdir() if p.is_file()}))
for v,r in results.items():print(v,{k:r[k] for k in ['persistent_span_sec','device_service_ms','launch_to_start_ms',
    'same_channel_overlapping_starts','late_gt50ms','late_near_same_channel_end','late_near_other_channel_end_only']})
