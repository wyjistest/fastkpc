"""Activity explanation, with an independent endpoint-sweep cross-check."""
import ast
import sys
from pathlib import Path
import numpy as np
from build import ROOT, prior, read, sha, write

out=Path(sys.argv[1]).resolve()
assert read(out/'campaign.json')['status']=='PASS_ISOLATED_RECOVER_TILE_CAMPAIGN'
parent=prior.TOOL/'analyze.py'
expected=read(out/'build.json')['dependencies'][str(parent.relative_to(ROOT))]
assert sha(parent)==expected
tree=ast.parse(parent.read_text())
helpers=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)],type_ignores=[])
exec(compile(helpers,str(parent),'exec'))
fields='kind device context stream correlation name detail thread gx gy gz bx by bz regs ss ds lmem channel'.split()
dtype=np.dtype([(k,'<u8') for k in ['start','end','bytes']]+[(k,'<u4') for k in fields])
assert dtype.itemsize==100


def independently_sweep(device,bidi,recover,lo,hi):
    changes=[]
    for tag,rows in enumerate([device,bidi,recover]):
        for row in rows:
            left,right=max(lo,int(row['start'])),min(hi,int(row['end']))
            if right>left:changes.extend([(left,tag,1),(right,tag,-1)])
    counts=[0,0,0];sums=[0,0,0,0,0];previous=lo
    for t,tag,delta in sorted(changes):
        width=t-previous
        for j in range(3):
            if counts[j]:sums[j]+=width
        if counts[1] and counts[2]:sums[3]+=width
        if counts[2] and counts[0]==counts[2]:sums[4]+=width
        counts[tag]+=delta
        assert counts[tag]>=0
        previous=t
    assert counts==[0,0,0]
    return [s/1e9 for s in sums]


result=dict(schema='recover-tile-activity-analysis-v1',tiles={},
    hardware_occupancy_measured=False,memory_traffic_measured=False,performance_from_trace=False,
    caveats=['One instrumented run per tile, separate from nine timing runs.',
             'Kernel intervals do not establish actual block co-residency.',
             'Family unions overlap; exclusive and trailing intervals are not a savings budget.',
             'Replay omits optimizer workspaces and intervening CI consumers.'])
hashes={};verified=0
for tile in [36,16,8]:
    dest=out/f'trace{tile}'
    summary=read(dest/'trace-summary.json')
    assert summary['dropped']==summary['invalid_timestamps']==0
    data=np.fromfile(dest/'events.bin',dtype=dtype)
    assert len(data)==summary['events']
    assert (data['start']>0).all() and (data['end']>=data['start']).all()
    names={int(s.split('\t',1)[0]):s.split('\t',1)[1] for s in (dest/'names.tsv').read_text().splitlines()}
    markers=[(int(t),int(level),label) for t,level,label in (s.split('\t') for s in (dest/'markers.tsv').read_text().splitlines())]
    levels=[]
    for level,count,wave_count in [(2,49919,78),(3,54878,86)]:
        lo=next(t for t,l,s in markers if l==level and s=='begin')
        hi=next(t for t,l,s in markers if l==level and s=='end')
        rows=data[(data['end']>lo)&(data['start']<hi)]
        kernels=rows[rows['kind']==1];device=rows[rows['kind']<=3]
        api_rows=rows[(rows['kind']==4)|(rows['kind']==5)]
        api={int(r['correlation']):r for r in api_rows}
        assert len(api)==len(api_rows)
        assert all(int(r['correlation']) in api for r in kernels)
        tags=np.array([family(names[int(key)]) for key in kernels['name']])
        groups={tag:kernels[tags==tag] for tag in sorted(set(tags))}
        bidi,rec,roots=groups['bidiagonal'],groups['recover'],groups['roots']
        assert len(bidi)==len(rec)==(count+63)//64
        assert (bidi['ds']==14040).all() and (rec['ds']==tile*351*8).all()
        assert (rec['bx']==64).all() and (rec['gy']==(36+tile-1)//tile).all()
        du=length(merge(device,lo,hi))
        d=dict(level=level,boundary_sec=(hi-lo)/1e9,device_union_sec=du,
            no_recorded_device_activity_sec=(hi-lo)/1e9-du,kernels={},kernel_api_correlations=len(kernels))
        for tag,selected in groups.items():
            others=device[~((device['kind']==1)&np.isin(device['name'],selected['name']))]
            configs=sorted({tuple(int(r[k]) for k in ['gx','gy','gz','bx','by','bz','regs','ss','ds','lmem']) for r in selected})
            d['kernels'][tag]=dict(count=len(selected),union_sec=length(merge(selected,lo,hi)),
                exclusive_recorded_activity_sec=du-length(merge(others,lo,hi)),
                duration_ms=quant((selected['end']-selected['start'])/1e6),
                config_fields=['gx','gy','gz','bx','by','bz','regs','ss','ds','lmem'],configs=configs)
        overlap=intersection(merge(bidi,lo,hi),merge(rec,lo,hi))
        d['bidiagonal_recover_overlap_sec']=overlap
        sweep=independently_sweep(device,bidi,rec,lo,hi)
        reference=[du,d['kernels']['bidiagonal']['union_sec'],d['kernels']['recover']['union_sec'],
            overlap,d['kernels']['recover']['exclusive_recorded_activity_sec']]
        assert all(abs(a-b)<=1e-8 for a,b in zip(sweep,reference)),(tile,level,sweep,reference)
        verified+=5
        waves=[];left=lo
        ends=sorted((int(s[9:]),t) for t,l,s in markers if l==level and s.startswith('wave_end_'))
        assert [i for i,t in ends]==list(range(wave_count))
        for index,right in ends:
            wb=bidi[(bidi['start']>=left)&(bidi['end']<=right)]
            wr=rec[(rec['start']>=left)&(rec['end']<=right)]
            wp=roots[(roots['start']>=left)&(roots['end']<=right)]
            assert len(wb)==len(wr)==len(wp)==min(10,(count-index*640+63)//64)
            first,last=int(wb['start'].min()),int(wb['end'].max())
            gaps=[];early=0
            for row in wr:
                pred=wp[wp['stream']==row['stream']];assert len(pred)==1
                end=int(pred[0]['end']);assert int(row['start'])>=end
                gaps.append((int(row['start'])-end)/1e6)
                early+=int(api[int(row['correlation'])]['end'])<=end
            offsets=sorted((int(r['start'])-first)/1e6 for r in wb)
            waves.append(dict(index=index,boundary_sec=(right-left)/1e9,lane_count=len(wb),
                starts_within_50ms=sum(v<=50 for v in offsets),start_spread_ms=max(offsets),
                roots_end_to_recover_start_ms=gaps,recover_launch_returned_before_roots_end=int(early),
                recover_union_after_last_bidiagonal_sec=length(merge(wr,last,right)),
                trailing_boundary_after_last_bidiagonal_sec=(right-last)/1e9))
            left=right
        normal=waves[1:-1]
        d.update(waves=waves,normal_wave_count=len(normal),
            normal_wave_boundary_ms=quant([w['boundary_sec']*1000 for w in normal]),
            normal_start_spread_ms=quant([w['start_spread_ms'] for w in normal]),
            normal_early_start_counts={str(v):sum(w['starts_within_50ms']==v for w in normal)
                for v in sorted({w['starts_within_50ms'] for w in normal})},
            recover_predecessor_gap_ms=quant([g for w in waves for g in w['roots_end_to_recover_start_ms']]),
            recover_launch_returned_before_roots_end=sum(w['recover_launch_returned_before_roots_end'] for w in waves),
            recover_union_after_last_bidiagonal_sec=sum(w['recover_union_after_last_bidiagonal_sec'] for w in waves),
            trailing_boundaries_after_last_bidiagonal_sec=sum(w['trailing_boundary_after_last_bidiagonal_sec'] for w in waves))
        levels.append(d)
        print(tile,level,'boundary',d['boundary_sec'],'recover exclusive',d['kernels']['recover']['exclusive_recorded_activity_sec'],
            'post bidiagonal',d['recover_union_after_last_bidiagonal_sec'],flush=True)
    result['tiles'][str(tile)]=dict(levels=levels,trace_summary=summary,hardware=read(dest/'hardware.json'))
    for name in ['events.bin','names.tsv','markers.tsv','trace-summary.json','hardware.json','tile.json','mode.json']:
        hashes[str((dest/name).relative_to(out))]=sha(dest/name)
result.update(input_sha256=hashes,independent_endpoint_checks=verified,independent_tolerance_sec=1e-8,
    analysis_tool_sha256=sha(Path(__file__)),campaign_sha256=sha(out/'campaign.json'))
assert verified==30
write(out/'analysis.json',result)
print('PASS',verified,'independent interval checks',flush=True)
