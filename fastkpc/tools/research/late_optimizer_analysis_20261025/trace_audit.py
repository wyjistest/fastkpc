"""Compare L8 worker kernel work and first-launch/drain signatures, not timing gates."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

out=Path(sys.argv[1]).resolve()
kd=np.dtype([('start','<u8'),('end','<u8'),('lmem_total','<u8'),('stream','<u4'),
 ('corr','<u4'),('name','<u4'),('lmem','<u4'),('channel','<u4'),('grid','<u4'),('block','<u4'),('regs','<u2')])
rd=np.dtype([('start','<u8'),('end','<u8'),('corr','<u4'),('cbid','<u4'),('thread','<u4')])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
rows={};inventory={};refs={}
for name in ['trace-off','trace-on']:
    trace=out/(name+'-trace');r=read(out/name/'report.json');h=read(out/name/'hook.json');p=read(out/(name+'-process.json'))
    assert all(r['gates'].values()) and all(h['gates'].values()) and p['exit_code']==0 and not p['unexpected_gpu_pids']
    K=np.fromfile(trace/'kernels.bin',dtype=kd);RT=np.fromfile(trace/'runtime.bin',dtype=rd)
    names={int(a):b for a,b in (line.split('\t',1) for line in (trace/'kernel-names.tsv').read_text().splitlines())}
    order=np.argsort(RT['corr']);corr=RT['corr'][order]
    pos=np.clip(np.searchsorted(corr,K['corr']),0,len(corr)-1)
    match=corr[pos]==K['corr'];thread=np.where(match,RT['thread'][order[pos]],0)
    threads,counts=np.unique(RT['thread'],return_counts=True);main=int(threads[np.argmax(counts)])
    funcs=h['after']['function_names'];wanted={i for i,n in names.items() if n in funcs}
    assert len(wanted)==5
    ids=np.nonzero(np.isin(K['name'],list(wanted))&(thread!=main)&match)[0]
    assert len(np.unique(K['stream'][ids]))==9
    inv=collections.Counter((names[int(K['name'][i])],int(K['grid'][i]),int(K['block'][i])) for i in ids)
    inventory[name]=inv
    by_stream=[]
    for s in sorted(set(K['stream'][ids])):
        ks=ids[K['stream'][ids]==s];start=int(K['start'][ks].min());end=int(K['end'][ks].max())
        by_stream.append(dict(stream=int(s),start=start,end=end,span_sec=(end-start)/1e9,
          busy_sec=float((K['end'][ks]-K['start'][ks]).sum())/1e9,kernels=len(ks)))
    first=[]
    for f in sorted(wanted):
        ks=ids[K['name'][ids]==f];assert len(ks)
        i=int(ks[np.argmin(RT['start'][order[pos[ks]]])]);api=RT[order[pos[i]]]
        a0=int(api['start']);running=ids[(K['start'][ids]<a0)&(K['end'][ids]>a0)]
        drain=int(K['end'][running].max()) if len(running) else a0
        later=ids[(K['start'][ids]>=drain)&(K['start'][ids]<=drain+2_000_000)&(K['stream'][ids]!=K['stream'][i])]
        later=[j for j in later if int(RT['start'][order[pos[j]]])>a0]
        first.append(dict(function=names[f],launch_api_ms=(int(api['end'])-a0)/1e6,
          launch_to_device_start_ms=(int(K['start'][i])-a0)/1e6,
          running_l8_kernels=len(running),running_kernel_drain_sec=(drain-a0)/1e9,
          later_launches_starting_within_2ms_of_drain=len(later)))
    rows[name]=dict(canonical_elapsed_traced_sec=r['elapsed_sec'],
      l8_host_sec=next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==8),
      l8_level_sec=next(x['elapsed_ms']/1000 for x in r['level_trace'] if x['level']==8),
      l8_device_span_sec=(max(x['end'] for x in by_stream)-min(x['start'] for x in by_stream))/1e9,
      l8_busy_sum_sec=sum(x['busy_sec'] for x in by_stream),l8_kernels=len(ids),l8_streams=by_stream,
      first_worker_launches=first,preload=h['after'],process=p,
      kernel_work=[dict(function=k[0],grid=k[1],block=k[2],count=v) for k,v in sorted(inv.items())])
    for f in ['kernels.bin','runtime.bin','kernel-names.tsv','profile-result.json','analysis.json','first-use.json']:
        refs[str((trace/f).relative_to(out))]=sha(trace/f)
    for f in ['report.json','hook.json']:
        refs[str((out/name/f).relative_to(out))]=sha(out/name/f)
assert inventory['trace-off']==inventory['trace-on'],'changed L8 kernel work'
result=dict(schema='late-optimizer-trace-audit-v1',status='PASS_UNCHANGED_L8_DEVICE_WORK',
  evidence_class='mechanism diagnostic; traced times excluded from timing campaign',
  interpretation='Drain signatures are observations, not a proof that every gap is loading or HOL.',
  runs=rows,references=refs)
with (out/'trace-audit.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps({n:{k:v for k,v in r.items() if k in ['l8_host_sec','l8_device_span_sec','l8_busy_sum_sec','l8_kernels','first_worker_launches']} for n,r in rows.items()},indent=2))
