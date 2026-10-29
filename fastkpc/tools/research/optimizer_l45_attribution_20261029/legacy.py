"""Read historical traces without inventing canonical request identities."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
KERNEL=np.dtype([('start','<u8'),('end','<u8'),('lmem_total','<u8'),('stream','<u4'),
 ('corr','<u4'),('name','<u4'),('lmem','<u4'),('channel','<u4'),('grid','<u4'),('block','<u4'),('regs','<u2')])
RUNTIME=np.dtype([('start','<u8'),('end','<u8'),('corr','<u4'),('cbid','<u4'),('thread','<u4')])
def read_trace(path):
    names={int(a):b for a,b in (line.split('\t',1) for line in (path/'kernel-names.tsv').read_text().splitlines())}
    k=np.fromfile(path/'kernels.bin',dtype=KERNEL);r=np.fromfile(path/'runtime.bin',dtype=RUNTIME)
    assert not (len(k)==0 or len(r)==0)
    assert np.all(k['end']>=k['start']) and np.all(k['start']>0)
    return names,k,r
def analyze(path):
    names,k,r=read_trace(path)
    report=json.loads((path/'profile-result.json').read_text())
    threads,counts=np.unique(r['thread'],return_counts=True);main=int(threads[np.argmax(counts)])
    order=np.argsort(r['corr']);corr=r['corr'][order]
    pos=np.minimum(np.searchsorted(corr,k['corr']),len(corr)-1)
    matched=corr[pos]==k['corr'];kt=np.where(matched,r['thread'][order][pos],main)
    multi=np.array(['mgcv_multi_penalty_gcv' in names[i] for i in k['name']])
    selected=np.where(multi&(kt!=main))[0]
    by_stream=collections.defaultdict(list)
    for i in selected[np.argsort(k['start'][selected])]:by_stream[int(k[i]['stream'])].append(int(i))
    streams=sorted(by_stream,key=lambda s:k[by_stream[s][0]]['start']);offset=0;result={}
    for w in report['prefill_batches']:
        if w['penalty_class']!='multi':continue
        setups=w['optimizer_setup_count'];width=min(32,setups) if setups>64 else setups
        segment=streams[offset:offset+width];offset+=width
        if w['level'] not in [4,5]:continue
        rows=[]
        for stream in segment:
            indexes=by_stream[stream];first=k[indexes[0]]['name'];bursts=[]
            for i in indexes:
                if k[i]['name']==first:bursts.append([])
                bursts[-1].append(i)
            for ordinal,burst in enumerate(bursts):
                opts=[i for i in burst if 'optimize_multi_penalty_targets_kernel' in names[int(k[i]['name'])]]
                assert len(opts)==1
                opt=k[opts[0]]
                rows.append(dict(stream=stream,slot_burst_ordinal=ordinal,target_count=int(opt['grid']),
                    start_ns=int(k[burst[0]]['start']),end_ns=int(k[burst]['end'].max()),
                    busy_sec=float((k[burst]['end']-k[burst]['start']).sum())/1e9,
                    optimizer_sec=float(opt['end']-opt['start'])/1e9,kernels=len(burst)))
        assert len(rows)==setups
        start=min(x['start_ns'] for x in rows);end=max(x['end_ns'] for x in rows)
        for i,x in enumerate(sorted(rows,key=lambda x:x['start_ns'])):
            x.update(observed_start_rank=i,start_sec=(x.pop('start_ns')-start)/1e9,end_sec=(x.pop('end_ns')-start)/1e9)
        result[str(w['level'])]=dict(setups=setups,targets=sum(x['target_count'] for x in rows),
            span_sec=(end-start)/1e9,busy_sum_sec=sum(x['busy_sec'] for x in rows),
            longest=sorted(rows,key=lambda x:x['optimizer_sec'],reverse=True)[:20],
            identity_status='anonymous stream/burst only; no canonical keys or request order recoverable')
    assert offset==len(streams)
    return dict(native_sha256=report.get('native_sha256'),
                native_identity_note='recorded in receipt' if 'native_sha256' in report else 'not in historical JSON; do not infer',levels=result)
if __name__=='__main__':
    out=Path(sys.argv[1]).resolve();base=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025'
    value={name:analyze(base/name) for name in ['ktrace-run','ktrace-run-cache-on']}
    with (out/'legacy.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({n:{lv:{'span_sec':v['span_sec'],'longest':v['longest'][0]} for lv,v in x['levels'].items()} for n,x in value.items()},indent=2))
