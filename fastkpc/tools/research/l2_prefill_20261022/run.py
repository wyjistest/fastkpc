"""Frozen L2 GPU replay. NumPy only indexes, hashes, and compares outputs."""
import argparse
import ctypes as ct
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
EXPECTED='89a2fab0bd02b695878556a74694fc522cddcbe2c5c2f22f74c7e4da6d5d56c3'
RESULT=np.dtype([('sequence','<u8')]+[(x,'<f8') for x in
 ('p','statistic','mean','variance','shape','scale')]+[(x,'<i4') for x in
 ('status','independent','iterations','flags')])
assert RESULT.itemsize==72
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def bits(a,b):return np.array_equal(np.ascontiguousarray(a).view(np.uint64),np.ascontiguousarray(b).view(np.uint64))
def ptr(a):return None if a is None else a.ctypes.data
def keybytes(keys):return ''.join(keys).encode()

class Library:
    def __init__(self,path):
        self.l=ct.CDLL(str(path.resolve()));l=self.l
        l.l2_error.restype=ct.c_char_p
        specs={
          'l2_arena':([ct.c_int,ct.c_int,ct.c_void_p],ct.c_void_p),
          'l2_arena_free':([ct.c_void_p],None),
          'l2_fixed':([ct.c_int]*4+[ct.c_void_p]*3,ct.c_void_p),
          'l2_fixed_free':([ct.c_void_p],None),
          'l2_evaluate':([ct.c_void_p]*2+[ct.c_int]*2+[ct.c_void_p]*2+[ct.c_char_p]+[ct.c_void_p]*3,ct.c_int),
          'l2_pool':([ct.c_void_p,ct.c_char_p,ct.c_void_p,ct.c_ulonglong,ct.c_int],ct.c_void_p),
          'l2_build':([ct.c_void_p]*2,ct.c_int),
          'l2_resolve':([ct.c_void_p]*2+[ct.c_int],ct.c_int),
          'l2_pairs':([ct.c_void_p]*4+[ct.c_int,ct.c_void_p],ct.c_int),
          'l2_finish':([ct.c_void_p],ct.c_int),
          'l2_pool_free':([ct.c_void_p],None),
          'l2_snapshot':([ct.c_void_p],ct.c_int)}
        for name,(args,result) in specs.items():getattr(l,name).argtypes=args;getattr(l,name).restype=result
    def check(self,result):
        if result:raise RuntimeError(self.l.l2_error().decode())
    def handle(self,result):
        if not result:raise RuntimeError(self.l.l2_error().decode())
        return result
    def snapshot(self):
        a=np.zeros(4,np.int64);self.check(self.l.l2_snapshot(ptr(a)));return a

class Inputs:
    def __init__(self,out):
        self.out=out;self.raw=out/'capture/raw'
        assert all(read(out/'capture/capture.json')['gates'].values())
        self.geometries={}
        for path in sorted(self.raw.glob('geometry-*.json')):
            g=read(path);g['data']=np.fromfile(path.with_suffix('.f64'),dtype='<f8');g['pivot']=np.array(g['pivot'],np.int32)
            n,q,r=g['n'],g['q'],g['rank'];assert len(g['data'])==2*n*q+q+2*q*q+q*r+1
            assert hashlib.sha256(g['data'][:n*q].tobytes()).hexdigest()==g['id']
            self.geometries[g['id']]=g
        dataset=np.fromfile(self.raw/'dataset.f64',dtype='<f8').reshape(48,351)
        dataset_id=digest('schema=full-cuda-ci-dataset-key-v1\nn=351\np=48\ntype=binary64-column-major\nvalues='+
          ''.join(f'{int(b):016x}' for b in dataset.view(np.uint64).flat)+'\n')
        self.keys=[];self.groups=[];self.keyinfo={};self.index={}
        for path in sorted(self.raw.glob('prefill-*.json'),key=lambda p:int(p.stem.split('-')[1])):
            m=read(path);g=self.geometries[m['geometry']];n,q,t=g['n'],g['q'],len(m['targets'])
            a=np.fromfile(path.with_suffix('.f64'),dtype='<f8');assert len(a)==n*t+2*t
            Y=a[:n*t].reshape(t,n);sp=a[n*t:n*t+t];log=a[n*t+t:]
            assert bits(Y,dataset[np.array(m['targets'])-1]) and np.isfinite(log).all()
            S='|'.join(map(str,g['S']))
            prepared=digest(f'schema=full-cuda-ci-prepared-s-key-v1\ndataset={dataset_id}\nS={S}\nformula_route=full-smooth\nsemantic=mgcv-1.9-1-tprs-native-setup-v1\n')
            keys=[]
            for target,value in zip(m['targets'],log):
                base=digest(f'schema=full-cuda-ci-target-key-v1\nprepared={prepared}\ntarget={target}\n')
                key=digest(f'schema=full-cuda-ci-residual-key-v2\ntarget={base}\nsolver=full-cuda-ci-mgcv-exact-fixed-evaluator-v1\nselected_log_sp_bits={int(value.view(np.uint64)):016x}\n')
                assert key not in self.index
                self.index[key]=len(self.keys);self.keys.append(key);keys.append(key)
                self.keyinfo[key]=dict(target=target,S=S,geometry=g['id'],prepared=prepared,log_bits=f'{int(value.view(np.uint64)):016x}')
            self.groups.append(dict(geometry=g['id'],Y=Y,log=log,sp=sp,keys=keys,offset=len(self.keys)-t))
        assert len(self.keys)==49919
        self.solves=[]
        for path in sorted(self.raw.glob('solve-*.json'),key=lambda p:int(p.stem.split('-')[1])):
            m=read(path);m['path']=path.with_suffix('.f64');self.solves.append(m)
            assert all(k in self.index for k in m['keys'])
            assert m['info']==[0]*m['count'] and m['route']==[2]*m['count']
        self.consumers=[json.loads(x) for x in (self.raw/'groups.jsonl').read_text().splitlines()]
        self.used=[];seen=set()
        for c in self.consumers:
            assert all(self.keyinfo[k]['prepared']==c['prepared'] for k in c['keys'])
            assert not seen.intersection(c['gram']);seen.update(c['gram']);self.used+=c['gram']
            if c['gram']:assert c['keys']==self.solves[c['solve_count']-1]['keys']
        assert len(self.solves)==9668 and sum(s['count'] for s in self.solves)==87110 and len(self.used)==42685
        self.used_index={key:i for i,key in enumerate(self.used)}
    def solve(self,m):
        n,q,t=m['n'],m['q'],m['count'];a=np.fromfile(m['path'],dtype='<f8')
        assert len(a)==t*(3*n+q+1)
        Y=a[:n*t].reshape(t,n);log=a[n*t:n*t+t];b=a[n*t+t:]
        coeff=b[:t*q].reshape(t,q);fitted=b[t*q:t*(q+n)].reshape(t,n);residual=b[t*(q+n):].reshape(t,n)
        for key,value in zip(m['keys'],log):assert f'{int(value.view(np.uint64)):016x}'==self.keyinfo[key]['log_bits']
        return Y,log,coeff,fitted,residual

def fixed_handle(lib,g,count):
    ms=np.zeros(1);p=lib.handle(lib.l.l2_fixed(g['n'],g['q'],g['rank'],count,ptr(g['data']),ptr(g['pivot']),ptr(ms)))
    return p,float(ms[0])
def evaluate(lib,p,g,Y,log,keys,arena=None,offset=0,shadow=True):
    count=len(keys);out=np.empty(count*(g['q']+2*g['n'])) if shadow else None
    meta=np.empty(3*count,np.int32) if shadow else None;ms=np.zeros(2)
    lib.check(lib.l.l2_evaluate(p,arena,offset,count,ptr(Y),ptr(log),keybytes(keys),ptr(out),ptr(meta),ptr(ms)))
    return out,None if meta is None else meta.reshape(3,count),ms

def fixed_campaign(inputs,lib,smoke=False,timing_only=False):
    out=inputs.out;initial=lib.snapshot();assert not initial.any()
    original=dict(batches=0,targets=0,prepare_ms=0.,evaluate_ms=0.,bitwise=True)
    selected=[] if timing_only else (inputs.solves[:3] if smoke else inputs.solves)
    for j,m in enumerate(selected):
        g=inputs.geometries[m['geometry']];Y,log,c,f,r=inputs.solve(m)
        p,ms=fixed_handle(lib,g,m['count']);original['prepare_ms']+=ms
        try:a,meta,times=evaluate(lib,p,g,Y,log,m['keys'])
        finally:lib.l.l2_fixed_free(p)
        assert bits(a,np.concatenate((c.ravel(),f.ravel(),r.ravel()))),('original shape not exact',j)
        assert np.array_equal(meta,np.array([m['info'],m['route'],m['rank']]))
        original['batches']+=1;original['targets']+=m['count'];original['evaluate_ms']+=float(times[0])
        if j%1000==0:print('original fixed anchor',j,len(selected),flush=True)
    # The candidate retains one prepared handle per real S and balances its
    # distinct planned targets into cohorts of width <=16. SP bits are frozen.
    groups=inputs.groups[:3] if smoke else inputs.groups
    total=sum(len(g['keys']) for g in groups)
    q_values={inputs.geometries[m['geometry']]['q'] for m in groups};assert len(q_values)==1
    q=q_values.pop();coefficient=np.empty((total,q));fitted=np.empty((total,351));residual=np.empty((total,351));metadata=np.empty((3,total),np.int32)
    candidates=[]
    for repetition in range(1 if smoke else (2 if timing_only else 3)):
        arena=lib.handle(lib.l.l2_arena(351,total,None));stats=dict(repetition=repetition,shadow=repetition==0,
          prepare_ms=0.,evaluate_ms=0.,detach_ms=0.,release_ms=0.,batches=0,targets=0,prepared=0)
        shadow_output=repetition==0 and not timing_only;stats['shadow']=shadow_output
        loop_started=time.monotonic()
        try:
            for j,m in enumerate(groups):
                g=inputs.geometries[m['geometry']];t=len(m['keys']);cohorts=(t+15)//16
                sizes=[t//cohorts+(i<t%cohorts) for i in range(cohorts)];assert min(sizes)>=2 and max(sizes)<=16
                p,ms=fixed_handle(lib,g,max(sizes));stats['prepare_ms']+=ms;stats['prepared']+=1;local=0
                try:
                    for size in sizes:
                        offset=m['offset']+local;keys=m['keys'][local:local+size]
                        a,meta,times=evaluate(lib,p,g,m['Y'][local:local+size],m['log'][local:local+size],keys,arena,offset,shadow_output)
                        stats['evaluate_ms']+=float(times[0]);stats['detach_ms']+=float(times[1]);stats['batches']+=1;stats['targets']+=size
                        if shadow_output:
                            coefficient[offset:offset+size]=a[:size*q].reshape(size,q)
                            fitted[offset:offset+size]=a[size*q:size*(q+351)].reshape(size,351)
                            residual[offset:offset+size]=a[size*(q+351):].reshape(size,351)
                            metadata[:,offset:offset+size]=meta
                        local+=size
                finally:
                    release_started=time.monotonic();lib.l.l2_fixed_free(p)
                    stats['release_ms']+=(time.monotonic()-release_started)*1000
                if j%250==0:print('candidate fixed',repetition,j,len(groups),flush=True)
        finally:lib.l.l2_arena_free(arena)
        stats['driver_wall_sec']=time.monotonic()-loop_started
        stats['stage_a_api_ms']=stats['prepare_ms']+stats['evaluate_ms']+stats['detach_ms']+stats['release_ms'];candidates.append(stats)
    assert np.array_equal(initial,lib.snapshot())
    if timing_only:
        save(out/'fixed-timing.json',dict(status='PASS',candidates=candidates,resources_zero=True,
          scope='no shadow D2H; includes prepared release and Python loop wall; isolated input, not integrated runtime'))
        print('PASS fixed timing',candidates,flush=True);return
    if smoke:
        save(out/'fixed-smoke.json',dict(status='PASS',original=original,candidates=candidates,resources_zero=True));return
    comparison=dict(target_occurrences=0,changed_coefficient=0,changed_fitted=0,changed_residual=0,
      metadata_mismatches=0,max_coefficient=0.,max_fitted=0.,max_residual=0.)
    original_residual=np.empty((len(inputs.used),351));original_rank=np.empty(len(inputs.used),np.int32)
    first_use={}
    for c in inputs.consumers:
        if c['gram']:first_use.setdefault(c['solve_count']-1,[]).extend(c['gram'])
    for j,m in enumerate(inputs.solves):
        _,_,c,f,r=inputs.solve(m);ids=np.array([inputs.index[k] for k in m['keys']]);comparison['target_occurrences']+=len(ids)
        for field,a,b in [('coefficient',c,coefficient[ids]),('fitted',f,fitted[ids]),('residual',r,residual[ids])]:
            changed=np.any(a.view(np.uint64)!=b.view(np.uint64),axis=1)
            comparison['changed_'+field]+=int(changed.sum());comparison['max_'+field]=max(comparison['max_'+field],float(np.max(np.abs(a-b))))
        comparison['metadata_mismatches']+=int(np.any(np.array([m['info'],m['route'],m['rank']])!=metadata[:,ids],axis=0).sum())
        for key in first_use.get(j,[]):
            k=m['keys'].index(key);i=inputs.used_index[key];original_residual[i]=r[k];original_rank[i]=m['rank'][k]
    assert comparison['changed_coefficient']==0 and comparison['metadata_mismatches']==0,comparison
    np.savez(out/'fixed-outputs.npz',coefficient=coefficient,fitted=fitted,residual=residual,metadata=metadata,
      original_residual=original_residual,original_rank=original_rank)
    save(out/'keys.json',dict(planned=inputs.keys,used=inputs.used,identity=inputs.keyinfo))
    save(out/'fixed.json',dict(status='PASS',original=original,candidates=candidates,comparison=comparison,
      resources_zero=True,native_sha256=EXPECTED,interpretation='isolated GPU API boundaries; not integrated outer timing'))
    print('PASS fixed',comparison,flush=True)

def component_run(inputs,lib,name,mode,original=False,smoke=False):
    arrays=np.load(inputs.out/'fixed-outputs.npz')
    keys=inputs.used if original else inputs.keys;index={k:i for i,k in enumerate(keys)}
    x=arrays['original_residual'] if original else arrays['residual'];rank=arrays['original_rank'] if original else arrays['metadata'][2]
    consumers=inputs.consumers
    if smoke:
        consumers=consumers[:3];wanted=set(k for c in consumers for k in c['keys']);keep=[i for i,k in enumerate(keys) if k in wanted]
        x=np.ascontiguousarray(x[keep]);rank=np.ascontiguousarray(rank[keep]);keys=[keys[i] for i in keep];index={k:i for i,k in enumerate(keys)}
    initial=lib.snapshot();assert not initial.any();arena=lib.handle(lib.l.l2_arena(351,len(keys),ptr(x)))
    budget=max(2,len(keys))*(351*35+35+2)*8 if smoke else 11811160064
    pool=None;results=[];references=[];identities=[];grams=0
    try:
        pool=lib.handle(lib.l.l2_pool(arena,keybytes(keys),ptr(np.ascontiguousarray(rank)),budget,mode))
        stats=np.zeros(16);start=time.monotonic();lib.check(lib.l.l2_build(pool,ptr(stats)));stage_b=time.monotonic()-start
        print(name,'built',stage_b,'accepted/fallback',stats[2:7].tolist(),flush=True)
        pair_start=time.monotonic()
        for j,c in enumerate(consumers):
            if c['gram']:
                ids=np.array([index[k] for k in c['gram']],np.int32);lib.check(lib.l.l2_resolve(pool,ptr(ids),len(ids)));grams+=1
            pairs=c['pairs'];left=np.array([index[c['keys'][p[1]]] for p in pairs],np.int32)
            right=np.array([index[c['keys'][p[2]]] for p in pairs],np.int32);sequence=np.array([p[0] for p in pairs],np.uint64)
            result=np.empty(len(pairs),RESULT);lib.check(lib.l.l2_pairs(pool,ptr(left),ptr(right),ptr(sequence),len(pairs),ptr(result)))
            assert np.array_equal(result['sequence'],sequence) and not result['status'].any()
            reference=np.array([p[3] for p in pairs]);results.append(result);references.append(reference)
            for p in pairs:
                a=inputs.keyinfo[c['keys'][p[1]]];b=inputs.keyinfo[c['keys'][p[2]]];assert a['S']==b['S']
                identities.append((min(a['target'],b['target']),max(a['target'],b['target']),a['S']))
            if j%5000==0:print(name,'original consumers',j,len(consumers),flush=True)
        consumer=time.monotonic()-pair_start
        lib.check(lib.l.l2_finish(pool))
    finally:
        if pool:lib.l.l2_pool_free(pool)
        lib.l.l2_arena_free(arena)
    assert np.array_equal(initial,lib.snapshot())
    values=np.concatenate(results);reference=np.concatenate(references)
    changed=values['p'].view(np.uint64)!=reference.view(np.uint64)
    error=float(np.max(np.abs(values['p']-reference)));flips=int(np.sum((values['p']>=.1)!=(reference>=.1)))
    report=dict(status='PASS' if error<=1e-10 and flips==0 else 'FAIL',mode=mode,original_residual=original,
      components=len(keys),consumer_groups=len(consumers),gram_groups=grams,physical_pairs=len(values),
      stage_b_sec=stage_b,consumer_replay_sec=consumer,full_component_pipeline_sec=stage_b+consumer,
      changed_physical_p=int(changed.sum()),max_difference_from_native=error,decision_flips=flips,
      resources_zero=True,native_sha256=EXPECTED)
    names=('stage_b_host_ms','produced','accepted','fallback','fallback_info','fallback_rotation','fallback_gap',
      'max_rotations','minimum_gap','pool_bytes','device_used_bytes_at_build','peak_inflight','waves',
      'fallback_batches','create_host_ms','dense_eig_ms')
    report['stats']=dict(zip(names,map(float,stats)))
    values.tofile(inputs.out/(name+'.results'))
    with (inputs.out/(name+'.tsv')).open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(['x','y','S_key','sequence','p','original_p'])
        for ident,row,ref in zip(identities,values,reference):writer.writerow([*ident,int(row['sequence']),format(float(row['p']),'.17g'),format(float(ref),'.17g')])
    save(inputs.out/(name+'.json'),report)
    if original:assert not changed.any(),'Original frozen residual/dense pipeline did not anchor bitwise'
    assert report['status']=='PASS',report
    print('PASS',name,report,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('phase',choices=['fixed','fixed-timing','original','dense1','structured1','structured2','dense2']);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.out.resolve();assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==EXPECTED
    inputs=Inputs(out);lib=Library(out/'build/probe.so')
    if args.phase in ('fixed','fixed-timing'):fixed_campaign(inputs,lib,args.smoke,args.phase=='fixed-timing')
    else:
        assert read(out/'fixed.json')['status']=='PASS'
        component_run(inputs,lib,args.phase+('-smoke' if args.smoke else ''),int(args.phase.startswith('structured')),args.phase=='original',args.smoke)
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==EXPECTED
if __name__=='__main__':main()
