"""Device-resident eig adapter qualification; Python compares GPU outputs only.

Residuals and kpcalg authority are immutable archived fixtures. No CPU eig,
Gram, statistic or gamma producer is used by this campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes as ct
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tarfile
import time

import numpy as np

HERE=Path(__file__).resolve().parent
REAL=Path('fastkpc/artifacts/full_cuda_ci/structured_real_20261020')
CANON=Path('fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
STATUS=np.dtype([(x,'<i4') for x in ('info','distinct','rotations','deflated','reason','reserved')]+
                [(x,'<f8') for x in ('kth','next','gap')])
RESULT=np.dtype([('sequence','<u8')]+[(x,'<f8') for x in
  ('p','statistic','mean','variance','shape','scale')]+[(x,'<i4') for x in
  ('status','independent','iterations','flags')])
assert STATUS.itemsize==48 and RESULT.itemsize==72


def checked(base,relative):
    m=json.loads((base/'manifest.json').read_text())['files'][relative]
    expected=m['sha256'] if isinstance(m,dict) else m
    p=base/relative
    assert sha(p)==expected, f'input identity changed: {p}'
    return p


def rows(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        return list(csv.DictReader(f,delimiter='\t' if '.tsv' in path.name else ','))


class Probe:
    def __init__(self,lib,x,keys,capacity=64,rotations=170662):
        self.lib=lib; self.x=np.ascontiguousarray(x,dtype='<f8')
        self.count,self.n=x.shape; self.k=35; self.capacity=capacity
        self.entry=self.n*self.k+self.k+2
        self.p=lib.adapter_create(self.n,self.k,self.count,capacity,self.x.ctypes.data,
          ''.join(keys).encode(),rotations)
        if not self.p: raise RuntimeError(lib.adapter_error().decode())
    def check(self,code):
        if code: raise RuntimeError(self.lib.adapter_error().decode())
    def close(self):
        if self.p: self.lib.adapter_destroy(self.p); self.p=None
    def reset(self): self.check(self.lib.adapter_reset(self.p))
    def begin(self,ids,mode=1,fault=-1,kind=0,fail_dense=0):
        ids=np.asarray(ids,dtype=np.int32)
        self.check(self.lib.adapter_begin(self.p,ids.ctypes.data,len(ids),mode,fault,kind,fail_dense))
    def finish(self,width):
        s=np.zeros(width,dtype=STATUS); self.check(self.lib.adapter_finish(self.p,s.ctypes.data)); return s
    def build(self,order=None,width=64,mode=1):
        if order is None: order=np.arange(self.count)
        all_status=np.zeros(self.count,dtype=STATUS)
        for start in range(0,len(order),width):
            ids=order[start:start+width]; self.begin(ids,mode); all_status[ids]=self.finish(len(ids))
        return all_status
    def shadow(self,ids=None):
        ids=np.asarray(np.arange(self.count) if ids is None else ids,dtype=np.int32)
        out=np.empty((len(ids),self.entry)); solvers=np.empty(len(ids),dtype=np.int32)
        self.check(self.lib.adapter_shadow(self.p,ids.ctypes.data,len(ids),out.ctypes.data,solvers.ctypes.data))
        return out,solvers
    def resolve(self,groups):
        for group in groups:
            ids=np.asarray(group,dtype=np.int32)
            self.check(self.lib.adapter_resolve(self.p,ids.ctypes.data,len(ids)))
    def pairs(self,pairs):
        results=[]
        for offset in range(0,len(pairs),4096):
            p=np.asarray(pairs[offset:offset+4096],dtype=np.int32)
            left=np.ascontiguousarray(p[:,0]); right=np.ascontiguousarray(p[:,1])
            ids=np.arange(offset+1,offset+1+len(p),dtype=np.uint64); out=np.zeros(len(p),dtype=RESULT)
            self.check(self.lib.adapter_pairs(self.p,left.ctypes.data,right.ctypes.data,
              ids.ctypes.data,len(p),out.ctypes.data)); results.append(out)
        return np.concatenate(results)


def load_lib(path):
    lib=ct.CDLL(str(path.resolve()))
    lib.adapter_error.restype=ct.c_char_p
    lib.adapter_create.argtypes=[ct.c_int]*4+[ct.c_void_p,ct.c_char_p,ct.c_int]
    lib.adapter_create.restype=ct.c_void_p
    lib.adapter_begin.argtypes=[ct.c_void_p,ct.c_void_p]+[ct.c_int]*5
    lib.adapter_finish.argtypes=[ct.c_void_p,ct.c_void_p]
    lib.adapter_reset.argtypes=[ct.c_void_p]
    lib.adapter_resolve.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int]
    lib.adapter_pairs.argtypes=[ct.c_void_p]*4+[ct.c_int,ct.c_void_p]
    lib.adapter_shadow.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_void_p]
    lib.adapter_snapshot.argtypes=[ct.c_void_p]
    lib.adapter_destroy.argtypes=[ct.c_void_p]
    lib.adapter_workspace.argtypes=[ct.c_void_p]; lib.adapter_workspace.restype=ct.c_size_t
    return lib


def snapshot(lib):
    values=np.zeros(4,dtype=np.int64); assert lib.adapter_snapshot(values.ctypes.data)==0
    return values.tolist()


def authority_gate(result,oracle):
    err=np.abs(result['p']-oracle)
    flips=(result['p']>=.1)!=(oracle>=.1)
    assert np.isfinite(result['p']).all() and (err<=1e-10).all() and not flips.any(), (
      'p contract failed',float(err.max()),np.flatnonzero(err>1e-10).tolist())
    near=(oracle>=.05)&(oracle<=.15)
    return dict(pairs=len(err),max_p_error=float(err.max()),alpha_flips=int(flips.sum()),
      near_alpha_count=int(near.sum()),near_alpha_max_p_error=float(err[near].max()) if near.any() else 0.)


def qualify_dataset(lib,work,name,x,keys,pairs,groups,oracle,*,smoke=False,captured=None):
    before=snapshot(lib); p=Probe(lib,x,keys,capacity=64)
    report=dict(name=name,n=x.shape[1],components=len(x),groups=[len(g) for g in groups])
    try:
        sd=p.build(mode=0); dense3,_=p.shadow()
        p.resolve(groups); dense=p.pairs(pairs); dense4,_=p.shadow()
        report['dense_authority']=authority_gate(dense,oracle)
        if captured is not None:
            # Actual canonical payload bytes, same residual. This verifies source-shared
            # dense eig/centering/total independently of the new solver.
            assert dense3[:,:-1].tobytes()==captured[:,:-1].tobytes(), 'dense three-field fixture mismatch'
            report['dense_original_three_fields_bitwise']=True
            report['dense_original_self_moment_changed']=int(np.sum(dense4[:,-1].view('u8')!=captured[:,-1].view('u8')))
        p.reset(); status=p.build(); baseline3,solvers=p.shadow()
        p.resolve(groups); actual=p.pairs(pairs); baseline4,_=p.shadow()
        report['structured_authority']=authority_gate(actual,oracle)
        report['max_p_vs_dense']=float(np.max(np.abs(actual['p']-dense['p'])))
        assert report['max_p_vs_dense']<=1e-10
        report['moments_vs_dense']={}
        for field,absolute in [('statistic',1e-9),('mean',1e-10),('variance',1e-10)]:
            error=np.abs(actual[field]-dense[field])
            relative=error/np.maximum(np.abs(dense[field]),1e-300)
            assert np.isfinite(actual[field]).all() and np.all(error<=absolute) and np.all(relative<=1e-8)
            report['moments_vs_dense'][field]=dict(max_absolute=float(error.max()),max_relative=float(relative.max()))
        assert baseline3[:,-2].tobytes()==dense3[:,-2].tobytes(), 'original total bits changed'
        report['original_total_bitwise']=True
        report['fallback_counts']={str(v):int(np.sum(status['reason']==v)) for v in range(5)}
        report['minimum_natural_gap']=float(status['gap'][status['reason']==0].min()) if np.any(status['reason']==0) else None
        report['rotations_max']=int(status['rotations'].max())
        report['workspace_bytes']=int(lib.adapter_workspace(p.p))
        report['three_fields_sha256']=digest(baseline3[:,:-1]); report['compact_sha256']=digest(actual)
        np.savez(work/(name+'.npz'),dense=dense,structured=actual,status=status,solver=solvers,
          dense_payload=dense4,structured_payload=baseline4,oracle=oracle)
        variants=[]
        # The original delayed Gram grouping/order is frozen across these variants.
        # Only three-field producer batch width/order changes.
        for width,label,order in [(1,'reverse',np.arange(len(x))[::-1]),
                                 (2,'forward',np.arange(len(x))),
                                 (17,'shuffle',np.random.default_rng(20261022).permutation(len(x))),
                                 (31,'reverse',np.arange(len(x))[::-1]),
                                 (64,'repeat',np.arange(len(x)))]:
            if smoke and width not in (1,64): continue
            p.reset(); current=p.build(order=order,width=width); fields,ss=p.shadow()
            assert fields[:,:-1].tobytes()==baseline3[:,:-1].tobytes(), ('three-field batch dependence',name,width,label)
            assert current.tobytes()==status.tobytes() and ss.tobytes()==solvers.tobytes()
            p.resolve(groups); out=p.pairs(pairs)
            assert out.tobytes()==actual.tobytes(), ('original GPU consumer repeat drift',name,width)
            variants.append(dict(width=width,order=label,three_fields_bitwise=True,compact_bitwise=True))
        report['batch_variants']=variants
    finally: p.close()
    assert snapshot(lib)==before, 'resource counters did not close'
    report['resources_closed']=True
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','batch_variants')}),flush=True)
    return report


def fail_expected(call,text):
    try: call()
    except RuntimeError as e:
        assert text in str(e), (text,str(e)); return
    raise AssertionError('expected rejection: '+text)


def fault_tests(lib,x,keys):
    before=snapshot(lib); p=Probe(lib,x[:4],keys[:4],capacity=4)
    groups=[[0,1],[2,3]]; pairs=[(0,1),(2,3)]; outcomes=[]
    try:
        p.build(width=4,mode=0); dense,_=p.shadow(); p.resolve(groups); densep=p.pairs(pairs)
        p.reset(); p.build(width=4); structured,_=p.shadow()
        for kind,reason in [(1,1),(2,2),(3,3)]:
            p.reset(); p.begin([0,1,2,3],fault=1,kind=kind)
            fail_expected(lambda:p.resolve([[0,1]]),'unfinished')
            status=p.finish(4); values,solvers=p.shadow()
            assert status['reason'].tolist()==[0,reason,0,0]
            assert solvers.tolist()==[1,2,1,1]
            assert values[1,:-1].tobytes()==dense[1,:-1].tobytes()
            assert values[[0,2,3],:-1].tobytes()==structured[[0,2,3],:-1].tobytes()
            fail_expected(lambda:p.pairs(pairs),'unresolved')
            p.resolve(groups); out=p.pairs(pairs)
            assert np.max(np.abs(out['p']-densep['p']))<=1e-10
            fail_expected(lambda:p.resolve([[0,1]]),'unresolved')
            outcomes.append(dict(kind=kind,reason=reason,dense_payload_bitwise=True,
              unaffected_payload_bitwise=True,rotations=int(status['rotations'][1])))
        p.reset(); p.begin([0,1,2,3],fault=1,kind=1,fail_dense=1)
        fail_expected(lambda:p.finish(4),'dense fallback failed closed')
        fail_expected(lambda:p.shadow(),'unfinished')
        fail_expected(lambda:p.pairs(pairs),'transaction failed')
        p.reset(); p.build(width=4); assert p.shadow()[0][:,:-1].tobytes()==structured[:,:-1].tobytes()
        p.reset(); p.begin([0,1,2,3]) # destroy with in-flight structured work
    finally: p.close()
    assert snapshot(lib)==before
    # Real tied/zero excluded spectrum triggers the gap guard, without injection.
    tied=np.tile(np.repeat([0.,1.],[175,176]),(2,1))
    tiedkeys=[hashlib.sha256(f'two-value-{i}'.encode()).hexdigest() for i in range(2)]
    p=Probe(lib,tied,tiedkeys,capacity=2)
    try:
        status=p.build(width=2); assert np.all(status['reason']==3)
        a,_=p.shadow(); p.reset(); p.build(width=2,mode=0); b,_=p.shadow()
        assert a[:,:-1].tobytes()==b[:,:-1].tobytes()
    finally:p.close()
    assert snapshot(lib)==before
    return dict(mixed=outcomes,pending_rejected=True,unresolved_rejected=True,
      dense_error_no_publication=True,recovery_after_error=True,abandoned_submit_teardown=True,
      real_tied_boundary_dense_bitwise=True,resources_closed=True)


def real_fixture(smoke):
    pairs=rows(checked(REAL,'pairs.csv'))
    x=np.fromfile(checked(REAL,'residuals.f64'),dtype='<f8').reshape(-1,351)
    if smoke:x=x[:4];pairs=pairs[:2]
    keys=[]
    for i,row in enumerate(pairs):
        for side in ('x','y'):
            j=len(keys)
            text=f"real-exact-fixed-v1|S={row['S_key']}|target={row[side]}|residual={digest(x[j])}"
            keys.append(hashlib.sha256(text.encode()).hexdigest())
    return x,keys,pairs


def concurrent_gate(lib,work,x,keys,smoke):
    # Ten bounded 64-component contexts are the actual historical throughput
    # shape. This tests actual D2D cache payload traffic, not an 11 GiB reservation.
    pool_size=2 if smoke else 10
    xs=np.array_split(x[:min(len(x),pool_size*64)],pool_size)
    ps=[];offset=0
    before=snapshot(lib)
    try:
        expected=[]
        for chunk in xs:
            p=Probe(lib,chunk,keys[offset:offset+len(chunk)],capacity=64); ps.append(p); offset+=len(chunk)
            p.build(); expected.append(p.shadow()[0][:,:-1].copy());p.reset()
        trials=[]
        def run(p):
            p.build(); return p.shadow()[0][:,:-1].copy()
        for _ in range(1 if smoke else 3):
            started=time.perf_counter()
            with ThreadPoolExecutor(max_workers=pool_size) as pool:actual=list(pool.map(run,ps))
            elapsed=time.perf_counter()-started
            assert all(a.tobytes()==b.tobytes() for a,b in zip(actual,expected)), 'concurrent payload drift'
            trials.append(elapsed)
            for p in ps:p.reset()
        return dict(contexts=pool_size,capacity_each=64,components=offset,
          structured_workspace_bytes=sum(lib.adapter_workspace(p.p) for p in ps),
          diagnostic_wall_seconds=trials,bitwise=True,
          timing_scope='includes diagnostic payload D2H and worker threads; not performance evidence')
    finally:
        for p in ps:p.close()
        assert snapshot(lib)==before


def canonical_fixture(work):
    # Actual canonical on-run residual bytes, not a re-fit with a new batch shape.
    components=rows(checked(CANON,'audit-on/components.tsv.gz'))
    attributes={r['key']:r for r in rows(checked(CANON,'component-attribution.csv'))}
    selected={};payload={};bykey={r['key']:r for r in components}
    with tarfile.open(checked(CANON,'audit-on/wide-raw.tar.gz'),'r:gz') as archive:
        for member in archive:
            if not member.isfile():continue
            key=Path(member.name).stem;row=bykey[key];q=int(row['q'])
            raw=np.frombuffer(archive.extractfile(member).read(),dtype='<f8')
            assert len(raw)==q+351*2+351*35+35+2
            residual=raw[q+351:q+702]
            assert digest(residual)==row['residual']
            fields=raw[q+702:]
            for field,part in [('vectors',fields[:351*35]),('selected_values',fields[351*35:351*35+35]),
                               ('total',fields[-2:-1]),('self_moment',fields[-1:])]:
                assert digest(part)==row[field]
            selected[key]=residual.copy();payload[key]=fields.copy()
    keys=sorted(selected);at={key:i for i,key in enumerate(keys)}
    x=np.stack([selected[key] for key in keys]);captured=np.stack([payload[key] for key in keys])
    # First-use request cohorts, ordered by original dataset target. The dense
    # self-moment comparison records whether this reconstruction reproduces bytes.
    request_groups={}
    for key in keys:request_groups.setdefault(bykey[key]['request'],[]).append(key)
    groups=[[at[key] for key in sorted(group,key=lambda z:int(attributes[z]['target']))]
            for group in request_groups.values()]
    assert len(keys)==951 and len(groups)==54 and all(len(g)<=64 for g in groups)
    pair_rows=[r for r in rows(checked(CANON,'audit-on/pairs.tsv.gz'))
               if r['left'] in at and r['right'] in at]
    pairs=[(at[r['left']],at[r['right']]) for r in pair_rows]
    authority=work/'canonical-authority.tsv'
    subprocess.run(['Rscript',str(HERE/'export_authority.R'),str(authority)],check=True)
    authority_map={}
    for r in rows(authority):
        key=(r['S_key'],int(r['x']),int(r['y']))
        p=float(r['p_value'])
        assert key not in authority_map or abs(authority_map[key]-p)<=1e-10
        authority_map[key]=p
    oracle=np.asarray([authority_map[(attributes[r['left']]['S_key'],
      int(attributes[r['left']]['target']),int(attributes[r['right']]['target']))] for r in pair_rows])
    (work/'canonical-input-identities.json').write_text(json.dumps(dict(
      component_keys=keys,component_metadata=[bykey[key] for key in keys],
      gram_groups=groups,physical_pairs=pair_rows,
      source_manifest_sha256=sha(CANON/'manifest.json'),authority_export_sha256=sha(authority)),indent=2)+'\n')
    return x,keys,pairs,groups,oracle,captured


def main():
    parser=argparse.ArgumentParser();parser.add_argument('work',type=Path)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();work=args.work
    lib=load_lib(work/'adapter.so');before=snapshot(lib)
    x,keys,rs=real_fixture(args.smoke)
    pairs=[(2*i,2*i+1) for i in range(len(rs))];groups=[list(p) for p in pairs]
    oracle=np.asarray([float(r['authority_p']) for r in rs])
    report=dict(schema='structured-device-adapter-qualification-v1',evidence='isolated development qualification',
      runtime_integration=False,production_default=False,phase10='ACTIVE',candidate='NOT_FROZEN',
      holdout='SEALED',recommended=False,guard_gap_threshold=1e-8,rows=[])
    report['rows'].append(qualify_dataset(lib,work,'real640' if not args.smoke else 'smoke-real',
      x,keys,pairs,groups,oracle,smoke=args.smoke))
    report['fault_tests']=fault_tests(lib,x,keys)
    report['concurrent']=concurrent_gate(lib,work,x,keys,args.smoke)
    for n in (242,253,351,364):
        witness=rows(checked(REAL,f'witnesses/{n}.csv'))
        xx=np.fromfile(checked(REAL,f'witnesses/{n}.f64'),dtype='<f8').reshape(-1,n)
        if args.smoke: witness=witness[:1];xx=xx[:2]
        kk=[hashlib.sha256(f"{witness[i//2]['id']}|{i%2}|{digest(xx[i])}".encode()).hexdigest()
            for i in range(len(xx))]
        pp=[(i,i+1) for i in range(0,len(xx),2)]
        report['rows'].append(qualify_dataset(lib,work,f'witness-{n}'+('-smoke' if args.smoke else ''),
          xx,kk,pp,[list(p) for p in pp],np.asarray([float(r['authority_p']) for r in witness]),smoke=args.smoke))
    if not args.smoke:
        xx,kk,pp,gg,oo,captured=canonical_fixture(work)
        report['rows'].append(qualify_dataset(lib,work,'canonical-actual951',xx,kk,pp,gg,oo,captured=captured))
    assert snapshot(lib)==before
    report['resources_closed']=True;report['status']='PASS_ADAPTER_GATES'
    report['native_sha256']=sha('fastkpc/build/fastkpc_cuda.so')
    report['build_sha256']=sha(work/'build.json')
    report['runner_sha256']=sha(__file__)
    name='smoke.json' if args.smoke else 'qualification.json'
    (work/name).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',name,flush=True)


if __name__=='__main__':main()
