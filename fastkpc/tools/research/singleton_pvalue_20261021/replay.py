"""Anchor the complete GPU pair pipeline, then replace only self_moment."""
import argparse
import csv
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
from collections import Counter
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('work',type=Path)
parser.add_argument('old',type=Path)
parser.add_argument('out',type=Path)
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
work=args.work; raw=work/'native/raw'
capture=json.loads((work/'native/capture.json').read_text())
build=json.loads((work/'build/build.json').read_text())
export=json.loads((work/'candidates/report.json').read_text())
assert all(capture['gates'].values())
assert capture['native_sha256']==build['native_sha256']==export['native_sha256']
assert sha(Path('fastkpc/build/fastkpc_cuda.so'))==build['native_sha256']
assert sha(work/'build/probe.so')==build['artifacts']['probe.so']
assert sha(work/'candidates/scalars.npz')==export['scalars_sha256']
keys=json.loads((work/'candidates/keys.json').read_text())
values=np.load(work/'candidates/scalars.npz')
scalar={key:values['candidate'][i] for i,key in enumerate(keys)}
changed={key for i,key in enumerate(keys) if values['original'][i:i+1].view('u8')[0]!=values['candidate'][i:i+1].view('u8')[0]}
requests=[json.loads(line) for line in (raw/'requests.jsonl').read_text().splitlines()]
assert sum(len(r['pairs']) for r in requests)==capture['physical_pairs']==24264
assert len({p['id'] for r in requests for p in r['pairs']})==24264
for r in requests:
    assert r['captured']==any(p['left'] in changed or p['right'] in changed for p in r['pairs'])
captured=[r for r in requests if r['captured']]
if args.smoke:
    # First 16 batches plus representatives of every pair count and cache category.
    selected={r['batch']:r for r in captured[:16]}
    for width in {len(r['pairs']) for r in captured}:
        r=next(r for r in captured if len(r['pairs'])==width); selected[r['batch']]=r
    r=next(r for r in captured if r['misses']==0); selected[r['batch']]=r
    captured=sorted(selected.values(),key=lambda r:r['batch'])
tasks=list(csv.DictReader((work/'native/tasks.tsv').open(),delimiter='\t'))
task_ids={(t['S_key'],int(t['x']),int(t['y'])):i for i,t in enumerate(tasks)}
assert len(task_ids)==len(tasks)==24264
task_p=np.fromfile(work/'native/task-pvalues.f64',dtype='<f8')
assert len(task_p)==len(tasks)
candidate_p=task_p.copy()
compact=np.dtype([('id','<u8'),('p','<f8'),('statistic','<f8'),('mean','<f8'),
 ('variance','<f8'),('shape','<f8'),('scale','<f8'),('status','<i4'),
 ('independent','<i4'),('iterations','<i4'),('flags','<i4')])
assert compact.itemsize==72
if '/usr/local/lib/R/lib' not in os.getenv('LD_LIBRARY_PATH','').split(':'):
    env=dict(os.environ); env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:'+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
ct.CDLL('/usr/local/lib/R/lib/libR.so',mode=ct.RTLD_GLOBAL)
ct.CDLL(str(Path('fastkpc/build/fastkpc_cuda.so').resolve()),mode=ct.RTLD_GLOBAL)
lib=ct.CDLL(str((work/'build/probe.so').resolve()))
dp=ct.POINTER(ct.c_double); ip=ct.POINTER(ct.c_int); up=ct.POINTER(ct.c_ulonglong)
lib.scalar_probe_create.argtypes=[ct.c_int]*4; lib.scalar_probe_create.restype=ct.c_void_p
lib.scalar_probe_run.argtypes=[ct.c_void_p,dp,ct.c_int,ip,ip,up,ct.c_int,ct.c_void_p,dp]
lib.scalar_probe_destroy.argtypes=[ct.c_void_p]
lib.scalar_probe_snapshot.argtypes=[ct.POINTER(ct.c_longlong)]
def snapshot():
    s=(ct.c_longlong*4)(); assert lib.scalar_probe_snapshot(s)==0; return list(s)
before=snapshot(); probe=lib.scalar_probe_create(351,35,22,484); assert probe
payload_cache={}
def payload(key):
    if key not in payload_cache:
        blob=(raw/(key+'.payload')).read_bytes()
        assert blob==(args.old/'native/raw'/(key+'.f64')).read_bytes(),('native payload drift',key)
        payload_cache[key]=np.frombuffer(blob,dtype='<f8')
    return payload_cache[key]
rows=[]; seen_tasks=set(); seen_changed=set(); repeats=2
try:
    for ordinal,r in enumerate(captured):
        pair_list=r['pairs']; p=len(pair_list)
        component_keys=list(dict.fromkeys(k for pair in pair_list for k in (pair['left'],pair['right'])))
        component_ids={key:i for i,key in enumerate(component_keys)}
        base=np.vstack([payload(key) for key in component_keys])
        new=base.copy()
        for j,key in enumerate(component_keys): new[j,-1]=scalar[key]
        left=np.array([component_ids[pair['left']] for pair in pair_list],dtype='<i4')
        right=np.array([component_ids[pair['right']] for pair in pair_list],dtype='<i4')
        ids=np.array([pair['id'] for pair in pair_list],dtype='<u8')
        expected=np.fromfile(raw/(str(r['batch'])+'.compact'),dtype=compact)
        original_cross=np.fromfile(raw/(str(r['batch'])+'.cross'),dtype='<f8')
        assert len(expected)==p and len(original_cross)==p*35*35
        assert np.array_equal(expected['id'],ids) and np.all(expected['status']==0)
        def gpu(data):
            out=np.empty(p,dtype=compact); cross=np.empty(p*35*35,dtype='<f8')
            assert lib.scalar_probe_run(probe,data.ctypes.data_as(dp),len(data),left.ctypes.data_as(ip),
              right.ctypes.data_as(ip),ids.ctypes.data_as(up),p,ct.c_void_p(out.ctypes.data),cross.ctypes.data_as(dp))==0
            assert cross.tobytes()==original_cross.tobytes(),('pair-cross drift',r['batch'])
            return out
        baseline=gpu(base)
        assert baseline.tobytes()==expected.tobytes(),('GPU baseline compact drift',r['batch'])
        result=gpu(new)
        for _ in range(repeats):
            assert gpu(new).tobytes()==result.tobytes(),('GPU repeat differs',r['batch'])
        for name in ('id','statistic','mean','flags'):
            assert result[name].tobytes()==baseline[name].tobytes(),('unchanged field differs',name,r['batch'])
        for j,pair in enumerate(pair_list):
            t=task_ids[(pair['S'],pair['x'],pair['y'])]
            assert t not in seen_tasks; seen_tasks.add(t)
            assert baseline['p'][j:j+1].tobytes()==task_p[t:t+1].tobytes(),('task p anchor differs',t)
            candidate_p[t]=result['p'][j]
            affected=pair['left'] in changed or pair['right'] in changed
            if not affected: assert result[j:j+1].tobytes()==baseline[j:j+1].tobytes()
            seen_changed.update(k for k in (pair['left'],pair['right']) if k in changed)
            row=dict(batch=r['batch'],physical_id=pair['id'],canonical_id=t+1,S=pair['S'],x=pair['x'],y=pair['y'],
              left=pair['left'],right=pair['right'],affected=affected,all_hit=r['misses']==0,
              baseline=float(baseline['p'][j]),candidate=float(result['p'][j]),
              baseline_bits=int(baseline['p'][j:j+1].view('u8')[0]),candidate_bits=int(result['p'][j:j+1].view('u8')[0]),
              delta=float(result['p'][j]-baseline['p'][j]),
              baseline_status=int(baseline['status'][j]),candidate_status=int(result['status'][j]),
              baseline_iterations=int(baseline['iterations'][j]),candidate_iterations=int(result['iterations'][j]),
              flip=bool(result['independent'][j]!=baseline['independent'][j]))
            rows.append(row)
        if ordinal%250==0: print('GPU pair batches',ordinal+1,'/',len(captured),'pairs',len(rows),flush=True)
finally:
    lib.scalar_probe_destroy(probe)
after=snapshot(); assert before==after
if not args.smoke: assert seen_changed==changed
affected=[r for r in rows if r['affected']]
def stats(group):
    errors=np.array([abs(r['delta']) for r in group])
    return dict(count=len(group),p_changed=sum(r['baseline_bits']!=r['candidate_bits'] for r in group),
      max_abs=float(max(errors,default=0)),flips=sum(r['flip'] for r in group),
      min_alpha_margin=min((abs(r['baseline']-.1) for r in group),default=None))
near=[r for r in affected if abs(r['baseline']-.1)<=.02]
band=[r for r in affected if .05<=r['baseline']<=.15]
bad=sum(r['candidate_status']!=0 or not np.isfinite(r['candidate']) or not 0<=r['candidate']<=1 for r in rows)
metrics=stats(affected)
report=dict(schema='singleton-gpu-pvalue-replay-v1',status='PASS_GPU_DIAGNOSTIC' if bad==0 and metrics['flips']==0 and metrics['max_abs']<=1e-13 else 'MEASURED_REVIEW_REQUIRED',
 scope='smoke' if args.smoke else 'all affected subset22 default-Inf physical/logical CI pairs',
 baseline_anchor='all captured payload bytes, pair-cross bytes, compact fields and canonical p bits exact',
 total_native_batches=len(requests),total_physical_pairs=24264,total_logical_tasks=24264,result_cache_hits=0,
 captured_batches=len(captured),captured_pairs=len(rows),affected_components=len(seen_changed),
 affected=metrics,near_alpha_002=stats(near),p005_to_015=stats(band),
 all_hit_affected=stats([r for r in affected if r['all_hit']]),
 unaffected_cobatch=stats([r for r in rows if not r['affected']]),
 by_level={str(level):stats([r for r in affected if len(r['S'].split('|'))==level]) for level in sorted({len(r['S'].split('|')) for r in affected})},
 gamma_status_failures=bad,gamma_iteration_changes=sum(r['baseline_iterations']!=r['candidate_iterations'] for r in rows),
 max_witness=max(affected,key=lambda r:abs(r['delta'])),repeats=repeats,
 resources_before=before,resources_after=after,
 native_sha256=build['native_sha256'],build_sha256=sha(work/'build/build.json'),
 native_capture_sha256=sha(work/'native/capture.json'),candidate_export_sha256=sha(work/'candidates/report.json'),
 runner_sha256=sha(Path(__file__)),
 limitations='scalar-only GPU diagnostic on fixed baseline task schedule; not new full-method, oracle, pMax, graph or performance qualification; original payload STOP unchanged')
(args.out/'pairs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
np.savez(args.out/'task-pvalues.npz',baseline=task_p,candidate=candidate_p)
report['outputs']={name:sha(args.out/name) for name in ('pairs.jsonl','task-pvalues.npz')}
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
