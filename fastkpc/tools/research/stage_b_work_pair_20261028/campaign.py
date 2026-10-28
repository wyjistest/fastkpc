"""Source-bound GPU parity followed by predeclared isolated native/candidate A/C."""
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from build import ROOT, TOOL, ACTIVITY, NATIVE, identity, sha, write

os.chdir(ROOT)
out = Path(sys.argv[1]).resolve()
b = json.loads((out/'build.json').read_text())
old = ACTIVITY/'attempt3'
capture = old/'capture'
old_manifest = json.loads((ACTIVITY/'manifest.json').read_text())
dependencies = [old/'observe.so',old/'replay'] + [capture/f'l{l}-{suffix}' for l in [2,3]
    for suffix in ['residuals.f64','targets.tsv','payload-sha256.txt']]
for p in dependencies:
    assert sha(p) == old_manifest['files'][str(p.relative_to(ACTIVITY))]
dep_hashes = {str(p):sha(p) for p in dependencies}

def frozen():
    identity()
    assert all(sha(ROOT/p)==h for p,h in b['tools'].items())
    assert all(sha(out/p)==h for p,h in b['outputs'].items())

def processes():
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    return {int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines() if x.strip()}

frozen()
os.sched_setaffinity(0,range(20))
env = os.environ.copy()
assert not any(env.get(k) for k in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_LAUNCH_BLOCKING'])
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',STAGE_B_NATIVE=str(NATIVE))
env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
assert not processes(), 'external compute task; no job started'
write(out/'protocol.json',dict(schema='fixed-vt-work-pair-qualification-v1',declared_epoch=time.time(),
    build_sha256=sha(out/'build.json'),native_sha256=b['native_sha256'],dependencies=dep_hashes,
    gpu=1,affinity=list(range(20)),sequence=['resources','qualification','A1','C1','C2','A2','A3','C3'],
    n=351,numCol=35,levels=[2,3],counts=[49919,54878],contexts=10,width=64,
    timing_gate='Both level medians improve, sum >=5% lower, all C totals below all A totals; otherwise conditional/stop',
    numerical_gate='GPU bitwise state/log/output/boundaries plus original native payload hashes on every timing run',
    production_changes=False,performance_promotion=False,
    environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}))

def job(name,cmd,extra=None):
    frozen()
    assert not processes(), 'external GPU task; no new job started'
    e=env.copy()
    if extra:e.update(extra)
    samples=[];unexpected=set();start=time.time();proc=None
    with (out/(name+'.log')).open('x') as log,(out/(name+'-gpu.csv')).open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            proc=subprocess.Popen(cmd,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                active=processes();samples.append(dict(epoch=time.time(),processes=active))
                for pid in active:
                    try:
                        if os.getpgid(pid)!=proc.pid:unexpected.add(pid)
                    except ProcessLookupError:pass
                if unexpected:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                try:proc.wait(timeout=3);break
                except subprocess.TimeoutExpired:pass
        finally:
            monitor.terminate();monitor.wait(timeout=10)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
    write(out/(name+'-process.json'),dict(command=cmd,exit_code=proc.returncode,start_epoch=start,end_epoch=time.time(),
        isolated=not unexpected,unexpected=sorted(unexpected),samples=samples))
    assert not unexpected and proc.returncode==0, name+' failed; evidence preserved'
    frozen()
    print('PASS',name,'process_seconds',round(time.time()-start,3),flush=True)

job('resources',[sys.executable,str(TOOL/'resources.py'),str(out)])
job('qualification',[str(out/'qualify'),str(out/'baseline_diag.so'),str(out/'candidate_diag.so'),str(capture),str(out/'qualification.json')])
qualification=json.loads((out/'qualification.json').read_text())
assert qualification['status']=='PASS_GPU_BITWISE_QUALIFICATION'
runs=[]
for name in ['A1','C1','C2','A2','A3','C3']:
    dest=out/name;dest.mkdir()
    preload=str(old/'observe.so')
    if name.startswith('C'):preload+=':'+str(out/'candidate_timing.so')
    job(name,[str(old/'replay'),str(capture)],dict(LD_PRELOAD=preload,STAGE_B_OUTPUT=str(dest),STAGE_B_MODE='replay'))
    rows=[]
    for level in [2,3]:
        row=json.loads((dest/f'l{level}-observation.json').read_text())
        assert row['components']==(49919 if level==2 else 54878) and row['three_field_hashes']==row['components']
        assert row['dense_fallback']==0 and row['contexts']==10 and row['width']==64
        assert sha(dest/f'l{level}-payload-sha256.txt')==dep_hashes[str(capture/f'l{level}-payload-sha256.txt')]
        rows.append(row)
    assert 'PASS native replay resource closure' in (out/(name+'.log')).read_text()
    if name.startswith('C'):
        variant=json.loads((dest/'variant.json').read_text())
        assert variant['creates']==variant['destroys']==10 and variant['submits']==1638
        assert variant['targets']==104797 and variant['work_factor']==2 and variant['dynamic_shared']==14040
    runs.append(dict(name=name,levels=rows,total=sum(r['stage_b_sec'] for r in rows)))
    print('TIMING',name,[r['stage_b_sec'] for r in rows],flush=True)
medians={x:[statistics.median(r['levels'][j]['stage_b_sec'] for r in runs if r['name'][0]==x) for j in [0,1]] for x in ['A','C']}
gain=1-sum(medians['C'])/sum(medians['A'])
separated=max(r['total'] for r in runs if r['name'][0]=='C')<min(r['total'] for r in runs if r['name'][0]=='A')
go=gain>=.05 and separated and all(c<a for a,c in zip(medians['A'],medians['C']))
decision='GO_ISOLATED_STAGE_B' if go else 'CONDITIONAL' if gain>0 else 'STOP_NO_STAGE_GAIN'
assert all(sha(p)==h for p,h in dep_hashes.items())
frozen()
write(out/'campaign.json',dict(status='PASS_COMPLETED_ISOLATED_CAMPAIGN',decision=decision,runs=runs,medians=medians,
    combined_fraction_reduction=gain,all_candidate_totals_below_baseline=separated,
    qualification=qualification,accepted_full_method_median_unchanged=329.078,production_changes=False,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
print('DONE',decision,medians,'reduction',gain,flush=True)
