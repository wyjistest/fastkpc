"""GPU1 only, source-closure binding, preserve logs and shared-host telemetry."""
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[4];os.chdir(root)
out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
assert mode in ['checks','memcheck','capacity','campaign']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text());native=root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['sources'].items())
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert all(sha(out/p)==h for p,h in build['outputs'].items())
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023'
assert sha(prior/'manifest.json')==build['fixture_manifest_sha256']
manifest=json.loads((prior/'manifest.json').read_text())
assert all(sha(prior/p)==h for p,h in manifest['files'].items() if p.startswith('raw/'))
if mode!='checks':
    assert json.loads((out/'checks-summary.json').read_text())['status']=='PASS'
if mode in ['capacity','campaign']:
    assert 'ERROR SUMMARY: 0 errors' in (out/'memcheck.log').read_text()
    assert '0 bytes leaked' in (out/'memcheck.log').read_text()
if mode=='campaign':
    assert json.loads((out/'capacity-summary.json').read_text())['status']=='PASS'
    rows=list(csv.DictReader((out/'capacity-runs.csv').open()));assert len(rows)==1
    assert int(rows[0]['physical_increment_bytes'])<=1024**3
device='1';os.sched_setaffinity(0,range(20))
uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
 FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
prefix=out/mode;assert not prefix.with_suffix('.log').exists(),'preserve previous evidence'
cmd=[str(out/'replay'),str(prior/'raw'),str(prefix),mode]
if mode=='memcheck':cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+cmd
samples=[];other=set();unexpected=set();started=time.time()
print('START',mode,time.strftime('%H:%M:%S'),flush=True)
with prefix.with_suffix('.log').open('x') as log,(out/(mode+'-gpu.csv')).open('x') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
        '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
    try:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            active={int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
            selected={p for p,u in active.items() if u==uuid};known={proc.pid}
            ps=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
            edges=[tuple(map(int,r.split())) for r in ps.splitlines()]
            for _ in range(8):known.update(p for p,parent in edges if parent in known)
            unexpected.update(selected-known);other.update(set(active)-selected)
            samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
            if unexpected:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            try:proc.wait(timeout=5);break
            except subprocess.TimeoutExpired:pass
    finally:monitor.terminate();monitor.wait(timeout=10)
(out/(mode+'-gpu-processes.json')).write_text(json.dumps(samples,indent=2)+'\n')
(out/(mode+'-process.json')).write_text(json.dumps(dict(command=cmd,start_epoch=started,end_epoch=time.time(),exit_code=proc.returncode,
 physical_gpu=1,shared_host=True,other_gpu_pids=sorted(other),unexpected_gpu_pids=sorted(unexpected),
 affinity=sorted(os.sched_getaffinity(0)),runner_sha256=sha(__file__),
 environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS',
 'FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}),indent=2)+'\n')
assert not unexpected and proc.returncode==0,'diagnostic failed; preserve '+str(prefix)+'.log'
assert sha(native)==build['native_sha256']
print('PASS',mode,flush=True)
