"""Run one external diagnostic on GPU1; preserve shared-host telemetry."""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4];os.chdir(root)
tool=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
build=json.loads((out/('capture-build.json' if mode=='capture' else 'replay-build.json')).read_text())
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['sources'].items())
assert all(sha(root/p)==h for p,h in build['source_closure'].items())
device='1';os.sched_setaffinity(0,range(20))
uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device,CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',FASTKPC_WINDOW_NATIVE=str(native),
 FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
prefix=out/mode
if mode=='capture':
    (out/'raw').mkdir()
    env.update(LD_PRELOAD=str(out/'capture.so'),FASTKPC_WINDOW_CAPTURE=str(out/'raw'))
    cmd=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(tool/'capture.R'),str(out)]
else:
    assert all(json.loads((out/'capture.json').read_text())['gates'].values())
    if mode in ('campaign','profile','profile-A','profile-R'):
        capacity=json.loads((out/'capacity-summary.json').read_text())
        assert capacity['status']=='PASS'
    if mode.startswith('profile-'):
        profile_build=json.loads((out/'profile-build.json').read_text())
        assert profile_build['native_sha256']==build['native_sha256']
        assert all(sha(root/p)==h for p,h in profile_build['sources'].items())
        assert all(sha(out/p)==h for p,h in profile_build['outputs'].items())
        cmd=[str(out/'profile-replay'),str(out/'raw'),str(prefix),mode[-1]]
    else:cmd=[str(out/'replay'),str(out/'raw'),str(prefix),mode]
    if mode=='memcheck':
        cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+cmd
    elif mode=='profile' or mode.startswith('profile-'):
        cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
             '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','-o',str(prefix)]+cmd
assert not prefix.with_suffix('.log').exists(),'preserve previous run'
assert not subprocess.check_output(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
started=time.time();samples=[];unexpected=set();other_pids=set()
print('START',mode,time.strftime('%H:%M:%S'),flush=True)
with prefix.with_suffix('.log').open('x') as log,(out/(mode+'-gpu.csv')).open('x') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
      '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
    try:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            active={int(row.split(',')[0]):row.split(',')[1].strip() for row in raw.splitlines() if row.strip()}
            selected={pid for pid,g in active.items() if g==uuid}
            # Profiler/sanitizer launch a child process; include only our process tree.
            known={p.pid}
            ps=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
            edges=[tuple(map(int,row.split())) for row in ps.splitlines()]
            for _ in range(8):known.update(pid for pid,parent in edges if parent in known)
            unexpected.update(selected-known);other_pids.update(set(active)-selected)
            samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
            if unexpected:
                # Terminate only this diagnostic process group, never external jobs.
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            try:p.wait(timeout=5);break
            except subprocess.TimeoutExpired:pass
    finally:monitor.terminate();monitor.wait(timeout=10)
(out/(mode+'-gpu-processes.json')).write_text(json.dumps(samples,indent=2)+'\n')
(out/(mode+'-process.json')).write_text(json.dumps(dict(command=cmd,start_epoch=started,end_epoch=time.time(),
 exit_code=p.returncode,physical_gpu=1,shared_host=True,other_gpu_pids=sorted(other_pids),
 unexpected_gpu_pids=sorted(unexpected),affinity=sorted(os.sched_getaffinity(0)),
 environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS','OPENBLAS_NUM_THREADS',
 'OMP_NUM_THREADS','MKL_NUM_THREADS','FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H')}),indent=2)+'\n')
assert not unexpected and p.returncode==0,'diagnostic failed: '+mode
assert sha(native)==build['native_sha256']
if mode=='profile' or mode.startswith('profile-'):
    subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True)
print('PASS',mode,flush=True)
