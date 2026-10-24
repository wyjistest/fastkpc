"""GPU1-only external L4 diagnostic with shared-host process telemetry."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4];os.chdir(root)
tool=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
assert mode in ['capture','capacity','memcheck','campaign']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=root/'fastkpc/build/fastkpc_cuda.so'
build=json.loads((out/('capture-build.json' if mode=='capture' else 'replay-build.json')).read_text())
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['sources'].items())
assert all(sha(root/p)==h for p,h in build['source_closure'].items())
assert all(sha(out/p)==h for p,h in build['outputs'].items())
os.sched_setaffinity(0,range(20));device='1'
uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device,CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',FASTKPC_WINDOW_NATIVE=str(native),
 FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
prefix=out/mode;policy='pending'
if mode=='capture':
 (out/'raw').mkdir()
 env.update(LD_PRELOAD=str(out/'capture.so'),FASTKPC_WINDOW_CAPTURE=str(out/'raw'))
 cmd=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(tool/'capture.R'),str(out)]
else:
 capture=json.loads((out/'capture.json').read_text());assert all(capture['gates'].values())
 assert all(sha(out/'raw'/p)==h for p,h in capture['fixture_sha256'].items())
 if mode!='capacity':
  capacity=json.loads((out/'capacity-summary.json').read_text())
  assert capacity['resources_closed'] and capacity['status'] in ['PASS','STOP_FULL_LAYER_CAPACITY']
  policy='resident' if capacity['status']=='PASS' else 'windows'
  if mode=='campaign':
   check=json.loads((out/'memcheck-summary.json').read_text());assert check['status']=='PASS'
 cmd=[str(out/'replay'),str(out/'raw'),str(prefix),mode,policy]
 if mode=='memcheck':
  cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+cmd
assert not prefix.with_suffix('.log').exists(),'preserve previous attempt'
assert not subprocess.check_output(['nvidia-smi','-i',device,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
started=time.time();samples=[];unexpected=set();other_pids=set()
print('START',mode,policy,time.strftime('%H:%M:%S'),flush=True)
with prefix.with_suffix('.log').open('x') as log,(out/(mode+'-gpu.csv')).open('x') as gpu:
 monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
  '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
 try:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while True:
   raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
   active={int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
   selected={pid for pid,g in active.items() if g==uuid};known={p.pid}
   edges=[tuple(map(int,row.split())) for row in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
   for _ in range(8):known.update(pid for pid,parent in edges if parent in known)
   unexpected.update(selected-known);other_pids.update(set(active)-selected)
   samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
   if unexpected:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    break
   try:p.wait(timeout=5);break
   except subprocess.TimeoutExpired:pass
 finally:monitor.terminate();monitor.wait(timeout=10)
(out/(mode+'-gpu-processes.json')).write_text(json.dumps(samples,indent=2)+'\n')
(out/(mode+'-process.json')).write_text(json.dumps(dict(command=cmd,policy=policy,start_epoch=started,end_epoch=time.time(),
 exit_code=p.returncode,physical_gpu=1,gpu_uuid=uuid,shared_host=True,other_gpu_pids=sorted(other_pids),
 unexpected_gpu_pids=sorted(unexpected),affinity=sorted(os.sched_getaffinity(0)),runner_sha256=sha(tool/'run.py'),
 native_sha256=sha(native),environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS',
 'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H')}),indent=2)+'\n')
assert not unexpected and p.returncode==0,'diagnostic failed: '+mode
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_closure'].items())
if mode=='memcheck':
 text=(out/'memcheck.log').read_text()
 assert 'ERROR SUMMARY: 0 errors' in text and '0 bytes leaked' in text
print('PASS',mode,flush=True)
