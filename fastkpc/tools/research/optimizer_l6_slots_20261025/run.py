"""Immutable GPU1 attempts; shared-host evidence is explicitly recorded."""
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import CAPTURE, NATIVE, ROOT, TOOL, check_identity, load, sha, write_new

out = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
assert mode in ('checks', 'memcheck', 'capacity', 'campaign')
os.chdir(ROOT)
build = check_identity(out)
protocol = dict(schema='l6-slots-protocol-v1', build_sha256=sha(out/'build.json'),
    declaration_sha256=sha(TOOL/'README.md'), capture_sha256=sha(CAPTURE/'capture-closure.json'),
    workers=32, capacity=5, setups=245, targets=1148, iterations=17062,
    original_windows=[64,64,64,53], warmup_each=1, balanced_pairs=6,
    maximum_go_lifecycle_ratio=0.90, required_paired_lifecycle_wins=6,
    max_physical_increment_bytes=1024**3, minimum_free_bytes=2*1024**3,
    capacity_basis='incremental above compute-warm production context',
    production_cache_present=False, scope='captured bounded-slot experiment only; shared host allowed')
if mode == 'checks':
    write_new(out/'protocol.json', protocol)
else:
    assert load(out/'protocol.json') == protocol
    assert load(out/'checks-summary.json')['status'] == 'PASS'
if mode in ('capacity','campaign'):
    assert load(out/'memcheck-summary.json')['status'] == 'PASS'
    log = (out/'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and '0 bytes leaked' in log
if mode == 'campaign':
    assert load(out/'capacity-summary.json')['status'] == 'PASS'
    rows = list(csv.DictReader((out/'capacity-runs.csv').open()))
    assert len(rows)==1 and int(rows[0]['physical_increment_bytes'])<=1024**3
os.sched_setaffinity(0,range(20))
uuid = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env = os.environ.copy()
for key in ('LD_PRELOAD','CUDA_LAUNCH_BLOCKING','CUDA_INJECTION64_PATH'):
    assert not env.get(key)
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_WINDOW_NATIVE=str(NATIVE),FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
prefix = out/mode
assert not prefix.with_suffix('.log').exists(),'preserve existing attempts'
cmd = [str(out/'replay'),str(CAPTURE/'capture-raw'),str(prefix),mode]
if mode == 'memcheck':
    cmd = ['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+cmd
started=time.time();samples=[];unexpected=set();others=set()
print('START',mode,time.strftime('%H:%M:%S'),flush=True)
with prefix.with_suffix('.log').open('x') as log,(out/(mode+'-gpu.csv')).open('x') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
        '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
    try:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            active={int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
            selected={pid for pid,device in active.items() if device==uuid};known={p.pid}
            edges=[tuple(map(int,r.split())) for r in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
            for _ in range(8):known.update(pid for pid,parent in edges if parent in known)
            unexpected.update(selected-known);others.update(set(active)-selected)
            samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
            if unexpected:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            try:p.wait(timeout=5);break
            except subprocess.TimeoutExpired:pass
    finally:monitor.terminate();monitor.wait(timeout=10)
write_new(out/(mode+'-gpu-processes.json'),samples)
write_new(out/(mode+'-process.json'),dict(command=cmd,start_epoch=started,end_epoch=time.time(),exit_code=p.returncode,
    physical_gpu=1,gpu_uuid=uuid,other_gpu_pids=sorted(others),unexpected_gpu_pids=sorted(unexpected),
    selected_gpu_isolated=not unexpected,isolated=not unexpected and not others,
    affinity=sorted(os.sched_getaffinity(0)),runner_sha256=sha(TOOL/'run.py'),native_sha256=sha(NATIVE),
    protocol_sha256=sha(out/'protocol.json'),environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_')) or k in
    ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH')}))
assert p.returncode==0 and not unexpected,'experiment failed; preserve logs'
check_identity(out)
print('PASS',mode,flush=True)
