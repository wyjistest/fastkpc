"""Identity and exclusive-GPU process accounting for the L3 experiment."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[3]
NATIVE=ROOT/'fastkpc/build/fastkpc_cuda.so'
PARENT=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l6_slot_runtime_20261026_isolated_completion'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load(path):return json.loads(Path(path).read_text())

def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')

def source_identity():
    b=load(PARENT/'build.json')
    assert b['status']=='PASS' and sha(NATIVE)==b['native_sha256']
    for p,h in b['source_sha256'].items():assert sha(ROOT/p)==h,p
    return b

def check_capture(out):
    source_identity();b=load(out/'capture-build.json')
    for p,h in b['inputs'].items():assert sha(ROOT/p)==h,p
    for p,h in b['outputs'].items():assert sha(out/p)==h,p
    c=load(out/'capture-closure.json');assert c['status']=='PASS'
    assert c['native_sha256']==sha(NATIVE)
    for p,h in c['files'].items():assert sha(out/p)==h,p
    return c

def check_identity(out):
    check_capture(out);b=load(out/'prototype-build.json')
    for p,h in {**b['source_sha256'],**b['execution_tools'],**b['dependencies']}.items():
        assert sha(ROOT/p)==h,p
    for p,h in b['outputs'].items():assert sha(out/p)==h,p
    assert b['native_sha256']==sha(NATIVE)
    return b

def environment():
    env={k:v for k,v in os.environ.items() if not k.startswith('FASTKPC_')}
    for k in ('LD_PRELOAD','CUDA_LAUNCH_BLOCKING','CUDA_INJECTION64_PATH'):
        assert not env.get(k),k
    env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        FASTKPC_WINDOW_NATIVE=str(NATIVE),FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
        FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
        LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
    return env

def monitored(out,name,cmd,env,extra=None):
    os.chdir(ROOT);os.sched_setaffinity(0,range(20))
    query=['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader']
    assert not subprocess.check_output(query,text=True).strip(),'external GPU task present'
    uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    started=time.time();samples=[];unexpected=set();p=None
    print('START',name,time.strftime('%H:%M:%S'),flush=True)
    with (out/(name+'.log')).open('x') as log,(out/(name+'-gpu.csv')).open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                raw=subprocess.check_output(query,text=True)
                active={int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
                known={p.pid}
                edges=[tuple(map(int,r.split())) for r in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
                for _ in range(8):known.update(pid for pid,parent in edges if parent in known)
                unexpected.update(set(active)-known)
                samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
                if unexpected:break
                try:p.wait(timeout=5);break
                except subprocess.TimeoutExpired:pass
        finally:
            if p is not None and p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            monitor.terminate();monitor.wait(timeout=10)
    write_new(out/(name+'-gpu-processes.json'),samples)
    write_new(out/(name+'-process.json'),dict(command=cmd,start_epoch=started,end_epoch=time.time(),
        exit_code=p.returncode,physical_gpu=1,gpu_uuid=uuid,unexpected_gpu_pids=sorted(unexpected),
        isolated=not unexpected,affinity=sorted(os.sched_getaffinity(0)),native_sha256=sha(NATIVE),
        environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_','KTRACE_')) or k in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH','LD_PRELOAD')},
        **(extra or {})))
    assert p.returncode==0 and not unexpected,'failed/interrupted attempt retained'
    source_identity();print('PASS',name,flush=True)
