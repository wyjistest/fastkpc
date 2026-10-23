"""Sequential, isolated capture/replay/profiling; GPU1 remains idle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4];os.chdir(root)
here=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
native=root/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874'
assert sha(native)==expected
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 FASTKPC_WINDOW_NATIVE=str(native),LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
def run(name,cmd,extra=None):
    assert not (out/(name+'.log')).exists(),'preserve existing evidence'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    e=env.copy();e.update(extra or {})
    start=time.time();print('START',name,flush=True)
    with (out/(name+'.log')).open('w') as log,(out/(name+'-gpu.csv')).open('w') as gpu:
        monitor=subprocess.Popen(['nvidia-smi',
          '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
          '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:r=subprocess.run(cmd,env=e,stdout=log,stderr=subprocess.STDOUT)
        finally:monitor.terminate();monitor.wait(timeout=10)
    (out/(name+'-process.json')).write_text(json.dumps(dict(command=cmd,start_epoch=start,
      end_epoch=time.time(),exit_code=r.returncode,affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
    assert r.returncode==0,f'{name} failed; see log'
    assert sha(native)==expected
    print('PASS',name,flush=True)
if mode=='capture':
    (out/'raw').mkdir()
    run('capture',['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript',str(here/'capture.R'),str(out)],
        dict(LD_PRELOAD=str(out/'capture.so'),FASTKPC_WINDOW_CAPTURE=str(out/'raw')))
elif mode in ('replay','profile','memcheck'):
    assert all(json.loads((out/'capture.json').read_text())['gates'].values())
    if mode=='replay':
        for name in ('L3-W31','L3-W32','L5-W7','L5-W8'):
            run('replay-'+name,[str(out/'replay'),str(out/'raw'/(name+'.bin')),str(out/('replay-'+name)),'campaign'])
    elif mode=='profile':
        for name in ('L3-W31','L5-W7'):
            prefix=out/('profile-'+name)
            run(prefix.name,['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
              '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false',
              '-o',str(prefix),str(out/'replay'),str(out/'raw'/(name+'.bin')),str(prefix),'profile'])
            subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True)
    else:
        for name in ('L3-W31','L5-W7'):
            run('memcheck-'+name,['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
              '--leak-check','full','--error-exitcode','99',str(out/'replay'),
              str(out/'raw'/(name+'.bin')),str(out/('memcheck-'+name)),'smoke'])
else:raise ValueError(mode)
