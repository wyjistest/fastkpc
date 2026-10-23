"""Run each adjacent pair sequentially on idle GPU0; preserve all evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent;root=here.parents[3];os.chdir(root)
out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
capture=root/'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'
native=root/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text())
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==v for p,v in build['sources'].items())
assert all(sha(root/p)==v for p,v in build['source_closure'].items())
assert all(sha(root/p)==v for p,v in build['inputs'].items())
assert all(sha(out/p)==v for p,v in build['outputs'].items())
os.sched_setaffinity(0,range(20))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 FASTKPC_WINDOW_NATIVE=str(native),LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
assert mode in ('campaign','profile','memcheck','smoke')
for level,a,b in ((3,31,32),(5,7,8)):
    name=f'{mode}-L{level}-W{a}+W{b}';prefix=out/name
    assert not prefix.with_suffix('.log').exists(),'preserve existing evidence'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    args=[str(out/'replay'),str(capture/'raw'/f'L{level}-W{a}.bin'),
          str(capture/'raw'/f'L{level}-W{b}.bin'),str(prefix),
          'smoke' if mode=='memcheck' else mode,'6']
    if mode=='profile':
        args=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
              '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false',
              '-o',str(prefix)]+args
    elif mode=='memcheck':
        args=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
              '--leak-check','full','--error-exitcode','99']+args
    start=time.time();print('START',name,flush=True)
    with (out/(name+'.log')).open('w') as log,(out/(name+'-gpu.csv')).open('w') as gpu:
        monitor=subprocess.Popen(['nvidia-smi',
          '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
          '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    (out/(name+'-process.json')).write_text(json.dumps(dict(command=args,start_epoch=start,
      end_epoch=time.time(),exit_code=result.returncode,affinity=sorted(os.sched_getaffinity(0)),
      environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS',
                   'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','FASTKPC_WINDOW_NATIVE')}),indent=2)+'\n')
    assert result.returncode==0,f'{name} failed; see log'
    assert sha(native)==build['native_sha256']
    if mode=='profile':
        subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True)
    print('PASS',name,flush=True)
