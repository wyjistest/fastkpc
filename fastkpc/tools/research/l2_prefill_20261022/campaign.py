"""Sequential GPU-only replay campaign; GPU1 stays idle, external memory sample."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4];os.chdir(root)
here=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
assert json.loads((out/'fixed.json').read_text())['status']=='PASS'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
query=['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu','--format=csv,noheader,nounits']
def run(name,command):
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not processes,processes
    print('START',name,flush=True);started=time.time()
    with (out/(name+'.log')).open('w') as log,(out/(name+'-gpu.csv')).open('w') as gpu:
        monitor=subprocess.Popen(query+['-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        finally:monitor.terminate();monitor.wait(timeout=10)
    (out/(name+'-process.json')).write_text(json.dumps(dict(exit_code=r.returncode,elapsed_sec=time.time()-started,command=command),indent=2)+'\n')
    if r.returncode:raise RuntimeError(f'{name} failed; see {name}.log')
    print('PASS',name,flush=True)
for mode in ('original','dense1','structured1','structured2','dense2'):
    run(mode,[sys.executable,str(here/'run.py'),str(out),mode])
run('memcheck-fixed',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full',
 '--error-exitcode','99',sys.executable,str(here/'run.py'),str(out),'fixed','--smoke'])
run('memcheck-components',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full',
 '--error-exitcode','99',sys.executable,str(here/'run.py'),str(out),'structured1','--smoke'])
for mode in ('memcheck-fixed','memcheck-components'):
    text=(out/(mode+'.log')).read_text();assert 'ERROR SUMMARY: 0 errors' in text and '0 bytes leaked' in text
subprocess.run([sys.executable,str(here/'map.py'),str(out)],env=env,check=True)
with (out/'qualification.log').open('w') as log:
    subprocess.run(['Rscript','--vanilla',str(here/'qualify.R'),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('PASS frozen L2 qualification; runtime not integrated',flush=True)
