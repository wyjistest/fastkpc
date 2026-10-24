"""Qualify the production L5 API, Stage A ownership, and unchanged device code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4];os.chdir(root)
tool=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text());native=root/'fastkpc/build/fastkpc_cuda.so'
assert build['status']=='PASS' and sha(native)==build['native_sha256']
assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
 OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64',
 FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
os.sched_setaffinity(0,range(20))
def run(name,cmd):
 print('START',name,flush=True)
 with (out/(name+'.log')).open('x') as log:
  subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('PASS',name,flush=True)
includes=['-Ifastkpc/src','-I/usr/local/cuda/include']
run('audit-build',['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*includes,
 'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.cpp',
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl','-o',str(out/'audit.so')])
run('qualify-build',['g++','-O2','-std=c++17','-pthread',*includes,str(tool/'qualify.cpp'),
 str(out/'audit.so'),str(native),'-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(native.parent),
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(out/'qualify')])
args=[str(out/'qualify'),str(root/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023/raw')]
run('qualify',args)
run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+args)
mem=(out/'memcheck.log').read_text();assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
run('configuration',['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',
 'fastkpc/tests/test_full_cuda_ci_l5_optimizer_slots.R'])
run('device-code',['python3','-I','fastkpc/tools/research/optimizer_status_wait_20261023/device_code.py',str(out)])
device=json.loads((out/'device-code.json').read_text());assert device['status']=='PASS'
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
(out/'checks.json').write_text(json.dumps(dict(status='PASS',native_sha256=sha(native),physical_gpu=1,
 real_target_counts=[8,2,4,3],optimizer_and_stage_a_target_comparisons_per_run=68,
 token_and_generation_checks=True,cancellation_cleanup=True,resource_closure=True,
 memcheck_errors=0,memcheck_leaked_bytes=0,device_functions=sum(r['new_sass']['functions'] for r in device['objects'].values()),
 outputs={p:sha(out/p) for p in ['audit.so','qualify']},
 test_sha256=sha(root/'fastkpc/tests/test_full_cuda_ci_l5_optimizer_slots.R')),indent=2)+'\n')
print('PASS production L5 checks',flush=True)
