"""All capacity buckets and new host-event lifecycle/failure checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
os.chdir(root)
out=Path(sys.argv[1]).resolve()
tool=Path(__file__).resolve().parent
native=root/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text())
assert build['status']=='PASS' and sha(native)==build['native_sha256']
check_device=os.environ.get('FASTKPC_STATUS_WAIT_CHECK_DEVICE','0')
assert check_device in ['0','1']
assert not subprocess.check_output(['nvidia-smi','-i',check_device,'--query-compute-apps=pid',
                                   '--format=csv,noheader'],text=True).strip()
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=check_device,CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
os.sched_setaffinity(0,range(20))
def run(name,command):
    print('START',name,flush=True)
    with (out/(name+'.log')).open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',name,flush=True)
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
command=['g++','-O2','-std=c++17','-pthread','-rdynamic','-Ifastkpc/src',
    '-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,str(tool/'qualify.cpp'),
    str(native),'-L/usr/local/cuda/lib64','-L/usr/local/lib/R/lib',
    '-Wl,-rpath,/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/lib/R/lib',
    '-Wl,-rpath,'+str(native.parent),'-lcudart','-lR','-ldl','-o',str(out/'qualify')]
run('qualify-build',command)
fixture=root/'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022/raw'
args=[str(out/'qualify'),str(fixture/'L3-W31.bin'),str(fixture/'L5-W7.bin')]
run('qualify',args)
run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
    '--leak-check','full','--error-exitcode','99']+args+['--resource-memcheck'])
sanitizer=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in sanitizer and '0 bytes leaked' in sanitizer
assert 'PASS all five capacity buckets' in (out/'qualify.log').read_text()
run('attribution-test',['python3','-I','fastkpc/tests/test_d2h_wait_attribution.py'])
run('build-lock-test',['Rscript','--vanilla','fastkpc/tests/test_cuda_build_lock_contract.R'])
assert sha(native)==build['native_sha256']
if not (out/'device-code.json').exists():
    run('device-code',['python3','-I',str(tool/'device_code.py'),str(out)])
device_code=json.loads((out/'device-code.json').read_text())
assert device_code['status']=='PASS'
assert all(row['exact'] and row['new_object_sha256']==build['objects'][name]
           for name,row in device_code['objects'].items())
checks=dict(status='PASS',native_sha256=sha(native),
    physical_gpu=check_device,performance_evidence=False,
    cases=dict(captured_q=[28,46],synthetic_capacity_q=[73,82,193,385]),
    gates=['bitwise optimizer state and coefficients/fitted/residual',
           'one prepared-owned blocking event; none in worker hot path',
           'downstream producer event unchanged; repeated prepare policy immutable',
           'fixed evaluator numerical output and wait/download path unchanged',
           'trace download does not add waits','busy output slot rejected',
           'record/wait errors fail before status download; prepared recovers',
           'event allocation failure and invalid configuration unwind',
           'multi-worker cancellation releases leases and wait events'],
    device_code=device_code,
    memcheck_errors=0,leaked_bytes=0,
    memcheck_scope='real q28/q46 complete paths; all five buckets allocate/fail/destroy; multi-worker cancellation',
    build_command=command,
    tool_sha256={str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()},
    fixture_sha256={str(p.relative_to(root)):sha(p) for p in [fixture/'L3-W31.bin',fixture/'L5-W7.bin']})
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('PASS status-wait checks',flush=True)
