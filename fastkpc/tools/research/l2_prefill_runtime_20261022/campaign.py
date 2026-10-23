"""Same-binary S3/S2/S2/S3 full canonical development qualification.

All production numerics run on GPU0; both GPUs are otherwise idle.
Serialization and archived CPU-authority comparisons are outside elapsed time.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[4]
os.chdir(root)
out = Path(sys.argv[1]).resolve()
tool = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
assert json.loads((out/'checks.json').read_text())['status']=='PASS'
historical = 'fastkpc/artifacts/full_cuda_ci/structured_prefill_canonical_20261022/structured1/receipt.rds'
runs = [('s3a','s3',historical,'-'),
        ('s2a','s2',str(out/'s3a/receipt.rds'),'-'),
        ('s2b','s2',str(out/'s3a/receipt.rds'),str(out/'s2a/receipt.rds')),
        ('s3b','s3',str(out/'s3a/receipt.rds'),str(out/'s3a/receipt.rds'))]
native = root/'fastkpc/build/fastkpc_cuda.so'
native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
assert native_sha==json.loads((out/'build.json').read_text())['native_sha256']
with (out/'environment.log').open('w') as log:
    for command in (["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.current.sm,temperature.gpu","--format=csv"],
                    ["/usr/local/cuda/bin/nvcc","--version"],
                    ["Rscript","-e","print(sessionInfo()); print(sapply(c('Rcpp','RcppArmadillo','RcppEigen','kpcalg','mgcv'),function(p) as.character(packageVersion(p))))"]):
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for name,mode,reference,repeat in runs:
    assert not (out/name).exists(),'do not overwrite a run, including incomplete evidence'
    compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not compute,'both GPUs must be free of compute jobs before each run'
    assert hashlib.sha256(native.read_bytes()).hexdigest()==native_sha
    started = time.time()
    print('START',name,time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    command = ['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript',
               str(tool/'run.R'),str(out/name),mode,reference,repeat]
    with (out/(name+'-gpu.csv')).open('w') as memory,(out/(name+'.log')).open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=memory,stderr=subprocess.STDOUT)
        try:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    report = json.loads((out/name/'report.json').read_text())
    assert report['gates'] and all(report['gates'].values())
    (out/name/'process.json').write_text(json.dumps(dict(start_epoch=started,end_epoch=time.time(),
        affinity=sorted(os.sched_getaffinity(0)),command=command),indent=2)+'\n')
    print('PASS',name,report['elapsed_sec'],report['oracle_errors'],flush=True)
print('PASS S3/S2/S2/S3 canonical development qualification',flush=True)
