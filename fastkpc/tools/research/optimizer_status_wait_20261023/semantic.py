"""Full canonical correctness on idle GPU1 while GPU0 is externally occupied.

These runs are explicitly ineligible for performance conclusions. The separate
four-run campaign still requires both GPUs free of other compute jobs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[4]
os.chdir(root)
out=Path(sys.argv[1]).resolve()
tool=Path(__file__).resolve().parent
native=root/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text())
assert json.loads((out/'checks.json').read_text())['status']=='PASS'
os.sched_setaffinity(0,range(20))
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
old='fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022/s2a/receipt.rds'
for name,mode,reference in [('semantic-off','0',old),
                           ('semantic-on','1',str(out/'semantic-off/receipt.rds'))]:
    assert not (out/name).exists() and sha(native)==build['native_sha256']
    assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid',
                                       '--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                      '--format=csv,noheader'],text=True)
    command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
             'Rscript','--vanilla',str(tool/'run.R'),str(out/name),mode,reference,'-']
    env['FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H']=mode
    record=dict(command=command,start_epoch=time.time(),physical_gpu=1,
        external_gpu_processes_before=processes,performance_eligible=False,
        evidence_class='NONISOLATED_CANONICAL_SEMANTICS_ONLY',native_sha256=build['native_sha256'])
    (out/(name+'-process.json')).write_text(json.dumps(record,indent=2)+'\n')
    print('START',name,'SEMANTICS ONLY',flush=True)
    with (out/(name+'.log')).open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode,end_epoch=time.time())
    (out/(name+'-process.json')).write_text(json.dumps(record,indent=2)+'\n')
    assert result.returncode==0,name
    report=json.loads((out/name/'report.json').read_text())
    assert report['gates'] and all(report['gates'].values())
    print('PASS',name,report['comparisons']['changed_ci_count'],report['oracle_errors'],flush=True)
print('PASS canonical semantic pair; NO performance conclusion',flush=True)
