"""Sequential isolated fresh-process comparisons with external memory sampling."""
import json
import os
from pathlib import Path
import subprocess
import sys

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
tool=Path(__file__).resolve().parent
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
historical='fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/subset-off/receipt.rds.gz'
authority=str(out/'authority.rds')
for name,mode,reference in [('off','off',historical),('dense','dense',str(out/'off/receipt.rds')),
                           ('structured','structured',str(out/'off/receipt.rds'))]:
    assert not (out/name/'receipt.rds').exists()
    command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript',
             str(tool/'run.R'),str(out/name),mode,reference,authority]
    with (out/(name+'-memory.csv')).open('w') as memory, (out/(name+'.log')).open('w') as log:
        monitor=subprocess.Popen(['nvidia-smi','--id=1',
            '--query-gpu=timestamp,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=memory,stderr=subprocess.STDOUT)
        try:
            print('START',name,flush=True)
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    report=json.loads((out/name/'report.json').read_text())
    assert report['gates'] and all(report['gates'].values())
    print('PASS',name,report['elapsed_sec'],report['oracle_errors'],flush=True)
