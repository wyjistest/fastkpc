"""Sequential L2 gates, retained regressions, lifecycle/memcheck, runtime smoke."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
os.chdir(root)
out = Path(sys.argv[1]).resolve()
tool = Path(__file__).resolve().parent
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FASTKPC_DCC_GAMMA_L2_COMPONENT_PREFILL='0')
env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/lib/R/lib:' + env.get('LD_LIBRARY_PATH', '')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'],text=True).strip()
def run(name, command):
    print('START', name, flush=True)
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PASS', name, flush=True)
tests = [('l2-configuration','test_full_cuda_ci_l2_component_prefill.R'),
         ('structured-configuration','test_full_cuda_ci_structured_prefill.R'),
         ('level-configuration','test_full_cuda_ci_level_component_prefill.R'),
         ('cache-regression','test_full_cuda_ci_legacy_component_cache.R')]
for name, path in tests:
    run(name, ['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
               'Rscript','fastkpc/tests/'+path])
run('lifecycle-build', ['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','-std=c++17',
    '-Ifastkpc/src/cuda','-Ifastkpc/src','-Xlinker=-Bsymbolic',str(tool/'lifecycle.cu'),
    'fastkpc/build/full_cuda_ci_structured_eig.o','fastkpc/build/fastkpc_cuda.so',
    '-L/usr/local/lib/R/lib','-lR','-lcudart','-lcublas','-lcusolver','-o',str(out/'lifecycle')])
fixture = root/'fastkpc/artifacts/full_cuda_ci/structured_prefill_20261022/fixtures'
args = [str(out/'lifecycle'),str(fixture/'residuals.f64'),str(fixture/'three-fields.f64')]
run('lifecycle',args)
run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
    '--leak-check','full','--error-exitcode','99']+args+['smoke'])
run('smoke',['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
    'Rscript',str(tool/'smoke.R'),str(out)])
for name in [name for name, _ in tests]+['lifecycle','memcheck','smoke']:
    assert 'PASS' in (out/(name+'.log')).read_text(),name
check = (out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in check and '0 bytes leaked' in check
(out/'checks.json').write_text(json.dumps(dict(status='PASS',tests=[n for n,_ in tests],
    l2_lifecycle_components=4,retained_adapter_components=640,
    sanitizer_adapter_components=4,sanitizer_mixed_components=70,
    true_gap_fallbacks=56,fallback_chunks=[48,8],memcheck_errors=0,leaked_bytes=0,
    runtime_smoke=json.loads((out/'smoke.json').read_text())),indent=2)+'\n')
print('PASS L2 runtime checks',flush=True)
