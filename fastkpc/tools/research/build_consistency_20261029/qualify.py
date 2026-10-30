"""Isolated GPU method parity, plus existing RNG and failure-order regressions."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ROOT, BASE, sha, read, write

TOOL = Path(__file__).resolve().parent
support_path = TOOL / 'numeric_jobs.py'
spec = importlib.util.spec_from_file_location('isolated_jobs', support_path)
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
mode = sys.argv[1]
assert mode in ('parent', 'full', 'incremental')
base = BASE / mode
identity = read(base / ('identity.json' if mode=='parent' else 'build.json'))
native = base / 'fastkpc_cuda.so'
out = base / 'qualification'
out.mkdir()
deps = {str(p): sha(p) for p in [native, TOOL/'methods.R', TOOL/'run_test.R',
    TOOL/'qualify.py', support_path,
    ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']}
for p in (ROOT/'fastkpc/R').glob('*.R'):
    deps[str(p)] = sha(p)
reference = '-' if mode=='parent' else str(BASE/'parent/qualification/receipt.rds')
if mode!='parent':
    if mode=='incremental':
        assert read(BASE/'object-payload-comparison.json')['status']=='PASS'
    else:
        assert identity['status']=='PASS' # numerical QA may overlap the incremental compile, never timing
    deps[reference] = sha(reference)
def frozen():
    assert sha(native)==identity['native_sha256']
    for p,h in deps.items():
        assert sha(p)==h,p
env = jobs.environment()
# These dcc.gamma-only options are irrelevant to the affected method tests.
for k in list(env):
    if k.startswith('FASTKPC_'):
        del env[k]
write(out/'protocol.json',dict(mode=mode,dependencies=deps,reference=reference,
    gpu=1,gpu1_exclusive=True,gpu0_background_allowed=True,fresh_R=True,all_numerical_work='GPU',
    no_skip_pass=True,performance_evidence=False))
jobs.job(out,'methods',['Rscript','--vanilla',str(TOOL/'methods.R'),str(out),str(native),reference],env,frozen)
assert read(out/'report.json')['status']=='PASS'
tests=[]
if mode=='full':
    for name in ['method_layered_identity','method_permutation_overlap','method_failure_order','method_deferred_svd']:
        path=ROOT/('fastkpc/tests/test_full_cuda_ci_'+name+'.R')
        deps[str(path)]=sha(path)
        jobs.job(out,name,['Rscript','--vanilla',str(TOOL/'run_test.R'),str(native),str(path)],env,frozen)
        log=(out/(name+'.log')).read_text()
        assert 'PASS' in log and 'SKIP' not in log
        tests.append(dict(name=name,status='PASS',source_sha256=sha(path),log_sha256=sha(out/(name+'.log'))))
write(out/'closure.json',dict(status='PASS',mode=mode,native_sha256=sha(native),
    report_sha256=sha(out/'report.json'),tests=tests,scope='affected-method numerical regression'))
print('PASS',mode,'qualification',flush=True)
