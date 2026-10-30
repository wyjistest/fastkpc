"""Resume only existing regressions after fixing the R cleanup harness.

The successful cross-native method results and first failed harness log are
retained. No numerical criterion changes and no repeated method campaign.
"""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,BASE,sha,read,write
TOOL=Path(__file__).resolve().parent
support=TOOL.parent/'stage_b_work_pair_runtime_20261028/support.py'
spec=importlib.util.spec_from_file_location('jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
build=read(BASE/'full/build.json');native=BASE/'full/fastkpc_cuda.so'
first=BASE/'full/qualification'
assert read(first/'report.json')['status']=='PASS'
out=BASE/'full/regressions2';out.mkdir()
names=['method_layered_identity','method_permutation_overlap','method_failure_order','method_deferred_svd']
tests=[ROOT/('fastkpc/tests/test_full_cuda_ci_'+n+'.R') for n in names]
deps={str(p):sha(p) for p in [native,TOOL/'run_test.R',TOOL/'regressions.py',support,*tests]}
for p in (ROOT/'fastkpc/R').glob('*.R'):deps[str(p)]=sha(p)
def frozen():
    for p,h in deps.items():assert sha(p)==h,p
env=jobs.environment()
for k in list(env):
    if k.startswith('FASTKPC_'):del env[k]
write(out/'protocol.json',dict(dependencies=deps,
    preserved_failure=str(first/'method_permutation_overlap.log'),
    correction='evaluate complete test expression vector; Sys.unsetenv receives a character vector',
    numerical_gates_unchanged=True,performance_evidence=False))
reports=[]
for name,path in zip(names,tests):
    jobs.job(out,name,['Rscript','--vanilla',str(TOOL/'run_test.R'),str(native),str(path)],env,frozen)
    log=(out/(name+'.log')).read_text();assert 'PASS' in log and 'SKIP' not in log
    reports.append(dict(name=name,status='PASS',source_sha256=sha(path),log_sha256=sha(out/(name+'.log'))))
write(out/'closure.json',dict(status='PASS',native_sha256=sha(native),tests=reports))
write(first/'closure.json',dict(status='PASS',mode='full',native_sha256=sha(native),
    report_sha256=sha(first/'report.json'),tests=reports,regression_closure_sha256=sha(out/'closure.json'),
    preserved_harness_failure=True,numerical_gates_unchanged=True))
print('PASS full-build method and RNG/failure regressions',flush=True)
