"""Source-shared GPU row/payload/fallback tests and compute-sanitizer."""
import importlib.util
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,PARENT,BASE,TOOL,sha,read,write
support=TOOL.parent/'build_consistency_20261029/numeric_jobs.py'
spec=importlib.util.spec_from_file_location('jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
out=BASE/'qualification';out.mkdir()
build=read(BASE/'candidate/build.json');native=Path(build['native_path'])
source=BASE/'candidate/source/fastkpc/src'
fixture=ROOT/'fastkpc/artifacts/full_cuda_ci/structured_prefill_20261022/fixtures'
manifest=read(fixture.parent/'manifest.json')
for name in ['residuals.f64','three-fields.f64']:
    assert sha(fixture/name)==manifest['files']['fixtures/'+name]['sha256']
command=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','-std=c++17',
    '-I'+str(source/'cuda'),'-I'+str(source),'-Xlinker=-Bsymbolic',str(TOOL/'lifecycle.cu'),
    str(BASE/'candidate/objects/full_cuda_ci_structured_eig.o'),str(native),
    '-L/usr/local/lib/R/lib','-Xlinker=-rpath=/usr/local/lib/R/lib',
    '-lRlapack','-lRblas','-lR','-lcudart','-lcublas','-lcusolver','-o',str(out/'lifecycle')]
write(out/'build-protocol.json',dict(command=command,source_sha256=build['source_sha256'],
    lifecycle_sha256=sha(TOOL/'lifecycle.cu'),native_sha256=sha(native)))
with (out/'build.log').open('x') as log:
    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
deps={str(p):sha(p) for p in [native,out/'lifecycle',TOOL/'lifecycle.cu',TOOL/'qualify.py',
    support,fixture/'residuals.f64',fixture/'three-fields.f64',TOOL/'README.md']}
def frozen():
    for p,h in deps.items():assert sha(p)==h,p
    for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
env=jobs.environment();env['FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR']='1'
write(out/'run-protocol.json',dict(dependencies=deps,gpu=1,gpu1_exclusive=True,gpu0_background_allowed=True,
    performance_evidence=False,
    diagnostic_D2H_only=True,full_components=640,mixed_components=70,fallback_components=56,
    row_batches=[64,1,63,17,64,9,2],sanitizer_payload_components=4))
args=[str(out/'lifecycle'),str(fixture/'residuals.f64'),str(fixture/'three-fields.f64')]
jobs.job(out,'lifecycle',args,env,frozen)
jobs.job(out,'memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
    '--leak-check','full','--error-exitcode','99',*args,'smoke'],env,frozen)
for name in ['lifecycle','memcheck']:
    assert 'PASS' in (out/(name+'.log')).read_text()
log=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in log and '0 bytes leaked' in log
write(out/'closure.json',dict(status='PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES',native_sha256=sha(native),
    row_totals_bitwise=True,full_payload_components=640,mixed_components=70,dense_fallback_chunks=[48,8],
    mixed_accepted_payload_order_exact=True,dense_failure_no_publication=True,
    sanitizer_errors=0,leaked_bytes=0,dependencies=deps))
print('PASS row-only GPU qualification',flush=True)
