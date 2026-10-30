"""One bridge qualification after affected-method gates, before row-only edits."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,BASE,sha,read,write
TOOL=Path(__file__).resolve().parent
support=TOOL/'numeric_jobs.py'
spec=importlib.util.spec_from_file_location('isolated_jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
for mode in ['parent','full','incremental']:
    assert read(BASE/mode/'qualification/closure.json')['status']=='PASS'
assert read(BASE/'object-payload-comparison.json')['status']=='PASS'
build=read(BASE/'full/build.json');native=BASE/'full/fastkpc_cuda.so'
reference=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029/attempt1/canonical/on3/receipt.rds'
base=BASE/'canonical';base.mkdir()
write(base/'build.json',build)
out=base/'qualification';out.mkdir()
deps={str(p):sha(p) for p in [native,reference,TOOL/'canonical.R',TOOL/'canonical.py',support,
  ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
  ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds',
  TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R',
  ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
  ROOT/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds']}
for p in (ROOT/'fastkpc/R').glob('*.R'):deps[str(p)]=sha(p)
for name in ['logical_ci_trace.rds','pmax.rds','adjacency.rds']:
    p=ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name;deps[str(p)]=sha(p)
def frozen():
    for p,h in deps.items():assert sha(p)==h,p
    for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
env=jobs.environment();env['FASTKPC_DCC_GAMMA_VECTOR_POSTBACK']='1'
env['FASTKPC_DIAG_SHARED_HOST_QUALIFICATION']='1'
write(base/'protocol.json',dict(status='PREDECLARED',dependencies=deps,source_sha256=build['source_sha256'],
  hard_gates='all existing canonical bitwise/authority/graph/RNG/resource/capacity gates',
  runtime_settings='accepted postback ON / work-pair ON / existing slots and lanes',
  repetitions=1,gpu1_exclusive=True,gpu0_background_allowed=True,performance_claim=False))
jobs.job(base,'canonical',['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
  'Rscript','--vanilla',str(TOOL/'canonical.R'),str(out),'1',str(reference),'-'],env,frozen)
report=read(out/'report.json');assert all(report['gates'].values())
write(BASE/'closure.json',dict(status='PASS_BUILD_FIX_PARENT',native_sha256=sha(native),
  source_sha256=build['source_sha256'],build_identity_sha256=sha(BASE/'object-payload-comparison.json'),
  method_qualification_sha256={m:sha(BASE/m/'qualification/closure.json') for m in ['parent','full','incremental']},
  canonical_report_sha256=sha(out/'report.json'),canonical_elapsed_sec=report['elapsed_sec'],
  canonical_gate_count=len(report['gates']),installed=False,
  evidence_class='full source-closure and actual incremental rebuild; not clean committed producer or promotion'))
print('PASS fixed build parent; canonical',report['elapsed_sec'],flush=True)
