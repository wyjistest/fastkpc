"""GPU1-only numerical/capacity qualification; never counts as a timing pair."""
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,PARENT,BASE,TOOL,sha,read,write
support=TOOL.parent/'build_consistency_20261029/numeric_jobs.py'
spec=importlib.util.spec_from_file_location('jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
assert read(BASE/'qualification/closure.json')['status']=='PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES'
build=read(BASE/'candidate/build.json');native=Path(build['native_path'])
reference=PARENT/'canonical/qualification/receipt.rds'
reference_report=read(reference.parent/'report.json')
runner=TOOL.parent/'build_consistency_20261029/canonical.R'
base=BASE/'semantic';base.mkdir();out=base/'qualification';out.mkdir()
write(base/'build.json',build)
deps={str(p):sha(p) for p in [native,reference,runner,TOOL/'canonical_qualification.py',support,
    BASE/'qualification/closure.json',ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
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
write(base/'protocol.json',dict(dependencies=deps,gpu1_exclusive=True,gpu0_background_allowed=True,
    performance_evidence=False,numerical_gate='all retained canonical gates',
    expected_pool_reduction_bytes=630789120,minimum_checkpoint_improvement_bytes=600*1024**2,
    repetitions=1,counts_toward_interleaved_campaign=False))
jobs.job(base,'canonical',['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
    'Rscript','--vanilla',str(runner),str(out),'1',str(reference),'-'],env,frozen)
r=read(out/'report.json');assert all(r['gates'].values())
levels=[];gates={}
for a,b in zip(reference_report['level_component_prefill'],r['level_component_prefill']):
    assert a['level']==b['level']
    x,y=a['structured'],b['structured']
    delta=x['pool_device_bytes']-y['pool_device_bytes']
    same_solver=x['solver_workspace_bytes']==y['solver_workspace_bytes']
    gates['level_'+str(a['level'])]=delta==630789120 and same_solver
    levels.append(dict(level=a['level'],parent_pool_bytes=x['pool_device_bytes'],
        candidate_pool_bytes=y['pool_device_bytes'],pool_reduction_bytes=delta,solver_workspace_exact=same_solver))
checkpoints={}
for pool in ['l3_pool','l4_pool','l6_pool']:
    a,b=reference_report['stages'][pool],r['stages'][pool]
    delta=b['minimum_free_bytes']-a['minimum_free_bytes']
    gates[pool]=delta>=600*1024**2
    checkpoints[pool]=dict(parent_minimum_free_bytes=a['minimum_free_bytes'],
        candidate_minimum_free_bytes=b['minimum_free_bytes'],improvement_bytes=delta)
write(base/'closure.json',dict(status='PASS_NUMERICAL_CAPACITY_PENDING_OUTER' if all(gates.values()) else 'FAIL_CAPACITY',
    native_sha256=sha(native),canonical_gates=len(r['gates']),capacity_gates=gates,
    levels=levels,checkpoints=checkpoints,maximum_authority_error=max(r['oracle_errors']),
    report_sha256=sha(out/'report.json'),performance_evidence=False,
    speed='NOT_TESTED',outer_nonregression='PENDING_ISOLATED_CAMPAIGN',installed=False))
assert all(gates.values()),gates
print('PASS full canonical numerical and measured capacity; isolated outer campaign pending',flush=True)
