"""Predeclared interleaved capacity/non-regression campaign; no setting sweep."""
import importlib.util
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,PARENT,BASE,TOOL,sha,read,write
support=TOOL.parent/'stage_b_work_pair_runtime_20261028/support.py'
spec=importlib.util.spec_from_file_location('jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
assert read(PARENT/'closure.json')['status']=='PASS_BUILD_FIX_PARENT'
assert read(BASE/'qualification/closure.json')['status']=='PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES'
base=BASE/'canonical';base.mkdir()
builds={'off':read(PARENT/'full/build.json'),'on':read(BASE/'candidate/build.json')}
snapshots={'off':PARENT/'source','on':BASE/'candidate/source'}
reference=PARENT/'canonical/qualification/receipt.rds'
runner=TOOL.parent/'build_consistency_20261029/canonical.R'
deps={str(p):sha(p) for p in [reference,runner,TOOL/'campaign.py',TOOL/'README.md',support,
    PARENT/'closure.json',BASE/'qualification/closure.json',
    ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
    ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds',
    TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R',
    ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    ROOT/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds']}
for p in (ROOT/'fastkpc/R').glob('*.R'):deps[str(p)]=sha(p)
for name in ['logical_ci_trace.rds','pmax.rds','adjacency.rds']:
    p=ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name;deps[str(p)]=sha(p)
for variant,b in builds.items():
    folder=base/variant;folder.mkdir()
    current=dict(b)
    current['previous_native_sha256']=builds['off']['native_sha256']
    current['source_sha256']={str(snapshots[variant]/p):h for p,h in b['source_sha256'].items()}
    write(folder/'build.json',current)
    deps[b['native_path']]=b['native_sha256']
    deps[str(folder/'build.json')]=sha(folder/'build.json')
def frozen():
    for p,h in deps.items():assert sha(p)==h,p
    for v,b in builds.items():
        for p,h in b['source_sha256'].items():assert sha(snapshots[v]/p)==h,p
    for p,h in builds['on']['source_sha256'].items():assert sha(ROOT/p)==h,p
env=jobs.environment();env['FASTKPC_DCC_GAMMA_VECTOR_POSTBACK']='1'
sequence=[('off',1),('on',1),('on',2),('off',2),('off',3),('on',3)]
gate=dict(median_nonregression_ratio_max=1.02,each_pair_ratio_max=1.05,
    speed_median_ratio_max=.98,speed_paired_wins=3,
    measured_pool_bytes_reduction=630789120,minimum_free_improvement_bytes=600*1024**2)
write(base/'protocol.json',dict(schema='structured-rows-capacity-campaign-v1',epoch=time.time(),
    sequence=sequence,dependencies=deps,gate=gate,physical_gpu=1,
    fresh_process=True,fresh_dataset_caches=True,postback=True,work_pair=True,
    changed_lanes_or_slots=False,performance_gate_predeclared=True,
    capacity_may_pass_without_speed=True,promotion=False))
reports={};previous={};runs=[]
for variant,cycle in sequence:
    name=variant+str(cycle);out=base/variant/name;out.mkdir()
    print('START',name,time.strftime('%H:%M:%S'),flush=True)
    jobs.job(base,name,['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
        'Rscript','--vanilla',str(runner),str(out),'1',str(reference),previous.get(variant,'-')],env,frozen)
    report=read(out/'report.json');assert all(report['gates'].values())
    assert report['native_sha256']==builds[variant]['native_sha256']
    reports[(variant,cycle)]=report
    previous[variant]=str(out/'receipt.rds')
    runs.append(dict(variant=variant,cycle=cycle,elapsed_sec=report['elapsed_sec'],report=str(out/'report.json')))
medians={v:statistics.median(reports[(v,i)]['elapsed_sec'] for i in [1,2,3]) for v in ['off','on']}
pairs=[];capacity_pass=True
for cycle in [1,2,3]:
    off,on=reports[('off',cycle)],reports[('on',cycle)]
    rows=[]
    for a,b in zip(off['level_component_prefill'],on['level_component_prefill']):
        assert a['level']==b['level']
        x,y=a['structured'],b['structured']
        delta=x['pool_device_bytes']-y['pool_device_bytes']
        same_solver=x['solver_workspace_bytes']==y['solver_workspace_bytes']
        capacity_pass &= delta==gate['measured_pool_bytes_reduction'] and same_solver
        rows.append(dict(level=a['level'],pool_reduction_bytes=delta,solver_workspace_exact=same_solver,
            off_pool_bytes=x['pool_device_bytes'],on_pool_bytes=y['pool_device_bytes'],
            off_sampled_used_bytes=x['sampled_device_used_bytes'],on_sampled_used_bytes=y['sampled_device_used_bytes']))
    checkpoints={}
    for pool in ['l3_pool','l4_pool','l6_pool']:
        a,b=off['stages'][pool],on['stages'][pool]
        delta=b['minimum_free_bytes']-a['minimum_free_bytes']
        capacity_pass &= delta>=gate['minimum_free_improvement_bytes']
        names=['free_before_create','free_after_create','minimum_free_bytes','maximum_pool_increment_bytes',
               'free_after_optimizer','capacity_checkpoints','stage_a_capacity_checkpoints']
        checkpoints[pool]=dict(off={k:a[k] for k in names if k in a},on={k:b[k] for k in names if k in b},
            minimum_free_improvement_bytes=delta)
    pairs.append(dict(cycle=cycle,ratio=on['elapsed_sec']/off['elapsed_sec'],levels=rows,checkpoints=checkpoints))
ratio=medians['on']/medians['off']
nonregression=ratio<=gate['median_nonregression_ratio_max'] and all(p['ratio']<=gate['each_pair_ratio_max'] for p in pairs)
wins=sum(p['ratio']<1 for p in pairs)
speed=ratio<=gate['speed_median_ratio_max'] and wins==gate['speed_paired_wins']
accepted=capacity_pass and nonregression
write(base/'closure.json',dict(status='GO_CAPACITY' if accepted else 'STOP_CAPACITY_OR_OUTER_REGRESSION',
    numerical_behavior='PASS',capacity='PASS' if capacity_pass else 'FAIL',
    outer_nonregression='PASS' if nonregression else 'FAIL',speed='GO' if speed else 'NOT_PROVEN',
    gate=gate,medians=medians,ratio=ratio,paired_wins=wins,pairs=pairs,runs=runs,
    native_sha256={v:b['native_sha256'] for v,b in builds.items()},
    stage_medians={v:{s:statistics.median(reports[(v,i)]['stages'][s] for i in [1,2,3])
        for s in ['optimizer_sec','stage_b_sec','stage_a_sec','native_setup_sec','consumer_dcov_sec']}
        for v in ['off','on']},
    production_default=False,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
print('COMPLETE',medians,'ratio',ratio,'capacity',capacity_pass,'nonregression',nonregression,'speed',speed,flush=True)
