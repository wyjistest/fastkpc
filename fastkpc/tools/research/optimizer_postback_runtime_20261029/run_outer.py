"""Predeclared same-native OFF/ON canonical campaign, conditional on GPU qualification."""
from pathlib import Path
import shutil
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import ROOT,TOOL,BASE,sha,read,write,environment,processes,job

attempt,qualification=sys.argv[1:]
out=BASE/attempt;qa=out/qualification;base=out/'canonical'
assert read(qa/'closure.json')['status']=='PASS_RUNTIME_QUALIFICATION'
assert read(qa/'static-gates.json')['status']=='PASS'
build=read(out/'build.json');native=out/'fastkpc_cuda.so'
assert sha(native)==build['native_sha256']
baseline=read(BASE/'baseline/identity.json')
reference=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/on2/receipt.rds'
reference_manifest=read(reference.parent.parent/'manifest.json')
assert sha(reference)==reference_manifest['files']['on2/receipt.rds']
base.mkdir()
write(base/'build.json',dict(status='PASS',source_sha256=build['source_sha256'],
  native_path=str(native),native_sha256=sha(native),previous_native_sha256=baseline['native_sha256'],
  qualification_closure_sha256=sha(qa/'closure.json'),evidence_class='same-native source-closure development'))
runner=TOOL/'run.R';shutil.copyfile(runner,base/'run.R')
deps={str(p):sha(p) for p in [native,reference,runner,base/'run.R',base/'build.json',qa/'closure.json',
  TOOL/'run_outer.py',TOOL/'support.py',TOOL.parent/'stage_b_work_pair_runtime_20261028/support.py',
  ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
  ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']}
for p in (ROOT/'fastkpc/R').glob('*.R'):deps[str(p)]=sha(p)
for p in [TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R',
  ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
  ROOT/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds']:
    deps[str(p)]=sha(p)
for name in ['logical_ci_trace.rds','pmax.rds','adjacency.rds']:
    p=ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name;deps[str(p)]=sha(p)
def frozen():
    for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in deps.items():assert sha(p)==h,p
env=environment();frozen();assert not processes(),'external GPU job; campaign not started'
sequence=[('off',1),('on',1),('on',2),('off',2),('off',3),('on',3)]
write(base/'protocol.json',dict(schema='postback-runtime-canonical-campaign-v1',epoch=time.time(),
  sequence=sequence,dependencies=deps,physical_gpu=1,fresh_process=True,fresh_result_and_target_caches=True,
  same_native=True,original_opt_in_settings=True,
  gates=['all original CI/pMax bitwise, authority, graph/RNG, physical/cohort, resource and capacity gates',
    'actual backend q37/q46 counts','OFF original raw capacity checkpoints exact','all ON other-q counts zero'],
  performance_gate=dict(on_median_ratio_max=.98,paired_wins=3),
  stage_only_times_not_savings=True,discard_L8_variation=False,production_default=False,promotion=False))
reports=[];previous={}
for variant,cycle in sequence:
    name=variant+str(cycle);dest=base/name;dest.mkdir()
    current=env.copy();current['FASTKPC_DCC_GAMMA_VECTOR_POSTBACK']='1' if variant=='on' else '0'
    command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(base/'run.R'),
      str(dest),'1',str(reference),previous.get(variant,'-')]
    print('START',name,time.strftime('%H:%M:%S'),flush=True);job(base,name,command,current,frozen)
    report=read(dest/'report.json');assert all(report['gates'].values())
    assert report['native_sha256']==build['native_sha256']
    reports.append(dict(variant=variant,cycle=cycle,elapsed_sec=report['elapsed_sec'],report=str(dest/'report.json')))
    previous[variant]=str(dest/'receipt.rds')
medians={v:statistics.median(r['elapsed_sec'] for r in reports if r['variant']==v) for v in ['off','on']}
wins=sum(next(r['elapsed_sec'] for r in reports if r['variant']=='on' and r['cycle']==i)<
  next(r['elapsed_sec'] for r in reports if r['variant']=='off' and r['cycle']==i) for i in [1,2,3])
ratio=medians['on']/medians['off'];ok=ratio<=.98 and wins==3
write(base/'closure.json',dict(status='GO_DEFAULT_OFF_RUNTIME_DEVELOPMENT' if ok else 'STOP_OUTER_WALL_TIME',
  medians=medians,ratio=ratio,paired_wins=wins,runs=reports,native_sha256=sha(native),
  production_default=False,promotion=False,candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
print('COMPLETE outer campaign',medians,'ratio',ratio,'paired wins',wins,flush=True)
