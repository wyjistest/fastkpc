"""Conditional isolated cross-binary canonical experiment, never installs candidate."""
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,TOOL,sha,write,replace
sys.path.insert(0,str(TOOL.parent/'stage_b_work_pair_runtime_20261028'))
from support import environment,processes,job
out=Path(sys.argv[1]).resolve()
gate=json.loads((out/'cohorts-timing/closure.json').read_text())
assert gate['status']=='GO_ISOLATED_OUTER_QUALIFICATION'
base=out/'canonical';base.mkdir(exist_ok=False)
identity=json.loads((out/'source.json').read_text())
reference=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/on2/receipt.rds'
original=ROOT/'fastkpc/build/fastkpc_cuda.so';candidate=out/'candidate/fastkpc_cuda.so'
paths={'reference':str(original),'candidate':str(candidate)}
hashes={k:sha(v) for k,v in paths.items()}
assert hashes['reference']==identity['native_sha256']
candidate_build=json.loads((out/'candidate/build.json').read_text())
assert hashes['candidate']==candidate_build['native_sha256']
overlay={str(p):sha(p) for p in (out/'source').rglob('*') if p.is_file()}
write(base/'build.json',dict(status='PASS',source_sha256=identity['production_source_sha256'],
    native_paths=paths,native_hashes=hashes,previous_native_sha256=hashes['reference'],
    candidate_inputs_sha256=overlay,candidate_build_sha256=sha(out/'candidate/build.json'),
    evidence_class='isolated research overlay, no production integration'))
runner_source=TOOL.parent/'stage_b_work_pair_runtime_20261028/run.R'
r=runner_source.read_text()
start=r.index('evidence_class <-');end=r.index('\nSys.setenv(',start)
r=r[:start]+'''evidence_class <- "isolated cross-binary research overlay; not production integration or promotion"'''+r[end:]
r=replace(r,'native <- "fastkpc/build/fastkpc_cuda.so"; native_sha <- sha(native)', '''variant <- Sys.getenv("FASTKPC_POSTBACK_VARIANT")
stopifnot(variant %in% c("reference","candidate"))
native <- build$native_paths[[variant]]; native_sha <- sha(native)
build$native_sha256 <- build$native_hashes[[variant]]
candidate_inputs <- unlist(build$candidate_inputs_sha256)
stopifnot(identical(candidate_inputs,setNames(vapply(names(candidate_inputs),sha,""),names(candidate_inputs))))
.fastkpc_cuda_so <- local({ path <- native; function() path })''')
r=replace(r,'''g$reference_native <- identical(native_sha,reference$native_sha256) ||
  (work_pair_mode=="0" && args[[4L]]=="-" && identical(reference$native_sha256,build$previous_native_sha256))''',
'''g$reference_native <- identical(reference$native_sha256,build$previous_native_sha256)
g$research_candidate_identity <- identical(native_sha,build$native_hashes[[variant]]) &&
  identical(candidate_inputs,setNames(vapply(names(candidate_inputs),sha,""),names(candidate_inputs)))''')
r=replace(r,'schema="structured-work-pair-runtime-canonical-v1"','schema="postback-isolated-canonical-v1"')
r=replace(r,'sha("fastkpc/tools/research/stage_b_work_pair_runtime_20261028/run.R")','sha('+json.dumps(str(base/'run.R'))+')')
(base/'run.R').write_text(r)
deps={str(p):sha(p) for p in [original,candidate,reference,base/'run.R',base/'build.json',TOOL/'run_outer.py',runner_source,
    ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
    ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']}
for p in (ROOT/'fastkpc/R').glob('*.R'):deps[str(p)]=sha(p)
for p in [TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R',ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    ROOT/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds']:
    deps[str(p)]=sha(p)
for name in ['logical_ci_trace.rds','pmax.rds','adjacency.rds']:
    p=ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name;deps[str(p)]=sha(p)
deps.update(overlay)
def frozen():
    for p,h in identity['production_source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in deps.items():assert sha(p)==h,p
env=environment();frozen();assert not processes(),'external GPU job; no job started'
sequence=[('reference',1),('candidate',1),('candidate',2),('reference',2),('reference',3),('candidate',3)]
write(base/'protocol.json',dict(schema='postback-isolated-canonical-campaign-v1',epoch=time.time(),
    sequence=sequence,dependencies=deps,physical_gpu=1,fresh_process=True,fresh_result_and_target_caches=True,
    original_opt_in_settings=True,hard_gates='all original runner gates, including all 240498 CI/pMax bits, authority, graph/RNG, physical cohorts and resources',
    changed_identity_gate='reference must match authenticated original native; current run matches declared variant hash',
    performance_gate=dict(candidate_median_ratio_max=.98,paired_wins=3),
    stage_only_times_not_savings=True,production_integration=False))
reports=[];previous={}
for variant,cycle in sequence:
    name=variant+str(cycle);dest=base/name;dest.mkdir();current=env.copy();current['FASTKPC_POSTBACK_VARIANT']=variant
    command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(base/'run.R'),
      str(dest),'1',str(reference),previous.get(variant,'-')]
    print('START',name,time.strftime('%H:%M:%S'),flush=True);job(base,name,command,current,frozen)
    report=json.loads((dest/'report.json').read_text());assert all(report['gates'].values())
    assert report['native_sha256']==hashes[variant]
    reports.append(dict(variant=variant,cycle=cycle,elapsed_sec=report['elapsed_sec'],report=str(dest/'report.json')))
    previous[variant]=str(dest/'receipt.rds')
medians={v:statistics.median(r['elapsed_sec'] for r in reports if r['variant']==v) for v in paths}
wins=sum(next(r['elapsed_sec'] for r in reports if r['variant']=='candidate' and r['cycle']==i)<
         next(r['elapsed_sec'] for r in reports if r['variant']=='reference' and r['cycle']==i) for i in [1,2,3])
ratio=medians['candidate']/medians['reference'];ok=ratio<=.98 and wins==3
write(base/'closure.json',dict(status='GO_ISOLATED_POSTBACK_DEVELOPMENT' if ok else 'STOP_OUTER_WALL_TIME',
    medians=medians,ratio=ratio,paired_wins=wins,runs=reports,
    production_native_unchanged=True,runtime_integration='NOT_YET_GO',promotion=False))
print('PASS outer campaign',json.dumps(medians),'ratio',ratio,'paired wins',wins,flush=True)
