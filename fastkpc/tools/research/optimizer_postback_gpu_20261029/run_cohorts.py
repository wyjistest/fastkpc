"""Full original-order 32-slot optimizer qualification and paired layer timings."""
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,TOOL,sha,write
sys.path.insert(0,str(TOOL.parent/'stage_b_work_pair_runtime_20261028'))
from support import environment,processes,job
out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
assert mode in ('qualification','timing','profile')
env=environment();env['FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H']='1'
base=out/('cohorts-'+mode);base.mkdir(exist_ok=False)
identity=json.loads((out/'source.json').read_text())
l4=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023'
l5=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023'
deps={}
for capture,count in [(l4,34),(l5,15)]:
    manifest=json.loads((capture/'manifest.json').read_text())
    files={p:h for p,h in manifest['files'].items() if p.startswith('raw/') and p.endswith('.bin')}
    assert len(files)==count
    for p,h in files.items():assert sha(capture/p)==h,p;deps[str(capture/p)]=h
    deps[str(capture/'manifest.json')]=sha(capture/'manifest.json')
variants=['reference-profile','candidate-profile'] if mode=='profile' else ['reference','candidate']
for v in variants:
    build=json.loads((out/('cohorts-'+v+'-build.json')).read_text())
    executable=out/('cohorts-'+v)
    assert sha(executable)==build['executable_sha256'] and sha(TOOL/'cohorts.cpp')==build['harness_sha256']
    native=ROOT/'fastkpc/build/fastkpc_cuda.so' if v=='reference' else out/v/'fastkpc_cuda.so'
    assert sha(native)==build['native_sha256']
    deps[str(executable)]=sha(executable);deps[str(native)]=sha(native)
deps[str(out/'audit.so')]=sha(out/'audit.so')
deps[str(TOOL/'cohorts.cpp')]=sha(TOOL/'cohorts.cpp')
deps[str(TOOL/'run_cohorts.py')]=sha(TOOL/'run_cohorts.py')
def frozen():
    for p,h in identity['production_source_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in deps.items():assert sha(p)==h,p
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==identity['native_sha256']
frozen();assert not processes(),'external GPU compute; no job started'
if mode=='qualification':
    sequence=[('reference-reduced','reference','reduced',None),('candidate-reduced','candidate','reduced',None),
      ('candidate-memcheck','candidate','reduced','memcheck'),('candidate-synccheck','candidate','reduced','synccheck'),
      ('reference-full','reference','verify',None),('candidate-full','candidate','verify',None)]
elif mode=='timing':
    assert json.loads((out/'cohorts-qualification/closure.json').read_text())['status']=='PASS_FULL_COHORT_BITWISE'
    sequence=[(v+'-warmup',v,'timing',None) for v in variants]
    for cycle in range(3):
        for v in (variants if cycle%2==0 else list(reversed(variants))):
            sequence.append((v+'-'+str(cycle+1),v,'timing',None))
else:
    assert json.loads((out/'cohorts-qualification/closure.json').read_text())['status']=='PASS_FULL_COHORT_BITWISE'
    sequence=[(v,v,'verify',None) for v in variants]
write(base/'protocol.json',dict(schema='postback-captured-cohort-v1',epoch=time.time(),mode=mode,dependencies=deps,
    sequence=sequence,level_counts={'4':[2154,17767],'5':[955,6816]},slots=32,
    numerical_authority='GPU only',original_request_order=True,original_cohorts=True,
    gates=['captured optimizer state/trajectory/coefficients/fitted/residual bitwise','resource closure',
      'no device resource mutations inside workers','device free >=2GiB'],
    timing_gate=dict(combined_work_ratio_max=.95,each_layer_ratio_max=1.03,
      combined_charged_lifecycle_ratio_max=.95,minimum_paired_combined_wins=3),
    full_outer_after_layer_go=True,production_integration=False))
records=[]
for name,v,kind,sanitizer in sequence:
    command=[str(out/('cohorts-'+v)),str(l4/'raw'),str(l5/'raw'),str(base/name),kind]
    if sanitizer:
        command=['/usr/local/cuda/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','86',
            *(['--leak-check','full'] if sanitizer=='memcheck' else []),*command]
    print('START',name,time.strftime('%H:%M:%S'),flush=True);job(base,name,command,env,frozen)
    if sanitizer:assert 'ERROR SUMMARY: 0 errors' in (base/(name+'.log')).read_text()
    layers=[json.loads((base/(name+'-L'+str(level)+'.json')).read_text()) for level in [4,5]]
    if kind!='reduced':
        assert [(x['setups'],x['targets']) for x in layers]==[(2154,17767),(955,6816)]
    assert all(x['status']=='PASS' and x['resource_closed'] for x in layers)
    records.append(dict(name=name,variant=v,layers=layers))
status='PASS_FULL_COHORT_BITWISE' if mode=='qualification' else 'PASS_PROFILE'
analysis={}
if mode=='timing':
    timed=[r for r in records if not r['name'].endswith('warmup')]
    for v in variants:
        runs=[r for r in timed if r['variant']==v]
        analysis[v]=dict(layer_work_medians=[statistics.median(r['layers'][i]['work_sec'] for r in runs) for i in [0,1]],
            combined_work_median=statistics.median(sum(x['work_sec'] for x in r['layers']) for r in runs),
            charged_lifecycle_median=statistics.median(sum(x['charged_lifecycle_sec'] for x in r['layers']) for r in runs))
    r,c=analysis['reference'],analysis['candidate']
    analysis['combined_work_ratio']=c['combined_work_median']/r['combined_work_median']
    analysis['layer_ratios']=[c['layer_work_medians'][i]/r['layer_work_medians'][i] for i in [0,1]]
    analysis['charged_lifecycle_ratio']=c['charged_lifecycle_median']/r['charged_lifecycle_median']
    wins=0
    for cycle in range(1,4):
        pair={v:sum(x['work_sec'] for x in next(r for r in timed if r['name']==v+'-'+str(cycle))['layers']) for v in variants}
        wins+=pair['candidate']<pair['reference']
    analysis['paired_wins']=wins
    ok=analysis['combined_work_ratio']<=.95 and max(analysis['layer_ratios'])<=1.03 and analysis['charged_lifecycle_ratio']<=.95 and wins==3
    status='GO_ISOLATED_OUTER_QUALIFICATION' if ok else 'STOP_LAYER_WALL_TIME'
write(base/'closure.json',dict(status=status,analysis=analysis,runs=records,
    production_native_unchanged=True,runtime_integration='NOT_YET_GO'))
print(status,json.dumps(analysis),flush=True)
