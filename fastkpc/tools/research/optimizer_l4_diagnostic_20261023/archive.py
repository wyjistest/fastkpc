"""Bind L4 captured diagnostic evidence; never change runtime/promotion status."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parents[4];tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda name:json.loads((out/name).read_text())
rows=lambda name:list(csv.DictReader((out/name).open()))
capture=read('capture.json');capacity=read('capacity-summary.json')
assert all(capture['gates'].values()) and capture['ci_count']==234152
assert (capture['l4_windows'],capture['l4_setups'],capture['l4_targets'])==(34,2154,17767)
assert capture['receipt_sha256']==sha(out/'capture.rds')
assert capture['runner_sha256']==sha(tool/'capture.R')
assert len(capture['fixture_sha256'])==34
assert all(sha(out/'raw'/p)==h for p,h in capture['fixture_sha256'].items())
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_slot_runtime_20261023'
assert sha(prior/'on1/receipt.rds')==capture['reference_sha256']
native=root/'fastkpc/build/fastkpc_cuda.so'
dependencies={}
for name in ['capture-build.json','replay-build.json']:
 build=read(name)
 assert sha(native)==build['native_sha256']==capture['native_sha256']
 assert all(sha(root/p)==h for p,h in build['source_closure'].items())
 assert all(sha(root/p)==h for p,h in build['sources'].items())
 assert all(sha(out/p)==h for p,h in build['outputs'].items())
 assert build['prior_manifest_sha256']==sha(prior/'manifest.json')
 dependencies.update(build['sources'])
assert capacity['status'] in ['PASS','STOP_FULL_LAYER_CAPACITY'] and capacity['resources_closed']
census=[r for r in rows('capacity-allocations.csv') if r['phase']=='individual-census']
attempt=[r for r in rows('capacity-allocations.csv') if r['phase']=='resident-attempt']
assert len(census)==capacity['census_setups']==capacity['setups']==2154
assert [int(r['setup']) for r in census]==list(range(2154))
assert sum(int(r['targets']) for r in census)==capacity['targets']==17767
assert sum(int(r['application_bytes']) for r in census)==capacity['application_bytes_sum']
assert len(attempt)==capacity['resident_created'] and 0<len(attempt)<=2154
assert [int(r['setup']) for r in attempt]==list(range(len(attempt)))
assert int(attempt[-1]['free_after'])==capacity['free_resident_boundary']
assert all(int(r['free_after'])>=capacity['headroom_required'] for r in attempt[:-1])
resident=capacity['status']=='PASS';policy='resident' if resident else 'windows'
if resident:
 assert len(attempt)==2154 and capacity['free_resident_boundary']>=capacity['headroom_required']
else:
 assert capacity['free_resident_boundary']<capacity['headroom_required']
checks={name:read(name+'-summary.json') for name in ['memcheck','campaign']}
for name,r in checks.items():
 assert r['status']=='PASS' and r['policy']==policy and r['all_outputs_bitwise_exact']
 assert r['resources_closed'] and r['output_leases_closed'] and r['forbidden_calls']==0
 assert r['minimum_free_after_create']>=capacity['headroom_required']
assert checks['campaign']['runs']==14 and checks['campaign']['setups_per_run']==2154
assert checks['campaign']['targets_per_run']==17767
assert checks['campaign']['compared_target_occurrences']==14*17767
mem=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
assert {r['targets'] for r in rows('memcheck-allocations.csv')}==set(capacity['target_capacities'])
process={name:read(name+'-process.json') for name in ['capture','capacity','memcheck','campaign']}
assert all(p['exit_code']==0 and p['physical_gpu']==1 and not p['unexpected_gpu_pids'] for p in process.values())
assert all(p['runner_sha256']==sha(tool/'run.py') and p['native_sha256']==sha(native) for p in process.values())
all_runs=rows('campaign-runs.csv');timed=[r for r in all_runs if r['kind']=='timing']
assert len(all_runs)==14 and len(timed)==12
assert all(r['parity']=='1' and r['forbidden_calls']=='0' and int(r['targets'])==17767 for r in all_runs)
candidate='R' if resident else 'W';metrics=['native_ms','reported_ms','pack_ms','verify_ms','token_release_ms',
 'pool_prepare_ms','pool_destroy_ms','segmented_work_ms','lifecycle_work_ms']
native_calls=rows('campaign-calls.csv')
for r in all_runs:
 group=[c for c in native_calls if (c['cycle'],c['kind'],c['variant'])==(r['cycle'],r['kind'],r['variant'])]
 assert len(group)==int(r['calls'])
 assert abs(sum(float(c['native_ms']) for c in group)-float(r['native_ms']))<1e-6
 assert sum(int(c['targets']) for c in group)==17767
 if r['variant']=='R':
  assert len(group)==1 and int(group[0]['setups'])==2154 and int(group[0]['workers'])==32
 else:
  assert [int(c['first_setup']) for c in group]==list(range(0,2154,64))
  assert [int(c['original_window']) for c in group]==list(range(1,35))
  assert [int(c['setups']) for c in group]==[64]*33+[42]
  assert all(int(c['workers'])==min(64 if r['variant']=='A' else 32,int(c['setups'])) for c in group)
seconds={}
for variant in ['A',candidate]:
 group=[r for r in timed if r['variant']==variant];assert len(group)==6
 assert all(int(r['calls'])==(1 if variant=='R' else 34) for r in group)
 seconds[variant]={m.removesuffix('_ms')+'_sec':dict(median=statistics.median(float(r[m])/1000 for r in group),
  minimum=min(float(r[m])/1000 for r in group),maximum=max(float(r[m])/1000 for r in group)) for m in metrics}
paired=[]
for cycle in range(6):
 order=[r['variant'] for r in timed if int(r['cycle'])==cycle]
 assert order==(['A',candidate] if cycle%2==0 else [candidate,'A'])
 group={r['variant']:r for r in timed if int(r['cycle'])==cycle};assert set(group)=={'A',candidate}
 assert group['A']['pool_prepare_ms']==group[candidate]['pool_prepare_ms']
 assert group['A']['pool_destroy_ms']==group[candidate]['pool_destroy_ms']
 paired.append(float(group[candidate]['native_ms'])/float(group['A']['native_ms']))
ratio=seconds[candidate]['native_sec']['median']/seconds['A']['native_sec']['median']
wins=sum(r<1 for r in paired)
tested='RESIDENT' if resident else 'WINDOW_WORKER'
status=('GO_L4_'+tested+'_DIAGNOSTIC' if ratio<=.90 and wins==6 else
 'CONDITIONAL_L4_'+tested+'_DIAGNOSTIC' if ratio<.97 else 'STOP_L4_'+tested+'_PERFORMANCE')
telemetry={}
for name in process:
 raw=list(csv.reader((out/(name+'-gpu.csv')).open()))
 telemetry[name]=dict(gpu1_sampled_peak_mib=max(int(r[2]) for r in raw if len(r)==7 and int(r[1])==1),
  gpu0_peak_utilization=max(int(r[3]) for r in raw if len(r)==7 and int(r[1])==0))
closure=dict(schema='optimizer-l4-diagnostic-closure-v1',status=status,hard_gates='PASS',
 evidence_class='GPU1 shared-host captured-layer source-closure development; not full-method or promotion',
 native_sha256=sha(native),source_sha256=build['source_closure'],layer=4,setups=2154,targets=17767,
 original_windows=34,capture_ci_count=capture['ci_count'],capacity=capacity,policy=policy,
 timing_variant=candidate,seconds=seconds,candidate_over_A_ratio=ratio,paired_ratios=paired,paired_wins=wins,
 predeclared_go_ratio=.90,predeclared_stop_ratio=.97,
 all_outputs_bitwise_exact=True,resources_closed=True,
 compared_target_occurrences=sum(r['compared_target_occurrences'] for r in checks.values()),
 memcheck=checks['memcheck'],memcheck_errors=0,memcheck_leaked_bytes=0,
 telemetry=telemetry,processes=process,device_overlap_claimed=False,
 window_pause_limitation=not resident,production_cache_capacity_qualified=False,
 runtime_changed=False,runtime_integration='NOT_YET_GO',phase10='ACTIVE',
 promotion_candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
for p in ['fastkpc/tools/with_phase10_exact_development_env.sh','fastkpc/tools/run_phase10_pvalue_contract_audit.R',
 'fastkpc/R/fast_kpc.R','fastkpc/R/full_cuda_ci_phase35_vertical.R']:
 dependencies[p]=sha(root/p)
manifest=dict(schema='optimizer-l4-diagnostic-manifest-v1',native_sha256=sha(native),
 source_sha256=build['source_closure'],tools={str(p.relative_to(root)):sha(p) for p in tool.iterdir() if p.is_file()},
 dependencies=dependencies,references={str(p.relative_to(root)):sha(p) for p in [prior/'manifest.json',prior/'on1/receipt.rds',
 root/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']},
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['source_sha256','processes','memcheck']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
