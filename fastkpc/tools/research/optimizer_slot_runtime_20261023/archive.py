"""Freeze L5 opt-in integration evidence without changing any promotion state."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parents[4];tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads((out/p).read_text())
build=read('build.json');checks=read('checks.json');device=read('device-code.json')
assert build['status']==checks['status']==device['status']=='PASS'
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']==checks['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert all(sha(out/p)==h for p,h in checks['outputs'].items())
qualified_runner=out/'qualified-run.R'
if not qualified_runner.exists():
 qualified_runner.write_bytes((tool/'run.R').read_bytes())
assert qualified_runner.read_bytes().rstrip(b'\n')+b'\n'==(tool/'run.R').read_bytes()
reports={};peaks={};processes={};input_references={}
for name in ['off1','on1','on2','off2','off3','on3']:
 r=read(name+'/report.json');p=read(name+'-process.json')
 assert r['gates'] and all(r['gates'].values()),name
 assert sha(out/name/'receipt.rds')==r['receipt_sha256']
 assert r['runner_sha256']==sha(qualified_runner)
 assert r['reference_sha256']==sha(root/p['command'][-2])
 input_references[str((root/p['command'][-2]).relative_to(root))]=r['reference_sha256']
 assert r['source_sha256']==build['source_sha256']
 for field,filename in [('logical_ci_trace','logical_ci_trace.rds'),('pmax','pmax.rds'),('adjacency','adjacency.rds')]:
  authority=root/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/filename
  assert r['authority_sha256'][field]==sha(authority)
  input_references[str(authority.relative_to(root))]=r['authority_sha256'][field]
 frozen=root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds'
 assert r['comparisons']['frozen_l2']['receipt_sha256']==sha(frozen)
 input_references[str(frozen.relative_to(root))]=r['comparisons']['frozen_l2']['receipt_sha256']
 assert r['native_sha256']==build['native_sha256']==p['native_sha256']
 assert p['exit_code']==0 and p['physical_gpu']==1 and not p['unexpected_gpu_pids']
 assert r['comparisons']['changed_ci_count']==r['comparisons']['changed_pmax_count']==0
 assert r['comparisons']['max_ci_difference']==0 and r['comparisons']['alpha_flips']==0
 assert len(r['oracle_errors'])==2 and max(r['oracle_errors'])<=1e-10
 rows=list(csv.reader((out/(name+'-gpu.csv')).open()))
 peaks[name]=max(int(row[2]) for row in rows if len(row)==7 and int(row[1])==1)
 reports[name]=r;processes[name]=p
medians={}
for mode in ['off','on']:
 group=[reports[mode+str(i)] for i in range(1,4)]
 medians[mode]={k:statistics.median(r['stages'][k] for r in group) for k in
  ['elapsed_sec','optimizer_sec','stage_a_sec','stage_b_sec','native_setup_sec','prepared_build_sec']}
 medians[mode]['l5_optimizer_sec']=statistics.median(
  next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==5) for r in group)
 medians[mode]['optimizer_by_level_sec']={str(level):statistics.median(
  next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==level)
  for r in group) for level in range(1,9)}
ratio=medians['on']['elapsed_sec']/medians['off']['elapsed_sec']
wins=sum(reports['on'+str(i)]['elapsed_sec']<reports['off'+str(i)]['elapsed_sec'] for i in range(1,4))
status='GO_L5_OPT_IN_RUNTIME_DEVELOPMENT' if ratio<=.97 and wins==3 else 'CONDITIONAL_L5_RUNTIME_PERFORMANCE'
closure=dict(schema='optimizer-l5-slot-runtime-closure-v1',status=status,hard_gates='PASS',
 evidence_class='GPU1 shared-host full-build source-closure development; not clean producer or promotion',
 native_sha256=build['native_sha256'],producer_head=build['producer_head'],
 runner_identity=dict(qualified_file='qualified-run.R',qualified_sha256=sha(qualified_runner),
  current_tool_sha256=sha(tool/'run.R'),
  change='trailing blank-line cleanup only' if sha(qualified_runner)!=sha(tool/'run.R') else 'unchanged'),
 full_canonical_runs=6,logical_ci_per_run=240498,ci_and_pmax_bits_exact=True,
 device_functions_exact=checks['device_functions'],checks=checks,
 seconds_medians=medians,on_over_off_outer_ratio=ratio,paired_on_wins=wins,
 elapsed_sec_by_run={name:r['elapsed_sec'] for name,r in reports.items()},
 predeclared_ratio_limit=.97,gpu1_sampled_total_peak_mib=peaks,
 l5_on_counters={name:r['stages']['l5_slots'] for name,r in reports.items() if name.startswith('on')},
 authority_max_ci_error=max(r['oracle_errors'][0] for r in reports.values()),
 authority_max_pmax_error=max(r['oracle_errors'][1] for r in reports.values()),
 processes=processes,default_enabled=False,qualified_scope='canonical S2 L5 only',
 other_layers_qualified=False,new_real_data_qualified=False,new_final_pdag_qualified=False,
 phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
dependencies=[root/p for p in [
 'fastkpc/tests/test_full_cuda_ci_l5_optimizer_slots.R',
 'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.cpp',
 'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.hpp',
 'fastkpc/tools/research/optimizer_layer_refill_20261023/layer_fixture.hpp',
 'fastkpc/tools/research/optimizer_windows_20261022/fixture.hpp',
 'fastkpc/tools/research/optimizer_status_wait_20261023/device_code.py',
 'fastkpc/tools/research/structured_prefill_canonical_20261022/build.py',
 'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
 'fastkpc/tools/with_phase10_exact_development_env.sh',
 'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
 'fastkpc/R/fast_kpc.R',
 'fastkpc/R/full_cuda_ci_phase35_vertical.R']]
references=[root/'fastkpc/artifacts/full_cuda_ci'/name/'manifest.json' for name in
 ['optimizer_slot_pool_20261023','optimizer_layer_refill_20261023','optimizer_status_wait_20261023']]
manifest=dict(schema='optimizer-l5-slot-runtime-manifest-v1',native_sha256=build['native_sha256'],
 source_sha256=build['source_sha256'],
 tools={str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()},
 dependencies={str(p.relative_to(root)):sha(p) for p in dependencies},
 references={**{str(p.relative_to(root)):sha(p) for p in references},**input_references},
 files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['processes','checks']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
