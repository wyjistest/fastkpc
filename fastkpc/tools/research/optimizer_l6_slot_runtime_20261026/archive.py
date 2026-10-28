"""Bind L6 bounded-slot integration results, build lineage and hard gates."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(__file__).resolve().parents[4]
tool = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads((out / p).read_text())
build, checks, protocol = (read(p) for p in ['build.json','checks.json','protocol.json'])
assert build['status'] == checks['status'] == 'PASS'
assert sha(root/'fastkpc/build/fastkpc_cuda.so') == build['native_sha256'] == checks['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert all(sha(root/p)==h for p,h in protocol['frozen_tools'].items())
assert all(sha(root/p)==h for p,h in protocol['frozen_dependencies'].items())
reports, processes, peaks, references = {}, {}, {}, {}
for name in protocol['sequence']:
    r, p = read(name+'/report.json'), read(name+'-process.json')
    assert r['gates'] and all(r['gates'].values()), name
    assert sha(out/name/'receipt.rds') == r['receipt_sha256']
    assert r['runner_sha256'] == sha(tool/'run.R')
    ref = root/p['command'][-2]
    assert r['reference_sha256'] == sha(ref)
    references[str(ref.relative_to(root))] = sha(ref)
    assert r['source_sha256'] == build['source_sha256']
    assert r['native_sha256'] == build['native_sha256'] == p['native_sha256']
    assert p['exit_code'] == 0 and p['physical_gpu'] == 1 and not p['unexpected_gpu_pids']
    assert p['l5_slots'] == '1' and r['l5_slots'] == '1'
    assert p['l4_workers32'] == p['l4_slots'] == p['properties_cache'] == '1'
    assert str(r['l6_slots']) == p['l6_slots'] == ('1' if name.startswith('on') else '0')
    assert p['isolated'], 'both GPUs must be isolated in this campaign'
    assert r['comparisons']['changed_ci_count'] == r['comparisons']['changed_pmax_count'] == 0
    assert r['comparisons']['max_ci_difference'] == r['comparisons']['alpha_flips'] == 0
    assert len(r['oracle_errors']) == 2 and max(r['oracle_errors']) <= 1e-10
    for field,filename in [('logical_ci_trace','logical_ci_trace.rds'),('pmax','pmax.rds'),('adjacency','adjacency.rds')]:
        path = root/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/filename
        assert r['authority_sha256'][field] == sha(path)
        references[str(path.relative_to(root))] = sha(path)
    frozen = root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds'
    assert r['comparisons']['frozen_l2']['receipt_sha256'] == sha(frozen)
    references[str(frozen.relative_to(root))] = sha(frozen)
    with (out/(name+'-gpu.csv')).open() as stream:
        peaks[name] = max(int(row[2]) for row in csv.reader(stream) if len(row)==7 and int(row[1])==1)
    reports[name], processes[name] = r, p
medians = {}
for mode in ['off','on']:
    group = [reports[mode+str(i)] for i in range(1,4)]
    medians[mode] = {k:statistics.median(r['stages'][k] for r in group) for k in
        ['elapsed_sec','optimizer_sec','stage_a_sec','stage_b_sec','native_setup_sec','prepared_build_sec']}
    medians[mode]['optimizer_by_level_sec'] = {str(level):statistics.median(
        next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==level)
        for r in group) for level in range(1,9)}
outer = medians['on']['elapsed_sec']/medians['off']['elapsed_sec']
l6 = medians['on']['optimizer_by_level_sec']['6']/medians['off']['optimizer_by_level_sec']['6']
wins = sum(reports['on'+str(i)]['elapsed_sec'] < reports['off'+str(i)]['elapsed_sec'] for i in range(1,4))
if outer<=protocol['outer_ratio_limit'] and l6<=protocol['l6_optimizer_ratio_limit'] and wins==3:
    status = 'GO_L6_BOUNDED_SLOTS_RUNTIME_DEVELOPMENT'
elif outer>=1 or l6>=1:
    status = 'STOP_L6_RUNTIME_WALL_TIME'
else:
    status = 'CONDITIONAL_L6_RUNTIME_PERFORMANCE'
closure = dict(schema='optimizer-l6-slot-runtime-closure-v1',status=status,hard_gates='PASS',
    evidence_class='GPU1 isolated audited incremental source-closure development; not clean producer or promotion',
    native_sha256=build['native_sha256'],producer_head=build['producer_head'],
    build_lineage=build['source_build'],full_canonical_runs=6,logical_ci_per_run=240498,
    ci_and_pmax_bits_exact=True,seconds_medians=medians,on_over_off_outer_ratio=outer,
    on_over_off_l6_optimizer_ratio=l6,paired_on_wins=wins,
    elapsed_sec_by_run={name:r['elapsed_sec'] for name,r in reports.items()},
    l6_optimizer_sec_by_run={name:next(x['host_sec'] for x in r['stages']['optimizer_by_level'] if x['level']==6)
                            for name,r in reports.items()},
    predeclared_outer_ratio_limit=protocol['outer_ratio_limit'],
    predeclared_l6_ratio_limit=protocol['l6_optimizer_ratio_limit'],
    gpu1_sampled_total_peak_mib=peaks,
    authority_max_ci_error=max(r['oracle_errors'][0] for r in reports.values()),
    authority_max_pmax_error=max(r['oracle_errors'][1] for r in reports.values()),
    checks=checks,processes=processes,default_enabled=False,
    qualified_scope='canonical S2 L6 bounded refill; properties cache, L4/L5 slots and L4 workers32 in both modes',
    minimum_l6_pool_free_bytes=min(r['stages']['l6_pool']['minimum_free_bytes']
                                 for name,r in reports.items() if name.startswith('on')),
    maximum_l6_pool_increment_bytes=max(r['stages']['l6_pool']['maximum_pool_increment_bytes']
                                       for name,r in reports.items() if name.startswith('on')),
    new_real_data_qualified=False,new_final_pdag_qualified=False,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
assert not (out/'closure.json').exists(), 'preserve previous closure'
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
dependencies = [root/p for p in [
    'fastkpc/tests/test_full_cuda_ci_l6_optimizer_slots.R',
    'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    'fastkpc/tools/with_phase10_exact_development_env.sh',
    'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    'fastkpc/R/fast_kpc.R','fastkpc/R/full_cuda_ci_phase35_vertical.R']]
for p in [root/build['parent_build'],
          root/'fastkpc/artifacts/full_cuda_ci/late_optimizer_20261025/manifest.json',
          root/'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025_isolated_completion/manifest.json',
          root/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']:
    references[str(p.relative_to(root))] = sha(p)
assert sha(root/build['parent_build']) == build['parent_build_sha256']
assert sha(out/'coexistence-replacement/closure.json') == build['coexistence_closure_sha256']
manifest = dict(schema='optimizer-l6-slot-runtime-manifest-v1',native_sha256=build['native_sha256'],
    source_sha256=build['source_sha256'],tools=protocol['frozen_tools'],
    dependencies={str(p.relative_to(root)):sha(p) for p in dependencies},references=references,
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['checks','processes']},indent=2))
print('manifest SHA256',sha(out/'manifest.json'))

