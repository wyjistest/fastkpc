"""Close only the predeclared captured-L4 experiment, with all gate receipts."""
import csv
import json
from pathlib import Path
import statistics
import sys
from common import CAPTURE, ROOT, TOOL, check_identity, load, sha, write_new

out = Path(sys.argv[1]).resolve()
build = check_identity(out)
protocol = load(out / 'protocol.json')
assert protocol['build_sha256'] == sha(out / 'build.json')
assert protocol['declaration_sha256'] == sha(TOOL / 'README.md')
device = load(out / 'device-code.json')
assert device['status'] == 'PASS' and device['original'] == device['prototype'] and len(device['original']) == 14
modes = ['checks', 'memcheck', 'capacity', 'campaign']
reports = {n: load(out / (n + '-summary.json')) for n in modes}
for n, r in reports.items():
    assert r['status'] == 'PASS' and r['resources_closed'] and r['outputs_bitwise_exact'] and r['forbidden_calls'] == 0, n
for n, runs in [('checks', 3), ('memcheck', 1)]:
    r = reports[n]
    assert r['setups_per_run'] == 15 and r['runs'] == runs and r['compared_target_occurrences'] == 136*runs
    assert r['negative_checks'] == 10 and r['contract_compared_targets'] == 22
    assert all(r[k] for k in ['cancellation_checked', 'partial_upload_checked', 'partial_detach_checked'])
for n, runs in [('capacity', 1), ('campaign', 14)]:
    assert reports[n]['setups_per_run'] == 2154 and reports[n]['runs'] == runs
    assert reports[n]['compared_target_occurrences'] == 17767*runs
mem = (out / 'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
processes = {n: load(out / (n + '-process.json')) for n in modes}
assert all(p['exit_code'] == 0 and p['physical_gpu'] == 1 and not p['unexpected_gpu_pids'] for p in processes.values())
read_rows = lambda n: list(csv.DictReader((out / n).open()))
capacity_rows = read_rows('capacity-runs.csv')
assert len(capacity_rows) == 1
capacity = capacity_rows[0]
assert int(capacity['physical_increment_bytes']) <= protocol['max_physical_increment_bytes']
assert int(capacity['free_after_create']) >= protocol['minimum_free_bytes']
assert tuple(int(capacity[k]) for k in ['allocations', 'streams', 'events', 'blas_handles']) == (33, 32, 96, 32)
assert int(capacity['application_bytes']) == 590676136
campaign = read_rows('campaign-runs.csv')
timed = [r for r in campaign if r['kind'] == 'timing']
assert len(campaign) == 14 and len(timed) == 12
assert [(r['variant'], int(r['cycle'])) for r in campaign] == [('W', -1), ('S', -1)] + [
    (v, cycle) for cycle in range(6) for v in ('WS' if cycle % 2 == 0 else 'SW')]
for r in campaign:
    assert int(r['targets']) == 17767 and r['iterations'] == capacity['iterations']
    assert int(r['rebindings']) == (2154 if r['variant'] == 'S' else 0)
    assert int(r['stale_checks']) == (2154 if r['variant'] == 'S' else 0)
    assert int(r['streams']) == (32 if r['variant'] == 'S' else 64)
    assert int(r['free_after_create']) >= protocol['minimum_free_bytes']
    if r['variant'] == 'S':
        assert int(r['physical_increment_bytes']) <= protocol['max_physical_increment_bytes']
details = read_rows('campaign-setups.csv')
for cycle in range(-1, 6):
    for variant in 'WS':
        g = [r for r in details if int(r['cycle']) == cycle and r['variant'] == variant]
        assert len(g) == 2154 and [int(r['request']) for r in g] == list(range(2154))
        if variant == 'S':
            generations = {}
            for r in g:
                k, generation = int(r['slot']), int(r['generation'])
                assert 0 <= k < 32 and generation > generations.get(k, 0)
                generations[k] = generation
            assert len(generations) == 32
summary = {}
for v in 'WS':
    g = [r for r in timed if r['variant'] == v]
    summary[v] = {k: dict(median=statistics.median(float(r[k])/1000 for r in g),
                         minimum=min(float(r[k])/1000 for r in g), maximum=max(float(r[k])/1000 for r in g))
                  for k in ['work_ms', 'prepare_ms', 'destroy_ms', 'packet_pack_ms', 'lifecycle_ms', 'verify_ms',
                            'bind_sum_ms', 'optimize_sum_ms', 'detach_sum_ms']}
pairs = []
for cycle in range(6):
    pair = {r['variant']: r for r in timed if int(r['cycle']) == cycle}
    pairs.append(dict(cycle=cycle,
        work_ratio=float(pair['S']['work_ms'])/float(pair['W']['work_ms']),
        lifecycle_ratio=float(pair['S']['lifecycle_ms'])/float(pair['W']['lifecycle_ms']),
        lifecycle_saved_sec=(float(pair['W']['lifecycle_ms'])-float(pair['S']['lifecycle_ms']))/1000))
work_ratio = summary['S']['work_ms']['median']/summary['W']['work_ms']['median']
life_ratio = summary['S']['lifecycle_ms']['median']/summary['W']['lifecycle_ms']['median']
wins = sum(p['lifecycle_saved_sec'] > 0 for p in pairs)
go = life_ratio <= protocol['maximum_go_lifecycle_ratio'] and wins == protocol['required_paired_lifecycle_wins']
status = ('GO_L4_BOUNDED_REFILL_ISOLATED_PROTOTYPE' if go else
          'CONDITIONAL_L4_BOUNDED_REFILL' if life_ratio < 0.98 else 'STOP_L4_BOUNDED_REFILL_PERFORMANCE')
telemetry = {}
for n in modes:
    gpu = list(csv.reader((out / (n + '-gpu.csv')).open()))
    selected = [r for r in gpu if len(r) == 7 and int(r[1]) == 1]
    other = [r for r in gpu if len(r) == 7 and int(r[1]) == 0]
    telemetry[n] = dict(gpu1_peak_mib=max(int(r[2]) for r in selected),
                        gpu0_peak_utilization=max(int(r[3]) for r in other),
                        gpu1_max_temperature_c=max(int(r[6]) for r in selected))
closure = dict(schema='optimizer-l4-slots-closure-v1', status=status,
    evidence_class='GPU1 captured-layer source-closure development; not canonical runtime or promotion',
    runtime_commit=build['runtime_commit'], native_sha256=build['native_sha256'],
    build_sha256=sha(out / 'build.json'), protocol_sha256=sha(out / 'protocol.json'),
    device_functions_exact=14, setups=2154, targets=17767, slots=32, target_capacity=18,
    actual_target_counts=[2,3,4,5,6,7,8,10,11,12,14,16,17,18],
    hard_gates='PASS', all_outputs_bitwise_exact=True, resources_closed=True, forbidden_calls=0,
    completed_full_layer_runs=15,
    compared_target_occurrences=sum(r['compared_target_occurrences']+r['contract_compared_targets'] for r in reports.values()),
    reduced_memcheck_errors=0, reduced_memcheck_leaked_bytes=0,
    failure_and_token_checks='PASS', capacity=capacity, full_output_arena_bytes=105038504,
    seconds=summary, S_over_W_work_ratio=work_ratio, S_over_W_lifecycle_ratio=life_ratio,
    paired_S_lifecycle_wins=wins, pairs=pairs, processes=processes, telemetry=telemetry,
    baseline_instrumentation='real production original-window scheduler with three full outputs saved D2D inside each batch; both arms defer shadow verification until layer end',
    production_sources_changed=False, joint_runtime_capacity_qualified=False,
    runtime_rolling='NOT_YET_GO', new_canonical_seconds=None, accepted_canonical_median_sec=461.663,
    new_pvalue_graph_pdag_realdata_qualification=False, phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False)
write_new(out / 'closure.json', closure)
refs = [CAPTURE / 'manifest.json', ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_tail_20261024/manifest.json']
write_new(out / 'manifest.json', dict(schema='optimizer-l4-slots-manifest-v1',
    native_sha256=build['native_sha256'], source_sha256=build['source_sha256'],
    tools={str(p.relative_to(ROOT)): sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'},
    references={str(p.relative_to(ROOT)): sha(p) for p in refs}))
print(json.dumps(dict(status=status, seconds=summary, work_ratio=work_ratio, lifecycle_ratio=life_ratio,
                     paired_wins=wins, manifest_sha256=sha(out / 'manifest.json')), indent=2))
