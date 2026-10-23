"""Close the frozen L2 diagnostic without changing the production runtime."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np

from run import EXPECTED, Inputs

root = Path(__file__).resolve().parents[4]
os.chdir(root)
tool = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
previous = root / 'fastkpc/artifacts/full_cuda_ci/structured_prefill_canonical_20261022'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


prior_build = read(previous / 'build.json')
assert prior_build['status'] == 'PASS'
assert prior_build['native_sha256'] == EXPECTED
assert sha(root / 'fastkpc/build/fastkpc_cuda.so') == EXPECTED
for name, expected in prior_build['source_sha256'].items():
    assert sha(root / name) == expected, ('production source changed', name)
    assert sha(previous / 'build-source' / name) == expected
for receipt, binary, field in [('build.json', 'capture.so', 'capture_sha256'),
                                ('probe-build.json', 'probe.so', 'probe_sha256')]:
    built = read(out / 'build' / receipt)
    assert built['native_sha256'] == EXPECTED
    assert sha(out / 'build' / binary) == built[field]
    assert all(sha(root / name) == value for name, value in built['sources'].items())
probe_build = read(out / 'build/probe-build.json')
assert sha(out / 'build/pair_stage.inc') == probe_build['pair_stage_sha256']
assert sha(root / 'fastkpc/build/full_cuda_ci_structured_eig.o') == probe_build['structured_object_sha256']
initial_capture = read(out / 'build/capture-build-at-run.json')
assert initial_capture['capture_sha256'] == sha(out / 'build/capture.so')

capture = read(out / 'capture/capture.json')
fixed = read(out / 'fixed.json')
timing = read(out / 'fixed-timing.json')
qualified = read(out / 'qualification.json')
assert all(capture['gates'].values())
assert capture['receipt_sha256'] == sha(out / 'capture/receipt.rds')
assert fixed['status'] == timing['status'] == 'PASS'
assert qualified['status'] == 'PASS_FROZEN_L2_QUALIFICATION'
assert all(qualified['repeat_gates'].values())
assert all(all(r['gates'].values()) for r in qualified['runs'].values())
names = ('original', 'dense1', 'structured1', 'structured2', 'dense2')
reports = {name: read(out / (name + '.json')) for name in names}
for name, report in reports.items():
    assert report['status'] == 'PASS' and report['resources_zero']
    assert report['native_sha256'] == EXPECTED
    assert read(out / (name + '-process.json'))['exit_code'] == 0
assert reports['original']['changed_physical_p'] == 0
sanitizer = {}
for name in ('memcheck-fixed', 'memcheck-components'):
    text = (out / (name + '.log')).read_text()
    sanitizer[name] = 'ERROR SUMMARY: 0 errors' in text and '0 bytes leaked' in text
    assert sanitizer[name] and read(out / (name + '-process.json'))['exit_code'] == 0

# Attribute changes by the *original* materialization width, separately from
# counting distinct first-consumed component payloads.
inputs = Inputs(out)
arrays = np.load(out / 'fixed-outputs.npz')
candidate = arrays['residual']
by_width = {}
by_range = {'le16': dict(occurrences=0, changed=0, maximum=0.),
            'ge17': dict(occurrences=0, changed=0, maximum=0.)}
for solve in inputs.solves:
    _, _, _, _, original = inputs.solve(solve)
    ids = [inputs.index[key] for key in solve['keys']]
    difference = np.abs(original - candidate[ids])
    changed = np.any(original.view(np.uint64) != candidate[ids].view(np.uint64), axis=1)
    width = solve['count']
    row = by_width.setdefault(str(width), dict(cohorts=0, occurrences=0, changed=0, maximum=0.))
    row['cohorts'] += 1
    for item in (row, by_range['le16' if width <= 16 else 'ge17']):
        item['occurrences'] += width
        item['changed'] += int(changed.sum())
        item['maximum'] = max(item['maximum'], float(difference.max()))
used_ids = [inputs.index[key] for key in inputs.used]
original = arrays['original_residual']
changed_unique = np.any(original.view(np.uint64) != candidate[used_ids].view(np.uint64), axis=1)
assert sum(r['changed'] for r in by_width.values()) == fixed['comparison']['changed_residual']
assert np.array_equal(arrays['original_rank'], arrays['metadata'][2, used_ids])
attribution = dict(original_width=by_width, width_ranges=by_range,
    first_consumed_unique_components=len(inputs.used),
    changed_first_consumed_unique_components=int(changed_unique.sum()),
    first_consumed_maximum=float(np.max(np.abs(original - candidate[used_ids]))),
    planned=len(inputs.keys), planned_unused=len(inputs.keys) - len(inputs.used),
    q=sorted({g['q'] for g in inputs.geometries.values()}),
    prepared_groups=len(inputs.groups))
save(out / 'residual-attribution.json', attribution)

memory = {}
for name in names:
    with (out / (name + '-gpu.csv')).open() as stream:
        rows = [r for r in csv.reader(stream) if len(r) == 5]
    a = [r for r in rows if r[0].strip() == '0']
    b = [r for r in rows if r[0].strip() == '1']
    assert a and b
    memory[name] = dict(sample_interval_ms=200,
        gpu0_peak_mib=max(float(r[1]) for r in a),
        gpu1_peak_mib=max(float(r[1]) for r in b),
        gpu1_peak_utilization_percent=max(float(r[2]) for r in b),
        gpu0_sm_clock_range_mhz=[min(float(r[3]) for r in a), max(float(r[3]) for r in a)],
        gpu0_peak_temperature_c=max(float(r[4]) for r in a))
    assert memory[name]['gpu1_peak_utilization_percent'] == 0

fields = ('stage_b_sec', 'consumer_replay_sec', 'full_component_pipeline_sec')
medians = {mode: {field: statistics.median(reports[mode + i][field] for i in ('1', '2'))
                  for field in fields} for mode in ('dense', 'structured')}
separated_gain = (
    max(reports['structured' + i]['stage_b_sec'] for i in ('1', '2')) <
    min(reports['dense' + i]['stage_b_sec'] for i in ('1', '2')) and
    medians['structured']['full_component_pipeline_sec'] <
    reports['original']['full_component_pipeline_sec'])
stage_a = {field: statistics.median(run[field] for run in timing['candidates'])
           for field in ('prepare_ms', 'evaluate_ms', 'detach_ms', 'release_ms',
                         'stage_a_api_ms', 'driver_wall_sec')}
with (out / 'timing.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['run'] + list(fields) + ['sampled_peak_mib'])
    writer.writeheader()
    for name, report in reports.items():
        writer.writerow(dict(run=name, **{f: report[f] for f in fields},
                             sampled_peak_mib=memory[name]['gpu0_peak_mib']))

closure = dict(schema='l2-prefill-diagnostic-closure-v1',
    status='PASS_ISOLATED_L2_QUALIFICATION',
    decision=('GO_BOUNDED_OPT_IN_RUNTIME_QUALIFICATION; NOT_YET_RUNTIME_ACCEPTANCE'
              if separated_gain else 'STOP_NO_SEPARATED_COMPONENT_GAIN'),
    evidence_class='frozen-cohort GPU diagnostic; two component repetitions per candidate; not outer performance',
    native_sha256=EXPECTED, production_sources_unchanged=True,
    source_sha256=prior_build['source_sha256'],
    configuration=dict(n=351, p=48, q=30, level=2, numCol=35,
        fixed_max_width=16, cache_bytes=11811160064, eig_lanes=8,
        structured_contexts=10, structured_width=64, cuda_connections=32, physical_gpu=0),
    capture=capture, fixed=fixed, stage_a_timing=timing, stage_a_two_run_median=stage_a,
    residual_attribution=attribution, run_order=list(names), component_runs=reports,
    component_two_run_medians=medians, qualification=qualified,
    separated_isolated_component_gain=separated_gain,
    sanitizer_smoke=sanitizer, sampled_memory=memory,
    resource_check_scope='vertical counters on every replay; fixed and structured sanitizer smoke; not full-campaign memcheck',
    timing_limits=['Stage A loop excludes initial arena allocation and input loading; includes teardown.',
                   'Stage B includes component build; replay includes original Gram and GPU pair/gamma plus Python control.',
                   'Component pipeline timer excludes fixture upload and pool creation/teardown.',
                   'No timing here is a complete runtime or an integrated L2 stage.'],
    provenance_notes=['Fixed numerical qualification preceded the additional timing-only release/driver counters.',
                      'Final tools are archived; fixed-timing.json supplements, and does not overwrite, fixed.json.',
                      'The observer build recipe was finalized after capture; the rebuilt observer must have identical bytes.',
                      'The initial post-check wrongly folded reciprocal tasks; the final map preserves directed capture identity.',
                      'The initial pMax mask check compared dimnames; final checks preserve matrix shape, nonfinite positions/bits and finite numeric tolerance.',
                      'GPU outputs were not rerun or changed for these post-check fixes; rejected logs remain archived.'],
    cpu_numerical_producer=False,
    diagnostic_large_d2h='explicit coefficient/fitted/residual shadows; comparison only',
    public_l2_lifecycle='NOT_INTEGRATED; probe calls existing level-3 lifecycle with L2 fixtures',
    full_runtime_l2_qualification='NOT_RUN', real_data_campaign='NOT_RUN', final_pdag_campaign='NOT_RUN',
    production_default=False, phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False)
save(out / 'closure.json', closure)

# Keep the observation-time build identity alongside the finalized recipe.
for path in sorted(tool.iterdir()):
    if path.is_file():
        dest = out / 'tools' / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
references = [previous / 'manifest.json', previous / 'build.json', previous / 'fastkpc_cuda.so.gz',
              previous / 'off/receipt.rds', previous / 'structured1/receipt.rds',
              root / 'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds',
              root / 'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/pmax.rds',
              root / 'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']
files = {str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=sha(path))
         for path in sorted(out.rglob('*'))
         if path.is_file() and path.name not in ('manifest.json', 'archive.log')}
save(out / 'manifest.json', dict(schema='l2-prefill-diagnostic-manifest-v1', status='PASS',
    native_sha256=EXPECTED, inputs={str(p.relative_to(root)): sha(p) for p in references}, files=files))
print(json.dumps(dict(status=closure['status'], stage_a_median=stage_a,
    component_medians=medians, residual_attribution=attribution,
    qualification=qualified, memory=memory), indent=2), flush=True)
print('PASS manifest', sha(out / 'manifest.json'), flush=True)
