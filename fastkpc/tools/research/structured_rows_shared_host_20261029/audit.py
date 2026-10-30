"""Recompute and seal this supplement without installing or changing isolation gates."""
import shutil
import statistics
import subprocess
from pathlib import Path
from support import ROOT, TOOL, BASE, ROWS, PARENT, sha, read, write, telemetry

protocol = read(BASE/'protocol.json')
closure = read(BASE/'closure.json')
pending = read(ROWS/'pending-outer-protocol.json')
assert protocol['gate'] == pending['gate'] == closure['gate']
assert protocol['sequence'] == pending['sequence']
assert protocol['gpu0_background_allowed'] and protocol['gpu1_exclusive']
assert not protocol['replaces_isolated_acceptance'] and not protocol['installed']
assert pending['both_gpus_free_of_external_compute'] and not pending['accepts_shared_host_timings']
for path, h in protocol['dependencies'].items():
    assert sha(path) == h, path
builds = {'off': read(PARENT/'full/build.json'), 'on': read(ROWS/'candidate/build.json')}
snapshots = {'off': PARENT/'source', 'on': ROWS/'candidate/source'}
for variant, build in builds.items():
    assert sha(build['native_path']) == build['native_sha256'] == closure['native_sha256'][variant]
    for path, h in build['source_sha256'].items():
        assert sha(snapshots[variant]/path) == h, path
        if variant == 'on':
            assert sha(ROOT/path) == h, path
    for name, h in build['objects'].items():
        folder = PARENT/'full/objects' if variant == 'off' else ROWS/'candidate/objects'
        assert sha(folder/name) == h, name
assert [(r['variant'], r['cycle']) for r in closure['runs']] == [tuple(x) for x in protocol['sequence']]
reports = {}
for row in closure['runs']:
    variant, cycle = row['variant'], row['cycle']
    name = variant + str(cycle)
    path = Path(row['report'])
    report = read(path)
    reports[(variant, cycle)] = report
    process = read(BASE/(name+'-process.json'))
    assert process['exit_code'] == 0 and process['gpu1_exclusive']
    assert not process['unexpected'] and process['failure'] is None
    assert process['shared_host'] and not process['isolated_acceptance']
    assert process['affinity'] == list(range(20))
    assert all(report['gates'].values())
    assert report['native_sha256'] == builds[variant]['native_sha256']
    assert report['elapsed_sec'] == row['elapsed_sec']
    assert report['comparisons']['changed_ci_count'] == report['comparisons']['changed_pmax_count'] == 0
    assert sha(path.parent/'receipt.rds') == report['receipt_sha256']
    assert report['evidence_class'].startswith('shared-host structured row-only development timing;')
    transform = read(path.parent/'runner-transformation.json')
    assert transform['changes'] == dict(label=1, start=1, end=1)
    assert sha(ROOT/transform['parent_runner']) == transform['parent_sha256']
    assert sha(path.parent/'effective-runner.rds') == transform['effective_sha256']
    window = read(path.parent/'compute-window.json')
    assert window['timestamps_outside_timer'] and window['elapsed_sec'] == report['elapsed_sec']
    assert abs(window['end_epoch']-window['start_epoch']-report['elapsed_sec']) < .1
    assert telemetry(name, window) == read(row['telemetry'])
medians = {v: statistics.median(reports[(v,i)]['elapsed_sec'] for i in [1,2,3]) for v in ['off','on']}
ratio = medians['on']/medians['off']
assert medians == closure['medians'] and ratio == closure['ratio']
gate = protocol['gate']
ratios, capacity = [], True
for cycle in [1,2,3]:
    off, on = reports[('off',cycle)], reports[('on',cycle)]
    paired_ratio = on['elapsed_sec']/off['elapsed_sec']
    ratios.append(paired_ratio)
    pair = closure['pairs'][cycle-1]
    assert pair['cycle'] == cycle and pair['ratio'] == paired_ratio
    assert len(off['level_component_prefill']) == len(on['level_component_prefill']) == 7
    for a, b, saved in zip(off['level_component_prefill'], on['level_component_prefill'], pair['levels']):
        assert a['level'] == b['level'] == saved['level']
        x, y = a['structured'], b['structured']
        delta = x['pool_device_bytes']-y['pool_device_bytes']
        solver = x['solver_workspace_bytes'] == y['solver_workspace_bytes']
        assert delta == saved['pool_reduction_bytes'] and solver == saved['solver_workspace_exact']
        capacity &= delta == gate['measured_pool_bytes_reduction'] and solver
    for pool in ['l3_pool','l4_pool','l6_pool']:
        a, b = off['stages'][pool]['minimum_free_bytes'], on['stages'][pool]['minimum_free_bytes']
        assert pair['checkpoints'][pool] == dict(off=a,on=b,improvement_bytes=b-a)
        capacity &= b-a >= gate['minimum_free_improvement_bytes']
nonregression = ratio <= gate['median_nonregression_ratio_max'] and all(x <= gate['each_pair_ratio_max'] for x in ratios)
wins = sum(x < 1 for x in ratios)
speed = ratio <= gate['speed_median_ratio_max'] and wins == gate['speed_paired_wins']
assert wins == closure['paired_wins']
assert closure['capacity'] == ('PASS' if capacity else 'FAIL')
assert closure['observed_outer_nonregression'] == ('PASS' if nonregression else 'FAIL')
assert closure['observed_speed_threshold'] == ('PASS' if speed else 'NOT_MET')
assert closure['status'] == ('PASS_SHARED_HOST_DEVELOPMENT' if capacity and nonregression else 'STOP_SHARED_HOST_GATE')
assert closure['isolated_outer_nonregression'] == 'PENDING' and closure['isolated_speed'] == 'NOT_TESTED'
assert not closure['installed'] and closure['accepted_development_median_sec'] == 290.491
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so') == read(PARENT/'parent/identity.json')['native_sha256']
for variant in ['off','on']:
    for stage, value in closure['stage_medians'][variant].items():
        assert value == statistics.median(reports[(variant,i)]['stages'][stage] for i in [1,2,3])
write(BASE/'audit.json', dict(status='PASS_RECOMPUTED_SHARED_HOST', result=closure['status'],
    medians=medians, ratio=ratio, paired_ratios=ratios, capacity=closure['capacity'],
    observed_outer_nonregression=closure['observed_outer_nonregression'],
    observed_speed_threshold=closure['observed_speed_threshold'],
    maximum_authority_error=max(max(r['oracle_errors']) for r in reports.values()),
    original_pending_protocol_sha256=sha(ROWS/'pending-outer-protocol.json'),
    installed=False, isolated_acceptance=False, no_new_real_input_or_final_pdag_campaign=True))
frozen_tools = BASE/'tools'
frozen_tools.mkdir()
for path in TOOL.iterdir():
    if path.is_file():
        shutil.copyfile(path, frozen_tools/path.name)
files = {str(path.relative_to(BASE)): sha(path) for path in BASE.rglob('*') if path.is_file()}
write(BASE/'manifest.json', dict(schema='structured-rows-shared-host-manifest-v1', files=files,
    dependencies=protocol['dependencies'], source_sha256={v:b['source_sha256'] for v,b in builds.items()},
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    evidence_class='uncommitted source closure; shared-host development supplement; not promotion'))
print('PASS shared-host audit', closure['status'], medians, 'ratio', ratio, flush=True)
print('closure', sha(BASE/'closure.json'), 'manifest', sha(BASE/'manifest.json'), flush=True)
