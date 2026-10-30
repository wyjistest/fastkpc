"""Six predeclared shared-host observations; preserves isolated acceptance."""
import statistics
import time
from support import ROOT, TOOL, BASE, ROWS, PARENT, SUPPORT, GPU0, sha, read, write
from support import environment, processes, job, telemetry


def main():
    assert read(PARENT/'closure.json')['status'] == 'PASS_BUILD_FIX_PARENT'
    assert read(ROWS/'qualification/closure.json')['status'] == 'PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES'
    assert read(ROWS/'pending-closure.json')['machine_audit'] == 'PASS_COMPLETED_SCOPE_ONLY'
    pending = read(ROWS/'pending-outer-protocol.json')
    assert pending['both_gpus_free_of_external_compute'] and not pending['accepts_shared_host_timings']
    assert all(x['gpu_uuid'] == GPU0 for x in processes()), 'GPU1 busy; no campaign started'
    builds = {'off': read(PARENT/'full/build.json'), 'on': read(ROWS/'candidate/build.json')}
    snapshots = {'off': PARENT/'source', 'on': ROWS/'candidate/source'}
    assert builds['off']['native_sha256'] == pending['parent_native_sha256']
    assert builds['on']['native_sha256'] == pending['candidate_native_sha256']
    reference = PARENT/'canonical/qualification/receipt.rds'
    runner = TOOL/'canonical_shared.R'
    deps = {str(p): sha(p) for p in [reference, SUPPORT,
        ROWS/'pending-outer-protocol.json', ROWS/'pending-closure.json', ROWS/'pending-manifest.json',
        PARENT/'closure.json', ROWS/'qualification/closure.json',
        ROOT/'fastkpc/build/fastkpc_cuda.so',
        TOOL.parent/'build_consistency_20261029/canonical.R',
        ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
        ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds',
        TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R',
        ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
        ROOT/'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds']}
    for p in list(TOOL.glob('*.py')) + list(TOOL.glob('*.R')) + [TOOL/'README.md']:
        deps[str(p)] = sha(p)
    for p in (ROOT/'fastkpc/R').glob('*.R'):
        deps[str(p)] = sha(p)
    for name in ['logical_ci_trace.rds', 'pmax.rds', 'adjacency.rds']:
        p = ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name
        deps[str(p)] = sha(p)
    BASE.mkdir()
    for variant, build in builds.items():
        folder = BASE/variant
        folder.mkdir()
        current = dict(build)
        current['previous_native_sha256'] = builds['off']['native_sha256']
        current['source_sha256'] = {str(snapshots[variant]/p): h for p, h in build['source_sha256'].items()}
        write(folder/'build.json', current)
        deps[build['native_path']] = build['native_sha256']
        deps[str(folder/'build.json')] = sha(folder/'build.json')

    def frozen():
        for path, h in deps.items():
            assert sha(path) == h, path
        for variant, build in builds.items():
            for path, h in build['source_sha256'].items():
                assert sha(snapshots[variant]/path) == h, path
        for path, h in builds['on']['source_sha256'].items():
            assert sha(ROOT/path) == h, path

    sequence = pending['sequence']
    gate = pending['gate']
    env = environment()
    write(BASE/'protocol.json', dict(schema='structured-rows-shared-host-v1', epoch=time.time(),
        sequence=sequence, gate=gate, dependencies=deps, physical_gpu=1, gpu1_exclusive=True,
        gpu0_background_allowed=True, evidence_class='shared-host development supplement',
        fresh_process=True, fresh_dataset_caches=True, cpu_affinity=list(range(20)), math_threads=1,
        postback=True, work_pair=True, changed_lanes_or_slots=False,
        thresholds_unchanged=True, isolated_protocol_unchanged=True, installed=False,
        replaces_isolated_acceptance=False, promotion=False))
    reports, previous, runs = {}, {}, []
    for variant, cycle in sequence:
        name = variant + str(cycle)
        out = BASE/variant/name
        out.mkdir()
        print('START', name, time.strftime('%H:%M:%S'), flush=True)
        job(name, ['sh', 'fastkpc/tools/with_phase10_exact_development_env.sh', 'Rscript', '--vanilla',
            str(runner), str(out), '1', str(reference), previous.get(variant, '-')], env, frozen)
        report = read(out/'report.json')
        assert all(report['gates'].values())
        assert report['native_sha256'] == builds[variant]['native_sha256']
        assert report['comparisons']['changed_ci_count'] == report['comparisons']['changed_pmax_count'] == 0
        window = read(out/'compute-window.json')
        assert window['elapsed_sec'] == report['elapsed_sec']
        load = telemetry(name, window)
        write(out/'telemetry-summary.json', load)
        reports[(variant, cycle)] = report
        previous[variant] = str(out/'receipt.rds')
        runs.append(dict(variant=variant, cycle=cycle, elapsed_sec=report['elapsed_sec'],
                         report=str(out/'report.json'), telemetry=str(out/'telemetry-summary.json')))
        print('RESULT', name, report['elapsed_sec'], 'seconds; hard gates', len(report['gates']),
              'PASS; GPU0 mean utilization', load['gpu']['0']['fields']['utilization_percent']['mean'], flush=True)
    medians = {v: statistics.median(reports[(v, i)]['elapsed_sec'] for i in [1, 2, 3]) for v in ['off', 'on']}
    pairs, capacity = [], True
    for cycle in [1, 2, 3]:
        off, on = reports[('off', cycle)], reports[('on', cycle)]
        assert len(off['level_component_prefill']) == len(on['level_component_prefill']) == 7
        rows = []
        for a, b in zip(off['level_component_prefill'], on['level_component_prefill']):
            assert a['level'] == b['level']
            x, y = a['structured'], b['structured']
            delta = x['pool_device_bytes'] - y['pool_device_bytes']
            solver = x['solver_workspace_bytes'] == y['solver_workspace_bytes']
            capacity &= delta == gate['measured_pool_bytes_reduction'] and solver
            rows.append(dict(level=a['level'], pool_reduction_bytes=delta, solver_workspace_exact=solver))
        checkpoints = {}
        for pool in ['l3_pool', 'l4_pool', 'l6_pool']:
            a, b = off['stages'][pool]['minimum_free_bytes'], on['stages'][pool]['minimum_free_bytes']
            capacity &= b-a >= gate['minimum_free_improvement_bytes']
            checkpoints[pool] = dict(off=a, on=b, improvement_bytes=b-a)
        pairs.append(dict(cycle=cycle, ratio=on['elapsed_sec']/off['elapsed_sec'],
                          levels=rows, checkpoints=checkpoints))
    ratio = medians['on']/medians['off']
    nonregression = ratio <= gate['median_nonregression_ratio_max'] and all(
        pair['ratio'] <= gate['each_pair_ratio_max'] for pair in pairs)
    wins = sum(pair['ratio'] < 1 for pair in pairs)
    speed = ratio <= gate['speed_median_ratio_max'] and wins == gate['speed_paired_wins']
    write(BASE/'closure.json', dict(
        status='PASS_SHARED_HOST_DEVELOPMENT' if capacity and nonregression else 'STOP_SHARED_HOST_GATE',
        evidence_class='shared-host observations only; original isolated acceptance remains pending',
        numerical_behavior='PASS', capacity='PASS' if capacity else 'FAIL',
        observed_outer_nonregression='PASS' if nonregression else 'FAIL',
        observed_speed_threshold='PASS' if speed else 'NOT_MET',
        isolated_outer_nonregression='PENDING', isolated_speed='NOT_TESTED',
        gate=gate, medians=medians, ratio=ratio, paired_wins=wins, pairs=pairs, runs=runs,
        native_sha256={v: b['native_sha256'] for v, b in builds.items()},
        stage_medians={v: {s: statistics.median(reports[(v,i)]['stages'][s] for i in [1,2,3])
            for s in ['optimizer_sec','stage_b_sec','stage_a_sec','native_setup_sec','consumer_dcov_sec']}
            for v in ['off','on']}, installed=False, accepted_development_median_sec=290.491,
        production_default=False, phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False))
    frozen()
    print('COMPLETE shared-host', medians, 'ratio', ratio, 'capacity', capacity,
          'observed nonregression', nonregression, 'observed speed threshold', speed, flush=True)


if __name__ == '__main__':
    main()
