"""Measured window tails and explicitly counterfactual fixed-service models."""
import bisect
import collections
import csv
import heapq
import json
from pathlib import Path
import sqlite3
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, TOOL, check_identity, load, sha, write_new


def schedule(durations, workers=32):
    free = [0.0] * workers
    for duration in durations:
        assert duration >= 0
        heapq.heapreplace(free, free[0] + duration)
    return max(free)


def interval_stats(intervals, lower, upper):
    events = collections.Counter({lower: 0, upper: 0})
    for start, end in intervals:
        start, end = max(lower, start), min(upper, end)
        if end > start:
            events[start] += 1
            events[end] -= 1
    active = peak = 0
    previous = lower
    union = area = low8 = under32 = 0
    for time in sorted(events):
        duration = time - previous
        if active:
            union += duration
            if active <= 8:
                low8 += duration
        if active < 32:
            under32 += duration
        area += active * duration
        active += events[time]
        assert active >= 0
        peak = max(peak, active)
        previous = time
    assert active == 0
    span = upper - lower
    return dict(union_sec=union / 1e9, no_active_sec=(span - union) / 1e9,
                peak=peak, time_with_1_to_8_active_sec=low8 / 1e9,
                time_below_32_active_sec=under32 / 1e9,
                mean_active=area / span if span else 0)


def distribution(values):
    values = sorted(values)
    assert values
    return dict(min=min(values), median=statistics.median(values),
                p95=values[min(len(values) - 1, int(len(values) * .95))], max=max(values))


def csv_write(path, rows):
    assert rows
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out, run, emit=True):
    prefix = 'profile-' + str(run)
    summary = load(out / (prefix + '-summary.json'))
    process = load(out / (prefix + '-process.json'))
    assert summary['status'] == 'PASS' and process['exit_code'] == 0
    assert not process['unexpected_gpu_pids']
    assert all(summary[k] for k in ['resources_closed', 'output_leases_closed', 'all_outputs_bitwise_exact'])
    assert summary['forbidden_calls'] == 0 and summary['compared_target_occurrences'] == 35534
    inputs = list(csv.DictReader((out / (prefix + '-inputs.csv')).open()))
    assert len(inputs) == 2154 and sum(int(r['targets']) for r in inputs) == 17767
    measured = list(csv.DictReader((out / (prefix + '-windows.csv')).open()))
    profiled = {int(r['window']): r for r in measured if r['pass'] == 'profile'}
    assert set(profiled) == set(range(1, 35))
    db = sqlite3.connect((out / (prefix + '.sqlite')).as_uri() + '?mode=ro', uri=True)
    names = dict(db.execute('select id,value from StringIds'))
    ranges = [(s, e, t or names.get(i, ''), tid) for s, e, t, i, tid in
              db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null')]
    layers = [r for r in ranges if r[2] == 'layer:L4:profile']
    assert len(layers) == 1
    layer_start, layer_end, _, _ = layers[0]
    window_ranges = {int(t.split(':W')[1]): (s, e) for s, e, t, _ in ranges if t.startswith('window:W')}
    assert set(window_ranges) == set(range(1, 35))
    apis = collections.defaultdict(list)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER']:
        if table in tables:
            for rid, s, e, tid, co, name in db.execute('select rowid,start,end,globalTid,correlationId,nameId from ' + table):
                apis[tid].append((s, e, co, names[name], table, rid))
    for rows in apis.values():
        rows.sort()
    starts = {tid: [r[0] for r in rows] for tid, rows in apis.items()}
    kernels = {}
    by_corr = collections.defaultdict(list)
    for rid, s, e, co, st, name, gx, gy, gz in db.execute(
            'select rowid,start,end,correlationId,streamId,shortName,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL'):
        kernels[rid] = dict(start=s, end=e, correlation=co, stream=st, name=names[name], grid_blocks=gx*gy*gz)
        by_corr[co].append(rid)
    copies = {}
    copies_corr = collections.defaultdict(list)
    for rid, s, e, co, st, kind, size in db.execute(
            'select rowid,start,end,correlationId,streamId,copyKind,bytes from CUPTI_ACTIVITY_KIND_MEMCPY'):
        copies[rid] = dict(start=s, end=e, correlation=co, stream=st, kind=kind, bytes=size)
        copies_corr[co].append(rid)
    db.close()
    setup_rows = []
    assigned_kernels, assigned_copies = set(), set()
    setup_kernels, setup_copies = {}, {}
    api_totals = collections.defaultdict(lambda: dict(calls=0, worker_sec=0.0, intervals=[]))
    for start, end, label, tid in ranges:
        if not label.startswith('setup:') or not (layer_start <= start < end <= layer_end):
            continue
        index = int(label.split(':')[1])
        assert index not in setup_kernels
        lo = bisect.bisect_left(starts[tid], start)
        hi = bisect.bisect_right(starts[tid], end)
        found_apis = [r for r in apis[tid][lo:hi] if r[1] <= end]
        correlations = {r[2] for r in found_apis}
        ids = {k for co in correlations for k in by_corr[co]}
        mids = {m for co in correlations for m in copies_corr[co]}
        assert ids and not assigned_kernels.intersection(ids)
        assert not assigned_copies.intersection(mids)
        assigned_kernels.update(ids)
        assigned_copies.update(mids)
        ks = sorted((kernels[k] for k in ids), key=lambda r: r['start'])
        ms = sorted((copies[m] for m in mids), key=lambda r: r['start'])
        first, last = min(k['start'] for k in ks), max(k['end'] for k in ks)
        window = int(inputs[index]['window'])
        ws, we = window_ranges[window]
        assert ws <= start < end <= we and ws <= first < last <= we
        api_return = {co: e for s, e, co, name, table, rid in found_apis if table == 'CUPTI_ACTIVITY_KIND_RUNTIME'}
        h2d = [m for m in ms if m['kind'] == 1]
        d2h = [m for m in ms if m['kind'] == 2]
        optimizer = [k for k in ks if 'optimize_multi_penalty_targets' in k['name']]
        assert len(optimizer) == 1, (index, [k['name'] for k in ks])
        setup_rows.append(dict(setup=index, window=window, targets=int(inputs[index]['targets']),
            iterations=int(inputs[index]['iterations']), host_start_sec=(start - layer_start)/1e9,
            host_end_sec=(end - layer_start)/1e9, host_service_sec=(end - start)/1e9,
            first_kernel_sec=(first - layer_start)/1e9, last_kernel_sec=(last - layer_start)/1e9,
            device_service_sec=(last - first)/1e9, kernel_count=len(ks),
            kernel_sum_sec=sum(k['end'] - k['start'] for k in ks)/1e9,
            optimizer_kernel_sec=sum(k['end'] - k['start'] for k in optimizer)/1e9,
            initial_host_to_kernel_sec=(first - start)/1e9,
            last_kernel_to_host_return_sec=(end - last)/1e9,
            kernel_enqueue_delay_sec=max(0, optimizer[0]['start'] - api_return.get(optimizer[0]['correlation'], optimizer[0]['start']))/1e9,
            h2d_bytes=sum(m['bytes'] for m in h2d), d2h_bytes=sum(m['bytes'] for m in d2h),
            h2d_device_sec=sum(m['end'] - m['start'] for m in h2d)/1e9,
            d2h_device_sec=sum(m['end'] - m['start'] for m in d2h)/1e9,
            event_wait_api_sec=sum(e-s for s,e,co,n,t,rid in found_apis if n.startswith('cudaEventSynchronize'))/1e9,
            stream_wait_api_sec=sum(e-s for s,e,co,n,t,rid in found_apis if n.startswith('cudaStreamSynchronize'))/1e9))
        setup_kernels[index], setup_copies[index] = ks, ms
        for s, e, co, name, table, rid in found_apis:
            if table != 'CUPTI_ACTIVITY_KIND_RUNTIME':
                continue
            entry = api_totals[name]
            entry['calls'] += 1
            entry['worker_sec'] += (e-s)/1e9
            entry['intervals'].append((s, e))
    setup_rows.sort(key=lambda r: r['setup'])
    assert [r['setup'] for r in setup_rows] == list(range(2154))
    native_kernel_ids = {rid for rid, k in kernels.items() if any(s <= k['start'] and k['end'] <= e for s,e in window_ranges.values())}
    native_copy_ids = {rid for rid, m in copies.items() if any(s <= m['start'] and m['end'] <= e for s,e in window_ranges.values())}
    assert assigned_kernels == native_kernel_ids, 'uncorrelated native kernels'
    assert assigned_copies == native_copy_ids, 'uncorrelated native copies'
    window_rows = []
    for window in range(1, 35):
        rows = [r for r in setup_rows if r['window'] == window]
        ids = [r['setup'] for r in rows]
        assert len(rows) == (42 if window == 34 else 64)
        ks = [k for i in ids for k in setup_kernels[i]]
        ws, we = window_ranges[window]
        first, last = min(k['start'] for k in ks), max(k['end'] for k in ks)
        last_setup_start = max(min(k['start'] for k in setup_kernels[i]) for i in ids)
        stats = interval_stats([(k['start'], k['end']) for k in ks], ws, we)
        tail = interval_stats([(k['start'], k['end']) for k in ks], last_setup_start, last)
        window_rows.append(dict(window=window, setups=len(rows), targets=sum(r['targets'] for r in rows),
            native_sec=(we-ws)/1e9, kernel_span_sec=(last-first)/1e9,
            first_kernel_delay_sec=(first-ws)/1e9, after_last_kernel_sec=(we-last)/1e9,
            last_setup_start_to_last_kernel_sec=(last-last_setup_start)/1e9,
            drain_tail_mean_active_kernels=tail['mean_active'],
            drain_tail_1_to_8_active_sec=tail['time_with_1_to_8_active_sec'],
            kernel_union_sec=stats['union_sec'], no_kernel_sec=stats['no_active_sec'],
            mean_active_kernels=stats['mean_active'], peak_active_kernels=stats['peak'],
            time_with_1_to_8_kernels_sec=stats['time_with_1_to_8_active_sec'],
            modeled_host_window_sec=schedule([r['host_service_sec'] for r in rows]),
            modeled_device_window_sec=schedule([r['device_service_sec'] for r in rows])))
    native_sec = sum(r['native_sec'] for r in window_rows)
    assert abs(native_sec-summary['profiled_native_sec']) < .1
    host = [r['host_service_sec'] for r in setup_rows]
    device = [r['device_service_sec'] for r in setup_rows]
    models = {}
    for label, values, measured_sec in [('host', host, native_sec), ('device', device, sum(r['kernel_span_sec'] for r in window_rows))]:
        windows = sum(r['modeled_' + label + '_window_sec'] for r in window_rows)
        refill = schedule(values)
        models[label] = dict(window_model_sec=windows, measured_boundary_sec=measured_sec,
            relative_fit_error=abs(windows-measured_sec)/measured_sec,
            refill_model_sec=refill, modeled_barrier_reduction_sec=windows-refill,
            modeled_reduction_over_measured_native=(windows-refill)/native_sec,
            fixed_service_lower_bound_sec=max(sum(values)/32, max(values)),
            budget_ms_per_setup_if_new_cost_is_fully_serial=1000*(windows-refill)/len(values),
            refill_plus_fully_serial_1ms_per_setup_sec=refill+len(values)*.001,
            refill_plus_fully_serial_5ms_per_setup_sec=refill+len(values)*.005)
    native_api = []
    for name, entry in sorted(api_totals.items()):
        native_api.append(dict(api=name, calls=entry['calls'], worker_sum_sec=entry['worker_sec'],
            union_across_workers_sec=interval_stats(entry['intervals'], layer_start, layer_end)['union_sec']))
    geometry_ranges = [(s,e) for s,e,t,tid in ranges if t.startswith('prepare:W')]
    geometry_h2d = [m for m in copies.values() if m['kind']==1 and any(s<=m['start'] and m['end']<=e for s,e in geometry_ranges)]
    copy_summary = {}
    for kind, label in [(1, 'H2D'), (2, 'D2H')]:
        ms = [copies[rid] for rid in assigned_copies if copies[rid]['kind'] == kind]
        copy_summary[label] = dict(count=len(ms), bytes=sum(m['bytes'] for m in ms),
            device_sum_sec=sum(m['end']-m['start'] for m in ms)/1e9,
            device_union_sec=interval_stats([(m['start'],m['end']) for m in ms],layer_start,layer_end)['union_sec'])
    components = {key: sum(float(r[key]) for r in profiled.values()) for key in
                  ['prepare_sec','pack_sec','native_sec','verify_sec','release_sec','destroy_sec']}
    report = dict(run=run, setup_count=2154, target_count=17767, window_count=34,
        unprofiled_warmup_native_sec=summary['unprofiled_warmup_native_sec'], measured_profile_native_sec=native_sec,
        capture_range_sec=(layer_end-layer_start)/1e9, segmented_host_boundaries=components,
        no_kernel_in_native_sec=sum(r['no_kernel_sec'] for r in window_rows),
        drain_tail_sum_sec=sum(r['last_setup_start_to_last_kernel_sec'] for r in window_rows),
        drain_tail_1_to_8_kernels_sum_sec=sum(r['drain_tail_1_to_8_active_sec'] for r in window_rows),
        drain_tail_not_automatically_hideable=True, correlated_kernels=len(assigned_kernels),
        correlated_memcopies=len(assigned_copies), all_native_kernels_and_copies_correlated=True,
        copy_summary=copy_summary, prepared_geometry_h2d=dict(count=len(geometry_h2d),
            bytes=sum(m['bytes'] for m in geometry_h2d), device_sum_sec=sum(m['end']-m['start'] for m in geometry_h2d)/1e9),
        host_api_totals_are_worker_sums_not_savings=True, models=models,
        distributions={name: distribution([r[name] for r in setup_rows]) for name in
                       ['host_service_sec','device_service_sec','initial_host_to_kernel_sec',
                        'last_kernel_to_host_return_sec','kernel_enqueue_delay_sec']},
        peak_application_bytes=summary['peak_application_bytes'],
        minimum_free_after_create=summary['minimum_free_after_create'], other_gpu_pids=process['other_gpu_pids'])
    if emit:
        csv_write(out/(prefix+'-setup-timeline.csv'), setup_rows)
        csv_write(out/(prefix+'-window-attribution.csv'), window_rows)
        csv_write(out/(prefix+'-host-api.csv'), native_api)
    return report, setup_rows


def main():
    out = Path(sys.argv[1]).resolve()
    check_identity(out)
    protocol = load(out/'protocol.json')
    reports, rows = [], []
    for run in (1,2):
        report, timeline = analyze(out,run)
        reports.append(report)
        rows.append(timeline)
    fit = all(r['models']['host']['relative_fit_error'] <= protocol['model_max_relative_fit_error'] for r in reports)
    opportunity = [r['models']['host']['modeled_reduction_over_measured_native'] >=
                   protocol['model_min_fraction_of_native_saved'] for r in reports]
    if not fit or opportunity[0] != opportunity[1]:
        status = 'CONDITIONAL_MODEL_NOT_VALIDATED'
    elif all(opportunity):
        status = 'GO_L4_BOUNDED_REFILL_EXPERIMENT_ONLY'
    else:
        status = 'STOP_L4_REFILL_OPPORTUNITY_TOO_SMALL'
    cross = {}
    for label, key in [('host','host_service_sec'),('device','device_service_sec')]:
        ratios = [b[key]/a[key] for a,b in zip(*rows)]
        assert all(a['setup']==b['setup'] and a['targets']==b['targets'] for a,b in zip(*rows))
        cross[label] = dict(per_setup_second_over_first=distribution(ratios),
            setup_ratios_over_1_10=sum(x>1.10 for x in ratios), setup_ratios_below_0_90=sum(x<.90 for x in ratios),
            first_services_vs_second_boundary_relative_error=abs(
                reports[0]['models'][label]['window_model_sec']-reports[1]['models'][label]['measured_boundary_sec'])/
                reports[1]['models'][label]['measured_boundary_sec'],
            second_services_vs_first_boundary_relative_error=abs(
                reports[1]['models'][label]['window_model_sec']-reports[0]['models'][label]['measured_boundary_sec'])/
                reports[0]['models'][label]['measured_boundary_sec'])
    result = dict(schema='l4-tail-attribution-v1',status=status,
        interpretation='fixed observed service-time screening only; no measured refill, slot costs or production-cache capacity qualification',
        host_model_fit_pass=fit, modeled_opportunity_pass=opportunity, runs=reports, cross_trace_service=cross,
        source_native_unchanged=True, full_layer_capacity_status='STOP_FULL_LAYER_CAPACITY',
        runtime_integration='NOT_YET_GO', phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
        analysis_sha256=sha(TOOL/'analyze.py'))
    write_new(out/'analysis.json',result)
    print(json.dumps(dict(status=status,runs=[dict(run=r['run'],native=r['measured_profile_native_sec'],
        tails=r['drain_tail_sum_sec'],no_kernel=r['no_kernel_in_native_sec'],models=r['models']) for r in reports]),indent=2))


if __name__ == '__main__':
    main()
