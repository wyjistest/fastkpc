"""Compare instrumented interval unions and original waves; no occupancy claims."""
from pathlib import Path
import sys
import numpy as np
from build import read, sha, write

out = Path(sys.argv[1]).resolve()
assert read(out / 'campaign.json')['status'] == 'PASS_CURRENT_NATIVE_WORK_PAIR_ACTIVITY'
fields = ['kind', 'device', 'context', 'stream', 'correlation', 'name', 'detail', 'thread',
          'gx', 'gy', 'gz', 'bx', 'by', 'bz', 'regs', 'ss', 'ds', 'lmem', 'channel']
dtype = np.dtype([(k, '<u8') for k in ['start', 'end', 'bytes']] + [(k, '<u4') for k in fields])
assert dtype.itemsize == 100


def merge(rows, lo, hi):
    result = []
    for row in rows[np.argsort(rows['start'], kind='stable')]:
        a, b = max(lo, int(row['start'])), min(hi, int(row['end']))
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1][1] = max(result[-1][1], b)
        else:
            result.append([a, b])
    return result


def length(intervals):
    return sum(b-a for a, b in intervals) / 1e9


def intersection(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total / 1e9


def quant(values):
    return {name: float(np.quantile(values, q)) for name, q in
            [('min', 0), ('p50', .5), ('p95', .95), ('max', 1)]} if len(values) else {}


def family(name):
    for word in ['build_distance_rows', 'reduce_rows_batch', 'bidiagonal', 'recover',
                 'roots', 'check_structured_boundary', 'center_structured_component', 'prepare']:
        if word in name:
            return word
    return name


results = dict(schema='current-work-pair-activity-analysis-v1', modes={},
    performance_claim=False, hardware_occupancy_measured=False, warp_stalls_measured=False,
    caveats=['Instrumented OFF/ON once each, not repeated performance evidence.',
             'Kernel intervals do not identify block residency or scheduling causality.',
             'Concurrent family/lane sums are not additive savings.',
             'No-recorded-activity gaps are not guaranteed idle or removable.',
             'Recover-only and post-bidiagonal intervals are descriptive, not a guaranteed savings pool.',
             'Replay excludes optimizer workspaces and CI consumers.'])
input_hashes = {}
for mode, expected_shared in [('off', 19656), ('on', 14040)]:
    dest = out / mode
    summary = read(dest / 'trace-summary.json')
    assert summary['dropped'] == summary['invalid_timestamps'] == 0
    assert (dest / 'events.bin').stat().st_size == summary['events'] * dtype.itemsize
    events = np.fromfile(dest / 'events.bin', dtype=dtype)
    assert (events['start'] > 0).all() and (events['end'] >= events['start']).all()
    names = {int(line.split('\t', 1)[0]): line.rstrip('\n').split('\t', 1)[1]
             for line in (dest / 'names.tsv').read_text().splitlines()}
    markers = [(int(t), int(level), label) for t, level, label in
               (line.split('\t') for line in (dest / 'markers.tsv').read_text().splitlines())]
    hardware = read(dest / 'hardware.json')
    levels = []
    for level, wave_count, target_count in [(2, 78, 49919), (3, 86, 54878)]:
        starts = [t for t, l, label in markers if l == level and label == 'begin']
        ends = [t for t, l, label in markers if l == level and label == 'end']
        assert len(starts) == len(ends) == 1
        lo, hi = starts[0], ends[0]
        rows = events[(events['end'] > lo) & (events['start'] < hi)]
        kernels = rows[rows['kind'] == 1]
        device = rows[rows['kind'] <= 3]
        api_rows = rows[(rows['kind'] == 4) | (rows['kind'] == 5)]
        api = {int(row['correlation']): row for row in api_rows}
        assert len(api) == len(api_rows), 'ambiguous API correlation IDs'
        assert all(int(row['correlation']) in api for row in kernels), 'unmatched kernel launch'
        tags = np.array([family(names[int(key)]) for key in kernels['name']])
        groups = {tag: kernels[tags == tag] for tag in sorted(set(tags))}
        assert len(groups['bidiagonal']) == len(groups['recover']) == (target_count+63)//64
        assert all(groups['bidiagonal']['ds'] == expected_shared)
        assert all(groups['recover']['ds'] == 101088)
        union = length(merge(device, lo, hi))
        d = dict(level=level, boundary_sec=(hi-lo)/1e9, device_union_sec=union,
                 no_recorded_device_activity_sec=(hi-lo)/1e9-union,
                 kernels={}, wave_count=wave_count, kernel_api_correlations=len(kernels))
        for tag, selected in groups.items():
            others = device[~((device['kind'] == 1) & np.isin(device['name'], selected['name']))]
            configs = sorted(set(tuple(int(row[key]) for key in
                ['gx', 'gy', 'gz', 'bx', 'by', 'bz', 'regs', 'ss', 'ds', 'lmem']) for row in selected))
            d['kernels'][tag] = dict(count=len(selected),
                union_sec=length(merge(selected, lo, hi)),
                exclusive_recorded_activity_sec=union-length(merge(others, lo, hi)),
                duration_ms=quant((selected['end']-selected['start'])/1e6),
                config_fields=['gx', 'gy', 'gz', 'bx', 'by', 'bz', 'regs', 'ss', 'ds', 'lmem'],
                configs=configs)
        d['bidiagonal_recover_overlap_sec'] = intersection(
            merge(groups['bidiagonal'], lo, hi), merge(groups['recover'], lo, hi))
        wave_markers = sorted((int(label[9:]), t) for t, l, label in markers
                              if l == level and label.startswith('wave_end_'))
        assert [i for i, _ in wave_markers] == list(range(wave_count))
        waves = []
        left = lo
        for index, right in wave_markers:
            wk = kernels[(kernels['start'] >= left) & (kernels['end'] <= right)]
            wt = np.array([family(names[int(key)]) for key in wk['name']])
            bidi = wk[wt == 'bidiagonal']
            recover = wk[wt == 'recover']
            roots = wk[wt == 'roots']
            expected = min(10, (target_count-index*640+63)//64)
            assert len(bidi) == len(recover) == len(roots) == expected, (mode, level, index)
            first = int(bidi['start'].min())
            last = int(bidi['end'].max())
            offsets = sorted((int(row['start'])-first)/1e6 for row in bidi)
            gaps, launched_before_roots_end = [], 0
            for rec in recover:
                pred = roots[roots['stream'] == rec['stream']]
                assert len(pred) == 1
                pred_end = int(pred[0]['end'])
                assert int(rec['start']) >= pred_end
                gaps.append((int(rec['start'])-pred_end)/1e6)
                launched_before_roots_end += int(api[int(rec['correlation'])]['end']) <= pred_end
            waves.append(dict(index=index, lane_count=len(bidi), boundary_sec=(right-left)/1e9,
                bidiagonal_start_offsets_ms=offsets, starts_within_50ms=sum(x <= 50 for x in offsets),
                bidiagonal_span_sec=(last-first)/1e9,
                roots_end_to_recover_start_ms=gaps,
                recover_launch_returned_before_roots_end=int(launched_before_roots_end),
                recover_union_after_last_bidiagonal_sec=length(merge(recover, last, right)),
                trailing_boundary_after_last_bidiagonal_sec=(right-last)/1e9))
            left = right
        normal = waves[1:-1]
        d.update(normal_wave_count=len(normal),
            normal_early_start_counts={str(v): sum(w['starts_within_50ms'] == v for w in normal)
                                       for v in sorted(set(w['starts_within_50ms'] for w in normal))},
            normal_start_spread_ms=quant([max(w['bidiagonal_start_offsets_ms']) for w in normal]),
            normal_wave_boundary_ms=quant([w['boundary_sec']*1000 for w in normal]),
            recover_predecessor_gap_ms=quant([v for w in waves for v in w['roots_end_to_recover_start_ms']]),
            recover_launch_returned_before_roots_end=sum(w['recover_launch_returned_before_roots_end'] for w in waves),
            recover_union_after_last_bidiagonal_sec=sum(w['recover_union_after_last_bidiagonal_sec'] for w in waves),
            trailing_boundaries_after_last_bidiagonal_sec=sum(w['trailing_boundary_after_last_bidiagonal_sec'] for w in waves),
            post_final_wave_sec=(hi-wave_markers[-1][1])/1e9,
            waves=waves)
        levels.append(d)
        print(mode, 'L'+str(level), 'boundary', round(d['boundary_sec'], 6),
              'bidiagonal union', round(d['kernels']['bidiagonal']['union_sec'], 6),
              'recover exclusive', round(d['kernels']['recover']['exclusive_recorded_activity_sec'], 6),
              'recover after bidiagonal', round(d['recover_union_after_last_bidiagonal_sec'], 6),
              'normal early counts', d['normal_early_start_counts'], flush=True)
    results['modes'][mode] = dict(trace_summary=summary, hardware=hardware, levels=levels)
    for name in ['events.bin', 'names.tsv', 'markers.tsv', 'trace-summary.json', 'hardware.json', 'mode.json']:
        input_hashes[str((dest/name).relative_to(out))] = sha(dest/name)
results['input_sha256'] = input_hashes
results['analysis_tool_sha256'] = sha(Path(__file__))
results['campaign_sha256'] = sha(out/'campaign.json')
write(out/'analysis.json', results)
