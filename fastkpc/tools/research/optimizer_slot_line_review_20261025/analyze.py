"""Per-setup multi-penalty optimizer service from one CUPTI trace (ktrace.cpp) of the
production canonical run, with an original-order list-scheduling model per level.

python3 -I analyze.py RUN_DIR    (kernels.bin, runtime.bin, kernel-names.tsv,
                                  profile-result.json written by trace.sh)

Setup identity in window mode: every setup owns a fresh prepared object and stream, so
setup = stream within a window. Optimizer-call kernels are multi-penalty kernels launched
from a non-main thread (the optimizer workers); Stage A/B and setup work run on the main
thread. Streams are ordered by first optimizer kernel and chunked by the receipt's
per-window setup counts (windows are barriered, so they never interleave; the code
asserts it). Slot levels (L4/L5) form one segment over 32 reused streams; there a setup
is one worker launch burst on its slot stream (split where the stream's first optimizer
kernel name recurs).
"""
import collections
import heapq
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
K = np.fromfile(run / 'kernels.bin', dtype=np.dtype([
    ('start', '<u8'), ('end', '<u8'), ('lmem_total', '<u8'), ('stream', '<u4'),
    ('corr', '<u4'), ('name', '<u4'), ('lmem', '<u4'), ('channel', '<u4'),
    ('grid', '<u4'), ('block', '<u4'), ('regs', '<u2')]))
RT = np.fromfile(run / 'runtime.bin', dtype=np.dtype([
    ('start', '<u8'), ('end', '<u8'), ('corr', '<u4'), ('cbid', '<u4'), ('thread', '<u4')]))
names = {}
for line in (run / 'kernel-names.tsv').read_text().splitlines():
    i, name = line.split('\t', 1)
    names[int(i)] = name
result = json.loads((run / 'profile-result.json').read_text())

order = np.argsort(RT['corr'])
rt_corr, rt_thread = RT['corr'][order], RT['thread'][order]
pos = np.clip(np.searchsorted(rt_corr, K['corr']), 0, len(rt_corr) - 1)
k_thread = np.where(rt_corr[pos] == K['corr'], rt_thread[pos], 0)
threads, counts = np.unique(RT['thread'], return_counts=True)
main_thread = int(threads[np.argmax(counts)])

def family(name):
    if 'mgcv_multi_penalty_gcv' in name:
        for tag in ('small64', 'ext192', 'ext384', 'ext559'):
            if 'mgcv_multi_penalty_gcv_' + tag in name:
                return 'multi-' + tag
        return 'multi-base'
    if 'mgcv_single_penalty_gcv' in name:
        return 'single'
    return 'other'

fam_of_name = {i: family(n) for i, n in names.items()}
k_family = np.array([fam_of_name[int(i)] for i in K['name']])
is_opt = np.char.startswith(k_family.astype(str), 'multi') & (k_thread != main_thread)
idx = np.nonzero(is_opt)[0]
idx = idx[np.argsort(K['start'][idx], kind='stable')]

by_stream = collections.defaultdict(list)
for i in idx:
    by_stream[int(K['stream'][i])].append(int(i))
streams = sorted(by_stream, key=lambda s: K['start'][by_stream[s][0]])
# Windows are barriered (the next window's prepared objects are created only after the
# optimizer call returns), so first-kernel order never interleaves windows; a setup
# whose first kernel is held back by a shared hardware channel can, however, start
# after every other setup of its window has finished. Chunk by the receipt's counts.
windows = [b for b in result['prefill_batches'] if b['penalty_class'] == 'multi']
segments, cursor = [], 0
for w in windows:
    n = w['optimizer_setup_count']
    width = min(32, n) if n > 64 else n  # slot levels reuse 32 streams
    segments.append(streams[cursor:cursor + width])
    cursor += width
assert cursor == len(streams), (cursor, len(streams))
for seg_a, seg_b in zip(segments, segments[1:]):
    assert max(int(K['end'][by_stream[s]].max()) for s in seg_a) <= \
        min(int(K['start'][by_stream[s][0]]) for s in seg_b), 'windows overlap'

def bursts(stream):
    """Split one stream's optimizer kernels into setups (calls)."""
    ks = by_stream[stream]
    first_name = K['name'][ks[0]]
    calls, cur = [], []
    for i in ks:
        if cur and K['name'][i] == first_name:
            calls.append(cur)
            cur = []
        cur.append(i)
    calls.append(cur)
    return calls

def record(ks):
    ks = np.asarray(ks)
    start, end = int(K['start'][ks].min()), int(K['end'][ks].max())
    busy = int((K['end'][ks] - K['start'][ks]).sum())
    longest = ks[np.argmax(K['end'][ks] - K['start'][ks])]
    return dict(start=start, end=end, span=(end - start) / 1e9, busy=busy / 1e9,
                kernels=len(ks), channel=int(K['channel'][ks[0]]),
                stream_id=int(K['stream'][ks[0]]),
                longest_kernel=names[int(K['name'][longest])][:80],
                longest_kernel_s=(int(K['end'][longest]) - int(K['start'][longest])) / 1e9,
                grid=int(K['grid'][longest]), block=int(K['block'][longest]),
                regs=int(K['regs'][longest]), lmem=int(K['lmem'][longest]),
                lmem_total=int(K['lmem_total'][longest]))

def list_schedule(services, slots):
    free = [0.0] * max(1, min(slots, len(services)))
    heapq.heapify(free)
    finish = 0.0
    for service in services:
        t = heapq.heappop(free)
        heapq.heappush(free, t + service)
        finish = max(finish, t + service)
    return finish

levels = collections.OrderedDict()
for seg, w in zip(segments, windows):
    slot_mode = len(seg) < w['optimizer_setup_count']
    if slot_mode:  # one original-order queue: start order is fetch order
        recs = sorted((record(call) for s in seg for call in bursts(s)), key=lambda r: r['start'])
    else:  # streams are created in request order (see predict.py, asserted there)
        recs = [record(by_stream[s]) for s in sorted(seg)]
    assert len(recs) == w['optimizer_setup_count'], (w['level'], len(recs), w['optimizer_setup_count'])
    span = (max(r['end'] for r in recs) - min(r['start'] for r in recs)) / 1e9
    lv = levels.setdefault(w['level'], dict(windows=[], recs=[], slot_mode=slot_mode))
    w0 = min(r['start'] for r in recs)
    late = [(r['start'] - w0) / 1e9 for r in recs]
    lv.setdefault('late_starts', []).extend(late)
    lv.setdefault('max_service', []).append(max(r['span'] for r in recs))
    lv['windows'].append(dict(span=span, setups=len(recs), targets=w['target_optimization_count'],
                              workers=w['optimizer_worker_count'] or len(recs),
                              host_s=w['optimizer_host_ms'] / 1e3,
                              channels=len({r['channel'] for r in recs}),
                              streams=len(seg)))
    lv['recs'].extend(recs)

out = dict(run=str(run), main_thread=main_thread, kernels=int(len(K)), runtime=int(len(RT)),
           optimizer_kernels=int(len(idx)), segments=len(segments), levels={})
for level, lv in levels.items():
    recs, ws = lv['recs'], lv['windows']
    span_sum = sum(w['span'] for w in ws)
    sv = [r['span'] for r in recs]
    row = dict(slot_mode=lv['slot_mode'], windows=len(ws), setups=len(recs),
               targets=sum(w['targets'] for w in ws),
               receipt_host_s=round(sum(w['host_s'] for w in ws), 3),
               trace_span_sum_s=round(span_sum, 3),
               window_spans_s=[round(w['span'], 3) for w in ws] if len(ws) <= 8 else None,
               window_channels=sorted({w['channels'] for w in ws}),
               service_sum_s=round(sum(sv), 3), busy_sum_s=round(sum(r['busy'] for r in recs), 3),
               service_p50_s=round(float(np.median(sv)), 4),
               service_p90_s=round(float(np.percentile(sv, 90)), 4),
               service_max_s=round(max(sv), 3),
               floor_s=round(max(sum(sv) / 32, max(sv)), 3),
               kernels_per_setup=sorted(collections.Counter(r['kernels'] for r in recs).items())[:6])
    if not lv['slot_mode']:
        late = lv['late_starts']
        row['late_start_over_10ms'] = sum(x > 0.010 for x in late)
        row['late_start_over_100ms'] = sum(x > 0.100 for x in late)
        row['late_start_max_s'] = round(max(late), 3)
        row['window_span_minus_max_service_s'] = round(span_sum - sum(lv['max_service']), 3)
    model_w, i = 0.0, 0
    for w in ws:
        model_w += list_schedule(sv[i:i + w['setups']], w['workers'])
        i += w['setups']
    row['model_original_s'] = round(model_w, 3)
    row['model_fit_error'] = round(model_w / span_sum - 1, 4)
    for slots in (32, 48, 64):
        row['model_refill_%d_s' % slots] = round(list_schedule(sv, slots), 3)
    row['model_refill_32_longest_first_s'] = round(list_schedule(sorted(sv, reverse=True), 32), 3)
    row['model_saving_refill_32_s'] = round(span_sum - row['model_refill_32_s'], 3)
    if lv['slot_mode']:
        # Host-side cost between consecutive setups on one slot stream (bind upload,
        # status download, detach, lease release): the per-setup refill overhead.
        by_slot = collections.defaultdict(list)
        for r in recs:
            by_slot[r['stream_id']].append(r)
        gaps = []
        for rs in by_slot.values():
            rs.sort(key=lambda r: r['start'])
            gaps += [(b['start'] - a['end']) / 1e9 for a, b in zip(rs, rs[1:])]
        row['slot_gap_p50_ms'] = round(float(np.median(gaps)) * 1e3, 3)
        row['slot_gap_p90_ms'] = round(float(np.percentile(gaps, 90)) * 1e3, 3)
        row['slot_gap_sum_over_slots_s'] = round(sum(gaps) / len(by_slot), 3)
        row['model_refill_32_plus_gap_s'] = round(list_schedule(
            [s + float(np.median(gaps)) for s in sv], 32), 3)
    else:
        for gap_ms in (0.5, 1.0, 2.0):
            row['model_refill_32_plus_%gms_s' % gap_ms] = round(list_schedule(
                [s + gap_ms / 1e3 for s in sv], 32), 3)
    top = sorted(recs, key=lambda r: -r['span'])[:3]
    row['longest'] = [dict(span_s=round(r['span'], 3), busy_s=round(r['busy'], 3),
                           kernels=r['kernels'], longest_kernel=r['longest_kernel'],
                           longest_kernel_s=round(r['longest_kernel_s'], 3), grid=r['grid'],
                           block=r['block'], regs=r['regs'], lmem=r['lmem'],
                           lmem_total=r['lmem_total']) for r in top]
    out['levels'][str(level)] = row

# Context local-memory reservation as seen by each optimizer kernel family.
lm = collections.defaultdict(set)
for i in idx[::max(1, len(idx) // 20000)]:
    lm[k_family[i]].add((int(K['regs'][i]), int(K['lmem'][i]), int(K['lmem_total'][i])))
out['optimizer_local_memory'] = {f: sorted(v)[-4:] for f, v in lm.items()}

# Whole-run GPU busy union and idle gaps.
o = np.argsort(K['start'])
starts, ends = K['start'][o].astype(np.int64), K['end'][o].astype(np.int64)
run_max = np.maximum.accumulate(ends)
gaps = starts[1:] - run_max[:-1]
gap_mask = gaps > 0
out['gpu_first_kernel_s'] = 0.0
out['gpu_span_s'] = round((int(run_max[-1]) - int(starts[0])) / 1e9, 3)
out['gpu_idle_s'] = round(float(gaps[gap_mask].sum()) / 1e9, 3)
for threshold_ms in (1, 10, 100, 1000):
    big = gaps > threshold_ms * 1e6
    out['gpu_idle_gaps_over_%dms_s' % threshold_ms] = round(float(gaps[big].sum()) / 1e9, 3)
(run / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
