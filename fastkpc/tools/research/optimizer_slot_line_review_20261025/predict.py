"""Busy-time refill predictions in true request order, L8 launch detail and host-call
accounting from the production trace.

python3 -I predict.py RUN_DIR

Request order. In window mode every request owns a fresh prepared object whose stream is
created on the main thread in request (metadata) order, and windows are consecutive
chunks of the level's std::map order. CUPTI stream IDs are assigned at creation, so the
level's original order is: windows in order, ascending stream ID within a window. The
script asserts that every window's stream IDs are consecutive and that window ranges
increase. In slot mode (L4/L5) requests leave one original-order queue, so burst start
order is fetch order. (Revision 2026-10-25b: the first version sorted window-mode setups
by first-kernel start time, which head-of-line waits perturb.)

Service time. Window-mode spans are inflated by head-of-line waits (64 streams on 32
hardware channels), so refill is modeled from kernel busy time x the slot-level
span/busy ratio plus the median slot gap, both measured on L4/L5 of the same run. This
assumes window-mode busy time persists under sustained refill load; it is not checked
for L3/L6 here.
"""
import collections
import heapq
import json
import random
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
names = dict((int(a), b) for a, b in (l.split('\t', 1) for l in
             (run / 'kernel-names.tsv').read_text().splitlines()))
result = json.loads((run / 'profile-result.json').read_text())

order = np.argsort(RT['corr'])
rt_corr, rt_thread = RT['corr'][order], RT['thread'][order]
rt_start, rt_end = RT['start'][order], RT['end'][order]
pos = np.clip(np.searchsorted(rt_corr, K['corr']), 0, len(rt_corr) - 1)
matched = rt_corr[pos] == K['corr']
k_thread = np.where(matched, rt_thread[pos], 0)
k_launch = np.where(matched, rt_start[pos], 0).astype(np.int64)
k_launch_end = np.where(matched, rt_end[pos], 0).astype(np.int64)
threads, counts = np.unique(RT['thread'], return_counts=True)
main_thread = int(threads[np.argmax(counts)])
multi = np.array(['mgcv_multi_penalty_gcv' in names[i] for i in range(len(names))])
is_opt = multi[K['name']] & (k_thread != main_thread)
idx = np.nonzero(is_opt)[0]
idx = idx[np.argsort(K['start'][idx], kind='stable')]
by_stream = collections.defaultdict(list)
for i in idx:
    by_stream[int(K['stream'][i])].append(int(i))
streams = sorted(by_stream, key=lambda s: K['start'][by_stream[s][0]])
windows = [b for b in result['prefill_batches'] if b['penalty_class'] == 'multi']
segments, cursor = [], 0
for w in windows:
    n = w['optimizer_setup_count']
    width = min(32, n) if n > 64 else n
    segments.append(streams[cursor:cursor + width])
    cursor += width
assert cursor == len(streams)
previous = -1
for seg in segments:
    ids = sorted(seg)
    assert ids[-1] - ids[0] == len(ids) - 1 and ids[0] > previous, 'stream ids not creation-ordered'
    previous = ids[-1]

def bursts(stream):
    ks = by_stream[stream]
    first = K['name'][ks[0]]
    calls, cur = [], []
    for i in ks:
        if cur and K['name'][i] == first:
            calls.append(cur)
            cur = []
        cur.append(i)
    calls.append(cur)
    return calls

def schedule(services, slots=32):
    free = [0.0] * max(1, min(slots, len(services)))
    heapq.heapify(free)
    finish = 0.0
    for s in services:
        t = heapq.heappop(free)
        heapq.heappush(free, t + s)
        finish = max(finish, t + s)
    return finish

def busy(c):
    return float((K['end'][c] - K['start'][c]).sum()) / 1e9

per_level = collections.OrderedDict()
for seg, w in zip(segments, windows):
    slot = w['optimizer_setup_count'] > 64
    if slot:
        calls = sorted((c for s in seg for c in bursts(s)), key=lambda c: K['start'][c[0]])
        start_order = calls
    else:
        calls = [by_stream[s] for s in sorted(seg)]                    # request order
        start_order = sorted(calls, key=lambda c: K['start'][c[0]])   # first version
    lv = per_level.setdefault(w['level'], dict(slot=slot, busy=[], busy_start_order=[],
                                               span=[], windows=[], measured=0.0,
                                               window_max_busy=0.0))
    s0 = min(int(K['start'][c[0]]) for c in calls)
    e0 = max(int(K['end'][c].max()) for c in calls)
    lv['measured'] += (e0 - s0) / 1e9
    lv['busy'] += [busy(c) for c in calls]
    lv['busy_start_order'] += [busy(c) for c in start_order]
    lv['span'] += [(int(K['end'][c].max()) - int(K['start'][c[0]])) / 1e9 for c in calls]
    lv['window_max_busy'] += max(busy(c) for c in calls)
    lv['windows'].append(len(calls))

gap = {}
for level in (4, 5):
    seg = segments[[w['level'] for w in windows].index(level)]
    gaps = []
    for s in seg:
        cs = bursts(s)
        gaps += [(int(K['start'][b[0]]) - int(K['end'][a].max())) / 1e9 for a, b in zip(cs, cs[1:])]
    gap[level] = float(np.median(gaps))
slot_gap = float(np.mean(list(gap.values())))
ratio = float(np.mean([sum(per_level[l]['span']) / sum(per_level[l]['busy']) for l in (4, 5)]))
service = lambda b: b * ratio + slot_gap

out = dict(order='window mode: windows in order, ascending stream id (creation order) '
                 'within a window; slot mode: burst start (fetch) order',
           slot_gap_median_ms={str(k): round(v * 1e3, 3) for k, v in gap.items()},
           slot_span_over_busy=round(ratio, 4), levels={})
rng = random.Random(20261025)
for level, lv in per_level.items():
    sv = [service(b) for b in lv['busy']]
    row = dict(slot_mode=lv['slot'], setups=len(sv), measured_s=round(lv['measured'], 3),
               busy_sum_s=round(sum(lv['busy']), 3), span_sum_s=round(sum(lv['span']), 3),
               busy_max_s=round(max(lv['busy']), 3),
               sum_of_window_max_busy_s=round(lv['window_max_busy'], 3),
               refill32_request_order_s=round(schedule(sv, 32), 3),
               refill32_start_order_s=round(schedule([service(b) for b in lv['busy_start_order']], 32), 3),
               refill32_floor_s=round(max(sum(sv) / 32, max(sv)), 3),
               refill32_longest_first_s=round(schedule(sorted(sv, reverse=True), 32), 3))
    perms = []
    for _ in range(200):
        p = sv[:]
        rng.shuffle(p)
        perms.append(schedule(p, 32))
    row['refill32_random_order_p5_p50_p95_s'] = [round(float(np.percentile(perms, q)), 3)
                                                  for q in (5, 50, 95)]
    if not lv['slot']:
        total, i = 0.0, 0
        for n in lv['windows']:
            total += schedule(sv[i:i + n], 32)
            i += n
        row['original_windows_workers32_s'] = round(total, 3)
        longest = int(np.argmax(lv['busy']))
        row['longest_request_position'] = [longest, len(sv)]
    out['levels'][str(level)] = row

# L8: every setup's optimizer launch (by correlation id) and its execution.
seg8 = segments[[w['level'] for w in windows].index(8)]
opt_ids = {i for i, n in names.items() if 'optimize_multi_penalty_targets' in n}
rows = []
for s in sorted(seg8):
    ks = [i for i in by_stream[s] if int(K['name'][i]) in opt_ids]
    assert len(ks) == 1
    i = ks[0]
    rows.append(dict(stream=s, channel=int(K['channel'][i]), targets=int(K['grid'][i]),
                     launch_api=int(k_launch[i]), launch_api_ms=(int(k_launch_end[i]) - int(k_launch[i])) / 1e6,
                     start=int(K['start'][i]), end=int(K['end'][i])))
t8 = min(r['start'] for r in rows)
out['l8_optimizer_launches'] = [dict(stream=r['stream'], channel=r['channel'], targets=r['targets'],
                                     launch_at_s=round((r['launch_api'] - t8) / 1e9, 4),
                                     launch_api_ms=round(r['launch_api_ms'], 3),
                                     start_s=round((r['start'] - t8) / 1e9, 4),
                                     duration_s=round((r['end'] - r['start']) / 1e9, 3))
                                for r in sorted(rows, key=lambda r: r['launch_api'])]

# Gaps with no recorded kernel (memcpy/memset activity was not traced).
o = np.argsort(K['start'])
st_, en_ = K['start'][o].astype(np.int64), K['end'][o].astype(np.int64)
run_max = np.maximum.accumulate(en_)
gl = st_[1:] - run_max[:-1]
out['no_kernel_gaps'] = dict(note='kernel and runtime-API activity only; copy engines not traced',
                             total_s=round(float(gl[gl > 0].sum()) / 1e9, 3),
                             over_10ms_s=round(float(gl[gl > 1e7].sum()) / 1e9, 3),
                             over_100ms_s=round(float(gl[gl > 1e8].sum()) / 1e9, 3))

# cudaGetDeviceProperties (cbid 440): cumulative, wall-clock union, main thread.
sel = RT['cbid'] == 440
a, b = RT['start'][sel].astype(np.int64), RT['end'][sel].astype(np.int64)
iv = sorted(zip(a.tolist(), b.tolist()))
union, cs, ce = 0, iv[0][0], iv[0][1]
for s, e in iv[1:]:
    if s > ce:
        union += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
union += ce - cs
out['get_device_properties'] = dict(calls=int(sel.sum()), cumulative_s=round(float((b - a).sum()) / 1e9, 3),
                                     union_s=round(union / 1e9, 3),
                                     main_thread_s=round(float((b - a)[RT['thread'][sel] == main_thread].sum()) / 1e9, 3),
                                     main_thread_calls=int((RT['thread'][sel] == main_thread).sum()))
(run / 'predict.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
