"""Per-window head-of-line comparison between two traced canonical runs.

python3 -I compare_windows.py RUN_A RUN_B

For every window-mode window (request order as in predict.py): measured kernel span,
the longest setup busy time, their difference (time the window spends beyond its
critical setup: late starts and channel waits), setups starting >100 ms late, and the
number of hardware channels used. Also the L8 optimizer launch/start pattern.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np


def load(run):
    run = Path(run)
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
    rc, rth, rst = RT['corr'][order], RT['thread'][order], RT['start'][order]
    pos = np.clip(np.searchsorted(rc, K['corr']), 0, len(rc) - 1)
    ok = rc[pos] == K['corr']
    kth = np.where(ok, rth[pos], 0)
    klaunch = np.where(ok, rst[pos], 0).astype(np.int64)
    th, c = np.unique(RT['thread'], return_counts=True)
    main = th[np.argmax(c)]
    multi = np.array(['mgcv_multi_penalty_gcv' in names[i] for i in range(len(names))])
    idx = np.nonzero(multi[K['name']] & (kth != main))[0]
    idx = idx[np.argsort(K['start'][idx], kind='stable')]
    by = collections.defaultdict(list)
    for i in idx:
        by[int(K['stream'][i])].append(int(i))
    streams = sorted(by, key=lambda s: K['start'][by[s][0]])
    windows = [b for b in result['prefill_batches'] if b['penalty_class'] == 'multi']
    out, cur = collections.defaultdict(list), 0
    for w in windows:
        n = w['optimizer_setup_count']
        width = min(32, n) if n > 64 else n
        seg = sorted(streams[cur:cur + width])
        cur += width
        if n > 64:
            continue
        s0 = min(int(K['start'][by[s][0]]) for s in seg)
        e0 = max(int(K['end'][by[s]].max()) for s in seg)
        busy = [float((K['end'][by[s]] - K['start'][by[s]]).sum()) / 1e9 for s in seg]
        late = [(int(K['start'][by[s][0]]) - s0) / 1e9 for s in seg]
        longest = int(np.argmax(busy))
        out[w['level']].append(dict(span=(e0 - s0) / 1e9, max_busy=max(busy),
                                    late_over_100ms=sum(x > 0.1 for x in late),
                                    longest_late_s=late[longest],
                                    channels=len({int(K['channel'][by[s][0]]) for s in seg})))
    opt = {i for i, n in names.items() if 'optimize_multi_penalty_targets' in n}
    seg8 = sorted(streams[cur - 9:cur])
    l8 = []
    for s in seg8:
        i = [j for j in by[s] if int(K['name'][j]) in opt][0]
        l8.append((int(klaunch[i]), int(K['start'][i]), int(K['end'][i])))
    t8 = min(x[1] for x in l8)
    l8 = [dict(launch_s=round((a - t8) / 1e9, 4), start_s=round((b - t8) / 1e9, 3),
               duration_s=round((c - b) / 1e9, 3)) for a, b, c in sorted(l8)]
    return out, l8, result['elapsed']


summary = {}
for label, run in (('A', sys.argv[1]), ('B', sys.argv[2])):
    windows, l8, elapsed = load(run)
    row = dict(run=run, elapsed=elapsed, l8_optimizer=l8)
    for level, ws in sorted(windows.items()):
        row['L%d' % level] = dict(
            windows=len(ws), span_s=round(sum(w['span'] for w in ws), 3),
            max_busy_s=round(sum(w['max_busy'] for w in ws), 3),
            beyond_critical_s=round(sum(w['span'] - w['max_busy'] for w in ws), 3),
            late_over_100ms=sum(w['late_over_100ms'] for w in ws),
            longest_setup_late_s=round(sum(w['longest_late_s'] for w in ws), 3),
            channels=sorted({w['channels'] for w in ws}),
            per_window=[dict(span=round(w['span'], 3), max_busy=round(w['max_busy'], 3),
                             longest_late=round(w['longest_late_s'], 3), channels=w['channels'])
                        for w in ws] if len(ws) <= 4 else None)
    summary[label] = row
print(json.dumps(summary, indent=2))
