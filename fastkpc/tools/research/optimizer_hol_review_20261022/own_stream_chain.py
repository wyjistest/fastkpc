"""For each optimizer kernel, inspect the preceding operations in its own stream and their delays."""
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds"))
ops = []
for r in c.execute("select start, end, streamId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    ops.append(dict(start=r[0], end=r[1], stream=r[2], corr=r[3], what='K:' + names.get(r[4], '?')[:40]))
kind = {1: 'H2D', 2: 'D2H', 8: 'D2D', 10: 'P2P'}
for r in c.execute("select start, end, streamId, correlationId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
    ops.append(dict(start=r[0], end=r[1], stream=r[2], corr=r[3], what='M:%s:%d' % (kind.get(r[4], r[4]), r[5])))
R = {r[0]: (r[1], r[2], names.get(r[3], '?')) for r in c.execute("select correlationId, start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME")}
by_stream = collections.defaultdict(list)
for o in ops: by_stream[o['stream']].append(o)
for s in by_stream.values(): s.sort(key=lambda o: o['start'])
opt = sorted([o for o in ops if 'optimize_multi_penalty_targets' in o['what']], key=lambda o: o['start'])
first = opt[:64]                       # first window run (64 setups)
t0 = min(o['start'] for o in first)
rows = []
for k in first:
    seq = by_stream[k['stream']]; i = seq.index(k)
    window_ops = [o for o in seq[:i] if o['start'] >= t0 - 1e9]
    chain = []
    for o in window_ops[-4:]:
        api = R.get(o['corr'])
        chain.append((o['what'], round((o['start'] - api[0]) / 1e6, 1) if api else None, round((o['end'] - o['start']) / 1e6, 2)))
    gap_prev = (k['start'] - window_ops[-1]['end']) / 1e6 if window_ops else None
    api = R.get(k['corr'])
    rows.append(((k['start'] - t0) / 1e6, k['stream'], round((k['start'] - api[0]) / 1e6, 1), round(gap_prev, 3) if gap_prev is not None else None, round((k['end'] - k['start']) / 1e6, 1), chain))
rows.sort()
for r in rows[:6] + rows[28:36] + rows[-6:]:
    print('start+%.0fms stream %d launch->start %.1fms gap-after-own-prev %sms dur %.0fms | own prev ops (what, api->start ms, dur ms): %s' % r)
