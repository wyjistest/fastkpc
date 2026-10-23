"""Do delayed H2D (Y upload) starts coincide with another stream's D2H/kernel completion?"""
import sqlite3, sys, collections, bisect
c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds"))
R = {r[0]: r[1] for r in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
M = [dict(start=r[0], end=r[1], stream=r[2], corr=r[3], kind=r[4], bytes=r[5]) for r in c.execute("select start, end, streamId, correlationId, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY")]
K = [dict(start=r[0], end=r[1], stream=r[2], name=names.get(r[3], '?')) for r in c.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
ends = sorted([(m['end'], 'D2H' if m['kind'] == 2 else ('H2D' if m['kind'] == 1 else 'M%d' % m['kind']), m['stream']) for m in M] +
              [(k['end'], 'K:' + k['name'][:28], k['stream']) for k in K])
et = [e[0] for e in ends]
h2d = sorted([m for m in M if m['kind'] == 1 and m['corr'] in R], key=lambda m: m['start'])
delayed = [m for m in h2d if (m['start'] - R[m['corr']]) > 5e6]
print('H2D total', len(h2d), 'delayed >5ms', len(delayed))
cnt = collections.Counter(); pairs = collections.Counter(); lag = []
for m in delayed:
    i = bisect.bisect_left(et, m['start'] - 50000); j = bisect.bisect_right(et, m['start'] + 1000)
    hits = [e for e in ends[i:j] if e[2] != m['stream']]
    if hits:
        h = max(hits, key=lambda e: e[0]); cnt[h[1]] += 1; pairs[(m['stream'], h[2])] += 1; lag.append((m['start'] - h[0]) / 1e3)
    else: cnt['none'] += 1
print('delayed H2D start preceded (<=50us) by another stream end of:', dict(cnt))
if lag: lag.sort(); print('lag from that end to H2D start (us): median %.1f max %.1f' % (lag[len(lag)//2], lag[-1]))
streams = sorted({m['stream'] for m in h2d})
print('streams', len(streams), streams[:8], '...')
pc = collections.Counter()
for (a, b), n in pairs.items(): pc[(min(a, b), max(a, b))] += n
print('blocking stream pairs (unordered) and counts:', sorted(pc.items(), key=lambda x: -x[1])[:16])
