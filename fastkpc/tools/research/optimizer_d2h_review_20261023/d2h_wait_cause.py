"""What does each delayed result D2H wait for? Per profiled variant (A/B/C):
coincidence of each D2H device start with the end of another stream's op,
D2H/D2H serialization, and host API ordering."""
import bisect, collections, sqlite3, statistics, sys

db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
api = {co: (s, e, tid, S[n]) for s, e, tid, co, n in
       db.execute('select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME')}
K = [dict(start=s, end=e, stream=st, corr=co, name=S[n], what='K') for s, e, st, co, n in
     db.execute('select start, end, streamId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL')]
M = [dict(start=s, end=e, stream=st, corr=co, what={1: 'H2D', 2: 'D2H'}.get(k, 'M%d' % k), src=sk, dst=dk, bytes=b)
     for s, e, st, co, k, sk, dk, b in
     db.execute('select start, end, streamId, correlationId, copyKind, srcKind, dstKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY')]
hw = collections.Counter(r[0] for r in db.execute('select hwId from TARGET_INFO_CUDA_STREAM'))
print('TARGET_INFO_CUDA_STREAM hwId values:', dict(hw))
pairs = sorted([(s, e, t) for s, e, t, ti in db.execute('select start, end, text, textId from NVTX_EVENTS')
                if e is not None and (t or '').startswith('pair:')])
pairs = [(s, e, t if t else S.get(ti)) for s, e, t in pairs]
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
for ps, pe, label in pairs:
    v = label.split(':')[-2]
    ops = [o for o in K + M if ps <= o['start'] and o['end'] <= pe]
    d2h = sorted([o for o in ops if o['what'] == 'D2H'], key=lambda o: o['start'])
    ends = sorted((o['end'], o['what'] if o['what'] != 'K' else 'K:' + o['name'][:26], o['stream']) for o in ops)
    et = [x[0] for x in ends]
    last_k = collections.defaultdict(int)
    for o in ops:
        if o['what'] == 'K': last_k[o['stream']] = max(last_k[o['stream']], o['end'])
    cause = collections.Counter(); lags = []; delays = []; api_before = 0
    for o in d2h:
        own = last_k[o['stream']]
        delay = (o['start'] - own) / 1e6; delays.append(delay)
        if delay <= 0.05: cause['<=50us after own kernel'] += 1; continue
        a = api[o['corr']]
        api_before += a[0] < own
        i = bisect.bisect_left(et, o['start'] - 50000); j = bisect.bisect_right(et, o['start'] + 1000)
        hits = [x for x in ends[i:j] if x[2] != o['stream']]
        if hits:
            h = max(hits); cause[h[1]] += 1; lags.append((o['start'] - h[0]) / 1e3)
        else:
            cause['none within 50us'] += 1
    overl = sum(1 for a, b in zip(d2h, d2h[1:]) if b['start'] < a['end'])
    print('\n== %s  wall %.3f s  D2H n=%d  dst kinds %s' % (label, (pe - ps) / 1e9, len(d2h),
          dict(collections.Counter((o['src'], o['dst']) for o in d2h))))
    print('  own last kernel end -> D2H start (ms): median %.1f p90 %.1f max %.1f; >50us: %d' %
          (statistics.median(delays), q(delays, .9), max(delays), sum(d > .05 for d in delays)))
    print('  delayed D2H whose API was entered before own last kernel ended: %d' % api_before)
    print('  delayed D2H start preceded (<=50us) by end of another stream op:', dict(cause))
    if lags: print('  lag from that end (us): median %.1f max %.1f' % (statistics.median(lags), max(lags)))
    print('  consecutive D2H device intervals overlapping: %d of %d' % (overl, len(d2h) - 1))
