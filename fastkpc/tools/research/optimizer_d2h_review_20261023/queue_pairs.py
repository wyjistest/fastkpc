"""Which streams share a hardware queue? For each H2D (first op of a setup chain)
that starts >5 ms after its API returned, find the other stream whose op ended
<=50 us before it. Report pairs by stream creation rank (streamId order)."""
import bisect, collections, sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
api = {co: (s, e) for s, e, co in db.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
ids = sorted(r[0] for r in db.execute('select streamId from TARGET_INFO_CUDA_STREAM'))
rank = {sid: i for i, sid in enumerate(ids)}
ops = [(e, st) for s, e, st in db.execute('select start, end, streamId from CUPTI_ACTIVITY_KIND_KERNEL')]
ops += [(e, st) for s, e, st in db.execute('select start, end, streamId from CUPTI_ACTIVITY_KIND_MEMCPY')]
ops.sort(); et = [o[0] for o in ops]
pairs = sorted([(s, e, t or S.get(ti)) for s, e, t, ti in db.execute('select start, end, text, textId from NVTX_EVENTS')
                if e is not None and (t or S.get(ti) or '').startswith('pair:')])
print('streams', len(ids), 'id range', ids[0], ids[-1], 'contiguous', ids == list(range(ids[0], ids[0] + len(ids))))
for ps, pe, label in pairs:
    h2d = [(s, st, co) for s, e, st, co, k in db.execute(
        'select start, end, streamId, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY') if k == 1 and ps <= s <= pe]
    diff = collections.Counter(); n = 0
    for s, st, co in h2d:
        if s - api[co][1] <= 5e6: continue
        n += 1
        i = bisect.bisect_left(et, s - 50000); j = bisect.bisect_right(et, s + 1000)
        hits = [o for o in ops[i:j] if o[1] != st]
        if hits:
            o = max(hits); diff[(rank[st] - rank[o[1]]) % 32] += 1
        else:
            diff['none'] += 1
    print(label, 'delayed H2D', n, ' (rank difference mod 32 -> count):', dict(diff))
