"""For delayed D2H starts that coincide (<=50 us) with another stream's
target_squared_norm_kernel end: was that kernel launched by the thread whose
D2H (the previous one in device order) had just returned?"""
import bisect, collections, sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
api = {co: (s, e, tid) for s, e, tid, co in db.execute('select start, end, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
K = [(e, st, co, S[n]) for s, e, st, co, n in db.execute('select start, end, streamId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL')]
M = list(db.execute('select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2'))
for ps, pe, label in sorted((s, e, t or S.get(ti)) for s, e, t, ti in
        db.execute('select start, end, text, textId from NVTX_EVENTS') if e is not None):
    if not (label or '').startswith('pair:'): continue
    last = collections.defaultdict(int)
    for e, st, co, n in K:
        if ps <= e <= pe: last[st] = max(last[st], e)
    d2h = sorted([m for m in M if ps <= m[0] <= pe])
    ends = sorted((e, st, co, n) for e, st, co, n in K if ps <= e <= pe); et = [x[0] for x in ends]
    same = other = 0
    for i, (s, e, st, co) in enumerate(d2h):
        if i == 0 or s - last[st] <= 50000: continue
        j0 = bisect.bisect_left(et, s - 50000); j1 = bisect.bisect_right(et, s + 1000)
        hits = [x for x in ends[j0:j1] if x[1] != st]
        if not hits: continue
        h = max(hits)
        if 'target_squared_norm' not in h[3]: continue
        prev_tid = api[d2h[i - 1][3]][2]
        same += api[h[2]][2] == prev_tid; other += api[h[2]][2] != prev_tid
    print('%-26s norm-kernel coincidences launched by the thread of the previous D2H: %d, by another thread: %d' % (label, same, other))
