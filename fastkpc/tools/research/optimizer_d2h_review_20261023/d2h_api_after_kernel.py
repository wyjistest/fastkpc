"""Count result D2H API calls entered before their own stream's last kernel ended
(production: the thread waits inside the pageable D2H; patched: should be zero)."""
import collections, sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
api = {co: s for s, co in db.execute('select start, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
K = list(db.execute('select start, end, streamId from CUPTI_ACTIVITY_KIND_KERNEL'))
M = list(db.execute('select start, end, streamId, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2'))
for ps, pe, label in sorted((s, e, t or S.get(ti)) for s, e, t, ti in
        db.execute('select start, end, text, textId from NVTX_EVENTS') if e is not None):
    if not (label or '').startswith('pair:'): continue
    last = collections.defaultdict(int)
    for s, e, st in K:
        if ps <= s and e <= pe: last[st] = max(last[st], e)
    d = [(api[co], last[st]) for s, e, st, co in M if ps <= s <= pe]
    print('%-26s result D2H API entered before own last kernel end: %3d / %d' % (label, sum(a < k for a, k in d), len(d)))
