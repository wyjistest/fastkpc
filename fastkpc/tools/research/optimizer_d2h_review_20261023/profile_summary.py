"""Per-variant shape metrics from an Nsight export of the double-window replay
(same definitions as Codex's analyze.py): wall, optimizer peak concurrency,
H2D starting >5 ms after its API returned, own last kernel end -> result D2H
start, and cross-window optimizer overlap (intersection of the two windows'
optimizer-kernel interval unions; not a saving)."""
import collections, sqlite3, statistics, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
api = {co: (s, e) for s, e, co in db.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
K = [(s, e, st, S[n]) for s, e, st, n in db.execute('select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL')]
M = [(s, e, st, co, k) for s, e, st, co, k in db.execute('select start, end, streamId, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY')]
ids = sorted(r[0] for r in db.execute('select streamId from TARGET_INFO_CUDA_STREAM'))
side = {sid: int(i >= 64) for i, sid in enumerate(ids)}  # prepared streams are created in setup order
pairs = sorted([(s, e, t or S.get(ti)) for s, e, t, ti in db.execute('select start, end, text, textId from NVTX_EVENTS')
                if e is not None and (t or S.get(ti) or '').startswith('pair:')])
def union(iv):
    tot, end = 0, None
    for a, b in sorted(iv):
        if end is None or a >= end: tot += b - a; end = b
        elif b > end: tot += b - end; end = b
    return tot
def peak(iv):
    ev = collections.Counter()
    for a, b in iv: ev[a] += 1; ev[b] -= 1
    cur = best = 0
    for t in sorted(ev): cur += ev[t]; best = max(best, cur)
    return best
for ps, pe, label in pairs:
    opt = [(s, e, st) for s, e, st, n in K if ps <= s and e <= pe and 'optimize_multi_penalty_targets' in n]
    last = collections.defaultdict(int)
    for s, e, st, n in K:
        if ps <= s and e <= pe: last[st] = max(last[st], e)
    h2d = [(s, co) for s, e, st, co, k in M if k == 1 and ps <= s <= pe]
    d2h = [(s, st) for s, e, st, co, k in M if k == 2 and ps <= s <= pe]
    wait = [(s - last[st]) / 1e6 for s, st in d2h]
    left = [(s, e) for s, e, st in opt if side[st] == 0]; right = [(s, e) for s, e, st in opt if side[st] == 1]
    print('%-26s wall %7.3f s | optimizer kernels %d peak %d | H2D >5 ms after API %3d | '
          'last kernel->D2H ms median %8.3f max %9.3f | cross-window overlap %.3f s' % (
          label, (pe - ps) / 1e9, len(opt), peak([(s, e) for s, e, st in opt]),
          sum(s - api[co][1] > 5e6 for s, co in h2d), statistics.median(wait), max(wait),
          (union(left) + union(right) - union(left + right)) / 1e9))
