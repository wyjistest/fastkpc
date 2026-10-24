"""Attribute pageable D2H serialization in a read-only Nsight trace.

The inferred occupied interval cannot precede this call's API entry. This
is a trace model, not a measurement of an internal driver lock. Report the
first call separately because it has no preceding D2H return in the range.
"""
import collections
import sqlite3
import statistics
import sys


def waiting_interval(previous_return, api_enter, own_kernel_end):
    start = max(previous_return, api_enter)
    return start, max(start, own_kernel_end)


def analyze(path):
    db = sqlite3.connect('file:' + path + '?mode=ro', uri=True)
    names = dict(db.execute('select id, value from StringIds'))
    api = {co: (s, e, tid, names[n]) for s, e, tid, co, n in
           db.execute('select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    kernels = list(db.execute('select start, end, streamId from CUPTI_ACTIVITY_KIND_KERNEL'))
    copies = list(db.execute('select start, end, streamId, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'))
    pairs = sorted((s, e, t or names.get(ti)) for s, e, t, ti in
                   db.execute('select start, end, text, textId from NVTX_EVENTS') if e is not None)
    db.close()
    for ps, pe, label in pairs:
        if not (label or '').startswith('pair:'):
            continue
        last = collections.defaultdict(int)
        for s, e, stream in kernels:
            if ps <= s and e <= pe:
                last[stream] = max(last[stream], e)
        d2h = sorted((dict(start=s, end=e, stream=st, api=api[co])
                      for s, e, st, co, kind in copies if kind == 2 and ps <= s and e <= pe),
                     key=lambda row: row['start'])
        h2d = [api[co] for s, e, st, co, kind in copies if kind == 1 and ps <= s and e <= pe]
        gaps, holds, first, worker_seconds = [], [], 0.0, 0.0
        for i, row in enumerate(d2h):
            previous = d2h[i - 1]['api'][1] if i else ps
            own = last[row['stream']]
            begin, end = waiting_interval(previous, row['api'][0], own)
            gaps.append((row['start'] - max(previous, row['api'][0], own)) / 1e3)
            if i == 0:
                first = (end - begin) / 1e9
                continue
            if end > begin:
                holds.append((end - begin) / 1e9)
                for other in d2h[i + 1:]:
                    lo = max(begin, other['api'][0], last[other['stream']])
                    hi = min(end, other['api'][1])
                    worker_seconds += max(0, hi - lo) / 1e9
        sorted_gaps = sorted(gaps)
        print('\n== %s  wall %.3f s' % (label, (pe - ps) / 1e9))
        print('  D2H start - max(previous return, own API entry, own kernel end): '
              'median %.1f us, p90 %.1f, max %.1f; within [0,100us]: %d/%d' %
              (statistics.median(gaps), sorted_gaps[min(len(gaps)-1, int(.9*len(gaps)))],
               max(gaps), sum(0 <= g <= 100 for g in gaps), len(gaps)))
        print('  inferred wait after first D2H: n=%d, total %.6f s, max %.6f s; '
              'first D2H wait %.6f s (separate); completed-worker seconds %.6f' %
              (len(holds), sum(holds), max(holds, default=0), first, worker_seconds))
        durations = [(e - s) / 1e6 for s, e, tid, name in h2d]
        print('  H2D API duration ms: median %.3f max %.3f; >5 ms: %d' %
              (statistics.median(durations), max(durations), sum(t > 5 for t in durations)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: d2h_lock_chain.py TRACE.sqlite')
    analyze(sys.argv[1])
