"""What limits the L5 32-worker refill (R)? From Codex's profile-kernel-timeline.csv:
per-setup service time (first kernel -> last kernel), the tail of R, and an
idealized 32-worker list-scheduling model (original order vs longest first),
validated against measured R. Model times are idealized, not predictions."""
import csv, heapq, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
by = {v: sorted([r for r in rows if r['variant'] == v], key=lambda r: int(r['setup'])) for v in 'AR'}
assert all(len(x) == 955 for x in by.values())
R = by['R']
svc = [float(r['last_kernel_sec']) - float(r['first_kernel_sec']) for r in R]
svcA = [float(r['last_kernel_sec']) - float(r['first_kernel_sec']) for r in by['A']]
start = [float(r['first_kernel_sec']) for r in R]; end = [float(r['last_kernel_sec']) for r in R]
t0, t1 = min(start), max(end)
print('R measured kernel span %.3f s; sum service %.1f s; sum/32 %.3f s; max service %.3f s' % (t1 - t0, sum(svc), sum(svc) / 32, max(svc)))
print('service time R vs A per setup: median ratio %.4f' % statistics.median([a / b for a, b in zip(svc, svcA) if b > 0]))
top = sorted(range(955), key=lambda i: -svc[i])[:8]
print('longest setups (index, window, service s, start in R s, end in R s):')
for i in top: print('  %4d  W%-2s  %6.3f  %7.3f  %7.3f' % (i, R[i]['window'], svc[i], start[i] - t0, end[i] - t0))
def conc(t): return sum(1 for s, e in zip(start, end) if s <= t < e)
last_start = max(start) - t0
print('last setup start in R: %.3f s; active setups then: %d' % (last_start, conc(max(start))))
for frac in (0.5, 0.7, 0.8, 0.9):
    t = t0 + frac * (t1 - t0); print('  active setups at %.0f%% of span (%.2f s): %d' % (frac * 100, t - t0, conc(t)))
# time with fewer than 32 active setups, at the end
grid = [t0 + k * 0.01 for k in range(int((t1 - t0) / 0.01))]
under = [t for t in grid if conc(t) < 32]
tail_from = next((t for t in grid if all(conc(u) < 32 for u in grid if u >= t)), t1)
print('tail with <32 active (continuous to end) begins at %.2f s; tail length %.2f s' % (tail_from - t0, t1 - tail_from))
def rolling(xs, w=32):
    free = [0.0] * w; heapq.heapify(free); fin = 0.0
    for x in xs:
        t = heapq.heappop(free) + x; fin = max(fin, t); heapq.heappush(free, t)
    return fin
print('model 32 workers, original order : %.3f s (measured R kernel span %.3f)' % (rolling(svc), t1 - t0))
print('model 32 workers, longest first  : %.3f s' % rolling(sorted(svc, reverse=True)))
print('lower bound max(sum/32, max)     : %.3f s' % max(sum(svc) / 32, max(svc)))
for w in (24, 40, 48, 64):
    print('model %d workers, original order : %.3f s   longest first %.3f s' % (w, rolling(svc, w), rolling(sorted(svc, reverse=True), w)))
# time with <32 active before the last setup start (ramp-up and refill gaps)
ev = sorted([(s, 1) for s in start] + [(e, -1) for e in end], key=lambda x: (x[0], x[1]))
cur, prev, under, low = 0, t0, 0.0, 99
for t, d in ev:
    if prev < max(start) and t > prev:
        if cur < 32: under += min(t, max(start)) - prev
        if prev > t0 + 0.5: low = min(low, cur)
    cur += d; prev = t
print('before last start: <32 active for %.3f s (%.2f%%); minimum active after first 0.5 s: %d'
      % (under, 100 * under / (max(start) - t0), low))
