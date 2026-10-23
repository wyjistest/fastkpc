import sqlite3, sys, bisect, collections
c = sqlite3.connect(sys.argv[1])
names = dict(c.execute("select id, value from StringIds"))
R = {r[0]: r[1] for r in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
M = [dict(start=r[0], end=r[1], stream=r[2], corr=r[3], kind=r[4]) for r in c.execute("select start, end, streamId, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")]
K = [dict(start=r[0], end=r[1], stream=r[2], name=names.get(r[3], '?')) for r in c.execute("select start, end, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
opt = sorted([k for k in K if 'optimize_multi_penalty_targets' in k['name']], key=lambda k: k['start'])
# group optimizer kernels into windows of 64 by start order
wins = [opt[i:i+64] for i in range(0, len(opt), 64)]
ends = sorted([(m['end'], m['stream']) for m in M] + [(k['end'], k['stream']) for k in K]); et = [e[0] for e in ends]
h2d = sorted([m for m in M if m['kind'] == 1 and m['corr'] in R], key=lambda m: m['start'])
for wi, w in enumerate(wins):
    t0 = min(k['start'] for k in w) - 50e6; t1 = max(k['end'] for k in w)
    hs = [m for m in h2d if t0 <= m['start'] <= t1]
    d = [m for m in hs if m['start'] - R[m['corr']] > 5e6]
    coinc = 0
    for m in d:
        i = bisect.bisect_left(et, m['start'] - 50000); j = bisect.bisect_right(et, m['start'] + 1000)
        if any(e[1] != m['stream'] for e in ends[i:j]): coinc += 1
    # concurrency profile of optimizer kernels
    ev = sorted([(k['start'], 1) for k in w] + [(k['end'], -1) for k in w]); cur = peak = 0
    for t, x in ev: cur += x; peak = max(peak, cur)
    durs = sorted((k['end'] - k['start']) / 1e9 for k in w)
    wall = (t1 - min(k['start'] for k in w)) / 1e9
    print(f"window {wi}: wall {wall:.3f}s peak concurrent {peak} | H2D {len(hs)} delayed>5ms {len(d)} (start right after other-stream end: {coinc}) | optimizer kernel dur: max {durs[-1]:.2f}s median {durs[len(durs)//2]:.2f}s sum {sum(durs):.1f}s -> sum/32 {sum(durs)/32:.2f}s")
