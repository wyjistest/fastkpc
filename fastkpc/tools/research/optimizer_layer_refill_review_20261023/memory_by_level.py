"""Peak GPU memory per level in a status-wait canonical run: telemetry (200 ms samples)
aligned to the level timeline. Alignment: method start = first sample where the run GPU
exceeds base+4 GiB (component cache allocation); levels follow level_trace order."""
import datetime as dt, json, sys
run_dir, gpu = sys.argv[1], int(sys.argv[2])
rep = json.load(open(run_dir + '/report.json'))
rows = []
for line in open(run_dir + '-gpu.csv'):
    f = [x.strip() for x in line.split(',')]
    if int(f[1]) != gpu: continue
    t = dt.datetime.strptime(f[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
    rows.append((t, int(f[2]), int(f[3])))
base = rows[0][1]
t0 = next(t for t, m, u in rows if m > base + 4096)
edges, acc = [], t0
for x in rep['level_trace']:
    edges.append((x['level'], acc, acc + x['elapsed_ms'] / 1e3)); acc += x['elapsed_ms'] / 1e3
peak = max(rows, key=lambda r: r[1])
print('base %d MiB; method start aligned at first sample > base+4 GiB; overall peak %d MiB at +%.1f s' % (base, peak[1], peak[0] - t0))
for lvl, a, b in edges:
    ms = [m for t, m, u in rows if a <= t < b]; us = [u for t, m, u in rows if a <= t < b]
    if ms: print('L%d  %6.1f–%6.1f s  peak %5d MiB  median %5d MiB  median util %3d%%' % (lvl, a - t0, b - t0, max(ms), sorted(ms)[len(ms) // 2], sorted(us)[len(us) // 2]))
