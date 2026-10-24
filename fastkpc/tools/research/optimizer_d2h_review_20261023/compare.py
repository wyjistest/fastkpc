"""Compare primary timings (6 balanced cycles) across campaign directories.
Usage: compare.py LABEL=DIR [LABEL=DIR ...]"""
import csv, statistics, sys
from pathlib import Path
for pair in ('L3-W31+W32', 'L5-W7+W8'):
    print('==', pair)
    for arg in sys.argv[1:]:
        label, d = arg.split('=', 1)
        p = Path(d) / f'campaign-{pair}-runs.csv'
        if not p.exists(): continue
        rows = list(csv.DictReader(p.open()))
        assert all(r['parity'] == '1' for r in rows)
        t = {v: [float(r['wall_ms']) / 1e3 for r in rows if r['kind'] == 'timing' and r['variant'] == v] for v in 'ABC'}
        cyc = {}
        for r in rows:
            if r['kind'] == 'timing': cyc.setdefault(int(r['cycle']), {})[r['variant']] = float(r['wall_ms'])
        wins = {v: sum(c[v] < c['A'] for c in cyc.values()) for v in 'BC'}
        print('  %-10s ' % label + '  '.join('%s median %.3f s (%.3f-%.3f)' % (v, statistics.median(x), min(x), max(x))
              for v, x in t.items()) + '   wins vs same-cycle A: B %d/6, C %d/6' % (wins['B'], wins['C']))
