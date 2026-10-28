"""First launch of each kernel function in the production trace, and the stall it causes.

python3 -I first_use.py RUN_DIR

Under CUDA lazy module loading the first launch of a function may synchronize the
context (lmem_growth_probe.cu, scenarios 2-4 vs CUDA_MODULE_LOADING=EAGER). Signature:
kernels launched on other streams after that first launch do not start until every
kernel running at that moment has finished. For each first use, report the kernels
running at its launch API, the latest end among them (the drain point), and how many
later-launched kernels started within 2 ms after the drain point.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
K = np.fromfile(run / 'kernels.bin', dtype=np.dtype([
    ('start', '<u8'), ('end', '<u8'), ('lmem_total', '<u8'), ('stream', '<u4'),
    ('corr', '<u4'), ('name', '<u4'), ('lmem', '<u4'), ('channel', '<u4'),
    ('grid', '<u4'), ('block', '<u4'), ('regs', '<u2')]))
RT = np.fromfile(run / 'runtime.bin', dtype=np.dtype([
    ('start', '<u8'), ('end', '<u8'), ('corr', '<u4'), ('cbid', '<u4'), ('thread', '<u4')]))
names = dict((int(a), b) for a, b in (l.split('\t', 1) for l in
             (run / 'kernel-names.tsv').read_text().splitlines()))
result = json.loads((run / 'profile-result.json').read_text())
launch = dict(zip(RT['corr'].tolist(), zip(RT['start'].tolist(), RT['end'].tolist())))
o = np.argsort(K['start'])
K = K[o]
st, en = K['start'].astype(np.int64), K['end'].astype(np.int64)
t0 = int(st[0])
bounds = np.cumsum([0.0] + [l['elapsed_ms'] / 1e3 for l in result['levels']]) * 1e9 + t0
rows, seen = [], set()
order_by_launch = sorted(range(len(K)), key=lambda i: launch.get(int(K['corr'][i]), (st[i],))[0])
for i in order_by_launch:
    f = int(K['name'][i])
    if f in seen:
        continue
    seen.add(f)
    la = launch.get(int(K['corr'][i]))
    if la is None:
        continue
    a0 = la[0]
    running = np.nonzero((st < a0) & (en > a0))[0]
    if len(running) == 0:
        continue
    drain = int(en[running].max())
    # later-launched kernels (by API time) on other streams that start right at the drain
    later = np.nonzero((st >= drain) & (st <= drain + 2_000_000) & (K['stream'] != K['stream'][i]))[0]
    later = [j for j in later if launch.get(int(K['corr'][j]), (0,))[0] > a0]
    if not later:
        continue
    held = (drain - a0) / 1e9
    rows.append(dict(t_s=round((a0 - t0) / 1e9, 3),
                     level=int(np.searchsorted(bounds, a0, side='right') - 1),
                     function=names[f][:90], launch_api_ms=round((la[1] - la[0]) / 1e6, 3),
                     running_at_launch=int(len(running)), held_s=round(held, 4),
                     released_together=len(later)))
total = sum(r['held_s'] for r in rows)
by_level = collections.defaultdict(float)
for r in rows:
    by_level[r['level']] += r['held_s']
out = dict(first_use_stalls=len(rows), held_sum_s=round(total, 3),
           held_by_level_s={str(k): round(v, 3) for k, v in sorted(by_level.items())},
           top=sorted(rows, key=lambda r: -r['held_s'])[:15])
(run / 'first-use.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
