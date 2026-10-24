"""Export measured per-window drain tails; no modeled speedup on this plot."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1]).resolve()
fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, sharey=True)
for run, ax in enumerate(axes, 1):
    rows = list(csv.DictReader((out / ('profile-' + str(run) + '-window-attribution.csv')).open()))
    windows = [int(r['window']) for r in rows]
    native = [float(r['native_sec']) for r in rows]
    tail = [float(r['last_setup_start_to_last_kernel_sec']) for r in rows]
    other = [n-t for n,t in zip(native,tail)]
    ax.bar(windows, other, color='#607d8b', label='Remaining native interval')
    ax.bar(windows, tail, bottom=other, color='#e5a33d', label='Drain: last setup starts to last kernel ends')
    ax.set_title(f'Trace {run}: 34 original L4 windows, 32 workers; native sum {sum(native):.3f} s', loc='left')
    ax.set_ylabel('Measured seconds')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=9, loc='upper right')
axes[-1].set_xticks(range(1,35))
axes[-1].set_xlabel('Original window (cohorts and order unchanged)')
fig.text(.5,.01,'Drain time contains useful GPU work; it is not a measured saving. Shadow checks occur between windows.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,1))
for suffix in ['png','svg']:
    path=out/('window-tails.'+suffix)
    assert not path.exists()
    fig.savefig(path,dpi=150)
plt.close(fig)
print('PASS measured-tail figure')
