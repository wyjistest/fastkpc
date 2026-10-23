"""Show actual optimizer intervals by original setup identity, not occupancy."""
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]).resolve()
fig,axes=plt.subplots(2,3,figsize=(15,10),sharey=True)
for row,name in enumerate(('L3-W31+W32','L5-W7+W8')):
    for col,variant in enumerate('ABC'):
        with (out/f'profile-{name}-{variant}-timeline.csv').open() as f:
            data=list(csv.DictReader(f))
        ax=axes[row,col]
        for r in data:
            index=int(r['setup']);start=float(r['optimizer_start_ms'])/1000
            end=float(r['optimizer_end_ms'])/1000;launch=float(r['launch_return_ms'])/1000
            color='#d9782f' if index<64 else '#427db6'
            ax.plot([launch,start],[index,index],color='#aaaaaa',linewidth=.6)
            ax.broken_barh([(start,end-start)],(index-.4,.8),facecolors=color)
        ax.axhline(63.5,color='#777777',linestyle='--',linewidth=.7)
        ax.set(title=name+' / '+variant,xlabel='Seconds from pair entry',ylim=(-1,128))
        ax.grid(axis='x',alpha=.2)
    axes[row,0].set_ylabel('Setup in original window order')
fig.suptitle('Adjacent optimizer windows: A sequential / B 64 workers / C 32 workers')
fig.text(.5,.015,'Orange: first window   Blue: second window   Grey: launch-return to optimizer start (includes prior same-stream work)',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.035,1,.95))
fig.savefig(out/'double-window-timeline.png',dpi=160)
