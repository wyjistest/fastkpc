"""Actual 64-worker optimizer launches and device intervals, not occupancy."""
from pathlib import Path
import sqlite3
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(sys.argv[1]).resolve()
fig,axes=plt.subplots(1,2,figsize=(13,8),sharey=True)
for ax,name in zip(axes,('L3-W31','L5-W7')):
    db=sqlite3.connect(out/('profile-'+name+'.sqlite'))
    strings=dict(db.execute('select id,value from StringIds'))
    cols=[r[1] for r in db.execute('pragma table_info(NVTX_EVENTS)')]
    ranges=[dict(zip(cols,r)) for r in db.execute('select * from NVTX_EVENTS')]
    for r in ranges:r['label']=r.get('text') or strings.get(r.get('textId'),'')
    w=next(r for r in ranges if r['label'].startswith('window:') and 'workers64:' in r['label'])
    setups=[r for r in ranges if r['label'].startswith('setup:') and r['start']>=w['start'] and r['end']<=w['end']]
    apis={c:(s,e,t) for s,e,t,c in db.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    found=0
    for start,end,n,c in db.execute('select start,end,shortName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        if 'optimize_multi_penalty' not in strings[n] or not (w['start']<=start<end<=w['end']):continue
        hs,he,tid=apis[c]
        r=next(r for r in setups if r['globalTid']==tid and r['start']<=hs and he<=r['end'])
        index=int(r['label'].split(':')[1]);origin=w['start'];found+=1
        ax.plot([(he-origin)/1e9,(start-origin)/1e9],[index,index],color='#a3a3a3',linewidth=1)
        ax.broken_barh([((start-origin)/1e9,(end-start)/1e9)],(index-.35,.7),facecolors='#dd8532')
        ax.plot((hs-origin)/1e9,index,'|',color='#225ea8',markersize=4)
    assert found==64
    ax.set(title=name+' / 64 workers',xlabel='Seconds from window entry',ylim=(-1,64))
    ax.grid(axis='x',alpha=.2);db.close()
axes[0].set_ylabel('Setup in original window order')
fig.suptitle('Fixed optimizer windows: host launch, queue delay, device execution')
fig.text(.5,.02,'Blue: host launch   Grey: enqueue-to-start delay   Orange: optimizer kernel interval (not SM occupancy)',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'optimizer-timeline.png',dpi=160)
