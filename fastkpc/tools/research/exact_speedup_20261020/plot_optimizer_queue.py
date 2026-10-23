"""Plot one optimizer window: host submission versus actual device start.

The grey spans are queue delay, not GPU utilization or removable time. No
numerical computation participates in this offline trace visualization.
"""
import argparse
import collections
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument("sqlite");p.add_argument("output")
    args=p.parse_args()
    c=sqlite3.connect(args.sqlite)
    names=dict(c.execute("select id,value from StringIds"))
    name_id=next(k for k,v in names.items() if v=="optimize_multi_penalty_targets_kernel")
    kernels=c.execute("select start,end,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL "
                      "where shortName=? order by start limit 64",(name_id,)).fetchall()
    cids={v[3] for v in kernels}
    apis={cid:(s,e) for s,e,cid in c.execute(
        "select start,end,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME") if cid in cids}
    kernels.sort(key=lambda r:apis[r[3]][0])
    origin=min(apis[r[3]][0] for r in kernels)
    fig,ax=plt.subplots(figsize=(11,9))
    for row,(start,end,stream,cid) in enumerate(kernels):
        hs,he=apis[cid]
        ax.plot([(he-origin)/1e6,(start-origin)/1e6],[row,row],color="#a3a3a3",linewidth=1)
        ax.broken_barh([((start-origin)/1e6,(end-start)/1e6)],(row-.36,.72),facecolors="#dd8532")
        ax.plot((hs-origin)/1e6,row,"|",color="#225ea8",markersize=6)
    ax.set(xlabel="Milliseconds from first host optimizer launch",ylabel="Setup (host submission order)",
           title="Pinned optimizer: submitted work still waits in device queues")
    ax.text(.99,.01,"Blue: host launch   Grey: enqueue-to-start delay   Orange: kernel",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=9)
    ax.grid(axis="x",alpha=.2);fig.tight_layout();fig.savefig(args.output,dpi=150)


if __name__=="__main__":main()
