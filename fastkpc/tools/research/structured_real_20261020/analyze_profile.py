"""Attribute kernel intervals, not SM occupancy or removable host wait."""
import json
from pathlib import Path
import sqlite3
import sys

work=Path(sys.argv[1])
db=sqlite3.connect(work/"real640.sqlite")
db.row_factory=sqlite3.Row
kernels=[dict(r) for r in db.execute("""select k.*,s.value name from
 CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName""")]
ranges=list(db.execute("select start,end,text from NVTX_EVENTS where text like 'structured.real640.trial%'"))
assert len(ranges)==3

def intervals(rows):
    events={}
    for r in rows:
        events[r["start"]]=events.get(r["start"],0)+1
        events[r["end"]]=events.get(r["end"],0)-1
    active=peak=union=multiple=weighted=0
    last=0
    for t,delta in sorted(events.items()):
        dt=t-last
        if active:
            union+=dt
            weighted+=dt*active
        if active>1:
            multiple+=dt
        active+=delta
        peak=max(peak,active)
        last=t
    return dict(union_ms=union/1e6,multiple_kernels_ms=multiple/1e6,
                summed_kernel_ms=weighted/1e6,peak_concurrent_kernels=peak,
                average_concurrent_kernels_when_busy=weighted/union if union else 0)

trials=[]
for start,end,name in ranges:
    rows=[r for r in kernels if start<=r["start"] and r["end"]<=end]
    groups={}
    for stage in sorted(set(r["name"] for r in rows)):
        subset=[r for r in rows if r["name"]==stage]
        groups[stage]=dict(count=len(subset),**intervals(subset),
            grid_x=sorted(set(r["gridX"] for r in subset)),
            block_x=sorted(set(r["blockX"] for r in subset)),
            registers_per_thread=sorted(set(r["registersPerThread"] for r in subset)),
            shared_bytes=sorted(set(r["staticSharedMemory"]+r["dynamicSharedMemory"] for r in subset)),
            local_bytes_per_thread=sorted(set(r["localMemoryPerThread"] for r in subset)))
    first=min(r["start"] for r in rows)
    last=max(r["end"] for r in rows)
    trials.append(dict(name=name,wall_ms=(end-start)/1e6,kernel_count=len(rows),
        before_first_kernel_ms=(first-start)/1e6,after_last_kernel_ms=(end-last)/1e6,
        kernel_interval=intervals(rows),stages=groups))
report=dict(schema="structured-real640-nsys-attribution-v1",trials=trials,
    measured_sm_occupancy=False,
    caveat="Kernel intervals may overlap. Neither summed duration nor launch concurrency is SM occupancy; profiled times are not performance results.")
(work/"profile-analysis.json").write_text(json.dumps(report,indent=2))
for trial in trials:
    print(trial["name"],"wall ms",round(trial["wall_ms"],2),
          "busy ms",round(trial["kernel_interval"]["union_ms"],2),
          "peak kernels",trial["kernel_interval"]["peak_concurrent_kernels"],
          "tail ms",round(trial["after_last_kernel_ms"],2))
    for name,stage in trial["stages"].items():
        print(" ",name,"union ms",round(stage["union_ms"],2))
