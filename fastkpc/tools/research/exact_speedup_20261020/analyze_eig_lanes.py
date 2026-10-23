"""Attribute steady-state eig lane profiles using exact NVTX API ranges.

Timing from a profiled run is diagnostic, not a performance receipt. Device
kernel concurrency is not SM utilization; a call span includes device gaps.
"""
import argparse
import bisect
import collections
import json
import sqlite3
import statistics


def union_length(intervals):
    total, lo, hi = 0, 0, 0
    for start,end in sorted(intervals):
        if start > hi:
            total += hi-lo
            lo,hi=start,end
        else:
            hi=max(hi,end)
    return total+hi-lo


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--output")
    args=parser.parse_args()
    c=sqlite3.connect(args.sqlite)
    names=dict(c.execute("select id,value from StringIds"))
    ranges=[(s,e,text or names.get(tid,""),thread) for s,e,text,tid,thread in c.execute(
        "select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null")]
    runtime=[(s,e,names[nid],tid,cid) for s,e,nid,tid,cid in c.execute(
        "select start,end,nameId,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")]
    kernels=[(s,e,names[nid],grid,stream,cid) for s,e,nid,grid,stream,cid in c.execute(
        "select start,end,shortName,gridX*gridY*gridZ,streamId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL")]
    calls=[r for r in ranges if r[2]=="cusolverDnDsyevd"]
    per_thread=collections.defaultdict(list)
    for start,end,_,tid in calls: per_thread[tid].append((start,end))
    for v in per_thread.values(): v.sort()

    def inside_solver(s,e,tid):
        v=per_thread[tid]
        i=bisect.bisect_right(v,(s,float("inf")))-1
        return i>=0 and v[i][0]<=s and e<=v[i][1]

    allocations=collections.defaultdict(lambda: [0,0,0.0,0.0])
    for s,e,name,tid,_ in runtime:
        if name.startswith(("cudaMalloc","cudaFree")):
            inside=inside_solver(s,e,tid)
            allocations[name][0 if inside else 1]+=1
            allocations[name][2 if inside else 3]+=(e-s)/1e6
    reports=[]
    for label in ("eig.serial","eig.lanes","eig.threaded"):
        outer=[r for r in ranges if r[2]==label]
        if not outer: continue
        span=sum(e-s for s,e,_,_ in outer)
        ks=[r for r in kernels if any(r[0]>=s and r[1]<=e for s,e,_,_ in outer)]
        apis=[r for r in runtime if any(r[0]>=s and r[1]<=e for s,e,_,_ in outer)]
        cs=[r for r in calls if any(r[0]>=s and r[1]<=e for s,e,_,_ in outer)]
        by_api=collections.defaultdict(list)
        by_kernel=collections.defaultdict(list)
        for r in apis: by_api[r[2]].append(r)
        for r in ks: by_kernel[r[2]].append(r)
        busy=union_length([(r[0],r[1]) for r in ks])
        events=[]
        for r in ks: events.extend([(r[0],1),(r[1],-1)])
        active=peak=0
        for _,delta in sorted(events): active+=delta;peak=max(peak,active)
        report=dict(label=label,trials=len(outer),range_ms=span/1e6,
                    syevd_calls=len(cs),syevd_host_ms_sum=sum(e-s for s,e,_,_ in cs)/1e6,
                    syevd_host_ms_median=statistics.median((e-s)/1e6 for s,e,_,_ in cs),
                    gpu_busy_fraction=busy/span,kernel_concurrency=sum(r[1]-r[0] for r in ks)/busy,
                    peak_simultaneous_kernels=peak,streams=len(set(r[4] for r in ks)),
                    apis={},kernels={})
        for name,rows in sorted(by_api.items(),key=lambda kv:-sum(r[1]-r[0] for r in kv[1]))[:8]:
            report['apis'][name]=dict(count=len(rows),sum_ms=sum(r[1]-r[0] for r in rows)/1e6,
                union_ms=union_length([(r[0],r[1]) for r in rows])/1e6)
        for name,rows in sorted(by_kernel.items(),key=lambda kv:-sum(r[1]-r[0] for r in kv[1]))[:8]:
            union=union_length([(r[0],r[1]) for r in rows])
            report['kernels'][name]=dict(count=len(rows),sum_ms=sum(r[1]-r[0] for r in rows)/1e6,
                union_ms=union/1e6,concurrency=sum(r[1]-r[0] for r in rows)/union,
                average_grid=statistics.mean(r[3] for r in rows))
        reports.append(report)
    output=dict(schema="eig-lane-range-attribution-v1",profile=args.sqlite,
                total_syevd_calls=len(calls),allocation_columns=["inside_solver_count","outside_solver_count",
                    "inside_solver_ms","outside_solver_ms"],allocations=dict(allocations),ranges=reports)
    print(json.dumps(output,indent=2))
    if args.output:
        with open(args.output,"x") as f: json.dump(output,f,indent=2)


if __name__=="__main__": main()
