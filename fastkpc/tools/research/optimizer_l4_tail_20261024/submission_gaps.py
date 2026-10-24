"""Separate gaps outside recorded CUDA APIs; do not infer a mutex/driver cause."""
import bisect
import collections
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze import distribution, interval_stats
from common import TOOL, sha, write_new

out=Path(sys.argv[1]).resolve()
reports=[]
for run in (1,2):
    db=sqlite3.connect((out/('profile-'+str(run)+'.sqlite')).as_uri()+'?mode=ro',uri=True)
    names=dict(db.execute('select id,value from StringIds'))
    apis=collections.defaultdict(list)
    for s,e,tid,name in db.execute('select start,end,globalTid,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
        apis[tid].append((s,e,names[name]))
    for rows in apis.values():rows.sort()
    starts={tid:[r[0] for r in rows] for tid,rows in apis.items()}
    gaps=collections.defaultdict(list)
    entries=[]
    ranges=list(db.execute("select start,end,text,globalTid from NVTX_EVENTS where text like 'setup:%'"))
    assert len(ranges)==2154
    for s,e,text,tid in ranges:
        calls=[r for r in apis[tid][bisect.bisect_left(starts[tid],s):bisect.bisect_right(starts[tid],e)] if r[1]<=e]
        assert calls
        entries.append((s,calls[0][0]))
        previous=s;name='setup entry'
        for begin,end,n in calls:
            if begin>previous:gaps[(name,n)].append((previous,begin))
            if end>=previous:previous=end;name=n
        if e>previous:gaps[(name,'setup return')].append((previous,e))
    db.close()
    lo=min(r[0] for r in ranges);hi=max(r[1] for r in ranges)
    reports.append(dict(run=run,entry_to_first_recorded_cuda_api_worker_sec=sum(e-s for s,e in entries)/1e9,
        entry_gap_union_sec=interval_stats(entries,lo,hi)['union_sec'],
        entry_gap_seconds=distribution([(e-s)/1e9 for s,e in entries]),
        gap_pairs=[dict(previous=name[0],following=name[1],count=len(values),
            worker_sum_sec=sum(e-s for s,e in values)/1e9,
            union_sec=interval_stats(values,lo,hi)['union_sec'])
            for name,values in sorted(gaps.items(),key=lambda item:sum(e-s for s,e in item[1]),reverse=True)]))
write_new(out/'submission-gaps.json',dict(schema='l4-submission-gaps-v1',runs=reports,
    interpretation='Time outside recorded runtime APIs is not proven CPU execution or driver-lock time; omitted APIs, library work and descheduling can contribute. Worker sums and overlapping unions are not savings. Already retained in the complete-host-span model.',
    tool_sha256=sha(TOOL/'submission_gaps.py')))
for r in reports:print(r['run'],r['entry_to_first_recorded_cuda_api_worker_sec'],r['entry_gap_union_sec'],r['entry_gap_seconds'])
