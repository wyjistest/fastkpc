"""Archive independent review calculations without changing prior receipts."""
import collections
import csv
import hashlib
import heapq
import json
from pathlib import Path
import statistics
import sys
root=Path(__file__).resolve().parents[4];out=Path(sys.argv[1]).resolve()
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=list(csv.DictReader((prior/'profile-kernel-timeline.csv').open()))
by={v:sorted([r for r in rows if r['variant']==v],key=lambda r:int(r['setup'])) for v in 'AR'}
svc={v:[float(r['last_kernel_sec'])-float(r['first_kernel_sec']) for r in by[v]] for v in 'AR'}
def rolling(values):
    free=[0.0]*32
    for d in values:heapq.heappush(free,heapq.heappop(free)+d)
    return max(free)
ratios=sorted(r/a for r,a in zip(svc['R'],svc['A']))
index=.95*(len(ratios)-1);lo=int(index)
memory=list(csv.DictReader((prior/'capacity-setups.csv').open()))
counts=collections.Counter(int(r['targets']) for r in memory)
campaign=list(csv.DictReader((prior/'campaign-runs.csv').open()))
p=[r for r in campaign if r['variant']=='P' and r['kind']=='timing']
report=dict(schema='slot-pool-review-corrections-v1',
 A_service_model_sec=rolling(svc['A']),R_service_model_sec=rolling(svc['R']),
 measured_R_kernel_span_sec=max(float(r['last_kernel_sec']) for r in by['R'])-min(float(r['first_kernel_sec']) for r in by['R']),
 median_R_over_A=statistics.median(ratios),p95_R_over_A=ratios[lo]+(index-lo)*(ratios[lo+1]-ratios[lo]),
 maximum_R_over_A=max(ratios),setups_above_1_10=sum(r>1.1 for r in ratios),
 service_sums_sec={v:sum(svc[v]) for v in 'AR'},target_capacity_counts=dict(counts),
 P_prepare_plus_destroy_median_sec=statistics.median((float(r['prepare_ms'])+float(r['destroy_ms']))/1000 for r in p),
 model_scope='descriptive L5 cross-configuration check; no cross-layer prediction qualification',
 inputs={str(p.relative_to(root)):sha(p) for p in [prior/'profile-kernel-timeline.csv',prior/'capacity-setups.csv',prior/'campaign-runs.csv']})
(out/'review-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='inputs'},indent=2))
