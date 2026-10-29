"""One smoke check then one full-gate, unchanged-native attribution run."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'stage_b_work_pair_runtime_20261028'))
from support import ROOT, NATIVE, sha, read, write, check_map, environment, processes, job, time

out=Path(sys.argv[1]).resolve();tool=Path(__file__).resolve().parent
b=read(out/'build.json')
env=environment()
tools={str(p.relative_to(ROOT)):sha(p) for p in tool.iterdir() if p.is_file()}
deps={str(out/p):h for p,h in b['outputs'].items()}
deps.update(b['capture_sha256'])
reference=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/on2/receipt.rds'
deps[str(reference)]=sha(reference)
def frozen():
    assert sha(NATIVE)==b['native_sha256']
    check_map(b['source_sha256']);check_map(tools);check_map(deps)
frozen();assert not processes(),'external compute; no job started'
write(out/'protocol.json',dict(schema='l45-existing-status-attribution-v1',declared_epoch=time.time(),
    native_sha256=b['native_sha256'],production_changes=False,device_changes=False,physical_gpu=1,
    preserved='canonical order/cohorts/32 slots and accepted opt-in settings',
    phases='existing per-target clock64 counters, not target lifetime or wall time',
    sequence=['smoke','canonical'],performance_claim=False,tools=tools,dependencies=deps,
    identity='sealed canonical keys/order, CUPTI external IDs; no inference from start order'))
base=env.copy()
base.update(LD_PRELOAD=str(out/'observer.so'),CUDA_INJECTION64_PATH=str(out/'trace.so'),
    FASTKPC_L45_NATIVE=str(NATIVE),FASTKPC_L45_REFERENCE=str(out/'reference.bin'),
    FASTKPC_L45_REFERENCE_TSV=str(out/'reference.tsv'),FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1')
for name in ['smoke','canonical']:
    dest=out/name;dest.mkdir(exist_ok=False)
    current=base.copy();current.update(FASTKPC_L45_OUTPUT=str(dest),KTRACE_DIR=str(dest))
    if name=='smoke':
        command=[str(out/'smoke-check'),str(ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023/raw'),
                 str(ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023/raw')]
    else:
        command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(out/'run.R'),str(dest),'1',str(reference),'-']
    print('START',name,time.strftime('%H:%M:%S'),flush=True)
    job(out,name,command,current,frozen)
    o=read(dest/'observer.json');t=read(dest/'trace.json')
    assert t['dropped']==t['invalid']==0
    assert o['passed_setups']==(4 if name=='smoke' else 3109)
    if name=='canonical':
        assert o['complete'] and o['targets']==24583 and o['canonical_request_order_exact']
        assert all(read(dest/'report.json')['gates'].values())
        assert (dest/'identity-order.txt').read_text().startswith('PASS')
print('PASS full-gate attribution campaign',flush=True)
