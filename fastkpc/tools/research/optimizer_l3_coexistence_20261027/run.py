"""Run one bound W/S canonical diagnostic on exclusive GPU1."""
from pathlib import Path
import sys
TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,PARENT,load,sha,write_new,check_identity,environment,monitored
out=Path(sys.argv[1]).resolve();mode=sys.argv[2];assert mode in ('W','S')
capture=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027'
check_identity(capture);b=load(out/'build.json')
for p,h in {**b['tools'],**b['dependencies'],**b['source_sha256']}.items():assert sha(ROOT/p)==h,p
for p,h in b['outputs'].items():assert sha(out/p)==h,p
assert not (out/mode).exists() and not (out/(mode+'-process.json')).exists(),'preserve attempt'
reference=out/'W/stage-a.bin'
if mode=='S':
    assert load(out/'W/probe.json')['status']=='PASS'
    assert all(load(out/'W/report.json')['gates'].values())
    assert load(out/'W-process.json')['isolated']
    reference_sha=sha(reference)
else:assert not reference.exists();reference_sha=None
env=environment()
env.update(LD_PRELOAD=str(out/'probe.so')+':'+str(capture/'audit.so'),
    FASTKPC_L3_PROBE_LIBRARY=str(out/'probe.so'),FASTKPC_L3_PROBE_OUT=str(out/mode),
    FASTKPC_DIAG_L3_REPLACEMENT='1' if mode=='S' else '0',
    FASTKPC_L3_PROBE_FIXTURES=str(capture/'capture-raw'),FASTKPC_L3_STAGE_A_REFERENCE=str(reference),
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
canonical_reference=PARENT/'on3/receipt.rds'
monitored(out,mode,['Rscript','--vanilla',str(out/'run.R'),str(out/mode),'1',str(canonical_reference),str(canonical_reference)],env,
    dict(evidence_class='L3 actual replacement coexistence diagnostic; no performance claim',
         diagnostic_overlay_sha256=sha(out/'probe.so'),protocol_sha256=sha(out/'protocol.json'),
         execution_tool_sha256=sha(Path(__file__)),stage_a_reference_before_sha256=reference_sha))
assert all(load(out/mode/'report.json')['gates'].values())
assert load(out/mode/'probe.json')['status']=='PASS'
if reference_sha:assert sha(reference)==reference_sha,'Stage A authority changed'
for p,h in b['outputs'].items():assert sha(out/p)==h,p
write_new(out/(mode+'-closure.json'),dict(status='PASS',native_sha256=sha(NATIVE),
    diagnostic_overlay_sha256=sha(out/'probe.so'),reference_sha256=sha(canonical_reference),
    stage_a_reference_sha256=sha(reference),probe=load(out/mode/'probe.json'),
    canonical_hard_gates=load(out/mode/'report.json')['gates'],
    report_sha256=sha(out/mode/'report.json'),process_sha256=sha(out/(mode+'-process.json'))))
print('PASS',mode,load(out/mode/'probe.json'),flush=True)
