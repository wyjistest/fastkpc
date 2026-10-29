"""Seal the isolated experiment, leaving parent artifacts untouched."""
import sys
from pathlib import Path
from build import ROOT, TOOL, identity, read, sha, write

out=Path(sys.argv[1]).resolve()
b=read(out/'build.json')
assert identity()['native_sha256']==b['native_sha256']
for group in ['tools','dependencies']:
    for name,expected in b[group].items():assert sha(ROOT/name)==expected,name
for name,expected in b['outputs'].items():assert sha(out/name)==expected,name
c=read(out/'campaign.json');a=read(out/'analysis.json');t=read(out/'timing.json')
assert c['status']=='PASS_ISOLATED_RECOVER_TILE_CAMPAIGN' and t['status']=='PASS_NINE_TIMINGS'
assert a['independent_endpoint_checks']==30 and a['campaign_sha256']==sha(out/'campaign.json')
assert a['analysis_tool_sha256']==sha(TOOL/'analyze.py')
for name,expected in a['input_sha256'].items():assert sha(out/name)==expected,name
processes=list(out.glob('*-process.json'))
assert len(processes)==18,len(processes)
for p in processes:
    r=read(p);assert r['exit_code']==0 and r['isolated'] and not r['unexpected'],str(p)
write(out/'closure.json',dict(status='ISOLATED_RECOVER_TILE_EXPERIMENT_COMPLETE',decisions=c['decisions'],
    build_sha256=sha(out/'build.json'),protocol_sha256=sha(out/'protocol.json'),campaign_sha256=sha(out/'campaign.json'),
    timing_sha256=sha(out/'timing.json'),analysis_sha256=sha(out/'analysis.json'),native_sha256=b['native_sha256'],
    captured_comparisons=2*104797,component_hash_comparisons=c['component_hash_comparisons'],
    independent_interval_checks=30,full_replays=13,production_changes=False,
    runtime_integration_qualified=False,hardware_occupancy_measured=False,memory_traffic_measured=False,
    accepted_opt_in_development_median_sec=311.440,default_enabled=False,phase10='ACTIVE',candidate='NOT_FROZEN',
    holdout='SEALED',recommended=False,l8_optimizer='STOP_UNCHANGED',
    evidence_class='isolated captured L2/L3 prototype; not full-runtime or promotion evidence'))
write(out/'manifest.json',dict(schema='isolated-recover-tile-manifest-v1',native_sha256=b['native_sha256'],
    source_sha256=b['source_sha256'],tools=b['tools'],dependencies=b['dependencies'],
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
print('PASS closure',sha(out/'closure.json'),'manifest',sha(out/'manifest.json'),flush=True)
