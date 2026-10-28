"""Close an already completed isolated campaign, without running numerical work."""
import json
from pathlib import Path
import sys
from build import ROOT, TOOL, ACTIVITY, NATIVE, identity, sha, write

out=Path(sys.argv[1]).resolve()
b=json.loads((out/'build.json').read_text())
p=json.loads((out/'protocol.json').read_text())
c=json.loads((out/'campaign.json').read_text())
q=json.loads((out/'qualification.json').read_text())
assert c['status']=='PASS_COMPLETED_ISOLATED_CAMPAIGN'
assert q['status']=='PASS_GPU_BITWISE_QUALIFICATION' and q['captured']['targets']==104797
assert not q['captured']['mismatches'] and not q['synthetic']['mismatches']
assert q['live_device_allocations']==q['live_device_bytes']==0
identity()
assert all(sha(ROOT/s)==h for s,h in b['tools'].items())
assert all(sha(out/s)==h for s,h in b['outputs'].items())
assert all(sha(s)==h for s,h in p['dependencies'].items())
processes=[json.loads(x.read_text()) for x in sorted(out.glob('*-process.json'))]
assert len(processes)==8 and all(r['isolated'] and r['exit_code']==0 for r in processes)
assert all(sha(out/r['name']/f'l{l}-payload-sha256.txt')==p['dependencies'][str(ACTIVITY/'attempt3/capture'/f'l{l}-payload-sha256.txt')]
           for r in c['runs'] for l in [2,3])
resources=json.loads((out/'resources.json').read_text())
write(out/'closure.json',dict(status='PASS_ISOLATED_WORK_PAIR_CLOSURE',decision=c['decision'],
    native_sha256=sha(NATIVE),production_inputs_verified=len(b['source_sha256']),
    generated_dbdsqr_unchanged_sha256=sha(ROOT/'fastkpc/build/recorded_dbdsqr.inc'),
    build_sha256=sha(out/'build.json'),protocol_sha256=sha(out/'protocol.json'),
    qualification_sha256=sha(out/'qualification.json'),campaign_sha256=sha(out/'campaign.json'),
    captured_intermediate_targets=q['captured']['targets'],synthetic_cases=q['synthetic']['targets'],
    native_payload_hash_comparisons=6*104797,resources=resources,medians=c['medians'],
    combined_fraction_reduction=c['combined_fraction_reduction'],
    production_changes=False,full_method_validation='NOT_RUN',actual_sm_occupancy='NOT_MEASURED',
    accepted_full_method_median_unchanged=329.078,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
    limitation='Captured Stage B excludes optimizer workspaces and CI consumers; no full-method speedup or capacity claim.'))
write(out/'manifest.json',dict(schema='fixed-vt-work-pair-manifest-v1',
    production_source_sha256=b['source_sha256'],dependencies=p['dependencies'],native_sha256=sha(NATIVE),
    tools={str(x.relative_to(ROOT)):sha(x) for x in sorted(TOOL.iterdir()) if x.is_file()},
    files={str(x.relative_to(out)):sha(x) for x in sorted(out.rglob('*')) if x.is_file()}))
print(json.dumps(dict(decision=c['decision'],closure_sha256=sha(out/'closure.json'),manifest_sha256=sha(out/'manifest.json')),indent=2))
