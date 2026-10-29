"""Close the diagnostic without rewriting previous evidence or production."""
from pathlib import Path
import sys
from build import ROOT, TOOL, identity, read, sha, write

out = Path(sys.argv[1]).resolve()
b = read(out/'build.json')
assert identity()['native_sha256'] == b['native_sha256']
for group in ['tools', 'dependencies']:
    for name, expected in b[group].items():
        assert sha(ROOT/name) == expected, name
for name, expected in b['outputs'].items():
    assert sha(out/name) == expected, name
c = read(out/'campaign.json')
a = read(out/'analysis.json')
assert c['status'] == 'PASS_CURRENT_NATIVE_WORK_PAIR_ACTIVITY'
for name, expected in a['input_sha256'].items():
    assert sha(out/name) == expected, name
assert a['analysis_tool_sha256'] == sha(TOOL/'analyze.py')
for label in ['off', 'on']:
    p = read(out/f'{label}-process.json')
    assert p['isolated'] and p['exit_code'] == 0
    assert c['runs'][label]['mode']['pass']
write(out/'closure.json', dict(status='AUTHENTICATED_CURRENT_WORK_PAIR_ACTIVITY_COMPLETE',
    build_sha256=sha(out/'build.json'), protocol_sha256=sha(out/'protocol.json'),
    campaign_sha256=sha(out/'campaign.json'), analysis_sha256=sha(out/'analysis.json'),
    native_sha256=b['native_sha256'], production_inputs=len(b['source_sha256']),
    full_level_replays=4, component_hash_comparisons=c['component_hash_comparisons'],
    hardware_occupancy_measured=False, performance_claim=False, prototype_qualified=False,
    accepted_opt_in_development_median_sec=311.440, default_enabled=False,
    phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False,
    l8_optimizer='STOP_UNCHANGED', evidence_class='isolated current-native instrumented captured Stage B diagnostic'))
write(out/'manifest.json', dict(schema='current-work-pair-activity-manifest-v1',
    native_sha256=b['native_sha256'], source_sha256=b['source_sha256'], tools=b['tools'],
    dependencies=b['dependencies'],
    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
print('PASS diagnostic closure', sha(out/'closure.json'), 'manifest', sha(out/'manifest.json'))
