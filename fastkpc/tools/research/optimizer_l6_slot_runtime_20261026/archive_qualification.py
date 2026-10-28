"""Archive the completed gates while the separate isolated campaign is pending.

This post-run inventory does not amend the frozen runner or acceptance gates.
"""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads((out/p).read_text())
destination = out/'qualification-manifest.json'
assert not destination.exists(), 'preserve the previous qualification archive'
closure = read('shared-qualification-closure.json')
assert closure['status'] == 'PASS_L6_RUNTIME_SHARED_HOST_HARD_GATES'
assert closure['performance_status'] == 'PENDING_ISOLATED_CAMPAIGN'
for field, name in [('build_sha256','build.json'), ('checks_sha256','checks.json'),
                    ('protocol_sha256','shared-qualification-protocol.json')]:
    assert closure[field] == sha(out/name)
for p, h in closure['files'].items():
    assert sha(out/p) == h
for mapping in [closure['source_sha256'], closure['tools'], closure['references']]:
    assert all(sha(root/p) == h for p,h in mapping.items())
build, checks = read('build.json'), read('checks.json')
assert build['native_sha256'] == checks['native_sha256'] == closure['native_sha256']
assert sha(root/'fastkpc/build/fastkpc_cuda.so') == build['native_sha256']
assert sha(out/'checks/device-code.json') == checks['device_code_sha256']
assert sha(out/'coexistence-replacement/closure.json') == build['coexistence_closure_sha256']
assert sha(root/build['parent_build']) == build['parent_build_sha256']
memcheck = (out/'checks/memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in memcheck and '0 bytes leaked' in memcheck
manifest = dict(schema='optimizer-l6-runtime-qualification-manifest-v1',
    status='PASS_L6_RUNTIME_QUALIFICATION_PERFORMANCE_PENDING',
    evidence_class='incremental source closure; shared-host full canonical hard gates, no performance acceptance',
    native_sha256=build['native_sha256'],
    source_sha256=build['source_sha256'], frozen_tools=closure['tools'],
    archive_tool_sha256=sha(__file__), references=closure['references'],
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False)
with destination.open('x') as stream:
    json.dump(manifest, stream, indent=2); stream.write('\n')
print(manifest['status'])
print('qualification manifest SHA256', sha(destination))
