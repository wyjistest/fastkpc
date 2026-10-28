"""Capture and qualify all missing L4–L8 components before production changes."""
import sys
from support import *

out = Path(sys.argv[1]).resolve()
b = read(out/'build.json')
def frozen():
    original_identity()
    check_map(b['tools'])
    check_map(b['dependencies'])
    check_map(b['outputs'], out)

env = environment()
frozen()
write(out/'protocol.json', dict(schema='work-pair-deep-qualification-v1', declared_epoch=time.time(),
    native_sha256=sha(NATIVE), build_sha256=sha(out/'build.json'), levels=COUNTS,
    sequence=['capture', 'qualify'], physical_gpu=1, other_gpu_idle=True,
    diagnostic_payload_d2h=True, performance_evidence=False,
    inherited_synthetic_evidence_sha256=sha(SEALED/'qualification.json')))
dest = out/'capture'
dest.mkdir()
env.update(STAGE_B_NATIVE=str(NATIVE), STAGE_B_MODE='capture', STAGE_B_OUTPUT=str(dest), LD_PRELOAD=str(out/'observe.so'))
job(out, 'capture', ['sh', 'fastkpc/tools/with_phase10_exact_development_env.sh', 'Rscript', '--vanilla',
    str(out/'capture_runner.R'), str(dest), '1', str(PARENT/'on3/receipt.rds'), str(PARENT/'on3/receipt.rds')], env, frozen)
report = read(dest/'report.json')
assert all(report['gates'].values())
for level, count in COUNTS.items():
    r = read(dest/f'l{level}-observation.json')
    assert r['components'] == r['three_field_hashes'] == count
    assert r['contexts'] == min(10, (count+63)//64) and r['dense_fallback'] == 0
    assert (dest/f'l{level}-residuals.f64').stat().st_size == count*351*8
    assert len((dest/f'l{level}-payload-sha256.txt').read_text().splitlines()) == count
capture_hashes = {str(p.relative_to(out)): sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
write(out/'capture-identity.json', dict(native_sha256=sha(NATIVE), files=capture_hashes))
env.pop('LD_PRELOAD')
job(out, 'qualify', [str(out/'qualify'), str(SEALED/'baseline_diag.so'), str(SEALED/'candidate_diag.so'),
    str(dest), str(out/'qualification.json')], env, frozen)
check_map(capture_hashes, out)
q = read(out/'qualification.json')
assert q['captured']['targets'] == sum(COUNTS.values()) and q['captured']['mismatches'] == 0
assert q['live_device_allocations'] == q['live_device_bytes'] == 0
write(out/'closure.json', dict(status='PASS_DEEP_WORK_PAIR_GPU_BITWISE', native_sha256=sha(NATIVE),
    production_inputs=len(b['source_sha256']), qualification=q, canonical_gates=report['gates'],
    capture_identity_sha256=sha(out/'capture-identity.json'), production_changes=False, performance_evidence=False))
write(out/'manifest.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    dependencies=b['dependencies'], tools=b['tools'], source_sha256=b['source_sha256']))
print('PASS deep qualification', q['captured'], flush=True)
