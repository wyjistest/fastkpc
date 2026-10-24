"""Audited host-only change, derived from the qualified L5 full build."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
snapshot = out / 'build-source'
assert not snapshot.exists(), 'preserve build attempts'
prior = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_slot_runtime_20261023'
old = json.loads((prior / 'build.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
native = root / 'fastkpc/build/fastkpc_cuda.so'
assert old['status'] == 'PASS' and sha(native) == old['native_sha256']
sources = [p for p in (root / 'fastkpc/src').rglob('*') if p.is_file() and
           p.suffix in ('.cpp', '.c', '.h', '.hpp', '.cu', '.cuh', '.inc', '.txt')]
sources += [root / 'fastkpc/tools/build_cuda_native.sh',
            root / 'fastkpc/tools/generate_recorded_dbdsqr.py']
hashes = {str(p.relative_to(root)): sha(p) for p in sorted(sources)}
assert hashes.keys() == old['source_sha256'].keys()
changed = [p for p, h in hashes.items() if h != old['source_sha256'][p]]
assert changed == ['fastkpc/src/full_cuda_ci_one_call.cpp'], changed
for source in sources:
    target = snapshot / source.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
build = snapshot / 'fastkpc/build'
build.mkdir()
prior_build = prior / 'build-source/fastkpc/build'
for name, expected in old['objects'].items():
    assert sha(prior_build / name) == expected
    shutil.copyfile(prior_build / name, build / name)
shutil.copyfile(prior_build / 'recorded_dbdsqr.inc', build / 'recorded_dbdsqr.inc')
assert sha(build / 'recorded_dbdsqr.inc') == old['generator_output_sha256']
with native.open('rb') as source, (out / 'previous-native.so.gz').open('xb') as dest:
    with gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as compressed:
        shutil.copyfileobj(source, compressed)
env = os.environ.copy()
for name in list(env):
    if name.startswith('FASTKPC_INCREMENTAL_') or name == 'FASTKPC_MULTI_PENALTY_BUCKETS':
        del env[name]
env['FASTKPC_INCREMENTAL_ONE_CALL_ONLY'] = '1'
started = time.time()
with (out / 'build.log').open('x') as log:
    subprocess.run(['sh', '-x', str(snapshot / 'fastkpc/tools/build_cuda_native.sh')],
                   cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
assert all(sha(root / p) == h and sha(snapshot / p) == h for p, h in hashes.items())
objects = {p.name: sha(p) for p in sorted(build.glob('*.o'))}
assert objects.keys() == old['objects'].keys()
rebuilt = ['full_cuda_ci_one_call.o', 'full_cuda_ci_structured_eig.o']
assert all(h == old['objects'][name] for name, h in objects.items() if name not in rebuilt)
assert sha(build / 'recorded_dbdsqr.inc') == old['generator_output_sha256']
for name in [*rebuilt, 'recorded_dbdsqr.inc', 'fastkpc_cuda.so']:
    shutil.copyfile(build / name, root / 'fastkpc/build' / name)
# The retained installed objects must have the same identity as the link inputs.
assert all(sha(root / 'fastkpc/build' / name) == h for name, h in objects.items())
receipt = dict(schema='optimizer-l4-incremental-source-build-v1', status='PASS',
    source_build='verified L5 full-build objects; rebuild one-call and structured eig; relink',
    evidence_class='audited incremental uncommitted source closure; not full rebuild or clean producer',
    parent_build=str((prior / 'build.json').relative_to(root)), parent_build_sha256=sha(prior / 'build.json'),
    previous_native_sha256=old['native_sha256'], native_sha256=sha(native),
    source_sha256=hashes, changed_sources=changed, objects=objects,
    rebuilt_objects=rebuilt, retained_objects_exact=True,
    generator_output_sha256=sha(build / 'recorded_dbdsqr.inc'),
    elapsed_sec=time.time()-started,
    producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
(out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS audited source-closure build', receipt['native_sha256'], flush=True)
