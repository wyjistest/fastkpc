"""Rebuild every properties caller; retain and verify all other parent objects."""
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
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior = root/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_slot_runtime_20261024'
old = json.loads((prior/'build.json').read_text())
native = root/'fastkpc/build/fastkpc_cuda.so'
assert old['status'] == 'PASS' and sha(native) == old['native_sha256']
snapshot = out/'build-source'
assert not snapshot.exists(), 'preserve earlier attempts'
sources = [p for p in (root/'fastkpc/src').rglob('*') if p.is_file() and p.suffix in
           ('.cpp', '.c', '.h', '.hpp', '.cu', '.cuh', '.inc', '.txt')]
sources += [root/'fastkpc/tools/build_cuda_native.sh', root/'fastkpc/tools/generate_recorded_dbdsqr.py']
hashes = {str(p.relative_to(root)):sha(p) for p in sorted(sources)}
allowed = {'fastkpc/src/cuda/'+p for p in ['cuda_status.cpp','cuda_device_properties.hpp',
    'dcov_batch_cuda.cu','mgcv_multi_penalty_gcv.cu','mgcv_fixed_sp_runtime.cu',
    'mgcv_single_penalty_gcv.cu']}
changed = {p for p,h in hashes.items() if old['source_sha256'].get(p) != h}
assert changed == allowed, (changed, allowed)
assert set(hashes) == set(old['source_sha256']) | {'fastkpc/src/cuda/cuda_device_properties.hpp'}
for p in sources:
    dest = snapshot/p.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
build = snapshot/'fastkpc/build'
build.mkdir()
parent_objects = prior/'build-source/fastkpc/build'
for name,h in old['objects'].items():
    assert sha(parent_objects/name) == h
    shutil.copyfile(parent_objects/name, build/name)
shutil.copyfile(parent_objects/'recorded_dbdsqr.inc', build/'recorded_dbdsqr.inc')
assert sha(build/'recorded_dbdsqr.inc') == old['generator_output_sha256']
with native.open('rb') as src, (out/'previous-native.so.gz').open('xb') as dest:
    with gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as zipped:
        shutil.copyfileobj(src, zipped)

# Extract unchanged commands/flags/link order from the production build recipe.
# Capacity templates instantiate the common source, so all five are rebuilt.
recipe = (snapshot/'fastkpc/tools/build_cuda_native.sh').read_text()
flat = recipe.replace('\\\n', '')
rebuilt = ['cuda_status.o','dcov_batch_cuda.o','mgcv_fixed_sp_runtime.o',
    'mgcv_single_penalty_gcv.o','mgcv_multi_penalty_gcv.o',
    'mgcv_multi_penalty_gcv_small64.o',*[f'mgcv_multi_penalty_gcv_ext{n}.o' for n in [192,384,559]]]
commands = []
for name in rebuilt:
    target = name if '_ext' not in name else 'mgcv_multi_penalty_gcv_ext${CAPACITY}.o'
    candidates = {' '.join(line.split()) for line in flat.splitlines() if '-c ' in line and
                  f'-o "$BUILD/{target}"' in line}
    assert len(candidates) == 1, (name, candidates)
    command = candidates.pop()
    if '_ext' in name:
        command = command.replace('${CAPACITY}', name.split('_ext')[1].split('.')[0])
    commands.append(command)
script = recipe.split('(\nflock 9')[0] + '(\nflock 9\n'
script += 'compile_pids=""\ntrap \'kill $compile_pids 2>/dev/null || true\' EXIT INT TERM\n'
for index, command in enumerate(commands):
    script += command + f' > "$BUILD/compile-{index}.log" 2>&1 &\n'
    script += 'compile_pids="$compile_pids $!"\n'
script += 'for compile_pid in $compile_pids; do wait "$compile_pid"; done\n'
script += 'compile_pids=""\n'
script += recipe[recipe.index('"$CXX" -shared'):]
generated = snapshot/'fastkpc/tools/build_device_properties.sh'
generated.write_text(script)
env = os.environ.copy()
for key in list(env):
    if key.startswith('FASTKPC_INCREMENTAL_') or key == 'FASTKPC_MULTI_PENALTY_BUCKETS':
        del env[key]
started = time.time()
with (out/'build.log').open('x') as log:
    subprocess.run(['sh','-x',str(generated)], cwd=snapshot, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
assert all(sha(root/p) == h and sha(snapshot/p) == h for p,h in hashes.items())
objects = {p.name:sha(p) for p in sorted(build.glob('*.o'))}
assert objects.keys() == old['objects'].keys()
assert all(h == old['objects'][n] for n,h in objects.items() if n not in rebuilt)
for name in [*rebuilt, 'fastkpc_cuda.so']:
    shutil.copyfile(build/name, root/'fastkpc/build'/name)
assert all(sha(root/'fastkpc/build'/name) == h for name,h in objects.items())
receipt = dict(schema='device-properties-cache-build-v1', status='PASS',
    source_build='nine rebuilt host-call-site objects; remaining parent objects verified exact',
    evidence_class='audited incremental source closure; not full rebuild or clean producer',
    parent_build=str((prior/'build.json').relative_to(root)), parent_build_sha256=sha(prior/'build.json'),
    previous_native_sha256=old['native_sha256'], native_sha256=sha(native),
    source_sha256=hashes, changed_sources=sorted(changed), objects=objects,
    rebuilt_objects=rebuilt, retained_objects_exact=True,
    generator_output_sha256=sha(build/'recorded_dbdsqr.inc'),
    build_commands=commands, generated_build_sha256=sha(generated), elapsed_sec=time.time()-started,
    producer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
with (out/'build.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print('PASS build',receipt['native_sha256'],flush=True)
