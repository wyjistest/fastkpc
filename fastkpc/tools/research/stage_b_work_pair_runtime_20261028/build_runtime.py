"""Two-object source-closure build, gated on sealed deep GPU qualification."""
import gzip
import shutil
import sys
from support import *

out = Path(sys.argv[1]).resolve()
deep = Path(sys.argv[2]).resolve()
assert read(deep/'closure.json')['status'] == 'PASS_DEEP_WORK_PAIR_GPU_BITWISE'
check_map(read(deep/'manifest.json')['files'], deep)
old = read(PARENT/'build.json')
assert sha(NATIVE) == old['native_sha256']
out.mkdir(parents=True, exist_ok=False)
snapshot = out/'build-source'
sources = [p for p in (ROOT/'fastkpc/src').rglob('*') if p.is_file() and p.suffix in
           ('.cpp', '.c', '.h', '.hpp', '.cu', '.cuh', '.inc', '.txt')]
sources += [ROOT/'fastkpc/tools/build_cuda_native.sh', ROOT/'fastkpc/tools/generate_recorded_dbdsqr.py']
hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
allowed = {'fastkpc/src/cuda/full_cuda_ci_structured_eig.cu',
           'fastkpc/src/cuda/full_cuda_ci_structured_eig.hpp',
           'fastkpc/src/cuda/full_cuda_ci_structured_launch.hpp',
           'fastkpc/src/full_cuda_ci_one_call.cpp',
           'fastkpc/tools/build_cuda_native.sh', 'fastkpc/tools/generate_recorded_dbdsqr.py'}
changed = {p for p, h in hashes.items() if old['source_sha256'].get(p) != h}
assert changed == allowed, (changed, allowed)
assert set(hashes) == set(old['source_sha256']) | {'fastkpc/src/cuda/full_cuda_ci_structured_launch.hpp'}
for p in sources:
    dest = snapshot/p.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
build = snapshot/'fastkpc/build'
build.mkdir()
prior_build = PARENT/'build-source/fastkpc/build'
for name, h in old['objects'].items():
    assert sha(prior_build/name) == h
    assert sha(ROOT/'fastkpc/build'/name) == h
    shutil.copyfile(prior_build/name, build/name)
with NATIVE.open('rb') as src, (out/'previous-native.so.gz').open('xb') as dest:
    with gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as zipped:
        shutil.copyfileobj(src, zipped)
env = os.environ.copy()
for key in list(env):
    if key.startswith('FASTKPC_INCREMENTAL_') or key == 'FASTKPC_MULTI_PENALTY_BUCKETS':
        del env[key]
env['FASTKPC_INCREMENTAL_ONE_CALL_ONLY'] = '1'
started = time.time()
with (out/'build.log').open('x') as log:
    subprocess.run(['sh', '-x', str(snapshot/'fastkpc/tools/build_cuda_native.sh')],
                   cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
check_map(hashes)
check_map(hashes, snapshot)
objects = {p.name: sha(p) for p in sorted(build.glob('*.o'))}
rebuilt = ['full_cuda_ci_one_call.o', 'full_cuda_ci_structured_eig.o']
assert objects.keys() == old['objects'].keys()
assert all(h == old['objects'][name] for name, h in objects.items() if name not in rebuilt)
assert sha(build/'recorded_dbdsqr.inc') == old['generator_output_sha256']
candidate = (build/'recorded_dbdsqr_fixed_vt.inc').read_text()
expected = (SEALED/'candidate_timing/recorded_dbdsqr.inc').read_text()
assert candidate.replace('recorded_dbdsqr_fixed_vt_(', 'recorded_dbdsqr_(', 1) == expected
for name in [*rebuilt, 'recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc', 'fastkpc_cuda.so']:
    shutil.copyfile(build/name, ROOT/'fastkpc/build'/name)
assert all(sha(ROOT/'fastkpc/build'/name) == h for name, h in objects.items())
write(out/'build.json', dict(schema='work-pair-runtime-source-closure-v1', status='PASS',
    source_build='verified parent objects; structured and one-call objects rebuilt and relinked',
    evidence_class='audited incremental source closure; not clean producer or full rebuild',
    parent_build_sha256=sha(PARENT/'build.json'), previous_native_sha256=old['native_sha256'],
    native_sha256=sha(NATIVE), source_sha256=hashes, changed_sources=sorted(changed),
    objects=objects, rebuilt_objects=rebuilt, retained_objects_exact=True,
    original_generator_output_sha256=sha(build/'recorded_dbdsqr.inc'),
    fixed_vt_generator_output_sha256=sha(build/'recorded_dbdsqr_fixed_vt.inc'),
    fixed_vt_matches_qualified_candidate_except_name=True, deep_closure_sha256=sha(deep/'closure.json'),
    elapsed_sec=time.time()-started,
    producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
print('PASS work-pair runtime source-closure', sha(NATIVE), flush=True)
