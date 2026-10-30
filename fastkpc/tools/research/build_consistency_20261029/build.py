"""Freeze the accepted parent, then exercise real full/incremental build entries.

Never installs into fastkpc/build. Both entries use the same frozen source path,
toolchain and output path; the full entry starts without any retained objects.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'fastkpc/artifacts/full_cuda_ci/build_consistency_20261029'
PRIOR = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029/attempt1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def linked_names(script):
    recipe = script.read_text().rsplit('"$CXX" -shared', 1)[1]
    return re.findall(r'"\$BUILD/([^/"\n]+\.o)"', recipe)

def source_map():
    paths = [p for p in (ROOT / 'fastkpc/src').rglob('*') if p.is_file() and
             p.suffix in ('.cpp', '.c', '.h', '.hpp', '.cu', '.cuh', '.inc', '.txt')]
    paths += [ROOT / 'fastkpc/tools/build_cuda_native.sh',
              ROOT / 'fastkpc/tools/generate_recorded_dbdsqr.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}

def freeze():
    dest = BASE / 'parent'
    dest.mkdir(parents=True)
    manifest = read(PRIOR / 'manifest.json')
    native = ROOT / 'fastkpc/build/fastkpc_cuda.so'
    assert sha(native) == manifest['native_sha256']
    for name, h in manifest['production_source_sha256'].items():
        assert sha(ROOT / name) == h, name
    sources = source_map()
    for name, h in sources.items():
        p = dest / 'source' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
        assert sha(p) == h
    names = linked_names(ROOT / 'fastkpc/tools/build_cuda_native.sh')
    previous = {Path(p).name: h for p, h in read(PRIOR / 'build.json')['objects'].items()}
    previous['mgcv_multi_penalty_gcv_postback.o'] = previous.pop('postback.o')
    assert set(names) == set(previous) and len(names) == 44
    (dest / 'objects').mkdir()
    for name in names:
        assert sha(ROOT / 'fastkpc/build' / name) == previous[name], name
        shutil.copyfile(ROOT / 'fastkpc/build' / name, dest / 'objects' / name)
    shutil.copyfile(native, dest / 'fastkpc_cuda.so')
    write(dest / 'identity.json', dict(status='PASS', source_sha256=sources,
        native_sha256=sha(native), objects=previous,
        parent_manifest_sha256=sha(PRIOR / 'manifest.json'),
        parent_closure_sha256=sha(PRIOR / 'closure.json'),
        producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    print('PASS parent frozen', sha(native), flush=True)

def build():
    parent = read(BASE / 'parent/identity.json')
    snapshot = BASE / 'source'
    assert not snapshot.exists()
    sources = source_map()
    for name in sources:
        p = snapshot / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    script = snapshot / 'fastkpc/tools/build_cuda_native.sh'
    names = linked_names(script)
    env = os.environ.copy()
    for k in list(env):
        if k.startswith('FASTKPC_INCREMENTAL_') or k == 'FASTKPC_MULTI_PENALTY_BUCKETS':
            del env[k]
    write(BASE / 'protocol.json', dict(schema='full-incremental-build-consistency-v1',
        epoch=time.time(), source_sha256=sources, parent_native_sha256=parent['native_sha256'],
        entries=['full', 'FASTKPC_INCREMENTAL_FIXED_SP_ONLY=1'],
        same_source_path=True, full_build_empty_directory=True,
        numerical_methods=['dcc.perm', 'hsic.gamma', 'hsic.perm'],
        identity_gates=['method TU command equal', 'method TU object equal',
                        'all linked objects equal', 'native equal'],
        semantic_gates=['p-values/pMax', 'task/deletion/graph/sepsets/counts',
                        'route/status', 'permutation RNG', 'zero fallback/resources'],
        installed_native_unchanged=True, promotion=False))
    build_dir = snapshot / 'fastkpc/build'
    assert not build_dir.exists()
    receipts = {}
    for mode in ('full', 'incremental'):
        dest = BASE / mode
        dest.mkdir()
        current = env.copy()
        if mode == 'incremental':
            current['FASTKPC_INCREMENTAL_FIXED_SP_ONLY'] = '1'
        command = ['sh', '-x', str(script)]
        print('START', mode, time.strftime('%H:%M:%S'), flush=True)
        started = time.time()
        with (dest / 'build.log').open('x') as log:
            subprocess.run(command, cwd=snapshot, env=current, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        commands = [line[2:] for line in (dest / 'build.log').read_text().splitlines()
                    if line.startswith('+ /usr/local/cuda/bin/nvcc ') and
                    '/full_cuda_ci_method_batch.cu ' in line]
        assert len(commands) == 1 and '--fmad=false' in commands[0]
        assert set(p.name for p in build_dir.glob('*.o')) == set(names)
        (dest / 'objects').mkdir()
        for name in names:
            shutil.copyfile(build_dir / name, dest / 'objects' / name)
        for name in ('fastkpc_cuda.so', 'recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc'):
            shutil.copyfile(build_dir / name, dest / name)
        receipt = dict(status='PASS', native_path=str(dest / 'fastkpc_cuda.so'),
            native_sha256=sha(dest / 'fastkpc_cuda.so'), source_sha256=sources,
            objects={name: sha(dest / 'objects' / name) for name in names},
            method_compile_command=commands[0], elapsed_sec=time.time()-started,
            build_log_sha256=sha(dest / 'build.log'),
            previous_native_sha256=parent['native_sha256'],
            evidence_class='full source-closure / incremental entry comparison; not clean committed producer')
        assert all(sha(ROOT / p) == h and sha(snapshot / p) == h for p, h in sources.items())
        assert sha(ROOT / 'fastkpc/build/fastkpc_cuda.so') == parent['native_sha256']
        write(dest / 'build.json', receipt)
        receipts[mode] = receipt
        print('PASS', mode, receipt['native_sha256'], receipt['elapsed_sec'], flush=True)
    a, b = receipts['full'], receipts['incremental']
    gates = dict(method_compile_command=a['method_compile_command'] == b['method_compile_command'],
        method_object=a['objects']['full_cuda_ci_method_batch.o'] == b['objects']['full_cuda_ci_method_batch.o'],
        all_objects=a['objects'] == b['objects'], native=a['native_sha256'] == b['native_sha256'])
    required = gates['method_compile_command'] and gates['native']
    write(BASE / 'build-comparison.json', dict(status='RAW_IDENTICAL' if all(gates.values()) else
        ('RAW_OBJECT_METADATA_REVIEW_REQUIRED' if required else 'FAIL'),
        gates=gates, changed_objects=[k for k in a['objects'] if a['objects'][k] != b['objects'][k]],
        full_sha256=a['native_sha256'], incremental_sha256=b['native_sha256']))
    assert required, gates
    print('PASS commands / native; compare_objects.py must also validate object payloads', flush=True)

if __name__ == '__main__':
    {'freeze': freeze, 'build': build}[sys.argv[1]]()
