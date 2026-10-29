"""Source snapshots and authenticated execution for default-off postback integration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
BASE = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_postback_runtime_20261029'
PRIOR = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_postback_gpu_20261029/attempt2'
spec = importlib.util.spec_from_file_location('postback_job_support',
    TOOL.parent / 'stage_b_work_pair_runtime_20261028/support.py')
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
environment, processes, job = jobs.environment, jobs.processes, jobs.job

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def freeze_baseline():
    dest = BASE / 'baseline'
    assert not dest.exists()
    manifest = read(PRIOR / 'manifest.json')
    for section in ['production_source_sha256', 'tools', 'shared_tools', 'files']:
        for name, digest in manifest[section].items():
            assert sha((PRIOR if section == 'files' else ROOT) / name) == digest, name
    old = read(ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime/build.json')
    assert sha(ROOT / 'fastkpc/build/fastkpc_cuda.so') == manifest['native_sha256'] == old['native_sha256']
    for name, digest in old['objects'].items():
        assert sha(ROOT / 'fastkpc/build' / name) == digest, name
    dest.mkdir(parents=True)
    for name in manifest['production_source_sha256']:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (dest / 'objects').mkdir()
    for name in old['objects']:
        shutil.copyfile(ROOT / 'fastkpc/build' / name, dest / 'objects' / name)
    shutil.copyfile(ROOT / 'fastkpc/build/fastkpc_cuda.so', dest / 'fastkpc_cuda.so')
    write(dest / 'identity.json', dict(native_sha256=manifest['native_sha256'],
        production_source_sha256=manifest['production_source_sha256'], objects=old['objects'],
        prior_manifest_sha256=sha(PRIOR / 'manifest.json'),
        prior_closure_sha256=sha(PRIOR / 'closure.json'),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    print('PASS baseline frozen', dest, flush=True)

if __name__ == '__main__':
    assert sys.argv[1:] == ['freeze-baseline']
    freeze_baseline()
