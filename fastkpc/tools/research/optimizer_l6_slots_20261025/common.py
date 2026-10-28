"""Identity checks shared by the isolated L6 experiment."""
import hashlib
import json
from pathlib import Path

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
NATIVE = ROOT / 'fastkpc/build/fastkpc_cuda.so'
CAPTURE = ROOT / 'fastkpc/artifacts/full_cuda_ci/late_optimizer_20261025'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def check_identity(out):
    b = load(out / 'build.json')
    assert sha(NATIVE) == b['native_sha256']
    for p,h in {**b['source_sha256'], **b['execution_tools'], **b['dependencies']}.items():
        assert sha(ROOT / p) == h, p
    for p,h in b['outputs'].items():
        assert sha(out / p) == h, p
    capture = load(CAPTURE / 'capture-closure.json')
    assert capture['status'] == 'PASS' and capture['native_sha256'] == b['native_sha256']
    for p,h in capture['files'].items():
        assert sha(CAPTURE / p) == h, p
    return b
