"""Identity helpers for this diagnostic, not a production configuration path."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
NATIVE = ROOT / 'fastkpc/build/fastkpc_cuda.so'
CAPTURE = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023'
ATTESTATION = ROOT / 'fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261024.json'


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
    build = load(out / 'build.json')
    assert sha(NATIVE) == build['native_sha256']
    for path, expected in {**build['source_sha256'], **build['execution_tools'], **build['dependencies']}.items():
        assert sha(ROOT / path) == expected, path
    for path, expected in build['outputs'].items():
        assert sha(out / path) == expected, path
    for path, expected in build['fixtures'].items():
        assert sha(CAPTURE / path) == expected, path
    return build
