"""Archive actual captures, replay evidence, and their source/binary closure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

work, out = (Path(p).resolve() for p in sys.argv[1:])
assert not out.exists(), 'refuse to overwrite evidence'
out.mkdir(parents=True)
root = Path.cwd()
here = Path(__file__).resolve().parent
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1<<20), b''):
            digest.update(chunk)
    return digest.hexdigest()
native = root/'fastkpc/build/fastkpc_cuda.so'
native_sha = sha(native)
assert native_sha == 'fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50'
build = json.loads((work/'build/build.json').read_text())
assert build['native_sha256'] == native_sha
assert all(sha(root/path) == digest for path,digest in build['sources'].items())
memcheck = (work/'memcheck.log').read_text()
assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in memcheck
assert 'ERROR SUMMARY: 0 errors' in memcheck
source_hashes = {str(p.relative_to(root)):sha(p) for p in sorted(here.iterdir()) if p.is_file()}
for name in ('capture','capture-inf','baseline-inf','build'):
    source = work/name
    assert source.is_dir(), source
    destination = out/name
    destination.mkdir()
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copy2(path,destination/path.name)
    if name.startswith('capture'):
        metadata=json.loads((source/'capture.json').read_text())
        replay=json.loads((source/'replay.json').read_text())
        assert all(metadata['gates'].values())
        assert replay['original_shape_all_exact'] and replay['coefficient_metadata_identity']
        assert replay['bounded16_bitwise_pass']
        assert replay['capture_sha256'] == sha(source/'capture.json')
        assert replay['replay_library_sha256'] == sha(work/'build/replay.so')
        assert replay['replay_script_sha256'] == sha(here/'replay.py')
        files=sorted((source/'raw').iterdir())
        raw_hashes={p.name:sha(p) for p in files}
        with tarfile.open(destination/'raw.tar.gz','w:gz',compresslevel=6) as tar:
            for path in files:
                tar.add(path,arcname='raw/'+path.name,recursive=False)
        (destination/'raw-manifest.json').write_text(json.dumps(raw_hashes,indent=2)+'\n')
        with tarfile.open(destination/'raw.tar.gz','r:gz') as tar:
            for member in tar:
                assert member.isfile() and member.name.startswith('raw/')
                digest=hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                assert digest == raw_hashes[Path(member.name).name]
for path in sorted(work.glob('*.log')):
    shutil.copy2(path,out/path.name)
baseline=Path('/tmp/fastkpc-exact-speedup-20261020/subset-restored-queues32.rds')
shutil.copy2(baseline,out/'baseline-subset22.rds')
assert source_hashes == {str(p.relative_to(root)):sha(p) for p in sorted(here.iterdir()) if p.is_file()}
manifest=dict(schema='production-cohort-frozen-evidence-v1',audit='PASS',
    native_sha256=native_sha,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    sources=source_hashes,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version',
                                 '--format=csv'],text=True),
    nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True),
    files={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(out.rglob('*')) if p.is_file()})
assert sha(native)==native_sha
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PASS archive',len(manifest['files']),'files; manifest SHA256',sha(out/'manifest.json'))
