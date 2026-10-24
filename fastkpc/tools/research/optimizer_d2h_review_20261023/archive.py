"""Bind tools, artifacts, research natives, and referenced manifests."""
import hashlib, json
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parents[3]
art = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_d2h_review_20261023'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
work = Path('/tmp/d2h_native')
prod = root / 'fastkpc/build/fastkpc_cuda.so'
natives = {'production': str(prod), 'relink': str(work / 'relink/fastkpc_cuda.so'),
           'control': str(work / 'control/fastkpc_cuda.so'), 'patched': str(work / 'patched/fastkpc_cuda.so')}
native_sha = {k: sha(v) for k, v in natives.items()}
assert native_sha['production'] == native_sha['relink'] == '8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874'
for v in ('control', 'patched'):
    assert json.loads((art / v / 'build.json').read_text())['native_sha256'] == native_sha[v]
refs = {'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022/manifest.json':
        'b846bcb3b99ed33f1445651ad0398206ac99e24b6998a74a7889a731dfc1212e',
        'fastkpc/artifacts/full_cuda_ci/optimizer_double_windows_20261022/manifest.json':
        '969e9cd611e1b80a3b8807490bcff3d82e6a7b28ceaea11f5a8298e307b4fc3c'}
assert all(sha(root / p) == h for p, h in refs.items())
manifest = dict(
    schema='optimizer-d2h-review-manifest-v1', date='2026-10-23',
    natives={k: dict(path=v, sha256=native_sha[k]) for k, v in natives.items()},
    small64_objects={'production': sha(root / 'fastkpc/build/mgcv_multi_penalty_gcv_small64.o'),
                     'control': sha(work / 'control/mgcv_multi_penalty_gcv_small64.o'),
                     'patched': sha(work / 'patched/mgcv_multi_penalty_gcv_small64.o')},
    references=refs,
    tools={str(p.relative_to(root)): sha(p) for p in sorted(here.iterdir()) if p.is_file()},
    files={str(p.relative_to(art)): sha(p) for p in sorted(art.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
(art / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('manifest', sha(art / 'manifest.json'))
