"""Build Codex's unchanged double-window replay against a given native.
Usage: build_replay.py OUT NATIVE   (OUT must not exist)"""
import hashlib, json, subprocess, sys
from pathlib import Path
here = Path(__file__).resolve().parent
root = here.parents[3]
original = here.parent / 'optimizer_windows_20261022'
double = here.parent / 'optimizer_double_windows_20261022'
capture = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'
codex = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_double_windows_20261022'
out = Path(sys.argv[1]).resolve(); native = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(capture / 'manifest.json') == 'b846bcb3b99ed33f1445651ad0398206ac99e24b6998a74a7889a731dfc1212e'
assert sha(codex / 'manifest.json') == '969e9cd611e1b80a3b8807490bcff3d82e6a7b28ceaea11f5a8298e307b4fc3c'
cap = json.loads((capture / 'manifest.json').read_text())
built = json.loads((codex / 'build.json').read_text())
for p, h in built['sources'].items():  # replay.cpp, build.py, fixture/observer sources
    assert sha(root / p) == h, p
inputs = ['raw/' + n + '.bin' for n in ('L3-W31', 'L3-W32', 'L5-W7', 'L5-W8')] + ['symbols.inc']
assert all(sha(capture / p) == cap['files'][p] for p in inputs)
(out / 'symbols.inc').write_bytes((capture / 'symbols.inc').read_bytes())
rcpp = subprocess.check_output(['Rscript', '--vanilla', '-e', 'cat(system.file("include",package="Rcpp"))'], text=True)
common = ['g++', '-O2', '-std=c++17', '-fPIC', '-pthread', '-I' + str(out),
          '-I' + str(original), '-I' + str(root / 'fastkpc/src'), '-I/usr/local/cuda/include',
          '-I/usr/local/lib/R/include', '-I' + rcpp, '-L/usr/local/cuda/lib64',
          '-Wl,-rpath,/usr/local/cuda/lib64', '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib']
commands = [common + ['-shared', str(original / 'observe.cpp'), '-lcudart', '-lnvToolsExt',
                      '-ldl', str(native), '-Wl,-rpath,' + str(native.parent), '-o', str(out / 'observe.so')],
            common + [str(double / 'replay.cpp'), str(out / 'observe.so'), str(native),
                      '-Wl,-rpath,' + str(out), '-Wl,-rpath,' + str(native.parent),
                      '-lR', '-lcudart', '-lnvToolsExt', '-ldl', '-o', str(out / 'replay')]]
for c in commands:
    subprocess.run(c, check=True)
needed = subprocess.check_output(['readelf', '-d', str(out / 'replay')], text=True)
assert str(native) in needed, 'replay must load the given native by absolute path'
(out / 'build.json').write_text(json.dumps(dict(
    schema='optimizer-d2h-review-replay-build-v1', native=str(native), native_sha256=sha(native),
    codex_build_sources=built['sources'], inputs={str((capture / p).relative_to(root)): sha(capture / p) for p in inputs},
    tools={str(p.relative_to(root)): sha(p) for p in (here / 'build_replay.py', here / 'campaign.py')},
    commands=commands, outputs={n: sha(out / n) for n in ('replay', 'observe.so', 'symbols.inc')}), indent=2) + '\n')
print('PASS build', out, sha(native)[:16])
