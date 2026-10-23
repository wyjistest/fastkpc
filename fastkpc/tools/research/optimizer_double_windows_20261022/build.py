"""Build a diagnostic driver using the original fixture/observer and unchanged native."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
root = here.parents[3]
original = here.parent / 'optimizer_windows_20261022'
capture = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = json.loads((capture / 'manifest.json').read_text())
assert sha(capture / 'manifest.json') == 'b846bcb3b99ed33f1445651ad0398206ac99e24b6998a74a7889a731dfc1212e'
native = root / 'fastkpc/build/fastkpc_cuda.so'
assert sha(native) == manifest['native_sha256']
assert all(sha(root / p) == h for p, h in manifest['source_closure'].items())
inputs = ['raw/' + n + '.bin' for n in ('L3-W31', 'L3-W32', 'L5-W7', 'L5-W8')]
inputs += ['capture.json', 'capture-authority.json', 'closure.json', 'symbols.inc']
assert all(sha(capture / p) == manifest['files'][p] for p in inputs)
assert all(json.loads((capture / 'capture.json').read_text())['gates'].values())
for name in ('fixture.hpp', 'observe.hpp', 'observe.cpp'):
    p = original / name
    assert sha(p) == manifest['tools'][str(p.relative_to(root))]
(out / 'symbols.inc').write_bytes((capture / 'symbols.inc').read_bytes())
rcpp = subprocess.check_output(['Rscript', '--vanilla', '-e', 'cat(system.file("include",package="Rcpp"))'], text=True)
common = ['g++', '-O2', '-std=c++17', '-fPIC', '-pthread', '-I'+str(out),
          '-I'+str(original), '-I'+str(root/'fastkpc/src'), '-I/usr/local/cuda/include',
          '-I/usr/local/lib/R/include', '-I'+rcpp, '-L/usr/local/cuda/lib64',
          '-Wl,-rpath,/usr/local/cuda/lib64', '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib']
commands = [common + ['-shared', str(original/'observe.cpp'), '-lcudart', '-lnvToolsExt',
                     '-ldl', str(native), '-Wl,-rpath,'+str(native.parent), '-o', str(out/'observe.so')],
            common + [str(here/'replay.cpp'), str(out/'observe.so'), str(native),
                      '-Wl,-rpath,'+str(out), '-Wl,-rpath,'+str(native.parent),
                      '-lR', '-lcudart', '-lnvToolsExt', '-ldl', '-o', str(out/'replay')]]
for command in commands:
    subprocess.run(command, check=True)
sources = [here/'build.py', here/'replay.cpp'] + [original/n for n in ('fixture.hpp','observe.hpp','observe.cpp')]
receipt = dict(schema='optimizer-double-window-build-v1', native_sha256=sha(native),
               capture_manifest_sha256=sha(capture/'manifest.json'), commands=commands,
               inputs={str((capture/p).relative_to(root)): sha(capture/p) for p in inputs},
               sources={str(p.relative_to(root)): sha(p) for p in sources},
               source_closure=manifest['source_closure'],
               outputs={n:sha(out/n) for n in ('replay','observe.so','symbols.inc')})
(out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS diagnostic build; native', sha(native))
