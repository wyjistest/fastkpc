"""Compile external NVTX companions against the committed qualified native."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ATTESTATION, CAPTURE, NATIVE, ROOT, TOOL, load, sha, write_new

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
a = load(ATTESTATION)
assert sha(NATIVE) == a['native_sha256']
for path, expected in a['source_sha256'].items():
    assert sha(ROOT / path) == expected, path
    blob = subprocess.check_output(['git', 'show', a['runtime_commit'] + ':' + path], cwd=ROOT)
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == expected, path
manifest = load(CAPTURE / 'manifest.json')
assert sha(CAPTURE / 'manifest.json') == a['archive_sha256'][str((CAPTURE / 'manifest.json').relative_to(ROOT))]
fixtures = {p: h for p, h in manifest['files'].items() if p.startswith('raw/')}
assert len(fixtures) == 34
for path, expected in fixtures.items():
    assert sha(CAPTURE / path) == expected, path
symbols = [r.split()[-1] for r in subprocess.check_output(['nm', '-D', '--defined-only', str(NATIVE)], text=True).splitlines()]
demangled = subprocess.check_output(['c++filt'], input='\n'.join(symbols), text=True).splitlines()
names = {'batch': 'fastkpc::multi_penalty_gcv_cuda_optimize_batch',
         'small': 'fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64'}
lines = []
for alias, name in names.items():
    matches = [s for s, d in zip(symbols, demangled) if d.startswith(name + '(')]
    assert len(matches) == 1, (alias, matches)
    lines.append('constexpr const char* SYM_' + alias + ' = "' + matches[0] + '";')
(out / 'symbols.inc').write_text('\n'.join(lines) + '\n')
rcpp = subprocess.check_output(['Rscript', '--vanilla', '-e', 'cat(system.file("include",package="Rcpp"))'], text=True)
old = TOOL.parent / 'optimizer_layer_refill_20261023'
windows = TOOL.parent / 'optimizer_windows_20261022'
common = ['g++', '-O2', '-std=c++17', '-fPIC', '-pthread', '-I' + str(out), '-I' + str(old), '-I' + str(windows),
          '-I' + str(ROOT / 'fastkpc/src'), '-I/usr/local/cuda/include', '-I/usr/local/lib/R/include', '-I' + rcpp,
          '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
          '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib']
commands = [common + ['-shared', str(old / 'nvtx_observe.cpp'), str(old / 'audit.cpp'), str(NATIVE),
                       '-Wl,-rpath,' + str(NATIVE.parent), '-lcudart', '-lcublas', '-lnvToolsExt', '-ldl', '-o', str(out / 'observe.so')],
            common + [str(TOOL / 'profile.cpp'), str(out / 'observe.so'), str(NATIVE), '-Wl,-rpath,' + str(out),
                      '-Wl,-rpath,' + str(NATIVE.parent), '-lR', '-lcudart', '-lnvToolsExt', '-ldl', '-o', str(out / 'profile')]]
for command in commands:
    subprocess.run(command, check=True)
deps = [ATTESTATION, CAPTURE / 'manifest.json', CAPTURE / 'capture.json',
        TOOL.parent / 'optimizer_l4_diagnostic_20261023/fixture.hpp',
        windows / 'fixture.hpp', windows / 'observe.hpp',
        old / 'layer_fixture.hpp', old / 'nvtx_observe.cpp', old / 'audit.cpp', old / 'audit.hpp']
tools = [TOOL / p for p in ['README.md', 'profile.cpp', 'common.py', 'build.py', 'run.py']]
write_new(out / 'build.json', dict(schema='l4-tail-build-v1',
    producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    runtime_commit=a['runtime_commit'], native_sha256=sha(NATIVE), commands=commands,
    source_sha256=a['source_sha256'], fixtures=fixtures,
    dependencies={str(p.relative_to(ROOT)): sha(p) for p in deps},
    execution_tools={str(p.relative_to(ROOT)): sha(p) for p in tools},
    outputs={p: sha(out / p) for p in ['symbols.inc', 'observe.so', 'profile']}))
print('PASS external profile build; production source/native unchanged', flush=True)
