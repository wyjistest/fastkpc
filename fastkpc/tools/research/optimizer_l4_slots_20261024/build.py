"""Build hidden numerical clone and production-output observer, not the runtime."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
tool = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, obj: p.write_text(json.dumps(obj, indent=2) + '\n')
attestation = root / 'fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261024.json'
identity = json.loads(attestation.read_text())
native = root / 'fastkpc/build/fastkpc_cuda.so'
assert sha(native) == identity['native_sha256']
assert all(sha(root / p) == h for p, h in identity['source_sha256'].items())
for p, h in identity['source_sha256'].items():
    blob = subprocess.check_output(['git', 'show', identity['runtime_commit'] + ':' + p], cwd=root)
    assert hashlib.sha256(blob).hexdigest() == h, p
capture = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023'
assert sha(capture / 'manifest.json') == 'd5c77e9c8bcd89c5ee5ac3fcd93fc8421a60231000bd355273b90e631d386ceb'
captured_files = {p: h for p, h in json.loads((capture / 'manifest.json').read_text())['files'].items() if p.startswith('raw/')}
assert len(captured_files) == 34 and all(sha(capture / p) == h for p, h in captured_files.items())
includes = ['-I' + str(root / 'fastkpc/src'), '-I' + str(tool), '-I' + str(out), '-I/usr/local/cuda/include']
cuda_inputs = [tool / 'slot_backend.cu', tool / 'slot_api.hpp',
               tool.parent / 'optimizer_windows_20261022/fixture.hpp']
cuda_inputs += [root / p for p in identity['source_sha256']]
cuda_hashes = {str(p.relative_to(root)): sha(p) for p in cuda_inputs}
cuda_cmd = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false', '-diag-suppress', '2464',
            '-Xcompiler', '-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden', '-std=c++17', *includes,
            '-c', str(tool / 'slot_backend.cu'), '-o', str(out / 'slot_backend.o')]
link_only = '--link-only' in sys.argv[2:]
if link_only:
    prior = json.loads((out / 'cuda-build.json').read_text())
    assert prior['inputs'] == cuda_hashes and prior['object_sha256'] == sha(out / 'slot_backend.o')
else:
    print('Compiling isolated CUDA object', flush=True)
    subprocess.run(cuda_cmd, check=True)
    write(out / 'cuda-build.json', dict(inputs=cuda_hashes, command=cuda_cmd, object_sha256=sha(out / 'slot_backend.o')))
if '--cuda-only' in sys.argv[2:]:
    sys.exit(0)

symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(native)], text=True).splitlines()
mangled = [s.split()[-1] for s in symbols]
demangled = subprocess.check_output(['c++filt'], input='\n'.join(mangled), text=True).splitlines()
selected = [s for s, d in zip(mangled, demangled)
            if d.startswith('fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64(')]
assert len(selected) == 1
(out / 'symbols.inc').write_text('#define SYM_small "' + selected[0] + '"\n')
links = ['-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-lcublas', '-ldl']
rpath = ['-Wl,-rpath,' + str(out), '-Wl,-rpath,' + str(native.parent)]
commands = [
    ['g++', '-shared', '-pthread', str(out / 'slot_backend.o'), str(native), *rpath, *links,
     '-o', str(out / 'slot_backend.so')],
    ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-pthread', *includes,
     str(tool.parent / 'optimizer_layer_refill_20261023/audit.cpp'), *links, '-o', str(out / 'audit.so')],
    ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-pthread', *includes, str(tool / 'native_observe.cpp'),
     str(out / 'slot_backend.so'), *rpath, *links, '-o', str(out / 'native_observe.so')],
    ['g++', '-O2', '-std=c++17', '-pthread', *includes, str(tool / 'replay.cpp'),
     str(out / 'audit.so'), str(out / 'native_observe.so'), str(out / 'slot_backend.so'), str(native),
     *rpath, *links, '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib',
     '-lR', '-lRlapack', '-lRblas', '-o', str(out / 'replay')],
]
for cmd in commands:
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
exports = subprocess.check_output(['nm', '-D', '--defined-only', str(out / 'slot_backend.so')], text=True)
assert 'create_multi_penalty_gcv_cuda_prepared_small64' not in exports
assert 'multi_penalty_gcv_cuda_optimize_batch_small64' not in exports

def sass(path):
    result, name = {}, None
    proc = subprocess.Popen(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(path)], stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if 'Function :' in line:
            name = line.split('Function :')[1].strip()
            name = re.sub(r'\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+_cu_[0-9a-f]{8}', '4ANON', name)
            assert name not in result
            result[name] = []
        elif name:
            result[name].extend(re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/', line))
    assert proc.wait() == 0 and len(result) == 14 and all(result.values())
    return {n: dict(words=len(x), sha256=hashlib.sha256('\n'.join(x).encode()).hexdigest()) for n, x in result.items()}

reference = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_runtime_20261023/build-source/fastkpc/build/mgcv_multi_penalty_gcv_small64.o'
assert sha(reference) == '24fd038cca7e1232db4dd3f5f01f9d823632c4b976502bf946ac5c7b08414452'
original, prototype = sass(reference), sass(out / 'slot_backend.o')
write(out / 'device-code.json', dict(status='PASS' if original == prototype else 'FAIL',
      original=original, prototype=prototype, reference_object_sha256=sha(reference),
      prototype_object_sha256=sha(out / 'slot_backend.o')))
assert original == prototype, 'GPU instruction mismatch'
deps = [attestation, tool.parent / 'optimizer_windows_20261022/fixture.hpp',
        tool.parent / 'optimizer_l4_diagnostic_20261023/fixture.hpp',
        tool.parent / 'optimizer_layer_refill_20261023/audit.cpp',
        tool.parent / 'optimizer_layer_refill_20261023/audit.hpp']
write(out / 'build.json', dict(native_sha256=sha(native), runtime_commit=identity['runtime_commit'],
      source_sha256=identity['source_sha256'], commands=[cuda_cmd, *commands],
      execution_tools={str(p.relative_to(root)): sha(p) for p in tool.iterdir() if p.suffix in ('.cu', '.hpp', '.cpp', '.py', '.md')},
      dependencies={str(p.relative_to(root)): sha(p) for p in deps},
      outputs={p: sha(out / p) for p in ['slot_backend.o', 'slot_backend.so', 'native_observe.so', 'audit.so', 'replay', 'symbols.inc', 'device-code.json']},
      fixtures={str(p.relative_to(capture)): sha(p) for p in [capture / 'manifest.json', *sorted((capture / 'raw').glob('L4-W*.bin'))]},
      device_code='PASS'))
print('PASS isolated build: 14 GPU functions match production', flush=True)
