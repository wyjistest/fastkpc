"""Build an isolated L6 host scheduler, checking all 14 device functions."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

tool = Path(__file__).resolve().parent
root = tool.parents[3]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, v: p.write_text(json.dumps(v, indent=2) + '\n')
parent = root / 'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025/build.json'
identity = json.loads(parent.read_text())
native = root / 'fastkpc/build/fastkpc_cuda.so'
assert sha(native) == identity['native_sha256']
assert all(sha(root / p) == h for p, h in identity['source_sha256'].items())
includes = ['-I' + str(root / 'fastkpc/src'), '-I' + str(tool), '-I' + str(out), '-I/usr/local/cuda/include']
cuda_inputs = [tool / 'slot_backend.cu', tool / 'slot_api.hpp', tool.parent / 'optimizer_windows_20261022/fixture.hpp']
cuda_inputs += [root / p for p in identity['source_sha256']]
cuda_hashes = {str(p.relative_to(root)): sha(p) for p in cuda_inputs}
cuda_cmd = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false', '-diag-suppress', '2464',
            '-Xcompiler', '-fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden', '-std=c++17', *includes,
            '-c', str(tool / 'slot_backend.cu'), '-o', str(out / 'slot_backend.o')]
if '--link-only' in sys.argv[2:]:
    prior = json.loads((out / 'cuda-build.json').read_text())
    assert prior['inputs'] == cuda_hashes and prior['object_sha256'] == sha(out / 'slot_backend.o')
else:
    assert not (out / 'cuda-build.json').exists(), 'preserve prior CUDA build'
    print('Compiling isolated CUDA object', flush=True)
    subprocess.run(cuda_cmd, check=True)
    write(out / 'cuda-build.json', dict(inputs=cuda_hashes, command=cuda_cmd, object_sha256=sha(out / 'slot_backend.o')))
if '--cuda-only' in sys.argv[2:]:
    sys.exit(0)
assert not (out / 'build.json').exists(), 'preserve existing build'
symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(native)], text=True).splitlines()
mangled = [s.split()[-1] for s in symbols]
demangled = subprocess.check_output(['c++filt'], input='\n'.join(mangled), text=True).splitlines()
selected = [s for s, d in zip(mangled, demangled) if d.startswith('fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64(')]
assert len(selected) == 1
(out / 'symbols.inc').write_text('#define SYM_small "' + selected[0] + '"\n')
links = ['-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-lcublas', '-ldl']
rpath = ['-Wl,-rpath,' + str(out), '-Wl,-rpath,' + str(native.parent)]
commands = [
    ['g++', '-shared', '-pthread', str(out / 'slot_backend.o'), str(native), *rpath, *links, '-o', str(out / 'slot_backend.so')],
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

reference = parent.parent / 'build-source/fastkpc/build/mgcv_multi_penalty_gcv_small64.o'
assert sha(reference) == identity['objects']['mgcv_multi_penalty_gcv_small64.o']
original, prototype = sass(reference), sass(out / 'slot_backend.o')
write(out / 'device-code.json', dict(status='PASS' if original == prototype else 'FAIL', original=original,
      prototype=prototype, reference_object_sha256=sha(reference), prototype_object_sha256=sha(out / 'slot_backend.o')))
assert original == prototype, 'GPU instruction mismatch'
deps = [parent, tool.parent / 'optimizer_windows_20261022/fixture.hpp', tool.parent / 'late_optimizer_20261025/fixture.hpp',
        tool.parent / 'optimizer_layer_refill_20261023/audit.cpp', tool.parent / 'optimizer_layer_refill_20261023/audit.hpp']
assert sha(native) == identity['native_sha256']
assert all(sha(root / p) == h for p, h in identity['source_sha256'].items())
write(out / 'build.json', dict(native_sha256=sha(native), producer_head=identity['producer_head'],
      evidence_class='source-closure isolated prototype, not production integration',
      source_sha256=identity['source_sha256'], commands=[cuda_cmd, *commands],
      execution_tools={str(p.relative_to(root)): sha(p) for p in tool.iterdir() if p.suffix in ('.cu', '.hpp', '.cpp', '.py', '.md')},
      dependencies={str(p.relative_to(root)): sha(p) for p in deps},
      outputs={p: sha(out / p) for p in ['slot_backend.o', 'slot_backend.so', 'native_observe.so', 'audit.so', 'replay', 'symbols.inc', 'device-code.json']},
      device_code='PASS'))
print('PASS isolated build: 14 GPU functions match production', flush=True)
