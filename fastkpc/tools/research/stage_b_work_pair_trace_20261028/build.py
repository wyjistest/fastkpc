"""Build host-only replay with current-mode authentication; preserve native."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
OLD_TOOL = ROOT / 'fastkpc/tools/research/stage_b_attribution_20261028'
SEALED = ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_attribution_20261028'
CAPTURE = SEALED / 'attempt3/capture'
PARENT = ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_runtime_20261028/runtime'
NATIVE = ROOT / 'fastkpc/build/fastkpc_cuda.so'


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


def identity():
    b = read(PARENT / 'build.json')
    assert sha(NATIVE) == b['native_sha256']
    assert b['native_sha256'] == '12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d'
    for name, expected in b['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for name, key in [('recorded_dbdsqr.inc', 'original_generator_output_sha256'),
                      ('recorded_dbdsqr_fixed_vt.inc', 'fixed_vt_generator_output_sha256')]:
        assert sha(ROOT / 'fastkpc/build' / name) == b[key]
    return b


if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    b = identity()
    sealed = read(SEALED / 'manifest.json')
    inputs = [CAPTURE / f'l{level}-{suffix}' for level in [2, 3]
              for suffix in ['residuals.f64', 'targets.tsv', 'payload-sha256.txt']]
    dependencies = [PARENT / 'build.json', PARENT / 'closure.json',
                    SEALED / 'manifest.json', SEALED / 'closure.json', *inputs]
    for p in inputs:
        assert sha(p) == sealed['files'][str(p.relative_to(SEALED))], str(p)
    for name in ['common.hpp', 'observe.cpp', 'replay.cpp', 'trace.cpp']:
        p = OLD_TOOL / name
        assert sha(p) == sealed['tools'][str(p.relative_to(ROOT))], str(p)
        dependencies.append(p)
    symbols = SEALED / 'attempt3/symbols.inc'
    assert sha(symbols) == sealed['files']['attempt3/symbols.inc']
    dependencies.append(symbols)
    includes = ['-I' + str(ROOT / 'fastkpc/src'), '-I' + str(OLD_TOOL),
                '-I' + str(SEALED / 'attempt3'), '-I/usr/local/cuda/include']
    libs = ['-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64',
            '-lcudart', '-lcrypto', '-ldl']
    base = ['g++', '-O2', '-std=c++17', '-pthread', *includes]
    commands = [base + ['-fPIC', '-shared', str(OLD_TOOL / 'observe.cpp'),
                       *libs, '-o', str(out / 'observe.so')],
                base + [str(TOOL / 'replay.cpp'), str(NATIVE),
                        '-Wl,-rpath,' + str(NATIVE.parent), *libs,
                        '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib',
                        '-lR', '-o', str(out / 'replay')],
                base + ['-fPIC', '-shared', str(OLD_TOOL / 'trace.cpp'),
                        *libs, '-lcupti', '-o', str(out / 'trace.so')]]
    for i, command in enumerate(commands):
        with (out / f'build-{i}.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    identity()
    write(out / 'build.json', dict(schema='work-pair-activity-build-v1', status='PASS',
        producer_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        native_sha256=b['native_sha256'], source_sha256=b['source_sha256'],
        parent_build_sha256=sha(PARENT / 'build.json'), commands=commands,
        kernels_recompiled=False, capture=str(CAPTURE),
        tools={str(p.relative_to(ROOT)): sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
        dependencies={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        outputs={name: sha(out / name) for name in ['observe.so', 'replay', 'trace.so']}))
    print('PASS current native and captured inputs; host-only build', flush=True)
