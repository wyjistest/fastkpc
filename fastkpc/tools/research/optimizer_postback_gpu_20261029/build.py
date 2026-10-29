"""Authenticated isolated source overlay; never writes production source/native."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
AUDIT = ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_postback_audit_20261029/attempt1'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()

def write(p, obj):
    with Path(p).open('x') as f: json.dump(obj, f, indent=2); f.write('\n')

def replace(s, old, new):
    assert s.count(old) == 1, old[:120]
    return s.replace(old, new)

def overlay(out):
    identity = json.loads((AUDIT/'manifest.json').read_text())
    assert sha(AUDIT/'manifest.json') == '0695f2192a90a33bb7505ad00f6fccf22ef5936d3057cbfb5a4eaf5ebfa66076'
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so') == identity['native_sha256']
    for p, h in identity['production_source_sha256'].items(): assert sha(ROOT/p) == h, p
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'
    shutil.copytree(ROOT/'fastkpc/src', source,
        ignore=shutil.ignore_patterns('*.o', '*.so', '*.a', '*.d'))
    (out/'generated').mkdir()
    shutil.copyfile(ROOT/'fastkpc/build/recorded_dbdsqr.inc', out/'generated/recorded_dbdsqr.inc')
    header = source/'cuda/lapack_312_small_dgesdd.cuh'
    s = header.read_text()
    candidate = (TOOL/'candidate.inc').read_text()
    s = replace(s, '__device__ integer apply_reference_blocked_q_left(',
                candidate + '\n__device__ integer apply_reference_blocked_q_left(')
    qstart = s.index('__device__ integer apply_reference_blocked_q_left(')
    qend = s.index('__device__ void apply_reference_blocked_q_left_transpose(', qstart)
    chunk = s[qstart:qend]
    chunk = replace(chunk, '  const integer lane =', '''#if POSTBACK_SLICES || POSTBACK_PROFILE
  if ((n == 37 || n == 46) && m >= n && reflector_count == n)
    return postback_apply<POSTBACK_SLICES != 0>(true, m, n, reflector_matrix,
      reflector_leading_dimension, tau, output, output_leading_dimension, scratch);
#endif
  const integer lane =''')
    s = s[:qstart] + chunk + s[qend:]
    pstart = s.index('__device__ integer apply_reference_blocked_p_right_transpose(')
    pend = s.index('__device__ int small_dbdsdc_upper_i_two_warp(', pstart)
    chunk = s[pstart:pend]
    chunk = replace(chunk, 'doublereal* scratch) {', 'doublereal* scratch, integer postback_deferred = 0) {')
    chunk = replace(chunk, '  const integer lane =', '''#if POSTBACK_SLICES || POSTBACK_PROFILE
  if (n == 37 || n == 46)
    return postback_apply<POSTBACK_SLICES != 0>(false, n, n, reflector_matrix,
      reflector_leading_dimension, tau, output, output_leading_dimension, scratch,
      postback_deferred);
#endif
  const integer lane =''')
    s = s[:pstart] + chunk + s[pend:]
    start = s.index('__device__ int complete_stable_right_vectors_after_score(')
    end = s.index('__device__ int small_dgesdd_left_direct_two_warp(', start)
    chunk = replace(s[start:end], 'workspace->bidiagonal_vt, n, workspace->work);',
                    'workspace->bidiagonal_vt, n, workspace->work, 1);')
    s = s[:start] + chunk + s[end:]
    header.write_text(s)
    write(out/'source.json', dict(native_sha256=identity['native_sha256'],
        production_source_sha256=identity['production_source_sha256'],
        audit_manifest_sha256=sha(AUDIT/'manifest.json'),
        candidate_header_sha256=sha(header),
        generated_sha256=sha(out/'generated/recorded_dbdsqr.inc'),
        tools={str(p.relative_to(ROOT)): sha(p) for p in TOOL.iterdir() if p.is_file()}))

def compile_unit(out):
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false',
        '-std=c++17', '-diag-suppress', '2464', '-Xptxas=-v',
        '-I'+str(out/'source'), '-I'+str(out/'source/cuda'), '-I'+str(out/'generated'),
        '-I'+str(TOOL), str(TOOL/'unit.cu'), '-o', str(out/'unit')]
    write(out/'unit-build-command.json', command)
    with (out/'unit-build.log').open('x') as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    print('PASS isolated GPU unit build', flush=True)

if __name__ == '__main__':
    destination = Path(sys.argv[1]).resolve()
    overlay(destination)
    compile_unit(destination)
