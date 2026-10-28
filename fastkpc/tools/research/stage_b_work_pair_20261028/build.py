"""Mechanical, source-bound isolated kernel variants; never overwrite production."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
PARENT = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028'
ACTIVITY = ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_attribution_20261028'
NATIVE = ROOT / 'fastkpc/build/fastkpc_cuda.so'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)
        f.write('\n')

def identity():
    b = json.loads((PARENT / 'build.json').read_text())
    assert sha(NATIVE) == b['native_sha256']
    assert all(sha(ROOT / p) == h for p, h in b['source_sha256'].items())
    assert sha(ROOT / 'fastkpc/build/recorded_dbdsqr.inc') == 'be76c5e06ce8bce4013a9759a3a54fc63feed9e421a1d3d44af77e11f32f53b9'
    return b

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)

def compact(text):
    # Four local store blocks; arithmetic and original store expressions retained.
    for forward, vals in [(True, ('cs','sn','oldcs','oldsn')),
                          (False, ('cs','-sn','oldcs','-oldsn')),
                          (True, ('cosr','sinr','cosl','sinl')),
                          (False, ('cosr','-sinr','cosl','-sinl'))]:
        idx = 'i__ - ll + 1' if forward else 'i__ - ll'
        rows = ['\t\twork[' + idx + suffix + '] = ' + value + ';'
                for suffix, value in zip(['',' + nm1',' + nm12',' + nm13'], vals)]
        kept = rows[:2] if forward else [rows[2].replace(' + nm12',''), rows[3].replace(' + nm13',' + nm1')]
        text = replace(text, '\n'.join(rows), '\n'.join(kept))
    text = replace(text, 'record_dlasr_(log, "L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[\n\t\t\tnm13 + 1]',
                   'record_dlasr_(log, "L", "V", "B", &i__1, ncvt, &work[1], &work[*n]', 2)
    # This is a dedicated entry, not a general DBDSQR with truncated scratch.
    text = replace(text, '    /* System generated locals */',
                   '    if (*uplo != \'L\' || *ncvt != 1 || *nru != 0 || *ncc != 0) { *info = -101; return 0; }\n    /* System generated locals */')
    return text

def instrument(text):
    text = replace(text, '    if (*n == 1) {', '    if (*n == 1) { log->coverage |= 1u;')
    text = replace(text, '    if (lower) {', '    if (lower) { log->coverage |= 2u;')
    text = replace(text, '/*        2 by 2 block, handle separately */',
                   'log->coverage |= 4u;\n/*        2 by 2 block, handle separately */')
    text = replace(text, '    iter = iter + m - ll;',
        '    ++log->sweeps;\n    if (log->previous_direction && log->previous_direction != idir) log->coverage |= 128u;\n'
        '    log->previous_direction = idir;\n    log->coverage |= shift == 0. ? (idir == 1 ? 8u : 16u) : (idir == 1 ? 32u : 64u);\n'
        '    iter = iter + m - ll;')
    text = replace(text, '\tif (d__[i__] < 0.) {', '\tif (d__[i__] < 0.) { log->coverage |= 256u;')
    text = replace(text, '\t    if (d__[j] <= smin) {', '\t    if (d__[j] <= smin) { if (d__[j] == smin) log->coverage |= 512u;')
    text = replace(text, 'L200:\n', 'L200:\n    log->coverage |= 1024u;\n')
    text = replace(text, '    maxitdivn = *n * 6;',
                   '    maxitdivn = *n * 6;\n    if (log->force_nonconvergence) maxitdivn = 0;')
    return text

def kernel_source(original, candidate, diagnostic):
    text = original
    if candidate:
        text = replace(text, '7*n*sizeof(double)', '5*n*sizeof(double)')
        text = replace(text, '7*c->n*sizeof(double)', '5*c->n*sizeof(double)')
    if diagnostic:
        text = '#include "diagnostic.hpp"\n' + text
        text = replace(text, '  int count, capacity, overflow;',
            '  int count, capacity, overflow;\n  unsigned coverage = 0;\n  int sweeps = 0, previous_direction = 0, force_nonconvergence = 0;')
        text = replace(text, 'Rotation* rotations, int capacity) {',
            'Rotation* rotations, int capacity, WorkPairState* diag, double* deq) {')
        text = replace(text, '  double* work = q+n;',
            '  double* work = q+n;\n  for (int j=threadIdx.x;j<8;j+=blockDim.x) work[WP_WORK_FACTOR*n+j]=0x1.2345p+9;\n  __syncthreads();')
        text = replace(text, '  z.rotations = log.count;',
            '  diag[component] = {info,log.count,log.overflow,int(log.coverage),log.sweeps,0,r,0};\n'
            '  for(int j=0;j<8;++j) if(work[WP_WORK_FACTOR*n+j]!=0x1.2345p+9) diag[component].canary=1;\n'
            '  for(int j=0;j<r;++j) { deq[size_t(component)*3*n+j]=d[j]; deq[size_t(component)*3*n+n+j]=e[j]; deq[size_t(component)*3*n+2*n+j]=q[j]; }\n'
            '  z.rotations = log.count;')
        text = replace(text, '  int n, k, count, capacity, tile;',
            '  int n, k, count, capacity, tile;\n  WorkPairState* diag; double* deq;')
        text = replace(text, '    c->n=n; c->k=k+1;',
            '    c->diag=static_cast<WorkPairState*>(allocate(owner,count*sizeof(WorkPairState)));\n'
            '    c->deq=static_cast<double*>(allocate(owner,size_t(count)*3*n*sizeof(double)));\n    c->n=n; c->k=k+1;')
        text = replace(text, 'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity);',
            'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity,c->diag,c->deq);')
        size = '5' if candidate else '7'
        text = replace(text, size+'*n*sizeof(double)', '('+size+'*n+8)*sizeof(double)')
        text = replace(text, size+'*c->n*sizeof(double)', '('+size+'*c->n+8)*sizeof(double)')
        text += '\n#include "diagnostic_footer.inc"\n'
    else:
        text = '#include "timing_state.hpp"\n' + text
        text = replace(text, '    return c;\n', '    wp_resource(bidiagonal,n); ++wp_creates;\n    return c;\n')
        text = replace(text, 'noexcept { delete p; }', 'noexcept { ++wp_destroys; delete p; }')
        text = replace(text, '  prepare<<<count,128,0,c->stream>>>',
            '  ++wp_submits; wp_targets += count;\n  prepare<<<count,128,0,c->stream>>>')
        text += '\nextern "C" void wp_resources(int* out) { structured_check(cudaSetDevice(0)); '
        text += 'structured_check(cudaFuncSetAttribute(bidiagonal,cudaFuncAttributeMaxDynamicSharedMemorySize,(3+WP_WORK_FACTOR)*351*8)); '
        text += 'wp_resource(bidiagonal,351); out[0]=wp_blocks;out[1]=wp_regs;out[2]=wp_local;out[3]=wp_static;out[4]=wp_dynamic; }\n'
    return '#define WP_WORK_FACTOR '+('2' if candidate else '4')+'\n'+text

if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    b = identity()
    original = (ROOT/'fastkpc/src/cuda/full_cuda_ci_structured_eig.cu').read_text()
    generated = (ROOT/'fastkpc/build/recorded_dbdsqr.inc').read_text()
    candidate = compact(generated)
    with (out/'work-pair.diff').open('x') as f:
        f.writelines(difflib.unified_diff(generated.splitlines(True), candidate.splitlines(True), fromfile='sealed/dbdsqr',tofile='isolated/fixed-vt-work-pair'))
    commands = []
    for name, is_candidate, diagnostic in [('baseline_diag',False,True),('candidate_diag',True,True),
                                            ('baseline_timing',False,False),('candidate_timing',True,False)]:
        folder = out/name
        folder.mkdir()
        inc = candidate if is_candidate else generated
        if diagnostic: inc = instrument(inc)
        (folder/'recorded_dbdsqr.inc').write_text(inc)
        (folder/'kernel.cu').write_text(kernel_source(original,is_candidate,diagnostic))
        cmd = ['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-DFASTKPC_STRUCTURED_FMAD_DISABLED=1',
               '-Xcompiler','-fPIC','-std=c++17','-shared','--cudart','shared','-diag-suppress','2464',
               '-I'+str(TOOL),'-I'+str(ROOT/'fastkpc/src/cuda'),str(folder/'kernel.cu'),
               '-o',str(out/(name+'.so'))]
        commands.append(cmd)
        with (out/(name+'-build.log')).open('x') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        with (out/(name+'-resources.txt')).open('x') as f:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(out/(name+'.so'))],stdout=f,check=True)
        print('built',name,flush=True)
    cmd=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-std=c++17','--cudart','shared',
         '-I'+str(TOOL),'-I'+str(ROOT/'fastkpc/src/cuda'),str(TOOL/'qualify.cu'),'-ldl','-o',str(out/'qualify')]
    commands.append(cmd)
    with (out/'qualify-build.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    identity()
    write(out/'build.json',dict(status='PASS',native_sha256=b['native_sha256'],source_sha256=b['source_sha256'],
        commands=commands,tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
        outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)))
