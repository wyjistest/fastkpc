"""Generate isolated copies of current source; never overwrite production."""
import difflib
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
WP = ROOT / 'fastkpc/tools/research/stage_b_work_pair_20261028'
TRACE = ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_trace_20261028/attempt1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prior = module('recover_parent_identity', ROOT / 'fastkpc/tools/research/stage_b_work_pair_trace_20261028/build.py')
wp = module('recover_parent_instrumentation', WP / 'build.py')
sha, read, write, identity = prior.sha, prior.read, prior.write, prior.identity
replace = wp.replace
NATIVE, CAPTURE = prior.NATIVE, prior.CAPTURE


def source(original, tile, diagnostic):
    text = replace(original,
        '    if (c->tile < 1) throw std::runtime_error("insufficient structured shared memory");',
        '    if (c->tile < 1) throw std::runtime_error("insufficient structured shared memory");\n'
        '    c->tile=std::min(c->tile,RT_TILE);\n'
        '    if (c->mode != StructuredDbdsqrMode::FixedVtWorkPair)\n'
        '      throw std::runtime_error("recover prototype requires work-pair ON");')
    if not diagnostic:
        return '#define RT_TILE '+str(tile)+'\n'+text+'\n#include "timing_footer.inc"\n'
    text = '#include "diagnostic.hpp"\n'+text
    text = replace(text, '  int count, capacity, overflow;',
        '  int count, capacity, overflow;\n  unsigned coverage = 0;\n'
        '  int sweeps = 0, previous_direction = 0, force_nonconvergence = 0;')
    text = replace(text, 'Rotation* rotations, int capacity) {',
        'Rotation* rotations, int capacity, WorkPairState* diag, double* deq) {')
    text = replace(text, '  double* work = q+n;',
        '  double* work = q+n;\n'
        '  for(int j=threadIdx.x;j<8;j+=blockDim.x) work[2*n+j]=0x1.2345p+9;\n'
        '  __syncthreads();')
    text = replace(text, '  z.rotations = log.count;',
        '  diag[component]={info,log.count,log.overflow,int(log.coverage),log.sweeps,0,r,0};\n'
        '  for(int j=0;j<8;++j) if(work[2*n+j]!=0x1.2345p+9) diag[component].canary=1;\n'
        '  for(int j=0;j<r;++j) {deq[size_t(component)*3*n+j]=d[j];\n'
        '    deq[size_t(component)*3*n+n+j]=e[j];deq[size_t(component)*3*n+2*n+j]=q[j];}\n'
        '  z.rotations = log.count;')
    text = replace(text, '  int n, k, count, capacity, tile;',
        '  int n, k, count, capacity, tile;\n  WorkPairState* diag; double* deq;')
    text = replace(text, '    c->n=n; c->k=k+1;',
        '    c->diag=static_cast<WorkPairState*>(allocate(owner,count*sizeof(WorkPairState)));\n'
        '    c->deq=static_cast<double*>(allocate(owner,size_t(count)*3*n*sizeof(double)));\n'
        '    c->n=n; c->k=k+1;')
    text = replace(text, 'const auto shared_bytes = structured_dbdsqr_shared_bytes(c->mode,n);',
        'const auto shared_bytes = structured_dbdsqr_shared_bytes(c->mode,n)+8*sizeof(double);')
    text = replace(text, 'bidiagonal<Mode><<<count,32,shared_bytes,c->stream>>>',
        'bidiagonal<Mode><<<count,32,shared_bytes+8*sizeof(double),c->stream>>>')
    text = replace(text, 'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity);',
        'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity,c->diag,c->deq);')
    footer = replace((WP / 'diagnostic_footer.inc').read_text(),
                     'fastkpc::lapack312::recorded_dbdsqr_(',
                     'fastkpc::lapack312::recorded_dbdsqr_fixed_vt_(')
    return '#define WP_WORK_FACTOR 2\n#define RT_TILE '+str(tile)+'\n'+text+'\n'+footer+'\n#include "diagnostic_extra.inc"\n'


def qualifier():
    text = (WP / 'qualify.cu').read_text()
    text = '#include <memory>\n#include "full_cuda_ci_structured_eig.hpp"\n'+text
    text = replace(text, 'std::vector<Lane*> lanes;', 'std::vector<std::unique_ptr<Lane>> lanes;')
    text = replace(text, 'lanes.push_back(new Lane(a,b));', 'lanes.push_back(std::make_unique<Lane>(a,b));')
    text = replace(text, 'for(auto* p:lanes)delete p;', 'lanes.clear();')
    text = replace(text, 'Lane(Library& x,Library& y):a(x),b(y)',
                   'Lane(Library& x,Library& y,int k=35,int capacity=170662):a(x),b(y)')
    text = replace(text, '.create(351,35,64,170662,stream,',
                   '.create(351,k,64,capacity,stream,', 2)
    text = replace(text, 'result=static_cast<Result*>(Memory::allocate(&memory,64*sizeof(Result)));}',
        'result=static_cast<Result*>(Memory::allocate(&memory,64*sizeof(Result)));\n'
        '    check(cudaStreamSynchronize(nullptr)); // Complete default-stream zero/redzone initialization.\n'
        '  }')
    text = replace(text, 'check(cudaMalloc(&p,bytes));',
        'check(cudaMalloc(&p,bytes+512));\n'
        '    check(cudaMemset(p,0xa5,256));p=static_cast<char*>(p)+256;\n'
        '    check(cudaMemset(static_cast<char*>(p)+bytes,0xa5,256));')
    text = replace(text, '~Memory(){for(auto v:items){cudaFree(v.first);--live_allocations;live_bytes-=v.second;}}',
        '~Memory(){for(auto v:items){\n'
        '    unsigned char guards[512];\n'
        '    check(cudaMemcpy(guards,static_cast<char*>(v.first)-256,256,cudaMemcpyDeviceToHost));\n'
        '    check(cudaMemcpy(guards+256,static_cast<char*>(v.first)+v.second,256,cudaMemcpyDeviceToHost));\n'
        '    for(auto x:guards)need(x==0xa5,"allocation redzone corrupted");\n'
        '    check(cudaFree(static_cast<char*>(v.first)-256));--live_allocations;live_bytes-=v.second;}}')
    text = replace(text, 'int main(int argc,char** argv) {',
                   '#include "boundary_cases.inc"\n\nint main(int argc,char** argv) {')
    text = replace(text, 'for(int r=1;r<=350;++r){auto v=',
        'for(int r=1;r<=350;++r){if(std::getenv("RECOVER_SMOKE")&&r!=1&&r!=2&&r!=16&&r!=350)continue;auto v=')
    text = replace(text, 'for(int kind=1;kind<=7;++kind){\n        auto v=',
        'for(int kind=1;kind<=7;++kind){\n        if(std::getenv("RECOVER_SMOKE")&&r!=2&&r!=350)continue;\n        auto v=')
    text = replace(text, '      for(int level:{2,3}) {',
        '      const int extra_cases=recover_boundaries(a,b,synthetic);\n'
        '      std::cout<<"PASS recover boundary cases "<<extra_cases<<std::endl;\n'
        '      const bool smoke=std::getenv("RECOVER_SMOKE")!=nullptr;\n'
        '      for(int level:{2,3}) {')
    text = replace(text, 'const int total=level==2?49919:54878;',
                   'const int total=smoke?8:(level==2?49919:54878);')
    text = replace(text, 'need(raw.peek()==std::char_traits<char>::eof(),"capture trailing bytes");',
                   'if(!smoke)need(raw.peek()==std::char_traits<char>::eof(),"capture trailing bytes");')
    text = replace(text, 'need(captured.targets==104797,"captured coverage");',
                   'need(captured.targets==(std::getenv("RECOVER_SMOKE")?16:104797),"captured coverage");')
    text = replace(text, 'need(((captured.coverage|synthetic.coverage)&127)==127,"missing sweep branch coverage");',
        'if(!std::getenv("RECOVER_SMOKE"))need(((captured.coverage|synthetic.coverage)&127)==127,"missing sweep branch coverage");')
    return text


if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = identity()
    sealed = read(TRACE / 'manifest.json')
    dependencies = [prior.TOOL/'build.py', WP/'build.py', WP/'qualify.cu', WP/'diagnostic.hpp',
                    WP/'diagnostic_footer.inc', TRACE/'manifest.json', TRACE/'closure.json',
                    prior.PARENT/'build.json', prior.PARENT/'closure.json', prior.TOOL/'analyze.py']
    assert sha(prior.TOOL/'analyze.py') == sealed['tools'][str((prior.TOOL/'analyze.py').relative_to(ROOT))]
    for name in ['observe.so', 'replay', 'trace.so']:
        p = TRACE / name
        assert sha(p) == sealed['files'][name]
        dependencies.append(p)
    for name, expected in sealed['dependencies'].items():
        p = ROOT / name
        assert sha(p) == expected, name
        dependencies.append(p)
    for name in ['recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc']:
        dependencies.append(ROOT / 'fastkpc/build' / name)
    original = (ROOT/'fastkpc/src/cuda/full_cuda_ci_structured_eig.cu').read_text()
    commands = []
    for tile in [36, 16, 8]:
        for diagnostic in [False, True]:
            name = f'tile{tile}_'+('diag' if diagnostic else 'timing')
            folder = out/name
            folder.mkdir()
            text = source(original, tile, diagnostic)
            (folder/'kernel.cu').write_text(text)
            for include in ['recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc']:
                inc = (ROOT/'fastkpc/build'/include).read_text()
                (folder/include).write_text(wp.instrument(inc) if diagnostic else inc)
            if not diagnostic:
                (folder/'source.diff').write_text(''.join(difflib.unified_diff(
                    original.splitlines(True), text.splitlines(True), fromfile='current-production', tofile=name)))
            command = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false',
                '-DFASTKPC_STRUCTURED_FMAD_DISABLED=1', '-Xcompiler', '-fPIC', '-std=c++17',
                '-shared', '--cudart', 'shared', '-diag-suppress', '2464', '-Xptxas', '-v',
                '-I'+str(TOOL), '-I'+str(WP), '-I'+str(ROOT/'fastkpc/src/cuda'),
                str(folder/'kernel.cu'), '-o', str(out/(name+'.so'))]
            commands.append(command)
            with (out/(name+'-build.log')).open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            with (out/(name+'-resources.txt')).open('x') as log:
                subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(out/(name+'.so'))], stdout=log, check=True)
            print('BUILT', name, flush=True)
    (out/'qualify.cu').write_text(qualifier())
    command = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false', '-std=c++17',
        '--cudart', 'shared', '-I'+str(TOOL), '-I'+str(WP), '-I'+str(ROOT/'fastkpc/src/cuda'),
        str(out/'qualify.cu'), '-ldl', '-o', str(out/'qualify')]
    commands.append(command)
    with (out/'qualify-build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    identity()
    write(out/'build.json', dict(schema='isolated-recover-tile-build-v1', status='PASS',
        native_sha256=parent['native_sha256'], source_sha256=parent['source_sha256'],
        producer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        commands=commands, production_changes=False, tiles=[36,16,8], threads=64,
        tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(dependencies))},
        outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print('PASS isolated recover builds; current production unchanged', flush=True)
