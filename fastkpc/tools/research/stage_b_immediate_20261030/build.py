"""Build source-bound isolated copies; no production mutation or GPU execution."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parent
WP=ROOT/'fastkpc/tools/research/stage_b_work_pair_20261028'
RT=ROOT/'fastkpc/tools/research/stage_b_recover_tile_20261029'
OLD=ROOT/'fastkpc/tools/research/stage_b_attribution_20261028'
SEALED=ROOT/'fastkpc/artifacts/full_cuda_ci/stage_b_attribution_20261028'
CAPTURE=SEALED/'attempt3/capture'
NATIVE=ROOT/'fastkpc/build/fastkpc_cuda.so'
EXPECTED='b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''): h.update(part)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def write(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2);f.write('\n')
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
wp=module('immediate_prior_coverage',WP/'build.py')

def identity():
    a=read(ROOT/'fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json')
    assert a['native_sha256']==EXPECTED and sha(NATIVE)==EXPECTED
    for name,expected in a['source_sha256'].items(): assert sha(ROOT/name)==expected,name
    for name,expected in a['generated_include_sha256'].items():
        assert sha(ROOT/'fastkpc/build'/name)==expected,name
    return a

def remove_blocks(text,needle,count,contains=None):
    positions=[];start=0
    while True:
        begin=text.find(needle,start)
        if begin<0:break
        brace=text.index('{',begin);depth=1;end=brace+1
        while depth:
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        if contains is None or contains in text[brace:end]:positions.append((begin,end))
        start=end
    assert len(positions)==count,(needle,len(positions),count)
    for begin,end in reversed(positions):text=text[:begin]+text[end:]
    return text

def candidate(text):
    # Entry guard already rejects these configurations. Preserve all remaining
    # parameter validation in its original order, including ldu/ldc checks.
    text=remove_blocks(text,'if (*nru > 0) {',7)
    text=remove_blocks(text,'if (*ncc > 0) {',7)
    text=remove_blocks(text,'if (! rotate) {',1)
    text=remove_blocks(text,'if (*ncvt > 0) {',4,'record_dlasr_')
    text=replace(text,'doublereal *work, integer *info','doublereal *unused_work, integer *info')
    text=replace(text,'    --work;\n','')
    text=replace(text,'\t    work[i__] = cs;\n\t    work[nm1 + i__] = sn;\n','')
    families=[(True,'cs','sn'),(False,'oldcs','-oldsn'),
              (True,'cosr','sinr'),(False,'cosl','-sinl')]
    for forward,c,s in families:
        index='i__ - ll + 1' if forward else 'i__ - ll'
        old=f'\t\twork[{index}] = {c};\n\t\twork[{index} + nm1] = {s};'
        left='i__ + vt_dim1' if forward else 'i__ - 1 + vt_dim1'
        right='i__ + 1 + vt_dim1' if forward else 'i__ + vt_dim1'
        text=replace(text,old,f'\t\tip_consume_sweep(log, &vt[{left}], &vt[{right}], {c}, {s});')
    code=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    assert not re.search(r'\bwork\b',code), 'candidate retains a work expression'
    assert 'record_dlasr_' not in text
    assert text.count('ip_consume_sweep(')==4
    return text

def checkpoints(text):
    text=wp.instrument(text)
    base='log, {stage}, *n, &d__[1], &e[1], &vt[vt_offset], *info'
    text=replace(text,'/*     Compute singular values to relative accuracy TOL */',
                 '    ip_mark('+base.format(stage=1)+');\n\n/*     Compute singular values to relative accuracy TOL */')
    text=replace(text,'\tm += -2;\n\tgoto L60;',
                 '\tm += -2;\n\tip_mark('+base.format(stage=3)+', ll,m,idir,iter,iterdivn,oldll,oldm,tol,thresh);\n\tgoto L60;')
    text=replace(text,'/*     QR iteration finished, go back and check convergence */',
                 '    ip_mark('+base.format(stage=2)+', ll,m,idir,iter,iterdivn,oldll,oldm,tol,thresh,shift);\n\n'
                 '/*     QR iteration finished, go back and check convergence */')
    text=replace(text,'/*     Sort the singular values into decreasing order (insertion sort on */',
                 '    ip_mark('+base.format(stage=4)+');\n\n/*     Sort the singular values into decreasing order (insertion sort on */')
    text=replace(text,'    goto L220;', '    ip_mark('+base.format(stage=5)+');\n    goto L220;')
    text=replace(text,'L220:\n', '    ip_mark('+base.format(stage=6)+');\nL220:\n')
    return text

MARK='''
__device__ void ip_mark(RotationLog* log,int stage,int r,const double* d,const double* e,
    const double* q,int info,int ll=0,int m=0,int direction=0,int iter=0,int iterdivn=0,
    int oldll=0,int oldm=0,double tol=0,double threshold=0,double shift=0) {
  ip_checkpoint_emit(log->checkpoint,stage,r,log->sweeps,ll,m,direction,iter,iterdivn,
    oldll,oldm,info,log->count,log->overflow,int(log->coverage),tol,threshold,shift,d,e,q);
}
'''

def kernel_source(original,is_candidate,diagnostic):
    factor=3 if is_candidate else 5
    text=replace(original,'#define TRUE_ 1',
                 '#include "rotation.inc"\n'+(MARK if diagnostic else '')+'\n#define TRUE_ 1')
    text=replace(text,'    if (c->tile < 1) throw std::runtime_error("insufficient structured shared memory");',
        '    if (c->tile < 1) throw std::runtime_error("insufficient structured shared memory");\n'
        '    if (c->mode != StructuredDbdsqrMode::FixedVtWorkPair)\n'
        '      throw std::runtime_error("isolated immediate test requires current work-pair mode");')
    if is_candidate:
        text=replace(text,'  double* work = q+n;',
                     '  double* work = Mode==StructuredDbdsqrMode::FixedVtWorkPair ? nullptr : q+n;')
    if not diagnostic:
        return f'#define IP_FACTOR {factor}\n'+text+'\n#include "timing_footer.inc"\n'
    text='#include "diagnostic.hpp"\n'+text
    text=replace(text,'  int count, capacity, overflow;',
        '  int count, capacity, overflow;\n  unsigned coverage=0;\n'
        '  int sweeps=0,previous_direction=0,force_nonconvergence=0;\n  CheckpointCursor checkpoint;')
    text=replace(text,'Rotation* rotations, int capacity) {',
        'Rotation* rotations, int capacity, WorkPairState* diag, double* deq,\n'
        '    CheckpointHeader* headers,unsigned long long* cp_values,CheckpointResult* cp_results,\n'
        '    const CheckpointResult* cp_reference,int cp_capacity,int cp_compare) {')
    marker='  double* work = Mode==StructuredDbdsqrMode::FixedVtWorkPair ? nullptr : q+n;' if is_candidate else '  double* work = q+n;'
    text=replace(text,marker,marker+'\n'
        '  double* guard=local+(Mode==StructuredDbdsqrMode::FixedVtWorkPair ? IP_FACTOR : 7)*n;\n'
        '  for(int j=threadIdx.x;j<8;j+=blockDim.x) guard[j]=0x1.2345p+9;\n  __syncthreads();')
    text=replace(text,'  RotationLog log{rotations+size_t(component)*capacity, q, 0, capacity, 0};',
        '  RotationLog log{rotations+size_t(component)*capacity, q, 0, capacity, 0};\n'
        '  log.checkpoint={headers+size_t(component)*cp_capacity,\n'
        '    cp_values+size_t(component)*cp_capacity*3*n,cp_results+component,\n'
        '    cp_reference ? cp_reference+component : nullptr,cp_capacity,n,cp_compare};')
    text=replace(text,'  z.rotations = log.count;',
        '  fastkpc::lapack312::ip_mark(&log,7,r,d,e,q,info);\n'
        '  diag[component]={info,log.count,log.overflow,int(log.coverage),log.sweeps,0,r,0};\n'
        '  for(int j=0;j<8;++j) if(guard[j]!=0x1.2345p+9)diag[component].canary=1;\n'
        '  for(int j=0;j<r;++j){deq[size_t(component)*3*n+j]=d[j];\n'
        '    deq[size_t(component)*3*n+n+j]=e[j];deq[size_t(component)*3*n+2*n+j]=q[j];}\n'
        '  z.rotations = log.count;')
    text=replace(text,'  int n, k, count, capacity, tile;',
        '  int n, k, count, capacity, tile;\n  WorkPairState* diag; double* deq;\n'
        '  CheckpointHeader* cp_headers=nullptr;unsigned long long* cp_values=nullptr;\n'
        '  CheckpointResult* cp_results;const CheckpointResult* cp_reference=nullptr;\n'
        '  int cp_capacity=0,cp_compare=0;')
    text=replace(text,'    c->n=n; c->k=k+1;',
        '    c->diag=static_cast<WorkPairState*>(allocate(owner,count*sizeof(WorkPairState)));\n'
        '    c->deq=static_cast<double*>(allocate(owner,size_t(count)*3*n*sizeof(double)));\n'
        '    c->cp_results=static_cast<CheckpointResult*>(allocate(owner,count*sizeof(CheckpointResult)));\n'
        '    c->n=n; c->k=k+1;')
    text=replace(text,'const auto shared_bytes = structured_dbdsqr_shared_bytes(c->mode,n);',
                 'const auto shared_bytes = structured_dbdsqr_shared_bytes(c->mode,n)+8*sizeof(double);')
    text=replace(text,'bidiagonal<Mode><<<count,32,shared_bytes,c->stream>>>',
                 'bidiagonal<Mode><<<count,32,shared_bytes+8*sizeof(double),c->stream>>>')
    text=replace(text,'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity);',
        'c->n,c->data,c->ints,c->meta,c->rotations,c->capacity,c->diag,c->deq,\n'
        '    c->cp_headers,c->cp_values,c->cp_results,c->cp_reference,c->cp_capacity,c->cp_compare);')
    return f'#define IP_FACTOR {factor}\n#define IP_CANDIDATE {int(is_candidate)}\n'+text+\
        '\n#include "diagnostic_footer.inc"\n#include "diagnostic_extra.inc"\n'

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    att=identity();sealed=read(SEALED/'manifest.json')
    deps=[ROOT/'fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json',WP/'build.py',
          RT/'diagnostic_extra.inc',RT/'boundary_cases.inc',SEALED/'manifest.json',SEALED/'closure.json',
          ROOT/'fastkpc/artifacts/full_cuda_ci/fixed_vt_dbdsqr_binary_20261030/indexed/fixed.sass']
    for level in [2,3]:
        for suffix in ['residuals.f64','targets.tsv','payload-sha256.txt']:
            p=CAPTURE/f'l{level}-{suffix}'
            assert sha(p)==sealed['files'][str(p.relative_to(SEALED))],p
            deps.append(p)
    for name in ['common.hpp','observe.cpp','replay.cpp']:
        p=OLD/name;assert sha(p)==sealed['tools'][str(p.relative_to(ROOT))];deps.append(p)
    symbols=SEALED/'attempt3/symbols.inc'
    assert sha(symbols)==sealed['files']['attempt3/symbols.inc'];deps.append(symbols)
    replay_source=ROOT/'fastkpc/tools/research/stage_b_work_pair_trace_20261028/replay.cpp'
    deps.append(replay_source)
    original=(ROOT/'fastkpc/src/cuda/full_cuda_ci_structured_eig.cu').read_text()
    generated=(ROOT/'fastkpc/build/recorded_dbdsqr_fixed_vt.inc').read_text()
    converted=candidate(generated)
    (out/'candidate.diff').write_text(''.join(difflib.unified_diff(generated.splitlines(True),converted.splitlines(True),
        fromfile='accepted-fixed-5N',tofile='isolated-immediate-3N')))
    commands=[]
    def run(label,command):
        commands.append(command)
        with (out/(label+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    for is_candidate in [False,True]:
        for diagnostic in [True,False]:
            name=('candidate' if is_candidate else 'baseline')+('_diag' if diagnostic else '_timing')
            folder=out/name;folder.mkdir()
            text=kernel_source(original,is_candidate,diagnostic)
            (folder/'kernel.cu').write_text(text)
            for header in ['full_cuda_ci_structured_eig.hpp','full_cuda_ci_structured_launch.hpp']:
                content=(ROOT/'fastkpc/src/cuda'/header).read_text()
                if is_candidate and header.endswith('launch.hpp'):
                    content=replace(content,'FixedVtWorkPair: return std::size_t(5)',
                                    'FixedVtWorkPair: return std::size_t(3)')
                (folder/header).write_text(content)
            inc=converted if is_candidate else generated
            (folder/'recorded_dbdsqr_fixed_vt.inc').write_text(checkpoints(inc) if diagnostic else inc)
            (folder/'recorded_dbdsqr.inc').write_text((ROOT/'fastkpc/build/recorded_dbdsqr.inc').read_text())
            (folder/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
                fromfile='accepted-production',tofile=name)))
            cmd=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false',
                 '-DFASTKPC_STRUCTURED_FMAD_DISABLED=1','-Xcompiler','-fPIC','-std=c++17',
                 '-shared','--cudart','shared','-diag-suppress','2464','-Xptxas','-v',
                 '-I'+str(TOOL),'-I'+str(RT),'-I'+str(ROOT/'fastkpc/src/cuda'),
                 str(folder/'kernel.cu'),'-o',str(out/(name+'.so'))]
            run(name+'-build',cmd)
            run(name+'-resources',['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(out/(name+'.so'))])
            if not diagnostic:
                run(name+'-sass',['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out/(name+'.so'))])
            print('BUILT',name,flush=True)
    run('qualify-build',['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-std=c++17',
        '--cudart','shared','-I'+str(TOOL),'-I'+str(ROOT/'fastkpc/src/cuda'),
        str(TOOL/'qualify.cu'),'-ldl','-o',str(out/'qualify')])
    includes=['-I'+str(ROOT/'fastkpc/src'),'-I'+str(OLD),'-I'+str(SEALED/'attempt3'),'-I/usr/local/cuda/include']
    libs=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcrypto','-ldl']
    base=['g++','-O2','-std=c++17','-pthread',*includes]
    run('observe-build',base+['-fPIC','-shared',str(OLD/'observe.cpp'),*libs,'-o',str(out/'observe.so')])
    run('replay-build',base+[str(replay_source),str(NATIVE),'-Wl,-rpath,'+str(NATIVE.parent),*libs,
        '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(out/'replay')])
    run('binary-audit',[sys.executable,str(TOOL/'resources.py'),str(out),'--offline'])
    identity()
    write(out/'build.json',dict(schema='isolated-immediate-build-v1',status='PASS',native_sha256=EXPECTED,
        source_sha256=att['source_sha256'],producer_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        generated_include_sha256=att['generated_include_sha256'],commands=commands,
        tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
        dependencies={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(deps))},
        outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        production_changes=False,baseline_factor=5,candidate_factor=3,
        nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)))
    print('PASS isolated build; accepted native unchanged',flush=True)

if __name__=='__main__':main()
