"""Compile independent ON backend, then link production source with original OFF objects."""
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import ROOT, TOOL, BASE, sha, read, write

mode, attempt = sys.argv[1:]
out = BASE / attempt
baseline = read(BASE / 'baseline/identity.json')

def command_log(command, path):
    with path.open('x') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

if mode == 'device':
    out.mkdir()
    shutil.copytree(ROOT / 'fastkpc/src', out / 'source',
        ignore=shutil.ignore_patterns('*.o', '*.so', '*.a', '*.d'))
    (out / 'generated').mkdir()
    for name in ['recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc']:
        shutil.copyfile(ROOT / 'fastkpc/build' / name, out / 'generated' / name)
    flags = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '--fmad=false',
        '-diag-suppress', '2464', '-Xcompiler', '-fPIC', '-std=c++17',
        '-I'+str(out/'source'), '-I'+str(out/'source/cuda'), '-I'+str(out/'generated')]
    source = out/'source/cuda/mgcv_multi_penalty_gcv_postback.cu'
    raw = subprocess.check_output([*flags, '-M', str(source)], text=True)
    local = {}
    for name in shlex.split(raw.replace('\\\n', ' '))[1:]:
        p = Path(name)
        if p.is_relative_to(out/'source'):
            local[str(p.relative_to(out/'source'))] = sha(p)
    cmd = [*flags, '-Xptxas=-v', '-c', str(source), '-o', str(out/'postback.o')]
    write(out/'device-protocol.json', dict(command=cmd, source_sha256=local,
        build_tool_sha256=sha(Path(__file__)), epoch=time.time()))
    print('START independent ON compilation', flush=True)
    command_log(cmd, out/'device-build.log')
    write(out/'device-build.json', dict(status='PASS', object_sha256=sha(out/'postback.o'),
        source_sha256=local, protocol_sha256=sha(out/'device-protocol.json')))
    print('PASS independent ON compilation', flush=True)
elif mode == 'link':
    device = read(out/'device-build.json')
    for p,h in device['source_sha256'].items():
        assert sha(ROOT/'fastkpc/src'/p)==h,p
    assert sha(out/'postback.o')==device['object_sha256']
    cppflags=shlex.split(subprocess.check_output(['R','CMD','config','--cppflags'],text=True))
    cxx=shlex.split(subprocess.check_output(['R','CMD','config','CXX17'],text=True))
    flags=shlex.split(subprocess.check_output(['R','CMD','config','CXX17FLAGS'],text=True))
    includes=shlex.split(subprocess.check_output(['Rscript','--vanilla','-e','cat(Rcpp:::CxxFlags())'],text=True))
    for package in ['RcppArmadillo','RcppEigen','RSpectra']:
        inc=subprocess.check_output(['Rscript','--vanilla','-e',
            'cat(system.file("include", package="'+package+'"))'],text=True)
        includes.append('-I'+inc)
    common=[*cxx,'-std=c++17',*flags,'-fPIC',*cppflags,*includes,
        '-I/usr/local/cuda/include','-I'+str(ROOT/'fastkpc/src'),'-I'+str(ROOT/'fastkpc/src/cuda')]
    replacements={'mgcv_multi_penalty_gcv_capacity.o':'cuda/mgcv_multi_penalty_gcv_capacity.cpp',
                  'full_cuda_ci_one_call.o':'full_cuda_ci_one_call.cpp'}
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'fastkpc/src').rglob('*')
        if p.is_file() and p.suffix in ['.cpp','.hpp','.cu','.cuh','.inc','.txt']}
    for p in ['fastkpc/tools/build_cuda_native.sh','fastkpc/tools/generate_recorded_dbdsqr.py']:
        sources[p]=sha(ROOT/p)
    commands=[]
    for obj,source in replacements.items():
        commands.append([*common,'-c',str(ROOT/'fastkpc/src'/source),'-o',str(out/obj)])
    objects=[]
    for name,h in baseline['objects'].items():
        assert sha(BASE/'baseline/objects'/name)==h,name
        objects.append(out/name if name in replacements else BASE/'baseline/objects'/name)
    objects.append(out/'postback.o')
    libs=[]
    for name in ['LAPACK_LIBS','BLAS_LIBS','FLIBS']:
        libs+=shlex.split(subprocess.check_output(['R','CMD','config',name],text=True))
    commands.append([*cxx,'-shared','-pthread','-o',str(out/'fastkpc_cuda.so'),*map(str,objects),*libs,
        '-lcrypto','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-lcusolver',
        '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR'])
    write(out/'link-protocol.json',dict(commands=commands,source_sha256=sources,
        previous_native_sha256=baseline['native_sha256'],original_objects=baseline['objects'],epoch=time.time()))
    for i,command in enumerate(commands):
        print('START host/link',i,flush=True)
        command_log(command,out/('link-'+str(i)+'.log'))
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    write(out/'build.json',dict(status='PASS',native_sha256=sha(out/'fastkpc_cuda.so'),
        source_sha256=sources,previous_native_sha256=baseline['native_sha256'],
        objects={str(p):sha(p) for p in objects},device_build_sha256=sha(out/'device-build.json'),
        evidence_class='default-off runtime source closure, not clean producer or promotion'))
    print('PASS linked runtime',sha(out/'fastkpc_cuda.so'),flush=True)
else:
    raise ValueError(mode)
