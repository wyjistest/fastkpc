"""Qualify shared helper, real metadata, captured numerical/resource parity and SASS."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
os.chdir(root)
tool = Path(__file__).resolve().parent
campaign = Path(sys.argv[1]).resolve()
out = campaign/'checks'
out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((campaign/'build.json').read_text())
native = root/'fastkpc/build/fastkpc_cuda.so'
assert build['status']=='PASS' and sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64',
    FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1')
assert not any(env.get(k) for k in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_LAUNCH_BLOCKING'])
os.sched_setaffinity(0,range(20))
commands={}
def run(name,command,config=None):
    commands[name]=dict(command=command,mode=config)
    runenv=env.copy()
    runenv.pop('FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE',None)
    if config is not None:runenv['FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE']=config
    print('START',name,flush=True)
    with (out/(name+'.log')).open('x') as log:
        subprocess.run(command,env=runenv,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',name,flush=True)
inc=['-Ifastkpc/src','-I/usr/local/cuda/include']
links=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl']
native_links=[str(native),'-Wl,-rpath,'+str(native.parent),*links,
    '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR']
run('mock-build',['g++','-O2','-std=c++17','-pthread',*inc,str(tool/'mock.cpp'),
    'fastkpc/src/cuda/cuda_status.cpp','-o',str(out/'mock')])
for name,mode,expected in [('default',None,'0'),('empty','','0'),('off','0','0'),
                           ('on','1','1'),('invalid','true','invalid')]:
    run('mock-'+name,[str(out/'mock'),expected],mode)
run('native-build',['g++','-O2','-std=c++17','-pthread',*inc,str(tool/'native.cpp'),
    *native_links,'-o',str(out/'native')])
for mode in ['0','1']:run('native-'+mode,[str(out/'native')],mode)
run('native-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
    '--leak-check','full','--error-exitcode','99',str(out/'native')],'1')
mem=(out/'native-memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
cpp=shlex.split(subprocess.check_output(['R','CMD','config','--cppflags'],text=True))
rcpp=shlex.split(subprocess.check_output(['Rscript','-e','cat(Rcpp:::CxxFlags())'],text=True))
run('stats-build',['g++','-O2','-std=c++17','-shared','-fPIC',*inc,*cpp,*rcpp,
    str(tool/'stats.cpp'),*native_links,'-o',str(campaign/'stats.so')])

# Reuse the frozen, substantive L4 fixture: original cohort split, coefficients,
# residuals, rank/status, token lifetime, injected failures and leak accounting.
oldtool=root/'fastkpc/tools/research/optimizer_l4_slot_runtime_20261024'
audit=root/'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.cpp'
run('audit-build',['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,
    str(audit),*links,'-o',str(out/'audit.so')])
run('fault-build',['g++','-O2','-std=c++17','-shared','-fPIC',*inc,str(oldtool/'fault.cpp'),
    *links,'-o',str(out/'fault.so')])
run('qualify-build',['g++','-O2','-std=c++17','-pthread',*inc,str(oldtool/'qualify.cpp'),
    str(out/'audit.so'),str(out/'fault.so'),'-Wl,-rpath,'+str(out),*native_links,
    '-o',str(out/'qualify')])
capture=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023'
manifest=json.loads((capture/'manifest.json').read_text())
assert sha(capture/'manifest.json')=='d5c77e9c8bcd89c5ee5ac3fcd93fc8421a60231000bd355273b90e631d386ceb'
assert all(sha(capture/p)==h for p,h in manifest['files'].items() if p.startswith('raw/'))
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_slot_runtime_20261024'
reference=prior/'checks-v2/stage-a-reference.bin'
old_checks=json.loads((prior/'checks.json').read_text())
assert sha(reference)==old_checks['stage_a_reference_sha256']
args=[str(out/'qualify'),str(capture/'raw'),str(reference)]
for mode in ['0','1']:run('captured-'+mode,args,mode)
run('captured-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
    '--leak-check','full','--error-exitcode','99',*args],'1')
mem=(out/'captured-memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem

def sass(path):
    functions={};name=None;digest=None;words=0
    process=subprocess.Popen(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(path)],stdout=subprocess.PIPE,text=True)
    def finish():
        if name is not None:
            assert words>0 and name not in functions
            functions[name]=dict(words=words,sha256=digest.hexdigest())
    for line in process.stdout:
        if 'Function :' in line:
            finish();name=line.split('Function :')[1].strip()
            name=re.sub(r'\d+_GLOBAL__N__[0-9a-f]{8}_\d+_\w+_cu_[0-9a-f]{8}','4ANON',name)
            digest=hashlib.sha256();words=0
        elif name:
            for word in re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',line):
                digest.update((word+'\n').encode());words+=1
    finish();assert process.wait()==0 and functions
    return functions
objects={}
old=json.loads((prior/'build.json').read_text())
for name in build['rebuilt_objects']:
    if name=='cuda_status.o':continue
    a=prior/'build-source/fastkpc/build'/name
    b=campaign/'build-source/fastkpc/build'/name
    assert sha(a)==old['objects'][name] and sha(b)==build['objects'][name]
    before,after=sass(a),sass(b)
    assert before==after,name+' device instructions changed'
    objects[name]=dict(functions=after,device_exact=True)
    print('PASS SASS',name,len(after),flush=True)
with (out/'device-code.json').open('x') as stream:
    json.dump(dict(status='PASS',objects=objects),stream,indent=2);stream.write('\n')
assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
receipt=dict(status='PASS',native_sha256=sha(native),physical_gpu=1,commands=commands,
    helper_concurrency_and_faults=True,metadata_exact=True,default_disabled=True,
    captured_bitwise_and_resource_parity=True,memcheck_errors=0,memcheck_leaked_bytes=0,
    device_code_sha256=sha(out/'device-code.json'),stats_sha256=sha(campaign/'stats.so'),
    capture_manifest_sha256=sha(capture/'manifest.json'),stage_a_reference_sha256=sha(reference),
    dependencies={str(p.relative_to(root)):sha(p) for p in
                  [audit,audit.with_suffix('.hpp'),oldtool/'qualify.cpp',oldtool/'fault.cpp',
                   oldtool/'fault.hpp',reference,
                   root/'fastkpc/tools/research/optimizer_l4_diagnostic_20261023/fixture.hpp',
                   root/'fastkpc/tools/research/optimizer_windows_20261022/fixture.hpp']})
with (campaign/'checks.json').open('x') as stream:
    json.dump(receipt,stream,indent=2);stream.write('\n')
print('PASS all cache qualification checks',flush=True)
