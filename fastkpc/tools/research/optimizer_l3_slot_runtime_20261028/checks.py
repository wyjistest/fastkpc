"""Native L3 ownership, Stage A, failures, memcheck, configuration and device-code parity."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[4];os.chdir(root)
tool=Path(__file__).resolve().parent;campaign_out=Path(sys.argv[1]).resolve()
attempt=sys.argv[2] if len(sys.argv)>2 else 'checks'
assert Path(attempt).name==attempt
out=campaign_out/attempt;out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((campaign_out/'build.json').read_text());native=root/'fastkpc/build/fastkpc_cuda.so'
assert build['status']=='PASS' and sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64',
    FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
os.sched_setaffinity(0,range(20))
commands={}
def run(name,command):
    commands[name]=command;print('START',name,flush=True)
    with (out/(name+'.log')).open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',name,flush=True)
inc=['-Ifastkpc/src','-I/usr/local/cuda/include']
links=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl']
run('capacity-build',['g++','-O2','-std=c++17',*inc,str(tool/'test_capacity.cpp'),'-o',str(out/'test-capacity')])
run('capacity',[str(out/'test-capacity')])
run('audit-build',['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,
    'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.cpp',*links,'-o',str(out/'audit.so')])
run('fault-build',['g++','-O2','-std=c++17','-shared','-fPIC',*inc,str(tool/'fault.cpp'),*links,'-o',str(out/'fault.so')])
run('qualify-build',['g++','-O2','-std=c++17','-pthread',*inc,str(tool/'qualify.cpp'),
    str(out/'audit.so'),str(out/'fault.so'),str(native),'-Wl,-rpath,'+str(out),
    '-Wl,-rpath,'+str(native.parent),*links,'-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib',
    '-lR','-o',str(out/'qualify')])
capture=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027'
manifest=json.loads((capture/'capture-closure.json').read_text())
assert sha(capture/'capture-closure.json')=='ecfa7c0ef306722e04b5a27fa46c1637b3451a90bef8bda091f702b2d0dde315'
assert all(sha(capture/p)==h for p,h in manifest['files'].items())
prior=root/'fastkpc/artifacts/full_cuda_ci/optimizer_l6_slot_runtime_20261026_isolated_completion'
old=json.loads((prior/'build.json').read_text())
parent_native=prior/'build-source/fastkpc/build/fastkpc_cuda.so'
assert sha(parent_native)==old['native_sha256']==build['previous_native_sha256']
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
    finish();assert process.wait()==0 and functions;return functions

objects={}
cuda_rebuilt={'mgcv_multi_penalty_gcv_small64.o','full_cuda_ci_vertical.o','full_cuda_ci_structured_eig.o'}
for name,h in build['objects'].items():
    a=prior/'build-source/fastkpc/build'/name;b=campaign_out/'build-source/fastkpc/build'/name
    assert sha(a)==old['objects'][name] and sha(b)==h
    if name in cuda_rebuilt:
        before,after=sass(a),sass(b);assert before==after,name+' numerical instructions changed'
        objects[name]=dict(old_sass=before,new_sass=after,device_exact=True)
    elif name not in build['rebuilt_objects']:
        assert sha(a)==sha(b);objects[name]=dict(object_sha256=h,object_exact=True)
device=dict(status='PASS',objects=objects)
with (out/'device-code.json').open('x') as stream:json.dump(device,stream,indent=2);stream.write('\n')

run('reference-build',['g++','-O2','-std=c++17','-pthread',*inc,str(tool/'stage_a_reference.cpp'),
    str(parent_native),'-Wl,-rpath,'+str(parent_native.parent),*links,
    '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(out/'reference')])
run('reference',[str(out/'reference'),str(capture/'capture-raw'),str(out/'stage-a-reference.bin')])
args=[str(out/'qualify'),str(capture/'capture-raw'),str(out/'stage-a-reference.bin')]
run('qualify',args)
run('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99',*args])
mem=(out/'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in mem and '0 bytes leaked' in mem
tests=['fastkpc/tests/test_full_cuda_ci_l3_optimizer_slots.R',
       'fastkpc/tests/test_full_cuda_ci_l6_optimizer_slots.R',
       'fastkpc/tests/test_full_cuda_ci_l4_optimizer_slots.R',
       'fastkpc/tests/test_full_cuda_ci_l5_optimizer_slots.R',
       'fastkpc/tests/test_full_cuda_ci_l4_optimizer_workers.R']
for i,test in enumerate(tests):
    run('configuration-'+str(i),['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',test])

assert sha(native)==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
receipt=dict(status='PASS',native_sha256=sha(native),physical_gpu=1,commands=commands,
    capacities=list(range(2,34)),selected_requests=33,
    optimizer_and_stage_a_comparisons_per_run=int(re.search(r'compared=(\d+)',(out/'qualify.log').read_text()).group(1)),
    capacity_boundary_checks=15,partial_creation_cleanup=True,
    token_generation_and_old_destructor=True,partial_upload_drain=True,partial_detach_drain=True,
    parallel_cancellation=True,resource_closed=True,worker_resource_mutations=0,
    memcheck_errors=0,memcheck_leaked_bytes=0,device_code_sha256=sha(out/'device-code.json'),
    parent_native_sha256=sha(parent_native),stage_a_reference_sha256=sha(out/'stage-a-reference.bin'),
    stage_a_comparison_basis='frozen parent native, original cohort split and finalization; no tolerance change',
    outputs={str((out/p).relative_to(campaign_out)):sha(out/p) for p in ['audit.so','fault.so','qualify','reference']},
    dependencies={p:sha(root/p) for p in [
      'fastkpc/tools/research/optimizer_l3_slots_20261027/fixture.hpp',
      'fastkpc/tools/research/optimizer_windows_20261022/fixture.hpp',
      'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.cpp',
      'fastkpc/tools/research/optimizer_layer_refill_20261023/audit.hpp']},
    tests={p:sha(root/p) for p in tests},capture_manifest_sha256=sha(capture/'capture-closure.json'))
with (campaign_out/'checks.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print('PASS all native L3 runtime checks',flush=True)
