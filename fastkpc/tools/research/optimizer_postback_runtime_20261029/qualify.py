"""Freeze and qualify the production helper and both runtime backends on GPU1."""
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import ROOT,TOOL,BASE,PRIOR,sha,read,write,environment,processes,job

mode,attempt=sys.argv[1:3]
out=BASE/attempt
qa=out/(sys.argv[3] if len(sys.argv)==4 else 'qualification')
build=read(out/'build.json')
native=out/'fastkpc_cuda.so'
assert sha(native)==build['native_sha256']
for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
l4=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023'
l5=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_layer_refill_20261023'
l3=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'

def log_run(command,name):
    with (qa/(name+'.log')).open('x') as log:
        subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)

if mode=='build':
    qa.mkdir()
    shared=['optimizer_l4_diagnostic_20261023/fixture.hpp',
      'optimizer_layer_refill_20261023/layer_fixture.hpp',
      'optimizer_layer_refill_20261023/audit.hpp','optimizer_layer_refill_20261023/audit.cpp',
      'optimizer_windows_20261022/fixture.hpp']
    originals=[TOOL/name for name in ['unit.cu','cohorts.cpp','contracts.cpp','qualify.py','support.py']]
    originals += [TOOL.parent/p for p in shared]
    originals += [TOOL.parent/'stage_b_work_pair_runtime_20261028/support.py']
    tool_identity={str(p.relative_to(ROOT)):sha(p) for p in originals}
    for p in originals:
        dest=qa/'frozen'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copytree(ROOT/'fastkpc/src',qa/'frozen/fastkpc/src')
    write(qa/'inputs.json',dict(epoch=time.time(),tools=tool_identity,production=build['source_sha256'],
        native_sha256=build['native_sha256']))
    ftools=qa/'frozen/fastkpc/tools/research'
    ftool=ftools/TOOL.name
    inc=['-I'+str(qa/'frozen/fastkpc/src'),'-I/usr/local/cuda/include']
    lib=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl',
      '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRlapack','-lRblas']
    commands=[['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,
        str(ftools/'optimizer_layer_refill_20261023/audit.cpp'),*lib,'-o',str(qa/'audit.so')],
      ['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','--fmad=false','-std=c++17',
        '-diag-suppress','2464',*inc,str(ftool/'unit.cu'),'-o',str(qa/'unit')]]
    for name in ['cohorts','contracts']:
        commands.append(['g++','-O2','-std=c++17','-pthread',*inc,str(ftool/(name+'.cpp')),
          str(qa/'audit.so'),str(native),'-Wl,-rpath,'+str(qa),'-Wl,-rpath,'+str(out),*lib,'-o',str(qa/name)])
    write(qa/'build-protocol.json',dict(commands=commands,inputs_sha256=sha(qa/'inputs.json')))
    for i,command in enumerate(commands):
        print('START harness build',i,flush=True);log_run(command,'build-'+str(i))
    resources={}
    for name,path in [('original',BASE/'baseline/objects/mgcv_multi_penalty_gcv_small64.o'),
                      ('postback',out/'postback.o'),('old-native',BASE/'baseline/fastkpc_cuda.so'),('native',native)]:
        raw=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','-res-usage',str(path)],text=True)
        (qa/(name+'-resources.txt')).write_text(raw)
        resources[name]=[(f.strip(),r.strip()) for f,r in re.findall(r'Function ([^\n]+)\n\s*(REG:[^\n]+)',raw)]
    for name,stack,evaluator in [('original',7120,288),('postback',7232,320)]:
        for prefix,expected in [('optimize_multi_penalty_targets_kernel',stack),('evaluate_multi_penalty_targets_kernel',evaluator)]:
            matches=[r for f,r in resources[name] if prefix in f]
            assert len(matches)==1 and re.search(r'REG:255 STACK:'+str(expected)+r'\b',matches[0]),(name,matches)
    assert not Counter(resources['old-native'])-Counter(resources['native']),'original device resources changed'
    write(qa/'static-gates.json',dict(status='PASS',off_stack=[7120,288],on_stack=[7232,320],
      all_original_device_resources_preserved=True,original_small64_sha256=sha(BASE/'baseline/objects/mgcv_multi_penalty_gcv_small64.o')))
    write(qa/'build.json',dict(status='PASS',tools=tool_identity,native_sha256=sha(native),
      executables={name:sha(qa/name) for name in ['audit.so','unit','cohorts','contracts']}))
    print('PASS static resources and harness build',flush=True)
elif mode=='run':
    qbuild=read(qa/'build.json');assert qbuild['status']=='PASS'
    assert read(qa/'static-gates.json')['status']=='PASS'
    deps={str(native):sha(native)}
    for p,h in qbuild['tools'].items():assert sha(ROOT/p)==h,p;deps[str(ROOT/p)]=h
    for p,h in qbuild['executables'].items():assert sha(qa/p)==h;deps[str(qa/p)]=h
    for capture,count in [(l4,34),(l5,15)]:
        manifest=read(capture/'manifest.json')
        files={p:h for p,h in manifest['files'].items() if p.startswith('raw/') and p.endswith('.bin')}
        assert len(files)==count
        for p,h in files.items():assert sha(capture/p)==h,p;deps[str(capture/p)]=h
        deps[str(capture/'manifest.json')]=sha(capture/'manifest.json')
    lm=read(l3/'manifest.json');f='raw/L3-W31.bin';assert sha(l3/f)==lm['files'][f];deps[str(l3/f)]=sha(l3/f)
    prior=read(PRIOR/'manifest.json');f='blocks.tsv';assert sha(PRIOR/f)==prior['files'][f]
    deps[str(PRIOR/f)]=sha(PRIOR/f)
    def frozen():
        for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
        for p,h in deps.items():assert sha(p)==h,p
    env=environment();env['FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H']='1';frozen()
    assert not processes(),'external compute; no qualification started'
    unit=[str(qa/'unit'),str(PRIOR/'blocks.tsv')]
    contract=[str(qa/'contracts'),str(l4/'raw'),str(l5/'raw'),str(l3/'raw/L3-W31.bin')]
    sequence=[('unit-full',unit,None,0),('unit-memcheck',unit+['--reduced'],'memcheck',0),
      ('unit-synccheck',unit+['--reduced'],'synccheck',0),('contracts',contract,None,0),
      ('contracts-memcheck',contract,'memcheck',0),('contracts-synccheck',contract,'synccheck',0)]
    for name,flag,kind,sanitizer in [('off-reduced',0,'reduced',None),('on-reduced',1,'reduced',None),
      ('on-memcheck',1,'reduced','memcheck'),('on-synccheck',1,'reduced','synccheck'),
      ('off-full',0,'verify',None),('on-full',1,'verify',None)]:
        command=[str(qa/'cohorts'),str(l4/'raw'),str(l5/'raw'),str(qa/name),kind]
        sequence.append((name,command,sanitizer,flag))
    write(qa/'run-protocol.json',dict(schema='postback-runtime-qualification-v1',epoch=time.time(),dependencies=deps,
      sequence=sequence,expected_cases=20628,expected_sanitizer_cases=1719,level_counts={'4':[2154,17767],'5':[955,6816]},
      same_native_off_on=True,GPU_numerical_authority_only=True,original_order_and_slots=True))
    records=[]
    for name,command,sanitizer,flag in sequence:
        current=env.copy();current['FASTKPC_DCC_GAMMA_VECTOR_POSTBACK']=str(flag)
        if sanitizer:command=['/usr/local/cuda/bin/compute-sanitizer','--tool',sanitizer,'--error-exitcode','86',
          *(['--leak-check','full'] if sanitizer=='memcheck' else []),*command]
        print('START',name,time.strftime('%H:%M:%S'),flush=True);job(qa,name,command,current,frozen)
        log=(qa/(name+'.log')).read_text()
        if sanitizer:assert 'ERROR SUMMARY: 0 errors' in log,name
        if name.startswith('unit'):
            result=json.loads(next(line for line in log.splitlines() if line.startswith('{')))
            assert result['cases']==result['expected']==(20628 if name=='unit-full' else 1719)
            assert all(result[k]==0 for k in ['bit_mismatches','readonly_or_guard_changes','finite_output_nan_count','info_errors'])
            write(qa/(name+'.json'),result)
        elif name.startswith('contracts'):assert 'PASS all runtime contracts' in log
        else:
            layers=[read(qa/(name+'-L'+str(level)+'.json')) for level in [4,5]]
            assert all(x['status']=='PASS' and x['resource_closed'] and x['vector_postback']==flag for x in layers)
            if name.endswith('full'):assert [(x['setups'],x['targets']) for x in layers]==[(2154,17767),(955,6816)]
        records.append(name)
    write(qa/'closure.json',dict(status='PASS_RUNTIME_QUALIFICATION',native_sha256=sha(native),runs=records,
      full_cases=20628,sanitizer_cases=1719,full_cohorts=[[2154,17767],[955,6816]],production_default=False))
    print('PASS runtime qualification',flush=True)
else:raise ValueError(mode)
