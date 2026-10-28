"""Build/capture/close all original L3 windows on the unchanged production native."""
from pathlib import Path
import subprocess
import sys
from common import ROOT,TOOL,NATIVE,PARENT,sha,load,write_new,source_identity,environment,monitored

out=Path(sys.argv[1]).resolve();mode=sys.argv[2];out.mkdir(parents=True,exist_ok=True)
identity=source_identity()
if mode=='build':
    assert not (out/'capture-build.json').exists()
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(NATIVE)],text=True).splitlines()
    mangled=[r.split()[-1] for r in symbols]
    demangled=subprocess.check_output(['c++filt'],input='\n'.join(mangled),text=True).splitlines()
    wanted={'plan':'make_layer_plan(', 'create':'fastkpc::create_multi_penalty_gcv_capacity_prepared(',
        'multi':'fastkpc::multi_penalty_gcv_capacity_optimize_multi(',
        'small':'fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64('}
    with (out/'symbols.inc').open('x') as stream:
        for key,prefix in wanted.items():
            matches=[m for m,d in zip(mangled,demangled) if d.startswith(prefix)]
            assert len(matches)==1,(key,matches)
            stream.write('#define SYM_'+key+' "'+matches[0]+'"\n')
    includes=['-I'+str(ROOT/'fastkpc/src'),'-I'+str(TOOL),'-I'+str(out),'-I/usr/local/cuda/include']
    links=[str(NATIVE),'-Wl,-rpath,'+str(NATIVE.parent),'-L/usr/local/cuda/lib64',
        '-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-ldl','-L/usr/local/lib/R/lib',
        '-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRlapack','-lRblas']
    commands=[['g++','-O2','-std=c++17','-shared','-fPIC',*includes,str(TOOL/'capture.cpp'),*links,'-o',str(out/'capture.so')],
        ['g++','-O2','-std=c++17',*includes,str(TOOL/'inspect.cpp'),*links,'-o',str(out/'inspect')]]
    for cmd in commands:subprocess.run(cmd,check=True)
    deps=[TOOL/p for p in ['common.py','capture.py','capture.cpp','fixture.hpp','inspect.cpp','README.md']]
    deps += [PARENT/'build.json',TOOL.parent/'optimizer_windows_20261022/fixture.hpp',
        TOOL.parent/'optimizer_l6_slot_runtime_20261026/run.R',TOOL.parent/'exact_speedup_20261020/pinned_optimizer_gates.R']
    write_new(out/'build.json',identity)
    write_new(out/'capture-build.json',dict(native_sha256=sha(NATIVE),commands=commands,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in deps},
        outputs={p:sha(out/p) for p in ['capture.so','inspect','symbols.inc','build.json']}))
elif mode=='run':
    b=load(out/'capture-build.json')
    for p,h in b['inputs'].items():assert sha(ROOT/p)==h,p
    for p,h in b['outputs'].items():assert sha(out/p)==h,p
    (out/'capture-raw').mkdir();env=environment()
    env.update(LD_PRELOAD=str(out/'capture.so'),FASTKPC_L3_CAPTURE=str(out/'capture-raw'),
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
    reference=PARENT/'on3/receipt.rds'
    monitored(out,'capture',['Rscript','--vanilla',str(TOOL.parent/'optimizer_l6_slot_runtime_20261026/run.R'),
        str(out/'capture'),'1',str(reference),str(reference)],env,
        dict(evidence_class='diagnostic capture only; elapsed is not performance evidence',
             capture_build_sha256=sha(out/'capture-build.json')))
elif mode=='verify':
    p=load(out/'capture-process.json');r=load(out/'capture/report.json')
    assert p['exit_code']==0 and p['isolated'] and all(r['gates'].values())
    census=__import__('json').loads(subprocess.check_output([str(out/'inspect'),str(out/'capture-raw')],text=True))
    assert (census['windows'],census['setups'],census['targets'])==(64,4065,54878)
    assert max(map(int,census['capacities']))==33 and census['total_root_rank']=={'24':4065}
    files=list((out/'capture-raw').glob('*.bin'))
    assert len(files)==64
    files += [out/'capture/report.json',out/'capture/receipt.rds',out/'capture-process.json',out/'capture-build.json']
    write_new(out/'capture-closure.json',dict(status='PASS',native_sha256=sha(NATIVE),census=census,
        canonical_hard_gates=r['gates'],evidence_class='diagnostic captured cohort; no performance claim',
        files={str(p.relative_to(out)):sha(p) for p in files}))
    print(census,flush=True)
else:raise ValueError(mode)
