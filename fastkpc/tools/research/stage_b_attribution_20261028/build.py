"""Build only host diagnostic observers/replay; link the accepted native directly."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
TOOL=Path(__file__).resolve().parent
PARENT=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028'
NATIVE=ROOT/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def identity():
    b=json.loads((PARENT/'build.json').read_text())
    assert sha(NATIVE)==b['native_sha256']
    assert all(sha(ROOT/p)==h for p,h in b['source_sha256'].items())
    return b

if __name__=='__main__':
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
    b=identity();assert not (out/'build.json').exists()
    runner=ROOT/'fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/run.R'
    text=runner.read_text()
    token='Sys.setenv(FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="1",'
    assert text.count(token)==1
    text=text.replace(token,'evidence_class <- "canonical diagnostic capture; extra observer D2H; not performance evidence"\n'+token)
    text=text.replace('optimizer-l3-slot-runtime-canonical-v1','stage-b-canonical-capture-v1')
    text=text.replace('sha("fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/run.R")',
                      'sha('+json.dumps(str(out/'capture_runner.R'))+')')
    with (out/'capture_runner.R').open('x') as f:f.write(text)
    raw=subprocess.check_output(['nm','-D','--defined-only',str(NATIVE)],text=True)
    symbols=[r.split()[-1] for r in raw.splitlines()]
    names=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
    with (out/'symbols.inc').open('x') as f:
        for short,term in [('begin','begin_full_cuda_ci_level_component_prefill'),
                           ('store','store_full_cuda_ci_level_component_residual'),
                           ('build','build_full_cuda_ci_level_components'),
                           ('info','full_cuda_ci_legacy_eig_component_cache_info')]:
            found=[s for s,n in zip(symbols,names) if 'fastkpc::'+term+'(' in n]
            assert len(found)==1
            f.write('constexpr const char* SYM_'+short+' = "'+found[0]+'";\n')
    includes=['-I'+str(ROOT/'fastkpc/src'),'-I'+str(TOOL),'-I'+str(out),'-I/usr/local/cuda/include']
    libs=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcrypto','-ldl']
    base=['g++','-O2','-std=c++17','-pthread',*includes]
    commands=[base+['-fPIC','-shared',str(TOOL/'observe.cpp'),*libs,'-o',str(out/'observe.so')],
      base+[str(TOOL/'replay.cpp'),str(NATIVE),'-Wl,-rpath,'+str(NATIVE.parent),*libs,
            '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(out/'replay')],
      base+['-fPIC','-shared',str(TOOL/'trace.cpp'),*libs,'-lcupti','-o',str(out/'trace.so')],
      base+[str(TOOL/'trace_smoke.cpp'),*libs,'-o',str(out/'trace-smoke')]]
    for i,cmd in enumerate(commands):
        with (out/f'build-{i}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    identity()
    write(out/'build.json',dict(status='PASS',native_sha256=b['native_sha256'],
        source_sha256=b['source_sha256'],parent_build_sha256=sha(PARENT/'build.json'),
        commands=commands,kernels_recompiled=False,
        runner_parent=str(runner.relative_to(ROOT)),runner_parent_sha256=sha(runner),
        tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
        outputs={p:sha(out/p) for p in ['symbols.inc','observe.so','replay','trace.so','trace-smoke','capture_runner.R']}))
    print('PASS unchanged native, host-only diagnostics',flush=True)
