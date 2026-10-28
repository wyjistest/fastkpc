"""Attribute the same rejected gate; do not proceed to Stage A or raise limits."""
from pathlib import Path
import subprocess
import sys
import time
TOOL=Path(__file__).resolve().parent;ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,PARENT,load,sha,write_new,check_identity,environment,monitored
out=Path(sys.argv[1]).resolve();extra=out/'failure-attribution';extra.mkdir(exist_ok=False)
b=load(out/'build.json');capture=ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027'
check_identity(capture)
assert load(out/'S-process.json')['exit_code']==1 and load(out/'S-process.json')['isolated']
assert 'optimizer slots exceeded the qualified memory budget' in (out/'S.log').read_text()
for p,h in {**b['tools'],**b['dependencies'],**b['source_sha256']}.items():assert sha(ROOT/p)==h,p
for p,h in b['outputs'].items():assert sha(out/p)==h,p
write_new(extra/'protocol.json',dict(schema='l3-rejected-capacity-logging-repeat-v1',declared_epoch=time.time(),
    unchanged_overlay_sha256=sha(out/'probe.so'),unchanged_protocol_sha256=sha(out/'protocol.json'),
    unchanged_limit_bytes=1536*1024**2,unchanged_headroom_bytes=2*1024**3,
    expected_result='same capacity rejection; never continue to Stage A',
    purpose='exact physical/application accounting and bitwise coverage before rejection; exception cleanup'))
symbols=[r.split()[-1] for r in subprocess.check_output(['nm','-D','--defined-only',str(out/'probe.so')],text=True).splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
match=[s for s,d in zip(symbols,demangled) if d.startswith('l3_probe::optimizer(')];assert len(match)==1
(extra/'symbols.h').write_text('#define ATTR_optimizer "'+match[0]+'"\n')
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
cmd=['g++','-O2','-std=c++17','-shared','-fPIC','-pthread','-include',str(extra/'symbols.h'),
    '-I'+str(TOOL),'-I'+str(ROOT/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
    str(TOOL/'failure_attribution.cpp'),str(out/'probe.so'),str(capture/'audit.so'),
    '-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(capture),'-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
    '-lcudart','-ldl','-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-o',str(extra/'tap.so')]
with (extra/'build.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
(extra/'run.R').write_text('''dyn.load(Sys.getenv("FASTKPC_L3_FAILURE_LIBRARY"))
failure <- tryCatch({source(Sys.getenv("FASTKPC_L3_ORIGINAL_RUNNER"),local=.GlobalEnv);NULL},error=function(e)e)
if(is.null(failure)) stop("expected the unchanged capacity rejection")
print(.Call("fastkpc_l3_failure_report"))
stop(conditionMessage(failure),call.=FALSE)
''')
write_new(extra/'build.json',dict(command=cmd,tap_sha256=sha(extra/'tap.so'),
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [TOOL/'failure_attribution.py',TOOL/'failure_attribution.cpp',TOOL/'probe.hpp',
        extra/'symbols.h',extra/'run.R',out/'probe.so',out/'run.R',out/'build.json']}))
name='S-attribution';assert not (out/name).exists()
env=environment();env.update(LD_PRELOAD=str(extra/'tap.so')+':'+str(out/'probe.so')+':'+str(capture/'audit.so'),
    FASTKPC_L3_FAILURE_LIBRARY=str(extra/'tap.so'),FASTKPC_L3_ORIGINAL_RUNNER=str(out/'run.R'),
    FASTKPC_L3_PROBE_LIBRARY=str(out/'probe.so'),FASTKPC_L3_PROBE_OUT=str(out/name),FASTKPC_DIAG_L3_REPLACEMENT='1',
    FASTKPC_L3_PROBE_FIXTURES=str(capture/'capture-raw'),FASTKPC_L3_STAGE_A_REFERENCE=str(out/'W/stage-a.bin'),
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
ref=PARENT/'on3/receipt.rds'
try:
    monitored(out,name,['Rscript','--vanilla',str(extra/'run.R'),str(out/name),'1',str(ref),str(ref)],env,
        dict(evidence_class='logging-only repeat of unchanged rejected gate; not qualification or performance',
             diagnostic_overlay_sha256=sha(out/'probe.so'),logging_tap_sha256=sha(extra/'tap.so')))
except AssertionError:
    process=load(out/(name+'-process.json'))
    assert process['isolated'] and process['exit_code']==1
    assert 'optimizer slots exceeded the qualified memory budget' in (out/(name+'.log')).read_text()
    attribution=load(out/name/'failure.json')
    assert attribution['optimizer_workers_joined'] and attribution['optimizer_setups_bitwise_exact']==4065
    assert attribution['optimizer_targets_bitwise_exact']==54878
    assert attribution['work_increment_bytes']>1536*1024**2 and attribution['free_after_optimizer_bytes']>=2*1024**3
    assert attribution['resources_closed_after_exception'] and attribution['worker_resource_mutations']==0
    for p,h in b['outputs'].items():assert sha(out/p)==h,p
    write_new(extra/'closure.json',dict(status='CONFIRMED_PREDECLARED_CAPACITY_STOP',attribution=attribution,
        original_stage_a_reference_sha256=sha(out/'W/stage-a.bin'),native_sha256=sha(NATIVE),
        diagnostic_overlay_sha256=sha(out/'probe.so'),threshold_changed=False,shape_changed=False))
    print('CONFIRMED_PREDECLARED_CAPACITY_STOP',attribution,flush=True)
else:raise AssertionError('unchanged rejection did not reproduce; retain for investigation')
