"""Fresh W/S qualification. GPU1 exclusive; external GPU0 tasks are permitted."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TOOL=Path(__file__).resolve().parent
ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,PARENT,sha,load,write_new,source_identity,environment
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_stack_campaign',TOOL.parent/'optimizer_l3_stack_20261027/campaign.py')
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
OLD=frozen.OLD
CAPTURE=frozen.CAPTURE


def validate(out):
    frozen.check_old()
    b=load(out/'build.json')
    for p,h in b['inputs'].items():
        assert sha(ROOT/p)==h,p
    for p,h in b['outputs'].items():
        assert sha(out/p)==h,p
    assert b['native_sha256']==sha(NATIVE)
    assert b['protocol_sha256']==sha(out/'protocol.json')
    return b


def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                 '--format=csv,noheader'],text=True)
    return {int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}


def monitored(out,mode,cmd,env):
    os.chdir(ROOT);os.sched_setaffinity(0,range(20))
    uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid',
                                 '--format=csv,noheader'],text=True).strip()
    initial=processes()
    assert uuid not in initial.values(),'GPU1 has an external compute task'
    started=time.time();samples=[];unexpected=set();external=set(initial);p=None
    print('START',mode,time.strftime('%H:%M:%S'),'GPU1 exclusive; host shared',flush=True)
    with (out/(mode+'.log')).open('x') as log,(out/(mode+'-gpu.csv')).open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                active=processes();known={p.pid}
                edges=[tuple(map(int,r.split())) for r in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines()]
                for _ in range(8):known.update(pid for pid,parent in edges if parent in known)
                unexpected.update(pid for pid,device in active.items() if device==uuid and pid not in known)
                external.update(pid for pid,device in active.items() if device!=uuid and pid not in known)
                samples.append(dict(epoch=time.time(),processes=active,own_pids=sorted(known)))
                if unexpected:break
                try:p.wait(timeout=5);break
                except subprocess.TimeoutExpired:pass
        finally:
            if p is not None and p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            monitor.terminate();monitor.wait(timeout=10)
    write_new(out/(mode+'-gpu-processes.json'),samples)
    receipt=dict(command=cmd,start_epoch=started,end_epoch=time.time(),exit_code=p.returncode,
        physical_gpu=1,gpu_uuid=uuid,gpu1_exclusive=not unexpected,unexpected_gpu1_pids=sorted(unexpected),
        external_other_gpu_pids=sorted(external),both_gpus_isolated=not unexpected and not external,
        evidence_class='GPU1-exclusive shared-host qualification; no performance evidence',
        affinity=sorted(os.sched_getaffinity(0)),native_sha256=sha(NATIVE),
        diagnostic_overlay_sha256=sha(out/'probe.so'),protocol_sha256=sha(out/'protocol.json'),
        environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_')) or k in
                     ('LD_PRELOAD','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    write_new(out/(mode+'-process.json'),receipt)
    assert not unexpected,'GPU1 interference; attempt preserved'
    source_identity()
    return receipt


def run(out,mode):
    validate(out);assert mode in ('W','S')
    assert not (out/mode).exists() and not (out/(mode+'-process.json')).exists()
    reference=out/'W/stage-a.bin'
    if mode=='S':
        assert load(out/'W-closure.json')['status']=='PASS'
        assert sha(reference)==sha(OLD/'W/stage-a.bin')
    env=environment()
    env.update(LD_PRELOAD=':'.join(map(str,[out/'probe.so',CAPTURE/'audit.so'])),
        FASTKPC_L3_PROBE_LIBRARY=str(out/'probe.so'),FASTKPC_L3_PROBE_OUT=str(out/mode),
        FASTKPC_DIAG_L3_REPLACEMENT='1' if mode=='S' else '0',
        FASTKPC_L3_PROBE_FIXTURES=str(CAPTURE/'capture-raw'),FASTKPC_L3_STAGE_A_REFERENCE=str(reference),
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',FASTKPC_DIAG_SHARED_HOST_QUALIFICATION='1')
    ref=PARENT/'on3/receipt.rds'
    process=monitored(out,mode,['Rscript','--vanilla',str(out/'run.R'),str(out/mode),'1',str(ref),str(ref)],env)
    if process['exit_code']!=0:
        failure=load(out/mode/'failure.json') if (out/mode/'failure.json').exists() else None
        write_new(out/(mode+'-closure.json'),dict(status='STOP_FAILED_QUALIFICATION',failure=failure,
            process_sha256=sha(out/(mode+'-process.json'))))
        raise RuntimeError('qualification failed; immutable attempt retained')
    report=load(out/mode/'report.json');probe=load(out/mode/'probe.json');cleanup=load(out/mode/'cleanup.json')
    assert len(report['gates'])==67 and all(report['gates'].values())
    assert probe['status']=='PASS' and probe['optimizer_setups']==4065 and probe['optimizer_targets']==54878
    assert probe['stage_a_targets']==54878 and probe['stage_a_batches']==5573
    assert cleanup['done'] and all(cleanup[k]==0 for k in ['allocations','bytes','streams','events','blas','worker_resource_mutations'])
    assert sha(reference)==sha(OLD/'W/stage-a.bin')
    validate(out)
    write_new(out/(mode+'-closure.json'),dict(status='PASS',canonical_gates=len(report['gates']),
        probe=probe,cleanup=cleanup,report_sha256=sha(out/mode/'report.json'),
        process_sha256=sha(out/(mode+'-process.json')),stage_a_reference_sha256=sha(reference)))
    print('PASS',mode,probe,flush=True)


if __name__=='__main__':
    action=sys.argv[1];out=Path(sys.argv[2]).resolve()
    if action=='prepare':
        original=frozen.check_old()
        builder_spec=importlib.util.spec_from_file_location('split_capacity_prepare',TOOL/'prepare.py')
        builder=importlib.util.module_from_spec(builder_spec);builder_spec.loader.exec_module(builder)
        builder.prepare(out,OLD,original,NATIVE,sha,write_new)
        validate(out)
    elif action=='run':run(out,sys.argv[3])
    else:raise ValueError(action)
