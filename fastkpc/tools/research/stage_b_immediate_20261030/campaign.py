"""Frozen 5N/3N qualification and six interleaved complete-stage timings."""
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from build import ROOT, TOOL, CAPTURE, NATIVE, identity, read, sha, write

out=Path(sys.argv[1]).resolve()
os.chdir(ROOT)
os.sched_setaffinity(0,range(20))
b=read(out/'build.json')
gpu_uuid='GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2'
order=[('A1','baseline'),('C1','candidate'),('C2','candidate'),
       ('A2','baseline'),('A3','baseline'),('C3','candidate')]


def frozen():
    identity()
    for group in ['tools','dependencies']:
        for name,expected in b[group].items():assert sha(ROOT/name)==expected,name
    for name,expected in b['outputs'].items():assert sha(out/name)==expected,name


def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    return {int(s.split(',')[0]):s.split(',')[1].strip() for s in raw.splitlines() if s.strip()}


frozen()
assert not processes(),'external GPU compute task; nothing started'
env=os.environ.copy()
assert not any(env.get(k) for k in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_LAUNCH_BLOCKING'])
for key in list(env):
    if key.startswith(('FASTKPC_','STAGE_B_')) or key in ['IP_SMOKE','RECOVER_SMOKE']:env.pop(key)
env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',STAGE_B_NATIVE=str(NATIVE),STAGE_B_MODE='replay')
env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
write(out/'protocol.json',dict(schema='isolated-immediate-protocol-v1',declared_epoch=time.time(),
    build_sha256=sha(out/'build.json'),declaration_sha256=sha(TOOL/'README.md'),native_sha256=b['native_sha256'],
    gpu=1,gpu_uuid=gpu_uuid,both_gpus_otherwise_idle=True,affinity=list(range(20)),
    baseline_factor=5,candidate_factor=3,recover_tile=36,recover_threads=64,
    qualification=['resources','memcheck','qualify','native_control','baseline_authentication'],
    timing_order=order,levels=[2,3],counts=[49919,54878],waves=[78,86],contexts=10,width=64,
    gate=dict(both_level_medians_improve=True,sum_of_level_median_ratio_max=.95,
              slowest_candidate_total_below_fastest_baseline_total=True,new_stack_local_spill_allowed=False),
    environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_','STAGE_B_')) or
                 k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH']},
    hardware_counters=False,production_changes=False,promotion=False))
with (out/'environment.log').open('x') as log:
    subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu',
                    '--format=csv'],stdout=log,check=True)


def job(name,command,extra=None):
    frozen();assert not processes(),'external compute task; no new job started'
    e=env.copy()
    if extra:e.update(extra)
    samples=[];unexpected=set();started=time.time();proc=None
    print('START',name,flush=True)
    with (out/f'{name}.log').open('x') as log,(out/f'{name}-gpu.csv').open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            proc=subprocess.Popen(command,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                active=processes();samples.append(dict(epoch=time.time(),processes=active))
                for pid,device in active.items():
                    try:
                        if os.getpgid(pid)!=proc.pid or device!=gpu_uuid:unexpected.add(pid)
                    except ProcessLookupError:pass
                if unexpected:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                try:proc.wait(timeout=3);break
                except subprocess.TimeoutExpired:pass
        finally:
            monitor.terminate();monitor.wait(timeout=10)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    write(out/f'{name}-process.json',dict(command=command,exit_code=proc.returncode,start_epoch=started,
          end_epoch=time.time(),isolated=not unexpected,unexpected=sorted(unexpected),samples=samples))
    assert not unexpected and proc.returncode==0,name+' failed; evidence retained'
    frozen();print('PASS',name,'process_sec',round(time.time()-started,3),flush=True)


def replay(name,variant=None):
    dest=out/name;dest.mkdir()
    preload=str(out/'observe.so')
    if variant:preload+=':'+str(out/(variant+'_timing.so'))
    job(name,[str(out/'replay'),str(CAPTURE)],dict(LD_PRELOAD=preload,STAGE_B_OUTPUT=str(dest)))
    mode=read(dest/'mode.json');assert mode['pass'] and mode['mode']==1
    if variant:
        v=read(dest/'immediate.json');factor=5 if variant=='baseline' else 3
        assert v['factor']==factor and v['dynamic_shared']==factor*351*8 and v['recover_tile']==36
        assert v['created']==v['destroyed']==10 and v['submissions']==1638 and v['targets']==104797
        assert all(v[k]==0 for k in ['other_created','other_destroyed','other_submissions','other_targets'])
    rows=[]
    for level,count in [(2,49919),(3,54878)]:
        row=read(dest/f'l{level}-observation.json')
        assert row['components']==row['three_field_hashes']==count
        assert row['dense_fallback']==0 and row['contexts']==10 and row['width']==64
        assert row['waves']==(count+639)//640
        assert sha(dest/f'l{level}-payload-sha256.txt')==sha(CAPTURE/f'l{level}-payload-sha256.txt')
        rows.append(row)
    assert 'PASS native replay resource closure' in (out/f'{name}.log').read_text()
    result=dict(name=name,variant=variant,levels=rows,total=sum(r['stage_b_sec'] for r in rows))
    print('STAGES',name,[r['stage_b_sec'] for r in rows],flush=True)
    write(out/(name+'-validated.json'),result)
    return result


def main():
    job('resources',[sys.executable,str(TOOL/'resources.py'),str(out)])
    command=[str(out/'qualify'),str(out/'baseline_diag.so'),str(out/'candidate_diag.so'),str(CAPTURE)]
    job('memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full',
                   '--error-exitcode','99',*command,str(out/'memcheck.json')],dict(IP_SMOKE='1',RECOVER_SMOKE='1'))
    log=(out/'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in log
    job('qualify',command+[str(out/'qualify.json')])
    q=read(out/'qualify.json')
    assert q['status']=='PASS_GPU_BITWISE_QUALIFICATION' and q['captured']['targets']==104797
    assert q['captured']['mismatches']==q['synthetic']['mismatches']==0
    assert all(q[k]==0 for k in ['live_device_allocations','live_device_bytes','live_streams','live_plans','cleanup_errors'])
    assert q['negative_controls']==2 and q['captured']['checkpoint_stages'][1]==q['captured']['sweeps']
    native_control=replay('native_control')
    baseline_authentication=replay('baseline_authentication','baseline')
    runs=[replay(name,variant) for name,variant in order]
    medians={v:[statistics.median(r['levels'][j]['stage_b_sec'] for r in runs if r['variant']==v)
                for j in [0,1]] for v in ['baseline','candidate']}
    total_medians={v:statistics.median(r['total'] for r in runs if r['variant']==v) for v in medians}
    ratio=sum(medians['candidate'])/sum(medians['baseline'])
    separated=max(r['total'] for r in runs if r['variant']=='candidate')<min(r['total'] for r in runs if r['variant']=='baseline')
    both=all(c<a for a,c in zip(medians['baseline'],medians['candidate']))
    decision='GO_ISOLATED_STAGE_B' if ratio<=.95 and separated and both else 'CONDITIONAL' if ratio<1 else 'STOP_NO_STAGE_GAIN'
    write(out/'timing.json',dict(status='PASS_SIX_TIMINGS',runs=runs,level_medians=medians,
          sum_of_level_medians={k:sum(v) for k,v in medians.items()},median_of_run_totals=total_medians,
          ratio=ratio,fraction_reduction=1-ratio,both_levels_improve=both,
          all_candidate_totals_below_baseline=separated,decision=decision,
          native_control=native_control,baseline_authentication=baseline_authentication))
    frozen()
    write(out/'campaign.json',dict(status='PASS_ISOLATED_IMMEDIATE_CAMPAIGN',decision=decision,
          build_sha256=sha(out/'build.json'),protocol_sha256=sha(out/'protocol.json'),
          native_sha256=b['native_sha256'],qualification_sha256=sha(out/'qualify.json'),timing_sha256=sha(out/'timing.json'),
          full_replays=8,component_hash_comparisons=8*104797,production_changes=False,
          accepted_development_median_sec=283.328,default_enabled=False,
          phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
    write(out/'closure.json',dict(schema='isolated-immediate-closure-v1',
          files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
          decision=decision,native_sha256=b['native_sha256']))
    print('PASS campaign complete',decision,flush=True)


try:main()
except Exception as error:
    write(out/'failure.json',dict(status='STOP_CAMPAIGN',message=str(error),epoch=time.time(),
          production_changes=False,performance_evidence_accepted=False))
    raise
