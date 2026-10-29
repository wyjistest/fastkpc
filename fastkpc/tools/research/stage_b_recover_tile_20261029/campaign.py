"""Frozen GPU qualification, balanced timings, then explanatory traces."""
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from build import ROOT, TOOL, TRACE, CAPTURE, NATIVE, identity, read, sha, write

out=Path(sys.argv[1]).resolve()
os.chdir(ROOT)
os.sched_setaffinity(0,range(20))
b=read(out/'build.json')
gpu_uuid='GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2'
order=[('A1',36),('B1',16),('C1',8),('B2',16),('C2',8),('A2',36),('C3',8),('A3',36),('B3',16)]


def frozen():
    identity()
    for group in ['tools','dependencies']:
        for name,expected in b[group].items():
            assert sha(ROOT/name)==expected,name
    for name,expected in b['outputs'].items():
        assert sha(out/name)==expected,name


def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    return {int(s.split(',')[0]):s.split(',')[1].strip() for s in raw.splitlines() if s.strip()}


frozen()
assert not processes(),'external compute task; no job started'
env=os.environ.copy()
assert not any(env.get(k) for k in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_LAUNCH_BLOCKING'])
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',STAGE_B_NATIVE=str(NATIVE),STAGE_B_MODE='replay')
env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
write(out/'protocol.json',dict(schema='isolated-recover-tile-protocol-v1',declared_epoch=time.time(),
    build_sha256=sha(out/'build.json'),declaration_sha256=sha(TOOL/'README.md'),native_sha256=b['native_sha256'],
    gpu=1,gpu_uuid=gpu_uuid,both_gpus_otherwise_idle=True,affinity=list(range(20)),threads=64,tiles=[36,16,8],
    qualification=['resources','memcheck16','memcheck8','qualify16','qualify8','native_control'],
    timing_order=order,trace_order=[36,16,8],levels=[2,3],counts=[49919,54878],waves=[78,86],contexts=10,width=64,
    gate='Each candidate: both level medians improve, combined sum decreases >=5%, all candidate totals below all A totals',
    environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_','STAGE_B_')) or k in
        ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH']},
    hardware_counters=False,production_changes=False,full_runtime_capacity_claim=False,promotion=False))
with (out/'environment.log').open('x') as log:
    subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu','--format=csv'],stdout=log,check=True)


def job(name,command,extra=None):
    frozen()
    assert not processes(),'external GPU task; no new job started'
    e=env.copy()
    if extra:e.update(extra)
    samples=[];unexpected=set();started=time.time();proc=None
    print('START',name,flush=True)
    with (out/f'{name}.log').open('x') as log,(out/f'{name}-gpu.csv').open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
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
    write(out/f'{name}-process.json',dict(command=command,exit_code=proc.returncode,start_epoch=started,end_epoch=time.time(),
        isolated=not unexpected,unexpected=sorted(unexpected),samples=samples))
    assert not unexpected and proc.returncode==0,name+' failed; evidence retained'
    frozen()
    print('PASS',name,'process_sec',round(time.time()-started,3),flush=True)


def replay(name,tile=None,trace=False):
    dest=out/name;dest.mkdir()
    preload=str(TRACE/'observe.so')
    if tile is not None:preload+=':'+str(out/f'tile{tile}_timing.so')
    extra=dict(LD_PRELOAD=preload,STAGE_B_OUTPUT=str(dest))
    if trace:extra.update(CUDA_INJECTION64_PATH=str(TRACE/'trace.so'),STAGE_B_TRACE=str(dest))
    job(name,[str(TRACE/'replay'),str(CAPTURE)],extra)
    mode=read(dest/'mode.json');assert mode['pass'] and mode['mode']==1
    if tile is not None:
        variant=read(dest/'tile.json')
        assert variant['tile']==tile and variant['threads']==64 and variant['dynamic_shared']==tile*351*8
        assert variant['created']==variant['destroyed']==10 and variant['submissions']==1638 and variant['targets']==104797
        assert all(variant[k]==0 for k in ['other_created','other_destroyed','other_submissions','other_targets'])
    observations=[]
    for level,count in [(2,49919),(3,54878)]:
        row=read(dest/f'l{level}-observation.json')
        assert row['components']==row['three_field_hashes']==count
        assert row['dense_fallback']==0 and row['contexts']==10 and row['width']==64
        assert row['waves']==(count+639)//640
        assert sha(dest/f'l{level}-payload-sha256.txt')==sha(CAPTURE/f'l{level}-payload-sha256.txt')
        observations.append(row)
    assert 'PASS native replay resource closure' in (out/f'{name}.log').read_text()
    if trace:
        summary=read(dest/'trace-summary.json')
        assert summary['dropped']==summary['invalid_timestamps']==0
    result=dict(name=name,tile=tile,trace=trace,levels=observations,total=sum(r['stage_b_sec'] for r in observations))
    print('STAGES',name,[r['stage_b_sec'] for r in observations],flush=True)
    return result


job('resources',[sys.executable,str(TOOL/'resources.py'),str(out)])
qualifications={}
for tile in [16,8]:
    command=[str(out/'qualify'),str(out/'tile36_diag.so'),str(out/f'tile{tile}_diag.so'),str(CAPTURE)]
    job(f'memcheck{tile}',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full',
        '--error-exitcode','99',*command,str(out/f'memcheck{tile}.json')],dict(RECOVER_SMOKE='1'))
    log=(out/f'memcheck{tile}.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in log
for tile in [16,8]:
    job(f'qualify{tile}',[str(out/'qualify'),str(out/'tile36_diag.so'),str(out/f'tile{tile}_diag.so'),str(CAPTURE),str(out/f'qualify{tile}.json')])
    q=read(out/f'qualify{tile}.json')
    assert q['status']=='PASS_GPU_BITWISE_QUALIFICATION' and q['captured']['targets']==104797
    assert q['captured']['mismatches']==q['synthetic']['mismatches']==q['live_device_allocations']==q['live_device_bytes']==0
    qualifications[str(tile)]=q
native_control=replay('native_control')
runs=[replay(name,tile) for name,tile in order]
medians={str(tile):[statistics.median(r['levels'][j]['stage_b_sec'] for r in runs if r['tile']==tile) for j in [0,1]]
         for tile in [36,16,8]}
decisions={}
for tile in [16,8]:
    gain=1-sum(medians[str(tile)])/sum(medians['36'])
    separated=max(r['total'] for r in runs if r['tile']==tile)<min(r['total'] for r in runs if r['tile']==36)
    both=all(c<a for a,c in zip(medians['36'],medians[str(tile)]))
    go=gain>=.05 and separated and both
    decisions[str(tile)]=dict(decision='GO_ISOLATED_STAGE_B' if go else 'CONDITIONAL' if gain>0 else 'STOP_NO_STAGE_GAIN',
        fraction_reduction=gain,both_levels_improve=both,all_candidate_totals_below_baseline=separated)
write(out/'timing.json',dict(status='PASS_NINE_TIMINGS',runs=runs,medians=medians,decisions=decisions,native_control=native_control))
print('TIMING_DECISION',decisions,flush=True)
traces=[replay(f'trace{tile}',tile,True) for tile in [36,16,8]]
frozen()
write(out/'campaign.json',dict(status='PASS_ISOLATED_RECOVER_TILE_CAMPAIGN',build_sha256=sha(out/'build.json'),
    protocol_sha256=sha(out/'protocol.json'),native_sha256=b['native_sha256'],qualifications=qualifications,
    timing_sha256=sha(out/'timing.json'),decisions=decisions,traces=traces,
    full_replays=13,component_hash_comparisons=13*104797,production_changes=False,
    accepted_opt_in_development_median_sec=311.440,default_enabled=False,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
print('PASS campaign complete',decisions,flush=True)
