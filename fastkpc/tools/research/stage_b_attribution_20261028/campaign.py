"""Frozen, GPU1-exclusive diagnostic capture and native Stage B replay."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from build import ROOT,TOOL,PARENT,NATIVE,sha,write,identity

os.chdir(ROOT)
out=Path(sys.argv[1]).resolve()
b=json.loads((out/'build.json').read_text());identity()
def frozen():
    identity()
    assert all(sha(ROOT/p)==h for p,h in b['tools'].items())
    assert all(sha(out/p)==h for p,h in b['outputs'].items())
    assert sha(ROOT/b['runner_parent'])==b['runner_parent_sha256']
frozen();os.sched_setaffinity(0,range(20))
env=os.environ.copy()
assert not env.get('LD_PRELOAD') and not env.get('CUDA_INJECTION64_PATH') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
    STAGE_B_NATIVE=str(NATIVE),LD_PRELOAD=str(out/'observe.so'))
env['LD_LIBRARY_PATH']='/usr/local/lib/R/lib:/usr/local/cuda/lib64:'+env.get('LD_LIBRARY_PATH','')
def processes():
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    return {int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines() if x.strip()}
assert not processes(),'wait for both GPUs to be free; no process started'
write(out/'protocol.json',dict(schema='stage-b-diagnostic-protocol-v1',declared_epoch=time.time(),
    native_sha256=b['native_sha256'],build_sha256=sha(out/'build.json'),declaration_sha256=sha(TOOL/'README.md'),
    physical_gpu=1,other_gpu_idle=True,sequence=['capture','replay1','replay2','replay3','trace'],
    levels=[2,3],component_counts=[49919,54878],n=351,numCol=35,contexts=10,width=64,
    dense_lanes=8,cache_bytes=11811160064,unprofiled_repetitions=3,
    performance_promotion=False,kernel_changes=False,hardware_counters='not available under current user permissions',
    correctness='all 104797 three-field payload hashes exact; original cohorts/order; zero fallback; resources close',
    environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_MAX_CONNECTIONS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
    tools=b['tools'],dependencies={str(PARENT/'on3/receipt.rds'):sha(PARENT/'on3/receipt.rds'),
        str(ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh'):sha(ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh')}))
with (out/'environment.log').open('x') as f:
    subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu','--format=csv'],stdout=f,check=True)
    subprocess.run(['/opt/nvidia/nsight-compute/2024.2.1/ncu','--version'],stdout=f,check=True)
    subprocess.run(['bash','-c',"rg -in 'profil' /proc/driver/nvidia/params\nrg '^CapEff:' /proc/self/status"],stdout=f,check=True)

def job(name,command,mode,traced=False):
    frozen();assert not processes(),'external GPU task; no new job started'
    dest=out/name;dest.mkdir();e=env.copy();e.update(STAGE_B_MODE=mode,STAGE_B_OUTPUT=str(dest))
    if traced:e.update(CUDA_INJECTION64_PATH=str(out/'trace.so'),STAGE_B_TRACE=str(dest))
    samples=[];unexpected=set();started=time.time()
    with (out/(name+'.log')).open('x') as log,(out/(name+'-gpu.csv')).open('x') as gpu:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        proc=None
        try:
            proc=subprocess.Popen(command,env=e,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                active=processes();samples.append(dict(epoch=time.time(),processes=active))
                for pid in active:
                    try:
                        if os.getpgid(pid)!=proc.pid:unexpected.add(pid)
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
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
    write(out/(name+'-process.json'),dict(command=command,exit_code=proc.returncode,start_epoch=started,end_epoch=time.time(),
        affinity=sorted(os.sched_getaffinity(0)),isolated=not unexpected,unexpected=sorted(unexpected),samples=samples))
    assert not unexpected and proc.returncode==0,name+' failed; evidence retained'
    frozen()
    rows=[]
    for level in [2,3]:
        row=json.loads((dest/f'l{level}-observation.json').read_text());rows.append(row)
        assert row['components']==(49919 if level==2 else 54878)
        assert row['three_field_hashes']==row['components'] and row['dense_fallback']==0
        assert len((dest/f'l{level}-payload-sha256.txt').read_text().splitlines())==row['components']
        if mode!='capture':
            assert sha(dest/f'l{level}-payload-sha256.txt')==sha(out/'capture'/f'l{level}-payload-sha256.txt')
    if mode=='capture':
        report=json.loads((dest/'report.json').read_text());assert all(report['gates'].values())
        write(out/'capture-identity.json',dict(native_sha256=b['native_sha256'],
            canonical_gates=report['gates'],hashes={str(p.relative_to(out)):sha(p) for p in sorted(dest.iterdir()) if p.is_file()}))
    else:assert 'PASS native replay resource closure' in (out/(name+'.log')).read_text()
    print('PASS',name,[(r['level'],r['stage_b_sec']) for r in rows],flush=True)

job('capture',['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(out/'capture_runner.R'),
    str(out/'capture'),'1',str(PARENT/'on3/receipt.rds'),str(PARENT/'on3/receipt.rds')],'capture')
for name in ['replay1','replay2','replay3','trace']:
    for p,h in json.loads((out/'capture-identity.json').read_text())['hashes'].items():assert sha(out/p)==h
    job(name,[str(out/'replay'),str(out/'capture')],'replay',traced=name=='trace')
trace=json.loads((out/'trace/trace-summary.json').read_text())
assert trace['dropped']==0 and trace['invalid_timestamps']==0
write(out/'campaign.json',dict(status='PASS_AUTHENTICATED_ACTIVITY_DIAGNOSTIC',native_sha256=b['native_sha256'],
    component_hashes_per_replay=104797,replays=4,hardware_counter_status='UNAVAILABLE_PRIVILEGE',
    performance_claim=False,trace=trace))
print('PASS authenticated activity campaign; hardware occupancy/stalls remain unmeasured',flush=True)
