"""Run one immutable GPU1-exclusive capture, timing or trace attempt.

python3 run.py ARTIFACT NAME 0|1 capture|timing|trace
Numerical/reference gates are the frozen cache-ON full canonical gates.
"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
tool=Path(__file__).resolve().parent;root=tool.parents[3]
out=Path(sys.argv[1]).resolve();name,mode,kind=sys.argv[2:]
assert mode in ('0','1') and kind in ('capture','timing','trace')
assert name and '/' not in name and not (out/name).exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
build=read(out/'observe-build.json');native=root/'fastkpc/build/fastkpc_cuda.so'
def identity():
    assert sha(native)==build['native_sha256']
    assert all(sha(root/p)==h for p,h in {**build['source_sha256'],**build['inputs']}.items())
    assert all(sha(out/p)==h for p,h in build['outputs'].items())
identity();os.chdir(root);os.sched_setaffinity(0,range(20))
assert not subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
env=os.environ.copy()
for key in ['LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_INJECTION32_PATH','CUDA_LAUNCH_BLOCKING','FASTKPC_LSCP_CANONICAL_AUDIT_DIR']:
    assert not env.get(key),(key,env.get(key))
env.update(CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_MAX_CONNECTIONS='32',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
 FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
 FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
 FASTKPC_LATE_L8_PRELOAD=mode,FASTKPC_LATE_OBSERVER=str(out/'observe.so'),LD_PRELOAD=str(out/'observe.so'),
 FASTKPC_WINDOW_NATIVE=str(native),FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
 FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS='1',FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32='1',FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS='1',
 FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE='0')
if kind=='capture':
    (out/(name+'-raw')).mkdir();env['FASTKPC_LATE_CAPTURE']=str(out/(name+'-raw'))
else:env.pop('FASTKPC_LATE_CAPTURE',None)
if kind=='trace':
    tracer=root/'fastkpc/artifacts/full_cuda_ci/optimizer_slot_line_review_20261025/ktrace.so'
    assert sha(tracer)=='fae0e66400d6a64487924000d7d29fd72ca0818ad3b32457a5382e3dd3db2adb'
    (out/(name+'-trace')).mkdir();env.update(CUDA_INJECTION64_PATH=str(tracer),KTRACE_DIR=str(out/(name+'-trace')))
reference=root/'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025/on1/receipt.rds'
command=['sh','fastkpc/tools/with_phase10_exact_development_env.sh','Rscript','--vanilla',str(tool/'run.R'),str(out/name),'1',str(reference),str(reference)]
frozen={str(p.relative_to(root)):sha(p) for p in tool.iterdir() if p.is_file()}
frozen.update({str(p.relative_to(root)):sha(p) for p in [
 tool.parent/'device_properties_cache_20261025/run.R',
 root/'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
 root/'fastkpc/tools/with_phase10_exact_development_env.sh',root/'fastkpc/R/fast_kpc.R',root/'fastkpc/R/full_cuda_ci_phase35_vertical.R']})
record=dict(schema='late-optimizer-run-protocol-v1',name=name,kind=kind,preload=mode,
 native_sha256=sha(native),observer_sha256=sha(out/'observe.so'),reference_sha256=sha(reference),
 tools=frozen,environment={k:v for k,v in env.items() if k.startswith(('FASTKPC_','CUDA_')) or k in ['LD_PRELOAD','LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
 sequence_role='diagnostic only' if kind!='timing' else 'predeclared OFF/ON/ON/OFF/OFF/ON',
 selected_gpu_exclusive=True,shared_host_allowed=True,affinity=sorted(os.sched_getaffinity(0)),command=command)
with (out/(name+'-protocol.json')).open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
started=time.time();samples=[];unexpected=set();other=set()
print('START',name,kind,'preload='+mode,time.strftime('%H:%M:%S'),flush=True)
with (out/(name+'.log')).open('x') as log,(out/(name+'-gpu.csv')).open('x') as gpu:
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu','--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
    try:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
            active={int(row.split(',')[0]):row.split(',')[1].strip() for row in raw.splitlines() if row.strip()}
            selected={pid for pid,value in active.items() if value==uuid}
            samples.append(dict(epoch=time.time(),processes=active));other.update(set(active)-selected);unexpected.update(selected-{proc.pid})
            if unexpected:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                break
            try:proc.wait(timeout=5);break
            except subprocess.TimeoutExpired:pass
    finally:monitor.terminate();monitor.wait(timeout=10)
with (out/(name+'-processes.json')).open('x') as stream:json.dump(samples,stream,indent=2);stream.write('\n')
with (out/(name+'-process.json')).open('x') as stream:
    json.dump(dict(exit_code=proc.returncode,start_epoch=started,end_epoch=time.time(),other_gpu_pids=sorted(other),unexpected_gpu_pids=sorted(unexpected),selected_gpu_isolated=not unexpected,isolated=not unexpected and not other),stream,indent=2);stream.write('\n')
assert not unexpected,'GPU1 interference; preserve invalidated attempt'
assert proc.returncode==0,'run failed; preserve log and any numerical receipt'
identity();assert all(sha(root/p)==h for p,h in frozen.items())
r=read(out/name/'report.json');h=read(out/name/'hook.json')
assert r['gates'] and all(r['gates'].values()) and all(h['gates'].values())
assert sha(out/name/'receipt.rds')==r['receipt_sha256']==h['receipt_sha256']
print('PASS',name,r['elapsed_sec'],json.dumps(h['after']),flush=True)
