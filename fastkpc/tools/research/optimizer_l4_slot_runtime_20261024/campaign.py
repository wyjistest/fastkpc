"""Full canonical S2, L4 slots off/on/on/off/off/on, same source-closure binary.

Change only L4 slots; keep L5 slots and L4 workers32 enabled; preserve all numerical gates.
Serialization and offline authority comparison remain outside outer elapsed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[4]
os.chdir(root)
out = Path(sys.argv[1]).resolve()
tool = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((out/'build.json').read_text())
assert build['status'] == 'PASS'
assert json.loads((out/'checks.json').read_text())['status'] == 'PASS'
native = root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native) == build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
os.sched_setaffinity(0, range(20))
device='1'
shared_host=False
assert device in ['0','1']
device_uuid=subprocess.check_output(['nvidia-smi','-i',device,'--query-gpu=uuid',
                                     '--format=csv,noheader'],text=True).strip()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=device, CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
assert not env.get('FASTKPC_LSCP_CANONICAL_AUDIT_DIR')
old = 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_runtime_20261023/on1/receipt.rds'
runs = [('off1','0',old,'-'),('on1','1',str(out/'off1/receipt.rds'),'-'),
        ('on2','1',str(out/'off1/receipt.rds'),str(out/'on1/receipt.rds')),
        ('off2','0',str(out/'off1/receipt.rds'),str(out/'off1/receipt.rds')),
        ('off3','0',str(out/'off1/receipt.rds'),str(out/'off1/receipt.rds')),
        ('on3','1',str(out/'off1/receipt.rds'),str(out/'on1/receipt.rds'))]
frozen_tools = {str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()}
frozen_dependencies = {p:sha(root/p) for p in [
    'fastkpc/tests/test_full_cuda_ci_l4_optimizer_slots.R',
    'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    'fastkpc/tools/with_phase10_exact_development_env.sh',
    'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    'fastkpc/R/fast_kpc.R','fastkpc/R/full_cuda_ci_phase35_vertical.R',
    'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']}
with (out/'protocol.json').open('x') as protocol:
    json.dump(dict(schema='optimizer-l4-slot-runtime-protocol-v1',native_sha256=sha(native),
        frozen_tools=frozen_tools,frozen_dependencies=frozen_dependencies,sequence=[x[0] for x in runs],
        outer_ratio_limit=.98,l4_optimizer_ratio_limit=.90,required_pair_wins=3,
        evidence_class='GPU1 isolated incremental source-closure development'),protocol,indent=2)
    protocol.write('\n')
with (out/'environment.log').open('x') as log:
    subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.current.sm,temperature.gpu',
                    '--format=csv'],stdout=log,stderr=subprocess.STDOUT,check=True)
for name, mode, reference, repeated in runs:
    assert not (out/name).exists(), 'preserve earlier runs'
    assert sha(native) == build['native_sha256']
    assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
    assert all(sha(root/p)==h for p,h in frozen_tools.items())
    assert all(sha(root/p)==h for p,h in frozen_dependencies.items())
    assert not subprocess.check_output(['nvidia-smi']+(['-i',device] if shared_host else [])+
                                      ['--query-compute-apps=pid',
                                       '--format=csv,noheader'],text=True).strip()
    env['FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H'] = '1'
    env['FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS'] = '1'
    env['FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32'] = '1'
    env['FASTKPC_DCC_GAMMA_L4_OPTIMIZER_SLOTS'] = mode
    env['FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE'] = '0'
    command = ['sh','fastkpc/tools/with_phase10_exact_development_env.sh',
               'Rscript','--vanilla',str(tool/'run.R'),str(out/name),mode,reference,repeated]
    started = time.time()
    process_samples=[]
    unexpected_pids=set()
    other_gpu_pids=set()
    print('START',name,time.strftime('%H:%M:%S'),flush=True)
    with (out/(name+'.log')).open('x') as log, (out/(name+'-gpu.csv')).open('x') as gpu:
        monitor = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits','-lms','200'],stdout=gpu,stderr=subprocess.DEVNULL)
        try:
            result = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            while True:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                                              '--format=csv,noheader'],text=True)
                active={int(row.split(',')[0]):row.split(',')[1].strip()
                        for row in raw.splitlines() if row.strip()}
                selected={pid for pid,uuid in active.items() if uuid==device_uuid}
                other=set(active)-selected
                process_samples.append(dict(epoch=time.time(),processes=active))
                other_gpu_pids.update(other)
                unexpected_pids.update(selected-{result.pid})
                if not shared_host: unexpected_pids.update(other)
                if unexpected_pids:
                    # Stop only our benchmark; never signal another user's job.
                    result.terminate()
                    try: result.wait(timeout=10)
                    except subprocess.TimeoutExpired: result.kill();result.wait()
                    break
                try: result.wait(timeout=5);break
                except subprocess.TimeoutExpired: pass
        finally:
            monitor.terminate();monitor.wait(timeout=10)
    (out/(name+'-gpu-processes.json')).write_text(json.dumps(process_samples,indent=2)+'\n')
    (out/(name+'-process.json')).write_text(json.dumps(dict(command=command,exit_code=result.returncode,
        start_epoch=started,end_epoch=time.time(),affinity=sorted(os.sched_getaffinity(0)),
        status_wait='1',l5_slots='1',l4_workers32='1',l4_slots=mode,native_sha256=build['native_sha256'],
        physical_gpu=int(device),shared_host_allowed=shared_host,
        selected_gpu_isolated=not unexpected_pids,
        isolated=not unexpected_pids and not other_gpu_pids,
        other_gpu_pids=sorted(other_gpu_pids),
        unexpected_gpu_pids=sorted(unexpected_pids)),indent=2)+'\n')
    assert not unexpected_pids, name+' invalidated by another GPU job'
    assert result.returncode == 0, name+' failed'
    report = json.loads((out/name/'report.json').read_text())
    assert report['gates'] and all(report['gates'].values())
    print('PASS',name,report['elapsed_sec'],report['oracle_errors'],flush=True)
print('PASS complete L4 slot off/on/on/off/off/on qualification',flush=True)

