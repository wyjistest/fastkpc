"""Run unchanged balanced replay under the documented warm-context capacity amendment."""
import csv
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import CAPTURE, NATIVE, ROOT, TOOL, check_identity, load, sha, write_new

out = Path(sys.argv[1]).resolve()
mode = 'campaign'
os.chdir(ROOT)
build = check_identity(out)
original_protocol = load(out / 'protocol.json')
assert load(out / 'capacity-process.json')['exit_code'] != 0
cold = load(out / 'capacity-diagnostic-summary.json')
warm = load(out / 'capacity-warm-diagnostic-summary.json')
assert not cold['capacity_gate_pass'] and cold['outputs_bitwise_exact'] and cold['resources_closed']
assert warm['warm_increment_within_limit'] and warm['outputs_bitwise_exact'] and warm['resources_closed']
assert warm['targets'] == 17767 and warm['native_warm_targets_exact'] == 18
assert warm['minimum_free_bytes'] >= 2*1024**3
for name in ('checks', 'memcheck'):
    report = load(out / (name + '-summary.json'))
    assert report['status'] == 'PASS' and report['forbidden_calls'] == 0 and report['resources_closed']
memcheck = (out / 'memcheck.log').read_text()
assert 'ERROR SUMMARY: 0 errors' in memcheck and '0 bytes leaked' in memcheck
for name in ('capacity-diagnostic', 'capacity-warm-diagnostic'):
    diagnostic_build = load(out / (name + '-build.json'))
    assert sha(out / name) == diagnostic_build['output_sha256']
    assert diagnostic_build['original_build_sha256'] == sha(out / 'build.json')
    assert load(out / (name + '-process.json'))['exit_code'] == 0
    for path, expected in diagnostic_build['inputs'].items():
        assert sha(path) == expected
protocol = {**original_protocol, 'schema': 'l4-slots-protocol-v2',
    'original_protocol_sha256': sha(out / 'protocol.json'),
    'amendment_sha256': sha(TOOL / 'CAPACITY_AMENDMENT.md'),
    'original_cold_capacity_gate_pass': False,
    'capacity_basis': 'additional physical memory above compute-warm production-native context',
    'cold_readout_sha256': sha(out / 'capacity-diagnostic-summary.json'),
    'warm_readout_sha256': sha(out / 'capacity-warm-diagnostic-summary.json'),
    'runner_sha256': sha(TOOL / 'run_revised.py'),
    'declaration_epoch': time.time()}
write_new(out / 'protocol-v2.json', protocol)
os.sched_setaffinity(0, range(20))
uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
assert not subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
env = os.environ.copy()
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FASTKPC_WINDOW_NATIVE=str(NATIVE), FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
           LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
prefix = out / mode
assert not prefix.with_suffix('.log').exists(), 'preserve existing attempts'
cmd = [str(out / 'replay'), str(CAPTURE / 'raw'), str(prefix), mode]
started = time.time()
samples, unexpected, others = [], set(), set()
print('START', mode, time.strftime('%H:%M:%S'), flush=True)
with prefix.with_suffix('.log').open('x') as log, (out / (mode + '-gpu.csv')).open('x') as gpu:
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
                                '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
    try:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
            active = {int(r.split(',')[0]): r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
            selected = {pid for pid, device in active.items() if device == uuid}
            known = {p.pid}
            edges = [tuple(map(int, r.split())) for r in subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines()]
            for _ in range(8):
                known.update(pid for pid, parent in edges if parent in known)
            unexpected.update(selected - known)
            others.update(set(active) - selected)
            samples.append(dict(epoch=time.time(), processes=active, own_pids=sorted(known)))
            if unexpected:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                break
            try:
                p.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
write_new(out / (mode + '-gpu-processes.json'), samples)
write_new(out / (mode + '-process.json'), dict(command=cmd,
    start_epoch=started, end_epoch=time.time(), exit_code=p.returncode, physical_gpu=1, gpu_uuid=uuid,
    other_gpu_pids=sorted(others), unexpected_gpu_pids=sorted(unexpected), affinity=sorted(os.sched_getaffinity(0)),
    runner_sha256=sha(TOOL / 'run_revised.py'), protocol_sha256=sha(out / 'protocol-v2.json'), native_sha256=sha(NATIVE),
    environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OPENBLAS_NUM_THREADS',
                                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H']}))
assert p.returncode == 0 and not unexpected, 'experiment failed; preserve logs'
check_identity(out)
print('PASS', mode, flush=True)
