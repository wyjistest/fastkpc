"""Run a fresh GPU1 attribution process and preserve process/GPU telemetry."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAPTURE, NATIVE, ROOT, TOOL, check_identity, sha, write_new

out = Path(sys.argv[1]).resolve()
run = int(sys.argv[2])
assert run in (1, 2)
os.chdir(ROOT)
build = check_identity(out)
prefix = out / ('profile-' + str(run))
assert not prefix.with_suffix('.log').exists(), 'preserve existing attempts'
if run == 1:
    write_new(out / 'protocol.json', dict(schema='l4-tail-protocol-v1',
        build_sha256=sha(out / 'build.json'), declaration_sha256=sha(TOOL / 'README.md'),
        independent_processes=2, unprofiled_warmup_per_process=1, profiled_pass_per_process=1,
        windows=34, setups=2154, targets=17767, workers=32,
        model_max_relative_fit_error=0.05, model_min_fraction_of_native_saved=0.10,
        decision_scope='separate bounded-slot experiment only; no runtime integration or measured speedup',
        production_cache_present=False))
else:
    from common import load
    assert load(out / 'profile-1-process.json')['exit_code'] == 0
    assert load(out / 'protocol.json')['build_sha256'] == sha(out / 'build.json')
os.sched_setaffinity(0, range(20))
uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
assert not subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
env = os.environ.copy()
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FASTKPC_WINDOW_NATIVE=str(NATIVE), FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
           LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
cmd = ['nsys', 'profile', '--sample=none', '--cpuctxsw=none', '--trace=cuda,nvtx',
       '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--export=sqlite',
       '--output=' + str(prefix), str(out / 'profile'), str(CAPTURE / 'raw'), str(prefix)]
started = time.time()
samples, unexpected, others = [], set(), set()
print('START profile', run, time.strftime('%H:%M:%S'), flush=True)
with prefix.with_suffix('.log').open('x') as log, (out / ('profile-' + str(run) + '-gpu.csv')).open('x') as gpu:
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
write_new(out / ('profile-' + str(run) + '-gpu-processes.json'), samples)
write_new(out / ('profile-' + str(run) + '-process.json'), dict(command=cmd,
    start_epoch=started, end_epoch=time.time(), exit_code=p.returncode, physical_gpu=1, gpu_uuid=uuid,
    other_gpu_pids=sorted(others), unexpected_gpu_pids=sorted(unexpected), affinity=sorted(os.sched_getaffinity(0)),
    runner_sha256=sha(TOOL / 'run.py'), native_sha256=sha(NATIVE),
    environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OPENBLAS_NUM_THREADS',
                                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H']}))
assert p.returncode == 0 and not unexpected, 'profile failed; preserve logs'
check_identity(out)
print('PASS profile', run, flush=True)
