"""Identity and isolated job monitoring shared by this bounded campaign."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
PARENT = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028'
SEALED = ROOT / 'fastkpc/artifacts/full_cuda_ci/stage_b_work_pair_20261028/attempt1'
NATIVE = ROOT / 'fastkpc/build/fastkpc_cuda.so'
COUNTS = {4: 17767, 5: 6816, 6: 1148, 7: 124, 8: 18}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

def check_map(mapping, root=ROOT):
    for p, h in mapping.items():
        assert sha(root / p) == h, p

def original_identity():
    b = read(PARENT / 'build.json')
    assert sha(NATIVE) == b['native_sha256']
    check_map(b['source_sha256'])
    assert sha(ROOT / 'fastkpc/build/recorded_dbdsqr.inc') == b['generator_output_sha256']
    return b

def environment():
    os.chdir(ROOT)
    os.sched_setaffinity(0, range(20))
    e = os.environ.copy()
    assert not any(e.get(k) for k in ('LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'CUDA_LAUNCH_BLOCKING'))
    e.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',
        FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1')
    e['LD_LIBRARY_PATH'] = '/usr/local/lib/R/lib:/usr/local/cuda/lib64:' + e.get('LD_LIBRARY_PATH', '')
    return e

def processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return {int(x.split(',')[0]): x.split(',')[1].strip() for x in raw.splitlines() if x.strip()}

def job(out, name, command, env, frozen):
    frozen()
    assert not processes(), 'external GPU task; no new job started'
    samples, unexpected = [], set()
    started = time.time()
    with (out / (name + '.log')).open('x') as log, (out / (name + '-gpu.csv')).open('x') as gpu:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
                                    '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
        proc = None
        try:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                active = processes()
                samples.append(dict(epoch=time.time(), processes=active))
                for pid in active:
                    try:
                        if os.getpgid(pid) != proc.pid or active[pid] != 'GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2':
                            unexpected.add(pid)
                    except ProcessLookupError:
                        pass
                if unexpected:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break
                try:
                    proc.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
    write(out / (name + '-process.json'), dict(command=command, exit_code=proc.returncode,
        start_epoch=started, end_epoch=time.time(), affinity=sorted(os.sched_getaffinity(0)),
        isolated=not unexpected, unexpected=sorted(unexpected), samples=samples))
    assert not unexpected and proc.returncode == 0, name + ' failed; evidence retained'
    frozen()
    print('PASS', name, flush=True)
