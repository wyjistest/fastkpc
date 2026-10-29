"""Two isolated, instrumented native replays; no performance acceptance claim."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from build import ROOT, TOOL, CAPTURE, NATIVE, identity, read, sha, write

out = Path(sys.argv[1]).resolve()
os.chdir(ROOT)
os.sched_setaffinity(0, range(20))
b = read(out / 'build.json')


def frozen():
    identity()
    for group in ['tools', 'dependencies']:
        for name, expected in b[group].items():
            assert sha(ROOT / name) == expected, name
    for name, expected in b['outputs'].items():
        assert sha(out / name) == expected, name


def processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return {int(line.split(',')[0]): line.split(',')[1].strip()
            for line in raw.splitlines() if line.strip()}


frozen()
assert not processes(), 'external GPU task; no job started'
env = os.environ.copy()
assert not any(env.get(k) for k in ['LD_PRELOAD', 'CUDA_INJECTION64_PATH', 'CUDA_LAUNCH_BLOCKING'])
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
           FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',
           STAGE_B_NATIVE=str(NATIVE), STAGE_B_MODE='replay',
           LD_PRELOAD=str(out / 'observe.so'), CUDA_INJECTION64_PATH=str(out / 'trace.so'))
env['LD_LIBRARY_PATH'] = '/usr/local/lib/R/lib:/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
write(out / 'protocol.json', dict(schema='current-work-pair-activity-protocol-v1',
    declared_epoch=time.time(), build_sha256=sha(out / 'build.json'),
    declaration_sha256=sha(TOOL / 'README.md'), sequence=['off', 'on'],
    source_sha256=b['source_sha256'], native_sha256=b['native_sha256'],
    physical_gpu=1, gpu_uuid='GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2',
    both_gpus_otherwise_idle=True, affinity=list(range(20)),
    levels=[2, 3], counts=[49919, 54878], waves=[78, 86], contexts=10, width=64,
    mode_counts=dict(created=10, destroyed=10, submissions=1638, targets=104797),
    instrumented=True, performance_claim=False, full_runtime_capacity_claim=False,
    hardware_counters=False, normal_waves='exclude first and last', early_start_ms=50,
    tools=b['tools'], dependencies=b['dependencies'], environment={key: value for key, value in env.items()
        if key.startswith(('CUDA_', 'FASTKPC_', 'STAGE_B_')) or key in
        ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_PRELOAD', 'LD_LIBRARY_PATH']}))
with (out / 'environment.log').open('x') as log:
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu',
                    '--format=csv'], stdout=log, check=True)


def job(label, mode):
    frozen()
    assert not processes(), 'external GPU task; no new job started'
    dest = out / label
    dest.mkdir()
    e = env.copy()
    e.update(STAGE_B_OUTPUT=str(dest), STAGE_B_TRACE=str(dest),
             FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR=str(mode))
    command = [str(out / 'replay'), str(CAPTURE)]
    samples, unexpected = [], set()
    started = time.time()
    print('START', label, 'current native, mode', mode, flush=True)
    with (out / f'{label}.log').open('x') as log, (out / f'{label}-gpu.csv').open('x') as gpu:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
                                    '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
        proc = None
        try:
            proc = subprocess.Popen(command, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                active = processes()
                samples.append(dict(epoch=time.time(), processes=active))
                for pid, device in active.items():
                    try:
                        if os.getpgid(pid) != proc.pid or device != 'GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2':
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
    write(out / f'{label}-process.json', dict(command=command, mode=mode, exit_code=proc.returncode,
        start_epoch=started, end_epoch=time.time(), isolated=not unexpected,
        unexpected=sorted(unexpected), samples=samples))
    assert not unexpected and proc.returncode == 0, label + ' failed; evidence retained'
    frozen()
    audit = read(dest / 'mode.json')
    assert audit['pass'] and audit['mode'] == mode
    observations = []
    for level, count in [(2, 49919), (3, 54878)]:
        r = read(dest / f'l{level}-observation.json')
        assert r['components'] == r['three_field_hashes'] == count
        assert r['dense_fallback'] == 0 and r['contexts'] == 10 and r['width'] == 64
        assert r['waves'] == (count + 639) // 640
        assert sha(dest / f'l{level}-payload-sha256.txt') == sha(CAPTURE / f'l{level}-payload-sha256.txt')
        observations.append(r)
    assert 'PASS native replay resource closure' in (out / f'{label}.log').read_text()
    trace = read(dest / 'trace-summary.json')
    assert trace['dropped'] == trace['invalid_timestamps'] == 0 and trace['events'] > 0
    print('PASS', label, 'payload/mode/resource/trace gates',
          [(r['level'], r['stage_b_sec']) for r in observations], flush=True)
    return dict(mode=audit, observations=observations, trace=trace)


runs = {label: job(label, mode) for label, mode in [('off', 0), ('on', 1)]}
write(out / 'campaign.json', dict(status='PASS_CURRENT_NATIVE_WORK_PAIR_ACTIVITY',
    native_sha256=b['native_sha256'], build_sha256=sha(out / 'build.json'),
    protocol_sha256=sha(out / 'protocol.json'), runs=runs,
    component_hash_comparisons=209594, performance_claim=False,
    hardware_occupancy_measured=False, production_changes=False))
print('PASS both isolated current-native activity replays', flush=True)
