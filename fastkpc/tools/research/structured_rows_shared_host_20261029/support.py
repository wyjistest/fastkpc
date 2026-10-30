"""Exclusive GPU1 jobs with explicit, sampled shared-host telemetry."""
import csv
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
BASE = ROOT / 'fastkpc/artifacts/full_cuda_ci/structured_rows_shared_host_20261029'
ROWS = ROOT / 'fastkpc/artifacts/full_cuda_ci/structured_rows_20261029'
PARENT = ROOT / 'fastkpc/artifacts/full_cuda_ci/build_consistency_20261029'
GPU0 = 'GPU-d1f5be95-b8a3-3c18-f919-90913038a1a3'
GPU1 = 'GPU-9c6883e3-fe8f-1a96-ebac-c8496069cba2'
SUPPORT = TOOL.parent / 'stage_b_work_pair_runtime_20261028/support.py'
spec = importlib.util.spec_from_file_location('original_jobs', SUPPORT)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
sha, read, write = original.sha, original.read, original.write


def environment():
    env = original.environment()
    env.update(FASTKPC_DCC_GAMMA_VECTOR_POSTBACK='1',
               FASTKPC_DIAG_SHARED_HOST_QUALIFICATION='0',
               FASTKPC_DIAG_SHARED_HOST_TIMING='1')
    return env


def processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return [dict(pid=int(row[0]), gpu_uuid=row[1].strip())
            for row in csv.reader(raw.splitlines()) if row]


def cpu_sample(active):
    counters = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields[0].startswith('cpu'):
            counters[fields[0]] = [int(x) for x in fields[1:]]
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[key + '_bytes'] = int(value.split()[0]) * 1024
    pressure = {name: (Path('/proc/pressure') / name).read_text().strip()
                for name in ('cpu', 'memory', 'io')}
    tasks = {}
    for pid in sorted({x['pid'] for x in active}):
        try:
            raw = Path(f'/proc/{pid}/stat').read_text()
            values = raw.rsplit(')', 1)[1].split()
            tasks[str(pid)] = dict(comm=raw.split('(', 1)[1].rsplit(')', 1)[0],
                pgid=int(values[2]), user_ticks=int(values[11]), system_ticks=int(values[12]),
                threads=int(values[17]), start_ticks=int(values[19]),
                affinity=sorted(os.sched_getaffinity(pid)))
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            tasks[str(pid)] = dict(unavailable=True)
    return dict(epoch=time.time(), cpu_ticks=counters,
        loadavg=Path('/proc/loadavg').read_text().strip(), memory=memory,
        pressure=pressure, gpu_processes=active, tasks=tasks)


def stop_own(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def job(name, command, env, frozen):
    frozen()
    initial = processes()
    assert all(x['gpu_uuid'] == GPU0 for x in initial), 'GPU1 occupied; no job started'
    background = {x['pid'] for x in initial}
    unexpected, reason = set(), None
    started = time.time()
    proc, monitor = None, None
    with (BASE / (name + '.log')).open('x') as log, \
         (BASE / (name + '-gpu.csv')).open('x') as gpu, \
         (BASE / (name + '-host.jsonl')).open('x') as host:
        monitor = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu',
            '--format=csv,noheader,nounits', '-lms', '200'],
            stdout=gpu, stderr=subprocess.DEVNULL)
        try:
            proc = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                active = processes()
                sample = cpu_sample(active)
                host.write(json.dumps(sample) + '\n')
                host.flush()
                for item in active:
                    pid, device = item['pid'], item['gpu_uuid']
                    try:
                        ours = os.getpgid(pid) == proc.pid
                    except ProcessLookupError:
                        continue
                    if device == GPU0 and not ours:
                        background.add(pid)
                    elif device != GPU1 or not ours:
                        unexpected.add(pid)
                if unexpected:
                    reason = 'unexpected GPU process; stopped only our process group'
                    stop_own(proc)
                    break
                if monitor.poll() is not None:
                    reason = 'GPU telemetry monitor exited unexpectedly'
                    stop_own(proc)
                    break
                try:
                    proc.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException as error:
            reason = repr(error)
            raise
        finally:
            stop_own(proc)
            if monitor.poll() is None:
                monitor.terminate()
            monitor.wait(timeout=10)
            write(BASE / (name + '-process.json'), dict(command=command,
                exit_code=proc.returncode if proc else None, start_epoch=started,
                end_epoch=time.time(), affinity=sorted(os.sched_getaffinity(0)),
                shared_host=True, isolated_acceptance=False,
                gpu1_exclusive=not unexpected, unexpected=sorted(unexpected),
                background_gpu0_pids=sorted(background), failure=reason,
                clk_tck=os.sysconf('SC_CLK_TCK'), installed=False))
    assert reason is None and proc.returncode == 0, name + ' failed; evidence preserved'
    frozen()
    print('PASS', name, flush=True)


def telemetry(name, window):
    samples = [json.loads(line) for line in
               (BASE / (name + '-host.jsonl')).read_text().splitlines()]
    samples = [x for x in samples if window['start_epoch'] <= x['epoch'] <= window['end_epoch']]
    assert len(samples) >= 2, 'insufficient compute-window host telemetry'
    fractions = {}
    for label, cpus in [('host', ['cpu']), ('affinity_0_19', [f'cpu{i}' for i in range(20)])]:
        first = [sum(samples[0]['cpu_ticks'][c][i] for c in cpus) for i in range(8)]
        last = [sum(samples[-1]['cpu_ticks'][c][i] for c in cpus) for i in range(8)]
        delta = [b-a for a, b in zip(first, last)]
        total = sum(delta)
        assert total > 0
        fractions[label] = dict(busy_percent=100*(total-delta[3]-delta[4])/total,
                                iowait_percent=100*delta[4]/total)
    gpu_samples = {0: [], 1: []}
    for row in csv.reader((BASE / (name + '-gpu.csv')).read_text().splitlines()):
        if len(row) != 7:
            continue
        epoch = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
        if window['start_epoch'] <= epoch <= window['end_epoch']:
            gpu_samples[int(row[1])].append([float(x.strip()) for x in row[2:]])
    gpu_summary = {}
    for index, rows in gpu_samples.items():
        assert rows, 'missing GPU samples'
        fields = ['memory_used_mib', 'utilization_percent', 'power_w', 'sm_clock_mhz', 'temperature_c']
        gpu_summary[str(index)] = dict(samples=len(rows), fields={field: dict(
            mean=statistics.mean(r[i] for r in rows), minimum=min(r[i] for r in rows),
            maximum=max(r[i] for r in rows)) for i, field in enumerate(fields)})
    return dict(scope='samples within compute window; not continuous peaks or contention proof',
        compute_window=window, host_samples=len(samples), cpu=fractions, gpu=gpu_summary,
        load1_mean=statistics.mean(float(x['loadavg'].split()[0]) for x in samples),
        minimum_host_available_bytes=min(x['memory']['MemAvailable_bytes'] for x in samples),
        background_gpu0_pids=sorted({p['pid'] for x in samples for p in x['gpu_processes']
                                     if p['gpu_uuid'] == GPU0}))
