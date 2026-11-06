"""Correctness jobs require an exclusive selected GPU, not an idle second GPU."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import time

spec = importlib.util.spec_from_file_location("sealed_jobs", Path(__file__).resolve().parent.parent /
    "stage_b_work_pair_runtime_20261028/support.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
ROOT, sha, read, write, check_map = base.ROOT, base.sha, base.read, base.write, base.check_map
environment, processes = base.environment, base.processes

def job(out, name, command, env, frozen):
    frozen()
    gpu_index = env["CUDA_VISIBLE_DEVICES"]
    assert gpu_index in ("0", "1")
    selected = subprocess.check_output(["nvidia-smi", "-i", gpu_index,
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    assert selected not in processes().values(), "external task on selected GPU; no job started"
    samples, unexpected, other_gpu = [], set(), set()
    started = time.time()
    with (out / (name + ".log")).open("x") as log, (out / (name + "-gpu.csv")).open("x") as gpu:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits", "-lms", "200"], stdout=gpu, stderr=subprocess.DEVNULL)
        proc = None
        try:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                active = processes()
                samples.append(dict(epoch=time.time(), processes=active))
                for pid, uuid in active.items():
                    if uuid != selected:
                        other_gpu.add(pid)
                        continue
                    try:
                        if os.getpgid(pid) != proc.pid: unexpected.add(pid)
                    except ProcessLookupError:
                        pass
                if unexpected:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try: proc.wait(timeout=10)
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
    write(out / (name + "-process.json"), dict(command=command, exit_code=proc.returncode,
        start_epoch=started, end_epoch=time.time(), affinity=sorted(os.sched_getaffinity(0)),
        isolated=not unexpected, isolation_scope="selected GPU only; correctness, not timing evidence",
        selected_gpu=selected, selected_gpu_index=gpu_index,
        machine_isolated=not unexpected and not other_gpu, other_gpu_processes=sorted(other_gpu),
        unexpected=sorted(unexpected), samples=samples))
    assert not unexpected and proc.returncode == 0, name + " failed; evidence retained"
    frozen()
    print("PASS", name, flush=True)
