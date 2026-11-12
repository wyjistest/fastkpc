"""Fresh original-R references, clean-native qualification, then interleaved timing."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clean_build", TOOL / "build.py")
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)
ROOT, BASE, CHECKOUT = build_module.ROOT, build_module.BASE, build_module.CHECKOUT
sha, write, git = build_module.sha, build_module.write, build_module.git
CASES = ["real179", "real243", "real300", "public500", "dense500", "blocks256", "dense1025"]
ORDER = [("A1", "off"), ("C1", "on"), ("C2", "on"), ("A2", "off"), ("A3", "off"), ("C3", "on")]

def read(path):
    return json.loads(Path(path).read_text())

def processes():
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    return {int(line.split(",")[0]): line.split(",")[1].strip() for line in text.splitlines() if line.strip()}

def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("FASTKPC_")}
    assert not any(env.get(k) for k in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_LAUNCH_BLOCKING"))
    env.update(CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_MAX_CONNECTIONS="32", OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE="1")
    env["LD_LIBRARY_PATH"] = "/usr/local/lib/R/lib:/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    return env

def stop_owned(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

def job(out, name, command, env, frozen, kind):
    frozen()
    selected = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    active = processes()
    if kind != "cpu":
        assert selected not in active.values(), "selected GPU occupied; no job started"
    if kind == "timing":
        assert not active, "performance requires both GPUs free; no job started"
    start = time.time(); samples = []; intruders = set(); other_gpu = set()
    with (out / (name + ".log")).open("x") as log, (out / (name + "-gpu.csv")).open("x") as gpu:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu",
            "--format=csv,noheader,nounits", "-lms", "200"], stdout=gpu, stderr=subprocess.DEVNULL)
        proc = None
        try:
            proc = subprocess.Popen(command, cwd=CHECKOUT, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            while True:
                active = processes(); samples.append(dict(epoch=time.time(), processes=active))
                for pid, uuid in active.items():
                    try:
                        own = os.getpgid(pid) == proc.pid
                    except ProcessLookupError:
                        continue
                    if own: continue
                    if uuid != selected: other_gpu.add(pid)
                    if kind != "cpu" and (uuid == selected or kind == "timing"):
                        intruders.add(pid)
                if intruders:
                    stop_owned(proc)
                    break
                try:
                    proc.wait(timeout=3)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            monitor.terminate(); monitor.wait(timeout=10)
            if proc is not None: stop_owned(proc)
    write(out / (name + "-process.json"), dict(command=command, cwd=str(CHECKOUT),
        exit_code=proc.returncode, start_epoch=start, end_epoch=time.time(), selected_gpu=selected,
        cpu_affinity="0-19", kind=kind, external_selected_or_timing_gpu=sorted(intruders),
        other_gpu_processes=sorted(other_gpu), machine_isolated=not intruders and not other_gpu,
        samples=samples))
    assert proc.returncode == 0 and not intruders, name + " failed; retained logs and receipt"
    frozen()
    report = read(out / name / "report.json")
    assert report["pass"], name
    print("PASS", name, "CI", report["ci_count"], flush=True)
    return report

def main():
    action, tag = sys.argv[1:3]
    assert action in ("references", "qualify", "timing") and Path(tag).name == tag
    out = BASE / tag; out.mkdir()
    assert git("rev-parse", "HEAD") == build_module.COMMIT and not git("status", "--porcelain")
    source_map = read(BASE / "build/protocol.json")["source_sha256"]
    inputs = read(BASE / "inputs/manifest.json")
    assert set(inputs) == set(CASES)
    files = {str(p): sha(p) for p in TOOL.glob("*") if p.is_file()}
    files.update({str(CHECKOUT / p): h for p, h in source_map.items()})
    files.update({spec["path"]: spec["file_sha256"] for spec in inputs.values()})
    snapshot = out / "tools"; snapshot.mkdir()
    for p in TOOL.glob("*"):
        if p.is_file(): shutil.copyfile(p, snapshot / p.name)
    build = None; previous = None
    if action != "references":
        build = read(BASE / "build/build.json")
        assert build["status"] == "PASS_CLEAN_FULL_BUILD" and build["source_sha256"] == source_map
        files[build["native_path"]] = build["native_sha256"]
        previous = BASE / sys.argv[3]
        prior = read(previous / "summary.json")
        assert prior["pass"] and set(prior["reports"]) == set(CASES)
        for case in CASES:
            stem = case if action == "qualify" else case + "-replay"
            path = previous / stem / "receipt.rds"
            files[str(path)] = sha(path)
    def frozen():
        assert not git("status", "--porcelain")
        for path, expected in files.items():
            assert sha(path) == expected, path
    write(out / "protocol.json", dict(action=action, producer_commit=build_module.COMMIT,
        native_sha256=None if build is None else build["native_sha256"], cases=CASES, inputs=inputs,
        dependencies=files, previous=None if previous is None else str(previous),
        order=ORDER if action == "timing" else ["off", "on"],
        cpu_authority="original kpcalg R, pcalg stable and mgcv/RSpectra; no numerical native substitution",
        requested_numCol="omitted/default floor(n/10)", max_conditioning_size="Inf/natural stopping",
        tolerance=1e-10, non_regression_ratio=1.05, speed_ratio=0.95,
        speed_requires_slowest_on_below_fastest_off=True,
        performance_boundary="full public WAN-PDAG; artifact operations excluded; unrelated warmup, fresh input caches",
        phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False))
    env = environment()
    def command(case, mode, stem, reference="-", phase="correctness"):
        return ["taskset", "-c", "0-19", "Rscript", "--vanilla", str(TOOL / "run.R"), mode,
            inputs[case]["path"], str(out / stem), str(reference), phase]
    reports = {}
    if action == "references":
        def one(case):
            return case, job(out, case, command(case, "cpu", case), env, frozen, "cpu")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for case, report in pool.map(one, CASES):
                reports[case] = report
    elif action == "qualify":
        for case in CASES:
            ref = previous / case / "receipt.rds"
            replay = case + "-replay"
            job(out, replay, command(case, "replay", replay, ref), env, frozen, "correctness")
            reports[case] = {}
            for mode in ("off", "on"):
                stem = case + "-" + mode
                report = job(out, stem, command(case, mode, stem, out / replay / "receipt.rds"),
                    env, frozen, "correctness")
                assert report["native_sha256"] == build["native_sha256"]
                reports[case][mode] = report
    else:
        for case in CASES:
            rows = {}
            for label, mode in ORDER:
                stem = case + "-" + label
                report = job(out, stem, command(case, mode, stem, previous / (case + "-replay/receipt.rds"), "timing"),
                    env, frozen, "timing")
                assert report["native_sha256"] == build["native_sha256"]
                rows[label] = report
            a = [rows[label]["outer_sec"] for label in ("A1", "A2", "A3")]
            c = [rows[label]["outer_sec"] for label in ("C1", "C2", "C3")]
            ratio = statistics.median(c) / statistics.median(a)
            reports[case] = dict(runs=rows, off_outer_sec=a, on_outer_sec=c, ratio=ratio,
                off_median=statistics.median(a), on_median=statistics.median(c),
                non_regression=ratio <= 1.05,
                speed="PROVEN" if ratio <= .95 and max(c) < min(a) else ("REGRESSION" if ratio > 1.05 else "NOT_PROVEN"))
    frozen()
    write(out / "summary.json", dict(action=action, reports=reports,
        producer_commit=build_module.COMMIT, native_sha256=None if build is None else build["native_sha256"],
        evidence_class="fresh original-reference correctness" if action != "timing" else "per-input six-run compute-warm development timing; no promotion",
        **{"pass": True}))
    print("PASS", action, len(reports), "inputs", flush=True)

if __name__ == "__main__":
    main()
