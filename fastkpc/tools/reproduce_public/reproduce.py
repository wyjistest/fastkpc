#!/usr/bin/env python3
"""Build and qualify public inputs without private data or old native artifacts."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

TOOL = Path(__file__).resolve().parent
REPO = TOOL.parents[2]
PRODUCER = "2be6a816d3039a5f9340ba7018166d737ab11d9a"
PUBLIC = ("public500", "blocks256", "dense500", "dense1025")
ORDER = (("A1", "off"), ("C1", "on"), ("C2", "on"), ("A2", "off"), ("A3", "off"), ("C3", "on"))
LEGACY_SHA = "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f"
NATIVE = "source/fastkpc/build/fastkpc_cuda.so"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def capture(command, cwd=None, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def cpus(value, available=None):
    available = set(os.sched_getaffinity(0) if available is None else available)
    if value is None:
        return sorted(available)
    result = set()
    for item in value.split(","):
        bounds = item.split("-")
        require(1 <= len(bounds) <= 2 and all(x.isdecimal() for x in bounds), "Invalid CPU list")
        left, right = int(bounds[0]), int(bounds[-1])
        require(left <= right and right - left <= len(available), "Invalid CPU range")
        result.update(range(left, right + 1))
    require(result and result <= available, "Requested CPUs are outside the current process affinity")
    return sorted(result)


def relative_file(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe manifest path: {name}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), f"Missing/escaping file: {name}")
    return resolved


def verify_manifest(root, manifest, expected=None):
    if expected:
        require(sha(manifest) == expected, "Manifest identity mismatch")
    evidence = read(manifest)["evidence"]
    require(bool(evidence), "Empty evidence manifest")
    for name, expected_hash in evidence.items():
        require(sha(relative_file(root, name)) == expected_hash, f"Evidence mismatch: {name}")
    return len(evidence)


def performance(runs):
    require(set(runs) == {label for label, _ in ORDER}, "Need all six predeclared timing runs")
    for label, mode in ORDER:
        require(runs[label].get("mode") == mode, f"Wrong A/C mode: {label}")
        for field in ("outer_sec", "skeleton_sec", "orientation_sec"):
            value = runs[label].get(field)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(value) and (value > 0 if field == "outer_sec" else value >= 0),
                    f"Invalid timing: {label}/{field}")
    off = [runs[label]["outer_sec"] for label in ("A1", "A2", "A3")]
    on = [runs[label]["outer_sec"] for label in ("C1", "C2", "C3")]
    require(all(math.isfinite(x) and x > 0 for x in off + on), "Invalid complete-call timing")
    a, c = statistics.median(off), statistics.median(on)
    ratio = c / a
    speed = "PROVEN" if ratio <= .95 and max(on) < min(off) else ("REGRESSION" if ratio > 1.05 else "NOT_PROVEN")
    stages = {mode: {stage: statistics.median(r[stage] for r in runs.values() if r["mode"] == mode)
                     for stage in ("skeleton_sec", "orientation_sec")} for mode in ("off", "on")}
    return dict(off_outer_sec=off, on_outer_sec=on, off_median=a, on_median=c, ratio=ratio,
                non_regression=ratio <= 1.05, speed=speed, stage_medians=stages)


def gpu_processes():
    output = capture(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"])
    # A single PID can own contexts on several devices; retain every row.
    return [dict(pid=int(row[0]), gpu_uuid=row[1].strip()) for row in csv.reader(output.splitlines()) if row]


def environment(gpu):
    env = {key: value for key, value in os.environ.items() if not key.startswith("FASTKPC_")}
    require(not any(env.get(key) for key in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_LAUNCH_BLOCKING")),
            "Unset LD_PRELOAD, CUDA_INJECTION64_PATH and CUDA_LAUNCH_BLOCKING before reproduction")
    env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_MAX_CONNECTIONS="32", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE="1")
    r_home = capture(["R", "RHOME"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(Path(r_home) / "lib"),
        "/usr/local/cuda/lib64", env.get("LD_LIBRARY_PATH", "")]))
    return env


def doctor(gpu, cpu_list):
    require(sys.platform == "linux", "The frozen producer requires Linux")
    for name in ("git", "R", "Rscript", "nvidia-smi", "taskset", "flock", "sh", "python3"):
        require(shutil.which(name), f"Missing executable: {name}")
    # These are requirements of the unchanged, frozen build entry.
    require(Path("/usr/local/cuda/bin/nvcc").is_file(), "CUDA must be available at /usr/local/cuda")
    inventory = list(csv.reader(capture(["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version",
                                        "--format=csv,noheader"]).splitlines()))
    matches = [row for row in inventory if gpu in (row[0].strip(), row[1].strip())]
    require(len(matches) == 1, "Select an available physical GPU index or full UUID")
    selected = [value.strip() for value in matches[0]]
    require(selected[3] == "8.9", "The frozen build targets sm_89; this entry qualifies compute capability 8.9 only")
    env = environment(selected[1])
    with tempfile.TemporaryDirectory(prefix="fastkpc-public-env-") as temporary:
        output = Path(temporary) / "R.json"
        subprocess.run(["Rscript", "--vanilla", str(TOOL / "environment.R"), str(output)], env=env, check=True)
        r_environment = read(output)
    compiler = capture(["R", "CMD", "config", "CXX17"])
    require(shutil.which(compiler), f"R CXX17 compiler is unavailable: {compiler}")
    return dict(platform=platform.platform(), python=sys.version, cpu_affinity=cpus(cpu_list),
                selected_gpu=dict(zip(("index", "uuid", "name", "compute_capability", "memory", "driver"), selected)),
                gpu_inventory=inventory, R=r_environment, CXX17=compiler,
                compiler_version=capture([compiler, "--version"]),
                nvcc=capture(["/usr/local/cuda/bin/nvcc", "--version"]))


def same_environment(original, current):
    # Inventory of other GPUs is observational, not a qualification requirement.
    for field in ("platform", "python", "cpu_affinity", "R", "CXX17", "compiler_version", "nvcc"):
        require(current[field] == original[field], f"Frozen environment changed: {field}")
    for field in ("uuid", "name", "compute_capability", "memory", "driver"):
        require(current["selected_gpu"][field] == original["selected_gpu"][field],
                f"Frozen GPU changed: {field}")


def terminate_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Campaign:
    def __init__(self, root):
        self.root = root.resolve()
        self.config = read(root / "protocol.json")
        require(sha(Path(__file__).resolve()) == self.config["frozen_files"]["tools/reproduce.py"],
                "Driver differs from frozen protocol; resume with work-dir/tools/reproduce.py")
        self.source = root / "source"
        self.env = environment(self.config["gpu_uuid"])
        self.cpu_list = ",".join(map(str, self.config["cpu_affinity"]))
        self.frozen()
        same_environment(read(root / "environment.json"), doctor(self.config["gpu_uuid"], self.cpu_list))

    def frozen(self):
        for name, expected in self.config["frozen_files"].items():
            require(sha(relative_file(self.root, name)) == expected, f"Frozen file changed: {name}")
        if (self.root / "build.json").exists():
            for name, expected in read(self.root / "build.json")["artifacts"].items():
                require(sha(relative_file(self.root, name)) == expected, f"Build identity changed: {name}")
        if (self.root / "inputs.json").exists():
            for name, expected in read(self.root / "inputs.json")["artifacts"].items():
                require(sha(relative_file(self.root, name)) == expected, f"Input identity changed: {name}")

    def execute(self, name, command, kind="host"):
        self.frozen()
        record = self.root / "records" / (name + ".json")
        require(not record.exists(), f"Existing process record: {name}; use a new work directory after a failed attempt")
        active = gpu_processes() if kind in ("correctness", "timing") else []
        require(kind != "correctness" or all(p["gpu_uuid"] != self.config["gpu_uuid"] for p in active), "Selected GPU is busy")
        require(kind != "timing" or not active, "Timing requires every GPU free of external compute")
        print("RUN", name, flush=True)
        start = time.time(); process = None; samples = []; unexpected = set(); other = set()
        try:
            with (self.root / "records" / (name + ".log")).open("x") as log:
                process = subprocess.Popen(["taskset", "-c", self.cpu_list, *command], cwd=self.source,
                    env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while True:
                    if kind in ("correctness", "timing"):
                        observed = gpu_processes()
                        samples.append(dict(epoch=time.time(), processes=observed))
                        for entry in observed:
                            pid, uuid = entry["pid"], entry["gpu_uuid"]
                            try:
                                own = os.getpgid(pid) == process.pid
                            except ProcessLookupError:
                                continue
                            if own:
                                continue
                            if uuid != self.config["gpu_uuid"]:
                                other.add(pid)
                            if kind == "timing" or uuid == self.config["gpu_uuid"]:
                                unexpected.add(pid)
                        if unexpected:
                            terminate_owned(process)
                            break
                    try:
                        process.wait(timeout=3)
                        break
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            if process is not None:
                terminate_owned(process)
            write(record, dict(command=["taskset", "-c", self.cpu_list, *command], kind=kind,
                start_epoch=start, end_epoch=time.time(), exit_code=None if process is None else process.returncode,
                gpu_uuid=self.config["gpu_uuid"], external_compute=sorted(unexpected), other_gpu_processes=sorted(other),
                machine_isolated=kind == "timing" and not unexpected and not other, samples=samples))
        require(process is not None and process.returncode == 0 and not unexpected,
                f"{name} failed; inspect records/{name}.log and .json; all evidence retained")
        self.frozen()

    def build(self):
        if (self.root / "build.json").exists():
            self.frozen()
            return
        build_dir = self.source / "fastkpc/build"
        require(not build_dir.exists(), "Full build requires no existing objects; failed build is retained")
        # The frozen shell entry has unquoted include flags. A temporary logical
        # path without spaces preserves its bytes while supporting spaced workdirs.
        with tempfile.TemporaryDirectory(prefix="fastkpc-public-build-") as temporary:
            link = Path(temporary) / "source"
            link.symlink_to(self.source, target_is_directory=True)
            self.execute("build", ["sh", "-x", str(link / "fastkpc/tools/build_cuda_native.sh")])
        commands = (self.root / "records/build.log").read_text().splitlines()
        compiles = [line for line in commands if line.startswith("+ ") and " -c " in line]
        objects = list(build_dir.glob("*.o"))
        require(len(compiles) == len(objects) == 44, "Expected 44 full compiler invocations and objects")
        method = [line for line in compiles if "full_cuda_ci_method_batch.cu" in line]
        require(len(method) == 1 and "--fmad=false" in method[0], "Method-batch numerical flags mismatch")
        artifacts = objects + [build_dir / name for name in
            ("fastkpc_cuda.so", "recorded_dbdsqr.inc", "recorded_dbdsqr_fixed_vt.inc")]
        write(self.root / "build.json", dict(status="PASS_CLEAN_FULL_BUILD", producer_commit=self.config["producer_commit"],
            compile_count=44, parent_objects_reused=False, native_sha256=sha(self.root / NATIVE),
            artifacts={str(path.relative_to(self.root)): sha(path) for path in artifacts}))
        print("PASS full build", sha(self.root / NATIVE), flush=True)

    def inputs(self):
        if (self.root / "inputs.json").exists():
            self.frozen()
            return
        self.execute("prepare", ["Rscript", "--vanilla", str(self.root / "tools/prepare.R"),
            str(self.root / "inputs"), ",".join(self.config["cases"])])
        inputs = read(self.root / "inputs/manifest.json")
        require(set(inputs) == set(self.config["cases"]), "Generated input set mismatch")
        artifacts = {"inputs/manifest.json": sha(self.root / "inputs/manifest.json")}
        for entry in inputs.values():
            path = relative_file(self.root, entry["path"])
            require(sha(path) == entry["file_sha256"], "Generated input hash mismatch")
            artifacts[entry["path"]] = entry["file_sha256"]
        write(self.root / "inputs.json", dict(artifacts=artifacts))

    def one(self, phase, case, label, mode, reference=None):
        self.frozen()
        directory = self.root / phase / f"{case}-{label}" if label else self.root / phase / case
        record_name = phase + "-" + directory.name
        binding = self.root / "records" / (record_name + "-result.json")
        if directory.exists():
            report_path = directory / "report.json"
            require(report_path.exists() and (self.root / "records" / (record_name + ".json")).exists(),
                    f"Incomplete job {record_name}; retained for inspection")
            record = read(self.root / "records" / (record_name + ".json"))
            require(record["exit_code"] == 0 and not record["external_compute"], f"Rejected job {record_name}")
            require(binding.exists(), f"Unsealed completed job {record_name}; inspect retained evidence")
            for name, expected in read(binding).items():
                require(sha(relative_file(self.root, name)) == expected, f"Completed result changed: {name}")
        else:
            path = self.root / read(self.root / "inputs/manifest.json")[case]["path"]
            expected = read(self.root / "inputs/manifest.json")[case]["file_sha256"]
            require(sha(path) == expected, "Input identity changed")
            command = ["Rscript", "--vanilla", str(self.root / "tools/run.R"), mode, str(path),
                       str(directory), str(reference) if reference else "-", "timing" if phase == "timing" else "correctness"]
            self.execute(record_name, command, "host" if mode == "cpu" else
                         ("timing" if phase == "timing" else "correctness"))
        report = read(directory / "report.json")
        require(report["pass"] and all(report.get("gates", {}).values()), f"Numerical gates failed: {record_name}")
        if mode != "cpu":
            require(report["native_sha256"] == read(self.root / "build.json")["native_sha256"], "Loaded native mismatch")
        if not binding.exists():
            write(binding, {str(path.relative_to(self.root)): sha(path) for path in
                            (directory / "report.json", directory / "receipt.rds",
                             self.root / "records" / (record_name + ".json"))})
        return report

    def qualify(self):
        reports = {}
        for phase in ("references", "qualification"):
            (self.root / phase).mkdir(exist_ok=True)
        for case in self.config["cases"]:
            self.one("references", case, "", "cpu")
            self.one("qualification", case, "replay", "replay", self.root / "references" / case / "receipt.rds")
            reference = self.root / "qualification" / f"{case}-replay/receipt.rds"
            reports[case] = {mode: self.one("qualification", case, mode, mode, reference) for mode in ("off", "on")}
        path = self.root / "qualification/summary.json"
        value = dict(pass_=True, native_sha256=read(self.root / "build.json")["native_sha256"], reports=reports)
        value["pass"] = value.pop("pass_")
        if path.exists():
            require(read(path) == value, "Qualification summary mismatch")
        else:
            write(path, value)

    def timing(self):
        (self.root / "timing").mkdir(exist_ok=True)
        reports = {}
        for case in self.config["cases"]:
            reference = self.root / "qualification" / f"{case}-replay/receipt.rds"
            runs = {label: self.one("timing", case, label, mode, reference) for label, mode in ORDER}
            reports[case] = dict(runs=runs, **performance(runs))
        value = {"pass": True, "native_sha256": read(self.root / "build.json")["native_sha256"], "reports": reports}
        path = self.root / "timing/summary.json"
        if path.exists():
            require(read(path) == value, "Timing summary mismatch")
        else:
            write(path, value)

    def seal(self):
        self.frozen()
        require(not (self.root / "manifest.json").exists(), "Already sealed; use verify")
        self.execute("receipt-audit", ["Rscript", "--vanilla", str(self.root / "tools/audit_receipts.R"),
            str(self.root / "references"), str(self.root / "qualification"), str(self.root / "timing"),
            str(self.root / "receipt_audit.json")])
        audited = read(self.root / "receipt_audit.json")
        require(audited["pass"] and not audited["native_loaded"], "Independent raw-receipt audit failed")
        timed = read(self.root / "timing/summary.json")["reports"]
        records = [read(self.root / "records" / f"timing-{case}-{label}.json")
                   for case in self.config["cases"] for label, _ in ORDER]
        require(all(p["exit_code"] == 0 and p["machine_isolated"] for p in records), "Timing isolation failed")
        require(all(b["start_epoch"] >= a["end_epoch"] for a, b in zip(records, records[1:])), "Timing order mismatch")
        rows = {case: performance(row["runs"]) for case, row in timed.items()}
        summary = dict(status="PASS_PUBLIC_INPUT_QUALIFICATION", producer_commit=self.config["producer_commit"],
            native_sha256=read(self.root / "build.json")["native_sha256"], cases=self.config["cases"],
            correctness_runs=2 * len(rows), timing_runs=6 * len(rows), raw_receipt_runs=audited["run_count"],
            ci_comparisons=audited["ci_comparisons"], orientation_ci_comparisons=audited["orientation_ci_comparisons"],
            max_ci_error=max(run["ci_error"] for runs in audited["cases"].values() for run in runs.values()),
            max_pmax_error=max(run["pmax_error"] for runs in audited["cases"].values() for run in runs.values()),
            rows=rows, all_inputs_non_regression=all(row["non_regression"] for row in rows.values()),
            evidence_class="fresh public-input development timing of two GPU configurations; not CPU-vs-GPU or promotion",
            phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False)
        write(self.root / "summary.json", summary)
        evidence = {}
        for path in sorted(self.root.rglob("*")):
            name = path.relative_to(self.root)
            if path.is_file() and not any(part in (".git", "__pycache__") for part in name.parts) and name.name != ".run.lock":
                relative_file(self.root, str(name))
                evidence[str(name)] = sha(path)
        write(self.root / "manifest.json", dict(schema="fastkpc-public-reproduction-v1",
            producer_commit=self.config["producer_commit"], native_sha256=summary["native_sha256"], evidence=evidence))
        print(json.dumps(summary, indent=2), flush=True)
        print("MANIFEST_SHA256", sha(self.root / "manifest.json"), flush=True)


def initialize(args):
    root = args.work_dir.resolve()
    require(not root.exists(), "Work directory already exists; use resume or a new directory")
    cases = args.cases.split(",")
    require(bool(cases) and len(set(cases)) == len(cases) and set(cases) <= set(PUBLIC), "Select public inputs only")
    host = doctor(args.gpu, args.cpus)
    producer = capture(["git", "rev-parse", args.producer + "^{commit}"], cwd=args.repo)
    root.mkdir(parents=True)
    (root / "records").mkdir(); (root / "tools").mkdir()
    write(root / "environment.json", host)
    with (root / "records/checkout.log").open("x") as log:
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(args.repo), str(root / "source")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", producer], cwd=root / "source",
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    tracked = capture(["git", "ls-files"], cwd=root / "source").splitlines()
    require(not (root / "source/fastkpc/build").exists(), "Checkout unexpectedly contains build outputs")
    require(sha(root / "source/kpcalg/R/dcovgamma.R") == LEGACY_SHA, "Original kpcalg reference identity mismatch")
    for name in ("reproduce.py", "prepare.R", "environment.R", "audit_receipts.R"):
        shutil.copyfile(TOOL / name, root / "tools" / name)
    shutil.copyfile(args.repo / "fastkpc/tools/research/clean_general_inputs_20261106/run.R", root / "tools/run.R")
    frozen = {"source/" + name: sha(root / "source" / name) for name in tracked}
    production = capture(["git", "ls-files", "fastkpc/src", "fastkpc/R", "kpcalg/R",
                          "fastkpc/tools/build_cuda_native.sh", "fastkpc/tools/generate_recorded_dbdsqr.py"],
                         cwd=root / "source").splitlines()
    for name in production:
        blob = subprocess.check_output(["git", "show", producer + ":" + name], cwd=root / "source")
        require(hashlib.sha256(blob).hexdigest() == frozen["source/" + name], f"Git source bytes mismatch: {name}")
    frozen.update({str(path.relative_to(root)): sha(path) for path in (root / "tools").iterdir()})
    frozen["environment.json"] = sha(root / "environment.json")
    write(root / "protocol.json", dict(schema="fastkpc-public-reproduction-v1", producer_commit=producer,
        harness_git_head=capture(["git", "rev-parse", "HEAD"], cwd=args.repo),
        cases=cases, gpu_uuid=host["selected_gpu"]["uuid"], cpu_affinity=host["cpu_affinity"],
        frozen_files=frozen, production_source_count=len(production), order=ORDER,
        ci_tolerance=1e-10, non_regression_ratio=1.05, speed_ratio=.95,
        speed_requires_slowest_on_below_fastest_off=True, threads=1, cuda_connections=32,
        result_cache="reset after unrelated 80x3 warmup", requested_numCol="omitted/default floor(n/10)",
        max_conditioning_size="Inf/natural stopping", primary_timing="full public WAN-PDAG call",
        parent_objects_reused=False, expected_compile_count=44, private_inputs_used=False,
        expected_native_sha256=None, performance_claim="per-input A/C, not CPU-vs-GPU; no promotion"))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="fresh checkout, full build, references, correctness, six-run timing and audit")
    run.add_argument("--repo", type=Path, default=REPO)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--producer", default=PRODUCER)
    run.add_argument("--gpu", default="0", help="physical index or UUID")
    run.add_argument("--cpus", help="allowed CPU list, e.g. 0-19; default: current affinity")
    run.add_argument("--cases", default="public500,blocks256", help=",".join(PUBLIC))
    run.add_argument("--until", choices=("prepare", "build", "qualify", "timing"), default="timing")
    resume = commands.add_parser("resume", help="continue completed stages/jobs using the frozen protocol")
    resume.add_argument("--work-dir", type=Path, required=True)
    resume.add_argument("--until", choices=("prepare", "build", "qualify", "timing"), default="timing")
    check = commands.add_parser("doctor", help="report required tools, R packages and selected GPU")
    check.add_argument("--gpu", default="0"); check.add_argument("--cpus")
    verify = commands.add_parser("verify", help="verify a relocated new or historical evidence tree without GPU/R")
    verify.add_argument("--work-dir", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-sha256", help="trusted manifest SHA from the evidence index")
    bundle = commands.add_parser("bundle", help="archive sealed public evidence, sources and native; excludes .git")
    bundle.add_argument("--work-dir", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(doctor(args.gpu, args.cpus), indent=2)); return
    if args.command == "verify":
        count = verify_manifest(args.work_dir.resolve(), args.manifest, args.expected_sha256)
        print("PASS", count, "evidence hashes; external manifest anchor:", bool(args.expected_sha256)); return
    if args.command == "bundle":
        root = args.work_dir.resolve(); manifest = root / "manifest.json"
        require(not args.output.exists() and not args.output.resolve().is_relative_to(root), "Use a new archive outside the evidence tree")
        require(read(manifest).get("schema") == "fastkpc-public-reproduction-v1", "Only public reproduction trees may be bundled")
        require(set(read(root / "protocol.json")["cases"]) <= set(PUBLIC), "Public inputs only")
        count = verify_manifest(root, manifest)
        with tarfile.open(args.output, "x:gz") as archive:
            for name in [*read(manifest)["evidence"], "manifest.json"]:
                archive.add(relative_file(root, name), arcname="public-reproduction/" + name, recursive=False)
        print("PASS public bundle", count, "files; archive SHA256", sha(args.output)); return
    root = initialize(args) if args.command == "run" else args.work_dir.resolve()
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(root)
        require(set(campaign.config["cpu_affinity"]) <= set(os.sched_getaffinity(0)), "Frozen CPU affinity unavailable")
        campaign.inputs()
        if args.until == "prepare": return
        campaign.build()
        if args.until == "build": return
        campaign.qualify()
        if args.until == "qualify": return
        campaign.timing()
        campaign.seal()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as error:
        print("FAIL:", error, file=sys.stderr)
        sys.exit(1)
