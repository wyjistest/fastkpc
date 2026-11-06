"""Verify all receipts and install only the qualified rank-general native."""
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rank_jobs", TOOL / "jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106"
build_dir, campaign, supplement, public, rank_one = [BASE / p for p in sys.argv[1:6]]
action = sys.argv[6]
assert action in ("audit", "install")
build = jobs.read(build_dir / "build.json")
native = Path(build["native_path"])
assert jobs.sha(native) == build["native_sha256"]
jobs.check_map(build["source_sha256"])
for name, digest in build["objects"].items():
    assert jobs.sha(build_dir / "objects" / name) == digest
protocol = jobs.read(campaign / "protocol.json")
jobs.check_map(protocol["dependencies"])
summary = jobs.read(campaign / "summary.json")
assert summary["status"] == "PASS" and len(summary["reports"]) == 50
assert protocol["native_sha256"] == build["native_sha256"]
parent_dir = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102/candidate4"
parent = jobs.read(parent_dir / "build.json")
assert build["objects"]["legacy_dcov_gamma_cpp.o"] == parent["objects"]["legacy_dcov_gamma_cpp.o"]
for reused in summary.get("reused_successes", []):
    original, name = Path(reused["original"]), reused["job"]
    assert not reused["rerun"]
    for relative, digest in ((name + "/report.json", reused["report_sha256"]),
                             (name + "-process.json", reused["process_sha256"])):
        assert jobs.sha(original / relative) == jobs.sha(campaign / relative) == digest
reports = []
shared_host_jobs = []
for case, rank in protocol["cases"]:
    modes = ["cpu", "off", "on"]
    if [case, rank] in protocol["uncached_cases"]: modes += ["uncached"]
    for mode in modes:
        stem = case + "-k" + rank + "-" + mode
        process = jobs.read(campaign / (stem + "-process.json"))
        report = jobs.read(campaign / stem / "report.json")
        assert process["isolated"] and process["exit_code"] == 0 and not process["unexpected"]
        assert report["pass"]
        if process.get("machine_isolated") is False: shared_host_jobs.append(stem)
        if mode == "cpu":
            assert report["native_sha256"] == protocol["cpu_native_sha256"]
        else:
            assert report["native_sha256"] == build["native_sha256"]
            assert all(report["gates"].values())
            assert report["ci"]["outside_tolerance"] == report["pmax"]["outside_tolerance"] == 0
            assert max(report["ci"]["max_abs_error"], report["pmax"]["max_abs_error"]) <= 1e-10
            reports.append(report)
supp = jobs.read(supplement / "closure.json")
assert supp["status"] == "PASS" and supp["native_sha256"] == build["native_sha256"]
jobs.check_map(supp["dependencies"])
for name in supp["tests"]:
    process = jobs.read(supplement / (name + "-process.json"))
    assert process["isolated"] and process["exit_code"] == 0 and not process["unexpected"]
    log = (supplement / (name + ".log")).read_text()
    assert "SKIP" not in log and "PASS" in log, name
assert "ERROR SUMMARY: 0 errors" in (supplement / "memcheck.log").read_text()
methods = jobs.read(supplement / "methods/report.json")
assert methods["status"] == "PASS" and len(methods["cases"]) == 28
assert methods["native_sha256"] == build["native_sha256"]
assert all(r["pass"] and all(r["gates"].values()) for r in methods["cases"].values())
canonical = jobs.read(supplement / "canonical/report.json")
assert canonical["status"] == "PASS" and all(canonical["gates"].values())
assert canonical["ci_count"] == 240498 and canonical["native_sha256"] == build["native_sha256"]
public_report = jobs.read(public / "run/report.json")
public_process = jobs.read(public / "run-process.json")
assert public_report["status"] == "PASS" and all(public_report["gates"].values())
assert public_report["native_sha256"] == build["native_sha256"]
assert public_process["exit_code"] == 0 and public_process["isolated"] and not public_process["unexpected"]
jobs.check_map(jobs.read(public / "protocol.json")["dependencies"])
rank_protocol = jobs.read(rank_one / "protocol.json")
jobs.check_map(rank_protocol["dependencies"])
rank_report = jobs.read(rank_one / "run/report.json")
rank_process = jobs.read(rank_one / "run-process.json")
assert rank_report["status"] == "PASS" and rank_report["native_sha256"] == build["native_sha256"]
assert rank_process["exit_code"] == 0 and rank_process["isolated"] and not rank_process["unexpected"]
assert len(rank_report["preserved_cases"]) == 21 and all(r["bitwise"] for r in rank_report["preserved_cases"])
assert len(rank_report["scalar_cases"]) == 3
assert all(check["pass"] for r in rank_report["scalar_cases"] for check in r.values())
assert len(rank_report["cases"]) == 2 and all(all(r["gates"].values()) for r in rank_report["cases"].values())
assert rank_report["parent_source_sha256"] == rank_protocol["original_source_sha256"]
assert rank_report["current_source_sha256"] == rank_protocol["current_source_sha256"] == jobs.sha(ROOT / "kpcalg/R/dcovgamma.R")
audit = dict(status="PASS_NUMCOL_GENERALIZATION", native_sha256=build["native_sha256"],
    source_sha256=build["source_sha256"], campaign=str(campaign), supplement=str(supplement),
    input_rank_cases=len(protocol["cases"]), gpu_runs=len(reports),
    ci_comparisons=sum(r["ci_count"] for r in reports),
    max_error=max(max(r["ci"]["max_abs_error"], r["pmax"]["max_abs_error"]) for r in reports),
    methods_cases=len(methods["cases"]), method_ci_comparisons=sum(r["ci_count"] for r in methods["cases"].values()),
    method_max_error=max(r["ci_error"] for r in methods["cases"].values()),
    canonical_ci_count=canonical["ci_count"], canonical_elapsed_sec=canonical["elapsed_sec"],
    public_default_ci_count=public_report["ci_count"], mathematical_flags_unchanged=True,
    rank_one_cases=len(rank_report["cases"]),
    rank_one_ci_count=sum(r["ci_count"] for r in rank_report["cases"].values()),
    rank_one_max_error=max(max(r["ci"]["max_abs_error"], r["pmax"]["max_abs_error"]) for r in rank_report["cases"].values()),
    legacy_r_source_sha256=rank_report["current_source_sha256"],
    legacy_r_parent_source_sha256=rank_report["parent_source_sha256"],
    reused_success_count=len(summary.get("reused_successes", [])), shared_host_jobs=shared_host_jobs,
    evidence_class="incremental development qualification; no new performance median",
    phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False)
if action == "audit":
    jobs.write(BASE / "audit.json", audit)
    print({k: v for k, v in audit.items() if k != "source_sha256"}, flush=True)
else:
    assert jobs.read(BASE / "audit.json") == audit
    active = jobs.processes()
    selected_gpu = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid",
                                           "--format=csv,noheader"], text=True).strip()
    assert selected_gpu not in active.values(), "selected GPU must be idle for installation"
    assert jobs.sha(parent_dir / "fastkpc_cuda.so") == parent["native_sha256"]
    dest = ROOT / "fastkpc/build"
    with (dest / "fastkpc_cuda.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert jobs.sha(dest / "fastkpc_cuda.so") == parent["native_sha256"]
        for name, digest in parent["objects"].items():
            assert jobs.sha(dest / name) == jobs.sha(parent_dir / "objects" / name) == digest
        mapping = {name: build_dir / "objects" / name for name in build["objects"]}
        for name in ("recorded_dbdsqr.inc", "recorded_dbdsqr_fixed_vt.inc"):
            mapping[name] = build_dir / "source/fastkpc/build" / name
        mapping["fastkpc_cuda.so"] = native
        for name, source in mapping.items():
            pending = dest / (name + ".numcol.tmp")
            assert not pending.exists()
            shutil.copy2(source, pending)
            assert jobs.sha(pending) == jobs.sha(source)
            os.replace(pending, dest / name)
        jobs.write(BASE / "installation.json", dict(status="PASS_INSTALLED", epoch=time.time(),
            previous_native_sha256=parent["native_sha256"], native_sha256=build["native_sha256"],
            other_gpu_processes=active, objects=build["objects"]))
    evidence = {}
    for directory in (campaign, supplement, public, rank_one):
        for path in directory.rglob("*"):
            if path.is_file(): evidence[str(path.relative_to(BASE))] = jobs.sha(path)
    for path in (BASE / "audit.json", BASE / "installation.json", build_dir / "build.json", build_dir / "protocol.json"):
        evidence[str(path.relative_to(BASE))] = jobs.sha(path)
    legacy_parent = BASE / "rank-one-parent/dcovgamma.R"
    evidence[str(legacy_parent.relative_to(BASE))] = jobs.sha(legacy_parent)
    legacy_current = BASE / "legacy-r-source/dcovgamma.R"
    legacy_current.parent.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "kpcalg/R/dcovgamma.R", legacy_current)
    evidence[str(legacy_current.relative_to(BASE))] = jobs.sha(legacy_current)
    for path in TOOL.iterdir():
        if path.is_file():
            target = BASE / "sealed-tools" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, target)
            evidence[str(target.relative_to(BASE))] = jobs.sha(target)
    jobs.write(BASE / "closure.json", audit)
    evidence["closure.json"] = jobs.sha(BASE / "closure.json")
    jobs.write(BASE / "manifest.json", dict(source_sha256=build["source_sha256"],
        native_sha256=build["native_sha256"], evidence=evidence))
    print("PASS installed", build["native_sha256"], flush=True)
