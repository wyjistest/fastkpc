"""Recompute qualification, then optionally install the tested local native."""
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shape_jobs", TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102"
build_dir, campaign, supplemental = [BASE / arg for arg in sys.argv[1:4]]
action = sys.argv[4]
assert action in ("audit", "install")
build = jobs.read(build_dir / "build.json")
native = Path(build["native_path"])
assert build["status"] == "PASS" and jobs.sha(native) == build["native_sha256"]
jobs.check_map(build["source_sha256"])
for name, digest in build["objects"].items():
    assert jobs.sha(build_dir / "objects" / name) == digest
protocol = jobs.read(campaign / "protocol.json")
summary = jobs.read(campaign / "summary.json")
assert protocol["native_sha256"] == build["native_sha256"]
assert summary["status"] == "PASS_NUMERICAL" and len(summary["cases"]) == 12
jobs.check_map(protocol["dependencies"])
rows = []
ci_total = 0
for case in summary["cases"]:
    reports = []
    for tag, mode in protocol["sequence"]:
        name = case["case"] + "-" + tag
        report = jobs.read(campaign / name / "report.json")
        process = jobs.read(campaign / (name + "-process.json"))
        assert process["isolated"] and process["exit_code"] == 0 and not process["unexpected"]
        assert report["pass"] and all(report["gates"].values()) and report["mode"] == mode
        assert report["native_sha256"] == build["native_sha256"]
        expected_orientation = tag == "c1" and case["case"] in protocol["orientation_cases"]
        assert report["orientation_checked"] == expected_orientation
        assert report["ci"]["outside_tolerance"] == report["pmax"]["outside_tolerance"] == 0
        assert report["ci"]["max_abs_error"] <= 1e-10 and report["pmax"]["max_abs_error"] <= 1e-10
        ci_total += report["ci_count"]
        reports.append(report)
    medians = {mode: statistics.median(r["elapsed_sec"] for r in reports if r["mode"] == mode)
               for mode in ("off", "on")}
    ratio = medians["on"] / medians["off"]
    assert medians == case["medians"] and ratio == case["ratio"]
    assert case["non_regression"] == (ratio <= protocol["non_regression_median_ratio"])
    assert case["speed"] == (ratio <= protocol["speed_median_ratio"])
    rows.append(dict(case=case["case"], n=reports[0]["n"], p=reports[0]["p"],
        ci_count=reports[0]["ci_count"], medians=medians, ratio=ratio,
        non_regression=case["non_regression"], speed=case["speed"],
        maximum_error=max(max(r["ci"]["max_abs_error"], r["pmax"]["max_abs_error"]) for r in reports)))
supplement = jobs.read(supplemental / "closure.json")
assert supplement["status"] == "PASS" and supplement["native_sha256"] == build["native_sha256"]
jobs.check_map(supplement["dependencies"])
for name in supplement["tests"]:
    process = jobs.read(supplemental / (name + "-process.json"))
    assert process["exit_code"] == 0 and process["isolated"] and not process["unexpected"]
canonical = jobs.read(supplemental / "canonical/report.json")
slots = jobs.read(supplemental / "slots/report.json")
assert canonical["status"] == slots["status"] == "PASS" and all(canonical["gates"].values())
assert len(slots["cases"]) == 80 and slots["targets_compared"] == 7808 and slots["negative_cases"] == 640
if slots["native_sha256"] != build["native_sha256"]:
    bridge = jobs.read(supplemental / "slot-native-bridge.json")
    assert bridge["reference_native_sha256"] == slots["native_sha256"]
    assert bridge["current_native_sha256"] == build["native_sha256"]
    assert bridge["changed_sources"] == ["fastkpc/src/full_cuda_ci_one_call.cpp"]
    assert bridge["unchanged_objects"] == {p: h for p, h in build["objects"].items() if p != "full_cuda_ci_one_call.o"}
audit = dict(status="PASS_GENERAL_SHAPE_QUALIFICATION", native_sha256=build["native_sha256"],
    source_sha256=build["source_sha256"], campaign=str(campaign), supplemental=str(supplemental),
    cases=rows, repeated_ci_comparisons=ci_total, canonical_ci_count=canonical["ci_count"],
    canonical_elapsed_sec=canonical["elapsed_sec"], slot_targets_compared=slots["targets_compared"],
    slot_native_sha256=slots["native_sha256"],
    all_inputs_nonregression=all(r["non_regression"] for r in rows),
    defaults_changed=False, phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False,
    evidence_class="incremental source-closure multi-input development qualification; no universal speed claim")
if action == "audit":
    jobs.write(BASE / "audit.json", audit)
    print("PASS audit", rows, flush=True)
else:
    assert jobs.read(BASE / "audit.json") == audit
    assert not jobs.processes(), "active GPU task; do not replace build inputs"
    old_dir = ROOT / "fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/candidate"
    old = jobs.read(old_dir / "build.json")
    assert jobs.sha(old_dir / "fastkpc_cuda.so") == old["native_sha256"]
    dest = ROOT / "fastkpc/build"
    with (dest / "fastkpc_cuda.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert jobs.sha(dest / "fastkpc_cuda.so") == old["native_sha256"]
        for name, digest in old["objects"].items():
            assert jobs.sha(dest / name) == jobs.sha(old_dir / "objects" / name) == digest
        mapping = {name: build_dir / "objects" / name for name in build["objects"]}
        for name in ("recorded_dbdsqr.inc", "recorded_dbdsqr_fixed_vt.inc"):
            mapping[name] = build_dir / "source/fastkpc/build" / name
        mapping["fastkpc_cuda.so"] = native  # replace the runnable native last
        for name, source in mapping.items():
            pending = dest / (name + ".general-shapes.tmp")
            assert not pending.exists()
            shutil.copy2(source, pending)
            assert jobs.sha(pending) == jobs.sha(source)
            os.replace(pending, dest / name)
        jobs.write(BASE / "installation.json", dict(status="PASS_INSTALLED", epoch=time.time(),
            previous_native_sha256=old["native_sha256"], native_sha256=jobs.sha(dest / "fastkpc_cuda.so"),
            objects=build["objects"], defaults_changed=False))
    evidence = {}
    for directory in (campaign, supplemental):
        for path in directory.rglob("*"):
            if path.is_file(): evidence[str(path.relative_to(BASE))] = jobs.sha(path)
    for path in (BASE / "audit.json", BASE / "installation.json", build_dir / "build.json", build_dir / "protocol.json"):
        evidence[str(path.relative_to(BASE))] = jobs.sha(path)
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
