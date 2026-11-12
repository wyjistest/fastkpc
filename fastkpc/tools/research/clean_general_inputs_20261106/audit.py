"""Seal the clean-build campaign without replacing or relabeling older evidence."""
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("clean_audit_build", TOOL / "build.py")
build_tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(build_tool)
ROOT, BASE, CHECKOUT = build_tool.ROOT, build_tool.BASE, build_tool.CHECKOUT
sha, write = build_tool.sha, build_tool.write

def read(path):
    return json.loads(Path(path).read_text())

def main():
    reference, qualification, timing, supplement, tag = sys.argv[1:6]
    directories = {name: BASE / value for name, value in
        (("references", reference), ("qualification", qualification), ("timing", timing), ("supplement", supplement))}
    out = BASE / tag; out.mkdir()
    build = read(BASE / "build/build.json")
    assert build["status"] == "PASS_CLEAN_FULL_BUILD" and build["compile_count"] == len(build["objects"]) == 44
    assert not build_tool.git("status", "--porcelain")
    assert build_tool.git("rev-parse", "HEAD") == build_tool.COMMIT
    for path, expected in build["source_sha256"].items():
        raw = subprocess.check_output(["git", "-C", str(CHECKOUT), "show", build_tool.COMMIT + ":" + path])
        assert hashlib.sha256(raw).hexdigest() == expected == sha(CHECKOUT/path), path
    for name, expected in {**build["objects"], **build["generated_sha256"]}.items():
        assert sha(CHECKOUT / "fastkpc/build" / name) == expected, name
    assert sha(build["native_path"]) == build["native_sha256"]
    assert sha(BASE / "build/build.log") == build["build_log_sha256"]
    prior_closure = BASE / "correctness_closure1"
    assert sha(prior_closure / "summary.json") == "6ecbacd1ed31a0384ec2a87bbe3124203f683b3fbb9a6c5693762f3cddca4e97"
    assert sha(prior_closure / "manifest.json") == "5bdbd0018da12d686e4272f7795871d952588e22dd7a6a249867d2c5e1084b66"
    prior_manifest = read(prior_closure / "manifest.json")
    for path, expected in prior_manifest["evidence"].items():
        assert sha(BASE / path) == expected, path
    summaries = {key: read(directory/"summary.json") for key, directory in directories.items()}
    tool_snapshot_bindings = {}
    for key, directory in directories.items():
        summary = summaries[key]
        assert summary.get("pass", summary.get("status") == "PASS"), key
        if key != "references": assert summary["native_sha256"] == build["native_sha256"]
        protocol = read(directory / "protocol.json")
        for path, expected in protocol["dependencies"].items():
            snapshot = directory / "tools" / Path(path).name
            if Path(path).parent == TOOL and snapshot.is_file():
                # Each completed stage retains the exact harness it ran. A
                # later harness fix must not relabel or rewrite that stage.
                assert sha(snapshot) == expected, path
                tool_snapshot_bindings[str(snapshot.relative_to(BASE))] = expected
            else:
                assert sha(path) == expected, path
        for path in directory.glob("*-process.json"):
            process = read(path)
            assert process["exit_code"] == 0, path
            assert not process.get("unexpected", process.get("external_selected_or_timing_gpu", [])), path
            if key == "timing": assert process["machine_isolated"], path
        for path in directory.glob("*-reference-bindings.json"):
            for dependency, expected in read(path)["dependencies"].items():
                assert sha(dependency) == expected, dependency
    cases = list(summaries["references"]["reports"])
    assert len(cases) == 7 and set(cases) == set(summaries["qualification"]["reports"]) == set(summaries["timing"]["reports"])
    timing_protocol = read(directories["timing"] / "protocol.json")
    order = [["A1", "off"], ["C1", "on"], ["C2", "on"], ["A2", "off"], ["A3", "off"], ["C3", "on"]]
    assert timing_protocol["order"] == order
    assert timing_protocol["non_regression_ratio"] == 1.05 and timing_protocol["speed_ratio"] == .95
    assert timing_protocol["speed_requires_slowest_on_below_fastest_off"] is True
    assert timing_protocol["producer_commit"] == build_tool.COMMIT
    assert timing_protocol["native_sha256"] == build["native_sha256"]
    assert Path(timing_protocol["previous"]) == directories["qualification"]
    assert len(list(directories["timing"].glob("*-process.json"))) == 42
    processes = []
    for case in timing_protocol["cases"]:
        for label, mode in order:
            process = read(directories["timing"] / f"{case}-{label}-process.json")
            command = process["command"]
            assert process["kind"] == "timing" and process["cpu_affinity"] == "0-19"
            assert command[:3] == ["taskset", "-c", "0-19"] and command[6] == mode
            assert command[-1] == "timing" and process["end_epoch"] > process["start_epoch"]
            assert not processes or process["start_epoch"] >= processes[-1]["end_epoch"]
            processes.append(process)
    assert len(processes) == 42 and len({p["selected_gpu"] for p in processes}) == 1
    rows = {}
    for case in cases:
        cpu = summaries["references"]["reports"][case]
        qual = summaries["qualification"]["reports"][case]
        timed = summaries["timing"]["reports"][case]
        runs = [*qual.values(), *timed["runs"].values()]
        assert set(qual) == {"off", "on"} and set(timed["runs"]) == {label for label, _ in order}
        for label, report in qual.items():
            assert report == read(directories["qualification"] / f"{case}-{label}/report.json")
        for label, mode in order:
            report = timed["runs"][label]
            assert report == read(directories["timing"] / f"{case}-{label}/report.json")
            assert report["mode"] == mode and report["phase"] == "timing"
            assert report["native_sha256"] == build["native_sha256"]
            assert all(math.isfinite(report[stage]) and report[stage] >= 0 for stage in
                       ("outer_sec", "skeleton_sec", "orientation_sec"))
        assert all(r["pass"] and all(r["gates"].values()) and r["numCol"] == cpu["numCol"] for r in runs)
        assert all(r["input_sha256"] == cpu["input_sha256"] and r["ci_count"] == qual["off"]["ci_count"] for r in runs)
        off = [timed["runs"][label]["outer_sec"] for label in ("A1", "A2", "A3")]
        on = [timed["runs"][label]["outer_sec"] for label in ("C1", "C2", "C3")]
        off_median, on_median = statistics.median(off), statistics.median(on)
        ratio = on_median / off_median
        speed = "PROVEN" if ratio <= .95 and max(on) < min(off) else ("REGRESSION" if ratio > 1.05 else "NOT_PROVEN")
        assert timed["off_outer_sec"] == off and timed["on_outer_sec"] == on
        assert (timed["off_median"], timed["on_median"], timed["ratio"]) == (off_median, on_median, ratio)
        assert timed["non_regression"] == (ratio <= 1.05) and timed["speed"] == speed
        stage_medians = {mode: {stage: statistics.median(
            r[stage] for r in timed["runs"].values() if r["mode"] == mode)
            for stage in ("skeleton_sec", "orientation_sec")} for mode in ("off", "on")}
        rows[case] = dict(n=cpu["n"], p=cpu["p"], numCol=cpu["numCol"], ci_count=qual["off"]["ci_count"],
            original_cpu_ci_count=cpu["ci_count"],
            orientation_ci_count=cpu["orientation_ci_count"], max_level=cpu["max_level"],
            max_ci_error=max(r["ci"]["max_abs_error"] for r in runs),
            max_pmax_error=max(r["pmax"]["max_abs_error"] for r in runs),
            off_outer_sec=off, on_outer_sec=on, off_outer_median=off_median, on_outer_median=on_median,
            stage_medians=stage_medians,
            outer_ratio=ratio, non_regression=ratio <= 1.05, speed=speed,
            slowest_on_below_fastest_off=max(on) < min(off),
            raw_pcalg_call_order_exact=all(r["raw_pcalg_call_order_exact"] for r in runs))
    receipt_dependencies = {}
    for path in ("fastkpc/R/full_cuda_ci_gate.R", "fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
                 "fastkpc/tools/research/real_data_parity_20261102/run.R",
                 "fastkpc/tools/research/numcol_20261106/common.R"):
        raw = subprocess.check_output(["git", "-C", str(CHECKOUT), "show", build_tool.COMMIT + ":" + path])
        receipt_dependencies[path] = sha(CHECKOUT / path)
        assert hashlib.sha256(raw).hexdigest() == receipt_dependencies[path], path
    command = ["Rscript", "--vanilla", str(TOOL / "check_final_receipts.R"),
               str(directories["references"]), str(directories["qualification"]),
               str(directories["timing"]), str(out / "receipt_audit.json")]
    with (out / "receipt_audit.log").open("x") as log:
        subprocess.run(command, cwd=CHECKOUT, stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt_audit = read(out / "receipt_audit.json")
    assert receipt_audit["pass"] and receipt_audit["run_count"] == 56 and not receipt_audit["native_loaded"]
    legacy = "kpcalg/R/dcovgamma.R"
    assert sha(ROOT/legacy) == sha(CHECKOUT/legacy) == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f"
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "kpcalg"], cwd=ROOT)
    installed = ROOT / "fastkpc/build/fastkpc_cuda.so"
    assert sha(installed) == "2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301"
    assert sha(ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106/closure.json") == \
        "f5969fc61d7d86fdbe5d3a2f6fa5c45031151449822cc4c260f9aa2003864870"
    supplement_summary = summaries["supplement"]
    assert supplement_summary["canonical"]["ci_count"] == 240498 and all(supplement_summary["canonical"]["gates"].values())
    assert supplement_summary["method_cases"] == 28 and supplement_summary["method_ci_count"] == 792
    assert supplement_summary["rank_one"]["pass"] and all(supplement_summary["rank_one"]["gates"].values())
    rejected = {
        "supplement1": "missing prepared_s shard in clean checkout; no numerical result from failed job",
        "supplement2": "missing historical ABI contract manifests; failed run not accepted",
        "supplement3": "rank-one harness omitted explicit full_cuda route; authority gate rejected legacy execution",
    }
    summary = dict(status="PASS_CLEAN_ORIGINAL_REFERENCE_QUALIFICATION", audited_utc=datetime.now(timezone.utc).isoformat(),
        producer_commit=build_tool.COMMIT,
        native_sha256=build["native_sha256"], installed_native_sha256=sha(installed), main_installation_replaced=False,
        legacy_r_sha256=sha(ROOT/legacy), source_count=len(build["source_sha256"]), object_count=len(build["objects"]),
        input_count=7, qualification_gpu_runs=14, timing_gpu_runs=42,
        fresh_original_cpu_ci_count=sum(row["original_cpu_ci_count"] for row in rows.values()),
        gpu_ci_comparisons=8*sum(row["ci_count"] for row in rows.values()),
        orientation_ci_comparisons=8*sum(row["orientation_ci_count"] for row in rows.values()),
        max_ci_error=max(row["max_ci_error"] for row in rows.values()),
        max_pmax_error=max(row["max_pmax_error"] for row in rows.values()), rows=rows,
        all_inputs_non_regression=all(row["non_regression"] for row in rows.values()),
        speed_proven_inputs=[case for case, row in rows.items() if row["speed"] == "PROVEN"],
        timing_started_utc=datetime.fromtimestamp(processes[0]["start_epoch"], timezone.utc).isoformat(),
        timing_completed_utc=datetime.fromtimestamp(processes[-1]["end_epoch"], timezone.utc).isoformat(),
        timing_campaign_wall_sec=processes[-1]["end_epoch"] - processes[0]["start_epoch"],
        timing_machine_isolated_runs=len(processes), performance_state="COMPLETED_PER_INPUT_DEVELOPMENT_TIMING",
        raw_receipt_audit=dict(path=str((out / "receipt_audit.json").relative_to(BASE)),
            sha256=sha(out / "receipt_audit.json"), run_count=receipt_audit["run_count"], native_loaded=False),
        canonical_ci_count=supplement_summary["canonical"]["ci_count"],
        canonical_gates=supplement_summary["canonical"]["gates"],
        method_ci_count=supplement_summary["method_ci_count"], method_max_error=supplement_summary["method_max_error"],
        rank_one_original_reference_supported=False, rank_one_isolated_fixture_pass=supplement_summary["rank_one"]["pass"],
        rejected_harness_attempts=rejected,
        phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False,
        evidence_class="clean producer/full build, fresh unmodified original-R correctness and per-input development timing; not promotion")
    assert summary["gpu_ci_comparisons"] == receipt_audit["ci_comparisons"]
    assert summary["orientation_ci_comparisons"] == receipt_audit["orientation_ci_comparisons"]
    write(out/"summary.json", summary)
    tools_dir = out / "tools"; tools_dir.mkdir()
    for name in ("build.py", "audit.py", "check_final_receipts.R"):
        shutil.copyfile(TOOL / name, tools_dir / name)
    evidence = {}
    for directory in [BASE/"build", BASE/"inputs", *directories.values(), prior_closure, out,
                      *[BASE/name for name in rejected]]:
        for path in directory.rglob("*"):
            if path.is_file(): evidence[str(path.relative_to(BASE))] = sha(path)
    evidence[str((out/"summary.json").relative_to(BASE))] = sha(out/"summary.json")
    write(out/"manifest.json", dict(producer_commit=build_tool.COMMIT, native_sha256=build["native_sha256"],
        source_sha256=build["source_sha256"], objects=build["objects"], evidence=evidence,
        generated_sha256=build["generated_sha256"], raw_receipt_dependency_sha256=receipt_dependencies,
        tool_snapshot_bindings=tool_snapshot_bindings,
        audit_tool_sha256=sha(__file__)))
    print(json.dumps({key: summary[key] for key in ("status", "native_sha256", "input_count",
        "gpu_ci_comparisons", "orientation_ci_comparisons", "max_ci_error", "all_inputs_non_regression",
        "speed_proven_inputs", "timing_machine_isolated_runs")}, indent=2))

if __name__ == "__main__":
    main()
