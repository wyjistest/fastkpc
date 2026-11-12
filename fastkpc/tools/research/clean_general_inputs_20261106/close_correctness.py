"""Seal correctness separately; never mark missing performance evidence as passed."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("correctness_audit", TOOL / "audit.py")
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
ROOT, BASE, CHECKOUT = audit.ROOT, audit.BASE, audit.CHECKOUT
sha, write, read = audit.sha, audit.write, audit.read

def main():
    reference, qualification, supplement, tag = sys.argv[1:5]
    directories = {"references": BASE/reference, "qualification": BASE/qualification, "supplement": BASE/supplement}
    out = BASE/tag; out.mkdir()
    build = read(BASE/"build/build.json")
    assert build["status"] == "PASS_CLEAN_FULL_BUILD" and build["compile_count"] == len(build["objects"]) == 44
    assert not audit.build_tool.git("status", "--porcelain")
    assert audit.build_tool.git("rev-parse", "HEAD") == audit.build_tool.COMMIT
    for path, expected in build["source_sha256"].items():
        blob = subprocess.check_output(["git", "-C", str(CHECKOUT), "show", audit.build_tool.COMMIT+":"+path])
        assert hashlib.sha256(blob).hexdigest() == expected == sha(CHECKOUT/path), path
    for path, expected in {**build["objects"], **build["generated_sha256"]}.items():
        assert sha(CHECKOUT/"fastkpc/build"/path) == expected, path
    assert sha(build["native_path"]) == build["native_sha256"]
    assert sha(BASE/"build/build.log") == build["build_log_sha256"]
    summaries = {key: read(directory/"summary.json") for key, directory in directories.items()}
    snapshots = {}
    shared_host_jobs = []
    for key, directory in directories.items():
        summary = summaries[key]
        assert summary.get("pass", summary.get("status") == "PASS"), key
        if key != "references": assert summary["native_sha256"] == build["native_sha256"]
        for path, expected in read(directory/"protocol.json")["dependencies"].items():
            snapshot = directory/"tools"/Path(path).name
            actual = snapshot if Path(path).parent == TOOL and snapshot.is_file() else Path(path)
            assert sha(actual) == expected, path
            if actual == snapshot: snapshots[str(snapshot.relative_to(BASE))] = expected
        for path in directory.glob("*-reference-bindings.json"):
            for dependency, expected in read(path)["dependencies"].items():
                assert sha(dependency) == expected, dependency
        for path in directory.glob("*-process.json"):
            process = read(path)
            assert process["exit_code"] == 0
            assert not process.get("unexpected", process.get("external_selected_or_timing_gpu", []))
            if not process["machine_isolated"]: shared_host_jobs.append(str(path.relative_to(BASE)))
    cases = summaries["references"]["reports"]
    assert len(cases) == 7 and set(cases) == set(summaries["qualification"]["reports"])
    rows = {}
    for case, cpu in cases.items():
        pair = summaries["qualification"]["reports"][case]
        assert set(pair) == {"off", "on"}
        assert all(r["pass"] and all(r["gates"].values()) and r["numCol"] == cpu["numCol"] and
            r["input_sha256"] == cpu["input_sha256"] for r in pair.values())
        rows[case] = dict(n=cpu["n"], p=cpu["p"], numCol=cpu["numCol"],
            original_cpu_ci_count=cpu["ci_count"], gpu_ci_count=pair["off"]["ci_count"],
            orientation_ci_count=cpu["orientation_ci_count"], max_level=cpu["max_level"],
            max_ci_error=max(r["ci"]["max_abs_error"] for r in pair.values()),
            max_pmax_error=max(r["pmax"]["max_abs_error"] for r in pair.values()))
    command = ["Rscript", "--vanilla", str(TOOL/"check_receipts.R"), str(directories["references"]),
        str(directories["qualification"]), str(out/"receipt_audit.json")]
    with (out/"receipt_audit.log").open("x") as log:
        subprocess.run(command, cwd=CHECKOUT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(out/"receipt_audit.json")["pass"]
    assert not subprocess.check_output(["git", "diff", "HEAD", "--", "kpcalg"], cwd=ROOT)
    legacy = "kpcalg/R/dcovgamma.R"
    assert sha(ROOT/legacy) == sha(CHECKOUT/legacy) == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f"
    installed = sha(ROOT/"fastkpc/build/fastkpc_cuda.so")
    assert installed == "2727ef51899688b54da697d0008f99a4908922dbcacec1c30ed58cef09198301"
    assert sha(ROOT/"fastkpc/artifacts/full_cuda_ci/numcol_20261106/closure.json") == \
        "f5969fc61d7d86fdbe5d3a2f6fa5c45031151449822cc4c260f9aa2003864870"
    supplementary = summaries["supplement"]
    assert supplementary["canonical"]["ci_count"] == 240498 and all(supplementary["canonical"]["gates"].values())
    assert supplementary["method_cases"] == 28 and supplementary["method_ci_count"] == 792
    assert supplementary["rank_one"]["pass"] and all(supplementary["rank_one"]["gates"].values())
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader"], text=True).strip()
    preflight = dict(observed_utc=datetime.now(timezone.utc).isoformat(), active_compute_processes=active,
        both_gpus_free=not active, required_for_timing="both GPUs free of external compute")
    write(out/"performance_preflight.json", preflight)
    summary = dict(status="PASS_CLEAN_ORIGINAL_REFERENCE_CORRECTNESS", producer_commit=audit.build_tool.COMMIT,
        native_sha256=build["native_sha256"], installed_native_sha256=installed, main_installation_replaced=False,
        legacy_r_sha256=sha(ROOT/legacy), source_count=len(build["source_sha256"]), object_count=44,
        input_count=7, gpu_correctness_runs=14, original_cpu_ci_count=sum(r["original_cpu_ci_count"] for r in rows.values()),
        gpu_ci_comparisons=2*sum(r["gpu_ci_count"] for r in rows.values()),
        orientation_ci_comparisons=2*sum(r["orientation_ci_count"] for r in rows.values()),
        max_ci_error=max(r["max_ci_error"] for r in rows.values()),
        max_pmax_error=max(r["max_pmax_error"] for r in rows.values()), cases=rows,
        canonical=supplementary["canonical"], method_ci_count=supplementary["method_ci_count"],
        method_max_error=supplementary["method_max_error"], rank_one=supplementary["rank_one"],
        shared_host_correctness_jobs=shared_host_jobs,
        performance_state="NOT_RUN_IN_THIS_CORRECTNESS_CLOSURE", performance_preflight=preflight,
        phase10="ACTIVE", candidate="NOT_FROZEN",
        holdout="SEALED", recommended=False, evidence_class="clean full build and fresh original-R correctness; no repeated performance claim")
    write(out/"summary.json", summary)
    tools_dir = out/"tools"; tools_dir.mkdir()
    for name in ("build.py", "audit.py", "close_correctness.py", "check_receipts.R"):
        shutil.copyfile(TOOL/name, tools_dir/name)
    evidence = {}
    for directory in [BASE/"build", BASE/"inputs", *directories.values(), out,
        BASE/"supplement1", BASE/"supplement2", BASE/"supplement3"]:
        for path in directory.rglob("*"):
            if path.is_file(): evidence[str(path.relative_to(BASE))] = sha(path)
    write(out/"manifest.json", dict(producer_commit=audit.build_tool.COMMIT, native_sha256=build["native_sha256"],
        source_sha256=build["source_sha256"], objects=build["objects"], evidence=evidence, tool_snapshot_bindings=snapshots))
    print(json.dumps({k: summary[k] for k in ("status", "native_sha256", "input_count", "gpu_ci_comparisons", "max_ci_error", "shared_host_correctness_jobs")}, indent=2))

if __name__ == "__main__":
    main()
