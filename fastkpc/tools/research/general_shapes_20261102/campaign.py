"""Freeze a multi-shape protocol before the first GPU run; retain all failures."""
import importlib.util
import os
from pathlib import Path
import statistics
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shape_jobs", TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102"
build = jobs.read(BASE / sys.argv[1] / "build.json")
out = BASE / sys.argv[2]
out.mkdir()
cases = sys.argv[3:] or ["real179", "real243", "real300", "public500", "dense500",
                         "boundary75", "boundary76", "large1025", "blocks256",
                         "dense75", "dense76", "dense1025"]
for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
native = Path(build["native_path"])
dependencies = {str(p.relative_to(ROOT)): jobs.sha(p) for p in [native, TOOL / "run.R", Path(__file__)]}
dependencies.update({str(p.relative_to(ROOT)): jobs.sha(p) for p in (ROOT / "fastkpc/R").glob("*.R")})
input_paths = [
    "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
    "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    "fastkpc/tests/fixtures/full_cuda_ci_development_500x50_v2.rds",
    "fastkpc/tests/test_compatible_cuda_wanpdag_authority.R",
    "fastkpc/tools/research/real_data_parity_20261102/run.R",
    "fastkpc/tools/research/stage_b_work_pair_runtime_20261028/support.py"]
input_paths += [f"fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102/attempt4/{case}/cpu-schedule-replay.rds"
                for case in ("20221022181603", "20220418113957", "20240315113222")]
dependencies.update({p: jobs.sha(ROOT / p) for p in input_paths})

def frozen():
    jobs.check_map(build["source_sha256"])
    jobs.check_map(dependencies)

sequence = [("a1", "off"), ("c1", "on"), ("c2", "on"), ("a2", "off"), ("a3", "off"), ("c3", "on")]
orientation_cases = ["real179", "real243", "real300", "public500"]
for case in cases:
    folder = "references2" if case in ("dense75", "dense76", "dense1025") else "references"
    ref = BASE / folder / case / "receipt.rds"
    if ref.exists(): dependencies[str(ref)] = jobs.sha(ref)
jobs.write(out / "protocol.json", dict(cases=cases, sequence=sequence,
    native_sha256=build["native_sha256"], dependencies=dependencies,
    ci_tolerance=1e-10, pmax_tolerance=1e-10, max_conditioning_size="Inf", numCol=35, alpha=.1,
    performance="per-input fresh-process skeleton medians; no pooled ratio or canonical promotion",
    non_regression_median_ratio=1.05, speed_median_ratio=.95,
    correctness="all CI/pMax, graph/trace/sepsets/counts, RNG, GPU authority, resources on every run",
    orientation_cases=orientation_cases, orientation_run="c1 only; unchanged CPU kpcalg authority"))
reports = []
for case in cases:
    reference = "-"
    if case not in ("real179", "real243", "real300", "public500"):
        folder = "references2" if case in ("dense75", "dense76", "dense1025") else "references"
        cpu_out = BASE / folder / case
        assert jobs.read(cpu_out / "report.json")["pass"]
        reference = str(cpu_out / "receipt.rds")
        dependencies[reference] = jobs.sha(reference)
    runs = []
    for tag, mode in sequence:
        name = case + "-" + tag
        print("START", name, flush=True)
        run_env = env.copy()
        run_env["FASTKPC_SHAPE_ORIENTATION"] = "1" if tag == "c1" and case in orientation_cases else "0"
        jobs.job(out, name, ["Rscript", "--vanilla", str(TOOL / "run.R"),
            case, mode, str(native), str(out / name), reference], run_env, frozen)
        report = jobs.read(out / name / "report.json")
        assert report["pass"]
        runs.append(report)
        print("RESULT", name, report["elapsed_sec"], "CI", report["ci_count"], flush=True)
    medians = {mode: statistics.median(r["elapsed_sec"] for r in runs if r["mode"] == mode)
               for mode in ("off", "on")}
    ratio = medians["on"] / medians["off"]
    summary = dict(case=case, medians=medians, ratio=ratio,
        non_regression=ratio <= 1.05, speed=ratio <= .95, runs=runs)
    reports.append(summary)
    jobs.write(out / (case + "-summary.json"), summary)
jobs.write(out / "summary.json", dict(status="PASS_NUMERICAL", cases=reports))
print("COMPLETE", [(r["case"], r["medians"], r["ratio"]) for r in reports], flush=True)
