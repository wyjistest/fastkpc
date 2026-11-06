"""Fresh CPU references and GPU qualification; timings are observations only."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rank_jobs", TOOL / "jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106"
CASES = [("real179", "default"), ("real243", "default"), ("real300", "default"),
         ("public500", "default"), ("dense500", "default"), ("large1025", "default")]
CASES += [("dense140", str(k)) for k in (1, 2, 7, 35, 50, 64, 65, 139)]
CASES += [("dense133", "64"), ("dense134", "64")]

def main():
    build = jobs.read(BASE / sys.argv[1] / "build.json")
    out = BASE / sys.argv[2]
    out.mkdir()
    reuse = BASE / sys.argv[3] if len(sys.argv) > 3 else None
    native = Path(build["native_path"])
    # Candidate3 fixes only host early-stop accounting in the CPU reference
    # scheduler. Its independent CPU dCov and mgcv numerical code is unchanged.
    cpu_native = native
    for key in list(os.environ):
        if key.startswith("FASTKPC_"): del os.environ[key]
    env = jobs.environment()
    deps = {str(p): jobs.sha(p) for p in [native, TOOL / "run.R", TOOL / "common.R", TOOL / "jobs.py", Path(__file__)]}
    deps.update({str(p): jobs.sha(p) for p in (ROOT / "fastkpc/R").glob("*.R")})
    deps.update({str(p): jobs.sha(p) for p in (ROOT / "kpcalg/R").glob("*.R")})
    for p in ["causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
              "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
              "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData"]:
        deps[p] = jobs.sha(ROOT / p)
    def frozen():
        jobs.check_map(build["source_sha256"])
        jobs.check_map(deps)
    jobs.write(out / "protocol.json", dict(cases=CASES, dependencies=deps,
        native_sha256=build["native_sha256"], parent_native_sha256=build["parent_native_sha256"],
        cpu_native_sha256=jobs.sha(cpu_native),
        cpu_authority="unchanged R mgcv and native CPU dCov Spectra; corrected host empty-graph stop",
        sequence=["cpu", "off", "on"], uncached_cases=[["dense140", "1"], ["dense140", "65"]],
        ci_tolerance=1e-10, pmax_tolerance=1e-10, max_conditioning_size="Inf",
        default="floor(n/10); option omitted", alpha=.1, index=1,
        gates="all CI/pMax, order/deletion/graph/sepsets/counts, RNG, GPU authority, resources, rank dispatch",
        performance="single observations; no speed or promotion claim",
        isolation="selected GPU exclusive; other GPU use recorded without aborting correctness checks",
        reuse_from=str(reuse) if reuse else None))
    reports, reused = [], []
    if reuse:
        previous = jobs.read(reuse / "protocol.json")
        assert previous["native_sha256"] == build["native_sha256"]
        for p, digest in previous["dependencies"].items():
            if Path(p) != Path(__file__): assert jobs.sha(ROOT / p) == digest, p
    for case, rank in CASES:
        stem = case + "-k" + rank
        cpu_out = out / (stem + "-cpu")
        for mode in ("cpu", "off", "on") + (("uncached",) if (case, rank) in (("dense140", "1"), ("dense140", "65")) else ()):
            name = stem + "-" + mode
            print("START", name, flush=True)
            prior_report = reuse / name / "report.json" if reuse else None
            if prior_report and prior_report.exists() and jobs.read(prior_report)["pass"]:
                prior_process = jobs.read(reuse / (name + "-process.json"))
                assert prior_process["exit_code"] == 0 and prior_process["isolated"]
                frozen()
                shutil.copytree(reuse / name, out / name)
                for suffix in (".log", "-process.json", "-gpu.csv"):
                    shutil.copyfile(reuse / (name + suffix), out / (name + suffix))
                reused.append(dict(job=name, original=str(reuse), rerun=False,
                    report_sha256=jobs.sha(prior_report), process_sha256=jobs.sha(reuse / (name + "-process.json"))))
                print("REUSE authenticated original", name, flush=True)
            else:
                jobs.job(out, name, ["Rscript", "--vanilla", str(TOOL / "run.R"), case, rank, mode,
                    str(cpu_native if mode == "cpu" else native), str(out / name),
                    "-" if mode == "cpu" else str(cpu_out / "receipt.rds")], env, frozen)
            report = jobs.read(out / name / "report.json")
            assert report["pass"]
            reports.append(report)
            print("RESULT", name, report["elapsed_sec"], "CI", report["ci_count"], flush=True)
    jobs.write(out / "summary.json", dict(status="PASS", reports=reports, reused_successes=reused))
    print("PASS campaign", len(reports), flush=True)

if __name__ == "__main__": main()
