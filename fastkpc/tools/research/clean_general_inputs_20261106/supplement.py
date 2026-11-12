"""Validate the full-build artifact with existing contracts and isolated rank-one fixture."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result
build_tool = module("clean_build_extra", TOOL / "build.py")
ROOT, BASE, CHECKOUT = build_tool.ROOT, build_tool.BASE, build_tool.CHECKOUT
driver = module("clean_campaign_extra", TOOL / "campaign.py")
jobs = module("clean_original_jobs", CHECKOUT / "fastkpc/tools/research/numcol_20261106/jobs.py")
sha, write, read = build_tool.sha, build_tool.write, driver.read

def main():
    out = BASE / sys.argv[1]; out.mkdir()
    build = read(BASE / "build/build.json")
    assert build["status"] == "PASS_CLEAN_FULL_BUILD"
    native = Path(build["native_path"])
    tests = ["numcol", "one_call", "phase35_legacy_eig_batch", "structured_prefill", "phase10_hardening"]
    baseline = ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106/supplement5/canonical/receipt.rds"
    patch = ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106/legacy-r-source/dcovgamma.R"
    # Only ignored input links are added; the clean producer's tracked tree stays intact.
    links = ["fastkpc/artifacts/kpc_tprs_real_zhu", "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2",
        "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1"]
    contract_fixtures = ["oracle_351x48_v1", "workload_census_351x48_v1",
        "prepared_s_contract_v1", "fixed_sp_cuda_oracle_sp_v1", "fixed_sp_cuda_full_shadow_v1"]
    links += ["fastkpc/artifacts/full_cuda_ci/" + name for name in contract_fixtures if name != "prepared_s_contract_v1"]
    for relative in links:
        target = CHECKOUT / relative; original = ROOT / relative
        assert original.is_dir()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.symlink_to(original, target_is_directory=True)
        assert target.resolve() == original.resolve()
    dependencies = {str(CHECKOUT/p): h for p, h in build["source_sha256"].items()}
    dependencies.update({str(p): sha(p) for p in (native, baseline, patch, Path(__file__),
        TOOL / "build.py", TOOL / "campaign.py", TOOL / "rank_one_fixture.R")})
    for relative in ("fastkpc/tools/research/numcol_20261106", "fastkpc/tools/research/general_shapes_20261102",
                     "fastkpc/tools/research/real_data_parity_20261102", "fastkpc/tools/research/stage_b_work_pair_runtime_20261028"):
        dependencies.update({str(p): sha(p) for p in (CHECKOUT / relative).iterdir() if p.is_file()})
    for p in (CHECKOUT / "fastkpc/tests").glob("*.R"):
        dependencies[str(p)] = sha(p)
    for p in (ROOT / "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2").iterdir():
        if p.is_file(): dependencies[str(p)] = sha(p)
    data = ROOT / "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds"
    dependencies[str(data)] = sha(data)
    shard = ROOT / "fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/shards/shard_34.rds"
    dependencies[str(shard)] = sha(shard)
    for name in contract_fixtures:
        path = ROOT / "fastkpc/artifacts/full_cuda_ci" / name / "manifest.json"
        dependencies[str(path)] = sha(path)
    trace = ROOT / "fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/logical_ci_trace.rds"
    dependencies[str(trace)] = sha(trace)
    for path in (CHECKOUT / "fastkpc/inst/contracts").rglob("*.json"):
        dependencies[str(path)] = sha(path)
    pinned = CHECKOUT / "fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R"
    dependencies[str(pinned)] = sha(pinned)
    def frozen():
        assert not build_tool.git("status", "--porcelain")
        for p, h in dependencies.items(): assert sha(p) == h, p
    env = driver.environment()
    env.update(FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES="11811160064",
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS="1",
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES="8")
    write(out / "protocol.json", dict(producer_commit=build_tool.COMMIT, native_sha256=build["native_sha256"],
        dependencies=dependencies, tests=tests, methods="28 cases across dcc.perm/hsic.gamma/hsic.perm; original R authority",
        canonical="single full regression; prior GPU bits and archived CPU 1e-10 oracle; original R final orientation",
        rank_one="archived patch loaded in an isolated environment only; independent full eig GPU comparison",
        environment={k: v for k, v in env.items() if k.startswith("FASTKPC_") or k in
            ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        main_native_unchanged=True, original_kpcalg_files_unchanged=True))
    os.chdir(CHECKOUT); os.sched_setaffinity(0, range(20))
    runner = CHECKOUT / "fastkpc/tools/research/numcol_20261106/run_test.R"
    for name in tests:
        jobs.job(out, name, ["Rscript", "--vanilla", str(runner), str(native),
            str(CHECKOUT / ("fastkpc/tests/test_full_cuda_ci_" + name + ".R"))], env, frozen)
        log = (out / (name + ".log")).read_text()
        assert "PASS" in log and "SKIP" not in log, name
    jobs.job(out, "methods", ["Rscript", "--vanilla",
        str(CHECKOUT / "fastkpc/tools/research/numcol_20261106/methods.R"), str(native), str(out / "methods"), "all"], env, frozen)
    methods = read(out / "methods/report.json")
    assert methods["status"] == "PASS" and len(methods["cases"]) == 28
    jobs.job(out, "rank-one", ["Rscript", "--vanilla", str(TOOL / "rank_one_fixture.R"), str(patch), str(out / "rank-one")], env, frozen)
    assert read(out / "rank-one/report.json")["pass"]
    jobs.job(out, "memcheck", ["/usr/local/cuda/bin/compute-sanitizer", "--tool", "memcheck", "--leak-check", "full",
        "--error-exitcode", "99", "Rscript", "--vanilla", str(runner), str(native),
        str(CHECKOUT / "fastkpc/tests/test_full_cuda_ci_numcol.R")], env, frozen)
    assert "ERROR SUMMARY: 0 errors" in (out / "memcheck.log").read_text()
    jobs.job(out, "canonical", ["Rscript", "--vanilla",
        str(CHECKOUT / "fastkpc/tools/research/general_shapes_20261102/canonical.R"),
        str(native), str(out / "canonical"), str(baseline)], env, frozen)
    canonical = read(out / "canonical/report.json")
    assert canonical["status"] == "PASS" and canonical["ci_count"] == 240498 and all(canonical["gates"].values())
    write(out / "summary.json", dict(status="PASS", native_sha256=build["native_sha256"],
        tests=tests, method_cases=28, method_ci_count=sum(r["ci_count"] for r in methods["cases"].values()),
        method_max_error=max(r["ci_error"] for r in methods["cases"].values()), canonical=canonical,
        rank_one=read(out / "rank-one/report.json"), evidence_class="clean full-build correctness; no new canonical median"))
    print("PASS supplemental clean-build qualification", flush=True)

if __name__ == "__main__":
    main()
