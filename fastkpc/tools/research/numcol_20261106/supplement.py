"""New ranks in other methods, cache/errors, sanitizer and old-rank regression."""
import importlib.util
import os
from pathlib import Path
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rank_jobs", TOOL / "jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/numcol_20261106"
build = jobs.read(BASE / sys.argv[1] / "build.json")
out = BASE / sys.argv[2]
out.mkdir()
for key in list(os.environ):
    if key.startswith("FASTKPC_"): del os.environ[key]
env = jobs.environment()
native = Path(build["native_path"])
parent = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102/candidate4/fastkpc_cuda.so"
canonical = TOOL.parent / "general_shapes_20261102/canonical.R"
baseline = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102/supplement6/canonical/receipt.rds"
tests = ["numcol", "one_call", "phase35_legacy_eig_batch", "structured_prefill", "phase10_hardening"]
deps = {str(p): jobs.sha(p) for p in [*TOOL.glob("*.R"), Path(__file__), native, parent, canonical, baseline]}
deps.update({str(p): jobs.sha(p) for p in (ROOT / "fastkpc/R").glob("*.R")})
deps.update({str(p): jobs.sha(p) for p in (ROOT / "kpcalg/R").glob("*.R")})
deps[str(TOOL / "jobs.py")] = jobs.sha(TOOL / "jobs.py")
for name in tests:
    p = ROOT / ("fastkpc/tests/test_full_cuda_ci_" + name + ".R")
    deps[str(p)] = jobs.sha(p)

def frozen():
    jobs.check_map(build["source_sha256"])
    jobs.check_map(deps)

jobs.write(out / "protocol.json", dict(native_sha256=build["native_sha256"], dependencies=deps,
    tests=tests, methods=["dcc.perm", "hsic.gamma", "hsic.perm"], ranks=[1, 7, 35, 50, 64, 65, 102, 139],
    method_cpu_authority="kpcalg/mgcv/kernlab/R permutations; all CIs and RNG",
    method_tolerance="gamma 1e-10; permutation exact", hsic_extra="k65/102 sig1e-5, actual CPU rank >64",
    canonical="old k35 bits + CPU oracle + final PDAG",
    sanitizer="memcheck/leak-check full, error exit 99; cross-rank cache and invalid parameters"))
for name in tests:
    jobs.job(out, name, ["Rscript", "--vanilla", str(TOOL / "run_test.R"), str(native),
        str(ROOT / ("fastkpc/tests/test_full_cuda_ci_" + name + ".R"))], env, frozen)
for name, binary, ranks in (("methods-parent", parent, "35"), ("methods", native, "all")):
    jobs.job(out, name, ["Rscript", "--vanilla", str(TOOL / "methods.R"), str(binary),
        str(out / name), ranks], env, frozen)
jobs.job(out, "methods-bits", ["Rscript", "--vanilla", str(TOOL / "compare_methods.R"),
    str(out / "methods-parent"), str(out / "methods")], env, frozen)
jobs.job(out, "memcheck", ["/usr/local/cuda/bin/compute-sanitizer", "--tool", "memcheck",
    "--leak-check", "full", "--error-exitcode", "99", "Rscript", "--vanilla",
    str(TOOL / "run_test.R"), str(native), str(ROOT / "fastkpc/tests/test_full_cuda_ci_numcol.R")], env, frozen)
jobs.job(out, "canonical", ["Rscript", "--vanilla", str(canonical), str(native),
    str(out / "canonical"), str(baseline)], env, frozen)
jobs.write(out / "closure.json", dict(status="PASS", native_sha256=build["native_sha256"], dependencies=deps,
    tests=tests + ["methods-parent", "methods", "methods-bits", "memcheck", "canonical"]))
print("PASS supplemental qualification", flush=True)
