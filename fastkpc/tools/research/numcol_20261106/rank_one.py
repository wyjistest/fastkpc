"""Qualify the legacy R scalar-diagonal correction and the public rank-one route."""
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
parent = BASE / "rank-one-parent/dcovgamma.R"
assert jobs.sha(parent) == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f"
out = BASE / sys.argv[2]
out.mkdir()
for key in list(os.environ):
    if key.startswith("FASTKPC_"): del os.environ[key]
env = jobs.environment()
native = Path(build["native_path"])
deps = {str(p): jobs.sha(p) for p in [native, parent, TOOL / "common.R", TOOL / "rank_one.R",
    TOOL / "jobs.py", Path(__file__), TOOL.parent / "real_data_parity_20261102/run.R"]}
for directory in (ROOT / "fastkpc/R", ROOT / "kpcalg/R"):
    deps.update({str(p): jobs.sha(p) for p in directory.glob("*.R")})
def frozen():
    jobs.check_map(build["source_sha256"])
    jobs.check_map(deps)
jobs.write(out / "protocol.json", dict(dependencies=deps, native_sha256=build["native_sha256"],
    original_source_sha256=jobs.sha(parent), current_source_sha256=jobs.sha(ROOT / "kpcalg/R/dcovgamma.R"),
    numerical_tolerance=1e-10, preserved_ranks=[2, 7, 35, 50, 64, 65, 139], pairs=3,
    gates="21 original/patched R bitwise comparisons; rank1 CPU full eig; public WAN-PDAG CI/pMax/order/graph/RNG/resources",
    evidence_class="boundary correction; no change to historical receipts"))
jobs.job(out, "run", ["Rscript", "--vanilla", str(TOOL / "rank_one.R"), str(native),
    str(parent), str(out / "run")], env, frozen)
