"""Qualify the actual public WAN-PDAG default, not an explicit-rank shortcut."""
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
reference = BASE / sys.argv[2] / "public500-kdefault-cpu/receipt.rds"
out = BASE / sys.argv[3]
out.mkdir()
for key in list(os.environ):
    if key.startswith("FASTKPC_"): del os.environ[key]
env = jobs.environment()
native = Path(build["native_path"])
deps = {str(p): jobs.sha(p) for p in [native, reference, TOOL / "common.R", TOOL / "public_default.R", TOOL / "jobs.py", Path(__file__)]}
deps.update({str(p): jobs.sha(p) for p in (ROOT / "fastkpc/R").glob("*.R")})
deps.update({str(p): jobs.sha(p) for p in (ROOT / "kpcalg/R").glob("*.R")})
def frozen():
    jobs.check_map(build["source_sha256"])
    jobs.check_map(deps)
jobs.write(out / "protocol.json", dict(dependencies=deps, native_sha256=build["native_sha256"],
    options="ci_method=dcc.gamma, trace_level=logical; numCol omitted", expected_numCol=50,
    method="public 500x50 full WAN-PDAG, all CI/pMax, task/graph/orientation/RNG/resource gates"))
jobs.job(out, "run", ["Rscript", "--vanilla", str(TOOL / "public_default.R"), str(native),
    str(reference), str(out / "run")], env, frozen)
