"""Independent CPU references for the predefined additional input shapes."""
import importlib.util
import os
from pathlib import Path
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shape_jobs", TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
out = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102" / (sys.argv[1] if len(sys.argv) > 1 else "references")
out.mkdir()
native = ROOT / "fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/candidate/fastkpc_cuda.so"
for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
deps = {str(p): jobs.sha(p) for p in [native, TOOL / "run.R", Path(__file__)]}

def frozen():
    jobs.check_map(deps)

cases = sys.argv[2:] or ["dense500", "boundary75", "boundary76", "large1025", "blocks256"]
jobs.write(out / "protocol.json", dict(cases=cases, dependencies=deps,
    authority="legacy R mgcv, native CPU dCov; unchanged parent native",
    max_conditioning_size="Inf", alpha=.1, index=1, numCol=35))
for case in cases:
    print("CPU START", case, flush=True)
    jobs.job(out, case, ["Rscript", "--vanilla", str(TOOL / "run.R"),
        case, "cpu", str(native), str(out / case), "-"], env, frozen)
