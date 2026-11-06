"""One isolated current-native 500x50 shape/correctness check; no rebuild."""
import importlib.util
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
support = TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py"
spec = importlib.util.spec_from_file_location("shape_jobs", support)
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
os.chdir(ROOT)
out = ROOT / "fastkpc/artifacts/full_cuda_ci/shape_500x50_20261102"
out.mkdir()
attestation = jobs.read(ROOT / "fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json")
native = ROOT / "fastkpc/build/fastkpc_cuda.so"
assert jobs.sha(native) == attestation["native_sha256"]
files = [native, TOOL / "run.R", Path(__file__), support,
    ROOT / "fastkpc/tests/fixtures/full_cuda_ci_development_500x50_v2.rds"]
files += list((ROOT / "fastkpc/R").glob("*.R"))
dependencies = {str(p.relative_to(ROOT)): jobs.sha(p) for p in files}

def frozen():
    jobs.check_map(attestation["source_sha256"])
    jobs.check_map(dependencies)

for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
env.update(FASTKPC_DCC_GAMMA_EXACT_PRIMARY_OPTIMIZER="1",
           FASTKPC_DCC_GAMMA_EXACT_PRIMARY_FIXED_REUSE="1")
jobs.write(out / "protocol.json", dict(
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    native_sha256=jobs.sha(native), dependencies=dependencies,
    evidence_class="current-native shape/correctness diagnostic against archived CPU oracle",
    fixture="public synthetic 500x50 development v2; not real data or holdout",
    physical_gpu=1, cpu_affinity=list(range(20)), math_threads=1,
    p_value_absolute_tolerance=1e-10, pmax_absolute_tolerance=1e-10,
    canonical_only_optimizations=False))
jobs.job(out, "gpu", ["Rscript", "--vanilla", str(TOOL / "run.R"), str(out / "run")], env, frozen)
report = jobs.read(out / "run/report.json")
assert report["pass"]
artifacts = {str(p.relative_to(ROOT)): jobs.sha(p) for p in out.rglob("*") if p.is_file()}
jobs.write(out / "manifest.json", dict(status="PASS", dependencies=dependencies,
    source_sha256=attestation["source_sha256"], artifact_sha256=artifacts))
print("PASS", report["logical_ci_count"], "logical CI;", report["gpu_elapsed_sec"], "s", flush=True)
