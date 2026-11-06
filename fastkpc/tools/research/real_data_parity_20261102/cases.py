"""Run fresh CPU/GPU comparisons and retain failures, logs, and identities."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102"
support = TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py"
spec = importlib.util.spec_from_file_location("parity_jobs", support)
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
os.chdir(ROOT)
assert len(sys.argv) >= 2, "cases.py OUTPUT_TAG [DATASET ...]"
tag = sys.argv[1]
assert tag and Path(tag).name == tag
cases = sys.argv[2:] or ["20221022181603", "20220418113957", "20240315113222", "canonical_orientation"]
base = BASE / tag
base.mkdir()
attestation = jobs.read(ROOT / "fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json")
native = ROOT / "fastkpc/build/fastkpc_cuda.so"
assert jobs.sha(native) == attestation["native_sha256"]
inputs = [
    ROOT / "causalDiscovery/20221022181603/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    ROOT / "causalDiscovery/20220418113957/logged-liver-exhausted-allData-causalDiscoveryInput.RData",
    ROOT / "causalDiscovery/20240315113222/CD8.c11.Tex.PDCD1-causalDiscoveryInput.RData",
    ROOT / "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
]
files = inputs + [native, TOOL / "run.R", Path(__file__), support,
    ROOT / "fastkpc/tests/test_compatible_cuda_wanpdag_authority.R"]
files += list((ROOT / "fastkpc/R").glob("*.R")) + list((ROOT / "kpcalg/R").glob("*.R"))
files += [p for p in (ROOT / "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2").glob("*") if p.is_file()]
if "canonical_orientation" in cases:
    files.append(BASE / "canonical/run/receipt.rds")
deps = {str(p): jobs.sha(p) for p in files}

def frozen():
    jobs.check_map(attestation["source_sha256"])
    for p, expected in deps.items():
        assert jobs.sha(p) == expected, p

for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
jobs.write(base / "protocol.json", dict(
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    native_sha256=jobs.sha(native), dependencies=deps, cases=cases,
    evidence_class="correctness only; one fresh CPU/GPU comparison per complete real input",
    ci_method="dcc.gamma", alpha=0.1, index=1, numCol=35, max_conditioning_size="Inf",
    physical_gpu=1, cpu_affinity=list(range(20)), math_threads=1,
    p_value_absolute_tolerance=1e-10, pmax_absolute_tolerance=1e-10,
    canonical_only_optimizations_on_other_shapes=False,
    orientation_authority="kpcalg CPU WAN-PDAG; not native GPU orientation",
))
reports = []
for case in cases:
    print("START", case, flush=True)
    jobs.job(base, case, ["Rscript", "--vanilla", str(TOOL / "run.R"), case, str(base / case)], env, frozen)
    report = jobs.read(base / case / "report.json")
    assert report["pass"]
    reports.append(report)
jobs.write(base / "summary.json", dict(status="PASS", reports=reports))
print("PASS all", len(cases), "cases", flush=True)
