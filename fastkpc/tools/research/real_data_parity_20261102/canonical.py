"""Rerun the accepted canonical configuration; no rebuild or promotion."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102"
support = TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py"
spec = importlib.util.spec_from_file_location("parity_jobs", support)
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
os.chdir(ROOT)
attestation = jobs.read(ROOT / "fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json")
native = ROOT / "fastkpc/build/fastkpc_cuda.so"
assert jobs.sha(native) == attestation["native_sha256"]
jobs.check_map(attestation["source_sha256"])
build = jobs.read(ROOT / "fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/candidate/build.json")
build["native_path"] = str(native)
base = BASE / "canonical"
base.mkdir(parents=True)
out = base / "run"
out.mkdir()
jobs.write(base / "build.json", build)
reference = ROOT / "fastkpc/artifacts/full_cuda_ci/build_consistency_20261029/canonical/qualification/receipt.rds"
runner = TOOL.parent / "build_consistency_20261029/canonical.R"
deps = {str(p): jobs.sha(p) for p in [
    Path(__file__), support, runner, reference, base / "build.json", native,
    ROOT / "fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
    TOOL.parent / "exact_speedup_20261020/pinned_optimizer_gates.R",
    ROOT / "fastkpc/tools/run_phase10_pvalue_contract_audit.R",
    ROOT / "fastkpc/tools/with_phase10_exact_development_env.sh",
]}
oracle = ROOT / "fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2"
for p in list((ROOT / "fastkpc/R").glob("*.R")) + list(oracle.glob("*")):
    if p.is_file():
        deps[str(p)] = jobs.sha(p)

def frozen():
    jobs.check_map(attestation["source_sha256"])
    for p, expected in deps.items():
        assert jobs.sha(p) == expected, p

for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
env["FASTKPC_DCC_GAMMA_VECTOR_POSTBACK"] = "1"
jobs.write(base / "protocol.json", dict(
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    native_sha256=jobs.sha(native), dependencies=deps,
    evidence_class="fresh-process real-data correctness rerun; saved CPU skeleton authority; no performance promotion",
    dataset="cancer_RD", shape=[351, 48], ci_method="dcc.gamma",
    alpha=0.1, index=1, numCol=35, max_conditioning_size="Inf",
    physical_gpu=1, fresh_process=True, fresh_caches=True, repetitions=1,
    p_value_absolute_tolerance=1e-10, pmax_absolute_tolerance=1e-10,
    production_configuration="accepted structured rows, work-pair, L3/L4/L5/L6 slots, q37/q46 postback",
))
jobs.job(base, "canonical", [
    "sh", "fastkpc/tools/with_phase10_exact_development_env.sh", "Rscript", "--vanilla",
    str(runner), str(out), "1", str(reference), "-",
], env, frozen)
report = jobs.read(out / "report.json")
assert all(report["gates"].values())
print(json.dumps(dict(status="PASS", elapsed_sec=report["elapsed_sec"],
    oracle_errors=report["oracle_errors"], gate_count=len(report["gates"])), indent=2), flush=True)
