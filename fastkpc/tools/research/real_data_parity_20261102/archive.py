"""Summarize completed comparisons; keep original failed harness attempts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/real_data_parity_20261102"

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")

attestation = read(ROOT / "fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261030.json")
for name, expected in attestation["source_sha256"].items():
    assert sha(ROOT / name) == expected, name
native_sha = sha(ROOT / "fastkpc/build/fastkpc_cuda.so")
assert native_sha == attestation["native_sha256"]
canonical = read(BASE / "canonical/run/report.json")
assert all(canonical["gates"].values())
batch = read(BASE / "attempt4/summary.json")
assert batch["status"] == "PASS"
assert all(r["pass"] and all(r["gates"].values()) for r in batch["reports"])
orientation = batch["reports"][-1]
assert orientation["dataset"] == "cancer_RD"
rows = []
for r in batch["reports"][:-1]:
    rows.append(dict(dataset=r["dataset"], n=r["n"], p=r["p"],
        logical_ci_count=r["logical_ci_count"],
        max_ci_error=r["ci"]["max_abs_error"], max_pmax_error=r["pmax"]["max_abs_error"],
        orientation_ci_count=r["orientation_ci_count"],
        gpu_elapsed_sec=r["gpu_elapsed_sec"], cpu_elapsed_sec=r["cpu_elapsed_sec"],
        raw_pcalg_call_order_exact=r["raw_pcalg_call_order_exact"],
        reference="fresh independent pcalg/kpcalg + native CPU-order replay", pass_=r["pass"]))
rows.append(dict(dataset="cancer_RD", n=351, p=48, logical_ci_count=240498,
    max_ci_error=canonical["oracle_errors"][0], max_pmax_error=canonical["oracle_errors"][1],
    gpu_skeleton_elapsed_sec=canonical["elapsed_sec"],
    orientation_ci_count=orientation["orientation_ci_count"],
    reference="archived full CPU skeleton + fresh CPU-authority orientation", pass_=True))
for path in [BASE / "canonical/canonical-process.json"] + list((BASE / "attempt4").glob("*-process.json")):
    record = read(path)
    assert record["exit_code"] == 0 and record["isolated"], path
write(BASE / "summary.json", dict(status="PASS_REAL_DATA_CORRECTNESS", rows=rows,
    total_logical_ci=sum(r["logical_ci_count"] for r in rows),
    maximum_ci_error=max(r["max_ci_error"] for r in rows),
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    native_sha256=native_sha, production_source_files_verified=len(attestation["source_sha256"]),
    canonical_gate_count=len(canonical["gates"]),
    numerical_tolerance=1e-10, physical_gpu=1,
    limitations=["dcc.gamma, alpha=0.1, index=1, numCol=35, m.max=Inf only",
        "canonical-only structured/slots/postback enabled only on 351x48",
        "orientation uses kpcalg CPU authority", "no new performance median or promotion",
        "raw pcalg node-major calls differ from native edge-major order; both preserved and independently checked"],
    retained_attempts=dict(attempt1="raw pcalg order versus native-order comparator mismatch",
        attempt2="new CPU replay provider did not handle an empty terminal plan",
        attempt3="generic CPU replay padded n.edgetests with empty levels",
        attempt4="all corrected harness gates pass; numerical and graph gates unchanged")))
files = [p for p in BASE.rglob("*") if p.is_file()]
files += [p for p in TOOL.iterdir() if p.is_file()]
write(BASE / "manifest.json", dict(native_sha256=native_sha,
    files={str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
    production_source_sha256=attestation["source_sha256"]))
print(json.dumps(read(BASE / "summary.json"), indent=2), flush=True)
