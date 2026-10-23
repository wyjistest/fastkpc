"""Audit immutable identities and collect a self-contained replay archive."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

work=Path(sys.argv[1]).resolve()
root=Path.cwd()
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):
    return json.loads((work/name).read_text())
def audit(mapping,base=root):
    for name,expected in mapping.items():
        assert sha(base/name)==expected, f"identity changed: {name}"
def copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert sha(source)==sha(target)

capture=read("capture.json")
benchmark=read("benchmark.json")
build=read("build.json")
witness=read("witness-qualification.json")
smoke=read("sanitizer-smoke.json")
assert benchmark["pass"] and witness["pass_"] and smoke["pass_"]
assert capture["resource_counters_closed"]
native=root/"fastkpc/build/fastkpc_cuda.so"
assert sha(native)==capture["native_sha256"]
audit(capture["source_files"])
audit(benchmark["source_sha256"])
audit(benchmark["artifact_sha256"],work)
audit(build["sources"])
audit(build["artifacts"],work)
audit(witness["source_sha256"])
audit(smoke["source_sha256"])
audit(read("profile-run.json")["sources"])
memcheck=Path("/tmp/fastkpc-structured-real-memcheck.log").read_text()
assert "ERROR SUMMARY: 0 errors" in memcheck
assert "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in memcheck
assert len(read("profile-analysis.json")["trials"])==3

sources=set(build["sources"])|set(benchmark["source_sha256"])
sources.update(str(p.relative_to(root)) for p in (root/"fastkpc/tools/research/structured_real_20261020").iterdir() if p.is_file())
sources.update([
 "fastkpc/tools/research/guarded_qr_20261020/common.R",
 "fastkpc/tools/research/exact_speedup_20261020/structured_dcov_reference.py",
 "fastkpc/src/cuda/third_party/lapack_3_12_small_dgesdd_device.inc",
 "fastkpc/src/cuda/third_party/lapack_3_12_block_reflector_device.inc",
 "fastkpc/src/cuda/third_party/lapack_3_12_dormbr_device.inc",
 "fastkpc/docs/PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md",
 "fastkpc/docs/PHASE10_STRUCTURED_EIG_SCALING_20261020.md",
 "fastkpc/inst/contracts/full_cuda_ci/numerical_contract_v1.json",
])
source_hashes={}
for name in sorted(sources):
    p=(root/name).resolve()
    relative=p.relative_to(root)
    copy(p,work/"sources"/relative)
    source_hashes[str(relative)]=sha(p)
for stem in ("capture","build","benchmark","export","witness","nsys-retry","memcheck"):
    copy(Path(f"/tmp/fastkpc-structured-real-{stem}.log"),work/"logs"/f"{stem}.log")
copy(Path("/tmp/fastkpc-exact-speedup-20261020/subset-restored-queues32.rds"),
     work/"inputs/subset-source-receipt.rds")
copy(root/"fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds",
     work/"inputs/canonical-data.rds")
files={str(p.relative_to(work)):sha(p) for p in sorted(work.rglob("*"))
       if p.is_file() and p!=work/"manifest.json"}
manifest=dict(schema="structured-real-source-evidence-v1",audit="PASS",
    created_utc=datetime.now(timezone.utc).isoformat(),
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    production_native_sha256=sha(native),source_files=source_hashes,files=files,
    evidence_class="isolated development diagnostic; no production integration or promotion",
    runtime_dirty=subprocess.check_output(["git","status","--porcelain","--","fastkpc/src","fastkpc/R"],text=True).strip(),
    decision="GO_BOUNDED_COMPONENT_REPLAY; FULL_METHOD_INTEGRATION_NOT_YET_GO")
assert not manifest["runtime_dirty"]
(work/"manifest.json").write_text(json.dumps(manifest,indent=2))
# Check the actual archived bytes, not only the source paths used for copying.
audit(files,work)
print("PASS",len(files),"artifacts; manifest SHA256",sha(work/"manifest.json"))
