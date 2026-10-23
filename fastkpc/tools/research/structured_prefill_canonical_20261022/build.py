"""Build an immutable production-only source snapshot without retained objects.

This is a full source-closure build, not a clean committed producer. Install
the resulting native and objects only after generation identity and the full
build pass. Research files are deliberately absent from the build snapshot.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
finish_only = len(sys.argv) == 3 and sys.argv[2] == "--finish"
out.mkdir(parents=True, exist_ok=True)
snapshot = out / "build-source"
assert snapshot.exists() if finish_only else not snapshot.exists(), "invalid build epoch"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources = [p for p in (root / "fastkpc/src").rglob("*")
           if p.is_file() and p.suffix in (".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".inc", ".txt")]
sources += [root / "fastkpc/tools/build_cuda_native.sh",
            root / "fastkpc/tools/generate_recorded_dbdsqr.py"]
hashes = {str(p.relative_to(root)): sha(p) for p in sorted(sources)}
if not finish_only:
    for source in sources:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert not (snapshot / "fastkpc/build").exists()
assert not (snapshot / "fastkpc/tools/research").exists()
native = root / "fastkpc/build/fastkpc_cuda.so"
if not finish_only:
    with native.open("rb") as source, (out / "previous-native.so.gz").open("wb") as dest:
        with gzip.GzipFile(fileobj=dest, mode="wb", mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
with gzip.open(out / "previous-native.so.gz", "rb") as source:
    old_native_sha = hashlib.sha256(source.read()).hexdigest()
generated = out / "generated-dbdsqr.inc"
subprocess.run(["python3", "-I", str(snapshot / "fastkpc/tools/generate_recorded_dbdsqr.py"),
                str(generated)], cwd="/tmp", check=True)
assert sha(generated) == sha(root / "fastkpc/build/recorded_dbdsqr.inc"), "generator changed bytes"
env = os.environ.copy()
for name in list(env):
    if name.startswith("FASTKPC_INCREMENTAL_") or name == "FASTKPC_MULTI_PENALTY_BUCKETS":
        del env[name]
started = time.time()
if not finish_only:
    with (out / "build.log").open("w") as log:
        subprocess.run(["sh", "-x", str(snapshot / "fastkpc/tools/build_cuda_native.sh")],
                       cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
else:
    assert (out / "build.log").read_text().splitlines()[-1].startswith("built: "), "full build did not finish"
build = snapshot / "fastkpc/build"
assert all(sha(root / path) == value and sha(snapshot / path) == value
           for path, value in hashes.items()), "production source changed during full build"
objects = {p.name: sha(p) for p in sorted(build.glob("*.o"))}
link_recipe = (snapshot / "fastkpc/tools/build_cuda_native.sh").read_text().rsplit('"$CXX" -shared', 1)[1]
expected_objects = set(re.findall(r'"\$BUILD/([^/"\n]+\.o)"', link_recipe))
assert set(objects) == expected_objects, "full-build object inventory does not match recipe"
assert sha(build / "recorded_dbdsqr.inc") == sha(generated)
for path in list(build.glob("*.o")) + [build / "recorded_dbdsqr.inc", build / "fastkpc_cuda.so"]:
    shutil.copyfile(path, root / "fastkpc/build" / path.name)
receipt = dict(schema="structured-prefill-canonical-full-build-v1", status="PASS",
               source_build="full build from empty directory; production-only source snapshot",
               evidence_class="uncommitted source-closure development; not clean producer",
               previous_native_sha256=old_native_sha, native_sha256=sha(native),
               source_sha256=hashes, objects=objects,
               generator_output_sha256=sha(generated), generator_output_unchanged=True,
               elapsed_since_snapshot_sec=time.time()-snapshot.stat().st_mtime,
               producer_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
(out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("PASS full source-closure build", receipt["native_sha256"], flush=True)
