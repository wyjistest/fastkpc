"""Bind native capture, direct payload replay, sources and sanitizer evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

working, destination = map(lambda p:Path(p).resolve(),sys.argv[1:])
destination.mkdir(parents=True,exist_ok=True)
root = Path.cwd()
here = Path(__file__).resolve().parent
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build = json.loads((working/"build/build.json").read_text())
capture = json.loads((working/"native/capture.json").read_text())
full = json.loads((working/"native-full/report.json").read_text())
assert build["native_sha256"] == capture["native_sha256"] == full["native_sha256"]
assert sha(root/"fastkpc/build/fastkpc_cuda.so") == build["native_sha256"]
assert all(capture["gates"].values())
assert full["unique_targets"] == 6632 and full["resources_before"] == full["resources_after"]
assert full["native_capture_sha256"] == sha(working/"native/capture.json")
assert full["build_sha256"] == sha(working/"build/build.json")
assert all(full["fields"][name] == 0 for name in ("vectors","selected_values","total"))
changed = full["unique_changed_keys"]["self_moment"]
assert len(changed) == 1002 and full["fields"]["self_moment"] == 4*len(changed)
for key in changed:
    record = json.loads((working/"native/raw"/(key+".json")).read_text())
    assert record["component_misses"] == 1
sanitizer = (working/"memcheck.log").read_text()
assert "ERROR SUMMARY: 0 errors" in sanitizer
assert "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in sanitizer
smoke = json.loads((working/"memcheck/report.json").read_text())
assert smoke["resources_before"] == smoke["resources_after"]
for path,digest in build["sources"].items():
    assert sha(root/path) == digest,("source drift",path)
assert sha(here/"capture_native.R") == capture["runner_sha256"]
assert sha(working/"build/capture.so") == capture["interposer_sha256"]
files = ["build/build.json","build/probe.so","build/capture.so","build/component_stage.inc",
         "native/capture.json","native/receipt.rds","native/keys.txt",
         "native-smoke/report.json","native-full/report.json",
         "memcheck/report.json","memcheck.log"]
for name in files:
    target = destination/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(working/name,target)
raw = working/"native/raw"
raw_files = sorted(p for p in raw.iterdir() if p.is_file())
assert len(raw_files) == 6632*3
raw_manifest = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in raw_files}
raw_archive = destination/"native/raw.tar.gz"
with tarfile.open(raw_archive,"w:gz",compresslevel=1) as archive:
    for path in raw_files:
        archive.add(path,arcname=path.name,recursive=False)
with tarfile.open(raw_archive,"r:gz") as archive:
    members = archive.getmembers()
    assert len(members) == len(raw_manifest)
    for member in members:
        content = archive.extractfile(member).read()
        expected = raw_manifest[member.name]
        assert len(content) == expected["bytes"]
        assert hashlib.sha256(content).hexdigest() == expected["sha256"]
(destination/"native/raw-manifest.json").write_text(json.dumps(raw_manifest,indent=2)+"\n")
sources = {str(p.relative_to(root)):sha(p) for p in sorted(here.iterdir()) if p.is_file()}
artifacts = {str(p.relative_to(destination)):dict(bytes=p.stat().st_size,sha256=sha(p))
             for p in sorted(destination.rglob("*")) if p.is_file() and p.name!="manifest.json"}
manifest = dict(schema="component-payload-campaign-v1",audit="PASS",decision=full["status"],
  head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  runtime_producer="dcb1749",native_sha256=build["native_sha256"],
  gpu_inventory=subprocess.check_output(["nvidia-smi",
    "--query-gpu=index,name,uuid,driver_version","--format=csv,noheader"],text=True).strip().splitlines(),
  sources=sources,artifacts=artifacts,
  evidence_class="accepted-native diagnostic capture and source-bound GPU payload replay; no performance or promotion claim",
  dependency="fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021/capture-inf/raw.tar.gz")
(destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("audit PASS",full["status"],"manifest",sha(destination/"manifest.json"),flush=True)
