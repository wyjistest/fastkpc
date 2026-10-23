"""Build a source-bound probe without changing production sources or native."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
here = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
native = root / "fastkpc/build/fastkpc_cuda.so"
source = root / "fastkpc/src/cuda/full_cuda_ci_vertical.cu"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = sha(native)
assert before == "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"
text = source.read_text()
start = text.index('      check_cuda(cudaEventRecord(execution_context->distance_start(), stream)')
end = text.index('                 "record legacy eig finalize end");', start)
end += len('                 "record legacy eig finalize end");')
stage = text[start:end]
assert stage.count("cublasDgemmStridedBatched") == 1
assert stage.count("component_cache->vectors(slot)") == 1
(out / "component_stage.inc").write_text(stage + "\n")
command = ["/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_89", "-std=c++17",
  "-Xcompiler=-fPIC", "-shared", str(here / "probe.cu"),
  "-I"+str(root / "fastkpc/src/cuda"), "-I"+str(out),
  "-Xlinker=-Bsymbolic", str(native), "-Xlinker=-rpath="+str(native.parent),
  "-lcublas", "-lcusolver", "-o", str(out / "probe.so")]
print(" ".join(command), flush=True)
capture_only = len(sys.argv) == 3 and sys.argv[2] == "--capture-only"
if capture_only:
    previous = json.loads((out/"build.json").read_text())
    assert previous["probe_sha256"] == sha(out/"probe.so")
    for p in (here/"probe.cu",source,root/"fastkpc/src/cuda/full_cuda_ci_vertical.hpp"):
        assert previous["sources"][str(p.relative_to(root))] == sha(p)
else:
    subprocess.run(command, check=True)
capture_command = ["g++","-O2","-std=c++17","-fPIC","-shared",str(here/"capture.cpp"),
  "-I"+str(root/"fastkpc/src/cuda"),"-I/usr/local/cuda/include",
  "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-ldl",
  "-o",str(out/"capture.so")]
subprocess.run(capture_command, check=True)
assert sha(native) == before
sources = [here / "probe.cu", here / "build.py", here/"capture.cpp", source,
           root / "fastkpc/src/cuda/full_cuda_ci_vertical.hpp",
           root / "fastkpc/src/full_cuda_ci_contract.hpp"]
report = dict(schema="component-payload-probe-build-v1", native_sha256=before,
  command=command, capture_command=capture_command,
  sources={str(p.relative_to(root)):sha(p) for p in sources},
  extracted_stage_sha256=sha(out / "component_stage.inc"),
  probe_sha256=sha(out / "probe.so"),capture_sha256=sha(out/"capture.so"))
(out / "build.json").write_text(json.dumps(report, indent=2)+"\n")
