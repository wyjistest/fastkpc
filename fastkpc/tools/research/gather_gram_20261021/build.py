"""Build the gather probe bound to the accepted native; no production change."""
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
gstart = text.index('      check_cublas(cublasDgemmStridedBatched(')
gend = text.index('      check_cuda(cudaGetLastError(), "launch legacy eig self moments");', gstart)
gend += len('      check_cuda(cudaGetLastError(), "launch legacy eig self moments");')
gram = text[gstart:gend]
assert gram.count("cublasDgemmStridedBatched") == 1 and gram.count("reduce_legacy_self_moment_kernel") == 1
assert stage.find(gram) >= 0, "gram statements must be a verbatim sub-range of the full stage"
(out / "gram_stage.inc").write_text(gram + "\n")
command = ["/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_89", "-std=c++17",
  "-Xcompiler=-fPIC", "-shared", str(here / "probe_gather.cu"),
  "-I"+str(root / "fastkpc/src/cuda"), "-I"+str(out),
  "-Xlinker=-Bsymbolic", str(native), "-Xlinker=-rpath="+str(native.parent),
  "-lcublas", "-lcusolver", "-o", str(out / "probe_gather.so")]
print(" ".join(command), flush=True)
subprocess.run(command, check=True)
assert sha(native) == before
sources = [here / "probe_gather.cu", here / "build.py", source,
           root / "fastkpc/src/cuda/full_cuda_ci_vertical.hpp"]
report = dict(schema="gather-gram-probe-build-v1", native_sha256=before, command=command,
  sources={str(p.relative_to(root)): sha(p) for p in sources},
  extracted_stage_sha256=sha(out / "component_stage.inc"),
  extracted_gram_sha256=sha(out / "gram_stage.inc"),
  probe_sha256=sha(out / "probe_gather.so"))
(out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("extracted_stage_sha256", "extracted_gram_sha256", "probe_sha256")}, indent=1))
