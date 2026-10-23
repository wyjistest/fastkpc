"""Build independent capture/replay libraries without touching production."""
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
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
before = sha(native)
assert before == "fe8286f7e6c6114ef66e040b16b693f5abcb23333dc9c0fd767c042c509c3c50"
cuda = Path("/usr/local/cuda")
commands = [
    ["g++", "-O2", "-std=c++17", "-fPIC", "-shared", str(here/"capture.cpp"),
     "-I"+str(root/"fastkpc/src/cuda"), "-I"+str(cuda/"include"),
     "-L"+str(cuda/"lib64"), "-Wl,-rpath,"+str(cuda/"lib64"),
     "-lcudart", "-lcublas", "-ldl", "-o", str(out/"capture.so")],
    [str(cuda/"bin/nvcc"), "-O3", "-arch=sm_89", "--fmad=false", "-std=c++17",
     "-Xcompiler=-fPIC", "-shared", str(here/"replay.cu"), "-lcublas",
     "-o", str(out/"replay.so")],
]
for command in commands:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
assert sha(native) == before
sources = [here/"build.py", here/"capture.cpp", here/"replay.cu",
           root/"fastkpc/src/cuda/mgcv_multi_penalty_gcv.hpp",
           root/"fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.hpp",
           root/"fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu",
           root/"fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.cpp",
           root/"fastkpc/src/full_cuda_ci_one_call.cpp"]
report = dict(schema="production-cohort-frozen-build-v1", native_sha256=before,
              commands=commands, sources={str(p.relative_to(root)):sha(p) for p in sources},
              binaries={p.name:sha(p) for p in (out/"capture.so", out/"replay.so")})
(out/"build.json").write_text(json.dumps(report, indent=2)+"\n")
