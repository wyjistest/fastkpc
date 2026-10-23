"""Build isolated diagnostics and record exact source/binary identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True,exist_ok=True)
root = Path.cwd()
base = root/"fastkpc/tools/research/exact_speedup_20261020"
here = Path(__file__).resolve().parent
nvcc = "/usr/local/cuda/bin/nvcc"
common = [nvcc,"-O3","-arch=sm_89","--fmad=false","-diag-suppress=2464",
          "-std=c++17","-Xcompiler=-fPIC","-shared"]
commands = [
    [sys.executable,"-I",str(base/"generate_recorded_dbdsqr.py"),str(work/"recorded_dbdsqr.inc")],
    common+["-I"+str(work),str(base/"structured_eig_gpu.cu"),"-o",str(work/"structured-eig.so")],
    common+[str(here/"dense_reference.cu"),"-lcusolver","-o",str(work/"dense-reference.so")],
]
files = [here/"build.py",here/"dense_reference.cu",base/"structured_eig_gpu.cu",
         base/"generate_recorded_dbdsqr.py",root/"fastkpc/tools/phase10_cuda_full_eig_lane_benchmark.cu"]
files += sorted((root/"fastkpc/src/cuda").glob("lapack_312*"))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
before = {str(p.relative_to(root)):sha(p) for p in files if p.is_file()}
for command in commands:
    print(" ".join(command),flush=True)
    subprocess.run(command,check=True)
assert before == {str(p.relative_to(root)):sha(p) for p in files if p.is_file()}
report = dict(schema="structured-real-isolated-build-v1",commands=commands,sources=before,
    nvcc=subprocess.check_output([nvcc,"--version"],text=True),
    artifacts={p.name:sha(p) for p in [work/"recorded_dbdsqr.inc",work/"structured-eig.so",work/"dense-reference.so"]})
(work/"build.json").write_text(json.dumps(report,indent=2))
