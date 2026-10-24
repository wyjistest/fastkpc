"""Build profile-only companions; preserve the primary timing executable."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[4];tool=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'replay-build.json').read_text())
assert all(sha(root/p)==h for p,h in build['sources'].items())
native=root/'fastkpc/build/fastkpc_cuda.so';assert sha(native)==build['native_sha256']
assert all(sha(out/p)==h for p,h in build['outputs'].items())
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
old=tool.parent/'optimizer_windows_20261022'
common=['g++','-O2','-std=c++17','-fPIC','-pthread','-I'+str(out),'-I'+str(tool),'-I'+str(old),
 '-I'+str(root/'fastkpc/src'),'-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib']
commands=[common+['-shared',str(tool/'nvtx_observe.cpp'),str(tool/'audit.cpp'),str(native),
 '-Wl,-rpath,'+str(native.parent),'-lcudart','-lcublas','-lnvToolsExt','-ldl','-o',str(out/'profile-observe.so')],
 common+[str(tool/'profile.cpp'),str(out/'profile-observe.so'),str(native),'-Wl,-rpath,'+str(out),
 '-Wl,-rpath,'+str(native.parent),'-lR','-lcudart','-lnvToolsExt','-ldl','-o',str(out/'profile-replay')]]
for cmd in commands:subprocess.run(cmd,check=True)
sources=[tool/p for p in ['profile_build.py','profile.cpp','nvtx_observe.cpp','audit.cpp','audit.hpp','layer_fixture.hpp']]+[old/'fixture.hpp',old/'observe.hpp']
(out/'profile-build.json').write_text(json.dumps(dict(native_sha256=sha(native),commands=commands,
 sources={str(p.relative_to(root)):sha(p) for p in sources},
 outputs={p:sha(out/p) for p in ['profile-observe.so','profile-replay']}),indent=2)+'\n')
print('PASS profile-only build; primary replay unchanged')
