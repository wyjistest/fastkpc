"""Add a warm-context attribution follow-up after the original capacity gate failed."""
from pathlib import Path
import subprocess
import sys
from common import NATIVE, ROOT, TOOL, check_identity, load, sha, write_new

out=Path(sys.argv[1]).resolve()
check_identity(out)
assert load(out/'capacity-process.json')['exit_code'] != 0
assert 'FAIL post-work standalone slot capacity limit' in (out/'capacity.log').read_text()
cmd=['g++','-O2','-std=c++17','-pthread','-I'+str(ROOT/'fastkpc/src'),'-I/usr/local/cuda/include',
     str(TOOL/'capacity_warm_diagnostic.cpp'),str(out/'audit.so'),str(out/'native_observe.so'),
     str(out/'slot_backend.so'),str(NATIVE),'-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(NATIVE.parent),
     '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl',
     '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRlapack','-lRblas',
     '-o',str(out/'capacity-warm-diagnostic')]
subprocess.run(cmd,check=True)
write_new(out/'capacity-warm-diagnostic-build.json',dict(command=cmd,
    original_build_sha256=sha(out/'build.json'),inputs={str(TOOL/'capacity_warm_diagnostic.cpp'):sha(TOOL/'capacity_warm_diagnostic.cpp'),
    str(TOOL/'build_capacity_warm_diagnostic.py'):sha(TOOL/'build_capacity_warm_diagnostic.py')},
    output_sha256=sha(out/'capacity-warm-diagnostic'),scope='quantify failed capacity gate and validate completed outputs; no campaign'))
print('PASS follow-up diagnostic build; original executable and evidence untouched',flush=True)
