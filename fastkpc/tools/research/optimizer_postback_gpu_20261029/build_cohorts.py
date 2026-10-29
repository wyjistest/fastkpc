"""Build a common host harness against either isolated candidate or accepted native."""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT, TOOL, sha, write
out=Path(sys.argv[1]).resolve();variants=sys.argv[2:] or ['reference','candidate']
inc=['-I'+str(ROOT/'fastkpc/src'),'-I/usr/local/cuda/include']
lib=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-lcublas','-ldl',
     '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lRlapack','-lRblas']
audit=out/'audit.so'
if not audit.exists():
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC','-pthread',*inc,
      str(TOOL.parent/'optimizer_layer_refill_20261023/audit.cpp'),*lib,'-o',str(audit)],check=True)
for variant in variants:
    native=ROOT/'fastkpc/build/fastkpc_cuda.so' if variant=='reference' else out/variant/'fastkpc_cuda.so'
    assert native.is_file()
    executable=out/('cohorts-'+variant);assert not executable.exists()
    command=['g++','-O2','-std=c++17','-pthread',*inc,str(TOOL/'cohorts.cpp'),str(audit),str(native),
      '-Wl,-rpath,'+str(out),'-Wl,-rpath,'+str(native.parent),*lib,'-o',str(executable)]
    with (out/('cohorts-'+variant+'-build.log')).open('x') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    write(out/('cohorts-'+variant+'-build.json'),dict(command=command,executable_sha256=sha(executable),
        native_sha256=sha(native),audit_sha256=sha(audit),
        harness_sha256=sha(TOOL/'cohorts.cpp'),build_tool_sha256=sha(TOOL/'build_cohorts.py')))
    print('PASS cohorts harness',variant,flush=True)
