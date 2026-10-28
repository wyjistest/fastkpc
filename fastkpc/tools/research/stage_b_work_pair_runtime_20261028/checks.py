"""Qualify native dispatch/lifetime and retain kernel resource records."""
import sys
from support import *

out = Path(sys.argv[1]).resolve()
deep = Path(sys.argv[2]).resolve()
b = read(out/'build.json')
def frozen():
    assert sha(NATIVE) == b['native_sha256']
    check_map(b['source_sha256'])

frozen()
command = ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '-std=c++17', '--cudart', 'shared',
    '-I'+str(ROOT/'fastkpc/src/cuda'), str(TOOL/'runtime_checks.cu'), str(NATIVE),
    '-Xlinker', '-rpath', '-Xlinker', str(NATIVE.parent), '-L/usr/local/lib/R/lib',
    '-Xlinker', '-rpath', '-Xlinker', '/usr/local/lib/R/lib',
    '-lR', '-o', str(out/'runtime-checks')]
attempt = len(list(out.glob('checks-build*.log'))) + 1
with (out/f'checks-build-{attempt}.log').open('x') as f:
    subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
write(out/'checks-build.json', dict(command=command, executable_sha256=sha(out/'runtime-checks'),
    source_sha256=sha(TOOL/'runtime_checks.cu'), native_sha256=b['native_sha256']))
env = environment()
job(out, 'checks', [str(out/'runtime-checks'), str(deep/'capture'), str(out/'checks.json')], env, frozen)
with (out/'kernel-resources.txt').open('x') as f:
    subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage',
        str(out/'build-source/fastkpc/build/full_cuda_ci_structured_eig.o')], stdout=f, check=True)
frozen()
print('PASS integrated native checks', flush=True)
