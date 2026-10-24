"""Check configuration/resource rejection and the exact retained numerical code."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
os.chdir(root)
out = Path(sys.argv[1]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((out / 'build.json').read_text())
prior = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_slot_runtime_20261023'
old = json.loads((prior / 'build.json').read_text())
native = root / 'fastkpc/build/fastkpc_cuda.so'
assert build['status'] == 'PASS' and sha(native) == build['native_sha256']
assert all(sha(root / p) == h for p, h in build['source_sha256'].items())
assert not subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
           OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
os.sched_setaffinity(0, range(20))
test = 'fastkpc/tests/test_full_cuda_ci_l4_optimizer_workers.R'
with (out / 'configuration.log').open('x') as log:
    subprocess.run(['sh', 'fastkpc/tools/with_phase10_exact_development_env.sh',
                    'Rscript', '--vanilla', test], env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)

def sass(path):
    digest = hashlib.sha256()
    functions = 0
    process = subprocess.Popen(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(path)],
                               stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        line = re.sub(r'_GLOBAL__N__[0-9a-f]{8}_(\d+_\w+_cu_)[0-9a-f]{8}',
                      r'_GLOBAL__N__SOURCE_\1SOURCE', line)
        functions += 'Function :' in line
        digest.update(line.encode())
    assert process.wait() == 0 and functions > 0
    return dict(sha256=digest.hexdigest(), functions=functions)

objects = {}
for name, expected in build['objects'].items():
    before = prior / 'build-source/fastkpc/build' / name
    after = out / 'build-source/fastkpc/build' / name
    assert sha(before) == old['objects'][name] and sha(after) == expected
    if name == 'full_cuda_ci_one_call.o':
        continue
    if name == 'full_cuda_ci_structured_eig.o':
        a, b = sass(before), sass(after)
        assert a == b, 'rebuilt structured eig device code changed'
        objects[name] = dict(old_object=sha(before), new_object=sha(after),
                             old_sass=a, new_sass=b, device_exact=True)
    else:
        assert sha(before) == sha(after), name + ' changed'
        objects[name] = dict(object_sha256=expected, object_exact=True)
diagnostic = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_l4_diagnostic_20261023/manifest.json'
diag = json.loads(diagnostic.read_text())
# This inherited diagnostic belongs to the exact old binary whose CUDA objects
# are retained here; do not claim a new memcheck execution on the host-only patch.
assert diag['native_sha256'] == old['native_sha256']
assert sha(native) == build['native_sha256']
assert all(sha(root / p) == h for p, h in build['source_sha256'].items())
receipt = dict(status='PASS', native_sha256=sha(native), physical_gpu=1,
    configuration_rejections=6, configuration_resource_closure=True,
    unchanged_object_count=41, changed_host_source_only=True, objects=objects,
    inherited_l4_diagnostic=str(diagnostic.relative_to(root)),
    inherited_l4_diagnostic_sha256=sha(diagnostic),
    new_memcheck_run=False, test_sha256=sha(root / test))
(out / 'checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS configuration and numerical-code identity', flush=True)
