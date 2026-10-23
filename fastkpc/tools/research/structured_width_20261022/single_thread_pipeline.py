"""One host thread submits every 64-wide batch before the first status checkpoint."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, 'fastkpc/tools/research/structured_adapter_20261022')
import run as R
import numpy as np
lib = R.load_lib(Path('fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/adapter.so'))
x, keys, _ = R.real_fixture(False)
out = {}
for pool in (1, 5, 10, 20):
    ps = []
    for j in range(pool):
        ids = np.arange(64 * (j % 10), 64 * (j % 10) + 64)
        ps.append(R.Probe(lib, x[ids], [keys[i] for i in ids], capacity=64))
    try:
        def once():
            for p in ps: p.begin(np.arange(64), 1)
            for p in ps: p.finish(64)
        once(); [p.reset() for p in ps]
        ts = []
        for _ in range(3):
            t = time.perf_counter(); once(); ts.append(time.perf_counter() - t); [p.reset() for p in ps]
        comps = 64 * pool; med = sorted(ts)[1]
        out[f'{pool}_contexts_one_thread'] = dict(components=comps, seconds=ts, components_per_s=comps / med)
        print(pool, 'contexts x64, one host thread,', comps, 'components', ['%.3f' % t for t in ts], '%.0f comp/s' % (comps / med), flush=True)
    finally:
        for p in ps: p.close()
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
