"""Time structured vs dense three-field production at different in-flight widths.
Uses Codex's adapter (fastkpc/artifacts/.../structured_adapter_20261022/adapter.so)
and runner classes; no shadow D2H inside timed regions. Run from repo root."""
import sys, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, 'fastkpc/tools/research/structured_adapter_20261022')
import run as R
import numpy as np

lib = R.load_lib(Path('fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/adapter.so'))
x, keys, _ = R.real_fixture(False)
assert x.shape == (640, 351)
out = {}

def timed_build(p, width, mode, reps=3):
    p.build(width=width, mode=mode); p.reset()          # warmup
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); p.build(width=width, mode=mode); ts.append(time.perf_counter() - t); p.reset()
    return ts

for cap in (22, 64):
    p = R.Probe(lib, x, keys, capacity=cap)
    try:
        for mode, label in ((1, 'structured'), (0, 'dense')):
            ts = timed_build(p, cap, mode)
            med = sorted(ts)[1]
            out[f'{label}_single_context_width{cap}'] = dict(seconds=ts, components_per_s=640 / med)
            print(label, 'single context width', cap, ['%.3f' % t for t in ts], '%.0f comp/s' % (640 / med), flush=True)
    finally:
        p.close()

for pool in (2, 5, 10):
    chunks = np.array_split(np.arange(640), pool)
    ps = [R.Probe(lib, x[c], [keys[i] for i in c], capacity=64) for c in chunks]
    try:
        def go(p): p.build(width=64, mode=1)
        with ThreadPoolExecutor(max_workers=pool) as ex: list(ex.map(go, ps))
        for p in ps: p.reset()
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            with ThreadPoolExecutor(max_workers=pool) as ex: list(ex.map(go, ps))
            ts.append(time.perf_counter() - t)
            for p in ps: p.reset()
        med = sorted(ts)[1]
        out[f'structured_{pool}_contexts_x64'] = dict(seconds=ts, components_per_s=640 / med)
        print('structured', pool, 'contexts', ['%.3f' % t for t in ts], '%.0f comp/s' % (640 / med), flush=True)
    finally:
        for p in ps: p.close()

Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
