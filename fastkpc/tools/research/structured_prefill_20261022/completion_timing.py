"""Compare completed three-field production, without shadow D2H or CPU solvers.

Uses the unchanged, qualified adapter binary. One final cudaDeviceSynchronize
joins every producer stream BEFORE stopping the timer; reset is outside it.
All shapes process the same 640 inputs once, using one host thread.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, 'fastkpc/tools/research/structured_adapter_20261022')
import run as R

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
adapter = Path('fastkpc/artifacts/full_cuda_ci/structured_adapter_20261022/adapter.so')
assert R.sha(adapter) == '2c21d39574a9407f471ff698473427c99670dbe13a9a9231619efcc2db38da4d'
lib = R.load_lib(adapter)
cuda = ct.CDLL('/usr/local/cuda/lib64/libcudart.so')
cuda.cudaMemGetInfo.argtypes = [ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_size_t)]


def join():
    assert cuda.cudaDeviceSynchronize() == 0


def used():
    free, total = ct.c_size_t(), ct.c_size_t()
    assert cuda.cudaMemGetInfo(ct.byref(free), ct.byref(total)) == 0
    return total.value - free.value


x, keys, _ = R.real_fixture(False)
assert x.shape == (640, 351)
report = dict(schema='structured-completed-timing-v1',
              adapter_sha256=R.sha(adapter), script_sha256=R.sha(__file__),
              boundary='begin through all payload stores completed; final device join; no shadow D2H',
              components=640, n=351, host_threads=1, cases={})
for mode, pool, width in [(0, 1, 22), (0, 1, 64), (1, 1, 22),
                          (1, 1, 64), (1, 5, 64), (1, 10, 32), (1, 10, 64)]:
    name = ('dense' if mode == 0 else 'structured') + f'_{pool}x{width}'
    chunks = np.array_split(np.arange(640), pool)
    before = R.snapshot(lib)
    ps = []
    started = time.perf_counter()
    try:
        for ids in chunks:
            ps.append(R.Probe(lib, x[ids], [keys[i] for i in ids], capacity=width))
        creation = time.perf_counter() - started
        workspace = sum(lib.adapter_workspace(p.p) for p in ps)
        memory = used()

        def once():
            statuses = []
            for offset in range(0, max(p.count for p in ps), width):
                active = [(p, min(width, p.count-offset)) for p in ps if offset < p.count]
                for p, count in active:
                    p.begin(np.arange(offset, offset+count), mode)
                for p, count in active:
                    statuses.append(p.finish(count))
            return np.concatenate(statuses)

        once(); join()
        for p in ps:
            p.reset()
        trials = []
        for _ in range(3):
            started = time.perf_counter()
            statuses = once()
            returned = time.perf_counter()
            join()
            completed = time.perf_counter()
            assert len(statuses) == 640
            assert np.all(statuses['reason'] == (4 if mode == 0 else 0))
            trials.append(dict(host_return_sec=returned-started,
                               completion_tail_sec=completed-returned,
                               completed_sec=completed-started))
            for p in ps:
                p.reset()
        median = sorted(t['completed_sec'] for t in trials)[1]
        report['cases'][name] = dict(trials=trials, median_completed_sec=median,
                                    completed_components_per_sec=640/median,
                                    context_creation_sec=creation,
                                    structured_workspace_bytes=workspace,
                                    device_used_bytes=memory)
        print(name, json.dumps(report['cases'][name]), flush=True)
    finally:
        for p in ps:
            p.close()
    assert R.snapshot(lib) == before, 'resource closure failed'
    out.write_text(json.dumps(report, indent=2)+'\n')
