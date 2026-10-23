"""Throughput of the structured-eig prototype at batch sizes beyond one
64-component context, versus the retained dense eig lane benchmark.

The prototype accepts at most 64 components per context, so larger batches
are emulated by running several contexts concurrently, one host thread and
CUDA stream each (ctypes releases the GIL).  Structured time is wall clock
from starting all threads to joining them: H2D, all four device stages, the
diagnostic D2H of every value/vector/mean, and host synchronization.  Dense
time is the lane benchmark's eig-only device median, which excludes distance
construction and centering.  Both choices favour dense.  Inputs are the same
synthetic series as benchmark_structured_eig.py, not real residuals.

Run from the repository root, with CUDA_DEVICE_MAX_CONNECTIONS=32 set:
  python3 -I fastkpc/tools/research/exact_speedup_20261020/benchmark_structured_eig_concurrency.py \
    STRUCTURED_LIBRARY DENSE_BENCHMARK OUTPUT_JSON [--totals 64,256,640,1280,2560]
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time

import numpy as np

spec = importlib.util.spec_from_file_location(
    "gpu", Path(__file__).with_name("verify_structured_eig_gpu.py"))
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)

PER_CONTEXT = 64


def inputs(n, offset, count):
    row = np.arange(1, n + 1, dtype=float)
    return np.array([np.sin(.017 * (c + 1) * row) + np.cos(.0031 * (c + 3) * row * row)
                     for c in range(offset, offset + count)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("library")
    p.add_argument("dense_benchmark")
    p.add_argument("output")
    p.add_argument("--totals", default="64,256,640,1280,2560")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--n", type=int, default=351)
    p.add_argument("--k", type=int, default=35)
    args = p.parse_args()
    n, k = args.n, args.k

    # Size the rotation log from one context at the default capacity; the
    # default 8*n*n entries per component does not fit many contexts in memory.
    probe = gpu.GpuComponents(args.library, n, k, PER_CONTEXT)
    try:
        _, stages, diag, _ = probe.run(inputs(n, 0, PER_CONTEXT))
    finally:
        probe.close()
    rotations = diag[:, 2]
    capacity = int(rotations.max() * 1.5) + 1024
    report = dict(n=n, k=k, trials=args.trials, per_context=PER_CONTEXT,
                  rotations_max=int(rotations.max()), rotations_mean=float(rotations.mean()),
                  rotation_capacity=capacity,
                  single_context_stage_ms=dict(zip(["prepare", "bidiagonal_qr", "roots", "recover"],
                                                   stages[:4].tolist())),
                  rows=[])
    print(f"rotations per component: max {report['rotations_max']}, mean "
          f"{report['rotations_mean']:.0f}; capacity {capacity}; single-context stage ms "
          + "/".join(f"{s:.1f}" for s in stages[:4]), flush=True)

    for total in (int(t) for t in args.totals.split(",")):
        if total % PER_CONTEXT:
            raise SystemExit(f"total {total} is not a multiple of {PER_CONTEXT}")
        count = total // PER_CONTEXT
        contexts = [gpu.GpuComponents(args.library, n, k, PER_CONTEXT, capacity)
                    for _ in range(count)]
        xs = [inputs(n, i * PER_CONTEXT, PER_CONTEXT) for i in range(count)]
        try:
            reference = []
            for c, x in zip(contexts, xs):
                parts, _, d, _ = c.run(x)
                if int(d[:, 2].max()) > capacity:
                    raise SystemExit("rotation capacity exceeded")
                reference.append(parts)
            walls = []
            for _ in range(args.trials):
                out = [None] * count

                def work(i):
                    out[i] = contexts[i].run(xs[i])[0]

                threads = [threading.Thread(target=work, args=(i,)) for i in range(count)]
                started = time.perf_counter()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                walls.append(time.perf_counter() - started)
                for got, ref in zip(out, reference):
                    assert all(all(np.array_equal(a, b) for a, b in zip(u, v))
                               for u, v in zip(got, ref)), "non-deterministic output"
        finally:
            for c in contexts:
                c.close()
        wall = statistics.median(walls)
        text = subprocess.check_output(
            [args.dense_benchmark, str(n), str(total), "8", str(args.trials)], text=True)
        record = dict(t.split("=", 1) for t in text.split() if "=" in t)
        assert record["eigenvalue_bit_mismatch_count"] == "0"
        assert record["eigenvector_bit_mismatch_count"] == "0"
        dense = float(record["concurrent_device_median_ms"]) / 1e3
        row = dict(components=total, contexts=count, structured_wall_s=wall,
                   structured_walls_s=walls, structured_per_s=total / wall,
                   dense_eig_only_s=dense, dense_per_s=total / dense, ratio=wall / dense)
        report["rows"].append(row)
        print(f"{total:5d} components ({count} x {PER_CONTEXT}): structured wall "
              f"{wall * 1e3:8.1f} ms ({row['structured_per_s']:5.0f}/s) | dense eig-only "
              f"{dense * 1e3:8.1f} ms ({row['dense_per_s']:4.0f}/s) | ratio {row['ratio']:.2f}",
              flush=True)
    Path(args.output).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
