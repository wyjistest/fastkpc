"""Real captured residual replay. GPU eig; CPU pair arithmetic is oracle-only.

The 11 GiB reservation is touched GPU memory, NOT a live production cache.
Its purpose is bounded-memory co-residency qualification, not pipeline timing.
"""
import argparse
import csv
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from scipy.special import gammainc

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("structured_gpu", HERE.parent /
    "exact_speedup_20261020/verify_structured_eig_gpu.py")
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_bits(parts):
    h = hashlib.sha256()
    for part in parts:
        for item in part:
            h.update(np.ascontiguousarray(item, dtype=np.float64).tobytes())
    return h.hexdigest()


class Reference:
    def __init__(self, path):
        self.lib = ct.CDLL(str(Path(path).resolve()))
        self.error = ct.create_string_buffer(512)
        self.lib.dense_create.argtypes = [ct.c_int]*3+[ct.c_void_p]*2
        self.lib.dense_create.restype = ct.c_void_p
        self.lib.dense_run.argtypes = [ct.c_void_p]*5
        self.lib.dense_run.restype = ct.c_int
        self.lib.dense_destroy.argtypes = [ct.c_void_p]
        self.lib.reserve_memory.argtypes = [ct.c_size_t, ct.c_void_p]
        self.lib.reserve_memory.restype = ct.c_void_p
        self.lib.release_memory.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
        self.lib.memory_info.argtypes = [ct.POINTER(ct.c_size_t)]*2

    def memory(self):
        free, total = ct.c_size_t(), ct.c_size_t()
        assert self.lib.memory_info(ct.byref(free), ct.byref(total)) == 0
        return dict(free=free.value, total=total.value, used=total.value-free.value)

    def reserve(self, size):
        ptr = self.lib.reserve_memory(size, self.error)
        if not ptr:
            raise RuntimeError(self.error.value.decode())
        return ptr

    def release(self, ptr, size):
        if self.lib.release_memory(ptr, size, self.error):
            raise RuntimeError(self.error.value.decode())

    def dense(self, x, trials):
        count, n = x.shape
        k = 35
        ctx = self.lib.dense_create(n, k, count, x.ctypes.data, self.error)
        if not ctx:
            raise RuntimeError(self.error.value.decode())
        values = np.empty((count,k))
        vectors = np.empty((count,k,n))
        stats = np.zeros(4)
        rows = []
        try:
            for iteration in range(trials+1):
                if self.lib.dense_run(ctx,values.ctypes.data,vectors.ctypes.data,
                                      stats.ctypes.data,self.error):
                    raise RuntimeError(self.error.value.decode())
                parts = [(values[c].copy(),vectors[c].T.copy(),
                          float(np.abs(x[c,:,None]-x[c,None,:]).mean()))
                         for c in range(count)]
                signature = output_bits(parts)
                if iteration == 0:
                    baseline = signature
                else:
                    assert signature == baseline, "dense repeat bits changed"
                    rows.append(stats.copy().tolist())
            memory = self.memory()
            return parts, dict(trials=rows, device_median_ms=statistics.median(r[0] for r in rows),
                diagnostic_wall_median_ms=statistics.median(r[2] for r in rows),
                explicit_device_bytes=int(stats[3]), device_memory=memory,
                bitwise_deterministic=True, output_sha256=baseline)
        finally:
            self.lib.dense_destroy(ctx)


def pair_values(parts, pairs, n):
    centered = []
    for values, vectors, mean in parts:
        w = vectors - vectors.mean(axis=0, keepdims=True)
        self_moment = float(np.sum(np.outer(values,values)*(w.T@w)**2))
        centered.append((values,w,mean,self_moment))
    out = []
    for i,j in pairs:
        a,b = centered[i],centered[j]
        statistic = np.sum(np.outer(a[0],b[0])*(a[1].T@b[1])**2)/n
        mean = a[2]*b[2]
        variance = ((2*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3))*a[3]*b[3]/n**4)*n**2
        p = 1.0-gammainc(mean**2/variance, statistic/(variance/mean))
        out.append([statistic,mean,variance,p])
    return np.asarray(out)


def compare(actual, expected):
    error = np.abs(actual-expected)
    rel = error/np.maximum(np.abs(expected),1e-300)
    gate = np.isfinite(actual).all(axis=1) & (error[:,:3] <= [1e-9,1e-10,1e-10]).all(axis=1)
    gate &= (rel[:,:3] <= 1e-8).all(axis=1) & (error[:,3] <= 1e-10)
    flips = (actual[:,3]>=.1) != (expected[:,3]>=.1)
    return dict(pairs=len(gate), pass_=bool(gate.all() and not flips.any()),
        failures=int((~gate).sum()), alpha_flips=int(flips.sum()),
        max_absolute_error=error.max(axis=0).tolist(),
        max_relative_error=rel.max(axis=0).tolist(),
        failed_indices=np.flatnonzero(~gate | flips).tolist())


def structured(library, x, capacity, trials, reference):
    contexts = []
    count,n = x.shape
    for start in range(0,count,64):
        contexts.append(gpu.GpuComponents(library,n,35,min(64,count-start),capacity))
    try:
        xs = [np.ascontiguousarray(x[s:s+64]) for s in range(0,count,64)]
        baseline, diagnostics = [], []
        work_bytes = 0
        for c,chunk in zip(contexts,xs):
            parts,stages,d,_ = c.run(chunk)
            baseline.extend(parts)
            diagnostics.extend(d.tolist())
            work_bytes += int(stages[4])
        signature = output_bits(baseline)
        walls, stage_rows = [], []
        # Match the earlier thread-start/join boundary; propagate worker errors.
        for _ in range(trials):
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=len(contexts)) as pool:
                futures = [pool.submit(c.run,chunk) for c,chunk in zip(contexts,xs)]
                runs = [f.result() for f in futures]
            walls.append(time.perf_counter()-started)
            actual = [p for run in runs for p in run[0]]
            assert output_bits(actual) == signature, "structured repeat bits changed"
            assert [d.tolist() for run in runs for d in run[2]] == diagnostics
            stage_rows.append([run[1][:4].tolist() for run in runs])
        diag = np.array(diagnostics)
        return baseline,dict(walls_s=walls, wall_median_s=statistics.median(walls),
            throughput=count/statistics.median(walls), stage_ms_by_trial_context=stage_rows,
            explicit_device_bytes=work_bytes, device_memory=reference.memory(),
            rotations_min=int(diag[:,2].min()), rotations_max=int(diag[:,2].max()),
            rotations_mean=float(diag[:,2].mean()), distinct_min=int(diag[:,1].min()),
            deflated_total=int(diag[:,3].sum()), diagnostics=diagnostics,
            bitwise_deterministic=True, output_sha256=signature)
    finally:
        for context in contexts:
            context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--trials",type=int,default=3)
    args = parser.parse_args()
    work = args.directory
    meta = json.loads((work/"capture.json").read_text())
    x = np.fromfile(work/"residuals.f64",dtype="<f8").reshape(meta["components"],meta["n"])
    assert sha(work/"residuals.f64") == meta["residual_sha256"]
    pair_rows = list(csv.DictReader((work/"pairs.csv").open()))
    pairs = [(int(r["component_x"]),int(r["component_y"])) for r in pair_rows]
    authority = np.array([[float(r[f]) for f in ("statistic","mean","variance","authority_p")]
                          for r in pair_rows])
    original_p = np.array([float(r["original_gpu_p"]) for r in pair_rows])
    ref = Reference(work/"dense-reference.so")
    library = work/"structured-eig.so"
    # Frozen from the earlier independent synthetic campaign, not tuned on
    # these residuals. Any overflow aborts this campaign instead of truncating.
    capacity = 170662
    report = dict(schema="structured-real-residual-replay-v1", production_integrated=False,
        n=meta["n"],k=35, rotation_capacity=capacity,trials=args.trials,
        memory_load_kind="11 GiB cudaMalloc + full cudaMemset; no production cache traffic",
        numerical_boundary="GPU eig outputs; offline pair/gamma validation, not full GPU method qualification",
        timing_boundary="dense eig-only device vs structured wall including H2D, diagnostic D2H and thread launch/join",
        environment={k:os.environ.get(k,"") for k in ("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS",
                        "OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,uuid,driver_version,memory.total",
                                     "--format=csv,noheader"],text=True),rows=[])
    # Explicitly recheck overflow fail-closed before the real run.
    bad = gpu.GpuComponents(library,meta["n"],35,1,1)
    try:
        report["overflow_failure_diagnostics"] = bad.run(x[:1],expect_failure=True)
    finally:
        bad.close()
    baselines = {}
    for count in (256,640):
        for reserve in (0,11*2**30):
            before = ref.memory()
            ptr = ref.reserve(reserve) if reserve else None
            try:
                loaded = ref.memory()
                # Alternate order to avoid always favouring the second solver.
                if reserve:
                    fast,st = structured(library,x[:count],capacity,args.trials,ref)
                    dense,dt = ref.dense(x[:count],args.trials)
                else:
                    dense,dt = ref.dense(x[:count],args.trials)
                    fast,st = structured(library,x[:count],capacity,args.trials,ref)
                signatures = (st["output_sha256"],dt["output_sha256"])
                if not reserve:
                    baselines[count] = signatures
                else:
                    assert signatures == baselines[count], "memory pressure changed eig output bits"
                actual = pair_values(fast,pairs[:count//2],meta["n"])
                expected = pair_values(dense,pairs[:count//2],meta["n"])
                dense_gate = compare(actual,expected)
                authority_gate = compare(actual,authority[:count//2])
                native_error = np.abs(actual[:,3]-original_p[:count//2])
                # Cross-cohort diagnostic pairs exercise combinations beyond
                # the 320 real same-S CI tasks; they are not graph trace rows.
                rng = np.random.default_rng(20261020)
                cross = rng.integers(0,count,size=(2048,2))
                cross = cross[cross[:,0]!=cross[:,1]]
                cross_gate = compare(pair_values(fast,cross,meta["n"]),
                                     pair_values(dense,cross,meta["n"]))
                row = dict(components=count,reserved_bytes=reserve,dense=dt,structured=st,
                    ratio=st["wall_median_s"]/(dt["device_median_ms"]/1000),
                    versus_gpu_dense=dense_gate,versus_kpcalg=authority_gate,
                    cross_cohort_vs_dense=cross_gate,max_error_original_gpu_p=float(native_error.max()),
                    near_alpha_005_015_count=int(((authority[:count//2,3]>=.05)&
                                                 (authority[:count//2,3]<=.15)).sum()),
                    memory_before=before,memory_with_reservation=loaded,
                    observed_used_bytes_max=max(loaded["used"],dt["device_memory"]["used"],st["device_memory"]["used"]),
                    # This is sampled at context/stage boundaries; vendor
                    # transient allocations can exceed the observed maximum.
                    memory_peak_kind="boundary-sampled; excludes unobserved vendor transient allocations")
                report["rows"].append(row)
                np.savez(work/f"outputs-{count}-{reserve}.npz",
                         structured_pair=actual,dense_pair=expected)
            finally:
                if ptr:
                    ref.release(ptr,reserve)
            row["memory_after_release"] = ref.memory()
            row["all_numerical_gates_pass"] = bool(dense_gate["pass_"] and authority_gate["pass_"]
                and cross_gate["pass_"] and (native_error<=1e-10).all())
            (work/"benchmark.json").write_text(json.dumps(report,indent=2))
            print(f"components={count} reserve={reserve/2**30:g}GiB structured={st['wall_median_s']:.6f}s "
                  f"dense={dt['device_median_ms']/1000:.6f}s ratio={row['ratio']:.3f} "
                  f"p_error={authority_gate['max_absolute_error'][3]:.3g} pass={row['all_numerical_gates_pass']}",flush=True)
            if not row["all_numerical_gates_pass"]:
                raise SystemExit("STOP: real residual numerical gate failed")
    report["pass"] = all(r["all_numerical_gates_pass"] for r in report["rows"])
    report["source_sha256"] = {str(p):sha(p) for p in [Path(__file__),HERE/"dense_reference.cu",
        HERE.parent/"exact_speedup_20261020/structured_eig_gpu.cu",
        HERE.parent/"exact_speedup_20261020/verify_structured_eig_gpu.py",
        HERE.parent.parent/"phase10_cuda_full_eig_lane_benchmark.cu"]}
    report["artifact_sha256"] = {p.name:sha(p) for p in [work/"capture.json",work/"pairs.csv",
        work/"residuals.f64",library,work/"dense-reference.so"]}
    (work/"benchmark.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
