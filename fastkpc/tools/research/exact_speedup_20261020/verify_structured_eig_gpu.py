"""GPU component qualification; CPU code is an offline oracle only.

python3 -I .../verify_structured_eig_gpu.py LIBRARY [--n 351 1000] [--output JSON]
The library never enters the production route and exposes explicit shadows.
"""
import argparse
import ctypes
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np

spec = importlib.util.spec_from_file_location(
    "structured_reference", Path(__file__).with_name("structured_dcov_reference.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


class GpuComponents:
    def __init__(self, library, n, k, count, capacity=None):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.structured_eig_create.restype = ctypes.c_void_p
        self.lib.structured_eig_create.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p]
        self.lib.structured_eig_run.restype = ctypes.c_int
        self.lib.structured_eig_run.argtypes = [ctypes.c_void_p] * 8
        self.lib.structured_eig_destroy.argtypes = [ctypes.c_void_p]
        self.error = ctypes.create_string_buffer(512)
        self.n, self.k, self.count = n, k, count
        self.handle = self.lib.structured_eig_create(
            n, k, count, capacity or 8*n*n, self.error)
        if not self.handle:
            raise RuntimeError(self.error.value.decode())

    def close(self):
        if self.handle:
            self.lib.structured_eig_destroy(self.handle)
            self.handle = None

    def run(self, inputs, expect_failure=False):
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        assert inputs.shape == (self.count, self.n)
        values = np.full((self.count, self.k), np.nan)
        vectors = np.full((self.count, self.k, self.n), np.nan)
        means = np.full(self.count, np.nan)
        stages = np.zeros(5)
        diagnostics = np.zeros((self.count, 4), dtype=np.int32)
        started = time.perf_counter()
        status = self.lib.structured_eig_run(
            self.handle, inputs.ctypes.data, values.ctypes.data, vectors.ctypes.data,
            means.ctypes.data, stages.ctypes.data, diagnostics.ctypes.data, self.error)
        elapsed = time.perf_counter()-started
        if expect_failure:
            assert status != 0 and np.isnan(values).all() and np.isnan(vectors).all()
            return diagnostics.tolist()
        if status:
            raise RuntimeError(f"{self.error.value.decode()}: {diagnostics.tolist()}")
        parts = [(values[j].copy(), vectors[j].T.copy(), means[j]) for j in range(self.count)]
        return parts, stages, diagnostics, elapsed


def pair_quantities(x, y, n):
    lx, wx, mx = x
    ly, wy, my = y
    nv = np.sum(np.outer(lx, ly)*(wx.T @ wy)**2)/n
    tx = np.sum(np.outer(lx, lx)*(wx.T @ wx)**2)
    ty = np.sum(np.outer(ly, ly)*(wy.T @ wy)**2)
    mu = mx*my
    var = ((2*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3))*tx*ty/n**4)*n**2
    return np.array([nv, mu, var, ref.legacy_pvalue(x,y,n)])


def qualify(library, n, cases, k=35):
    names = list(cases)
    inputs = np.array(list(cases.values()))
    gpu = GpuComponents(library, n, k, len(cases))
    try:
        parts, stages, diagnostics, elapsed = gpu.run(inputs)
        repeated, stages2, diagnostics2, _ = gpu.run(inputs)
        deterministic = all(all(np.array_equal(a,b) for a,b in zip(p,q))
                            for p,q in zip(parts,repeated)) and np.array_equal(diagnostics,diagnostics2)
    finally:
        gpu.close()
    dense = [ref.dense_topk(x,k) for x in inputs]
    centered = [ref.centered(p) for p in parts]
    centered_dense = [ref.centered(p) for p in dense]
    operators = []
    for fast, baseline in zip(centered,centered_dense):
        actual = fast[1] @ (fast[0][:,None]*fast[1].T)
        expected = baseline[1] @ (baseline[0][:,None]*baseline[1].T)
        operators.append(float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
    diffs = []
    passes = []
    flips = 0
    for i,j in itertools.combinations(range(len(cases)),2):
        actual = pair_quantities(centered[i],centered[j],n)
        expected = pair_quantities(centered_dense[i],centered_dense[j],n)
        error = np.abs(actual-expected)
        tolerance = np.maximum([1e-9,1e-10,1e-10,1e-10],
                               np.abs(expected)*[1e-8,1e-8,1e-8,0])
        diffs.append(error)
        passes.append(bool(np.isfinite(actual).all() and (error<=tolerance).all()))
        flips += int((actual[-1]>=0.1)!=(expected[-1]>=0.1))
    result = dict(n=n,k=k,components=len(cases),names=names,deterministic=deterministic,
                  diagnostic_columns=["info","distinct_values","rotations","deflated"],
                  diagnostics=diagnostics.tolist(),
                  stage_names=["sort_prepare","bidiagonal_qr_log","secular","vector_replay"],
                  stage_ms=stages[:4].tolist(),steady_stage_ms=stages2[:4].tolist(),
                  device_workspace_bytes=int(stages[4]),host_run_with_shadows_sec=elapsed,
                  operator_relative_errors=operators,pair_count=len(diffs),
                  max_stat_mean_var_p_abs_error=np.max(diffs,axis=0).tolist(),
                  alpha_flips=flips,pass_=bool(deterministic and all(passes) and flips==0))
    print(json.dumps(result),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--n", nargs="+",type=int,default=[351,1000])
    parser.add_argument("--output")
    parser.add_argument("--fixture", help="TSV: one component per column, no header")
    args = parser.parse_args()
    rng = np.random.default_rng(20261020)
    results = [qualify(args.library,n,ref.stress_cases(rng,n)) for n in args.n]
    if args.fixture:
        x = np.loadtxt(args.fixture,delimiter="\t")
        # Bound peak rotation-log memory; cross-batch pairs are a separate gate.
        for start in range(0,x.shape[1],16):
            cases={f"component{j}":x[:,j] for j in range(start,min(start+16,x.shape[1]))}
            results.append(qualify(args.library,x.shape[0],cases))
    failures={}
    for name,x,capacity in [("constant",np.ones((1,351)),None),
                            ("nonfinite",np.full((1,351),np.nan),None),
                            ("rotation_capacity",rng.standard_normal((1,351)),1)]:
        gpu=GpuComponents(args.library,351,35,1,capacity)
        try: failures[name]=gpu.run(x,expect_failure=True)
        finally: gpu.close()
    # k=1 exactly cuts the equal-|lambda| pair: the positive root must win.
    gpu=GpuComponents(args.library,1000,1,1)
    try:
        parts,_,_,_=gpu.run((np.arange(1000)>=500).astype(float)[None,:])
        assert parts[0][0][0]>0 and np.linalg.norm(parts[0][1]-parts[0][1].mean())<1e-12
    finally: gpu.close()
    report=dict(schema="structured-eig-gpu-isolated-v1",results=results,
                fail_closed=failures,tie_break_pass=True,production_integrated=False,
                full_contract_qualified=False)
    if args.output:
        path=Path(args.output)
        with path.open("x") as f: json.dump(report,f,indent=2)
    if not all(r["pass_"] for r in results):
        raise SystemExit("STOP: isolated numerical gates failed")


if __name__=="__main__":
    main()
