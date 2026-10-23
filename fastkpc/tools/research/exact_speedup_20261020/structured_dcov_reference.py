"""Exact structured top-k eigenpairs of the raw 1-D distance matrix used by
legacy kpcalg::dcov.gamma, checked against dense eigh on an adversarial
corpus.  Reference/validation code for
fastkpc/docs/PHASE10_EXACT_SPEEDUP_FINDINGS.md; not production code.

Per component x (n samples, k = numCol):
  1. sort; distinct values v, counts c, gaps h, cumulative counts t
  2. L_Q = closed-form Cholesky of the Brownian-bridge precision Q;
     B = diag(h)^-1/2 L_Q (lower bidiagonal)
  3. Demmel-Kahan bidiagonal QR (LAPACK dbdsqr): singular values sigma with
     high relative accuracy and right singular vectors P.  Production would
     record the right rotations and replay them onto k vectors; forming P
     explicitly applies exactly the same rotations.
  4. HDH eigenpairs: a = -2 / sigma^2, U = (H F L_Q) P (orthonormal)
  5. raw D in the basis [e, U] is the arrowhead [[alpha, b'], [b, diag(a)]];
     top-k |lambda| from a secular solve in nearest-pole offset coordinates
     with dlaed2-style deflation
  6. eigenvectors y = b / (lambda - a) in P coordinates, v = e + O P y

Run from anywhere:  python3 -I structured_dcov_reference.py [--bench]
"""
import argparse
import ctypes
import sys
import time

import numpy as np
import scipy.linalg.cython_lapack as cython_lapack
from scipy.special import gammainc


def _lapack_fn(name, nargs):
    capsule = cython_lapack.__pyx_capi__[name]
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    get_ptr.restype = ctypes.c_void_p
    get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * nargs))(
        get_ptr(capsule, get_name(capsule)))


_DBDSQR = _lapack_fn("dbdsqr", 15)


def dbdsqr_apply(d, e, vt):
    """Lower-bidiagonal B (diagonal d, subdiagonal e).  Returns the singular
    values (descending) and P^T @ vt, where B = Q diag(sigma) P^T."""
    m = len(d)
    d = np.array(d, dtype=np.float64, order="F")
    e = np.array(e, dtype=np.float64, order="F")
    vt = np.array(vt, dtype=np.float64, order="F")
    ncvt = vt.shape[1]
    dummy = np.zeros(1)
    work = np.zeros(4 * m)
    info = ctypes.c_int(0)

    def ip(value):
        return ctypes.byref(ctypes.c_int(value))

    _DBDSQR(ctypes.c_char_p(b"L"), ip(m), ip(ncvt), ip(0), ip(0),
            d.ctypes.data, e.ctypes.data, vt.ctypes.data, ip(m),
            dummy.ctypes.data, ip(1), dummy.ctypes.data, ip(1),
            work.ctypes.data, ctypes.byref(info))
    if info.value != 0:
        raise RuntimeError(f"dbdsqr info={info.value}")
    return d, vt


def secular_root(alpha, a, b2, i_lo, i_hi):
    """Root of f(l) = alpha - l - sum(b2 / (a - l)) in (a[i_lo], a[i_hi]);
    None marks an outer interval.  Returns (origin, delta) with
    lambda = a[origin] + delta and origin the nearest pole, so that
    lambda - a_j = (a[origin] - a_j) + delta stays accurate within an ulp of
    the pole (dlaed4 idea; bisection instead of rational interpolation)."""
    def f(origin, delta):
        return alpha - (a[origin] + delta) - np.sum(b2 / ((a - a[origin]) - delta))

    span = np.sqrt(b2.sum()) + abs(alpha) + 1.0
    if i_lo is None:
        origin = i_hi
        lo, hi = -2.0 * (abs(a[origin]) + span), 0.0
    elif i_hi is None:
        origin = i_lo
        lo, hi = 0.0, 2.0 * (abs(a[origin]) + span)
    else:
        half = 0.5 * (a[i_hi] - a[i_lo])
        if f(i_lo, half) > 0:
            origin = i_hi
            lo, hi = -half, 0.0
        else:
            origin = i_lo
            lo, hi = 0.0, half
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if f(origin, mid) > 0:
            lo = mid
        else:
            hi = mid
    return origin, 0.5 * (lo + hi)


def structured_topk(x, k):
    """Top-k (by |lambda|) eigenpairs of D = |x_i - x_j| without forming D.
    Returns (values, vectors, mean(D)); vectors are not centered."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    v, cnt = np.unique(xs, return_counts=True)
    m = len(v)
    if m < 2:
        raise ValueError("constant input: the distance matrix is zero")
    h = np.diff(v)
    t = np.cumsum(cnt).astype(np.float64)
    c = cnt.astype(np.float64)
    ld = np.sqrt(t[1:] / (t[:-1] * c[1:]))            # diag(L_Q)
    le = -np.sqrt(t[:-2] / (t[1:-1] * c[1:-1]))        # subdiag(L_Q)
    gi = 1.0 / np.sqrt(h)
    group = np.repeat(np.arange(m), cnt)

    # Row sums of D from gaps: positive terms only, translation invariant.
    g = np.diff(xs)
    kpos = np.arange(1, n, dtype=np.float64)
    rs = (np.concatenate([[0.0], np.cumsum(kpos * g)]) +
          np.concatenate([np.cumsum(((n - kpos) * g)[::-1])[::-1], [0.0]]))
    alpha = rs.sum() / n                               # e'De, e = 1/sqrt(n)
    s = rs / np.sqrt(n)                                # De

    # q = O^T s with O = H F L_Q  (F^T: cumulative sums over distinct values)
    ft = np.cumsum(np.bincount(group, weights=s - s.mean(), minlength=m))[:-1]
    q = ld * ft
    q[:-1] += le * ft[1:]

    sigma, pt = dbdsqr_apply(gi * ld, gi[1:] * le, np.eye(m - 1))
    bt = pt @ q                                        # b = P^T q
    a = -2.0 / sigma**2                                # eigenvalues of HDH
    idx = np.argsort(a)                                # most negative first
    a, bt, pt = a[idx], bt[idx], pt[idx]

    eps = np.finfo(np.float64).eps
    scale = max(abs(alpha), np.abs(a).max(), np.linalg.norm(bt))
    deflated = np.abs(bt) <= 8.0 * eps * scale
    live = np.flatnonzero(~deflated)
    al, bl = a[live], bt[live]
    b2 = bl**2
    kk = min(k, m)
    candidates = []
    if len(al):
        candidates.append(("root",) + secular_root(alpha, al, b2, len(al) - 1, None))
        candidates.append(("root",) + secular_root(alpha, al, b2, None, 0))
        for i in range(min(kk, len(al) - 1)):
            candidates.append(("root",) + secular_root(alpha, al, b2, i, i + 1))
    else:
        candidates.append(("e", -1, 0.0))              # e itself is an eigenvector
    for j in np.flatnonzero(deflated)[: kk + 1]:
        candidates.append(("pole", j, 0.0))

    def eigenvalue(cand):
        if cand[0] == "root":
            return al[cand[1]] + cand[2]
        return a[cand[1]] if cand[0] == "pole" else alpha

    candidates.sort(key=lambda cand: -abs(eigenvalue(cand)))
    values = np.zeros(k)
    vectors = np.zeros((n, k))
    for col, cand in enumerate(candidates[:kk]):
        y = np.zeros(m - 1)
        if cand[0] == "root":
            origin, delta = cand[1], cand[2]
            y[live] = bl / ((al[origin] - al) + delta)
            y0 = 1.0
        elif cand[0] == "pole":
            y[cand[1]] = 1.0
            y0 = 0.0
        else:
            y0 = 1.0
        w = pt.T @ y                                   # P y
        lw = ld * w
        lw[1:] += le * w[:-1]                          # L_Q P y
        rc = np.concatenate([np.cumsum(lw[::-1])[::-1], [0.0]])
        vs = rc[group]
        vs -= vs.mean()                                # H F L_Q P y
        vs += y0 / np.sqrt(n)
        vs /= np.linalg.norm(vs)
        vectors[order, col] = vs
        values[col] = eigenvalue(cand)
    return values, vectors, alpha / n


def dense_topk(x, k):
    d = np.abs(x[:, None] - x[None, :])
    w, v = np.linalg.eigh(d)
    idx = np.argsort(-np.abs(w), kind="stable")[:k]
    return w[idx], v[:, idx], d.mean()


def centered(parts):
    values, vectors, dmean = parts
    return values, vectors - vectors.mean(axis=0, keepdims=True), dmean


def legacy_pvalue(px, py, n):
    """kpcalg::dcov.gamma statistic, moments and 1 - pgamma, from centered
    top-k parts."""
    lx, wx, mx = px
    ly, wy, my = py
    nv2 = np.sum(np.outer(lx, ly) * (wx.T @ wy) ** 2) / n
    tx = np.sum(np.outer(lx, lx) * (wx.T @ wx) ** 2)
    ty = np.sum(np.outer(ly, ly) * (wy.T @ wy) ** 2)
    mean = mx * my
    var = ((2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)) *
           tx * ty / n**4) * n**2
    shape, scale = mean**2 / var, var / mean
    return 1.0 - gammainc(shape, nv2 / scale)


def stress_cases(rng, n):
    base = rng.standard_normal(n)
    return {
        "normal": rng.standard_normal(n),
        "cauchy_heavy_tail": rng.standard_cauchy(n),
        "lognormal_skew": np.exp(2.5 * rng.standard_normal(n)),
        "exact_ties_50pct": np.round(rng.standard_normal(n), 1),
        "zero_inflated": np.where(rng.random(n) < 0.45, 0.0,
                                  rng.gamma(0.7, 2.0, n)),
        "near_ties_1e-12": (np.repeat(rng.standard_normal(n // 3 + 1), 3)[:n] +
                            1e-12 * rng.standard_normal(n)),
        "near_ties_1e-15rel": (np.repeat(rng.standard_normal(n // 2 + 1), 2)[:n] *
                               (1 + 1e-15 * rng.standard_normal(n))),
        "clustered_scales": np.concatenate([1e-6 * rng.standard_normal(n // 2),
                                            1e3 + rng.standard_normal(n - n // 2)]),
        "uniform_grid": np.linspace(-1, 1, n),
        "outlier_1e8": np.concatenate([rng.standard_normal(n - 1), [1e8]]),
        "large_offset_1e6": 1e6 + rng.standard_normal(n),
        "dependent_on_base": base**2 + 0.3 * rng.standard_normal(n),
        "base": base,
        "binary_unequal": (rng.random(n) < 0.3).astype(np.float64),
        "binary_half_split": (np.arange(n) >= n // 2).astype(np.float64),
        "three_values": rng.choice([-1.0, 0.0, 2.5], n),
    }


def run_stress():
    rng = np.random.default_rng(20261020)
    worst = 0.0
    for n in (351, 1000):
        k = n // 10
        cases = stress_cases(rng, n)
        names = list(cases)
        dense = {name: dense_topk(cases[name], k) for name in names}
        fast = {name: structured_topk(cases[name], k) for name in names}
        print(f"n={n} k={k}")
        print(f"  {'case':22s} {'top-k eigval rel err':>21s} {'centered operator rel err':>26s}")
        for name in names:
            # compare the nonzero eigenvalues in value order: equal-|lambda|
            # pairs (two-valued inputs give +-h*sqrt(c1*c2)) have no |lambda| order
            dv, fv = dense[name][0], fast[name][0]
            dnz = np.sort(dv[np.abs(dv) > 1e-12 * np.abs(dv).max()])
            fnz = np.sort(fv[np.abs(fv) > 1e-12 * np.abs(fv).max()])
            rel = (np.max(np.abs(dnz - fnz) / np.abs(dnz))
                   if len(dnz) == len(fnz) else np.inf)
            dc, fc = centered(dense[name]), centered(fast[name])
            hd = dc[1] @ (dc[0][:, None] * dc[1].T)
            hf = fc[1] @ (fc[0][:, None] * fc[1].T)
            op = np.linalg.norm(hd - hf) / np.linalg.norm(hd)
            print(f"  {name:22s} {rel:21.2e} {op:26.2e}")
        max_dp, worst_pair = 0.0, None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pd = legacy_pvalue(centered(dense[names[i]]),
                                   centered(dense[names[j]]), n)
                pf = legacy_pvalue(centered(fast[names[i]]),
                                   centered(fast[names[j]]), n)
                if abs(pd - pf) > max_dp:
                    max_dp, worst_pair = abs(pd - pf), (names[i], names[j])
        worst = max(worst, max_dp)
        pairs = len(names) * (len(names) - 1) // 2
        print(f"  max |p_structured - p_dense| over {pairs} pairs = {max_dp:.3e}"
              f"  ({worst_pair[0]} x {worst_pair[1]})")
    print(f"WORST |dp| = {worst:.3e}  (numerical_contract_v1 p-value bound 1e-10)")
    return worst


def run_bench(reps=20, k=35):
    """Single-thread CPU core cost: dense eigh(D) versus bidiagonal QR that
    applies P^T to k vectors (same flops as replaying the rotations)."""
    rng = np.random.default_rng(1)
    for n in (351, 500, 1000):
        x = rng.standard_normal(n)
        d = np.abs(x[:, None] - x[None, :])
        start = time.perf_counter()
        for _ in range(reps):
            np.linalg.eigh(d)
        dense = (time.perf_counter() - start) / reps
        h = np.diff(np.sort(x))
        t = np.arange(1, n + 1, dtype=np.float64)
        ld = np.sqrt(t[1:] / t[:-1])
        le = -np.sqrt(t[:-2] / t[1:-1])
        gi = 1.0 / np.sqrt(h)
        timings = {}
        for ncvt in (0, 1, k):
            block = rng.standard_normal((n - 1, max(ncvt, 1)))[:, :ncvt]
            start = time.perf_counter()
            for _ in range(reps):
                dbdsqr_apply(gi * ld, gi[1:] * le, block)
            timings[ncvt] = (time.perf_counter() - start) / reps
        print(f"n={n}: dense eigh {dense * 1e3:7.2f} ms | bidiagonal QR values "
              f"{timings[0] * 1e3:6.2f} ms, +P^T q {timings[1] * 1e3:6.2f} ms, "
              f"+{k} vectors {timings[k] * 1e3:6.2f} ms | ratio {dense / timings[k]:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true",
                        help="also time dense eigh vs bidiagonal QR on one CPU thread")
    args = parser.parse_args()
    worst = run_stress()
    if args.bench:
        run_bench()
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    sys.exit(main())
