# Strict CI methods 351x48 optimized evidence v1

This directory indexes the first production optimization pass for the
default-Inf 351x48 `hsic.gamma`, `dcc.perm`, and `hsic.perm` full-CUDA
skeleton routes.

The implementation adds two exact-shape optimizations:

```text
fixed-SP residuals:
  bounded one-call device slab
  any miss -> execute the complete original SVD cohort
  all hit  -> D2D gather in original target order

HSIC components:
  parallel independent row work
  unchanged pivot, solve, mean, and reduction order
```

Measured outer elapsed times on the same reference RTX 4090:

```text
method       previous     optimized    speedup
hsic.gamma   1562.052 s    979.870 s     1.594x
dcc.perm     2131.261 s   1455.935 s     1.464x
hsic.perm    3877.316 s   1875.133 s     2.068x
```

All established strict contracts pass: 834,467 skeleton p-values, logical and
deletion traces, graph outputs, permutation RNG terminal state, 19 kpcalg CPU
WAN-PDAG orientation CI tests, and final PDAGs. The three runs have zero CPU
skeleton numerical authority, fallback, residual/component D2H, and cache
eviction.

`manifest.json` is tracked. Dense candidate and WAN-PDAG RDS payloads remain
under `payload/` locally and are ignored by Git; the manifest records their
sizes and SHA-256 hashes. These are development receipts, not promotion or
sealed-holdout evidence. Phase 10 remains active and the recommended route is
unchanged.
