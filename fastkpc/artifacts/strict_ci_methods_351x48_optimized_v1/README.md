# Strict CI methods 351x48 optimized evidence v1

This directory indexes the current production optimization pass for the
default-Inf 351x48 `hsic.gamma`, `dcc.perm`, and `hsic.perm` full-CUDA
skeleton routes.

The retained implementation combines these strict-shape optimizations:

```text
fixed-SP residuals:
  bounded one-call device slab
  any miss -> execute the complete original cohort
  all hit  -> D2D gather in original target order
  <=2 penalties -> qualified augmented QR
  >=3 penalties -> authoritative stable SVD

persistent method context:
  reuse CUDA streams, events, buffers, and method workspaces

HSIC components and hsic.perm:
  parallel independent row work
  unchanged pivot, solve, mean, and reduction order
  bounded 384 MiB deterministic component LRU

permutation host path:
  exact inline R Rejection index generation for hsic.perm
  OpenSSL SHA-256 authentication without payload string copies
```

Measured outer elapsed times on the same reference RTX 4090:

```text
method       previous     optimized    speedup
hsic.gamma   1562.052 s    631.522 s     2.473x
dcc.perm     2131.261 s    945.359 s     2.254x
hsic.perm    3877.316 s   1152.905 s     3.363x
```

All established strict contracts pass: 834,467 skeleton p-values, logical and
deletion traces, graph outputs, permutation RNG terminal state, 19 kpcalg CPU
WAN-PDAG orientation CI tests, and final PDAGs. The three runs have zero CPU
skeleton numerical authority, fallback, residual/component D2H, and residual
cache eviction. The bounded `hsic.perm` component LRU has authenticated hit,
miss, transfer, and deterministic-eviction accounting.

`manifest.json` is tracked. Dense candidate and WAN-PDAG RDS payloads remain
under `payload/` locally and are ignored by Git; the manifest records their
sizes and SHA-256 hashes. These are development receipts, not promotion or
sealed-holdout evidence. Phase 10 remains active and the recommended route is
unchanged.
