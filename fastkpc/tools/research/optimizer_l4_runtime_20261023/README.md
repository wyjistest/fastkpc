# L4 original-window worker runtime qualification

Default-off `FASTKPC_DCC_GAMMA_L4_OPTIMIZER_WORKERS32=1` caps only the
canonical S2 L4 optimizer workers at 32. All 34 windows (33 x 64 + 42),
2,154 prepared setups and 17,767 original target requests remain unchanged.
There is no slot pool, cross-window work, new CUDA resource or numerical edit.
Both comparison modes keep the qualified L5 slots enabled.

Predeclared gates, before any new GPU run:

- Only `full_cuda_ci_one_call.cpp` may differ from the prior L5 full-build
  source closure. Rebuild that translation unit and relink in a source snapshot
  using hash-verified prior full-build objects. The build script also rebuilds
  structured eig; verify its normalized device code, all retained objects and
  generated DBDSQR. This is an audited incremental source-closure build, not a
  new full build or clean producer. Freeze sources/native during qualification.
- Reject invalid flags and unqualified configuration without resource leaks.
  The existing L4 diagnostic covers all 14 target capacities and reduced
  memcheck; unchanged CUDA objects bind that numerical/lifetime evidence.
- Six fresh-process canonical runs: OFF/ON/ON/OFF/OFF/ON, three per mode.
  All 240,498 CI and pMax bits, task/deletion/skeleton/sepset/count/RNG,
  physical cohort and Gram order must equal the qualified L5 result and
  same-native baseline. All CI/pMax authority errors <=1e-10, zero CPU
  numerical authority/fallback/large D2H, unchanged exact-primary and cache
  route/status ledgers, all application resources closed.
- Every original window and cohort remains exact. L4 requested/effective
  worker counts must be 64 (42 in final window) OFF, 32 ON. Other-layer
  worker counts must be unchanged. Peak host calls are not GPU overlap.
- GO only if ON outer median <=0.99 OFF, all three adjacent pairs improve,
  and L4 optimizer host median <=0.95 OFF. Correct but smaller/inconsistent
  outer gains are CONDITIONAL; no L4 improvement or outer regression is STOP.
  Any hard-gate failure stops the campaign; never relax a gate after results.
- Record joint production memory; L5 pool retains its >=2 GiB free-memory
  gate. No OOM and no unexpected process on GPU1. Existing GPU0 work is preserved.

GPU1, CPU affinity 0-19, CUDA connections=32, one BLAS/OMP/MKL thread,
11 GiB cache, residual bypass, eight eig lanes, status-wait=1, L5 slots=1.
No inter-window shadow verification or profiling injection in timed runs.
Artifact writes/compression and offline authority checks are outside elapsed.
Evidence is GPU1 shared-host, uncommitted source-closure development, not
isolated performance or promotion. Defaults stay off. No new final PDAG or
real-data qualification. Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED,
recommended=false. No commit or push is part of this task.

Run from the repository root into a fresh output directory:

```sh
python3 -I fastkpc/tools/research/optimizer_l4_runtime_20261023/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_l4_runtime_20261023/checks.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_l4_runtime_20261023/campaign.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_l4_runtime_20261023/archive.py OUTPUT
```
