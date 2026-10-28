# L3 replacement coexistence: split raw capacity contract

This is a new, predeclared diagnostic protocol. The original 1.5 GiB raw-total
experiment remains `STOP_L3_REPLACEMENT_POOL_INCREMENT`. Thresholds below were
chosen after its observations; this campaign does not reinterpret old receipts.

Only canonical 351 x 48, default-Inf, exact-primary S2 is qualified: L3 n=351,
q=28, three penalties, total root rank=24, original cohorts of 2--33 targets,
32 capacity-33 slots, no decomposition trace. Outside this envelope the slot
diagnostic fails closed. Original target identity/order, optimizer arithmetic,
real coefficient arena, Stage A cohorts and all downstream gates are unchanged.

Capacity uses raw cudaMemGetInfo observations, without stack subtraction:

* Baseline: after the real coefficient arena reserve, before pool construction.
* Creation increment <= 1.5 GiB, before any L3 optimizer launch.
* At optimizer completion, growth from creation-complete <= 1.25 GiB.
* At optimizer completion, total increment from original baseline <= 2.75 GiB.
* Entire L3 optimizer and Stage A: every checkpoint and 10 ms sample has
  free >= 2 GiB. This is sampled coverage, not a continuous-peak guarantee.
* Stage A raw total increment is reported, but has no 2.75 GiB ceiling.
* W/S minimum-free comparison is descriptive, not a noninferiority gate.

No new warmup, stack-limit change, baseline reset, extra full-layer GPU output
arena, CPU numerical fallback, or production/default modification is allowed.
L4/L6 retain their original 1 GiB guards. Shared retained component cache,
coefficient arena and all original consumer resources remain in scope.

Run fresh W then S on exclusive GPU1, allowing other tasks on GPU0. Monitor both
devices and preserve any failed/interfered attempt. Host sharing and diagnostic
output copies preclude performance claims. Saved CPU authority is only an offline
comparison; diagnostic D2H shadows are separately counted, never CI authority.

W must pass every original canonical gate and reproduce the frozen Stage A
binary. S must match every captured optimizer output and all original Stage A
cohorts, real arena gather, coefficient/fitted/residual bits, route/rank/info,
all 240,498 CI/pMax bits, task/deletion/skeleton/sepsets/count/RNG, authority,
physical cohorts/Gram order, and resource closure. No new final-PDAG or real-data
qualification is claimed. The new split guards also receive CPU-only boundary
tests; these are policy tests, not numerical algorithm execution.

PASS permits only `GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT`. Runtime
integration and isolated full-method performance remain subsequent work. The
accepted 368.851 s median and Phase 10/candidate/holdout/recommended states do not
change. No commit/push is part of this campaign.

```
python3 fastkpc/tools/research/optimizer_l3_coexistence_v2_20261027/campaign.py prepare ARTIFACT/attempt1
python3 fastkpc/tools/research/optimizer_l3_coexistence_v2_20261027/campaign.py run ARTIFACT/attempt1 W
python3 fastkpc/tools/research/optimizer_l3_coexistence_v2_20261027/campaign.py run ARTIFACT/attempt1 S
python3 fastkpc/tools/research/optimizer_l3_coexistence_v2_20261027/close.py ARTIFACT/attempt1
```
