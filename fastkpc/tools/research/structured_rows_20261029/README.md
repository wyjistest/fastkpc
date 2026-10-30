# Structured row-only capacity experiment (predeclared)

Parent: `build_consistency_20261029/closure.json` must first report
`PASS_BUILD_FIX_PARENT`. No q55 expansion, new lanes/slots or recover-tile
changes are in scope. Structured remains behind its existing opt-in gate.

Only remove the structured lane's unused distance matrix allocation/store.
Keep fabs, per-thread accumulation, the shared reduction tree, totals, solver,
payload, fallback and publication arithmetic/order. The dense kernel and its
matrix allocation remain intact.

Required GPU gates:

- Bitwise rows/totals with full and short batches and repeated lane reuse.
- 640 archived component payloads; original target order and Gram lifecycle.
- Mixed 56 dense GPU fallbacks among 70 components, chunked 48+8; fallback
  failure publishes no wave and releases resources. Also rotation overflow
  and destruction with abandoned submission.
- Memcheck and complete resource closure; diagnostic D2H stays in the test.
- Every canonical CI/pMax bit, task/deletion/graph/sepset/count/RNG and authority
  gate, original cohorts, all accepted lanes/slots and postback configuration.

Unit/fallback/sanitizer qualification may use exclusive GPU1 while unrelated
GPU0 work is recorded. That mode supplies no performance evidence. The full
OFF/ON timing campaign below still requires both GPUs free of other compute.

Use two natives differing only in the rebuilt consumer TU, with the qualified
build fix in both. Run fresh processes and caches on physical GPU1, both GPUs
otherwise free, fixed affinity 0-19 and one math-library thread. Sequence:
OFF1, ON1, ON2, OFF2, OFF3, ON3. Persist artifacts outside the skeleton timer;
retain all valid repetitions including L7/L8 variation.

Declare three separate verdicts before inspecting timings:

1. Numerical/behavior PASS requires all gates; no tolerance relaxation.
2. Capacity PASS requires solver workspace unchanged, measured structured
   pool reduction of exactly 630,789,120 bytes (601.5673828125 MiB) in every
   level, paired L3/L4/L6 minimum-free checkpoints each improving by at least
   600 MiB, and all original capacity and lifecycle gates. Report raw checkpoints
   and 200-ms external samples separately; neither is an instantaneous peak proof.
3. Outer non-regression requires ON/OFF median <= 1.02 and every paired ratio
   <= 1.05. These are predeclared practical noise allowances, not a proof of
   zero slowdown. A speed GO additionally requires median <= 0.98 and all three
   pairs faster. Capacity may be accepted without speed GO if 1/2 and the
   non-regression gate pass. Otherwise retain the parent and report STOP.

Freed memory stays as headroom. No new dataset envelope, final PDAG, promotion,
candidate freeze or holdout release is claimed. Historical 290.491 seconds keeps
its original native identity; this experiment establishes its own OFF/ON median.

After the build-fix parent qualifies and the row-only patch is applied, run:

```sh
python3 fastkpc/tools/research/structured_rows_20261029/build.py
python3 fastkpc/tools/research/structured_rows_20261029/qualify.py
python3 fastkpc/tools/research/structured_rows_20261029/campaign.py
python3 fastkpc/tools/research/structured_rows_20261029/archive.py audit
python3 fastkpc/tools/research/structured_rows_20261029/archive.py install
python3 fastkpc/tools/research/structured_rows_20261029/archive.py seal
```

The scripts require the local authenticated parent, canonical authority,
dataset and archived structured fixtures. Output directories must be fresh;
missing inputs or a failed gate stop the campaign. They do not report SKIP as
success. The install step is gated on all declared acceptance checks and keeps
the prior native/objects in the frozen parent archive.
