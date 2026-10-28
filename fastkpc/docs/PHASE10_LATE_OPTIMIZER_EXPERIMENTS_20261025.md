# Late optimizer experiments, 2026-10-25–26

Final statuses: **STOP_L8_PRELOAD_PERFORMANCE** and
**GO_CAPTURED_L6_SLOTS**. Capture, trace and all six L8 full canonical hard
gates PASS; all L6 parity, lifecycle, capacity, memcheck and six-pair gates
PASS. Both experiments are GPU1-exclusive, shared-host development evidence.
The original isolated device-properties pair remains pending.

## Scope and identity

The authorized sequence is the remaining isolated device-properties-cache
OFF3/ON3 pair, targeted L8 preload with that cache ON, then an original-order
L6 bounded-slot captured prototype. GPU0 remains occupied by another user's
training job, PID 1422670. Independent preparation and exclusive-GPU1 runs
therefore proceed as explicitly shared-host development evidence. They do not
complete the original isolated cache performance gate.

HEAD is `d27e59b`; current native SHA-256 is
`994c43d386c859f23c34509b857db22064fb9a009b8fe871c719dd608f820df9`.
These experiments do not rebuild or replace that native, change numerical
operations, alter target cohorts, or introduce CPU numerical authority.
Existing property-cache source changes and all earlier evidence are preserved.
This is the existing uncommitted incremental cache build, not a clean producer
run. No additional production source changes were made for these experiments.
Accepted full-method median remains **407.531 seconds**. Phase 10 ACTIVE,
candidate NOT_FROZEN, holdout SEALED, recommended=false.

## Original-order capture

An independent full canonical run saved all 14 floating and 19 integer output
vectors for each original request, including selected SP, coefficients,
fitted/residual, route/rank/info and optimizer trajectory counters.

| Level | Original windows | Setups | Targets | Iterations | q | Total root rank |
|---|---|---:|---:|---:|---:|---:|
| L6 | 64/64/64/53 | 245 | 1,148 | 17,062 | 55 | 48 |
| L8 | 9 | 9 | 18 | 378 | 73 | 64 |

L6 target capacities 2/3/5 occur in 7/28/210 setups. All L8 setups contain two
targets. Complete canonical gates pass: all 240,498 CI/pMax bits match the
cache-ON reference, maximum authority error is `3.05425129631942e-12`, and
task/skeleton/sepset/count/RNG, GPU authority and resource gates pass.
Capture includes 7,063,888 bytes of explicitly diagnostic shadow D2H;
its elapsed time is not performance evidence.

Artifact root: `fastkpc/artifacts/full_cuda_ci/late_optimizer_20261025`.
Capture closure SHA-256:
`ea55f3951c07aa34e8e51808cea7cfb2dc7c35931560ea1f30721ca850709f81`.

## L8 targeted preload

A host-only research interposer delegates all numerical calls to the existing
native. After the first actual L8 prepared creation and before its workers,
ON queries attributes of exactly five registered base80 kernel functions:
target norm, QR projection, optimizer, coefficient gather, residual formation.
OFF uses the identical interposer without the queries. Preload cost is inside
the regular outer timer. No global eager-loading policy or stack limit changes.

Separate OFF/ON traced runs both pass all canonical gates and each record
569,524 device kernels, with zero dropped CUPTI records. L8 keeps exactly
45 kernels with identical names/grid/block counts across nine streams.

| Trace measurement | OFF | ON |
|---|---:|---:|
| L8 optimizer host boundary, s | 16.651912 | 14.918208 |
| L8 device span, s | 16.632292 | 14.917029 |
| Summed L8 kernel busy time, s | 35.693114 | 35.658734 |
| Latest setup start relative to first, s | 1.703 | 0.016 |

The OFF first-use diagnostic records a 1.6857-second drain signature, with
16 later-launched kernels starting within 2 ms of the drain point. That L8
signature is absent in ON. ON preloading costs 36.826681 ms and 6 MiB, leaving
more than 2 GiB free. This supports removing the observed first-use stall;
it does not prove that every scheduling gap is lazy loading or HOL. Ordinary
same-stream waits for an optimizer's own result are not counted as avoidable
stall. The traced outer difference is not assigned entirely to L8 and is not
used in the timing gate.

The predeclared fresh-process timing sequence is
OFF1/ON1/ON2/OFF2/OFF3/ON3, without CUPTI or capture D2H. GO requires all hard
gates, three paired outer wins, median saving at least
`max(2 s, 0.5% of OFF median, OFF range, ON range)`, and at least 1 s lower
median L8 elapsed. Positive inconclusive results are CONDITIONAL. Shared-host
evidence cannot establish isolated runtime qualification or promotion.

All six declared fresh-process timings completed, in the fixed order below.
Every hard gate passes; all six preserve the native and have no capture D2H.
GPU0's external job is recorded throughout. Times are seconds; the small
outer-minus-level-sum remainder is also retained in the raw reports.

| Run | Outer | L0–L5 | L6–L8 | L8 |
|---|---:|---:|---:|---:|
| OFF1 | 389.146 | 316.970 | 70.355 | 16.912 |
| ON1 | 393.150 | 319.194 | 72.107 | 15.283 |
| ON2 | 397.108 | 320.126 | 75.160 | 15.304 |
| OFF2 | 401.697 | 319.470 | 80.392 | 18.741 |
| OFF3 | 392.890 | 317.642 | 73.373 | 18.624 |
| ON3 | 394.397 | 318.976 | 73.599 | 15.310 |

Outer OFF/ON medians are **392.890/394.397 seconds**, so ON is 1.507 seconds
slower (0.384%). Paired savings are -4.004/+4.589/-1.507 seconds: **one win
out of three**. The predeclared minimum saving is 12.551 seconds, set by the
OFF within-mode range; ON range is 3.958 seconds. L8 median does improve
18.624366 → 15.303958 seconds, a 3.320408-second local reduction.

Thus the mechanism is supported but the full-method performance gate fails.
Do not integrate this preload or advertise a method speedup from this campaign.
This is a STOP for the present experiment, not a claim that preloading can
never help. Earlier-layer variation predates the L8 intervention and cannot
be assigned to it; the measured shared-host regressions also do not prove a
causal slowdown. A future changed baseline would require a separate declared
qualification. No extra repetitions were selected after seeing the result.

Closure: `closure.json` under the artifact root above.
Manifest SHA-256:
`e3a8b3cea29d67a8cb31046aab0242167c93ec4125bbde5190bbcf6a17a4f36e`.

## L6 bounded-slot prototype

W runs the real production scheduler with the original four windows. S uses
32 reusable capacity-5 slots and an original-order request queue. Both retain
complete outputs into equivalent D2D arenas and verify them after timing.
The independently built hidden numerical clone matches all 14 production GPU
functions' normalized instruction words.

Before timing, the declared protocol requires full output/trajectory parity,
invalid/busy/stale-token and partial-failure cleanup tests, zero-error/leak
memcheck, and resource closure. Slot plus output-arena capacity must add at
most 1 GiB above a verified compute-warm production context and leave at least
2 GiB free. Standalone capacity does not qualify coexistence with production
component caches.

After qualification, one W/S warmup precedes six balanced W/S versus S/W pairs.
GO requires median full-lifecycle S/W <=0.90 and all six paired wins. Packing,
allocation, uploads, computation, detach and release are charged. Capture reads
and delayed output comparison are excluded. No full-method gain or runtime
integration is inferred from the model or this prototype alone.

All qualification gates passed. Four reduced original requests cover all
three target capacities plus the final request; ordinary checks include three
verified executions, ten negative checks and nine contract-test target
comparisons. Reduced memcheck reports **0 errors / 0 bytes leaked** and checks
the same partial-failure and cancellation paths. Across ordinary checks,
memcheck, capacity, warmups and timing, **17,310 target-output comparison
instances** pass, including each process's five-target native warmup. Worker
resource mutations are zero and all tracked resources close.

One full W/S warmup was followed by the six declared balanced pairs, with
every original setup and all 17,062 iterations retained in each execution.

| Full-lifecycle measurement | W: original windows | S: 32 slots |
|---|---:|---:|
| Median, s | 43.739627 | 18.040455 |
| Minimum, s | 42.324802 | 17.999531 |
| Maximum, s | 48.882878 | 18.119635 |
| Work-only median, s | 43.574878 | 18.009794 |

S/W full-lifecycle ratio is **0.412451**, with **six paired wins**: a
**58.75%** reduction (about **2.425x**) in this captured L6 boundary. W includes
the declared in-worker full-output retention observer. The experiment changes
both 64-worker window barriers and request refill into 32 reusable slots;
it does not identify the entire gain as pure HOL removal.

Ordinary native warmup retains **1,152 MiB** of common context memory, recorded
separately from slot capacity (the sanitizer has its own recorded context
overhead). The independent capacity process measures **472 MiB** incremental
physical memory, 155,482,976 application bytes, and 22,723,493,888 bytes still
free. It has 33 tracked allocations, 32 streams, 96 events and 32 BLAS handles.
Every timed S pass measures **462 MiB** incremental memory. The independent
capacity process's first full S execution is 30.121785 seconds; the campaign
W/S warmups are 42.1817/18.8215 seconds. These are preserved separately and
excluded from the declared six-pair medians; no unmeasured cause is assigned
to their difference.

Result: **GO_CAPTURED_L6_SLOTS**, with runtime integration **NOT_YET_GO**.
The next scoped step is production component-cache/arena coexistence and
bounded lifetime validation. Only then consider a separate opt-in runtime
qualification with full canonical CI/graph/RNG gates and repeated outer
timing. This prototype does not establish a new canonical elapsed time or
new real-data/final-PDAG qualification.

Final L6 artifact root:
`fastkpc/artifacts/full_cuda_ci/optimizer_l6_slots_20261025/qualified`.
The preliminary build and its tool snapshot remain in the parent directory;
the final host-only context-memory receipt addition preceded every GPU check.
All 14 device functions remain exact. Closure SHA-256:
`8462b981aa0cf9d9412e7d790a6cdb21b11e2d53f8ca0caf58097da28f8ab40a`.
Manifest SHA-256:
`0ff3cd25c96c7bdca6e0c2c554b44feb130ab843c20fd2851fa975f2676d0d51`.

## Pending isolation and handoff

At completion, GPU0 still hosts PID 1422670; no test process from this work
remains on GPU1. Do not combine these shared-host results with the original
four isolated cache runs to pass that gate. Its already prepared continuation
tree is `fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025_isolated_completion`.
Once both GPUs are idle, run the existing `resume.py`, `audit_continuation.py`
and `archive.py` from `tools/research/device_properties_cache_20261025` against
that tree. The original interrupted evidence and shared-host supplement stay
unchanged. Accepted median, defaults, Phase 10 and all promotion states remain
unchanged; no commit or push is part of this handoff.

Tools: `fastkpc/tools/research/late_optimizer_20261025`,
`fastkpc/tools/research/late_optimizer_analysis_20261025`, and
`fastkpc/tools/research/optimizer_l6_slots_20261025`.
