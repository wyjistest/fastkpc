# L3 bounded-slot runtime qualification

Prerequisite: frozen L3 coexistence v2 `GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT`.
Add a default-off `FASTKPC_DCC_GAMMA_L3_OPTIMIZER_SLOTS` opt-in on canonical S2:
n=351, q=28, three penalties, total root rank24, original unpadded cohorts2–33,
32 capacity-33 slots, original-order queue and Stage A. Keep L4/L5/L6 slots,
properties cache, status wait, exact-primary coefficient reuse and retained
full-eig configuration ON. Unsupported opt-ins fail closed. No numerical
algorithm changes, preload, warmup, stack-limit change or extra layer arena.

The baseline is after the actual coefficient arena reserve and before pool
creation. Raw creation increment before any L3 launch must be <=1.5 GiB.
Optimizer completion growth from creation must be <=1.25 GiB, and total
increment from that original baseline <=2.75 GiB. Free >=2 GiB at creation,
optimizer completion, each Stage A setup entry and before pool close. Stage A
raw total is recorded without a total-increment ceiling. No subtraction or
baseline reset. These are checkpoints, not continuous peak coverage; no sampler
thread is added. Existing L4/L6 1 GiB and 2 GiB guards are unchanged.

Before timing, require captured optimizer/Stage A bitwise parity for all actual
cohort capacities plus the final request, unchanged device instructions,
generation/lease isolation, partial creation/upload/detach failure cleanup,
worker cancellation/drain, zero worker resource mutations, zero leaked resources,
zero-error memcheck, and configuration/capacity boundary rejection tests.
Stage A authority is the frozen parent binary's original finalization, not an
assumption that optimizer residuals are interchangeable. CPU comparison is
offline saved authority only; GPU numerical authority and no fallback remain.

Freeze one native and tools before six fresh-process canonical runs in order
OFF1/ON1/ON2/OFF2/OFF3/ON3. GPU1 exclusive, both GPUs otherwise idle, CPU affinity
0–19, one BLAS/OMP/MKL thread, CUDA connections32. No diagnostic shadow or
decomposition trace. Persistence and offline authority checks are outside outer
elapsed. Full240,498 CI/pMax bitwise, task/deletion/skeleton/sepset/count/RNG,
physical cohort/Gram order, all-layer resource and saved authority <=1e-10 gates
must pass. No final-PDAG/real-data campaign is claimed from skeleton-only checks.

Predeclared performance GO: ON/OFF outer median <=0.98, L3 optimizer median
ratio <=0.90, and all three paired outer wins. Smaller positive/inconsistent
gain is CONDITIONAL; no median gain or any numerical/resource failure is STOP.
Keep every completed result, including unfavorable ones, and every interrupted
attempt. Other users' jobs are never terminated. Shared-host qualification, if
needed, is separately labeled and excluded from performance acceptance.

Report L3 optimizer, L3 remainder and outside-L3 separately. Pool creation has
its own timer, within total slot preparation; the old prepared-build counter
does not include it and cannot establish net creation savings. Downstream free
comparisons are descriptive; absolute headroom is the hard gate.

Evidence is incremental source closure, not a clean producer or promotion.
Phase10 ACTIVE, candidate NOT_FROZEN, holdout SEALED, recommended=false remain.
The opt-in stays default-off even if this development campaign passes.
