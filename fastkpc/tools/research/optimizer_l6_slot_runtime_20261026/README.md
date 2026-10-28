# L6 bounded-slot runtime qualification

First pass the separately declared replacement coexistence probe on the frozen
parent. Only then add a default-off canonical S2 runtime envelope: n=351, q=55,
six penalties, total root rank48, original target counts2/3/5, 32 capacity-5
slots and one original-order queue. Keep properties cache, L4/L5 slots,
status-wait, exact-primary coefficient reuse and the retained full-eig stack
enabled in both comparison modes. No L8 preload or GEMM warmup is added.

The slot API must preserve numerical instructions, cohorts, selected-SP and
optimizer trajectories. Stage A must materialize the same selected coefficients
in original target/cohort order. Require generation/lease safety, rejection of
invalid geometry, partial-upload and detach failure cleanup, cancellation/drain,
zero worker resource mutations, zero leaked resources and zero-error memcheck.
Compare Stage A output against the frozen parent's actual finalization, not
against an assumed interchangeable optimizer residual. Cover every L6 target
capacity and the final request. Retain L4/L5 configuration regression checks.

Runtime memory gates remain >=2 GiB joint free and <=1 GiB warm pool increment,
checked during creation, after work, throughout Stage A and before close. Original
window handles are replaced; the pool ends before Stage B. Unsupported opt-ins
fail closed. The option remains off by default even after development GO.

After those gates, run six full canonical fresh-process executions in fixed
order OFF1/ON1/ON2/OFF2/OFF3/ON3, same frozen binary, GPU1 exclusive, both GPUs
otherwise idle, CPU affinity0-19 and one BLAS/OMP/MKL thread. No trace or shadow
D2H in timings; persistence and offline authority checks are outside outer time.
Require all240,498 CI and pMax bits identical, task/deletion/skeleton/sepset/count/
RNG exact, physical cohort and Gram order exact, unchanged other-layer settings,
GPU-only numerical authority, no fallback, no production large D2H, resources
closed, and all finite authority errors <=1e-10.

Predeclared performance GO: ON/OFF outer median <=0.98, L6 optimizer median
ratio <=0.90, and all three paired outer wins. A smaller positive or inconsistent
gain is CONDITIONAL; no median gain or any numerical/resource failure is STOP.
Keep all attempts. External interference invalidates only the interrupted
incomplete attempt, never a completed unfavorable timing. No selected repeats.

If GPU0 becomes occupied before isolated timing, the optional
`campaign.py OUT --shared-host-qualification` runs exactly one OFF/ON pair on
exclusive GPU1 for full-method correctness and resource checks. It writes separate
`shared-*` artifacts, explicitly labels the shared host, and never supplies a
performance decision or contributes timings to the six isolated runs. Freeze
this distinction before either supplemental run; the isolated gate is unchanged.

Evidence is audited incremental source closure, not a clean producer, new real
data/final-PDAG campaign or promotion. Phase10 ACTIVE, candidate NOT_FROZEN,
holdout SEALED and recommended=false remain unchanged. L8 requires a future
separately declared experiment after reviewing the resulting full-method profile.
