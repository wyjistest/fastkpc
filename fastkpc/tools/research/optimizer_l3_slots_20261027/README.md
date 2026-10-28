# L3 original-order captured-cohort slot experiment

Predeclared 2026-10-27. Research only; no production-source/default changes.
Current native is 313d9f3c…510a1b. Capture all 64 original L3 windows (63 × 64
+ 33 setups), 4,065 setups / 54,878 targets, n=351, q=28, three penalties.
Preserve actual target cohorts, their request order, controls and GPU arithmetic.
Do not extrapolate the approximately 35-second busy-time model to performance.

W uses the actual production multi-setup entrypoint and original window sizes.
S uses 32 independent reusable capacity-33 slots, an original-order atomic queue,
and the unchanged small64 numerical implementation. Compare every captured
floating/integer field, coefficients, fitted values, residuals, route/rank/info,
selected-SP and optimizer trajectory bits. Both arms retain identical full GPU
outputs to a device arena; diagnostic D2H/comparison is outside timing. Retention
cost is inside both arms. No CPU numerical execution or fallback is permitted.

Require all 14 device functions' normalized instruction words to match the
current production object. Before timing: reduced checks spanning every actual
target count and the final request, busy/invalid bind and stale-token rejection,
old-token destruction, partial-upload/detach and cancellation drain, resource
closure, no resource mutation during workers, zero-error/leak reduced memcheck.

Measure capacity relative to a compute-warm production context after verifying
one actual maximum-capacity native request. Keep that common context cost
separate. Pool + full diagnostic output arena must add <=2 GiB, with >=2 GiB
free at pre/post work checkpoints. The larger standalone allowance explicitly
includes the 33-target slots and 54,878-target full-output arena; it is NOT a
production capacity allowance. Record application bytes, pool/arena sizes,
physical increment and sampled GPU occupancy. Actual cache/arena replacement
coexistence remains a separate gate before any runtime integration.

After checks and capacity: one full W/S warmup, then six balanced W/S, S/W pairs.
GO requires median S/W full lifecycle <=0.90 and six paired wins with every hard
gate. Smaller positive/inconsistent improvement is CONDITIONAL; no median gain
or any numerical/resource failure is STOP. Lifecycle includes packing, creation,
uploads, computation, D2D retention, release and destruction; fixture reads and
delayed comparison are excluded. Report each setup's host service duration and
contention change. Do not use traced runs in performance medians.

Separate fresh-process W and S CUPTI traces measure launch-to-start waits,
device service times, first-use GEMM waits and same-/other-channel end alignment.
Include late-start statistics for slot mode, not just concurrent channel use.
The ordinary replay's common native warmup is omitted in these trace modes so
the isolated first-use cost remains visible; this is not a full-method context.

GPU1, affinity 0–19, connections=32, status wait and device-properties cache ON,
single BLAS/OMP/MKL thread. Require both GPUs free of external compute jobs;
monitor and preserve interrupted attempts. No sorting, L7/L8 experiments,
full-layer residency, production integration, commit or push in this task.
Accepted canonical development median stays 368.851 s. Phase10 ACTIVE,
candidate NOT_FROZEN, holdout SEALED and recommended=false remain unchanged.
