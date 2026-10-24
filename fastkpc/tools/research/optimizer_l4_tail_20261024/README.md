# L4 original-window tail attribution

Predeclared scope, 2026-10-24: profile the qualified 32-worker L4 original
windows using the unchanged native bound to runtime commit `33e2d3e`.
Reuse all 34 captured windows, 2,154 setups and 17,767 targets, including
all fourteen actual cohort sizes (2 through 18, not every integer).
No numerical kernel clone, production edit, new scheduler or slot prototype.

Run two independent GPU1 processes. Each does one complete unprofiled warmup
and one complete profiled pass. Keep CPU affinity 0-19, connections=32,
status-wait=1, and one BLAS/OMP/MKL thread. Record both GPUs' process and
memory telemetry; never interrupt an external job. Stop our process if an
unrelated compute process appears on GPU1. Other-GPU jobs invalidate an
isolated-host claim, but are allowed in this development diagnostic.

Each original window creates only its own prepared objects, calls the actual
production optimizer with 32 workers, validates every output, releases every
lease and destroys its objects. Require at least 2 GiB physical headroom.
NVTX setup ranges add no CUDA events or stream synchronization. CUPTI API
correlation supplies device intervals. Resource interposition must observe
zero allocation/stream/event/BLAS mutations inside optimizer calls. All
application-visible allocations, bytes, streams, events and handles must
close. Verify all optimizer states, selected SP, route/rank/info, trajectory
counters, coefficients, fitted and residual bits against the archived capture.
GPU-only authority checks use the existing snapshot validator.

Report separately: native host boundaries; prepared lifecycle; packaging;
shadow verification; output lease release; device kernels; H2D/D2H bytes and
intervals; per-setup host API waits; API-to-device delays; window drain tails
after the last setup starts; and no-kernel gaps. Summed worker/API durations
are not wall-time savings. A setup's first-to-last-kernel span is observed
service time, not pure compute or a concurrency-invariant property. Shadow
verification interrupts windows, so segmented native totals and the profile
process duration are not canonical outer elapsed. No production cache/arena
is resident, no new p-value/full-method/real-data/PDAG qualification is claimed.

Use an original-order, 32-slot fixed-service-time model to compare the same
34 window barriers with continuous refill. Compute both a device-span model
and a complete host-setup-span model; the latter retains observed upload,
metadata/wait and return cost. Check each model against measured native/kernel
boundaries before using its projection. Also cross-check the two captures.
Do not duration-sort, split/pad targets, use a partial layer, or equate a
model projection with measured overlap or net performance.

Screening rule fixed before profiling: all coverage/parity/resource gates
must pass; the complete-host-span window model must reproduce segmented
native time within 5% in both traces. A continuous-refill projection saving
at least 10% of measured L4 native time in both traces supports a separate
bounded-slot experiment (`GO_L4_BOUNDED_REFILL_EXPERIMENT_ONLY`). A smaller
projection is `STOP_L4_REFILL_OPPORTUNITY_TOO_SMALL`; failed model fit or
conflicting traces is `CONDITIONAL_MODEL_NOT_VALIDATED`. Numerical/resource
failures are STOP. Report projected milliseconds per setup available for new
slot upload/detach/lease costs; these costs are not yet measured, so even a
GO does not qualify an implementation or promise outer-time savings.

Before any runtime integration, a separate prototype must cover all 14
cohort sizes, bounded geometry upload and output detach, stale-token/failure
cleanup, production cache/arena coexistence and >=2 GiB headroom. Require
actual full-lifecycle speedup and then full canonical outer-time/semantic
qualification. Existing `STOP_FULL_LAYER_CAPACITY` remains in force.

From the repository root, using a fresh OUTPUT:

```sh
python3 -I fastkpc/tools/research/optimizer_l4_tail_20261024/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_l4_tail_20261024/run.py OUTPUT 1
python3 -I fastkpc/tools/research/optimizer_l4_tail_20261024/run.py OUTPUT 2
python3 -I fastkpc/tools/research/optimizer_l4_tail_20261024/analyze.py OUTPUT
```

Build/runner pin the native, all 98 production inputs, original capture
manifest and fixture hashes, and execution tools before GPU execution.
Analysis and final archive are separately hashed. Datasets, binaries and
large receipts stay in local artifact storage. Phase 10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED, recommended=false; production defaults unchanged.
