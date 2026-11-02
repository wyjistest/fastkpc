# Fixed-VT immediate rotation consumption: isolated 5N versus 3N

Protocol declared before GPU work, 2026-10-30. Parent source 1fb26aa; accepted
native b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b.
This directory does not modify production sources, includes, native or defaults.
The preceding binary audit is local evidence, not a candidate qualification.

A is the CURRENT fixed-VT work-pair, 5N / 14,040 bytes at N=351. C consumes
each sweep's c/s by value at the original work-store position, 3N / 8,424 bytes.
Original 7N is not the performance denominator. The candidate removes lower
conversion's unused stores, all work expressions and the work pointer decrement;
the ABI argument is retained but unused. No work pointer arithmetic on nullptr.
Fixed-entry rejection and remaining argument-error precedence remain unchanged.
The four sweep families retain signs, identity skip, old q operands and overflow
behavior. Direct 2x2 drot, sign/scal and swap keep their existing wrappers.
No q register cache, restrict changes, numerical helper changes, tolerances,
recover tile changes, extra lanes/slots/cache or optimizer expansion.

Build baseline_diag, candidate_diag, baseline_timing and candidate_timing with
-O3 -arch=sm_89 --fmad=false and the current numerical helpers. A/C timing
libraries use identical host interposition/counters. Diagnostic code, guards,
checkpoints and failure injection are absent from timing libraries. Authenticate
A against one unchanged-native L2/L3 replay before the formal six runs; neither
this native replay nor the A authentication replay enters formal medians.

GPU qualification compares d/e/q and control fields at lower-conversion end,
every COMPLETE sweep after threshold handling and q consumption, completed 2x2,
sign completion, sort completion and exits. Event keys use stage + independent
ordinal + sweep number, not log.count. A stores bounded device checkpoints;
C compares each integer and FP64 bit pattern on device against the same buffer.
Metadata and numerical fields are compared directly, never by hash. The limit
is 2,048 events/component, 64 targets/context, ten diagnostic contexts. Exhaustion
is a failed qualification, not truncation. Only compact comparison/status and
allocation guard results leave the GPU; no CPU numerical authority.
Whole stored rotation fields/order/count/overflow, post-DBDSQR state, root/recover
payload and status also use GPU byte comparison. Include constructed missing
branches, n=0/1/errors, capacity 0/1/count-1/count, forced nonconvergence, signed
zero/ties, target tails/reuse, late recover failure and shared/allocation guards.
Inherit the default-stream initialization join and exception-safe RAII cleanup.
Run a reduced compute-sanitizer memcheck before full qualification.

The input capture retains its original identity: stage_b_attribution_20261028,
attempt3/capture, L2 49,919 and L3 54,878 components, original order/cohorts.
Timing uses the CURRENT native consumer/cache/wave implementation with ten
contexts, width 64, 78/86 waves, recover tile36, 64 recover threads, 11 GiB cache,
8 dense eig lanes and work-pair mode 1. Stage B includes roots, recover, status,
joins and cache stores. Payload D2H/hash occurs after the elapsed boundary.
All 104,797 sealed payload hashes and fallback=0 must pass on EVERY replay.

Resource gate: timing A must match the accepted fixed kernel's 82 REG and zero
STACK/LOCAL. Record C REG and live ranges without an automatic 82-REG ceiling.
New nonzero STACK/LOCAL or local spill is STOP_RESOURCE for this first candidate;
any revised allowance requires a new protocol, not post-hoc acceptance. Verify
3N entry/shared binding and absence of candidate work accesses. Register-only
changes with zero spill proceed to complete-stage measurement.

GPU1 exclusive, GPU0 free of compute work; CPU affinity 0-19; one math-library
thread; CUDA connections=32. Record existing clock/power policy without changing
machine policy. An external compute task stops only this campaign process and
preserves incomplete evidence. Fresh process for every run. Freeze build,
protocol, tools, capture and native hashes before GPU work.

Formal timing order: A1 C1 C2 A2 A3 C3. Define M_A and M_C as the sum of the
independent L2 and L3 medians; separately report median of per-run sums.
GO_ISOLATED_STAGE_B requires all correctness/resource gates, both level medians
lower, M_C/M_A <= 0.95, and max(C run totals) < min(A run totals). Positive but
insufficient/noisy benefit is CONDITIONAL; no net gain is STOP_NO_STAGE_GAIN.
Any hard failure stops before further performance work. Do not tune thresholds
or select a favorable subset after seeing results.

Passing L2/L3 would only support later L4-L8 qualification and default-off runtime
work. It cannot update the 283.328 s accepted development median. L8 optimizer
STOP does not exempt shared DBDSQR from future L8 correctness checks. q55 remains
deferred; recover tile16/8 and L8 optimizer remain STOP. Phase10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED, recommended=false. No production integration or
commit/push is included here.
