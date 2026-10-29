# Isolated recover tile qualification

Use current runtime 1d8a11d, native
12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d,
and the sealed L2/L3 capture: 49,919 + 54,878 components, original cohorts,
78/86 waves, ten contexts, width 64, n=351, numCol=35. Work-pair is ON.
Production source, generated includes, native and defaults remain unchanged.

The only numerical execution change in the timing libraries is c->tile:
36 (unchanged control), 16, or 8. All use 64 threads/block. Each column retains
its rotation order, scalar arithmetic, normalization, output and status logic.
Compile the same current source with -O3 -arch=sm_89 --fmad=false. Three
uninstrumented interposers use the production consumer/cache/wave implementation.
One unchanged-native replay authenticates the compiled tile36 control before
timing. Host counters authenticate mode, tile, creates/destroys and submits.

GPU qualification compares tile36 with each candidate on all captured inputs:
d/e/q, rotation records/order, metadata, candidates, values/vectors/means and
status. Reuse the sealed GPU comparator and DBDSQR branch instrumentation.
Add full-pipeline repeated/signed-zero/extreme/precheck cases, column seams,
rotation overflow and injected recover nonfinite failure. Failed output is
not a numerical result; injected late failures must report info=50 and reject.
Guard every diagnostic allocation with 256-byte sentinels. Run reduced CUDA
memcheck before complete qualification; sanitizer execution is not timed.
Diagnostic instrumentation and injected failures are absent from timing code.

Harness amendment after attempts1/2: wait for default-stream allocation
initialization before submitting to a nonblocking test stream. The prior
harness had no such dependency and failed at fresh plans, including k=1 where
both libraries actually use tile2. Preserve both failed attempts; neither
reached timing. Use RAII lane ownership on all exception paths. Reduced
memcheck covers selected column/target seams and 16 captured components;
complete branch and full-capture coverage remain mandatory in qualification.

GPU1 exclusive; GPU0 also free of compute work; CPU affinity 0-19, one math
thread, CUDA connections=32, 11 GiB cache, eight dense lanes. An external GPU
job stops only this campaign's process, preserving incomplete evidence.

Predeclare nine fresh-process timings in a balanced order:
A1 B1 C1, B2 C2 A2, C3 A3 B3; A=36, B=16, C=8.
All three variants use the same interposition and host-counter mechanism.
All 104,797 sealed three-field hashes, fallback=0, cohort order and resource
closure must pass on every replay. Payload D2H/hashing stays outside Stage B.
No CUPTI injection, diagnostic export or sanitizer enters these medians.

Evaluate each candidate independently: both level medians must improve, their
sum must decrease >=5%, and every candidate total must beat every A total.
GO only recommends a later runtime experiment. Positive but insufficient/noisy
benefit is CONDITIONAL; no net gain or any correctness/resource failure is STOP.
Do not select one noisy run or combine candidate improvements.

After timings, obtain one separate activity replay per tile with the same
sealed CUPTI recorder. Report roots-to-recover gaps, bidiagonal/recover union
and intersection, recover-only activity, post-bidiagonal recover and complete
wave tail, recover duration, and no-recorded-device-activity intervals. These
are explanations, not occupancy measurements, memory-traffic counters or a
savings budget. Hardware counters remain unmeasured; no privilege changes.

Scope is captured L2/L3 only, excluding optimizer workspaces/CI consumers.
No full-runtime capacity or new p-value/graph/RNG/outer qualification follows.
Accepted opt-in development median remains 311.440 s; defaults off, L8 STOP,
Phase10 ACTIVE, candidate NOT_FROZEN, holdout SEALED, recommended=false.

Run build.py OUTPUT, campaign.py OUTPUT, analyze.py OUTPUT, archive.py OUTPUT.
Build and protocol identities are frozen before GPU work. Never overwrite a
previous attempt. No commit/push or production integration is part of this task.
