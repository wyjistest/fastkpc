# Current-native work-pair OFF/ON Stage B activity diagnostic

Diagnostic only. Bind committed runtime 1d8a11d and native
12acc3cd01cce2dcac5f2bfb5ec02f27056e92154cbab2534c7f03bdc22caa7d,
all 104 production inputs and both generated DBDSQR includes. Compile only
host observers/replay against this native; no CUDA numerical source changes.

Reuse the sealed stage_b_attribution_20261028/attempt3/capture L2/L3 inputs,
original ordered keys/cohorts/route/rank/info and all 104,797 component hashes.
Authenticate the inputs against the old manifest, not a newly invented digest.
Reuse the original observer, replay input loop and CUPTI activity recorder.
Add host counters to prove the selected mode actually executes: 10 plans,
1,638 submits, 104,797 targets and 10 destroyed plans, with other-mode zeros.

Run exactly two fresh processes, OFF then ON, GPU1 exclusive with both GPUs
otherwise idle, CPU affinity 0-19, one math thread, CUDA connections=32,
11 GiB cache, eight dense eig lanes, ten structured contexts of width 64.
Preserve 78/86 waves. Stop only this diagnostic if external GPU work appears;
preserve incomplete attempts. Protocol/build/tool identities precede GPU work.

Both runs use concurrent CUPTI activities; these timings are instrumented
diagnostics, not a fresh performance campaign. Require every three-field
payload hash to match the original native, zero dense fallback, closed tracked
resources, zero dropped/invalid activities and exact launch correlations.
Observer payload D2H/hashing is explicit and outside each Stage B boundary.
There is no CPU numerical producer. Optimizer workspaces and CI consumers are
absent, so this replay does not qualify full-runtime capacity or outer speed.

Report full-level/wave boundaries, kernel-family interval unions, overlap,
exclusive recorded activity and no-recorded-device-activity gaps. For normal
waves (excluding first and last), report bidiagonal launch start spreads and
the count starting within 50 ms of the first bidiagonal start. Pair kernels by
stream within their original wave. Record roots-end to recover-start gaps and
API timing separately; neither proves a hardware scheduling cause. Separate
recover activity after the last bidiagonal completion from the entire trailing
wave boundary, which also contains roots/status/stores and other work.

Resource limits are static upper bounds. Activity overlap does not measure SM
occupancy, and family durations or concurrent lane sums are not additive
critical-path savings. No NCU permission changes, tile sweep, recover rewrite,
new warmup, lane/wave/cache change or L8 experiment is authorized here. The
result only recommends whether a separate captured recover prototype merits
testing. Default-off 311.440-second development evidence and Phase 10/candidate/
holdout/recommendation states remain unchanged.

Run build.py OUTPUT, campaign.py OUTPUT, analyze.py OUTPUT, archive.py OUTPUT.
Use a new output directory for every attempt. Sealed historical tools and
artifacts are read-only dependencies.
