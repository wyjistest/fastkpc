# Fixed optimizer-window concurrency diagnostic

This diagnostic uses the existing production cross-setup scheduler and native
`8e457e811eaa01ae85326b8ff3cc0bcfa56f5b907ad6840a302a8205370d3874`.
It changes no production source, numerical kernel, default, or runtime flag.

The preselected contiguous windows are L3 W31/W32 and L5 W7/W8 from the
canonical 351x48 S2 route. Each contains exactly 64 setups. An external
interposer records their actual prepared geometry, input Y, ordered target
keys, and complete optimizer control. After the original production call it
downloads diagnostic snapshots of the selected state, coefficients, fitted
values and residuals. The capture must reproduce the qualified S2 CI prefix
through L5 bitwise; capture elapsed and diagnostic downloads are not production
performance or authority evidence.

Each replay calls `multi_penalty_gcv_capacity_optimize_multi` in the unchanged
native, changing only its requested worker count. Setup membership, target
cohorts, numeric input and final selected evaluation are fixed. Prepared handles
are created once, then reused with output leases released between repetitions.
Creation and destruction are measured separately. No allocation or lifecycle
optimization is implemented.

The primary timing sequence is 64/32/16, 32/16/64, 16/64/32, after one 64-worker
warmup. Each result must match the captured production output bitwise, including
all exported floating optimizer fields, coefficients/fitted/residuals, ranks,
statuses, iteration counts and trajectory counters. Diagnostic shadow D2H,
comparison and serialization happen after the timer. Later observed runs add
host timestamps, thread IDs and CUDA events without inserting per-setup host
waits. The added events and profiler overhead are excluded from primary timing.

CUDA event spans include queueing and submission gaps. Nsight kernel interval
coverage measures whether any kernel is executing, not SM utilization. A sum
of active launch grids is only an upper bound on resident blocks. Host waits
summed across workers are never treated as removable wall time.

Run from the repository root with a fresh output directory. The scripts require
both GPUs to have no compute jobs before each experiment; only GPU0 is used.

```sh
python3 -I fastkpc/tools/research/optimizer_windows_20261022/build.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_windows_20261022/campaign.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_windows_20261022/build.py OUTPUT replay
python3 -I fastkpc/tools/research/optimizer_windows_20261022/campaign.py OUTPUT replay
python3 -I fastkpc/tools/research/optimizer_windows_20261022/campaign.py OUTPUT profile
python3 -I fastkpc/tools/research/optimizer_windows_20261022/campaign.py OUTPUT memcheck
python3 -I fastkpc/tools/research/optimizer_windows_20261022/analyze.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_windows_20261022/plot.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_windows_20261022/archive.py OUTPUT
```

The profiler observes one L3 and one L5 window at all three worker counts.
Memcheck uses the first two setups of these windows; its reduced shape is not
performance evidence. This experiment does not qualify complete runtime
integration, new datasets or a changed numerical contract. L2 Stage A and
optimizer/component overlap remain outside its scope.
