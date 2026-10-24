# Whole-L5 optimizer refill diagnostic

Scope: unchanged status-wait native `ba8f806b...d9298f`, original L5 inputs,
955 setups / 6,816 targets / 15 windows (14×64+59). No runtime scheduler or
slot-pool integration. Wait is on for every variant. GPU1 only, CPU 0–19,
32 CUDA connections, one BLAS/OMP/MKL thread; GPU0 may host external work.
All timings are shared-host source-closure development evidence.

Capture reproduces the complete qualified S2 prefix through L5, including
every CI/pMax bit, task/deletion/sepset/graph/count/RNG and GPU authority/resource
gate. Diagnostic residual shadows are outside numerical production authority.
Fixtures preserve geometry, Y, optimizer controls, target identity/order,
complete optimizer state, coefficients, fitted and residual output bits.

Before timing, capacity mode creates every actual prepared object with at least
2 GiB of physical free-memory headroom. It records each exact application-visible
allocation increment, CUDA free-memory readings and resource counts. Any failure
stops the full-layer experiment; no smaller workload is silently substituted.
This replay has no 11 GiB component cache and is not runtime capacity evidence.

Three configurations:

* P / A-prod: original 64/64 windows, create and destroy handles per window.
* A / A-resident: original 64/64 windows, all 955 handles resident.
* R / R-resident: one original-order request list, 32 workers continuously
  acquire work, using the same resident handles as A in that cycle.

A/R measure the combined change of 64 to 32 workers and removal of original
window barriers; they do not isolate the contribution of each factor.

A/R use the same pool per cycle; it is destroyed before P when P comes last.
P never runs with a spare resident pool. A/R retain all output slots until all
native calls finish; shadow downloads and comparison occur after timed work.
The existing public consumer view does not expose device fitted values. P must
therefore download/compare complete shadows between windows before destroying
each prepared object. P reports raw total, diagnostic time and segmented work
time separately; these pauses can affect scheduling/thermal conditions. P is a
lifecycle reference, not an instrumentation-free end-to-end speedup baseline.
Native call sum, input packaging, token release, handle build/release and shared
resident lifecycle costs are reported separately. No output arena is introduced.

One warmup per variant followed by six frozen cycle orders:
`PAR / RAP / ARP / PRA / PAR / RAP`. A/R order is balanced 3/3, P first/last 3/3.
Native calls must create/free no application-visible GPU allocations, streams,
events or BLAS handles. Every output is checked against capture after each run;
no numerical drift is accepted. Resource counts must close after every cycle.
Resource accounting covers interposed runtime APIs; driver/library caches are
not claimed to be zero. A reduced four-setup numerical memcheck supplements
full-layer ordinary capacity, output and lifecycle checks.

Predeclared interpretation: all hard gates and capacity must pass. Recommend
further bounded slot-pool design only if R's median native-call sum is at most
0.90× A and R's time including resident build/release remains below P's segmented
work-time reference. P's comparison is a diagnostic estimate with the above
pause limitation; runtime net benefit remains unqualified. Otherwise close as
CONDITIONAL/STOP based on measured effect.
This is a diagnostic decision, not runtime integration GO or an outer-method
speedup claim. Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED,
recommended=false remain unchanged.

From repository root, use a fresh OUTPUT:

```sh
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/build.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/build.py OUTPUT replay
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT capacity
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT memcheck
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT campaign
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/profile_build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT profile-A
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/run.py OUTPUT profile-R
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/analyze_profile.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_layer_refill_20261023/archive.py OUTPUT
```

The initial combined `profile` attempt exited 139 after the P trace; the kernel
log locates a fault in `libcupti.so.12.5`, with the exact trigger unproven. All
initial profile-prefixed evidence and `PROFILE_V1_FAILURE.md` are retained,
not treated as PASS. The replacement A/R profiles use separate executables,
all prepared objects allocated before capture, and NVTX-only setup scopes (no
diagnostic CUDA events). The primary replay executable and timing receipts are
unchanged. Profiler data never enters primary performance medians.

Completed evidence: six timed runs per variant give P/A/R native medians of
106.124 / 106.154 / 31.787 seconds. R wins all six paired cycles; charging the
full resident build/release cost gives 33.580 seconds. Both replacement profiles
pass complete bitwise/resource checks and correlate all 5,730 kernels to all
955 setups. Cross-original-window kernel overlap is 0 seconds in A and 27.042
seconds in R. The 25 successful full-layer runs plus the reduced memcheck
compare 170,430 target occurrences; the initial incomplete profile is excluded.
This supports further bounded slot-pool design/diagnostics only. It does not
qualify runtime capacity with component caches, full-method elapsed time or
promotion. The detailed report is
[PHASE10_OPTIMIZER_LAYER_REFILL_20261023.md](../../../docs/PHASE10_OPTIMIZER_LAYER_REFILL_20261023.md).
