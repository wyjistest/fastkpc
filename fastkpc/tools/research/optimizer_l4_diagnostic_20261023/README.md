# Captured L4 scheduling and capacity diagnostic

Predeclared scope, 2026-10-23: unchanged production native from the completed
L5 opt-in qualification (`b0d35232...cdf58`), GPU1 only, shared host with GPU0
external training preserved. No production edits, default changes, other-layer
slot integration, commit/push or promotion. CPU affinity 0-19, CUDA connections
32, status-wait=1, one BLAS/OMP/MKL thread.

Capture the complete canonical S2 prefix through L4. Preserve all 34 original
windows (33x64+42), 2,154 setups, 17,767 targets, n=351, q=37, four penalties,
every actual target cohort, geometry, Y, controls, target order and full output.
Compare the prefix CI/pMax bits, task/deletion/skeleton/sepset/count/RNG and
authority/resource gates to the qualified L5-runtime ON1 receipt. Shadow D2H
is diagnostic only; no CPU numerical production is allowed. Capture elapsed
is not a new full-method performance result.

Before timing, census all target capacities and application allocation sizes.
Then attempt original-order whole-layer prepared residency, requiring at least
2 GiB physical free memory. Stop at the headroom boundary and drain/destroy
all objects; preserve the attempted coverage and resource closure. Do not
silently drop setups or treat a partial resident layer as a successful one.
This test excludes the production component cache and arena.

The capacity outcome selects exactly one full-layer timing experiment:

- If the entire layer fits: A uses original windows/64 workers; R uses all
  original-order requests/32 workers. Both share the same complete resident
  pool per cycle. This measures worker-count and barrier removal together.
- If whole-layer residency fails: explicitly STOP that resident experiment.
  A/W then use the original 34 windows with 64/32 workers, respectively.
  Only one original window's prepared objects exist at a time; each pair uses
  the same objects and actual cohorts. This tests worker count alone and
  provides no cross-window refill evidence.

One warmup per variant, then six paired cycles with alternating order (three
of each order). Every pass covers the entire captured layer and compares all
optimizer state, coefficients/fitted/residual, rank/status and trajectory bits.
Report native call sums, packaging, diagnostics, token release, pool build and
destroy separately. Window-only timings contain between-window shadow pauses;
do not call their sum an instrumentation-free outer runtime. Resident A/R
retain all output leases until timing ends. No duration sorting or cohort edits.

Audited allocations/frees, stream/event and BLAS-handle mutations inside
native optimizer calls must be zero; all exposed resources and leases close.
A reduced memcheck covers every observed target capacity plus endpoint
setups, with zero errors/leaks. All original fixtures participate in ordinary
full-layer validation. Any numerical or unexpected resource failure stops.

A candidate ratio <=0.90 of A's six-run median and six paired wins supports
the next separate diagnostic; [0.90,0.97) or inconsistent wins is CONDITIONAL;
>=0.97 is STOP for the tested scheduling policy. These are captured diagnostic
decisions, not slot-pool or runtime integration GO. No GPU overlap is claimed
without a separate device trace. Record whole-run sampled GPU1 memory peak,
not just application allocations or a creation delta.

Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED, recommended=false.
L5's prior runtime result remains unchanged. Final PDAG and real-data campaigns
are outside this diagnostic.

From repository root, with a fresh OUTPUT:

```sh
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/build.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/run.py OUTPUT capture
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/build.py OUTPUT replay
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/run.py OUTPUT capacity
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/run.py OUTPUT memcheck
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/run.py OUTPUT campaign
python3 -I fastkpc/tools/research/optimizer_l4_diagnostic_20261023/archive.py OUTPUT
```

## Completed result

`STOP_FULL_LAYER_CAPACITY` plus `GO_L4_WINDOW_WORKER_DIAGNOSTIC`.
The unchanged native capture passes all 234,152 prefix CI/pMax bits and
task/skeleton/sepset/count/RNG, authority and resource gates. All 2,154 setups
and 17,767 targets are captured; 1,006 setups exceed eight targets, including
999 eleven-target cohorts, with an actual maximum of eighteen.

The standalone full-layer attempt crosses the 2 GiB headroom gate at 1,298
prepared objects and releases all of them. Free memory falls from
24,424,284,160 to 2,142,044,160 bytes, a 20.752 GiB physical increment for that
prefix alone. Whole-layer application buffers would sum to 14.247 GiB;
this excludes driver/library costs and is not a capacity pass. No resident
A/R timing or partial-input substitute is included.

The window-only experiment retains all 34 original windows and compares A=64
with W=32 workers (A's last window uses 42). Six A/W native-call timings:

| Cycle | Order | A seconds | W seconds |
| --- | --- | ---: | ---: |
| 1 | A/W | 92.313378 | 80.959929 |
| 2 | W/A | 93.323565 | 80.398927 |
| 3 | A/W | 91.457737 | 80.875274 |
| 4 | W/A | 96.020865 | 80.333885 |
| 5 | A/W | 89.020633 | 80.870667 |
| 6 | W/A | 91.776521 | 80.520311 |

Medians are 92.044950/80.695489 seconds, W/A=0.876696546 (12.330% lower),
with six W wins. Including full shared prepared lifecycle cost gives
96.231291/84.884677 seconds of segmented work. The between-window shadow
pause limitation applies; this is not a full-method measurement or proof
of device overlap.

Fourteen full replays including warmup compare 248,738 target output
instances bitwise. Reduced memcheck covers all fourteen target capacities
and endpoints (15 setups, 136 targets per variant, 272 comparisons), with
zero errors/leaks. All output leases and exposed resources close; audited
resource mutation count inside native calls is zero. Sampled GPU1 peaks for
capture/capacity/memcheck/campaign are 21,624/20,430/2,202/3,146 MiB; 200 ms
telemetry does not supersede the per-creation free-memory capacity checks.

This supports a separate L4 original-window 32-worker opt-in runtime
qualification. Production source/native/defaults and the existing 472.080s
full-method baseline are unchanged. Shared-host development only; no runtime
integration, production cache/arena capacity, final PDAG or promotion claim.
See [the diagnostic result](../../../docs/PHASE10_OPTIMIZER_L4_DIAGNOSTIC_20261023.md).
