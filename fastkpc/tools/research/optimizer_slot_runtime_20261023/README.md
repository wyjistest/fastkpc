# L5 opt-in slot runtime qualification

Default-off `FASTKPC_DCC_GAMMA_L5_OPTIMIZER_SLOTS=1` is restricted to the
canonical S2 351x48 dcc.gamma shape, exact-primary coefficient reuse and
status-wait=1. L5 requests must retain their original 2/3/4/8 targets, q=46,
five penalties; other shapes fail closed. Other layers keep their scheduler.

Prepare all owned L5 geometry on the R thread, reserve coefficient-arena space,
then create at most 32 capacity-eight slots and blocking detach events.
Workers take original-order requests, rebind geometry, call the unchanged
exact-primary kernel and save coefficients into the existing arena. Arena
insertion is serialized; each slot's event must finish before its old output
lease is released. No CUDA resource grows in the worker interval. No R API or
target-state cache is accessed by workers. Packet lifetimes extend through
thread join and failure drain.

After all workers join, the original completion logic publishes target states
and runs Stage A in the original setup/target order. One pool slot is rebound
for each Stage A producer, using the existing coefficient gather/finalizer and
residual detach. The pool is drained and destroyed before Stage B. This is one
real optimizer boundary at L5; receipts must not label it as 15 old windows.
No sorting, Stage B overlap or changes to floating-point kernels.

Predeclared gates:

- Full clean-directory source-closure build, all five capacity objects' GPU
  instruction words unchanged. Production source hashes frozen while running.
- Real 8/2/4/3-target captures: optimizer state, coefficients, fitted, residual,
  rank/status/trajectory exact; busy and malformed binding rejection; stale
  tokens and old-token destruction safe; cancellation cleanup and reduced
  CUDA memcheck zero errors/leaks. No CPU numerical authority or fallback.
- Full canonical 240,498 CI/pMax bits, tasks/deletions/graph/sepsets/counts/RNG
  exact versus qualified S2; all CI/pMax authority errors <=1e-10. Original
  physical cohort and Gram order hashes exact. Exact-primary/replay and cache
  route/status ledgers unchanged. All application resources close.
- Joint production cache/arena/slot memory: >=2 GiB free after pool creation,
  no OOM, record free-memory boundaries and sampled GPU1 peak. Creation
  increment must not be described as total occupancy.
- Six full fresh-process runs in OFF/ON/ON/OFF/OFF/ON order, three per mode.
  ON median outer time <=0.97 OFF and all three adjacent OFF/ON pairs improve
  before declaring a development performance GO. Every run passes hard gates.
  Any failure stops the campaign without relaxing checks.

GPU1 only, affinity 0-19, CUDA connections=32, one BLAS/OMP/MKL thread,
11 GiB retained component budget, bypass=1, eight eig lanes, status-wait=1.
GPU0 external training is preserved: shared-host development, not isolated
performance or promotion. Final PDAG/real-data campaigns are not substituted
by this skeleton qualification. Defaults stay off. Phase 10 ACTIVE, candidate
NOT_FROZEN, holdout SEALED, recommended=false.

Run from repository root into a fresh directory:

```sh
python3 -I fastkpc/tools/research/structured_prefill_canonical_20261022/build.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_slot_runtime_20261023/checks.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_slot_runtime_20261023/campaign.py OUTPUT
python3 -I fastkpc/tools/research/optimizer_slot_runtime_20261023/archive.py OUTPUT
```

## Completed result (2026-10-23)

`GO_L5_OPT_IN_RUNTIME_DEVELOPMENT`. Full-build native SHA-256:
`b0d3523285bc8098286d00350c5d4204edec31aab13f6336090fab2d203cdf58`.
The 98-file source closure includes uncommitted changes on HEAD `d982dcf`;
this is not a clean producer result. All 74 GPU functions across five capacity
objects have unchanged instruction words. Normal and reduced memcheck checks
each compare 68 optimizer/Stage A target instances bitwise and pass 24 negative
or stale-token checks; memcheck reports zero errors and zero leaked bytes.

| Execution order | Mode | Outer seconds | L5 optimizer host seconds |
| --- | --- | ---: | ---: |
| 1 | OFF1 | 554.703 | 107.075796 |
| 2 | ON1 | 472.080 | 31.827456 |
| 3 | ON2 | 474.505 | 31.800844 |
| 4 | OFF2 | 550.152 | 109.892481 |
| 5 | OFF3 | 548.008 | 104.332973 |
| 6 | ON3 | 470.261 | 31.814829 |

OFF/ON outer medians are 550.152/472.080 seconds (14.191% reduction,
1.1654x speedup); all three adjacent pairs improve. L5 host medians are
107.075796/31.814829 seconds. This jointly changes worker count, whole-layer
refill and slot reuse; it is not an allocation-only attribution or pure GPU time.

All six full canonical runs pass every hard gate, with 240,498 CI per run,
CI/pMax bits unchanged, maximum authority error 3.05425129631942e-12,
original cohorts/Gram order, GPU-only production and closed resources.
Each ON run creates/destroys 32 slots for 955 setups/6,816 targets and leaves
2,372,730,880 bytes free after pool creation. Physical creation delta is
516 MiB; all six whole-run GPU1 sampled peaks are 22,366 MiB. Capacity remains
qualified only for this canonical S2 L5 configuration. GPU0 external training
remained active; shared-host development and default-off boundaries still apply.

The first standalone check launch failed before GPU execution because the
Rlapack loader path was absent. Its logs remain under `loader-attempt/`,
excluded from PASS counts; the runner environment fix did not change production
source or native. `closure.json` and `manifest.json` bind the completed results,
source/native, tools and references. The exact campaign runner is retained as
`qualified-run.R`: final tool cleanup removes one blank line at EOF only,
which `archive.py` verifies byte-for-byte; original receipt runner hashes are
unchanged and bind that archived copy. See the
[runtime result](../../../docs/PHASE10_OPTIMIZER_SLOT_RUNTIME_20261023.md).
