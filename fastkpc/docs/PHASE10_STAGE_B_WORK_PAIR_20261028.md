# Stage B fixed-VT work-pair isolated qualification

Date: 2026-10-28. This follows the
[static audit](PHASE10_STAGE_B_DBDSQR_WORK_AUDIT_20261028.md) and its
[review](PHASE10_STAGE_B_DBDSQR_WORK_AUDIT_REVIEW_20261028.md).

Status: **`GO_ISOLATED_STAGE_B`**. GPU bitwise qualification and all six
predeclared timing runs pass. The sum of L2/L3 medians decreases from
**114.957718 to 101.630583 seconds**, saving **13.327135 seconds (11.59%)**.
Production integration and full-method performance qualification remain pending.

## Scope and identity

Production native remains
`e8c39703f01e09260d330167b49267c295aa2221167d5248d0980a97c80ca045`.
All 103 production inputs and production `recorded_dbdsqr.inc` are unchanged.
Generated DBDSQR SHA is
`be76c5e06ce8bce4013a9759a3a54fc63feed9e421a1d3d44af77e11f32f53b9`.

The isolated fixed `UPLO=L, NCVT=1, NRU=NCC=0` implementation removes unused
forward W2/W3 stores, removes unused backward W0/W1 stores, and remaps retained
backward W2/W3 values (including sine signs) and consumers to W0/W1. A dedicated
entry rejects unsupported configurations. Lower-to-upper conversion stores,
scalar arithmetic, QR convergence, sorting, rotation wrappers, d/e/q layout,
recover, lane count, wave order, status handling and cache policy are retained.

Only research tools, artifacts and documentation are added or updated. Production defaults and native
are not replaced. The timing process interposes the three structured-plan APIs
while retaining the original native cache/wave/consumer implementation. Counters
must demonstrate ten creates/destroys, 1,638 submits and 104,797 targets in each
candidate replay. This is a complete captured Stage B replay, not a full-method
run or production integration.

## Compiled resources

Both variants use nvcc `-O3 -arch=sm_89 --fmad=false`, the original helper source
and the same compiler. The diagnostic builds are separate from timing builds.

| Uninstrumented kernel | Baseline | Work pair |
|---|---:|---:|
| Dynamic shared bytes, N=351 | 19,656 | 14,040 |
| Registers/thread | 100 | 82 |
| STACK / LOCAL bytes | 0 / 0 | 0 / 0 |
| CUDA occupancy API blocks/SM, block size 32 | 4 | 6 |

These are compiled resources and API upper bounds. Actual SM occupancy, stalls
and block overlap remain unmeasured; no profiler permissions were changed.
The change also alters register allocation and removes stores, so any timing
gain cannot be assigned exclusively to the shared-memory residency bound.

## GPU correctness

The two diagnostic libraries execute the same instrumented production pipeline
on GPU1. They compare GPU memory directly and download compact comparison
results. CPU code constructs synthetic inputs and checks bytes/hashes; it does
not provide numerical QR/SVD authority.

- All **104,797** captured components pass, in original order and width-64
  groups: 49,919 L2 and 54,878 L3 components.
- **10,515,312,078** recorded rotations are compared field by field and in order
  across those components; no mismatch occurs.
- Compared state includes d/e/q, rotation count/overflow/info, scalar branch
  coverage, a/b, scale/live metadata, candidates, selected values/vectors/means
  and boundary status. Shared-memory end canaries remain intact.
- **433** synthetic/boundary cases pass, including r=1..350, small active blocks,
  zero/nonzero shifts in both directions, direction changes, repeated values,
  sign/tie/signed-zero inputs, extreme finite scales and non-finite prechecks.
- Four rotation-capacity tests cover 0, 1, exact count minus one, and exact
  count; overflow behavior and continued q updates match.
- Nonconvergence is tested once by explicitly lowering the diagnostic iteration
  budget in both variants. This is fault injection, not naturally observed
  nonconvergence, and is absent from timing code.
- Tracked diagnostic allocations and bytes return to zero.

Coverage bits are, in order: r=1, lower conversion, two-by-two, zero forward,
zero backward, shifted forward, shifted backward, direction switch, sign,
sort tie, nonconvergence. Captured counts are
`[0,104797,104797,21163,0,104797,100556,100556,104796,0,0]`.
Synthetic counts (including the one injected failure) are
`[9,420,399,8,5,230,379,222,370,173,1]`.
Thus r=1, zero-backward and sorting ties rely on constructed cases; no claim is
made that the natural captures cover them.

## Predeclared timing decision

GPU1 is exclusive and GPU0 is free of compute tasks. CPU affinity is 0–19,
math-library threads are one, CUDA connections are 32. Each timing run starts
a fresh process. External compute activity stops only our process and preserves
the interrupted attempt.

The fixed order is **A1, C1, C2, A2, A3, C3**, where A is unchanged native and C
is the uninstrumented candidate. Intermediate diagnostic exports/canaries are
absent. Every run validates all native captured component payload hashes, zero
dense fallback, original cohorts and native resource closure. Final payload
hashing occurs after each measured Stage B, as in the sealed attribution tool.

The Stage B boundary includes recover, status/checkpoint, wave joins and cache
stores. Stage GO requires both level medians to improve, their sum to decrease
by at least 5%, and every candidate total to be below every baseline total.
Smaller/noisy positive gain is CONDITIONAL; no combined gain or a correctness
failure is STOP. This gate only determines whether later production testing is
warranted.

All six runs completed in the declared order on GPU1, with no external compute
process observed by the isolation monitor. Times below are complete Stage B
seconds, not end-to-end process time (which also includes input loading and
post-boundary diagnostic payload hashing).

| Run | L2 | L3 | L2 + L3 |
|---|---:|---:|---:|
| A1 | 54.610112 | 60.214745 | 114.824857 |
| C1 | 48.463689 | 53.226766 | 101.690455 |
| C2 | 48.422729 | 53.196172 | 101.618901 |
| A2 | 54.697374 | 60.260344 | 114.957718 |
| A3 | 54.866117 | 60.275344 | 115.141461 |
| C3 | 48.434411 | 53.188908 | 101.623319 |

| Per-stage median | Native A | Work pair C | Reduction |
|---|---:|---:|---:|
| L2 | 54.697374 | 48.434411 | 11.45% |
| L3 | 60.260344 | 53.196172 | 11.72% |
| Sum of the two medians | 114.957718 | 101.630583 | **11.59%** |

The last row is the predeclared sum of per-stage medians, not a median of
full-method elapsed times. Even the slowest candidate total (101.690455 seconds)
is below the fastest baseline total (114.824857 seconds). All gates pass:
both levels improve, the combined reduction exceeds 5%, and the run ranges
are separated.

Every run matches all 104,797 sealed native three-field hashes: **628,782** hash
comparisons across six runs. Dense fallback remains zero and native resource
ledgers close. Every candidate reports exactly ten plan creates/destroys,
1,638 submits and 104,797 targets; measured code is the intended two-segment work
variant, with 14,040 B shared, 82 registers and the occupancy upper bound of six.
The maximum sampled GPU1 usage across timing runs is 15,969 MiB; this is sampled
isolated-replay usage, not a continuous peak or a full-runtime capacity result.

The numerical result establishes parity on the captured and constructed cases;
it is not a universal floating-point proof. The performance result establishes
an isolated Stage B benefit. A later, separately qualified default-off runtime
integration must still pass complete canonical p-value/pMax, graph/sepset/count,
RNG, route/status, GPU-only authority, capacity and resource gates, followed by
repeated outer elapsed measurements. Do not subtract 13.327135 seconds from the
accepted full-method result and label the difference a measured runtime.

## Limits and retained state

The captured replay excludes optimizer workspaces and intervening CI consumers.
No integrated full-method capacity, p-value/graph/RNG campaign, new real-data
qualification or outer elapsed improvement is claimed here. The accepted
opt-in development median remains **329.078 seconds**. L8 STOP, defaults off,
Phase 10 ACTIVE, candidate NOT_FROZEN, holdout SEALED and recommended=false
remain unchanged.

Tools: [stage_b_work_pair_20261028](../tools/research/stage_b_work_pair_20261028/).
Evidence: [attempt1](../artifacts/full_cuda_ci/stage_b_work_pair_20261028/attempt1/).

| Sealed object | SHA-256 |
|---|---|
| closure.json | `0246f2f49ae3fc609c7e3cfe630c2de4289e8534a55c6b70c939238793f06e3c` |
| manifest.json | `cf9618361f00d1ea455f6337b5a500d43f10824390a49d0e12fc8f80056e9671` |

The manifest binds artifacts, source inputs, captured dependencies and tools.
Independent verification passes all **218** entries (98 artifact files, 103
production inputs, eight dependencies and nine tools), artifact file-set
equality, and the native/closure/manifest hashes.
The archive-only helper was added after the execution build and is bound by
the final manifest; no frozen build/campaign tool was edited during execution.
This report and the goal-document entry are written after archive and are not
part of its sealed artifact file set. All own GPU/monitor processes have ended;
no compute tasks remain on either GPU at closure. No commit or push.
