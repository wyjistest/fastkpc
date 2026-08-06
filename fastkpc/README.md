# fastkpc

Staged fast kPC backend for this workspace.

This directory is intentionally separate from `kpcalg/R`. The legacy package
files stay unchanged while the fast backend is developed and validated.

## Full-CUDA legacy-compatible skeleton and WAN-PDAG candidate

An explicit one-call full-CUDA route now implements the numerical CI data plane
for the exact `regrXonS`/KPC subset. The historical Phase 10 v1 campaign passed
canonical correctness, CUDA authority, hardening, and replay-latency gates, but
did not measure acceptable fresh-data performance. The sealed holdout remains
`SEALED_NOT_RELEASED` and must not be opened for this candidate. The route is
explicit, not the default, and not recommended.

The current Phase 9 v2 and hardening v2 artifacts use the actual default
`kpcalg` conditioning contract (`m.max = Inf`) and cover the canonical graph
through its natural stop after level 8.

```r
source("fastkpc/R/fast_kpc.R")

skeleton <- fastkpc_compatible_cuda_skeleton(
  data,
  alpha = 0.1,
  options = list(
    route = "full_cuda",
    compatible_cuda_strict = TRUE,
    max_conditioning_size = Inf,
    index = 1,
    numCol = 35L,
    trace_level = "logical"
  )
)
```

For a strict end-to-end KPC result, use the same CUDA skeleton followed by the
kpcalg WAN-PDAG authority in one process:

```r
result <- fastkpc_compatible_cuda_wanpdag(
  data,
  alpha = 0.1,
  options = list(
    max_conditioning_size = Inf,
    index = 1,
    numCol = 35L,
    trace_level = "logical",
    ci_method = "hsic.perm",
    hsic_params = list(sig = 1),
    permutation_params = list(
      replicates = 100L, seed = 707L, include_observed = TRUE
    )
  )
)
```

The skeleton remains CUDA-authoritative. The short orientation stage uses
`kpcalg::udag2wanpdag` and `kpcalg::kernelCItest` as CPU numerical authority,
records every orientation CI identity, p-value, and decision, and preserves the
legacy R RNG stream for permutation methods. The native fastkpc orientation
engine remains available for development, but it is not the strict authority
for the canonical 351 x 48 route.

This API is separate from `fast_kpc(precision = "compatible")`, which remains
the older mgcvExtractGPU compatibility bridge documented below. The full-CUDA
candidate uses one native skeleton call, a C++ canonical control plane, native
response-independent setup construction, CUDA target-specific GCV and stable
residual solves, device-resident CUDA dCov/HSIC components, method-specific
gamma or permutation p-values, and canonical compact-result replay.

Qualified semantic envelope:

```text
input: finite binary64 matrix; n > 35; 2 <= p <= 64
alpha: exactly 0.1
index / numCol: 1 / 35
maximum conditioning size: defaults to Inf and resolves to p - 2
multi-penalty CUDA capacity: |S| through 62, with q = 1 + 9|S| < n
model: Gaussian, identity, unweighted, zero offset
smooth route: joint thin-plate smooth for |S| <= 2
              additive thin-plate smooths for |S| > 2
graph stage: CUDA skeleton, optionally followed by kpcalg-authority WAN-PDAG
```

This is a compatible implementation of that exact subset, not a full mgcv
clone. Unsupported values fail before canonical replay. Strict skeleton
execution has no legacy-mgcv target fit or setup call in the CI loop, no R
callback, no CPU residual/CI numerical fallback, and no approximate fallback.
The optional WAN-PDAG completion deliberately uses kpcalg CPU authority; that
expected orientation work is not a skeleton fallback.

`Inf` is a finite graph-search contract, not an instruction to execute
unbounded levels. With `p` variables, `kpcalg` can use at most `p - 2`
conditioning variables, and the stable graph normally stops earlier when no
live edge has enough neighbors. On the canonical 351x48 input the search needs
level 8: its per-level test counts are
`2213,52659,125293,40694,13293,5422,835,80,9`, after which it stops naturally.
The CUDA optimizer is compiled in q/penalty buckets `64/7`, `80/8`, `192/21`,
`384/42`, and `559/62`; larger buckets reduce concurrent setups to bound VRAM.
If a reached additive model violates `q = 1 + 9|S| < n`, strict mode fails
closed instead of silently truncating the `Inf` search.

The candidate uses these bounded caches:

```text
compact CI result cache: 262144 entries
target optimizer state cache: 131072 entries
native Prepared-S active device cache: 2048 single-penalty + 8192 multi-penalty
per-call native Prepared-S host cache: 16384 entries
device dCov component capacity: 47 entries
strict-method residual cache: <= 384 MiB, <= 131072 entries
strict hsic.perm component cache: <= 384 MiB, deterministic LRU
```

The strict `hsic.gamma`, `dcc.perm`, and `hsic.perm` routes use a
one-call-scoped residual slab and persistent CUDA execution context. A batch
with any residual-cache miss still executes its complete original fixed-SP
cohort. Only an all-hit batch bypasses the solve, and its residuals are gathered
D2D in the original target order. The cache identity includes the authenticated
ResidualKey and solver route. Conditioning systems with at most two penalties
use the fully qualified augmented-QR residual path; larger systems retain the
authoritative stable-SVD path.

The HSIC incomplete-Cholesky component kernel also parallelizes independent
row work across one 256-thread block per component. Pivot selection,
triangular solves, mean accumulation, and final reductions retain their
original order. `hsic.perm` additionally keeps a bounded persistent component
LRU. Its R 4.4 Rejection-sampler index operation is inlined for `n <= 32768`
with the exact `R_unif_index()` draw/rejection order; unsupported sample kinds
or larger `n` fall back to the R API. Set
`FASTKPC_STRICT_HSIC_PERM_INLINE_R_UNIF_INDEX=0` to disable that path.

Permutation request authentication still covers every table byte. A C++17
move-only builder owns the payload, hashes each completed row incrementally,
and seals it into an immutable handle. The trusted one-call consumer uses the
sealed OpenSSL SHA-256 attestation without scanning the payload a second time;
untrusted boundaries still require a complete rehash. The portable SHA-256
implementation remains a compile-time fallback and produces the same digest.

Clean default-Inf 351x48 development receipts from producer `830cfbe` measured:

```text
method       original     50595ce      830cfbe      latest reduction
hsic.gamma   1562.052 s    631.522 s    593.814 s       5.97%
dcc.perm     2131.261 s    945.359 s    849.693 s      10.12%
hsic.perm    3877.316 s   1152.905 s   1042.358 s       9.59%
total        7570.629 s   2729.786 s   2485.865 s       8.94%
```

The final `hsic.gamma` maximum absolute p-value difference is
`8.271161533457416e-14`, below its `1e-10` contract with zero decision flips;
both permutation methods remain bitwise exact. Logical traces, adjacency,
undirected sepsets, `pMax`, `n.edgetests`, permutation RNG terminal state, the
19 kpcalg-authority orientation CI tests, and final PDAGs are exact. All three
runs report zero residual/component D2H, zero CPU numerical authority, zero
fallback, and zero residual-cache eviction. The bounded `hsic.perm` component
LRU records 240,685 hits and 234,199 deterministic evictions. The clean
receipt and source closure are indexed by
`artifacts/strict_ci_methods_351x48_optimized_v2/manifest.json`.

These routes are not exhausted. The remaining dominant exact work is the
level-3-and-deeper stable-SVD residual path: residual solving still costs
`383.800`, `455.173`, and `433.361` seconds respectively. Permutation table
generation plus incremental SHA-256 costs `117.275` seconds for `dcc.perm` and
`255.772` seconds for `hsic.perm`; trusted request-identity build and validation
now add only `1.469` and `1.764` seconds. The development path now submits one
GPU residual/component preparation before each R permutation build and queues
permutation finalization on the same stream, with no intermediate host event
wait. Stable-SVD cohorts use the qualified nonblocking submission path; QR
cohorts may still wait inside submission. The overlap is enabled by default for
`dcc.perm` and `hsic.perm` and can be disabled with
`FASTKPC_STRICT_METHOD_PERMUTATION_GPU_OVERLAP=0`. Fixture-level bitwise,
failure-order, RNG, resource, and WAN-PDAG gates pass, but the canonical
`351x48` wall times have not yet been rerun for this implementation. The prior
timings remain the current clean performance evidence and do not replace the
`dcc.gamma` Phase 10 baseline.

The historical canonical artifact is
`artifacts/full_cuda_ci/promotion_351x48_v1/`. Across five fresh-process cold
runs, five same-data replay-warm runs, and five fresh correct baselines, every
run had 110 edges, SHD 0, exact adjacency, exact normalized sepsets, exact
`n.edgetests`, and an exact deletion/logical trace. Timings on
`reference_machine_v1` were:

```text
fresh-process cold median  1290.664 seconds
replay-warm median             0.699 seconds
baseline median              601.431 seconds
replay/baseline ratio          0.001162228
```

The v1 replay-warm timing is a second call after one complete same-data call in
a fresh process. It replays authenticated, capacity-bounded compact results and
performs zero physical CI tests or residual fits. It is not evidence for the
latency of a previously unseen input matrix. `performance_budget_v2` preserves
this result as report-only replay latency and requires a separate fresh-data
compute-warm boundary with empty dataset-specific caches.

The v2 hard gates are:

```text
fresh-data compute-warm median <= 120 seconds
fresh-data compute-warm median <= 0.80 * same-run correct baseline median
fresh-process cold median <= same-run correct baseline median
replay-warm: report only
```

The current development producer has passed the first optimization checkpoint,
but it is not a frozen candidate. A fresh-data compute-warm development run on
the canonical 351x48 input with explicit `max_conditioning_size = 7` completed
in `235.058` seconds, down from the prior `263.305`-second best. Score-first
line-search evaluation first reduced the optimizer boundary from `173.246` to
`158.189` seconds. A one-call-scoped fixed-SP penalty-root cache then reused
only roots produced and validated by the authoritative fixed-SP eigensolver.
The retained run measured 23.642 seconds for native setup, 159.511 seconds for
the optimizer boundary, 15.375 seconds for residual solves, and 29.933 seconds
for dCov. All 240,489 task rows and p-value bits, graph fields, sepsets, counts,
levels, and optimizer trajectory counters remain exact.

The default-`Inf` development run with the same implementation took `233.848`
seconds, down from the score-first `252.616`-second run and the Phase 9
`277.868`-second artifact. It resolved `Inf` to 46, executed levels 0 through
8, and produced 240,498 logical tests. Every task row and p-value bit, graph
field, sepset, count, and level semantic is exact against the older authority
artifact; CPU numerical authority, fallback, residual D2H, and component D2H
remain zero.

An opt-in CPU-only setup/optimizer pipeline now applies to all supported
full-CUDA CI methods and provides the best `dcc.gamma` development result. It
keeps R and Rcpp objects on the main thread, passes only owned
optimizer inputs through one background future, and preserves every original
window, setup cohort, target vector, and target order. With a fixed 5 ms
optimizer head start, level 3 fell from `114.038` to `102.770` seconds and the
default-`Inf` run fell from `233.848` to `219.513` seconds. The latter overlaps
`19.319` seconds of setup work across 138 windows. All 240,498 task and p-value
bits, graph/sepset/count/level fields, and optimizer trajectory counters are
exact; authority and transfer gates remain zero. The route is enabled with
`FASTKPC_PHASE10_SETUP_OPTIMIZER_PIPELINE=1` and
`FASTKPC_PHASE10_SETUP_OPTIMIZER_PRODUCER_DELAY_US=5000`; it is not yet the
production default or a frozen candidate. GPU-side setup overlap, CPU affinity,
and synchronous level-tail variants were rejected on measured wall time.

The fixed-SP root cache is enabled for all supported one-call CI methods and
can be disabled deterministically with
`FASTKPC_PHASE10_FIXED_SP_ROOT_CACHE=0`. It authenticates the coefficient
dimension, offset, rank, and every penalty-block bit on each hit. Each runtime
is bounded to 4,096 entries and 256 MiB. The default-`Inf` receipt records
28,611 lookups, 26,925 hits, 1,686 inserts, zero bypasses, zero identity
rejections, and 8,691,624 live device bytes at peak. The clean strict-method
receipts record hit/lookup counts of `29,544/31,325`, `25,250/26,967`, and
`24,844/26,646` for `hsic.gamma`, `dcc.perm`, and `hsic.perm`, with zero bypass
or identity rejection. Cached roots move only D2D; the runtime resource ledger
returns to its pre-call active counts at teardown.

The default-Inf cold run consumed 8,643 unique PreparedS keys and physically
built 8,646 setups (three bounded speculative builds). It used 138 optimizer
host boundaries, built and released 7,472 right-sized multi-penalty handles
with a peak target capacity of 33, and recorded 22,291 bounded excess target
optimizations out of 132,926. The additive setup path made 27,436 primitive
requests, built exactly 48 primitives, and served the remaining 27,388 requests
from the per-call cache.
Adjacency, normalized sepsets, `n.edgetests`, task order, and deletion decisions
matched the prior canonical result; the maximum p-value difference was
zero (all consumed p-values were bitwise identical), and every CPU numerical
authority, fallback, and residual/component D2H counter was zero. The public
500x50 v2 production-shape fixture omits the conditioning limit, resolves the
default `Inf` ceiling to 48, stops naturally at level 3, and also passes with
empty dataset-specific caches and one native skeleton call. The default-Inf
hardening artifact is `artifacts/full_cuda_ci/failure_injection_default_inf_v2/`.
Its opt-in extended test executes q192, q384, and the maximum
`n=600,p=64,|S|=62,q=559` capacity shape; all remain CUDA-authoritative with
zero fallback.

A non-formal max-conditioning-size-3 profile now exposes the optimizer work
instead of treating the CUDA boundary as one opaque stage. With score-first
evaluation it completed in `118.729` seconds, including `36.883` seconds in the
multi-penalty optimizer, down from `40.939` seconds under the immediately prior
penalty-hoist build.
Across 54,878 targets it recorded 563,638 optimizer iterations, 700,893 score
calls, 1,053,964 objective calls, 137,255 step halvings, 941,534 guarded-QR
evaluations, and 85,596 stable-SVD evaluations. Multi-penalty prepared-handle
construction accounted for `5.812` seconds, so pooling those handles alone
cannot close the remaining performance gap.

Within the `1.316e12` stage-0 QR/reduction cycles, the new profile attributes
`63.6%` to QR factorization, `25.7%` to explicit Q generation, `5.4%` to the
guard, and `5.3%` to fallback bidiagonal reduction. Two exact-trajectory
`DGER` lane-remapping experiments, a 32-thread target block, and a five-block
launch bound were rejected because full level-3 wall time stayed flat or
regressed. These experiments are not present in the production path. The next
optimizer change therefore needs decomposition reuse or a larger execution-
shape change while preserving the authenticated optimizer transcript.

A guarded-QR Q-copy-elision prototype also preserved every one of the 220,859
level-3 task rows and p-value bits, graph field, semantic level field, and
optimizer trajectory counter. Its `114.002`-second wall time was effectively
identical to the `114.038`-second fixed-root-cache baseline, however, and the
instrumented QR/Q cycle changes were within run-to-run noise. The prototype was
reverted as `STOP_GUARDED_Q_COPY_ELISION_WALL_TIME`; removing the final Q copy
is not a substitute for restructuring QR factorization or Q generation.

Three subsequent small64 resource prototypes also preserved every level-3 task
and p-value bit, graph/level field, and optimizer trajectory counter, but failed
against the pipelined `102.770`-second baseline. Dynamic-shared augmented QR
through q=46 took `103.084` seconds; QR/SVD cycles fell `2.58%`, but the
multi-optimizer boundary was flat at `36.137` versus `36.159` seconds
(`STOP_SHARED_QR_MATRIX_WALL_TIME`). Fusing each left Householder dot/update
reduced Q-generation cycles `6.67%`, but increased per-thread stack from 672 to
704 bytes and regressed to `103.529` seconds and `36.740` multi-optimizer seconds
(`STOP_FUSED_LEFT_REFLECTOR_WALL_TIME`). A `--maxrregcount=192` build traded 255
registers for a 928-byte thread stack and regressed to `104.373` seconds and
`37.429` multi-optimizer seconds (`STOP_OPTIMIZER_REGISTER_CAP_WALL_TIME`). The
three prototypes are absent from retained source and binary. Their receipts are
`fresh-data-level-03-shared-qr-q46-v1-development.rds`,
`fresh-data-level-03-fused-left-reflector-v1-development.rds`, and
`fresh-data-level-03-optimizer-reg192-v1-development.rds`.

A same-stream `cudaMallocAsync/cudaFreeAsync` exact-dCov prototype also
preserved every level-3 task bit, graph field, and level semantic. It reduced
exact dCov host time only from `26.844` to `26.111` seconds and teardown from
`2.994` to `2.719` seconds, while fixed-SP residual time rose from `13.271` to
`26.601` seconds as allocator-pool work entered the producer/consumer critical
path. Total time regressed from `102.770` to `116.948` seconds. The decision is
`STOP_EXACT_ASYNC_ALLOCATOR_CRITICAL_PATH`; the prototype is absent from
retained source and binary. Any future persistent context must preallocate a
bounded arena outside both stage boundaries rather than allocate per batch.

Compile-time specialization of the optimizer into separate primary and
stability-replay kernel instances also preserved all task bits, graph/level
fields, and trajectory counters. It did not release the dominant resource
pressure: the primary instance remained at 255 registers and its thread stack
increased from 672 to 704 bytes. Level 3 regressed to `104.125` seconds and the
multi-optimizer boundary to `37.234` seconds. This is recorded as
`STOP_PRIMARY_REPLAY_SPECIALIZATION_WALL_TIME`; the prototype is absent from
retained source and binary.

The current numerical-contract epoch therefore stops before Checkpoint B. The
retained exact default-`Inf` best is `219.513` seconds, leaving `39.513` seconds
to the `<180` gate. Even treating all measured dCov teardown (`3.459` seconds)
and otherwise unattributed dCov host time (`7.416` seconds) as free leaves
`208.638` seconds and a `28.638`-second optimizer deficit. Setup overlap has
already recovered `19.319` seconds, while GPU-side setup overlap raised
fixed-SP H2D to `38.169` seconds. The decision is
`STOP_PHASE10_STRICT_180_NO_QUALIFIED_ROUTE`: no candidate is frozen, the
external holdout remains sealed, and the route remains unrecommended. This is
not a claim that the RTX 4090 cannot reach 180 seconds; reopening requires a
real-shape single-matrix evaluator showing at least `1.5x` QR/Q throughput and
an expected `>=1.35x` full multi-optimizer speedup, or a separately versioned
and qualified numerical contract.

An opt-in exact-bit decomposition trace then measured whether response-
independent decomposition reuse is large enough to justify that first route.
The full max-conditioning-size-7 development trace recorded 1,691,603 actual
decomposition requests, 1,618,329 unique setup/capability/log-sp keys, and
73,274 reusable requests (`4.33%`). Initial evaluations supplied 73,272 of
those requests; only two later steepest-descent requests matched exactly.
Newton, step-halving, boundary-probe, terminal-confirmation, stability-replay,
and selected-fit stages had zero exact reuse. Reuse also declined as penalty
count increased: `4.93%`, `3.67%`, `3.08%`, `2.34%`, and `1.55%` for three
through seven penalties. Unique-key group sizes had p50 and p95 equal to one
and a maximum of 33. Trace overflow and route mismatch counts were zero, and
the full traced result was bitwise identical to the `263.305`-second result for
every consumed p-value, graph field, sepset, count, and task.

The full traced call took `421.065` seconds because it downloads and groups
every diagnostic record; that value is not performance evidence. A level-3
trace-disabled rerun completed in `121.940` seconds versus the prior `121.848`
seconds (`+0.08%`), with bitwise-identical tasks and all optimizer work counters
unchanged, so the dormant trace branch has no observed default-path regression.
Exact decomposition memoization can remain a secondary initial-evaluation
optimization, but a maximum observed `4.33%` request reduction cannot close the
optimizer gap.

A development-only staged evaluator then tested the proposed grouped execution
shape before changing the optimizer state machine. It retains the certified
per-matrix `dgeqr2`/R-copy/`dorg2r`/guard arithmetic, assigns one independent
matrix to each warp, compacts guard failures into a device queue, and sends
those failures through the existing exact 64-thread/two-warp stable-SVD
continuation. Separate baseline and grouped workspaces are compared before
post-processing for bitwise-identical R, explicit Q, left basis, singular
values, right basis, condition diagnostics, route, and status. The completed
RSS, EDF, score, gradient, Hessian, and coefficients are also compared bitwise
against both the staged baseline and the existing standalone Phase 6 evaluator.

On `reference_machine_v1` (RTX 4090), a 512-target campaign covered real
Prepared-S shapes `q=28/37/46/55/64`, three through seven penalties, guarded-QR
and stable-SVD routes, and 2/4/8 warps per CTA. All 15 configurations had zero
internal or public-output mismatches. Throughput, however, did not clear the
prototype gate:

```text
best QR/Q speedup                  1.032x  (q=37, 4 warps/CTA)
2/4-warp configurations           0.962x to 1.032x
8-warp configurations             0.638x to 0.824x
required to justify integration  >=1.500x
stop threshold                    <1.300x
decision                          STOP_BEFORE_OPTIMIZER_INTEGRATION
```

The result shows that merely packing independent certified QR evaluations into
larger CTAs does not recover the nominally idle second warp: with enough work,
CUDA already overlaps the existing 64-thread target CTAs effectively. The
failure queue adds modest overhead on stable-SVD-heavy shapes, and eight-warp
CTAs reduce useful occupancy. The prototype therefore remains development-only
and is not called by `full_cuda_ci_one_call.cpp`; the optimizer state machine,
candidate identity, and promotion boundary are unchanged. Any next optimizer
route needs new evidence for a materially different decomposition kernel or
work organization, not another direct per-warp CTA packing variant.

Compute-profile v5 now attributes the existing whole-level target prefill
without changing its schedule. A full max-conditioning-size-7 diagnostic run
completed in `269.426` seconds. This is an instrumented development run, not a
new performance claim. Its adjacency, normalized sepsets, pMax,
`n.edgetests`, task structure, and every consumed p-value were bitwise
identical to the `263.305`-second result. CPU numerical authority, fallback,
and residual/component D2H counters remained zero.

The attribution closes every physical optimizer target and boundary:

```text
physical target optimizations             132908
whole-level prefill target optimizations   132908
prefill TargetKeys eventually consumed     110617
prefill TargetKeys never consumed           22291
frontier-required target optimizations          0
singleton requests skipped by prefill           0
singleton padding targets                       0

prefill optimizer host time                162.321 seconds
prefill end-to-end batch wall time          178.574 seconds
frontier optimizer host time                  0.000 seconds
singleton-padding batch host time             0.000 seconds
```

Thus all 22,291 excess optimizations in this run are distinct TargetKeys that
whole-level prefill computed before canonical deletion made them unreachable;
none comes from singleton padding or repeated optimization. The per-window
receipt records level, penalty class, conditioning groups, optimizer setups,
optimized TargetKeys, eventually consumed/unconsumed TargetKeys, and exact
batch/optimizer host time. It deliberately does not assign a fabricated
per-target time to mixed batches.

A development-only zero-lookahead frontier implementation was then tested and
rejected before integration. It optimized only currently ready TargetKeys and
closed its accounting with zero lookahead, zero on-demand optimization, and
zero unconsumed frontier keys. On the canonical max-conditioning-size-3
workload, however, it took `394.993` seconds versus `179.914` seconds for the
whole-level v5 route. Physical target optimizations fell only from `107,053`
to `102,110`, because `14,507` singleton padding targets replaced most of the
removed speculative work. Optimizer boundaries increased from 83 to 1,898 and
setup submissions from 5,239 to 46,691; optimizer host time increased from
`50.112` to `287.610` seconds. The structural task trace, graph, sepsets,
deletion decisions, and `n.edgetests` remained exact, but 120,991 consumed
p-values differed in low bits (maximum absolute difference `3.33e-14`), so the
experiment also failed the bitwise numerical gate. The v5 production scheduler
was restored, and no max-conditioning-size-7 run was attempted.

Do not retry synchronous zero-lookahead or add depth-one/two lookahead directly
to the production scheduler. Any later frontier design must first show, on the
level-3 development workload, that setup-aware coalescing keeps optimizer
boundaries and setup submissions near the v5 shape, does not exchange
speculative targets for singleton padding, preserves the original canonical
group execution shape, and restores bitwise-identical consumed p-values. A
dedicated multi-penalty singleton tail is not justified as a standalone fix;
even eliminating all 14,507 padding targets would leave the repeated setup and
boundary costs dominant.

A subsequent setup-aware opportunity diagnostic rebuilt the original v5 plan
without executing any new CUDA CI work. It used the exported native layer
planner plus the stored canonical `p_used` trace, reconstructed all 8,637 setup
cohorts and 137 original 64-setup windows, and matched every persisted v5
window count exactly. The result closes the scheduler opportunity:

```text
v5 windows                                      137
windows ever demanded                          137
windows never demanded                           0
recorded wall time in skippable windows       0 ms

v5 setup cohorts                              8637
setup cohorts ever demanded                   8634
setup cohorts never demanded                     3
targets in never-demanded cohorts                9

unconsumed targets inside demanded cohorts   22282
total unconsumed targets                     22291
```

Thus 99.96% of the unconsumed targets are distributed inside setup cohorts
whose original windows must execute. At max conditioning size 3, lazy original-
window activation still requires the same 83 optimizer boundaries, 5,239 setup
submissions, and 107,053 target optimizations as v5, with zero recorded batch
wall time available to skip. Frontier density also does not justify dynamic
setup flushing at that checkpoint: newly ready setup cohorts have p95 at most
two, a maximum of 51 at level 3, and no level-3 epoch exposes a full 64-setup
inactive window.

The diagnostic decision is `STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL`. Do not
implement lazy original-window activation or setup-cohort coalescing for this
contract epoch. Preserving the v5 cohort/window numerical shape removes almost
all scheduler-side savings; trimming inside demanded cohorts reopens the
bitwise and batch-amortization failures already observed.

The next diagnostic tested whether the multi-penalty optimizer's accepted
device residual can replace the later fixed-SP residual solve. A real
`q=64`, seven-penalty, two-target fixture compared both residual buffers on the
device and returned only 64 bytes of compact diagnostics. All 702 residual
values differed, both targets failed bitwise parity, the maximum absolute
difference was `0.2399546`, and relative L2 difference was `0.0288145`.
Optimizer and fixed solver statuses were successful and route/status metadata
was consistent, so this is numerical producer drift rather than an error path.

Development-only residual-view entrypoints then fed the still-device-resident
optimizer payload directly into the exact screen and legacy full-eig CUDA
consumers. They bypassed the fixed-SP solve with zero residual payload D2H, but
neither p-value was bitwise identical to its fixed-SP counterpart:

```text
consumer       fixed-SP p-value       optimizer-residual p-value
exact screen   3.58933903915984e-14   6.41771967688960e-14
legacy eig     4.48319202981132e-14   7.89308561131647e-14
```

There were zero decision flips on this fixture, but the current contract
requires consumed p-values to remain bitwise identical. The qualification
decision is therefore `STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY`. A released
optimizer token was also rejected as stale, residual shadow D2H remained zero,
and the original fixed-SP exact/legacy and persistent optimizer tests still
pass. D2D detach and arena retention were not attempted because the numerical
gate fails before memory/lifetime qualification. The production one-call path
still releases optimizer residuals and uses the authenticated fixed-SP solver.
The optimizer residual remains optimizer-internal evidence: it does not define
production ResidualKey authority and must not enter dCov or the residual cache.

A trace-free full level-7 compute-profile v6 then split authoritative fixed-SP
residual time both by penalty class and by consumer stage. Its `274.728`-second
development runtime includes the new diagnostic accounting and is not a new
performance baseline; the best explicit-max-7 development runtime is now
`235.058` seconds, and the default-`Inf` development run is `233.848` seconds.
The graph, sepsets, counts, all 240,489 task rows (including consumed p-values),
and the semantic level trace are bitwise identical to the primitive-cache
artifact. CPU numerical authority, fallback, residual D2H, and component D2H
remain zero.

```text
fixed-SP residual total                  15.315 s
single-penalty                           10.272 s
multi-penalty                             5.043 s
exact screen                             14.833 s
guarded legacy-eig refinement             0.482 s

exact residual batches                   38,613
guarded refinement residual batches         553
exact residual target fits              228,015
guarded refinement residual target fits   9,939
unique ResidualKeys                     110,665
excess exact residual fits              117,350
```

Guarded refinement accounts for only `3.15%` of residual-solve time, below the
`3-5` second implementation gate. Do not build a large exact/refinement sharing
route for a measured upper bound of `0.482` seconds.

A device-only cross-batch qualification then tested repeated, permuted,
subset, singleton, and mixed-route fixed-SP cohorts. Identical and permuted
complete cohorts were bitwise exact. QR and SVD singleton subsets were also
exact, but changing a batched-Cholesky cohort to a subset or singleton changed
913 residual values, 332,530 exact-component values/moments, three solver
statuses, and two unconditional legacy-eig p-values. Final guarded p-values and
decisions happened to remain unchanged, but that does not satisfy the strict
contract. The decision was `CONDITIONAL_ALL_HIT_BATCH_ONLY`: target-granular
miss-only execution is rejected, and only a previously observed complete
cohort is qualified for possible reuse.

The follow-up trace-free level-7 opportunity run closed that remaining case.
Its `266.801`-second runtime contains diagnostic hashing and is not a new
performance baseline. Adjacency, sepsets, pMax, `n.edgetests`, all 240,489 task
rows, and every semantic level field were bitwise identical to the prior v6
artifact; only per-level elapsed times differed. Authority, fallback, residual
D2H, and component D2H remained zero. The four exact-screen categories were:

```text
category                  batches  targets  solve ms  component ms  total ms
all miss                     8925    76750  3330.604      1214.481  4545.086
mixed                       16047    87666  6132.209      1820.599  7952.808
all hit, new cohort         13595    61746  5338.055      1455.815  6793.871
all hit, repeated cohort       46     1853    54.209        17.145    71.354
```

The category counts sum exactly to 38,613 batches and 228,015 targets; their
solve and component times sum to the exact-screen totals. Only the last row is
numerically qualified. Its combined measured upper bound is `71.354` ms, far
below the `5`-second stop gate. The final decision is
`STOP_CROSS_BATCH_FIXED_RESIDUAL_CACHE_OPPORTUNITY`. Do not build a residual
slab, exact-component cache, capacity trace, or whole-cohort reuse prototype
for this contract epoch; the apparent `51.47%` target-level duplication is not
convertible under the required batch-cohort numerical shape.

Persistent dCov resources remain a separate low-risk candidate. The v6 run
records 39,166 dCov calls, `3.440` seconds of teardown, and `7.416` seconds of
dCov host time not attributed to metadata H2D, component kernels, pair/gamma,
or compact D2H. Their sum is an orchestration upper bound, not a promised
speedup. A pure-C++ owned Prepared-S pipeline that overlaps CPU setup with the
unchanged v5 GPU optimizer windows remains the next system-level route. A later
single-matrix QR/Q kernel prototype is still necessary if these routes cannot
close Checkpoint B.

Rebuild the opportunity receipt without CUDA numerical work with:

```bash
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_coalescing_opportunity.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_coalescing_opportunity.R
```

Reproduce the optimizer accepted-residual qualification with:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_optimizer_residual_reuse.R
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tests/test_full_cuda_ci_phase10_optimizer_residual_reuse.R
```

Reproduce the fixed-SP cross-batch identity and opportunity STOP with:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_fixed_residual_identity.R
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tests/test_full_cuda_ci_phase10_fixed_residual_identity.R

CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_fresh_data.R \
  7 \
  fastkpc/artifacts/full_cuda_ci/phase10_profile_v2/fresh-data-level-07-fixed-residual-cohort-opportunity-profile-v6-development.rds \
  development
Rscript \
  fastkpc/tools/profile_full_cuda_ci_phase10_fixed_residual_cohort_opportunity.R \
  fastkpc/artifacts/full_cuda_ci/phase10_profile_v2/fresh-data-level-07-fixed-residual-cohort-opportunity-profile-v6-development.rds
```

Reproduce the exactness and stop/go campaign with:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tests/test_full_cuda_ci_grouped_qr_prototype.R
```

Set `FASTKPC_GROUPED_QR_REQUIRE_GO=1` to make the campaign fail unless the best
measured configuration reaches the `1.5x` integration gate.

Reproduce the development-only reuse trace with a distinct artifact path:

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
FASTKPC_PHASE10_DECOMPOSITION_TRACE_CAPACITY=4096 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_fresh_data.R \
  7 \
  fastkpc/artifacts/full_cuda_ci/phase10_profile_v2/fresh-data-level-07-exact-decomposition-reuse-development.rds \
  development
```

Reproduce the prefill attribution profile with a distinct ignored artifact:

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_fresh_data.R \
  7 \
  fastkpc/artifacts/full_cuda_ci/phase10_profile_v2/fresh-data-level-07-prefill-attribution-profile-v5-development.rds \
  development
```

This is single-run development evidence: Checkpoint A (`< 600` seconds) passes,
while Checkpoint B (`< 180` seconds) and the final five-run 120-second gate do
not. Reproduce the development profile without creating formal evidence with:

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
Rscript fastkpc/tools/profile_full_cuda_ci_phase10_fresh_data.R \
  7 \
  fastkpc/artifacts/full_cuda_ci/phase10_profile_v2/fresh-data-level-07-univariate-primitive-cache-development.rds \
  development
```

Build and focused validation:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_full_cuda_ci_one_call.R
Rscript fastkpc/tests/test_compatible_cuda_wanpdag_authority.R
Rscript fastkpc/tests/test_default_max_conditioning_size.R
FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_default_inf_level9_capacity.R
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  FASTKPC_RUN_FULL_DEFAULT_INF_CUDA_TEST=1 \
  Rscript fastkpc/tests/test_full_cuda_ci_default_inf_production.R
Rscript fastkpc/tests/test_full_cuda_ci_one_call_cache.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_hardening_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_campaign_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_completion_audit.R
```

The completion audit intentionally fails until the v2 fresh-data campaign, the
public 500x50 development fixture, the sealed holdout artifact, and the final
`COMPLETE` roadmap state all exist.

Do not provide a custody envelope or release token for the v1 candidate. The
holdout may be opened once, and only after the v2 performance contract, public
500x50 fixture, fresh-data compute campaign, source/native configuration, and
all contract hashes are frozen. A successful holdout may then recommend this
explicit route; making it the default still requires separate approval.

## Current Scope

Implemented:

- R exact distance-covariance gamma reference: `R/dcov_exact.R`
- Legacy R baseline helpers: `R/legacy_runner.R`
- C++ exact distance-covariance p-value backend
- C++ skeleton MVP with stable level semantics
- R wrappers around the native entry points
- Tests for exact dCov and skeleton behavior
- Opt-in CUDA batched exact dCov backend: `R/cuda_native.R`
- Opt-in CUDA skeleton path using per-level batched dCov and deterministic replay
- CPU-vs-CUDA validation helpers: `R/cuda_validation.R`
- Per-run residual cache for CPU and CUDA skeleton paths
- Residual cache validation helpers: `R/residual_validation.R`
- Opt-in `fastSpline` residual backend on CPU
- Backend-aware CPU/CUDA skeleton wrappers
- fastSpline validation and benchmark helpers: `R/fastspline_validation.R`
- Opt-in native WAN-PDAG orientation stage
- CPU/CUDA skeleton-to-WAN-PDAG pipeline wrappers
- WAN-PDAG validation and benchmark helpers: `R/wanpdag_validation.R`
- Opt-in CUDA `fastSpline` residual device for skeleton residual cache
- Layer-batched CUDA scheduler with legacy-order replay diagnostics
- True-batched CUDA `fastSpline` residual groups for scheduler residual prefetch
- Opt-in CUDA WAN-PDAG orientation residual/dCov execution for fastSpline
- Scheduler validation and benchmark helpers: `R/scheduler_validation.R`
- Opt-in native CPU HSIC gamma and HSIC permutation CI methods
- Opt-in native CPU and fixed-seed CUDA dCov permutation CI method
- CUDA HSIC kernels for `hsic.gamma` and fixed-seed `hsic.perm`
- CUDA HSIC skeleton and WAN-PDAG orientation diagnostics
- CUDA HSIC validation and benchmark helpers: `R/hsic_cuda_validation.R`
- Operational precision-ladder reports, timing attribution, routing policy,
  compatibility envelope, workload structure stats, and true-batched-kernel
  decision artifacts for the fastSplineCUDA / mgcvExtractGPU stack

Not implemented:

- A general-purpose CUDA clone of all mgcv families and formula semantics
- Multi-GPU scheduling
- Replacement of exported `kpcalg::kpc()`
- `tprsApproxCUDA`, unless future attribution evidence reverses the current
  defer decision

## Operational precision ladder

The public backend positioning is:

```text
precision = "fast":
    fastSplineCUDA
    fastest approximate CUDA primary backend
    not mgcv-compatible

precision = "compatible":
    mgcvExtractGPU where the version/semantic envelope is supported
    mgcvExtractCPU / legacy mgcv fallback otherwise

precision = "hybrid":
    fastSplineCUDA primary
    mgcvExtractGPU near-alpha verifier
    canonical replay preserved
```

This section describes the older public precision ladder, not the explicit
full-CUDA candidate above. `mgcvExtractGPU` is a version-pinned compatibility
bridge: mgcv constructs the restricted setup and fastkpc uses GPU numerical
paths where supported. It is not a full mgcv clone and it is not a pure GPU
approximation backend.

The precision `mgcvExtractGPU` executor now uses same-setup x/y pair batching
for selected fixed-sp CUDA solves after per-target spectral GCV selection.
It also uses an on-demand same-S prepared setup/spectral cache inside each run,
so later misses reuse response-free design and spectral state while preserving
target-specific `X'y`, GCV, `sp`, and solve work. This is not eager same-S group
planning/batching, it is not capacity-bounded prepared-cache eviction, and it
is not a true fused/batched GPU kernel. Diagnostics must keep
`true_batched_kernel = false` until a fused kernel exists.

Native CUDA validation remains opt-in. Running
`FASTKPC_RUN_CUDA_TESTS=1 fastkpc/tools/run_mgcv_gate_b_tests.sh` exercises the
native precision E2E gate and writes CPU/GPU parity evidence through
`fastkpc_run_native_cuda_precision_parity()`.

`fastSplineCUDA` remains the frozen approximate baseline. `tprsApproxCUDA`
remains deferred unless projection-floor, oracle-lambda, timing, and graph-level
evidence justify building a new pure GPU approximation.

Operational artifacts can be generated with:

```bash
Rscript fastkpc/tools/run_precision_ladder_summary_report.R
Rscript fastkpc/tools/run_precision_ladder_timing_campaign.R
Rscript fastkpc/tools/run_hybrid_policy_calibration_report.R
Rscript fastkpc/tools/run_workload_structure_stats.R
Rscript fastkpc/tools/run_true_batched_kernel_decision.R
FASTKPC_RUN_CUDA_TESTS=1 fastkpc/tools/run_precision_end_to_end_benchmark.sh
```

The end-to-end benchmark writes `runs.csv`, `stage_timing.csv`, `cache.csv`,
`graph_agreement.csv`, `mode_summary.csv`, `comparison_summary.csv`,
`bottleneck_decision.csv`, and `summary.{json,md}` under
`fastkpc/artifacts/precision_end_to_end_benchmark/`. It compares legacy mgcv,
fastSplineCUDA, precision-scheduler primary-only CUDA, compatible CUDA, and
hybrid CUDA rows, with warm-up, randomized measured mode order, and robust
per-mode summaries. Set `FASTKPC_PRECISION_E2E_REAL_DATA=/path/to/data.csv` or
`.rds` to append an external numeric workload.

CUDA-specific tests remain opt-in. GitHub Actions are intentionally absent
unless reintroduced by explicit request.

The precision policy and skeleton data plane are now wired into `fast_kpc()`:

```text
precision = "fast":
    preserves fastSpline primary execution

precision = "compatible":
    routes through the authoritative resolver
    fails closed when semantic/version/runtime envelope checks fail
    executes CPU and CUDA skeleton data-plane slices for |S| <= 2
    uses mgcvExtractGPU where supported
    falls back through mgcvExtractCPU/GCVBridge and legacy mgcv

precision = "hybrid":
    keeps fastSpline primary execution
    executes near-alpha verifier residualization for skeleton |S| <= 2
    records verifier and fallback receipts
    preserves canonical replay
    uses verifier p-values in real skeleton edge/sepset decisions
```

The default remains the existing behavior until held-out validation is accepted.
The full-CUDA candidate is selected only through its explicit facade and does
not alter these `fast_kpc()` precision routes.

Diagnostics distinguish `backend_planned` from `backend_executed`;
`backend_used` refers to the actual executor. Current precision data-plane scope
is skeleton only, CPU/CUDA, and single-penalty `|S| <= 2`. On-demand same-S
prepared setup/spectral caching is implemented for the precision GPU executor;
eager same-S group planning/batching, capacity-bounded prepared-cache eviction,
WAN-PDAG, `|S| > 2` multi-penalty GCV, and true fused/batched
`mgcvExtractGPU` kernels remain future work pending native CUDA parity and
timing evidence.

## Build

The first slice uses `Rcpp::sourceCpp()` rather than a package build system.
The wrapper builds the native code on first use:

```bash
Rscript -e 'source("fastkpc/R/native.R"); build_fastkpc_native(rebuild=TRUE)'
```

`sourceCpp()` creates object files in `fastkpc/src` during this local build.
Those files are build artifacts, not source deliverables; rebuild them with the
command above after removing them.

## CUDA Scope

The CUDA backend is opt-in through `fastkpc/R/cuda_native.R`. It adds:

- `fast_dcov_batch_cuda()`: evaluates many exact dCov gamma tests in one CUDA call.
- `fast_skeleton_cuda()`: runs the C++ skeleton MVP with per-level CUDA dCov batches.

The CUDA skeleton preserves the CPU exact MVP's stable-level replay semantics:
candidate tests are batched for evaluation, then replayed in deterministic CPU
order so adjacency, sepsets, pMax, and `n.edgetests` match the CPU reference.

## Layer-batched CUDA scheduler

The CUDA skeleton backend supports an explicit scheduler mode:

```r
fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           scheduler = "layer")
fast_kpc(data, engine = "cuda", scheduler = "layer")
```

Scheduler modes:

```text
scheduler="layer": build one task plan per PC skeleton level, prefetch residuals,
                   evaluate dCov in CUDA batches, then replay in legacy order.
scheduler="legacy": keep the previous CUDA skeleton loop as a reference path.
scheduler="auto": use layer scheduling for CUDA skeleton execution and legacy
                  scheduling for CPU execution.
```

`scheduler="layer"` preserves legacy replay order. It may evaluate speculative
tasks, but only accepted replayed tests update adjacency, sepsets, pMax, and
`n.edgetests`. `scheduler="legacy"` remains available as a reference path for
graph-level comparisons.

`residual_backend` and `residual_device` keep their previous meanings:
`residual_backend` selects the statistical residual model, while
`residual_device` selects where supported residual fits execute. The
`residual_batch_size` option controls materialization grouping for residual
vectors; it does not change the statistical residual model. `batch_size`
continues to control dCov batch grouping.

## Scheduler diagnostics

CUDA skeleton results include:

```text
scheduler
scheduler_requested
scheduler_diagnostics$summary
scheduler_diagnostics$levels
scheduler_diagnostics$batches
scheduler_diagnostics$residuals
```

The diagnostics distinguish planned work, evaluated work, replayed tests, and
ignored speculative work. The summary records counts such as `tasks_planned`,
`tasks_evaluated`, `tests_replayed`, `tasks_ignored_after_delete`,
`dcov_batches`, `unique_residual_requests`, and `residual_batches`.

## CI methods

The default conditional-independence method remains exact dCov gamma:

```r
fast_kpc(data, ci_method = "dcc.gamma")
```

Permutation dCov and HSIC methods are opt-in. Set `ci_diagnostics = TRUE` to
keep CI method counts and backend-resolution fields in the result:

```r
fast_kpc(data, ci_method = "hsic.gamma", hsic_params = list(sig = 1))
fast_kpc(data, ci_method = "dcc.perm",
         permutation_params = list(replicates = 100, seed = 42,
                                   include_observed = TRUE))
fast_kpc(data, ci_method = "hsic.perm",
         permutation_params = list(replicates = 100, seed = 42,
                                   include_observed = TRUE))
```

Native HSIC uses dense RBF Gram matrices on CPU. Conditional tests reuse the
selected residual backend: raw vectors are tested when the conditioning set is
empty, and residual vectors are tested otherwise. CUDA dCov is used for
`ci_method = "dcc.gamma"` and fixed-seed `ci_method = "dcc.perm"`.

## CUDA CI kernels

CUDA HSIC kernels are available for CUDA runs with `ci_method = "hsic.gamma"`
and for `ci_method = "hsic.perm"` when an explicit permutation seed is supplied.
Successful GPU execution records `ci_backend = "cuda-hsic"` and positive CUDA
HSIC counters such as `ci_hsic_cuda_batches` and `ci_hsic_cuda_pairs`.

CUDA dCov permutation is available with `ci_method = "dcc.perm"` when an
explicit permutation seed is supplied. Successful execution records
`ci_backend = "cuda-dcov"`, positive `ci_dcc_perm_cuda_tests`, and the exact
number of executed permutation replicates. A missing seed fails closed to the
native CPU path with an explicit backend reason, as for CUDA HSIC permutation.
The observed dCov statistic follows `energy::dcov.test`, while fixed-seed
fastkpc runs use a deterministic CPU/CUDA permutation plan. That plan does not
claim bitwise identity with the legacy `energy` global-RNG permutation stream.

The specialized Phase 9/10 full-CUDA one-call path used by the 351 x 48
performance campaign remains separate from the generic CUDA path. Its strict
one-call skeleton supports `dcc.gamma`, `dcc.perm`, `hsic.gamma`, and
`hsic.perm`; use `fastkpc_compatible_cuda_wanpdag()` when the final graph must
also follow the kpcalg WAN-PDAG authority. The generic native WAN-PDAG path is a
separate implementation and does not inherit that strict claim.

The R precision data plane also accepts all four methods. In that path,
`engine = "cuda"` may use CUDA residualization while reporting
`ci_backend = "native-cpu"` for these opt-in CI methods; this is distinct from
the generic CUDA skeleton path, which executes `dcc.perm` on `cuda-dcov` and
HSIC on `cuda-hsic`.

`hsic.perm` on CUDA requires an explicit permutation seed. If the seed is
missing, the run uses native-cpu fallback and records the reason
`CUDA HSIC permutation requires explicit seed in this stage`. Other unsupported
conditions, such as configured sample-size limits, also use `ci_backend =
"native-cpu"` with `ci_backend_reason` unless `cuda_memory_fallback = FALSE`.

Useful CUDA HSIC controls and diagnostics:

```r
fast_kpc(
  data,
  engine = "cuda",
  ci_method = "hsic.gamma",
  hsic_params = list(sig = 1,
                     cuda_max_n = 2048,
                     cuda_max_batch_pairs = 64,
                     cuda_memory_fallback = TRUE)
)

fast_kpc(
  data,
  engine = "cuda",
  ci_method = "hsic.perm",
  permutation_params = list(replicates = 100, seed = 42,
                            include_observed = TRUE)
)
```

Top-level results expose `config$ci_backend`, `config$cuda_hsic_used`,
`diagnostics$cuda_hsic_available`, and `diagnostics$cuda_hsic_reason`.
Skeleton results expose `skeleton$ci_backend`,
`skeleton$ci_backend_reason`, and `skeleton$ci_diagnostics`.
WAN-PDAG orientation results expose `orientation$ci_backend`,
`orientation$ci_backend_reason`, and `orientation$ci_diagnostics`.

Validation reports include `hsic_cuda_backend_diagnostics.csv`,
`hsic_cuda_cpu_fallbacks.csv`, and `hsic_cuda_perf.csv`.
The legacy `kpcalg/R/*.R` files are not modified.

Validation campaigns and reports include `ci_method`, `ci_method_diffs.csv`,
`ci_method_diagnostics.csv`, `hsic_cuda_backend_diagnostics.csv`,
`hsic_cuda_cpu_fallbacks.csv`, and `hsic_cuda_perf.csv`. Helper functions in
`R/hsic_validation.R` validate native HSIC numerics and deterministic
permutation behavior. Helper functions in `R/hsic_cuda_validation.R` validate
CUDA HSIC numerics, fixed-seed CUDA permutation behavior, CPU/CUDA skeleton
agreement, and non-strict benchmark timings.

Validation campaigns can compare schedulers with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = "cuda",
  schedulers = c("legacy", "layer")
)
```

Generated reports include `scheduler_diffs.csv`, `scheduler_levels.csv`,
`scheduler_batches.csv`, `scheduler_residuals.csv`, and
`true_batched_residuals.csv`.

## True-Batched CUDA fastSpline Residuals

`fastspline_residual_batch_cuda()` groups residual requests by sample count,
fastSpline parameter set, and design column count. Compatible groups with two or
more fits run through a grouped CUDA path for crossproducts, penalized systems,
cuSOLVER batched Cholesky solves, GCV scoring, and residual computation.
Singleton groups keep using the single-fit CUDA path.

Batch results include `batch_diagnostics`:

```text
requested_fits
groups
true_batched_groups
true_batched_fits
single_fit_calls
cpu_fallback_fits
unique_designs
duplicate_design_fits
max_fits_per_design
group_table
```

Layer scheduler summaries include the corresponding residual counters:

```text
cuda_residual_batch_groups
cuda_residual_true_batched_groups
cuda_residual_true_batched_fits
cuda_residual_single_fit_calls
cuda_residual_cpu_fallback_fits
cuda_residual_unique_designs
cuda_residual_duplicate_design_fits
cuda_residual_max_fits_per_design
```

`residual_batch_size = 0` lets a scheduler level materialize all unique
compatible residual requests together. `residual_batch_size = 1` remains useful
as a one-at-a-time validation reference. If a requested design exceeds the
current true-batch solver limit or a CUDA batch solve fails, `fallback = TRUE`
uses the CPU fastSpline residual path and records the fallback reason; with
`fallback = FALSE`, the batch call errors.

Run a focused validation:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); x <- validate_cuda_fastspline_residual_batch(); print(x$cases); print(x$batch_diagnostics[c("groups","true_batched_groups","true_batched_fits","single_fit_calls")])'
```

Run a non-strict timing smoke:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); print(benchmark_cuda_fastspline_residual_batch(repeats=3)$summary)'
```

## CUDA WAN-PDAG Orientation Residuals

WAN-PDAG graph mutation remains sequential and deterministic, but the numerical
work inside generalized `regrVonPS` checks can now run on CUDA for CUDA
fastSpline WAN-PDAG runs:

```r
fast_kpc(
  data,
  engine = "cuda",
  residual_backend = "fastSpline",
  residual_device = "cuda",
  orientation_residual_device = "cuda",
  orientation_batch_size = 0,
  graph_stage = "wanpdag"
)
```

`orientation_residual_device` controls only the orientation residual/dCov work.
It does not change skeleton residual-device selection. Supported values are:

```text
orientation_residual_device = "auto" | "cpu" | "cuda"
```

`"auto"` resolves to CUDA for CUDA fastSpline WAN-PDAG runs and to CPU for CPU
engine, skeleton-only runs, or linear residuals. A linear request for
`orientation_residual_device = "cuda"` resolves to CPU with reason
`linear orientation residual CUDA device is not implemented`.

`orientation_batch_size = 0` batches all residual-vs-S dCov pairs in one
orientation check. `orientation_batch_size = 1` is the one-at-a-time validation
reference. The graph search order, accepted orientations, rule replay, and pdag
updates remain CPU-ordered and deterministic.

`orientation_diagnostics = TRUE` returns orientation-device resolution,
`regrVonPS` CPU/CUDA counters, residual fit counters, dCov batch/pair counters,
fallback counters, and orientation cache counters.

Orientation diagnostics are returned in:

```text
result$config$orientation_residual_device_requested
result$config$orientation_residual_device_used
result$config$orientation_residual_device_reason
result$orientation$residual_device
result$orientation$diagnostics
```

Key counters include:

```text
regrvonps_calls
regrvonps_cuda_calls
regrvonps_cpu_calls
orientation_dcov_batches
orientation_dcov_pairs
orientation_residual_fits
orientation_cuda_residual_fits
orientation_cpu_fallback_fits
orientation_cache_requests
orientation_cache_hits
orientation_cache_computations
```

Validation campaigns can compare orientation devices with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = "cuda",
  orientation_residual_devices = c("cpu", "cuda")
)
```

Generated reports include `orientation_device_diffs.csv` and
`orientation_device_diagnostics.csv`.
`kpcalg/R` files are not modified by fastkpc staged backend work.

## CUDA Build

Build CUDA native code:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
```

Or build through R:

```bash
Rscript -e 'source("fastkpc/R/cuda_native.R"); build_fastkpc_cuda_native(rebuild=TRUE)'
```

The build uses `/usr/local/cuda/bin/nvcc`, `-arch=sm_89`, and writes
`fastkpc/build/fastkpc_cuda.so`.

`fastkpc/build/*.o` and `fastkpc/build/fastkpc_cuda.so` are local build
artifacts. They can be removed with:

```bash
bash fastkpc/tools/clean_cuda_native.sh
```

## CUDA Tests

Run CUDA tests serially:

```bash
Rscript fastkpc/tests/test_cuda_build_contract.R
Rscript fastkpc/tests/test_dcov_cuda_batch.R
Rscript fastkpc/tests/test_skeleton_cuda_batch.R
```

## CPU-vs-CUDA Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/cuda_validation.R"); print(validate_cuda_dcov_batch()); print(validate_cuda_skeleton_scenario())'
```

Expected fixed-scenario graph behavior:

```text
adjacency_identical TRUE
sepsets_identical TRUE
n_edgetests_identical TRUE
max_abs_pmax_diff < 1e-8
```

## Residual Cache Scope

The residual cache is shared by the CPU exact skeleton and CUDA skeleton paths.
It caches conditional residual vectors within one skeleton run using:

```text
target variable
sorted conditioning set
residual backend name
residual backend parameters
sample count
variable count
```

Unconditional tests do not use the cache. The supported native residual
backends are:

```text
linear:     intercept=true;ridge=1e-8
fastSpline: degree=3;knots=10;lambda_grid=1e-4:1e4:25;ridge=1e-8;mode=auto
```

## Residual Cache API

CPU:

```r
source("fastkpc/R/native.R")
fast_skeleton_cpp_cached(data, alpha, max_conditioning_size,
                         residual_cache = TRUE)
```

CUDA:

```r
source("fastkpc/R/cuda_native.R")
fast_skeleton_cuda_cached(data, alpha, max_conditioning_size,
                          batch_size = 0,
                          residual_cache = TRUE)
```

Both return the normal skeleton result plus:

```text
residual_backend
residual_cache$enabled
residual_cache$requests
residual_cache$hits
residual_cache$misses
residual_cache$computations
residual_cache$stored_vectors
residual_cache$stored_values
residual_cache$backend_name
```

## Residual Cache Tests

Run:

```bash
Rscript fastkpc/tests/test_residual_cache_core.R
Rscript fastkpc/tests/test_skeleton_residual_cache.R
Rscript fastkpc/tests/test_cuda_residual_cache.R
```

## Residual Cache Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/native.R"); source("fastkpc/R/cuda_native.R"); source("fastkpc/R/residual_validation.R"); print(validate_cpu_residual_cache()); print(validate_cuda_residual_cache())'
```

Expected:

```text
adjacency_identical TRUE
sepsets_identical TRUE
n_edgetests_identical TRUE
cache_stats$hits > 0
cache_stats$computations < cache_stats$requests
```

## Residual Cache Known Limits

- The cache is per skeleton run and is not global across R calls.
- The cache does not implement mgcv equivalence.
- The cache does not implement GPU residualization.
- The cache does not change exported `kpcalg::kpc()`.

## fastSpline Residual Backend Scope

`fastSpline` is an opt-in residual backend for conditional tests. It uses a
native C++ cubic spline design and ridge-regularized penalized least squares:

```text
|S| = 1: cubic spline basis
|S| = 2: tensor-product cubic spline basis
|S| > 2: additive cubic spline blocks
lambda: fixed log grid selected by GCV
```

CUDA skeleton runs may use `fastSpline`, but residuals are still computed on
CPU and then packed into CUDA dCov batches. No CUDA spline kernels are
implemented in this stage.

## fastSpline API

CPU backend-aware skeleton:

```r
source("fastkpc/R/native.R")
fast_skeleton_cpp_backend(data, alpha, max_conditioning_size,
                          residual_backend = "fastSpline",
                          residual_cache = TRUE)
```

CUDA backend-aware skeleton:

```r
source("fastkpc/R/cuda_native.R")
fast_skeleton_cuda_backend(data, alpha, max_conditioning_size,
                           residual_backend = "fastSpline",
                           residual_cache = TRUE,
                           batch_size = 0)
```

One-off residual fit:

```r
source("fastkpc/R/native.R")
fastspline_residual(y, S)
```

Both skeleton wrappers return the normal skeleton result plus:

```text
residual_backend
residual_backend_params
residual_cache
```

## fastSpline Tests

Run:

```bash
Rscript fastkpc/tests/test_fastspline_basis.R
Rscript fastkpc/tests/test_fastspline_solver.R
Rscript fastkpc/tests/test_residual_backend_registry.R
Rscript fastkpc/tests/test_skeleton_fastspline_cpu.R
Rscript fastkpc/tests/test_skeleton_fastspline_cuda.R
Rscript fastkpc/tests/test_fastspline_mgcv_validation.R
Rscript fastkpc/tests/test_fastspline_benchmark.R
```

## fastSpline Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/fastspline_validation.R"); print(validate_fastspline_against_mgcv()); print(compare_fastspline_linear_graph()); print(compare_fastspline_cpu_cuda_graph())'
```

Expected fixed-scenario behavior:

```text
mgcv one-dimensional residual correlation >= 0.97
mgcv two-dimensional residual correlation >= 0.85
CPU-vs-CUDA fastSpline adjacency identical
CPU-vs-CUDA fastSpline max_abs_pmax_diff < 1e-8
```

Graph differences from legacy `mgcv` are reported, not hidden.

## fastSpline Benchmark

Run:

```bash
Rscript -e 'source("fastkpc/R/fastspline_validation.R"); print(benchmark_fastspline_backends())'
```

The helper returns timing rows, cache stats, a linear-vs-fastSpline graph diff,
and a CPU-vs-CUDA fastSpline graph diff.

## fastSpline Known Limits

- fastSpline is opt-in and not the default backend.
- fastSpline is not mgcv and graph differences from legacy mgcv can occur.
- fastSpline residuals are computed on CPU even when CUDA dCov is used.
- No CUDA spline kernels are implemented in this goal.
- No WAN-PDAG migration is implemented in this goal.
- `kpcalg::kpc()` is not replaced.

## WAN-PDAG Orientation Scope

WAN-PDAG orientation is implemented as an opt-in fastkpc stage. It consumes the
native skeleton result, builds a partially directed integer adjacency matrix,
orients collider triples, applies the native orientation rules, and runs the
generalized transitive orientation checks with the existing residual backend and
per-run residual cache.

The strict compatible route is separate: `fastkpc_compatible_cuda_wanpdag()`
uses the full-CUDA skeleton but delegates orientation to the original kpcalg R
implementation. It records the complete `regrVonPS` CI trace and is the
production authority when process-level parity is required. Native CPU/CUDA
orientation remains useful for development and small fixtures, but is not
strict-compatible on every canonical method.

The orientation engine lives under `fastkpc/src` and does not modify legacy
package files. kpcalg/R/*.R files are not modified, and kpcalg::kpc() is not replaced.

## WAN-PDAG API

Strict CUDA skeleton plus kpcalg-authority orientation:

```r
source("fastkpc/R/fast_kpc.R")
fastkpc_compatible_cuda_wanpdag(
  data,
  alpha = 0.1,
  options = list(
    max_conditioning_size = Inf,
    numCol = 35L,
    trace_level = "logical",
    ci_method = "dcc.perm",
    permutation_params = list(
      replicates = 100L, seed = 707L, include_observed = TRUE
    )
  )
)
```

CPU skeleton plus native orientation:

```r
source("fastkpc/R/native.R")
fast_kpc_wanpdag_cpp(data, alpha, max_conditioning_size,
                     residual_backend = "fastSpline",
                     residual_cache = TRUE)
```

Orient an existing fastkpc skeleton:

```r
source("fastkpc/R/native.R")
fast_orient_wanpdag_cpp(skeleton_result, data,
                        residual_backend = "fastSpline",
                        residual_cache = TRUE)
```

CUDA skeleton plus the same native orientation engine:

```r
source("fastkpc/R/cuda_native.R")
fast_kpc_wanpdag_cuda(data, alpha, max_conditioning_size,
                      residual_backend = "fastSpline",
                      residual_cache = TRUE,
                      batch_size = 0)
```

The returned object has:

```text
skeleton
orientation$pdag
orientation$events
orientation$counts
orientation$residual_cache
```

## WAN-PDAG Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(validate_wanpdag_against_legacy()); print(compare_wanpdag_cpu_cuda())'
```

`validate_wanpdag_against_legacy()` compares against
`kpcalg::udag2wanpdag()` when `pcalg` and `graph` are available. If those
packages are missing, the helper returns `available = FALSE` with an explicit
missing-package reason and still reports native event counts, cache stats, and
a deterministic hand-written fixture check.

`compare_wanpdag_cpu_cuda()` verifies that CPU and CUDA skeleton paths feed the
same native orientation stage and reports pdag equality, orientation-count
equality, skeleton pMax drift, and cache stats.

## WAN-PDAG Benchmark

Run:

```bash
Rscript -e 'source("fastkpc/R/wanpdag_validation.R"); print(benchmark_wanpdag_pipelines())'
```

The helper returns timing rows, orientation cache rows, orientation counts, a
CPU-vs-CUDA WAN-PDAG diff, and a linear-vs-fastSpline WAN-PDAG diff.

## WAN-PDAG Known Limits

- WAN-PDAG orientation is opt-in.
- `kpcalg::kpc()` is not replaced.
- kpcalg/R/*.R files are not modified.
- HSIC gamma and HSIC permutation tests are implemented as opt-in native CPU
  CI methods and as CUDA HSIC kernels for supported CUDA runs.
- CUDA orientation residual/dCov execution is opt-in and limited to fastSpline.
- WAN-PDAG graph mutation remains sequential; CUDA is used only for numerical checks.
- `unfVect` conservative/majority exclusions are accepted as input but not fully implemented in native orientation.

## Public fast_kpc API

`R/fast_kpc.R` provides the opt-in public wrapper over the completed fastkpc
CPU/CUDA skeleton and WAN-PDAG stages:

```r
source("fastkpc/R/fast_kpc.R")
result <- fast_kpc(
  data,
  alpha = 0.2,
  max_conditioning_size = 2,
  engine = "auto",
  residual_backend = "fastSpline",
  graph_stage = "wanpdag"
)
```

`engine = "auto"` uses CUDA only when `fastkpc_cuda_available()` succeeds;
otherwise it falls back to CPU. `graph_stage = "skeleton"` returns the skeleton
only, while `graph_stage = "wanpdag"` also returns native orientation output.

## fastkpc_result

`fast_kpc()` returns class `fastkpc_result`. The normalized result contains:

```text
config
data_info
engine
skeleton
orientation
metrics
timings
cache
validation
benchmark
diagnostics
```

Helpers:

```r
validate_fastkpc_result(result)
fastkpc_result_summary(result)
fastkpc_extract_skeleton(result)
fastkpc_extract_pdag(result)
```

`print.fastkpc_result()` and `summary.fastkpc_result()` provide compact
interactive views.

## Validation Campaign

`R/validation_scenarios.R` defines deterministic synthetic scenarios:

```text
chain
fork
collider
independent
additive
```

`R/validation_campaign.R` runs reproducible CPU/CUDA and residual-backend
comparisons:

```r
source("fastkpc/R/validation_campaign.R")
campaign <- run_fastkpc_validation_campaign(
  seeds = c(11, 12, 13),
  n_values = c(80, 140),
  scenarios = c("chain", "fork", "collider", "independent", "additive"),
  engines = c("cpu", "cuda"),
  residual_backends = c("linear", "fastSpline"),
  legacy = TRUE,
  benchmark = TRUE
)
```

The campaign reports run rows, graph metrics, pairwise diffs, CPU-vs-CUDA
diffs, linear-vs-fastSpline diffs, legacy diagnostics, timing rows, cache rows,
orientation counts, and errors. Missing `pcalg` or `graph` packages are
reported as explicit legacy diagnostics, not as package-independent failures.

## Validation Reports

`R/report_writer.R` writes campaign artifacts:

```r
source("fastkpc/R/report_writer.R")
write_fastkpc_validation_report(campaign, "fastkpc/reports/example")
```

`write_fastkpc_validation_report()` writes `summary.md`, CSV tables for each
campaign section, and `campaign.rds`.

## Command Line Tools

Run one public-wrapper job from CSV:

```bash
Rscript fastkpc/tools/run_fast_kpc.R \
  --input data.csv \
  --output result.rds \
  --engine cpu \
  --residual-backend fastSpline \
  --alpha 0.2 \
  --max-conditioning-size 1 \
  --graph-stage wanpdag
```

Run a validation campaign and write a report:

```bash
Rscript fastkpc/tools/run_validation_campaign.R \
  --output-dir fastkpc/reports/example \
  --seeds 11,12 \
  --n-values 80 \
  --scenarios chain,independent \
  --engines cpu,cuda \
  --residual-backends linear,fastSpline \
  --orientation-residual-devices cpu,cuda \
  --alpha 0.2 \
  --max-conditioning-size 1 \
  --legacy TRUE
```

## Public Wrapper Known Limits

- kpcalg::kpc() is not replaced.
- kpcalg/R/*.R files are not modified.
- CUDA residual kernels are opt-in.
- Validation campaign reports graph differences; it does not force equality.
- Legacy comparison requires pcalg and graph.

## CUDA Residual Device

CUDA residual kernels are opt-in through the `residual_device` argument. The
statistical residual model remains selected by `residual_backend`, while
`residual_device` selects where supported residual fits run:

```text
residual_backend = "fastSpline"
residual_device = "auto" | "cpu" | "cuda"
```

`residual_device = "cuda"` uses CUDA fastSpline residual fitting in the CUDA
skeleton path. `residual_device = "cpu"` keeps the previous CPU residual path.
`residual_device = "auto"` resolves to CUDA for CUDA fastSpline skeleton runs
when CUDA is available. linear residual CUDA device is not implemented; linear
residual requests resolve to CPU with an explicit diagnostic reason.

## Standalone CUDA Residual API

Standalone helpers are available for validating residual numerics directly:

```r
source("fastkpc/R/cuda_native.R")
fit <- fastspline_residual_cuda(y, S, fallback = TRUE)
batch <- fastspline_residual_batch_cuda(data, targets, conditioning_sets)
```

`fastspline_residual_cuda()` returns residuals, fitted values, selected lambda,
GCV, RSS, EDF, design column count, ridge attempts, backend, residual backend,
residual device, fallback status, and diagnostics.

`fastspline_residual_batch_cuda()` returns the same quantities for a batch of
target/conditioning-set residual fits.

## Residual Device In fast_kpc

The public wrapper accepts:

```r
fast_kpc(data,
         engine = "cuda",
         residual_backend = "fastSpline",
         residual_device = "cuda")
```

`fastkpc_result$config` records `residual_device_requested` and
`residual_device_used`. CUDA residual fallback can resolve the used value to
`cuda-fallback-cpu` when fallback is enabled and a CUDA residual fit fails.

## CUDA Residual Validation

Run:

```bash
Rscript -e 'source("fastkpc/R/cuda_residual_validation.R"); print(validate_cuda_fastspline_residuals()); print(benchmark_cuda_fastspline_residuals(repeats=2))'
```

Expected fixed-scenario behavior:

```text
max_abs_residual_diff < 1e-7
max_abs_fitted_diff < 1e-7
relative_rss_diff < 1e-8
```

Validation campaigns can compare residual devices with:

```r
run_fastkpc_validation_campaign(
  engines = "cuda",
  residual_backends = "fastSpline",
  residual_devices = c("cpu", "cuda")
)
```

Generated reports include `residual_device_diffs.csv`.

## CUDA Residual Known Limits

- CUDA residual kernels are opt-in.
- linear residual CUDA device is not implemented.
- CUDA residual fallback is explicit and may report `cuda-fallback-cpu`.
- CUDA WAN-PDAG orientation residuals are opt-in and limited to fastSpline.
- kpcalg::kpc() is not replaced.
- kpcalg/R/*.R files are not modified.

## mgcv-Compatible Residual Oracle And Hybrid Verification

fastSpline CUDA is a high-throughput approximate backend. It is useful as the
primary fast path for batched kPC conditional-independence workloads, but it is
not mgcv-equivalent and graph differences from legacy `mgcv::gam()` residuals
can occur.

mgcvExtractCPU is a version-pinned extraction oracle for compatibility
validation. It is restricted to the `kpcalg::regrXonS()` residualization
surface, may depend on `mgcv` internals, and is not the final portable product
backend. The protected baseline is:

```text
Baseline: mgcv Gate B fixed-sp self-solve + hybrid canonical replay
Commit: 5da2313
Tag: mgcv-gate-b-v1
```

`fastkpc_mgcv_extract_capabilities()` returns a machine-readable capability
and version boundary object for diagnostics, bug reports, and campaign output.

The mgcv fixed-sp reference calls `mgcv::gam(..., sp=sp, fit=TRUE)`.
The mgcvExtract fixed-sp self-solve uses `mgcv::gam(fit=FALSE)` setup data,
then solves the fixed-sp Gaussian penalized least-squares problem inside
fastkpc. For mgcv parity, that setup solve follows mgcv's all-fixed
`L/lsp0` semantics and records the version-pinned `mgcv-C-magic-fixed-sp`
kernel path in diagnostics; it does not call `mgcv::gam(fit=TRUE)` or
`mgcv::magic()`. mgcvExtractGCVBridge uses mgcv for smoothing-parameter
selection and fastkpc only for the fixed-sp solve; it is not a self-contained
GCV implementation.

The restricted compatibility target is:

```text
|S| == 0: direct CI test; no residualization
|S| <= 2: X_i ~ s(S variables jointly)
|S| > 2:  X_i ~ s(S_1) + s(S_2) + ... + s(S_k)
family: Gaussian identity
output: residuals only
```

The default s(s1, s2) is not a tensor-product smooth. It is mgcv's default
isotropic smooth semantics. Tensor-product smooths such as `te`, `ti`, or `t2`
are not part of legacy `kpcalg::regrXonS()` formula construction.

The near-alpha verifier runs a fast primary backend first and verifies tests
whose p-values are close to alpha on a log scale. Verification may replace the
p-value source, but it must preserve canonical edge and sepset replay order.

Gate B campaign:
`fastkpc/tools/run_mgcv_gate_b_campaign.sh` runs fixed-sp setup/self-solve
parity scenarios across formula classes, smoothing-parameter scales, sample
sizes, collinearity, near-constant conditioning variables, and tied values. It
writes `mgcv_gate_b_fixed_sp_campaign.csv`.

canonical hybrid replay:
the near-alpha verifier may replace p-values but not replay order. Primary rows
define `canonical_test_order_id`; verifier rows are joined by that id and
replayed deterministically. Sepsets are recorded from the canonical first
separating set after p-value replacement.

hybrid calibration campaign:
`fastkpc/tools/run_hybrid_calibration_campaign.sh` evaluates tau values such as
`log(1.5)`, `log(2)`, `log(3)`, and `log(5)` and writes
`hybrid_calibration_summary.csv` plus `hybrid_policy_summary.txt`. The summary
tracks near-alpha verifier rate, decision flip reduction, skeleton/sepset/WAN-
PDAG drift proxies, runtime proxies, speedup versus legacy, and a recommended
tau per scenario group.

graph-level golden snapshots:
`fastkpc_hybrid_golden_snapshots()` provides deterministic canonical replay
snapshots for linear, nonlinear additive, pairwise full-smooth, and near-alpha
flip scenarios. These protect test order, p-value source selection, edge
deletion logs, sepsets, skeleton adjacency, and WAN-PDAG adjacency summaries.

Non-goals:

```text
No full mgcv clone
No bamGPU
No non-Gaussian family
No summary.gam/vcov/SE/prediction interval compatibility
No GAMM
No by-smooth or factor-smooth support
No pretending fastSpline is mgcv-equivalent
No sharing smoothing parameters across targets
```

## Tests

Run all first-slice tests:

```bash
Rscript fastkpc/tests/test_dcov_exact.R
Rscript fastkpc/tests/test_skeleton_mvp.R
Rscript fastkpc/tests/test_diff_report.R
```

Run these commands serially. The current first-slice native wrapper uses
`Rcpp::sourceCpp()`, and parallel test processes can race while building object
files in `fastkpc/src`.

Run the external GPU numeric baseline:

```bash
Rscript gpu-dcov/validate.R
```

## Behavior Notes

The exact dCov backend removes the old `RSpectra::eigs()` truncation from
`kpcalg::dcov.gamma`. It computes the full double-centered distance covariance
statistic:

```text
nV2 = sum((HKH) * (HLH)) / n
```

The `legacy_index` option controls the old kpcalg index behavior:

- `legacy_index = TRUE`: preserve old behavior and ignore the distance exponent.
- `legacy_index = FALSE`: apply the documented distance exponent.

The skeleton MVP uses a native linear residualization path for conditional
tests. This validates scheduler mechanics and is not a replacement for the
legacy `mgcv` residual backend.

`R/legacy_runner.R` provides optional helpers for running the legacy kpcalg R
functions from `kpcalg/R` as validation baselines. They do not modify or replace
the legacy package files.
