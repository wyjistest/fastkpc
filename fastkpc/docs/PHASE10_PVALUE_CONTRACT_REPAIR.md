# Phase 10 p-value contract repair

## Status

The historical `dcc.gamma` Phase 10 route is invalid under
`numerical_contract_v1`. Graph parity and zero decision flips did not satisfy
the contract requiring every canonical CI p-value and every finite `pMax` cell
to have absolute error at most `1e-10` from the legacy reference.

```text
historical Phase 10 candidate: INVALIDATED_NUMERICAL_CONTRACT
GPU-only correctness repair:  FRESH_REBUILD_CANONICAL_PASS
full-eig component status:     CLOSED_STRICT_SINGLE_GPU
Checkpoint B <180s:             FAIL
candidate:                      NOT_FROZEN
recommended route:              false
sealed holdout:                 SEALED_NOT_RELEASED
```

## Defect

The invalid route ran an exact CUDA screen for every pair and invoked the CUDA
legacy full-eigensolver only when the screen p-value was in `[0.05, 0.15]`.
Pairs outside that interval returned the screen p-value directly. A
retrospective canonical audit found:

```text
CI tests                         240,489
CI absolute error >1e-10          79,012
CI absolute error >1e-6           35,233
CI absolute error >1e-3            5,282
maximum CI absolute error       0.045461329947135143
pMax cells above 1e-10              1,398
decision flips                           0
```

The absence of decision flips explains why the defect escaped the earlier
graph-only gate, but does not make the implementation numerically compatible.

## Repair

The repaired production shape remains GPU-only:

```text
all dcc.gamma pairs
  -> CUDA fixed-SP residual
  -> CUDA raw-distance matrix
  -> CUDA full symmetric eigensolver
  -> CUDA component moments and gamma p-value
  -> compact result D2H only
```

The exact-screen decision path and guarded refinement are no longer used by
`dcc.gamma`. Conditional fixed-SP residuals use augmented QR through two
penalties and CUDA stable-SVD for deeper additive systems. No CPU residual,
dCov component, eigensolver, pair statistic, gamma calculation, or numerical
fallback is permitted.

## Canonical evidence

The complete default-`Inf` 351x48 development audit executed all 240,498
logical CI tests and produced:

```text
maximum CI absolute error        6.9555444737190442e-11
CI mismatches above 1e-10        0
maximum pMax absolute error      6.9555444737190442e-11
pMax mismatches above 1e-10      0
task/deletion identity           exact
adjacency                        exact
semantic sepsets                 exact
n.edgetests                      exact
CPU numerical authority counts  0
residual/component D2H           0
CUDA legacy-compatible pairs     240,498 / 240,498
exact-screen pairs               0
guard-refinement pairs           0
elapsed                          2293.965 s
```

The complete audit was rerun after a full native rebuild. The audited native
binary SHA-256 is
`20c29a5724cb5a931d79b0f4e413d09ae3525be407319daadad359ecb31de265`.
The local canonical receipt SHA-256 is
`cc1e8b7029252dfed0c6b7f01ca523008878b69f7f449374cbaf10bb7714a4b2`;
its semantic-sepset revalidation attestation SHA-256 is
`2de8ebc5b38987cb0d4d1d36426870781c2f12f9fe5cf01b386edb49d90297c8`;
and the independent formal cold-candidate validation attestation SHA-256 is
`ed6bf0c7771f42886cccbceae0e1db7f20f5774e93854fe0b4d710ffc2a506d8`.
The run fixed `CUDA_VISIBLE_DEVICES=0` and all supported BLAS/OpenMP thread
variables to one. These remain development receipts because the repair source
is not yet committed as a frozen producer; they are not promotion evidence.

Three independent real-input CPU-oracle/GPU-production comparisons also pass:

```text
shape    CI tests    maximum CI absolute error
179x8         338    1.6017187576267133e-12
243x9         413    1.3618425831474212e-11
300x10        657    1.5052667445836221e-11
```

All three have exact decisions, adjacency, semantic sepsets, final PDAG, and
RNG state, with zero CPU skeleton numerical authority in the GPU route.

## Performance reset

The corrected canonical stage totals show why the historical 219.513-second
number cannot be retained:

```text
CUDA legacy component build      1701.543 s
CUDA fixed-SP residual solve      340.900 s
CUDA optimizer boundary           155.392 s
native setup                       22.606 s
full outer elapsed               2293.965 s
```

Subsequent strict optimization reduced this repaired baseline to `928.720`
seconds by retaining all unique components in a bounded device cache, bypassing
residual solves for complete all-hit batches, and running deterministic
`cusolverDnDsyevd` through eight persistent lanes. The final current-binary
audit preserves the same maximum CI and `pMax` error shown above, with zero CPU
authority, fallback, or large-payload D2H and closed CUDA resource counters.

The remaining full-eig component candidates are now closed by measured
evidence. Selective two-group scheduling has only a `1.469`-second projected net
upper bound after observed frontier fragmentation; expanding beyond eight
lanes is at most `1.0024x`; per-lane host threads do not improve device
makespan; and Jacobi eig is slower and numerically ineligible. Full details,
artifact hashes, and reopen conditions are recorded in
`PHASE10_FULL_EIG_COMPONENT_CLOSURE.md`.

Candidate freezing, recommendation, and holdout release remain blocked. The
full-eig component is closed, but broader Phase 10 promotion status is still
active and has not passed Checkpoint B or the final repeated-run gate.

## Exact mgcv semantic closure

Producer `377676f` subsequently closed the selected-SP and fixed-SP semantic
gap on the GPU-only route. Its fully rebuilt native SHA-256 is
`6f417bce389fa1d852e6aba8d35fb2a836e9f5a9ba30cabdbbb4c5ba789b5697`.
The complete default-`Inf` audit now has maximum CI and `pMax` error
`3.0547508966805026e-12`; three larger real-input campaigns add 223,217 tests
with zero errors above `1e-10` and maximum error
`4.8635392588258775e-12`. CPU numerical authority and fallback remain zero.

The attempted KKT/iterative-refinement certificate is retained only as a
test-only fail-closed diagnostic. Real rank-deficient witnesses prove that a
small KKT residual is not sufficient to identify mgcv's pseudoinverse result,
including one certificate pass with p-value error
`1.454857345706273e-10`. Production therefore uses the GPU exact fixed-SP
evaluator. Full evidence and receipt hashes are recorded in
`MGCV_GPU_SEMANTIC_CLOSURE.md`.

The earlier `928.720`-second result remains historical full-eig optimization
evidence and is not a performance measurement for producer `377676f`.
Promotion status remains unchanged.

## Exact-primary production update

Runtime producer `305d7a7` removes the redundant selected-SP stability replay
for `dcc.gamma` by running its deterministic stable-SVD path once as the exact
primary. With the retained 11-GiB component cache, all-hit residual bypass, and
eight eig lanes, the current canonical source-closure audit passes all 240,498
CI tests in `1272.360s`; maximum CI and `pMax` error remain
`3.0547508966805026e-12`.

All 80,751 multi-penalty targets carry exact-primary identity, while replay and
terminal-confirmation counts are zero. The canonical result is bitwise
identical to the qualified test-only prototype, and the current 12-shape
selected-SP/fixed-finalization campaign also passes. Full counters, native and
receipt SHA-256 values, real-input prototype evidence, and the v8 residual
bypass accounting rule are recorded in `MGCV_GPU_SEMANTIC_CLOSURE.md`.

This is source-closure development evidence, not a clean-checkout repeated
campaign or promotion evidence. Checkpoint B, the five-run final gate,
candidate freeze, recommendation, and sealed holdout state remain unchanged.
