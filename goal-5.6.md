# Full-CUDA Legacy-Compatible CI Goal Document

> **Campaign:** replace the complete numerical conditional-independence data plane used by the KPC skeleton with a legacy-compatible CUDA implementation.
>
> **Document origin baseline:** `main` at `7b36668` (`feat: add CUDA Spectra handle projection primitive`).
>
> **Current accepted implementation/evidence snapshot:** Phase 9 is COMPLETE. The historical Phase 10 v1 campaign producer `592dd982673c62d996ebff404dd68f9bdde71f83d5784e4235325cc1a2ffc556` is accepted for canonical correctness, CUDA authority, hardening, artifact integrity, and replay latency only. Its `0.699`-second measurement is replay-warm after a complete same-data call; fresh-process cold is `1290.664` seconds versus a `601.431`-second correct baseline. Under `performance_budget_v2`, the public 500x50 development fixture passes and the best full fresh-data compute-warm development run remains `263.305` seconds. Per-call univariate primitive reuse reduced native setup from `77.650` to `38.997` seconds while preserving bitwise-identical consumed p-values, exact graph/sepset/count/trace parity, and zero CPU numerical authority or fallback. A development-only grouped guarded-QR/stable-SVD-queue prototype has exact internal and public-output parity across real q=28/37/46/55/64 shapes, but its best 512-target QR/Q speedup is only `1.032x`, below the `1.3x` stop threshold; it is not integrated into the optimizer. Compute-profile v5 proves that all `132,908` physical target optimizations occur in whole-level prefill: `110,617` TargetKeys are eventually consumed and `22,291` are never consumed, while frontier live optimization and singleton padding are both zero. Its `269.426`-second diagnostic run is bitwise identical to the `263.305`-second result and is not a new performance claim. A subsequent zero-lookahead canonical-frontier prototype was rejected at max conditioning size 3: it took `394.993` seconds versus `179.914` seconds for v5, increased optimizer boundaries from 83 to 1,898 and setup submissions from 5,239 to 46,691, created 14,507 singleton padding targets, and changed 120,991 consumed p-values in low bits (maximum `3.33e-14`) despite exact structural graph semantics and zero decision flips. The v5 production path is restored. Checkpoint A passes, but Checkpoint B (`<180` seconds) and the final five-run median gate (`<=120` seconds) still fail. No new candidate is frozen; Phase 10 is not complete, and the external promotion holdout remains `SEALED_NOT_RELEASED` and must stay unopened.
>
> **Latest scheduler opportunity diagnostic:** A no-CUDA-CI native-plan replay reconstructed all 137 original v5 windows, 8,637 setup cohorts, 132,908 target optimizations, and 110,617 consumed TargetKeys exactly. Every v5 window is eventually demanded; only three setup cohorts are never demanded, containing nine targets. Of the 22,291 unconsumed targets, 22,282 are inside demanded cohorts. Lazy original-window activation therefore skips zero recorded windows and zero recorded batch wall time; through level 3 it still requires exactly 83 boundaries, 5,239 setup submissions, and 107,053 target optimizations. The decision is `STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL`: stop scheduler-prefill work for this contract epoch.
>
> **Latest optimizer-residual qualification:** A development-only real `q=64`, seven-penalty, two-target fixture compared the accepted optimizer residual with the later selected-SP fixed solve entirely on the device. All 702 values differed; maximum absolute difference was `0.2399546` and relative L2 difference was `0.0288145`, with zero optimizer/fixed status failures and zero route/status mismatches. Direct device-residual-view exact and legacy-eig consumers had zero residual payload D2H but both produced non-bitwise-identical p-values (`3.58933903915984e-14` versus `6.41771967688960e-14`, and `4.48319202981132e-14` versus `7.89308561131647e-14`). There were zero decision flips, but the numerical producers are not semantically interchangeable under the current contract. The decision is `STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY`; do not implement D2D detach/arena retention or production reuse in this contract epoch. The optimizer residual remains optimizer-internal evidence and does not define production ResidualKey authority. Fixed-SP residual solving remains authoritative; the subsequent v6 attribution is recorded below.
>
> **Latest fixed-SP residual attribution:** The trace-free full level-7 compute-profile v6 takes `274.728` seconds as a development diagnostic and does not replace the `263.305`-second best runtime. Its graph, sepsets, counts, 240,489 task rows including consumed p-values, and semantic level trace are bitwise identical to the primitive-cache artifact, with zero CPU numerical authority, fallback, residual D2H, or component D2H. Authoritative fixed-SP residual solve time is `15.315` seconds: single-penalty `10.272`, multi-penalty `5.043`, exact screen `14.833`, and guarded legacy-eig refinement only `0.482`. The run executes 38,613 exact residual batches and 553 refinement batches; refinement is only `3.15%` of residual time, so exact/refinement residual sharing is below the `3-5` second implementation gate and stops as a main route. Exact screens nevertheless perform 228,015 residual target fits for 110,665 unique ResidualKeys, leaving 117,350 excess fits (`51.47%`). That apparent target-level opportunity motivated the cross-batch qualification recorded below; it was not treated as reusable without evidence. The same run records 39,166 dCov calls, `3.440` seconds of teardown, and `7.416` seconds of otherwise unattributed dCov host time; this is only an upper bound for a persistent dCov execution context.
>
> **Latest fixed-SP cross-batch qualification:** Device-only repeat/permutation/subset/singleton/route-mix fixtures show that complete identical or permuted cohorts are bitwise exact, but batched-Cholesky miss-only subsets are not. Across the fixtures there are 913 residual-value mismatches, 332,530 exact-component value/moment mismatches, three solver-status mismatches, and two unconditional legacy-eig p-value mismatches. Final guarded p-values and decisions remain unchanged, but the target-granular strict gate fails; the intermediate decision is `CONDITIONAL_ALL_HIT_BATCH_ONLY`. A trace-free full level-7 opportunity run then classifies all 38,613 exact batches and 228,015 target fits with exact accounting. Only 46 batches and 1,853 targets are both all-hit and a repeated complete cohort. Their measured residual solve plus exact-component upper bound is only `71.354` ms (`54.209 + 17.145`), below the 5-second stop gate. The final decision is `STOP_CROSS_BATCH_FIXED_RESIDUAL_CACHE_OPPORTUNITY`: do not implement target-granular caching, residual slabs, exact-component caching, capacity traces, or whole-cohort reuse in this contract epoch. The diagnostic run takes `266.801` seconds and does not replace the `263.305`-second baseline; its graph/task outputs and semantic level fields remain bitwise exact with zero CPU authority, fallback, or large-payload D2H.
>
> **Active roadmap phase:** Phase 10, full performance gate, hardening, and promotion.
>
> **Hard rule:** **0 SHD is mandatory.** A faster result with a different skeleton is a failed result.

This file is the active implementation roadmap for the full-CUDA CI campaign. `final_goal.md` remains useful as historical evidence and an experiment log, but Codex should use this file to decide what to implement next and which gates must pass.

---

## 0. One-line goal

Build a **legacy-compatible, full-CUDA conditional-independence engine for the KPC skeleton**:

```text
R prepares the input and calls one native skeleton entrypoint.
C++ owns canonical PC/KPC control flow, adjacency, sepsets, and logical test counts.
CUDA performs the repeated numerical CI work:
  mgcv-compatible GAM smoothing-parameter selection
  penalized fitting and residualization
  legacy-compatible dCov.gamma component construction
  gamma moments and p-values
C++ replays results in the exact legacy order.
R receives the skeleton and sepsets and continues orientation.
```

The final compatible route must reproduce the legacy CPU skeleton with **SHD = 0**. It must not use `fastSplineCUDA`, `tprsApproxCUDA`, or any other approximate smoother as a silent substitute for `mgcv::gam`.

The guiding principle is:

```text
First 0 SHD, then speed.
```

---

## 1. Hard correctness contract

### 1.1 Canonical full-data gate

For the canonical real benchmark:

```text
n                   = 351
p                   = 48
alpha               = 0.1
reference edge count = 110
reference n.edgetests = 2213,52659,125293,40694,13293,5422,835,80
```

Every route proposed for compatible-CUDA promotion must satisfy:

```text
candidate edge count        = 110
reference edge count        = 110
adjacency identical         = TRUE
SHD                         = 0
normalized sepsets identical = TRUE
logical n.edgetests identical = TRUE
logical n.edgetests          = 2213,52659,125293,40694,13293,5422,835,80
canonical deletion trace identical = TRUE
unknown fallback count      = 0
approximate backend count   = 0
```

A timeout, crash, missing candidate graph, or incomplete artifact is not a correctness pass.

### 1.2 `SHD = 0` is necessary but not sufficient

A matching skeleton can occur by accident even when CI decisions or separating sets differ. Therefore promotion requires all of the following:

```text
same skeleton
same canonical logical tests consumed
same edge-deletion level
same first accepted separating set for every deleted edge
same normalized sepsets
same logical n.edgetests
same stop level
```

When the orientation stage is included in an artifact, the final oriented graph must also be identical after canonical normalization.

### 1.3 Graph decisions outrank average numerical error

The primary acceptance criterion is not:

```text
mean residual correlation
mean p-value error
mean fitted-value error
average speedup
```

The primary criterion is:

```text
no CI decision flip that changes canonical replay
no sepset drift
no graph drift
SHD = 0
```

Residual, fitted-value, smoothing-parameter, score, EDF, eigenvalue, statistic, and p-value errors must still be recorded. They are diagnostics and local phase gates, but no small average error can waive a graph mismatch.

### 1.4 Exact legacy semantic source

The compatibility oracle is the current legacy path:

```text
conditional residuals:
  kpcalg::regrXonS
  mgcv::gam

formula route:
  |S| <= 2: target ~ s(S1, ..., Sk)
  |S| > 2:  target ~ s(S1) + ... + s(Sk)

family/link:
  gaussian / identity

smoothing selection:
  the version-pinned mgcv behavior used by the canonical run
  normally method = "GCV.Cp"

independence test:
  legacy dcov.gamma semantics

search semantics:
  stable/canonical skeleton replay
```

The executable code is authoritative when comments and old documentation disagree. In particular, the current `regrXonS` implementation switches to additive smooths when `|S| > 2`.

### 1.5 Approximate paths are forbidden in compatible mode

The following are never accepted as the authority for this campaign:

```text
fastSplineCUDA residuals
tprsApproxCUDA residuals
simplified B-spline or P-spline replacement
coarse-grid-only smoothing selection when it changes legacy decisions
exact dCov substituted for legacy dcov.gamma without a full zero-drift gate
near-alpha heuristics used without canonical verification
float32, TF32, or --use_fast_math numerical shortcuts
```

They may remain separate experimental or `precision="fast"` backends, but they must not enter `precision="compatible.cuda"` execution.

### 1.6 Strict fallback policy

Development modes may use explicit oracle fallback, but every fallback must be counted and explained.

```text
shadow mode:
  legacy remains authoritative
  candidate is compared

oracle-fallback mode:
  allowed only for development artifacts
  every fallback emits a reason and target/S key

strict compatible.cuda mode:
  no approximate fallback
  no silent CPU fallback
  unsupported semantics fail closed
```

For the canonical 351x48 final gate:

```text
legacy mgcv fit count in CI loop = 0
R callback count in native skeleton loop = 0
CPU residual solve count = 0
CPU dCov component count = 0
CPU dCov eigen/low-rank count = 0
CPU dCov pair-statistic count = 0
CPU gamma p-value count = 0
CPU Spectra count = 0
unknown fallback count = 0
```

A documented C++ host control plane is not a fallback.

### 1.7 Determinism

The promoted route must pass at least five measured repetitions after warm-up:

```text
SHD = 0 in every repetition
adjacency identical in every repetition
sepsets identical in every repetition
logical n.edgetests identical in every repetition
no intermittent CUDA fallback
no non-finite numerical result
```

GPU task scheduling may be parallel. Logical replay and final graph output must remain deterministic.

---

## 2. Definition of “the whole CI test is CUDA”

The project does not need to put graph bookkeeping on the GPU. It does need to remove the repeated numerical CI hot path from R/CPU.

### 2.1 Must execute on CUDA in the final route

```text
target-specific smoothing-parameter scoring and selection
penalized GAM solves
residual formation
residual cache payloads used by dCov
distance-matrix/component construction
legacy-compatible low-rank or eigen computation
dCov statistic and gamma moments
batched gamma p-value computation
```

The final device-resident path should avoid downloading residual vectors. Only compact results such as p-values, status flags, selected smoothing parameters for diagnostics, and optional sampled shadow payloads should return to the host.

### 2.2 May remain on C++ host

```text
formula-route selection from |S|
response-independent setup key construction
canonical edge and conditioning-set enumeration
round/level state machine
logical test counting
adjacency mutation
sepset storage
canonical result replay
cache policy and memory budgeting
error handling and fail-closed decisions
```

A response-independent basis/penalty setup may initially be constructed on the C++ host once per unique `S`, provided there is no per-target `mgcv::gam` call and all repeated target-dependent fitting/GCV work is on CUDA. Phase 7 removes the R/mgcv setup dependency from the native hot path.

### 2.3 May remain in R outside the CI engine

```text
input validation and data preparation
calling the one native skeleton entrypoint
result packaging
post-skeleton orientation
report generation
```

---

## 3. Current foundation and known gaps

The existing repository already contains important substrate. Codex must reuse it rather than restart from an unrelated implementation.

### 3.1 Correct compatible baseline

The best completed full one-call compatible facade currently recorded is approximately:

```text
route:
  native one-call threaded round dCov
  legacy mgcv/regrXonS residual authority
  C++ Spectra legacy dCov authority

elapsed_sec       = 592.259
edge_count        = 110 / 110
SHD               = 0
n.edgetests exact = TRUE
pMax max abs diff = 0
```

This is the performance baseline to beat on the same hardware and software environment. A fresh same-machine baseline must be included in any promotion artifact.

### 3.2 Approximate fast CUDA path

The repository has a fast approximate CUDA stack, including `fastSplineCUDA`. It is useful for performance comparison but is not legacy compatible and has previously produced skeleton drift.

Status for this campaign:

```text
keep frozen as a separate approximate backend
do not use it as a compatible fallback
do not tune it and claim mgcv equivalence
```

### 3.3 Existing GAM-compatible substrate

Existing files include:

```text
fastkpc/R/mgcv_compat_contract.R
fastkpc/R/mgcv_extract_oracle.R
fastkpc/R/mgcv_residual_oracle_trace.R
fastkpc/R/mgcv_residual_replay_spec.R
fastkpc/R/precision_data_plane.R
fastkpc/src/cuda/mgcv_extract_fixed_sp_cuda.cu
```

Existing capabilities include:

```text
version-pinned mgcv setup extraction
fixed-sp CPU replay
fixed-sp CUDA solve
constraint-nullspace handling
single-penalty spectral scoring helpers
same-S prepared setup and spectral cache
same-setup x/y and target-list APIs
oracle trace and replay artifacts
```

Historical limitations of the pre-Phase 3 precision prototype included:

```text
precision CUDA scope was mainly |S| <= 2
multi-penalty GPU GCV was not implemented
smoothing scoring/selection used r-cpu-spectral in the main path
same-setup CUDA solve repeated work instead of using a true fused batch kernel
true_batched_kernel remained FALSE for that path
fixed-sp CUDA code allocated/copied/created solver resources too often
residuals were materialized back on the host
R/mgcv supplied setup semantics
```

Phase 3 removed the repeated-resource and false-batch limitations for the
accepted oracle-selected fixed-sp runtime. The current campaign gaps are:

```text
autonomous single-penalty and multi-penalty CUDA GCV are not implemented
R/mgcv still supplies response-independent setup semantics
the promotable device-resident dCov architecture is not yet selected
the capacity-bounded final cache split has not been accepted
the final one-call compatible.cuda service is not integrated
the conservative 120-second feasibility budget has not passed
```

### 3.4 Residual workload evidence

Historical full-run diagnostics report:

```text
conditional residual requests                  = 476,552
canonical global unique conditional target|S  = 110,617
S-affinity executed mgcv fits                  = 273,284
S-affinity current native |S|<=2 fits          = 157,122
S-affinity current |S|>2 fallback fits         = 116,162
canonical nonempty same-S groups               = 8,634
S-affinity same-S fit reuse opportunity        = 264,650
S-affinity same-S fit reuse ratio              = 0.968406
```

`110,617` is the canonical global key count. `273,284` is a route-specific
executed-fit count produced by the historical S-affinity worker/cache layout;
it is not a unique-request count. The evidence still means same-S,
multi-target execution is the required unit of work. Per-target `mgcv::gam`
and per-target CUDA allocation cannot reach the final goal.

### 3.5 Existing dCov substrate

Existing files include:

```text
fastkpc/src/legacy_dcov_gamma_cpp.cpp
fastkpc/src/cuda/dcov_batch_cuda.cu
fastkpc/src/cuda/legacy_dcov_spectra_matvec_cuda.cu
fastkpc/R/compatible_cuda_skeleton_artifact.R
```

Existing capabilities include:

```text
legacy dCov C++ oracle and Spectra backend
exact CUDA dCov backend
CUDA legacy-low-rank parity primitives
native one-call skeleton facade
round-based canonical replay
component cache experiments
resident CUDA matrix handle
multi-RHS cuBLAS matvec
Q -> A Q -> Q' A Q projection primitive
small-fixture zero-SHD CUDA-low-rank gates
```

Current limitation:

```text
the full 351x48 cuda_spectra route does not complete inside the accepted wall-time envelope
RSpectra/Arnoldi remains host-driven
per-iteration host/device traffic and synchronization remain
local primitive parity has not produced a promotable full CUDA dCov route
```

### 3.6 Existing native skeleton substrate

The repository already has native skeleton planning, provider entrypoints, one-call facades, level/round dCov batching, deterministic replay, and detailed diagnostics.

The remaining architectural gap is not graph control. It is replacing the hidden residual provider and CPU dCov authority with a device-resident compatible CI service.

---

## 4. Target architecture

### 4.1 Final call shape

Suggested public shape:

```r
fast_kpc(
  data,
  precision = "compatible.cuda",
  compatible_cuda_strict = TRUE,
  graph_stage = "skeleton"
)
```

Suggested native boundary:

```text
.Call("fastkpc_compatible_cuda_skeleton_run", data, config)
```

The exact exported name may follow existing repository conventions. The architectural requirements are mandatory even if the symbol name changes.

### 4.2 Runtime layout

```text
R
  prepares data and config
  calls one native entrypoint

C++ canonical control plane
  owns adjacency and sepsets
  enumerates tests in legacy order
  gathers only currently reachable work
  groups residual requests by same S
  submits CUDA work
  replays p-values in canonical order
  counts only logically consumed tests

CUDA compatible CI service
  PreparedS cache
    basis and penalty state
    constraint/null-space state
    stable factorization / spectral state
    version and semantic fingerprint

  Target residual service
    target-specific y and X'y
    target-specific smoothing parameters
    target-specific convergence state
    device-resident residual handle

  dCov component service
    distance component per residual handle
    cached eigen/low-rank representation
    pairwise statistic and gamma moment batch
    device-side p-values

  compact result batch
    p-value
    numerical status
    diagnostic flags
```

### 4.3 Required keys and identities

```text
DatasetKey:
  data hash + dimensions + column order + numeric type

PreparedSKey:
  DatasetKey + sorted S + formula route + mgcv semantic version

TargetKey:
  PreparedSKey + target column

ResidualKey:
  TargetKey + selected-sp fingerprint + solver semantic version

DcovComponentKey:
  ResidualKey + index + numCol + dCov semantic version

LogicalCiKey:
  level + canonical edge + ordered conditioning set + logical sequence id
```

Cache reuse must never depend on non-semantic pointer identity.

### 4.4 Same-S batching rule

For a fixed `S`:

```text
basis is shared
penalty structure is shared
constraints are shared
response-independent transforms are shared
X'X or stable equivalent is shared
```

But:

```text
y is target-specific
X'y is target-specific
selected smoothing parameter(s) are target-specific
GCV convergence can be target-specific
```

Therefore the correct unit is:

```text
prepare(S) once
score_and_fit_targets(S, Y_batch) with one target state per column
```

Do not incorrectly force one smoothing parameter vector on all targets in a same-S group.

### 4.5 Device-resident residual-to-dCov path

The final hot path is:

```text
same-S target batch
  -> CUDA GCV / smoothing selection
  -> CUDA penalized fit
  -> CUDA residual handles
  -> cached CUDA dCov components
  -> CUDA pair statistics / gamma p-values
  -> compact p-value batch
```

The host must not download every residual merely to upload it again for dCov.

### 4.6 Logical versus physical work

CUDA may evaluate speculative work to form useful batches, but diagnostics must separate:

```text
logical_tests_consumed
physical_ci_tasks_evaluated
speculative_tasks_ignored
logical_residual_requests
physical_residual_fits
logical n.edgetests
```

Only canonical replay increments `n.edgetests` or changes the graph.

---

## 5. Supported semantic envelope

The final canonical route only needs to implement the actual KPC/regrXonS subset, not all of mgcv.

### 5.1 Required

```text
numeric matrix input
finite double values
Gaussian family
identity link
unweighted observations
zero offset
residual output only
method used by the legacy canonical run, normally GCV.Cp
thin-plate regression smooth semantics used by s(...)
|S| <= 2 full smooth
|S| > 2 additive smooths
single-penalty and multi-penalty setups
mgcv-compatible identifiability constraints
rank-deficient and ill-conditioned handling seen in the workload
```

### 5.2 Not required

```text
non-Gaussian families
IRLS
random effects
gamm
bam
by-variable smooths
factor smooths
tensor-product replacement semantics
standard errors
vcov
prediction intervals
summary.gam
arbitrary user formulas
full mgcv model object reconstruction
```

### 5.3 Numerical policy

Compatible CUDA uses:

```text
double precision
no TF32
no float32 fallback
no --use_fast_math
explicit finite checks
stable rank handling
recorded tolerances and convergence status
```

After Phase 3.5, all numerical thresholds, denominator rules, condition
buckets, boundary semantics, and non-finite policies come from the tracked
`numerical_contract_v1`; phase-local prose cannot silently override it.

Do not rely only on normal equations for difficult cases. A Cholesky path may be used when the measured condition and parity gates support it. Ill-conditioned cases require a stable QR/SVD or equivalent augmented-system path on C++/CUDA.

---

## 6. Phase status overview

Existing code may provide substrate for a phase, but a phase is not complete until its explicit artifact and exit gates pass.

| Phase | Name | Current status |
|---|---|---|
| 0 | Freeze oracle and zero-SHD comparator | COMPLETE — standardized oracle and first-divergence gate pass |
| 1 | Full workload and risk census | COMPLETE - full 110,617-key metadata artifact passes all gates |
| 2 | Response-independent GAM setup contract | COMPLETE - full structural artifact and qualification exact-parity/restart gates pass |
| 3 | Persistent stable fixed-sp CUDA residual runtime | COMPLETE - full 110,617-target oracle and 240,489-test shadow artifacts independently validate with 0 flips and SHD 0 |
| 3.5 | Full-CUDA architecture feasibility and performance-budget gate | COMPLETE - guarded exact-screen/full-eig CUDA architecture is GO for Phase 8 qualification; measured cache/memory and conservative 98.529-second campaign feasibility gates pass |
| 4 | Full-CUDA single-penalty GCV for `|S|<=2` | COMPLETE - all 44,941 targets use CUDA scoring/selection and fitting; full-shadow graph semantics and the five-run backend gate pass |
| 5 | C++ multi-penalty GAM semantic replica | COMPLETE - all 7,460 setups, 65,676 targets, and 60,324 logical rows pass with zero optimizer drift, decision flips, or fallback; mixed replay has SHD 0 |
| 6 | CUDA multi-penalty same-S target batches | COMPLETE - all 65,676 multi-penalty targets use persistent CUDA optimization and residual solves; the full residual route has SHD 0 and a 0.7693 same-trace performance ratio |
| 7 | Native setup builder; remove R/mgcv from CI loop | COMPLETE - all 8,634 native setups and the complete residual/graph route pass with zero R/mgcv setup authority or fallback |
| 8 | Legacy-compatible device-resident CUDA dCov | COMPLETE - all 240,489 logical tests use guarded device-resident CUDA dCov with zero final flips, zero CPU numerical authority, SHD 0, and a 17.499-second dCov boundary |
| 9 | Fused one-call compatible CUDA skeleton | COMPLETE - one native call reproduces all 240,489 canonical tests, exact graph semantics, and zero R/CPU numerical authority |
| 10 | Full gate, hardening, and promotion | ACTIVE - v1 correctness/hardening/replay evidence passes; the public 500x50 fixture and a 263.305-second single-run fresh-data development profile pass, but Checkpoint B, the formal five-run 120-second gate, refreeze, sealed holdout, and completion audit remain |

**Codex starts at the earliest phase whose exit gate is not complete.** Do not skip Phase 0 because later code already exists.

---

## 7. Phase plan

## Phase 0 — Freeze the oracle and build the zero-SHD gate

### Goal

Create one immutable, machine-readable reference and one comparator that every later phase must use.

### Required implementation

Add a standardized full-CUDA campaign artifact generator and comparator. Suggested files:

```text
fastkpc/R/full_cuda_ci_gate.R
fastkpc/tools/run_full_cuda_ci_oracle.R
fastkpc/tests/test_full_cuda_ci_oracle_gate.R
```

Reuse existing helpers where possible.

### Required oracle artifact

```text
fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/
  manifest.json
  summary.json
  summary.md
  adjacency.rds
  adjacency.csv
  sepsets.rds
  n_edgetests.csv
  logical_ci_trace.rds
  deletion_trace.csv
  pmax.rds
  near_alpha_cases.csv
  environment.txt
```

### Manifest fields

```text
data hash
data dimensions
column order
alpha
max conditioning size
index
numCol
R version
mgcv version
package/session information
compiler versions
CUDA driver/runtime version
GPU model
CPU model
thread counts
source commit
oracle route environment
```

### Comparator output

```text
adjacency_identical
SHD
edge_count_reference
edge_count_candidate
sepsets_identical
n_edgetests_identical
deleting_test_identical
first_divergence_found
first_divergence_level
first_divergence_edge
first_divergence_S
reference_p
candidate_p
reference_decision
candidate_decision
fallback summary
```

### Hard gate

Run the oracle against itself and against the current 592.259-second correct route:

```text
edge_count = 110 / 110
SHD = 0
adjacency_identical = TRUE
sepsets_identical = TRUE
n.edgetests exact = TRUE
deletion trace identical = TRUE
```

### Exit condition

Phase 0 is complete only when one command can fail a candidate run on the first graph-semantic mismatch and emit a first-divergence artifact.

### Accepted Phase 0 result

```text
oracle artifact:
  fastkpc/artifacts/full_cuda_ci/oracle_351x48_v1/

qualified current-route comparison:
  fastkpc/artifacts/full_cuda_ci/current_correct_route_351x48_v1/

qualified logical trace source:
  fastkpc/artifacts/full_cuda_ci/trace_source_351x48_v1/result.rds

full gate:
  elapsed_sec = 591.152
  logical CI trace rows = 240,489
  edge_count = 110 / 110
  SHD = 0
  adjacency identical = TRUE
  normalized sepsets identical = TRUE
  n.edgetests identical = TRUE
  canonical deletion trace identical = TRUE
  logical CI trace identical = TRUE
  unknown fallback count = 0
  approximate backend count = 0
```

The standard runner is `Rscript fastkpc/tools/run_full_cuda_ci_oracle.R`.
It defaults to immutable validation of the pinned oracle bundle; regeneration
requires the explicit opt-in
`FASTKPC_FULL_CUDA_CI_ORACLE_MODE=refresh`. Validation input can be separated
from comparison output with `FASTKPC_FULL_CUDA_CI_ORACLE_DIR`.
Focused mismatch tests prove fail-closed first-divergence output for adjacency,
sepset, logical test-count, deletion-trace, and missing-candidate failures. The
native `trace_level="logical"` mode records only canonically consumed tests in
reserved C++ vectors, avoiding the quadratic Rcpp-vector growth observed with
the old full physical-task trace. The standard runner pins the canonical data,
column order, oracle result, adjacency, normalized sepsets, and deletion trace
by hash; logical rows must match both per-level `n.edgetests` and an independently
rebuilt canonical layer plan. Explicit unknown/approximate/backend fallback and
error counters are fail-closed, and a missing candidate emits a persisted
`candidate_missing` first-divergence artifact. Phase 0 is complete; Phase 1 is
next.

### Do not do in this phase

```text
no CUDA performance change
no oracle change
no tolerance weakening
no backend promotion
```

---

## Phase 1 — Capture the complete CI workload and numerical risk census

### Goal

Turn the full canonical run into a complete implementation specification, not an eight-case sample.

### Required artifact

```text
fastkpc/artifacts/full_cuda_ci/workload_census_351x48_v1/
```

### Completion evidence (2026-07-10)

```text
status                                  = COMPLETE
artifact source commit                  = 1560068ba8d635e806612554e11bbed92c0b8843
metadata schema                         = full-cuda-ci-metadata-v4
artifact manifest SHA-256               = b0990cfc932a5fcabc09ad25e352e7babb67fc8a127f11c7d2b88887c4940574
artifact summary SHA-256                = 4f71d1bbbbdd2436e3576b728363120bb9b911897b9dee3ecf6f8a5d3379eb24
same-S setup RDS SHA-256                = 8b35a463b17a64512d653da949f5ac74f7cc21223f346a304ac52fdfe8434a3f
target-fit RDS SHA-256                  = af09b5dc4c6a34d7ec126e1fe7f3f1f9c3d7fcb6316ada759a293abe76d8323c
risk-cases RDS SHA-256                  = 1e0951e9856bea3c9a1b7ba83ec03b79a678e7aa60464d7f6808397ab8d9a7bc
oracle input bundle SHA-256             = 7700bc78240984c36f8ae5ca281362a0afb8d7dedd34a5711ce4ab76a2ebee0e
canonical logical census SHA-256        = c9b48074dd59a439fceb9d5e64806adda5620cc4abe32095371abc447ef98634
canonical key corpus SHA-256            = b843630969f116da63f7fad095c54de2ff471540159ff97ca56c3871d6b2e1fa

logical tests                           = 240,489
conditional logical tests               = 238,276
conditional residual requests           = 476,552
canonical conditional target|S keys     = 110,617
nonempty same-S groups                  = 8,634
same-S setup metadata rows              = 8,634
target-fit metadata rows                = 110,617
pre-compression setup observations      = 110,617
pre-filter target-risk metadata rows    = 110,617

mgcv fit errors                         = 0
same-S invariant violations             = 0
required field coverage                 = 100%
legacy two-target layout parity          = exact
all target fit_status values             = success
target/request/setup/risk joins          = exact
target risk semantics                    = exact
target near-constant semantics           = exact
authenticated metadata tables           = exact
authenticated shard payloads             = 64 / 64
warning/nonfinite classification         = exact
unclassified warnings/nonfinite values  = 0 / 0
unsupported/fallback counters zero       = 15 / 15
coefficient outputs finite               = 110,617 / 110,617
fitted outputs finite                    = 110,617 / 110,617
residual outputs finite                  = 110,617 / 110,617
oracle inherited graph gate             = TRUE
new candidate graph gate                = NOT_APPLICABLE

requested/actual workers                = 20 / 20
completed shards                        = 64 / 64
executed keys                           = 110,617
internal elapsed                        = 769.640 sec
external wall                           = 770.31 sec
maximum RSS                             = 2,090,784 KiB
artifact size                           = 448,379,500 bytes (427.6 MiB)
metadata shards / merge / artifact      = 421.433 / 50.090 / 256.440 sec
```

The risk corpus contains 68,204 rows; flags overlap:

```text
high penalized-system condition         = 33,249
rank-deficient / infinite condition     = 1,162
multi-penalty                           = 65,676
near-alpha logical tests                = 1,529
mgcv.conv not fully converged           = 980
mgcv warnings                           = 0
near-constant targets/conditioners      = 0 / 0
```

All 1,162 non-finite penalized-system conditions are classified and coincide
with rank-deficient cases. Every authoritative fit still reports
`fit_status=success` and `fit$converged=TRUE`. The historical S-affinity count
of `273,284` remains a provenance-only route metric and is not a Phase 1 gate.

### Record every logical CI test

```text
logical sequence id
source sequence id
source task index
level
x
y
sorted S
formula route
reference p-value
explicit reference decision and independence flag
signed/absolute distance from alpha
signed/absolute log ratio from alpha
whether the test deletes the edge
selected sepset status
```

Level 0 uses the existing `direct-ci` formula class and does not create a GAM
residual key. Conditional rows reuse the existing `full-smooth` and
`additive-smooth` formula classes.

### Record every unique conditional target|S residual request

```text
target
S
|S|
full-smooth versus additive-smooth formula class
canonical residual-key payload and SHA-256
same-S group id
same-S group size
request multiplicity
```

### Separate response-independent setup from target-specific fit state

`same_s_setup_metadata` contains one row per nonempty S group:

```text
model-matrix dimensions
penalty count and block dimensions
constraint dimensions
rank metadata
condition estimates
penalty nullity
constraint rank and nullspace dimension
setup fingerprint and invariant hashes
```

`target_fit_metadata` contains one row per canonical conditional target|S key:

```text
near-constant target flags
selected smoothing parameter vector
GCV/Cp score
EDF
mgcv convergence fields
fit time
residual hash
fitted hash
penalized-system condition at selected sp
```

### Required summaries

```text
counts by |S|
counts by penalty count
counts by model dimension
counts by rank/condition bucket
same-S group-size distribution
near-alpha test distribution
time weighted by |S| and penalty count
unsupported-envelope list
```

Known historical sanity values include:

```text
logical tests total                                = 240,489
conditional logical tests                          = 238,276
conditional residual requests                      = 476,552
canonical global unique conditional target|S       = 110,617
nonempty same-S groups                              = 8,634
S-affinity executed mgcv fits (historical metric)  = 273,284
```

The first five values are canonical hard gates. `273,284` is provenance-only
unless a hash-protected S-affinity execution trace is supplied and independently
recounted. If a canonical value differs, stop and explain the oracle or key
contract difference before continuing.

### Hard correctness gate

Phase 1 is offline and does not execute a new skeleton. It must independently
validate and inherit the frozen Phase 0 graph evidence:

```text
oracle_inherited_graph_gate = TRUE
oracle inherited SHD = 0
oracle inherited sepsets identical = TRUE
oracle inherited n.edgetests exact = TRUE
oracle inherited deletions identical = TRUE
new_candidate_graph_gate = NOT_APPLICABLE
```

Before the complete metadata pass, a representative parity gate must prove
that offline one-target fitting matches the actual two-target `regrXonS` data
layout for `|S|=1,2,3`, a larger additive multi-penalty case, rank-deficient
and near-constant cases, and a canonical near-alpha case.

### Exit condition

The artifact must identify every canonical case that later phases must support, including all high-condition, rank-deficient, near-constant, multi-penalty, and near-alpha cases.

---

## Phase 2 — Define a response-independent GAM setup contract

### Goal

Separate target-independent same-S structure from target-specific fitting state without changing mgcv semantics.

### Required abstraction

Create a versioned `PreparedSSetup` contract containing the response-independent state required by the actual regrXonS formula:

```text
semantic version
DatasetKey
PreparedSKey
sorted S
formula route
model matrix or equivalent basis representation
penalty blocks
penalty offsets
constraint representation
constraint null space or stable equivalent
rank metadata
centering/scaling metadata
smooth labels and dimensions
mgcv semantic/version fingerprint
```

Create a separate `TargetState`:

```text
target id
y
X'y or stable projected RHS
selected smoothing parameter vector
score
EDF
convergence state
residual/fitted fingerprints
```

### Required behavior

For targets sharing `S`:

```text
PreparedSSetup is built once
response-independent state is identical
TargetState remains independent
selected sp is not shared unless it is actually equal
```

### Implementation guidance

Extend/reuse:

```text
fastkpc_mgcv_extract_setup
fastkpc_mgcv_extract_retarget_setup
fastkpc_prepare_gpu_setup_state
setup_fingerprint and target_fingerprint helpers
```

Do not use a raw matrix hash as the only semantic check. Equivalent bases may differ by sign or orthogonal rotation. Validate column space, constraints, penalty action, and fixed-sp fitted/residual behavior.

### Required artifact

```text
fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/
```

### Required tests

Suggested:

```text
fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
fastkpc/tests/test_full_cuda_ci_target_retarget.R
```

### Gates

For all census same-S groups selected for the gate:

```text
response-independent setup reuse is valid
fixed-sp oracle residual parity passes
legacy dCov decision flips = 0
setup fingerprint collisions = 0
unsupported setup is explicit
```

The phase must include difficult cases from the census, not only `|S|<=2` easy examples.

### Exit condition

A later CUDA solver can accept one `PreparedSSetup` plus a matrix of targets without calling `mgcv::gam` once per target.

### Closure

Phase 2 is complete with the authenticated artifact:

```text
PreparedSSetup rows             = 8,634
TargetState rows                = 110,617
iteration setup groups          = 44
iteration target keys           = 270
iteration logical tests         = 44
qualification target keys       = 6,143
qualification logical tests     = 3,808
fixed-sp residual hashes exact  = TRUE
dCov decision flips             = 0
unsupported/fallback count      = 0
artifact path                   =
  fastkpc/artifacts/full_cuda_ci/prepared_s_contract_v1/
```

The artifact preserves version-pinned mgcv/regrXonS as the residual authority
and inherited Phase 0 graph evidence. It does not claim Phase 3 CUDA solve
authority.

---

## Phase 3 — Build a persistent, stable fixed-sp CUDA residual runtime

### Goal

Replace the current one-fit-at-a-time fixed-sp CUDA prototype with a persistent same-S, multi-target runtime while using oracle-selected smoothing parameters. This isolates linear algebra from smoothing-selection correctness.

### Phase 3A — Persistent resources

Refactor the current fixed-sp CUDA path so that it reuses:

```text
CUDA stream(s)
cuBLAS handle
cuSOLVER handle
prepared setup device buffers
workspace buffers
factorization buffers
output buffers
```

Prohibited hot-path behavior:

```text
create/destroy cuSOLVER handle per target
cudaMalloc/cudaFree per target
upload the same model matrix per target
download residuals before the dCov stage
cudaDeviceSynchronize after every small kernel
```

#### Verified Phase 3A milestone (2026-07-15)

Phase 3A is complete. It did not complete Phase 3; the verified Phase 3B
milestone is recorded below, and Phase 3C remains open.

```text
authenticated Phase 3 catalog and native DTO
one persistent CUDA stream/cuBLAS/cuSOLVER context
explicit deterministic cuSOLVER and pedantic/no-atomics cuBLAS config
one-time Prepared-S uploads
CUDA-built X0'Y RHS and safe single-target Cholesky
leased device-resident residual token and explicit shadow materializer
post-warm-up per-target allocation/handle creation = 0
persistent path faster than repeated single-target CUDA prototype
iteration safe targets = 172 with parity < 1e-7
stable targets = 98 explicit ERR_STABLE_PATH_NOT_IMPLEMENTED
```

Phase 3B status is recorded below.

### Phase 3B — True multi-target fixed-sp execution

The batch must support:

```text
one PreparedSSetup
many target columns
one target-specific sp vector per target
target-specific RHS
target-specific status
```

Set `true_batched_kernel = TRUE` only when the implementation actually launches fused/batched work rather than an R/C++ loop that repeats the old single-fit call.

#### Verified Phase 3B milestone (2026-07-16)

Phase 3B is complete. This does not complete Phase 3. Stable routes have not
been implemented by this milestone and remain explicit errors until Phase 3C.
The legacy repeated-handle bridge remains truthfully non-batched.

```text
one same-S native Y/SP upload phase per setup batch
one CUDA X0'Y RHS build per setup batch
fused target-specific system construction
true batched potrf/potrs
canonical mixed-batch output order
iteration true-batched targets = 160
iteration single safe targets = 12
iteration stable targets remain explicit errors = 98
post-warm-up allocation/handle creation = 0
```

Active next task: Phase 3C penalty roots and augmented QR/SVD.

### Phase 3C — Stable solve path

The current null-space normal-equation Cholesky path is not sufficient for every canonical case.

Implement:

```text
fast path:
  double-precision Cholesky when the measured system is safe

stable path:
  augmented QR/SVD or equivalent rank-revealing CUDA solve
  deterministic rank tolerance
  explicit penalty null-space handling
```

High-condition cases must not silently return a normal-equation answer that differs from mgcv.

#### Verified Phase 3C milestone (2026-07-20)

Phase 3C qualification is complete. The full Phase 3 artifact and graph
closure is recorded below.

```text
one-time individual QR roots, qualification matrices/rows 6,272 / 63,552
augmented QR and deterministic aggregate-penalty augmented SVD
aggregate root rank/pivot exact against test-only CPU LAPACK
C_magic numeric reference for every route and target
C_magic sqrt(double epsilon) SVD rank threshold
aggregate SVD one factor / two B builds per executed target
iteration 270/270 targets OK
qualification 6,143/6,143 targets OK
planned routes 3,889 / 190 / 2,064 exact
executed routes 3,889 / 190 / 2,064; declared reroutes 0 / 0
dCov 3,808 pairs, near-alpha 1,478, decision flips 0
unknown/CPU/approximate fallback 0
```

Active next task: Phase 4 full-CUDA single-penalty GCV objective and optimizer
parity for `|S| <= 2`, consuming the accepted Phase 3.5 contracts and guarded
dCov architecture decision.

### Required API shape

Suggested internal shape:

```text
PreparedSGpuHandle create_prepared_s_gpu(setup)
FixedSpBatchResult solve_fixed_sp_batch(handle, Y, SP)
DeviceResidualHandle[] result.residual_handles
```

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/
fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/
```

### Numerical gates

For oracle-selected sp:

```text
all outputs finite
residual max-abs and relative-L2 within recorded phase tolerance
fitted-value parity within recorded phase tolerance
legacy dCov p-value parity within recorded phase tolerance
decision flips = 0
high-condition cases use the declared stable path
```

The initial target tolerance should be no weaker than the current replay evidence unless a documented condition-specific bound is justified. Tolerance cannot waive a decision flip.

### Full graph gate

Use the fixed-sp CUDA runtime in shadow or oracle-sp backend mode:

```text
edge_count = 110 / 110
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
```

### Operational gates

```text
per-target CUDA allocation count = 0 after handle warm-up
per-target solver-handle creation count = 0
setup H2D upload count is O(same-S groups), not O(targets)
unknown fallback count = 0
```

### Exit condition

Fixed-sp fitting is a correct, stable, persistent CUDA service. Smoothing-parameter selection remains oracle-supplied until Phase 4/6.

### Completion evidence (2026-07-27)

Phase 3 is complete for the oracle-selected fixed-sp scope. Both production
artifacts were generated from source commit
`2b61721b63616f5100084ccafdd243a8c3647b82` and independently validated from
disk. Commit `558043f4908f0b0ed66b5d9bed6ff1bdb2e270a8` is a post-artifact
publication/reuse fix; it does not change the CUDA runtime or native binary,
and it does not relabel or rewrite the frozen artifact identities.

```text
oracle artifact                       =
  fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_oracle_sp_v1/
oracle manifest SHA-256               =
  7ba91e115243e4b4551b0d80c234ba18b3bc85c60e6dc583753df742bb94bab6
oracle summary SHA-256                =
  38ada38620ba45cfd70ed55e37b6b1b84f043f74c78f82eb036ac3276879fa82

shadow artifact                       =
  fastkpc/artifacts/full_cuda_ci/fixed_sp_cuda_full_shadow_v1/
shadow manifest SHA-256               =
  0efbbdd2dba998da792fd80f035bd703c87ca888f226a3cab2a501782f464c35
shadow summary SHA-256                =
  8df1ff4b0e7a800e5ecdd897256759bd1be7b743ed94080a5c4b4c66f991af57
executed native library SHA-256       =
  29a303e7d22bfd6b7de45a4456b239c93dc1d62c274f5a1114f5c98217151ab3

shards / PreparedSSetup / TargetState = 64 / 8,634 / 110,617
planned Cholesky / QR / SVD            = 73,158 / 4,210 / 33,249
executed Cholesky / QR / SVD           = 73,158 / 4,210 / 33,249
Cholesky->SVD / QR->SVD reroutes       = 0 / 0
non-OK / nonfinite / numeric failures  = 0 / 0 / 0
unknown / CPU / approximate fallback   = 0 / 0 / 0
per-target allocations/handles         = 0 / 0 after warm-up
implicit residual D2H / device sync     = 0 / 0

max fitted abs / relative-L2 diff       = 1.1930435e-08 / 8.5685251e-10
max residual abs / relative-L2 diff     = 1.1930435e-08 / 2.6844550e-09
qualification dCov rows                 = 3,808
qualification max p-value abs diff      = 6.9555695e-11
qualification decision flips/errors     = 0 / 0

logical CI rows                         = 240,489
direct / conditional rows               = 2,213 / 238,276
near-alpha all / conditional             = 1,529 / 1,478
decision flips / backend errors          = 0 / 0
Spectra fallbacks                        = 0
edge_count                               = 110 / 110
SHD                                      = 0
normalized sepsets identical             = TRUE
n.edgetests exact                        = TRUE
deletion trace exact                     = TRUE
first divergence                         = NOT_APPLICABLE
```

The oracle pure resume reused the completed artifact without creating a new
session or CUDA context. The shadow resume probe likewise created no new
session or context and left all 64 shard pairs unchanged, but the old
publisher then rejected volatile cross-session `input_hashes.csv` evidence.
`558043f` fixes completed-publication reuse, exact-JSON numeric round trips,
and the runner's completion fast path; focused, publication, artifact-contract,
and hostile stable-source/native-SHA tests pass. No full workload was rerun for
that post-artifact tooling fix.

---

## Phase 3.5 — Full-CUDA architecture feasibility and performance-budget gate

### Status and blocking rule

Phase 3 remains complete. Phase 3.5 passed its blocking gate on 2026-07-28.
Phase 4 may now become accepted campaign authority when its own exit gates
pass, using compatible versions of the frozen Phase 3.5 contracts.

#### Accepted Phase 3.5 closure (2026-07-28)

The selected architecture is:

```text
exact CUDA decision screen for every conditional pair
  -> screen p in closed interval [0.05, 0.15]
     -> deterministic CUDA legacy full-eig refinement
  -> final canonical-order decision replay
```

This architecture is `GO for Phase 8 implementation and full qualification`.
That GO is an architecture-feasibility decision, not a production-backend or
Phase 10 promotion. Exact CUDA values outside the guard are internal decision
screens; they are not claimed to be universal legacy p-values.

Candidate decisions:

```text
Candidate A full eig globally       REJECT; 827,349.8 ms diagnostic bound
Candidate A two-sided partial eig   REJECT; 1,634,351.4 ms diagnostic bound
Candidate B block Krylov core       REJECT/NOT-GO; 34,647.2 ms core-only bound
Candidate C exact CUDA authority    REJECT; 92 complete-conditional flips
guarded C-screen/A-refinement       GO; zero final flips and 25,527.978 ms dCov bound
```

Candidate B's bound excludes convergence, finalization, pair work, and parity;
it is rejection evidence and is not used as the feasibility bound. No best or
average microbenchmark is multiplied by the global workload in the accepted
performance model.

Measured numerical and scale evidence:

```text
Scale A qualification       3,808 pairs / 2,061 groups / 6,143 components
  screen flips 92; guarded pairs 1,142; final flips 0
  refined components/groups 1,532 / 385
  maximum refined p-value error 6.955136e-11
  dCov host boundary 12,845.19 ms

Scale B campaign slice      21,380 pairs / 192 groups / 7,236 components
  48 groups in each reuse quartile
  screen flips 9; guarded pairs 78; final flips 0
  refined components/groups 124 / 29
  maximum refined p-value error 1.826568e-11
  dCov host boundary 1,136.45 ms

complete conditional trace 238,276 pairs / 8,634 groups / 110,617 components
  levels 1 through 7 complete
  all 92 exact-screen flips inside the guard
  guarded pairs 1,142; final flips 0
  maximum refined p-value error 6.955136e-11
  component CUDA 13,042.082 ms
  pair/gamma CUDA 721.365 ms
  dCov host boundary 16,766.36 ms
```

All measured routes report zero residual/component D2H and zero CPU numerical
dCov/gamma authority. The deleting fixture at logical id 239277 restores the
legacy independent decision with p-value error `8.2768e-12` and statistic
error `2.68519e-11`.

Accepted cache, memory, and conservative performance evidence:

```text
LRU capacities               2 / 8 / 16 / 47 components
misses                  224,619 / 201,455 / 166,794 / 110,617
evictions               207,351 / 141,879 / 83,736 / 0
declared peak device bytes   632,071,874
declared RTX 4090 headroom   25,125,148,990 bytes

component p95 bound          18,337.636 ms <= 35,000 ms
pair/gamma p95 bound          7,190.342 ms <= 12,000 ms
dCov total bound             25,527.978 ms <= 47,000 ms
full campaign used/reserved  98,529 ms <= 120,000 ms
```

The full campaign bound retains the entire allocation for every unimplemented
Phase 4/7/9/10 owner. The reusable development corpus is measured, the
metamorphic protocol is frozen without claiming complete Phase 3.5 execution
of every future transformation, and promotion holdout v1 remains
`SEALED_NOT_RELEASED` with no repository payload.

Accepted identities:

```text
native binary SHA-256
  9b28d0045eedf013955c673f68e1a5b138b4063d901920bb720cb85a30f688e5
prepublication evidence manifest SHA-256
  102768a44b09e6afe43d8eca50bb3d96fd81d63e1249bffd99c819c5d0d19de2
Phase 3.5D vertical artifact manifest SHA-256
  ae36e739d9674ad3c20123d90f25a2573def925ca7f0edbeda58ef466fc631f3
Phase 3.5 feasibility artifact manifest SHA-256
  87c66555edb0912d5d50e80eb9d7601e4da9800c61b7de0f9691d76d8c332185
feasibility producer semantic identity
  66a03682777339be730068be1f471ed7b2839cef5290e7542900ebea5e8a82a8
```

The accepted on-disk artifact is
`fastkpc/artifacts/full_cuda_ci/phase35_feasibility_v1/`. Its validator binds
the tracked contract snapshots, producer/attestation/receipt namespaces,
source closure, native and benchmark binaries, evidence hashes, complete pair
decisions, cache/memory tables, corpus policy, and conservative performance
classes. Payload and manifest tamper tests fail closed.

### Goal

Before implementing full GCV and multi-penalty support, prove that the eventual
one-call, device-resident architecture has:

```text
a stable but evolvable semantic interface
an auditable producer/validator/receipt identity model
at least one viable CUDA dCov architecture
a capacity-bounded memory and cache design
a conservative path to the 120-second campaign target
```

### Explicit non-goals

Phase 3.5 does not:

```text
implement complete single-penalty or multi-penalty GCV
implement the native setup builder
implement or promote the final production dCov backend
replace Phase 8 full dCov qualification
replace Phase 10 complete campaign benchmarking
reopen or repeat Phase 3 fixed-sp correctness closure
```

Phase 3.5 ends when one architecture is demonstrably feasible. It does not
require optimizing every candidate or completing every production integration
detail.

### 3.5A — Version-controlled semantic contracts

Contracts are tracked source, not untracked artifact authority. The intended
layout is:

```text
fastkpc/inst/contracts/full_cuda_ci/
  architecture_contract_v1.json
  numerical_contract_v1.json
  artifact_identity_contract_v1.json
  reference_machine_v1.json
  performance_budget_v1.json
  development_qualification_corpus_v1.json
  metamorphic_contract_v1.json
  promotion_holdout_manifest_v1.json
```

Tracked R/C++ code must parse, canonically serialize, hash, and validate these
contracts. Every artifact records:

```text
contract name and semantic version
canonical contract SHA-256
exact contract snapshot
producer source commit
```

The validator recomputes every snapshot hash and checks it against the tracked
contract. Artifact contents cannot become the sole contract authority.

Contract versions use:

```text
major: breaking semantic or API change
minor: backward-compatible capability addition
patch/attestation revision: validator or explanatory change that does not
  change producer semantics
```

Accepted phases consume a compatible ABI version. An incompatible ABI change
requires a major-version increment, an explicit migration or adapter decision,
renewed qualification for affected phases, and no reinterpretation of old
artifacts.

The numerical contract must define explicit formulas and policies for:

| Numerical layer | Required gate |
|---|---|
| RSS and GCV/Cp score | absolute plus relative error |
| EDF | absolute error |
| selected log-sp | diagnostic compatibility; bit identity only if required by semantics |
| fitted and residual vectors | max-absolute plus relative-L2 error |
| dCov statistics and moments | absolute plus relative error |
| p-value | absolute error with a separate near-alpha rule |
| decision and graph semantics | zero flips and exact graph-semantic equality |

It also freezes denominator floors, condition buckets, boundary handling,
NaN/Inf policy, and the rule for changing a tolerance. Any semantic tolerance
change requires a new contract version and renewed affected qualification.
Observed Phase 3 maxima are evidence inputs, not automatic future thresholds.

### 3.5B — Opaque semantic ABI

Define these logical objects:

```text
PreparedSGpuHandle
TargetOptimizerStateHandle
DeviceResidualHandle
DcovComponentHandle
LogicalCiBatchHandle
CompactCiResult
```

Handles are opaque, versioned, capability-queryable, lifetime-controlled,
backend-owned objects. Raw pointer values and internal layouts are not artifact
formats. The ABI freezes:

```text
creation, destruction, ownership, and lease semantics
stream/event dependency and asynchronous completion semantics
cache identity and semantic fingerprints
error/status propagation
allowed compact host-visible diagnostics
large-payload host-transfer prohibition
capability discovery and version negotiation
eviction and reconstruction semantics
selected-sp coordinates, penalty scaling, rank, and null-space semantics
```

It does not freeze:

```text
struct or buffer layout
matrix leading dimensions
workspace layout
eigensolver representation
kernel launch topology
```

At minimum, capability queries expose:

```text
abi_major / abi_minor
capabilities
backend_semantic_version
producer_contract_hash
device_residency_flags
```

`TargetOptimizerStateHandle` remains opaque until Phase 4 establishes optimizer
semantics. `DcovComponentHandle` promises only a device-side semantic object
consumable by the compatible pair evaluator; it does not promise full
eigenvectors, partial eigenpairs, low-rank factors, block-Krylov state, or a
dense centered matrix.

### 3.5C — Three-layer artifact identity

Every later campaign artifact uses three separate identities.

Producer semantic identity describes the implementation that produced the
payload:

```text
producer source closure hash
native binary SHA-256
route semantic version
ABI and numerical contract hashes
dataset/corpus and oracle identities
backend configuration
compiler/build recipe identity where required
```

It is immutable after publication.

Validator/attestation identity describes a validation act:

```text
validator source closure hash and semantic version
validator contract hashes
validation timestamp and environment summary
validation result
attested producer identity
```

One producer artifact may have multiple append-only attestations. A new
validator may validate an old producer artifact, but it must not replace the
producer identity or rewrite the producer manifest.

Volatile execution receipts are audit evidence only:

```text
PID and session ID
CUDA context/session identifiers
filesystem path and inode
temporary/staging directories
timestamps and host-process details
```

They must not define semantic equality, impersonate producer identity, or
prevent valid cross-session linkage.

### 3.5D — Minimal vertical structural prototype

Build the minimum final-shape slice:

```text
one native control-plane call
  -> authenticated canonical request slice
  -> Phase 3 device residual handles
  -> candidate dCov component handles
  -> CUDA pair evaluation
  -> compact host results
  -> deterministic replay
```

This slice may use oracle-selected sp and Phase 2 prepared setups. It must
demonstrate:

```text
residuals remain device-resident
production-sized residual/component D2H is zero
CPU numerical dCov authority is zero
only compact p-value/status diagnostics return to the host
handle lifetimes are bounded and leak-free
cache eviction does not alter results
```

The prototype validates object boundaries, ownership, residency, and timing
instrumentation. It does not repeat the Phase 3 full-graph correctness claim.

### 3.5E — dCov architecture bake-off

Evaluate the Phase 8 candidates before committing to a production
implementation. Each candidate records:

```text
component representation and construction parity
pair statistic parity
gamma moment and p-value parity
near-alpha and declared risk-corpus decisions
component-build and pair-evaluation time
H2D/D2H bytes and host synchronization count
workspace, persistent, and cacheable bytes
failure and convergence behavior
```

Two real measured scales are mandatory:

```text
Scale A — qualification scale
  covers every declared semantic and risk class

Scale B — campaign-slice scale
  materially larger canonical trace slice
  representative reuse distribution and dense skeleton level
  realistic batching, cache pressure, and scheduling
```

Also cover component-build-dominated and pair/reuse-dominated workloads, the
complete near-alpha corpus, and the complete declared risk corpus. Neither
scale may use simulated backend timing.

The conservative full-workload upper bound includes measured fixed overhead,
class-stratified component and pair costs, observed reuse/cache behavior,
allocator/workspace and synchronization overhead, a conservative throughput
bound, and declared contingency. It must not multiply one best or average
microbenchmark time by a global count.

A candidate is `GO for Phase 8 implementation and full qualification` only
when:

```text
representative component, pair, moment, and p-value parity pass
complete near-alpha decision flips = 0
complete declared risk-corpus decision flips = 0
both measured scales fit the allocated dCov budget
the conservative full-workload upper bound fits that budget
CPU numerical dCov authority = 0
production large-payload residual/component D2H = 0
unsupported and numerical failure behavior is fail-closed
```

This GO decision selects an architecture for Phase 8. It does not promote a
final backend.

### 3.5F — Cache and memory model

Use measured allocation data, not raw matrix dimensions alone. Account for:

```text
allocator metadata, alignment, pitch, and fragmentation
cuBLAS/cuSOLVER workspaces
stream and event state
prepared setup and optimizer state
residual and dCov component persistent bytes
temporary pair-batch workspace
peak concurrent batch state
```

Use the canonical trace and intended legal schedules to compute:

```text
reuse-distance and future-use distributions
peak live set and dense-level pressure
cache miss curves by capacity
eviction/rebuild cost
residual/component memory split
```

The accepted policy is capacity-bounded, memory-accounted, deterministic in
semantics, result-invariant under eviction, and independent of allocator luck.
It declares reference-GPU headroom, fails closed on OOM, and never publishes a
partial graph after OOM.

### 3.5G — Corpus policy

Keep three distinct corpora.

The reusable development qualification corpus covers rank deficiency,
condition buckets, extreme sp, near constants, representative penalty counts,
and declared numerical risks.

The reusable metamorphic corpus covers:

```text
conditioning-column permutation
basis sign flip and equivalent orthogonal rotation
batch split/merge
stream-count and cache-capacity variation
standalone versus batched execution
```

The sealed promotion holdout is opened only in Phase 10 after candidate source,
configuration, and contracts are frozen. Its tracked manifest records identity,
hash, and release protocol; ordinary development cannot access the payload.
Opening is recorded. If results cause an implementation change, that corpus
becomes regression evidence and a new sealed version is required for another
unbiased promotion claim.

### 3.5H — Performance budget

`performance_budget_v1` separates feasibility from final measurement:

```text
Phase 3.5 feasibility gate:
  measured component budgets
  + conservative full-workload bounds
  + declared contingency
  <= 120 seconds

Phase 10 actual campaign gate:
  five-run real complete warm campaign median <= 120 seconds
```

The tracked reference-machine contract freezes timing boundaries, input-copy
and native-setup inclusion, cache construction, build exclusion, cold/warm
definitions, GPU power/clock policy, CPU affinity, and thread policy. Cold and
warm metrics are reported independently.

Final promotion must simultaneously satisfy:

```text
warm median <= 120 seconds on reference_machine_v1
warm median <= 0.80 * same-run correct baseline median
every measured run passes every graph-semantic and authority gate
```

The stretch target is a warm median of at most 60 seconds. Phase budgets must
leave explicit contingency and a credible allocation for not-yet-implemented
GCV and native setup work.

### Phase 3.5 exit gate

Phase 3.5 passes only when all of the following versioned contracts and
artifacts are accepted:

```text
architecture_contract_v1
  opaque semantic ABI and compatibility rules accepted

numerical_contract_v1
  tracked tolerances, decision rules, snapshots, and hashes accepted

artifact_identity_contract_v1
  producer, validator/attestation, and volatile receipt identities separated

dcov_architecture_bakeoff_v1
  at least one candidate is GO for Phase 8 under two-scale measured evidence

cache_memory_model_v1
  measured memory, declared headroom, deterministic eviction, and OOM gates pass

performance_budget_v1
  cold/warm boundaries frozen and measured budgets plus contingency <= 120 sec

corpus_policy_v1
  development, metamorphic, and sealed promotion protocols frozen
```

Passing Phase 3.5 proves architectural feasibility. It does not claim
completion of Phase 4, Phase 7, Phase 8, Phase 9, or the actual Phase 10
120-second campaign gate.

---

## Phase 4 — Implement full-CUDA single-penalty GCV for `|S| <= 2`

### Status

Phase 4 passed its local, full-shadow, identity, and performance gates on
2026-07-29. Phase 5 is now the earliest incomplete phase. This is a Phase 4
scope decision, not promotion of the final production backend.

#### Accepted Phase 4 closure (2026-07-29)

The accepted route performs response-independent setup preparation once, then
uses CUDA for batched target projection, dense objective evaluation,
continuous `log(sp)` refinement, risk-gated exact replay, and the selected-sp
residual solve. Refinement uses bounded proposals, rejected-trial bracketing
with deterministic step halving, explicit convergence/flat-objective states,
iteration limits, and boundary probes. Exact replay is selected only from
generic numerical state; no setup key or target ID controls routing.

Full single-penalty oracle evidence:

```text
setups / targets                    1,174 / 44,941
dense objective cells               9,020,884
maximum RSS absolute error          4.001777e-11
maximum EDF absolute error          9.947598e-14
maximum GCV score absolute error    1.292300e-13
maximum selected log-sp error       2.231308e-10
optimizer iteration mismatches      0
optimizer Hessian-state mismatches  0
required/preserved transcripts      5,622 / 5,622
legacy mgcv target calls            0
CPU score/optimizer/fallback calls  0 / 0 / 0
```

The compact full-shadow route covered all `177,952` logical `|S| <= 2` CI
tests. It used 36,365 spectral-only targets and 8,576 exact CUDA replays,
including 6,718 generic numerical-risk replays. Maximum residual oracle
relative-L2 error was `4.174897e-10`, maximum downstream p-value absolute
error was `4.889811e-11`, and both residual and downstream dCov decision-flip
counts were zero. It performed no selected-sp R roundtrip and no implicit
residual D2H materialization.

Mixed-route graph evidence:

```text
edge count                         110 / 110
SHD                                0
adjacency / sepsets identical      TRUE / TRUE
n.edgetests / deletions identical  TRUE / TRUE
explicit legacy fallbacks          60,324, all |S| > 2
unknown / approximate fallbacks    0 / 0
```

Five measured repetitions on physical GPU 0 had a CUDA median of `7,378 ms`
versus `13,058 ms` for the same-corpus `r-cpu-spectral` baseline, a ratio of
`0.5650`. Every candidate result and counter signature was identical, the
warm median was below the Phase 4 `25,000 ms` ceiling, and both absolute and
relative backend gates passed.

Accepted identities:

```text
producer source commit             54c2267a01debde8abfe4477eeb613e13d74ed2c
producer source closure SHA-256    cdcaf2ef36c2d037e3c25ffec5146bcb1966f7853371eaff7622424c4a97cac8
native binary SHA-256              2cfbdef063f757b95a0c1a6a0161a5066ccb808e4bdb4f41f2c4e85747cc225f
oracle producer identity           8ad2d090301e21ddc3f50b1c671f5fa505bf3adb948242b895e54aaa3bdac2d9
full-shadow producer identity      9a610468edff3219fe8c0b92013e9ffb4d4d3bcb39d55e22403dcc39b9e05464
backend producer identity          fb8bc399dc9a609cd4fa9f9a4343962f2e0825fc600262716299fbdc0ba865bd
```

The accepted artifacts are the three directories listed under Required
artifacts below. Their validators bind the tracked contracts, 91-file source
closure, native binary, semantic payload hashes, validator attestations, and
volatile receipts. Payload, manifest, attestation, and receipt tamper tests
all fail closed. Phase 4 retires `r-cpu-spectral` authority for the canonical
single-penalty envelope; `|S| > 2` remains an explicit legacy oracle fallback
until Phases 5 and 6 pass.

### Goal

Remove `r-cpu-spectral` smoothing scoring and selection from the single-penalty compatible path.

Phase 3.5 now permits Phase 4 to become accepted campaign authority when the
Phase 4 gates pass. Phase 4 must consume compatible versions of the tracked
architecture, numerical, identity, machine, budget, and corpus contracts.

### Current starting point

The repository already has response-free spectral preparation, target batching helpers, grid scoring helpers, selected-sp fixed CUDA solves, and same-S caches. These are substrate, not the final route.

### Phase 4A — Objective parity

For each single-penalty setup, implement CUDA evaluation of:

```text
RSS(sp)
EDF(sp)
GCV/Cp score(sp)
validity and rank status
```

Validate the full score curve against the version-pinned oracle and
`numerical_contract_v1` over a dense log-sp grid, including extreme and
near-optimal regions.

### Phase 4B — Target-batched scoring

For one `PreparedSSetup` and many targets:

```text
project all target RHS values in batched GEMM
score all target x candidate-sp combinations on CUDA
keep target-specific minima and status
avoid host materialization of the score matrix unless in sampled shadow mode
```

### Phase 4C — Continuous compatible refinement

A fixed coarse grid is not sufficient as the final authority.

Implement a deterministic log-sp refinement that reproduces the legacy optimum closely enough to pass full graph gates. It must include:

```text
bracketing
flat-objective handling
step acceptance / step halving
convergence criteria
iteration limits
boundary status
```

Record selected `sp`, score, EDF, iteration count, and convergence reason for every target.

For the development risk corpus, near-alpha cases, boundary cases, and every
oracle non-fully-converged case, also preserve an optimizer transcript:

```text
iteration and current log-sp
objective, gradient, and Hessian or approximation
proposed and accepted steps
step-halving count
boundary and rank-path flags
termination reason
```

The CUDA optimizer need not reproduce every internal mgcv step. It must explain
exceptional states and preserve compatible final objective, fit, residual,
decision, and accepted boundary/convergence semantics. Reporting every case as
converged is not an acceptable substitute.

### Phase 4D — Integrate with the persistent fixed-sp solver

The selected target-specific sp values must feed the Phase 3 CUDA batch solver without returning to R.

### Required diagnostics

```text
single_penalty_targets
cuda_gcv_targets
cuda_gcv_batches
cuda_gcv_iterations
cuda_gcv_nonconverged
cuda_gcv_boundary_targets
cuda_gcv_score_ms
cuda_selected_sp_solve_ms
sp_selection_backend_executed
gcv_score_backend_executed
legacy_mgcv_target_calls
```

For this phase:

```text
sp_selection_backend_executed = "cuda"
gcv_score_backend_executed = "cuda"
legacy_mgcv_target_calls = 0 for |S|<=2
```

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/single_penalty_cuda_gcv_oracle_v1/
fastkpc/artifacts/full_cuda_ci/single_penalty_cuda_gcv_full_shadow_v1/
fastkpc/artifacts/full_cuda_ci/single_penalty_cuda_gcv_backend_v1/
```

### Local gates

For the entire `|S|<=2` census envelope:

```text
objective curves agree under numerical_contract_v1
selected-sp diagnostics are explained
residual decision flips = 0
downstream legacy dCov decision flips = 0
unsupported/fallback count = 0 for the canonical |S|<=2 envelope
```

### Full graph gate

Run a mixed development route:

```text
|S|<=2: CUDA GCV + CUDA residuals
|S|>2:  explicit legacy oracle fallback
legacy dCov authority retained
```

Required:

```text
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
all fallback keys are |S|>2 and explicitly recorded
```

### Performance gate

The `|S|<=2` CUDA path must beat the current `r-cpu-spectral + repeated fixed-sp solve` path on the same request corpus. Do not promote a local kernel win if full residual wall time regresses.

### Exit condition

Every canonical single-penalty target is selected and fitted on CUDA with zero graph-semantic drift.

---

## Phase 5 — Implement a C++ multi-penalty GAM semantic replica

### Goal

Reproduce the actual `|S|>2` additive-smooth mgcv residual semantics in a stable C++ shadow implementation before attempting CUDA acceleration.

This is the main correctness phase for the difficult GAM cases.

### Scope

```text
target ~ s(S1) + ... + s(Sk)
Gaussian identity
multiple penalty blocks
GCV.Cp semantics used by the oracle
residual output only
```

### Required algorithmic pieces

```text
constraint-aware penalized least squares
stable rank-revealing solve
log smoothing-parameter vector
objective evaluation
objective gradient
usable Hessian or stable approximation
Newton or quasi-Newton step
step halving / steepest-descent fallback
boundary and non-convergence handling
mgcv-compatible rank tolerance
```

The implementation may initially consume the Phase 2 mgcv-extracted `PreparedSSetup`. It must not rebuild a different statistical model.

### Numerical safety rule

Do not use only:

```text
X'X + sum(lambda_j S_j)
```

with unconditional Cholesky for every case. The canonical workload includes very ill-conditioned examples. Use an augmented-system QR/SVD or another numerically stable formulation when required.

### Required modes

```text
shadow:
  legacy mgcv authoritative
  C++ computes selected sp and residuals

backend-dev:
  C++ authoritative only for fully qualified cases
  explicit legacy fallback otherwise
```

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/multi_penalty_cpp_oracle_v1/
fastkpc/artifacts/full_cuda_ci/multi_penalty_cpp_full_shadow_v1/
fastkpc/artifacts/full_cuda_ci/multi_penalty_cpp_backend_v1/
```

### Required diagnostics

```text
penalty_count
optimizer_iterations
step_halving_count
rank_path
condition bucket
convergence code
selected log-sp vector
score and EDF
residual errors
p-value errors
near-alpha status
fallback reason
```

### Local gates

For every canonical `|S|>2` target from the census:

```text
finite result or explicit fail-closed status
residual decision flips = 0
downstream legacy dCov decision flips = 0
first-divergence artifact empty
high-condition cases use stable path
```

### Full graph gate

Use:

```text
|S|<=2: accepted Phase 4 CUDA residual path
|S|>2:  C++ multi-penalty candidate
legacy dCov authority retained
```

Required:

```text
edge_count = 110 / 110
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
canonical fallback count = 0 before Phase 5 completion
```

### Performance policy

Correctness is the goal. A slower C++ semantic replica is acceptable as a shadow milestone. Do not optimize or port to CUDA until the full graph gate is clean.

### Exit condition

The complete regrXonS semantic subset has a non-R, stable, zero-drift numerical specification.

### Accepted closure evidence

Accepted on 2026-07-29 from producer `cc336a9`: all 16 authenticated-shard
partitions exactly cover 7,460 setups, 65,676 targets, and 60,324 logical CI
rows. The C++ optimizer has zero iteration, score-call, convergence, Hessian,
and rank mismatches; zero fallback; zero downstream or near-alpha decision
flips; and no unconditional normal-equation path. Mixed replay with the
accepted Phase 4 route returns 110 / 110 edges, SHD 0, and identical adjacency,
sepsets, `n.edgetests`, and deletion trace. The three required artifacts above
independently validate and reject payload, manifest, attestation, and receipt
tampering. The canonical Phase 5 corpus has identity constraints and `H=NULL`
for every setup, as authenticated by the Phase 1 metadata. Phase 5 is COMPLETE;
Phase 6 may use this immutable C++ result as its numerical oracle.

---

## Phase 6 — Port multi-penalty same-S target batches to CUDA

### Status

Phase 6 passed its local, full-corpus residual, graph, identity, and performance
gates on 2026-07-30. Phase 7 is now the earliest incomplete phase. This closes
target-dependent GAM numerical work; it does not yet remove the Phase 2
R/mgcv setup provider or promote the final Phase 10 route.

#### Accepted Phase 6 closure (2026-07-30)

The accepted runtime uploads each `PreparedSSetup` once, gives every target an
independent optimizer state, executes same-S target batches on CUDA, and uses
bounded independent prepared streams across setups. Objective, gradient,
Hessian, step acceptance, boundary probes, guarded QR/SVD solves, GEMM fitted
values, and residual formation are CUDA authoritative. The reference machine
run used physical GPU 0 with 64 concurrent setup streams.

All 16 authenticated partitions merged with exact coverage:

```text
setups / targets / logical rows       7,460 / 65,676 / 60,324
penalty-count range                   3 through 7
maximum coefficient dimension        64
maximum selected log-sp error         9.827357e-07
maximum score absolute error          2.481615e-11
maximum EDF absolute error            9.985751e-09
maximum downstream p-value error      9.055423e-11
iteration / score-call mismatches     0 / 0
objective / step-halving mismatches   0 / 0
boundary / convergence mismatches     0 / 0
Hessian-state / rank mismatches       0 / 0
legacy mgcv target calls              0
CPU multi-penalty solves / fallback   0 / 0
CUDA optimizer errors                 0
```

Generic stability telemetry identified 182 replay candidates across all six
risk reasons. It screened 119 low-risk candidates before replay, executed 63,
selected 10 numerically material corrections, and recorded zero replay
errors. All 10 corrections retained by the unscreened qualification corpus
remain selected. Thirty-three terminal boundary confirmations produced 31
accepted and two rejected probes through the strong-delta, identity-tie, and
delta-identity modes. Physical evaluation and factorization counts both equal
`1,394,102`; no hidden evaluation or fallback is present.

Full residual graph evidence:

```text
edge count                         110 / 110
SHD                                0
adjacency / sepsets identical      TRUE / TRUE
n.edgetests / deletions identical  TRUE / TRUE
residual numerical fallbacks       0
downstream / near-alpha flips      0 / 0
```

The authenticated same-trace residual performance evidence includes the
accepted Phase 4 optimizer and selected fit plus Phase 6 setup upload,
optimization, GEMM, and residual work:

```text
candidate residual wall time       252,393.596 ms
legacy-mgcv 20-core baseline       328,075.687 ms
candidate / baseline ratio         0.769315
relative performance gate          PASS
```

Accepted identities:

```text
producer source commit             1ff656a7de649d0ea762fc0d8679e586aaeaef39
producer source closure SHA-256    c8730a438617eeffdf983d678a33292ec6edd9c44b55880d05d9f98f45cff686
native binary SHA-256              e29b997c9e92da92c92482e9e456d7eec1a814c945292d055ccfd7d651d0f285
oracle producer identity           08399f638dd994903d3761e8e386334ab3813ab806e0fa59a41623904280a7d1
full-shadow producer identity      b600fe36e4d198313b4bcfc2bf3946bf1b528ac428b66bd3b53d9f42257f3f51
backend producer identity          4c591e33f344bc89e36d1ea06b941ac2ebf7fa35c2a5009ef6bca0b5f209631e
```

The three required artifact directories below independently validate their
payload manifests, producer envelopes, source/native identities, validator
attestations, and execution receipts. Payload, manifest, attestation, and
receipt tampering all fail closed. Phase 6 is COMPLETE; Phase 7 may replace
setup construction while treating this residual route as immutable authority.

### Goal

Move Phase 5 smoothing optimization and residual solves to a persistent CUDA runtime.

### Required execution shape

For each same-S group:

```text
upload PreparedSSetup once
batch target RHS/projections
maintain one log-sp vector per target
maintain one active/converged mask per target
batch objective/gradient/Hessian work
apply target-specific step acceptance
batch stable penalized solves
keep residuals on device
```

Targets may converge in different iteration counts. Do not force one global optimizer state.

### Required CUDA features

```text
persistent workspace
batched GEMM
batched small-matrix factorization where safe
stable QR/SVD path for difficult targets
deterministic double-precision reductions
bounded stream concurrency
explicit CUDA error/status per target
```

### Forbidden shortcut

Do not mark the phase complete by looping over targets and invoking the old single-fit CUDA entrypoint. That can remain a reference path, but final diagnostics must distinguish it from true batched execution.

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/multi_penalty_cuda_oracle_v1/
fastkpc/artifacts/full_cuda_ci/multi_penalty_cuda_full_shadow_v1/
fastkpc/artifacts/full_cuda_ci/full_residual_cuda_backend_v1/
```

### Local gates

```text
all canonical |S|>2 targets covered
legacy mgcv target calls = 0
CPU multi-penalty solve count = 0
CUDA non-convergence count = 0 or explicitly resolved by a stable CUDA path
residual decision flips = 0
legacy dCov decision flips = 0
```

### Full residual gate

At the end of this phase, the full canonical residualization path must be CUDA authoritative, although setup extraction may still be supplied by the version-pinned Phase 2 provider:

```text
all conditional targets use CUDA GCV/fit
edge_count = 110 / 110
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
residual numerical fallback count = 0
```

### Performance gate

The full CUDA residual route must beat the best correct legacy-mgcv residual provider on the same full skeleton trace. Use end-to-end residual wall time, not only summed kernel time.

### Exit condition

All target-dependent GAM numerical work for the canonical workload executes on CUDA with zero graph-semantic drift.

---

## Phase 7 — Build the native setup generator and remove R/mgcv from the CI loop

### Goal

Remove the remaining runtime dependence on R/mgcv setup construction while preserving the exact regrXonS subset.

### Required native setup support

```text
|S|=1 thin-plate smooth setup
|S|=2 joint thin-plate smooth setup
|S|>2 additive one-dimensional smooth setup
legacy default basis dimension semantics
centering and scaling
penalty construction
identifiability constraints
null-space/rank metadata
versioned semantic fingerprint
```

The setup builder may execute on the C++ host. Basis evaluation may move to CUDA later if profiling proves it material. The hard requirement is no R callback and no `mgcv::gam`/`smoothCon` call inside the native skeleton run.

### Validation method

Raw basis columns may differ by sign or rotation. Compare:

```text
model-space/projector equivalence
constraint satisfaction
penalty operator equivalence in a canonical subspace
rank and null-space dimensions
fixed-sp fitted/residual parity
selected-sp fitted/residual parity
legacy dCov decisions
```

### Required modes

```text
native-setup-shadow:
  build both mgcv oracle setup and native setup
  compare semantics

native-setup-backend:
  native setup authoritative
  fail closed if unsupported
```

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/native_setup_oracle_v1/
fastkpc/artifacts/full_cuda_ci/native_setup_full_shadow_v1/
fastkpc/artifacts/full_cuda_ci/native_setup_backend_v1/
```

### Hard gates

```text
R callback count in skeleton CI loop = 0
legacy mgcv fit count = 0
legacy mgcv setup count = 0
native setup unsupported count = 0 on canonical workload
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
```

### Exit condition

The compatible GAM residual service is self-contained in C++/CUDA for the complete canonical semantic envelope.

### Accepted Phase 7 closure (2026-07-31)

The native C++ setup builder now covers every canonical `|S|=1..7` setup:

```text
setups                                  = 8,634
single- / multi-penalty setups          = 1,174 / 7,460
targets                                 = 110,617
logical tests                           = 240,489
native setup unsupported count          = 0
legacy mgcv setup / fit count           = 0 / 0
R callback count                        = 0
CPU residual numerical solve count      = 0
unknown / approximate fallback count    = 0 / 0
edge count                              = 110 / 110
SHD                                     = 0
adjacency / sepsets / n.edgetests       = exact / exact / exact
canonical deletion trace                = exact
```

All 8,634 oracle-shadow setup comparisons have exact model matrices,
projectors, constraints, penalty operators, rank/null-space metadata, QR,
penalty roots, and initial smoothing parameters. The authoritative native
setup route then passes the complete Phase 4 and Phase 6 selected-fit and
downstream decision gates. Maximum selected log-sp error is
`2.231309e-10` for single-penalty targets and `9.827357e-07` for
multi-penalty targets; maximum single-penalty fitted/residual error is
`6.223058e-10`, and maximum multi-penalty fitted GEMM error is
`1.110223e-14` with exact residual identity.

Accepted artifacts:

```text
fastkpc/artifacts/full_cuda_ci/native_setup_oracle_v1/
  manifest SHA-256 = a01230a39177e995c2c7290c587706f5478c2a7905ddcba60a012a7299925389

fastkpc/artifacts/full_cuda_ci/native_setup_full_shadow_v1/
  manifest SHA-256 = 6342295fd09081c2515c6380c6deae9d3d5b5195c1644d44d9e3f8c7d848bc7e

fastkpc/artifacts/full_cuda_ci/native_setup_backend_v1/
  manifest SHA-256 = e84e8597f8876c78b63050932190600267fd858c260d6918b8740ad89ea760dc
```

Each artifact contains all 26 standard and phase-specific files and validates
its 110,617 case rows, 8,634 rank/condition rows, 1,478 near-alpha rows, 96
authenticated raw runs, graph payloads, producer identity, source closure,
binary identity, tracked contract snapshots, and content hashes. The accepted
source evidence SHA-256 is
`65ee9f8cd579b090bd27656a62cc642ef229705a723dc4c89566740bad4d91c3`.
Phase 7 is complete; Phase 8 is next.

---

## Phase 8 — Build a promotable legacy-compatible CUDA dCov backend

### Goal

Replace the CPU Spectra legacy dCov authority with a device-resident CUDA implementation that completes the full workload and preserves every canonical decision.

### Important policy

Phase 8 implements and fully qualifies the architecture selected as GO by the
Phase 3.5 bake-off. It is not the first architecture-selection point. The
candidate descriptions below define the Phase 3.5 comparison set and retained
research alternatives; a different architecture requires a new accepted
bake-off artifact.

Do not assume the current host-driven `cuda_spectra` design must be the final solution. Benchmark architectures by full-route wall time and zero-drift correctness.

### Required component-level architecture

A residual component is reused across many CI pairs. The preferred design is:

```text
DeviceResidualHandle
  -> distance component built once
  -> centered/eigen or low-rank component built once
  -> DcovComponentHandle cached by ResidualKey

Dcov pair batch
  -> consume two component handles
  -> compute nV2, mean, variance, alpha, beta, p-value
```

Do not recompute a residual component eigensystem for every pair when the same target|S residual recurs.

### Candidate A — CUDA full/partial symmetric eigensolver

Because the canonical sample size is 351, benchmark a stable cuSOLVER-based full or selected eigen route per unique residual component. Cache only the representation needed by legacy dcov.gamma.

### Candidate B — Device-resident block Krylov

Use the existing resident matrix, multi-RHS GEMM, and `Q -> A Q -> Q' A Q` projection substrate to implement a block/device-resident eigensolver with:

```text
no host-driven matvec loop
no per-Arnoldi-vector H2D/D2H
GPU orthogonalization
GPU convergence checks
compact host status only
```

### Candidate C — Exact CUDA dCov shadow only

The existing exact CUDA dCov may be evaluated as a competitor. It can enter compatible mode only if the full oracle gate proves:

```text
decision flips = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
SHD = 0
```

Any decision flip rejects it as the compatible authority, even if it is statistically attractive or much faster.

### Legacy semantic details to preserve

```text
distance index
numCol behavior
selected eigenvalue/eigenvector semantics
centering order
nV2
nV2 mean
nV2 variance
gamma alpha/beta
p-value tail
finite/error behavior
```

### Device-side p-values

The final route must compute the batched gamma tail on CUDA or through an equivalent CUDA numerical primitive. A temporary C++ scalar p-value step is acceptable only in an intermediate shadow checkpoint and must be separately timed and counted.

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/dcov_cuda_component_oracle_v1/
fastkpc/artifacts/full_cuda_ci/dcov_cuda_full_shadow_v1/
fastkpc/artifacts/full_cuda_ci/dcov_cuda_backend_v1/
```

### Required diagnostics

```text
unique residual components
component cache requests/hits/misses/evictions
component build distance_ms
component eig/lowrank_ms
pair statistic_ms
gamma pvalue_ms
matrix H2D bytes
residual D2H bytes
host synchronization count
CPU dCov component count
CPU dCov eigen/low-rank count
CPU dCov pair-statistic count
CPU gamma p-value count
CPU Spectra count
CUDA dCov component count
CUDA dCov pair count
CUDA gamma p-value count
CUDA eig convergence/failure count
```

### Primitive gates

```text
all existing legacy dCov oracle fixtures pass
real residual fixture grid passes
near-alpha dCov cases pass
all outputs finite
no unaccounted convergence failure
```

### Full graph gate

Use the accepted full-CUDA residual service plus the CUDA dCov candidate:

```text
edge_count = 110 / 110
SHD = 0
sepsets identical = TRUE
n.edgetests exact = TRUE
canonical deletion trace identical = TRUE
CPU dCov component count = 0
CPU dCov eigen/low-rank count = 0
CPU dCov pair-statistic count = 0
CPU gamma p-value count = 0
CPU Spectra count = 0
residual/component large-payload D2H = 0
unknown fallback count = 0
```

### Performance gate

The CUDA dCov route must complete the full run, beat the C++ Spectra dCov
portion on the same machine, and satisfy the dCov allocation in the accepted
`performance_budget_v1`. A faster microbenchmark that violates the Phase 3.5
full-workload bound or still causes a full-run timeout is a failed route.

### Exit condition

The Phase 3.5-selected numerical dCov architecture is fully implemented,
device-resident, complete on 351x48, and graph-identical.

### Accepted Phase 8 closure (2026-07-31)

The Phase 3.5-selected exact-CUDA decision screen with guarded CUDA legacy
full-eig refinement now runs against the accepted Phase 7 native setup and
device-resident residual service. Level zero uses a distance-invariant
intercept-shift adapter; all conditional groups use Phase 7 native setup DTOs
and accepted CUDA-selected smoothing parameters.

```text
logical tests                           = 240,489
direct / conditional tests              = 2,213 / 238,276
exact-screen residual components        = 110,665
guarded pairs                           = 1,188
full-eig refined components             = 1,559
exact-screen decision flips             = 92
final decision flips                    = 0
near-alpha rows / final flips            = 1,529 / 0
maximum refined p-value error            = 6.955545e-11
residual / component large-payload D2H  = 0 / 0 bytes
CPU dCov component / eig / pair / gamma = 0 / 0 / 0 / 0
CPU Spectra calls                        = 0
unknown / approximate fallback count    = 0 / 0
edge count                              = 110 / 110
SHD                                     = 0
adjacency / sepsets / n.edgetests       = exact / exact / exact
canonical deletion trace                = exact
```

The measured CUDA dCov host boundary is `17,498.876 ms`, within the accepted
`47,000 ms` Phase 8 allocation. The same canonical workload records
`4,026,920 ms` of summed Phase 7 CPU Spectra dCov work, for a measured ratio of
`0.00434547`. This is a Phase 8 dCov-component gate, not the Phase 10 complete
one-call campaign timing claim. The full R-orchestrated qualification run took
`509.690` seconds; Phase 9 owns removal of that per-group host orchestration.

Accepted artifacts:

```text
fastkpc/artifacts/full_cuda_ci/dcov_cuda_component_oracle_v1/
  manifest SHA-256 = 0608a6710bf2cd0bfa1f238093783242c172e6f12ee62356678276c13102de8d

fastkpc/artifacts/full_cuda_ci/dcov_cuda_full_shadow_v1/
  manifest SHA-256 = dc620c6025bc03107744e362ead3d4bbaea274e155797e5cc7320f252585d3b2

fastkpc/artifacts/full_cuda_ci/dcov_cuda_backend_v1/
  manifest SHA-256 = 9df0a8c0829f831ccb983f9ce95cd479bc9c55e80aaeea1d8875c5e6d7699d4d
```

Each artifact contains all 26 required files and validates 240,489 case rows,
1,529 near-alpha rows, 8,635 group diagnostic rows, exact graph payloads,
cache accounting, performance evidence, tracked contracts, producer identity,
attestation, receipt, source closure, and native binary identity. The accepted
source evidence SHA-256 is
`2ac5df9bb660cb8a50bcb49b61ff479c993c364099168393f10737031e9e9324`.
Phase 8 is complete; Phase 9 is next.

---

## Phase 9 — Fuse the compatible CI service into the native one-call skeleton

### Goal

Replace the hidden R residual provider and CPU dCov authority in the existing native one-call facade with the accepted CUDA CI service.

The integration must consume compatible Phase 3.5 semantic handles and the
accepted identity, numerical, cache, corpus, and performance contracts. Phase 9
is not the first validation of ownership, device residency, dCov architecture,
or memory feasibility.

### C++ control-plane responsibilities

```text
canonical level/round loop
current-edge state
conditioning-set enumeration
logical sequence ids
same-S request grouping
bounded submission batches
p-value replay
adjacency mutation
sepset write
logical n.edgetests
stop condition
```

### CUDA data-plane responsibilities

```text
native PreparedS setup handles
single- and multi-penalty target batches
device residual cache
device dCov component cache
pair p-value batches
compact status results
```

### Scheduling policy

Reuse the successful round-based principle:

```text
advance only currently reachable edge states
batch currently requested work
avoid unbounded eager level prefill
preserve canonical replay by logical sequence id
```

Historical level-prefetch and eager-prefill experiments reduced theoretical fit counts but regressed full wall time. Do not revive them without a new bounded design and full artifact.

### Cache policy

Start from the accepted `cache_memory_model_v1` capacity split, accounting,
headroom, and deterministic eviction policy. Phase 9 may tune that policy only
through a versioned replacement model and renewed affected gates.

Caches must be:

```text
capacity-bounded
memory-accounted
deterministic in semantics
safe under eviction
scoped by dataset/run
reported in the artifact
```

Eviction may change performance but must never change results.

### Final native diagnostics

```text
r_callback_count
legacy_mgcv_fit_count
legacy_mgcv_setup_count
cpu_residual_solve_count
cpu_dcov_component_count
cpu_dcov_eigen_or_lowrank_count
cpu_dcov_pair_stat_count
cpu_gamma_pvalue_count
cpu_spectra_count
cuda_single_penalty_target_count
cuda_multi_penalty_target_count
cuda_residual_batch_count
cuda_dcov_component_count
cuda_dcov_pair_count
cuda_gamma_pvalue_count
logical_tests_consumed
physical_tests_evaluated
speculative_tests_ignored
residual_h2d_bytes
residual_d2h_bytes
unknown_fallback_count
```

Required canonical values include:

```text
r_callback_count = 0
legacy_mgcv_fit_count = 0
legacy_mgcv_setup_count = 0
cpu_residual_solve_count = 0
cpu_dcov_component_count = 0
cpu_dcov_eigen_or_lowrank_count = 0
cpu_dcov_pair_stat_count = 0
cpu_gamma_pvalue_count = 0
cpu_spectra_count = 0
cuda_dcov_component_count > 0 and is fully cache-accounted
cuda_dcov_pair_count = physical_tests_evaluated
cuda_gamma_pvalue_count = physical_tests_evaluated
residual_d2h_bytes = 0 in the production path
unknown_fallback_count = 0
logical n.edgetests exact = TRUE
```

Sampled shadow downloads are allowed only when an explicit diagnostic flag is enabled and must not be part of measured production timing.

### Proposed route name

```text
precision = "compatible.cuda"
compatible_cuda_strict = TRUE
```

Keep this route env-gated or explicitly selected until Phase 10 promotion.

### Required artifact

```text
fastkpc/artifacts/full_cuda_ci/one_call_full_cuda_351x48_v1/
```

### Hard gate

```text
run_status = ok
candidate graph returned
edge_count = 110 / 110
adjacency_identical = TRUE
SHD = 0
sepsets_identical = TRUE
n.edgetests exact = TRUE
canonical deletion trace identical = TRUE
unknown fallback count = 0
approximate backend count = 0
```

### Exit condition

One native call runs the complete compatible skeleton with a CUDA numerical CI data plane and no hidden R/CPU numerical authority.

### Accepted Phase 9 closure (2026-07-31)

The explicit `compatible.cuda` / `route="full_cuda"` facade now makes one
native call. C++ owns the complete stable skeleton state machine and canonical
replay; the accepted Phase 7 residual and Phase 8 dCov services remain the only
numerical authorities. Physical CUDA rounds are recorded by task slot and then
replayed in exact canonical layer-plan order.

```text
native calls                            = 1
logical tests                           = 240,489
logical n.edgetests                     = 2213,52659,125293,40694,13293,5422,835,80
physical dCov/gamma evaluations         = 241,677
guarded full-eig refinements            = 1,188
logical trace / deletion trace          = exact / exact
edge count                              = 110 / 110
SHD                                     = 0
adjacency / sepsets / n.edgetests       = exact / exact / exact
final decision flips                    = 0
R callbacks / legacy mgcv fits/setups   = 0 / 0 / 0
CPU residual / dCov / gamma / Spectra   = 0 / 0 / 0 / 0
residual / component large-payload D2H  = 0 / 0 bytes
unknown / approximate fallback count    = 0 / 0
```

Accepted artifact:

```text
fastkpc/artifacts/full_cuda_ci/one_call_full_cuda_351x48_v1/
  manifest SHA-256 = f32a4ae5275da20912e709d766ad4e12a887446c34ccaf89f9d2358383159831
  source evidence SHA-256 = bb8d0e401dfa2ef59bbc4f73c656aa1201b5e7ad8ecb0d805d5cdd2b16becb47
  producer identity = d248433eeba4aa66bf18d709f5231b6638e9dfd1dcf91477f6dbed682206bbc3
```

The first complete measured call took `3,074.062` seconds, so it explicitly
fails the Phase 10 performance gate. Diagnostics identify `80,175` native setup
misses/rebuilds, `294,877` physical residual fits, `699.592` seconds of setup
time, and `2,257.702` seconds at the CUDA optimizer host boundary. Phase 9
accepts correctness and authority only; Phase 10 must remove this repeated work
and cannot promote the route at the recorded timing.

---

## Phase 10 — Full performance gate, hardening, and promotion

### Goal

Prove that the correct full-CUDA route is repeatable, faster, fail-closed, and supportable.

### Benchmark protocol

Use the accepted `reference_machine_v1` and `performance_budget_v2`. The v1
budget and `promotion_351x48_v1` remain immutable historical evidence; v1 warm
is classified as replay-warm and cannot satisfy a fresh-data promotion gate.
Measure fresh-process cold, fresh-data compute-warm, replay-warm, and the
correct baseline independently. The contracts define cache preconditions,
include complete numerical CI and skeleton work in both fresh-data boundaries,
exclude build time, and freeze GPU power/clock policy, CPU affinity, thread
counts, and cache state.

On the same machine and software environment:

```text
five fresh-process cold repetitions from uninitialized CUDA state
five fresh-data compute-warm repetitions after noncanonical CUDA prewarm
five replay-warm repetitions after a complete same-data call
five fresh correct-baseline repetitions
fresh process for every measured repetition
empty dataset-specific caches before each compute-warm measurement
zero preexisting entries for the measured DatasetKey
same data and config
same CPU/GPU affinity policy
same thread counts
same cache policy
record raw per-run timings
report cold and warm median, min, max, and MAD/IQR separately
```

Always run a fresh correct baseline in the same campaign. Do not compare only with an old artifact from another environment.

### Mandatory performance gate

Final promotion must satisfy all of these conditions simultaneously:

```text
fresh-data compute-warm median <= 120 seconds on reference_machine_v1
fresh-data compute-warm median <= 0.80 * same-run correct baseline median
fresh-process cold median <= same-run correct baseline median
every candidate run passes all correctness and authority gates
replay-warm is reported but is not a promotion gate
```

The fresh-data compute-warm stretch target is at most 60 seconds. The relative
gate remains mandatory even when the absolute gate is stricter, so a hardware
or baseline change cannot hide relative regression. Machine counters must prove
positive physical CI, setup, optimizer, residual, dCov component, pair, and
gamma work; a result-cache replay cannot pass this boundary. None of these
targets weakens correctness: a 30-second run with SHD > 0 is a failure.

### Required hardening

```text
CUDA OOM handling
unsupported semantic fail-closed behavior
non-finite input checks
rank-deficient cases
near-constant variables
near-alpha cases
cache-capacity sweeps
determinism across stream/thread counts
CUDA error injection
resource cleanup
repeated-run leak checks
version mismatch receipts
```

Capacity sweeps validate the accepted Phase 3.5 memory model under the complete
route; they are not the first memory-feasibility analysis.

### Held-out validation

Use `corpus_policy_v1`. Reusable development qualification and metamorphic
corpora must already have run throughout the affected phases. After source,
configuration, and contracts are frozen, open the sealed promotion holdout
under its recorded release protocol. It covers:

```text
different n and p
|S|=1,2,>2
collinearity
near constants
multiple penalty counts
near-alpha decisions
```

The canonical 351x48 gate remains mandatory. Holdout success cannot replace it.
If holdout results cause an implementation change, the opened corpus becomes
regression evidence and a new sealed holdout version is required for another
unbiased promotion claim.

### Promotion ladder

```text
shadow
  -> explicit experimental compatible.cuda
  -> recommended compatible.cuda
  -> possible default only after explicit approval
```

### Required artifacts

```text
fastkpc/artifacts/full_cuda_ci/promotion_351x48_v1/
fastkpc/artifacts/full_cuda_ci/promotion_351x48_v2/
fastkpc/artifacts/full_cuda_ci/development_500x50_v1/
fastkpc/artifacts/full_cuda_ci/sealed_promotion_holdout_v1/
fastkpc/artifacts/full_cuda_ci/failure_injection_v1/
```

### Documentation updates

Update:

```text
README.md
fastkpc/README.md
backend routing policy
precision capabilities
build instructions
validation commands
fallback policy
known supported semantic envelope
```

Do not describe `compatible.cuda` as a full mgcv clone. It is a compatible implementation of the exact regrXonS/KPC subset.

### Historical v1 canonical/replay evidence (2026-07-31)

The v1 correctness, authority, repeatability, hardening, artifact-integrity,
and replay-latency claims are accepted independently of the still-sealed
promotion holdout. Its performance claim is not sufficient under v2:

```text
artifact = fastkpc/artifacts/full_cuda_ci/promotion_351x48_v1/
producer identity SHA-256 =
  592dd982673c62d996ebff404dd68f9bdde71f83d5784e4235325cc1a2ffc556
freeze identity SHA-256 =
  1b1f19333ab2df038d6177bbbb40f9213afe8ed94bd0411fcc088638ec232a75
source closure SHA-256 =
  9f5a85968bbb61390288a62775c3b0ea930cb8c679b36dda86f3661abc5c1e36
native binary SHA-256 =
  c24881ffd1acedd095b83c54271bcc9c11c0f7bc0843e7f9900cb38f6c32623d
manifest SHA-256 =
  4ec1858dee56bdfc8af6073548adf86a08dfd7c2b479ac4d5db51fcad711b6b4
summary SHA-256 =
  313bfbd27d3dd48beb3be700c65d490feae50adfc52ab6a8d59c97452337045c

fresh-process cold repetitions / median = 5 / 1290.664 sec
replay-warm repetitions / median = 5 / 0.699 sec
fresh correct baselines / median = 5 / 601.431 sec
replay-warm / correct-baseline median ratio = 0.001162228
v2 fresh-data compute-warm gate = NOT_MEASURED
v2 fresh-process cold ratio gate = FALSE

edge count = 110 / 110
SHD = 0 in all 15 measured runs
adjacency / sepsets / n.edgetests exact = TRUE / TRUE / TRUE
deletion / logical trace exact = TRUE / TRUE
repeatability gate = TRUE
every-run correctness gate = TRUE
every-candidate authority gate = TRUE

legacy mgcv target fits / setup calls in CI loop = 0 / 0
R callbacks / CPU residual solves = 0 / 0
CPU dCov component / eig / pair / gamma / Spectra calls = 0 / 0 / 0 / 0 / 0
residual / component D2H bytes = 0 / 0
unknown / approximate fallback count = 0 / 0

hardening artifact = fastkpc/artifacts/full_cuda_ci/failure_injection_v1/
hardening gate = TRUE
campaign artifact identity and tamper gate = TRUE
holdout state / gate = SEALED_NOT_RELEASED / FALSE
phase10 v1 canonical/replay claim = TRUE
phase10 promotion claim = FALSE
recommended route = FALSE
```

The five replay-warm measurements each follow one unmeasured complete same-data
call in a fresh process. They reuse authenticated, capacity-bounded
compact-result and target-state caches and perform zero physical CI tests or
residual fits. Cold timing includes input validation, native setup, CUDA work,
cache construction, and packaging. Under v2, the old `0.699`-second value is
report-only and the old `1290.664`-second cold median fails the new ratio gate.
No sealed holdout release is allowed until the v2 campaign and public 500x50
fixture pass and a new candidate is frozen.

### Final exit condition

The final success definition in Section 13 is satisfied.

---

## 8. Standard artifact schema

Every phase artifact that makes a correctness or performance claim must contain:

```text
manifest.json
summary.json
summary.md
commands.txt
environment.txt
graph_agreement.csv
sepset_agreement.csv
n_edgetests.csv
first_divergence.json
fallbacks.csv
stage_timing.csv
raw_runs.csv
```

Every artifact after Phase 3.5 must consume
`artifact_identity_contract_v1` and keep these namespaces separate:

```text
producer semantic manifest        # immutable producer authority
validator attestations            # append-only validation acts
volatile execution receipts       # audit-only session/process evidence
```

The producer manifest records every applicable tracked contract name, semantic
version, canonical SHA-256, and exact snapshot. Validators recompute contract
and payload hashes from disk. A validator upgrade may add a new attestation but
must not rewrite the producer manifest. Volatile path, inode, PID, session,
context, staging, and timestamp values never define semantic equality.

Phase-specific numerical artifacts should additionally include:

```text
case_results.csv
near_alpha_results.csv
rank_condition_results.csv
cache.csv
```

### Required summary fields

```text
run_status
timeout
source_commit
oracle_artifact
candidate_route
edge_count_reference
edge_count_candidate
SHD
adjacency_identical
sepsets_identical
n_edgetests_identical
deletions_identical
unknown_fallback_count
approximate_backend_count
architecture_contract_sha256
numerical_contract_sha256
artifact_identity_contract_sha256
reference_machine_contract_sha256
performance_budget_contract_sha256
elapsed_sec
pass
```

`pass` must be false when the graph is missing or any hard correctness field is unknown.

---

## 9. First-divergence protocol

Any graph mismatch must immediately produce a minimal reproducer.

### Required first-divergence payload

```text
logical sequence id
level
edge x-y
conditioning set S
reference residual hashes
candidate residual hashes
reference selected sp
candidate selected sp
reference score/EDF
candidate score/EDF
reference dCov statistic/moments/p-value
candidate dCov statistic/moments/p-value
alpha
reference decision
candidate decision
setup fingerprint
solver path
rank/condition metadata
fallback/status flags
```

### Required response to a mismatch

```text
1. Stop promotion work.
2. Do not increase tolerance just to pass.
3. Reproduce the single target/S or CI case outside the full skeleton.
4. Identify whether the first divergence is setup, sp selection, solve,
   residual, dCov component, moment, p-value, or replay.
5. Add the case permanently to the oracle suite.
6. Fix and rerun local, phase, and full graph gates.
```

Do not continue optimizing later phases while the earliest unexplained decision flip remains.

---

## 10. Codex execution rules

### 10.1 Work on one phase at a time

Every Codex task must state:

```text
active phase
single question being answered
files expected to change
artifact to produce
commands to run
exit gate
```

Do not combine:

```text
new diagnostics + default promotion
new optimizer + scheduler rewrite
new setup semantics + dCov rewrite
shadow implementation + removal of oracle fallback
correctness change + unrelated cleanup
```

### 10.2 Test-driven phase workflow

For each implementation slice:

```text
RED:
  add a focused failing test or artifact assertion

GREEN:
  implement the smallest complete slice

REGRESSION:
  run all relevant prior phase gates

ARTIFACT:
  run the required real fixture or full gate

DECISION:
  promote, keep shadow-only, or reject based on the artifact
```

### 10.3 Commit rule

Commit only when:

```text
git diff --check passes
all focused tests pass
all relevant previous phase tests pass
required artifact exists
hard correctness fields pass for the claimed scope
route remains explicit/env-gated unless promotion is authorized
summary text matches artifact evidence
```

Suggested commit prefixes:

```text
test:
diag:
feat:
perf:
fix:
docs:
```

### 10.4 Do not commit a performance claim when

```text
SHD != 0
candidate graph is missing
sepsets differ
n.edgetests differ
first divergence is unexplained
full wall time regresses despite lower kernel time
fallback is hidden
true_batched_kernel is falsely reported
artifact uses a different oracle/config
```

Negative experiments may be preserved in a separate experiment note or artifact, but they must not be presented as phase completion.

### 10.5 Never modify the oracle to fit the candidate

The following are forbidden without explicit human approval:

```text
changing alpha
changing data columns or order
changing numCol/index
changing legacy formula route
changing reference sepsets
changing reference n.edgetests
changing the SHD comparator
removing near-alpha cases
loosening a hard graph gate
```

### 10.6 Keep the goal document concise

Do not append thousands of lines of raw experiment logs to this file. Update only:

```text
phase status
accepted artifact path
one-paragraph decision
next phase
```

Store detailed run evidence under `fastkpc/artifacts/full_cuda_ci/` or a separate experiment log.

### 10.7 Fail closed

When a semantic case is unsupported:

```text
strict mode: error with a structured reason
shadow mode: use oracle only and record the key
```

Never route an unsupported compatible case to `fastSplineCUDA` or another approximation.

---

## 11. Validation commands

Existing relevant checks include:

```bash
git diff --check

Rscript fastkpc/tests/test_mgcv_residual_oracle_trace.R
Rscript fastkpc/tests/test_mgcv_residual_replay_spec.R
Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R
Rscript fastkpc/tests/test_mgcv_extract_same_setup_fixed_sp_batch_cpp.R
Rscript fastkpc/tests/test_mgcv_extract_gpu_handle_batch_solve.R
Rscript fastkpc/tests/test_legacy_mgcv_same_s_fixed_sp_batch_provider.R

Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_oracle.R
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_batch_oracle.R
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_spectra_oracle.R
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_shadow_route.R

Rscript fastkpc/tests/test_skeleton_native_layer_plan.R
Rscript fastkpc/tests/test_skeleton_native_residual_provider_legacy_dcov.R
Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R
Rscript fastkpc/tests/test_compatible_cuda_skeleton_artifact.R
```

When CUDA code changes:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh

FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_legacy_dcov_spectra_matvec_cuda.R

FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_legacy_dcov_cuda_lowrank_gamma_parity.R

FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_legacy_dcov_cuda_lowrank_backend_route.R

FASTKPC_RUN_CUDA_TESTS=1 \
  Rscript fastkpc/tests/test_compatible_cuda_skeleton_native_cuda_lowrank.R
```

New campaign commands to create phase by phase:

```bash
Rscript fastkpc/tests/test_full_cuda_ci_oracle_gate.R
Rscript fastkpc/tests/test_full_cuda_ci_prepared_s_contract.R
Rscript fastkpc/tests/test_full_cuda_ci_fixed_sp_cuda.R
Rscript fastkpc/tests/test_full_cuda_ci_single_penalty_gcv.R
Rscript fastkpc/tests/test_full_cuda_ci_multi_penalty_cpp.R
Rscript fastkpc/tests/test_full_cuda_ci_multi_penalty_cuda.R
Rscript fastkpc/tests/test_full_cuda_ci_native_setup.R
Rscript fastkpc/tests/test_full_cuda_ci_dcov_cuda.R

FASTKPC_RUN_CUDA_TESTS=1 \
  bash fastkpc/tools/run_full_cuda_ci_gate.sh
```

The exact script names may be adjusted to repository conventions, but one standard full-gate command must exist before promotion.

---

## 12. Immediate next actions

Codex should begin with the earliest phase whose exit gate is incomplete. Phase
0 through Phase 9 are complete, so the current starting point is Phase 10
performance, hardening, repeatability, and promotion.

### Task 1

Complete: the standardized immutable oracle artifact and comparator reuse the
current 351x48 legacy reference and qualified correct one-call trace source.

### Task 2

Complete: normalized sepset equality, canonical deletion-trace equality, and
first-divergence reporting are hard-gate fields.

### Task 3

Complete: Phase 1 produced the full 110,617-key mgcv workload/risk census with
8,634 same-S setup rows, zero fit errors, exact legacy-layout parity, complete
field coverage, a pinned logical-census hash, authenticated metadata/shard
payloads, independently recomputed risk semantics, deterministic 64-shard
closure, and inherited SHD 0 evidence.

### Task 4

Complete: Phase 2 produced 8,634 authenticated `PreparedSSetup` objects and
110,617 `TargetState` rows, with exact qualification fixed-sp hashes, zero dCov
decision flips, zero unsupported/fallback counts, deterministic 64-shard
restart closure, and inherited Phase 0 graph evidence.

### Task 5

Complete: Phase 3A established the verified persistent fixed-sp CUDA resource
and safe single-target Cholesky milestone recorded above. It is incorporated
into the completed Phase 3 runtime.

### Task 6

Complete: Phase 3B established the verified true same-S multi-target Cholesky
milestone recorded above. It is incorporated into the completed Phase 3
runtime.

### Task 7

Complete: Phase 3C established the verified augmented QR/SVD stable routes and
the 6,143-target qualification milestone recorded above, with zero declared
reroutes and zero dCov decision flips.

### Task 8

Complete: the full Phase 3 closure produced and independently validated all
64 shards, 8,634 setups, 110,617 target solves, and 240,489 logical CI rows.
It reproduced 110 edges with zero decision flips, exact sepsets and
`n.edgetests`, and SHD 0.

### Task 9

Complete: Phase 3.5 froze the tracked campaign contracts, opaque semantic ABI,
and three-layer identity model; selected the guarded exact-screen/full-eig CUDA
architecture as GO for Phase 8 qualification under two real measured scales;
and accepted the cache/memory and conservative 98.529-second full-campaign
feasibility model with zero final decision drift.

### Task 10

Complete: Phase 4 produced full-corpus CUDA objective/optimizer, full-shadow,
and five-repetition backend artifacts for all 1,174 single-penalty setups and
44,941 targets. It retired `r-cpu-spectral` authority with zero CPU target
scoring/optimization/fallback, zero downstream decision flips, exact mixed
graph semantics, SHD 0, and a `0.5650` same-corpus performance ratio.

### Task 11

Complete: Phase 5 reproduced the `|S| > 2` additive multi-penalty mgcv
objective, optimizer trajectory, selected smoothing parameters, and residuals
for the full canonical corpus. All 16 partitions and three authenticated
artifacts pass with zero fallback, zero decision flips, exact mixed graph
semantics, and SHD 0.

### Task 12

Complete: Phase 6 produced all 16 authenticated partitions and three required
artifacts for 7,460 setups, 65,676 targets, and 60,324 logical rows. Persistent
CUDA optimization and residual solves have zero trajectory mismatches, zero
fallback, exact graph semantics, SHD 0, and a `0.769315` same-trace residual
performance ratio against the correct 20-core legacy-mgcv provider.

### Task 13

Complete: Phase 7 produced all 64 setup shards, 16 Phase 4 partitions, 16
Phase 6 partitions, and three schema-complete authenticated artifacts. The
native C++ setup builder covers all 8,634 setups and 110,617 targets with exact
setup geometry, zero unsupported cases, zero runtime R/mgcv setup or fit
authority, exact graph semantics, and SHD 0.

### Task 14

Complete: Phase 8 qualified the Phase 3.5-selected guarded exact-CUDA screen
plus CUDA legacy full-eig refinement against the Phase 7 native residual
service. All 240,489 logical rows pass with 92 screen flips guarded, zero final
flips, zero CPU numerical dCov/gamma authority, zero residual/component
large-payload D2H, exact graph semantics, SHD 0, and three authenticated
schema-complete artifacts. The measured dCov boundary is 17.499 seconds and
passes the accepted 47-second allocation.

### Task 15

Complete: Phase 9 fused Phase 7 native setup, live CUDA target optimization and
device residual handles, and Phase 8 guarded CUDA dCov into one native skeleton
call. The authenticated canonical artifact has exact logical order, graph,
sepsets, counts, and deletion trace, zero final decision flips, and zero hidden
R/CPU numerical authority. Its `3,074.062`-second timing is retained as a failed
Phase 10 performance baseline, not as a promotion claim.

### Task 16

Active: bounded compact-result/target-state caching and the cache-aware frontier
scheduler make same-data replay fast, but the historical `0.699`-second value
is replay-warm rather than fresh-data compute. The v1 cold path takes
`1290.664` seconds versus a `601.431`-second correct baseline. The v2 cache-state
proof and public 500x50 fixture pass. Lazy multi-penalty handles and per-call
univariate primitive reuse reduce the current single-run fresh-data development
profile to `263.305` seconds with exact numerical and graph parity, but
Checkpoint B and the final five-run gate still fail. A non-formal
max-conditioning-size-3 profile attributes 63.6% of optimizer stage-0 cycles to
QR factorization and 25.7% to explicit Q generation; prepared-handle builds are
only 5.812 seconds of the 40.330-second multi-penalty boundary. An opt-in exact-
bit full level-7 trace finds only 73,274 reusable requests among 1,691,603
decompositions (`4.33%`): 73,272 occur during initial evaluation and only two
occur later. Reuse declines from 4.93% at three penalties to 1.55% at seven,
with group-size p50/p95 both equal to one. The traced result remains bitwise
identical and has zero overflow or route mismatch, but its instrumented
421.065-second wall time is not performance evidence. Treat initial
decomposition sharing as secondary and continue with a trajectory-preserving
grouped/persistent QR execution shape. A development-only one-warp-per-matrix
prototype now proves exact R/Q/condition, route, score, derivative, Hessian,
coefficient, and nonfinite/status parity across real q=28/37/46/55/64 shapes.
Its 512-target 2/4/8-warp campaign reaches at best `1.032x` QR/Q throughput;
2/4-warp configurations are effectively flat and 8-warp configurations
regress. This is below the `1.3x` stop threshold, so do not convert the
optimizer into a grouped state machine from this prototype. Require evidence
for a materially different decomposition kernel/work organization before the
next optimizer integration attempt. Compute-profile v5 closes the existing
optimizer work accounting: all `132,908` physical target optimizations are
issued by whole-level prefill, `110,617` distinct TargetKeys are eventually
consumed, and `22,291` are not consumed. Frontier-required optimization,
prefill singleton skips, and singleton padding are all zero. The diagnostic
run takes `269.426` seconds, records `162.321` seconds of prefill optimizer host
time and `178.574` seconds of end-to-end prefill batch wall time, and remains
bitwise identical to the `263.305`-second result with zero CPU authority,
fallback, or large-payload D2H. Do not invent per-target time attribution for
mixed windows. A zero-lookahead canonical-frontier prototype was rejected on
the level-3 development workload: wall time regressed from `179.914` to
`394.993` seconds, optimizer boundaries increased from 83 to 1,898, setup
submissions increased from 5,239 to 46,691, and 14,507 singleton padding
targets replaced most of the removed speculative work. It also changed 120,991
consumed p-values in low bits (maximum absolute difference `3.33e-14`) despite
zero decision flips and exact structural graph semantics. The v5 whole-level
production scheduler is restored, and the failed prototype was not run through
level 7. Do not retry synchronous zero-lookahead or direct depth-one/two
lookahead. A subsequent no-CUDA-CI opportunity replay rebuilt all 137 original
v5 windows and 8,637 setup cohorts exactly. All 137 windows are eventually
demanded; only three cohorts and nine targets are never demanded. The remaining
22,282 unconsumed targets are inside demanded cohorts. Lazy original-window
activation therefore skips zero recorded batch wall time and, through level 3,
retains exactly the v5 shape of 83 boundaries, 5,239 setup submissions, and
107,053 target optimizations. The scheduler opportunity decision is
`STOP_SCHEDULER_OPPORTUNITY_TOO_SMALL`. Stop this optimization family for the
current contract epoch. A subsequent device-only optimizer accepted-residual
qualification also stops before integration: on a real q=64, seven-penalty,
two-target fixture all 702 residual values differ from the selected-SP fixed
solve (maximum absolute difference `0.2399546`, relative L2 `0.0288145`). The
exact screen and legacy full-eig consumers both produce non-bitwise-identical
p-values with zero residual payload D2H. Zero decision flips do not waive this
contract failure. The decision is `STOP_OPTIMIZER_RESIDUAL_NUMERICAL_PARITY`;
do not implement detached residual arenas or production optimizer-residual
reuse in this epoch. The subsequent trace-free full level-7 compute-profile v6
is complete: fixed-SP residual time is `15.315` seconds, of which guarded
refinement accounts for only `0.482` seconds, below the implementation gate. Do
not build the exact/refinement sharing route. Cross-batch identity qualification
also rejects target-granular caching: batched-Cholesky subset execution changes
913 residual values, 332,530 component values/moments, three statuses, and two
unconditional legacy-eig p-values. Only repeated complete cohorts qualify. The
full level-7 opportunity receipt finds just 46 such batches and 1,853 targets,
with `71.354` ms of residual solve plus exact-component time. The final decision
is `STOP_CROSS_BATCH_FIXED_RESIDUAL_CACHE_OPPORTUNITY`; stop the residual and
component-cache family without a capacity trace or production prototype. A
persistent dCov context still has an orchestration upper bound of
`3.440` seconds teardown plus `7.416` seconds otherwise unattributed dCov host
time. Then continue with a pure-C++ setup/optimizer pipeline and, if needed,
single-matrix QR/Q work. Pass the v2 checkpoints and refreeze before opening
the external sealed holdout. Promotion remains forbidden.

---

## 13. Final success definition

The campaign is complete only when all of the following are true.

### Product shape

```text
R prepares data and calls one native skeleton function.
C++ owns canonical graph control.
CUDA owns the repeated numerical CI data plane.
R receives skeleton and sepsets and continues orientation.
```

### Contract authority

```text
compatible architecture and ABI contract accepted
numerical contract accepted and all artifact snapshots authenticated
producer/validator/receipt identity contract enforced
reference machine and performance budget contracts accepted
development, metamorphic, and sealed holdout corpus policy enforced
```

### Canonical correctness

```text
full 351x48 run_status = ok
edge_count = 110 / 110
adjacency_identical = TRUE
SHD = 0
normalized sepsets identical = TRUE
n.edgetests exact = TRUE
n.edgetests = 2213,52659,125293,40694,13293,5422,835,80
canonical deletion trace identical = TRUE
```

### CUDA authority

```text
legacy mgcv target fits in CI loop = 0
legacy mgcv setup calls in CI loop = 0
R callbacks in native skeleton loop = 0
CPU residual numerical solves = 0
CPU dCov component builds = 0
CPU dCov eigen/low-rank calls = 0
CPU dCov pair-statistic calls = 0
CPU gamma p-value calls = 0
CPU Spectra calls = 0
approximate residual backend calls = 0
unknown fallback count = 0
residual D2H materialization in production path = 0
```

### Repeatability

```text
five measured repetitions
SHD = 0 every time
same adjacency every time
same sepsets every time
same logical n.edgetests every time
no intermittent fallback or non-finite result
```

### Performance

```text
fresh-data compute-warm median <= 120 seconds on reference_machine_v1
fresh-data compute-warm median <= 0.80 * same-run correct baseline median
fresh-process cold median <= same-run correct baseline median
every measured candidate run passes all correctness/authority gates
fresh-data compute-warm stretch target <= 60 seconds
fresh-process cold, fresh-data compute-warm, replay-warm, and baseline raw
timings are independently reported
replay-warm is report-only and cannot satisfy a promotion performance gate
```

### Failure behavior

```text
unsupported external semantics fail closed or use an explicitly requested,
fully recorded legacy-compatible fallback mode
no unsupported case silently changes to an approximate CI definition
```

### Final rule

```text
A route with SHD > 0 is not a near-success.
It is a correctness failure and cannot be promoted.
```
