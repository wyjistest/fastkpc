# Full-CUDA Legacy-Compatible CI Goal Document

> **Campaign:** replace the complete numerical conditional-independence data plane used by the KPC skeleton with a legacy-compatible CUDA implementation.
>
> **Document origin baseline:** `main` at `7b36668` (`feat: add CUDA Spectra handle projection primitive`).
>
> **Current accepted implementation/evidence snapshot:** Phase 3.5 is COMPLETE. Its accepted feasibility manifest is `5618db7966d0be4c3762a17a937a370842f61ba76294986ead967fed1c98be9f`, built on the Phase 3.5D native baseline at `ded3333`.
>
> **Active roadmap phase:** Phase 4, full-CUDA single-penalty GCV for `|S| <= 2`.
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
| 4 | Full-CUDA single-penalty GCV for `|S|<=2` | ACTIVE - CPU spectral selection remains to be removed |
| 5 | C++ multi-penalty GAM semantic replica | NOT COMPLETE |
| 6 | CUDA multi-penalty same-S target batches | NOT STARTED |
| 7 | Native setup builder; remove R/mgcv from CI loop | NOT STARTED |
| 8 | Legacy-compatible device-resident CUDA dCov | PARTIAL — primitives/smoke gates exist |
| 9 | Fused one-call compatible CUDA skeleton | PARTIAL — facade exists, CI service does not |
| 10 | Full gate, hardening, and promotion | NOT STARTED |

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
Candidate A full eig globally       REJECT; 832,508.19 ms diagnostic bound
Candidate A two-sided partial eig   REJECT; 1,633,730.54 ms diagnostic bound
Candidate B block Krylov core       REJECT/NOT-GO; 34,439.81 ms core-only bound
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
  bca5ca4efde7f5377b2da764e24a74d0611cf4c106ccc1555d6a616a46bd42e5
Phase 3.5 feasibility artifact manifest SHA-256
  5618db7966d0be4c3762a17a937a370842f61ba76294986ead967fed1c98be9f
feasibility producer semantic identity
  e056099c6fb456349f787a0025891fc86fd0eed6db7962b24e13e8d2407f82e3
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

---

## Phase 6 — Port multi-penalty same-S target batches to CUDA

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

---

## Phase 10 — Full performance gate, hardening, and promotion

### Goal

Prove that the correct full-CUDA route is repeatable, faster, fail-closed, and supportable.

### Benchmark protocol

Use the accepted `reference_machine_v1` and `performance_budget_v1`. Measure
cold and warm product boundaries independently. The contracts define whether
input copies, native setup, and cache construction are included, exclude build
time, and freeze GPU power/clock policy, CPU affinity, thread counts, and cache
state.

On the same machine and software environment:

```text
five measured cold repetitions from the declared cold state
one prescribed warm-up before each warm measurement
five measured warm repetitions
fresh process for each repetition with the declared in-process warm-up
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
warm candidate median <= 120 seconds on reference_machine_v1
warm candidate median <= 0.80 * same-run correct baseline median
every cold and warm measured run passes all correctness and authority gates
```

The warm stretch target is at most 60 seconds. The relative gate remains
mandatory even when the absolute gate is stricter, so a hardware or baseline
change cannot hide relative regression. The Phase 3.5 feasibility bound is not
the Phase 10 actual campaign measurement. None of these targets weakens
correctness: a 30-second run with SHD > 0 is a failure.

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
0 through Phase 3.5 are complete, so the current starting point is Phase 4
single-penalty GCV.

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

Active: implement Phase 4 full-CUDA single-penalty GCV for `|S| <= 2`, starting
with objective parity and deterministic optimizer semantics. Retire
`r-cpu-spectral` selection from the accepted route while preserving the Phase
3.5 numerical, identity, machine, budget, and corpus contracts.

The current CUDA Spectra projection primitive remains useful substrate for
diagnostics, but the immediate critical path is accepted Phase 4 GCV objective
and optimizer parity, not another Phase 3.5 dCov architecture bake-off.

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
warm median <= 120 seconds on reference_machine_v1
warm median <= 0.80 * same-run correct baseline median
every cold and warm measured run passes all correctness/authority gates
warm stretch target <= 60 seconds
cold and warm boundaries and raw timings are independently reported
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
