# compatible.cuda Goal Document

## 0. One-line goal

Build a **legacy-compatible GPU-accelerated KPC skeleton engine**:

> R prepares the data, calls one C++/CUDA skeleton engine, the engine runs the full skeleton with legacy-compatible CI semantics, returns skeleton/sepsets to R, and R continues orientation. The hard correctness gate is **0 SHD versus the legacy CPU skeleton**.

This is not an extension of the current `precision="fast"` CUDA path. The current fast CUDA path is a high-performance approximate backend. The new goal is a **compatible CUDA path**.

---

## 1. Correctness policy

### 1.1 Hard rule

```text
skeleton mismatch = correctness failure
```

For the canonical 351x48 benchmark, every promoted compatible acceleration route must satisfy:

```text
edge_count_actual = 110
edge_count_ref    = 110
SHD               = 0
n.edgetests       = 2213,52659,125293,40694,13293,5422,835,80
n.edgetests exact = TRUE
```

The primary acceptance criterion is not average p-value error, not pMax, and not residual correlation. The primary acceptance criterion is **graph decision equivalence**:

```text
same skeleton
same deletion decisions where applicable
same n.edgetests
same sepset behavior where recorded
```

### 1.2 CPU legacy is oracle, not the target implementation

The CPU legacy path defines correctness:

```text
legacy residual: regrXonS / mgcv
legacy CI:       dcov.gamma
legacy search:   stable/canonical skeleton decisions
```

But the implementation goal is:

```text
CPU legacy oracle -> C++/CUDA compatible implementation
```

The CPU legacy path should be used for oracle, shadow, and fallback during development. It is not the desired production execution path.

### 1.3 Any new route must be env-gated first

No new backend becomes default directly. All new routes start as env-gated:

```text
shadow     -> env-gated backend -> recommended env route -> possible default later
```

A route can only be promoted if it passes full 351x48 correctness and wall-time gates.

---

## 2. Current known state

### 2.1 Fast CUDA approximate campaign is closed

The previous low-risk fast CUDA campaign optimized:

- fastSpline CUDA residual path
- CUDA exact dCov path
- dCov pvalue-only route
- CUDA workspaces, staging, grid cache
- dCov abs-fast no-pow rowsum path
- 64-thread rowsum block tuning

That route is fast, but not legacy-compatible:

```text
fast CUDA residual: fastSpline
fast CUDA dCov:     exact CUDA dCov
legacy CPU:         mgcv/regrXonS + legacy dcov.gamma
```

Therefore current `precision="fast"` CUDA must not be used as the correctness path if SHD=0 is required.

### 2.2 Current recommended compatible acceleration route

As of this document, the recommended SHD=0 compatible acceleration route is:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
```

Known full 351x48 milestones:

```text
original compatible legacy:           ~42.5 min, SHD=0
C++ Spectra legacy dCov backend:      ~19.8 min, SHD=0
+ target|S residual cache:            ~16.3 min, SHD=0
+ S-affinity residual scheduling:      ~14.8 min, SHD=0
```

### 2.3 Experimental or non-recommended routes

`FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=target_s`:

```text
correctness: passed
worker-sum: improved
wall time: regressed on full 351x48
status: experimental only; not recommended
```

Hybrid target-S scheduling:

```text
correctness: passed in experiment
worker-sum: improved
wall time: still worse than S-affinity
status: not committed as perf; keep as negative artifact/patch only
```

`FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH=level`:

```text
correctness: passed full 351x48
worker-sum / fit-count: improved
wall time: regressed badly on full 351x48
status: experimental only; not recommended
```

`FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=chunk` with
`FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE=cpp`:

```text
correctness: passed full 351x48
fixed-sp batch replay: exercised successfully inside same-S provider
wall time: only slightly better than same-S chunk baseline, still worse than
           recommended S-affinity route
status: experimental only; not recommended
```

`FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=chunk`:

```text
correctness: passed full 351x48
native batch backend: exercised successfully with zero errors/fallbacks
batch shape: fragmented into 54,233 small worker-chunk batches
wall time: regressed versus recommended S-affinity route
status: experimental only; not recommended
```

`FASTKPC_FASTSPLINE_EDF_TRACE_MODE=cholesky_cuda`:

```text
status: experimental fast CUDA/fastSpline path only
not relevant to legacy-compatible correctness route
```

---

## 3. Non-goals

Do not pursue these as the compatible.cuda mainline:

1. **Do not tune fastSpline to approximate mgcv.**
   It is a different CI definition. If skeleton differs, it is unusable for this goal.

2. **Do not replace mgcv with simplified B-spline/P-spline GAM as the compatible route.**
   It may be statistically reasonable and GPU-friendly, but if it is not legacy mgcv/regrXonS equivalent, it cannot guarantee 0 SHD.

3. **Do not replace legacy dcov.gamma with exact dCov by default unless full skeleton shadow proves 0 decision flips.**
   Exact dCov may be faster and more statistically exact, but it changes legacy behavior.

4. **Do not call R/mgcv directly inside OpenMP worker threads.**
   Treat R/mgcv as an oracle/setup provider, not as a thread-safe C++ worker function.

5. **Do not attempt a full mgcv clone as the first target.**
   Only reproduce the actual regrXonS/kpc subset needed by the target workloads.

6. **Do not require all graph bookkeeping to be CUDA device-side in the first version.**
   C++ host canonical replay plus CUDA CI data plane is the recommended architecture.

---

## 4. Target architecture

### 4.1 Final product shape

```text
R
  prepares data / parameters / labels
  calls .Call("fastkpc_compatible_cuda_skeleton_run", ...)

C++ compatible skeleton engine
  owns adjacency / sepsets / level loop
  enumerates CI tests in canonical order
  batches tests by level / S / shape
  replays p-values in legacy order

CUDA / C++ CI data plane
  mgcv-compatible residual executor
  legacy-compatible dCov.gamma executor
  returns p-value vectors

R
  receives skeleton + sepsets
  continues orientation
```

### 4.2 What must be on GPU

The GPU should accelerate the heavy data plane:

```text
batched residualization
batched linear algebra / solves
batched legacy-compatible dCov.gamma
large CI batches by skeleton level
```

### 4.3 What may remain on C++ host

The following can remain on C++ host without compromising the goal:

```text
skeleton level loop
edge and conditioning-set enumeration
canonical p-value replay
adjacency mutation
sepset storage
fallback/shadow decisions
```

From the R user's perspective this is still a one-call CUDA skeleton engine.

---

## 5. Phase plan

## Phase 0 — Lock the oracle and recommended compatible route

### Goal

Keep a stable, documented, SHD=0 baseline for every later phase.

### Required artifacts

```text
legacy_cpu_oracle_351x48_v1
legacy_compatible_recommended_route_v1
```

### Required checks

```text
edge_count = 110 / 110
SHD = 0
n.edgetests exact = TRUE
n.edgetests = 2213,52659,125293,40694,13293,5422,835,80
```

### Recommended env for current compatible acceleration route

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
```

### Promotion rule

Do not promote another route over this one unless it beats wall time and passes all correctness gates.

---

## Phase 1 — Level-prefetch gate closed as negative

### Background

A diagnostic estimate showed:

```text
current S-affinity mgcv fits:      273,284
level-prefetch unique target|S:    110,617
additional fit reduction:          162,667
raw residual payload total:        ~296 MiB
max level payload:                 ~114 MiB
```

An env-gated prototype exists:

```bash
FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH=level
```

### Goal

Determine if level-scoped parent residual prefetch converts theoretical fit reduction into wall-time improvement.

### Required full artifact

```text
fastkpc/artifacts/legacy_mgcv_residual_level_prefetch_v1
```

### Required env

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH=level
```

### Hard correctness gate

```text
edge_count = 110 / 110
SHD = 0
n.edgetests exact = TRUE
prefetch_error_count = 0
```

### Performance gate

Compare against current recommended S-affinity route:

```text
elapsed must be lower than the S-affinity baseline
mgcv_fit_count should be much lower than 273,284
ideally close to 110,617
```

### Gate result

Full artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_level_prefetch_v1
```

Result:

```text
elapsed_sec:        3280.205
edge_count:         110 / 110
SHD:                0
n.edgetests exact:  TRUE
prefetch errors:    0
dCov cpp errors:    0
Spectra failures:   0

mgcv fits:           132,908
consumed keys:       110,617
unused keys:          22,291
payload:             355.9 MiB
max level payload:   147.0 MiB
```

The route reduced mgcv fit count, but wall time regressed from the S-affinity baseline:

```text
S-affinity baseline:  ~882-890 sec
level-prefetch:       3280.205 sec
```

Additional finding:

```text
same dCov call count: 239,404
C++ dCov backend worker-ms inflated to ~51.8M
```

This means the current dense parent residual matrix / CI consumption path disrupts the dCov hot path and cannot be promoted as-is.

### Phase 1 decision

- Keep recommended route as S-affinity.
- Stop R-level worker scheduling/cache experiments for now.
- Move to Phase 2 and Phase 3.

---

## Phase 2 — Legacy-compatible dCov GPU/batch backend

### Goal

Turn the already-correct C++ Spectra dCov backend into a batched/GPU-compatible data-plane backend.

### Current state

Completed correctness ladder:

```text
R legacy dcov.gamma timed oracle
fixed residual oracle fixture
C++ scalar oracle parity
C++ batch oracle parity
compatible-route C++ shadow parity
full 351x48 Spectra shadow parity
C++ Spectra production backend
```

Production env:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
```

Known win:

```text
original compatible legacy: ~42.5 min
C++ Spectra dCov backend:  ~19.8 min
SHD = 0
```

### Phase 2A — Real batched C++ dCov workspace

#### Goal

Reduce per-call C++/R overhead and prepare data layout for CUDA.

#### Tasks

- Group dCov calls by `n`, `numCol`, and compatible path shape.
- Use one C++ call per batch, not one wrapper call per CI test.
- Reuse C++ workspace for distance, lowrank, centering, statistic, moments.
- Preserve output order and canonical replay.

#### Diagnostic checkpoint

Initial diagnostic counters are wired into the legacy-compatible C++ dCov
backend summary:

```text
legacy_dcov_cpp_batch_potential_call_count
legacy_dcov_cpp_batch_potential_group_count
legacy_dcov_cpp_batch_potential_max_group_size
legacy_dcov_cpp_batch_potential_mean_group_size
legacy_dcov_cpp_batch_potential_reuse_opportunity_count
legacy_dcov_cpp_batch_potential_reuse_ratio
```

These counters do not change dCov authority or execution. They estimate, at
the current skeleton level boundaries, how many scalar C++ dCov backend calls
could be grouped by the same batch shape. The next Phase 2A artifact should
run the current recommended route and record these fields before implementing
the real batched C++ workspace.

Full 351x48 diagnostic artifact:

```text
fastkpc/artifacts/legacy_dcov_gamma_cpp_batch_potential_v1

route:
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s

correctness:
  edge_count = 110 / 110
  adjacency_identical = TRUE
  SHD = 0
  n.edgetests exact = TRUE
  n.edgetests = 2213,52659,125293,40694,13293,5422,835,80

runtime:
  elapsed_sec = 908.895
  residual_worker_ms = 9067761
  mgcv_fit_count = 273284
  cache hits / misses = 203268 / 273284
  dCov cpp backend ms = 3977327
  dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
  Spectra count / converged / failed = 478808 / 478808 / 0

batch-potential:
  scalar C++ dCov calls = 239404
  level-local batch shape groups = 8
  max group size = 125293
  mean group size = 29925.5
  reuse opportunity = 239396
  reuse ratio = 0.9999666
```

By level, each active skeleton level collapses to one dCov batch shape group:

```text
level 0:   1128 calls -> 1 group
level 1:  52659 calls -> 1 group
level 2: 125293 calls -> 1 group
level 3:  40694 calls -> 1 group
level 4:  13293 calls -> 1 group
level 5:   5422 calls -> 1 group
level 6:    835 calls -> 1 group
level 7:     80 calls -> 1 group
```

Conclusion:

```text
The scalar C++ dCov backend is shape-homogeneous within skeleton levels on the
full 351x48 route. Phase 2A should proceed to a level-local batched C++ dCov
workspace that preserves per-level canonical output order while reducing
per-call Rcpp overhead, repeated allocation, distance/lowrank workspace setup,
and wrapper dispatch.
```

Native batch primitive checkpoint:

```text
status: implemented as a C++ oracle/batch substrate, not a production backend
scope:
  legacy_dcov_gamma_cpp_oracle_export()
  legacy_dcov_gamma_cpp_oracle_batch_export()

change:
  scalar and batch exports now share an internal native compute kernel
  batch export no longer calls the exported scalar/list wrapper per column
  batch diagnostics now aggregate stage timings and lowrank counters:
    input / distance / lowrank / statistic / moment / pgamma
    full eig / Spectra counts and failures
    accounted / unaccounted / total timing

tests:
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_batch_oracle.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_oracle.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_spectra_oracle.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_dcov_cpp_backend.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_shadow_route.R

status:
  This removes the per-column Rcpp list wrapper inside the C++ batch oracle and
  gives the next scheduler integration enough diagnostics to verify true batch
  execution. It is still not the production Phase 2A backend because the legacy
  skeleton does not yet route level-local dCov tasks through the batch export.
```

Env-gated scheduler integration checkpoint:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=chunk
```

Status:

```text
status: implemented as an env-gated worker-chunk prototype, not default
scope:
  compatible legacy dcc.gamma route
  C++ legacy dCov backend only
  affinity worker chunks on conditional levels

behavior:
  each worker chunk advances the existing edge-local CI state machine
  only currently reached CI tests are evaluated
  residual generation remains legacy mgcv/regrXonS or the selected residual
    backend for that route
  reached dCov tests in the current chunk round are sent through the native
    C++ batch oracle
  p-values are replayed back into the unchanged edge state machine
  parent level replay remains canonical by edge index

diagnostics:
  legacy_dcov_cpp_batch_backend_enabled
  legacy_dcov_cpp_batch_backend_count
  legacy_dcov_cpp_batch_backend_pair_count
  legacy_dcov_cpp_batch_backend_ms
  legacy_dcov_cpp_batch_backend_error_count
  legacy_dcov_cpp_batch_backend_fallback_count
  legacy_dcov_cpp_batch_backend_max_batch_size
  legacy_dcov_cpp_batch_backend_mean_batch_size

targeted gate:
  Rscript fastkpc/tests/test_precision_compatible_legacy_dcov_cpp_backend.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_batch_oracle.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_oracle.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_spectra_oracle.R
  Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_shadow_route.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_parallel_runtime_breakdown.R

status:
  The targeted gate proves the env-gated chunk batch path preserves adjacency
  and n.edgetests on the tested compatible skeleton while reporting batch
  counters and zero batch errors/fallbacks. This is still not a promoted route.
```

Full 351x48 worker-chunk batch artifact:

```text
fastkpc/artifacts/legacy_dcov_gamma_cpp_chunk_batch_backend_v1

route:
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=chunk

correctness:
  edge_count = 110 / 110
  adjacency_identical = TRUE
  SHD = 0
  n.edgetests exact = TRUE
  n.edgetests = 2213,52659,125293,40694,13293,5422,835,80

runtime:
  elapsed_sec = 933.183
  baseline elapsed_sec = 908.895
  elapsed_delta_vs_baseline_sec = +24.288
  residual_worker_ms = 9137472
  mgcv_fit_count = 273284
  cache hits / misses = 203268 / 273284

dCov:
  dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
  level 0 scalar backend ms = 280031
  chunk batch calls / pairs = 54233 / 238276
  chunk batch max / mean size = 63 / 4.393561
  chunk batch ms / errors / fallbacks = 3877138 / 0 / 0
  scalar plus chunk-batch dCov worker-ms = 4157169
  Spectra count / converged / failed / fallback full eig =
    478808 / 478808 / 0 / 0

residual boundary:
  mgcv same-S setup provider chunk enabled / count = FALSE / 0
```

Decision:

```text
FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=chunk is correctness-clean but
performance-negative on the full 351x48 gate. It must remain experimental and
must not replace the current recommended S-affinity route.

The negative result explains why: worker-chunk scheduling does not realize the
level-local batch potential. The theoretical diagnostic found 8 level-local
shape groups, but the worker-chunk prototype produced 54,233 small batch calls
with mean batch size 4.39 and late levels near size 1. The extra wrapper,
matrix materialization, and tiny-batch overhead erase the native batch benefit.

The next Phase 2A attempt should not tune this worker-chunk route. It should
use a broader level/round-local dCov batching plan that can approach the
observed level-local shape groups while preserving canonical replay.
```

Round-local batch potential diagnostic:

```text
fastkpc/artifacts/legacy_dcov_gamma_cpp_round_batch_potential_v1

route:
  FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
  FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
  FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s

correctness:
  edge_count = 110 / 110
  adjacency_identical = TRUE
  SHD = 0
  n.edgetests exact = TRUE
  n.edgetests = 2213,52659,125293,40694,13293,5422,835,80
  dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
  Spectra count / converged / failed = 478808 / 478808 / 0

runtime:
  elapsed_sec = 917.610
  residual_worker_ms = 9122027
  mgcv_fit_count = 273284
  cache hits / misses = 203268 / 273284
  dCov cpp backend ms = 4200561

level-shape potential:
  calls / groups = 239404 / 8
  max / mean group size = 125293 / 29925.5
  reuse / ratio = 239396 / 0.9999666

round-local potential:
  calls / rounds = 239404 / 4059
  max / mean round size = 1128 / 58.9810
  reuse / ratio = 235345 / 0.9830454
```

By-level round-local potential:

```text
level 0: calls 1128   -> rounds    1, max 1128, mean 1128.000
level 1: calls 52659  -> rounds   92, max 1085, mean  572.380
level 2: calls 125293 -> rounds 1126, max  539, mean  111.273
level 3: calls 40694  -> rounds 1529, max  193, mean   26.615
level 4: calls 13293  -> rounds  750, max  127, mean   17.724
level 5: calls 5422   -> rounds  468, max   79, mean   11.585
level 6: calls 835    -> rounds   85, max   23, mean    9.824
level 7: calls 80     -> rounds    8, max   17, mean   10.000
```

Conclusion:

```text
The round-local diagnostic narrows the next implementation target. The
worker-chunk prototype produced 54,233 batch calls with mean batch size 4.39.
The synchronous round-local potential reduces that to 4,059 candidate batches
with mean batch size 58.98 while preserving the exact set of executed dCov
tests. This is still far from the theoretical 8 level-local shape groups, but
it is a much more realistic execution boundary because edge deletion can be
replayed after each round.

The next Phase 2A implementation should target an env-gated round-local dCov
batch scheduler, not worker-chunk batching. It must preserve per-edge state,
apply p-values after each synchronous round, and keep the parent canonical
replay semantics unchanged.
```

Env-gated round-local scheduler checkpoint:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=round
```

Status:

```text
status: implemented as an env-gated R-level prototype, not default
scope:
  compatible legacy dcc.gamma route
  C++ legacy dCov backend only

behavior:
  parent owns the edge-local state machine for the active skeleton level
  active CI tests are collected synchronously by state-machine round
  residual/input preparation is parallelized across fork workers
  parent sends prepared current-round vectors through one C++ batch dCov call
  p-values are replayed back into the unchanged edge state machine
  parent level replay remains canonical by edge index

diagnostics:
  legacy_dcov_cpp_batch_round_enabled
  legacy_dcov_cpp_batch_round_prepare_worker_count
  legacy_dcov_cpp_batch_round_prepare_task_count

targeted gate:
  Rscript fastkpc/tests/test_precision_compatible_legacy_dcov_cpp_backend.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_parallel_runtime_breakdown.R
  git diff --check

status:
  The targeted gate proves the env-gated round batch path preserves adjacency
  and n.edgetests on the tested compatible skeleton while reporting multi-worker
  round preparation and zero batch errors/fallbacks.
```

Full 351x48 gate attempt:

```text
fastkpc/artifacts/legacy_dcov_gamma_cpp_round_batch_backend_v1
```

Status:

```text
status: no completed artifact
attempt:
  full 351x48 route was started with:
    FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
    FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
    FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
    FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=round

result:
  initial parent-serial implementation exposed an architecture flaw:
    round mode serialized mgcv residual/input preparation in the parent
  fixed implementation parallelized current-round preparation across workers,
    but the full route still exceeded the existing negative chunk-batch gate
    wall-time window before completing

decision:
  FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=round must remain experimental.
  It is not a promotion candidate and should not replace the recommended
  S-affinity route.

reason:
  R-level synchronous round batching reduces intended dCov batch count, but it
  introduces too much parent orchestration / per-round fork and materialization
  overhead for the full 351x48 route. The next dCov batching attempt should
  move the round/level batch boundary into a C++/persistent-worker scheduler or
  C++ skeleton data-plane boundary, not continue tuning R-level round batching.
```

#### Artifact

```text
legacy_dcov_gamma_cpp_chunk_batch_backend_v1
```

#### Gates

```text
SHD = 0
n.edgetests exact = TRUE
worker-ms lower than scalar C++ backend: failed
wall time lower than scalar C++ backend: failed
```

### Phase 2B — CUDA legacy-compatible dCov backend

#### Goal

GPU-accelerate the legacy-compatible dCov backend, using C++ Spectra/backend results as correctness oracle.

#### Two possible subpaths

1. **Legacy lowrank-compatible CUDA**

   ```text
   replicate legacy dcov.gamma decision semantics
   preserve numCol=floor(n/10) behavior
   use R/C++ oracle for parity
   ```

2. **Direct exact dCov with verifier**

   ```text
   shadow exact direct dCov against legacy dcov.gamma
   if decision_flip_count = 0 on full 351x48, candidate route is viable
   otherwise use near-alpha legacy verifier or fall back to lowrank-compatible path
   ```

#### Required shadow artifact before production

```text
legacy_dcov_gamma_cuda_shadow_full_351x48_v1
```

#### Required production artifact before promotion

```text
legacy_dcov_gamma_cuda_backend_full_351x48_v1
```

#### Gates

```text
edge_count = 110 / 110
SHD = 0
n.edgetests exact = TRUE
legacy_dcov_gamma_count matches
error_count = 0
fallback_count recorded and acceptable
wall time improves over C++ backend
```

---

## Phase 3 — mgcv residual oracle trace and replay specification

### Goal

Define the exact mgcv/regrXonS residual semantics that C++/CUDA must reproduce.

This is the most important correctness phase for the final compatible CUDA engine.

### Important policy

Do not replace mgcv with a different smoother as the compatible path. Simplified B-spline/P-spline/fastSpline routes may be useful as approximate fast backends, but not for SHD=0 legacy-compatible execution.

### Phase 3A — Capture mgcv residual oracle traces

#### Artifact

```text
fastkpc/artifacts/mgcv_residual_oracle_v1
```

Initial artifact status:

```text
status: created
case_count: 8
error_count: 0
coverage:
  |S| = 1
  |S| = 2
  |S| > 2
  full_smooth legacy formula route
  additive_smooth legacy formula route
  full 351x48 skeleton deletion source
  near-alpha deletion decisions
  hot level-2 deletion
  late sparse deletion
  largest observed |S| deletion
```

The current oracle trace is still intentionally small, but its cases are now selected from the full 351x48 skeleton deletion log and carry source `pMax` / level diagnostics. It should be expanded further if rank-deficient, collinear, near-constant, or other envelope-risk examples are discovered.

Current generator coverage also records per-case envelope diagnostics needed by
the C++/CUDA residual executor:

```text
conditioning rank / rank-deficient flag
conditioning condition-kappa
near-constant conditioning / target counts
mgcv family, link, convergence, smooth labels, smooth counts
linear predictor matrix dimensions, rank, rank-deficiency, condition-kappa
downstream legacy dCov alpha, p-value, alpha decision, log-alpha distance
```

#### Cases to include

- `|S| = 1`
- `|S| = 2`
- `|S| > 2`
- hot level-2 cases
- late sparse levels
- near-alpha deletion decisions
- rank-deficient / collinear / near-constant examples if present
- representative cases from the 351x48 run

#### For each case record

```text
target
conditioning set S
actual formula route
mgcv/regrXonS parameters
residual vector
fitted vector if available
edf / rank / smoothing info if accessible
conditioning rank / near-constant / lpmatrix diagnostics
runtime
downstream legacy dCov p-value
decision at alpha
```

### Phase 3B — R-level narrow executable spec

#### Goal

Build a small R-level spec for the actual regrXonS/kpc subset only. This is not a full mgcv clone and not a performance backend.

#### Artifact

```text
fastkpc/artifacts/mgcv_residual_replay_spec_v1
```

Initial artifact status:

```text
status: created
case_count: 8
residual_pair_match_count: 8
dcov_p_match_count: 8
decision_flip_count: 0
max_residual_x_abs_diff: 0
max_residual_y_abs_diff: 0
max_dcov_p_abs_diff: 0
pass: TRUE
```

Current replay artifact schema also carries the oracle/replay envelope
diagnostics added to `mgcv_residual_oracle_v1`:

```text
dcov alpha and log-alpha distance
conditioning rank / rank-deficient flags
near-constant conditioning / target counts
lpmatrix rank / rank-deficient flags
smooth labels and smooth counts
summary-level rank-deficient / near-constant counts
```

This is an executable R-level replay spec over the current
`mgcv_residual_oracle_v1` cases. It replays the same legacy formula route,
residual generation, downstream legacy dCov p-value, alpha decision, and
envelope diagnostics. It is not a replacement residual backend.

#### Gate

For oracle cases:

```text
residual parity or decision parity must match mgcv oracle
legacy dCov p-values must not flip decisions
```

### Phase 3C — C++ mgcv-compatible replay executor

#### Goal

Given a captured or extracted mgcv setup, C++ should compute residuals matching mgcv for supported envelopes.

#### Scope order

1. `|S| = 1`
2. `|S| = 2`
3. `|S| > 2` additive/full formulas

#### Initial mode

Shadow only:

```text
C++ computes residuals
legacy mgcv remains authority
compare residuals, p-values, decisions
```

#### Artifacts

```text
mgcv_residual_cpp_replay_oracle_v1
mgcv_residual_cpp_shadow_full_351x48_v1
```

Initial captured-setup shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_shadow_v1
```

Status:

```text
status: created
mode: shadow only, not authoritative
backend: mgcvCapturedCppReplay / captured-setup-matvec-v1
case_count: 8
cpp_supported_count: 8
residual_pair_match_count: 8
dcov_p_match_count: 8
decision_match_count: 8
decision_flip_count: 0
max_residual_x_abs_diff: 3.048122e-10
max_residual_y_abs_diff: 2.142921e-10
max_dcov_p_abs_diff: 2.261868e-11
residual_tol: 1e-8
p_tol: 1e-9
pass: TRUE
```

This is not yet a full mgcv setup extractor or production residual backend.
It proves that captured `lpmatrix + coefficients + response` can be replayed
in C++ for the current oracle cases without decision flips. The next residual
work must extract/reconstruct the mgcv setup rather than relying on captured
R `predict(type = "lpmatrix")` output.

Setup-extracted fixed-sp shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_setup_shadow_v1
```

Status:

```text
status: created
mode: shadow only, not authoritative
backend: mgcvExtractCPU fixed-sp setup self-solve
setup provider: mgcv::gam(fit = FALSE) with selected oracle sp
solver kernel: mgcv C_magic fixed-sp path
case_count: 8
setup_supported_count: 8
residual_pair_match_count: 8
dcov_p_match_count: 8
decision_match_count: 8
decision_flip_count: 0
max_residual_x_abs_diff: 0
max_residual_y_abs_diff: 0
max_dcov_p_abs_diff: 0
residual_tol: 1e-5
p_tol: 1e-5
pass: TRUE
```

This is the first setup-extracted replay artifact over the current full
skeleton oracle cases. It still uses mgcv as the setup provider and the mgcv
fixed-sp C kernel as the solve authority, so it is not the final C++/CUDA
numeric executor. It does prove that the current oracle cases can be reduced
from full `mgcv::gam()` fitting to extracted fixed-sp setup plus replayed
residual solve without changing downstream legacy dCov decisions.

Explicit C++ fixed-sp numeric shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_v1
```

Status:

```text
status: created
mode: shadow only, not authoritative
setup provider: mgcv::gam(fit = FALSE) with selected oracle sp
numeric executor: native C++ penalized fixed-sp normal-equation solver
case_count: 8
setup_supported_count: 8
residual_pair_match_count: 8
dcov_p_match_count: 8
decision_match_count: 8
decision_flip_count: 0
max_residual_x_abs_diff: 1.237126e-10
max_residual_y_abs_diff: 2.477574e-12
max_dcov_p_abs_diff: 3.99325e-11
residual_tol: 1e-5
p_tol: 1e-5
pass: TRUE
```

This is the first explicit native C++ numeric executor checkpoint for the
current extracted fixed-sp oracle cases. It is still not production because
the setup provider is mgcv and the coverage is the small Phase 3 oracle set,
but it removes the mgcv `C_magic` solve path from the shadow residual replay.
The next Phase 3C step is to expand this C++ numeric shadow across more
full-skeleton residual requests and unsupported-envelope diagnostics before
considering any env-gated residual backend.

Expanded explicit C++ numeric shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_expanded_v1
```

Status:

```text
status: created
mode: shadow only, not authoritative
source: expanded cases from full 351x48 skeleton deletion log
case_count: 28
setup_supported_count: 28
setup_unsupported_count: 0
residual_pair_match_count: 27
residual_pair_mismatch_count: 1
dcov_p_match_count: 27
dcov_p_mismatch_count: 1
decision_match_count: 28
decision_mismatch_count: 0
decision_flip_count: 0
max_residual_x_abs_diff: 0.06386512
max_residual_y_abs_diff: 1.029583e-10
max_dcov_p_abs_diff: 0.003665505
residual_tol: 1e-5
p_tol: 1e-5
solver: cpp
pass: FALSE
```

Mismatch case:

```text
case_id: expanded351_22_s4_x8_y9
source_level: 4
S_size: 4
S_key: 1|4|5|6
target_x: 8
target_y: 9
source_pmax / oracle p: 0.1130094
C++ numeric p: 0.1093439
p_abs_diff: 0.003665505
decision_match: TRUE
setup_status: mismatch
```

This expanded artifact is deliberately not a promotion gate pass. It proves
the C++ fixed-sp numeric executor has strong coverage on the sampled full
skeleton cases and no decision flips, but it also exposes a real deeper-level
numeric drift envelope. The next Phase 3C work is to isolate this `|S|=4`
drift against the mgcv `C_magic` self-solve and decide whether the C++ normal
equation solver needs a closer mgcv-equivalent kernel, stricter supported
envelope, or fallback policy.

C++ numeric drift isolation artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_drift_isolation_v1
```

Status:

```text
status: created
case_count: 1
target_count: 2
case_id: expanded351_22_s4_x8_y9
S_size: 4
source_level: 4
cpp_matches_r_normal_target_count: 2 / 2
cmagic_matches_oracle_target_count: 2 / 2
cpp_matches_cmagic_target_count: 1 / 2
decision_flip_count: 0
normal_equation_vs_mgcv_magic_count: 1
max_cpp_vs_r_normal_abs_diff: 1.998401e-13
max_cpp_vs_cmagic_abs_diff: 0.06386512
max_cmagic_vs_oracle_abs_diff: 0
max_normal_matrix_condition: 4.267963e13
```

Conclusion:

```text
native C++ normal-equation solve == R normal-equation solve
mgcv C_magic fixed-sp replay == full mgcv oracle
drift layer = normal_equation_vs_mgcv_magic
```

This means the current native C++ normal-equation solver is not simply buggy;
it is solving a numerically different/unstable path than mgcv's fixed-sp
kernel for an ill-conditioned deeper-level additive setup. Promotion must
therefore use one of these routes:

```text
1. implement a closer mgcv-equivalent C++ solve kernel;
2. fail closed / fallback for high-condition or deeper additive envelopes;
3. keep native C++ fixed-sp solve as shadow-only until full expanded shadow
   has zero strict residual/p mismatches or an accepted fallback policy.
```

Guarded C++ fixed-sp numeric shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_expanded_v1
```

Status:

```text
status: created
mode: shadow only, not authoritative
source: expanded cases from full 351x48 skeleton deletion log
solver: cpp_guarded
condition_threshold: 1e12
case_count: 28
setup_supported_count: 28
setup_unsupported_count: 0
residual_pair_match_count: 28
residual_pair_mismatch_count: 0
dcov_p_match_count: 28
dcov_p_mismatch_count: 0
decision_match_count: 28
decision_flip_count: 0
fallback_count: 19 targets
high_condition_fallback_count: 19 targets
cpp_guarded_count: 37 targets
max_normal_matrix_condition: 9.20319e13
max_residual_x_abs_diff: 7.194023e-12
max_residual_y_abs_diff: 2.175637e-11
max_dcov_p_abs_diff: 5.538792e-12
pass: TRUE
```

The previous expanded mismatch case is now covered by the guarded high-condition
fallback:

```text
case_id: expanded351_22_s4_x8_y9
normal_matrix_condition_x: 4.267963e13
normal_matrix_condition_y: 3.799319e13
fallback_reason_x: high_normal_matrix_condition
fallback_reason_y: high_normal_matrix_condition
residual_pair_match: TRUE
dcov_p_match: TRUE
decision_flip: FALSE
```

This is a useful fail-closed supported-envelope policy, but it is still not a
production residual backend. It proves that high-condition guardrails can make
the current native C++ fixed-sp executor safe under the expanded shadow sample
by falling back to mgcv C_magic replay. The next promotion step must either
expand this guarded shadow across more full-skeleton cases or replace the
normal-equation path with a closer mgcv-equivalent numeric kernel.

Wider guarded C++ fixed-sp numeric shadow artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_expanded_wide_v1
```

Selection profile:

```text
near_alpha_count: 64
per_s_size_count: 32
per_level_count: 20
max_cases: 160
selected cases: 116
```

Status:

```text
status: created
mode: shadow only, not authoritative
source: expanded cases from full 351x48 skeleton deletion log
solver: cpp_guarded
condition_threshold: 1e12
case_count: 116
setup_supported_count: 116
setup_unsupported_count: 0
residual_pair_match_count: 116
residual_pair_mismatch_count: 0
dcov_p_match_count: 116
dcov_p_mismatch_count: 0
decision_match_count: 116
decision_flip_count: 0
fallback_count: 52 targets
high_condition_fallback_count: 52 targets
cpp_guarded_count: 180 targets
max_normal_matrix_condition: 2.083868e14
max_residual_x_abs_diff: 1.399334e-10
max_residual_y_abs_diff: 1.659735e-10
max_dcov_p_abs_diff: 2.134748e-11
pass: TRUE
```

Coverage:

```text
cases by |S| / level:
  |S|=1: 32
  |S|=2: 32
  |S|=3: 32
  |S|=4: 16
  |S|=5:  3
  |S|=6:  1

fallback targets by |S| / level:
  |S|=1:  0
  |S|=2:  0
  |S|=3: 28
  |S|=4: 20
  |S|=5:  3
  |S|=6:  1
```

This strengthens the current supported-envelope interpretation:

```text
|S| <= 2 full-smooth sampled targets:
  native C++ fixed-sp normal-equation solve matched mgcv oracle under strict
  residual and p-value tolerances with zero guarded fallback.

|S| >= 3 additive sampled targets:
  high-condition guardrails are frequently required. The fail-closed fallback
  policy preserves strict residual/p-value parity and decisions, but native
  normal-equation solve is not yet a production substitute for mgcv C_magic in
  this deeper additive envelope.
```

Strict `|S|<=2` guarded envelope artifact:

```text
fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_s2_envelope_wide_v1
```

Selection profile:

```text
near_alpha_count: 64
per_s_size_count: 32
per_level_count: 20
max_cases: 160
selected cases: 116
native_s_size_limit: 2
```

Status:

```text
status: created
mode: shadow only, not authoritative
source: expanded cases from full 351x48 skeleton deletion log
solver: cpp_guarded
condition_threshold: 1e12
case_count: 116
setup_supported_count: 116
setup_unsupported_count: 0
residual_pair_match_count: 116
residual_pair_mismatch_count: 0
dcov_p_match_count: 116
dcov_p_mismatch_count: 0
decision_match_count: 116
decision_flip_count: 0
fallback_count: 104 targets
high_condition_fallback_count: 0 targets
outside_envelope_fallback_count: 104 targets
cpp_guarded_count: 128 targets
max_normal_matrix_condition: 2.083868e14
max_residual_x_abs_diff: 1.387113e-11
max_residual_y_abs_diff: 2.538347e-11
max_dcov_p_abs_diff: 1.429523e-12
pass: TRUE
```

Fallback distribution:

```text
cases by |S| / level:
  |S|=1: 32
  |S|=2: 32
  |S|=3: 32
  |S|=4: 16
  |S|=5:  3
  |S|=6:  1

fallback targets by |S| / level:
  |S|=1:  0
  |S|=2:  0
  |S|=3: 64
  |S|=4: 32
  |S|=5:  6
  |S|=6:  2
```

This gives a concrete fail-closed residual shadow policy:

```text
native C++ fixed-sp solve:
  allowed for sampled |S|<=2 full-smooth setups

mgcv C_magic fallback:
  required for |S|>2 additive setups under this strict envelope

promotion status:
  still shadow-only, but now the supported native envelope is explicit and
  observable through fallback diagnostics.
```

Initial full-route guarded residual shadow artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_synthetic_v1
```

Status:

```text
status: created
mode: full legacy-parallel skeleton shadow, not authoritative
data: synthetic n=66, p=6
alpha: 0.08
max_conditioning_size: 2
residual authority: legacy regrXonS / mgcv
shadow solver: cpp_guarded
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 24,19,0
shadow_n.edgetests:   24,19,0
baseline_edge_count: 3
shadow_edge_count:   3

residual_request_count: 38
shadow_count: 38
native_count: 38
fallback_count: 0
high_condition_fallback_count: 0
outside_envelope_fallback_count: 0
error_count: 0
residual_mismatch_count: 0
max_abs_diff: 4.563017e-12
max_rel_l2: 3.869293e-12
```

This is the first full-route skeleton replay check for the guarded native
residual envelope. The authoritative residuals and skeleton decisions still
come from legacy `regrXonS`; the C++ fixed-sp path only shadows each conditional
target residual and records parity diagnostics. It verifies that the `|S|<=2`
native envelope can be exercised inside the legacy scheduler without changing
canonical replay on a small skeleton. It is not a full 351x48 gate.

Real 351x48 subset guarded residual shadow artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_real_subset_v1
```

Status:

```text
status: created
mode: full legacy-parallel skeleton shadow, not authoritative
data: real 351x48 fixture, 8 hot-column subset
columns: 1,2,3,4,5,6,9,12
n / p: 351 / 8
alpha: 0.1
max_conditioning_size: 2
num_cores: 2
dCov backend: C++ Spectra
residual authority: legacy regrXonS / mgcv
shadow solver: cpp_guarded
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 55,274,357
shadow_n.edgetests:   55,274,357
baseline_edge_count: 16
shadow_edge_count:   16

residual_request_count: 1262
shadow_count: 1262
native_count: 1262
fallback_count: 0
high_condition_fallback_count: 0
outside_envelope_fallback_count: 0
error_count: 0
residual_mismatch_count: 0
max_abs_diff: 2.103081e-10
max_rel_l2: 1.145284e-10
elapsed_ms: 47348
```

This is the first real-data full-route guarded residual shadow. It validates
the `|S|<=2` native C++ fixed-sp residual replay envelope inside the legacy
scheduler on a real 351-row subset while preserving canonical replay and graph
output. It is still a subset gate, not the full 351x48 acceptance gate.

Deep real 351x48 subset guarded residual shadow artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_real_subset_deep_v1
```

Status:

```text
status: created
mode: full legacy-parallel skeleton shadow, not authoritative
data: real 351x48 fixture, 12 hot-column subset
columns: 1,2,3,4,5,6,9,12,15,16,17,18
n / p: 351 / 12
alpha: 0.1
max_conditioning_size: 3
num_cores: 4
dCov backend: C++ Spectra
residual authority: legacy regrXonS / mgcv
shadow solver: cpp_guarded
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 131,994,1453,243
shadow_n.edgetests:   131,994,1453,243
baseline_edge_count: 20
shadow_edge_count:   20

residual_request_count: 5380
shadow_count: 5380
native_count: 4894
fallback_count: 486
high_condition_fallback_count: 0
outside_envelope_fallback_count: 486
error_count: 0
residual_mismatch_count: 0
max_abs_diff: 2.103081e-10
max_rel_l2: 1.145284e-10
elapsed_ms: 94902
```

This is the first real-data guarded residual shadow that exercises both the
native `|S|<=2` C++ fixed-sp envelope and the fail-closed `|S|>2` mgcv
fallback path inside the legacy scheduler. Canonical replay, adjacency, and
`n.edgetests` remain unchanged, and all 5,380 residual shadow targets match
without errors or mismatches. It is still a subset gate, not the full 351x48
acceptance gate.

Full 351x48 guarded residual shadow artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_full_351x48_v1
```

Status:

```text
status: created
mode: full legacy-parallel skeleton shadow, not authoritative
data: real 351x48 fixture, all columns
n / p: 351 / 48
alpha: 0.1
max_conditioning_size: 46
num_cores: 20
dCov backend: C++ Spectra
residual cache: enabled
residual affinity: s
residual authority: legacy regrXonS / mgcv
shadow solver: cpp_guarded
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
shadow_n.edgetests:   2213,52659,125293,40694,13293,5422,835,80
baseline_edge_count: 110
shadow_edge_count:   110

residual_request_count: 476552
mgcv_fit_count: 273284
residual_cache_hit_count: 203268
shadow_count: 273284
native_count: 157122
fallback_count: 116162
high_condition_fallback_count: 0
outside_envelope_fallback_count: 116162
error_count: 0
residual_mismatch_count: 0
max_abs_diff: 6.223067e-10
max_rel_l2: 4.174908e-10

legacy_dcov_gamma_count: 239404
legacy_dcov_cpp_backend_count: 239404
legacy_dcov_cpp_backend_error_count: 0
legacy_dcov_cpp_backend_fallback_count: 0
legacy_dcov_cpp_lowrank_spectra_count: 478808
legacy_dcov_cpp_lowrank_spectra_converged_count: 478808
legacy_dcov_cpp_lowrank_spectra_failed_count: 0
elapsed_ms: 1848722
```

This is the first full 351x48 legacy-compatible guarded residual shadow pass.
The authoritative residuals still come from legacy `regrXonS` / mgcv, and the
`cpp_guarded` residual path remains shadow-only. Within the current recommended
compatible route, every worker-local mgcv cache miss was shadowed: `|S|<=2`
targets used the native C++ fixed-sp solve, while `|S|>2` targets failed closed
to mgcv fallback. Canonical replay, adjacency, `n.edgetests`, dCov backend
selection, and Spectra convergence remained unchanged.

Initial env-gated guarded residual backend prototype:

```bash
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT=2
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD=1e300
```

Artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_real_subset_deep_v1
```

Status:

```text
status: created
mode: env-gated backend prototype, not default
data: real 351x48 fixture, 12 hot-column subset
columns: 1,2,3,4,5,6,9,12,15,16,17,18
n / p: 351 / 12
alpha: 0.1
max_conditioning_size: 3
num_cores: 4
dCov backend: C++ Spectra
residual authority: cpp_guarded for native envelope, legacy regrXonS fallback
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 131,994,1453,243
backend_n.edgetests:  131,994,1453,243
baseline_edge_count: 20
backend_edge_count:  20

baseline_elapsed_sec: 90.126
backend_elapsed_sec:  73.678
residual_request_count: 5380
cpp_backend_count: 5380
native_count: 4894
r_fallback_count: 486
fallback_count: 486
high_condition_fallback_count: 0
outside_envelope_fallback_count: 486
error_count: 0

legacy_dcov_cpp_backend_count: 2756
legacy_dcov_cpp_backend_error_count: 0
legacy_dcov_cpp_backend_fallback_count: 0
legacy_dcov_cpp_lowrank_spectra_count: 5512
legacy_dcov_cpp_lowrank_spectra_failed_count: 0
```

This is the first real-data env-gated guarded residual backend prototype. It
does switch residual authority for supported `|S|<=2` targets to the native C++
fixed-sp solve, while unsupported `|S|>2` targets fail closed to legacy
`regrXonS`. On the 12-column deep subset it preserves canonical replay and
shows a subset wall-time win. It is still not recommended or default; promotion
requires a full 351x48 artifact with unchanged replay, zero backend errors, and
a wall-time win over the current recommended compatible route.

Full 351x48 env-gated guarded residual backend artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_full_351x48_v1
```

Status:

```text
status: created
mode: env-gated backend prototype, not default
data: real 351x48 fixture, all columns
n / p: 351 / 48
alpha: 0.1
max_conditioning_size: 46
num_cores: 20
dCov backend: C++ Spectra
residual cache: enabled
residual affinity: s
residual authority: cpp_guarded for native envelope, legacy regrXonS fallback
native_s_size_limit: 2
condition_threshold: 1e300

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
backend_n.edgetests:  2213,52659,125293,40694,13293,5422,835,80
baseline_edge_count: 110
backend_edge_count:  110

baseline_elapsed_sec: 898.110
backend_elapsed_sec:  1067.308
elapsed_speedup: 0.841x
residual_request_count: 476552
cache_hit_count: 203268
cache_miss_count: 273284
mgcv_fit_count: 273284
cpp_backend_count: 273284
native_count: 157122
r_fallback_count: 116162
fallback_count: 116162
high_condition_fallback_count: 0
outside_envelope_fallback_count: 116162
error_count: 0

baseline_residual_worker_ms: 9115972
backend_residual_worker_ms: 12734441
legacy_dcov_cpp_backend_count: 239404
legacy_dcov_cpp_backend_error_count: 0
legacy_dcov_cpp_backend_fallback_count: 0
legacy_dcov_cpp_lowrank_spectra_count: 478808
legacy_dcov_cpp_lowrank_spectra_converged_count: 478808
legacy_dcov_cpp_lowrank_spectra_failed_count: 0
```

This full gate is correctness-clean but not performance-viable. The guarded
backend preserves canonical replay, edge count, `n.edgetests`, dCov backend
selection, and Spectra convergence, with zero residual backend errors. However,
full wall time regresses from 898.110 sec to 1067.308 sec and residual worker-ms
increases from 9.12M to 12.73M. Therefore `cpp_guarded` must remain
experimental and must not become the recommended compatible route.

Guarded residual backend timing split diagnostic:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_timing_split_subset_v1
```

Status:

```text
status: created
mode: env-gated backend timing diagnostic
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
backend targets: 5380
native / fallback targets: 4894 / 486
|S| counts 1 / 2 / >2: 1988 / 2906 / 486

backend_worker_ms: 251446
input_formula_setup_ms: 98314
mgcv_gam_fit_ms: 65368
sp_extract_ms: 101
setup_extract_ms: 53598
condition_check_ms: 1850
native_fixed_sp_solve_ms: 2919
fallback_regrXonS_ms: 28691
```

This diagnostic explains the full backend regression. The native C++ fixed-sp
numeric solve is not the expensive part; on the subset it accounts for only
about 2.9s worker-ms. The dominant costs are repeated per-target input/formula
setup, `mgcv::gam`, and setup extraction before the native solve. Therefore the
next acceleration attempt should reuse or batch same-S setup/extraction work.
Optimizing the scalar native solve alone will not address the full-route
regression.

Guarded residual same-S setup reuse potential diagnostic:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_reuse_potential_subset_v1
```

Status:

```text
status: created
mode: env-gated backend same-S reuse potential diagnostic
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
backend / native / fallback targets: 5380 / 4894 / 486
native same-S groups: 78
native same-S targets: 4894
native same-S max targets: 182
native same-S mean targets: 62.74359
native same-S reuse opportunity: 4816
native same-S setup reuse ratio: 0.9840621

input_formula_setup_ms: 99512
mgcv_gam_fit_ms: 66999
setup_extract_ms: 53907
native_fixed_sp_solve_ms: 5020
fallback_regrXonS_ms: 28819
```

This shows that almost all native guarded backend targets share conditioning
sets with other native targets. On the subset, 4,894 native target residuals
collapse to only 78 same-S groups, so a same-S setup reuse design has a 98.4%
setup reuse opportunity before considering load balance and cache boundaries.
This is the strongest evidence so far that the next residual acceleration path
should batch or reuse setup by conditioning set, not optimize scalar solves.

Guarded residual same-S plus selected-sp reuse potential diagnostic:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_sp_reuse_potential_subset_v1
```

Status:

```text
status: created
mode: env-neutral diagnostic over cpp_guarded backend route
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
num_cores: 8

edge_count: 20
n.edgetests: 131,994,1453,243

backend / native / fallback / error targets: 2756 / 2297 / 459 / 0

same-S groups / targets / max / mean:
  78 / 2297 / 56 / 29.448718
same-S reuse opportunity / ratio:
  2219 / 0.96604266

same-S+selected-sp groups / targets / max / mean:
  765 / 2297 / 7 / 3.0026144
same-S+selected-sp reuse opportunity / ratio:
  1532 / 0.6669569

input_formula_setup_ms: 15305
mgcv_gam_fit_ms:        34600
setup_extract_ms:       28499
native_fixed_sp_solve:   2061
fallback_regrXonS_ms:   17482
```

This refines the previous same-S diagnostic. Conditioning set alone
overstates complete fixed-sp setup reuse because selected smoothing parameters
are target-dependent. On this subset, adding the selected-sp signature splits
78 same-S groups into 765 same-S+sp groups and reduces the complete fixed-sp
reuse opportunity from 96.6% to 66.7%. There is still meaningful exact reuse,
but the next serious design should distinguish:

```text
target-independent reuse:
  formula, model frame, basis/penalty metadata for same S

target/sp-dependent reuse:
  fixed-sp setup/factorization and solve batches for same S + selected sp
```

Therefore a production attempt should not assume one fixed-sp setup per S.
It should either batch by same-S+selected-sp for full setup/factorization reuse,
or split the executor so same-S basis/model-frame work is reused while
target-specific smoothing parameter selection remains faithful to mgcv.

Guarded residual same-S setup-structure reuse diagnostic:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_structure_subset_v1
```

Status:

```text
status: created
mode: env-neutral diagnostic over cpp_guarded backend route
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
num_cores: 8

edge_count: 20
n.edgetests: 131,994,1453,243

backend / native / fallback / error targets:
  2756 / 2297 / 459 / 0

same-S groups / targets / reuse / ratio:
  78 / 2297 / 2219 / 0.96604266

same-S+setup-structure groups / targets / max / mean:
  78 / 2297 / 56 / 29.448718
same-S+setup-structure reuse / ratio:
  2219 / 0.96604266

same-S+selected-sp groups / reuse / ratio:
  765 / 1532 / 0.6669569

input_formula_setup_ms: 15369
mgcv_gam_fit_ms:        34235
setup_extract_ms:       28419
native_fixed_sp_solve:   2213
fallback_regrXonS_ms:   17231
```

This confirms the split design implied above. On this subset, the
target-independent setup structure hash is invariant within each same-S group:
same-S and same-S+setup-structure both have 78 groups and 96.6% reuse
opportunity. The selected smoothing parameter still splits those groups into
765 same-S+sp groups. Therefore the next implementation should not treat
`sp` selection as reusable by S alone, but it can target same-S reuse for:

```text
formula construction
model frame / predictor data
model matrix / basis metadata
penalty and constraint structure
```

Then it should batch the fixed-sp numerical replay under:

```text
same S + selected sp
```

This is the current lowest-risk path toward an exact same-S residual executor:
reuse target-independent setup by S, preserve per-target mgcv smoothing
parameter selection, and only share fixed-sp factorization/solve work when the
selected-sp signature also matches.

Guarded residual same-S setup timing potential diagnostic:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_timing_potential_subset_v1
```

Status:

```text
status: created
mode: env-neutral diagnostic over cpp_guarded backend route
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
num_cores: 8

edge_count: 20
n.edgetests: 131,994,1453,243

backend / native / fallback / error targets:
  2756 / 2297 / 459 / 0

same-S groups / targets / reuse / ratio:
  78 / 2297 / 2219 / 0.9660427

same-S+setup groups / reuse / ratio:
  78 / 2219 / 0.9660427

same-S+selected-sp groups / reuse / ratio:
  765 / 1532 / 0.6669569

input_setup_ms:       6378
mgcv_gam_fit_ms:     34388
setup_extract_ms:    28313
condition_ms:         1073
native_solve_ms:      2711
fallback_ms:         13148

same-S setup input potential saved ms:       6378
same-S setup extract potential saved ms:    27414
same-S setup condition potential saved ms:   1073
same-S setup structure potential saved ms:  34865
same-S+sp native solve potential saved ms:   2408
same-S+sp native solve reuse ratio:          0.8882331
gam fit preserved ms:                       34388
```

This timing diagnostic makes the next implementation boundary sharper. The
same-S target-independent setup path has a large estimated worker-ms ceiling on
the subset:

```text
input/formula + setup extraction + condition check potential:
  34.865s worker-ms
```

But the diagnostic also shows that `mgcv::gam` time is preserved:

```text
mgcv_gam_fit_ms = gam_fit_preserved_ms = 34.388s worker-ms
```

Therefore same-S setup reuse is not allowed to skip per-target mgcv smoothing
parameter selection. The first serious prototype should reuse only
target-independent same-S structure:

```text
conditioning-set data/model frame
formula / term structure
model matrix / basis
penalty and constraint structure
condition metadata where semantically identical
```

It must still run target-specific `mgcv::gam` or an exactly equivalent
selected-sp provider for each target. Fixed-sp numeric replay or factorization
can only be shared under the narrower key:

```text
same S + selected sp
```

Same-S guarded residual prefill prototype:

```bash
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT=2
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD=1e300
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL=1
```

Artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_prefill_subset_v1
```

Status:

```text
status: created
mode: env-gated worker/chunk-local prefill prototype, not recommended
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
num_cores: 8

adjacency_identical: TRUE
n.edgetests_identical: TRUE

baseline_elapsed_sec: 21.106
prefill_elapsed_sec:  20.343

baseline_mgcv_fit_count: 2756
prefill_mgcv_fit_count:  3488
baseline_cache_hits:     2624
prefill_cache_hits:      4921

prefill_groups:  599
prefill_targets: 3029
prefill_inserts: 3029
prefill_unused:   732
prefill_errors:     0

baseline_backend_ms:  99222
prefill_backend_ms: 108493
prefill_ms:          96729
```

This prototype preserves canonical replay on the subset and slightly improves
wall time, but it is not true same-S setup reuse. It still runs the guarded
backend per target during prefill, and it overcomputes unused residual keys.
The higher fit/backend counts confirm that prefill is only a scheduling/cache
experiment. It should remain experimental and must not be promoted without a
full 351x48 wall-time win. The next serious residual path should reuse or batch
the same-S setup itself rather than merely precomputing per-target residuals.

Same-S target-independent setup provider prototype:

```bash
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT=2
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD=1e300
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_PREFILL=1
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=1
```

Artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_subset_v1
```

Status:

```text
status: created
mode: env-gated same-S target-independent setup provider prototype,
      not default and not recommended yet
data: real 351x48 fixture, 12 hot-column subset
max_conditioning_size: 3
num_cores: 8

adjacency_identical: TRUE
n.edgetests_identical: TRUE
baseline_n.edgetests: 131,994,1453,243
provider_n.edgetests: 131,994,1453,243
baseline/provider edge_count: 20 / 20

baseline_elapsed_sec: 54.676
provider_elapsed_sec: 16.793

baseline/provider backend targets: 3488 / 3488
baseline/provider native targets: 3029 / 3029
baseline/provider fallback targets: 459 / 459
provider_error_count: 0
setup_provider_error_count: 0

setup_provider groups / targets / templates / reuse:
  599 / 3029 / 599 / 2430

setup_provider_setup_ms: 8283
baseline/provider setup_extract_ms: 38712 / 8283
baseline/provider gam_fit_ms:       46113 / 46753
```

This is the first true same-S setup reuse prototype in the legacy-compatible
residual line. It reuses target-independent setup structure inside same-S
groups while preserving per-target `mgcv::gam` selected-sp authority. On the
real 12-column subset it preserves canonical replay and cuts setup extraction
worker-ms substantially:

```text
setup_extract_ms: 38.7s -> 8.3s
```

The subset wall-time win is large, but this route is still experimental. It is
implemented through the existing same-S prefill vehicle, so it may still
overcompute unused residuals and must not be promoted without a full 351x48
artifact. The next gate is:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_full_351x48_v1
```

Required full-gate checks:

```text
edge_count = 110 / 110
SHD = 0
n.edgetests exact = TRUE
setup_provider_error_count = 0
residual backend error_count = 0
elapsed < current recommended S-affinity route
```

Full 351x48 same-S setup provider artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_full_351x48_v1
```

Status:

```text
status: created
mode: env-gated same-S target-independent setup provider prototype,
      not default and not recommended
data: real 351x48 fixture, all columns
n / p: 351 / 48
alpha: 0.1
max_conditioning_size: 46
num_cores: 20

adjacency_identical: TRUE
SHD: 0
n.edgetests_identical: TRUE
baseline_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
provider_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
baseline/provider edge_count: 110 / 110

baseline_elapsed_sec: 899.461
provider_elapsed_sec: 1152.628
elapsed_speedup: 0.780x

baseline/provider residual_worker_ms:
  9010282 / 14602256

baseline/provider mgcv_fit_count:
  273284 / 402019

provider backend / native / fallback / error targets:
  402019 / 285857 / 116162 / 0

setup_provider groups / targets / templates / reuse:
  22585 / 285857 / 22585 / 263272

setup_provider_setup_ms: 516777
setup_provider_error_count: 0
prefill_unused_count: 128735
prefill_ms: 9792094

provider input/setup/gam/native/fallback ms:
  141770 / 516777 / 7702648 / 483903 / 4721014

dCov cpp backend count / errors / fallbacks:
  239404 / 0 / 0

Spectra count / converged / failed:
  478808 / 478808 / 0
```

This full gate is correctness-clean but performance-negative. It proves the
same-S setup provider preserves canonical replay and graph output on the full
351x48 benchmark, but the route must not be promoted:

```text
elapsed: 899.461s -> 1152.628s
residual worker-ms: 9.01M -> 14.60M
```

The failure is not dCov and not residual mismatch. It is the current prefill
vehicle overcomputing residuals:

```text
baseline mgcv fits: 273284
provider mgcv fits: 402019
prefill unused residual keys: 128735
```

Therefore the setup provider idea remains useful, but the prefill vehicle is
not viable for full 351x48. The next prototype must avoid precomputing unused
target|S residuals. It should move same-S setup reuse into an on-demand or
consumed-key execution path where only residual keys actually required by
canonical CI tasks are fitted.

Consumed-key same-S setup provider artifact:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_NATIVE_S_SIZE_LIMIT=2
FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND_CONDITION_THRESHOLD=1e300
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=consumed
```

Artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_consumed_keys_v1
```

Status:

```text
status: created
mode: env-gated consumed-key same-S setup provider prototype,
      not default and not recommended
data: real 351x48 fixture, all columns
n / p: 351 / 48
alpha: 0.1
max_conditioning_size: 46
num_cores: 20

adjacency_identical: TRUE
SHD: 0
n.edgetests_identical: TRUE
baseline_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
provider_n.edgetests: 2213,52659,125293,40694,13293,5422,835,80
baseline/provider edge_count: 110 / 110

baseline_elapsed_sec: 909.418
provider_elapsed_sec: 1075.924
elapsed_speedup: 0.845x

baseline/provider residual_worker_ms:
  9083787 / 12154374

baseline/provider mgcv_fit_count:
  273284 / 273284

baseline/provider cache_hits:
  203268 / 203268

provider backend / native / fallback / error targets:
  273284 / 157122 / 116162 / 0

setup_provider groups / targets / templates / reuse:
  42418 / 84836 / 42418 / 42418

setup_provider_setup_ms: 940660
setup_provider_error_count: 0
prefill_enabled: FALSE
prefill_unused_count: 0

provider input/setup/gam/native/fallback ms:
  126829 / 2326981 / 3978266 / 240563 / 4877692

dCov cpp backend count / errors / fallbacks:
  239404 / 0 / 0

Spectra count / converged / failed:
  478808 / 478808 / 0
```

This gate proves the consumed-key variant fixes the prefill overcompute problem:

```text
prefill_unused_count: 128735 -> 0
mgcv_fit_count:       402019 -> 273284
```

It is still performance-negative versus the recommended S-affinity route:

```text
elapsed:            909.418s -> 1075.924s
residual worker-ms: 9.08M    -> 12.15M
```

The failure mode changed. It is no longer unused residual precomputation.
Instead, the provider is still only pair-local in the current consumed path:
each CI call can reuse setup across at most the two residual targets in that
pair, so it does not capture the large same-S group opportunity measured in the
diagnostics. It also still routes all misses through the slower `cpp_guarded`
residual backend, including many outside-envelope fallbacks. Do not promote
`FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=consumed`.

Actual consumed cache-miss same-S grouping artifact:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
```

Artifact:

```text
fastkpc/artifacts/legacy_mgcv_residual_cache_miss_key_grouping_v1
```

Status:

```text
status: created
mode: current recommended compatible route with actual cache hit/miss-key
      diagnostics
data: real 351x48 fixture, all columns
n / p: 351 / 48
alpha: 0.1
max_conditioning_size: 46
num_cores: 20

elapsed_sec: 899.077
edge_count: 110
n.edgetests exact: TRUE
n.edgetests: 2213,52659,125293,40694,13293,5422,835,80

residual_worker_ms: 9103781
mgcv_fit_count: 273284
residual_request_count: 476552
cache hits / misses: 203268 / 273284
cache hit / miss keys: 203268 / 273284

actual miss same-S groups / targets:
  8634 / 273284

actual miss same-S max / mean targets:
  637 / 31.652

actual miss same-S reuse opportunity / ratio:
  264650 / 0.968406

all request same-S groups / targets / reuse:
  8634 / 476552 / 467918

dCov cpp backend count / errors / fallbacks:
  239404 / 0 / 0

Spectra count / converged / failed:
  478808 / 478808 / 0
```

By level, the actual cache-miss same-S opportunity is concentrated in the
early hot levels:

```text
level 1:
  miss keys: 23008
  miss same-S groups: 48
  reuse opportunity / ratio: 22960 / 0.997914

level 2:
  miss keys: 134114
  miss same-S groups: 1126
  reuse opportunity / ratio: 132988 / 0.991604

level 3:
  miss keys: 77808
  miss same-S groups: 4064
  reuse opportunity / ratio: 73744 / 0.947769
```

This artifact proves the large same-S reuse opportunity is present in the
actual consumed cache-miss stream, not only in over-prefetch estimates. The
pair-local consumed provider fails because it cannot group across CI calls.
The next serious prototype must change execution shape: collect actual misses
within a worker chunk or level, group those misses by S, fit each consumed
target once through a same-S provider batch, then run dCov and parent canonical
replay unchanged.

#### Gate before production use

```text
decision_flip_count = 0
full skeleton SHD = 0
edge_count = 110 / 110
n.edgetests exact = TRUE
```

---

## Phase 4 — CUDA mgcv-compatible residual executor

### Goal

GPU-accelerate the mgcv-compatible residual executor without changing residual semantics.

### Architecture

```text
CPU/mgcv side:
  setup semantics extraction
  formula / smooth / basis / penalty / constraint metadata
  version-pinned behavior

CUDA side:
  same-S batched solves
  batched residualization
  batched score/GCV if semantically pinned
```

### Initial supported envelope

Start with:

```text
|S| <= 2
single-penalty setups
same-S groups
Gaussian identity case used by regrXonS
```

This covers a large fraction of current requests:

```text
|S|=1 requests: 105,318
|S|=2 requests: 250,586
combined:       ~74.7% of residual requests
```

### Env gate

```bash
FASTKPC_COMPATIBLE_CUDA_RESIDUAL=1
```

### Required diagnostics

```text
mgcv_cuda_residual_supported_count
mgcv_cuda_residual_fallback_count
mgcv_cuda_residual_fallback_reason
mgcv_cuda_residual_shadow_count
mgcv_cuda_residual_decision_flip_count
mgcv_cuda_residual_max_abs_diff
mgcv_cuda_residual_max_p_diff
mgcv_cuda_residual_ms
mgcv_legacy_residual_ms
```

### Artifacts

```text
mgcv_residual_cuda_shadow_supported_v1
mgcv_residual_cuda_backend_supported_v1
```

### Gates

```text
supported cases: residual/p-value/decision parity
full 351x48: SHD=0
edge_count=110/110
n.edgetests exact=TRUE
fallback_count recorded
wall time improves over current recommended compatible route
```

### Promotion rule

Do not default-enable CUDA residual executor until:

```text
full skeleton SHD=0
no unexplained decision flips
fallback coverage acceptable
wall time improves
```

---

## Phase 5 — compatible.cuda one-call skeleton engine

### Goal

Expose a one-call skeleton engine to R:

```text
R prepares data
C++/CUDA runs complete skeleton
R receives skeleton/sepsets and continues orientation
```

### C++ host responsibilities

```text
upload data once
own GPU/C++ context
run level loop
generate CI task list
batch tasks by S / shape / backend capability
canonical p-value replay
adjacency mutation
sepset writeback
return diagnostics
```

### Native p-table replay checkpoint

The first C++ host boundary already has a native p-value table replay artifact:

```text
fastkpc/artifacts/skeleton_ptable_parity
```

Status:

```text
status: targeted host replay parity gate, not a full skeleton engine
scope:
  synthetic p-table CI values
  native C++ replay of canonical task rows
  R reference replay over the same task table

coverage:
  default p=6, max_conditioning_size=2 scenario
  explicit conditioning_size task column
  level-2 deletion
  level-2 post-delete ignored task rows

gate:
  adjacency_identical = TRUE
  sepsets_identical = TRUE
  n.edgetests identical = TRUE
  pMax max abs diff < 1e-12
```

Decision:

```text
This proves the native host replay primitive can preserve canonical deletion,
sepset, pMax, n.edgetests, and ignored-after-delete semantics beyond the
small |S|=1 smoke case. It is a useful boundary for the future one-call
compatible CUDA skeleton, but it is not yet the full C++/CUDA skeleton engine:
CI task generation, residual batching, dCov batching, and full 351x48 execution
still have to move behind the native entrypoint before Phase 5 can be promoted.
```

### Native layer planning checkpoint

The next C++ host boundary exposes the native scheduler layer planner to R:

```text
precision_make_layer_plan_native()
```

Status:

```text
status: targeted task-generation parity gate, not a full skeleton engine
scope:
  native C++ make_layer_plan()
  R reference fastkpc_batched_precision_make_layer_plan()
  levels 0, 1, and 2 on a p=6 partially pruned adjacency
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_layer_plan.R
```

Coverage:

```text
task_id
edge_x / edge_y
orientation x / y
conditioning set S
S_key
conditioning_size
conditioning_target_side
summary task counts
```

Decision:

```text
This moves canonical layer task generation behind the native boundary and
checks it against the existing R planner. Together with the p-table replay
gate, Phase 5 now has targeted native host parity for task generation and
p-value replay. Residual generation, dCov p-value generation, full level loop
ownership, and full 351x48 one-call execution still remain outside the native
entrypoint.
```

### Native p-table one-call skeleton checkpoint

The next C++ host boundary combines the native planner and replay primitive
inside a single native skeleton control entrypoint:

```text
precision_run_skeleton_ptable_native()
```

Status:

```text
status: targeted one-call host-control parity gate, not a real CI engine
scope:
  C++ owns complete graph initialization
  C++ owns skeleton level loop
  C++ generates each level with make_layer_plan()
  C++ applies synthetic p-table p-values
  C++ replays deletion / ignored-after-delete / pMax / sepsets
  R reference remains fastkpc_run_skeleton_ptable_parity()
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_ptable_native_one_call.R
```

Coverage:

```text
p = 6
max_conditioning_size = 2
level-2 deletion
level-2 ignored post-delete task rows
adjacency identical to R/native p-table reference
sepsets identical to R/native p-table reference
n.edgetests identical to R/native p-table reference
pMax max abs diff < 1e-12
global canonical task trace emitted by the one-call native entrypoint
```

Decision:

```text
This is the first one-call native skeleton-control checkpoint. It proves that
C++ can own the level loop, task generation, adjacency mutation, sepset writes,
pMax updates, and ignored-after-delete semantics for a deterministic p-table
CI oracle. It still does not execute real mgcv residuals, dCov p-values, CUDA
residual batches, or the full 351x48 compatible route, so Phase 5 remains open.
The next boundary must replace the synthetic p-table oracle with a real
compatible CI data-plane call while preserving the same native control flow.
```

### Native provider skeleton checkpoint

The next C++ host boundary replaces the synthetic p-table oracle with a
provider seam that receives native-planned task tables and returns p-values:

```text
precision_run_skeleton_provider_native()
```

Status:

```text
status: targeted one-call host-control + real legacy CI provider gate
scope:
  C++ owns complete graph initialization
  C++ owns skeleton level loop
  C++ generates each level with make_layer_plan()
  C++ calls an R p-value provider with the native task table
  provider computes legacy mgcv/regrXonS + dcc.gamma p-values in the test gate
  C++ replays deletion / ignored-after-delete / pMax / sepsets
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_provider_legacy_ci.R
```

Coverage:

```text
p = 4 real-valued data
max_conditioning_size = 1
legacy mgcv residual authority
legacy dcc.gamma CI p-values
conditional task rows exercised
adjacency identical to R reference replay over the same provider
sepsets identical to R reference replay over the same provider
n.edgetests identical to R reference replay over the same provider
pMax max abs diff < 1e-12
```

Decision:

```text
This moves Phase 5 beyond synthetic p-tables: the native one-call control loop
can now consume p-values produced by a real legacy-compatible CI provider while
retaining native ownership of planning and replay. The provider is still an R
callback and therefore not the final compatible CUDA data plane. The next
boundary must replace this R provider seam with a native C++/CUDA-compatible CI
executor that can generate p-values from data behind the same one-call skeleton
entrypoint.
```

### Native dCov0 one-call checkpoint

The next C++ host boundary removes the R p-value provider for the
unconditional level and computes p-values directly from data inside the native
entrypoint:

```text
precision_run_skeleton_dcov0_native()
```

Status:

```text
status: targeted one-call host-control + native CI data-plane gate
scope:
  C++ owns complete graph initialization
  C++ owns level-0 skeleton control
  C++ generates level 0 with make_layer_plan()
  C++ computes unconditional p-values from the data matrix
  C++ uses the existing native exact dCov gamma approximation helper
  C++ replays deletion / ignored-after-delete / pMax / sepsets
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_dcov0_one_call.R
```

Coverage:

```text
real-valued data
max_conditioning_size = 0
native exact dCov p-value generation
adjacency identical to R exact-dCov p-value replay
n.edgetests identical to R exact-dCov p-value replay
pMax max abs diff < 1e-10
at least one deletion
ignored post-delete task rows
```

Decision:

```text
This is the first one-call native skeleton checkpoint that generates CI
p-values from data without an R provider callback. It covers only
unconditional tests and uses the native exact dCov helper, not the
legacy-compatible lowrank dcov.gamma backend. Therefore it is a host/data-plane
integration checkpoint, not a promotion route for the final 351x48 compatible
engine. The next boundaries must add conditional residual generation and then
swap the exact dCov helper for the legacy-compatible C++/CUDA dcov.gamma data
plane while preserving the same one-call native control flow.
```

### Native exact-CI conditional one-call checkpoint

The next C++ host boundary extends the native data-plane checkpoint from
unconditional dCov to conditional CI tests:

```text
precision_run_skeleton_exact_ci_native()
```

Status:

```text
status: targeted one-call host-control + native residual/CI data-plane gate
scope:
  C++ owns complete graph initialization
  C++ owns skeleton level loop through max_conditioning_size = 1
  C++ generates each level with make_layer_plan()
  C++ computes unconditional exact dCov p-values from the data matrix
  C++ computes conditional residuals with native linear lm residualization
  C++ computes exact dCov p-values from those residuals
  C++ replays deletion / ignored-after-delete / pMax / sepsets
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_exact_ci_one_call.R
```

Coverage:

```text
real-valued data
max_conditioning_size = 1
native conditional residual generation exercised
native exact dCov p-value generation exercised
adjacency identical to R lm-residual + exact-dCov reference replay
sepsets identical to R lm-residual + exact-dCov reference replay
n.edgetests identical to R lm-residual + exact-dCov reference replay
pMax max abs diff < 1e-10
```

Decision:

```text
This is the first one-call native skeleton checkpoint with conditional
residual generation and p-value generation behind the native entrypoint. It is
still not legacy-compatible: the residual backend is native linear lm, and the
CI backend is native exact dCov, not mgcv/regrXonS plus legacy-compatible
dcov.gamma. The next Phase 5 boundary must replace native-linear-lm residuals
with the mgcv-compatible residual executor/provider path, and then replace the
exact dCov helper with the legacy-compatible dCov gamma C++/CUDA data plane.
```

### Native residual-provider + exact-dCov checkpoint

The next C++ host boundary replaces native linear residualization with a
residual provider seam that can supply legacy mgcv/regrXonS residuals while
keeping p-value generation inside native C++:

```text
precision_run_skeleton_residual_provider_native()
```

Status:

```text
status: targeted one-call host-control + mgcv residual-provider + native CI gate
scope:
  C++ owns complete graph initialization
  C++ owns skeleton level loop through max_conditioning_size = 1
  C++ generates each level with make_layer_plan()
  C++ enumerates unique target|S residual requests per level
  R residual provider supplies legacy mgcv/regrXonS residual vectors
  C++ computes exact dCov p-values from those residuals
  C++ replays deletion / ignored-after-delete / pMax / sepsets
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_residual_provider_exact_dcov.R
```

Coverage:

```text
real-valued data
max_conditioning_size = 1
legacy mgcv/regrXonS residual authority via provider
unique target|S residual request table exercised
native exact dCov p-value generation exercised
adjacency identical to R residual-provider + exact-dCov reference replay
sepsets identical to R residual-provider + exact-dCov reference replay
n.edgetests identical to R residual-provider + exact-dCov reference replay
pMax max abs diff < 1e-10
```

Decision:

```text
This restores mgcv-compatible residual authority behind the native one-call
skeleton boundary while keeping p-value generation in native C++. It is still
not the final compatible route because the CI backend is native exact dCov, not
the legacy-compatible lowrank dcov.gamma C++/CUDA backend. The next Phase 5
boundary must replace exact dCov with the legacy-compatible dCov gamma data
plane while preserving native skeleton control and the residual-provider
request contract.
```

### Native residual-provider + legacy-dCov checkpoint

The next C++ host boundary replaces the exact dCov helper with the
legacy-compatible C++ `dcov.gamma` implementation while keeping the same native
skeleton control loop and residual-provider request contract:

```text
precision_run_skeleton_residual_provider_legacy_dcov_native()
```

Status:

```text
status: targeted one-call host-control + mgcv residual-provider + legacy CI gate
scope:
  C++ owns complete graph initialization
  C++ owns skeleton level loop through max_conditioning_size = 1
  C++ generates each level with make_layer_plan()
  C++ enumerates unique target|S residual requests per level
  R residual provider supplies legacy mgcv/regrXonS residual vectors
  C++ computes legacy-compatible lowrank dcov.gamma p-values from those vectors
  C++ replays deletion / ignored-after-delete / pMax / sepsets
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_residual_provider_legacy_dcov.R
```

Coverage:

```text
real-valued data
max_conditioning_size = 1
legacy mgcv/regrXonS residual authority via provider
unique target|S residual request table exercised
shared C++ legacy dcov.gamma oracle used by R and native paths
Spectra lowrank route exercised through FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK
adjacency identical to R residual-provider + legacy-dCov reference replay
sepsets identical to R residual-provider + legacy-dCov reference replay
n.edgetests identical to R residual-provider + legacy-dCov reference replay
pMax max abs diff < 1e-10
```

Decision:

```text
This closes the targeted Phase 5 host/data-plane boundary that still used
native exact dCov: native skeleton control can now consume legacy mgcv residuals
and compute legacy-compatible C++ dcov.gamma p-values behind one native
entrypoint. It is still not the final compatible.cuda route because residual
authority remains an R provider seam, execution is only a targeted
max_conditioning_size = 1 gate, and full 351x48 one-call execution has not yet
passed SHD=0 / n.edgetests-exact gates. The next boundary must move residual
generation or residual batches further behind the native entrypoint while
preserving this legacy dcov.gamma CI data-plane contract.
```

Native legacy dCov level-batch checkpoint:

```text
env gate: FASTKPC_NATIVE_LEGACY_DCOV_BATCH=level
R-facing one-call option:
  precision_run_skeleton_legacy_mgcv_legacy_dcov_native(dcov_batch = "level")
scope: precision_run_skeleton_residual_provider_legacy_dcov_native()

behavior:
  native skeleton control remains unchanged
  R residual provider remains the legacy mgcv/regrXonS authority
  C++ batches each nonempty skeleton level through the shared legacy
    dcov.gamma batch oracle
  p-values are replayed by the same native canonical state machine
  the R-facing one-call wrapper scopes the env toggle during the native call
    and restores the caller environment afterward
  default dcov_batch = "env" preserves existing env-gated behavior

diagnostics:
  legacy_dcov_native_batch_enabled
  legacy_dcov_native_batch_count
  legacy_dcov_native_batch_pair_count
  legacy_dcov_native_batch_ms
  legacy_dcov_native_batch_workspace_reuse_count
  legacy_dcov_native_batch_distance_workspace_reuse_count
  legacy_dcov_native_batch_statistic_moment_workspace_reuse_count
  legacy_dcov_native_batch_lowrank_output_workspace_reuse_count
  legacy_dcov_native_batch_lowrank_eig_workspace_reuse_count
  legacy_dcov_native_batch_oracle_column_copy_count
  legacy_dcov_native_batch_column_materialize_count

gate:
  Rscript fastkpc/tests/test_skeleton_native_residual_provider_legacy_dcov.R
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R

status:
  This moves the level-batch dCov data-plane boundary into the native
  one-call skeleton checkpoint while preserving legacy dCov authority and
  canonical replay. It remains experimental: the native path still uses an R
  residual provider seam, materializes level batch matrices on the host, and
  has not passed full 351x48 SHD=0 / wall-time gates.
```

Native level residual batch-provider contract checkpoint:

```text
scope: precision_run_skeleton_residual_provider_legacy_dcov_native()
wrapper: precision_run_skeleton_legacy_mgcv_legacy_dcov_native()

behavior:
  native residual-provider entrypoint accepts both legacy raw matrix
    responses and structured level residual batch responses
  structured response contract:
    list(
      residuals = <n x request_count matrix>,
      contract = "level-residual-matrix-v1",
      backend = "legacy-mgcv-regrXonS-level-batch",
      level = <int>,
      request_count = <int>,
      n = <int>
    )
  R-facing one-call wrapper now uses the structured legacy mgcv batch
    provider internally
  residual vectors are still generated by legacy mgcv/regrXonS
  native skeleton control, legacy dcov.gamma authority, and canonical replay
    remain unchanged

summary fields:
  residual_provider_contract
  residual_provider_response_mode
  residual_provider_response_backend
  residual_provider_batch_count
  residual_provider_batch_max_requests
  residual_provider_batch_mean_requests
  residual_provider_matrix_cell_count

gate:
  Rscript fastkpc/tests/test_skeleton_native_residual_provider_legacy_dcov.R
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R

status:
  Boundary metadata and contract checkpoint only. It makes the native
  one-call residual batch boundary explicit so a future native/CUDA residual
  executor can replace the R provider without changing skeleton replay. It is
  not a full 351x48 promotion route and does not change the current
  recommended compatible acceleration environment.
```

Native guarded C++ residual-provider backend checkpoint:

```text
env gate: FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
scope: precision_run_skeleton_legacy_mgcv_legacy_dcov_native()

behavior:
  the hidden structured level residual batch provider now honors the legacy
    guarded C++ residual backend env gate
  default backend remains legacy R regrXonS
  cpp_guarded backend still uses mgcv::gam to select smoothing parameters and
    extract the compatible setup, then replays the fixed-sp solve through the
    existing native C++ residual solver when inside the guarded envelope
  fallback remains legacy R regrXonS for unsupported or high-condition cases
  native skeleton control, legacy dcov.gamma authority, and canonical replay
    remain unchanged

summary fields:
  residual_provider_mgcv_backend
  residual_provider_mgcv_cpp_backend_enabled
  residual_provider_mgcv_cpp_backend_count
  residual_provider_mgcv_cpp_backend_native_count
  residual_provider_mgcv_cpp_backend_fallback_count
  residual_provider_mgcv_cpp_backend_error_count
  residual_provider_mgcv_cpp_backend_high_condition_fallback_count
  residual_provider_mgcv_cpp_backend_outside_envelope_fallback_count
  residual_provider_mgcv_cpp_backend_ms
  residual_provider_mgcv_cpp_backend_native_solve_ms

gate:
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R

status:
  Boundary integration checkpoint. It moves the existing guarded C++ residual
  executor behind the native one-call residual batch-provider contract, but it
  is not a recommended full 351x48 route: mgcv setup/GCV authority is still R
  mgcv, the backend remains env-gated, and full compatible.cuda promotion
  still requires SHD=0 / n.edgetests-exact / wall-time gates.
```

### R-facing legacy-mgcv + legacy-dCov one-call checkpoint

The next API boundary hides the residual-provider callback behind a single
R-facing native skeleton call:

```text
precision_run_skeleton_legacy_mgcv_legacy_dcov_native()
```

Status:

```text
status: targeted one-call API wrapper over native skeleton + legacy CI gate
scope:
  R caller passes data / alpha / max_conditioning_size / dCov parameters
  wrapper constructs the legacy mgcv/regrXonS residual provider
  native entrypoint owns skeleton level loop, residual request enumeration,
    legacy-compatible C++ dcov.gamma p-values, canonical replay, and sepsets
  summary records entrypoint = legacy-mgcv-legacy-dcov-native
```

Gate:

```text
Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R
```

Coverage:

```text
real-valued data
max_conditioning_size = 1
legacy mgcv/regrXonS residual authority via hidden provider
native legacy dcov.gamma CI data plane
adjacency identical to explicit residual-provider native route
sepsets identical to explicit residual-provider native route
n.edgetests identical to explicit residual-provider native route
pMax max abs diff < 1e-12
```

Decision:

```text
This gives Phase 5 a single R-facing call for the current native-compatible
legacy residual + legacy dCov gate. It improves the final API shape but is not
yet the final compatible.cuda engine: internally, residual generation still
uses an R provider callback, execution is still a small max_conditioning_size =
1 gate, and full 351x48 one-call execution still has not passed SHD=0 /
n.edgetests-exact / wall-time gates. The next boundary should either validate
this one-call wrapper on a real 351-row subset or move residual batching/setup
further behind the native entrypoint.
```

Experimental compatible CUDA facade checkpoint:

```text
fastkpc_compatible_cuda_skeleton(data, alpha, labels = NULL, options = list(...))

status: experimental R-facing facade over the current native one-call wrapper
scope:
  accepts the proposed final API shape
  requires options$max_conditioning_size explicitly
  passes options$index, options$numCol, options$trace_level, and
    options$dcov_batch through to the native one-call wrapper
  scopes options$mgcv_residual_backend and guarded backend envelope options
    around the native one-call wrapper without leaking caller env
  applies labels to adjacency and pMax
  records compatible_cuda_facade = TRUE
  records compatible_cuda_entrypoint = fastkpc-compatible-cuda-skeleton
  records compatible_cuda_route = legacy-mgcv-provider-native-legacy-dcov
  records compatible_cuda_residual_authority = legacy-mgcv-regrXonS-provider
  records compatible_cuda_ci_authority = native-legacy-dcov.gamma
  records compatible_cuda_mgcv_residual_backend
  preserves native_entrypoint = legacy-mgcv-legacy-dcov-native

gate:
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_one_call.R

status note:
  This is an API-shape checkpoint, not a completed compatible.cuda engine.
  The facade still routes to the R hidden-provider legacy mgcv residual path
  and native legacy dCov data plane. Full 351x48 SHD=0 / wall-time gates and
  native/CUDA residual generation remain open.
```

### Real 351-row subset one-call checkpoint

The next gate validates the R-facing one-call wrapper on the real 351-row
fixture, using the established 8 hot-column subset plus the 12-column hot
subset used by earlier residual-backend gates:

```text
precision_run_skeleton_legacy_mgcv_legacy_dcov_native()
```

Status:

```text
status: opt-in real-data subset gate for one-call API wrapper
scope:
  data: real 351x48 fixture
  hot8 columns:  1,2,3,4,5,6,9,12
  hot12 columns: 1,2,3,4,5,6,9,12,15,16,17,18
  n / p: 351 / 8 and 351 / 12
  max_conditioning_size = 2
  wrapper hides the legacy mgcv residual provider
  native entrypoint owns skeleton loop, residual request enumeration,
    legacy-compatible C++ dcov.gamma p-values, canonical replay, and sepsets
```

Gate:

```text
FASTKPC_RUN_REAL_SUBSET_TESTS=1 \
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_real_subset.R
```

For quicker local iteration, the gate can be narrowed to one case:

```bash
FASTKPC_RUN_REAL_SUBSET_TESTS=1 \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_CASES=hot8 \
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_real_subset.R
```

To validate the experimental compatible CUDA facade on the same real subset:

```bash
FASTKPC_RUN_REAL_SUBSET_TESTS=1 \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_ROUTE=facade \
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_real_subset.R
```

To validate the guarded C++ residual backend through the facade on the same
real subset:

```bash
FASTKPC_RUN_REAL_SUBSET_TESTS=1 \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_ROUTE=facade \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_MGCV_BACKEND=cpp_guarded \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_MGCV_NATIVE_S_SIZE_LIMIT=2 \
FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_MGCV_CONDITION_THRESHOLD=1e300 \
  Rscript fastkpc/tests/test_skeleton_native_legacy_mgcv_legacy_dcov_real_subset.R
```

For quicker facade iteration, the same gate can be narrowed with
`FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_CASES=hot8` or
`FASTKPC_NATIVE_LEGACY_ONE_CALL_REAL_SUBSET_CASES=hot12`.

Coverage:

```text
real fixture rows, not synthetic data
nontrivial residual-provider traffic on both hot8 and hot12 subsets
adjacency identical to explicit residual-provider native route
sepsets identical to explicit residual-provider native route
n.edgetests identical to explicit residual-provider native route
pMax max abs diff < 1e-12 on finite entries
legacy dCov native task count preserved
summary records residual_provider_hidden = TRUE
```

Facade hot8 / hot12 checkpoint:

```text
route = facade
scenario = hot8
p = 8
residual provider requests = 224
adjacency identical to explicit provider route = TRUE
n.edgetests identical to explicit provider route = TRUE
summary compatible_cuda_facade = TRUE
summary compatible_cuda_route = legacy-mgcv-provider-native-legacy-dcov
summary legacy_dcov_native_batch_enabled = TRUE

scenario = hot12
p = 12
residual provider requests = 792
adjacency identical to explicit provider route = TRUE
n.edgetests identical to explicit provider route = TRUE
summary compatible_cuda_facade = TRUE
summary compatible_cuda_route = legacy-mgcv-provider-native-legacy-dcov
summary legacy_dcov_native_batch_enabled = TRUE

status: real 351-row subset facade gate pass for hot8 and hot12; still not full 351x48
```

Facade guarded C++ residual hot8 / hot12 checkpoint:

```text
route = facade
mgcv_backend = cpp_guarded
scenario = hot8
p = 8
residual provider requests = 224
reference residual provider = legacy R regrXonS
candidate residual provider = legacy-mgcv-cpp-guarded-level-batch
adjacency identical to explicit R-provider route = TRUE
n.edgetests identical to explicit R-provider route = TRUE
summary compatible_cuda_residual_authority = legacy-mgcv-cpp-guarded-provider
summary residual_provider_mgcv_cpp_backend_enabled = TRUE
summary residual_provider_mgcv_cpp_backend_native_count > 0
summary residual_provider_mgcv_cpp_backend_error_count = 0

scenario = hot12
p = 12
residual provider requests = 792
reference residual provider = legacy R regrXonS
candidate residual provider = legacy-mgcv-cpp-guarded-level-batch
adjacency identical to explicit R-provider route = TRUE
n.edgetests identical to explicit R-provider route = TRUE
summary compatible_cuda_residual_authority = legacy-mgcv-cpp-guarded-provider
summary residual_provider_mgcv_cpp_backend_enabled = TRUE
summary residual_provider_mgcv_cpp_backend_native_count > 0
summary residual_provider_mgcv_cpp_backend_error_count = 0

status: real 351-row hot8/hot12 guarded-residual facade gate pass; still not full 351x48
```

Decision:

```text
This extends the R-facing one-call wrapper from a synthetic smoke gate to a
real 351-row subset pair while preserving native skeleton replay and the
legacy-compatible C++ dcov.gamma data plane. It is still not a promotion route:
the residual provider remains an R seam, the gate is limited to 8- and
12-column subsets for both the default residual provider and the guarded C++
residual provider, and full 351x48 SHD=0 / n.edgetests-exact / wall-time gates
remain open.
```

### CUDA responsibilities

```text
mgcv-compatible residual batches
legacy-compatible dCov batches
p-value vector generation
```

### Do not require device-side graph mutation initially

Graph control can remain on C++ host. This is the best tradeoff for SHD=0.

### Proposed API

```r
fastkpc_compatible_cuda_skeleton(
  data,
  alpha,
  labels = NULL,
  options = list(
    max_conditioning_size = <int>,
    index = 1,
    numCol = floor(nrow(data) / 10),
    trace_level = "summary",
    dcov_batch = "env",
    mgcv_residual_backend = c("env", "r", "cpp_guarded"),
    mgcv_residual_backend_native_s_size_limit = NULL,
    mgcv_residual_backend_condition_threshold = NULL
  )
)
```

Current implementation status:

```text
An experimental R facade with this name exists and forwards to the current
legacy-mgcv + legacy-dCov native one-call checkpoint. It is not yet the final
C++/CUDA engine because residual generation remains behind a hidden structured
provider, although the facade can now explicitly scope the guarded C++
fixed-sp residual replay backend for candidate artifacts. Full 351x48
promotion gates remain open.
```

Internal C++ entrypoint:

```cpp
fastkpc_compatible_cuda_skeleton_run(...)
```

### Artifact

```text
compatible_cuda_skeleton_full_351x48_v1
```

Artifact runner:

```text
fastkpc/R/compatible_cuda_skeleton_artifact.R
```

Current status:

```text
status: artifact runner scaffold implemented with scoped residual backend metadata
scope:
  runs fastkpc_compatible_cuda_skeleton()
  runs the explicit residual-provider native legacy-dCov route as reference
  writes summary.csv, result.rds, and summary.md
  records SHD, edge counts, n.edgetests exactness, pMax max abs diff,
    residual-provider request counts, native legacy dCov counts,
    compatible CUDA route metadata, scoped mgcv residual backend metadata,
    provider C++ residual backend counters, and elapsed seconds
targeted gate:
  Rscript fastkpc/tests/test_compatible_cuda_skeleton_artifact.R
```

Full 351x48 command:

```bash
Rscript -e 'source("fastkpc/R/compatible_cuda_skeleton_artifact.R"); fastkpc_run_compatible_cuda_skeleton_artifact(output_dir = "fastkpc/artifacts/compatible_cuda_skeleton_full_351x48_v1", artifact_name = "compatible_cuda_skeleton_full_351x48_v1", alpha = 0.1, max_conditioning_size = 46L, dcov_batch = "level", reference_result_path = "fastkpc/artifacts/legacy_mgcv_residual_cache_s_affinity_v1/compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds", expected_edge_count = 110L, expected_n_edgetests = c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L))'
```

Guarded C++ residual-provider candidate command:

```bash
Rscript -e 'source("fastkpc/R/compatible_cuda_skeleton_artifact.R"); fastkpc_run_compatible_cuda_skeleton_artifact(output_dir = "fastkpc/artifacts/compatible_cuda_skeleton_cpp_guarded_residual_full_351x48_v1", artifact_name = "compatible_cuda_skeleton_cpp_guarded_residual_full_351x48_v1", alpha = 0.1, max_conditioning_size = 46L, dcov_batch = "level", mgcv_residual_backend = "cpp_guarded", mgcv_residual_backend_native_s_size_limit = Inf, mgcv_residual_backend_condition_threshold = 1e12, reference_result_path = "fastkpc/artifacts/legacy_mgcv_residual_cache_s_affinity_v1/compatible_legacy_cpp_dcov_mgcv_cache_s_affinity_result.rds", expected_edge_count = 110L, expected_n_edgetests = c(2213L, 52659L, 125293L, 40694L, 13293L, 5422L, 835L, 80L))'
```

Additional artifact fields:

```text
mgcv_residual_backend
mgcv_residual_backend_native_s_size_limit
mgcv_residual_backend_condition_threshold
residual_provider_response_backend
residual_provider_mgcv_backend
residual_provider_mgcv_cpp_backend_enabled
residual_provider_mgcv_cpp_backend_count
residual_provider_mgcv_cpp_backend_native_count
residual_provider_mgcv_cpp_backend_fallback_count
residual_provider_mgcv_cpp_backend_error_count
residual_provider_mgcv_cpp_backend_ms
residual_provider_mgcv_cpp_backend_native_solve_ms
```

Full 351x48 attempt status:

```text
status: attempted, no completed artifact
reference: loaded from legacy_mgcv_residual_cache_s_affinity_v1 result.rds
reference_source: rds
candidate: fastkpc_compatible_cuda_skeleton() facade
candidate route: legacy-mgcv-provider-native-legacy-dcov
attempt result:
  candidate exceeded the current recommended S-affinity wall-time baseline
  before producing a completed summary
  process remained CPU-bound and was stopped after the performance gate was
  already failed
artifact files written: none

decision:
  current facade is a useful API-shape and subset-correctness checkpoint, but
  it is not a viable full 351x48 promotion route. It still runs the hidden R
  legacy mgcv residual-provider path in a single candidate pass. The artifact
  runner can now explicitly scope and measure the guarded C++ residual-provider
  backend, but no full 351x48 guarded-residual candidate has yet passed the
  SHD=0 / n.edgetests-exact / wall-time promotion gate. Phase 5 must keep
  moving residual generation/batching further behind the native/CUDA boundary
  before promotion.
```

### Gate

```text
edge_count = 110 / 110
SHD = 0
n.edgetests exact = TRUE
sepset/deletion trace exact where recorded
cpu_fallback_count reported
wall time improves over best compatible CPU/C++ route
```

---

## Phase 6 — Hardening and promotion

### Goal

Move from env-gated experimental route to recommended compatible.cuda route.

### Required artifacts

```text
compatible_cuda_skeleton_full_351x48_v1
compatible_cuda_skeleton_reproducibility_v1
compatible_cuda_skeleton_near_alpha_audit_v1
compatible_cuda_fallback_coverage_v1
```

### Required checks

```text
SHD = 0
edge_count = 110 / 110
n.edgetests exact = TRUE
no decision flips in shadow-supported cases
fallback reasons documented
no route drift across repeated runs
```

### Documentation updates

Update README / docs with:

```text
precision="fast": approximate fast backend, not skeleton-compatible guarantee
precision="compatible": legacy CPU-compatible baseline
precision="compatible.cuda": GPU-accelerated legacy-compatible backend, if gates pass
```

---

## 6. Decision rules for Codex

### 6.1 When to commit

Commit only if:

```text
all relevant tests pass
git diff --check passes
route is env-gated unless already accepted
correctness gates pass for required scope
artifact supports the performance claim
```

### 6.2 When not to commit

Do not commit perf code if:

```text
worker-sum improves but full wall time regresses
SHD != 0
n.edgetests changes unexpectedly
route changes default behavior without explicit approval
artifact contradicts expected win
```

Preserve negative experiments as:

```text
/tmp/*.patch
local artifact
summary in notes if useful
```

### 6.3 Always separate phases

Do not combine:

```text
diagnostics + production switch
shadow + default backend
C++ replica + CUDA backend
mgcv semantic change + scheduler optimization
```

Each phase should have a single clear question and artifact.

---

## 7. Standard validation commands

Use relevant subsets for small changes, but full-route promotion requires full 351x48 artifacts.

Common checks:

```bash
git diff --check

Rscript fastkpc/tests/test_precision_compatible_legacy_parallel_runtime_breakdown.R
Rscript fastkpc/tests/test_precision_compatible_legacy_dcov_cpp_backend.R
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_shadow_route.R
Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_breakdown.R
Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_cache.R
Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_affinity.R

FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_fast_cuda_stage_breakdown.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_dcc_gamma_cuda_parity_artifact.R
```

When changing CUDA code:

```bash
bash fastkpc/tools/build_cuda_native.sh
```

When changing basis/mgcv-related code:

```bash
Rscript fastkpc/tests/test_fastspline_basis.R
```

---

## 8. Current immediate next actions

### 8.1 Keep recommended route fixed

The current recommended route remains:

```bash
FASTKPC_LEGACY_DCOV_GAMMA_BACKEND=cpp
FASTKPC_LEGACY_DCOV_GAMMA_CPP_LOW_RANK=spectra
FASTKPC_LEGACY_MGCV_RESIDUAL_CACHE=1
FASTKPC_LEGACY_MGCV_RESIDUAL_AFFINITY=s
```

Do not promote `FASTKPC_LEGACY_MGCV_RESIDUAL_PREFETCH=level`, `target_s`, or hybrid affinity based on worker-sum reductions. Full wall time regressed.

### 8.2 Expand mgcv residual oracle trace

```text
fastkpc/artifacts/mgcv_residual_oracle_v1
```

Current artifact exists and is sourced from the full 351x48 skeleton deletion log.
The generator and gate now add:

```text
rank-deficient / collinear / near-constant diagnostics
formula route and richer mgcv metadata
lpmatrix rank / condition metadata
downstream legacy dCov alpha, p-value, log-alpha distance, and alpha decision
summary-level risk counts
```

Gate:

```bash
Rscript fastkpc/tests/test_mgcv_residual_oracle_trace.R
```

Default artifact generation:

```bash
Rscript fastkpc/tools/run_mgcv_residual_oracle_trace.R
```

Current generated summary:

```text
case_count:                         8
success_count:                      8
error_count:                        0
full_skeleton_source_count:         8
near_alpha_source_count:            7
rank_deficient_case_count:          0
lpmatrix_rank_deficient_case_count: 0
near_constant_case_count:           0
max_conditioning_condition_kappa:   3.883344
min_dcov_log_alpha_distance:        0.0001601918
```

Still open for future expansion:

```text
add concrete rank-deficient / collinear / near-constant examples if discovered
add more late sparse levels if they expose new mgcv formula/setup behavior
```

### 8.3 Start mgcv replay executable spec

Use the oracle trace to build the narrow executable spec for the actual legacy regrXonS/kpc subset:

```text
residual parity or decision parity against oracle cases
legacy dCov p-values must not flip decisions
```

Current status:

```text
fastkpc/artifacts/mgcv_residual_replay_spec_v1 exists
8 / 8 residual pairs match
8 / 8 dCov p-values match
decision_flip_count = 0
rank_deficient_case_count = 0
lpmatrix_rank_deficient_case_count = 0
near_constant_case_count = 0
min_dcov_log_alpha_distance_oracle = 0.0001601918
min_dcov_log_alpha_distance_replay = 0.0001601918

fastkpc/artifacts/mgcv_residual_cpp_shadow_v1 exists
8 / 8 captured setup C++ replays supported
decision_flip_count = 0

fastkpc/artifacts/mgcv_residual_setup_shadow_v1 exists
8 / 8 extracted fixed-sp setup replays supported
decision_flip_count = 0

fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_v1 exists
8 / 8 extracted setup native C++ numeric replays supported
decision_flip_count = 0

fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_expanded_v1 exists
28 / 28 expanded extracted setups supported
27 / 28 residual pairs match under strict tolerance
decision_flip_count = 0
status: diagnostic mismatch present; not promotable

fastkpc/artifacts/mgcv_residual_cpp_numeric_drift_isolation_v1 exists
1 / 1 mismatch case isolated
C++ normal solve matches R normal solve
mgcv C_magic fixed-sp replay matches full mgcv oracle
drift layer = normal_equation_vs_mgcv_magic

fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_expanded_v1 exists
28 / 28 expanded extracted setups supported
28 / 28 residual pairs match under strict tolerance
28 / 28 dCov p-values match under strict tolerance
decision_flip_count = 0
fallback_count = 19 high-condition targets
status: guarded shadow pass; still not production

fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_expanded_wide_v1 exists
116 / 116 expanded extracted setups supported
116 / 116 residual pairs match under strict tolerance
116 / 116 dCov p-values match under strict tolerance
decision_flip_count = 0
fallback_count = 52 high-condition targets
|S|<=2 fallback_count = 0
|S|>=3 fallback_count = 52
status: wider guarded shadow pass; still not production

fastkpc/artifacts/mgcv_residual_cpp_numeric_shadow_guarded_s2_envelope_wide_v1 exists
116 / 116 expanded extracted setups supported
116 / 116 residual pairs match under strict tolerance
116 / 116 dCov p-values match under strict tolerance
decision_flip_count = 0
native_s_size_limit = 2
cpp_guarded_count = 128 native |S|<=2 targets
fallback_count = 104 outside-envelope |S|>=3 targets
high_condition_fallback_count = 0
status: strict supported-envelope shadow pass; still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_synthetic_v1 exists
full legacy-parallel skeleton shadow
adjacency identical = TRUE
n.edgetests identical = TRUE
38 / 38 residual shadow targets matched
native_count = 38
fallback_count = 0
error_count = 0
residual_mismatch_count = 0
status: synthetic full-route shadow pass; still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_real_subset_v1 exists
real 351x48 fixture, 8 hot-column subset
adjacency identical = TRUE
n.edgetests identical = TRUE
1262 / 1262 residual shadow targets matched
native_count = 1262
fallback_count = 0
error_count = 0
residual_mismatch_count = 0
status: real subset full-route shadow pass; still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_real_subset_deep_v1 exists
real 351x48 fixture, 12 hot-column subset, max_conditioning_size = 3
adjacency identical = TRUE
n.edgetests identical = TRUE
5380 / 5380 residual shadow targets matched
native_count = 4894
fallback_count = 486
outside_envelope_fallback_count = 486
error_count = 0
residual_mismatch_count = 0
status: deep real subset full-route shadow pass with |S|>2 fallback traffic;
still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_shadow_full_351x48_v1 exists
real 351x48 fixture, full compatible skeleton shadow
adjacency identical = TRUE
n.edgetests identical = TRUE
edge_count = 110
273284 / 273284 cache-miss residual shadow targets matched
native_count = 157122
fallback_count = 116162
outside_envelope_fallback_count = 116162
error_count = 0
residual_mismatch_count = 0
dCov C++ backend count = 239404
dCov C++ backend errors/fallbacks = 0 / 0
Spectra converged/failed = 478808 / 0
status: full 351x48 guarded residual shadow pass; still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_real_subset_deep_v1 exists
real 351x48 fixture, 12 hot-column subset, max_conditioning_size = 3
env gate = FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
adjacency identical = TRUE
n.edgetests identical = TRUE
5380 / 5380 residual backend targets covered
native_count = 4894
r_fallback_count = 486
fallback_count = 486
outside_envelope_fallback_count = 486
error_count = 0
baseline_elapsed_sec = 90.126
backend_elapsed_sec = 73.678
status: real subset env-gated backend prototype pass; still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_full_351x48_v1 exists
real 351x48 fixture, full compatible skeleton backend prototype
env gate = FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
adjacency identical = TRUE
n.edgetests identical = TRUE
edge_count = 110
273284 / 273284 cache-miss residual backend targets covered
native_count = 157122
r_fallback_count = 116162
fallback_count = 116162
outside_envelope_fallback_count = 116162
error_count = 0
dCov C++ backend count = 239404
dCov C++ backend errors/fallbacks = 0 / 0
Spectra converged/failed = 478808 / 0
baseline_elapsed_sec = 898.110
backend_elapsed_sec = 1067.308
elapsed_speedup = 0.841x
status: full 351x48 backend prototype correctness pass but performance fail;
not recommended and still not production

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_timing_split_subset_v1 exists
real 351x48 fixture, 12 hot-column subset, backend timing diagnostic
backend targets = 5380
native / fallback targets = 4894 / 486
backend_worker_ms = 251446
input_formula_setup_ms = 98314
mgcv_gam_fit_ms = 65368
setup_extract_ms = 53598
native_fixed_sp_solve_ms = 2919
fallback_regrXonS_ms = 28691
status: regression attributed to repeated per-target setup/extraction, not
native numeric solve

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_reuse_potential_subset_v1 exists
real 351x48 fixture, 12 hot-column subset, same-S native reuse potential
backend / native / fallback targets = 5380 / 4894 / 486
native same-S groups = 78
native same-S targets = 4894
native same-S max targets = 182
native same-S mean targets = 62.74359
native same-S reuse opportunity = 4816
native same-S setup reuse ratio = 0.9840621
status: strong same-S setup reuse opportunity; proceed to prototype

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_timing_potential_subset_v1 exists
real 351x48 fixture, 12 hot-column subset, same-S setup timing potential
backend / native / fallback / error targets = 2756 / 2297 / 459 / 0
same-S groups / native targets / reuse ratio = 78 / 2297 / 0.9660427
same-S+setup groups / reuse ratio = 78 / 0.9660427
same-S+selected-sp groups / reuse ratio = 765 / 0.6669569
input_setup_ms = 6378
mgcv_gam_fit_ms = 34388
setup_extract_ms = 28313
condition_ms = 1073
native_solve_ms = 2711
same-S setup structure potential saved ms = 34865
same-S+sp native solve potential saved ms = 2408
gam_fit_preserved_ms = 34388
status: same-S target-independent setup reuse has measurable potential, but
per-target mgcv::gam / selected-sp selection remains preserved and cannot be
skipped by an S-only setup cache

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_subset_v1 exists
real 351x48 fixture, 12 hot-column subset, env-gated same-S setup provider
adjacency_identical = TRUE
n.edgetests_identical = TRUE
baseline/provider n.edgetests = 131,994,1453,243 / 131,994,1453,243
baseline/provider edge_count = 20 / 20
baseline_elapsed_sec = 54.676
provider_elapsed_sec = 16.793
backend targets = 3488
native targets = 3029
fallback targets = 459
provider_error_count = 0
setup_provider groups / targets / templates / reuse = 599 / 3029 / 599 / 2430
baseline/provider setup_extract_ms = 38712 / 8283
baseline/provider gam_fit_ms = 46113 / 46753
status: subset correctness and wall-time pass; still experimental because the
same-S setup provider is running through the same-S prefill vehicle and needs
the full 351x48 gate before promotion

fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_full_351x48_v1 exists
real 351x48 fixture, full compatible skeleton, env-gated same-S setup provider
adjacency_identical = TRUE
SHD = 0
n.edgetests_identical = TRUE
baseline/provider n.edgetests = 2213,52659,125293,40694,13293,5422,835,80 /
  2213,52659,125293,40694,13293,5422,835,80
baseline/provider edge_count = 110 / 110
baseline_elapsed_sec = 899.461
provider_elapsed_sec = 1152.628
elapsed_speedup = 0.780x
baseline/provider residual_worker_ms = 9010282 / 14602256
baseline/provider mgcv_fit_count = 273284 / 402019
provider backend / native / fallback / error targets = 402019 / 285857 /
  116162 / 0
setup_provider groups / targets / templates / reuse = 22585 / 285857 /
  22585 / 263272
setup_provider_error_count = 0
prefill_unused_count = 128735
dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
Spectra count / converged / failed = 478808 / 478808 / 0
status: full correctness pass but wall-time and residual worker-ms fail; do not
promote. The current prefill vehicle overcomputes 128735 unused residual keys.
```

Gate:

```bash
Rscript fastkpc/tests/test_mgcv_residual_replay_spec.R
```

Default artifact generation:

```bash
Rscript fastkpc/tools/run_mgcv_residual_replay_spec.R
```

Next Phase 3 step:

```text
do not promote FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded. The regression
is now attributed to repeated per-target setup/extraction, not the native
fixed-sp numeric solve. Same-S diagnostics show strong target-independent setup
reuse opportunity on the deep real subset, but selected smoothing parameters
are target-specific. An env-gated same-S target-independent setup provider
prototype exists and passes the 12-column subset gate, but the full 351x48 gate
fails wall time because it is currently attached to the same-S prefill vehicle.
Do not promote FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=1.
```

The consumed-key/on-demand pair-local prototype has now been run:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_consumed_keys_v1

correctness: passed
prefill overcompute: fixed
SHD = 0
edge_count = 110 / 110
n.edgetests exact = TRUE
baseline_elapsed_sec = 909.418
provider_elapsed_sec = 1075.924
baseline/provider mgcv_fit_count = 273284 / 273284
prefill_unused_count = 0
wall time: failed
status: experimental only; not recommended
```

The next attempt should not be another pair-local wrapper around
`fastkpc_legacy_run_mgcv_residual_pair()`. It needs to group actual consumed
miss keys across a worker chunk or skeleton level by S, then call the setup
provider once per consumed same-S group. If that still fails wall time, move to
same-S setup extraction for a C++/CUDA numeric executor rather than further
R-level task scheduling.

The actual consumed miss-key grouping diagnostic has now been run:

```text
fastkpc/artifacts/legacy_mgcv_residual_cache_miss_key_grouping_v1

route: current recommended S-affinity route
edge_count = 110
n.edgetests exact = TRUE
elapsed_sec = 899.077
cache miss keys = 273284
miss same-S groups = 8634
miss same-S reuse opportunity = 264650
miss same-S reuse ratio = 0.968406
status: actual consumed miss stream has strong same-S grouping opportunity
```

Next Phase 3 implementation target:

```text
perf/exp: batch actual legacy mgcv cache misses by S inside compatible workers

requirements:
  group actual residual cache misses by S across a worker chunk or level
  fit only consumed target|S keys
  preserve per-target mgcv::gam selected-sp authority
  preserve parent canonical replay
  keep default and recommended routes unchanged until full gate passes

gate:
  edge_count = 110 / 110
  SHD = 0
  n.edgetests exact = TRUE
  elapsed < current recommended S-affinity route
```

Implementation checkpoint:

```text
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=chunk

status: implemented as env-gated worker-chunk consumed-miss same-S batching
scope: compatible legacy dcc.gamma residual cache + cpp_guarded residual
       backend + affinity worker chunks
behavior:
  each worker chunk advances edge-local CI states round by round
  only currently reached CI tests contribute residual miss keys
  missing target|S keys are grouped by S and computed through the same-S setup
    provider before the existing CI path consumes worker-local cache hits
  singleton misses remain on the existing pair path
  parent level replay remains by canonical edge index

targeted tests:
  Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_cpp_backend.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_affinity.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_cpp_shadow.R
  git diff --check

full 351x48 gate:
  artifact: fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_setup_provider_chunk_v1
  correctness: passed
  edge_count = 110 / 110
  adjacency_identical = TRUE
  SHD = 0
  n.edgetests exact = TRUE
  elapsed_sec = 1058.716
  baseline S-affinity elapsed_sec = 899.077
  result: wall-time failed; do not promote

chunk/provider metrics:
  residual_worker_ms = 11329001
  baseline S-affinity residual_worker_ms = 9103781
  mgcv_fit_count = 273284
  baseline S-affinity mgcv_fit_count = 273284
  cache hits/misses = 336507 / 140045
  chunk groups / targets / inserts / errors = 103359 / 247269 / 133239 / 0
  setup provider groups / targets / templates / reuse / errors =
    46891 / 133239 / 46891 / 86348 / 0
  setup_provider_setup_ms = 1030507
  chunk_ms = 5095590
  dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
  Spectra count / converged / failed = 478808 / 478808 / 0

status:
  FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=chunk is correctness-clean but
  experimental only. It converts many same-S residual misses into provider
  cache hits, but it does not reduce total mgcv_fit_count versus S-affinity and
  adds substantial R-level chunk/provider overhead. The recommended route stays
  unchanged until a full artifact beats S-affinity wall time.

next:
  stop iterating on R-level worker-local task scheduling without a stronger
  diagnostic. The remaining mainline should move toward real same-S setup
  extraction for a C++/CUDA numeric executor, or a true batched same-S executor
  that reduces setup/extraction overhead without the R chunk state-machine cost.
```

Executor substrate checkpoint:

```text
fastkpc_mgcv_extract_same_setup_batch_fixed_sp_cpp()

status: implemented as a narrow non-CUDA same-S fixed-sp batch prototype
scope: mgcvExtract CPU/C++ helper only; not connected to the legacy skeleton
behavior:
  accepts a same-S target matrix Y, conditioning data S_data, and one fixed
    scalar sp per target
  extracts one template mgcv setup
  retargets that template for each target and per-target fixed sp
  solves each retargeted setup through the existing native C++ fixed-sp solver
  reports setup reuse while explicitly not claiming a true batched kernel

test:
  Rscript fastkpc/tests/test_mgcv_extract_same_setup_fixed_sp_batch_cpp.R

neighbor checks:
  Rscript fastkpc/tests/test_mgcv_extract_gpu_handle_batch_solve.R
  Rscript fastkpc/tests/test_mgcv_extract_batch_cpu.R

status:
  This is a substrate for the next residual executor iteration, not a promoted
  compatible route. It preserves fixed-sp semantics and proves a reusable
  same-S setup contract on CPU/C++ before any full skeleton integration.
```

Provider integration checkpoint:

```text
FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE=cpp

status: implemented as an env-gated same-S setup provider branch
scope: legacy same-S setup provider only; recommended route unchanged
behavior:
  same-S provider still runs per-target mgcv::gam to preserve selected-sp
    authority
  when all targets in a provider group have scalar positive sp and pass the
    condition threshold, the fixed-sp replay step uses
    fastkpc_mgcv_extract_same_setup_batch_fixed_sp_cpp()
  if the batch branch cannot safely run, provider falls back to the existing
    per-target path
  diagnostics report batch_solve enabled/group/target/ms/error counters

tests:
  Rscript fastkpc/tests/test_legacy_mgcv_same_s_fixed_sp_batch_provider.R
  Rscript fastkpc/tests/test_mgcv_extract_same_setup_fixed_sp_batch_cpp.R
  Rscript fastkpc/tests/test_precision_compatible_legacy_mgcv_residual_cpp_backend.R

status:
  This connects the fixed-sp batch substrate to the experimental provider path
  without promoting it. It still does not reduce mgcv::gam selected-sp calls;
  the next gate is a subset/full artifact showing whether batched fixed-sp
  replay reduces provider overhead enough to justify wider skeleton use.
```

Provider batch-solve subset gate:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_batch_provider_subset_v1

real 351x48 fixture, 16 hot-column subset
route: FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=chunk
comparison: SAME_S_BATCH_SOLVE unset vs SAME_S_BATCH_SOLVE=cpp

correctness:
  adjacency_identical = TRUE
  n.edgetests_identical = TRUE
  baseline/batch n.edgetests = 236,2243,3607,641,60 /
    236,2243,3607,641,60
  baseline/batch edge_count = 32 / 32

performance:
  baseline_elapsed_sec = 71.694
  batch_elapsed_sec = 37.177
  baseline/batch residual_worker_ms = 172305 / 167086
  baseline/batch mgcv_fit_count = 6517 / 6517
  baseline/batch cache_hits = 11087 / 11087
  baseline/batch chunk_ms = 108801 / 105549

provider:
  baseline/batch groups = 1555 / 1555
  baseline/batch targets = 4502 / 4502
  baseline/batch templates = 1555 / 1555
  baseline/batch reuse = 2947 / 2947
  baseline/batch provider_setup_ms = 20340 / 20893
  batch_solve enabled / groups / targets / ms / errors =
    TRUE / 1555 / 4502 / 98191 / 0
  baseline/batch native/fallback/error targets =
    5161,1356,0 / 5161,1356,0
  dCov cpp backend count/errors = 6671 / 0
  Spectra count/failed = 13342 / 0

status:
  subset correctness and wall-time pass, but this is not yet a recommended
  route. Worker-sum only improves modestly, provider setup time does not fall,
  and mgcv_fit_count is intentionally unchanged because selected-sp authority
  remains mgcv::gam. The result was enough to justify the full 351x48 artifact
  below before any promotion decision.
```

Provider batch-solve full 351x48 gate:

```text
fastkpc/artifacts/legacy_mgcv_residual_cpp_backend_same_s_batch_provider_full_351x48_v1

real 351x48 fixture, full compatible skeleton
route:
  FASTKPC_LEGACY_MGCV_RESIDUAL_BACKEND=cpp_guarded
  FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_SETUP=chunk
  FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE=cpp

correctness:
  adjacency_identical = TRUE
  SHD = 0
  n.edgetests_exact = TRUE
  n.edgetests = 2213,52659,125293,40694,13293,5422,835,80
  edge_count = 110 / 110

performance:
  elapsed_sec = 1045.958
  chunk baseline elapsed_sec = 1058.716
  residual_worker_ms = 11122609
  chunk baseline residual_worker_ms = 11329001
  mgcv_fit_count = 273284
  chunk baseline mgcv_fit_count = 273284
  cache hits / misses = 336507 / 140045

provider:
  setup provider groups / targets / templates / reuse / errors =
    46891 / 133239 / 46891 / 86348 / 0
  setup_provider_setup_ms = 1048208
  batch_solve enabled / groups / targets / ms / errors =
    TRUE / 46891 / 133239 / 4844318 / 0
  chunk groups / targets / inserts / ms / errors =
    103359 / 247269 / 133239 / 4977699 / 0
  backend native / fallback / error targets =
    157122 / 116162 / 0

dCov:
  dCov cpp backend count / errors / fallbacks = 239404 / 0 / 0
  Spectra count / converged / failed = 478808 / 478808 / 0

status:
  full correctness passes and the route is a small improvement over the
  same-S chunk baseline:

    1058.716s -> 1045.958s

  It is not a promotion candidate because it remains much slower than the
  current recommended S-affinity route, which is approximately 882-899s on the
  same 351x48 gate family. The batch fixed-sp replay branch does not reduce
  mgcv_fit_count, because selected-sp authority remains per-target mgcv::gam,
  and R-level chunk/provider overhead still dominates. Keep
  FASTKPC_LEGACY_MGCV_RESIDUAL_SAME_S_BATCH_SOLVE=cpp as experimental only.
```

### 8.4 Continue dCov backend improvement separately

Next dCov steps:

```text
real batched C++ dCov workspace
CUDA legacy-compatible dCov shadow
CUDA legacy-compatible dCov backend
```

Do not block mgcv residual work on dCov once current C++ Spectra backend is stable.

Current dCov batch substrate status:

```text
fastkpc_legacy_dcov_gamma_cpp_oracle_batch() exists
FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=chunk exists
FASTKPC_LEGACY_DCOV_GAMMA_CPP_BATCH=round exists
```

The C++ batch oracle has started moving beyond the original scalar-loop
wrapper. It now keeps shared C++ distance, eig/Spectra intermediate,
lowrank output/centering, and statistic/moment cross-product workspaces across
batch columns and reads matrix columns by pointer, so the oracle avoids
per-column Rcpp vector copies while preserving the scalar C++ dCov authority
for each pair. It still is not a promoted skeleton backend batch route: the
full chunk/round skeleton batch modes remain experimental until they show a
full 351x48 wall-time win.

It reports:

```text
scalar_total_ms
wrapper_overhead_ms
batch_overhead_ms
workspace_reuse_enabled
distance_workspace_reuse_count
statistic_moment_workspace_reuse_count
lowrank_output_workspace_reuse_count
lowrank_eig_workspace_reuse_count
column_copy_count
stage accounted timing
lowrank Spectra/full-eig aggregate counts
```

The env-gated chunk/round skeleton batch summaries also expose the same
workspace reuse counters with `legacy_dcov_cpp_batch_*` prefixes, including
workspace reuse calls, distance/statistic-moment/lowrank workspace reuse, and
column-copy counts. This is diagnostic plumbing for the experimental batch
routes only; it does not promote chunk or round batching over the recommended
S-affinity route without a full 351x48 wall-time win.

Gate:

```bash
Rscript fastkpc/tests/test_legacy_dcov_gamma_cpp_batch_oracle.R
Rscript fastkpc/tests/test_precision_compatible_legacy_dcov_cpp_backend.R
```

Next implementation target:

```text
continue moving batch execution from the current shared distance, lowrank
eig internals, output/centering, and statistic/moment workspaces toward full
skeleton batch promotion while preserving legacy C++ scalar parity and the
env-gated chunk/round skeleton replay semantics.
```

---

## 9. Final success definition

The final goal is reached when:

```text
R prepares data
one C++/CUDA skeleton call runs full skeleton
R receives skeleton/sepsets
orientation continues in R

full 351x48:
  edge_count = 110 / 110
  SHD = 0
  n.edgetests exact = TRUE

performance:
  materially faster than current compatible CPU/C++ route
  ideally moves from minutes toward tens of seconds

route:
  no unexplained CPU fallback
  all fallback reasons documented if any
  reproducible across repeated runs
```

The guiding principle remains:

```text
First 0 SHD, then speed.
```
