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

#### Artifact

```text
legacy_dcov_gamma_cpp_batched_backend_v1
```

#### Gates

```text
SHD = 0
n.edgetests exact = TRUE
worker-ms lower than scalar C++ backend
wall time lower than scalar C++ backend
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

This is an executable R-level replay spec over the current `mgcv_residual_oracle_v1` cases. It replays the same legacy formula route, residual generation, downstream legacy dCov p-value, and alpha decision. It is not a replacement residual backend.

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
  options = list(...)
)
```

Internal C++ entrypoint:

```cpp
fastkpc_compatible_cuda_skeleton_run(...)
```

### Artifact

```text
compatible_cuda_skeleton_full_351x48_v1
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

Current artifact exists and is sourced from the full 351x48 skeleton deletion log. Expand it further with:

```text
rank-deficient / collinear / near-constant examples if present
additional late sparse levels if they expose new mgcv formula/setup behavior
formula route and mgcv metadata
downstream legacy dCov p-value and alpha decision
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
```

Next Phase 3 step:

```text
expand the guarded residual shadow from the 8-column real subset to a larger
real subset that includes |S|>2 fallback traffic, then run the full 351x48
compatible skeleton shadow. Promotion requires canonical replay to remain
unchanged and all shadow-supported residual targets to match without errors or
decision drift. The route remains shadow-only until that gate passes.
```

### 8.4 Continue dCov backend improvement separately

Next dCov steps:

```text
real batched C++ dCov workspace
CUDA legacy-compatible dCov shadow
CUDA legacy-compatible dCov backend
```

Do not block mgcv residual work on dCov once current C++ Spectra backend is stable.

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
