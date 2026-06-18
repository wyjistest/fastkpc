# Goal: Build a precision ladder and mgcvExtractGPU bridge for fastkpc

## Motivation

`fastkpc` now has a fast CUDA primary path:

```text
fastSplineCUDA residualization
true-batched cuSOLVER fastSpline solves
CUDA dCov / HSIC
residual cache
layer scheduler
canonical replay
WAN-PDAG opt-in path
mgcv Gate B CPU fixed-sp self-solve
near-alpha hybrid replay scaffolding
```

The latest performance work fixed the major CUDA residual bottleneck: on the
real cancer dataset, `mmax = 2` dropped from about 235 seconds to about 50
seconds while preserving the fastkpc skeleton exactly. That means remaining
differences against legacy `kpcalg` should be treated primarily as residual
semantics and accuracy issues, not as CUDA performance bugs.

The next stage should not mutate `fastSplineCUDA` until its role is fixed:

```text
fastSplineCUDA:
    frozen high-throughput approximate baseline

mgcvExtractGPU:
    mgcv setup anchored compatibility bridge
    not an approximate backend

tprsApproxCUDA / mgcvInspiredCUDA:
    future pure-GPU higher-accuracy approximate backend
    only justified after attribution data
```

The goal is to establish a precision ladder that can explain where residual,
p-value, skeleton, sepset, and WAN-PDAG drift comes from before investing in a
new pure GPU approximation.

## Non-goals

```text
No full mgcv GPU clone.
No bamGPU.
No mutation of the current fastSplineCUDA baseline.
No raw mgcv sp reuse across different basis/penalty parameterizations.
No immediate tprsApproxCUDA implementation before attribution evidence.
No self-contained multi-penalty GCV optimizer in the first milestone.
No claim that fastSplineCUDA is mgcv-compatible.
```

`mgcvExtractGPU` must be described as a compatibility bridge: mgcv constructs
the model setup, and fastkpc uses GPU numerical kernels to solve the restricted
Gaussian residualization problem. It is allowed to depend on mgcv internals and
version fingerprints. It is not a pure GPU approximation.

## Backend taxonomy

```text
legacy mgcv:
    direct kpcalg-compatible reference
    slow but authoritative

Gate B CPU:
    mgcv setup + fastkpc CPU fixed-sp solve
    verifies basis / penalty / constraint / solve semantics

mgcvExtractGPUFixedSP:
    mgcv setup + fastkpc GPU fixed-sp solve
    verifies GPU numerical parity against Gate B CPU

mgcvExtractGPUGCVBridge:
    mgcv selects sp
    fastkpc GPU solves at selected sp
    compatibility verifier, not self-contained GCV

mgcvExtractGPUGCV:
    mgcv setup
    fastkpc selects smoothing parameter on GPU for supported single-penalty setups
    future high-compatibility production candidate

fastSplineCUDA:
    fast approximate primary backend
    frozen baseline

tprsApproxCUDA:
    future pure-GPU approximate thin-plate-like backend
    only after projection-floor / oracle-lambda evidence justifies it
```

## Phase 0: Accuracy attribution campaign

Before writing new approximation kernels, build an attribution campaign that
separates basis mismatch, penalty mismatch, smoothing parameter selection,
solver drift, and CI-test amplification.

### Experiment ladder

```text
legacy mgcv:
    basis / penalty = mgcv
    smoothing parameter = mgcv
    solver = mgcv

Gate B CPU:
    basis / penalty = mgcv
    smoothing parameter = mgcv
    solver = fastkpc CPU

mgcvExtractGPUFixedSP:
    basis / penalty = mgcv
    smoothing parameter = mgcv
    solver = fastkpc GPU

mgcvExtractGPUGCV:
    basis / penalty = mgcv
    smoothing parameter = fastkpc GPU
    solver = fastkpc GPU

fastSpline oracle-lambda:
    basis / penalty = fastSpline
    smoothing parameter = oracle chosen to minimize residual drift to mgcv
    solver = fastSpline CUDA or CPU reference

tprsApprox candidate:
    basis / penalty = approximate thin-plate-like GPU setup
    smoothing parameter = fastkpc
    solver = GPU
```

Do not compare raw `sp` values across different basis and penalty
parameterizations. Across backends, compare fitted values, residuals, EDF,
GCV-like scores, p-values, and graph outputs.

### Basis projection floor

For each target and conditioning set, define the fitted-value approximation
floor of a candidate basis `B` against the legacy mgcv fitted vector `f_m`:

```text
e_basis = ||f_m - P_B f_m||_2 / ||f_m||_2
```

This estimates the best possible fitted-value error for that basis family,
before smoothing parameter selection is considered.

Interpretation:

```text
high projection floor:
    basis / penalty geometry likely dominates drift
    smoothing optimizer work will not solve the main issue

low projection floor:
    basis can represent the mgcv fit
    smoothing selection or penalty scaling may be the larger problem
```

### Oracle-lambda gap

Within the fixed fastSpline basis, search a richer lambda grid and choose the
lambda that minimizes residual drift against legacy mgcv residuals.

Record:

```text
current_lambda_residual_rel_l2
oracle_lambda_residual_rel_l2
current_lambda_log_p_drift
oracle_lambda_log_p_drift
current_lambda_decision_flip
oracle_lambda_decision_flip
```

Interpretation:

```text
large oracle improvement:
    smoothing selection is a major source of drift

small oracle improvement:
    basis / penalty geometry is the likely floor
```

### Frozen and native CI diagnostics

Every residual comparison should produce two CI-test views:

```text
frozen CI configuration:
    same kernel bandwidth
    same distance normalization
    same permutation / seed / index
    isolates residual drift

native CI configuration:
    each backend follows normal CI parameter estimation
    measures end-to-end drift
```

Record:

```text
kernel_bandwidth_legacy
kernel_bandwidth_candidate
test_stat_legacy
test_stat_candidate
p_frozen_config
p_native_config
decision_flip_frozen
decision_flip_native
distance_to_alpha_log
```

This separates residual error from downstream p-value amplification.

## Phase 1: mgcvExtractGPUFixedSP

Implement the first GPU compatibility bridge:

```text
CPU:
    mgcv generates setup
    X / S / C / rank / L / lsp0 / sp / fingerprints

GPU:
    penalty assembly
    constrained fixed-sp solve
    batched fitted / residual generation
```

The first gate is numerical parity:

```text
same extracted setup
same effective penalty multipliers
same target-specific sp
GPU residuals ~= Gate B CPU residuals
GPU fitted values ~= Gate B CPU fitted values
```

Implementation should be same-S multi-target first:

```text
setup_handle(S):
    transformed X or null-space X
    X'X
    penalties
    constraint / null-space transform
    rank metadata
    setup fingerprint

payload:
    many target columns Y
    per-target sp
```

The handle should avoid uploading setup for every CI test. It should be
reusable within a scheduler level or campaign group. Residuals should be able
to remain on device for CUDA dCov / HSIC when the caller asks for an all-CUDA
pipeline.

### Gate C: fixed-sp GPU parity

Required metrics:

```text
coef_rel_l2 when parameterization is comparable
fitted_rel_l2
residual_rel_l2
max_abs_residual_diff
edf_diff
rss_diff
warning / rank diagnostics
setup_fingerprint
target_fingerprint
```

Suggested first tolerance:

```text
residual_rel_l2 <= 1e-6 to 1e-5 on non-pathological cases
fitted_rel_l2 <= 1e-6 to 1e-5 on non-pathological cases
```

If this gate fails, stop GCV work and debug setup transform, constraint
handling, penalty assembly, rank handling, and GPU solve numerics.

## Phase 2: single-penalty extracted-setup GPU GCV

Only after fixed-sp GPU parity, implement smoothing parameter selection for the
single-penalty mgcv setup cases:

```text
|S| = 1:
    y ~ s(s1)

|S| = 2:
    y ~ s(s1, s2)
```

These are the best first candidates because they are joint `s(...)` smooths
with a single main smoothing parameter in the restricted kpcalg semantics.

Preferred numerical form:

```text
1. transform parameters into the constraint null space
2. whiten X'X
3. diagonalize the penalty once per setup
4. evaluate many target-specific lambda candidates by elementwise shrinkage
5. compute fitted values, RSS, EDF, and GCV in batch
```

This Demmler-Reinsch-style representation should avoid repeated Cholesky for
each lambda and target in the single-penalty case.

Record:

```text
sp_source = "fastkpc-gpu"
gcv_source = "fastkpc-gpu"
solve_source = "mgcvExtractGPU"
is_self_contained_gcv = TRUE
```

Compare against:

```text
legacy mgcv
Gate B CPU with mgcv-selected sp
mgcvExtractGPUFixedSP at mgcv-selected sp
mgcvExtractGPUGCVBridge
```

## Phase 3: graph-level hybrid comparison

Run graph-level comparisons across:

```text
legacy mgcv
fastSplineCUDA
mgcvExtractGPUFixedSP / GCVBridge
mgcvExtractGPUGCV for |S| <= 2
hybrid fastSplineCUDA -> mgcvExtractGPU near alpha
```

Key graph metrics:

```text
skeleton SHD
skeleton precision / recall / F1
edge deletion mismatch
first separating set mismatch
sepset mismatch rate
WAN-PDAG orientation mismatch
arrowhead agreement
near-alpha verifier call count
verifier-induced decision changes
runtime by backend
```

The hybrid invariant remains:

```text
fallback may replace p-value source
fallback must not change canonical test order
edge deletion and sepset selection must replay in canonical order
```

## Phase 4: decide whether tprsApproxCUDA is justified

Only design `tprsApproxCUDA` if attribution proves that:

```text
basis / penalty mismatch dominates fastSplineCUDA drift
mgcvExtractGPU dependence on CPU mgcv setup is too expensive for target workloads
a pure GPU approximate backend can preserve graph-level accuracy gains
```

Potential scope:

```text
tprsApproxCUDA-1D:
    covers |S| = 1
    also useful for additive |S| > 2 one-dimensional terms

tprsApproxCUDA-2D:
    covers |S| = 2 joint isotropic smooth

tprsApproxCUDA-additive:
    multi-penalty additive path for |S| > 2
```

Do not treat tensor B-spline as equivalent to mgcv default `s(s1, s2)`. A
future approximate backend should be explicitly described as thin-plate-like or
mgcv-inspired, not mgcv-compatible.

## Acceptance gates

### Gate A: attribution data exists

```text
basis projection floor implemented
oracle-lambda gap implemented
frozen/native CI diagnostics implemented
residual / p-value / graph summaries written to CSV
```

### Gate B: existing CPU compatibility remains protected

```text
mgcv fixed-sp Gate B CPU campaign still passes
existing canonical replay tests still pass
fastSplineCUDA baseline output remains stable
```

### Gate C: mgcvExtractGPUFixedSP parity

```text
GPU fixed-sp residuals match Gate B CPU residuals
GPU fixed-sp fitted values match Gate B CPU fitted values
diagnostics explain failures by setup / rank / solve stage
```

### Gate D: single-penalty GPU GCV usefulness

```text
|S| = 1 and |S| = 2 selected smoothness is close enough to legacy mgcv
residual drift decreases relative to fastSplineCUDA
p-value drift decreases relative to fastSplineCUDA
near-alpha decision flips decrease
```

### Gate E: graph-level value

```text
hybrid with mgcvExtractGPU improves or controls skeleton SHD
sepset mismatch decreases or is explained
WAN-PDAG orientation drift is measured
runtime remains meaningfully below legacy mgcv for target workloads
```

### Gate F: tprsApproxCUDA decision

```text
projection-floor and oracle-lambda results justify or reject pure-GPU TPRS work
decision is documented before implementation
```

## Recommended issues

```text
Issue 1: Freeze fastSplineCUDA as approximate baseline and expose backend version
Issue 2: Add basis projection floor metrics
Issue 3: Add fastSpline oracle-lambda attribution runner
Issue 4: Add frozen/native CI diagnostics
Issue 5: Implement mgcvExtractGPUFixedSP setup handle
Issue 6: Validate GPU fixed-sp solve against Gate B CPU
Issue 7: Add device-resident same-S multi-target extracted setup batching
Issue 8: Implement single-penalty extracted-setup GPU GCV for |S| = 1
Issue 9: Extend single-penalty GPU GCV to |S| = 2 joint smooth
Issue 10: Run graph-level hybrid comparison against legacy mgcv
Issue 11: Write tprsApproxCUDA go/no-go memo from attribution evidence
```

## Success criteria

```text
1. fastSplineCUDA remains a stable high-throughput baseline.
2. Attribution campaign identifies whether smoothing selection or basis/penalty geometry dominates drift.
3. mgcvExtractGPUFixedSP matches Gate B CPU at numerical tolerance.
4. Single-penalty extracted-setup GPU GCV reduces residual and p-value drift for |S| <= 2.
5. Hybrid fastSplineCUDA -> mgcvExtractGPU near alpha reduces graph drift against legacy mgcv.
6. The project has evidence for whether tprsApproxCUDA is worth building.
```

## Final positioning

The next stage should make fastkpc's backend story explicit:

```text
fastSplineCUDA:
    fastest approximate primary backend
    frozen and benchmarked

mgcvExtractGPU:
    high-compatibility bridge anchored by mgcv setup
    verifier and future production candidate for selected cases

tprsApproxCUDA:
    optional future pure GPU higher-accuracy approximation
    starts only after attribution evidence
```

The near-term win is not a new spline kernel. It is a precision ladder that can
explain, reproduce, and reduce graph-level drift while preserving the speed
already gained from CUDA batching.
