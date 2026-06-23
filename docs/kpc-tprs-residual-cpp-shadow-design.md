# kpcTprsResidualCPP shadow design

Date: 2026-06-23
Status: design spike

## Positioning

`kpcTprsResidualCPP` is a standalone C++ residual engine for the restricted single-smooth TPRS semantics used by kPC. It is developed first as a shadow candidate against a version-pinned mgcv oracle and does not drive graph decisions until residual-, CI-, and graph-level compatibility gates pass.

This is not a C++ implementation of `mgcv::gam()`. It is not a generic GAM framework and is not a drop-in replacement for mgcv.

## First-Version Contract

Input:

```text
finite numeric target y
finite numeric conditioning matrix S
```

Supported:

```text
Gaussian identity
residual output only
|S| = 1 or 2
one joint isotropic thin-plate regression spline
one quadratic penalty
target-specific smoothing parameter
repeated targets sharing the same S
```

Unsupported:

```text
weights
offset
missing values
non-Gaussian families
multiple smooth terms
|S| > 2
te / ti / t2
summary / vcov / SE / prediction
GAMM
```

The correct compatibility claim is:

> mgcv-compatible residual semantics for the restricted kPC single-smooth TPRS subset.

## Return Contract

The oracle and candidate should return the same shape:

```r
list(
  residuals,
  fitted,
  coefficients,
  edf,
  selected_sp,
  score,
  basis_rank,
  null_space_rank,
  setup_fingerprint,
  diagnostics
)
```

## Shadow Architecture

Shadow mode runs both paths:

```text
raw y, S
   |-- mgcvExtract oracle
   `-- kpcTprsResidualCPP candidate
```

Graph decisions always use the mgcvExtract oracle during shadow mode. The candidate only records differences and does not affect `p_used`, edge deletion, sepsets, `pMax`, skeleton, or WAN-PDAG orientation.

Initial campaigns may run full shadow. Larger real workloads may shadow only:

```text
fixed sample ratio
near-alpha tests
first occurrence of each setup fingerprint
numerically difficult cases
```

## Implementation Phases

### Phase 1: C++ TPRS Setup

Build only:

```text
S
-> unique covariate handling
-> thin-plate radial kernel
-> polynomial null space
-> low-rank truncation
-> penalty
-> centering / identifiability constraint
-> prepared setup
```

Reuse the existing C++ spectral scoring, CUDA fixed-sp batch solve, and CUDA dCov paths. Do not add new CUDA kernels in this phase.

### Phase 2: Fixed-Sp Parity

Compare:

```text
mgcv setup + fastkpc solve
vs
C++ setup + fastkpc solve
```

Use fixed smoothing strengths so setup semantics are isolated from smoothing-parameter optimizer differences.

### Phase 3: Continuous Single-Penalty GCV

Use the existing 17-point grid only as the bracket source, then refine in `log(sp)` with a one-dimensional method such as Brent search.

Each target selects its own `sp`; setup and spectral state remain shared across targets for the same `S`.

### Phase 4: Shadow Campaign

Compare, in order:

```text
setup geometry
fixed-sp residual
GCV / EDF / selected smoothing
CI statistic / p-value
decision
sepset
skeleton
```

The candidate remains non-authoritative.

### Phase 5: Limited Switch

Only after all gates pass:

```text
kpcTprsResidualCPP drives |S| <= 2 compatible backend
mgcvExtract remains sampled oracle / fallback
legacy mgcv remains final fallback
```

## Acceptance Metrics

Do not require raw `sp_cpp == sp_mgcv` unless penalty scaling has been intentionally matched. Equivalent bases can rotate or scale the function space while preserving the smoother.

Prioritize:

```text
basis column-space projector
generalized penalty eigenvalues
effective smoother / fitted values
EDF
residuals
GCV curve
p-value and graph behavior
```

Setup-level projector distance:

```text
||P_cpp - P_mgcv||_F / ||P_mgcv||_F
```

where `P` is the basis column-space projector.

## Gates

### Gate A: Function Space

```text
null-space dimension matches
effective rank matches
projector distance is small
penalty generalized spectrum aligns
```

### Gate B: Fixed-Sp

```text
fitted relative L2 is small
residual relative L2 is small
EDF is close
fixed-sp CI decision does not regress
```

### Gate C: Own GCV

```text
GCV minimum is in an equivalent location
EDF is close
residual / p-value is close
near-alpha decision flips are controlled
```

### Gate D: Graph-Level Shadow

```text
introduced flips do not exceed threshold
corrected / introduced flips are explainable
skeleton does not regress
first/all sepset mismatch does not regress
pMax drift does not regress
```

### Gate E: Switch Qualification

```text
multiple seeds / n / data scales pass
|S| = 1 and |S| = 2 pass
difficult numeric cases fail closed
mgcv fallback remains available
candidate does not depend on mgcv setup
```

## Difficult Cases

Campaigns must cover:

```text
repeated covariate rows
many ties
very different variable scales
translation, scaling, and 2D rotation
near collinearity
k close to available unique combinations
small samples
near rank-deficient penalty
GCV minimum on boundary
near-alpha CI tests
```

For two-dimensional isotropic smooths, rotated inputs should preserve residual, p-value, and graph behavior up to numerical tolerance. Failure here suggests an accidental tensor-product geometry rather than isotropic TPRS behavior.

## Go / No-Go

Go:

```text
C++ setup fixed-sp residuals are stable against the oracle
projector and penalty geometry align
shadow graph campaign does not regress
```

No-go or redesign:

```text
error is dominated by unstable rank / truncation semantics
2D rotation invariance fails
fixed-sp parity fails but GCV work begins anyway
```

## Explicit Non-Goals

Do not implement:

```text
generic mgcv::gam clone
multiple smooth classes
multi-penalty |S| > 2 optimizer
te / ti / t2
families beyond Gaussian identity
summary / vcov / SE / prediction
WAN-PDAG precision data-plane integration
new CUDA kernels
```
