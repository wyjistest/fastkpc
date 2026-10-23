# Residual cohort bit-identity experiment, 2026-10-21

Isolated GPU diagnostic for the prerequisite question raised in
[the component lifecycle replay](../../../docs/PHASE10_COMPONENT_LIFECYCLE_REPLAY_20261021.md)
and its review: with the same prepared-S geometry, the same target columns and
the same exact-primary coefficient bits, does the *cohort shape* of the
residual materialization (how many targets are finalized together, and in
which column position) change the fitted or residual bytes? Results are in
[`PHASE10_RESIDUAL_COHORT_IDENTITY_20261021.md`](../../../docs/PHASE10_RESIDUAL_COHORT_IDENTITY_20261021.md).

Only the accepted native `fe8286f7…` and its public
`C_full_cuda_ci_multi_penalty_capacity_qualify` entry are used: the same
exact-primary optimizer (one block per target), the same cohort-wide
`cublasDgemm(n, target_count, q)` finalization as the production
`finalize_selected_coefficients` path, and the exact-fixed stable-SVD
evaluator, with full host shadows. No production source, object or library is
modified. Run from the repository root with one idle RTX 4090.

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/research/residual_cohort_identity_20261021/run.R \
  fastkpc/artifacts/full_cuda_ci/residual_cohort_identity_20261021

Rscript fastkpc/tools/research/residual_cohort_identity_20261021/analyze.R \
  fastkpc/artifacts/full_cuda_ci/residual_cohort_identity_20261021

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/research/residual_cohort_identity_20261021/threshold.R \
  fastkpc/artifacts/full_cuda_ci/residual_cohort_identity_20261021
```

`run.R` builds 40 cases with real conditioning sets: the first 24 archived
level-3 pairs of the structured-eig real replay (22-column data, q = 28, whose
archived production-dispatcher outputs serve as anchors and whose prepared
geometry SHA must match), plus 16 deeper S from the canonical 32-connection
receipt (48-column data; levels 4–8, q = 37, 46, 55, 64, 73). For every case
it finalizes the same targets in 15 cohort shapes: the archived pair, the pair
repeated, the pair swapped, prefixes of widths 3/5/7/10/16/24/32 of a
pair-first order, two 5-wide windows at other positions, a reordered 3-wide
prefix, and every non-S column in natural, reversed and permuted order (19 or
40–44 targets). Each call is checked for solver info 0, stable-SVD route,
zero CPU/fallback counts and resource closure. Per column it records SHA-256
digests of optimizer coefficients, fitted values, residuals, exact-fixed
coefficients/fitted/residuals and selected/fixed log-SP, and the maximum
absolute difference against the pair cohort.

`analyze.R` tabulates which shapes change the fitted digests, whether the
change depends on width or column order, and the difference magnitudes.
`threshold.R` reruns widths 16–20 on six cases to locate the class boundary.
Artifacts: `cases/NNN.rds`, `summary.csv/rds`, `analysis.rds`,
`fitted_mismatch_by_cohort.csv`, `fitted_mismatch_by_width.csv`,
`width_threshold.csv/rds`, `run.log`.

Scope limits: the multi-penalty route (levels >= 3) only; the single-penalty
route used at levels 1–2 (`C_full_cuda_ci_single_penalty_gcv_multi_cuda`,
fused exact replay with its own `target_count`-shaped projection GEMMs) is not
covered. Production one-call cohorts are not captured directly (production
never downloads residuals); the archived 2-target dispatcher outputs are the
anchor.
