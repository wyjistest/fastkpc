# Legacy eig batch-shape invariance (p-level gate 0), 2026-10-21

Necessary-condition check for the level component prefill prototype
([plan](../../../docs/PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)):
with one fixed residual token per conditioning set, does the legacy eig batch
shape change any pair's p-value, statistic, mean or variance bits? The
component build forms Gram matrices with `cublasDgemmStridedBatched`
(`batchCount = miss_count`) and the pair stage uses `cublasDgemmBatched`
(`batchCount = pair_count`), so batch width and pair count are the two shape
parameters. Results: [`PHASE10_COMPONENT_BATCH_SHAPE_20261021.md`](../../../docs/PHASE10_COMPONENT_BATCH_SHAPE_20261021.md).

Only accepted-native public entries are used (no production change): the
capacity qualification entry for every target's selected log-SP, the
persistent exact-fixed evaluator for one residual token holding all 19
non-S targets, and `C_full_cuda_ci_phase35_legacy_eig_batch_from_optimizer_residual`
for every request. Run from the repository root on an idle GPU:

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/research/component_batch_shape_20261021/run.R \
  fastkpc/artifacts/full_cuda_ci/component_batch_shape_20261021
```

Per case (the first 12 archived level-3 conditioning sets of the structured
real replay, 22-column data, q = 28) the script issues 29 requests on the
same token: nine single pairs, four 4-wide/2-pair, two 8-wide/4-pair, one
16-wide/8-pair, one 19-wide/9-pair, one 19-wide with all 171 pairs, the same
171 pairs in reverse order, one 17-wide/10-pair window, and the nine single
pairs again. Every request is checked for status OK, component build count
equal to the referenced targets, and zero residual/component D2H. Per pair
row it records the SHA-256 of the (p, statistic, mean, variance) bits and the
values. Outputs: `summary.csv`, `pair_rows.csv`, `results.rds`.

This is a p-level necessary condition, not a byte-level comparison of the
cache payload (vectors, selected values, total, self moment). Direct diagnostic
payload comparison is required; switch-off / switch-on p-value equality cannot
substitute for it. This public entry uses a temporary single-lane context, not
the retained eight-lane cache context. Of the 2,088 directed pairs, 108 occur
at multiple component widths; 924 rows have a matching single-pair reference.
