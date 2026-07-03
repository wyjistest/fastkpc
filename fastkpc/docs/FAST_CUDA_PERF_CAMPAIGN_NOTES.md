# Fast CUDA CI Performance Campaign Notes

## Final status

- Mainline low-risk fast CUDA CI performance campaign is closed.
- Final baseline artifact:
  `fastkpc/artifacts/fast_cuda_stage_breakdown_post_dcov_rowsum_block_tuned_final_v1`
- Final mainline commit at closeout:
  `c1639f7 perf: tune dcov rowsum block size`
- Remaining fast CUDA CI performance work should be treated as experimental or
  env-gated kernel/algorithm work, not low-risk mainline plumbing.

## Final artifact summary

Median values from the final artifact:

| scenario | n | skeleton_total_ms | ci_eval_total_ms | dcov_total_ms | dcov_rowsum_ms | rowsum_threads | rowsum_launch_ms | rowsum_max_chunk_ms | abs_fast_count | pow_generic_count | residual_prefetch_ms | accounted_share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| breakdown-n100-p12-m2 | 100 | 5.018 | 0.996 | 0.768 | 0.228 | 64 | 0.076 | 0.155 | 508 | 0 | 3.903 | 0.9990 |
| breakdown-n300-p12-m2 | 300 | 7.006 | 3.118 | 2.574 | 1.390 | 64 | 0.348 | 0.980 | 711 | 0 | 3.735 | 0.9992 |
| breakdown-n1000-p12-m2 | 1000 | 36.682 | 23.308 | 20.114 | 16.839 | 64 | 4.210 | 7.397 | 1158 | 0 | 13.132 | 0.9998 |

Final route invariants:

- `cpu_fallback_count = 0`
- `precision_overlay_used = FALSE`
- dCov rowsum default route uses the branch-free abs-fast kernel.
- dCov semantic `index != 1` generic pow path remains covered by tests.
- dCov abs-fast rowsum uses 64-thread blocks by default.
- dCov rowsum block override remains available via
  `FASTKPC_DCOV_ROWSUM_BLOCK=64|128|256`.

## Effective optimizations merged to main

### dCov CUDA path

- Fused raw aggregate rowsum/reduce path.
- dCov CUDA workspace reuse.
- p-value-only scheduler path.
- Process-wide dCov grid-limit cache.
- Rowsum chunk-shape diagnostics.
- Branch-free abs-fast rowsum path for `legacy_index || index == 1.0`.
- 64-thread abs-fast rowsum block tuning.

### fastSpline residual CUDA path

- Run-scoped residual workspace reuse.
- CI staging buffer reuse.
- Unique design X reuse in true-batch residual path.
- Residual cache prefetch move/key reuse.
- Algebraic RSS scoring.
- Residual-only scheduler path.
- Small-p RHS solve.
- D2H score metadata coalescing and batching.
- Candidate RHS solve fused into algebraic RSS.
- Batched fused candidate scoring.
- Typed slab workspace for true-batch buffers.
- H2D metadata coalescing.
- Run-scoped design cache.
- Run-scoped basis cache.
- Cubic basis eval micro-optimization.
- Structural grouping keys.
- Shared design objects across cache and groups.
- Large-p candidate RHS solves batched across lambda grid.

## Experiments rejected by artifact

These were tested and intentionally not merged as default mainline behavior:

- Small-k basis eval/normalize specialized path.
- Fixed 10-column basis helper.
- Sorted-column cache for knot build.
- Atomic-free dCov rowsum raw aggregate reduce.

The atomic-free rowsum experiment did not reduce `dcov_rowsum_distance`; it
added row raw partial writes and row scalar reduce cost, which made
`dcov_measured_total` and `ci_eval_total` worse in the final scenarios.

## Env-gated or experimental paths retained

- `FASTKPC_FASTSPLINE_EDF_TRACE_SHADOW=1`
  - Shadow EDF trace diagnostics only.
  - Does not affect winner selection, GCV scoring, residuals, or graph output.

- `FASTKPC_FASTSPLINE_EDF_TRACE_MODE=cholesky_cuda`
  - Experimental CUDA Cholesky EDF trace mode.
  - Skips candidate full inverse.
  - Not default.

- Future candidate:
  - `exp: tiled dcov rowsum raw aggregate kernel`
  - Should stay env-gated until parity, p90, and route diagnostics justify a
    default switch.

## Remaining hotspots

- `dcov_rowsum_distance`
  - Largest remaining global hotspot, especially at `n = 1000`.
  - Next work should be a tiled/memory-reuse kernel experiment, not a default
    mainline change.

- `residual_factor_inverse_solve`
  - Full inverse remains default for EDF trace.
  - `cholesky_cuda` has net benefit in targeted artifacts, but the CUDA trace
    kernel currently eats back much of the saved inverse time.

- `residual_grouping / design_build`
  - Remaining cost is mostly genuine cold basis/design construction.
  - Cheap string, copy, cache, and object-sharing overhead has already been
    removed or measured.

## Final validation checklist

Closeout validation command block:

```bash
bash fastkpc/tools/build_cuda_native.sh

Rscript fastkpc/tests/test_fastspline_basis.R

FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_dcov_cuda_batch.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_fast_cuda_stage_breakdown.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_dcc_gamma_cuda_parity_artifact.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_fast_cuda_data_plane_validation_runner.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_cuda_fastspline_true_batch_contract.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_cuda_fastspline_residual_batch.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_cuda_layer_scheduler_true_residual_batch.R
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_cuda_fastspline_large_p_true_batch.R

FASTKPC_FASTSPLINE_EDF_TRACE_MODE=cholesky_cuda \
FASTKPC_RUN_CUDA_TESTS=1 Rscript fastkpc/tests/test_dcc_gamma_cuda_parity_artifact.R

git diff --check
```
