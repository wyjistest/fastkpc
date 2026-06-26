# Fast CUDA Baseline V1

Date: 2026-06-26

Commit: `1eb4b0a` plus validation commits through the current working tree.

Environment:

- GPU: 2 x NVIDIA GeForce RTX 4090, driver 555.42.06, 24564 MiB each
- R: 4.4.1
- CUDA native backend: `fastkpc/build/fastkpc_cuda.so`

## Scope

This baseline qualifies the default production speed path:

```text
fast_kpc(engine = "cuda", precision = "fast", graph_stage = "skeleton")
```

The measured path is the native CUDA skeleton data plane:

```text
native C++ skeleton controller
CUDA fastSpline residualization
CUDA dCov / dcc.gamma
native C++ replay
```

It does not include `hybrid`, `compatible`, `mgcvExtract`, or
`kpcTprsResidualCPP` verifier paths.

## Artifacts

Generated locally, not committed:

- `fastkpc/artifacts/fast_cuda_performance_baseline_default`
- `fastkpc/artifacts/fast_cuda_performance_baseline_full_grid`
- `fastkpc/artifacts/fast_cuda_performance_baseline_real_n100_p12`

The validation gate is:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
Rscript fastkpc/tools/run_fast_cuda_data_plane_validation.R
```

The performance baseline runner is:

```bash
FASTKPC_RUN_CUDA_TESTS=1 \
bash fastkpc/tools/run_fast_cuda_performance_baseline.sh <output_dir>
```

## Results

Default synthetic baseline:

```text
runs: 120
ok: 120
skipped: 0
errors: 0
fast_cuda_route_violations: 0
median speedup vs fast_cpu: 3.90x
```

Full synthetic grid:

```text
runs: 405
ok: 310
skipped: 95 legacy_mgcv rows
errors: 0
fast_cuda_route_violations: 0
median speedup vs fast_cpu: 7.72x
```

Real `cancer_RD-causalDiscoveryInput_n100_p12` baseline:

```text
fast_cuda median: 255 ms
fast_cpu median: 1542 ms
legacy_mgcv median: 14902 ms
speedup vs fast_cpu: 6.05x
speedup vs legacy_mgcv: 58.44x
```

## Correctness

Route purity:

```text
fast_cuda scheduler: layer
precision overlay used: false
CPU fallback count: 0
route violations: 0
```

Graph agreement against `fast_cpu`:

```text
synthetic full grid:
    fast_cuda adjacency exact: 135 / 135
    fast_cuda sepsets exact: 135 / 135
    fast_cuda max pMax drift: 9.43e-13

real n100/p12:
    fast_cuda adjacency exact: 5 / 5
    fast_cuda sepsets exact: 5 / 5
    fast_cuda max pMax drift: 3.23e-12
```

`legacy_mgcv` is a reference/compatibility path, not the fast-path oracle.
Its pMax and sepsets differ from the fastSpline fast path as expected.

## Performance Shape

Small synthetic cases can be neutral or slightly slower due to GPU setup and
batching overhead:

```text
synthetic-n100-p8-m2: 0.82x vs fast_cpu
synthetic-n100-p8-m1: 1.15x vs fast_cpu
```

Larger cases show the intended scaling:

```text
n=300 cases: about 6.7x to 8.6x vs fast_cpu
n=1000 cases: about 34x to 67x vs fast_cpu
```

Full-grid stage timing for `fast_cuda`:

```text
median skeleton:          59.0 ms
median residual_prefetch: 25.7 ms
median ci_eval:           28.5 ms
median native_replay:      0.04 ms
```

Real n100/p12 stage timing:

```text
median skeleton:          255.0 ms
median residual_prefetch: 158.0 ms
median ci_eval:            92.7 ms
median native_replay:       0.08 ms
```

The dominant costs are fastSpline residualization and CUDA dCov/dcc.gamma.
Native replay is not a bottleneck.

## Decision

`fast_cuda` is qualified as the production fast path for skeleton execution:

```text
recommended speed path:
    fast_kpc(engine = "cuda", precision = "fast")

accuracy overlay:
    precision = "hybrid"

compatibility/reference:
    mgcvExtract / kpcTprsResidualCPP
```

Future speed work should be driven by stage timing:

```text
residual_prefetch high -> optimize fastSplineCUDA batching
ci_eval high           -> optimize CUDA dCov/dcc.gamma batching
native_replay high     -> optimize C++ skeleton replay
```

No further `kpcTprs` or `mgcvExtract` work is required to advance the
production fast path.

## Known Limits

- `legacy_mgcv` is skipped for large full-grid rows by configured limits:
  `n > 300`, `p > 12`, or `max_conditioning_size > 2`.
- Small `n/p` workloads may not benefit from CUDA.
- Baseline covers skeleton execution, not WANPDAG orientation.
- Baseline covers `dcc.gamma`, not HSIC modes.
