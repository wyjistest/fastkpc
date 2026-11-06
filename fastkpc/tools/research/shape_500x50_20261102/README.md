# Current-native 500x50 shape check, 2026-11-02

This diagnostic distinguishes general strict CUDA input support from the
351x48-only structured-prefill and optimizer-slot opt-ins. It does not modify
or rebuild production code, generalize those opt-ins, or claim promotion.

Run from the repository root with GPU1 available:

```bash
python3 fastkpc/tools/research/shape_500x50_20261102/run.py
```

The runner refuses to overwrite existing evidence. It clears inherited
`FASTKPC_*` settings, enables exact-primary, coefficient reuse, 11 GiB retained
component cache, residual bypass and eight eig lanes, and leaves canonical-only
options disabled. Both GPUs are monitored for external compute processes.
CPU affinity is 0–19 and math libraries use one thread.

The existing public synthetic `full_cuda_ci_development_500x50_v2.rds` fixture
and its archived legacy-mgcv CPU oracle are authenticated before use. The R
call explicitly passes `numCol=35`: the facade's general `floor(n/10)` default
would be 50, outside the current strict CUDA parameter contract. The omitted
conditioning limit exercises default `Inf`, resolving to 48 and stopping
naturally at level 3.

On native `b2b79fc6219841a7b9de193e312de27bbdb8bca77a9d3a9d4a626aa1e135b12b`,
all 2,722 logical CI tests passed, with maximum CI/pMax error
`2.639000129534e-13` against the archived reference. Graph, semantic sepsets,
test counts, task/deletion identity, RNG preservation, GPU authority, cache
metadata and resource closure passed. One fresh skeleton call took 10.137 s.

Evidence is under `fastkpc/artifacts/full_cuda_ci/shape_500x50_20261102/`:
`protocol.json`, `run/report.json`, `run/receipt.rds`, GPU/process logs and
`manifest.json`. This is a single correctness check of synthetic data, with an
archived CPU reference. It is not a new real-data run, final-PDAG check or
repeated performance benchmark.
