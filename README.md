# fastkpc workspace

This repository keeps the legacy `kpcalg` sources and the staged `fastkpc`
backend work in one workspace. The active fast backend code is under
`fastkpc/`.

## Full-CUDA compatible skeleton candidate

The explicit full-CUDA skeleton route has passed the frozen Phase 10 canonical
351x48 campaign. It is not promoted or recommended yet: the externally held
sealed promotion corpus remains `SEALED_NOT_RELEASED`.

The route is separate from the older `fast_kpc(precision = "compatible")`
bridge described below. Invoke it explicitly:

```r
source("fastkpc/R/fast_kpc.R")

skeleton <- fastkpc_compatible_cuda_skeleton(
  data,
  alpha = 0.1,
  options = list(
    route = "full_cuda",
    compatible_cuda_strict = TRUE,
    max_conditioning_size = 7L,
    index = 1,
    numCol = 35L,
    trace_level = "logical"
  )
)
```

This route implements the exact Gaussian/identity `regrXonS` subset used by
the KPC skeleton: a joint thin-plate smooth for `|S| <= 2`, additive smooths
for `|S| > 2`, target-specific GCV selection, CUDA residual formation, and
legacy-compatible CUDA `dcov.gamma`. It is not a general mgcv clone.

The currently qualified public envelope is deliberately narrow:

```text
finite binary64 matrix; n > 35; 2 <= p <= 64
alpha = 0.1; index = 1; numCol = 35
0 <= max_conditioning_size <= 7
Gaussian family; identity link; unweighted; zero offset
skeleton stage only
```

Strict mode fails closed outside that envelope. It does not silently call
legacy mgcv, a CPU numerical CI path, `fastSplineCUDA`, or another approximate
backend. C++ owns canonical skeleton replay; CUDA owns repeated numerical CI
work. The result and target-state caches are capacity-bounded and keyed by
semantic dataset/test identities.

Build and validate the frozen candidate with:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_full_cuda_ci_one_call.R
Rscript fastkpc/tests/test_full_cuda_ci_one_call_cache.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_hardening_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_campaign_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_completion_audit.R
```

The completion audit intentionally fails until the sealed holdout artifact and
final `COMPLETE` roadmap state exist.

Canonical campaign evidence is under
`fastkpc/artifacts/full_cuda_ci/promotion_351x48_v1/`. Five cold, five warm,
and five same-campaign correct-baseline runs all reproduced 110 edges,
SHD 0, exact sepsets, exact logical test counts, and the exact deletion trace.
The measured warm median was 0.699 seconds versus a 601.431-second baseline
median; the cold median was 1290.664 seconds and is reported separately without
a promotion threshold. All CPU numerical authority, approximate-backend, and
unknown-fallback counters were zero.

Final promotion additionally requires an external custodian release directory
and token through `FASTKPC_PROMOTION_HOLDOUT_RELEASE_DIR` and
`FASTKPC_PROMOTION_HOLDOUT_RELEASE_TOKEN`. The standard gate fails when those
inputs or the resulting sealed-holdout artifact are absent:

```bash
FASTKPC_RUN_CUDA_TESTS=1 bash fastkpc/tools/run_full_cuda_ci_gate.sh
```

## Operational backend positioning

```text
precision = "fast":
    fastSplineCUDA
    fastest approximate CUDA primary backend
    not mgcv-compatible

precision = "compatible":
    mgcvExtractGPU where the version/semantic envelope is supported
    mgcvExtractCPU / legacy mgcv fallback otherwise

precision = "hybrid":
    fastSplineCUDA primary
    mgcvExtractGPU near-alpha verifier
    canonical replay preserved
```

`mgcvExtractGPU` is a version-pinned compatibility bridge. It relies on mgcv
setup semantics and does not claim to be a full mgcv clone or a pure GPU
approximation.

The same-setup native batch path is not a true fused/batched GPU kernel; its
diagnostics must keep `true_batched_kernel = false` until a fused kernel exists.

`tprsApproxCUDA` remains deferred unless projection-floor, oracle-lambda,
timing, and graph-level evidence justify a new pure GPU approximation.

CUDA-specific tests remain opt-in. GitHub Actions are intentionally absent
unless reintroduced by explicit request.

## Precision ladder data-plane integration

The precision policy and skeleton data plane are integrated into `fast_kpc()`:

```text
precision = "fast":
    preserves fastSpline primary execution

precision = "compatible":
    routes through the authoritative resolver
    fails closed when semantic/version/runtime envelope checks fail
    executes CPU and CUDA skeleton data-plane slices for |S| <= 2
    uses mgcvExtractGPU where supported
    falls back through mgcvExtractCPU/GCVBridge and legacy mgcv

precision = "hybrid":
    keeps fastSpline primary execution
    executes near-alpha verifier residualization for skeleton |S| <= 2
    records verifier and fallback receipts
    preserves canonical replay
    uses verifier p-values in real skeleton edge/sepset decisions
```

The default remains the existing legacy-compatible fastkpc behavior unless
`precision` is explicitly requested. This precision ladder is distinct from
the explicit full-CUDA candidate above. Diagnostics distinguish
`backend_planned` from `backend_executed`; `backend_used` refers to the actual
executor. Current precision data-plane scope is skeleton only, CPU/CUDA, and
single-penalty `|S| <= 2`. CUDA precision tests include an opt-in native E2E
gate and a CPU/GPU parity artifact. The `mgcvExtractGPU` precision executor
uses same-setup x/y pair batching for selected fixed-sp CUDA solves and an
on-demand same-S prepared setup/spectral cache within each run. Eager same-S
group planning/batching, capacity-bounded prepared-cache eviction, WAN-PDAG,
`|S| > 2` multi-penalty GCV, and true fused/batched `mgcvExtractGPU` kernels
remain future work pending broader workload timing evidence.

End-to-end performance evidence is generated by:

```bash
FASTKPC_RUN_CUDA_TESTS=1 fastkpc/tools/run_precision_end_to_end_benchmark.sh
```

The benchmark writes `runs.csv`, `stage_timing.csv`, `cache.csv`,
`graph_agreement.csv`, `mode_summary.csv`, `comparison_summary.csv`,
`bottleneck_decision.csv`, and `summary.{json,md}` under
`fastkpc/artifacts/precision_end_to_end_benchmark/`. It compares legacy mgcv,
fastSplineCUDA, precision-scheduler primary-only CUDA, compatible CUDA, and
hybrid CUDA modes when native CUDA is enabled. By default it uses warm-up,
randomized measured mode order, and five repeats; set
`FASTKPC_PRECISION_E2E_REAL_DATA=/path/to/data.csv` or `.rds` to append an
external numeric workload.
