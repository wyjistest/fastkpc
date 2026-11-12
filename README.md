# fastkpc workspace

This repository keeps the legacy `kpcalg` sources and the staged `fastkpc`
backend work in one workspace. The active fast backend code is under
`fastkpc/`.

## Full-CUDA compatible skeleton and WAN-PDAG candidate

The [clean-checkout qualification and repeated timing](fastkpc/docs/PHASE10_CLEAN_GENERAL_INPUTS_20261106.md)
use committed producer `2be6a81` and full-build native `d8793173…10d7350`.
All 44 objects were rebuilt. Seven full inputs, including three real datasets,
500x50 at default rank 50 and 1025x4 at default rank 102, pass 14 correctness
and 42 isolated timing runs against fresh unmodified R references. All 63,056
CI comparisons (maximum error `1.83624e-13`), 146,576 orientation CI checks,
final PDAG/RNG, authority and resource gates pass. The November 12 independent
audit rechecks all 56 raw receipts and seals their identities.

All seven inputs meet the predeclared outer non-regression gate (ON/OFF median
<=1.05). Only real 300x10 meets the speed gate: **3.353/2.893 seconds** (13.72%
less elapsed time). Public 500x50 is **24.986/25.265 seconds**; the other six
inputs are speed `NOT_PROVEN`. These compare two GPU configurations, each with
three full WAN-PDAG timings, and do not establish a general GPU-vs-CPU speedup
or promotion. Original `kpcalg/R` is restored; installed native
`2727ef51…9198301` remains. See the [final evidence index](fastkpc/docs/PHASE10_CLEAN_GENERAL_INPUTS_ATTESTATION_20261112.json).

The [public reproduction entry](fastkpc/tools/reproduce_public/README.md) builds
the frozen producer in a separate checkout, generates public inputs, computes
fresh original-R references and runs the A/C protocol without private data or
historical binaries. It also packages relative-path evidence for independent
hash verification and raw-receipt auditing after relocation.
Its [local validation](fastkpc/docs/PHASE10_PUBLIC_REPRODUCTION_20261112.md)
passes a new full build, 4 correctness and 12 timing runs, and relocation of
1,643 evidence files. Both public inputs meet non-regression; neither proves
speed. This does not replace the seven-input evidence above.

Development priority was corrected on 2026-11-02: production optimizations must
dispatch on model dimensions and resource capacity, with validation across
multiple input shapes. The 351 x 48 input is one regression case. See the
[general-shape implementation and validation](fastkpc/docs/PHASE10_GENERAL_SHAPES_20261102.md).
The [2026-11-06 rank qualification](fastkpc/docs/PHASE10_NUMCOL_GENERALIZATION_20261106.md)
also removes the fixed `numCol=35` requirement. Qualified development native
`2727ef51…9198301` passes 16 input/rank cases in 50 CPU/GPU runs, including three
complete real inputs and the actual 500x50 default rank 50, plus other-method
and complete canonical regressions. The previous `476ad65e…5e5a37b` twelve-input /
72-run evidence retains its explicit rank-35 scope. Speed depends on workload;
switches remain opt-in. That earlier qualification established no new performance median. Historical
timings below retain their original input and binary scope.

The Git handoff preserves the original `kpcalg/R` reference. The earlier November 6 local
qualification used a separate, uncommitted scalar-diagonal correction in
`dcov.gamma`; its receipts retain that R source identity. Rank-one GPU results
were checked against independent CPU full eig and the patched R function, not
the original R function, which errors at that rank. Rank-one CPU-authority
orientation remains unqualified with the original R reference. That patch is
now retained only as an archived fixture; see the rank report for its historical
reproducibility boundary.

Runtime/tool commit `8d139b8` is bound to the local native by the
[November 6 source attestation](fastkpc/docs/PHASE10_SOURCE_ATTESTATION_20261106.json):
107 build inputs and 97 fastkpc R files match Git; the local legacy R patch is
listed separately. This handoff did not rerun the GPU campaign.

The preceding runtime's committed source and evidence identities are recorded in the
[2026-10-30 development handoff](fastkpc/docs/PHASE10_DEVELOPMENT_HANDOFF_20261030.md).
Build fix `89a8099` and runtime `ae3687f` match the measured source closure;
the [isolated capacity acceptance](fastkpc/docs/PHASE10_STRUCTURED_ROWS_ISOLATED_20261030.md)
retains its original precommit run identity.
The [build-entry qualification](fastkpc/docs/PHASE10_BUILD_CONSISTENCY_20261029.md)
passes full/incremental native identity and numerical gates. The structured
row-only change was installed as native `b2b79fc6…e135b12b`: **GO_CAPACITY**,
with 601.57 MiB less requested pool storage and a recorded 699.125 MiB of L6 checkpoint
headroom above the 2 GiB gate. All six isolated runs pass numerical, capacity,
resource and predeclared outer non-regression gates. Matched OFF/ON medians are
**283.985/283.328 seconds**; speed is **NOT_PROVEN** (0.23% observed decrease,
2/3 paired wins). The capacity version's 283.328-second median is local
source-closure development evidence, not a performance run from a clean checkout or promotion.
Those historical slot results cover the specified `dcc.gamma` 351 x 48 input.
The current multi-shape implementation and its separate evidence are linked above.

The [2026-10-29 handoff](fastkpc/docs/PHASE10_DEVELOPMENT_HANDOFF_20261029.md)
preserves runtime `c469ac5`, attestation `e15be9d`, and the historical
**290.491-second** result bound to native `a1e16a9c…7b73bd34`. Its difference
from 283.328 seconds is not attributed to this change. The separate
[shared-host supplement](fastkpc/docs/PHASE10_STRUCTURED_ROWS_SHARED_HOST_20261029.md)
retains its original receipts and is not mixed into isolated acceptance.
Phase 10 remains `ACTIVE`, candidate `NOT_FROZEN`, recommended `false`, and
the externally held holdout `SEALED_NOT_RELEASED`.

The historical Phase 10 v1 correctness claim is withdrawn. Its retrospective
audit found 79,012 CI p-values and 1,398 `pMax` cells outside the absolute
`1e-10` contract (maximum error `0.045461329947135143`). Graph parity did not
establish numerical correctness. Those receipts and the old 219.513-second
result are invalid as correctness or performance-candidate evidence. See the
[semantic repair](fastkpc/docs/MGCV_GPU_SEMANTIC_CLOSURE.md) and current handoff.

The active correctness contract is now default `kpcalg`: `m.max = Inf`, resolved
to `p - 2`, with graph-driven natural stopping. Current Phase 9 and hardening v2
artifacts cover all 240,498 canonical tests through level 8, including the nine
tests omitted by the historical max-7 campaign.

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
    max_conditioning_size = Inf,
    index = 1,
    trace_level = "logical"
  )
)
```

The strict end-to-end entrypoint keeps that CUDA skeleton and uses the original
kpcalg implementation for the CPU WAN-PDAG orientation stage:

```r
result <- fastkpc_compatible_cuda_wanpdag(
  data,
  alpha = 0.1,
  options = list(
    max_conditioning_size = Inf,
    index = 1,
    trace_level = "logical",
    ci_method = "hsic.gamma",
    hsic_params = list(sig = 1)
  )
)
```

This authority route records every orientation CI identity, p-value, decision,
and RNG boundary. The native fastkpc orientation implementation remains an
explicit development route and is not used for strict 351 x 48 claims.

This route implements the exact Gaussian/identity `regrXonS` subset used by
the KPC skeleton: a joint thin-plate smooth for `|S| <= 2`, additive smooths
for `|S| > 2`, target-specific GCV selection, CUDA residual formation, and
legacy-compatible CUDA evaluation for `dcc.gamma`, `dcc.perm`, `hsic.gamma`,
and `hsic.perm`. It is not a general mgcv clone.

The public input contract is:

```text
finite binary64 matrix; n > 5; 2 <= p <= 64
alpha = 0.1; index = 1
numCol: caller-specified integer, 1 <= numCol < n; default floor(n / 10)
max_conditioning_size defaults to Inf and resolves to p - 2
additive CUDA capacity through |S| = 62, subject to q = 1 + 9|S| < n
Gaussian family; identity link; unweighted; zero offset
CUDA skeleton plus optional kpcalg-authority WAN-PDAG orientation
```

The examples omit `numCol`: 500 rows use rank 50, while 351 rows use rank 35.
An explicit rank is required when `floor(n / 10)` is zero. Rank describes the
CI spectral/low-rank approximation, not the number of input variables. A rank
outside a structured kernel's capacity uses the dense GPU implementation without
changing the requested parameter. Qualification covers the recorded inputs in
the [rank validation report](fastkpc/docs/PHASE10_NUMCOL_GENERALIZATION_20261106.md).

Strict skeleton mode fails closed outside the input/model contract. It does not silently
call legacy mgcv, a CPU numerical skeleton CI path, `fastSplineCUDA`, or another
approximate backend. C++ owns canonical skeleton replay; CUDA owns repeated
skeleton CI work. When WAN-PDAG is requested, kpcalg CPU orientation is an
explicit authority stage rather than a fallback. The result and target-state
caches are capacity-bounded and keyed by semantic dataset/test identities.

Build and validate the current explicit route with:

```bash
bash fastkpc/tools/clean_cuda_native.sh
bash fastkpc/tools/build_cuda_native.sh
Rscript fastkpc/tests/test_full_cuda_ci_one_call.R
Rscript fastkpc/tests/test_full_cuda_ci_one_call_cache.R
Rscript fastkpc/tests/test_compatible_cuda_wanpdag_authority.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_hardening_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_campaign_artifact.R
Rscript fastkpc/tests/test_full_cuda_ci_phase10_completion_audit.R
```

Qualification is specific to each recorded source, native and configuration.
The completion audit intentionally fails until the v2 five-run performance
campaign, the sealed holdout artifact, and the final `COMPLETE` roadmap state
all exist. Artifact tests can report `SKIP` with exit code zero when local
materials are absent; a successful exit alone is not a qualification `PASS`.

Historical v1 campaign evidence is under
`fastkpc/artifacts/full_cuda_ci/promotion_351x48_v1/`. Five fresh-process cold,
five replay-warm, and five same-campaign correct-baseline runs all reproduced
110 edges, SHD 0, exact sepsets, exact logical test counts, and the exact
deletion trace. The `0.699`-second result is replay-warm: it follows a complete
same-data call and performs zero physical CI tests or residual fits. The
fresh-process cold median was `1290.664` seconds versus a `601.431`-second CPU
baseline median. All CPU numerical authority, approximate-backend, and
unknown-fallback counters were zero.

The v1 receipts remain historical diagnostics only; their original labels
do not override the retrospective numerical failure. Promotion requires five fresh-data compute-warm runs with
empty dataset-specific caches, a median at most 120 seconds and at most 0.80 of
the same-campaign correct baseline, plus a cold median no slower than that
baseline. Replay-warm is report-only. The standard gate remains unavailable
for promotion until the v2 campaign and public 500x50 development fixture pass:

```bash
FASTKPC_RUN_CUDA_TESTS=1 bash fastkpc/tools/run_full_cuda_ci_gate.sh
```

The current 351x48 default-`Inf` qualification covers 240,498 logical CI tests
through natural stopping after level 8. The latest six-run OFF/ON campaign
preserves CI/pMax bits, task/deletion/graph/sepset/count/RNG semantics and
resource closure; maximum authority error is `3.05425129631942e-12`. The
283.328-second capacity-version median still fails both the `<180s` checkpoint and the
five-run 120-second promotion gate. Historical max-7, v1, replay-warm and
pre-semantic-closure timings are not current producer performance.

Only after those prerequisites and a new source/native/contract freeze may an
external custodian release be supplied. No holdout release input should be
provided for the current candidate.

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
