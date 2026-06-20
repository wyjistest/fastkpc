# fastkpc workspace

This repository keeps the legacy `kpcalg` sources and the staged `fastkpc`
backend work in one workspace. The active fast backend code is under
`fastkpc/`.

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
`precision` is explicitly requested. Diagnostics distinguish
`backend_planned` from `backend_executed`; `backend_used` refers to the actual
executor. Current precision data-plane scope is skeleton only, CPU/CUDA, and
single-penalty `|S| <= 2`. CUDA precision tests include an opt-in native E2E
gate and a CPU/GPU parity artifact. The `mgcvExtractGPU` precision executor
uses same-setup x/y pair batching for selected fixed-sp CUDA solves and an
on-demand same-S prepared setup/spectral cache within each run. Eager same-S
group planning/batching, capacity-bounded prepared-cache eviction, WAN-PDAG,
`|S| > 2` multi-penalty GCV, and true fused/batched `mgcvExtractGPU` kernels
remain future work pending broader workload timing evidence.
