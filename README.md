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

## Precision ladder control-plane integration

The precision policy control plane is integrated into `fast_kpc()`:

```text
precision = "fast":
    preserves fastSpline primary execution

precision = "compatible":
    routes through the authoritative resolver
    fails closed when semantic/version/runtime envelope checks fail
    executes a CPU skeleton vertical slice for |S| <= 1 through direct CI
    and mgcvExtractCPU/GCVBridge receipts

precision = "hybrid":
    keeps fastSpline primary execution
    records verifier and fallback plans
    preserves canonical replay
    does not yet execute verifier residualization or replace p-values in the
    real skeleton/WAN-PDAG data plane
```

The default remains the existing legacy-compatible fastkpc behavior unless
`precision` is explicitly requested. Diagnostics distinguish
`backend_planned` from `backend_executed`; `backend_used` refers to the actual
executor. The compatible data-plane slice is currently CPU skeleton only for
`max_conditioning_size <= 1`; |S| = 2, WAN-PDAG, CUDA, and hybrid verifier
execution remain future work. True fused/batched `mgcvExtractGPU` kernel work
remains blocked on scenario-aligned timing/workload evidence and broader
data-plane integration.
