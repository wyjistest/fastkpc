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
