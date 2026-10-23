# Structured eig device adapter qualification

This is an isolated research library. It does not install a runtime switch,
modify the production native, enable prefill by default, or qualify a complete
skeleton. It accepts authenticated GPU residual captures as fixtures and keeps
all component/pair numerical production on CUDA. Archived kpcalg p-values are
read only for comparison; there is no CPU numerical fallback.

The device adapter uses the existing structured solver with k+1 eigenpairs.
Its CUDA translation unit preserves `--fmad=false`. The consumer translation
unit includes the current production vertical CUDA source with its usual build
flags; centering and pair-stage statements are extracted verbatim at build
time. It reuses the original total reduction, dense eig lanes, delayed Gram,
self-moment reduction, cuBLAS pair GEMM, and GPU gamma kernel.

The normal path is device residual -> structured eig -> compact status -> GPU
centering -> D2D three-field cache store -> completion event -> original delayed
Gram -> consumer stream event wait -> GPU pair/gamma -> compact result. The
fixture upload and explicit diagnostic payload shadows are outside this path.
There is one structured status checkpoint per bounded batch, not per target.
This experiment makes no claim of a host-wait-free asynchronous producer.

The prototype stores `component_solver_identity` separately from residual
identity: 1 is `structured-device-three-field-v1`, 2 is
`retained-dense-three-field-v1`. The research metadata wraps the existing cache
payload layout; it does not change the production cache lookup schema. A real
runtime integration must add that identity to its own entry metadata.

The fixed, predeclared fallback policy is:

- structured info nonzero, including true rotation-log overflow;
- nonfinite gap or `(abs(lambda[k-1])-abs(lambda[k])) /
  max(abs(lambda[k-1]),abs(lambda[k]),DBL_MIN) < 1e-8`.

Fallback reruns only the affected component through the original GPU dense
producer. No pending entry can be read or resolved, and no entry in the batch
is published if dense reports an error. A sufficient gap is a guard, not a
proof of the p-value bound; the independent authority gate is mandatory.

From the repository root:

```sh
python3 fastkpc/tools/research/structured_adapter_20261022/build.py \
  /tmp/fastkpc-structured-adapter-20261022

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/cuda/lib64 \
python3 fastkpc/tools/research/structured_adapter_20261022/run.py \
  /tmp/fastkpc-structured-adapter-20261022
```

`--smoke` runs four real components, one pair for each of four n values, all
fallback/error tests and two concurrent contexts. Run that mode under
`/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full
--error-exitcode 77` for resource/error-path qualification. This is not a
full-R/full-method sanitizer run.

The full replay includes 640 independent exact-fixed real components, 72
deep/real-input pairs, and 951 actual canonical prefill residuals. Inputs are
authenticated against the two earlier archive manifests. Canonical fixtures
retain full residual identities, native provenance, route/rank/info and the
actual three/four-field payloads. Their first-use cohorts are reconstructed
from request identity and dataset-target order; the dense self-moment check
reports whether the reconstruction reproduces the captured bytes. No full
canonical graph or scheduler qualification is inferred from this replay.

Batch widths 1, 2, 17, 31 and 64 vary component order and position while keeping
the delayed Gram groups/order fixed. Only the three fields are required to be
independent of producer batch shape. The resulting compact GPU outputs must
also repeat exactly. Ten independent contexts of capacity 64 test the actual
concurrent execution shape; each has its own stream/workspace and real D2D
cache stores. Diagnostic D2H and Python thread overhead are included in those
recorded walls, so they are not performance evidence or a prediction of whole
method speedup. This experiment does not model an 11 GiB live cache or overlap
with an optimizer.

The next gate after a complete PASS is an opt-in 22-column full method with
unchanged scheduler/delayed Gram, full authority/graph/RNG/resource gates and
same-binary off/dense/structured comparisons. Canonical full method follows
that gate. Optimizer overlap and level-2 prefill remain separate later work.
