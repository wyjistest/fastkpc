# Opt-in level component prefill: 351×22 qualification

Implements path (i): finalize residuals inside the existing optimizer window,
prefill only vectors/eigenvalues/total, and compute Gram/self_moment on first
use with the original ordered miss cohort. The production flag
`FASTKPC_DCC_GAMMA_LEVEL_COMPONENT_PREFILL` defaults to **0**. Enabling it is
restricted to `dcc.gamma`, 351 rows, 22 columns, fresh semantic caches,
exact-primary coefficient reuse, component cache with residual bypass, eight
eig lanes, and the original single-group scheduler. Canonical qualification
and production default enablement are outside this prototype.

All numerical production is CUDA. CPU code manages identities, compares
bytes/receipts, and prepares existing input columns. Unsupported settings,
insufficient level capacity, or a level ≥3 consumer miss fail closed.

From the repository root, with an otherwise idle GPU:

```sh
LSCP_WORK=/tmp/fastkpc-level-prefill-20261021
mkdir -p "$LSCP_WORK"
FASTKPC_INCREMENTAL_EIG_ONLY=1 sh fastkpc/tools/build_cuda_native.sh \
  > "$LSCP_WORK/build-final.log" 2>&1

sh fastkpc/tools/with_phase10_exact_development_env.sh \
  Rscript fastkpc/tools/research/level_component_prefill_20261021/run.R \
  "$LSCP_WORK/final-off" off \
  fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021/baseline-inf/receipt.rds

sh fastkpc/tools/with_phase10_exact_development_env.sh \
  Rscript fastkpc/tools/research/level_component_prefill_20261021/run.R \
  "$LSCP_WORK/final-on" on "$LSCP_WORK/final-off/receipt.rds"
```

The build mode recompiles every translation unit that includes the changed
vertical header: one-call, CUDA vertical, and R API. Other objects are retained
from the accepted build. These receipts are source-closure development evidence,
not clean-checkout promotion evidence. The runner stores uncompressed receipts
after the compute timer and checks all CI/pMax bits, task/deletion identity,
adjacency, sepsets, counts, RNG, resource closure, and physical group / ordered
component key hashes. It does not run a new CPU oracle or PDAG campaign.

`audit_payload.cpp` is an **external diagnostic observer** of the real native
pair consumer. Compile it against the new header, then run a separate `on`
qualification with `LD_PRELOAD`. Set:

```text
FASTKPC_LSCP_NATIVE       absolute path to the newly built fastkpc_cuda.so
FASTKPC_LSCP_REFERENCE    extracted accepted component_payload_20261021/native/raw
FASTKPC_LSCP_PAYLOAD_LOG  output log (must start absent)
```

Compile with `g++ -O2 -std=c++17 -fPIC -shared`, includes `fastkpc/src/cuda`
and `/usr/local/cuda/include`, libraries `cudart`, `cublas`, `dl`, and CUDA
lib64 rpath. The observer reads all four fields after the real pair completion,
compares them against the accepted `.f64` files, and requires 6,632 unique PASS
records. Its explicit large diagnostic D2H is outside native production counters
and **must not be used as performance evidence**. No replacement p-value is
returned to the algorithm.

`cache_lifecycle.cu` includes the production CUDA implementation verbatim and
tests the same cache state machine under compute-sanitizer. It covers singleton
and two-entry D2D gather/Gram, duplicate resolution rejection, pending entries
discarded at reset, a level with no consumers, insufficient capacity, invalid
rank/duplicate key, and resource closure during exception unwinding. This is a
source-shared sanitizer smoke, not a complete native R run under sanitizer.

```sh
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -std=c++17 \
  fastkpc/tools/research/level_component_prefill_20261021/cache_lifecycle.cu \
  -Ifastkpc/src/cuda -Xlinker=-Bsymbolic fastkpc/build/fastkpc_cuda.so \
  -L/usr/local/lib/R/lib -Xlinker=-rpath=/usr/local/lib/R/lib \
  -lR -lcudart -lcublas -lcusolver -o "$LSCP_WORK/cache_lifecycle_final"

LD_LIBRARY_PATH=/usr/local/lib/R/lib CUDA_VISIBLE_DEVICES=0 \
  CUDA_DEVICE_MAX_CONNECTIONS=32 OPENBLAS_NUM_THREADS=1 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 "$LSCP_WORK/cache_lifecycle_final" \
  "$LSCP_REFERENCE/e526b295f52461bf3b30f515bdc61f58ee5741304e63124bdea93462423d3208.residual.f64"
```

Also run `fastkpc/tests/test_full_cuda_ci_level_component_prefill.R` for native
configuration rejection / resource cleanup, and the existing
`fastkpc/tests/test_full_cuda_ci_legacy_component_cache.R` for disabled-path
cache/bypass/lane regression coverage. Keep GPU runs sequential.

`FASTKPC_LSCP_REFERENCE=/path/to/native/raw Rscript archive.R WORK OUTPUT`
verifies every observed reference payload against its accepted SHA manifest,
the final off/on/observer receipts, hashes,
payload coverage and test logs, and saves the native binary, source snapshot,
runtime patch, receipts, tools' hashes and manifest. The expected final run/log
names are explicit in that script. Historical eager-payload STOP and Phase 10,
candidate, holdout and recommendation statuses remain unchanged.
