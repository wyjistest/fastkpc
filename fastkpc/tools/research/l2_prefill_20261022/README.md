# Frozen L2 prefill diagnostic

This tool qualifies real canonical L2 selected-SP and residual cohorts before
any runtime L2 integration. It does not install a native, change a production
gate, rerun an optimizer in the probe, or use a CPU fit/eig/gamma producer.
It requires the source-closure native `89a2fab0…` and its canonical receipts.

The observer captures the actual L0–L2 run, saving only L2 numerical fixtures.
Diagnostic D2H shadows are explicit and excluded from performance claims.
The prefix must preserve all 180,165 CI bits, pMax, task/deletion identity,
adjacency, counts, RNG, GPU-only production and tracked resource closure.

The probe directly calls the production exact fixed evaluator with captured
geometry, Y and selected log-SP. It first repeats every original cohort and
requires bitwise coefficients/fitted/residual plus exact route/rank/info. Then
it groups the planned distinct targets by real S into balanced cohorts of at
most 16, retaining one prepared handle per S and detaching residuals to a GPU
arena. The original consumer and Gram order is preserved. Original residuals
plus dense components must reproduce every original physical p-value bit.
Regrouped residuals plus dense and structured components must pass all logical
L2 p-values against independent archived kpcalg authority at `1e-10`, with no
decision flips. Each mode is repeated bitwise.

Logical matching uses directed `(x,y,S)` recovered from the captured consumer
order by `map.py`. Reciprocal `(y,x,S)` computations are separate tests and can
differ in low bits even in the unchanged baseline. The presentation TSV has
sorted endpoints; it is checked against, but never substitutes for, this map.
The initial unordered-map post-check was rejected on the unchanged original
anchor. A second comparison detected only authority matrix dimnames on the
pMax finite mask. The corrected check retains shape, nonfinite positions/bits
and the finite-value tolerance. Rejected logs are retained. The existing GPU
outputs are immutable; only the post-checks are rerun for these fixes.

The component probe includes unchanged production lifecycle code, the linked
production structured object, and the extracted production GPU pair stage.
It calls the existing level-3 lifecycle with L2 fixtures; it does **not** qualify
a public L2 gate. The existing dense GPU fallback policy is unchanged.

From the repository root, with both GPUs free of compute jobs, use a fresh OUT:

```sh
export L2_OUT=fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/cuda/lib64
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8
export FASTKPC_L2_NATIVE="$PWD/fastkpc/build/fastkpc_cuda.so"
python3 fastkpc/tools/research/l2_prefill_20261022/build.py "$L2_OUT/build"
env LD_PRELOAD="$PWD/$L2_OUT/build/capture.so" FASTKPC_L2_CAPTURE="$PWD/$L2_OUT/capture/raw" \
  Rscript --vanilla fastkpc/tools/research/l2_prefill_20261022/capture.R "$L2_OUT/capture"
python3 fastkpc/tools/research/l2_prefill_20261022/build.py "$L2_OUT/build" --probe
python3 fastkpc/tools/research/l2_prefill_20261022/run.py "$L2_OUT" fixed
python3 fastkpc/tools/research/l2_prefill_20261022/run.py "$L2_OUT" fixed-timing
python3 fastkpc/tools/research/l2_prefill_20261022/campaign.py "$L2_OUT"
```

The component order is original/dense/structured/structured/dense. The campaign
stops on any numerical or resource failure, then runs two small sanitizer
fixtures and the logical qualification. GPU0 runs sequentially; GPU1 remains
idle. Both cards are sampled every 200 ms. Samples are not a strict instantaneous
VRAM bound, nor an integrated-runtime capacity qualification.

`fixed.json` is the complete residual comparison; `fixed-timing.json` adds two
no-shadow repetitions including prepared teardown. Its driver time excludes
initial arena allocation and input loading. Component timings separate build
from original Gram/pair replay; they exclude fixture upload and pool setup and
teardown. Python orchestration is included in the replay boundary. None of these
timers is a complete method wall time. The full pMax check substitutes candidate
L2 p-values into a frozen higher-level trace; no new full skeleton or PDAG run is
claimed.

For this campaign the observer recipe was finalized after capture. Preserve
`build/build.json` as `build/capture-build-at-run.json`, rebuild the observer,
and require the same binary SHA before running `archive.py "$L2_OUT"`. The archive
validates all production source hashes against the prior full source snapshot,
records the fixed numerical and timing phases separately, derives residual
changes by original width and unique first consumption, and binds raw inputs,
outputs, tools, native and prior authority receipts by SHA. It makes no claim
that the pre-timing fixed qualification used the final timing-instrumented
Python driver verbatim.

Full-runtime opt-in L2 qualification remains a separate step. Production
defaults, Phase 10, candidate, holdout and recommendation are unchanged.
