# Default-off canonical dense prefill qualification

Completed campaign: [canonical qualification result](../../../docs/PHASE10_LEVEL_COMPONENT_PREFILL_CANONICAL_20261021.md).
All 65,694 consumed components and all 240,498 logical CI passed their gates;
the 906 p-value bit differences are fully attributed. The single uninstrumented
off/on pair was 892.468 s / 842.873 s. The flag remains default off.

Extends the qualified 351x22 prototype to 351x48, keeping the same dense eig
producer and original consumer grouping. Numerical production remains CUDA.
This campaign does not enable a production default, integrate structured eig,
freeze a candidate or change Phase 10/holdout/recommendation state.

Runtime changes are limited to jointly skipping begin/build/finish for an empty
layer, restoring the caller's device in prefill helpers, and allowing the two
explicit input shapes. The existing p<=64 entry gate already protects the
65-element diagnostic arrays. No general input qualification is implied.

Build with `FASTKPC_INCREMENTAL_EIG_ONLY=1 sh fastkpc/tools/build_cuda_native.sh`.
This rebuilds one-call, vertical CUDA and R API. Retained objects come from the
previous accepted build; the evidence class is source-closure development.

First run the existing configuration rejection and component-cache tests, then
`../level_component_prefill_20261021/run.R` in fresh off/on processes on all 22
columns and depths, using the previous 22-column receipt as the off reference.

`empty_layer.cpp` is a test-only planner interposer: level 3 is deliberately
empty while levels 0-2 execute unchanged on GPU. Compile with g++ C++17, `-shared
-fPIC -Ifastkpc/src -ldl`, then run `test_empty_layer.R OUTPUT off|on [REFERENCE]`
under `LD_PRELOAD`, setting `FASTKPC_LSCP_TEST_NATIVE` to the absolute native path.
Both modes must preserve the complete lower-level result and release resources.
This injected lifecycle test is not a real-data numerical qualification.

`device_lifecycle.cu` reuses the prior source-shared lifecycle smoke and adds
caller-device restoration on success, exceptions and destruction. Compile as
the previous `cache_lifecycle.cu`, with `-Ifastkpc/src/cuda`, linking the native,
R, cudart, cublas and cusolver. Under compute-sanitizer memcheck, expose GPUs 0,1:
the cache's work stays on GPU 0; GPU 1 is the calling thread's device. This is a
device-ownership test, not a multi-GPU algorithm. Both zero errors and zero
leaked bytes are required. The closure archive includes the exact 351-value
input as `device_lifecycle.residual.f64` (SHA-256 `bd87f2b6…7fa63`); pass it as the
test binary's only argument. `FASTKPC_LSCP_LIFECYCLE_FIXTURE` can point `archive.R`
to that file on reproduction, and its full SHA is checked before archiving.

Canonical timing and qualification, from the repository root on an idle GPU:

```sh
LSCP_WORK=/tmp/fastkpc-level-prefill-canonical-20261021
LSCP_TOOL=fastkpc/tools/research/level_component_prefill_canonical_20261021
sh fastkpc/tools/with_phase10_exact_development_env.sh Rscript "$LSCP_TOOL/run.R" \
  "$LSCP_WORK/canonical-off" off \
  fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/canonical-queues32.rds
sh fastkpc/tools/with_phase10_exact_development_env.sh Rscript "$LSCP_TOOL/run.R" \
  "$LSCP_WORK/canonical-on" on "$LSCP_WORK/canonical-off/receipt.rds"
```

The historical receipt is an output reference, not the timing baseline. Both
new runs use the same native, fresh R processes/caches, 11 GiB component cache,
bypass, 8 lanes and connections=32. Artifact serialization is outside the
timer. The on run is checked against the archived kpcalg authority for every
CI and pMax (absolute error <=1e-10); its bit differences from off are retained
explicitly for attribution. Task/deletion, graph/sepsets/counts/RNG, physical
group and ordered Gram-key hashes must stay exact. No CPU fallback is allowed.

The external `audit_payload.cpp` then independently observes actual finalizers
and pair consumers. Compile with g++ C++17, `-shared -fPIC`, CUDA include/library
paths, `-Ifastkpc/src/cuda`, and `-lcudart -lcublas -ldl -lcrypto`. Run separate
off/on diagnostic processes with the observer in `LD_PRELOAD` and set:

```text
FASTKPC_LSCP_TEST_NATIVE               absolute new native path
FASTKPC_LSCP_CANONICAL_AUDIT_DIR       fresh audit-off or audit-on directory
FASTKPC_LSCP_CANONICAL_AUDIT_MODE      capture or compare
FASTKPC_LSCP_CANONICAL_REFERENCE       audit-off directory (compare only)
```

Create each output directory's `raw/` subdirectory before the run. Pass the
corresponding uninstrumented receipt as the fourth runner argument. Diagnostic
results must match that timing receipt exactly; **diagnostic elapsed is never
performance evidence**. All extra large D2H is external to native counters.

After compiling the observer as `$LSCP_WORK/audit_payload.so` and completing the
timing pair, the sequential driver performs the 22-column level-3 observer
smoke, canonical diagnostic off/on, and offline attribution:

```sh
sh "$LSCP_TOOL/run_diagnostics.sh" "$LSCP_WORK"
```

The observer hashes coefficients, fitted, residual, vectors, selected_values,
total and self_moment for every first-consumed multi-penalty component. It
records ordered pair-to-component identities for all subsequent consumers.
Raw arrays are retained for first-miss targets produced by baseline cohorts
of width >=17. Other components must remain bitwise identical. Any changed
payload must have both an observed changed residual and a wide baseline
cohort; coefficient/route/rank/info changes fail closed.

After the diagnostic processes finish, `Rscript "$LSCP_TOOL/attribute.R"
"$LSCP_WORK"` joins full component identities back to dataset columns and S,
checks all 65,694 consumed components, and attributes every changed logical CI
and pMax edge. Physical pair output bits must equal the mapped logical CI bits;
the maximizing CI for each changed pMax edge is independently checked. It does
not treat a column number alone as an allowed exception.
The canonical authority threshold and graph gates must pass independently.

`archive.R WORK NEW_ARTIFACT_DIRECTORY` closes successful reports, hashes,
source/native identity, diagnostics and test logs. It records the single off/on
timing comparison without converting it into a repeated median or promotion.
