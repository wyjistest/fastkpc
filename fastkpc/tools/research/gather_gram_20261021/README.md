# Gather-then-Gram step 0 for path (i), 2026-10-21

Step 0 of [the level component prefill plan](../../../docs/PHASE10_LEVEL_COMPONENT_PREFILL_PROTOTYPE_PLAN_20261021.md)
(path (i): prefill eig ahead, defer Gram/self_moment to the original consumer
miss grouping). Results: [`PHASE10_GATHER_GRAM_STEP0_20261021.md`](../../../docs/PHASE10_GATHER_GRAM_STEP0_20261021.md).

The probe includes the production `full_cuda_ci_vertical.cu` and two verbatim
extracts written by `build.py`: `component_stage.inc` (the full production
component stage, same anchors and SHA as the payload gate probe) and
`gram_stage.inc` (only the `cublasDgemmStridedBatched` Gram call and the
`reduce_legacy_self_moment_kernel` launch, asserted to be a sub-range of the
full stage). `run_full` reproduces the payload gate's wide prefill;
`run_gram` uploads supplied vectors/selected values into the execution
context's miss buffers (same layout, lda, stride) and runs only the Gram
statements with `miss_count = count`. The accepted native is linked, not
modified.

Inputs are the archived frozen default-Inf residual capture
(`production_cohort_frozen_20261021/capture-inf/raw.tar.gz`) and the archived
accepted-native cache payload capture
(`component_payload_20261021/native/raw.tar.gz`), each extracted into its own
fresh directory. From the repository root on an idle GPU:

```sh
python3 -I fastkpc/tools/research/gather_gram_20261021/build.py "$WORK/build"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 fastkpc/tools/research/gather_gram_20261021/run.py \
  "$FROZEN_INF" "$NATIVE_PAYLOAD" "$WORK/build" "$WORK/out"

# sanitizer smoke on the first 40 original miss cohorts
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --target-processes all --error-exitcode 99 \
  python3 fastkpc/tools/research/gather_gram_20261021/run.py \
  "$FROZEN_INF" "$NATIVE_PAYLOAD" "$WORK/build" "$WORK/memcheck" --smoke 40
```

Phases: (1) 22-wide prefill of all 6,632 components, every field compared
with the native payload; (2) gather the prefilled vectors/selected values in
the original 2,592 miss cohorts (original order by `component_position`) and
run the Gram stage with the original `miss_count`, comparing self_moment with
the native payload; (3) control: the Gram stage in 22-wide groups, which must
reproduce the known singleton differences; (4)/(5) repeat phase 2 and repeat
it with reversed cohort processing order. `status` is PASS only if the three
prefilled fields and every gathered self_moment are bitwise identical in
phases 1, 2, 4 and 5 and the repeats are bitwise identical to each other.

Artifacts: `fastkpc/artifacts/full_cuda_ci/gather_gram_20261021/` (report,
build record, extracted includes, probe binary, sanitizer log and smoke
report, hashed manifest). Diagnostic H2D/D2H and timings are not production
measurements; this is isolated source-shared evidence, not native integration
qualification.
