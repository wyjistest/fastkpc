# Persistent eight-lane component payload gate

This diagnostic compares all four cache fields (`vectors`, `selected_values`,
`total`, `self_moment`) byte for byte. It uses frozen real GPU residuals from the
22-column default-Inf capture and the production eig execution context, kernels,
Gram/self-moment computation and cache D2D statements. `build.py` extracts those
statements verbatim, records their hash, and builds outside the production tree.
The accepted native remains unchanged.

The reference is an actual accepted-native one-call capture of **cache-miss
cohorts**, not merely the larger residual-finalization cohorts. Mixed cache
batches can build only one component even when multiple residuals were formed.
`capture.cpp` intercepts the exact-primary consumer and its cache D2D writes;
`capture_native.R` requires full p/pMax, task/deletion, graph/sepset/count, RNG,
authority and resource parity against the existing uninstrumented reference.
All payload/residual downloads are explicit external diagnostics. Neither their
bytes nor their timings are production measurements.

From the repository root, with the prior frozen capture extracted under `FROZEN`
and a fresh working directory `WORK`:

```sh
python3 fastkpc/tools/research/component_payload_20261021/build.py "$WORK/build"
python3 fastkpc/tools/research/component_payload_20261021/run.py \
  "$FROZEN" "$WORK/build" "$WORK/native" --prepare-capture

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8 \
FASTKPC_PAYLOAD_CAPTURE_NATIVE="$PWD/fastkpc/build/fastkpc_cuda.so" \
FASTKPC_PAYLOAD_CAPTURE_KEYS="$WORK/native/keys.txt" \
FASTKPC_PAYLOAD_CAPTURE_DIR="$WORK/native/raw" \
LD_PRELOAD="$WORK/build/capture.so" \
Rscript fastkpc/tools/research/component_payload_20261021/capture_native.R "$WORK/native"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fastkpc/tools/research/component_payload_20261021/run.py \
  "$FROZEN" "$WORK/build" "$WORK/native-smoke" --native-anchor "$WORK/native"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fastkpc/tools/research/component_payload_20261021/run.py \
  "$FROZEN" "$WORK/build" "$WORK/native-full" --native-anchor "$WORK/native" --full
```

Smoke coverage uses 174 distinct targets across q=28/37/46/55 and widths
1/2/3/7/8/9/16/17/19/21/22, in both orders and with repetitions. Full coverage
first reproduces every original cache-miss cohort against the actual native
payloads, then compares all 6,632 consumed targets in 22-wide batches, reverse
order and two additional repeats. The full tail is 10. No repeated or fabricated
residual columns are added as padding. The probe reserves the actual 11 GiB
cache and uses eight persistent eig lanes; small batches activate fewer lanes.

Run the smoke command under `/usr/local/cuda/bin/compute-sanitizer --tool memcheck
--leak-check full --target-processes all --error-exitcode 99`, using the same
environment and `$WORK/memcheck` output. Save the diagnostic log as
`$WORK/memcheck.log`. `archive.py WORK DESTINATION` archives and verifies raw
payloads, receipts, probe binaries, sources and sanitizer evidence.

The JSON `status` is the numerical decision (`PASS` or `STOP_PAYLOAD_BITS`). A
normal process exit means the diagnostic completed, including a measured STOP;
it is not itself a numerical PASS. Integration is conditional on the JSON gate.
