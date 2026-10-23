# GPU singleton self-moment sensitivity

This research-only campaign keeps the accepted native unchanged. It exports
the actual 22-wide candidate scalars, captures all original consumer batches
that reference one of the 1,002 changed components (including all-hit batches),
and replays the production pair GEMM and GPU gamma kernel. CPU code performs
identity checks, byte comparisons and serialization; it produces no CI values.

`capture.cpp` interposes the common legacy-eig implementation, so both the exact
residual and all-hit public entries are covered. It records all native requests
for coverage. For relevant batches it copies actual component payloads, pair
cross matrices and compact outputs after original completion. The SHA helper
interposition observes target/prepared identities, preserving each hash result,
to bind physical pairs to canonical `(S, x, y)` tasks without matching by p value.
Explicit diagnostic D2H is outside native counters; timings are not evidence.

`probe.cu` uses the production execution context and kernels. `build.py` extracts
the exact pair stage into `pair_stage.inc`. Each batch retains its original pair
count and order. Before any scalar substitution, every pair-cross byte, every
compact field and the canonical p bits must match the accepted-native capture.
Only `self_moment` is replaced. All other payload bytes, the pair-cross bytes,
statistic and mean must remain exact; two additional candidate repeats must match.

Run from the repository root. `OLD` is the previous payload campaign working
directory with `native/raw` extracted; `WORK` is a fresh directory. GPU 0 is used.

```sh
python3 fastkpc/tools/research/singleton_pvalue_20261021/build.py "$WORK/build"
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fastkpc/tools/research/singleton_pvalue_20261021/export_candidates.py \
  "$OLD" "$WORK/candidates"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1 \
FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8 \
FASTKPC_SCALAR_CAPTURE_NATIVE="$PWD/fastkpc/build/fastkpc_cuda.so" \
FASTKPC_SCALAR_CAPTURE_KEYS="$WORK/candidates/changed-keys.txt" \
FASTKPC_SCALAR_CAPTURE_DIR="$WORK/native/raw" \
LD_PRELOAD="$WORK/build/capture.so" \
Rscript fastkpc/tools/research/singleton_pvalue_20261021/capture_native.R "$WORK/native"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 fastkpc/tools/research/singleton_pvalue_20261021/replay.py \
  "$WORK" "$OLD" "$WORK/full"

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
LD_LIBRARY_PATH=/usr/local/lib/R/lib:/usr/local/cuda/lib64 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full \
  --error-exitcode 99 --log-file "$WORK/memcheck.log" \
python3 fastkpc/tools/research/singleton_pvalue_20261021/replay.py \
  "$WORK" "$OLD" "$WORK/memcheck" --smoke

python3 fastkpc/tools/research/singleton_pvalue_20261021/archive.py \
  "$WORK" "$OLD" fastkpc/artifacts/full_cuda_ci/singleton_pvalue_20261021
```

The scalar export repeats natural order, reverse order and natural order again.
The full replay covers all affected CI tasks and unaffected pairs sharing their
batches; `--smoke` retains representatives of each pair count and all-hit use.
`PASS_GPU_DIAGNOSTIC` requires valid GPU status, no alpha flips and measured
delta <= 1e-13 on this fixed task schedule. It does not relax the existing
payload bitwise gate, establish agreement with kpcalg, or qualify a new full
method, pMax, graph, scheduler, Gram implementation or production baseline.
