# Structured eig: bounded real-residual replay

This is an isolated diagnostic, not a production implementation. The accepted
native, optimizer, residual producer, method scheduler and cache are unchanged.
The result is recorded in
[the real replay report](../../../docs/PHASE10_STRUCTURED_EIG_REAL_REPLAY_20261020.md).

Run from the repository root. All GPU work below uses one RTX 4090; the other
GPU should be idle. Use a new artifact directory for a new campaign. The
examples show the recorded directory for clarity.

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/research/structured_real_20261020/capture.R \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020

python3 -I fastkpc/tools/research/structured_real_20261020/build.py \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -I fastkpc/tools/research/structured_real_20261020/benchmark.py \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020
```

`capture.R` requires the accepted native matching the saved subset receipt.
It uses the existing native capacity dispatcher, exact-primary optimizer and
exact-fixed stable-SVD evaluator, with two targets per independently replayed
cohort. The original 22-column task receipt is read from
`/tmp/fastkpc-exact-speedup-20261020/subset-restored-queues32.rds`; the archive
also preserves this receipt as `inputs/subset-source-receipt.rds`. The canonical
dataset is preserved as `inputs/canonical-data.rds`. Previously completed
cohorts can resume only with identical task and native identities.

640 distinct `(target,S)` components come from 320 unique three-variable S
sets. Selection uses 64 near-alpha development pairs and 256 evenly spaced
remaining sets; it is fixed before comparing eig outputs. Residual bytes are
stored as little-endian doubles, one complete component after another.
`pairs.csv` binds components to original task IDs and offline kpcalg oracles.
These diagnostic downloads and CPU oracle calculations are outside production
and never supply a fallback residual or eig result.

The dense adapter includes the existing eig-lane benchmark's implementation;
it uses eight deterministic cuSOLVER lanes on exactly the same residual bytes.
Dense timing is eig-only device time. Structured timing includes H2D, all four
GPU stages, diagnostic D2H, and thread start/join. Neither includes context
creation or represents a full-method time. All repeats compare output bytes,
including across zero-reservation and 11 GiB-reservation configurations.

The reservation is `cudaMalloc` plus a full `cudaMemset`, with head/tail checks
on release. It models the resident capacity of the component cache, not its
traffic, identities, replacement policy, optimizer or setup workspaces.
Memory observations are sampled at boundaries, not continuous vendor peak
allocation measurements. Rotation capacity is frozen at 170,662 from the prior
synthetic campaign. Overflow is tested explicitly and aborts with no output.

Supplemental exact witnesses use authenticated, previously captured residuals
from 72 cohorts spanning canonical hard cases and three real datasets. They
have independent production-capacity residual-bit parity in the prior archive.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/research/structured_real_20261020/export_witnesses.R \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020

CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -I fastkpc/tools/research/structured_real_20261020/qualify_witnesses.py \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020
```

For memory checking, prefix the last Python command with
`/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full --error-exitcode 77`
and append `--sanitizer-smoke`. This tests one pair for each of four n values,
both eig implementations, repetition, and the rotation-overflow path.

Profile separately from performance measurements:

```sh
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o fastkpc/artifacts/full_cuda_ci/structured_real_20261020/real640 \
  python3 -I fastkpc/tools/research/structured_real_20261020/profile.py \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020

nsys export --type sqlite \
  --output fastkpc/artifacts/full_cuda_ci/structured_real_20261020/real640.sqlite \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020/real640.nsys-rep

python3 -I fastkpc/tools/research/structured_real_20261020/analyze_profile.py \
  fastkpc/artifacts/full_cuda_ci/structured_real_20261020
```

`archive.py` checks recorded source/native/build identities and numerical gates,
then archives source snapshots, input receipts and logs with a SHA-256 manifest.
It expects this campaign's logs at their documented `/tmp/fastkpc-structured-real-*`
locations. The final archive has everything needed to replay the eig experiment
without recapturing residuals. It is development evidence, not a clean producer
performance campaign, full-method qualification or promotion evidence.
