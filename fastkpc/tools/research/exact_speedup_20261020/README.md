# Exact speedup research, 2026-10-20

Results and evidence limits are recorded in
[`PHASE10_EXACT_SPEEDUP_EXECUTION_20261020.md`](../../../docs/PHASE10_EXACT_SPEEDUP_EXECUTION_20261020.md).
Run commands from the repository root. Production numerical work remains on
GPU; Python/R dense references and explicit component downloads are diagnostic
oracles only. The structured library is not loaded by the production runtime.

## Retained development configuration

The complete canonical comparison found a benefit from
`CUDA_DEVICE_MAX_CONNECTIONS=32`, set before CUDA context creation. The helper
also supplies the retained 11 GiB component cache, residual bypass and 8 eig
lanes. Explicit environment overrides take precedence:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  sh fastkpc/tools/with_phase10_exact_development_env.sh \
  Rscript fastkpc/tools/run_phase10_pvalue_contract_audit.R NEW_OUTPUT.rds
```

This is a development configuration for the measured RTX 4090. It is not a
promotion decision, a change to library defaults, or evidence for other GPU
memory capacities. Start a fresh R process, use fresh per-dataset caches and
leave other GPU/CPU workloads idle for timing comparisons.

## Reproducing the retired pinned experiment

Pinned optimizer buffers removed the memcpy mutex waits but provided no
incremental full-method gain. The production source has been restored.
`pinned_optimizer_io_experiment.patch` is the exact experiment diff against
`main@6a6a4b3`; it changes no numerical kernels. To reproduce, use a separate
checkout/build and apply only after checking for overlapping edits:

```sh
git apply --check fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_io_experiment.patch
git apply fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_io_experiment.patch
FASTKPC_INCREMENTAL_MULTI_PENALTY_ONLY=1 bash fastkpc/tools/build_cuda_native.sh
python3 -I fastkpc/tools/research/exact_speedup_20261020/run_canonical_campaign.py NEW_DIRECTORY
```

The campaign runs default/pageable, connections32/pageable, then
connections32/pinned sequentially, using a frozen R runner. It saves raw
receipts before offline auditing; do not edit the runner or rebuild the native
while a run is active. The individual runner rejects pinned=1 if the native
does not contain the experimental switch. `audit_saved_canonical.R` performs
the original canonical audit on a saved receipt without rerunning GPU work.

The archived campaign native SHA starts `c4f599ef`; the restored accepted native
starts `fe8286f7`. The 880.659 s full run belongs to the former, with pinned=0.
The latter has a separate subset parity receipt, not a new full canonical time.

## Structured eig isolation

The current GPU implementation passes the recorded component numerical smoke
gates but fails the performance progression gate. It is kept only for research.
`generate_recorded_dbdsqr.py` derives its QR iteration from the repository's
vendored LAPACK implementation, recording transformations instead of storing a
full right-singular-vector matrix.

```sh
structured_build_dir=$(mktemp -d /tmp/fastkpc-structured-eig.XXXXXX)
python3 -I fastkpc/tools/research/exact_speedup_20261020/generate_recorded_dbdsqr.py \
  "$structured_build_dir/recorded_dbdsqr.inc"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 --fmad=false -diag-suppress=2464 \
  -std=c++17 -Xcompiler=-fPIC -shared -I"$structured_build_dir" \
  fastkpc/tools/research/exact_speedup_20261020/structured_eig_gpu.cu \
  -o "$structured_build_dir/structured-eig.so"
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -I fastkpc/tools/research/exact_speedup_20261020/verify_structured_eig_gpu.py \
  "$structured_build_dir/structured-eig.so" --n 351 1000 \
  --output "$structured_build_dir/qualification.json"
```

`benchmark_structured_eig.py` compares identical inputs with the existing dense
lane benchmark. It reports dense **eig-only** device time versus structured
**complete-component** device time, excluding diagnostic D2H. Its ratio is not
an outer-method or identical-boundary speedup.

## Profiling and archive

`analyze_eig_lanes.py` uses NVTX call ranges to identify allocations inside
cuSOLVER. `analyze_nsys_profile.py` uses marker-kernel windows with a 200 ms gap
rule: these are approximate phase windows, not hardware occupancy measurements.
Summed host waits across threads must not be interpreted as removable elapsed
time. `plot_optimizer_queue.py` visualizes the original pinned queue evidence.

The immutable campaign manifest is under
`fastkpc/artifacts/full_cuda_ci/exact_speedup_20261020/manifest.json`.
`archive_execution.R` was run before reverting the experiment so its source
closure describes the measured binary. Restoration/profile receipts are
indexed separately; do not rewrite the original manifest to describe the
restored runtime as the producer of the full campaign.
