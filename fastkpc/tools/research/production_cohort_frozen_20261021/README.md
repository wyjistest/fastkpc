# Actual production cohorts and frozen-coefficient GPU replay

This diagnostic supplements `residual_cohort_identity_20261021`. It captures
the actual accepted 22-column, max-level-3 one-call execution; it does not use
independently reconstructed two-target cohorts as the production baseline.

`capture.cpp` interposes three calls without rebuilding the accepted native:

1. The capacity multi-setup optimizer returns the real selected coefficients
   that will enter the exact-primary arena. Capture those bytes, Y, selected
   log-SP and route/rank/info, once per producer cohort.
2. The actual capacity coefficient finalizer records its ordered residual keys.
3. Its original cuBLAS GEMM runs unchanged. Capture its X, coefficient and
   fitted bytes, and then the original GPU residual after its completion event.

All additional transfers are explicit diagnostic D2H, outside production
authority counters. Capture timing is not performance evidence. The tool
checks all CI p-values, pMax, task/deletion identity, adjacency, sepsets,
n.edgetests, RNG and native resource closure against the archived subset run.

`replay.py` recreates every residual-v2 key from the captured base target key
and selected log-SP bits. It requires the actual finalizer coefficient bytes,
Y and metadata to equal the frozen producer values, anchors the GPU replay to
every original fitted/residual output, and then changes only target grouping
and column order. No optimizer or CPU numerical producer executes during
replay. The same n, q, cuBLAS math/atomics modes and residual `__dadd_rn` are
used. New groups contain distinct real targets from the same captured
producer; there is no synthetic padding. Unconsumed targets are not assigned
invented production references.

Run from the repository root on an idle GPU:

```sh
python3 -I fastkpc/tools/research/production_cohort_frozen_20261021/build.py \
  /tmp/fastkpc-production-cohort-frozen-20261021/build

env LD_PRELOAD=/tmp/fastkpc-production-cohort-frozen-20261021/build/capture.so \
  FASTKPC_COHORT_CAPTURE_NATIVE=/data/wenyujianData/kpcalg/fastkpc/build/fastkpc_cuda.so \
  FASTKPC_COHORT_CAPTURE_DIR=/tmp/fastkpc-production-cohort-frozen-20261021/capture/raw \
  FASTKPC_MULTI_PENALTY_PINNED_IO=0 \
  sh fastkpc/tools/with_phase10_exact_development_env.sh \
  Rscript fastkpc/tools/research/production_cohort_frozen_20261021/run_capture.R \
  /tmp/fastkpc-production-cohort-frozen-20261021/capture \
  /tmp/fastkpc-exact-speedup-20261020/subset-restored-queues32.rds

env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -I fastkpc/tools/research/production_cohort_frozen_20261021/replay.py \
  /tmp/fastkpc-production-cohort-frozen-20261021/capture \
  /tmp/fastkpc-production-cohort-frozen-20261021/build/replay.so
```

For deeper shapes on the same 22 columns, run `run_default_inf.R OUT -`
without interception to create a fresh reference, then run the same script
with interception and `OUT BASELINE/receipt.rds`. Use separate `baseline-inf`
and `capture-inf` directories under the work directory. The captured run must
match every p-value and graph field of that uninstrumented reference. Replay
`capture-inf` with the same `replay.py`; no source or optimizer change is
involved. The concrete alternative tested by the replay is a balanced
partition of each captured producer's targets into distinct-target chunks of
at most 16 columns, repeated twice. Output includes both original-shape
anchors and every regrouped comparison; unconsumed planned targets have no
invented reference.

Run the replay with `--memcheck-smoke` under
`/usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check full
--error-exitcode 99`. This selects the widest and deepest captured source
cohorts, and writes separate `smoke-*` reports. Finally:

```sh
python3 -I fastkpc/tools/research/production_cohort_frozen_20261021/archive.py \
  /tmp/fastkpc-production-cohort-frozen-20261021 \
  fastkpc/artifacts/full_cuda_ci/production_cohort_frozen_20261021
```

The artifact includes the actual raw captures as tar archives, original
receipts, per-file digests, replay libraries and source identities. Extract
each `raw.tar.gz` inside its capture directory to repeat the replay without
rerunning any optimizer.

This covers actual multi-penalty finalization on these 22 columns at n=351.
It does not qualify single-penalty projections, other inputs, regrouped
full-method p-values, or structured eig integration. The accepted runtime,
Phase 10, candidate, recommended route and sealed holdout remain unchanged.
