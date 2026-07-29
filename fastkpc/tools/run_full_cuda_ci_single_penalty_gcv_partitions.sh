#!/bin/sh
set -eu

partition_count=${FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT:-16}
partition_dir=${FASTKPC_FULL_CUDA_PHASE4_PARTITION_DIR:-/tmp/fastkpc-phase4-full-shadow-partitions-v1}
gpu_count=${FASTKPC_FULL_CUDA_PHASE4_GPU_COUNT:-2}

case "$partition_count:$gpu_count" in
  *[!0-9:]*|0:*|*:0)
    echo "partition and GPU counts must be positive integers" >&2
    exit 1
    ;;
esac

mkdir -p "$partition_dir"
pids=""
partition_id=0
while [ "$partition_id" -lt "$partition_count" ]; do
  gpu_id=$((partition_id % gpu_count))
  output=$(printf '%s/partition-%03d-of-%03d.rds' \
    "$partition_dir" "$partition_id" "$partition_count")
  log=$(printf '%s/partition-%03d-of-%03d.log' \
    "$partition_dir" "$partition_id" "$partition_count")
  CUDA_VISIBLE_DEVICES=$gpu_id \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  FASTKPC_FULL_CUDA_PHASE4_PARTITION_ID=$partition_id \
  FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT=$partition_count \
  FASTKPC_FULL_CUDA_PHASE4_SHADOW_RDS=$output \
    Rscript fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_shadow.R \
      >"$log" 2>&1 &
  pids="$pids $!"
  partition_id=$((partition_id + 1))
done

failed=0
partition_id=0
for pid in $pids; do
  if ! wait "$pid"; then
    log=$(printf '%s/partition-%03d-of-%03d.log' \
      "$partition_dir" "$partition_id" "$partition_count")
    echo "Phase 4 shadow partition failed: $partition_id" >&2
    tail -n 40 "$log" >&2
    failed=1
  fi
  partition_id=$((partition_id + 1))
done
if [ "$failed" -ne 0 ]; then
  exit 1
fi

FASTKPC_FULL_CUDA_PHASE4_PARTITION_DIR=$partition_dir \
FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT=$partition_count \
  Rscript fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_merge.R
