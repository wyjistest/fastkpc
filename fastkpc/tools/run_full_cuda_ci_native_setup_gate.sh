#!/bin/sh
set -eu

phase4_count=${FASTKPC_PHASE7_PHASE4_PARTITION_COUNT:-16}
phase6_count=${FASTKPC_PHASE7_PHASE6_PARTITION_COUNT:-16}
gpu_count=${FASTKPC_PHASE7_GPU_COUNT:-2}
phase4_dir=${FASTKPC_PHASE7_PHASE4_PARTITION_DIR:-/tmp/fastkpc-phase7-native-phase4-partitions-v1}
phase6_dir=${FASTKPC_PHASE7_PHASE6_PARTITION_DIR:-/tmp/fastkpc-phase7-native-phase6-partitions-v1}
setup_dir=${FASTKPC_PHASE7_SETUP_CORPUS_DIR:-/tmp/fastkpc-phase7-native-setup-corpus-v1}
merged=${FASTKPC_PHASE7_MERGED_EVIDENCE:-/tmp/fastkpc-phase7-native-setup-full-evidence-v1.rds}

case "$phase4_count:$phase6_count:$gpu_count" in
  *[!0-9:]*|0:*|*:0:*|*:*:0)
    echo "Phase 7 partition and GPU counts must be positive integers" >&2
    exit 1
    ;;
esac

if [ "${FASTKPC_PHASE7_SKIP_BUILD:-0}" != "1" ]; then
  bash fastkpc/tools/build_cuda_native.sh
fi

FASTKPC_PHASE7_SETUP_CORPUS_DIR=$setup_dir \
FASTKPC_PHASE7_REBUILD_ORACLE=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
  Rscript fastkpc/tools/run_full_cuda_ci_native_setup_corpus.R

FASTKPC_PHASE7_NATIVE_SETUP=1 \
FASTKPC_FULL_CUDA_PHASE4_PARTITION_COUNT=$phase4_count \
FASTKPC_FULL_CUDA_PHASE4_PARTITION_DIR=$phase4_dir \
FASTKPC_FULL_CUDA_PHASE4_GPU_COUNT=$gpu_count \
FASTKPC_FULL_CUDA_PHASE4_MERGED_RDS=$phase4_dir/merged.rds \
  bash fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_partitions.sh

mkdir -p "$phase6_dir"
pids=""
partition_id=0
while [ "$partition_id" -lt "$phase6_count" ]; do
  gpu_id=$((partition_id % gpu_count))
  output=$(printf '%s/partition_%02d.rds' "$phase6_dir" "$partition_id")
  log=$(printf '%s/partition_%02d.log' "$phase6_dir" "$partition_id")
  CUDA_VISIBLE_DEVICES=$gpu_id \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  FASTKPC_PHASE7_NATIVE_SETUP=1 \
  FASTKPC_PHASE6_PARTITION_ID=$partition_id \
  FASTKPC_PHASE6_PARTITION_COUNT=$phase6_count \
  FASTKPC_PHASE6_CONCURRENCY=64 \
  FASTKPC_PHASE6_RUN_DCOV=1 \
  FASTKPC_PHASE6_PARTITION_OUTPUT=$output \
    Rscript fastkpc/tools/run_full_cuda_ci_multi_penalty_cuda_partition.R \
      >"$log" 2>&1 &
  pids="$pids $!"
  partition_id=$((partition_id + 1))
done

failed=0
partition_id=0
for pid in $pids; do
  if ! wait "$pid"; then
    log=$(printf '%s/partition_%02d.log' "$phase6_dir" "$partition_id")
    echo "Phase 7 Phase 6 partition failed: $partition_id" >&2
    tail -n 60 "$log" >&2
    failed=1
  fi
  partition_id=$((partition_id + 1))
done
if [ "$failed" -ne 0 ]; then
  exit 1
fi

FASTKPC_PHASE7_SETUP_CORPUS_MERGED=$setup_dir/source_evidence.rds \
FASTKPC_PHASE7_PHASE4_PARTITION_DIR=$phase4_dir \
FASTKPC_PHASE7_PHASE4_PARTITION_COUNT=$phase4_count \
FASTKPC_PHASE7_PHASE6_PARTITION_DIR=$phase6_dir \
FASTKPC_PHASE7_MERGED_EVIDENCE=$merged \
  Rscript fastkpc/tools/merge_full_cuda_ci_native_setup_evidence.R

FASTKPC_PHASE7_MERGED_EVIDENCE=$merged \
  Rscript fastkpc/tools/run_full_cuda_ci_native_setup_artifacts.R

echo "PASS Phase 7 native setup full gate: $merged"
