#!/bin/sh
set -eu

output=${FASTKPC_FULL_CUDA_PHASE4_BACKEND_RDS:-/tmp/fastkpc-phase4-backend-current-v1.rds}
gpu_samples=${FASTKPC_FULL_CUDA_PHASE4_GPU_SAMPLES:-/tmp/fastkpc-phase4-backend-gpu0-current-v1.csv}

if [ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]; then
  echo "Phase 4 backend benchmark requires physical GPU 0" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export FASTKPC_FULL_CUDA_PHASE4_BACKEND_RDS=$output

rm -f "$gpu_samples"
nvidia-smi --id=0 \
  --query-gpu=timestamp,index,uuid,name,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits --loop-ms=100 --filename="$gpu_samples" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

Rscript fastkpc/tools/run_full_cuda_ci_single_penalty_gcv_backend.R
